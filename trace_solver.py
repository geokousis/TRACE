#!/usr/bin/env python3
"""
TRACE: Thermodynamic Resolution of Ancestry and Continuous Evolution.

This module solves a discretized one-dimensional variational ancestry model
for unphased diploid genotype likelihoods:

    S(phi) = 0.5 * D * sum_i (phi[i + 1] - phi[i])**2 / d_cM[i]
             - sum_i log(P(GL_i | phi_i, f_A_i, f_B_i))

where phi is bounded to [0, 1]. The implementation uses SciPy's L-BFGS-B
optimizer with an analytical gradient.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from collections.abc import Iterable
from typing import Final

import numpy as np


LOGGER = logging.getLogger("TRACE")

try:
    from scipy.optimize import Bounds, check_grad, curve_fit, minimize
except ModuleNotFoundError as exc:  # pragma: no cover - exercised by environment
    raise SystemExit(
        "SciPy is required. Install dependencies with: "
        "python3 -m pip install -r requirements.txt"
    ) from exc


EPS: Final[float] = 1e-12
MIN_CM_DELTA: Final[float] = 1e-6


@dataclass(frozen=True)
class TraceResult:
    phi: np.ndarray
    action: float
    converged: bool
    message: str
    iterations: int
    seconds: float


@dataclass(frozen=True)
class ReferencePanel:
    chromosomes: np.ndarray
    positions: np.ndarray
    frequencies: np.ndarray
    population_names: tuple[str, ...]
    allele_counts: np.ndarray | None = None
    refs: np.ndarray | None = None
    alts: np.ndarray | None = None


@dataclass(frozen=True)
class InferenceChunk:
    chromosome: str
    positions: np.ndarray
    frequencies: np.ndarray
    genotype_likelihoods: np.ndarray
    c_m_positions: np.ndarray
    ploidy: np.ndarray
    population_names: tuple[str, ...]
    allele_counts: np.ndarray | None = None


def frequency_site_weights(
    frequencies: np.ndarray,
    allele_counts: np.ndarray | None = None,
) -> np.ndarray:
    """Down-weight SNPs whose reference panels provide little ancestry information."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    if frequencies.ndim != 2:
        raise ValueError("frequencies must have shape (N, K).")
    counts = None if allele_counts is None else np.asarray(allele_counts, dtype=np.float64)
    if counts is not None:
        if counts.shape != frequencies.shape:
            raise ValueError("allele_counts must have the same shape as frequencies.")
        variance = frequencies * (1.0 - frequencies) / np.clip(counts + 1.0, 1.0, None)
    if counts is not None and frequencies.shape[1] == 2:
        separation = np.abs(frequencies[:, 0] - frequencies[:, 1])
        uncertainty = np.sqrt(variance[:, 0] + variance[:, 1] + EPS)
        information = separation / uncertainty
    elif counts is not None:
        information = np.std(frequencies, axis=1) / np.sqrt(np.mean(variance, axis=1) + EPS)
    elif frequencies.shape[1] == 2:
        information = np.abs(frequencies[:, 0] - frequencies[:, 1])
    else:
        information = np.std(frequencies, axis=1)
    positive = information[information > 0.0]
    if len(positive) == 0:
        return np.ones(len(frequencies), dtype=np.float64)
    scale = np.median(positive)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = float(np.mean(positive))
    weights = information / max(scale, EPS)
    return np.clip(weights, 0.1, 2.0)


def _append_unknown_to_frequencies(
    frequencies: np.ndarray,
    *,
    mode: str,
    flat_frequency: float,
) -> np.ndarray:
    if mode == "flat":
        unknown = np.full(len(frequencies), flat_frequency, dtype=np.float64)
    elif mode == "midpoint":
        unknown = np.mean(frequencies, axis=1)
    elif mode == "shrink-midpoint":
        unknown = 0.5 * flat_frequency + 0.5 * np.mean(frequencies, axis=1)
    else:
        raise ValueError(f"Unknown ancestry mode {mode!r} is not supported.")
    unknown = np.clip(unknown, EPS, 1.0 - EPS)
    return np.column_stack((frequencies, unknown))


def add_unknown_state_to_chunks(
    chunks: list[InferenceChunk],
    *,
    mode: str,
    flat_frequency: float,
    label: str,
) -> list[InferenceChunk]:
    out: list[InferenceChunk] = []
    for chunk in chunks:
        out.append(
            InferenceChunk(
                chromosome=chunk.chromosome,
                positions=chunk.positions,
                frequencies=_append_unknown_to_frequencies(
                    chunk.frequencies,
                    mode=mode,
                    flat_frequency=flat_frequency,
                ),
                genotype_likelihoods=chunk.genotype_likelihoods,
                c_m_positions=chunk.c_m_positions,
                ploidy=chunk.ploidy,
                population_names=(*chunk.population_names, label),
                allele_counts=None if chunk.allele_counts is None else np.column_stack((
                    chunk.allele_counts,
                    np.mean(chunk.allele_counts, axis=1),
                )),
            )
        )
    return out


@dataclass(frozen=True)
class AdmixtureTimeResult:
    model: str                  # "single_pulse" or "multi_pulse"
    g_estimates: list[float]    # [g] or [g_recent, g_ancient] in generations
    g_std_errors: list[float]   # 1-sigma standard errors from covariance matrix
    weights: list[float]        # mixture weights, sum to 1
    amplitude: float            # fitted signal amplitude A
    baseline: float             # fitted background constant C
    delta_aic: float            # AIC_single - AIC_double; positive favours multi-pulse
    lags_morgan: np.ndarray
    acf: np.ndarray

    @property
    def weighted_g(self) -> float:
        """Weighted-mean generations since admixture across all components."""
        return sum(g * w for g, w in zip(self.g_estimates, self.weights))


def beta_smoothed_frequencies(
    derived_counts: np.ndarray,
    allele_counts: np.ndarray,
    *,
    alpha: float = 0.5,
    beta: float = 0.5,
) -> np.ndarray:
    """
    Convert reference-panel allele counts to frequencies with a Beta prior.

    Jeffreys smoothing, alpha=beta=0.5, is the default. Laplace smoothing is
    alpha=beta=1.0. This prevents exact 0 or 1 frequencies from finite panels.
    """
    derived_counts = np.asarray(derived_counts, dtype=np.float64)
    allele_counts = np.asarray(allele_counts, dtype=np.float64)

    if derived_counts.shape != allele_counts.shape:
        raise ValueError("derived_counts and allele_counts must have the same shape.")
    if np.any(allele_counts <= 0.0):
        raise ValueError("allele_counts must be positive.")
    if np.any((derived_counts < 0.0) | (derived_counts > allele_counts)):
        raise ValueError("derived_counts must be in [0, allele_counts].")
    if alpha <= 0.0 or beta <= 0.0:
        raise ValueError("alpha and beta must be positive.")

    return (derived_counts + alpha) / (allele_counts + alpha + beta)


def require_optional_module(module_name: str, package_hint: str):
    try:
        return __import__(module_name)
    except ModuleNotFoundError as exc:
        raise SystemExit(
            f"{module_name} is required for this input mode. "
            f"Install it with: micromamba install -c conda-forge {package_hint}"
        ) from exc


def configure_logging(verbose: bool = False) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s - TRACE - %(levelname)s - %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def progress_iter(iterable, *, total: int, desc: str):
    try:
        from tqdm import tqdm
    except ModuleNotFoundError:
        LOGGER.info("%s: %d tasks", desc, total)
        return iterable
    return tqdm(iterable, total=total, desc=desc)


def reverse_complement(allele: str) -> str:
    table = str.maketrans("ACGTNacgtn", "TGCANtgcan")
    return allele.translate(table)[::-1].upper()


def is_palindromic_pair(ref: str, alt: str) -> bool:
    pair = {ref.upper(), alt.upper()}
    return pair == {"A", "T"} or pair == {"C", "G"}


def add_reference_index_key(
    index: dict[tuple[int, str, str], int],
    position: int,
    ref: str,
    alt: str,
    row_index: int,
) -> None:
    ref = ref.upper()
    alt = alt.upper()
    index[(position, ref, alt)] = row_index
    if len(ref) == 1 and len(alt) == 1 and not is_palindromic_pair(ref, alt):
        index[(position, reverse_complement(ref), reverse_complement(alt))] = row_index


def shrink_frequencies_to_grand_mean(
    frequencies: np.ndarray,
    allele_counts: np.ndarray,
    *,
    strength: float,
) -> np.ndarray:
    """Empirical-Bayes shrinkage of per-population frequencies toward the
    cross-population mean, more strongly for smaller panels.

    For each site and population, ``f' = w*f + (1-w)*grand`` where
    ``grand`` is the mean frequency across populations at that site and
    ``w = n / (n + strength)`` with ``n`` the panel size. A small
    (bottlenecked) panel has ``w`` near 0, so its noisy/biased frequency is
    pulled toward the consensus; a large panel has ``w`` near 1 and is left
    essentially untouched. ``strength = 0`` disables shrinkage.

    This corrects the *mean* of an unreliable small-panel frequency, which the
    genotype-likelihood variance marginalization cannot do (it only widens the
    uncertainty around a possibly-biased point estimate).
    """
    if strength <= 0.0:
        return frequencies
    grand = frequencies.mean(axis=1, keepdims=True)
    w = allele_counts / (allele_counts + strength)
    shrunk = w * frequencies + (1.0 - w) * grand
    return np.clip(shrunk, EPS, 1.0 - EPS)


def read_reference_panel(
    path: str, *, alpha: float = 0.5, beta: float = 0.5, freq_shrinkage: float = 0.0
) -> ReferencePanel:
    """
    Read a tab-delimited reference panel table.

    Required columns: chrom, pos.
    Optional columns for BAM mode: ref, alt.

    Frequency format:
      chrom pos f_WHG f_EEF f_STEPPE

    Count format:
      chrom pos derived_WHG total_WHG derived_EEF total_EEF ...

    ``freq_shrinkage`` (count-format panels only) applies empirical-Bayes
    shrinkage of per-population frequencies toward the cross-population mean,
    scaled by panel size, to limit false ancestry from small/bottlenecked
    panels. See :func:`shrink_frequencies_to_grand_mean`.
    """
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("Reference panel table is missing a header.")

        fields = set(reader.fieldnames)
        if "chrom" not in fields or "pos" not in fields:
            raise ValueError("Reference panel table requires chrom and pos columns.")

        freq_cols = [name for name in reader.fieldnames if name.startswith("f_")]
        if freq_cols:
            population_names = tuple(name[2:] for name in freq_cols)
            count_pairs: list[tuple[str, str]] = []
        else:
            derived_cols = [name for name in reader.fieldnames if name.startswith("derived_")]
            population_names = tuple(name.removeprefix("derived_") for name in derived_cols)
            count_pairs = [
                (f"derived_{pop}", f"total_{pop}")
                for pop in population_names
                if f"total_{pop}" in fields
            ]
            if len(count_pairs) != len(population_names):
                raise ValueError(
                    "Count-format reference panels require derived_POP and total_POP "
                    "columns for every population."
                )

        if len(population_names) < 2:
            raise ValueError("Reference panel must contain at least two populations.")

        chromosomes: list[str] = []
        positions: list[int] = []
        refs: list[str] = []
        alts: list[str] = []
        frequency_rows: list[list[float]] = []
        allele_count_rows: list[list[float]] | None = [] if count_pairs else None

        for row in reader:
            chromosomes.append(row["chrom"])
            positions.append(int(row["pos"]))
            refs.append(row.get("ref", ""))
            alts.append(row.get("alt", ""))

            if freq_cols:
                frequency_rows.append([float(row[col]) for col in freq_cols])
            else:
                derived = np.array([float(row[d_col]) for d_col, _ in count_pairs])
                totals = np.array([float(row[t_col]) for _, t_col in count_pairs])
                frequency_rows.append(
                    beta_smoothed_frequencies(derived, totals, alpha=alpha, beta=beta).tolist()
                )
                if allele_count_rows is not None:
                    allele_count_rows.append(totals.tolist())

    frequencies = np.asarray(frequency_rows, dtype=np.float64)
    allele_counts = None if allele_count_rows is None else np.asarray(allele_count_rows, dtype=np.float64)
    if freq_shrinkage > 0.0 and allele_counts is not None:
        frequencies = shrink_frequencies_to_grand_mean(
            frequencies, allele_counts, strength=freq_shrinkage
        )
    if np.any((frequencies <= 0.0) | (frequencies >= 1.0)):
        raise ValueError(
            "Reference frequencies must be strictly inside (0, 1). "
            "Use count-format input or pre-smoothed f_POP columns."
        )

    refs_array = np.asarray(refs, dtype=object)
    alts_array = np.asarray(alts, dtype=object)
    if not np.any(refs_array) or not np.any(alts_array):
        refs_array = None
        alts_array = None

    order = np.lexsort((np.asarray(positions), np.asarray(chromosomes, dtype=str)))
    return ReferencePanel(
        chromosomes=np.asarray(chromosomes, dtype=str)[order],
        positions=np.asarray(positions, dtype=np.int64)[order],
        frequencies=frequencies[order],
        population_names=population_names,
        allele_counts=None if allele_counts is None else allele_counts[order],
        refs=None if refs_array is None else refs_array[order],
        alts=None if alts_array is None else alts_array[order],
    )


def write_reference_panel_zarr(
    panel: ReferencePanel,
    output_path: str,
    *,
    chunk_rows: int = 100_000,
) -> None:
    zarr = require_optional_module("zarr", "zarr")
    root = zarr.open_group(output_path, mode="w")
    n_rows, n_populations = panel.frequencies.shape
    max_chrom_len = max(1, max(len(str(chrom)) for chrom in panel.chromosomes))
    root.create_dataset("chrom", data=panel.chromosomes.astype(f"U{max_chrom_len}"), chunks=(min(chunk_rows, n_rows),))
    root.create_dataset("pos", data=panel.positions.astype(np.int64), chunks=(min(chunk_rows, n_rows),))
    root.create_dataset(
        "frequencies",
        data=panel.frequencies.astype(np.float32),
        chunks=(min(chunk_rows, n_rows), n_populations),
    )
    if panel.allele_counts is not None:
        root.create_dataset(
            "allele_counts",
            data=panel.allele_counts.astype(np.float32),
            chunks=(min(chunk_rows, n_rows), n_populations),
        )
    if panel.refs is not None and panel.alts is not None:
        root.create_dataset("ref", data=panel.refs.astype("U64"), chunks=(min(chunk_rows, n_rows),))
        root.create_dataset("alt", data=panel.alts.astype("U64"), chunks=(min(chunk_rows, n_rows),))
    root.attrs["population_names"] = list(panel.population_names)


def convert_reference_tsv_to_zarr(
    input_path: str,
    output_path: str,
    *,
    alpha: float,
    beta: float,
    chunk_rows: int,
) -> None:
    panel = read_reference_panel(input_path, alpha=alpha, beta=beta)
    write_reference_panel_zarr(panel, output_path, chunk_rows=chunk_rows)


def read_reference_panel_zarr_slice(root, start: int, end: int) -> ReferencePanel:
    population_names = tuple(root.attrs["population_names"])
    refs = np.asarray(root["ref"][start:end], dtype=object) if "ref" in root else None
    alts = np.asarray(root["alt"][start:end], dtype=object) if "alt" in root else None
    allele_counts = np.asarray(root["allele_counts"][start:end], dtype=np.float64) if "allele_counts" in root else None
    return ReferencePanel(
        chromosomes=np.asarray(root["chrom"][start:end], dtype=str),
        positions=np.asarray(root["pos"][start:end], dtype=np.int64),
        frequencies=np.asarray(root["frequencies"][start:end], dtype=np.float64),
        population_names=population_names,
        allele_counts=allele_counts,
        refs=refs,
        alts=alts,
    )


def reference_zarr_population_names(path: str) -> tuple[str, ...]:
    zarr = require_optional_module("zarr", "zarr")
    root = zarr.open_group(path, mode="r")
    return tuple(root.attrs["population_names"])


def read_genetic_map(path: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Read a genetic map and return chrom -> (bp_positions, cM_positions).

    Supported:
      - PLINK .map without header: chrom marker cM bp
      - Header tables with chrom/chr, pos/bp/position, and cm/cM/genetic_position
    """
    map_rows: dict[str, list[tuple[int, float]]] = {}
    with open(path, encoding="utf-8") as handle:
        first = handle.readline().strip().split()
        if not first:
            raise ValueError("Genetic map is empty.")
        handle.seek(0)

        has_header = any(token.lower() in {"chrom", "chr", "pos", "position", "bp", "cm", "c_m"} for token in first)
        if has_header:
            header = handle.readline().strip().split()
            lower = {name.lower(): idx for idx, name in enumerate(header)}
            chrom_idx = next((lower[key] for key in ("chrom", "chr", "chromosome") if key in lower), None)
            pos_idx = next((lower[key] for key in ("pos", "position", "bp") if key in lower), None)
            cm_idx = next((lower[key] for key in ("cm", "c_m", "genetic_position") if key in lower), None)
            if chrom_idx is None or pos_idx is None or cm_idx is None:
                raise ValueError("Header genetic map needs chrom, pos/bp, and cM columns.")
            for line in handle:
                parts = line.strip().split()
                if len(parts) <= max(chrom_idx, pos_idx, cm_idx):
                    continue
                map_rows.setdefault(parts[chrom_idx], []).append(
                    (int(parts[pos_idx]), float(parts[cm_idx]))
                )
        else:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 4:
                    continue
                chrom, _marker, c_m, bp = parts[:4]
                map_rows.setdefault(chrom, []).append((int(bp), float(c_m)))

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for chrom, rows in map_rows.items():
        rows.sort(key=lambda item: item[0])
        bp = np.asarray([row[0] for row in rows], dtype=np.float64)
        cm = np.asarray([row[1] for row in rows], dtype=np.float64)
        out[chrom] = (bp, cm)
    return out


def interpolate_cm(
    chromosome: str,
    positions: np.ndarray,
    genetic_map: dict[str, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    if chromosome not in genetic_map:
        raise ValueError(f"No genetic map entries found for chromosome {chromosome}.")
    bp, cm = genetic_map[chromosome]
    if len(bp) < 2:
        raise ValueError(f"Chromosome {chromosome} genetic map needs at least two rows.")
    positions_f = positions.astype(np.float64)
    result = np.interp(positions_f, bp, cm)
    # Linear extrapolation beyond map boundaries using the terminal recombination rate,
    # avoiding the flat-extrapolation artifact that produces delta_cM = 0 at telomeres.
    left_rate = (cm[1] - cm[0]) / max(float(bp[1] - bp[0]), 1.0)
    left_mask = positions_f < bp[0]
    result[left_mask] = cm[0] + left_rate * (positions_f[left_mask] - bp[0])
    right_rate = (cm[-1] - cm[-2]) / max(float(bp[-1] - bp[-2]), 1.0)
    right_mask = positions_f > bp[-1]
    result[right_mask] = cm[-1] + right_rate * (positions_f[right_mask] - bp[-1])
    return result


def reference_index_for_chrom(panel: ReferencePanel, chromosome: str) -> dict[tuple[int, str, str], int]:
    mask = panel.chromosomes == chromosome
    if panel.refs is None or panel.alts is None:
        return {(int(pos), "", ""): idx for idx, pos in zip(np.flatnonzero(mask), panel.positions[mask])}
    index: dict[tuple[int, str, str], int] = {}
    for idx in np.flatnonzero(mask):
        add_reference_index_key(
            index,
            int(panel.positions[idx]),
            str(panel.refs[idx]),
            str(panel.alts[idx]),
            int(idx),
        )
    return index


def chrom_ploidy(
    chromosome: str,
    *,
    sample_sex: str,
    haploid_chroms: set[str],
) -> int:
    chrom = chromosome.removeprefix("chr").upper()
    if chromosome in haploid_chroms or chrom in haploid_chroms:
        return 1
    if sample_sex == "male" and chrom in {"X", "Y", "M", "MT"}:
        return 1
    return 2


def gls_from_vcf(
    vcf_path: str,
    sample: str | None,
    *,
    sample_sex: str,
    haploid_chroms: set[str],
) -> Iterable[tuple[str, int, str, str, np.ndarray, int]]:
    cyvcf2 = require_optional_module("cyvcf2", "cyvcf2")
    vcf = cyvcf2.VCF(vcf_path)
    sample_index = 0
    if sample is not None:
        if sample not in vcf.samples:
            raise ValueError(f"Sample {sample!r} not found in VCF.")
        sample_index = vcf.samples.index(sample)

    for variant in vcf:
        if len(variant.ALT or []) != 1:
            continue
        ploidy = chrom_ploidy(
            variant.CHROM,
            sample_sex=sample_sex,
            haploid_chroms=haploid_chroms,
        )
        gl = None
        try:
            gl_field = variant.format("GL")
        except (KeyError, ValueError):
            gl_field = None
        if gl_field is not None:
            gl = np.power(10.0, np.asarray(gl_field[sample_index, :3], dtype=np.float64))
        else:
            try:
                pl_field = variant.format("PL")
            except (KeyError, ValueError):
                pl_field = None
            if pl_field is not None:
                gl = np.power(10.0, -np.asarray(pl_field[sample_index, :3], dtype=np.float64) / 10.0)
        if gl is None:
            gt = variant.genotypes[sample_index]
            alleles = [int(allele) for allele in gt[:-1] if int(allele) >= 0]
            gl = genotype_likelihood_from_gt(alleles, ploidy)
        yield variant.CHROM, int(variant.POS), variant.REF, variant.ALT[0], gl, ploidy


def gls_from_bam(
    bam_path: str,
    panel: ReferencePanel,
    chromosome: str,
    positions: np.ndarray,
    *,
    base_error: float = 0.01,
    min_base_quality: int = 20,
    sample_sex: str = "unknown",
    haploid_chroms: set[str] | None = None,
    fasta_path: str | None = None,
) -> Iterable[tuple[str, int, str, str, np.ndarray, int]]:
    if panel.refs is None or panel.alts is None:
        raise ValueError("BAM mode requires ref and alt columns in the reference panel.")

    pysam = require_optional_module("pysam", "pysam")
    bam = pysam.AlignmentFile(bam_path, "rb", reference_filename=fasta_path)
    haploid_chroms = haploid_chroms or set()
    ref_alt_by_pos = {
        int(panel.positions[idx]): (str(panel.refs[idx]).upper(), str(panel.alts[idx]).upper())
        for idx in np.flatnonzero(panel.chromosomes == chromosome)
    }
    ploidy = chrom_ploidy(
        chromosome,
        sample_sex=sample_sex,
        haploid_chroms=haploid_chroms,
    )

    for pos in sorted(set(int(position) for position in positions)):
        if pos not in ref_alt_by_pos:
            continue
        ref, alt = ref_alt_by_pos[pos]
        original_observed: list[bool] = []
        original_errors: list[float] = []
        flipped_observed: list[bool] = []
        flipped_errors: list[float] = []
        rc_ref = reverse_complement(ref)
        rc_alt = reverse_complement(alt)
        pos0 = pos - 1

        for pileup_column in bam.pileup(
            chromosome,
            pos0,
            pos0 + 1,
            truncate=True,
            stepper="all",
        ):
            if int(pileup_column.reference_pos) != pos0:
                continue
            for pileup_read in pileup_column.pileups:
                if pileup_read.is_del or pileup_read.is_refskip:
                    continue
                query_pos = pileup_read.query_position
                if query_pos is None:
                    continue
                read = pileup_read.alignment
                quality = (
                    read.query_qualities[query_pos]
                    if read.query_qualities is not None
                    else None
                )
                if quality is not None and quality < min_base_quality:
                    continue
                base = read.query_sequence[query_pos].upper()
                error_prob = 10.0 ** (-quality / 10.0) if quality is not None else base_error
                if base in {ref, alt}:
                    original_observed.append(base == alt)
                    original_errors.append(error_prob)
                if not is_palindromic_pair(ref, alt) and base in {rc_ref, rc_alt}:
                    flipped_observed.append(base == rc_alt)
                    flipped_errors.append(error_prob)

        if len(flipped_observed) > len(original_observed):
            observed_is_alt = flipped_observed
            error_probs = flipped_errors
        else:
            observed_is_alt = original_observed
            error_probs = original_errors

        gl = gl_from_base_observations(
            np.asarray(observed_is_alt, dtype=bool),
            np.asarray(error_probs, dtype=np.float64),
            fallback_error=base_error,
        )
        yield chromosome, pos, ref, alt, gl, ploidy

    bam.close()


def gl_from_base_observations(
    observed_is_alt: np.ndarray,
    error_probs: np.ndarray,
    *,
    fallback_error: float = 0.01,
) -> np.ndarray:
    if not 0.0 < fallback_error < 0.5:
        raise ValueError("fallback_error must be in (0, 0.5).")
    if len(observed_is_alt) == 0:
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)

    errors = np.asarray(error_probs, dtype=np.float64)
    if errors.shape != observed_is_alt.shape:
        raise ValueError("observed_is_alt and error_probs must have the same shape.")
    errors = np.clip(np.where(np.isfinite(errors), errors, fallback_error), EPS, 0.5 - EPS)

    p_alt_given_g = np.column_stack(
        (
            errors,
            np.full_like(errors, 0.5),
            1.0 - errors,
        )
    )
    probs = np.where(observed_is_alt[:, None], p_alt_given_g, 1.0 - p_alt_given_g)
    log_likelihood = np.sum(np.log(np.clip(probs, EPS, None)), axis=0)
    log_likelihood -= np.max(log_likelihood)
    gl = np.exp(log_likelihood)
    return gl / np.sum(gl)


def holdout_log_likelihood(
    phi: np.ndarray,
    frequencies: np.ndarray,
    genotype_likelihoods: np.ndarray,
    ploidy: np.ndarray,
    mask: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    if phi.ndim == 1:
        p = phi * frequencies[:, 0] + (1.0 - phi) * frequencies[:, 1]
    else:
        p = np.sum(phi * frequencies, axis=1)
    likelihood, _d_likelihood_dp = TraceSolver._likelihood_and_derivative(
        p,
        genotype_likelihoods,
        ploidy,
    )
    log_likelihood = np.log(likelihood[mask])
    if weights is None:
        return float(np.sum(log_likelihood))
    mask_weights = np.asarray(weights, dtype=np.float64)[mask]
    denom = float(np.sum(mask_weights))
    if denom <= 0.0:
        return float(np.mean(log_likelihood))
    return float(np.sum(mask_weights * log_likelihood) / denom)


def cm_site_weights(c_m_positions: np.ndarray) -> np.ndarray:
    """Trapezoidal per-SNP cM weights, normalized to sum to one."""
    c_m_positions = np.asarray(c_m_positions, dtype=np.float64)
    if len(c_m_positions) == 0:
        return np.array([], dtype=np.float64)
    if len(c_m_positions) == 1:
        return np.ones(1, dtype=np.float64)
    diffs = np.clip(np.diff(c_m_positions), 0.0, None)
    weights = np.empty(len(c_m_positions), dtype=np.float64)
    weights[0] = diffs[0] / 2.0
    weights[-1] = diffs[-1] / 2.0
    weights[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(len(c_m_positions), 1.0 / len(c_m_positions), dtype=np.float64)
    return weights / total


def make_calibration_mask(
    c_m_positions: np.ndarray,
    *,
    mask_fraction: float,
    seed: int,
    mode: str,
    block_cm: float,
) -> np.ndarray:
    if not 0.0 < mask_fraction < 1.0:
        raise ValueError("mask_fraction must be in (0, 1).")
    if mode not in {"snp", "block"}:
        raise ValueError("calibration mask mode must be 'snp' or 'block'.")
    rng = np.random.default_rng(seed)
    n_sites = len(c_m_positions)
    if n_sites == 0:
        return np.array([], dtype=bool)
    if mode == "snp":
        mask = rng.random(n_sites) < mask_fraction
        if not np.any(mask):
            mask[rng.integers(0, n_sites)] = True
        return mask

    c_m_positions = np.asarray(c_m_positions, dtype=np.float64)
    weights = cm_site_weights(c_m_positions)
    span = float(np.max(c_m_positions) - np.min(c_m_positions))
    effective_block_cm = block_cm if block_cm > 0.0 else max(span * mask_fraction, MIN_CM_DELTA)
    mask = np.zeros(n_sites, dtype=bool)
    attempts = 0
    while float(np.sum(weights[mask])) < mask_fraction and attempts < max(20, n_sites * 2):
        start_idx = int(rng.integers(0, n_sites))
        start_cm = float(c_m_positions[start_idx])
        end_cm = start_cm + effective_block_cm
        block_mask = (c_m_positions >= start_cm) & (c_m_positions <= end_cm)
        if not np.any(block_mask):
            block_mask[start_idx] = True
        mask |= block_mask
        attempts += 1
    if not np.any(mask):
        mask[rng.integers(0, n_sites)] = True
    return mask


def calibrate_d_for_chunk(
    chunk: InferenceChunk,
    *,
    d_grid: list[float],
    mask_fraction: float,
    seed: int,
    calibration_score: str,
    calibration_mask_mode: str,
    calibration_block_cm: float,
    logit_l2: float,
    gl_weight: float,
    prior_weight: float,
    prior_center: float | None,
    maxiter: int,
) -> tuple[float, list[tuple[float, float]]]:
    if not d_grid:
        raise ValueError("d_grid cannot be empty.")
    if calibration_score not in {"likelihood", "cm-likelihood"}:
        raise ValueError("calibration_score must be 'likelihood' or 'cm-likelihood'.")

    mask = make_calibration_mask(
        chunk.c_m_positions,
        mask_fraction=mask_fraction,
        seed=seed,
        mode=calibration_mask_mode,
        block_cm=calibration_block_cm,
    )
    score_weights = cm_site_weights(chunk.c_m_positions) if calibration_score == "cm-likelihood" else None

    train_gls = np.array(chunk.genotype_likelihoods, copy=True)
    train_gls[mask] = 1.0

    scores: list[tuple[float, float]] = []
    best_d = float(d_grid[0])
    best_score = -np.inf

    for d_value in d_grid:
        if chunk.frequencies.shape[1] == 2:
            solver = TraceSolver(
                d_coupling=d_value,
                gl_weight=gl_weight,
                prior_weight=prior_weight,
                prior_center=prior_center,
                maxiter=maxiter,
            )
            result = solver.solve(
                chunk.frequencies[:, 0],
                chunk.frequencies[:, 1],
                train_gls,
                chunk.c_m_positions,
                ploidy=chunk.ploidy,
            )
            phi_for_score = np.column_stack((result.phi, 1.0 - result.phi))
        else:
            solver = TraceMultidimensionalSolver(
                d_coupling=d_value,
                gl_weight=gl_weight,
                logit_l2=logit_l2,
                maxiter=maxiter,
            )
            result = solver.solve(
                chunk.frequencies,
                train_gls,
                chunk.c_m_positions,
                ploidy=chunk.ploidy,
            )
            phi_for_score = result.phi

        score = holdout_log_likelihood(
            phi_for_score,
            chunk.frequencies,
            chunk.genotype_likelihoods,
            chunk.ploidy,
            mask,
            weights=score_weights,
        )
        scores.append((float(d_value), score))
        if score > best_score:
            best_score = score
            best_d = float(d_value)

    return best_d, scores


def calibrate_params_for_chunk(
    chunk: InferenceChunk,
    *,
    d_grid: list[float],
    gl_weight_grid: list[float],
    robust_cap_grid: list[float],
    mask_fraction: float,
    seed: int,
    calibration_score: str,
    calibration_mask_mode: str,
    calibration_block_cm: float,
    logit_l2: float,
    prior_weight: float,
    prior_center: float | None,
    site_weighting: bool,
    posterior_init: bool,
    init_smooth_window: int,
    unknown_penalty: float,
    maxiter: int,
) -> tuple[float, float, float, list[tuple[float, float, float, float]]]:
    if not d_grid or not gl_weight_grid or not robust_cap_grid:
        raise ValueError("Parameter grids cannot be empty.")
    if calibration_score not in {"likelihood", "cm-likelihood"}:
        raise ValueError("calibration_score must be 'likelihood' or 'cm-likelihood'.")
    if any(value <= 0.0 for value in gl_weight_grid):
        raise ValueError("GL-weight grid values must be positive.")
    if any(value < 0.0 for value in robust_cap_grid):
        raise ValueError("Robust-cap grid values must be non-negative.")

    mask = make_calibration_mask(
        chunk.c_m_positions,
        mask_fraction=mask_fraction,
        seed=seed,
        mode=calibration_mask_mode,
        block_cm=calibration_block_cm,
    )
    score_weights = cm_site_weights(chunk.c_m_positions) if calibration_score == "cm-likelihood" else None

    train_gls = np.array(chunk.genotype_likelihoods, copy=True)
    train_gls[mask] = 1.0
    site_weights = (
        frequency_site_weights(chunk.frequencies, chunk.allele_counts)
        if site_weighting
        else None
    )

    scores: list[tuple[float, float, float, float]] = []
    best_d = float(d_grid[0])
    best_gl_weight = float(gl_weight_grid[0])
    best_cap = float(robust_cap_grid[0])
    best_score = -np.inf

    for d_value in d_grid:
        for gl_value in gl_weight_grid:
            for cap_value in robust_cap_grid:
                if chunk.frequencies.shape[1] == 2:
                    solver = TraceSolver(
                        d_coupling=d_value,
                        gl_weight=gl_value,
                        prior_weight=prior_weight,
                        prior_center=prior_center,
                        site_weights=site_weights,
                        robust_likelihood_cap=cap_value,
                        maxiter=maxiter,
                    )
                    f_a = chunk.frequencies[:, 0]
                    f_b = chunk.frequencies[:, 1]
                    initial_phi = (
                        posterior_initial_phi(
                            f_a,
                            f_b,
                            train_gls,
                            chunk.ploidy,
                            smooth_window=init_smooth_window,
                        )
                        if posterior_init
                        else None
                    )
                    result = solver.solve(
                        f_a,
                        f_b,
                        train_gls,
                        chunk.c_m_positions,
                        ploidy=chunk.ploidy,
                        initial_phi=initial_phi,
                    )
                    phi_for_score = np.column_stack((result.phi, 1.0 - result.phi))
                else:
                    solver = TraceMultidimensionalSolver(
                        d_coupling=d_value,
                        gl_weight=gl_value,
                        logit_l2=logit_l2,
                        prior_weight=prior_weight,
                        prior_center=None,
                        unknown_index=(chunk.frequencies.shape[1] - 1 if unknown_penalty > 0.0 else None),
                        unknown_penalty=unknown_penalty,
                        site_weights=site_weights,
                        robust_likelihood_cap=cap_value,
                        maxiter=maxiter,
                    )
                    initial_logits = (
                        posterior_initial_logits(
                            chunk.frequencies,
                            train_gls,
                            chunk.ploidy,
                            smooth_window=init_smooth_window,
                        )
                        if posterior_init
                        else None
                    )
                    result = solver.solve(
                        chunk.frequencies,
                        train_gls,
                        chunk.c_m_positions,
                        ploidy=chunk.ploidy,
                        initial_logits=initial_logits,
                    )
                    phi_for_score = result.phi

                score = holdout_log_likelihood(
                    phi_for_score,
                    chunk.frequencies,
                    chunk.genotype_likelihoods,
                    chunk.ploidy,
                    mask,
                    weights=score_weights,
                )
                scores.append((float(d_value), float(gl_value), float(cap_value), score))
                if score > best_score:
                    best_score = score
                    best_d = float(d_value)
                    best_gl_weight = float(gl_value)
                    best_cap = float(cap_value)

    return best_d, best_gl_weight, best_cap, scores


def parse_d_grid(value: str) -> list[float]:
    grid = [float(part) for part in value.split(",") if part.strip()]
    if any(item < 0.0 for item in grid):
        raise ValueError("D grid values must be non-negative.")
    return grid


def parse_positive_grid(value: str, *, name: str, allow_zero: bool = False) -> list[float]:
    grid = [float(part) for part in value.split(",") if part.strip()]
    if not grid:
        raise ValueError(f"{name} cannot be empty.")
    if allow_zero:
        bad = any(item < 0.0 for item in grid)
        requirement = "non-negative"
    else:
        bad = any(item <= 0.0 for item in grid)
        requirement = "positive"
    if bad:
        raise ValueError(f"{name} values must be {requirement}.")
    return grid


def inverse_tridiagonal_diag(
    diagonal: np.ndarray,
    off_diagonal: np.ndarray,
    *,
    jitter: float = 1e-8,
) -> np.ndarray:
    """
    Return diag(inv(H)) for a symmetric positive tridiagonal H in O(N).

    Uses an LDL' factorization. If local non-convex likelihood curvature makes
    H nearly indefinite, a small diagonal jitter is increased until the
    recurrence is positive. This keeps the reported uncertainty conservative
    rather than exploding from numerical sign flips.
    """
    diagonal = np.asarray(diagonal, dtype=np.float64)
    off_diagonal = np.asarray(off_diagonal, dtype=np.float64)
    n = len(diagonal)
    if off_diagonal.shape != (max(0, n - 1),):
        raise ValueError("off_diagonal must have length N - 1.")
    if n == 0:
        return np.array([], dtype=np.float64)

    extra = jitter
    for _attempt in range(8):
        d_factor = np.empty(n, dtype=np.float64)
        l_factor = np.empty(max(0, n - 1), dtype=np.float64)
        d_factor[0] = diagonal[0] + extra
        ok = np.isfinite(d_factor[0]) and d_factor[0] > EPS
        if ok:
            for i in range(n - 1):
                l_factor[i] = off_diagonal[i] / d_factor[i]
                d_factor[i + 1] = diagonal[i + 1] + extra - l_factor[i] * off_diagonal[i]
                if not np.isfinite(d_factor[i + 1]) or d_factor[i + 1] <= EPS:
                    ok = False
                    break
        if ok:
            inv_diag = np.empty(n, dtype=np.float64)
            inv_diag[-1] = 1.0 / d_factor[-1]
            for i in range(n - 2, -1, -1):
                inv_diag[i] = 1.0 / d_factor[i] + l_factor[i] * l_factor[i] * inv_diag[i + 1]
            return np.clip(inv_diag, 0.0, None)
        extra *= 10.0

    return 1.0 / np.clip(diagonal + extra, EPS, None)


def gl_from_allele_counts(ref_count: int, alt_count: int, *, base_error: float = 0.01) -> np.ndarray:
    if not 0.0 < base_error < 0.5:
        raise ValueError("base_error must be in (0, 0.5).")
    counts = np.array([ref_count, alt_count], dtype=np.float64)
    if np.sum(counts) == 0.0:
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)

    p_alt_given_g = np.array([base_error, 0.5, 1.0 - base_error], dtype=np.float64)
    log_likelihood = (
        alt_count * np.log(p_alt_given_g)
        + ref_count * np.log(1.0 - p_alt_given_g)
    )
    log_likelihood -= np.max(log_likelihood)
    return np.exp(log_likelihood)


def build_chunks_from_records(
    records: Iterable[tuple[str, int, str, str, np.ndarray, int]],
    panel: ReferencePanel,
    genetic_map: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    chunk_size: int,
    chunk_overlap: int,
    max_imputation_gap_cm: float,
) -> list[InferenceChunk]:
    if chunk_size < 2:
        raise ValueError("chunk_size must be at least 2.")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative.")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    if max_imputation_gap_cm <= 0.0:
        raise ValueError("max_imputation_gap_cm must be positive.")

    by_chrom: dict[str, list[tuple[int, np.ndarray, np.ndarray | None, np.ndarray, int]]] = {}
    indices_by_chrom: dict[str, dict[tuple[int, str, str], int]] = {}

    for chrom, pos, ref, alt, gl, ploidy in records:
        if chrom not in indices_by_chrom:
            indices_by_chrom[chrom] = reference_index_for_chrom(panel, chrom)
        index = indices_by_chrom[chrom].get((pos, ref, alt))
        if index is None:
            index = indices_by_chrom[chrom].get((pos, "", ""))
        if index is None:
            continue

        allele_counts = None if panel.allele_counts is None else panel.allele_counts[index]
        by_chrom.setdefault(chrom, []).append(
            (pos, panel.frequencies[index], allele_counts, np.asarray(gl, dtype=np.float64), ploidy)
        )

    chunks: list[InferenceChunk] = []
    step = chunk_size - chunk_overlap
    for chrom, rows in by_chrom.items():
        rows.sort(key=lambda row: row[0])
        positions = np.asarray([row[0] for row in rows], dtype=np.int64)
        frequencies = np.asarray([row[1] for row in rows], dtype=np.float64)
        allele_counts = (
            None
            if rows[0][2] is None
            else np.asarray([row[2] for row in rows], dtype=np.float64)
        )
        gls = np.asarray([row[3] for row in rows], dtype=np.float64)
        ploidy = np.asarray([row[4] for row in rows], dtype=np.int8)
        cm_positions = interpolate_cm(chrom, positions, genetic_map)

        split_points = np.flatnonzero(np.diff(cm_positions) > max_imputation_gap_cm) + 1
        segment_starts = np.concatenate(([0], split_points))
        segment_ends = np.concatenate((split_points, [len(positions)]))

        for seg_start, seg_end in zip(segment_starts, segment_ends):
            if seg_end - seg_start < 2:
                continue
            start = seg_start
            while start < seg_end:
                end = min(seg_end, start + chunk_size)
                if end - start < 2:
                    break
                chunks.append(
                    InferenceChunk(
                        chromosome=chrom,
                        positions=positions[start:end],
                        frequencies=frequencies[start:end],
                        genotype_likelihoods=gls[start:end],
                        c_m_positions=cm_positions[start:end],
                        ploidy=ploidy[start:end],
                        population_names=panel.population_names,
                        allele_counts=None if allele_counts is None else allele_counts[start:end],
                    )
                )
                if end == seg_end:
                    break
                start += step
    return chunks


def build_chunks_from_vcf(
    vcf_path: str,
    panel: ReferencePanel,
    genetic_map: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    sample: str | None,
    chunk_size: int,
    chunk_overlap: int,
    max_imputation_gap_cm: float,
    sample_sex: str,
    haploid_chroms: set[str],
) -> list[InferenceChunk]:
    return build_chunks_from_records(
        gls_from_vcf(
            vcf_path,
            sample,
            sample_sex=sample_sex,
            haploid_chroms=haploid_chroms,
        ),
        panel,
        genetic_map,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        max_imputation_gap_cm=max_imputation_gap_cm,
    )


def build_chunks_from_bam(
    bam_path: str,
    panel: ReferencePanel,
    genetic_map: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    chunk_size: int,
    base_error: float,
    min_base_quality: int,
    chunk_overlap: int,
    max_imputation_gap_cm: float,
    sample_sex: str,
    haploid_chroms: set[str],
    fasta_path: str | None,
) -> list[InferenceChunk]:
    chunks: list[InferenceChunk] = []
    for chrom in np.unique(panel.chromosomes):
        positions = panel.positions[panel.chromosomes == chrom]
        if len(positions) < 2:
            continue
        records = gls_from_bam(
            bam_path,
            panel,
            str(chrom),
            positions,
            base_error=base_error,
            min_base_quality=min_base_quality,
            sample_sex=sample_sex,
            haploid_chroms=haploid_chroms,
            fasta_path=fasta_path,
        )
        chunks.extend(
            build_chunks_from_records(
                records,
                panel,
                genetic_map,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_imputation_gap_cm=max_imputation_gap_cm,
            )
        )
    return chunks


def gls_from_vcf_panel_slice(
    vcf_path: str,
    panel: ReferencePanel,
    *,
    sample: str | None,
    sample_sex: str,
    haploid_chroms: set[str],
    hap_index: int | None = None,
) -> Iterable[tuple[str, int, str, str, np.ndarray, int]]:
    cyvcf2 = require_optional_module("cyvcf2", "cyvcf2")
    if len(np.unique(panel.chromosomes)) != 1:
        raise ValueError("VCF panel slices must contain exactly one chromosome.")

    chrom = str(panel.chromosomes[0])
    sample_index = 0
    vcf = cyvcf2.VCF(vcf_path)
    if sample is not None:
        if sample not in vcf.samples:
            raise ValueError(f"Sample {sample!r} not found in VCF.")
        sample_index = vcf.samples.index(sample)

    gl_by_key: dict[tuple[int, str, str], np.ndarray] = {}
    region = f"{chrom}:{int(panel.positions[0])}-{int(panel.positions[-1])}"
    has_index = any(
        os.path.exists(candidate)
        for candidate in (
            f"{vcf_path}.tbi",
            f"{vcf_path}.csi",
            vcf_path.removesuffix(".gz") + ".tbi",
            vcf_path.removesuffix(".gz") + ".csi",
        )
    )
    if has_index:
        iterator = vcf(region)
    else:
        LOGGER.warning("VCF is not indexed or region query failed; scanning VCF for %s", region)
        iterator = vcf

    for variant in iterator:
        if variant.CHROM != chrom or len(variant.ALT or []) != 1:
            continue
        pos = int(variant.POS)
        if pos < panel.positions[0] or pos > panel.positions[-1]:
            continue
        ploidy = chrom_ploidy(
            variant.CHROM,
            sample_sex=sample_sex,
            haploid_chroms=haploid_chroms,
        )
        gl = None
        try:
            gl_field = variant.format("GL")
        except (KeyError, ValueError):
            gl_field = None
        if gl_field is not None:
            gl = np.power(10.0, np.asarray(gl_field[sample_index, :3], dtype=np.float64))
        else:
            try:
                pl_field = variant.format("PL")
            except (KeyError, ValueError):
                pl_field = None
            if pl_field is not None:
                gl = np.power(10.0, -np.asarray(pl_field[sample_index, :3], dtype=np.float64) / 10.0)
        if gl is None or hap_index is not None:
            gt = variant.genotypes[sample_index]
            alleles = [int(allele) for allele in gt[:-1] if int(allele) >= 0]
            if hap_index is not None:
                # Extract a single phased allele and treat the site as haploid
                allele = alleles[min(hap_index, len(alleles) - 1)] if alleles else 0
                gl = genotype_likelihood_from_gt([allele], ploidy=1)
                ploidy = 1
            elif gl is None:
                gl = genotype_likelihood_from_gt(alleles, ploidy)
        gl_by_key[(pos, variant.REF.upper(), variant.ALT[0].upper())] = (gl, ploidy)
        if not is_palindromic_pair(variant.REF, variant.ALT[0]):
            gl_by_key[
                (pos, reverse_complement(variant.REF), reverse_complement(variant.ALT[0]))
            ] = (gl, ploidy)

    vcf.close()

    uninformative = np.array([1.0, 1.0, 1.0], dtype=np.float64)
    for idx, pos in enumerate(panel.positions):
        ref = "" if panel.refs is None else str(panel.refs[idx]).upper()
        alt = "" if panel.alts is None else str(panel.alts[idx]).upper()
        site_ploidy = chrom_ploidy(chrom, sample_sex=sample_sex, haploid_chroms=haploid_chroms)
        if hap_index is not None:
            site_ploidy = 1
        entry = gl_by_key.get((int(pos), ref, alt))
        gl = entry[0] if entry is not None else uninformative
        yield chrom, int(pos), ref, alt, gl, site_ploidy


def build_chunks_from_reference_zarr(
    zarr_path: str,
    genetic_map: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    vcf_path: str | None,
    bam_path: str | None,
    sample: str | None,
    chunk_size: int,
    chunk_overlap: int,
    max_imputation_gap_cm: float,
    base_error: float,
    min_base_quality: int,
    sample_sex: str,
    haploid_chroms: set[str],
    fasta_path: str | None,
    hap_index: int | None = None,
) -> list[InferenceChunk]:
    if (vcf_path is None) == (bam_path is None):
        raise ValueError("Provide exactly one of --vcf or --bam.")
    zarr = require_optional_module("zarr", "zarr")
    root = zarr.open_group(zarr_path, mode="r")
    if "chrom" not in root or "pos" not in root or "frequencies" not in root:
        raise ValueError("Reference Zarr requires chrom, pos, and frequencies arrays.")
    if "population_names" not in root.attrs:
        raise ValueError("Reference Zarr requires population_names metadata.")

    chroms = np.asarray(root["chrom"][:], dtype=str)
    positions = np.asarray(root["pos"][:], dtype=np.int64)
    chunks: list[InferenceChunk] = []
    step = chunk_size - chunk_overlap

    for chrom in np.unique(chroms):
        chrom_indices = np.flatnonzero(chroms == chrom)
        if len(chrom_indices) < 2:
            continue
        chrom_positions = positions[chrom_indices]
        cm_positions = interpolate_cm(str(chrom), chrom_positions, genetic_map)
        split_points = np.flatnonzero(np.diff(cm_positions) > max_imputation_gap_cm) + 1
        segment_starts = np.concatenate(([0], split_points))
        segment_ends = np.concatenate((split_points, [len(chrom_indices)]))

        for seg_start, seg_end in zip(segment_starts, segment_ends):
            start = seg_start
            while start < seg_end:
                end = min(seg_end, start + chunk_size)
                if end - start < 2:
                    break
                global_start = int(chrom_indices[start])
                global_end = int(chrom_indices[end - 1]) + 1
                panel_slice = read_reference_panel_zarr_slice(root, global_start, global_end)
                if vcf_path is not None:
                    records = gls_from_vcf_panel_slice(
                        vcf_path,
                        panel_slice,
                        sample=sample,
                        sample_sex=sample_sex,
                        haploid_chroms=haploid_chroms,
                        hap_index=hap_index,
                    )
                else:
                    records = gls_from_bam(
                        bam_path or "",
                        panel_slice,
                        str(chrom),
                        panel_slice.positions,
                        base_error=base_error,
                        min_base_quality=min_base_quality,
                        sample_sex=sample_sex,
                        haploid_chroms=haploid_chroms,
                        fasta_path=fasta_path,
                    )
                chunks.extend(
                    build_chunks_from_records(
                        records,
                        panel_slice,
                        genetic_map,
                        chunk_size=chunk_size,
                        chunk_overlap=0,
                        max_imputation_gap_cm=max_imputation_gap_cm,
                    )
                )
                if end == seg_end:
                    break
                start += step
    return chunks


def solve_inference_chunk(
    chunk: InferenceChunk,
    *,
    d_coupling: float,
    logit_l2: float,
    gl_weight: float,
    prior_weight: float,
    prior_center: float | None,
    site_weighting: bool,
    posterior_init: bool,
    init_smooth_window: int,
    robust_likelihood_cap: float,
    unknown_penalty: float,
    maxiter: int,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray | None, bool, str]:
    site_weights = (
        frequency_site_weights(chunk.frequencies, chunk.allele_counts)
        if site_weighting
        else None
    )
    if chunk.frequencies.shape[1] == 2:
        solver = TraceSolver(
            d_coupling=d_coupling,
            gl_weight=gl_weight,
            prior_weight=prior_weight,
            prior_center=prior_center,
            site_weights=site_weights,
            robust_likelihood_cap=robust_likelihood_cap,
            maxiter=maxiter,
        )
        f_a = chunk.frequencies[:, 0]
        f_b = chunk.frequencies[:, 1]
        initial_phi = (
            posterior_initial_phi(
                f_a,
                f_b,
                chunk.genotype_likelihoods,
                chunk.ploidy,
                smooth_window=init_smooth_window,
            )
            if posterior_init
            else None
        )
        # Marginalize the likelihood over reference-frequency uncertainty when
        # the panel carries per-population counts. Beta-posterior variance
        # f(1-f)/(n+alpha+beta+1); the Laplace default (alpha=beta=1) gives the
        # +3 denominator. Small/bottlenecked panels => large variance => those
        # sites become uninformative and kinetic smoothing dominates instead of
        # forcing false ancestry switches. Large panels => negligible variance.
        if chunk.allele_counts is not None:
            counts = np.asarray(chunk.allele_counts, dtype=np.float64)
            if counts.shape == chunk.frequencies.shape and np.all(np.isfinite(counts)):
                solver.freq_variance_a = f_a * (1.0 - f_a) / (counts[:, 0] + 3.0)
                solver.freq_variance_b = f_b * (1.0 - f_b) / (counts[:, 1] + 3.0)
        result = solver.solve(
            f_a, f_b,
            chunk.genotype_likelihoods,
            chunk.c_m_positions,
            ploidy=chunk.ploidy,
            initial_phi=initial_phi,
        )
        phi = np.column_stack((result.phi, 1.0 - result.phi))
        d_cm = solver._cm_deltas(chunk.c_m_positions)
        # Normalize GLs the same way _validate_inputs does before computing Hessian
        gls = chunk.genotype_likelihoods.astype(np.float64)
        gls = gls / np.sum(gls, axis=1, keepdims=True)
        se = solver.hessian_standard_errors(result.phi, f_a, f_b, gls, d_cm, chunk.ploidy)
        stderr = np.column_stack((se, se))  # same uncertainty for both complementary pops
    else:
        solver = TraceMultidimensionalSolver(
            d_coupling=d_coupling,
            gl_weight=gl_weight,
            logit_l2=logit_l2,
            prior_weight=prior_weight,
            prior_center=None,
            unknown_index=(chunk.frequencies.shape[1] - 1 if unknown_penalty > 0.0 else None),
            unknown_penalty=unknown_penalty,
            site_weights=site_weights,
            robust_likelihood_cap=robust_likelihood_cap,
            maxiter=maxiter,
        )
        initial_logits = (
            posterior_initial_logits(
                chunk.frequencies,
                chunk.genotype_likelihoods,
                chunk.ploidy,
                smooth_window=init_smooth_window,
            )
            if posterior_init
            else None
        )
        result = solver.solve(
            chunk.frequencies,
            chunk.genotype_likelihoods,
            chunk.c_m_positions,
            ploidy=chunk.ploidy,
            initial_logits=initial_logits,
        )
        phi = result.phi
        stderr = None  # Hessian UQ not yet implemented for K>2
    return chunk.chromosome, chunk.positions, phi, stderr, result.converged, result.message


def _solve_inference_chunk_star(args):
    (
        chunk,
        d_coupling,
        logit_l2,
        gl_weight,
        prior_weight,
        prior_center,
        site_weighting,
        posterior_init,
        init_smooth_window,
        robust_likelihood_cap,
        unknown_penalty,
        maxiter,
    ) = args
    return solve_inference_chunk(
        chunk,
        d_coupling=d_coupling,
        logit_l2=logit_l2,
        gl_weight=gl_weight,
        prior_weight=prior_weight,
        prior_center=prior_center,
        site_weighting=site_weighting,
        posterior_init=posterior_init,
        init_smooth_window=init_smooth_window,
        robust_likelihood_cap=robust_likelihood_cap,
        unknown_penalty=unknown_penalty,
        maxiter=maxiter,
    )


def stitch_chunk_results(
    results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, bool, str]],
    *,
    chunk_overlap: int,
) -> tuple[
    list[tuple[str, np.ndarray, np.ndarray]],
    list[tuple[str, np.ndarray, np.ndarray]] | None,
]:
    has_stderr = any(r[3] is not None for r in results)
    by_chrom: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray | None, bool, str]]] = {}
    for chrom, positions, phi, stderr, converged, message in results:
        if not converged:
            LOGGER.warning("chunk %s:%s-%s did not converge: %s", chrom, positions[0], positions[-1], message)
        by_chrom.setdefault(chrom, []).append((positions, phi, stderr, converged, message))

    stitched: list[tuple[str, np.ndarray, np.ndarray]] = []
    stitched_stderr: list[tuple[str, np.ndarray, np.ndarray]] = []

    for chrom, chrom_chunks in by_chrom.items():
        chrom_chunks.sort(key=lambda item: int(item[0][0]))
        sum_by_pos: dict[int, np.ndarray] = {}
        var_by_pos: dict[int, np.ndarray] = {}
        weight_by_pos: dict[int, float] = {}

        for idx, (positions, phi, stderr, _converged, _message) in enumerate(chrom_chunks):
            weights = np.ones(len(positions), dtype=np.float64)
            if chunk_overlap > 0 and idx > 0:
                previous_positions = set(int(pos) for pos in chrom_chunks[idx - 1][0])
                front_overlap = 0
                for pos in positions:
                    if int(pos) not in previous_positions:
                        break
                    front_overlap += 1
                if front_overlap > 0:
                    weights[:front_overlap] *= np.linspace(0.0, 1.0, front_overlap)
            if chunk_overlap > 0 and idx + 1 < len(chrom_chunks):
                next_positions = set(int(pos) for pos in chrom_chunks[idx + 1][0])
                tail_overlap = 0
                for pos in positions[::-1]:
                    if int(pos) not in next_positions:
                        break
                    tail_overlap += 1
                if tail_overlap > 0:
                    weights[-tail_overlap:] *= np.linspace(1.0, 0.0, tail_overlap)

            for i, (pos, row, weight) in enumerate(zip(positions, phi, weights)):
                if weight == 0.0:
                    continue
                pos_int = int(pos)
                sum_by_pos[pos_int] = sum_by_pos.get(pos_int, np.zeros_like(row)) + weight * row
                weight_by_pos[pos_int] = weight_by_pos.get(pos_int, 0.0) + weight
                if stderr is not None:
                    var_by_pos[pos_int] = (
                        var_by_pos.get(pos_int, np.zeros_like(stderr[i]))
                        + (weight * stderr[i]) ** 2
                    )

        sorted_positions = np.asarray(sorted(sum_by_pos), dtype=np.int64)
        phi_rows = np.vstack(
            [sum_by_pos[int(pos)] / max(weight_by_pos[int(pos)], EPS) for pos in sorted_positions]
        )
        stitched.append((chrom, sorted_positions, phi_rows))

        if has_stderr and var_by_pos:
            se_rows = np.vstack(
                [np.sqrt(var_by_pos.get(int(pos), np.zeros_like(phi_rows[0]))) / max(weight_by_pos[int(pos)], EPS)
                 for pos in sorted_positions]
            )
            stitched_stderr.append((chrom, sorted_positions, se_rows))

    stitched.sort(key=lambda item: (item[0], int(item[1][0])))
    stitched_stderr.sort(key=lambda item: (item[0], int(item[1][0])))
    return stitched, (stitched_stderr if has_stderr else None)


def write_results_tsv(
    output_path: str,
    stitched: list[tuple[str, np.ndarray, np.ndarray]],
    population_names: tuple[str, ...],
    *,
    stderr: list[tuple[str, np.ndarray, np.ndarray]] | None = None,
) -> None:
    se_lookup: dict[tuple[str, int], np.ndarray] = {}
    if stderr is not None:
        for chrom, positions, se in stderr:
            for pos, se_row in zip(positions, se):
                se_lookup[(chrom, int(pos))] = se_row

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        se_cols = [f"phi_{pop}_stderr" for pop in population_names] if stderr is not None else []
        writer.writerow(["chrom", "pos", *[f"phi_{pop}" for pop in population_names], *se_cols])
        for chrom, positions, phi in stitched:
            for pos, row in zip(positions, phi):
                values = [chrom, int(pos), *[f"{value:.8g}" for value in row]]
                if stderr is not None:
                    se_row = se_lookup.get((chrom, int(pos)))
                    se_vals = [f"{v:.8g}" for v in se_row] if se_row is not None else [""] * len(population_names)
                    values.extend(se_vals)
                writer.writerow(values)


def write_results_zarr(
    output_path: str,
    stitched: list[tuple[str, np.ndarray, np.ndarray]],
    population_names: tuple[str, ...],
) -> None:
    zarr = require_optional_module("zarr", "zarr")
    total_rows = sum(len(positions) for _chrom, positions, _phi in stitched)
    n_populations = len(population_names)
    root = zarr.open_group(output_path, mode="w")
    phi_ds = root.create_dataset(
        "phi",
        shape=(total_rows, n_populations),
        chunks=(min(100_000, max(1, total_rows)), n_populations),
        dtype="f4",
    )
    pos_ds = root.create_dataset(
        "pos",
        shape=(total_rows,),
        chunks=(min(100_000, max(1, total_rows)),),
        dtype="i8",
    )
    max_chrom_len = max(1, max(len(chrom) for chrom, _positions, _phi in stitched))
    chrom_ds = root.create_dataset(
        "chrom",
        shape=(total_rows,),
        chunks=(min(100_000, max(1, total_rows)),),
        dtype=f"U{max_chrom_len}",
    )
    root.attrs["population_names"] = list(population_names)

    offset = 0
    for chrom, positions, phi in stitched:
        end = offset + len(positions)
        phi_ds[offset:end, :] = phi.astype(np.float32)
        pos_ds[offset:end] = positions
        chrom_ds[offset:end] = [chrom] * len(positions)
        offset = end


def compute_global_ancestry(
    stitched: list[tuple[str, np.ndarray, np.ndarray]],
    genetic_map: dict[str, tuple[np.ndarray, np.ndarray]],
    population_names: tuple[str, ...],
) -> dict[str, float]:
    """cM-weighted global ancestry: integral of phi(x) dx / total cM."""
    total_weight = 0.0
    weighted_phi = np.zeros(len(population_names), dtype=np.float64)
    for chrom, positions, phi in stitched:
        if len(positions) < 2:
            continue
        cm_positions = interpolate_cm(chrom, positions, genetic_map)
        d_cm = np.diff(cm_positions)
        weights = np.zeros(len(positions), dtype=np.float64)
        weights[:-1] += 0.5 * d_cm
        weights[1:] += 0.5 * d_cm
        total_weight += float(np.sum(weights))
        weighted_phi += np.sum(phi * weights[:, None], axis=0)
    if total_weight == 0.0:
        return {pop: float("nan") for pop in population_names}
    return {pop: float(weighted_phi[i] / total_weight) for i, pop in enumerate(population_names)}


def estimate_g_from_autocorrelation(
    stitched: list[tuple[str, np.ndarray, np.ndarray]],
    genetic_map: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    grid_step_cm: float = 0.05,
    max_lag_cm: float | None = None,
    delta_aic_threshold: float = 10.0,
) -> AdmixtureTimeResult:
    """
    Estimate generations since admixture via Wiener-Khinchin FFT autocorrelation.

    Fits rho(r) = A*exp(-g*r) + C to the ancestry ACF computed per chromosome
    and averaged by chromosome span. AIC selects between single- and multi-pulse
    models; degeneracy checks guard against spurious two-component results.
    """
    # Pass 1: build per-chromosome interpolated signals, record spans
    chrom_signals: list[tuple[np.ndarray, float]] = []
    total_cm_span = 0.0
    for chrom, positions, phi in stitched:
        if len(positions) < 10:
            continue
        cm = interpolate_cm(chrom, positions, genetic_map)
        span = float(cm[-1] - cm[0])
        if span < grid_step_cm * 5:
            continue
        col = phi[:, int(np.argmax(np.var(phi, axis=0)))] if phi.ndim == 2 else phi
        grid_cm = np.arange(cm[0], cm[-1], grid_step_cm)
        chrom_signals.append((np.interp(grid_cm, cm, col), span))
        total_cm_span += span

    if not chrom_signals:
        raise ValueError("No chromosomes with sufficient SNP density for autocorrelation.")

    eff_max_lag = max_lag_cm if max_lag_cm is not None else min(total_cm_span * 0.4, 50.0)
    max_bin = max(int(eff_max_lag / grid_step_cm), 5)
    lags = np.arange(1, max_bin) * grid_step_cm / 100.0  # Morgans

    # Pass 2: per-chromosome FFT ACF, per-lag weighted average.
    # Track weights per lag bin so short chromosomes don't dilute long-lag estimates.
    acf_sum = np.zeros(max_bin, dtype=float)
    weight_per_lag = np.zeros(max_bin, dtype=float)
    for signal, span in chrom_signals:
        signal = signal - signal.mean()
        var = float(np.var(signal))
        if var < 1e-12:
            continue
        n = len(signal)
        fft_s = np.fft.rfft(signal, n=2 * n)
        raw = np.fft.irfft(fft_s * np.conj(fft_s))[:n]
        chrom_acf = (raw / np.arange(n, 0, -1, dtype=float)) / var
        bins = min(max_bin, len(chrom_acf))
        acf_sum[:bins] += span * chrom_acf[:bins]
        weight_per_lag[:bins] += span
    if not np.any(weight_per_lag > 0):
        raise ValueError("All chromosomes have near-zero phi variance.")
    acf_full = np.where(weight_per_lag > 0, acf_sum / weight_per_lag, 0.0)
    acf = acf_full[1:]  # drop lag-0 (always 1 by definition)

    if len(lags) < 5:
        raise ValueError("Insufficient lag range; use more chromosomes or a larger dataset.")

    # --- Model 1: A * exp(-g*r) + C  (k=3) ---
    def single(r: np.ndarray, g: float, A: float, C: float) -> np.ndarray:
        return A * np.exp(-g * r) + C

    popt1, pcov1 = curve_fit(
        single, lags, acf,
        p0=[10.0, float(np.clip(acf[0], 0.0, 1.0)), 0.0],
        bounds=([0.5, 0.0, -0.5], [5000.0, 1.05, 0.5]),
        maxfev=10_000,
    )
    g1, A1, C1 = popt1
    g1_err = float(np.sqrt(max(pcov1[0, 0], 0.0)))
    n_pts = len(lags)
    rss1 = float(np.sum((acf - single(lags, *popt1)) ** 2))
    aic1 = 2 * 3 + n_pts * np.log(rss1 / n_pts + 1e-30)

    # --- Model 2: A*(w*exp(-ga*r) + (1-w)*exp(-gb*r)) + C  (k=5) ---
    def double(r: np.ndarray, ga: float, gb: float, w: float, A: float, C: float) -> np.ndarray:
        return A * (w * np.exp(-ga * r) + (1.0 - w) * np.exp(-gb * r)) + C

    aic2 = np.inf
    g2a, g2b, w2, A2, C2 = g1, g1, 0.5, A1, C1
    g2a_err = g2b_err = g1_err
    try:
        popt2, pcov2 = curve_fit(
            double, lags, acf,
            p0=[max(g1 * 0.3, 1.0), g1 * 3.0, 0.5, float(np.clip(acf[0], 0.0, 1.0)), 0.0],
            bounds=([0.5, 0.5, 0.01, 0.0, -0.5], [5000.0, 5000.0, 0.99, 1.05, 0.5]),
            maxfev=20_000,
        )
        g2a, g2b, w2, A2, C2 = popt2
        if g2a > g2b:          # keep ga = recent, gb = ancient
            g2a, g2b = g2b, g2a
            w2 = 1.0 - w2
        g2a_err = float(np.sqrt(max(pcov2[0, 0], 0.0)))
        g2b_err = float(np.sqrt(max(pcov2[1, 1], 0.0)))
        rss2 = float(np.sum((acf - double(lags, *popt2)) ** 2))
        aic2 = 2 * 5 + n_pts * np.log(rss2 / n_pts + 1e-30)
    except Exception:
        pass

    delta_aic = float(aic1 - aic2)
    separation_ok = (g2b / max(g2a, 0.1)) > 2.0
    weight_ok = 0.05 < w2 < 0.95

    if delta_aic > delta_aic_threshold and separation_ok and weight_ok:
        return AdmixtureTimeResult(
            model="multi_pulse",
            g_estimates=[float(g2a), float(g2b)],
            g_std_errors=[g2a_err, g2b_err],
            weights=[float(w2), float(1.0 - w2)],
            amplitude=float(A2),
            baseline=float(C2),
            delta_aic=delta_aic,
            lags_morgan=lags,
            acf=acf,
        )
    return AdmixtureTimeResult(
        model="single_pulse",
        g_estimates=[float(g1)],
        g_std_errors=[g1_err],
        weights=[1.0],
        amplitude=float(A1),
        baseline=float(C1),
        delta_aic=delta_aic,
        lags_morgan=lags,
        acf=acf,
    )


def write_global_ancestry_summary(
    summary_path: str,
    global_ancestry: dict[str, float],
    sample_name: str | None,
    *,
    admixture_time: AdmixtureTimeResult | None = None,
) -> None:
    label = sample_name or "sample"
    lines = [f"TRACE global ancestry summary: {label}"]
    for pop, proportion in global_ancestry.items():
        lines.append(f"  {pop}: {proportion:.6f}")
    if admixture_time is not None:
        lines.append("admixture time (FFT autocorrelation):")
        lines.append(
            f"  model: {admixture_time.model}  "
            f"delta_AIC={admixture_time.delta_aic:.2f}  "
            f"weighted_g={admixture_time.weighted_g:.1f}"
        )
        if admixture_time.model == "single_pulse":
            g, ge = admixture_time.g_estimates[0], admixture_time.g_std_errors[0]
            lines.append(f"  g = {g:.1f} +/- {ge:.1f} generations")
        else:
            tags = ("recent", "ancient")
            for tag, g, ge, w in zip(
                tags,
                admixture_time.g_estimates,
                admixture_time.g_std_errors,
                admixture_time.weights,
            ):
                lines.append(f"  pulse ({tag}): g = {g:.1f} +/- {ge:.1f} gen, weight = {w:.3f}")
        lines.append(
            "  note: std errors assume independent lag bins; true uncertainty is larger"
        )
        lines.append(
            "  caution: g-estimator is exploratory — multi-pulse calls have ~30% false-positive"
            " rate on single-pulse genomes; validate with ALDER or dadi before drawing"
            " demographic conclusions"
        )
    text = "\n".join(lines) + "\n"
    print(text, end="")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(text)


def write_results_bed(
    output_path: str,
    stitched: list[tuple[str, np.ndarray, np.ndarray]],
    population_names: tuple[str, ...],
    *,
    threshold: float = 0.8,
) -> None:
    """Write hard ancestry calls as a 4-column BED file (0-based half-open intervals)."""
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        for chrom, positions, phi in stitched:
            if len(positions) == 0:
                continue
            max_phi = np.max(phi, axis=1)
            calls = np.where(max_phi >= threshold, np.argmax(phi, axis=1), -1)
            i = 0
            while i < len(positions):
                j = i + 1
                while j < len(positions) and calls[j] == calls[i]:
                    j += 1
                call_label = population_names[int(calls[i])] if calls[i] >= 0 else "Unknown"
                writer.writerow([chrom, int(positions[i]) - 1, int(positions[j - 1]), call_label])
                i = j


def write_results_msp(
    output_path: str,
    stitched: list[tuple[str, np.ndarray, np.ndarray]],
    population_names: tuple[str, ...],
    genetic_map: dict[str, tuple[np.ndarray, np.ndarray]],
    sample_name: str,
    *,
    phased: bool = False,
    threshold: float = 0.8,
) -> None:
    """Write hard ancestry calls in RFMix .msp.tsv format.

    For phased data, two haplotype columns (.0 and .1) are written with
    independent hard calls. For unphased diploid data, a single .diploid
    column is written with the argmax population index.
    """
    pop_codes = " ".join(f"{pop}={i}" for i, pop in enumerate(population_names))
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(f"#Subpopulation order/codes: {pop_codes}\n")
        if phased:
            handle.write(
                f"#chrom\tsgpos\tegpos\tspos\tepos\tn snps\t{sample_name}.0\t{sample_name}.1\n"
            )
        else:
            handle.write(
                f"#chrom\tsgpos\tegpos\tspos\tepos\tn snps\t{sample_name}.diploid\n"
            )
        writer = csv.writer(handle, delimiter="\t")
        for chrom, positions, phi in stitched:
            if len(positions) == 0:
                continue
            cm_positions = interpolate_cm(chrom, positions, genetic_map)
            max_phi = np.max(phi, axis=1)
            calls = np.where(max_phi >= threshold, np.argmax(phi, axis=1), -1)
            i = 0
            while i < len(positions):
                j = i + 1
                while j < len(positions) and calls[j] == calls[i]:
                    j += 1
                call = int(calls[i])
                row = [
                    chrom,
                    f"{cm_positions[i]:.6f}",
                    f"{cm_positions[j - 1]:.6f}",
                    int(positions[i]),
                    int(positions[j - 1]),
                    j - i,
                    call,
                ]
                if phased:
                    row.append(call)  # second haplotype column
                writer.writerow(row)
                i = j


def write_results_msp_phased(
    output_path: str,
    hap0: list[tuple[str, np.ndarray, np.ndarray]],
    hap1: list[tuple[str, np.ndarray, np.ndarray]],
    population_names: tuple[str, ...],
    genetic_map: dict[str, tuple[np.ndarray, np.ndarray]],
    sample_name: str,
    *,
    threshold: float = 0.8,
) -> None:
    """Write one RFMix-style MSP with separate haplotype columns."""
    pop_codes = " ".join(f"{pop}={i}" for i, pop in enumerate(population_names))

    def calls_by_chrom(stitched):
        out: dict[str, dict[int, int]] = {}
        for chrom, positions, phi in stitched:
            max_phi = np.max(phi, axis=1)
            calls = np.where(max_phi >= threshold, np.argmax(phi, axis=1), -1)
            out[str(chrom)] = {int(pos): int(call) for pos, call in zip(positions, calls)}
        return out

    hap0_calls = calls_by_chrom(hap0)
    hap1_calls = calls_by_chrom(hap1)

    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        handle.write(f"#Subpopulation order/codes: {pop_codes}\n")
        handle.write(
            f"#chrom\tsgpos\tegpos\tspos\tepos\tn snps\t{sample_name}.0\t{sample_name}.1\n"
        )
        writer = csv.writer(handle, delimiter="\t")
        for chrom in sorted(set(hap0_calls) & set(hap1_calls)):
            shared_positions = sorted(set(hap0_calls[chrom]) & set(hap1_calls[chrom]))
            if not shared_positions:
                continue
            positions = np.asarray(shared_positions, dtype=np.int64)
            cm_positions = interpolate_cm(chrom, positions, genetic_map)
            pairs = np.asarray(
                [(hap0_calls[chrom][pos], hap1_calls[chrom][pos]) for pos in shared_positions],
                dtype=np.int64,
            )
            i = 0
            while i < len(positions):
                j = i + 1
                while j < len(positions) and np.array_equal(pairs[j], pairs[i]):
                    j += 1
                writer.writerow([
                    chrom,
                    f"{cm_positions[i]:.6f}",
                    f"{cm_positions[j - 1]:.6f}",
                    int(positions[i]),
                    int(positions[j - 1]),
                    j - i,
                    int(pairs[i, 0]),
                    int(pairs[i, 1]),
                ])
                i = j


def gls_from_vcf_haplotype(
    vcf_path: str,
    sample: str | None,
    hap_index: int,
    *,
    sample_sex: str,
    haploid_chroms: set[str],
) -> Iterable[tuple[str, int, str, str, np.ndarray, int]]:
    """Yield haploid GL records for one phased haplotype (hap_index 0 or 1)."""
    cyvcf2 = require_optional_module("cyvcf2", "cyvcf2")
    vcf = cyvcf2.VCF(vcf_path)
    sample_index = 0
    if sample is not None:
        if sample not in vcf.samples:
            raise ValueError(f"Sample {sample!r} not found in VCF.")
        sample_index = vcf.samples.index(sample)
    for variant in vcf:
        if len(variant.ALT or []) != 1:
            continue
        ploidy = chrom_ploidy(variant.CHROM, sample_sex=sample_sex, haploid_chroms=haploid_chroms)
        try:
            gt = variant.genotypes[sample_index]
        except Exception as exc:
            raise ValueError(
                "--phased requires VCF GT calls; genotype-likelihood-only records "
                f"cannot be split into haplotypes at {variant.CHROM}:{variant.POS}."
            ) from exc
        alleles = [int(a) for a in gt[:-1] if int(a) >= 0]
        if ploidy == 2 and len(alleles) >= 2:
            allele = alleles[min(hap_index, len(alleles) - 1)]
            gl = genotype_likelihood_from_gt([allele], ploidy=1)
            yield variant.CHROM, int(variant.POS), variant.REF, variant.ALT[0], gl, 1
        elif hap_index == 0:
            # Haploid chrom: emit once only for hap_index 0
            gl = genotype_likelihood_from_gt(alleles, ploidy)
            yield variant.CHROM, int(variant.POS), variant.REF, variant.ALT[0], gl, ploidy


def _haplotype_path(path: str, hap_idx: int) -> str:
    dot = path.rfind(".")
    sep = max(path.rfind("/"), path.rfind("\\"))
    if dot > sep:
        return path[:dot] + f"_hap{hap_idx}" + path[dot:]
    return path + f"_hap{hap_idx}"


def _solve_chunks(
    chunks: list[InferenceChunk],
    *,
    d_coupling: float,
    logit_l2: float,
    gl_weight: float,
    prior_weight: float,
    prior_center: float | None,
    site_weighting: bool,
    posterior_init: bool,
    init_smooth_window: int,
    robust_likelihood_cap: float,
    unknown_penalty: float,
    maxiter: int,
    workers: int,
    chunk_overlap: int,
) -> tuple[
    list[tuple[str, np.ndarray, np.ndarray]],
    list[tuple[str, np.ndarray, np.ndarray]] | None,
]:
    task_args = [
        (
            chunk,
            d_coupling,
            logit_l2,
            gl_weight,
            prior_weight,
            prior_center,
            site_weighting,
            posterior_init,
            init_smooth_window,
            robust_likelihood_cap,
            unknown_penalty,
            maxiter,
        )
        for chunk in chunks
    ]
    results: list[tuple[str, np.ndarray, np.ndarray, np.ndarray | None, bool, str]] = []
    if workers == 1:
        results = [_solve_inference_chunk_star(args) for args in task_args]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_solve_inference_chunk_star, args) for args in task_args]
            for future in progress_iter(
                as_completed(futures), total=len(futures), desc="Optimizing chunks"
            ):
                results.append(future.result())
    return stitch_chunk_results(results, chunk_overlap=chunk_overlap)


def generate_zarr_report(zarr_path: str, output_html: str) -> None:
    zarr = require_optional_module("zarr", "zarr")
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib is required for reports. Install it with: "
            "python -m pip install matplotlib"
        ) from exc

    root = zarr.open_group(zarr_path, mode="r")
    phi = np.asarray(root["phi"][:], dtype=np.float64)
    positions = np.asarray(root["pos"][:], dtype=np.int64)
    chroms = np.asarray(root["chrom"][:], dtype=str)
    population_names = list(root.attrs.get("population_names", [f"K{i}" for i in range(phi.shape[1])]))

    images: list[tuple[str, str]] = []
    for chrom in np.unique(chroms):
        mask = chroms == chrom
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.stackplot(positions[mask], phi[mask].T, labels=population_names)
        ax.set_title(f"TRACE chromosome painting: {chrom}")
        ax.set_xlabel("Position (bp)")
        ax.set_ylabel("Ancestry fraction")
        ax.set_ylim(0.0, 1.0)
        ax.legend(loc="upper right", ncol=min(len(population_names), 4), fontsize=8)
        images.append((f"Chromosome {chrom}", figure_to_base64(fig)))
        plt.close(fig)

    if phi.shape[1] == 3:
        x = phi[:, 1] + 0.5 * phi[:, 2]
        y = (np.sqrt(3.0) / 2.0) * phi[:, 2]
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3.0) / 2.0, 0], color="black", linewidth=1)
        ax.scatter(x, y, c=np.arange(len(x)), s=8, cmap="viridis")
        ax.text(-0.04, -0.04, population_names[0], ha="right", va="top")
        ax.text(1.04, -0.04, population_names[1], ha="left", va="top")
        ax.text(0.5, np.sqrt(3.0) / 2.0 + 0.04, population_names[2], ha="center", va="bottom")
        ax.set_title("TRACE K=3 simplex trajectory")
        ax.set_axis_off()
        images.append(("Ternary trajectory", figure_to_base64(fig)))
        plt.close(fig)

    with open(output_html, "w", encoding="utf-8") as handle:
        handle.write("<!doctype html><html><head><meta charset='utf-8'><title>TRACE report</title>")
        handle.write("<style>body{font-family:sans-serif;margin:2rem}img{max-width:100%;border:1px solid #ddd}</style>")
        handle.write("</head><body><h1>TRACE Ancestry Report</h1>")
        for title, image in images:
            handle.write(f"<h2>{title}</h2><img src='data:image/png;base64,{image}' alt='{title}'>")
        handle.write("</body></html>")


def figure_to_base64(fig) -> str:
    buffer = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buffer, format="png", dpi=150)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def run_file_pipeline(
    *,
    vcf_path: str | None,
    bam_path: str | None,
    sample: str | None,
    reference_panel_path: str,
    reference_panel_zarr_path: str | None,
    genetic_map_path: str,
    output_path: str,
    output_zarr_path: str | None,
    output_bed_path: str | None,
    output_msp_path: str | None,
    summary_path: str | None,
    hard_call_threshold: float,
    phased: bool,
    chunk_size: int,
    chunk_overlap: int,
    max_imputation_gap_cm: float,
    workers: int,
    d_coupling: float,
    auto_d: bool,
    auto_params: bool,
    d_grid: list[float],
    gl_weight_grid: list[float],
    robust_cap_grid: list[float],
    calibration_mask_fraction: float,
    calibration_seed: int,
    calibration_score: str,
    calibration_mask_mode: str,
    calibration_block_cm: float,
    logit_l2: float,
    gl_weight: float,
    prior_weight: float,
    prior_center: float | None,
    site_weighting: bool,
    posterior_init: bool,
    init_smooth_window: int,
    robust_likelihood_cap: float,
    unknown_state: bool,
    unknown_mode: str,
    unknown_frequency: float,
    unknown_penalty: float,
    unknown_label: str,
    maxiter: int,
    base_error: float,
    min_base_quality: int,
    sample_sex: str,
    haploid_chroms: set[str],
    fasta_path: str | None,
    alpha: float,
    beta: float,
    freq_shrinkage: float = 0.0,
) -> None:
    if (vcf_path is None) == (bam_path is None):
        raise ValueError("Provide exactly one of --vcf or --bam.")

    genetic_map = read_genetic_map(genetic_map_path)

    if phased:
        if vcf_path is None:
            raise ValueError("--phased requires --vcf input.")
        if reference_panel_zarr_path is not None:
            population_names = reference_zarr_population_names(reference_panel_zarr_path)
        else:
            panel = read_reference_panel(reference_panel_path, alpha=alpha, beta=beta, freq_shrinkage=freq_shrinkage)
            population_names = panel.population_names
        phased_msp_tracks: list[list[tuple[str, np.ndarray, np.ndarray]]] = []
        for hap_idx in range(2):
            if reference_panel_zarr_path is not None:
                hap_chunks = build_chunks_from_reference_zarr(
                    reference_panel_zarr_path,
                    genetic_map,
                    vcf_path=vcf_path,
                    bam_path=None,
                    sample=sample,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    max_imputation_gap_cm=max_imputation_gap_cm,
                    base_error=base_error,
                    min_base_quality=min_base_quality,
                    sample_sex=sample_sex,
                    haploid_chroms=haploid_chroms,
                    fasta_path=fasta_path,
                    hap_index=hap_idx,
                )
            else:
                records = gls_from_vcf_haplotype(
                    vcf_path, sample, hap_idx,
                    sample_sex=sample_sex,
                    haploid_chroms=haploid_chroms,
                )
                hap_chunks = build_chunks_from_records(
                    records, panel, genetic_map,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    max_imputation_gap_cm=max_imputation_gap_cm,
                )
            if not hap_chunks:
                raise ValueError(f"No overlapping variants found for haplotype {hap_idx}.")
            if (auto_params or auto_d) and hap_idx == 0:
                cal_chunk = max(hap_chunks, key=lambda item: len(item.positions))
                if auto_params:
                    d_coupling, gl_weight, robust_likelihood_cap, scores = calibrate_params_for_chunk(
                        cal_chunk,
                        d_grid=d_grid,
                        gl_weight_grid=gl_weight_grid,
                        robust_cap_grid=robust_cap_grid,
                        mask_fraction=calibration_mask_fraction,
                        seed=calibration_seed,
                        calibration_score=calibration_score,
                        calibration_mask_mode=calibration_mask_mode,
                        calibration_block_cm=calibration_block_cm,
                        logit_l2=logit_l2,
                        prior_weight=prior_weight,
                        prior_center=prior_center,
                        site_weighting=site_weighting,
                        posterior_init=posterior_init,
                        init_smooth_window=init_smooth_window,
                        unknown_penalty=unknown_penalty,
                        maxiter=maxiter,
                    )
                    LOGGER.info(
                        "auto-calibrated params (phased, from hap0): D=%g gl_weight=%g robust_cap=%g",
                        d_coupling, gl_weight, robust_likelihood_cap,
                    )
                else:
                    d_coupling, _ = calibrate_d_for_chunk(
                        cal_chunk,
                        d_grid=d_grid,
                        mask_fraction=calibration_mask_fraction,
                        seed=calibration_seed,
                        calibration_score=calibration_score,
                        calibration_mask_mode=calibration_mask_mode,
                        calibration_block_cm=calibration_block_cm,
                        logit_l2=logit_l2,
                        gl_weight=gl_weight,
                        prior_weight=prior_weight,
                        prior_center=prior_center,
                        maxiter=maxiter,
                    )
                    LOGGER.info("auto-calibrated D=%g (phased, from hap0)", d_coupling)
            hap_stitched, hap_stderr = _solve_chunks(
                hap_chunks,
                d_coupling=d_coupling,
                logit_l2=logit_l2,
                gl_weight=gl_weight,
                prior_weight=prior_weight,
                prior_center=prior_center,
                site_weighting=site_weighting,
                posterior_init=posterior_init,
                init_smooth_window=init_smooth_window,
                robust_likelihood_cap=robust_likelihood_cap,
                unknown_penalty=unknown_penalty,
                maxiter=maxiter,
                workers=workers,
                chunk_overlap=chunk_overlap,
            )
            phased_msp_tracks.append(hap_stitched)
            if output_zarr_path:
                hap_zarr = _haplotype_path(output_zarr_path, hap_idx)
                write_results_zarr(hap_zarr, hap_stitched, population_names)
                print(f"wrote {hap_zarr}")
            if output_path:
                hap_tsv = _haplotype_path(output_path, hap_idx)
                write_results_tsv(hap_tsv, hap_stitched, population_names, stderr=hap_stderr)
                print(f"wrote {hap_tsv}")
            hap_summary = _haplotype_path(summary_path, hap_idx) if summary_path else None
            if hap_summary is None and output_path:
                dot = output_path.rfind(".")
                sep = max(output_path.rfind("/"), output_path.rfind("\\"))
                stem = output_path[:dot] if dot > sep else output_path
                hap_summary = _haplotype_path(stem + "_summary.txt", hap_idx)
            if hap_summary:
                global_ancestry = compute_global_ancestry(hap_stitched, genetic_map, population_names)
                g_result = None
                try:
                    g_result = estimate_g_from_autocorrelation(hap_stitched, genetic_map)
                except Exception as exc:
                    LOGGER.debug("admixture time estimation skipped for hap%d: %s", hap_idx, exc)
                write_global_ancestry_summary(
                    hap_summary, global_ancestry, f"{sample or 'sample'}_hap{hap_idx}",
                    admixture_time=g_result,
                )
            if output_bed_path:
                write_results_bed(
                    _haplotype_path(output_bed_path, hap_idx),
                    hap_stitched, population_names,
                    threshold=hard_call_threshold,
                )
        if output_msp_path:
            write_results_msp_phased(
                output_msp_path,
                phased_msp_tracks[0],
                phased_msp_tracks[1],
                population_names,
                genetic_map,
                sample or "sample",
                threshold=hard_call_threshold,
            )
            print(f"wrote {output_msp_path}")
        return

    if reference_panel_zarr_path is not None:
        population_names = reference_zarr_population_names(reference_panel_zarr_path)
        chunks = build_chunks_from_reference_zarr(
            reference_panel_zarr_path,
            genetic_map,
            vcf_path=vcf_path,
            bam_path=bam_path,
            sample=sample,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_imputation_gap_cm=max_imputation_gap_cm,
            base_error=base_error,
            min_base_quality=min_base_quality,
            sample_sex=sample_sex,
            haploid_chroms=haploid_chroms,
            fasta_path=fasta_path,
        )
    else:
        panel = read_reference_panel(reference_panel_path, alpha=alpha, beta=beta, freq_shrinkage=freq_shrinkage)
        population_names = panel.population_names
        if vcf_path is not None:
            chunks = build_chunks_from_vcf(
                vcf_path,
                panel,
                genetic_map,
                sample=sample,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_imputation_gap_cm=max_imputation_gap_cm,
                sample_sex=sample_sex,
                haploid_chroms=haploid_chroms,
            )
        else:
            chunks = build_chunks_from_bam(
                bam_path or "",
                panel,
                genetic_map,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_imputation_gap_cm=max_imputation_gap_cm,
                base_error=base_error,
                min_base_quality=min_base_quality,
                sample_sex=sample_sex,
                haploid_chroms=haploid_chroms,
                fasta_path=fasta_path,
            )

    if not chunks:
        raise ValueError("No overlapping variants found between input, reference panel, and map.")

    if unknown_state:
        if not 0.0 < unknown_frequency < 1.0:
            raise ValueError("--unknown-frequency must be in (0, 1).")
        chunks = add_unknown_state_to_chunks(
            chunks,
            mode=unknown_mode,
            flat_frequency=unknown_frequency,
            label=unknown_label,
        )
        population_names = (*population_names, unknown_label)

    if auto_params:
        calibration_chunk = max(chunks, key=lambda item: len(item.positions))
        d_coupling, gl_weight, robust_likelihood_cap, scores = calibrate_params_for_chunk(
            calibration_chunk,
            d_grid=d_grid,
            gl_weight_grid=gl_weight_grid,
            robust_cap_grid=robust_cap_grid,
            mask_fraction=calibration_mask_fraction,
            seed=calibration_seed,
            calibration_score=calibration_score,
            calibration_mask_mode=calibration_mask_mode,
            calibration_block_cm=calibration_block_cm,
            logit_l2=logit_l2,
            prior_weight=prior_weight,
            prior_center=prior_center,
            site_weighting=site_weighting,
            posterior_init=posterior_init,
            init_smooth_window=init_smooth_window,
            unknown_penalty=unknown_penalty,
            maxiter=maxiter,
        )
        LOGGER.info("auto-calibrated params by masked hold-out likelihood:")
        top_scores = sorted(scores, key=lambda item: item[3], reverse=True)[:10]
        for d_value, gl_value, cap_value, score in top_scores:
            LOGGER.info("  D=%g\tgl_weight=%g\tcap=%g\tscore=%.6f", d_value, gl_value, cap_value, score)
        LOGGER.info(
            "selected D=%g gl_weight=%g robust_cap=%g",
            d_coupling, gl_weight, robust_likelihood_cap,
        )
    elif auto_d:
        calibration_chunk = max(chunks, key=lambda item: len(item.positions))
        d_coupling, scores = calibrate_d_for_chunk(
            calibration_chunk,
            d_grid=d_grid,
            mask_fraction=calibration_mask_fraction,
            seed=calibration_seed,
            calibration_score=calibration_score,
            calibration_mask_mode=calibration_mask_mode,
            calibration_block_cm=calibration_block_cm,
            logit_l2=logit_l2,
            gl_weight=gl_weight,
            prior_weight=prior_weight,
            prior_center=prior_center,
            maxiter=maxiter,
        )
        LOGGER.info("auto-calibrated D by masked hold-out likelihood:")
        for d_value, score in scores:
            LOGGER.info("  D=%g\tscore=%.6f", d_value, score)
        LOGGER.info("selected D=%g", d_coupling)

    stitched, stitched_stderr = _solve_chunks(
        chunks,
        d_coupling=d_coupling,
        logit_l2=logit_l2,
        gl_weight=gl_weight,
        prior_weight=prior_weight,
        prior_center=prior_center,
        site_weighting=site_weighting,
        posterior_init=posterior_init,
        init_smooth_window=init_smooth_window,
        robust_likelihood_cap=robust_likelihood_cap,
        unknown_penalty=unknown_penalty,
        maxiter=maxiter,
        workers=workers,
        chunk_overlap=chunk_overlap,
    )
    if output_zarr_path is not None:
        write_results_zarr(output_zarr_path, stitched, population_names)
        print(f"wrote {output_zarr_path}")
    if output_path:
        write_results_tsv(output_path, stitched, population_names, stderr=stitched_stderr)
        print(f"wrote {output_path}")

    eff_summary = summary_path
    if eff_summary is None and output_path:
        dot = output_path.rfind(".")
        sep = max(output_path.rfind("/"), output_path.rfind("\\"))
        stem = output_path[:dot] if dot > sep else output_path
        eff_summary = stem + "_summary.txt"
    if eff_summary:
        global_ancestry = compute_global_ancestry(stitched, genetic_map, population_names)
        g_result = None
        try:
            g_result = estimate_g_from_autocorrelation(stitched, genetic_map)
        except Exception as exc:
            LOGGER.debug("admixture time estimation skipped: %s", exc)
        write_global_ancestry_summary(eff_summary, global_ancestry, sample, admixture_time=g_result)

    if output_bed_path:
        write_results_bed(output_bed_path, stitched, population_names, threshold=hard_call_threshold)
    if output_msp_path:
        write_results_msp(
            output_msp_path, stitched, population_names, genetic_map,
            sample or "sample", threshold=hard_call_threshold,
        )


def _estimate_prior_center(
    f_a: np.ndarray,
    f_b: np.ndarray,
    genotype_likelihoods: np.ndarray,
    ploidy: np.ndarray,
) -> float:
    """Rough GL-weighted estimate of global ancestry fraction; used as auto prior center."""
    gls = genotype_likelihoods / np.sum(genotype_likelihoods, axis=1, keepdims=True)
    diploid = ploidy == 2
    p_obs = np.empty(len(f_a))
    p_obs[diploid] = 0.5 * gls[diploid, 1] + gls[diploid, 2]
    p_obs[~diploid] = gls[~diploid, 2]
    delta_f = f_a - f_b
    safe_delta = np.where(np.abs(delta_f) > 1e-6, delta_f, np.nan)
    phi_est = np.clip((p_obs - f_b) / safe_delta, 0.0, 1.0)
    valid = np.isfinite(phi_est)
    return float(np.mean(phi_est[valid])) if np.any(valid) else 0.5


def smooth_vector(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return np.asarray(values, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    window = min(int(window), len(values))
    kernel = np.full(window, 1.0 / window, dtype=np.float64)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, (pad_left, pad_right), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def posterior_initial_phi(
    f_a: np.ndarray,
    f_b: np.ndarray,
    genotype_likelihoods: np.ndarray,
    ploidy: np.ndarray,
    *,
    smooth_window: int,
) -> np.ndarray:
    likelihood_a, _ = TraceSolver._likelihood_and_derivative(f_a, genotype_likelihoods, ploidy)
    likelihood_b, _ = TraceSolver._likelihood_and_derivative(f_b, genotype_likelihoods, ploidy)
    phi = likelihood_a / np.clip(likelihood_a + likelihood_b, EPS, None)
    phi = smooth_vector(phi, smooth_window)
    return np.clip(phi, 1e-4, 1.0 - 1e-4)


def posterior_initial_logits(
    frequencies: np.ndarray,
    genotype_likelihoods: np.ndarray,
    ploidy: np.ndarray,
    *,
    smooth_window: int,
) -> np.ndarray:
    likelihoods = []
    for pop_idx in range(frequencies.shape[1]):
        likelihood, _ = TraceSolver._likelihood_and_derivative(
            frequencies[:, pop_idx],
            genotype_likelihoods,
            ploidy,
        )
        likelihoods.append(likelihood)
    posterior = np.column_stack(likelihoods)
    posterior = posterior / np.clip(np.sum(posterior, axis=1, keepdims=True), EPS, None)
    if smooth_window > 1:
        posterior = np.column_stack(
            [smooth_vector(posterior[:, pop_idx], smooth_window) for pop_idx in range(posterior.shape[1])]
        )
        posterior = posterior / np.clip(np.sum(posterior, axis=1, keepdims=True), EPS, None)
    posterior = np.clip(posterior, 1e-6, 1.0)
    return np.log(posterior[:, :-1] / posterior[:, -1:])


class TraceSolver:
    """Vectorized L-BFGS-B solver for the TRACE diploid GL objective."""

    def __init__(
        self,
        d_coupling: float = 1.0,
        *,
        huber_delta: float = 0.05,
        gl_weight: float = 1.0,
        prior_weight: float = 0.0,
        prior_center: float | None = None,
        site_weights: np.ndarray | None = None,
        robust_likelihood_cap: float = 0.0,
        min_cm_delta: float = MIN_CM_DELTA,
        ftol: float = 1e-8,
        gtol: float = 1e-5,
        maxiter: int = 1000,
        maxcor: int = 20,
    ) -> None:
        if d_coupling < 0.0:
            raise ValueError("d_coupling must be non-negative.")
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive.")
        if gl_weight <= 0.0:
            raise ValueError("gl_weight must be positive.")
        if prior_weight < 0.0:
            raise ValueError("prior_weight must be non-negative.")
        if robust_likelihood_cap < 0.0:
            raise ValueError("robust_likelihood_cap must be non-negative.")
        if min_cm_delta <= 0.0:
            raise ValueError("min_cm_delta must be positive.")

        self.d_coupling = float(d_coupling)
        self.huber_delta = float(huber_delta)
        self.gl_weight = float(gl_weight)
        self.prior_weight = float(prior_weight)
        self.prior_center = float(prior_center) if prior_center is not None else None
        self.site_weights = None if site_weights is None else np.asarray(site_weights, dtype=np.float64)
        self.robust_likelihood_cap = float(robust_likelihood_cap)
        self.min_cm_delta = float(min_cm_delta)
        self.ftol = ftol
        self.gtol = gtol
        self.maxiter = maxiter
        self.maxcor = maxcor
        # Per-site reference-frequency variances (Beta posterior). When set
        # before solve(), the genotype likelihood is marginalized over panel
        # uncertainty so small/bottlenecked panels stop forcing false ancestry
        # switches. None => point-estimate frequencies (zero variance).
        self.freq_variance_a: np.ndarray | None = None
        self.freq_variance_b: np.ndarray | None = None

    @staticmethod
    def _expected_allele_freq(
        phi: np.ndarray,
        f_a: np.ndarray,
        f_b: np.ndarray,
    ) -> np.ndarray:
        return phi * f_a + (1.0 - phi) * f_b

    @staticmethod
    def _genotype_priors(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = 1.0 - p
        return q * q, 2.0 * p * q, p * p

    def _cm_deltas(self, c_m_positions: np.ndarray) -> np.ndarray:
        c_m_positions = np.asarray(c_m_positions, dtype=np.float64)
        if c_m_positions.ndim != 1:
            raise ValueError("cM positions must be a one-dimensional array.")
        if len(c_m_positions) < 2:
            raise ValueError("TRACE requires at least two SNPs.")
        if not np.all(np.isfinite(c_m_positions)):
            raise ValueError("cM positions must be finite.")

        d_cm = np.diff(c_m_positions)
        if np.any(d_cm < 0.0):
            raise ValueError("cM positions must be sorted in non-decreasing order.")
        return np.clip(d_cm, self.min_cm_delta, None)

    @staticmethod
    def _validate_inputs(
        f_a: np.ndarray,
        f_b: np.ndarray,
        genotype_likelihoods: np.ndarray,
        c_m_positions: np.ndarray,
        ploidy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        f_a = np.asarray(f_a, dtype=np.float64)
        f_b = np.asarray(f_b, dtype=np.float64)
        gls = np.asarray(genotype_likelihoods, dtype=np.float64)
        c_m_positions = np.asarray(c_m_positions, dtype=np.float64)
        ploidy = np.asarray(ploidy, dtype=np.int8)

        if f_a.ndim != 1 or f_b.ndim != 1:
            raise ValueError("f_a and f_b must be one-dimensional arrays.")
        if f_a.shape != f_b.shape:
            raise ValueError("f_a and f_b must have the same shape.")
        if gls.shape != (len(f_a), 3):
            raise ValueError("genotype_likelihoods must have shape (N, 3).")
        if c_m_positions.shape != f_a.shape:
            raise ValueError("c_m_positions must have length N.")
        if ploidy.shape != f_a.shape:
            raise ValueError("ploidy must have length N.")
        if np.any((ploidy != 1) & (ploidy != 2)):
            raise ValueError("ploidy values must be 1 or 2.")
        if np.any((f_a <= 0.0) | (f_a >= 1.0) | (f_b <= 0.0) | (f_b >= 1.0)):
            raise ValueError(
                "Allele frequencies must be strictly inside (0, 1). "
                "Use beta_smoothed_frequencies for reference-panel counts."
            )
        if np.any(gls < 0.0) or not np.all(np.isfinite(gls)):
            raise ValueError("genotype_likelihoods must be finite and non-negative.")
        if np.any(np.sum(gls, axis=1) <= 0.0):
            raise ValueError("Each genotype-likelihood row must contain positive mass.")

        # Genotype likelihoods can be arbitrarily scaled per site. Row
        # normalization improves numerical conditioning without changing phi.
        gls = gls / np.sum(gls, axis=1, keepdims=True)
        return f_a, f_b, gls, c_m_positions, ploidy

    @staticmethod
    def _likelihood_and_derivative(
        p: np.ndarray,
        genotype_likelihoods: np.ndarray,
        ploidy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        diploid = ploidy == 2
        likelihood = np.empty_like(p, dtype=np.float64)
        d_likelihood_dp = np.empty_like(p, dtype=np.float64)

        p_g0, p_g1, p_g2 = TraceSolver._genotype_priors(p)
        likelihood[diploid] = (
            genotype_likelihoods[diploid, 0] * p_g0[diploid]
            + genotype_likelihoods[diploid, 1] * p_g1[diploid]
            + genotype_likelihoods[diploid, 2] * p_g2[diploid]
        )
        d_likelihood_dp[diploid] = (
            genotype_likelihoods[diploid, 0] * (-2.0 * (1.0 - p[diploid]))
            + genotype_likelihoods[diploid, 1] * (2.0 - 4.0 * p[diploid])
            + genotype_likelihoods[diploid, 2] * (2.0 * p[diploid])
        )

        haploid = ~diploid
        likelihood[haploid] = (
            genotype_likelihoods[haploid, 0] * (1.0 - p[haploid])
            + genotype_likelihoods[haploid, 2] * p[haploid]
        )
        d_likelihood_dp[haploid] = (
            -genotype_likelihoods[haploid, 0] + genotype_likelihoods[haploid, 2]
        )
        return np.clip(likelihood, EPS, None), d_likelihood_dp

    def _genotype_terms(
        self,
        phi: np.ndarray,
        f_a: np.ndarray,
        f_b: np.ndarray,
        gls: np.ndarray,
        ploidy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Genotype likelihood Z(phi) and its 1st/2nd phi-derivatives.

        The diploid HWE genotype prior uses the mixed allele frequency
        ``p = phi*f_a + (1-phi)*f_b``, marginalized over the reference-frequency
        Beta posteriors. Marginalization leaves the mean ``E[p] = p`` unchanged
        and only inflates the second moment::

            E[p^2] = p^2 + phi^2 * Var(f_a) + (1-phi)^2 * Var(f_b)

        so a noisy (small-n) panel flattens the per-site phi likelihood instead
        of pulling ancestry toward whichever source the point estimate happens
        to resemble. With zero variance this is exactly the point-estimate
        model. Haploid sites are linear in p (mean only), so frequency variance
        does not affect them.
        """
        var_a = 0.0 if self.freq_variance_a is None else self.freq_variance_a
        var_b = 0.0 if self.freq_variance_b is None else self.freq_variance_b

        p = phi * f_a + (1.0 - phi) * f_b
        dp = f_a - f_b                       # dE[p]/dphi, constant in phi
        one_minus = 1.0 - phi
        s = p * p + phi * phi * var_a + one_minus * one_minus * var_b
        ds = 2.0 * p * dp + 2.0 * phi * var_a - 2.0 * one_minus * var_b
        d2s = 2.0 * dp * dp + 2.0 * var_a + 2.0 * var_b

        z = np.empty_like(phi, dtype=np.float64)
        dz = np.empty_like(phi, dtype=np.float64)
        d2z = np.zeros_like(phi, dtype=np.float64)

        g0, g1, g2 = gls[:, 0], gls[:, 1], gls[:, 2]
        # Diploid: P(g=0)=1-2p+s, P(g=1)=2(p-s), P(g=2)=s.
        dip = ploidy == 2
        z[dip] = (
            g0[dip] * (1.0 - 2.0 * p[dip] + s[dip])
            + g1[dip] * 2.0 * (p[dip] - s[dip])
            + g2[dip] * s[dip]
        )
        dz[dip] = (
            g0[dip] * (-2.0 * dp[dip] + ds[dip])
            + g1[dip] * 2.0 * (dp[dip] - ds[dip])
            + g2[dip] * ds[dip]
        )
        d2z[dip] = (g0[dip] - 2.0 * g1[dip] + g2[dip]) * d2s[dip]

        # Haploid: P(0)=1-p, P(1)=p; linear in phi, so d2z stays zero.
        hap = ~dip
        z[hap] = g0[hap] * (1.0 - p[hap]) + g2[hap] * p[hap]
        dz[hap] = (g2[hap] - g0[hap]) * dp[hap]

        return np.clip(z, EPS, None), dz, d2z

    def action_functional(
        self,
        phi: np.ndarray,
        f_a: np.ndarray,
        f_b: np.ndarray,
        genotype_likelihoods: np.ndarray,
        d_cm: np.ndarray,
        ploidy: np.ndarray,
        prior_mu: float,
    ) -> float:
        dphi = np.diff(phi)
        d = self.huber_delta
        # Standard pseudo-Huber: ~Delta^2/(2d) for |Delta|<<d (smooths noise),
        # ~|Delta| for |Delta|>>d (cheap domain walls). The linear-regime slope
        # is 1, so d_coupling is the true total-variation strength and does not
        # vanish with d (the previous d^2 prefactor made smoothing ~d weak).
        kinetic = self.d_coupling * np.sum(
            (np.sqrt(d * d + dphi * dphi) - d) / d_cm
        )

        likelihood, _dz, _d2z = self._genotype_terms(
            phi, f_a, f_b, genotype_likelihoods, ploidy
        )
        nll = -np.log(likelihood)
        if self.robust_likelihood_cap > 0.0:
            nll = np.minimum(nll, self.robust_likelihood_cap)
        if self.site_weights is not None:
            nll = self.site_weights * nll
        potential = self.gl_weight * np.sum(nll)
        prior = self.prior_weight * float(np.sum((phi - prior_mu) ** 2))
        return float(kinetic + potential + prior)

    def action_gradient(
        self,
        phi: np.ndarray,
        f_a: np.ndarray,
        f_b: np.ndarray,
        genotype_likelihoods: np.ndarray,
        d_cm: np.ndarray,
        ploidy: np.ndarray,
        prior_mu: float,
    ) -> np.ndarray:
        grad = np.zeros_like(phi, dtype=np.float64)

        dphi = np.diff(phi)
        d = self.huber_delta
        edge_force = self.d_coupling * (dphi / np.sqrt(d * d + dphi * dphi)) / d_cm
        grad[:-1] -= edge_force
        grad[1:] += edge_force

        likelihood, d_likelihood_dphi, _d2z = self._genotype_terms(
            phi, f_a, f_b, genotype_likelihoods, ploidy
        )

        pull = d_likelihood_dphi / likelihood
        if self.robust_likelihood_cap > 0.0:
            pull = np.where(-np.log(likelihood) < self.robust_likelihood_cap, pull, 0.0)
        if self.site_weights is not None:
            pull = self.site_weights * pull
        grad -= self.gl_weight * pull
        if self.prior_weight > 0.0:
            grad += 2.0 * self.prior_weight * (phi - prior_mu)
        return grad

    def hessian_standard_errors(
        self,
        phi: np.ndarray,
        f_a: np.ndarray,
        f_b: np.ndarray,
        genotype_likelihoods: np.ndarray,
        d_cm: np.ndarray,
        ploidy: np.ndarray,
    ) -> np.ndarray:
        """Per-site standard errors from diag(inv(H)) of the tridiagonal Hessian.

        This accounts for nearest-neighbor kinetic coupling exactly for the
        binary 1D solver. The result is still a local Laplace approximation,
        not a fully calibrated confidence interval.
        """
        d = self.huber_delta
        dphi = np.diff(phi)
        huber_curv = (d * d) / (d * d + dphi * dphi) ** 1.5  # shape (n-1,)

        kinetic_diag = np.zeros(len(phi))
        edge_curv = self.d_coupling * huber_curv / d_cm
        kinetic_diag[:-1] += edge_curv
        kinetic_diag[1:] += edge_curv
        kinetic_offdiag = -edge_curv

        # Genotype likelihood and its phi-derivatives, marginalized over
        # reference-frequency uncertainty (same model as the objective).
        z, dz, d2z = self._genotype_terms(phi, f_a, f_b, genotype_likelihoods, ploidy)

        # d²(-log Z)/dphi² = (dZ/dphi)²/Z² - (d²Z/dphi²)/Z.
        # This can be locally negative for uncertain diploid mixture sites, so
        # the LDL routine below adds jitter if needed.
        potential_diag = self.gl_weight * ((dz / z) ** 2 - d2z / z)
        if self.robust_likelihood_cap > 0.0:
            potential_diag = np.where(-np.log(z) < self.robust_likelihood_cap, potential_diag, 0.0)
        if self.site_weights is not None:
            potential_diag = self.site_weights * potential_diag

        h_diag = kinetic_diag + potential_diag + 2.0 * self.prior_weight
        covariance_diag = inverse_tridiagonal_diag(h_diag, kinetic_offdiag)
        # phi is bounded to [0, 1], so a local uncertainty larger than 0.5 is
        # not informative on the probability scale. Cap it and let users treat
        # 0.5 as "effectively unconstrained".
        return np.minimum(np.sqrt(covariance_diag), 0.5)

    def hessian_diagonal(
        self,
        phi: np.ndarray,
        f_a: np.ndarray,
        f_b: np.ndarray,
        genotype_likelihoods: np.ndarray,
        d_cm: np.ndarray,
        ploidy: np.ndarray,
    ) -> np.ndarray:
        return self.hessian_standard_errors(phi, f_a, f_b, genotype_likelihoods, d_cm, ploidy)

    def solve(
        self,
        f_a: np.ndarray,
        f_b: np.ndarray,
        genotype_likelihoods: np.ndarray,
        c_m_positions: np.ndarray,
        *,
        ploidy: np.ndarray | None = None,
        initial_phi: np.ndarray | None = None,
    ) -> TraceResult:
        if ploidy is None:
            ploidy = np.full(len(f_a), 2, dtype=np.int8)
        f_a, f_b, genotype_likelihoods, c_m_positions, ploidy = self._validate_inputs(
            f_a,
            f_b,
            genotype_likelihoods,
            c_m_positions,
            ploidy,
        )
        d_cm = self._cm_deltas(c_m_positions)
        if self.site_weights is not None and self.site_weights.shape != f_a.shape:
            raise ValueError("site_weights must have length N.")

        if initial_phi is None:
            phi0 = np.full(len(f_a), 0.5, dtype=np.float64)
        else:
            phi0 = np.asarray(initial_phi, dtype=np.float64)
            if phi0.shape != f_a.shape:
                raise ValueError("initial_phi must have length N.")
            phi0 = np.clip(phi0, 0.0, 1.0)

        if self.prior_weight > 0.0 and self.prior_center is None:
            prior_mu = _estimate_prior_center(f_a, f_b, genotype_likelihoods, ploidy)
        else:
            prior_mu = self.prior_center if self.prior_center is not None else 0.0

        start = time.perf_counter()
        result = minimize(
            fun=self.action_functional,
            x0=phi0,
            args=(f_a, f_b, genotype_likelihoods, d_cm, ploidy, prior_mu),
            jac=self.action_gradient,
            bounds=Bounds(0.0, 1.0),
            method="L-BFGS-B",
            options={
                "ftol": self.ftol,
                "gtol": self.gtol,
                "maxiter": self.maxiter,
                "maxcor": self.maxcor,
            },
        )
        seconds = time.perf_counter() - start

        return TraceResult(
            phi=result.x,
            action=float(result.fun),
            converged=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            seconds=seconds,
        )


class TraceMultidimensionalSolver:
    """
    Multi-population TRACE solver for K reference panels.

    The optimizer works on N x (K - 1) unconstrained logits. A zero-valued
    baseline column is appended before softmax, which removes the redundant
    shift degree of freedom in a full N x K softmax parameterization.
    """

    def __init__(
        self,
        d_coupling: float = 1.0,
        *,
        huber_delta: float = 0.05,
        gl_weight: float = 1.0,
        logit_l2: float = 1e-3,
        prior_weight: float = 0.0,
        prior_center: np.ndarray | None = None,
        unknown_index: int | None = None,
        unknown_penalty: float = 0.0,
        site_weights: np.ndarray | None = None,
        robust_likelihood_cap: float = 0.0,
        min_cm_delta: float = MIN_CM_DELTA,
        ftol: float = 1e-8,
        gtol: float = 1e-5,
        maxiter: int = 1000,
        maxcor: int = 20,
    ) -> None:
        if d_coupling < 0.0:
            raise ValueError("d_coupling must be non-negative.")
        if huber_delta <= 0.0:
            raise ValueError("huber_delta must be positive.")
        if gl_weight <= 0.0:
            raise ValueError("gl_weight must be positive.")
        if logit_l2 < 0.0:
            raise ValueError("logit_l2 must be non-negative.")
        if prior_weight < 0.0:
            raise ValueError("prior_weight must be non-negative.")
        if unknown_penalty < 0.0:
            raise ValueError("unknown_penalty must be non-negative.")
        if robust_likelihood_cap < 0.0:
            raise ValueError("robust_likelihood_cap must be non-negative.")
        if min_cm_delta <= 0.0:
            raise ValueError("min_cm_delta must be positive.")

        self.d_coupling = float(d_coupling)
        self.huber_delta = float(huber_delta)
        self.gl_weight = float(gl_weight)
        self.logit_l2 = float(logit_l2)
        self.prior_weight = float(prior_weight)
        self.prior_center = None if prior_center is None else np.asarray(prior_center, dtype=np.float64)
        self.unknown_index = unknown_index
        self.unknown_penalty = float(unknown_penalty)
        self.site_weights = None if site_weights is None else np.asarray(site_weights, dtype=np.float64)
        self.robust_likelihood_cap = float(robust_likelihood_cap)
        self.min_cm_delta = float(min_cm_delta)
        self.ftol = ftol
        self.gtol = gtol
        self.maxiter = maxiter
        self.maxcor = maxcor

    @staticmethod
    def _softmax_from_logits(logits: np.ndarray) -> np.ndarray:
        baseline = np.zeros((logits.shape[0], 1), dtype=logits.dtype)
        theta = np.column_stack((logits, baseline))
        theta = theta - np.max(theta, axis=1, keepdims=True)
        exp_theta = np.exp(theta)
        return exp_theta / np.sum(exp_theta, axis=1, keepdims=True)

    @staticmethod
    def _expected_allele_freq(phi: np.ndarray, frequencies: np.ndarray) -> np.ndarray:
        return np.sum(phi * frequencies, axis=1)

    @staticmethod
    def _genotype_priors(p: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q = 1.0 - p
        return q * q, 2.0 * p * q, p * p

    def _cm_deltas(self, c_m_positions: np.ndarray) -> np.ndarray:
        c_m_positions = np.asarray(c_m_positions, dtype=np.float64)
        if c_m_positions.ndim != 1:
            raise ValueError("cM positions must be one-dimensional.")
        if len(c_m_positions) < 2:
            raise ValueError("TRACE requires at least two SNPs.")
        if not np.all(np.isfinite(c_m_positions)):
            raise ValueError("cM positions must be finite.")

        d_cm = np.diff(c_m_positions)
        if np.any(d_cm < 0.0):
            raise ValueError("cM positions must be sorted in non-decreasing order.")
        return np.clip(d_cm, self.min_cm_delta, None)

    @staticmethod
    def _validate_inputs(
        frequencies: np.ndarray,
        genotype_likelihoods: np.ndarray,
        c_m_positions: np.ndarray,
        ploidy: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frequencies = np.asarray(frequencies, dtype=np.float64)
        gls = np.asarray(genotype_likelihoods, dtype=np.float64)
        c_m_positions = np.asarray(c_m_positions, dtype=np.float64)
        ploidy = np.asarray(ploidy, dtype=np.int8)

        if frequencies.ndim != 2:
            raise ValueError("frequencies must have shape (N, K).")
        n_sites, n_populations = frequencies.shape
        if n_populations < 2:
            raise ValueError("frequencies must include at least two populations.")
        if gls.shape != (n_sites, 3):
            raise ValueError("genotype_likelihoods must have shape (N, 3).")
        if c_m_positions.shape != (n_sites,):
            raise ValueError("c_m_positions must have length N.")
        if ploidy.shape != (n_sites,):
            raise ValueError("ploidy must have length N.")
        if np.any((ploidy != 1) & (ploidy != 2)):
            raise ValueError("ploidy values must be 1 or 2.")
        if np.any((frequencies <= 0.0) | (frequencies >= 1.0)):
            raise ValueError(
                "All reference frequencies must be strictly inside (0, 1). "
                "Use beta_smoothed_frequencies on reference-panel counts."
            )
        if np.any(gls < 0.0) or not np.all(np.isfinite(gls)):
            raise ValueError("genotype_likelihoods must be finite and non-negative.")
        if np.any(np.sum(gls, axis=1) <= 0.0):
            raise ValueError("Each genotype-likelihood row must contain positive mass.")

        gls = gls / np.sum(gls, axis=1, keepdims=True)
        return frequencies, gls, c_m_positions, ploidy

    def _grad_phi(
        self,
        phi: np.ndarray,
        frequencies: np.ndarray,
        genotype_likelihoods: np.ndarray,
        d_cm: np.ndarray,
        ploidy: np.ndarray,
    ) -> np.ndarray:
        grad_phi = np.zeros_like(phi, dtype=np.float64)

        dphi = np.diff(phi, axis=0)
        d = self.huber_delta
        edge_force = self.d_coupling * (dphi / np.sqrt(d * d + dphi * dphi)) / d_cm[:, None]
        grad_phi[:-1, :] -= edge_force
        grad_phi[1:, :] += edge_force

        p = self._expected_allele_freq(phi, frequencies)
        likelihood, d_likelihood_dp = TraceSolver._likelihood_and_derivative(
            p,
            genotype_likelihoods,
            ploidy,
        )
        likelihood_pull = (d_likelihood_dp / likelihood)[:, None] * frequencies
        if self.robust_likelihood_cap > 0.0:
            active = (-np.log(likelihood) < self.robust_likelihood_cap).astype(np.float64)
            likelihood_pull = active[:, None] * likelihood_pull
        if self.site_weights is not None:
            likelihood_pull = self.site_weights[:, None] * likelihood_pull
        grad_phi -= self.gl_weight * likelihood_pull
        if self.prior_weight > 0.0:
            center = self._effective_prior_center(phi)
            grad_phi += 2.0 * self.prior_weight * (phi - center)
        if self.unknown_index is not None and self.unknown_penalty > 0.0:
            grad_phi[:, self.unknown_index] += self.unknown_penalty
        return grad_phi

    def _effective_prior_center(self, phi: np.ndarray) -> np.ndarray:
        if self.prior_center is not None:
            center = self.prior_center
            if center.ndim == 1:
                return center[None, :]
            return center
        return np.full((1, phi.shape[1]), 1.0 / phi.shape[1], dtype=phi.dtype)

    def action_functional(
        self,
        logits_flat: np.ndarray,
        frequencies: np.ndarray,
        genotype_likelihoods: np.ndarray,
        d_cm: np.ndarray,
        ploidy: np.ndarray,
        n_sites: int,
        n_populations: int,
    ) -> float:
        logits = logits_flat.reshape((n_sites, n_populations - 1))
        phi = self._softmax_from_logits(logits)

        dphi = np.diff(phi, axis=0)
        d = self.huber_delta
        huber_loss = np.sqrt(d * d + dphi * dphi) - d
        kinetic = self.d_coupling * np.sum(np.sum(huber_loss, axis=1) / d_cm)
        logit_penalty = 0.5 * self.logit_l2 * np.sum(logits * logits)

        p = self._expected_allele_freq(phi, frequencies)
        likelihood, _d_likelihood_dp = TraceSolver._likelihood_and_derivative(
            p,
            genotype_likelihoods,
            ploidy,
        )
        nll = -np.log(likelihood)
        if self.robust_likelihood_cap > 0.0:
            nll = np.minimum(nll, self.robust_likelihood_cap)
        if self.site_weights is not None:
            nll = self.site_weights * nll
        potential = self.gl_weight * np.sum(nll)
        prior = 0.0
        if self.prior_weight > 0.0:
            center = self._effective_prior_center(phi)
            prior = self.prior_weight * float(np.sum((phi - center) ** 2))
        unknown_prior = 0.0
        if self.unknown_index is not None and self.unknown_penalty > 0.0:
            unknown_prior = self.unknown_penalty * float(np.sum(phi[:, self.unknown_index]))
        return float(kinetic + logit_penalty + potential + prior + unknown_prior)

    def action_gradient(
        self,
        logits_flat: np.ndarray,
        frequencies: np.ndarray,
        genotype_likelihoods: np.ndarray,
        d_cm: np.ndarray,
        ploidy: np.ndarray,
        n_sites: int,
        n_populations: int,
    ) -> np.ndarray:
        logits = logits_flat.reshape((n_sites, n_populations - 1))
        phi = self._softmax_from_logits(logits)
        grad_phi = self._grad_phi(phi, frequencies, genotype_likelihoods, d_cm, ploidy)

        weighted_sum = np.sum(grad_phi * phi, axis=1, keepdims=True)
        grad_theta_full = phi * (grad_phi - weighted_sum)
        grad_logits = grad_theta_full[:, :-1] + self.logit_l2 * logits
        return grad_logits.ravel()

    def solve(
        self,
        frequencies: np.ndarray,
        genotype_likelihoods: np.ndarray,
        c_m_positions: np.ndarray,
        *,
        ploidy: np.ndarray | None = None,
        initial_logits: np.ndarray | None = None,
    ) -> TraceResult:
        if ploidy is None:
            ploidy = np.full(len(frequencies), 2, dtype=np.int8)
        frequencies, genotype_likelihoods, c_m_positions, ploidy = self._validate_inputs(
            frequencies,
            genotype_likelihoods,
            c_m_positions,
            ploidy,
        )
        d_cm = self._cm_deltas(c_m_positions)
        n_sites, n_populations = frequencies.shape
        if self.site_weights is not None and self.site_weights.shape != (n_sites,):
            raise ValueError("site_weights must have length N.")
        if self.prior_center is not None and self.prior_center.shape != (n_populations,):
            raise ValueError("prior_center for K-state solver must have length K.")
        if self.unknown_index is not None and not 0 <= self.unknown_index < n_populations:
            raise ValueError("unknown_index must be a valid population column.")

        if initial_logits is None:
            logits0 = np.zeros((n_sites, n_populations - 1), dtype=np.float64)
        else:
            logits0 = np.asarray(initial_logits, dtype=np.float64)
            if logits0.shape != (n_sites, n_populations - 1):
                raise ValueError("initial_logits must have shape (N, K - 1).")

        start = time.perf_counter()
        result = minimize(
            fun=self.action_functional,
            x0=logits0.ravel(),
            args=(frequencies, genotype_likelihoods, d_cm, ploidy, n_sites, n_populations),
            jac=self.action_gradient,
            method="L-BFGS-B",
            options={
                "ftol": self.ftol,
                "gtol": self.gtol,
                "maxiter": self.maxiter,
                "maxcor": self.maxcor,
            },
        )
        seconds = time.perf_counter() - start
        phi = self._softmax_from_logits(result.x.reshape((n_sites, n_populations - 1)))

        return TraceResult(
            phi=phi,
            action=float(result.fun),
            converged=bool(result.success),
            message=str(result.message),
            iterations=int(result.nit),
            seconds=seconds,
        )


def simulate_reference_panel(
    rng: np.random.Generator,
    true_freq: np.ndarray,
    n_haplotypes: int,
) -> tuple[np.ndarray, np.ndarray]:
    derived_counts = rng.binomial(n_haplotypes, true_freq)
    allele_counts = np.full(np.shape(true_freq), n_haplotypes, dtype=np.float64)
    return derived_counts.astype(np.float64), allele_counts


def sample_diploid_genotypes(
    rng: np.random.Generator,
    allele_frequencies: np.ndarray,
) -> np.ndarray:
    p_g0 = (1.0 - allele_frequencies) ** 2
    p_g1 = 2.0 * allele_frequencies * (1.0 - allele_frequencies)
    p_g2 = allele_frequencies**2
    genotype_probs = np.column_stack((p_g0, p_g1, p_g2))
    cumulative = np.cumsum(genotype_probs, axis=1)
    return (rng.random(len(allele_frequencies))[:, None] > cumulative[:, :-1]).sum(axis=1)


def genotype_likelihoods_from_calls(
    genotypes: np.ndarray,
    *,
    error: float = 0.01,
) -> np.ndarray:
    if not 0.0 <= error < 1.0:
        raise ValueError("error must be in [0, 1).")
    gls = np.full((len(genotypes), 3), error / 2.0, dtype=np.float64)
    gls[np.arange(len(genotypes)), genotypes] = 1.0 - error
    return gls


def genotype_likelihood_from_gt(alleles: list[int], ploidy: int, *, error: float = 0.01) -> np.ndarray:
    if any(allele < 0 for allele in alleles):
        return np.array([1.0, 1.0, 1.0], dtype=np.float64)
    alt_count = int(np.sum(alleles))
    if ploidy == 1:
        if alt_count == 0:
            return np.array([1.0 - error, error / 2.0, error / 2.0], dtype=np.float64)
        return np.array([error / 2.0, error / 2.0, 1.0 - error], dtype=np.float64)
    return genotype_likelihoods_from_calls(np.array([alt_count]), error=error)[0]


def simulate_breakpoint(
    n_snps: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)

    true_f_a = rng.uniform(0.6, 0.9, n_snps)
    true_f_b = rng.uniform(0.1, 0.4, n_snps)

    a_counts, a_total = simulate_reference_panel(rng, true_f_a, n_haplotypes=200)
    b_counts, b_total = simulate_reference_panel(rng, true_f_b, n_haplotypes=200)
    f_a = beta_smoothed_frequencies(a_counts, a_total)
    f_b = beta_smoothed_frequencies(b_counts, b_total)

    c_m_steps = rng.exponential(scale=0.001, size=n_snps - 1)
    c_m_positions = np.concatenate(([0.0], np.cumsum(c_m_steps)))

    midpoint = n_snps // 2
    local_p = np.empty(n_snps, dtype=np.float64)
    local_p[:midpoint] = true_f_a[:midpoint]
    local_p[midpoint:] = true_f_b[midpoint:]

    genotypes = sample_diploid_genotypes(rng, local_p)
    gls = genotype_likelihoods_from_calls(genotypes)
    return f_a, f_b, gls, c_m_positions


def simulate_multik_mosaic(
    n_snps: int,
    k_populations: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if k_populations < 2:
        raise ValueError("k_populations must be at least 2.")

    rng = np.random.default_rng(seed)
    true_frequencies = rng.beta(0.7, 0.7, size=(n_snps, k_populations))
    true_frequencies = np.clip(true_frequencies, 0.02, 0.98)

    derived_counts, allele_counts = simulate_reference_panel(
        rng,
        true_frequencies,
        n_haplotypes=200,
    )
    frequencies = beta_smoothed_frequencies(derived_counts, allele_counts)

    c_m_steps = rng.exponential(scale=0.001, size=n_snps - 1)
    c_m_positions = np.concatenate(([0.0], np.cumsum(c_m_steps)))

    segment_ids = np.minimum(
        (np.arange(n_snps) * k_populations) // n_snps,
        k_populations - 1,
    )
    local_p = true_frequencies[np.arange(n_snps), segment_ids]
    genotypes = sample_diploid_genotypes(rng, local_p)
    gls = genotype_likelihoods_from_calls(genotypes)
    return frequencies, gls, c_m_positions


def _simulate_admixture_stitched(
    g_true: float,
    n_chroms: int,
    chrom_cm: float,
    *,
    grid_step_cm: float = 0.05,
    seed: int = 0,
) -> tuple[list[tuple[str, np.ndarray, np.ndarray]], dict[str, tuple[np.ndarray, np.ndarray]]]:
    """
    Simulate diploid ancestry under the Poisson-breakpoint admixture model.

    Breakpoints occur at rate g_true per Morgan; ancestry is independently
    re-assigned (not alternating) at each break, matching the admixture-model
    ACF = exp(-g*r) that estimate_g_from_autocorrelation assumes.
    """
    rng = np.random.default_rng(seed)
    n_pts = int(chrom_cm / grid_step_cm) + 1
    pos_cm = np.linspace(0.0, chrom_cm, n_pts)
    # Positions in bp (1 bp/cM proxy keeps genetic_map trivial)
    pos_bp = (pos_cm * 1_000).astype(np.int64)

    stitched = []
    genetic_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for i in range(n_chroms):
        chrom = f"chr{i + 1}"
        mat = np.zeros(n_pts)
        pat = np.zeros(n_pts)
        for strand in (mat, pat):
            ptr = 0.0
            while ptr < chrom_cm:
                length = rng.exponential(100.0 / g_true)  # cM
                new_ancestry = float(rng.integers(0, 2))   # random re-assignment
                strand[(pos_cm >= ptr) & (pos_cm < ptr + length)] = new_ancestry
                ptr += length
        phi = 0.5 * (mat + pat)
        stitched.append((chrom, pos_bp.copy(), phi[:, None]))
        genetic_map[chrom] = (pos_bp.astype(np.float64), pos_cm.copy())
    return stitched, genetic_map


def run_acf_check(*, n_chroms: int = 10, g_true: float = 25.0, chrom_cm: float = 200.0) -> None:
    stitched, genetic_map = _simulate_admixture_stitched(g_true, n_chroms, chrom_cm, seed=42)
    result = estimate_g_from_autocorrelation(stitched, genetic_map)
    g_est = result.weighted_g
    ratio = g_est / g_true
    print(
        f"ACF estimator check: g_true={g_true:.0f}, "
        f"model={result.model}, "
        f"weighted_g={g_est:.1f}, "
        f"ratio={ratio:.2f}"
    )
    if not 0.5 <= ratio <= 2.0:
        raise RuntimeError(
            f"ACF estimator bias too large: weighted_g={g_est:.1f} vs true g={g_true:.0f} "
            f"(ratio={ratio:.2f}); expected 0.5–2.0"
        )
    print("  OK")


def run_gradient_check(k_populations: int) -> None:
    if k_populations == 2:
        f_a, f_b, gls, c_m_positions = simulate_breakpoint(100, seed=7)
        solver = TraceSolver(d_coupling=0.001)
        d_cm = solver._cm_deltas(c_m_positions)
        phi = np.linspace(0.2, 0.8, len(f_a))
        ploidy = np.full(len(f_a), 2, dtype=np.int8)
        prior_mu = 0.5
        error = check_grad(
            lambda z: solver.action_functional(z, f_a, f_b, gls, d_cm, ploidy, prior_mu),
            lambda z: solver.action_gradient(z, f_a, f_b, gls, d_cm, ploidy, prior_mu),
            phi,
        )
        print(f"finite-difference gradient error: {error:.6e}")
        return

    frequencies, gls, c_m_positions = simulate_multik_mosaic(
        100,
        k_populations,
        seed=7,
    )
    solver = TraceMultidimensionalSolver(d_coupling=0.001)
    d_cm = solver._cm_deltas(c_m_positions)
    n_sites, n_populations = frequencies.shape
    logits = np.zeros((n_sites, n_populations - 1), dtype=np.float64)
    ploidy = np.full(n_sites, 2, dtype=np.int8)
    error = check_grad(
        lambda z: solver.action_functional(z, frequencies, gls, d_cm, ploidy, n_sites, n_populations),
        lambda z: solver.action_gradient(z, frequencies, gls, d_cm, ploidy, n_sites, n_populations),
        logits.ravel(),
    )
    print(f"finite-difference gradient error: {error:.6e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACE ancestry inference.")
    parser.add_argument("--vcf", help="Input VCF/BCF path. Requires cyvcf2.")
    parser.add_argument("--bam", help="Input BAM/CRAM path. Requires pysam and ref/alt columns.")
    parser.add_argument("--sample", help="Sample name for multi-sample VCFs.")
    parser.add_argument("--reference-panel", help="Tab-delimited reference panel table.")
    parser.add_argument("--reference-panel-zarr", help="Zarr-backed reference panel.")
    parser.add_argument("--build-reference-zarr", help="Build this Zarr reference panel from --reference-panel.")
    parser.add_argument("--reference-zarr-chunk-rows", type=int, default=100_000)
    parser.add_argument("--genetic-map", help="PLINK .map or header genetic-map table.")
    parser.add_argument("--output", help="Output TSV path for inferred ancestry fractions.")
    parser.add_argument("--output-zarr", help="Output Zarr group path for inferred ancestry fractions.")
    parser.add_argument("--output-bed", help="Output BED file for hard ancestry tract calls.")
    parser.add_argument("--output-msp", help="Output RFMix-style .msp.tsv for hard ancestry tract calls.")
    parser.add_argument("--summary", help="Output global ancestry summary path (default: derived from --output).")
    parser.add_argument("--hard-call-threshold", type=float, default=0.8,
                        help="Confidence threshold for BED/MSP hard calls (default: 0.8).")
    parser.add_argument("--phased", action="store_true",
                        help="Split phased VCF into two independent haploid tracks.")
    parser.add_argument("--report-zarr", help="Input TRACE output Zarr group for HTML report generation.")
    parser.add_argument("--report-output", help="Output HTML report path.")
    parser.add_argument("--chunk-size", type=int, default=100_000)
    parser.add_argument("--chunk-overlap", type=int, default=10_000)
    parser.add_argument("--max-imputation-gap-cm", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    parser.add_argument("--base-error", type=float, default=0.01)
    parser.add_argument("--min-base-quality", type=int, default=20)
    parser.add_argument("--sample-sex", choices=("unknown", "male", "female"), default="unknown")
    parser.add_argument("--haploid-chroms", default="")
    parser.add_argument("--fasta", help="Reference FASTA for CRAM input.")
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--freq-shrinkage", type=float, default=8.0,
        help="Empirical-Bayes shrinkage of count-format reference frequencies "
        "toward the cross-population mean, scaled by panel size (w=n/(n+S)). "
        "Limits false ancestry from small/bottlenecked panels; large panels are "
        "barely affected. 0 disables. Count-format panels only.",
    )
    parser.add_argument("--n-snps", type=int, default=100_000)
    parser.add_argument("--d", type=float, default=0.05)
    parser.add_argument("--auto-d", action="store_true")
    parser.add_argument(
        "--auto-params", action="store_true",
        help="Cross-validate D, GL weight, and robust likelihood cap on masked SNPs.",
    )
    parser.add_argument("--d-grid", default="0.003,0.01,0.03,0.1,0.3,1.0,3.0")
    parser.add_argument("--gl-weight-grid", default="0.15,0.25,0.5,1.0")
    parser.add_argument("--robust-cap-grid", default="0,1.5,2.5,4.0")
    parser.add_argument("--calibration-mask-fraction", type=float, default=0.10)
    parser.add_argument(
        "--calibration-score",
        choices=("likelihood", "cm-likelihood"),
        default="cm-likelihood",
        help="Held-out objective for auto calibration. cm-likelihood weights held-out SNPs by cM span.",
    )
    parser.add_argument(
        "--calibration-mask-mode",
        choices=("snp", "block"),
        default="block",
        help="Mask random SNPs or contiguous cM blocks during auto calibration.",
    )
    parser.add_argument(
        "--calibration-block-cm",
        type=float,
        default=0.05,
        help="Approximate cM length for block masking. <=0 uses mask fraction of calibration chunk span.",
    )
    parser.add_argument("--calibration-seed", type=int, default=123)
    parser.add_argument("--k", type=int, default=3)
    parser.add_argument("--logit-l2", type=float, default=1e-3)
    parser.add_argument(
        "--gl-weight", type=float, default=1.0,
        help="Scale factor for genotype-likelihood potential (tau). "
             "<1 trusts GLs less (smoother); >1 trusts more (sharper). Default: 1.0.",
    )
    parser.add_argument(
        "--prior-weight", type=float, default=0.0,
        help="L2 penalty pulling phi toward the genome-wide ancestry mean. "
             "Use for clinal/isolation-by-distance scenarios. Default: 0 (disabled).",
    )
    parser.add_argument(
        "--prior-center", type=float, default=None,
        help="Center for --prior-weight penalty. "
             "Defaults to the GL-weighted ancestry estimate (auto).",
    )
    parser.add_argument(
        "--site-weighting", action="store_true",
        help="Down-weight SNPs with weak or uncertain reference-panel allele-frequency separation.",
    )
    parser.add_argument(
        "--posterior-init", action="store_true",
        help="Initialize optimization from per-site ancestry posteriors instead of a flat field.",
    )
    parser.add_argument(
        "--init-smooth-window", type=int, default=25,
        help="Moving-average window for --posterior-init. Default: 25 SNPs.",
    )
    parser.add_argument(
        "--robust-likelihood-cap", type=float, default=0.0,
        help="Cap each SNP's negative log-likelihood contribution. 0 disables. "
             "Useful for bad markers/panel mismatch.",
    )
    parser.add_argument(
        "--unknown-state", action="store_true",
        help="Append an unknown/ghost ancestry state to the reference frequencies.",
    )
    parser.add_argument(
        "--unknown-mode",
        choices=("flat", "midpoint", "shrink-midpoint"),
        default="shrink-midpoint",
        help="Allele-frequency model for --unknown-state.",
    )
    parser.add_argument("--unknown-frequency", type=float, default=0.5)
    parser.add_argument("--unknown-penalty", type=float, default=0.02)
    parser.add_argument("--unknown-label", default="UNKNOWN")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--maxiter", type=int, default=1000)
    parser.add_argument("--gradient-check", action="store_true")
    parser.add_argument("--acf-check", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(args.verbose)
    if args.build_reference_zarr:
        if args.reference_panel is None:
            raise SystemExit("--build-reference-zarr requires --reference-panel")
        convert_reference_tsv_to_zarr(
            args.reference_panel,
            args.build_reference_zarr,
            alpha=args.alpha,
            beta=args.beta,
            freq_shrinkage=args.freq_shrinkage,
            chunk_rows=args.reference_zarr_chunk_rows,
        )
        LOGGER.info("wrote %s", args.build_reference_zarr)
        return

    if args.report_zarr:
        if args.report_output is None:
            raise SystemExit("--report-zarr requires --report-output")
        generate_zarr_report(args.report_zarr, args.report_output)
        LOGGER.info("wrote %s", args.report_output)
        return

    if args.gradient_check:
        run_gradient_check(args.k)
        return

    if args.acf_check:
        run_acf_check()
        return

    if args.vcf or args.bam:
        missing = [
            name
            for name, value in {
                "--genetic-map": args.genetic_map,
            }.items()
            if value is None
        ]
        if args.reference_panel is None and args.reference_panel_zarr is None:
            missing.append("--reference-panel or --reference-panel-zarr")
        if args.output is None and args.output_zarr is None:
            missing.append("--output or --output-zarr")
        if missing:
            raise SystemExit(f"Missing required file-pipeline arguments: {', '.join(missing)}")
        run_file_pipeline(
            vcf_path=args.vcf,
            bam_path=args.bam,
            sample=args.sample,
            reference_panel_path=args.reference_panel or "",
            reference_panel_zarr_path=args.reference_panel_zarr,
            genetic_map_path=args.genetic_map,
            output_path=args.output or "",
            output_zarr_path=args.output_zarr,
            output_bed_path=args.output_bed,
            output_msp_path=args.output_msp,
            summary_path=args.summary,
            hard_call_threshold=args.hard_call_threshold,
            phased=args.phased,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_imputation_gap_cm=args.max_imputation_gap_cm,
            workers=args.workers,
            d_coupling=args.d,
            auto_d=args.auto_d,
            auto_params=args.auto_params,
            d_grid=parse_d_grid(args.d_grid),
            gl_weight_grid=parse_positive_grid(args.gl_weight_grid, name="GL-weight grid"),
            robust_cap_grid=parse_positive_grid(args.robust_cap_grid, name="robust-cap grid", allow_zero=True),
            calibration_mask_fraction=args.calibration_mask_fraction,
            calibration_seed=args.calibration_seed,
            calibration_score=args.calibration_score,
            calibration_mask_mode=args.calibration_mask_mode,
            calibration_block_cm=args.calibration_block_cm,
            logit_l2=args.logit_l2,
            gl_weight=args.gl_weight,
            prior_weight=args.prior_weight,
            prior_center=args.prior_center,
            site_weighting=args.site_weighting,
            posterior_init=args.posterior_init,
            init_smooth_window=args.init_smooth_window,
            robust_likelihood_cap=args.robust_likelihood_cap,
            unknown_state=args.unknown_state,
            unknown_mode=args.unknown_mode,
            unknown_frequency=args.unknown_frequency,
            unknown_penalty=args.unknown_penalty,
            unknown_label=args.unknown_label,
            maxiter=args.maxiter,
            base_error=args.base_error,
            min_base_quality=args.min_base_quality,
            sample_sex=args.sample_sex,
            haploid_chroms={
                chrom.strip()
                for chrom in args.haploid_chroms.split(",")
                if chrom.strip()
            },
            fasta_path=args.fasta,
            alpha=args.alpha,
            beta=args.beta,
            freq_shrinkage=args.freq_shrinkage,
        )
        return

    print(f"Optimizing trajectory for {args.n_snps:,} SNPs and K={args.k} panels...")

    if args.k == 2:
        f_a, f_b, gls, c_m_positions = simulate_breakpoint(args.n_snps, args.seed)
        solver = TraceSolver(d_coupling=args.d, maxiter=args.maxiter)
        result = solver.solve(f_a, f_b, gls, c_m_positions)
    else:
        frequencies, gls, c_m_positions = simulate_multik_mosaic(
            args.n_snps,
            args.k,
            args.seed,
        )
        solver = TraceMultidimensionalSolver(
            d_coupling=args.d,
            logit_l2=args.logit_l2,
            maxiter=args.maxiter,
        )
        result = solver.solve(frequencies, gls, c_m_positions)

    print(f"converged: {result.converged}")
    print(f"message: {result.message}")
    print(f"iterations: {result.iterations}")
    print(f"seconds: {result.seconds:.4f}")
    print(f"action: {result.action:.6f}")

    start_idx = min(1000, args.n_snps - 1)
    end_idx = max(0, args.n_snps - 1000)
    print(f"phi[{start_idx}]: {np.array2string(np.asarray(result.phi[start_idx]), precision=4)}")
    print(f"phi[{end_idx}]: {np.array2string(np.asarray(result.phi[end_idx]), precision=4)}")


if __name__ == "__main__":
    main()
