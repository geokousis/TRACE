#!/usr/bin/env python3
"""
Generate reproducible TRACE benchmark fixtures with msprime.

This generator simulates the admixed benchmark individual *directly* in the
coalescent. Reference populations, recombination, mutation, and the admixed
sample all come out of a single ``msprime`` tree sequence. Local-ancestry
truth is read back from that tree sequence via a census event, and the
sample's genotypes are taken from its real simulated haplotypes.

This matters for validity. An earlier version of this script drew the sample's
genotypes from ``p = phi*f_A + (1-phi)*f_B`` followed by ``binomial(2, p)`` --
which is exactly TRACE's own likelihood model. Scoring against that truth only
measured the optimizer, never model misspecification. Here the truth is a
coalescent local-ancestry mosaic and the genotypes carry real linkage
disequilibrium, so the benchmark can actually probe where the
allele-frequency-only model breaks down.

How local ancestry truth is obtained:

- The admixed population ``ADM`` is formed by an admixture pulse from the
  reference sources at ``t_admix``.
- A census event just after the pulse inserts a node on every extant lineage
  and labels it with its source population.
- For each of the admixed individual's two haplotypes, at every position we
  walk up the local tree to its census ancestor; that ancestor's population is
  the local ancestry of that haplotype there.
- ``phi_A`` at a SNP is the fraction of the two haplotypes whose census
  ancestor is population ``A`` (so it takes values in {0, 0.5, 1}).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

try:
    import msprime
    import tskit
except ModuleNotFoundError as exc:
    raise SystemExit(
        "msprime and tskit are required for benchmark generation. Install with: "
        "python -m pip install -r requirements-benchmark.txt"
    ) from exc


# Per-base-pair recombination rate used for both the coalescent and the
# emitted genetic map, so the map is consistent with the simulated genetics.
RECOMB_RATE = 1e-8
MUTATION_RATE = 8e-7
ANCESTRAL_NE = 10_000
# Time of the A/B split (generations). Admixture times must be more recent.
AB_SPLIT_TIME = 2_000
GHOST_SPLIT_TIME = 3_000


@dataclass
class ScenarioParams:
    """Demographic and observation knobs for one benchmark scenario."""

    name: str
    description: str
    t_admix: float = 8.0
    # Admixture proportions, aligned with (A, B) or (A, B, GHOST) when ghost.
    proportions: tuple[float, ...] = (0.5, 0.5)
    use_ghost: bool = False
    # If set, the focal individual is sampled directly from this source
    # population with no admixture (truth is 100% that population).
    focal_pure: str | None = None
    n_ref_a: int = 40
    n_ref_b: int = 40
    # None => clean hard-call genotype likelihoods; otherwise mean read depth.
    coverage: float | None = None
    error_rate: float = 0.01
    # Extra cM inserted into the genetic map at the chromosome midpoint.
    map_gap_cm: float = 0.0
    # Optional second (older) admixture pulse of B into the admixed population,
    # producing a mix of long and short tracts. None disables the second pulse.
    second_pulse_time: float | None = None
    second_pulse_b_fraction: float = 0.0


SCENARIOS: dict[str, ScenarioParams] = {
    "recent_admixture": ScenarioParams(
        name="recent_admixture",
        description=(
            "Recent 50/50 pulse admixture (t=8). Long coalescent ancestry "
            "tracts; tests sharp domain-wall recovery from real haplotypes."
        ),
        t_admix=8.0,
        proportions=(0.5, 0.5),
    ),
    "ancient_mosaic": ScenarioParams(
        name="ancient_mosaic",
        description=(
            "Old 50/50 admixture (t=150). Many short coalescent tracts; tests "
            "local-ancestry resolution under dense breakpoints."
        ),
        t_admix=150.0,
        proportions=(0.5, 0.5),
    ),
    "isolation_by_distance": ScenarioParams(
        name="isolation_by_distance",
        description=(
            "Very old 50/50 admixture (t=600). Ancestry is a fine, stable "
            "near-50/50 mosaic; tests that TRACE does not invent recent "
            "hard-switch tracts where the truth is diffuse."
        ),
        t_admix=600.0,
        proportions=(0.5, 0.5),
    ),
    "ghost_introgression": ScenarioParams(
        name="ghost_introgression",
        description=(
            "Three-way admixture from A, B and an unsampled GHOST population "
            "(t=40). The reference panel contains only A and B, so ghost "
            "tracts have no matching source; tests graceful behaviour under "
            "unsampled ancestry."
        ),
        t_admix=40.0,
        proportions=(0.45, 0.30, 0.25),
        use_ghost=True,
    ),
    "bottleneck_drift": ScenarioParams(
        name="bottleneck_drift",
        description=(
            "Focal individual is 100% A, but the A reference panel is a small "
            "bottlenecked sample of haplotypes, so its allele frequencies are "
            "drift/sampling-noisy. Tests Beta smoothing and resistance to "
            "false ancestry shifts."
        ),
        focal_pure="A",
        n_ref_a=8,
        n_ref_b=40,
    ),
    "ancient_dna": ScenarioParams(
        name="ancient_dna",
        description=(
            "Recent 50/50 admixture (t=8) observed at low coverage (0.5x) with "
            "elevated error. Tests whether the kinetic term bridges weak "
            "genotype-likelihood evidence."
        ),
        t_admix=8.0,
        proportions=(0.5, 0.5),
        coverage=0.5,
        error_rate=0.02,
    ),
    "centromere_abyss": ScenarioParams(
        name="centromere_abyss",
        description=(
            "Recent 50/50 admixture (t=8) with a large cM gap inserted at the "
            "chromosome midpoint. Tests --max-imputation-gap-cm splitting."
        ),
        t_admix=8.0,
        proportions=(0.5, 0.5),
        map_gap_cm=5.0,
    ),
    # --- Starter scenarios for a larger evaluation ---
    "lopsided_minor": ScenarioParams(
        name="lopsided_minor",
        description=(
            "Lopsided 85/15 pulse admixture (t=12). Minority B ancestry is rare "
            "and patchy; stresses detection of the under-represented source."
        ),
        t_admix=12.0,
        proportions=(0.85, 0.15),
    ),
    "small_both_panels": ScenarioParams(
        name="small_both_panels",
        description=(
            "Recent 50/50 admixture (t=10) where BOTH reference panels are "
            "small (10 haplotypes each), so A and B frequencies are noisy. "
            "Stresses --freq-shrinkage and panel-uncertainty handling on both "
            "sources at once, not just one bottlenecked panel."
        ),
        t_admix=10.0,
        proportions=(0.5, 0.5),
        n_ref_a=10,
        n_ref_b=10,
    ),
    "two_pulse": ScenarioParams(
        name="two_pulse",
        description=(
            "Two admixture pulses from B into A (t=6 and t=80). Produces a "
            "mixture of long (recent) and short (old) ancestry tracts in the "
            "same genome; stresses a single global smoothing strength D."
        ),
        t_admix=6.0,                 # recent pulse; second pulse added in demography
        proportions=(0.7, 0.3),
        second_pulse_time=80.0,
        second_pulse_b_fraction=0.3,
    ),
    "deep_aDNA": ScenarioParams(
        name="deep_aDNA",
        description=(
            "Old 60/40 admixture (t=120) at very low coverage (0.25x) with high "
            "error. Combines short tracts with weak genotype evidence -- the "
            "hardest case for a frequency-only method."
        ),
        t_admix=120.0,
        proportions=(0.6, 0.4),
        coverage=0.25,
        error_rate=0.03,
    ),
}


def build_demography(params: ScenarioParams) -> tuple[msprime.Demography, float | None]:
    """Build the scenario demography. Returns (demography, census_time)."""
    demography = msprime.Demography()
    demography.add_population(name="A", initial_size=ANCESTRAL_NE)
    demography.add_population(name="B", initial_size=ANCESTRAL_NE)
    if params.use_ghost:
        demography.add_population(name="GHOST", initial_size=ANCESTRAL_NE)
    if params.focal_pure is None:
        demography.add_population(name="ADM", initial_size=ANCESTRAL_NE)
    demography.add_population(name="AB_ANC", initial_size=ANCESTRAL_NE)
    if params.use_ghost:
        demography.add_population(name="ANC", initial_size=ANCESTRAL_NE)

    census_time: float | None = None
    if params.focal_pure is not None:
        pass  # pure-source individual: no admixture, constant ancestry
    elif params.second_pulse_time is not None:
        # Two B-into-A pulses via mass migrations (backward in time): a fraction
        # of ADM lineages move to B at the recent pulse, another fraction of the
        # remainder at the older pulse, then all residual ADM lineages are moved
        # into A just before the census. This yields a mix of long (recent) and
        # short (old) B tracts on an A background.
        demography.add_mass_migration(
            time=params.t_admix, source="ADM", dest="B", proportion=params.proportions[1]
        )
        demography.add_mass_migration(
            time=params.second_pulse_time, source="ADM", dest="B",
            proportion=params.second_pulse_b_fraction,
        )
        demography.add_mass_migration(
            time=params.second_pulse_time + 0.005, source="ADM", dest="A", proportion=1.0
        )
        census_time = params.second_pulse_time + 0.01
        demography.add_census(time=census_time)
    else:
        ancestral = ["A", "B"] + (["GHOST"] if params.use_ghost else [])
        demography.add_admixture(
            time=params.t_admix,
            derived="ADM",
            ancestral=ancestral,
            proportions=list(params.proportions),
        )
        # Census just after the pulse so lineages have been reassigned to their
        # source populations (A/B/GHOST) rather than ADM.
        census_time = params.t_admix + 0.01
        demography.add_census(time=census_time)

    demography.add_population_split(
        time=AB_SPLIT_TIME, derived=["A", "B"], ancestral="AB_ANC"
    )
    if params.use_ghost:
        demography.add_population_split(
            time=GHOST_SPLIT_TIME, derived=["AB_ANC", "GHOST"], ancestral="ANC"
        )
    demography.sort_events()
    return demography, census_time


def simulate_tree_sequence(
    params: ScenarioParams,
    *,
    sequence_length: int,
    seed: int,
) -> tuple[tskit.TreeSequence, np.ndarray, np.ndarray, np.ndarray, float | None]:
    """Simulate the scenario. Returns ts plus node arrays and the census time."""
    demography, census_time = build_demography(params)

    focal_population = params.focal_pure if params.focal_pure is not None else "ADM"
    samples = [
        msprime.SampleSet(params.n_ref_a, population="A", ploidy=1),
        msprime.SampleSet(params.n_ref_b, population="B", ploidy=1),
        msprime.SampleSet(1, population=focal_population, ploidy=2),
    ]
    ancestry = msprime.sim_ancestry(
        samples=samples,
        demography=demography,
        sequence_length=sequence_length,
        recombination_rate=RECOMB_RATE,
        random_seed=seed,
    )
    mutated = msprime.sim_mutations(ancestry, rate=MUTATION_RATE, random_seed=seed + 1)

    all_samples = mutated.samples()
    a_nodes = np.asarray(all_samples[: params.n_ref_a], dtype=int)
    b_nodes = np.asarray(
        all_samples[params.n_ref_a : params.n_ref_a + params.n_ref_b], dtype=int
    )
    focal_nodes = np.asarray(all_samples[-2:], dtype=int)
    return mutated, a_nodes, b_nodes, focal_nodes, census_time


def census_population_map(ts: tskit.TreeSequence, census_time: float) -> dict[int, str]:
    """Map every census node id to its source population name."""
    pop_name = {pop.id: pop.metadata.get("name", str(pop.id)) for pop in ts.populations()}
    node_time = ts.tables.nodes.time
    node_population = ts.tables.nodes.population
    return {
        node_id: pop_name[node_population[node_id]]
        for node_id in range(ts.num_nodes)
        if node_time[node_id] == census_time
    }


def haplotype_ancestry_segments(
    ts: tskit.TreeSequence,
    hap_node: int,
    census_time: float,
    census_pop: dict[int, str],
    node_time: np.ndarray,
) -> tuple[np.ndarray, list[str | None]]:
    """Return (segment_left, segment_pop) for one haplotype across the genome.

    ``segment_pop[i]`` is the source population of ``hap_node`` on the genomic
    interval ``[segment_left[i], segment_left[i + 1])``.
    """
    seg_left: list[float] = []
    seg_pop: list[str | None] = []
    last: str | None = None
    have_segment = False
    for tree in ts.trees():
        node = hap_node
        while node != tskit.NULL and node_time[node] < census_time:
            node = tree.parent(node)
        pop = census_pop.get(node) if node != tskit.NULL else None
        if not have_segment or pop != last:
            seg_left.append(tree.interval.left)
            seg_pop.append(pop)
            last = pop
            have_segment = True
    return np.asarray(seg_left, dtype=float), seg_pop


def focal_phi_a(
    ts: tskit.TreeSequence,
    focal_nodes: np.ndarray,
    positions: np.ndarray,
    census_time: float | None,
) -> np.ndarray:
    """True ancestry fraction phi_A at each SNP position (values in {0,0.5,1})."""
    if census_time is None:
        # Pure-source focal individual: ancestry is constant. The only
        # pure-source scenario samples from A, so phi_A is 1 everywhere.
        return np.ones(len(positions), dtype=float)

    census_pop = census_population_map(ts, census_time)
    node_time = ts.tables.nodes.time
    phi = np.zeros(len(positions), dtype=float)
    for hap in focal_nodes:
        seg_left, seg_pop = haplotype_ancestry_segments(
            ts, int(hap), census_time, census_pop, node_time
        )
        idx = np.searchsorted(seg_left, positions, side="right") - 1
        idx = np.clip(idx, 0, len(seg_pop) - 1)
        is_a = np.array([seg_pop[i] == "A" for i in idx], dtype=float)
        phi += is_a
    return phi / len(focal_nodes)


def collect_variants(
    ts: tskit.TreeSequence,
    a_nodes: np.ndarray,
    b_nodes: np.ndarray,
    focal_nodes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extract biallelic sites. Returns (positions, count_a, count_b, focal_geno)."""
    positions: list[int] = []
    count_a: list[int] = []
    count_b: list[int] = []
    focal_geno: list[int] = []
    for variant in ts.variants():
        if len(variant.alleles) != 2:
            continue
        pos = int(round(variant.site.position))
        if pos <= 0:
            continue
        genotypes = variant.genotypes  # allele index per sample node, 0 or 1
        positions.append(pos)
        count_a.append(int(np.sum(genotypes[a_nodes])))
        count_b.append(int(np.sum(genotypes[b_nodes])))
        focal_geno.append(int(np.sum(genotypes[focal_nodes])))
    return (
        np.asarray(positions, dtype=int),
        np.asarray(count_a, dtype=int),
        np.asarray(count_b, dtype=int),
        np.asarray(focal_geno, dtype=int),
    )


def select_informative(
    positions: np.ndarray,
    count_a: np.ndarray,
    count_b: np.ndarray,
    focal_geno: np.ndarray,
    *,
    n_ref_a: int,
    n_ref_b: int,
    num_snps: int,
    min_freq_diff: float = 0.10,
) -> np.ndarray:
    """Pick ancestry-informative SNPs spread evenly across the genome.

    Selecting the globally most informative SNPs clusters them spatially (high
    |f_A - f_B| sites are linked), which badly misrepresents genome-wide local
    ancestry. Instead, split the candidate span into ``num_snps`` position bins
    and keep the most informative candidate in each bin, so the benchmark SNP
    set covers the chromosome uniformly. ``positions`` must be sorted ascending.
    """
    f_a = count_a / n_ref_a
    f_b = count_b / n_ref_b
    informative = np.abs(f_a - f_b)
    candidates = np.nonzero(informative >= min_freq_diff)[0]
    if len(candidates) < num_snps:
        raise RuntimeError(
            f"Only {len(candidates)} informative SNPs available "
            f"(need {num_snps}); increase --sequence-length or lower --num-snps."
        )

    cand_pos = positions[candidates]
    edges = np.linspace(cand_pos[0], cand_pos[-1] + 1, num_snps + 1)
    bin_of = np.clip(np.searchsorted(edges, cand_pos, side="right") - 1, 0, num_snps - 1)

    chosen: list[int] = []
    for b in range(num_snps):
        members = candidates[bin_of == b]
        if len(members):
            chosen.append(int(members[np.argmax(informative[members])]))

    # Fill any gaps left by empty bins with the most informative leftovers.
    if len(chosen) < num_snps:
        remaining = np.setdiff1d(candidates, np.asarray(chosen, dtype=int))
        need = num_snps - len(chosen)
        extra = remaining[np.argsort(informative[remaining])[::-1][:need]]
        chosen.extend(int(i) for i in extra)

    chosen_arr = np.asarray(chosen, dtype=int)
    return chosen_arr[np.argsort(positions[chosen_arr])]


def hard_call_gls(genotypes: np.ndarray, *, error: float) -> np.ndarray:
    gls = np.full((len(genotypes), 3), error / 2.0, dtype=float)
    gls[np.arange(len(genotypes)), genotypes] = 1.0 - error
    return gls


def low_coverage_gls(
    genotypes: np.ndarray,
    *,
    rng: np.random.Generator,
    coverage: float,
    error_rate: float,
) -> np.ndarray:
    gls = np.empty((len(genotypes), 3), dtype=float)
    for i, genotype in enumerate(genotypes):
        depth = rng.poisson(coverage)
        if depth == 0:
            gls[i] = [1.0, 1.0, 1.0]
            continue
        alt_prob = genotype / 2.0
        alt_reads = 0
        ref_reads = 0
        for _ in range(depth):
            true_alt = rng.random() < alt_prob
            observed_alt = true_alt
            if rng.random() < error_rate:
                observed_alt = not observed_alt
            alt_reads += observed_alt
            ref_reads += not observed_alt
        p_alt_given_g = np.array([error_rate, 0.5, 1.0 - error_rate])
        log_l = alt_reads * np.log(p_alt_given_g) + ref_reads * np.log(1.0 - p_alt_given_g)
        log_l -= np.max(log_l)
        gl = np.exp(log_l)
        gls[i] = gl / np.sum(gl)
    return gls


def genotype_likelihoods(
    params: ScenarioParams,
    focal_geno: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    if params.coverage is not None:
        return low_coverage_gls(
            focal_geno, rng=rng, coverage=params.coverage, error_rate=params.error_rate
        )
    return hard_call_gls(focal_geno, error=params.error_rate)


def write_scenario(
    out_dir: Path,
    params: ScenarioParams,
    positions: np.ndarray,
    count_a: np.ndarray,
    count_b: np.ndarray,
    phi_a: np.ndarray,
    gls: np.ndarray,
    *,
    seed: int,
) -> None:
    scenario_dir = out_dir / params.name
    scenario_dir.mkdir(parents=True, exist_ok=True)

    with open(scenario_dir / "reference_panel.tsv", "w", encoding="utf-8") as handle:
        handle.write("chrom\tpos\tref\talt\tderived_A\ttotal_A\tderived_B\ttotal_B\n")
        for pos, a, b in zip(positions, count_a, count_b):
            handle.write(
                f"1\t{int(pos)}\tA\tG\t{int(a)}\t{params.n_ref_a}\t{int(b)}\t{params.n_ref_b}\n"
            )

    c_m = positions * RECOMB_RATE * 100.0
    if params.map_gap_cm > 0.0:
        midpoint = len(c_m) // 2
        c_m[midpoint:] += params.map_gap_cm

    with open(scenario_dir / "genetic.map", "w", encoding="utf-8") as handle:
        for i, (pos, cm) in enumerate(zip(positions, c_m)):
            handle.write(f"1\trs{i}\t{cm:.8f}\t{int(pos)}\n")

    with open(scenario_dir / "sample.vcf", "w", encoding="utf-8") as handle:
        handle.write("##fileformat=VCFv4.2\n")
        handle.write("##contig=<ID=1>\n")
        handle.write('##FORMAT=<ID=GL,Number=G,Type=Float,Description="Genotype likelihoods">\n')
        handle.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tsample\n")
        for i, (pos, gl) in enumerate(zip(positions, gls)):
            gl_text = ",".join(f"{np.log10(max(value, 1e-300)):.6g}" for value in gl)
            handle.write(f"1\t{int(pos)}\trs{i}\tA\tG\t.\tPASS\t.\tGL\t{gl_text}\n")

    with open(scenario_dir / "truth.tsv", "w", encoding="utf-8") as handle:
        handle.write("chrom\tpos\tphi_A\tphi_B\n")
        for pos, phi in zip(positions, phi_a):
            handle.write(f"1\t{int(pos)}\t{phi:.8g}\t{1.0 - phi:.8g}\n")

    metadata = {
        "scenario": params.name,
        "seed": seed,
        "num_snps": int(len(positions)),
        "t_admix": params.t_admix if params.focal_pure is None else None,
        "proportions": list(params.proportions),
        "use_ghost": params.use_ghost,
        "focal_pure": params.focal_pure,
        "n_ref_a": params.n_ref_a,
        "n_ref_b": params.n_ref_b,
        "coverage": params.coverage,
        "map_gap_cm": params.map_gap_cm,
        "second_pulse_time": params.second_pulse_time,
        "second_pulse_b_fraction": params.second_pulse_b_fraction,
        "recomb_rate": RECOMB_RATE,
        "truth_source": "msprime_tree_sequence_census",
        "description": params.description,
    }
    with open(scenario_dir / "metadata.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)


def generate_scenario(
    out_dir: Path,
    params: ScenarioParams,
    *,
    num_snps: int,
    sequence_length: int,
    seed: int,
) -> None:
    ts, a_nodes, b_nodes, focal_nodes, census_time = simulate_tree_sequence(
        params, sequence_length=sequence_length, seed=seed
    )
    positions, count_a, count_b, focal_geno = collect_variants(
        ts, a_nodes, b_nodes, focal_nodes
    )
    selected = select_informative(
        positions,
        count_a,
        count_b,
        focal_geno,
        n_ref_a=params.n_ref_a,
        n_ref_b=params.n_ref_b,
        num_snps=num_snps,
    )
    sel_positions = positions[selected]
    sel_count_a = count_a[selected]
    sel_count_b = count_b[selected]
    sel_geno = focal_geno[selected]

    phi_a = focal_phi_a(ts, focal_nodes, sel_positions, census_time)

    rng = np.random.default_rng(seed + 5000)
    gls = genotype_likelihoods(params, sel_geno, rng)

    write_scenario(
        out_dir,
        params,
        sel_positions,
        sel_count_a,
        sel_count_b,
        phi_a,
        gls,
        seed=seed,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate TRACE msprime benchmark fixtures.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scenario", choices=("all", *SCENARIOS), default="all")
    parser.add_argument("--num-snps", type=int, default=2000)
    parser.add_argument("--sequence-length", type=int, default=20_000_000)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument(
        "--n-ref-haplotypes",
        type=int,
        default=None,
        help="Override reference haplotypes per source population. The "
        "bottleneck_drift scenario keeps its deliberately small A panel.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    names = tuple(SCENARIOS) if args.scenario == "all" else (args.scenario,)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for offset, name in enumerate(names):
        params = SCENARIOS[name]
        if args.n_ref_haplotypes is not None:
            # Only scale panels left at the default size (40); deliberately small
            # panels (bottleneck_drift, small_both_panels) keep their values so
            # the scenario still tests what it is meant to.
            new_a = args.n_ref_haplotypes if params.n_ref_a == 40 else params.n_ref_a
            new_b = args.n_ref_haplotypes if params.n_ref_b == 40 else params.n_ref_b
            params = replace(params, n_ref_a=new_a, n_ref_b=new_b)
        seed = args.seed + 1000 + offset
        generate_scenario(
            out_dir,
            params,
            num_snps=args.num_snps,
            sequence_length=args.sequence_length,
            seed=seed,
        )
        print(f"wrote {out_dir / name}")


if __name__ == "__main__":
    main()
