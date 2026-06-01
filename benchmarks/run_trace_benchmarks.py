#!/usr/bin/env python3
"""
Run TRACE on benchmark fixtures and compute local-ancestry accuracy metrics.

This script intentionally calls the public TRACE CLI instead of importing
private internals. That keeps the benchmark close to how users will run the
tool on real data.
"""

from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
from pathlib import Path

import numpy as np


def read_phi_table(path: Path, column: str) -> dict[tuple[str, int], float]:
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if column not in (reader.fieldnames or []):
            raise ValueError(f"{path} is missing required column {column!r}")
        return {
            (row["chrom"], int(row["pos"])): float(row[column])
            for row in reader
        }


def load_genetic_map_cm(map_path: Path) -> dict[tuple[str, int], float]:
    """Return {(chrom, pos_bp): pos_cm} from a PLINK-style genetic map."""
    result: dict[tuple[str, int], float] = {}
    with open(map_path, encoding="utf-8", newline="") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 4:
                continue
            chrom, pos_bp, pos_cm = parts[0], int(parts[3]), float(parts[2])
            result[(chrom, pos_bp)] = pos_cm
    return result


def _interpolate_cm(pos_bp: int, chrom: str, map_pos: list[int], map_cm: list[float]) -> float:
    """Linear interpolation of cM position for a bp position on one chromosome."""
    idx = int(np.searchsorted(map_pos, pos_bp))
    if idx == 0:
        return map_cm[0]
    if idx >= len(map_pos):
        return map_cm[-1]
    bp0, bp1 = map_pos[idx - 1], map_pos[idx]
    cm0, cm1 = map_cm[idx - 1], map_cm[idx]
    frac = (pos_bp - bp0) / (bp1 - bp0) if bp1 != bp0 else 0.0
    return cm0 + frac * (cm1 - cm0)


def _cm_positions_for_snps(
    shared_keys: list[tuple[str, int]],
    gmap: dict[tuple[str, int], float],
) -> np.ndarray:
    """Absolute cM position of each SNP, interpolated from the genetic map.

    SNPs on a chromosome with no map coverage fall back to their 1-based order
    index so that downstream interval/tolerance logic still has a monotone axis.
    """
    # Build per-chrom sorted map arrays for interpolation
    by_chrom: dict[str, tuple[list[int], list[float]]] = {}
    for (chrom, bp), cm in gmap.items():
        if chrom not in by_chrom:
            by_chrom[chrom] = ([], [])
        by_chrom[chrom][0].append(bp)
        by_chrom[chrom][1].append(cm)
    for chrom in by_chrom:
        bps, cms = by_chrom[chrom]
        order = np.argsort(bps)
        by_chrom[chrom] = ([bps[i] for i in order], [cms[i] for i in order])

    cm_pos = np.arange(len(shared_keys), dtype=float)
    for i, (chrom, pos_bp) in enumerate(shared_keys):
        if chrom not in by_chrom:
            continue  # falls back to index order
        map_pos, map_cm = by_chrom[chrom]
        cm_pos[i] = _interpolate_cm(pos_bp, chrom, map_pos, map_cm)
    return cm_pos


def _cm_weights_for_snps(
    shared_keys: list[tuple[str, int]],
    gmap: dict[tuple[str, int], float],
) -> np.ndarray:
    """
    Trapezoidal cM weights: each SNP gets the half-interval to its neighbours.
    Falls back to uniform weights when the map has no coverage for a chromosome.
    """
    cm_pos = _cm_positions_for_snps(shared_keys, gmap)
    interval = np.zeros(len(shared_keys))
    if len(shared_keys) == 1:
        interval[0] = 1.0
    else:
        # half-interval to each neighbour
        diffs = np.diff(cm_pos)
        interval[0] = diffs[0] / 2.0
        interval[-1] = diffs[-1] / 2.0
        interval[1:-1] = (diffs[:-1] + diffs[1:]) / 2.0
        # clamp negatives (unsorted map edge case) to tiny positive
        interval = np.clip(interval, 1e-9, None)

    total = interval.sum()
    return interval / total if total > 0 else np.full(len(shared_keys), 1.0 / len(shared_keys))


def read_reference_freqs(path: Path) -> dict[tuple[str, int], tuple[float, float]]:
    """Return {(chrom,pos): (f_A, f_B)} with Laplace (alpha=beta=1) smoothing.

    This mirrors TRACE's default count-panel smoothing so the baselines see the
    same reference frequencies the solver does.
    """
    freqs: dict[tuple[str, int], tuple[float, float]] = {}
    with open(path, encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        for row in reader:
            f_a = (float(row["derived_A"]) + 1.0) / (float(row["total_A"]) + 2.0)
            f_b = (float(row["derived_B"]) + 1.0) / (float(row["total_B"]) + 2.0)
            freqs[(row["chrom"], int(row["pos"]))] = (f_a, f_b)
    return freqs


def read_vcf_gls(path: Path) -> dict[tuple[str, int], np.ndarray]:
    """Return {(chrom,pos): [L0,L1,L2]} from a single-sample GL VCF (log10 GL)."""
    gls: dict[tuple[str, int], np.ndarray] = {}
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 10:
                continue
            chrom, pos = parts[0], int(parts[1])
            fmt = parts[8].split(":")
            sample = parts[9].split(":")
            field = dict(zip(fmt, sample))
            if "GL" not in field:
                continue
            log10 = np.array([float(x) for x in field["GL"].split(",")], dtype=float)
            gls[(chrom, pos)] = np.power(10.0, log10)
    return gls


def _site_loglik(phi: np.ndarray, f_a: np.ndarray, f_b: np.ndarray, gls: np.ndarray) -> np.ndarray:
    """Per-site diploid log-likelihood log P(GL | phi) under HWE allele mixing.

    ``phi`` broadcasts against the (n_sites,) frequency/GL arrays, so a scalar
    gives one value per site and a (grid, 1) column gives a (grid, n_sites)
    matrix.
    """
    p = np.clip(phi * f_a + (1.0 - phi) * f_b, 1e-9, 1.0 - 1e-9)
    likelihood = (
        gls[:, 0] * (1.0 - p) ** 2
        + gls[:, 1] * 2.0 * p * (1.0 - p)
        + gls[:, 2] * p ** 2
    )
    return np.log(np.clip(likelihood, 1e-300, None))


def global_mle_phi(f_a: np.ndarray, f_b: np.ndarray, gls: np.ndarray, grid_size: int = 201) -> float:
    """Single constant ancestry that best fits all sites (no local resolution)."""
    grid = np.linspace(0.0, 1.0, grid_size)
    total = np.array([float(np.sum(_site_loglik(g, f_a, f_b, gls))) for g in grid])
    return float(grid[int(np.argmax(total))])


def persite_mle_phi(f_a: np.ndarray, f_b: np.ndarray, gls: np.ndarray, grid_size: int = 101) -> np.ndarray:
    """Independent per-site ancestry MLE — TRACE with the smoothing prior off."""
    grid = np.linspace(0.0, 1.0, grid_size)
    ll = _site_loglik(grid[:, None], f_a, f_b, gls)  # (grid, n_sites)
    return grid[np.argmax(ll, axis=0)]


def _point_metrics(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray | None) -> dict[str, float | None]:
    """Pointwise accuracy of one prediction against truth."""
    residual = y_pred - y_true
    mse = float(np.mean(residual ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    metrics: dict[str, float | None] = {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(residual))),
        "max_abs_error": float(np.max(np.abs(residual))),
        "bias": float(np.mean(residual)),
        "hard_call_accuracy_0_5": float(np.mean((y_true >= 0.5) == (y_pred >= 0.5))),
        # 3-class dosage accuracy: snap both to the nearest copy-number level
        # {0, 0.5, 1} and compare. This is how local-ancestry dosage is actually
        # scored. Unlike the >=0.5 hard call, it does not penalize a continuous
        # estimate that sits correctly near 0.5 (heterozygous) just for landing
        # on the "wrong" side of the threshold, and it does not reward a tool
        # merely for emitting discrete calls.
        "dosage_accuracy_3class": float(
            np.mean(np.round(np.clip(y_true, 0.0, 1.0) * 2.0)
                    == np.round(np.clip(y_pred, 0.0, 1.0) * 2.0))
        ),
        "r2": float(1.0 - np.sum(residual ** 2) / ss_tot) if ss_tot > 0.0 else None,
        "pearson": (
            float(np.corrcoef(y_true, y_pred)[0, 1])
            if np.std(y_true) > 0 and np.std(y_pred) > 0 else None
        ),
        "cm_mse": None,
        "cm_rmse": None,
        "cm_mae": None,
    }
    if w is not None:
        cm_mse = float(np.sum(w * residual ** 2))
        metrics["cm_mse"] = cm_mse
        metrics["cm_rmse"] = float(np.sqrt(cm_mse))
        metrics["cm_mae"] = float(np.sum(w * np.abs(residual)))
    return metrics


def _dosage_state(phi: np.ndarray) -> np.ndarray:
    """Snap a continuous ancestry field to the nearest dosage level {0, 0.5, 1}."""
    return np.round(np.clip(phi, 0.0, 1.0) * 2.0) / 2.0


def _breakpoints_cm(cm_pos: np.ndarray, state: np.ndarray) -> np.ndarray:
    """cM midpoints of every ancestry-state change."""
    change = np.nonzero(np.abs(np.diff(state)) > 1e-9)[0]
    return (cm_pos[change] + cm_pos[change + 1]) / 2.0


def _mean_tract_cm(cm_pos: np.ndarray, state: np.ndarray) -> float | None:
    """Mean ancestry-tract length in cM (span between consecutive state changes)."""
    bounds = _breakpoints_cm(cm_pos, state)
    edges = np.concatenate(([cm_pos[0]], bounds, [cm_pos[-1]]))
    lengths = np.diff(edges)
    return float(np.mean(lengths)) if len(lengths) else None


def breakpoint_metrics(
    cm_pos: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    tol_cm: float,
) -> dict[str, float | int | None]:
    """Precision/recall/F1 of inferred ancestry breakpoints within ``tol_cm``."""
    true_bp = _breakpoints_cm(cm_pos, _dosage_state(y_true))
    pred_bp = _breakpoints_cm(cm_pos, _dosage_state(y_pred))

    def matched(query: np.ndarray, reference: np.ndarray) -> int:
        if len(query) == 0 or len(reference) == 0:
            return 0
        nearest = np.abs(query[:, None] - reference[None, :]).min(axis=1)
        return int(np.sum(nearest <= tol_cm))

    n_true, n_pred = len(true_bp), len(pred_bp)
    precision = matched(pred_bp, true_bp) / n_pred if n_pred else None
    recall = matched(true_bp, pred_bp) / n_true if n_true else None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1: float | None = 2.0 * precision * recall / (precision + recall)
    else:
        f1 = None
    return {
        "n_true_breakpoints": n_true,
        "n_pred_breakpoints": n_pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tract_mean_cm_true": _mean_tract_cm(cm_pos, _dosage_state(y_true)),
        "tract_mean_cm_pred": _mean_tract_cm(cm_pos, _dosage_state(y_pred)),
    }


def _skill(mse_model: float, mse_baseline: float | None) -> float | None:
    """Skill score: 1 - mse_model/mse_baseline. >0 means the model beats it."""
    if not mse_baseline:
        return None
    return float(1.0 - mse_model / mse_baseline)


def score_scenario(scenario_dir: Path, *, breakpoint_tol_cm: float = 0.5) -> dict[str, float | int | None]:
    truth = read_phi_table(scenario_dir / "truth.tsv", "phi_A")
    inferred = read_phi_table(scenario_dir / "trace_output.tsv", "phi_A")

    shared_keys = sorted(set(truth) & set(inferred), key=lambda key: (key[0], key[1]))
    if not shared_keys:
        raise ValueError(f"No overlapping SNPs between truth and TRACE output in {scenario_dir}")

    y_true = np.asarray([truth[key] for key in shared_keys], dtype=float)
    y_pred = np.asarray([inferred[key] for key in shared_keys], dtype=float)

    # cM weights/positions — density-invariant scoring and breakpoint tolerance.
    gmap_path = scenario_dir / "genetic.map"
    w: np.ndarray | None = None
    cm_pos: np.ndarray | None = None
    if gmap_path.exists():
        gmap = load_genetic_map_cm(gmap_path)
        w = _cm_weights_for_snps(shared_keys, gmap)
        cm_pos = _cm_positions_for_snps(shared_keys, gmap)

    trace = _point_metrics(y_true, y_pred, w)

    metrics: dict[str, float | int | None] = {
        "n_snps": len(shared_keys),
        "mse_phi_A": trace["mse"],
        "rmse_phi_A": trace["rmse"],
        "mae_phi_A": trace["mae"],
        "max_abs_error_phi_A": trace["max_abs_error"],
        "bias_phi_A": trace["bias"],
        "hard_call_accuracy_0_5": trace["hard_call_accuracy_0_5"],
        "dosage_accuracy_3class": trace["dosage_accuracy_3class"],
        "r2_phi_A": trace["r2"],
        "pearson_phi_A": trace["pearson"],
        "cm_mse_phi_A": trace["cm_mse"],
        "cm_rmse_phi_A": trace["cm_rmse"],
        "cm_mae_phi_A": trace["cm_mae"],
    }

    # --- Baselines: a trivial floor and two reference-frequency baselines that
    # bound what "no local resolution" and "no smoothing" achieve. ---
    panel_path = scenario_dir / "reference_panel.tsv"
    vcf_path = scenario_dir / "sample.vcf"
    if panel_path.exists() and vcf_path.exists():
        freqs = read_reference_freqs(panel_path)
        gl_map = read_vcf_gls(vcf_path)
        usable = [k for k in shared_keys if k in freqs and k in gl_map]
        if usable:
            idx = np.array([shared_keys.index(k) for k in usable])
            f_a = np.array([freqs[k][0] for k in usable])
            f_b = np.array([freqs[k][1] for k in usable])
            gls = np.array([gl_map[k] for k in usable])
            yt = y_true[idx]
            wb = w[idx] if w is not None else None

            const = _point_metrics(yt, np.full(len(usable), 0.5), wb)
            gphi = global_mle_phi(f_a, f_b, gls)
            glob = _point_metrics(yt, np.full(len(usable), gphi), wb)
            ps_pred = persite_mle_phi(f_a, f_b, gls)
            persite = _point_metrics(yt, ps_pred, wb)

            metrics.update({
                "base_const0.5_mse": const["mse"],
                "base_const0.5_cm_mse": const["cm_mse"],
                "base_global_mle_phi": gphi,
                "base_global_mle_mse": glob["mse"],
                "base_global_mle_cm_mse": glob["cm_mse"],
                "base_persite_mle_mse": persite["mse"],
                "base_persite_mle_cm_mse": persite["cm_mse"],
                "base_persite_mle_hardcall": persite["hard_call_accuracy_0_5"],
                "skill_vs_global_mle": _skill(trace["mse"], glob["mse"]),
                "skill_vs_persite_mle": _skill(trace["mse"], persite["mse"]),
                "skill_vs_const": _skill(trace["mse"], const["mse"]),
            })

            if cm_pos is not None:
                cmb = cm_pos[idx]
                ps_bp = breakpoint_metrics(cmb, yt, ps_pred, tol_cm=breakpoint_tol_cm)
                metrics["base_persite_mle_bp_f1"] = ps_bp["f1"]
                metrics["base_persite_mle_n_pred_breakpoints"] = ps_bp["n_pred_breakpoints"]

    # --- Breakpoint / tract metrics for TRACE: a method that advertises sharp
    # tract recovery should be judged on boundaries, not just pointwise MSE. ---
    if cm_pos is not None:
        bp = breakpoint_metrics(cm_pos, y_true, y_pred, tol_cm=breakpoint_tol_cm)
        metrics.update({
            "bp_tolerance_cm": breakpoint_tol_cm,
            "n_true_breakpoints": bp["n_true_breakpoints"],
            "n_pred_breakpoints": bp["n_pred_breakpoints"],
            "bp_precision": bp["precision"],
            "bp_recall": bp["recall"],
            "bp_f1": bp["f1"],
            "tract_mean_cm_true": bp["tract_mean_cm_true"],
            "tract_mean_cm_pred": bp["tract_mean_cm_pred"],
        })

    return metrics


def run_trace_for_scenario(
    scenario_dir: Path,
    *,
    python: str,
    trace_script: str,
    workers: int,
    maxiter: int,
    chunk_size: int,
    chunk_overlap: int,
    auto_d: bool,
    d_grid: str,
    gl_weight: float = 1.0,
    prior_weight: float = 0.0,
    prior_center: float | None = None,
    d_fixed: float | None = None,
    trace_extra_args: list[str] | None = None,
) -> None:
    cmd = [
        python,
        trace_script,
        "--vcf",
        str(scenario_dir / "sample.vcf"),
        "--sample",
        "sample",
        "--reference-panel",
        str(scenario_dir / "reference_panel.tsv"),
        "--genetic-map",
        str(scenario_dir / "genetic.map"),
        "--output",
        str(scenario_dir / "trace_output.tsv"),
        "--output-zarr",
        str(scenario_dir / "trace_output.zarr"),
        "--chunk-size",
        str(chunk_size),
        "--chunk-overlap",
        str(chunk_overlap),
        "--workers",
        str(workers),
        "--maxiter",
        str(maxiter),
        "--alpha",
        "1",
        "--beta",
        "1",
        "--gl-weight",
        str(gl_weight),
        "--prior-weight",
        str(prior_weight),
    ]
    if prior_center is not None:
        cmd.extend(["--prior-center", str(prior_center)])

    # Scenario-aware flags, driven by metadata.json so new scenarios get the
    # right treatment automatically (no hard-coded scenario names).
    meta = {}
    meta_path = scenario_dir / "metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
    if meta.get("map_gap_cm", 0.0) and meta["map_gap_cm"] > 0.0:
        cmd.extend(["--max-imputation-gap-cm", "1.0"])
    if meta.get("use_ghost"):
        # Unsampled ("ghost") ancestry: give TRACE a third unknown population
        # column so ghost tracts have somewhere to go instead of being forced
        # onto A or B. Skip if the caller already passed --unknown-state.
        extra = trace_extra_args or []
        if not any(a.startswith("--unknown-state") for a in extra):
            cmd.extend(["--unknown-state", "--unknown-penalty", "0.8"])

    if auto_d:
        cmd.extend(["--auto-d", "--d-grid", d_grid])
    elif d_fixed is not None:
        cmd.extend(["--d", str(d_fixed)])
    if trace_extra_args:
        cmd.extend(trace_extra_args)

    subprocess.run(cmd, check=True)

    report_cmd = [
        python,
        trace_script,
        "--report-zarr",
        str(scenario_dir / "trace_output.zarr"),
        "--report-output",
        str(scenario_dir / "trace_report.html"),
    ]
    subprocess.run(report_cmd, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run TRACE on benchmark fixtures and score MSE.")
    parser.add_argument("--benchmark-dir", required=True)
    parser.add_argument("--trace-script", default="trace_solver.py")
    parser.add_argument("--python", default="python")
    parser.add_argument("--scenario", default="all")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--no-auto-d", action="store_true",
                        help="Disable auto-D calibration (use fixed --d instead).")
    parser.add_argument("--d-grid", default="0.003,0.01,0.03,0.1,0.3,1.0,3.0")
    parser.add_argument("--gl-weight", type=float, default=1.0)
    parser.add_argument("--prior-weight", type=float, default=0.0)
    parser.add_argument("--prior-center", type=float, default=None)
    parser.add_argument(
        "--trace-extra-args",
        default="",
        help="Additional TRACE CLI flags appended to each benchmark run, parsed with shell-like quoting.",
    )
    parser.add_argument(
        "--breakpoint-tol-cm",
        type=float,
        default=0.5,
        help="Distance (cM) within which an inferred ancestry breakpoint counts as matched.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    if args.scenario == "all":
        scenario_dirs = sorted(
            path for path in benchmark_dir.iterdir()
            if path.is_dir() and (path / "sample.vcf").exists()
        )
    else:
        scenario_dirs = [benchmark_dir / args.scenario]

    summary: dict[str, dict[str, float | int | None]] = {}
    trace_extra_args = shlex.split(args.trace_extra_args)
    for scenario_dir in scenario_dirs:
        print(f"running TRACE benchmark: {scenario_dir.name}")
        run_trace_for_scenario(
            scenario_dir,
            python=args.python,
            trace_script=args.trace_script,
            workers=args.workers,
            maxiter=args.maxiter,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            auto_d=not args.no_auto_d,
            d_grid=args.d_grid,
            gl_weight=args.gl_weight,
            prior_weight=args.prior_weight,
            prior_center=args.prior_center,
            trace_extra_args=trace_extra_args,
        )
        metrics = score_scenario(scenario_dir, breakpoint_tol_cm=args.breakpoint_tol_cm)
        summary[scenario_dir.name] = metrics
        with open(scenario_dir / "metrics.json", "w", encoding="utf-8") as handle:
            json.dump(metrics, handle, indent=2)

        def fmt(value: float | int | None) -> str:
            return "n/a" if value is None else f"{value:.4g}"

        print(
            f"  MSE(phi_A)={fmt(metrics['mse_phi_A'])}  "
            f"baselines[const={fmt(metrics.get('base_const0.5_mse'))} "
            f"global_mle={fmt(metrics.get('base_global_mle_mse'))} "
            f"persite_mle={fmt(metrics.get('base_persite_mle_mse'))}]  "
            f"skill_vs_global={fmt(metrics.get('skill_vs_global_mle'))}  "
            f"bp_f1={fmt(metrics.get('bp_f1'))}"
        )

    with open(benchmark_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"wrote {benchmark_dir / 'summary_metrics.json'}")


if __name__ == "__main__":
    main()
