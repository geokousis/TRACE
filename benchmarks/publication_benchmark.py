#!/usr/bin/env python3
"""
Run a publication-style TRACE benchmark matrix.

The script generates replicate fixture sets with the existing msprime simulator,
runs TRACE through the public CLI, and writes aggregate JSON/CSV summaries.
It is intentionally an orchestrator: it does not import TRACE internals.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from pathlib import Path
from statistics import mean, stdev


SCENARIOS = (
    "recent_admixture",
    "ancient_mosaic",
    "isolation_by_distance",
    "ghost_introgression",
    "bottleneck_drift",
    "ancient_dna",
    "centromere_abyss",
)

PROFILE_SIZES = {
    "smoke": (80,),
    "small": (500, 2000),
    "medium": (2000, 10000),
    "large": (10000, 50000),
    "publication": (2000, 10000, 50000),
}

METRIC_KEYS = (
    "mse_phi_A",
    "rmse_phi_A",
    "mae_phi_A",
    "max_abs_error_phi_A",
    "bias_phi_A",
    "hard_call_accuracy_0_5",
    "r2_phi_A",
    "pearson_phi_A",
    "cm_mse_phi_A",
    "cm_rmse_phi_A",
    "cm_mae_phi_A",
    # Baselines, skill scores, and breakpoint metrics.
    "base_const0.5_mse",
    "base_global_mle_mse",
    "base_persite_mle_mse",
    "skill_vs_global_mle",
    "skill_vs_persite_mle",
    "skill_vs_const",
    "n_true_breakpoints",
    "n_pred_breakpoints",
    "bp_precision",
    "bp_recall",
    "bp_f1",
    "tract_mean_cm_true",
    "tract_mean_cm_pred",
)


def parse_csv_ints(value: str) -> tuple[int, ...]:
    parsed = tuple(int(part) for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("must contain at least one integer")
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("all values must be positive")
    return parsed


def parse_csv_strings(value: str) -> tuple[str, ...]:
    if value == "all":
        return SCENARIOS
    parsed = tuple(part.strip() for part in value.split(",") if part.strip())
    unknown = sorted(set(parsed) - set(SCENARIOS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown scenarios: {', '.join(unknown)}")
    return parsed


def size_to_sequence_length(num_snps: int) -> int:
    # The coalescent generator yields a limited density of ancestry-informative
    # SNPs, so allow ample sequence per requested SNP (and headroom for the
    # per-bin selection to spread them across the chromosome).
    return max(2_000_000, num_snps * 6000)


def run_command(cmd: list[str], *, cwd: Path | None = None) -> float:
    start = time.perf_counter()
    subprocess.run(cmd, cwd=cwd, check=True)
    return time.perf_counter() - start


def generate_replicate(
    *,
    python: str,
    simulator: Path,
    out_dir: Path,
    scenario: str,
    num_snps: int,
    seed: int,
    n_ref_haplotypes: int,
) -> float:
    cmd = [
        python,
        str(simulator),
        "--out-dir",
        str(out_dir),
        "--scenario",
        scenario,
        "--num-snps",
        str(num_snps),
        "--sequence-length",
        str(size_to_sequence_length(num_snps)),
        "--n-ref-haplotypes",
        str(n_ref_haplotypes),
        "--seed",
        str(seed),
    ]
    return run_command(cmd)


def run_trace_replicate(
    *,
    python: str,
    runner: Path,
    trace_script: Path,
    replicate_root: Path,
    workers: int,
    maxiter: int,
    chunk_size: int,
    chunk_overlap: int,
    auto_d: bool,
    d_grid: str,
    gl_weight: float,
    prior_weight: float,
    prior_center: float | None,
) -> float:
    cmd = [
        python,
        str(runner),
        "--benchmark-dir",
        str(replicate_root),
        "--trace-script",
        str(trace_script),
        "--python",
        python,
        "--workers",
        str(workers),
        "--maxiter",
        str(maxiter),
        "--chunk-size",
        str(chunk_size),
        "--chunk-overlap",
        str(chunk_overlap),
        "--gl-weight",
        str(gl_weight),
        "--prior-weight",
        str(prior_weight),
    ]
    if prior_center is not None:
        cmd.extend(["--prior-center", str(prior_center)])
    cmd.extend(["--d-grid", d_grid])
    if not auto_d:
        cmd.append("--no-auto-d")
    return run_command(cmd)


def load_metrics(replicate_root: Path, scenario: str) -> dict[str, float | int | None]:
    with open(replicate_root / scenario / "metrics.json", encoding="utf-8") as handle:
        return json.load(handle)


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["scenario"]), int(row["num_snps"])), []).append(row)

    summary_rows: list[dict[str, object]] = []
    for (scenario, num_snps), group in sorted(grouped.items(), key=lambda item: (item[0][1], item[0][0])):
        out: dict[str, object] = {
            "scenario": scenario,
            "num_snps": num_snps,
            "replicates": len(group),
        }
        for key in METRIC_KEYS:
            values = [
                float(row[key])
                for row in group
                if row.get(key) is not None
            ]
            out[f"{key}_mean"] = mean(values) if values else None
            out[f"{key}_sd"] = stdev(values) if len(values) > 1 else 0.0 if values else None
        out["trace_seconds_mean"] = mean(float(row["trace_seconds"]) for row in group)
        out["generate_seconds_mean"] = mean(float(row["generate_seconds"]) for row in group)
        summary_rows.append(out)
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a TRACE publication benchmark matrix.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--python", default="/home/kousis/work/trace/.mamba-env/bin/python")
    parser.add_argument("--trace-script", default="/home/kousis/work/trace/trace_solver.py")
    parser.add_argument("--simulator", default="/home/kousis/work/trace/benchmarks/simulate_msprime_benchmarks.py")
    parser.add_argument("--runner", default="/home/kousis/work/trace/benchmarks/run_trace_benchmarks.py")
    parser.add_argument("--profile", choices=tuple(PROFILE_SIZES), default="small")
    parser.add_argument("--sizes", type=parse_csv_ints,
                        help="Comma-separated SNP counts. Overrides --profile.")
    parser.add_argument("--scenarios", type=parse_csv_strings, default=SCENARIOS)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--n-ref-haplotypes", type=int, default=80)
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.replicates <= 0:
        raise SystemExit("--replicates must be positive")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    sizes = args.sizes if args.sizes is not None else PROFILE_SIZES[args.profile]
    scenarios = args.scenarios

    detail_rows: list[dict[str, object]] = []
    for num_snps in sizes:
        for scenario in scenarios:
            for replicate in range(args.replicates):
                seed = args.seed + num_snps * 10_000 + replicate * 101 + SCENARIOS.index(scenario)
                replicate_root = out_dir / f"n{num_snps}" / scenario / f"rep{replicate:03d}"
                replicate_root.mkdir(parents=True, exist_ok=True)

                print(
                    f"benchmark n={num_snps} scenario={scenario} "
                    f"replicate={replicate} seed={seed}",
                    flush=True,
                )
                generate_seconds = generate_replicate(
                    python=args.python,
                    simulator=Path(args.simulator),
                    out_dir=replicate_root,
                    scenario=scenario,
                    num_snps=num_snps,
                    seed=seed,
                    n_ref_haplotypes=args.n_ref_haplotypes,
                )
                trace_seconds = run_trace_replicate(
                    python=args.python,
                    runner=Path(args.runner),
                    trace_script=Path(args.trace_script),
                    replicate_root=replicate_root,
                    workers=args.workers,
                    maxiter=args.maxiter,
                    chunk_size=args.chunk_size,
                    chunk_overlap=args.chunk_overlap,
                    auto_d=not args.no_auto_d,
                    d_grid=args.d_grid,
                    gl_weight=args.gl_weight,
                    prior_weight=args.prior_weight,
                    prior_center=args.prior_center,
                )
                metrics = load_metrics(replicate_root, scenario)
                row: dict[str, object] = {
                    "scenario": scenario,
                    "num_snps": num_snps,
                    "replicate": replicate,
                    "seed": seed,
                    "generate_seconds": generate_seconds,
                    "trace_seconds": trace_seconds,
                    **metrics,
                }
                detail_rows.append(row)
                write_csv(out_dir / "detail_metrics.csv", detail_rows)
                summary_rows = summarize(detail_rows)
                write_csv(out_dir / "summary_metrics.csv", summary_rows)
                with open(out_dir / "detail_metrics.json", "w", encoding="utf-8") as handle:
                    json.dump(detail_rows, handle, indent=2)
                with open(out_dir / "summary_metrics.json", "w", encoding="utf-8") as handle:
                    json.dump(summary_rows, handle, indent=2)

    print(f"wrote {out_dir / 'detail_metrics.csv'}")
    print(f"wrote {out_dir / 'summary_metrics.csv'}")


if __name__ == "__main__":
    main()
