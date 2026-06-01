#!/usr/bin/env python3
"""
D × tau (gl_weight) grid search over existing benchmark fixtures.

This script re-uses already-simulated scenario directories (each containing
sample.vcf, reference_panel.tsv, genetic.map, truth.tsv) and sweeps over a
Cartesian product of D coupling values and GL weight (tau) values.  It does
NOT re-simulate; run simulate_msprime_benchmarks.py first.

Example usage
-------------
# Sweep all scenarios in benchmark_runs with default grids:
.mamba-env/bin/python benchmarks/grid_search_benchmark.py \\
    --benchmark-dir benchmark_runs \\
    --out-dir grid_results \\
    --d-grid 0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03 \\
    --tau-grid 0.25,0.5,1.0,2.0 \\
    --workers 2

# Narrow scan after seeing initial results:
.mamba-env/bin/python benchmarks/grid_search_benchmark.py \\
    --benchmark-dir benchmark_runs \\
    --out-dir grid_results_narrow \\
    --d-grid 0.0001,0.0003,0.001 \\
    --tau-grid 0.5,1.0 \\
    --scenarios recent_admixture,ancient_dna
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np


SCENARIOS = (
    "recent_admixture",
    "ancient_mosaic",
    "isolation_by_distance",
    "ghost_introgression",
    "bottleneck_drift",
    "ancient_dna",
    "centromere_abyss",
)

METRIC_KEYS = (
    "n_snps",
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
)


def parse_csv_floats(value: str) -> list[float]:
    parsed = [float(part) for part in value.split(",") if part.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("must contain at least one float")
    if any(v <= 0 for v in parsed):
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


def score_scenario_dir(scenario_dir: Path) -> dict[str, float | int | None]:
    """Import and run the same scoring logic as run_trace_benchmarks."""
    import sys
    benchmarks_dir = str(Path(__file__).parent)
    if benchmarks_dir not in sys.path:
        sys.path.insert(0, benchmarks_dir)
    from run_trace_benchmarks import score_scenario
    return score_scenario(scenario_dir)


def run_trace(
    *,
    python: str,
    trace_script: str,
    scenario_dir: Path,
    output_tsv: Path,
    output_zarr: Path,
    workers: int,
    maxiter: int,
    chunk_size: int,
    chunk_overlap: int,
    d: float,
    gl_weight: float,
    prior_weight: float,
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
        str(output_tsv),
        "--output-zarr",
        str(output_zarr),
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
        "--d",
        str(d),
        "--gl-weight",
        str(gl_weight),
        "--prior-weight",
        str(prior_weight),
    ]
    if scenario_dir.name == "centromere_abyss":
        cmd.extend(["--max-imputation-gap-cm", "1.0"])
    subprocess.run(cmd, check=True, capture_output=True)


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


def pivot_best(rows: list[dict[str, object]], metric: str = "cm_mse_phi_A") -> list[dict[str, object]]:
    """For each scenario, report best D, best tau, and metric value for all (D, tau) pairs."""
    from collections import defaultdict
    by_scenario: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_scenario[str(row["scenario"])].append(row)

    pivot_rows = []
    for scenario, srows in sorted(by_scenario.items()):
        best = min((r for r in srows if r.get(metric) is not None), key=lambda r: float(r[metric]), default=None)
        default = next((r for r in srows if r["d"] == 0.001 and r["tau"] == 1.0), None)
        pivot_rows.append({
            "scenario": scenario,
            f"best_{metric}": best[metric] if best else None,
            "best_d": best["d"] if best else None,
            "best_tau": best["tau"] if best else None,
            f"default_{metric}": default[metric] if default else None,
            "improvement": (
                (float(default[metric]) - float(best[metric])) / float(default[metric])
                if best and default and default.get(metric) and best.get(metric)
                else None
            ),
        })
    return pivot_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="D × tau grid search over existing benchmark scenario directories.",
    )
    parser.add_argument("--benchmark-dir", required=True,
                        help="Directory containing scenario subdirs with sample.vcf, truth.tsv, etc.")
    parser.add_argument("--out-dir", required=True,
                        help="Directory to write grid_results.csv, grid_best.csv, grid_results.json.")
    parser.add_argument("--trace-script",
                        default="/home/kousis/work/trace/trace_solver.py")
    parser.add_argument("--python",
                        default="/home/kousis/work/trace/.mamba-env/bin/python")
    parser.add_argument("--scenarios", type=parse_csv_strings, default="all")
    parser.add_argument("--d-grid", type=parse_csv_floats,
                        default="0.00003,0.0001,0.0003,0.001,0.003,0.01,0.03")
    parser.add_argument("--tau-grid", type=parse_csv_floats,
                        default="0.25,0.5,1.0,2.0",
                        help="GL weight (tau) values to sweep.")
    parser.add_argument("--prior-weight", type=float, default=0.0,
                        help="Fixed prior weight for all runs (default: 0 = disabled).")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--pivot-metric", default="cm_mse_phi_A",
                        help="Metric used to identify the best (D, tau) pair per scenario.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark_dir = Path(args.benchmark_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    d_grid: list[float] = args.d_grid
    tau_grid: list[float] = args.tau_grid

    # Discover scenario directories
    scenarios = args.scenarios
    scenario_dirs: list[Path] = []
    for name in scenarios:
        path = benchmark_dir / name
        if path.is_dir() and (path / "sample.vcf").exists():
            scenario_dirs.append(path)
        else:
            print(f"  warning: {path} not found or missing sample.vcf — skipping")

    if not scenario_dirs:
        raise SystemExit(f"No scenario directories found under {benchmark_dir}")

    total = len(scenario_dirs) * len(d_grid) * len(tau_grid)
    done = 0
    rows: list[dict[str, object]] = []

    for scenario_dir in scenario_dirs:
        scenario_name = scenario_dir.name
        for d in d_grid:
            for tau in tau_grid:
                done += 1
                label = f"[{done}/{total}] {scenario_name} D={d} tau={tau}"
                print(label, flush=True)

                with tempfile.TemporaryDirectory() as tmpdir:
                    tmp = Path(tmpdir)
                    output_tsv = tmp / "trace_output.tsv"
                    output_zarr = tmp / "trace_output.zarr"
                    # truth.tsv and genetic.map needed by scorer — use absolute paths so
                    # symlinks resolve correctly regardless of cwd.
                    truth_link = tmp / "truth.tsv"
                    truth_link.symlink_to((scenario_dir / "truth.tsv").resolve())
                    map_link = tmp / "genetic.map"
                    map_link.symlink_to((scenario_dir / "genetic.map").resolve())

                    try:
                        run_trace(
                            python=args.python,
                            trace_script=args.trace_script,
                            scenario_dir=scenario_dir,
                            output_tsv=output_tsv,
                            output_zarr=output_zarr,
                            workers=args.workers,
                            maxiter=args.maxiter,
                            chunk_size=args.chunk_size,
                            chunk_overlap=args.chunk_overlap,
                            d=d,
                            gl_weight=tau,
                            prior_weight=args.prior_weight,
                        )
                        metrics = score_scenario_dir(tmp)
                    except Exception as exc:
                        print(f"  ERROR: {exc}")
                        metrics = {k: None for k in METRIC_KEYS}

                row: dict[str, object] = {
                    "scenario": scenario_name,
                    "d": d,
                    "tau": tau,
                    **metrics,
                }
                rows.append(row)
                # Incremental writes so partial results survive interruption
                write_csv(out_dir / "grid_results.csv", rows)
                with open(out_dir / "grid_results.json", "w", encoding="utf-8") as f:
                    json.dump(rows, f, indent=2)

        print(f"  scenario {scenario_name} done")

    pivot_rows = pivot_best(rows, metric=args.pivot_metric)
    write_csv(out_dir / "grid_best.csv", pivot_rows)
    with open(out_dir / "grid_best.json", "w", encoding="utf-8") as f:
        json.dump(pivot_rows, f, indent=2)

    print(f"\nwrote {out_dir / 'grid_results.csv'} ({len(rows)} rows)")
    print(f"wrote {out_dir / 'grid_best.csv'}")

    # Print a quick summary table
    print("\nBest (D, tau) per scenario:")
    print(f"{'scenario':<28} {'best_d':>10} {'best_tau':>9} {f'best_{args.pivot_metric}':>18} {'improvement':>12}")
    for row in pivot_rows:
        pct = f"{float(row['improvement']) * 100:.1f}%" if row.get("improvement") is not None else "n/a"
        best_val = f"{float(row[f'best_{args.pivot_metric}']):.5f}" if row.get(f"best_{args.pivot_metric}") is not None else "n/a"
        print(f"  {row['scenario']:<26} {row['best_d']:>10} {row['best_tau']:>9} {best_val:>18} {pct:>12}")


if __name__ == "__main__":
    main()
