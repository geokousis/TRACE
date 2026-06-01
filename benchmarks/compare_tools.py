#!/usr/bin/env python3
"""
Compare TRACE against an external local-ancestry tool (RFMix 2) on the same
coalescent benchmark scenarios.

TRACE consumes reference *allele frequencies* and genotype *likelihoods*. RFMix
consumes phased reference and query *haplotypes*. Both are derived from the
exact same msprime tree sequence and the exact same selected SNP set, so the
comparison is apples-to-apples on identical ground truth.

For each scenario this script:
  1. re-simulates the deterministic tree sequence (same seed as the fixture),
  2. writes RFMix query/reference VCFs, a sample map, and a genetic map at the
     fixture's selected SNP positions,
  3. runs RFMix, parses its .msp.tsv into per-SNP phi_A for the admixed sample,
  4. scores RFMix with the same metrics used for TRACE, and
  5. prints a side-by-side table (reading TRACE metrics from metrics.json,
     which run_trace_benchmarks.py must have produced first).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


sim = _load("sim", HERE / "simulate_msprime_benchmarks.py")
rtb = _load("run_trace_benchmarks", HERE / "run_trace_benchmarks.py")


def regenerate_scenario(name: str, *, num_snps: int, sequence_length: int, seed: int):
    """Reproduce the tree sequence and selected SNPs for one fixture.

    Returns everything needed to emit RFMix inputs: the tree sequence, the
    reference/focal node ids, the selected SNP positions, and the genotype
    matrix (rows = selected SNPs, cols = all reference + focal sample nodes).
    """
    params = sim.SCENARIOS[name]
    ts, a_nodes, b_nodes, focal_nodes, census_time = sim.simulate_tree_sequence(
        params, sequence_length=sequence_length, seed=seed
    )
    positions, count_a, count_b, focal_geno = sim.collect_variants(
        ts, a_nodes, b_nodes, focal_nodes
    )
    selected = sim.select_informative(
        positions, count_a, count_b, focal_geno,
        n_ref_a=params.n_ref_a, n_ref_b=params.n_ref_b, num_snps=num_snps,
    )
    sel_positions = positions[selected]

    # Build a SNP x node allele matrix for the selected sites. Map each selected
    # site's genomic position back to its variant via a position lookup.
    site_index = {int(round(v.site.position)): k for k, v in enumerate(ts.variants())}
    # collect_variants filtered to biallelic/pos>0 in order, so rebuild aligned
    # genotype rows for the selected variants directly from the tree sequence.
    all_nodes = np.concatenate([a_nodes, b_nodes, focal_nodes])
    geno_rows = []
    var_list = list(ts.variants())
    pos_to_var = {}
    for v in var_list:
        if len(v.alleles) != 2:
            continue
        p = int(round(v.site.position))
        if p <= 0:
            continue
        pos_to_var.setdefault(p, v)  # first variant at this rounded position
    for p in sel_positions:
        v = pos_to_var[int(p)]
        geno_rows.append(v.genotypes[all_nodes])
    geno = np.asarray(geno_rows, dtype=np.int8)  # (n_snps, n_nodes)

    return {
        "params": params,
        "ts": ts,
        "n_ref_a": params.n_ref_a,
        "n_ref_b": params.n_ref_b,
        "positions": sel_positions,
        "geno": geno,  # columns: [A haplotypes..., B haplotypes..., focal hap0, focal hap1]
        "recomb_rate": sim.RECOMB_RATE,
        "map_gap_cm": params.map_gap_cm,
    }


def _diploid_pairs(n_hap: int) -> list[tuple[int, int]]:
    """Pair consecutive haplotype columns into diploid individuals."""
    return [(i, i + 1) for i in range(0, n_hap - n_hap % 2, 2)]


def write_rfmix_inputs(scenario_dir: Path, data: dict) -> dict:
    """Write query VCF, reference VCF, sample map, and RFMix genetic map.

    Returns paths and the index of population A in the sample map ordering.
    """
    positions = data["positions"]
    geno = data["geno"]
    n_a, n_b = data["n_ref_a"], data["n_ref_b"]
    n_snps = len(positions)

    a_cols = list(range(0, n_a))
    b_cols = list(range(n_a, n_a + n_b))
    focal_cols = [n_a + n_b, n_a + n_b + 1]

    a_pairs = _diploid_pairs(len(a_cols))
    b_pairs = _diploid_pairs(len(b_cols))

    rfmix_dir = scenario_dir / "rfmix"
    rfmix_dir.mkdir(parents=True, exist_ok=True)

    def vcf_header(sample_names: list[str]) -> str:
        cols = "\t".join(sample_names)
        return (
            "##fileformat=VCFv4.2\n"
            "##contig=<ID=1>\n"
            '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">\n'
            f"#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{cols}\n"
        )

    # Reference VCF: A and B reference individuals (phased diploids).
    ref_names = [f"A_{i}" for i in range(len(a_pairs))] + [f"B_{i}" for i in range(len(b_pairs))]
    ref_path = rfmix_dir / "reference.vcf"
    with open(ref_path, "w") as h:
        h.write(vcf_header(ref_names))
        for s in range(n_snps):
            calls = []
            for (h0, h1) in a_pairs:
                calls.append(f"{geno[s, a_cols[h0]]}|{geno[s, a_cols[h1]]}")
            for (h0, h1) in b_pairs:
                calls.append(f"{geno[s, b_cols[h0]]}|{geno[s, b_cols[h1]]}")
            h.write(
                f"1\t{int(positions[s])}\trs{s}\tA\tG\t.\tPASS\t.\tGT\t" + "\t".join(calls) + "\n"
            )

    # Query VCF: the single admixed focal individual (phased).
    query_path = rfmix_dir / "query.vcf"
    with open(query_path, "w") as h:
        h.write(vcf_header(["sample"]))
        for s in range(n_snps):
            gt = f"{geno[s, focal_cols[0]]}|{geno[s, focal_cols[1]]}"
            h.write(f"1\t{int(positions[s])}\trs{s}\tA\tG\t.\tPASS\t.\tGT\t{gt}\n")

    # Sample map: <sample>\t<population>. A listed first -> code 0.
    sample_map = rfmix_dir / "sample_map.tsv"
    with open(sample_map, "w") as h:
        for i in range(len(a_pairs)):
            h.write(f"A_{i}\tA\n")
        for i in range(len(b_pairs)):
            h.write(f"B_{i}\tB\n")

    # Genetic map for RFMix: chrom, physical pos, genetic pos (cM).
    cm = positions * data["recomb_rate"] * 100.0
    if data["map_gap_cm"] > 0.0:
        cm = cm.copy()
        cm[len(cm) // 2:] += data["map_gap_cm"]
    gmap_path = rfmix_dir / "genetic_map.tsv"
    with open(gmap_path, "w") as h:
        for p, c in zip(positions, cm):
            h.write(f"1\t{int(p)}\t{c:.8f}\n")

    return {
        "reference": ref_path,
        "query": query_path,
        "sample_map": sample_map,
        "genetic_map": gmap_path,
        "out_base": str(rfmix_dir / "rfmix_out"),
        "a_code": 0,  # A written first in sample map
    }


def run_rfmix(paths: dict, *, rfmix_bin: str, generations: int) -> Path:
    cmd = [
        rfmix_bin,
        "-f", str(paths["query"]),
        "-r", str(paths["reference"]),
        "-m", str(paths["sample_map"]),
        "-g", str(paths["genetic_map"]),
        "-o", paths["out_base"],
        "--chromosome=1",
        f"--generations={generations}",
        "-n", "5",          # small terminal node size (tiny reference panels)
        "--reanalyze-reference",
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    return Path(paths["out_base"] + ".msp.tsv")


def parse_rfmix_phi_a(msp_path: Path, positions: np.ndarray, a_code: int) -> dict[tuple[str, int], float]:
    """Parse RFMix .msp.tsv into phi_A at each fixture SNP position.

    The .msp.tsv gives, per genomic window, the most-likely ancestry code for
    each query haplotype (columns sample.0, sample.1). phi_A at a SNP is the
    fraction of the two haplotypes whose window assignment equals the A code.
    """
    windows: list[tuple[int, int, int, int]] = []  # spos, epos, hap0_code, hap1_code
    with open(msp_path) as h:
        for line in h:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 8:
                continue
            spos, epos = int(parts[1]), int(parts[2])
            hap0, hap1 = int(parts[6]), int(parts[7])
            windows.append((spos, epos, hap0, hap1))

    win_start = np.array([w[0] for w in windows])
    out: dict[tuple[str, int], float] = {}
    for p in positions:
        idx = int(np.searchsorted(win_start, p, side="right") - 1)
        idx = max(0, min(idx, len(windows) - 1))
        _s, _e, c0, c1 = windows[idx]
        phi_a = ((c0 == a_code) + (c1 == a_code)) / 2.0
        out[("1", int(p))] = float(phi_a)
    return out


def score_external(scenario_dir: Path, phi_pred: dict, *, breakpoint_tol_cm: float) -> dict:
    """Score an external tool's phi_A predictions with the TRACE metrics."""
    truth = rtb.read_phi_table(scenario_dir / "truth.tsv", "phi_A")
    shared = sorted(set(truth) & set(phi_pred), key=lambda k: (k[0], k[1]))
    y_true = np.array([truth[k] for k in shared])
    y_pred = np.array([phi_pred[k] for k in shared])

    gmap_path = scenario_dir / "genetic.map"
    w = cm_pos = None
    if gmap_path.exists():
        gmap = rtb.load_genetic_map_cm(gmap_path)
        w = rtb._cm_weights_for_snps(shared, gmap)
        cm_pos = rtb._cm_positions_for_snps(shared, gmap)

    pm = rtb._point_metrics(y_true, y_pred, w)
    metrics = {
        "n_snps": len(shared),
        "mse_phi_A": pm["mse"],
        "rmse_phi_A": pm["rmse"],
        "mae_phi_A": pm["mae"],
        "hard_call_accuracy_0_5": pm["hard_call_accuracy_0_5"],
        "dosage_accuracy_3class": pm["dosage_accuracy_3class"],
        "pearson_phi_A": pm["pearson"],
    }
    if cm_pos is not None:
        bp = rtb.breakpoint_metrics(cm_pos, y_true, y_pred, tol_cm=breakpoint_tol_cm)
        metrics.update({
            "n_pred_breakpoints": bp["n_pred_breakpoints"],
            "bp_f1": bp["f1"],
        })
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare TRACE vs RFMix on benchmark scenarios.")
    p.add_argument("--benchmark-dir", required=True,
                   help="Directory of scenario fixtures (must already contain TRACE metrics.json).")
    p.add_argument("--scenarios", default="all")
    p.add_argument("--num-snps", type=int, default=400)
    p.add_argument("--sequence-length", type=int, default=20_000_000)
    p.add_argument("--seed-base", type=int, default=13,
                   help="Must match the seed used to generate the fixtures (seed = base+1000+offset).")
    p.add_argument("--rfmix-bin", default=str(REPO / ".mamba-env" / "bin" / "rfmix"))
    p.add_argument("--generations", type=int, default=8)
    p.add_argument("--breakpoint-tol-cm", type=float, default=0.5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    bench = Path(args.benchmark_dir)
    names = list(sim.SCENARIOS) if args.scenarios == "all" else args.scenarios.split(",")

    rows = []
    for name in names:
        scenario_dir = bench / name
        if not (scenario_dir / "truth.tsv").exists():
            print(f"skip {name}: no fixture in {bench}")
            continue
        offset = list(sim.SCENARIOS).index(name)
        seed = args.seed_base + 1000 + offset
        print(f"[{name}] regenerating tree sequence (seed={seed}) and running RFMix...", flush=True)
        try:
            data = regenerate_scenario(
                name, num_snps=args.num_snps,
                sequence_length=args.sequence_length, seed=seed,
            )
            paths = write_rfmix_inputs(scenario_dir, data)
            msp = run_rfmix(paths, rfmix_bin=args.rfmix_bin, generations=args.generations)
            phi_pred = parse_rfmix_phi_a(msp, data["positions"], paths["a_code"])
            rfmix_metrics = score_external(
                scenario_dir, phi_pred, breakpoint_tol_cm=args.breakpoint_tol_cm
            )
        except subprocess.CalledProcessError as exc:
            print(f"  RFMix failed for {name}: {exc.stderr[-400:] if exc.stderr else exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - surface and continue
            print(f"  error for {name}: {exc!r}")
            continue

        trace_metrics = {}
        mfile = scenario_dir / "metrics.json"
        if mfile.exists():
            trace_metrics = json.load(open(mfile))

        with open(scenario_dir / "rfmix_metrics.json", "w") as h:
            json.dump(rfmix_metrics, h, indent=2)

        rows.append((name, trace_metrics, rfmix_metrics))

    # Side-by-side report.
    print("\n" + "=" * 96)
    print(f"{'scenario':22} | {'metric':16} | {'TRACE':>10} | {'RFMix':>10} | winner")
    print("-" * 96)
    keys = [("mse_phi_A", "MSE", "lower"), ("pearson_phi_A", "Pearson", "higher"),
            ("hard_call_accuracy_0_5", "hardcall acc", "higher"),
            ("dosage_accuracy_3class", "dosage acc(3cls)", "higher"),
            ("bp_f1", "breakpoint F1", "higher")]
    summary = {}
    for name, tm, rm in rows:
        for k, label, better in keys:
            tv, rv = tm.get(k), rm.get(k)
            win = ""
            if isinstance(tv, (int, float)) and isinstance(rv, (int, float)):
                if better == "lower":
                    win = "TRACE" if tv < rv else "RFMix" if rv < tv else "tie"
                else:
                    win = "TRACE" if tv > rv else "RFMix" if rv > tv else "tie"
                summary[win] = summary.get(win, 0) + 1
            fmt = lambda x: "n/a" if not isinstance(x, (int, float)) else f"{x:.4f}"
            print(f"{name:22} | {label:16} | {fmt(tv):>10} | {fmt(rv):>10} | {win}")
        print("-" * 96)
    print(f"metric-wins: {summary}")
    out_path = bench / "tool_comparison.json"
    json.dump(
        {name: {"trace": tm, "rfmix": rm} for name, tm, rm in rows},
        open(out_path, "w"), indent=2,
    )
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
