# TRACE Benchmark Suite

Reproducible simulation benchmarks for validating TRACE against known
local-ancestry truth, with trivial/statistical baselines and a head-to-head
comparison against the gold-standard haplotype tool **RFMix 2**.

## Why this benchmark is trustworthy

The admixed benchmark individual is simulated **directly in the coalescent**.
A single `msprime` tree sequence produces the reference populations,
recombination, mutation, and the admixed sample together. Local-ancestry truth
is read back from that tree sequence via a **census event**, and the sample's
genotypes are its real simulated haplotypes.

This matters. An earlier version drew the sample's genotypes from
`p = phi*f_A + (1-phi)*f_B` followed by `binomial(2, p)` — which is *exactly*
TRACE's own likelihood model. Scoring against that truth only measured the
optimizer, never model misspecification, and inflated apparent accuracy. With
coalescent truth the genotypes carry real linkage disequilibrium and the
ancestry is a real recombination mosaic, so the benchmark can probe where the
allele-frequency-only model actually breaks down.

How truth is obtained:

- The admixed population is formed by an admixture pulse from the reference
  sources at `t_admix`.
- A census event just after the pulse labels every extant lineage with its
  source population.
- For each of the two haplotypes, at each SNP we walk up the local tree to its
  census ancestor; that ancestor's population is the local ancestry there.
- `phi_A` is the fraction of the two haplotypes whose census ancestor is
  population `A` (values in {0, 0.5, 1}).

Informative SNPs are chosen per genomic bin (most differentiated site per bin)
so the SNP set covers the chromosome uniformly instead of clustering on the few
most differentiated loci.

## Install

From the repository root:

```bash
/home/kousis/work/trace/.mamba-env/bin/python -m pip install -r requirements-benchmark.txt
```

For the RFMix comparison (optional), install RFMix 2 from bioconda:

```bash
MAMBA_PKGS_DIRS=/home/kousis/work/trace/.mamba-pkgs \
  micromamba install -y -p /home/kousis/work/trace/.mamba-env \
  -c bioconda -c conda-forge rfmix
```

## 1. Generate scenarios

```bash
/home/kousis/work/trace/.mamba-env/bin/python benchmarks/simulate_msprime_benchmarks.py \
  --out-dir benchmark_runs \
  --scenario all \
  --num-snps 2000 \
  --sequence-length 20000000 \
  --seed 13
```

Each scenario directory gets:

- `reference_panel.tsv` — count-format reference panel (`derived_*`/`total_*`).
- `genetic.map` — PLINK-style genetic map.
- `sample.vcf` — unphased diploid genotype likelihoods for the focal sample.
- `truth.tsv` — true local ancestry `phi_A` at each SNP (coalescent census).
- `metadata.json` — scenario parameters, including `use_ghost`, `map_gap_cm`,
  and `n_ref_*`, which the runner uses to choose scenario-aware TRACE flags.

## 2. Run TRACE and score

```bash
/home/kousis/work/trace/.mamba-env/bin/python benchmarks/run_trace_benchmarks.py \
  --benchmark-dir benchmark_runs \
  --trace-script /home/kousis/work/trace/trace_solver.py \
  --python /home/kousis/work/trace/.mamba-env/bin/python \
  --workers 2 --maxiter 300
```

The runner reads each scenario's `metadata.json` and applies the right TRACE
options automatically (no hard-coded scenario names):

- `use_ghost: true` → adds `--unknown-state` (a third "unknown" ancestry column)
  so unsampled ghost tracts have somewhere to go instead of being forced onto A
  or B.
- `map_gap_cm > 0` → passes `--max-imputation-gap-cm 1.0` to split across the
  centromere-like gap.
- `--freq-shrinkage` (TRACE default 8.0) shrinks small/bottlenecked reference
  panels toward the cross-population mean, which matters for `bottleneck_drift`.

### Metrics

For each scenario the runner writes `metrics.json`. Alongside MSE/RMSE/MAE,
bias, R², and Pearson on `phi_A`, it reports:

- **Baselines** so the MSE is interpretable: a constant-0.5 floor, a single
  global-MLE ancestry (best constant fit — no local resolution), and an
  independent per-site MLE (TRACE with the smoothing prior off). `skill_vs_*`
  keys give `1 - mse_TRACE / mse_baseline` (positive means TRACE wins).
- **`dosage_accuracy_3class`** — snap both truth and prediction to the nearest
  copy-number level {0, 0.5, 1} and compare. This is how local-ancestry dosage
  is actually scored. Unlike the `>=0.5` hard call it does not penalize a
  continuous estimate that sits correctly near 0.5 (heterozygous) for landing
  on the "wrong" side of the threshold, and it does not reward a tool merely
  for emitting discrete calls.
- **Breakpoint metrics** (`bp_precision`, `bp_recall`, `bp_f1`) matching
  inferred ancestry-tract boundaries to truth within `--breakpoint-tol-cm`
  (default 0.5 cM), plus mean tract length in cM. These judge the sharp-tract
  recovery that pointwise MSE rewards over-smoothing on.

## 3. Compare against RFMix 2

TRACE consumes reference *allele frequencies* and genotype *likelihoods*; RFMix
consumes phased reference and query *haplotypes*. Both are derived from the
exact same tree sequence and the exact same SNP set, so the comparison is
apples-to-apples on identical ground truth. Run TRACE first (step 2), then:

```bash
/home/kousis/work/trace/.mamba-env/bin/python benchmarks/compare_tools.py \
  --benchmark-dir benchmark_runs \
  --scenarios all \
  --num-snps 2000 \
  --seed-base 13
```

`--num-snps` and `--seed-base` must match what was used to generate the
fixtures (the simulator seed is `seed_base + 1000 + scenario_offset`).
`compare_tools.py` re-simulates the deterministic tree sequence, writes RFMix
inputs, runs RFMix, parses its `.msp.tsv` into per-SNP `phi_A`, scores it with
the same metrics, and prints a side-by-side table plus `tool_comparison.json`.

### Representative result (400 SNPs, seed 13)

| scenario | MSE (TRACE / RFMix) | Pearson (T / R) | dosage 3-class (T / R) | breakpoint F1 (T / R) |
|---|---|---|---|---|
| recent_admixture | **0.014** / 0.015 | **0.89** / 0.87 | **0.96** / 0.94 | 0.67 / 0.40 |
| ancient_mosaic | **0.047** / 0.109 | **0.67** / 0.20 | **0.77** / 0.57 | **0.67** / 0.21 |
| isolation_by_distance | **0.121** / 0.237 | **0.44** / 0.14 | **0.53** / 0.38 | **0.96** / 0.23 |
| ghost_introgression | **0.038** / 0.062 | **0.89** / 0.81 | **0.80** / 0.75 | **0.52** / 0.45 |
| bottleneck_drift | 0.030 / **0.000** | n/a | 0.79 / **1.00** | n/a |
| ancient_dna | **0.020** / 0.041 | **0.82** / 0.71 | **0.94** / 0.84 | **0.67** / n/a |

TRACE leads on continuous accuracy (MSE/Pearson) in every scenario, on dosage
in 5/6, and on breakpoint F1 wherever it is defined. RFMix wins
`bottleneck_drift`, where the truth is a single constant (100% A) and a discrete
classifier with reference reanalysis is near-perfect by construction. Numbers
will shift with `--num-snps`, seed, and replicate count; treat this as
illustrative, not a published result.

## Publication matrix

For replicate seeds and multiple SNP densities:

```bash
/home/kousis/work/trace/.mamba-env/bin/python benchmarks/publication_benchmark.py \
  --out-dir publication_runs \
  --profile publication \
  --replicates 5 \
  --workers 2 --maxiter 300
```

Profiles: `smoke` (80), `small` (500, 2000), `medium` (2000, 10000),
`large` (10000, 50000), `publication` (2000, 10000, 50000). Override with
`--sizes 1000,5000,20000` and `--scenarios recent_admixture,ancient_dna`.
Outputs `detail_metrics.csv`/`.json` (per replicate) and
`summary_metrics.csv`/`.json` (mean and sd per scenario and size).

## Scenarios

`recent_admixture` — Recent single-pulse admixture (t=8) with long coalescent
ancestry blocks. Sharp domain-wall recovery from real haplotypes.

`ancient_mosaic` — Old admixture (t=150) with many short coalescent tracts.
Local-ancestry resolution under dense breakpoints.

`isolation_by_distance` — Very old admixture (t=600) giving a fine, stable
near-50/50 mosaic. Tests that TRACE does not invent recent hard-switch tracts
where the truth is diffuse.

`ghost_introgression` — Three-way admixture from A, B and an unsampled GHOST
population (t=40). The panel contains only A and B. Tests graceful behaviour
under unsampled ancestry (TRACE uses `--unknown-state` here).

`bottleneck_drift` — Focal individual is 100% A, but the A reference panel is a
small bottlenecked sample (8 haplotypes) with drift-noisy, biased frequencies.
Tests resistance to false ancestry (TRACE uses `--freq-shrinkage`).

`ancient_dna` — Recent admixture (t=8) at low coverage (0.5x) with elevated
error. Tests whether the kinetic term bridges weak GL evidence.

`centromere_abyss` — Recent admixture with a large cM gap at the chromosome
midpoint. Tests `--max-imputation-gap-cm` splitting.

## Adding your own scenarios

Scenarios are plain `ScenarioParams` dataclass entries in the `SCENARIOS` dict
in `simulate_msprime_benchmarks.py`. To add one, append an entry — no other
file needs editing; the runner and comparison pick it up from `metadata.json`.

```python
SCENARIOS["my_scenario"] = ScenarioParams(
    name="my_scenario",
    description="One-line description shown in metadata.json.",
    t_admix=25.0,             # generations since the admixture pulse
    proportions=(0.7, 0.3),   # (A, B), or (A, B, GHOST) when use_ghost=True
    use_ghost=False,          # True adds an unsampled GHOST source
    focal_pure=None,          # e.g. "A" for a 100%-A individual (no admixture)
    n_ref_a=40, n_ref_b=40,   # reference haplotypes per source population
    coverage=None,            # None = clean hard calls; else mean read depth
    error_rate=0.01,
    map_gap_cm=0.0,           # >0 inserts a genetic-map gap at the midpoint
)
```

Knobs and what they probe:

- **`t_admix`** — small = long tracts (easy), large = short diffuse tracts
  (hard). The single biggest difficulty lever.
- **`proportions`** — global ancestry balance; lopsided mixes (e.g. 0.9/0.1)
  stress rare-ancestry detection.
- **`use_ghost`** — adds a third, unsampled source; the runner auto-enables
  `--unknown-state`.
- **`focal_pure`** — a non-admixed individual; truth is 100% that population
  (a strong test against inventing tracts).
- **`n_ref_a` / `n_ref_b`** — shrink one to simulate a small/biased panel;
  exercises `--freq-shrinkage`.
- **`coverage`** — low values (0.3–1.0) produce realistic low-coverage / aDNA
  genotype likelihoods.
- **`map_gap_cm`** — inserts a centromere-like gap; the runner auto-enables
  `--max-imputation-gap-cm`.

Then generate and run as usual:

```bash
.../python benchmarks/simulate_msprime_benchmarks.py --out-dir my_runs --scenario my_scenario --num-snps 2000
.../python benchmarks/run_trace_benchmarks.py --benchmark-dir my_runs --trace-script ../trace_solver.py --python .../python
```

Ideas for a bigger evaluation: a lopsided 90/10 pulse; a small B panel as well
as A; two-pulse admixture (would need a second `add_admixture` in
`build_demography`); higher SNP densities (`--num-snps 10000+`); and several
seeds per scenario via `publication_benchmark.py --replicates`.

## Notes

This is a validation scaffold, not a published benchmark. The truth is
coalescent-derived rather than drawn from TRACE's own likelihood model, the
scorer reports trivial/statistical baselines and breakpoint + dosage metrics,
and `compare_tools.py` provides an external RFMix baseline on identical
fixtures. For a paper, extend it with many admixed individuals per scenario,
larger chromosomes, more external tools, and confidence intervals.
