# TRACE Solver

A vectorized Python/SciPy solver for **local ancestry inference from genotype
likelihoods**, using reference allele frequencies plus a genetic-distance
smoothing prior. It estimates a continuous per-SNP ancestry field `phi` along
the genome from unphased low-coverage data (VCF/BAM/CRAM), with K=2 and K≥3
support.

The binary K=2 solver minimizes:

```text
S(phi) = D * sum_i  (sqrt(huber_delta^2 + (phi[i+1] - phi[i])^2) - huber_delta) / d_cM_i
         - sum_i log P(GL_i | phi_i, f_A_i, f_B_i)
```

`phi` is bounded to `[0, 1]` and optimized with SciPy's L-BFGS-B backend using
an analytical gradient. The first term is a **pseudo-Huber total-variation
smoothing prior** along genetic distance (`d_cM`): near-quadratic for small
ancestry changes (smooths noise), near-linear for large jumps (keeps tract
boundaries sharp). `D` is the smoothing strength; `huber_delta` (default `0.05`)
sets the noise/edge corner. The second term is the genotype-likelihood data fit
under `p = phi*f_A + (1-phi)*f_B`, marginalized over reference-frequency
uncertainty. The multi-population solver optimizes `N x (K - 1)` unconstrained
logits, appends a zero baseline population, and maps to simplex ancestry
fractions with softmax.

### What this is, in standard terms

Stripped of framing, TRACE is a **MAP estimate of a continuous ancestry field
under a Gauss–Markov / total-variation (fused-lasso-style) smoothing prior**,
fit by L-BFGS-B. It deliberately uses only marginal reference *allele
frequencies*, not reference *haplotypes*, so unlike RFMix/Loter it carries no
LD/phasing information — by design. That makes its niche **fast, frequency-only,
low-coverage / ancient-DNA local ancestry** where haplotype phasing is
unreliable. On the included msprime benchmark it matches or beats RFMix 2 on
continuous accuracy (MSE/Pearson), copy-number dosage, and tract-boundary
recovery in the short-tract, diffuse, and low-coverage regimes; see
[benchmarks/README.md](benchmarks/README.md).

## Run

Create a local environment:

```bash
MAMBA_PKGS_DIRS=/home/kousis/work/trace/.mamba-pkgs \
  micromamba create -y -p /home/kousis/work/trace/.mamba-env \
  -c conda-forge python=3.11 numpy scipy pip
/home/kousis/work/trace/.mamba-env/bin/python -m pip install cyvcf2 pysam 'zarr<3'
/home/kousis/work/trace/.mamba-env/bin/python -m pip install tqdm matplotlib
```

Run a gradient check:

```bash
/home/kousis/work/trace/.mamba-env/bin/python trace_solver.py --gradient-check
```

Run the multi-population simulation demo:

```bash
/home/kousis/work/trace/.mamba-env/bin/python trace_solver.py --n-snps 100000 --k 3 --d 0.001
```

Run from a VCF:

```bash
/home/kousis/work/trace/.mamba-env/bin/python trace_solver.py \
  --vcf sample.vcf.gz \
  --sample SAMPLE_ID \
  --reference-panel reference_panel.tsv \
  --genetic-map genetic.map \
  --output ancestry.tsv \
  --output-zarr ancestry.zarr \
  --chunk-size 100000 \
  --chunk-overlap 10000 \
  --max-imputation-gap-cm 1.0 \
  --sample-sex male \
  --workers 8
```

Build a Zarr-backed reference panel:

```bash
/home/kousis/work/trace/.mamba-env/bin/python trace_solver.py \
  --reference-panel reference_panel.tsv \
  --build-reference-zarr reference_panel.zarr
```

Run from a VCF using lazy Zarr reference-panel windows:

```bash
/home/kousis/work/trace/.mamba-env/bin/python trace_solver.py \
  --vcf sample.vcf.gz \
  --sample SAMPLE_ID \
  --reference-panel-zarr reference_panel.zarr \
  --genetic-map genetic.map \
  --output-zarr ancestry.zarr \
  --chunk-size 100000 \
  --chunk-overlap 10000 \
  --workers 8
```

Run from a BAM:

```bash
/home/kousis/work/trace/.mamba-env/bin/python trace_solver.py \
  --bam sample.bam \
  --reference-panel reference_panel.tsv \
  --genetic-map genetic.map \
  --output ancestry.tsv \
  --output-zarr ancestry.zarr \
  --chunk-size 100000 \
  --chunk-overlap 10000 \
  --max-imputation-gap-cm 1.0 \
  --sample-sex male \
  --fasta reference.fa \
  --workers 8
```

Generate an HTML report:

```bash
/home/kousis/work/trace/.mamba-env/bin/python trace_solver.py \
  --report-zarr ancestry.zarr \
  --report-output ancestry_report.html
```

The VCF parser uses `GL`, then `PL`, then `GT` fallback. The BAM/CRAM parser
uses `pysam` targeted pileups at the exact reference-panel SNP coordinates;
it does not discover novel variants or scan for de novo SNPs. BAM/CRAM mode
requires `ref` and `alt` columns in the reference panel. CRAM inputs should
pass `--fasta reference.fa`.

Allele harmonization handles reverse-complement REF/ALT pairs. Palindromic
A/T and C/G SNPs are not strand-flipped automatically because their strand is
ambiguous.

Chunking uses overlapping windows. In overlap regions, chunk results are
linearly blended to reduce optimizer edge effects. Large genetic-map gaps are
not bridged: if adjacent SNPs differ by more than `--max-imputation-gap-cm`,
TRACE starts a separate optimization segment.

Ploidy is controlled by `--sample-sex` and `--haploid-chroms`. For example,
`--sample-sex male` treats X/Y/MT as haploid. Extra haploid contigs can be
specified as comma-separated names:

```bash
--haploid-chroms chrX,chrY,chrM
```

Reference panel frequency format:

```text
chrom	pos	ref	alt	f_WHG	f_EEF	f_STEPPE
1	100	A	G	0.85	0.20	0.10
```

Reference panel count format:

```text
chrom	pos	ref	alt	derived_WHG	total_WHG	derived_EEF	total_EEF	derived_STEPPE	total_STEPPE
1	100	A	G	42	50	10	50	5	50
```

Count-format panels are Beta-smoothed automatically with `--alpha` and
`--beta`, both defaulting to `1.0` for Laplace smoothing. Frequency-format
panels must already be smoothed so all values are strictly inside `(0, 1)`.

For count-format panels, `--freq-shrinkage` (default `8.0`) applies
empirical-Bayes shrinkage of each population's frequency toward the
cross-population mean, weighted by panel size `w = n / (n + S)`. A small or
bottlenecked panel (few haplotypes) has noisy, often biased frequencies that
otherwise drag local ancestry toward whichever source the point estimate
resembles; shrinkage corrects that *mean* while leaving large panels (high `n`)
essentially untouched. Set `--freq-shrinkage 0` to disable. This is distinct
from the likelihood's built-in marginalization over reference-frequency
*variance*, which widens uncertainty but cannot fix a biased mean.

For samples with unsampled ("ghost") ancestry that matches neither reference
source, `--unknown-state` appends a third, flat "unknown" ancestry column so
ghost tracts have somewhere to go instead of being forced onto A or B.
`--unknown-penalty` (default `0.02`; `~0.8` works well on the ghost benchmark)
controls how strongly the unknown state is discouraged.

TRACE can auto-calibrate the kinetic coupling `D` from the input data by masked
cross-validation:

```bash
--auto-d --d-grid 0.003,0.01,0.03,0.1,0.3,1.0,3.0 --calibration-mask-fraction 0.10
```

For each candidate `D`, TRACE hides a random fraction of genotype likelihoods,
fits ancestry on the remaining sites, then scores the held-out true likelihoods.
The selected `D` is used for the main chunked run.

The kinetic smoothing uses a standard pseudo-Huber penalty whose large-jump
slope is `1`, so `D` is the genuine total-variation smoothing strength and is
independent of the corner scale `huber_delta` (default `0.05`, the ancestry
difference above which a step is treated as a cheap domain wall rather than
noise). Useful `D` values are typically in the `0.03`–`1.0` range; far smaller
values leave the data term essentially unsmoothed.

Genetic maps can be PLINK `.map` files:

```text
1	rs100	0.00	100
1	rs200	0.01	200
```

or whitespace-delimited tables with `chrom`, `pos`/`bp`, and `cM` columns.

## Status

This is executable numerical code, not pseudocode. It has:

- A concrete objective function.
- A matching analytical gradient.
- Bounded optimization through SciPy L-BFGS-B.
- Input validation.
- cM-weighted pseudo-Huber total-variation smoothing.
- Diploid genotype likelihoods marginalized over reference-frequency
  uncertainty (Beta-posterior second moment), so noisy small panels do not
  force false ancestry switches.
- Beta-smoothed reference-panel frequencies, with optional empirical-Bayes
  `--freq-shrinkage` toward the cross-population mean for small/bottlenecked
  panels (panel-size weighted; large panels are barely affected).
- Multi-population simplex ancestry through a `K - 1` softmax parameterization.
- VCF parsing through `cyvcf2`.
- BAM pileup parsing through `pysam`.
- CRAM input through `pysam` with `--fasta`.
- Zarr-backed reference-panel input with lazy frequency-window loading.
- Reverse-complement allele harmonization.
- PLINK/header genetic-map parsing with cM interpolation.
- Chromosome/chunk execution with `ProcessPoolExecutor`.
- Progress bars through `tqdm` and structured logging.
- Overlap stitching to reduce chunk-boundary effects.
- Max-gap splitting across centromeres or other uninformative spans.
- Haploid/diploid likelihood switching for sex chromosomes and organellar DNA.
- Masked hold-out cross-validation for `D` auto-calibration.
- Optional Zarr output for large on-disk `phi` matrices.
- HTML visual reports with chromosome painting and K=3 ternary trajectory plots.
- A finite-difference gradient check.
- Binary K=2 TSV outputs include local standard-error columns from the
  tridiagonal inverse Hessian. Values near `0.5` mean the site is effectively
  unconstrained on the `[0, 1]` ancestry scale.
- Phased VCF mode writes one combined MSP file with separate `.0` and `.1`
  haplotype call columns.

It is still a research prototype, not a validated production genetics model.
For production ancestry inference, it still needs deeper reference-panel QC,
large-cohort benchmarking against known truth sets, and broader real-world
BAM/CRAM fixture coverage.

The multi-population solver includes a small default `--logit-l2 0.001`
penalty. This keeps softmax logits finite when the likelihood wants pure
simplex-boundary ancestry. Lower values are less biased but may converge
slowly; higher values converge faster but shrink ancestry fractions away from
zero and one.

## Benchmarks

Reproducible `msprime` benchmark scripts live in
[benchmarks/README.md](benchmarks/README.md). They generate documented
validation scenarios where local-ancestry truth comes directly from the
coalescent tree sequence (a census event), run TRACE through the public CLI,
and score inferred ancestry against that truth with trivial/statistical
baselines, copy-number dosage accuracy, and tract-breakpoint metrics.

`benchmarks/compare_tools.py` runs a head-to-head comparison against **RFMix 2**
on identical fixtures (same tree sequence, same SNPs). On the included scenarios
TRACE leads on continuous accuracy (MSE/Pearson) everywhere, on dosage in 5 of
6 scenarios, and on tract-breakpoint F1 wherever defined; RFMix wins the
constant-ancestry `bottleneck_drift` case. New scenarios are added by appending
one `ScenarioParams` entry — see the benchmark README.

For a publication-style benchmark matrix:

```bash
/home/kousis/work/trace/.mamba-env/bin/python benchmarks/publication_benchmark.py \
  --out-dir publication_runs \
  --profile publication \
  --replicates 5 \
  --workers 2 \
  --maxiter 300
```

The orchestrator writes per-replicate and aggregate metrics to
`detail_metrics.csv`, `summary_metrics.csv`, and matching JSON files.
