[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

# own-baseline

A single-cell potency or stemness score is usually validated by correlating it
against a pseudotime, a marker panel, or a known hierarchy. A score that
restates sequencing depth passes that test, and so does a score carrying real
ordering information: the criterion cannot separate them, because the thing
you are worried about correlates with the gold standard too. This repository
implements the check that can: residualize the score on the low-order statistic
its own authors say it approximates, and ask whether what is left still orders
cells. A score at rank correlation 0.93 with node degree can carry substantial
skill beyond it; another at 0.998 carries none. Pairwise correlation does not
tell you which you have, in either direction. Also in here: the field audit
that came out of applying this to thirteen published scores, and the finding
that our own strong hypothesis, that the whole family reduces to a fixed
low-order statistic, is false for most of them.

## Quick start

```bash
pip install own-baseline            # the diagnostic, the command and the null grid
```

For the audit, the figures and the reproduction chain, work from the
repository:

```bash
git clone https://github.com/IskakovDamir/ownbaseline
cd ownbaseline
pip install -e ".[full]"
```

### From the command line

```bash
ownbaseline check cells.h5ad \
    --score   obs:cytotrace \
    --ordinal obs:stage \
    --ordinal-source experimental \
    --json report.json
```

All four primitives are computed unless you name one, the direction check is
printed above the first margin, and every value is placed against its own
measured null floor rather than against zero. `--ordinal-source` is required and
has no override: a pseudotime computed from the same expression matrix is not a
valid ground truth for this test, because the primitive predicts such an ordinal
too.

```
ownbaseline floors --n 39505 --rho 0.4844      # the floor at that cell
ownbaseline floors --list-grid                 # the design points that exist
ownbaseline primitives cells.h5ad --out prim.npz
ownbaseline check cells.h5ad --primitives prim.npz ...   # reuse them
ownbaseline verify report.json --rerun ...     # does it still reproduce?
```

`primitives` and `check` compose: the first writes the low-order statistics once,
with a provenance block recording the scaffold, the cell count and a sha256 per
vector, and the second reads them back instead of recomputing. On a large atlas
the degree primitive is the expensive one and this is the difference between
computing it once and computing it on every run.

The degree primitive needs an interaction scaffold, and none is bundled: an
interactome is 100 MB and the release you pick changes every number downstream.
Point at a prepared one, or build it from a STRING release:

```bash
ownbaseline check cells.h5ad --score obs:ccat --ordinal obs:stage \
    --ordinal-source experimental \
    --scaffold prepared/string_cols.npz          # col_idx + degree
# or
    --string-links protein.links.v12.0.txt.gz \
    --string-info  protein.info.v12.0.txt.gz --string-threshold 700
```

`--scaffold-null N` adds the degree-permutation control on top of that: it
permutes which gene carries which degree, keeping the degree distribution and
breaking the correspondence with expression. Read what it prints before using
the number. On the systems in the paper the gap over that null was 0.120 for
signalling entropy against 0.057 for CCAT while both sat at a conditional skill
of about zero, so a rule reading a larger gap as more genuine would have ranked
them backwards, and the entropy null itself came in below chance. It is a
diagnostic. The own baseline is the arbiter.

`check` wraps the conditional-skill estimator. It does not wrap
`run_own_baseline`, the marginal gap, which returns a pass when a score is
indistinguishable from gene counts and on sorted haematopoietic progenitors
returns the exact false positive this repository exists to expose.

Conventions, mostly from [clig.dev](https://clig.dev):

| | |
|---|---|
| `--json -` | the report on stdout, everything human on stderr, so it pipes |
| `-q`, `--quiet` | the numbers and the verdict, nothing else, and no progress line |
| `--no-color` | as does `NO_COLOR`; `FORCE_COLOR` turns it back on |
| exit `0` / `1` / `2` / `3` | ran / something asked for was unavailable / bad command line / the test does not apply |

Both global flags are accepted before or after the verb. Colour is off already
when stdout is not a terminal. The bootstrap prints its progress on stderr and
Ctrl-C stops it without a traceback and without writing anything.

### From Python

```python
import numpy as np
from own_baseline import conditional_skill_report

rng = np.random.default_rng(0)
n = 2000
ordinal   = np.floor(6 * rng.random(n))                # stage, fixed before sequencing
primitive = ordinal + 1.5 * rng.standard_normal(n)     # e.g. gene count per cell

honest = 0.5 * primitive + 1.5 * ordinal + rng.standard_normal(n)
sham   = np.exp(primitive / 2)                         # a monotone function of it

conditional_skill_report(honest, ordinal, {'gene_count': primitive},
                         score_name='honest_score', n_boot=(100, 300))
conditional_skill_report(sham, ordinal, {'gene_count': primitive},
                         score_name='sham_score', n_boot=(100, 300))
```

Real output, about thirty seconds:

```
[conditional_skill] score=honest_score n=2000

  vs primitive: gene_count
    scipy_weightedtau  marginal gap  +0.1015   conditional skill +0.6786 CI[+0.6421,+0.7009]  ADDS-BEYOND
    kang_wdm_taub      marginal gap  +0.2365   conditional skill +0.4316 CI[+0.4100,+0.4506]  ADDS-BEYOND
    => ADDS-BEYOND
[conditional_skill] score=sham_score n=2000

  vs primitive: gene_count
    scipy_weightedtau  marginal gap  +0.0000   conditional skill -0.5439 CI[-0.5770,+0.8255]  TAUTOLOG
    kang_wdm_taub      marginal gap  +0.0000   conditional skill -0.6385 CI[-0.6491,+0.6440]  TAUTOLOG
    => TAUTOLOG
```

`sham_score` is `exp(primitive / 2)`, a monotone reparametrization that adds
nothing. Its marginal gap is exactly `+0.0000` and its verdict is `TAUTOLOG`,
which is right. But its conditional skill should also be zero and reports
`-0.54` with an interval spanning most of [-1, 1]. That is not a near-null; it
is rounding debris being scored as signal, and the verdict lands correctly only
because the interval is wide enough to contain zero anyway. It is a real defect
in the estimator, documented below as **F-2**, recorded by a failing test, and
deliberately not patched.

Two entry points, in increasing strength:

| function | what it computes | when to use it |
|---|---|---|
| `run_own_baseline(adata, score, gt_ordinal)` | marginal gap `tau(score, ordinal) − tau(gene counts, ordinal)` | a first look; one line, no residualization |
| `conditional_skill_report(score, ordinal, primitives)` | rank residual on the primitive(s), scored under two kernels with a bootstrap interval, plus a direction check | the test the paper reports, **prefer this** |

The verdict vocabulary says which way the evidence points and is not a
pass/fail: `REDUCES-TO-PRIMITIVE`, `INCONCLUSIVE`, `ADDS-BEYOND-PRIMITIVE` for
the marginal gap; `TAUTOLOG`, `ADDS-BEYOND`, `SIGN-FLIPPED`,
`INCONCLUSIVE-kernel-disagree` for conditional skill.

**Check the direction before reading any margin.** In sorted haematopoietic
progenitors the gene-count primitive orders cells *below* chance: HSCs carry a
median 916 detected genes, the GMPs below them carry 1,404, and the primitive
scores AUROC 0.378 against the known hierarchy. A score that beats that
baseline has supplied a sign correction, not biology, and nothing in the margin
shows it. `conditional_skill_report` flags this; `run_own_baseline` does not.

## Install

```bash
pip install own-baseline
ownbaseline                         # the splash, and the four verbs
```

For an isolated install that brings its own interpreter and puts the command on
your PATH without touching a project environment:

```bash
uv tool install --python 3.12 own-baseline
```

`uv` is a single binary: `curl -LsSf https://astral.sh/uv/install.sh | sh`, or
`brew install uv`. Pinning 3.12 keeps you on interpreters that numpy, scipy and
scikit-learn publish wheels for; on a very new Python those wheels may not exist
yet and pip will try to build them from source.

For what is on `main` and not yet released:

```bash
pip install "git+https://github.com/IskakovDamir/ownbaseline"
```

**If pip says it cannot find the package at all.** An error naming
`Requires-Python >=3.10` followed by `No matching distribution found` means the
pip you ran belongs to a Python older than 3.10. On macOS that is usually
`/Library/Developer/CommandLineTools/usr/bin/python3`, which is 3.9 and comes
with the Xcode command line tools; `python3 -V` says which one you have. Nothing
is wrong with the package, and pip is reporting this correctly. `uv` above fixes
it without touching that interpreter, because it brings its own.

**If pipx fails before it reaches this package.** An error naming `ensurepip`,
`venv --clear` or pipx's own `shared` directory is pipx repairing itself against
a Python it cannot bootstrap, and it happens with the newest Homebrew Python.
Nothing here is involved yet. Either use `uv` above, or repair pipx:

```bash
rm -rf "$HOME/Library/Application Support/pipx/shared"
brew install python@3.12
pipx install --python "$(brew --prefix python@3.12)/bin/python3.12" own-baseline
```

Or skip the installers entirely, which always works:

```bash
python3 -m venv ~/.ownbaseline
~/.ownbaseline/bin/pip install own-baseline
~/.ownbaseline/bin/ownbaseline
```

That gives the diagnostic, the shipped 200-seed null grid and the `ownbaseline`
command. It does not give the reproduction chain: the score implementations,
`reproduce.py` and the figures need the `[full]` extra and the datasets, which
is what the rest of this section is about.

From a checkout:

```bash
pip install -e ".[full]"            # or: pip install -r requirements.txt
python3 tests/run_tests.py          # 25 pass, 1 known defect (F-2), 0 fail
python3 reproduce.py --self-test    # the whole reproduction chain, synthetic data
```

Verified on 2026-08-13 in a fresh `python3 -m venv` on macOS 15 (Darwin
25.2.0), Python 3.14.3, resolving from PyPI: `numpy 2.5.2`, `scipy 1.18.0`,
`scikit-learn 1.9.0`, `pandas 3.0.5`, `matplotlib 3.11.1`, `anndata 0.13.2`.
Both commands above pass in that environment, and `import own_baseline` works
from outside the repository directory.

The core diagnostic needs only numpy, scipy and scikit-learn: with just those
three installed, 22 of the 25 tests pass and the 3 that do not are the ones
constructing an `AnnData`. `pandas`, `anndata` and `matplotlib` come with the
`[full]` extra and are needed by the score implementations, `reproduce.py` and
the figures. `requirements.txt` pins the exact versions the audit ran on.

The test suite needs no test runner: `python3 tests/run_tests.py` uses only the
runtime dependencies. `pytest tests/` collects the same functions if you prefer.

Two environment variables control where anything is read or written; both
default inside the repository and both defaults are git-ignored:

| variable | default | what lives there |
|---|---|---|
| `OWNBASELINE_DATA_ROOT` | `<repo>/data/runs` | downloads, prepared matrices, results |
| `OWNBASELINE_SCRATCH` | `<repo>/data/scratch` | the R ↔ Python handoff |

The R score wrappers read the same two variables, so setting them once lines
both halves up.

## Reproduction

```bash
python3 reproduce.py --ledger        # what can and cannot be regenerated
python3 reproduce.py --self-test     # whole chain on synthetic data, ~1 min
python3 reproduce.py                 # download ~135 MB and run
```

**Reproducible end to end**, from public accessions with `pip` and no R: the
three primitives (gene count, `PCC(x, degree)` on STRING v12, Shannon entropy),
**CytoTRACE v1** (the full MVG → GCS → NNLS → Markov-diffusion port in
`scores/cytotrace_full.py`), and **SCENT's SR and CCAT** (the Python
implementations in `own_baseline/potency_metrics.py`).

This was run, not asserted. On 2026-08-13 `reproduce.py` downloaded
GSE106474 and STRING v12, rebuilt the scaffold, prepared **39,505 cells across
12 stages**, the paper's n exactly, and regenerated four rows of the headline
table. They match the values behind the manuscript to every digit reported,
including both bootstrap bounds under both kernels:

| row | τ_b (reproduced) | 95% CI | weighted τ | 95% CI | verdict |
|---|---|---|---|---|---|
| CytoTRACE v1 \| gene count | +0.6531 | [+0.6479, +0.6579] | +0.7901 | [+0.7672, +0.8067] | ADDS-BEYOND |
| SR \| `PCC(x, degree)` | +0.4300 | [+0.4248, +0.4350] | +0.7366 | [+0.7174, +0.7611] | ADDS-BEYOND |
| SR \| CCAT | +0.4678 | [+0.4626, +0.4727] | +0.7522 | [+0.7347, +0.7766] | ADDS-BEYOND |
| CCAT \| `PCC(x, degree)` | −0.2209 | [−0.2267, −0.2151] | +0.1716 | [+0.1186, +0.2104] | INCONCLUSIVE-kernel-disagree |

That is the audit's largest effect (CytoTRACE, τ_b 0.653), the signalling-entropy
row the refutation turns on (SR, 0.430), and the CCAT row the reduction thesis
survives on. Read the last row with the scope note above and defect **F-2**
below in mind: the two kernels disagree at a near-null, and that disagreement is
also the signature of the F-2 artefact.

**Not reproducible here, and why.** Nothing below is silently skipped;
`reproduce.py` names the blocker for each.

| row | blocker |
|---|---|
| ORIGINS (τ_b 0.402) | needs `diff_edges.tsv`, `ORIGINS::differentiation_edges` dumped to TSV by hand. No script in this repository writes it. |
| SLICE (ρ 0.932) | needs the SLICE R package and its bundled `hs_kappasim.rda`; nothing here fetches it |
| dpath (τ_b 0.120) | GitHub-only R package with a documented R ≥ 4.2 incompatibility |
| NCG (τ_b 0.402) | needs the NCG repo cloned **with Git-LFS**; a plain clone yields pointer files |
| MCE | no public implementation exists |
| SPIDE | the formula is paywalled and was never obtained |
| scEnergy | requires MATLAB |
| StemID, cmEntropy | never measured. They equal a transcriptome-entropy primitive by construction, and the manuscript declines to report a number for them. |

The R cross-check of the Python SR/CCAT implementations is a separate matter
from computing them. The published check, Spearman 1.000, max |Δ| < 1e-13 on a
500-cell subsample, needs the SCENT R sources placed under
`$OWNBASELINE_SCRATCH/scent_src`, which no script here does.
`reproduce.py` computes the scores and writes `scent_validation.json` recording
that the cross-check was **skipped**.

`reproduce.py` writes `reproduction/table.md`, `reproduction/table.json` and a
regenerated Figure 2 carrying the bootstrap intervals, computed from the raw
data. `figures/make_paper_a_figures.py` is a separate path to the same numbers:
it reads the frozen run artefacts in `data/run_record/` and does not recompute
anything. The two agree where both can run; `make_paper_a_figures.py
--crosscheck` prints the comparison.

## What the audit found

On a 12-stage visually staged zebrafish ordinal (GSE106474, 39,505
whole-embryo cells), of the seven scores that could be placed on the ordinal,
five carry ordering skill beyond their own primitive, Kendall τ_b after rank
residualization, with the value after residualizing on all four low-order
statistics at once in brackets:

| score | primitive | τ_b | joint-4 |
|---|---|---|---|
| CytoTRACE v1 | gene count | 0.653 | 0.288 |
| SCENT SR | `PCC(x, degree)` | 0.430 | 0.183 |
| ORIGINS | `PCC(x, degree)` | 0.402 | 0.156 |
| NCG † | its own GO connectome | 0.402 | - |
| dpath † | Shannon entropy | 0.120 | 0.134 |
| CCAT | `PCC(x, degree)` | no measurable residual (ρ = 0.998) | - |
| SLICE † | Shannon entropy | near zero (ρ = 0.932) | - |

† on a fixed 3,000-cell subsample, not the full 39,505.

**The strong hypothesis is refuted.** This work set out to show that every
unsupervised potency score is a reparametrization of a fixed low-order
statistic, so that agreement between such scores across contexts is guaranteed
by construction rather than by biology. That holds for the identity and entropy
constructions, CCAT is its primitive by definition; StemID and cmEntropy equal
a transcriptome entropy by construction; SLICE tracks one it does not improve
on, and it is false for the entropy-rate and diffusion constructions.
Signalling entropy sits at ρ = 0.93 with the degree correlation and keeps skill
after four primitives are removed at once. What the surviving residual consists
of is open: a staged timecourse traces a developmental manifold, and a score
tracking that manifold keeps skill whether or not it measures potency in any
deeper sense.

## Scope, what this does not do

- **It does not tell you a score is good.** `ADDS-BEYOND` means the score
  orders cells beyond the primitive you named. It says nothing about whether
  what it adds is potency rather than, say, position on a developmental
  manifold.
- **It controls a realization of a primitive, not the construct.** The degree
  primitive here is STRING v12 at confidence 700, largest connected component.
  Threshold and interactome were not varied, and part of any residual may be
  scaffold error.
- **It only nulls monotone reparametrizations.** A score that is an exact
  non-monotone deterministic function of its primitive carries no information
  and will still register skill.
- **A score reducing to a fifth statistic passes.** The primitive set is four,
  each declared by the scores' own authors.
- **It needs a score-independent ordinal.** A pseudotime derived from the same
  expression matrix is not one.
- **It is calibrated on development and the scores are deployed on patients.**
  That gap is deliberate and it is still a gap.

## Known defects

Recorded rather than patched, because fixing them changes what the estimator
computes and the manuscript's numbers were traced to specific runs of it. Full
detail in the commit messages and in `tests/test_estimator.py`.

- **F-2: an exact monotone function of its primitive does not yield zero
  conditional skill, and both kernels are affected.** When
  `rank(score) == rank(primitive)` the least-squares fit is exact and the
  residual is the rounding error of the subtraction, on the order of 1e-12
  rather than an exact zero. That debris is monotone in the primitive rank, so
  what a kernel returns depends on whether the primitive predicts the ordinal.
  Where it does not, `weightedtau` returns up to +0.51 and Kendall τ_b stays
  near 0.02, which is the fixture the defect was first recorded on and the
  narrower case. Where it does, which is Null B and the case the audit faces,
  the debris inherits that association: across four seeds and two coupling
  strengths both kernels reach 0.87 in magnitude, with a sign that changes with
  the seed. How large it is depends on the machine, because the debris is the
  rounding error of a LAPACK least-squares fit: aarch64 with numpy 2.2 puts all
  sixteen measured cells between 0.71 and 0.87, x86_64 with numpy 2.4 puts
  fourteen there and collapses two toward zero. The sham score in the Quick
  start above is this: its τ_b of −0.6385 is the rounding pattern, not a
  measurement. What separates a residual from debris is its magnitude, not its
  τ: `own_baseline.cli.residual_scale` returns `max|resid| / (eps * n)`, real
  residuals sit at 1e14 to 1e15 on that ratio, and `ownbaseline check` refuses
  to report a value below 1e6. Both residual implementations in this repository
  have the defect. Pinned by
  `tests/test_estimator.py::test_exact_monotone_function_of_primitive_has_zero_conditional_skill`
  (expected failure) and by four tests in `tests/test_cli.py`.
- **`figures/make_paper_a_figures.py` contained no data, fixed 2026-08-29.**
  Every number in it was a hard-coded literal; it had no `errorbar` call despite
  the caption promising bootstrap intervals; it plotted StemID and cmEntropy at
  ρ 0.990 / 1.000 and skill 0.000 as though measured, and NCG as pending
  although the fix-1 run had measured it at τ_b +0.402. The script now reads
  every value from `data/run_record/` and the null-calibration grid, draws each
  score against its own measured null floor, and emits its caption and a
  `fig2_values.json` from what it actually drew. The old script is kept at
  `figures/deprecated/make_paper_a_figures.py` because its literals are what the
  pre-2026-08 manuscript was traced against. The full before/after is
  `04-experiments/2026-08-29-figure2-provenance.md` in the vault.
- **The figures were drawn to no page in particular, retargeted 2026-08-30.**
  Figure 2 was emitted at 10.2 in wide, 1.6× the ML4H text block and 3.2× a
  column, with a side table set in 6.6pt type. Nothing about it survived
  reduction to a 3.18 in column. Figure 2 is now two variants built from the
  same rows: `fig2_body.pdf` at 3.18 in (`\begin{figure}`) carrying only the
  seven scores measured on the ordinal, and `fig2_appendix.pdf` at 6.50 in
  (`\begin{figure*}`) carrying all fourteen rows, the n / floor / verdict table
  and the by-construction band. Figure 3 is 6.50 in for a reason the script
  measures and prints on every run: side by side in one column each panel gets
  1.03 in of plotting width, against 2.06 in for its own title and 1.38 in for
  its two x tick labels at 7pt. Nothing on a retargeted figure is set below 7pt
  at final size, `FontLedger.enforce()` raises rather than shrinking type,
  and each run reads the rendered width back out of the written PDF's MediaBox
  instead of trusting the `figsize` it asked for.
- **Two residual implementations, not merged.** `rank_resid_multi` (the
  track-2 estimator) fits by least squares with an intercept and takes any
  number of covariates. `rank_residual` / `_rank_residual` in `experiments/w4`,
  `experiments/track4`, `experiments/cytotrace_v1` and `experiments/stemsc`
  uses `np.cov(...)/np.var(...)`, which mixes `ddof=1` with `ddof=0` and so
  inflates the slope by `n/(n−1)`, fits no intercept, and takes exactly one
  covariate. Measured difference: |Δτ| up to 3.7e-2 at n = 106, 6.6e-07 at
  n = 39,505. Which is correct is a numerical decision for the authors.

## Layout

```
own_baseline/          the diagnostic
  own_baseline.py      marginal gap + the Test-A error-model flavour
  conditional_skill.py the track-2 conditional-skill estimator + the report API
  potency_metrics.py   the primitives, and SR / CCAT / CytoTRACE reimplementations
  atlas_run/wdm_tau.py Kang wdm weighted-Kendall kernel (== R wdm::wdm)
  paths.py / paths.R   where this repository reads and writes
  synthetic.py         synthetic generators for the self-tests
scores/                score implementations used in the audit
  cytotrace_full.py    R-faithful CytoTRACE v1 port
  run_*.R              wrappers calling the real R packages
experiments/           every run script, verbatim, organized as it was run
figures/               the manuscript's figures, each value read from a run artefact
  make_paper_a_figures.py  reads data/run_record/ + the null-calibration grid
  fig2_body.pdf        3.18 in, figure  - the 7 rows measured on the ordinal
  fig2_appendix.pdf    6.50 in, figure* - all 14 rows, side table, band
  fig3.pdf             6.50 in, figure* - the depth confound, two panels
  fig2_*_caption.txt   the caption generated for each variant
  deprecated/          the pre-2026-08-29 literal-only script, kept for tracing
tests/                 known-answer tests
reproduce.py           the reproduction entry point
data/README.md         accessions and how to fetch them; no data is stored here
data/run_record/       the run JSONs the figures read, committed verbatim
```

`experiments/` is a run record, not a library. Those scripts are kept as they
ran, apart from the path changes needed to make them import off the author's
machine.

## Scores (R packages)

The audit ran published implementations wherever one existed: SCENT
(`aet21/SCENT`, SR + CCAT), SLICE (`xu-lab/SLICE`), dpath (`gongx030/dpath`),
ORIGINS (`danielasenraoka/ORIGINS`), NCG (`Xinzhe-Ni/NCG`). CytoTRACE v1 is a
port of the Gulati 2020 R algorithm. The SR, CCAT and ORIGINS **Python**
implementations were checked against the R packages on a 500-cell subsample
before use at scale, SR and CCAT to machine precision (Spearman 1.000,
max |Δ| < 1e-13), ORIGINS to rank agreement only (Spearman 1.000).

## Citation

The manuscript is under review; this section will carry the DOI and BibTeX
entry once one is minted. Until then, cite the repository by URL and commit.

## License

MIT, see [LICENSE](LICENSE).
