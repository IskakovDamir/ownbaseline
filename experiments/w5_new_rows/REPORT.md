# W5 new rows: report

**P1 is refuted under the registered rule**: MCE's tau_b conditional skill on PCC(x, degree) is -0.4898 against a floor of +0.0325 at n = 39,505, NOT ABOVE FLOOR, where the entropy-rate branch predicted ABOVE FLOOR; the refutation comes from the rule's alignment step, not from a missing residual: MCE's marginal Spearman with the stage ordinal is +0.035, so it was kept as it is, and its residual beyond PCC(x, degree) orders cells in the direction its authors state at +0.4898 [+0.4850, +0.4948], which a direction-aware reading would count as skill.

Pre-registration: `PREREG.md`, first added in commit 71605a4 2026-10-09 15:28:14 +0500 and pushed before any new score met a label; later commits to it: 9eded68 2026-10-09 16:30:58 +0500 audit: amend PREREG for MCE now that MCE.m is in hand, before the row runs. Seed 42; bootstrap 1000 resamples for tau_b and 300 for weighted tau; plan.json SHA256 943e68952b02af815ec789847cccd1509b0490dd1b673fa8a3f2b294456dfda0.

## Results

Conditional skill is Kendall tau_b of the aligned score's rank residual on the declared primitive (the joint four-primitive residual for rows that declare none), read against the 97.5th percentile of the estimator's measured null at the value's n, rho, number of covariates and number of ordinal levels. Grid floors come from the nearest grid cell (the Floors table names it); design floors were computed at the value's own rho. For a joint value the rho is the declared primitive's |rho| when the row has one and otherwise the largest single-primitive |rho|, named in the column. Brackets after a floor give the 12-level floor where the ordinal has another level count. Marginal tau_b is the raw score against the raw ordinal (GSE106474 and GSE113074 run early to late stage, GSE117498 and GSE125970 low to high potency), so its sign mixes the two orientations; the direction check is potency-oriented. Weighted tau (rank=True) with its 95% interval is compared with the same cell's rank=True column and is reported, not a verdict. Numbered notes are caveats, under the table.

| score | class | dataset | n | rho with primitive | marginal tau_b | conditional tau_b [95% CI] | floor | verdict under tau_b | weighted tau [95% CI] / floor | joint four-primitive residual tau_b [95% CI] / floor | direction check |
|---|---|---|---|---|---|---|---|---|---|---|---|
| MCE | unsupervised (in tally) | GSE106474 (12 levels) | 39,505 | +0.802 with PCC(x, degree) | +0.0215 | -0.4898 [-0.4948, -0.4850] | +0.0325 | NOT ABOVE FLOOR | -0.3480 [-0.3611, -0.3353] / +0.2959 (not above) | -0.1311 [-0.1363, -0.1260] / +0.0233 (NOT ABOVE FLOOR) | score runs AGAINST its authors' stated direction (Spearman vs potency -0.035); primitives below chance: PCC(x, degree) |
| StemFinder [1] | unsupervised (in tally) | GSE106474 (12 levels) | 39,505 | -0.568 with cell-cycle gene-set score | +0.2924 | +0.0977 [+0.0912, +0.1043] | +0.0247 | ABOVE FLOOR | +0.4389 [+0.3920, +0.4810] / +0.2238 (above) | +0.1669 [+0.1625, +0.1715] / +0.0171 (ABOVE FLOOR) | score runs as its authors state (Spearman vs potency -0.445); primitives below chance: PCC(x, degree) |
| mRNAsi [2] | trained (boundary, outside tally) | GSE117498 (4 levels) | 12,354 | +0.634 (largest single: Shannon entropy) | -0.0148 | +0.0800 [+0.0671, +0.0915] | +0.0263 (12 levels: +0.0245) | ABOVE FLOOR | +0.4115 [+0.3883, +0.4302] / +0.2820 (above) | +0.0800 [+0.0671, +0.0915] / +0.0263 (ABOVE FLOOR) | score runs AGAINST its authors' stated direction (Spearman vs potency -0.021); primitives below chance: gene count, PCC(x, degree), log10 library size |
| mRNAsi [3] | trained (boundary, outside tally), sensitivity line | GSE117498h (4 levels) | 12,354 | +0.648 (largest single: Shannon entropy) | +0.0095 | -0.0578 [-0.0693, -0.0446] | +0.0256 (12 levels: +0.0254) | NOT ABOVE FLOOR | +0.1992 [+0.1240, +0.2641] / +0.2889 (not above) | -0.0578 [-0.0693, -0.0446] / +0.0256 (NOT ABOVE FLOOR) | score runs as its authors state (Spearman vs potency +0.010); primitives below chance: gene count, PCC(x, degree), log10 library size |
| FitDevo [4, 5] | trained (boundary, outside tally) | GSE106474 (12 levels) | 39,505 | +0.942 (largest single: gene count) | -0.2731 | +0.0631 [+0.0580, +0.0679] | +0.0227 | ABOVE FLOOR | +0.3467 [+0.2590, +0.3858] / +0.3234 (above) | +0.0631 [+0.0580, +0.0679] / +0.0227 (ABOVE FLOOR) | score runs as its authors state (Spearman vs potency +0.377); primitives below chance: PCC(x, degree) |
| FitDevo [2] | trained (boundary, outside tally) | GSE117498 (4 levels) | 12,354 | +0.782 (largest single: Shannon entropy) | -0.0650 | +0.1494 [+0.1359, +0.1617] | +0.0301 (12 levels: +0.0266) | ABOVE FLOOR | +0.4704 [+0.4518, +0.4852] / +0.2981 (above) | +0.1494 [+0.1359, +0.1617] / +0.0301 (ABOVE FLOOR) | score runs AGAINST its authors' stated direction (Spearman vs potency -0.087); primitives below chance: gene count, PCC(x, degree), log10 library size |
| FitDevo [3] | trained (boundary, outside tally), sensitivity line | GSE117498h (4 levels) | 12,354 | +0.784 (largest single: gene count) | -0.0658 | +0.1425 [+0.1290, +0.1550] | +0.0303 (12 levels: +0.0274) | ABOVE FLOOR | +0.4697 [+0.4499, +0.4841] / +0.2911 (above) | +0.1425 [+0.1290, +0.1550] / +0.0303 (ABOVE FLOOR) | score runs AGAINST its authors' stated direction (Spearman vs potency -0.088); primitives below chance: gene count, PCC(x, degree), log10 library size |
| FitDevo | trained (boundary, outside tally) | GSE125970 (3 levels) | 14,537 | +0.944 (largest single: PCC(x, degree)) | +0.5621 | +0.1527 [+0.1425, +0.1630] | +0.0246 (12 levels: +0.0235) | ABOVE FLOOR | +0.3572 [+0.3108, +0.3862] / +0.3296 (above) | +0.1527 [+0.1425, +0.1630] / +0.0246 (ABOVE FLOOR) | score runs as its authors state (Spearman vs potency +0.708); primitives below chance: none |
| FitDevo [6] | trained (boundary, outside tally) | GSE113074 (10 levels) | 127,607 | +0.814 (largest single: Shannon entropy) | +0.0811 | -0.0681 [-0.0711, -0.0651] | +0.0215 (12 levels: +0.0216) | NOT ABOVE FLOOR | +0.2274 [+0.1637, +0.2593] / +0.2902 (not above) | -0.0681 [-0.0711, -0.0651] / +0.0215 (NOT ABOVE FLOOR) | score runs AGAINST its authors' stated direction (Spearman vs potency -0.116); primitives below chance: gene count, PCC(x, degree), log10 library size |

1. nothing is fitted, but the authors chose among method variants (Gini, SD, variance; the k range) on a benchmark that includes the Farrell zebrafish set GSE106587, the SuperSeries of GSE106474
2. GSE117498 through the stemsc code path: the four broad-gate files name genes in R make.names form (HLA.A) and the seven sorted files in HGNC form (HLA-A); load_c1 unions by exact name, so in broad-gate cells (tier 4 among them) those genes read zero. The GSE117498h line is the registered sensitivity with the names merged.
3. registered sensitivity: gene names merged (prepare_inputs.ds_gse117498h)
4. not in the training set, but the authors evaluated their weights on this study's stage labels once (Spearman 0.228, GSE106587)
5. zebrafish symbols matched to human by upper case; the authors left zebrafish out of training for too few homologous genes
6. Xenopus symbols matched to human by upper case; the homology concern the authors raised for zebrafish applies here too

StemFinder's joint value sits at its declared primitive's rho; its weighted joint value and every other number not in the table are in the per-row JSON (`$OWNBASELINE_DATA_ROOT/w5_new_rows/results/<row>_<dataset>.json`).

### Reading the table

The verdicts are the registered rule's. The notes are derived from the same registered numbers and change none of them. The estimator is sign-agnostic: it points each score the way its marginal Spearman correlation with the ordinal runs, then residualizes. Because the rank residual and tau_b are both odd in the score's sign, the raw score's residual is the aligned value, negated when the score was negated. Read against the authors' stated polarity:

- MCE on GSE106474: the raw score's residual beyond its declared primitive orders cells in the direction its authors state (tau_b +0.4898 [+0.4850, +0.4948] with the authors' polarity counted positive); the marginal Spearman with the ordinal is +0.035, so the score was kept as it is before residualizing, and the registered verdict is NOT ABOVE FLOOR.
- StemFinder on GSE106474: the raw score's residual beyond its declared primitive orders cells in the direction its authors state (tau_b +0.0977 [+0.0912, +0.1043] with the authors' polarity counted positive); the marginal Spearman with the ordinal is +0.445, so the score was kept as it is before residualizing, and the registered verdict is ABOVE FLOOR.
- mRNAsi on GSE117498: the raw score's residual beyond the four primitives orders cells against the direction its authors state (tau_b -0.0800 [-0.0915, -0.0671] with the authors' polarity counted positive); the marginal Spearman with the ordinal is -0.021, so the score was negated before residualizing, and the registered verdict is ABOVE FLOOR.
- mRNAsi on GSE117498h: the raw score's residual beyond the four primitives orders cells against the direction its authors state (tau_b -0.0578 [-0.0693, -0.0446] with the authors' polarity counted positive); the marginal Spearman with the ordinal is +0.010, so the score was kept as it is before residualizing, and the registered verdict is NOT ABOVE FLOOR.
- FitDevo on GSE106474: the raw score's residual beyond the four primitives orders cells in the direction its authors state (tau_b +0.0631 [+0.0580, +0.0679] with the authors' polarity counted positive); the marginal Spearman with the ordinal is -0.377, so the score was negated before residualizing, and the registered verdict is ABOVE FLOOR.
- FitDevo on GSE117498: the raw score's residual beyond the four primitives orders cells against the direction its authors state (tau_b -0.1494 [-0.1617, -0.1359] with the authors' polarity counted positive); the marginal Spearman with the ordinal is -0.087, so the score was negated before residualizing, and the registered verdict is ABOVE FLOOR.
- FitDevo on GSE117498h: the raw score's residual beyond the four primitives orders cells against the direction its authors state (tau_b -0.1425 [-0.1550, -0.1290] with the authors' polarity counted positive); the marginal Spearman with the ordinal is -0.088, so the score was negated before residualizing, and the registered verdict is ABOVE FLOOR.
- FitDevo on GSE125970: the raw score's residual beyond the four primitives orders cells in the direction its authors state (tau_b +0.1527 [+0.1425, +0.1630] with the authors' polarity counted positive); the marginal Spearman with the ordinal is +0.708, so the score was kept as it is before residualizing, and the registered verdict is ABOVE FLOOR.
- FitDevo on GSE113074: the raw score's residual beyond the four primitives orders cells in the direction its authors state (tau_b +0.0681 [+0.0651, +0.0711] with the authors' polarity counted positive); the marginal Spearman with the ordinal is +0.116, so the score was kept as it is before residualizing, and the registered verdict is NOT ABOVE FLOOR.

- mRNAsi: the verdict differs between GSE117498 (ABOVE FLOOR) and the name-harmonized GSE117498h (NOT ABOVE FLOOR), but what the score carries beyond the primitives does not change direction: relative to the authors' polarity the residual is -0.0800 and -0.0578. The split moves the marginal Spearman from -0.021 to +0.010, which flips the alignment, and the registered rule reads only an aligned residual above its floor. The decisive verdict therefore rests on the sign of a marginal correlation close to zero.
- FitDevo: the GSE117498 line and its name-harmonized sensitivity GSE117498h agree (ABOVE FLOOR, conditional tau_b +0.1494 and +0.1425); the gene-name split does not move the verdict.

Audit tally, unsupervised rows only: MCE NOT ABOVE FLOOR; StemFinder ABOVE FLOOR. mRNAsi and FitDevo are trained and stay outside it.

### Floors

| value | n | k | levels | rho | floor source |
|---|---|---|---|---|---|
| MCE_GSE106474 declared | 39,505 | 1 | 12 | +0.8024 | shipped grid: Null B, align on, 12 levels, n=39,505, rho=0.9, kendalltau, 1 covariate, 3 coupling strengths, 600 seeds |
| MCE_GSE106474 joint4 | 39,505 | 4 | 12 | +0.8024 | shipped grid: Null B, align on, 12 levels, n=39,505, rho=0.9, kendalltau, 4 covariates, 1 coupling strength, 200 seeds |
| StemFinder_GSE106474 declared | 39,505 | 1 | 12 | -0.5684 | shipped grid: Null B, align on, 12 levels, n=39,505, rho=0.5, kendalltau, 1 covariate, 3 coupling strengths, 600 seeds |
| StemFinder_GSE106474 joint4 | 39,505 | 4 | 12 | +0.5684 | shipped grid: Null B, align on, 12 levels, n=39,505, rho=0.5, kendalltau, 4 covariates, 1 coupling strength, 200 seeds |
| mRNAsi_GSE117498 joint4 | 12,354 | 4 | 4 | +0.6335 | design point `n=12354,rho=0.634,levels=4,k=4` (run_missing_cells.py, 200 seeds) |
| mRNAsi_GSE117498 joint4 (12-level sensitivity) | 12,354 | 4 | 12 | +0.6335 | design point `n=12354,rho=0.634,levels=12,k=4` (run_missing_cells.py, 200 seeds) |
| mRNAsi_GSE117498h joint4 | 12,354 | 4 | 4 | +0.6485 | design point `n=12354,rho=0.648,levels=4,k=4` (run_missing_cells.py, 200 seeds) |
| mRNAsi_GSE117498h joint4 (12-level sensitivity) | 12,354 | 4 | 12 | +0.6485 | design point `n=12354,rho=0.648,levels=12,k=4` (run_missing_cells.py, 200 seeds) |
| FitDevo_GSE106474 joint4 | 39,505 | 4 | 12 | +0.9420 | shipped grid: Null B, align on, 12 levels, n=39,505, rho=0.93, kendalltau, 4 covariates, 1 coupling strength, 200 seeds |
| FitDevo_GSE117498 joint4 | 12,354 | 4 | 4 | +0.7822 | design point `n=12354,rho=0.782,levels=4,k=4` (run_missing_cells.py, 200 seeds) |
| FitDevo_GSE117498 joint4 (12-level sensitivity) | 12,354 | 4 | 12 | +0.7822 | design point `n=12354,rho=0.782,levels=12,k=4` (run_missing_cells.py, 200 seeds) |
| FitDevo_GSE117498h joint4 | 12,354 | 4 | 4 | +0.7843 | design point `n=12354,rho=0.784,levels=4,k=4` (run_missing_cells.py, 200 seeds) |
| FitDevo_GSE117498h joint4 (12-level sensitivity) | 12,354 | 4 | 12 | +0.7843 | design point `n=12354,rho=0.784,levels=12,k=4` (run_missing_cells.py, 200 seeds) |
| FitDevo_GSE125970 joint4 | 14,537 | 4 | 3 | +0.9440 | design point `n=14537,rho=0.944,levels=3,k=4` (run_missing_cells.py, 200 seeds) |
| FitDevo_GSE125970 joint4 (12-level sensitivity) | 14,537 | 4 | 12 | +0.9440 | design point `n=14537,rho=0.944,levels=12,k=4` (run_missing_cells.py, 200 seeds) |
| FitDevo_GSE113074 joint4 | 127,607 | 4 | 10 | +0.8143 | design point `n=127607,rho=0.814,levels=10,k=4` (run_missing_cells.py, 200 seeds) |
| FitDevo_GSE113074 joint4 (12-level sensitivity) | 127,607 | 4 | 12 | +0.8143 | design point `n=127607,rho=0.814,levels=12,k=4` (run_missing_cells.py, 200 seeds) |

`design_points.csv` holds 12 design points (36 rows), all computed before the PREREG commit in two passes of `run_rows.py nulls`: the first plan's points, then the points the reviewed plan added (the GSE117498h lines and the 12-level sensitivities). Weighted-tau floors as `ownbaseline check --kernel weighted` would read them (the rank=False column) are in each per-row JSON under `floors.weighted_rankFalse_cli`.

## Blocked rows

- none

## MCE port validation

`scores/mce.py` is a port of the authors' MCE.m (default rule "mce_m": MCE.m's start, iteration, 1e-2 stopping rule and returned pair, each cell stopping on its own). Its fixed point, solved to convergence, against analytic answers (`discovery/mce_validation.json`):

| case | expected | got | abs. deviation |
|---|---|---|---|
| self-loops only (P = I): MCE = H(pi) | 1.7471706669 | 1.7471706669 | 0.0e+00 |
| complete graph: MCE = 2 H(pi) | 2.87539980601 | 2.87539980601 | 0.0e+00 |
| pi proportional to self-loop degree: normalised MCE = 1 | 1 | 1 | 2.2e-16 |
| random 6-node graph vs SLSQP on Eq 3 | 2.67829421109 | 2.67829421109 | 0.0e+00 |

Against MCE.m (SHA256 `c8ad11f2cdadc979cc303a21785f65eba77a41dfe565ace2beffaeae265fdce2`) run unmodified in GNU Octave (aarch64-apple-darwin25.4.0) version 11.3.0, on 500 cells of GSE106474 (the SCENT validation subsample, seed 42), SR's input and network, as was done for SCENT:

| comparison | value |
|---|---|
| Spearman, port against MCE.m | 1.000000 |
| Pearson, port against MCE.m | 1.000000 |
| maximum absolute deviation | 1.8e-13 |
| identical step counts, all cells | yes |
| seconds inside Octave | 158.7 |

MCE.m's 1e-2 rule leaves the score within 8.3e-08 of the fixed point solved to a residual of 1e-6 (Spearman 1.000000).

Timing under MCE.m's rule, one core: 500 cells in 26.2 s, 137 to 547 steps per cell (median 263), projected 0.58 h for 39,505 cells, under 8 hours, so the MCE row uses every cell (PREREG.md, Amendment 1). The full run took 1,944 s on 8 processes, 109 to 659 steps per cell, 0 NaN.

Before MCE.m arrived, the equations-based solver was timed by stopping tolerance (1e-06: 480 iterations, 0.89 h; 1e-08: 16,950 iterations, 31.42 h); that projection is superseded by the one above.

## Commands

From the repository root, in this order, with `$OWNBASELINE_DATA_ROOT` and `$OWNBASELINE_SCRATCH` unset or at their defaults (`data/runs`, `data/scratch`): the R library path below is written for the default scratch root.

Inputs read from the data root and not made here: `track2/data/prepared/` (GSE106474 after the repository's QC, from `python3 reproduce.py --stages fetch-string scaffold fetch-zebrafish prep`), `w4/scaffolds/human_string_v12_thr700_lcc.npz` (stage `scaffold`), `<scratch>/scent_io/adjMC.npz` (`experiments/track2/code/prep_scent_inputs.py`, SR's network, used by the MCE timing), `w4/data/c1_gse117498/` and `w4/data/c2_gse125970/` (GEO GSE117498 and GSE125970), and `fix3/data/prepared/` (`experiments/fix3/code/prep_briggs.py` on GSE113074). `data/README.md` has the accessions.

```
python3 tests/run_tests.py
python3 tests/test_mce.py
python3 experiments/w5_new_rows/fetch_sources.py
python3 experiments/w5_new_rows/discovery_facts.py
python3 experiments/w5_new_rows/validate_mce.py      # needs reference/MCE.m and octave-cli
python3 experiments/w5_new_rows/prepare_inputs.py
printf 'CC=clang -std=gnu17\nCC17=clang -std=gnu17\nCC23=clang -std=gnu17\n' > data/scratch/Makevars.w5
R_MAKEVARS_USER=$PWD/data/scratch/Makevars.w5 Rscript --vanilla -e 'lib <- "data/scratch/Rlib"; .libPaths(c(lib, .libPaths())); install.packages(c("Seurat", "qlcMatrix"), lib = lib, repos = "https://cloud.r-project.org", Ncpus = 8)'
python3 experiments/w5_new_rows/score_mce.py --procs 8
Rscript --vanilla experiments/w5_new_rows/score_stemfinder.R GSE106474
Rscript --vanilla experiments/w5_new_rows/score_mrnasi.R GSE117498
Rscript --vanilla experiments/w5_new_rows/score_mrnasi.R GSE117498h
Rscript --vanilla experiments/w5_new_rows/score_fitdevo.R GSE106474
Rscript --vanilla experiments/w5_new_rows/score_fitdevo.R GSE117498
Rscript --vanilla experiments/w5_new_rows/score_fitdevo.R GSE117498h
Rscript --vanilla experiments/w5_new_rows/score_fitdevo.R GSE125970
Rscript --vanilla experiments/w5_new_rows/score_fitdevo.R GSE113074
python3 experiments/w5_new_rows/run_rows.py plan
python3 experiments/w5_new_rows/run_rows.py nulls
python3 experiments/w5_new_rows/check_design_extension.py
python3 experiments/w5_new_rows/run_rows.py values
python3 experiments/w5_new_rows/make_report.py
```

Per row: MCE needs `validate_mce.py` (with MCE.m in `reference/`) and `score_mce.py`; StemFinder needs `prepare_inputs.py`, `score_stemfinder.R GSE106474`, then `run_rows.py plan`, `nulls`, `values`; mRNAsi needs `score_mrnasi.R` on GSE117498 and GSE117498h; FitDevo needs `score_fitdevo.R` on the five inputs; every row is read by the same `run_rows.py values`. Homebrew's R 4.6.1 asks for `-std=gnu23`, which the Command Line Tools clang 15 rejects; the Makevars line works around it for this install only.

## Software and provenance

| item | value |
|---|---|
| Python, numpy, scipy, scikit-learn | 3.14.3, 2.4.3, 1.18.0, 1.9.0 |
| R | R version 4.6.1 (2026-06-24) |
| Seurat, SeuratObject, qlcMatrix | 5.6.0, 5.4.0, 0.9.9 |
| GNU Octave | GNU Octave (aarch64-apple-darwin25.4.0) version 11.3.0 |
| SHA256 of MCE.m | c8ad11f2cdadc979cc303a21785f65eba77a41dfe565ace2beffaeae265fdce2 |
| FitDevo commit | 0c757e609489867e83f32f6571ebcfce704c3124, 2024-04-07 11:06:59 +0800 |
| stemFinder commit | db8ef0e1b4ad02d03165c734954bc9beb28e5900 (version 0.1.0), 2024-09-21 09:12:17 -0400 |
| TCGAbiolinks | 2f9d2496241af1ad3950a23bfce7d434d1834516 (version 2.40.0), 2026-04-28 08:41:41 -0400 |
| SC_PCBC_stemSig.rda SHA256 | a9900d8001414737a0f8f1a7a89d788eaf341130ad39c20833091659042d0777 (12,956 weights) |
| FitDevo BGW.rds SHA256 | cccb4b62013f3ede24fdc6dfdb62053a7fb54a12566b4c89acd7371a8d79de43 (14,717 genes) |

Per-score run records (runtime, genes matched, parameters chosen by rule):

- mce on GSE106474: seconds 1944.3, procs 8, n_cells 39505, rule mce_m (epsilon 0.01, max_iter 1000000), steps_min 109, steps_median 262, steps_max 659, err_max 1.000e-02, n_nan 0, numpy 2.4.3, scipy 1.18.0, python 3.14.3
- stemfinder on GSE106474: seconds 1352.2, n_cells 39505, n_genes_after_gene_steps 20576, n_markers 82, n_markers_unique 81, n_markers_listed 98, n_variable_features_used_for_pca 2487, pcs 7, k 199
- mrnasi on GSE117498: seconds 50.9, n_cells 21412, n_weights 12956, n_weights_matched 12160, n_nan 0
- mrnasi on GSE117498h: seconds 49.5, n_cells 21412, n_weights 12956, n_weights_matched 12161, n_nan 0
- fitdevo on GSE106474: seconds 19.4, n_cells 39505, n_genes_input 23974, n_used_gw 9620, ssgw_0 18, ssgw_1 1827, ssgw_2 7775
- fitdevo on GSE117498: seconds 7.3, n_cells 21412, n_genes_input 19517, n_used_gw 13305, ssgw_0 83, ssgw_1 2840, ssgw_2 10382
- fitdevo on GSE117498h: seconds 7.2, n_cells 21412, n_genes_input 18819, n_used_gw 13309, ssgw_0 83, ssgw_1 2832, ssgw_2 10394
- fitdevo on GSE125970: seconds 6.8, n_cells 14537, n_genes_input 17416, n_used_gw 12006, ssgw_0 111, ssgw_1 2291, ssgw_2 9604
- fitdevo on GSE113074: seconds 27.1, n_cells 127607, n_genes_input 26550, n_used_gw 9682, ssgw_0 70, ssgw_1 2048, ssgw_2 7564

