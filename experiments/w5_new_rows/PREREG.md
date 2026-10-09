# W5 new rows: pre-registration

Written, committed and pushed on 2026-10-09, before any of the four new scores met a stage or tier label. Discovery is in `DISCOVERY.md`. The rows were added after feedback from Qingyang Wang on 9 Oct 2026: the MCE code is in the supplement of Shi et al. 2020, and StemFinder, mRNAsi and FitDevo were missing from the audit.

What had been computed when this was written, none of it comparing a score with an ordinal: the scores of the three methods that run (StemFinder on GSE106474; mRNAsi on GSE117498 and on its name-harmonized version GSE117498h; FitDevo on GSE106474, GSE117498, GSE117498h, GSE125970 and GSE113074), MCE from the published equations on the 500-cell SCENT subsample only, the four primitives of every dataset, the plan below (`run_rows.py plan`: for every value its n, its score-primitive rank correlation and its floor lookup, with the ordinal used only to know which cells are ranked and how many levels it has), the missing null design points it names (`run_rows.py nulls`), and a check of the design-point code against four cells the shipped grid already has (`check_design_extension.py`).

An adversarial review of the draft, before this file was pushed and without any label, led to these changes: the GSE117498 gene-name split is registered below as a sensitivity line; StemFinder's marker vector is now the vignette's (E2F8 counted twice) and its score was recomputed; the joint rows sit at the paper's rho; the levels argument is declared as a departure; and `run_rows.py values` checks more before it reads a value (listed below).

## The decision rule, unchanged from the paper

Applied to every row by `experiments/w5_new_rows/run_rows.py`, which calls the paper's own functions in `own_baseline/conditional_skill.py` and adds no estimator.

1. **Cells.** The ranked cells of the dataset for which the score, the four primitives and the declared primitive are all finite. Every registered cell set is "full": every ranked cell must have a finite score, or `run_rows.py` refuses. n is their count.
2. **Alignment.** `align(score, ordinal)`: the score is negated if its Spearman correlation with the ordinal is negative, so polarity conventions do not matter.
3. **Conditional skill.** `rank_resid_multi(aligned score, [primitive])`, rank OLS with an intercept, then Kendall tau_b (`kang_taub`) of the residual against the ordinal. This is the value that decides.
4. **Interval.** `_unit` with the published resample counts, 1,000 for tau_b and 300 for weighted tau, seed 42, cells resampled with replacement and the residual refitted in every resample, 2.5th and 97.5th percentiles. Seed 42 governs the audit (the bootstrap, and StemFinder's PCA); each method keeps its authors' internal seed (FitDevo SEED 123), because changing it would change the method; the design-point nulls use `run_missing_cells.py`'s hashed seed streams.
5. **Floor.** The 97.5th percentile of the estimator's measured Null B at the value's own n and |rho| (score-primitive Spearman on the same cells), with k covariates: `python3 -m own_baseline.cli floors --n N --rho R --kernel taub --covariates K --levels L` (the repository CLI, `ownbaseline floors`, run from this checkout).
6. **Missing design points.** A design point is missing when the lookup returns no floor, or marks the grid point as `distant` (|n - grid n| > grid n / 2, or |rho - grid rho| > 0.15, as `own_baseline/floors.py` computes it), or as `permissive` in n (the nearest grid n is larger, so its floor is too low). A missing point is computed before the value is read, with `experiments/null_calibration/run_missing_cells.py --design n=N,rho=R,levels=L,k=K` and 200 seeds: for k = 1, `null_grid.run_cell` at the grid's three coupling strengths, floor = the largest of the three 97.5th percentiles (how `own_baseline/floors.py` reads the grid); for k = 4, `null_covariates.run_cell` at its strength 0.55 with L ordinal levels. R is |rho| rounded to three decimals. Re-measuring four cells the grid already has through this path gives tau_b floors within 0.004 of the shipped ones (`discovery/design_extension_check.json`).
7. **Verdict under tau_b.** A value counts only if conditional tau_b is strictly above its floor: ABOVE FLOOR, otherwise NOT ABOVE FLOOR. A value whose residual is rounding debris (`own_baseline.cli.residual_scale` below 1e6) is not read under either kernel, as the CLI refuses it.
8. **Reported with it**, none of which changes the verdict:
   - weighted tau (`scipy_wtau`, scipy's `weightedtau` with its default rank=True, the kernel of every existing row) with its interval, compared with the `weightedtau_rankTrue` column of the same null cell, because that is the kernel computed. `ownbaseline check --kernel weighted` also computes rank=True but reads the `weightedtau_rankFalse` column; that number is recorded beside it in the per-row JSON.
   - the joint four-primitive residual: the score residualized on gene count, PCC(x, degree), Shannon entropy and log10 library size at once, as in `experiments/track4/code/fix2_joint_primitive.py`, both kernels, same bootstrap, floor at k = 4. Its rho is the declared primitive's |rho| when the row declares one, which is where the paper placed its joint rows (CytoTRACE at its gene-count rho, SR at its PCC rho; `experiments/null_calibration/paper_values.py`), and the largest single-primitive |rho| when it declares none, the convention of `ownbaseline check`.
   - the marginal tau_b of the score against the ordinal; the per-row JSON also carries the marginal weighted tau and Spearman.
   - the direction check: the Spearman correlation of each primitive with the ordinal oriented so that higher = more potent, flagged when negative (the primitive orders below chance); and whether the score runs in the direction its authors state.
   - for an ordinal with L != 12 levels, the 12-level floor at the same n, rho and k (see the departure below).

**Departure from the paper's lookup: the number of ordinal levels.** The brief's command, `ownbaseline floors --n N --rho R`, defaults to 12 levels, the zebrafish ordinal every floored row of the paper sits on. GSE117498 has 4 tiers, GSE125970 3 and GSE113074 10, and the paper kept its one row on a sorted atlas (StemSC on GSE117498) off the floor axis altogether. Here the floor is read at the ordinal's own level count L, and the 12-level floor is reported beside it. The shipped level sweep (n = 3,000, k = 1, tau_b) shows the floor moves little with L: 0.0408 at 4 levels against 0.0411 at 12 for rho 0.5, and 0.0244 against 0.0240 at rho 0; the largest gap in the sweep is at rho 0.998 (0.0308 against 0.0264). Both null generators bin the latent ordinal into L levels of equal occupancy, which the real tiers are not (most GSE125970 cells sit in tier 1, most ranked GSE117498 cells in tier 4); that approximation is the one the 12-level grid already makes for the zebrafish stages, and it is stated here as a limitation. Values on GSE117498, GSE125970 and GSE113074 are therefore not on the paper's 12-level axis, and all of them belong to the trained boundary rows.

The four primitives of each dataset follow `experiments/track2/code/prep_data.py`: detected genes; Pearson(log1p(CPM), STRING v12 degree) over the genes in both the dataset and the thr700 scaffold, CPM target = median library size; Shannon entropy of raw counts in natural log; log10 library size (`prepare_inputs.py`).

Ordinals are the ones the repository already uses, unmodified: GSE106474 `rank_kimmel` (12 levels, 1 = high stage, 12 = 6-somite); GSE117498 the W4 lock `C1_POP_TO_RANK` (HSC 5, MPP 4, CD34+/CD164+ 4, MLP 3, CMP 3, GMP 2, MEP 2, PreB/NK 2; three CD34-negative or low populations unranked, so 4 levels); GSE125970 the W4 lock `C2_LABEL_TO_RANK` (stem 4, TA 3, progenitor 3, differentiated 1; 3 levels); GSE113074 `rank_ordinal` (10 Nieuwkoop-Faber stages).

**Integrity checks in `run_rows.py values`.** It refuses to run unless PREREG.md and `run_rows.py` are tracked and identical to `origin/main`, PREREG.md carries no placeholder, every input file still has the SHA256 the plan recorded (`plan.json`, whose SHA256 is printed by `plan` and recorded in the summary), n, levels and every rho recompute to the plan's values, and every missing design point is complete in `design_points.csv` (all coupling strengths, all three kernels, 200 seeds each). The report names the commit that first added PREREG.md and lists any later amendment commits.

## The rows

Every value the rows will report, as `run_rows.py plan` measured it before this file was pushed (`$OWNBASELINE_DATA_ROOT/w5_new_rows/plan.json`, SHA256 `e2d30f3abcde806f2719cb990a67c03d71ec9d6b35f83d07977b04e4db8695f5`; it also records the SHA256 of every score and primitive file). rho is the Spearman correlation of the score with the primitive on the cells used; a joint value takes the declared primitive's |rho| when there is one and the largest single-primitive |rho| otherwise. "grid" means the shipped null grid has the cell; "design" means `run_missing_cells.py --design` computed it (200 seeds) before this file was pushed. Rows marked L = 12 are the reported 12-level sensitivity, not the decision.

| row | dataset | value | n | levels | k | rho | floor source |
|---|---|---|---|---|---|---|---|
| MCE | GSE106474 | - | - | - | - | - | not run: waiting for the author: MCE.m is not in reference/, so no validated port has written w5_new_rows/scores/GSE106474/mce.csv |
| StemFinder | GSE106474 | declared primitive (cell-cycle gene-set score) | 39,505 | 12 | 1 | -0.5684 | grid: Null B, align on, 12 levels, n=39,505, rho=0.5, kendalltau, 1 covariate, 3 coupling strengths, 600 seeds |
| StemFinder | GSE106474 | joint four-primitive residual | 39,505 | 12 | 4 | +0.5684 | grid: Null B, align on, 12 levels, n=39,505, rho=0.5, kendalltau, 4 covariates, 1 coupling strength, 200 seeds |
| mRNAsi | GSE117498 | joint four-primitive residual | 12,354 | 4 | 4 | +0.6335 | design: `n=12354,rho=0.634,levels=4,k=4` |
| mRNAsi | GSE117498 | joint four-primitive residual (L = 12) | 12,354 | 12 | 4 | +0.6335 | design: `n=12354,rho=0.634,levels=12,k=4` |
| mRNAsi | GSE117498h | joint four-primitive residual | 12,354 | 4 | 4 | +0.6485 | design: `n=12354,rho=0.648,levels=4,k=4` |
| mRNAsi | GSE117498h | joint four-primitive residual (L = 12) | 12,354 | 12 | 4 | +0.6485 | design: `n=12354,rho=0.648,levels=12,k=4` |
| FitDevo | GSE106474 | joint four-primitive residual | 39,505 | 12 | 4 | +0.9420 | grid: Null B, align on, 12 levels, n=39,505, rho=0.93, kendalltau, 4 covariates, 1 coupling strength, 200 seeds |
| FitDevo | GSE117498 | joint four-primitive residual | 12,354 | 4 | 4 | +0.7822 | design: `n=12354,rho=0.782,levels=4,k=4` |
| FitDevo | GSE117498 | joint four-primitive residual (L = 12) | 12,354 | 12 | 4 | +0.7822 | design: `n=12354,rho=0.782,levels=12,k=4` |
| FitDevo | GSE117498h | joint four-primitive residual | 12,354 | 4 | 4 | +0.7843 | design: `n=12354,rho=0.784,levels=4,k=4` |
| FitDevo | GSE117498h | joint four-primitive residual (L = 12) | 12,354 | 12 | 4 | +0.7843 | design: `n=12354,rho=0.784,levels=12,k=4` |
| FitDevo | GSE125970 | joint four-primitive residual | 14,537 | 3 | 4 | +0.9440 | design: `n=14537,rho=0.944,levels=3,k=4` |
| FitDevo | GSE125970 | joint four-primitive residual (L = 12) | 14,537 | 12 | 4 | +0.9440 | design: `n=14537,rho=0.944,levels=12,k=4` |
| FitDevo | GSE113074 | joint four-primitive residual | 127,607 | 10 | 4 | +0.8143 | design: `n=127607,rho=0.814,levels=10,k=4` |
| FitDevo | GSE113074 | joint four-primitive residual (L = 12) | 127,607 | 12 | 4 | +0.8143 | design: `n=127607,rho=0.814,levels=12,k=4` |

### MCE

- **Class:** unsupervised. Enters the audit tally.
- **Declared primitive:** PCC(x, degree). Shi et al. attribute about 70% of MCE's variance to it (Fig 4D, R2 = 0.67).
- **Dataset:** GSE106474, `rank_kimmel`, 12 levels.
- **Score:** `scores/mce.py` once validated against MCE.m, on SR's network object (`<scratch>/scent_io/adjMC.npz`) with A_ii = 1, and on SR's input, log2(CPM + 1.1) over the network genes, unless MCE.m prescribes its own transform, in which case MCE.m's. Normalised by log(d). The score file `scores/GSE106474/mce.csv` lists all 39,505 cells in input order, NA outside the registered cell set.
- **Status: waiting for the author.** MCE.m could not be downloaded (DISCOVERY.md 1.2). The row runs only after, in this order: MCE.m is placed in `reference/` and its SHA256 recorded; DISCOVERY.md 1.3 is checked against it; the port takes MCE.m's input transform, start, tolerance and iteration cap; MCE.m runs in Octave on the 500 SCENT validation cells and the port matches it (Spearman and maximum absolute deviation reported; a mismatch is fixed in the port, never in the comparison); the port is timed on those 500 cells under MCE.m's stopping rule.
- **Full set or subsample:** the projection is single-core wall time, computed as in DISCOVERY.md 1.4: seconds per iteration per cell times the iterations MCE.m's stopping rule needs times 39,505. All 39,505 cells if that is under 8 hours; otherwise the 3,000-cell subsample of `experiments/track4/code/make_subsample.py` (seed 42, the one SLICE, dpath and NCG used). `run_rows.py` holds the choice in `MCE_CELLS` ("full" or "subsample") and refuses any other pattern of finite scores.
- All of this is written under Amendments, dated, before the MCE row runs: MCE.m's SHA256, any change from DISCOVERY.md 1.3, the port's settings, the validation numbers, the timing and the cell set.

### StemFinder

- **Class:** unsupervised. Nothing in scoring is fitted to labels. Enters the audit tally, with a caveat carried on its line: the authors chose among method variants (Gini, SD and variance; the k range) on a benchmark that includes the Farrell zebrafish set GSE106587, the SuperSeries of GSE106474, whose stages decide this row.
- **Declared primitive:** the cell-cycle gene-set score, the mean log-normalized expression of the same S and G2M markers (`gene_set_score()` from the package), which is the statistic the authors build the score from and benchmark it against.
- **Dataset:** GSE106474, `rank_kimmel`, 12 levels. Full set, because the measured full run took 1,352 s (22.5 minutes) on all 39,505 cells (under 8 hours).
- **Score:** `score_stemfinder.R`, the authors' `run_stemFinder()` and `gene_set_score()` sourced unmodified from the pinned checkout: the gene-level steps of `process_cyto_adata` (genes in at least 5 cells; `^MT-` genes and MALAT1 removed; no cell filter, since the cells are the repository's 39,505), LogNormalize 1e4, 2,500 vst variable genes, the bundled human S and G2M lists concatenated as the README vignette does (E2F8 is on both, so it enters twice, as in the authors' construction) and matched by upper-case symbol, markers removed from the variable genes, scaling of the variable genes plus markers, 50 PCs (seed 42), the number of PCs used = the elbow of the 50 standard deviations taken as the point farthest from the chord from the first to the last (the authors choose it by eye), k = round(sqrt(N)), `FindNeighbors`, `run_stemFinder(thresh = 0, method = "gini")`. The reported score `stemFinder` (higher = more differentiated) is the score; alignment takes care of its sign.

### mRNAsi

- **Class:** trained. The weights are a one-class logistic regression fitted to the PCBC pluripotent stem-cell class. Boundary row outside the tally, by the rule that put CytoTRACE 2 there.
- **Declared primitive:** none. Its baseline, and its decisive value, is the joint four-primitive residual.
- **Dataset:** GSE117498 with the tiers and the code path of `experiments/stemsc`: `w4_gate2_run.load_c1`, genes detected in at least 10 cells, scored on all 21,412 cells, evaluated on the ranked ones. From that path the loading, the gene filter and the tiers are taken; the estimator is the decider's (above), because the stemsc estimator has no tau_b.
- **Score:** `score_mrnasi.R`, TCGAbiolinks' own `TCGAanalyze_Stemness` and `SC_PCBC_stemSig` from TCGAbiolinks 2.40.0 (Bioconductor RELEASE_3_23, commit 2f9d249), on raw counts. No retraining, no symbol remapping: the function's own intersection of weight names and gene symbols decides which genes enter.
- **Registered sensitivity, GSE117498h.** The stemsc code path splits genes in two. The seven sorted-population files name them in HGNC form (HLA-A) and the four broad-gate files in R `make.names` form (HLA.A); `load_c1` unions by exact name, so in every broad-gate cell, including the 6,343 ranked CD34+/CD164+ cells at tier 4, those genes read zero, for the score and for PCC(x, degree) alike. GSE117498h merges each broad-gate name into its hyphenated twin (the rule in `prepare_inputs.ds_gse117498h`) and is otherwise identical. Both lines are reported side by side; the GSE117498 line is the one the brief specifies and the decisive one. The sensitivity line exists so the effect of the split is measured rather than guessed.

### FitDevo

- **Class:** trained. The background gene weights were fitted to timepoint labels in 17 training samples. Boundary row outside the tally.
- **Declared primitive:** none. Decisive value: the joint four-primitive residual.
- **Datasets:** every audit dataset absent from its training set, which is all four (DISCOVERY.md 1.7): GSE106474 (12 levels), GSE117498 (4 levels, stemsc code path as for mRNAsi, with the same GSE117498h sensitivity line), GSE125970 (3 levels, W4 `load_c2`, genes in at least 10 cells), GSE113074 (10 levels, the fix3 prepared matrix). One line per dataset. Caveats carried on the lines: on GSE106474 the authors once evaluated their weights against the stage labels of the same study (Spearman 0.228, not used in training); zebrafish (GSE106474) and Xenopus (GSE113074) symbols reach the human weights only by upper-case matching, and the authors left zebrafish out of training for too few homologous genes.
- **Score:** `score_fitdevo.R`, the authors' `fitdevo()` from `fitdevo.R` v1.2 (commit 0c757e6) sourced unmodified with the shipped `BGW.rds` and its defaults (NORM, PCNUM 50, VARGENE 2000, tooLargeLimit 50,000, SEED 123, SS), on raw counts. Seurat assays are created as v3 Assay objects so that the code's `@data` slot exists. No retraining.

## Predictions

**P1.** DISCOVERY.md 1.3, from the published equations, found that MCE is the entropy of the stationary edge distribution of a Markov chain on the network, MCE = H(pi) + h(P*): the static entropy of the expression distribution plus the entropy rate of the maximum-entropy chain that keeps it invariant. The brief's rule has two branches, entropy rate and static entropy, and does not say what to do with a construction that contains both. That gap is resolved here by a registered judgment, not by the rule: MCE's defining object is a Markov chain chosen by its transition entropy, so it is classified as an entropy-rate construction. **P1: MCE keeps tau_b conditional skill above its floor on PCC(x, degree), as SR does.** If MCE.m, once read, computes only H(pi), or anything without the entropy-rate term, the classification and the prediction become the static-entropy branch (MCE does not keep conditional skill above its floor), and that is written under Amendments, dated, before the MCE row runs. The risk to P1, recorded in DISCOVERY.md before any label: H(pi) <= MCE <= 2 H(pi) for every cell, so much of MCE's variance may be static entropy.

**StemFinder, mRNAsi, FitDevo:** no directional prediction. None of them is an entropy-rate, diffusion, identity or static-entropy construction in the paper's sense, and discovery found nothing that would predict their conditional skill.

## Amendments

Any change to a primitive, subsample, kernel, threshold, seed, input, parameter or dataset is written here with its date and reason before the rerun, and the first result stays in the report.

### Amendment 1, 2026-10-09: MCE.m arrived; the MCE row's settings, fixed before it runs

**Reason.** After the PREREG commit (71605a4) the author downloaded the supplement and placed it in `reference/bby093_supp/`. The steps this file required before the MCE row runs were then carried out, without any label, and are recorded here before the row runs.

- **MCE.m.** `reference/MCE.m`, SHA256 `c8ad11f2cdadc979cc303a21785f65eba77a41dfe565ace2beffaeae265fdce2` (the zip's other file, `ADF1.pdf`, is the supplementary tables and legends).
- **DISCOVERY.md 1.3 against MCE.m.** Line 30 computes the entropy rate h(P) of P_ij = lambda_i theta_j A_ij and line 32 adds the static entropy H(p0): MCE.m computes h(P*) + H(pi), as 1.3 found from the equations. The switch condition above (MCE.m computing only H(pi)) is not met, so **P1 stays on the entropy-rate branch: MCE keeps tau_b conditional skill above its floor on PCC(x, degree).**
- **The port's settings, taken from MCE.m** (`scores/mce.py`, `rule="mce_m"`): start lambda0 = p0, theta0 = 1; each step lambda = 1 ./ (A theta0), theta = p0 ./ (B' lambda); stop when the infinity norm of the change in (lambda, theta) is below 1e-2 or after more than 1e6 steps, each cell on its own; the score of lines 30 and 32, divided by log(nnz(A + I)).
- **Input transform.** MCE.m prescribes none: it divides by the sum and needs strictly positive entries, and ADF1.pdf names no preprocessing. So the registered default stands: SR's input, log2(CPM + 1.1) over SR's 8,468 network genes, CPM target = median library size.
- **Validation** (`discovery/mce_validation.json`). On the 500 SCENT validation cells, the port against MCE.m run unmodified in GNU Octave (aarch64-apple-darwin25.4.0) version 11.3.0: Spearman 1.000000, maximum absolute deviation 1.8e-13, step counts identical in all 500 cells. MCE.m's 1e-2 rule leaves the score within 8.3e-08 of the converged fixed point (Spearman 1.000000).
- **Timing and cell set.** The port scored the 500 cells in 26.2 s on one core under MCE.m's rule, 137 to 547 steps per cell, which projects to 0.58 h for 39,505 cells. That is under 8 hours, so the MCE row uses all 39,505 cells: `MCE_CELLS = "full"` in `run_rows.py`. The full run (`score_mce.py`, 8 processes on a machine shared with two other jobs) then took 1,944 s, 109 to 659 steps per cell, no NaN; the projection, not the run, decided the cell set, as registered.
- **The MCE values, from `run_rows.py plan`** after `scores/GSE106474/mce.csv` was written (SHA256 `3e8f92f7dbe93296e52b33250269b90ac4f6980e9db3a4894d37c17668db095b`), with the ordinal used only to know which cells are ranked. Both grid points are within the registered distance, so no design point was needed. The joint value sits at the declared primitive's rho, as the paper's joint rows do.

| row | dataset | value | n | levels | k | rho | floor source |
|---|---|---|---|---|---|---|---|
| MCE | GSE106474 | declared primitive (PCC(x, degree)) | 39,505 | 12 | 1 | +0.8024 | grid: Null B, align on, 12 levels, n=39,505, rho=0.9, kendalltau, 1 covariate, 3 coupling strengths, 600 seeds |
| MCE | GSE106474 | joint four-primitive residual | 39,505 | 12 | 4 | +0.8024 | grid: Null B, align on, 12 levels, n=39,505, rho=0.9, kendalltau, 4 covariates, 1 coupling strength, 200 seeds |

- **The plan.** `plan.json` now has SHA256 `943e68952b02af815ec789847cccd1509b0490dd1b673fa8a3f2b294456dfda0` (it was `e2d30f3abcde806f2719cb990a67c03d71ec9d6b35f83d07977b04e4db8695f5` at 71605a4). Every other row's entry is unchanged: same input hashes, n, rho and floor sources.
- **The other rows.** `run_rows.py values` recomputes every row. Its inputs and its seed (42) are unchanged, so it reproduces their first results exactly; that is checked against the committed `results/summary.json` and reported. The first results stay in the report.

## Outputs

- per row: `$OWNBASELINE_DATA_ROOT/w5_new_rows/results/<row>_<dataset>.json`
- summary: `experiments/w5_new_rows/results/summary.json`
- the missing design points: `$OWNBASELINE_DATA_ROOT/w5_new_rows/nulls/design_points.csv`
- the plan the floors were looked up from: `$OWNBASELINE_DATA_ROOT/w5_new_rows/plan.json`, summarized in the table above
- the report: `experiments/w5_new_rows/REPORT.md`
