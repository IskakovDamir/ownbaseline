# Specificity, the replicated unit and sequencing depth: pre-registration

Written on 2026-11-22, before any statistic below met a real stage label. The plan was drafted by an AI agent (Claude, through Claude Code) acting as analyst, under the author's instruction of the same day to design and run the missing experiments. It is committed on the branch `revision/specificity-controls` before `analyse.py` runs without `--dry-run`, and that commit is the only lock it has: the commit is local and unpushed, and the author had not read this file when the analysis ran. Whether the results count as confirmatory is the author's ruling to make after reading it. Until then they are registered by the analyst and unratified.

## What is already known

Every value, floor and interval in the preprint of 2026-11-05 is known, so this plan is written with all of them in view: the nine registered rows on GSE106474, the permutation floors, the within-batch rows, the bootstrap over samples and the replication on GSE112294 and GSE113074. The per-stage median library sizes of GSE106474 were printed by its preparation script months ago; the analyst has not looked at them.

What has never been compared with a stage label: any negative-control score; any permutation of stage labels between samples; any gene count or CytoTRACE value computed after depth equalisation or on the shared window of hours; any stratified concordance.

What was computed before this file was committed: the control scores, the thinned gene counts and the CytoTRACE reruns (`controls.py`, `window.py`). `controls.py` reads no stage. `window.py` reads the hours post fertilisation to select cells and compares nothing with them. `analyse.py --dry-run` was run to test the code path; it shuffles the stage labels between cells before anything is computed. The synthetic calibration described at the end reads the design of GSE106474 (which sample holds how many cells at which stage in which batch) and no expression value or score, so its timing against the lock does not matter; it was started before this file was committed and its tables are in `results/calibration_*.json`.

## The adversarial review before the lock

A second agent with no part in writing the plan reviewed this file and the code before any label was read. It changed five things.

1. **The residual of the sample test.** As first drafted, Part B averaged the least-squares residual of Eq. (2) within samples. The reviewer simulated scores with zero conditional skill on the real 28-sample design and found that test rejecting in 13 of 30 replicates for a straight-line score and 30 of 30 for a quadratic one, with no sample effect at all: a line on ranks leaves a function of the primitive in the residual, sample means of the primitive trend with stage, and Kendall's S over 28 means ignores how small the trend is. With the stratum-mean residual the count was 0 to 3 of 30. The analyst reproduced the quadratic case (p from 0.002 to 0.009 against 0.30 to 1.0). The stratum-mean residual now decides and the least-squares one is reported as sensitivity.
2. **What the control comparison can say.** A random gene set correlates little with PCC(x, degree), so residualising it on that primitive removes little, while signalling entropy loses most of its rank variance to the same step. The comparison is therefore between increments over a primitive, and the plan now words it that way, reports each control family's correlation with the covariates, and lets no row be called indistinguishable from a gene set.
3. **The lock.** The control files' hashes are now in `blind_manifest.json`, committed with this file. `analyse.py` reads the manifest from the commit, refuses a dirty tree, checks every file it opens and the number of controls in each.
4. **The registered floor for a control.** The floor grid holds five correlations and the audit takes the nearest. Most controls sit between grid points, where the nearest floor can be the lower one. For controls the plan now registers the larger of the two floors that bracket the control's own correlation, and reports the nearest-point reading beside it.
5. **Rules that could be argued either way.** The paper rules below now name their thresholds.

Unchanged by the review: the control families, their sizes and seeds, every blind file already computed, and the predictions other than S6, S7 and S8, which are restated with the changes above.

## The objections

1. **Specificity.** The registered rule asks whether a score's residual is distinguishable from the estimator's null. On a staged series most of the transcriptome moves with stage, so a statistic nobody would call a potency score may clear the same floor. Without the distribution of the statistic over such negative controls, "clears its floor" cannot be read as "adds biology". The same question was put to prognostic signatures in breast cancer, where most random gene sets were significantly associated with outcome (Venet et al. 2011).
2. **The replicated unit.** A stage is assigned to a sample of pooled embryos, and GSE106474 has 28 samples. The registered floor and the permutation floor both treat cells as exchangeable given the primitive. An effect shared by the cells of a sample and unrelated to stage is then counted 39,505 times. The preprint reports a bootstrap interval over samples beside each value and lets it decide nothing.
3. **Direction.** CytoTRACE runs at Spearman -0.933 with stage on GSE106474 and +0.939 on GSE112294, and the preprint does not say why. Lin et al. (2024, SCTC) had already reported that early cells of GSE112294 express fewer genes than later ones and that CytoTRACE infers its early stages wrongly; the preprint did not credit that. Two cheap explanations have not been tested: the libraries differ in depth by stage, and the two series cover different hours (3.3 to 12 against 4 to 24).
4. **A statistic that needs a floor table.** The residual statistic of Eq. (3) has a positive null mean that depends on n and on the score-primitive correlation. A reader cannot interpret its size.

## Part A: negative controls

### Control families

Computed by `controls.py` on the prepared matrices of the deposit, log1p of counts scaled to the median library size, the transform of the audit's primitives.

- **A1, random gene sets.** The mean log-normalised expression of g genes drawn uniformly without replacement from the genes detected in at least 10 cells. Primary: g = 200, 1,000 sets per series. Sensitivity: g = 20 and g = 2,000, 200 sets each, on the rows that use all cells.
- **A2, degree-permuted network statistic.** PCC(x, degree) with the STRING degree vector permuted among the scaffold genes of the series, 1,000 permutations. This is the primitive of SR, CCAT, ORIGINS, NCG and MCE on a network that carries no information. The unpermuted statistic recomputed by this code matches the deposit's `pcc_x_degree` to 1e-14 on all three series.
- **A3, CytoTRACE with random genes.** The CytoTRACE v1 port of the audit with one change: the 200 genes most correlated with the gene count are replaced by 200 random genes from the A1 universe. The similarity matrix, the non-negative least squares, the diffusion and the chunking seed are unchanged. 100 controls on each zebrafish series. The same code with the real gene-count signature returns the deposit's `cytotrace_v1` exactly (Spearman 1.000, largest absolute difference 0). The 100 gene sets without the two smoothing steps are kept as well, so that gene selection and smoothing can be separated. A3 is evaluated on gene count and on the four primitives jointly.

A control family describes what arbitrary statistics of this transcriptome do on this ordinal. It is a competitive comparison in the sense of Goeman and Buhlmann (2007): genes are correlated, so the spread over gene sets is not a sampling distribution, and no p-value is computed from it.

### What is computed

For every control, on the cells of the row it is compared with: conditional skill, Eqs. (1) to (3) with alignment, on each of gene count, PCC(x, degree), Shannon entropy and log10 library size alone (k = 1), on StemFinder's cell-cycle score where the series has it, and on the four jointly (k = 4); its marginal tau_b with the stages; its largest absolute Spearman correlation with the covariates removed. On GSE106474 each control's value is compared with the registered floor of Eq. (4) at its own n and k, read two ways: the larger of the floors at the two grid correlations that bracket the control's own (registered here as the reading that decides S1), and the nearest grid point, the audit's lookup, with the share of controls more than 0.15 from a grid point reported. On all three series the first 50 A1 controls (20 on GSE113074) are compared with a permutation floor, 200 draws, strata as in the preprint.

For NCG, dpath and SLICE the A1 (g = 200) and A2 controls are evaluated on the same 3,000 cells as the row.

### Decisions

A control is not built on the primitive it is residualised on. Removing PCC(x, degree) from a random gene set takes little away, and removing it from signalling entropy takes most of the rank variance. What the comparison measures is the **increment**: the stage ordering that is left once the covariates are removed.

- **Specificity of the floor.** The share of A1 (g = 200) controls above the registered floor, bracketed reading, per conditioning set, and above the permutation floor.
- **A row's increment against the controls' increments.** A row's *increment exceeds the controls'* on a set of covariates if its conditional skill on them lies above the 97.5th percentile of the A1 (g = 200) controls residualised on the same covariates on the same cells. This is reported on the declared primitive and on the four primitives jointly, with the row's percentile, the row's correlation with the covariates and the controls' median correlation with them. The joint comparison is the more even one, because the controls correlate with gene count, entropy and library size as the scores do.
- **The marginal comparison.** The share of controls whose absolute marginal tau_b with stage is larger than the row's. This is the analogue of the breast-cancer result and is reported for every row.
- **CytoTRACE against its own pipeline.** CytoTRACE is compared with the A3 distribution on gene count and on the joint set, and its unsmoothed gene-count signature with the unsmoothed random sets. This gives a two-by-two table: genes chosen by gene count or at random, smoothing on or off.

Series: GSE106474 (nine rows and the gene-count signature), GSE112294 (CytoTRACE, SR, CCAT, MCE), GSE113074 (CytoTRACE, SR, ORIGINS, CCAT, and dpath and SLICE on their subsample).

## Part B: the trend over samples

For each row of GSE106474 and GSE112294 the stratum-mean residual is averaged within each sample: the rank of the aligned score minus its mean rank in its stratum of the declared primitive, strata of 100 cells, 25 on the subsample. The statistic is Kendall's S between the sample means and the sample stages, summed over sequencing batches (DS2 to DS5 on GSE106474; one group on GSE112294, where each library holds one stage and batch cannot be separated from stage). The null permutes stage labels between the samples of a batch, 100,000 draws, two-sided, with the +1 correction. Samples with fewer than five cells of the row are dropped, which matters only on the subsample.

A row is *supported at the sample level* if p < 0.05 and the sign of S agrees with the sign of its registered conditional skill. For a row with a negative conditional skill this means that its negative residual trend holds over samples; it does not make the row clear a floor. Holm-adjusted p-values over the nine rows are reported beside the raw ones. Sensitivity, without a verdict: the same test without batch strata; the least-squares residual of Eq. (2); and for the joint set, a residual from a cross-fitted boosted regression of the score's rank on the ranks of the four primitives. GSE113074 names no sample, so it has no row here.

The test is scale-free: it reads only the order of the sample means, so with little variation between samples it detects a trend of any size. It establishes that a trend over the replicated unit exists and says nothing about how large it is; the size is the conditional skill.

## Part C: depth and the shared window

- **Depth by stage.** Median library size and detected genes per stage, the Spearman correlation of each with stage, and of detected genes with stage after library size is removed by rank residualisation. All three series.
- **Equal depth.** Every cell with at least 1,000 counts is drawn down to exactly 1,000 without replacement; cells with fewer are dropped and the number kept per stage is reported. Detected genes are recounted, and on the two zebrafish series the gene-count signature and CytoTRACE v1 are recomputed on the thinned matrix. Because dropping shallow cells is itself a selection, the same three quantities are also recomputed on the kept cells at raw depth. Sensitivity for the gene count: 500 and 2,000 counts.
- **Shared window.** On the two zebrafish series, the cells collected between 4 and 12 hours post fertilisation: dome to 6-somite in GSE106474, and the 4, 6, 8 and 10 hour collections in GSE112294. Gene count, gene-count signature and CytoTRACE are recomputed on those cells at raw depth and at 1,000 counts.

The result is a table per series: all hours or the shared window, raw or equal depth, holding the Spearman correlation with stage of the gene count, the gene-count signature and CytoTRACE.

**How it is read.** A direction counts only when the absolute Spearman correlation is at least 0.1 on both zebrafish series; the two series *agree* when both count and have the same sign.

- If CytoTRACE's direction agrees at equal depth over all hours, and does not agree on the kept cells at raw depth, depth accounts for the reversal.
- If it agrees on the shared window at raw depth, the hours outside the window account for it.
- If both hold, the paper states both and gives neither priority; the shared window at equal depth is reported as the combined case.
- If it agrees only on the shared window at equal depth, the paper says the two together account for it and neither does alone.
- If none holds, the paper says neither depth nor the hours covered accounts for the reversal.

## Part D: stratified concordance

For each row, Kendall's tau restricted to pairs of cells that fall in the same stratum of the declared primitive and come from different stages: concordant minus discordant pairs over the number of such pairs, on the aligned score, with the strata of the preprint. This is Kendall's tau with a blocking variable. Its null is the same statistic after the score is permuted within strata and aligned again, 200 draws. The null mean is close to zero when the score's marginal association with stage is strong and a fraction of a standard deviation above zero when it is weak, because alignment then picks the sign; the decision is therefore taken against the null's 97.5th percentile, and z is descriptive. It carries no verdict of its own in the paper. Permuting within bins of a continuous covariate is the local permutation test of Kim et al. (2022); the preprint attributed it to Berrett et al. (2020), whose test draws from a model of the covariate instead.

## Predictions

Registered here, each a guess that can fail.

- **S1.** On GSE106474 more than half of the A1 (g = 200) controls exceed the registered floor, bracketed reading, on each of the four primitives and on the four jointly.
- **S2.** CytoTRACE's increment over gene count exceeds the controls' on GSE106474 and on GSE112294 and does not on GSE113074.
- **S3.** CytoTRACE's increment over gene count does not exceed the 97.5th percentile of A3 on either zebrafish series.
- **S4.** The increments of dpath and StemFinder over their declared primitives do not exceed the controls' on GSE106474.
- **S5.** The increments of SR, ORIGINS and NCG over PCC(x, degree) do not exceed the controls' on GSE106474. The review notes that this one is close to settled in advance by the low correlation of random gene sets with PCC(x, degree).
- **S6.** CytoTRACE, SR and ORIGINS are supported at the sample level on GSE106474; NCG, dpath and StemFinder are not.
- **S7.** At 1,000 counts per cell, detected genes fall with stage on both zebrafish series, at a Spearman of -0.1 or below, and CytoTRACE recomputed at that depth agrees in direction on the two.
- **S8.** On the shared window at raw depth CytoTRACE's correlation with stage has opposite signs on the two series, each at 0.1 or more in absolute value.
- **S9.** The stratified concordance is above its null's 97.5th percentile for the six rows that clear the registered floor and not for CCAT, SLICE and MCE.

## What this does to the paper

The registered rule, Eqs. (1) to (4), and its verdicts and counts on GSE106474 stay as they are, labelled as registered. Everything here is reported beside them as added on 2026-11-22 after all registered values were known, in a table of deviations in the form of Table A5, with every failed prediction listed as failed.

- **The floor.** The share of A1 controls above the registered floor (bracketed) is stated in the abstract and in section 3 for the gene-count and PCC(x, degree) sets and in the appendix for all five. Where that share is above 50 percent on a row's conditioning set, the paper says that clearing the floor on that set does not separate a potency score from an arbitrary gene set, and no row is described as adding biology on the strength of clearing it. Between 5 and 50 percent the paper gives the share and draws no such conclusion. Below 5 percent it says the floor is specific on that set.
- **Increments.** The number of the six clearing rows whose joint increment exceeds the controls' goes into the abstract, with the number whose increment over the declared primitive does. A row that clears its floor and whose increments do not exceed the controls' is described in those words: its increment over the named covariates is within the range of arbitrary gene sets. Its marginal comparison is given in the same sentence. No row is called indistinguishable from a gene set.
- **CytoTRACE.** The two-by-two table goes into section 4, per series. Where CytoTRACE does not exceed A3 the text says that its increment over gene count lies inside the range of the 100 random-gene controls run through the same pipeline on that series. Where it does, the text says the gene-count signature contributes beyond the smoothing on that series. If the two series differ the paper says so.
- **Samples.** A clearing row that is not supported at the sample level is named in section 5 and in Limitations as not established at the level of the replicated unit. No registered verdict changes. If the calibration shows the sample test rejecting in more than 10 percent of zero-skill replicates of any case without a planted trend, the paper says the test is anti-conservative in that case and its p-values are given without the word "supported".
- **Depth.** Section 5 reports the table and the reading fixed in Part C, and credits Lin et al. (2024) whatever it shows.

No further variant is run after the results are seen. A new variant needs its own plan.

## What this does not address

Three staged series, two of them zebrafish, remain a narrow base, and stage is not potency. Two published resources would widen it and are not part of this plan: a zebrafish atlas with 62 stages (Sur et al. 2023, GSE223922) and a mouse gastrulation series with 153 individually staged embryos (Mittnenzweig et al. 2021, GSE169210), whose label unit is the embryo. CytoTRACE 2, the current supervised method, is not run. A3 exists only for CytoTRACE: no pipeline-matched control was built for the network scores beyond A2, and SR on a permuted network was not computed. The sample test has 28 units over 12 stages on GSE106474, with five or six samples in three of its four batches, so its power is limited and a row it does not support may still carry skill. Its stratum-mean residual removes a function of the primitive only down to the width of a stratum. On GSE112294 batch and stage cannot be separated by any test. No one outside the author has rerun any of this.

## Blind files

`blind_manifest.json`, committed with this file, lists every control file with its SHA256, its size and the shape of every array in it. `analyse.py` reads the manifest from the commit and refuses to open a file that differs.

## Synthetic calibration

`calibrate.py`, two modes, neither reading a real score.

**cluster.** The design of GSE106474 is kept: which sample holds how many cells, at which stage, in which sequencing batch. Everything else is simulated. The primitive tracks stage at coupling 0.55. A score is a function of its primitive, plus noise, plus an effect shared by the cells of a sample and drawn independently of stage, so its true conditional skill is zero. The function is a straight line at rank correlation 0.5 (sample effect of standard deviation 0, 0.05, 0.1, 0.2 or 0.5 of the cell-level noise) or 0.9 (0, 0.1, 0.5), a cube (0, 0.1) or a square (0, 0.1). One further case adds an effect per sequencing batch with standard deviation 0.3. For each case, 200 replicates: how often the value clears the registered floor, how often the sample test gives p < 0.05 on the stratum-mean residual with and without batch strata, and how often it does on the least-squares residual. For four of the cases the permutation floor over cells is measured as well, 40 replicates of 200 draws. Three cases with a planted trend over stages, 0.05, 0.1 and 0.2 standard deviations per unit of standardised stage, measure the power of the sample test.

**concordance.** The coupled and curved nulls of `experiments/conditional_null`, at rank correlation 0.5 and 0.9, 200 replicates at n = 3,000 and 20 at n = 39,505 (200 and 100 permutation draws): how often the stratified concordance exceeds the 97.5th percentile of its own permutation null. Nominal is 2.5 percent. If it does so in more than 10 percent of replicates of a case, that case is named beside every concordance value.
