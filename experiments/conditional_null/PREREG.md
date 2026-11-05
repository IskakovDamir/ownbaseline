# Floors without the simulator, and a staged series the audit has not seen: pre-registration

Written on 2026-11-05, before any draw or value below met a real stage label. This file is committed and pushed before the `real` run of `conditional_null.py` and before any score of GSE112294 is compared with its stages.

## What is already known

Every value and floor in the preprint of 2026-10-22 is known, so Part 1 is written with the registered results in view. The isotonic fit was run on seven rows (Appendix I). The Xenopus conditional-skill values of `fix3/results/gate2_conditional_skill.json` are known.

What has never been computed on real stage labels: a conditional-permutation draw, a stratum-mean residual, an isotonic value for MCE or StemFinder, a leave-one-stage-out value, a bootstrap over samples.

GSE112294 (Part 2) has not been analysed at all. Its counts were prepared and CytoTRACE, SR, CCAT and MCE were scored on it before this file was written, none of it reading the hours post fertilisation. The stage of a cell is in its table and has not been compared with any score or primitive.

## The objections

1. A floor is read from a simulator (Appendix D), so it is as good as the simulator's resemblance to real cells.
2. A least-squares fit on ranks removes a straight line. A curved trace of the primitive stays in the residual, and if the ordinal shares that curve a kernel scores it as skill.
3. The ordinal is one staged series, so a statement about the field rests on one experiment.
4. The ordinal is assigned per sample and the 28 Drop-seq samples are the replicated unit. An interval over cells ignores that.
5. Stage and sequencing batch are partly confounded on GSE106474, and nothing in the primitives removes a batch effect.

Objections 1 and 2 are answered by one null built from the real vectors. Under "the score carries nothing beyond its primitive", score and ordinal are independent given the primitive, so permuting the score among cells with the same primitive value leaves the joint law unchanged (Berrett et al. 2020, JRSS B 82:175). A draw keeps the real primitive, the real ordinal, their dependence, and the real score's distribution and nonlinear relation to the primitive at each primitive level. Nothing is simulated.

The synthetic calibration at the end of this file shows objection 2 is real for the shipped floor: with zero true conditional skill and a curve shared by score and ordinal, the shipped floor was cleared in every replicate. That holds whatever the real data show, and the paper states it.

## Part 1: floors without the simulator

### Rows

- GSE106474: the nine placed rows on their declared primitives, on the audit's cells (39,505, and the shared seed-42 3,000-cell subsample for NCG, dpath and SLICE), and the six joint rows (CytoTRACE, SR, ORIGINS, StemFinder, MCE, dpath).
- GSE113074: CytoTRACE, SR, CCAT, ORIGINS at n = 127,607 and dpath, SLICE on their 3,000-cell subsample, on the primitives the Xenopus run declared, and the joint rows of CytoTRACE, SR, ORIGINS, dpath.
- The seven trained boundary rows of Table A3, at k = 4, on their ranked cells, by the generalised covariance measure.

Inputs are the score and cell tables of the data deposit, plus the Xenopus score vectors of `fix3/results`. No count matrix is read.

### Method

`experiments/conditional_null/conditional_null.py`, using `align` and `rank_resid_multi` from `own_baseline/conditional_skill.py` unchanged.

**Strata, k = 1.** Cells sorted by the declared primitive, ties broken by a seeded shuffle, cut into consecutive strata of m cells; the last stratum takes the remainder. Primary m = 25 at n up to 3,000 and m = 100 above it, chosen on the synthetic calibration below. Sensitivity at half and at twice the primary m.

**A draw.** The score is permuted within strata. Eqs. (1) to (3) are applied to the draw exactly as to the real score, so alignment is recomputed in every draw. 200 draws, seeded streams keyed on row, m and draw.

**Permutation floor.** The 97.5th percentile of the 200 draws.

**Residualizers.** The registered least-squares fit; an isotonic fit on ranks, direction from the sign of the rank correlation, primary m and k = 1 only; the stratum mean of the score's ranks, which follows a non-monotone trace as well. Each value is compared with a floor computed under the same residualizer.

**Second reading.** Each unsupervised row also read with the authors' polarity, alignment off, as in Appendix G. Trained rows are read under the registered reading only.

**The generalised covariance measure, k = 1 and k = 4.** Strata do not hold at k = 4. A first attempt, strata along a cross-fitted prediction of the score from the four primitives, was calibrated on synthetic data, cleared the curved case in 52 to 92 percent of replicates, and was dropped before any real label was read. In its place the aligned score's rank and the ordinal's rank are each regressed on the ranks of the primitives by gradient boosting, five folds, cross-fitted, and T is the studentized mean of the product of the two residuals (Shah and Peters 2020). A row clears at T > 3.5, for the reason given under the calibration. Both regressions see only primitives and the variable they predict.

**Inside each batch.** Stage and sequencing batch are partly confounded on GSE106474: batch DS5 holds seven stages and DS2, DS3 and DS4 each hold the other five. The nine rows are repeated inside each of the four batches, with their permutation floors, so that every comparison is between stages sequenced together. A row read this way needs its value above the permutation floor in every batch to be called robust to batch.

**Leave one stage out.** Registered conditional skill recomputed with each stage removed, k = 1, GSE106474 and GSE112294 and GSE113074.

**Bootstrap over samples.** 1,000 resamples of samples with replacement, read from the barcode, Eqs. (1) to (3) refitted in each. Not stratified by stage, so a stage held by one sample can vanish from a resample. GSE106474, 28 samples, and GSE112294, by library.

### Primary comparisons

For objection 1: the registered conditional skill against its permutation floor, least-squares residualizer, primary m.

For objection 2: the stratum-mean residual against its permutation floor, primary m; at k = 4, the generalised covariance measure.

For the batch confound: the registered conditional skill against its permutation floor in each of the four batches.

Everything else is sensitivity and is reported without a verdict of its own.

## Part 2: a staged series the audit has not seen

**Data.** GSE112294 (Wagner et al. 2018, Science 360:981): zebrafish whole embryos collected at 4, 6, 8, 10, 14, 18 and 24 hours post fertilisation, inDrops, another laboratory and platform than GSE106474. The seven `GSM30671NN_*hpf.csv.gz` raw count tables, 36,749 cells, all with at least 200 detected genes. The two TracerSeq libraries are left out. `experiments/wagner2018/prep_wagner.py` is the preparation. Symbols are upper-cased to match STRING, the audit's convention, and 8,618 intersect the scaffold.

**Ordinal.** The seven collection times, 1 = 4 hpf to 7 = 24 hpf, fixed before dissociation. Earlier is more potent.

**Scores.** CytoTRACE v1, SR, CCAT and MCE, by the audit's own code, `run_cytotrace.py`, `run_scent_full.py`'s functions and `score_mce.py`. ORIGINS, NCG, dpath, SLICE and StemFinder need R runs or inputs that were not repeated here; they are not part of this plan. Each score is residualized on the primitive it declares in Table A1.

**Rule.** The registered rule, Eqs. (1) to (4), with the floor from a design point at this series' own n, the row's own rho and L = 7, computed by `run_missing_cells.py --design` before the value is read. Part 1's primary comparisons and sensitivities apply as well.

**Predictions, one per row, registered here.**

- W1. CytoTRACE clears on gene count.
- W2. SR clears on PCC(x, degree).
- W3. CCAT does not clear on PCC(x, degree).
- W4. MCE does not clear on PCC(x, degree).
- W5. PCC(x, degree) rises with stage, as on GSE106474, and gene count falls with stage.
- W6. SR's marginal ordering runs against its authors' polarity and CytoTRACE's runs with it.
- W7. CytoTRACE and SR stay above their k = 4 floors on the four primitives, and T > 3.5 for both.

## What this does to the paper

Eq. (4) stays the registered rule, and the registered verdicts and counts on GSE106474 stay as they are. Part 1 is reported beside them, labelled as added after the registered values were known. Part 2 is a pre-registered replication.

Whatever these show, all values go into the paper, in new appendix tables and in a table of deviations in the form of Table A5.

- If a row that clears Eq. (4) on GSE106474 fails a primary comparison at k = 1 or the k = 4 measure, or falls below its floor in any batch, it is named in the abstract, in section 3 and in Limitations, and the claim of section 6 that the reduction is false for that row is withdrawn.
- If a row that does not clear Eq. (4) clears a primary comparison, that is reported and changes no count.
- If removing one stage takes a value below its registered floor, the row and the stage are named in section 3.
- An interval over samples that includes zero is reported as such. Intervals take no part in the decision, so it changes no verdict.
- If a prediction of Part 2 fails, it is named in the abstract and in section 3 and the row is described as not replicated.

No further variant is run after the results are seen. A new variant needs its own plan.

## What this does not address

Three staged series, two of them zebrafish, remain a narrow base. Permutation within strata treats cells in a stratum as exchangeable given the primitive, and a stratum still spans a small range of the primitive. Sample structure is not represented in the draws; the bootstrap over samples looks at it separately. The within-batch rows have fewer stages, five or seven, and the five-stage batches span 5.3 to 10 hours only. No one outside the author has rerun any of this.

## Synthetic calibration, run before this file was committed

Zero true conditional skill in every case. "Coupled" is the null of Eq. (5). "Curved" keeps the score independent of the ordinal given the primitive, but both share a curved function of it, s = p + h(p) + noise and an ordinal cut from h(p), with h(p) = (p^2 - 1)/sqrt(2). A false clear is a value above its floor. Nominal is 2.5 percent. Files: `results/synthetic_k1_n*.json`, `results/synthetic_gcm_n*.json`.

**Shipped floor and permutation floor, k = 1, least-squares residual unless stated.**

| case | rho parameter | n | reps | m | shipped floor | permutation floor | stratum mean, its floor |
|---|---|---|---|---|---|---|---|
| coupled | 0.5 | 3,000 | 200 | 25 | 0.030 | 0.040 | 0.045 |
| coupled | 0.9 | 3,000 | 200 | 25 | 0.010 | 0.020 | 0.025 |
| curved | 0.5 | 3,000 | 200 | 25 | 1.000 | 0.020 | 0.025 |
| curved | 0.9 | 3,000 | 200 | 25 | 1.000 | 0.050 | 0.035 |
| coupled | 0.5 | 3,000 | 200 | 100 | 0.030 | 0.050 | 0.055 |
| coupled | 0.9 | 3,000 | 200 | 100 | 0.010 | 0.025 | 0.035 |
| curved | 0.5 | 3,000 | 200 | 100 | 1.000 | 0.095 | 0.105 |
| curved | 0.9 | 3,000 | 200 | 100 | 1.000 | 0.265 | 0.275 |
| coupled | 0.5 | 39,505 | 40 | 100 | 0.000 | 0.000 | 0.000 |
| coupled | 0.9 | 39,505 | 40 | 100 | 0.000 | 0.000 | 0.000 |
| curved | 0.5 | 39,505 | 40 | 100 | 1.000 | 0.000 | 0.000 |
| curved | 0.9 | 39,505 | 40 | 100 | 1.000 | 0.000 | 0.050 |

m = 100 at n = 3,000 fails the curved case, 0.095 and 0.265, because its strata are too wide; hence m = 25 there. At n = 39,505, m = 100 holds. The 40 replicates at n = 39,505 are few: the rates there are good to about 0.05.

**Generalised covariance measure, false clears at T > 1.96.**

| case | rho parameter | n = 3,000, 200 reps | n = 39,505, 20 reps | mean T at 39,505 |
|---|---|---|---|---|
| coupled | 0.5 | 0.060 | 0.000 | +0.18 |
| coupled | 0.9 | 0.015 | 0.050 | +0.32 |
| coupled_k4 | 0.5 | 0.055 | 0.050 | +0.01 |
| coupled_k4 | 0.9 | 0.045 | 0.000 | -0.07 |
| curved | 0.5 | 0.035 | 0.000 | -0.35 |
| curved | 0.9 | 0.030 | 0.000 | +0.00 |
| curved_k4 | 0.5 | 0.055 | 0.100 | +0.91 |
| curved_k4 | 0.9 | 0.090 | 0.350 | +1.45 |

The boosted fits do not remove the curve entirely at k = 4 and large n, so T drifts upward: mean +1.45 at n = 39,505 with the score nearly deterministic in the primitives. A row is therefore read as clearing the measure at T > 3.5, that mean plus two standard deviations, with T and the partial rank correlation reported either way. This threshold is fixed here, before any real T is seen.

