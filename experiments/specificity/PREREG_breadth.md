# Breadth, a mouse series with the embryo as the unit and a later zebrafish series: pre-registration

Written on 2026-11-22, after the plans of `PREREG.md` and `PREREG_depth_2.md` had run and their results were known, and before any score, primitive or control of the two series below met a stage label. Same analyst, same status: drafted by an AI agent (Claude, through Claude Code) under the author's instruction of the same day, locked by a local unpushed commit, unratified until the author reads it. The author approved the downloads from GEO by file name and size. He has not read this file. No second agent reviewed it; the first plan had such a review and this one reuses its statistics unchanged.

## What prompted it

`PREREG.md` closed by naming its base as narrow: three staged series, two of them zebrafish, at most 32 replicated units, and it named two published series that would widen it. This plan runs the registered quantities on those two.

## What is already known

Everything in `results/REPORT.md`, with every number in view while the predictions below were written.

Of the new series, before the lock: the GEO file lists; the metadata tables, read for their columns and their design (which cell belongs to which embryo, lane and collection time); the totals the preparation script prints for GSE169210 (33,700 cells kept, median library size 4,543.5, all 33,700 cells with 1,000 counts); and the paper's account of how the mouse embryos were timed. The metadata table of GSE223922 carries the authors' own UMI and gene count per cell beside the collection time. The analyst printed their overall median and the share of cells with 1,000 counts (5,259 and 91 percent) and did not tabulate them by time.

Never compared with a stage label: any score, primitive or control of either series.

Computed before this file was committed, by code that opens no label: the count matrix, the four primitives, the scores and the controls of GSE169210 (`prep_mittnenzweig.py`, `breadth_blind.py`). `breadth_analyse.py --dry-run` was run once, on eight controls per family, to test the code path; it shuffles the stage labels between cells before anything is computed, and the report was rendered once from that shuffled output. For GSE223922 nothing had been computed when this file was committed. A guard on free disk space stopped the transfer of its count matrix at 0.98 of 2.55 GiB, while the machine's swap grew to 15 GB under another job; whether it resumes depends on the machine, as stated below.

## The labels, and a correction to the brief

The brief for this work described both series as carrying stage labels fixed outside the transcriptome. That is true of one of them.

**GSE223922 (Sur et al. 2023).** The label is the collection time in hours post fertilisation, fixed at the bench. GEO states that the cells of 3 to 12 hpf are the cells of Farrell et al. 2018 remapped. Those are GSE106474, which the audit already holds, so the 38,731 Drop-seq cells are left out. What remains is 10x only: 450,955 cells, 53 collection times from 14 to 120 hpf, 58 lanes. In 26 lanes embryos of four or six collection times were hashed and loaded together; 32 lanes hold one collection time.

**GSE169210 (Mittnenzweig et al. 2021).** 153 wild-type embryos were dissociated one by one and the embryo of every cell was recorded at the sort, so the unit is fixed outside the transcriptome. The time is not. GEO gives one `developmental_time` per embryo, and the paper says that "each embryo was assigned a calculated timestamp based on size and transcription" (Results). The embryos were ordered by the transcriptional similarity of their cells; the authors compared that order with a ranking by morphology alone, report a rank correlation of 0.97 between the two, and converted it to days with size measurements. The analyst read the Results and the figure legends; the STAR Methods were not read in full. GEO carries no morphological rank per embryo, and the plate records give only "E6.5 to E8.25".

What follows from that is fixed here, before any result:

- No audited score or primitive enters the authors' ordering, but a statistic that orders these cells by time agrees with an order built from the same transcriptomes. Every number from GSE169210 is described as coming from "an embryo-resolved series timed by its authors from transcription and size". It is never described as a label fixed outside the transcriptome.
- A high share of random gene sets above the floor is expected on this series for that reason alone. It is reported and is not counted as independent support for the statement that the floor is unspecific. GSE223922 carries that question.
- What GSE169210 adds is the unit: 153 embryos in 18 sort sessions, against 28 and 32 samples before.

## Cells, units and strata

**GSE169210.** Cells: the wells of an embryo with a developmental time that the authors assigned to a metacell, their own quality filter (33,700 wells), then at least 200 detected genes, which all of them have: 33,700 cells, 19 to 1,278 per embryo, median 134. Every one of them holds at least 1,000 counts, so thinning to 1,000 drops no cell here. Unit: the embryo, 153 of them. Stratum: the sort session, the first token of the embryo's identifier, 18 sessions, of which 17 hold embryos of more than one time. Ordinal: `developmental_time` in days, 137 distinct values. Mouse symbols are upper-cased to meet the human STRING scaffold, the convention of the audit's other non-human series; of several aliases joined by ';' the first on the scaffold is taken.

**GSE223922.** The atlas does not fit this machine, so a subsample is fixed from the metadata table alone, before the count matrix is opened: from each of the 184 lane-by-time units, 220 cells drawn uniformly without replacement among the unit's cells with at least 200 detected genes, every cell of a unit that has fewer (seed in `prep_sur.py`). That gives 40,108 cells before the check on the loaded matrix; two units hold fewer than 220 cells (57 and 11). The draw reads the collection time to form the units and reads no expression value. Unit: the lane-by-time pool. Stratum: the lane, so that stage labels are permuted only between pools that shared a lane and a lane's technical effect cannot produce a trend; the 32 single-time lanes then contribute nothing. Sensitivity: strata by experiment (the lane name's first token, TC1 to TC8), and no strata. Ordinal: hours post fertilisation, 53 values.

The zebrafish part runs only if the download completes and the machine holds it. If it does not, the Z rows below are void and the report says so.

## The blind stage

Per series, by `breadth_blind.py` through the functions of `controls.py`, on log1p of counts scaled to the median library size:

- the four primitives: detected genes, PCC(x, degree) on the deposit's STRING scaffold, Shannon entropy, log10 library size;
- CytoTRACE v1 and its gene-count signature, through the chunked pipeline that reproduced the deposit's values exactly; SR and CCAT through the functions of `run_scent_full.py`; MCE through `score_mce.py` where the machine allows. It is not run on GSE169210: the machine is shared with another job and MCE took half an hour on eight idle cores for GSE106474, so M5 and M9 are void. On GSE223922 it is run only if the machine is free by then. A score that is not in the manifest has no row, and its predictions are void;
- A1, 1,000 random sets of 200 genes among the genes detected in at least 10 cells; A2, PCC(x, degree) under 1,000 permutations of the degree vector;
- every cell with at least 1,000 counts drawn down to 1,000 without replacement, detected genes recounted, the gene-count signature and CytoTRACE recomputed on the thinned matrix and on the same cells at raw depth; 500 and 2,000 counts for the gene count.

On GSE169210 the blind stage found 16,853 genes detected in at least 10 cells and 12,559 genes on the scaffold; PCC(x, degree) recomputed from the deposit's degree table agrees with the primitive to 7e-15; 31,593 cells hold 2,000 counts.

`blind_manifest_breadth.json` records the SHA256 and the array shapes of each file, and the SHA256 of the label table. `breadth_analyse.py` reads the manifest from the commit and opens nothing that differs. The entries of GSE169210 are committed with this file. Those of GSE223922 are added by a later commit, before its labels are read, and that commit changes nothing else in this plan; if something must change it is appended here as a dated amendment.

## What is computed

The statistics of `PREREG.md`, through the same functions.

- **Part A, controls.** Conditional skill of every control on each of the four primitives and on the four jointly. The permutation floor of the first 50 A1 controls on each primitive, 200 draws, strata as in the preprint: this is the floor that decides here. The audit's simulated floor was built for 12 ordinal levels; both series have many more, so its reading (bracketed, all 1,000 controls) is printed as an extrapolation and decides nothing. For each audited row (CytoTRACE and its gene-count signature on gene count; SR, CCAT and MCE on PCC(x, degree)): conditional skill, its own permutation floor, and where its increment sits among the controls' increments over the declared primitive and over the four jointly.
- **Part B, the trend over the unit.** The stratum-mean residual averaged within each unit; Kendall's S between the unit means and the unit stages, summed over strata; stage labels permuted between the units of a stratum, 100,000 draws, two-sided, with the +1 correction. Units with fewer than five cells are dropped. A row is supported over the unit if p < 0.05 and the sign of S agrees with the sign of its conditional skill; for a negative conditional skill that means the negative residual trend holds over units. Sensitivity without a verdict: no strata; the least-squares residual, which the first plan's calibration found anti-conservative; for GSE223922, strata by experiment.
- **Part C, depth.** Spearman correlation with stage of library size, of detected genes, of detected genes after library size is removed by rank residualisation, of the gene-count signature and of CytoTRACE, at raw depth and at 1,000 counts, and on the kept cells at raw depth. The same correlations over per-unit medians, as description.

Not carried over: A3, A1 at 20 and 2,000 genes, the stratified concordance, the boosted joint residual, the shared window, PCC(x, degree) at equal depth. ORIGINS, NCG, dpath, SLICE and StemFinder are not scored on these series.

## Predictions

One per row, the same thirteen for each series: M1 to M13 on GSE169210, Z1 to Z13 on GSE223922. Each is a guess that can fail, and each is marked held, failed or void by `breadth_analyse.py`.

| row | prediction | on the three earlier series | fails if |
|---|---|---|---|
| 1 | More than half of the first 50 random gene sets are above the permutation floor on each of the four primitives. | 52 to 98 percent over twelve series-by-primitive cells | the share is 50 percent or less on any primitive |
| 2 | CytoTRACE's joint increment is above the 97.5th percentile of the random gene sets. | above on 3 of 3 | it is not above |
| 3 | SR's joint increment is above that percentile. | above on 3 of 3 | it is not above |
| 4 | CCAT's joint increment is not above it. | not above on 3 of 3 | it is above |
| 5 | MCE's joint increment is not above it. | not above on 2 of 2 | it is above |
| 6 | CytoTRACE's residual trend is supported over the unit. | supported on 2 of 2 | p of 0.05 or more, or the sign disagrees |
| 7 | SR's residual trend is supported over the unit. | supported on GSE112294, not on GSE106474 (p = 0.086) | the same |
| 8 | CCAT's negative residual trend is supported over the unit. | supported on 2 of 2 | the same |
| 9 | MCE's negative residual trend is supported over the unit. | supported on 2 of 2 | the same |
| 10 | At raw depth detected genes fall with stage, Spearman -0.1 or below. | -0.454, +0.085, +0.040 | Spearman above -0.1 |
| 11 | At 1,000 counts detected genes fall with stage, Spearman -0.1 or below. | -0.747, -0.771, -0.564 | Spearman above -0.1 |
| 12 | At raw depth CytoTRACE falls with stage, Spearman -0.1 or below. | -0.933, +0.939, -0.255 | Spearman above -0.1 |
| 13 | At 1,000 counts CytoTRACE falls with stage, Spearman -0.1 or below. | -0.912, -0.920 on the two zebrafish series | Spearman above -0.1 |

Rows 10 and 12 are the least secure: on two of the three earlier series the gene count did not fall with stage at raw depth.

## How the results are read

The analyst does not edit the manuscript. The report states, per earlier finding, whether it replicates on each new series, by these rules.

- **Arbitrary gene sets clear the floor.** Replicates on a series if row 1 holds. For GSE169210 the sentence about its label accompanies the number.
- **Joint increments.** The earlier pattern is rows 2 to 5. Each row is reported on its own; the pattern is said to replicate on a series only if all of its scored rows hold.
- **Support over the replicated unit.** Rows 6 to 9, each on its own. A row that fails is named as not established over the unit on that series. The test is scale-free: it establishes that a trend over units exists and says nothing about its size.
- **Direction at equal depth.** The earlier finding is that at 1,000 counts detected genes and CytoTRACE fall with stage on every series where both were computed. It replicates on a series if rows 11 and 13 hold. If row 10 or 12 fails while 11 and 13 hold, the raw-depth direction on that series is attributed to depth, as on GSE112294, and the kept-cells row at raw depth is quoted to show that dropping shallow cells does not account for it.

Every failed row is listed as failed. No further variant is run after the results are seen; a new variant needs its own plan.

## What this does not address

The time of the mouse embryos is derived from their transcriptomes, as stated above, and nothing here removes that. The mouse cells are those the authors kept. Mouse and zebrafish symbols meet a human scaffold by name. The mouse ordinal is close to continuous, which the permutation floor handles and the audit's simulated floor does not. A sort session whose embryos span a few hours gives the within-session permutation little to work with, and a lane with one collection time gives it nothing. The zebrafish result rests on 220 cells per unit of one draw. The hashed lanes were demultiplexed by the authors. Stage is not potency. No one other than the analyst has rerun any of this.
