# Depth in the sorted atlases and in the degree correlation: pre-registration

Written on 2026-11-22, after the plan of `PREREG.md` (commit `833acba`) had run on the two zebrafish series and before any quantity below met a label. Same author, same analyst, same status: drafted by an AI agent under the author's instruction, locked by a local unpushed commit, unratified until the author reads it.

## What prompted it

`PREREG.md` found that the reversal of CytoTRACE between the two zebrafish series is sequencing depth. At raw depth CytoTRACE runs at Spearman -0.933 with stage on GSE106474 and +0.939 on GSE112294. Drawn down to 1,000 counts per cell it runs at -0.912 and -0.920, and detected genes fall with stage on both, -0.747 and -0.771. The 4-hour libraries of GSE112294 hold a median of 1,891 counts against 6,228 to 8,079 at the later times. The shared window of hours does not account for it.

Two statements of the paper are now open to the same explanation and neither has been tested:

1. Section 2 reports that in sorted haematopoietic progenitors (GSE117498) gene count orders the hierarchy below chance, AUROC 0.378 for stem cells against the bottom tier, and above chance in intestinal epithelium (GSE125970), 0.798. Both are read at raw depth.
2. Section 3 reports that PCC(x, degree) rises with stage on all three staged series, which is why signalling entropy, ORIGINS and CCAT run against their authors' polarity. That is also read at raw depth.

## What is already known

Everything in the preprint and everything `analyse.py` wrote for GSE106474 and GSE112294. What has never been computed: a detected-gene count or a gene-count signature of either sorted atlas at equal depth, the library size of each sorted population, and PCC(x, degree) at equal depth on any series. The analyst has not looked at the library sizes of the sorted populations.

`sorted_depth.py blind` computed the thinned quantities before this file was committed, reading no label; `blind_manifest_depth_2.json` records their hashes.

## What is computed

**Sorted atlases, GSE117498h and GSE125970.** Every cell with at least 1,000 counts is drawn down to exactly 1,000 without replacement, with 500 and 2,000 as sensitivity; cells with fewer are dropped. On the thinned matrix: detected genes, and at 1,000 counts the gene-count signature, the mean log expression of the 200 genes most correlated with the gene count, recomputed. The same two quantities at raw depth on all cells and on the cells that survive the cut. For each: the AUROC of the top tier against the bottom tier as section 2 defines it (stem cells against GMP, MEP and Pre-B/NK in GSE117498h; stem cells against the differentiated tier in GSE125970), ties counted half, and Kendall's tau_b over all ranked cells. Per population: cells, cells with 1,000 counts, median library size, median detected genes at raw depth and at 1,000 counts. The deposit's own detected-gene column is read first as a check that the AUROCs of section 2 recompute.

**Staged series.** PCC(x, degree) recomputed at 1,000 counts per cell on GSE106474, GSE112294 and GSE113074, on the scaffold genes and with the draw `controls.py thin` made. Its Spearman correlation with stage, beside the raw value on all cells and on the kept cells, and its rank correlation with the raw statistic.

## Predictions

- **T1.** In GSE117498h at 1,000 counts the AUROC of detected genes, stem cells against the bottom tier, is above 0.5.
- **T2.** In GSE125970 at 1,000 counts it stays above 0.5.
- **T3.** PCC(x, degree) at 1,000 counts rises with stage at a Spearman of +0.1 or more on all three staged series.

## What this does to the paper

- If T1 holds, section 2 says that the below-chance ordering of sorted blood progenitors is depth: at equal depth stem cells detect more genes than the bottom tier. The reported raw values stay, the equal-depth values are printed beside them, and the example is recast as depth turning the primitive round.
- If T1 fails, section 2 says that the reversal in blood survives depth equalisation, so there the gene count runs against the hierarchy for a reason other than depth.
- T2 is reported either way.
- If T3 holds, section 3 says the direction of the degree correlation on staged embryos does not depend on depth. If it fails on a series, the paper says on which, and the statement that signalling entropy, ORIGINS and CCAT run against their authors' polarity is restricted to raw depth there.
- If more than a third of the cells of the top or the bottom tier of an atlas are lost at 1,000 counts, the 500-count reading is reported beside it in the main text.

No score other than the gene count, its signature and PCC(x, degree) is recomputed at equal depth here. Signalling entropy, CCAT, ORIGINS and the other rows at equal depth need the scores rerun and their own plan.
