# Signalling entropy, CCAT and MCE at equal depth: pre-registration

Written on 2026-11-22, after `PREREG.md` (commit `833acba`) and `PREREG_depth_2.md` (commit `4f8304b`) had run and their results were committed (`92dd551`), and before any score below met a label at equal depth. The author asked for this analysis after reading those results. Same status as the two earlier plans: drafted and run by an AI agent, locked by a local unpushed commit.

## What prompted it

The first two plans equalised depth for the gene count, its signature, CytoTRACE and PCC(x, degree), and found that CytoTRACE's direction on GSE112294 follows the number of counts per cell while the direction of PCC(x, degree) does not. The paper's Limitations then say that no other score was recomputed at equal depth. Three of the audited scores have Python ports in this repository and can be: signalling entropy (SR), CCAT and MCE.

## What is already known

Everything in the preprint and in the results of the two earlier plans, including PCC(x, degree) at 1,000 counts and its correlation with stage (+0.876, +0.862, +0.675). What has never been computed: SR, CCAT or MCE on a thinned matrix.

`equal_depth_scores.py blind` computed them before this file was committed and read no label. It rebuilds the interaction network of each series from the deposit's scaffold and checks the score code at raw depth against the deposit: on 10,000 cells per series for SR and CCAT and 300 cells for MCE. The thinned matrix is the draw `controls.py thin` made, every cell with at least 1,000 counts drawn down to exactly 1,000. `blind_manifest_depth_3.json` holds the hashes and the fidelity figures.

## What is computed

On GSE106474, GSE112294 and GSE113074, SR and CCAT at 1,000 counts; on the two zebrafish series, MCE as well, on signalling entropy's input as in the audit. For each score: its Spearman correlation with stage at 1,000 counts beside the raw value; its conditional skill, Eqs. (1) to (3) with alignment, on PCC(x, degree) computed at the same depth; and the permutation floor of that value, 200 draws within strata of 100 cells of the thinned primitive. A row clears when its value is above that floor.

No control family was thinned, so no comparison with random gene sets is made at equal depth.

## Predictions

- **V1.** SR at 1,000 counts rises with stage, at a Spearman of +0.1 or more, on all three series, against its authors' polarity, as at raw depth.
- **V2.** SR's conditional skill on PCC(x, degree), both at 1,000 counts, is above its permutation floor on all three series.
- **V3.** CCAT's conditional skill on the same primitive is negative on all three series.
- **V4.** MCE's is negative on both zebrafish series.

## What this does to the paper

The values go into the appendix on depth and the outcome of each prediction into its deviations table. If V1 and V2 hold, section 5 says that signalling entropy's direction and its skill beyond the degree correlation do not depend on depth on these series. If either fails on a series, the paper names the series and restricts the corresponding statement to raw depth there. V3 and V4 are reported either way. The sentence in Limitations that no other score was recomputed at equal depth is narrowed to the scores still not recomputed: ORIGINS, NCG, dpath, SLICE and StemFinder.
