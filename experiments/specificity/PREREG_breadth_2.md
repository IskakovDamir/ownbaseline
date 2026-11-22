# Breadth, a second draw of the later zebrafish series and MCE on the mouse series: pre-registration

Written on 2026-11-22, after `PREREG_breadth.md` had run on both of its series (results at `9d3dcbe` and `a7aa129`) and with every number of its report in view. Same analyst and status as before: drafted by an AI agent (Claude, through Claude Code) on the author's instruction to strengthen the two added series, locked by a commit before any quantity below meets a stage label, unratified until the author reads it. No second agent reviewed it; it reuses the statistics and the code of `PREREG_breadth.md`.

## What prompted it

Two things in the report of `PREREG_breadth.md` rest on little. The later zebrafish series, GSE223922, was analysed on one draw of 220 cells per lane-by-time pool, and two of its outcomes turn on a margin of 0.002 and 0.009 against the controls' 97.5th percentile of 0.115 (CytoTRACE's joint increment, 0.117, and MCE's, 0.124). And MCE was not scored on the mouse series, so M5 and M9 were void.

## Part 1: a second draw of GSE223922

`prep_sur.py` with draw 2 takes, from each lane-by-time pool, 220 cells drawn uniformly without replacement among the cells the first draw did not take, under a new seed, and every remaining cell where fewer are left. The two pools the first draw exhausted (57 and 11 cells) drop out. No cell is in both draws. The draw reads the metadata table alone. Everything after it is `PREREG_breadth.md` unchanged, under the series key `GSE223922b` and its own seeds: primitives, CytoTRACE, SR, CCAT, MCE, the 1,000 random gene sets, the 1,000 degree permutations, thinning, the permutation floor, the test over pools inside lanes, depth.

The two draws share the embryos, the lanes and the sequencing, so the second is a check on sampling of cells and on the random gene sets, and it adds no independent pool.

**Predictions, R1 to R13.** Each of the thirteen rows comes out on the second draw as it did on the first. In the wording `breadth_analyse.py` prints: R1, more than half of the first 50 random gene sets are above the permutation floor on each primitive; R2, CytoTRACE's joint increment is above the controls' 97.5th percentile; R3, SR's is not; R4, CCAT's is not; R5, MCE's is; R6, CytoTRACE's residual trend is supported over the pools; R7 to R9, those of SR, CCAT and MCE are not; R10 and R11, the gene count does not fall with stage at raw depth or at 1,000 counts (Spearman above -0.1); R12, CytoTRACE falls with stage at raw depth (-0.1 or below); R13, it does not at 1,000 counts (above -0.1). R2 and R5 are the least secure, for the margins named above, and R13 next, the first draw having read -0.096.

**How it is read.** A statement about the later zebrafish series stays in the paper as a finding only where its row holds on both draws. Where the draws disagree the paper gives both values and says that the outcome depends on the draw.

## Part 2: MCE on GSE169210

MCE is computed on the mouse cells by `score_mce.py`, on the network and the input of SR, reading no label, and collected into the series' score file, whose hash changes for that reason alone. `breadth_analyse.py` is then rerun on the series. Its random streams are keyed by row, so every number already reported must return unchanged; the report states the largest difference. M5 and M9 are the rows registered at `b9c190f`, scored now with the other eleven outcomes known: MCE's joint increment is not above the controls' 97.5th percentile, and its negative residual trend is supported over the embryos.

## The lock

This file and the code are committed first. The hashes of the new blind files are added to `blind_manifest_breadth.json` by a second commit, before `breadth_analyse.py` reads a label of the second draw or reruns the mouse series. No further variant is run after the results are seen.
