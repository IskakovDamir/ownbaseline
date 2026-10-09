# GSE117498 gene-name split: what it does to the W4 rows and StemSC

`w4_gate2_run.load_c1` unions the eleven GSE117498 files by exact gene name. The seven sorted-population files name genes in HGNC form (HLA-A), the four broad-gate files in R `make.names` form (HLA.A), so every such gene becomes two columns, one zero in every broad-gate cell and the other zero in every sorted cell (`experiments/w5_new_rows/DISCOVERY.md` section 1.8). This note measures what that did to the results that went through `load_c1` before the split was found: the W4 rows on C1 (CytoTRACE proxy, SR, CCAT; `data/run_record/w4/c1_gse117498_results.json`, Figure 3) and StemSC on C1 (`data/run_record/stemsc/summary.json`). No paper text, README number, figure or tracked result file is changed by it.

## Answer

- **StemSC does not move.** No hyphenated atlas symbol maps to any of its 437 reference genes, so its 15,670 pairs read the same columns in both conditions; the score and the REO primitive are identical in every cell.
- **The depth facts and the AUROCs do not move at printed precision.** Per-population median detected genes (916 HSC, 1,404 GMP, ...) are identical, the 12,354 ranked cells are the same, and the gene-count primitive's AUROC is 0.378 either way (0.37837 split, 0.37831 merged).
- **Every tau_b value moves, and no tau_b verdict changes.** For CytoTRACE the marginal gap falls from 0.229 to 0.181, the marginal skill from +0.140 to +0.092, the conditional skill from 0.232 [0.218, 0.247] to 0.196 [0.182, 0.210], and the score's AUROC from 0.272 to 0.269. Those four printed numbers of the manuscript's section 2 change; the argument they carry does not (the table below goes statement by statement). SR and CCAT move more (SR marginal gap 0.383 to 0.173; SR's primitive goes from -0.103 to +0.008), but no published sentence quotes them.
- **Under weighted tau two verdicts swap**: SR from SIGN-FLIPPED SCAFFOLD to inconclusive, CCAT from inconclusive to SIGN-FLIPPED SCAFFOLD. Weighted tau is the kernel of the manuscript-era copy of the W4 run, not of the run record or the current manuscript, and neither verdict is quoted in the paper or a tracked file. The weighted CytoTRACE marginal gap, 0.141, is quoted in the `own_baseline/conditional_skill.py` docstring and becomes 0.157.

## The split

From the files' gene names alone (`measure_split.py facts`):

| | sorted files (7) | broad-gate files (4) |
|---|---|---|
| genes per file | 25,464 | 24,719 |
| with a hyphen | 4,355 | 0 |
| with a dot | 3,862 | 4,916 |

The union is 29,601 names (the run record's `n_genes_loaded`), 20,582 of them shared. All 4,137 broad-only names are the dotted twins of a hyphenated sorted name, which is the 4,137 pairs `prepare_inputs.ds_gse117498h` merges; no broad-only name is left unpaired. The other 745 sorted-only names (218 of them hyphenated) are not listed in the broad-gate files under any spelling (see "What the merge does not fix").

## How it was measured

"split" is `load_c1` as the published runs used it. "merged" is `prepare_inputs.ds_gse117498h()` called as it is: its merged matrix is taken at the moment it enters `w4_gate2_run.preprocess_and_filter`, so the merge is that function's own rule and not a copy. Both conditions then go through `w4_gate2_run.run_atlas` and `stemsc_run.run_atlas` unchanged, W4 at both kernels and StemSC at weighted tau, the kernel of its record. Seed 42, 1,000 bootstrap resamples, as in the originals.

Checks before any merged number was read:

| check | result |
|---|---|
| split, tau_b, against `data/run_record/w4/c1_gse117498_results.json` | every field equal (max abs difference 0.0; `elapsed_seconds` skipped) |
| split, weighted tau, against the original run's weighted-tau copy (not in the run record; SHA256 `6e00a41857f024deeb64dfb77f7011b07f1809b82e4e10743504477e272ceb88`) | every field equal (0.0) |
| split StemSC against `data/run_record/stemsc/summary.json` | every field equal (0.0) |
| merged gene list after the 10-cell filter against `$OWNBASELINE_DATA_ROOT/w5_new_rows/inputs/GSE117498h/genes.txt` | identical, 18,819 genes |
| cells, population labels and ordinal, split against merged | identical |

## W4 rows, tau_b (the run record's and the manuscript's kernel)

12,354 ranked cells in both conditions. Brackets are 95% bootstrap intervals; AUROC is the top tier (HSC) against the bottom tier (GMP, MEP, PreB/NK).

| score | quantity | split (published) | merged | change |
|---|---|---|---|---|
| CytoTRACE proxy | marginal skill | +0.1400 [+0.1253, +0.1543] | +0.0915 [+0.0770, +0.1061] | -0.0485 |
| | primitive (gene count) | -0.0889 [-0.1010, -0.0776] | -0.0891 [-0.1011, -0.0778] | -0.0002 |
| | marginal gap | +0.2289 [+0.2165, +0.2418] | +0.1806 [+0.1691, +0.1927] | -0.0483 |
| | conditional skill | +0.2316 [+0.2175, +0.2466] | +0.1955 [+0.1815, +0.2100] | -0.0361 |
| | AUROC score / primitive | 0.2721 / 0.3784 | 0.2691 / 0.3783 | -0.0030 / -0.0001 |
| | verdict | SCORE-BEATS-PRIMITIVE | SCORE-BEATS-PRIMITIVE | same |
| SR | marginal skill | +0.2798 [+0.2672, +0.2938] | +0.1812 [+0.1687, +0.1946] | -0.0986 |
| | primitive (x . degree) | -0.1029 [-0.1173, -0.0880] | +0.0080 [-0.0066, +0.0227] | +0.1109 |
| | marginal gap | +0.3827 [+0.3565, +0.4099] | +0.1733 [+0.1521, +0.1979] | -0.2094 |
| | conditional skill | +0.2541 [+0.2407, +0.2674] | +0.1814 [+0.1679, +0.1949] | -0.0727 |
| | AUROC score / primitive | 0.4560 / 0.7948 | 0.4560 / 0.7947 | +0.0000 / -0.0000 |
| | verdict | SCORE-BEATS-PRIMITIVE | SCORE-BEATS-PRIMITIVE | same |
| CCAT | marginal skill | -0.0454 [-0.0576, -0.0337] | -0.0506 [-0.0630, -0.0387] | -0.0052 |
| | primitive (Pearson(x, degree)) | +0.2471 [+0.2334, +0.2606] | +0.1668 [+0.1533, +0.1806] | -0.0803 |
| | marginal gap | -0.2925 [-0.3088, -0.2764] | -0.2173 [-0.2319, -0.2022] | +0.0751 |
| | conditional skill | -0.1305 [-0.1422, -0.1193] | -0.1227 [-0.1349, -0.1113] | +0.0078 |
| | AUROC score / primitive | 0.5229 / 0.4414 | 0.5230 / 0.4414 | +0.0001 / -0.0000 |
| | verdict | SIGN-FLIPPED SCAFFOLD | SIGN-FLIPPED SCAFFOLD | same |

## W4 rows, weighted tau (the manuscript-era copy)

| score | quantity | split | merged | change |
|---|---|---|---|---|
| CytoTRACE proxy | marginal skill | +0.1841 [+0.1567, +0.2023] | +0.2001 [+0.1731, +0.2184] | +0.0160 |
| | primitive | +0.0430 [-0.0071, +0.0980] | +0.0427 [-0.0082, +0.0978] | -0.0003 |
| | marginal gap | +0.1411 [+0.0710, +0.1923] | +0.1574 [+0.0887, +0.2086] | +0.0163 |
| | conditional skill | +0.1904 [+0.1382, +0.2213] | +0.2202 [+0.1734, +0.2495] | +0.0298 |
| | verdict | SCORE-BEATS-PRIMITIVE | SCORE-BEATS-PRIMITIVE | same |
| SR | marginal skill | +0.2198 [+0.1974, +0.2381] | +0.3228 [+0.3017, +0.3397] | +0.1030 |
| | primitive | +0.3760 [+0.2528, +0.4568] | +0.3986 [+0.2752, +0.4792] | +0.0227 |
| | marginal gap | -0.1561 [-0.2370, -0.0310] | -0.0758 [-0.1554, +0.0539] | +0.0803 |
| | conditional skill | +0.3507 [+0.2886, +0.3897] | +0.2838 [+0.2361, +0.3482] | -0.0669 |
| | verdict | SIGN-FLIPPED SCAFFOLD | inconclusive | **changed** |
| CCAT | marginal skill | +0.2217 [+0.1521, +0.2556] | +0.1605 [+0.0751, +0.2128] | -0.0612 |
| | primitive | +0.2852 [+0.2539, +0.3065] | +0.3399 [+0.3124, +0.3589] | +0.0547 |
| | marginal gap | -0.0635 [-0.1361, -0.0158] | -0.1793 [-0.2677, -0.1177] | -0.1159 |
| | conditional skill | +0.1468 [+0.0530, +0.1958] | +0.1678 [+0.0698, +0.2202] | +0.0211 |
| | verdict | inconclusive | SIGN-FLIPPED SCAFFOLD | **changed** |

The AUROCs are kernel-free and equal those in the tau_b table. CytoTRACE's conditional skill moves in opposite directions under the two kernels (tau_b -0.036, weighted +0.030).

## StemSC (weighted tau, the record's kernel)

| quantity | split (published) | merged |
|---|---|---|
| marginal skill | +0.2736 [+0.2015, +0.3283] | identical |
| REO primitive | +0.0920 [+0.0134, +0.1405] | identical |
| marginal gap | +0.1816 [+0.0889, +0.2751] | identical |
| conditional skill | +0.3344 [+0.2664, +0.3765] | identical |
| AUROC score / primitive | 0.5006 / 0.3510 | identical |
| verdict | SCORE-BEATS-PRIMITIVE | SCORE-BEATS-PRIMITIVE |
| reference pairs present | 15,670 of 16,893 | 15,670 |

Identical per cell, not only in the summary. The merge does change which atlas columns map to an Entrez ID (17,077 of 19,517 filtered genes split, 17,238 of 18,819 merged), but none of the changed ones is a reference gene. The reference files are not in the repository; the run used `ref_pairs.tsv` (SHA256 `e1767adc92e42e519c6c95c0343a2d18ef9203f122f154ee6aad8ab58399422e`) and NCBI `Homo_sapiens.gene_info.gz` (SHA256 `7f63e0bac7ec3d75ad213a655da90e5ce33d8944a29bfb207e5b20447ca4a1fe`), the files that reproduce the record above.

## Published statements, old against new

Manuscript means section 2 ("A primitive that orders below chance") and the Figure 3 caption. Kernel tau_b unless stated.

| statement | where | published | merged | at printed precision |
|---|---|---|---|---|
| median detected genes: 916 HSC, 1,045 MPP, 1,359 MLP, 992 PreB/NK, 882 MEP, 1,063 CMP, 1,404 GMP | manuscript, Figure 3 bars, README, `own_baseline.py` docstring | as stated | identical (max abs difference 0.0) | unchanged |
| 12,354 ranked cells | manuscript, Figure 3 titles | 12,354 | 12,354 | unchanged |
| gene-count AUROC 0.378 | manuscript, Figure 3, README, `own_baseline.py` and `conditional_skill.py` docstrings | 0.3784 | 0.3783 | unchanged |
| gene-count tau_b -0.089 | manuscript | -0.0889 | -0.0891 | unchanged |
| CytoTRACE marginal skill +0.140 | manuscript | +0.1400 | +0.0915 | **changes to +0.092** |
| marginal gap 0.229 | manuscript; Figure 3 note ("CT marginal delta = +0.229") | +0.2289 | +0.1806 | **changes to 0.181** |
| conditional skill 0.232, interval 0.218 to 0.247 | manuscript | +0.2316 [+0.2175, +0.2466] | +0.1955 [+0.1815, +0.2100] | **changes to 0.196, 0.182 to 0.210** |
| it clears every tau_b floor measured, the largest 0.0804 | manuscript | yes | yes, lower bound 0.1815 | unchanged |
| CytoTRACE AUROC 0.272 | manuscript, Figure 3 caption | 0.2721 | 0.2691 | **changes to 0.269** |
| direction check: score and primitive both below 0.5 | manuscript | 0.272, 0.378 | 0.269, 0.378 | unchanged |
| the marginal gap reads as a pass (Delta > 0.10, SCORE-BEATS-PRIMITIVE) | manuscript; README ("returns the exact false positive"); `cli.py`, `own_baseline.py` docstrings | 0.229 | 0.181 | unchanged |
| CytoTRACE "beats" gene count by 0.141 (weighted tau) | `own_baseline/conditional_skill.py` docstring; `figures/deprecated/make_paper_a_figures.py` | +0.1411 | +0.1574 | **changes to 0.157** |
| weighted conditional skill +0.190 [+0.138, +0.221], gap +0.141 (listed as manuscript values, not placeable) | `experiments/null_calibration/paper_values.py` | +0.1904 [+0.1382, +0.2213], +0.1411 | +0.2202 [+0.1734, +0.2495], +0.1574 | **changes** |
| StemSC measured on the sorted atlases; a row in Figure A1 and no bar | Figure A1, appendix | conditional skill +0.3344 | identical | unchanged |

The intestinal numbers (AUROC 0.798, 3,263 against 1,862 genes) come from `load_c2` and are not affected.

## Why the AUROCs hold and the taus move

Each cell comes from one file, and a file lists either the hyphenated or the dotted name of a pair, never both, so merging adds a zero to a value. Per-cell detected genes and library size before the gene filter cannot change, which is why the depth medians are exact.

What the split did change:

1. **The scaffold input of broad-gate cells.** 36 of the 12,952 STRING nodes in the split are paired HGNC names, among them all 13 MT- genes and 18 HLA genes, and in every broad-gate cell they read zero. The 42 paired genes that are scaffold nodes once merged carry a median 8.0% of a CD34+/CD164+ cell's counts (8.4% across all broad-gate cells, 14.9% in sorted cells), almost all of it in the 13 MT- genes (7.7%, 8.1%, 14.5%), and the library size those counts belong to stays in the denominator. The tier-4 gate's median SR primitive goes from 349,670 to 373,184 when they come back (MPP, its tier-4 neighbour: 422,029 to 421,947); its median CCAT primitive goes from 0.338 to 0.321. The merged scaffold has 12,958 nodes (HLA-G, ERVFRD-1 and four NKX genes join).
2. **CytoTRACE's gene set.** The top 200 genes correlated with gene count are chosen over all 21,412 cells. The split set held four broad-only dotted columns (MT.CO2, MT.ATP8, RP11.620J15.3, LRRC75A.AS1), nonzero only in broad-gate cells. The merged set shares 198 of 200 with it once names are matched (out: G3BP1, RP11-620J15.3; in: MT-ATP6, MT-CO3).
3. **The 10-cell filter.** It counts cells over both file groups, so a gene the split filtered out on one side can come back merged. Gene count rises in 4,551 cells (2,252 of 6,011 sorted, 2,299 of 15,401 broad-gate), by at most 23 genes, and falls in none.

The AUROC compares HSC (1,282 cells) with tier 2 (GMP, MEP, PreB/NK; 2,815 cells), all sorted, where only items 2 and 3 apply. Tau runs over all four tiers, and 6,343 of the 6,558 tier-4 cells are the broad CD34+/CD164+ gate, which item 1 altered. A primitive shifted in one tier's cells moves tau and leaves this AUROC alone, which is the table above.

## What the merge does not fix

745 sorted-file genes are not listed in the broad-gate files under any name, so they read zero in every broad-gate cell after the merge too. 322 of them pass the 10-cell filter and 28 are scaffold nodes; none is in either CytoTRACE gene set. In sorted cells, where both sets are measured, they carry a median 0.27% of counts against 15.8% for the 4,137 paired genes (0.02% against 14.9% within the scaffold), so what remains is small next to what the merge restores. Whether these genes are absent from the broad-gate files because the files were made against a different annotation is not established here.

## Not measured here

Three more scripts load C1 through `load_c1`: `experiments/track4/code/gate2b_origins_c1c2.py` (ORIGINS, 3-cell gene filter), `experiments/track4/code/gate2c_resolve.py` and `experiments/cytotrace_v1/ct_full_run.py`. None of them feeds a figure, the run record or a sentence of the manuscript, so they are listed and not rerun. The w5 rows already carry their own sensitivity lines (GSE117498h).

## Commands

From the repository root, with `$OWNBASELINE_DATA_ROOT` holding `w4/data/c1_gse117498/` (GEO GSE117498_RAW.tar, unpacked) and `w4/scaffolds/human_string_v12_thr700_lcc.npz`. The CytoTRACE proxy and StemSC each hold several dense float64 copies of the 21,412 x 19,517 matrix at once (3.3 GB each), which is more than a 16 GB machine holds without swapping, so run the `w4` and `stemsc` steps one at a time.

```
python3 experiments/w4_name_split/measure_split.py facts
python3 experiments/w4_name_split/measure_split.py w4 split
python3 experiments/w4_name_split/measure_split.py w4 merged
python3 experiments/w4_name_split/measure_split.py stemsc split  --stemsc-ref DIR
python3 experiments/w4_name_split/measure_split.py stemsc merged --stemsc-ref DIR
python3 experiments/w4_name_split/measure_split.py compare --w4-weighted FILE
```

`DIR` holds StemSC's `ref_pairs.tsv` and `mapping/Homo_sapiens.gene_info.gz`; `FILE` is the original weighted-tau W4 C1 JSON, needed only for that reproduction check. Per-condition JSON and per-cell arrays go to `$OWNBASELINE_DATA_ROOT/w4_name_split/`; `compare` writes `split_vs_merged.json` here, the source of every number above.

Python 3.14.3, numpy 2.4.3, scipy 1.18.0, scikit-learn 1.9.0, pandas 2.3.3, anndata 0.13.2.
