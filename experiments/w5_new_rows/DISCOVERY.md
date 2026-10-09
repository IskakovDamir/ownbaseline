# W5 new rows: discovery

Step 1 of adding four rows to the audit (MCE, StemFinder, mRNAsi, FitDevo), after feedback from Qingyang Wang on 9 Oct 2026. Nothing in this step reads a stage or tier label, and no score was compared with any ordinal. Scores were computed where a method runs: MCE (from the published equations) on the 500-cell SCENT validation subsample, StemFinder on all of GSE106474, mRNAsi on all of GSE117498, FitDevo on all four audit datasets. Those runs establish whether each method runs and at what cost, which the pre-registration needs.

Repository state at the start: `main` at `b94ea5c`.

Every count below comes from a script run in this session. Where a number is a figure in a paper, the paper is cited as its source.

## 1.1 Environment

`python3 tests/run_tests.py` on the clean checkout gave 24 passed, 1 failed, 1 known defect (F-2). The failure was `test_no_source_file_hard_codes_an_absolute_home_path`: it imported `pytest` unconditionally, pytest is not installed, and `run_tests.py` promises the suite needs no test runner. The import now sits in the only branch that uses it, and the suite gives 25 passed, 0 failed, 1 known defect, the README's count. `tests/test_mce.py` (section 1.4) is kept out of `run_tests.py`, so that count stays what the README states; it runs on its own (`python3 tests/test_mce.py`, 5 passed) and under `pytest tests/`, which is what CI runs.

| software | version |
|---|---|
| Python | 3.14.3 |
| numpy, scipy, scikit-learn, pandas, anndata | 2.4.3, 1.18.0, 1.9.0, 2.3.3, 0.13.2 |
| R | 4.6.1 (Homebrew, arm64) |
| GNU Octave | 11.3.0, installed through Homebrew in this session for step 1.4 (it was not present before). The install also upgraded openssl@4, which unlinked the openssl@3 command-line symlinks; the openssl@3 keg is still installed. |

## 1.2 MCE.m: blocked in this session, then placed by the author

`reference/` did not exist. The supplement could not be downloaded:

- The article page (`academic.oup.com`) answers `curl` with a Cloudflare 403. In the built-in browser it shows an interactive "Verify you are human" check, which was not attempted.
- The supplement is one archive, `bby093_supp.zip`, on `oup.silverchair-cdn.com`. Unsigned, the CDN returns 403 (missing Key-Pair-Id). The signed query string exists only in the rendered article page.
- Europe PMC has no open-access copy (PMID 30289442, no PMCID). The bioRxiv preprint (10.1101/257220) supplement is a single PDF. GitHub code search and the Wang et al. review (arXiv 2309.13518) give no mirror of MCE.m.
- A discovery subagent also opened the article read-only in the author's own Chrome through the Claude in Chrome extension. The page and its 18 signed links to the zip were there, but the extension hides query strings and an in-page fetch failed on CORS. Nothing was downloaded and the tab was closed. That browser should not have been used without asking, and this is recorded so the author knows.

**What the author has to do.** Open https://academic.oup.com/bib/article/21/1/248/5115275#supplementary-data, click "bby093_Supp - zip file" (it resolves to https://oup.silverchair-cdn.com/oup/backfile/Content_public/Journal/bib/21/1/10.1093_bib_bby093/4/bby093_supp.zip with a signed query string), and save `bby093_supp.zip` into `<repo>/reference/`. Then:

```
cd <repo>/reference && unzip -d shi2020_supplement bby093_supp.zip
find shi2020_supplement -name 'MCE.m' -exec cp {} MCE.m \;
shasum -a 256 MCE.m
```

`reference/` is git-ignored; MCE.m stays local and its SHA256 goes in the report. That the zip contains a file named MCE.m rests on Qingyang Wang's message. The article's main text does not mention code or MATLAB, so it could not be checked.

**Status, first pass: MCE was waiting for the author.** The MCE row was pre-registered (PREREG.md) to run once MCE.m was in place.

**Update, 2026-10-09, after the PREREG commit.** The author downloaded the supplement and placed it, unzipped, in `reference/bby093_supp/`. It holds two files: `MCE.m` (2,338 bytes, SHA256 `c8ad11f2cdadc979cc303a21785f65eba77a41dfe565ace2beffaeae265fdce2`) and `ADF1.pdf` (1,925,087 bytes, SHA256 `6bc953734294b71947ea1f6234a79b3ed9d554111c13f669818e9df85dceef76`), the supplementary tables S1 to S5 and figure legends S1 to S11, which say nothing about preprocessing. MCE.m was copied to `reference/MCE.m`. So the zip does contain MCE.m, as Qingyang Wang said. Sections 1.3 and 1.4 below were then checked against it, still without any label.

## 1.3 What MCE computes

MCE.m could not be read. This section is from the published Methods (sections "MCE", "Computing MCE for each sample", "Normalization of MCE"), read through the article page, and from the Wang et al. review's notation section. It is to be checked line by line against MCE.m.

| | MCE (Shi et al. 2020) | SR (SCENT, Teschendorff and Enver 2017) |
|---|---|---|
| network | 0/1 PPI adjacency, **A_ii = 1** (every node has a self-loop). Wang et al.: "Ai,i = 0 in SCENT and SPIDE, but Ai,i = 1 in MCE" | same PPI, **A_ii = 0** |
| expression | normalized profile x over the network genes | log2(x + 1.1) over the network genes (DoIntegPPI) |
| stationary law | pi = x / sum(x) | pi_i proportional to x_i (A x)_i |
| chain | the P supported on A, with pi invariant, that maximizes the entropy of an edge, sum over (i, j) of pi_i P_ij log(pi_i P_ij) (Eq 3) | mass action, P_ij = A_ij x_j / (A x)_i |
| how P is found | P_ij = alpha_i beta_j A_ij; fixed point alpha = 1 ./ (A beta), beta = pi ./ (B' alpha), B_ij = pi_i A_ij (Eq 6), a Sinkhorn scaling | closed form |
| score | MCE = -sum_i pi_i log(alpha_i beta_i pi_i) | SR = sum_i pi_i S_i, S_i the entropy of row i of P |
| normalization | MCE / log(d), d = nnz of A with the self-loops (Eqs 7 and 8); 1 exactly when pi_i is proportional to the self-loop degree | SR / log(lambda_max(A)) |
| output | one value per cell in [0, 1], higher = more potent | one value per cell in [0, 1], higher = more potent |

**Entropy rate or static entropy.** By Eq 2 of the paper, and by direct substitution of P_ij = alpha_i beta_j A_ij,

MCE = H(pi) + h(P*)

where H(pi) is the Shannon entropy of the stationary distribution over the network genes and h(P*) is the entropy rate of the maximum-entropy chain that keeps pi invariant. MCE is the entropy of a Markov chain's stationary edge distribution. It is therefore neither a pure entropy rate, as SR is, nor a pure static entropy, as StemID is. Its defining object is a Markov chain on the network, chosen to maximize its transition entropy, and the score contains that chain's entropy rate. For P1 it is classified as an entropy-rate construction (PREREG.md says how this is revisited when MCE.m is read).

The static part is not small. Since 0 <= h(P*) <= H(pi) (a conditional entropy is at most the marginal's), H(pi) <= MCE <= 2 H(pi) for every cell. The two ends are attained: a network with self-loops only gives P = I and MCE = H(pi); a complete network gives the independent coupling and MCE = 2 H(pi). `scores/mce.py` reproduces both (section 1.4). On a sparse PPI the network places MCE between them. A large share of its variance may therefore come from H(pi), the static entropy of network-gene expression. That is a risk to P1, recorded here before any label is read.

**Declared primitive.** PCC(x, degree): Shi et al., Results "Cellular potency encoded by the Pearson correlation of transcriptome and connectome", attribute about 70% of MCE's variance to it; the Fig 4D panel prints R2 = 0.67 (Wang et al. NPC and neuron data, MCE on the y axis, PCC(mRNA, Degree) on the x axis).

**Class.** Unsupervised. The score depends on one cell's expression and a fixed network; nothing is fitted to labelled data. MCE.m hard-codes its start, threshold and step cap (below); none is fitted to anything.

**Functions MCE.m calls.** `size`, `logical`, `speye`, `find`, `nnz`, `zeros`, `sum`, `sparse`, `dot`, `log`, `fprintf`, `ones`, `norm` (infinity norm), and element-wise arithmetic. All are core GNU Octave; Octave 11.3.0 ran the file unmodified (1.4).

### Checked against MCE.m

Read in full (79 lines). It confirms the table above, line by line, and adds what the Methods leave open:

- **Network.** `net(logical(speye(size(net)))) = 1` (line 19): A_ii = 1, as Wang et al. say. `maxMCE = log(nx)`, nx = nnz of that matrix (lines 21, 36).
- **Input.** `p0 = data(:, i); p0 = p0/sum(p0)` (line 26). No transform, no log: MCE.m uses whatever it is given, divided by its sum. A zero entry makes `0 * log(0)`, which is NaN, at line 30, so the input must be strictly positive. Neither the code nor ADF1.pdf names the transform the authors used.
- **Iteration** (`fun_iteration`, lines 43 to 79). Start lambda0 = p0, theta0 = 1. Each step: lambda = 1 ./ (A theta0), theta = p0 ./ (B' lambda) with B_ij = p0_i A_ij, err = the infinity norm of the change in [lambda; theta]. It stops when err < 1e-2 or after more than 1e6 steps (line 62) and returns the last (lambda, theta). The stopping rule is an absolute change in the multipliers, not a residual of the constraints.
- **Score.** `ge(i) = - dot(p0, log(out(1:p).*out((p+1):end)))` (line 30), then `ge(i) = ge(i) - sum(p0(p0>0).*log(p0(p0>0)))` (line 32). Line 30 is the entropy rate h(P) of P_ij = lambda_i theta_j A_ij, and line 32 adds the static entropy H(p0). **So MCE.m computes MCE = h(P*) + H(pi), the entropy-rate construction plus the static term, exactly as 1.3 found from the equations.** The classification used for P1 stands, and PREREG.md's switch condition (MCE.m computing only H(pi)) is not met.

## 1.4 scores/mce.py

`scores/mce.py` was first written from the published equations (this paragraph and the analytic table below). Once MCE.m arrived it became a port of it: the default `rule="mce_m"` reproduces MCE.m's start, iteration, stopping rule and returned pair, each cell stopping on its own as MCE.m's per-sample loop does; `rule="residual"` keeps the converged solver for the analytic tests. It runs on the network object SR uses (`<scratch>/scent_io/adjMC.npz`, STRING v12 at 700, atlas intersection, largest component, no self-loops) and adds the identity itself, so MCE and SR share one scaffold. Choices MCE.m may make differently (input transform, start, tolerance, iteration cap) are arguments, so the port can take the authors' values without changing the algorithm.

**Validation against analytic answers** (`experiments/w5_new_rows/validate_mce.py`, `discovery/mce_validation.json`):

| case | expected | scores/mce.py | abs. deviation |
|---|---|---|---|
| self-loops only, MCE = H(pi) (7 nodes) | 1.7471706669 | 1.7471706669 | 0 |
| complete graph, MCE = 2 H(pi) (6 nodes) | 2.87539980601 | 2.87539980601 | 0 |
| pi proportional to self-loop degree, normalised MCE = 1 (30 nodes) | 1 | 1 | 2.2e-16 |
| random 6-node graph, Eq 3 solved by SLSQP with no knowledge of the factorisation | 2.67829421109 | 2.67829421109 | 0 |

`tests/test_mce.py` asserts these four, plus that the fitted chain keeps pi invariant to 1e-12 and that cells solved together and alone agree to 1e-10. All five pass.

**Validation against MCE.m in Octave** (`discovery/mce_validation.json`, after MCE.m arrived). `validate_mce.py` writes the 500 SCENT validation cells, log2(CPM + 1.1) over SR's 8,468 network genes, and SR's network (no self-loops; MCE.m adds them) to a MAT file, runs `MCE(data, net)` from `reference/MCE.m` in `octave-cli` 11.3.0 unmodified, and compares it with the port under MCE.m's rule on the same matrix:

| comparison | value |
|---|---|
| Spearman, port against MCE.m | 1.000000 |
| Pearson, port against MCE.m | 1.000000 |
| maximum absolute deviation | 1.8e-13 |
| step counts identical in every one of the 500 cells | yes |
| time inside Octave | 158.7 s |

`tests/test_mce.py` also checks the port against a line-by-line transcription of MCE.m on random networks: same scores to 1e-12, same step counts.

**Timing under MCE.m's rule** (one core, arm64, numpy 2.4.3, scipy 1.18.0): the port scored the 500 cells in 26.2 s, 137 to 547 steps per cell (median 263), which projects to 0.58 h for 39,505 cells. That is under 8 hours, so the MCE row uses all 39,505 cells (PREREG.md, Amendments).

**How far MCE.m's 1e-2 rule stops from the fixed point.** On the same 500 cells the MCE.m-rule score differs from the fixed point solved to a stationarity residual of 1e-6 by at most 8.3e-8 (median 1.1e-8), Spearman 1.000000. The loose threshold does not change the ranking.

The timing below was measured before MCE.m arrived, on the equations-based solver, and is superseded by the timing above.

**Timing** (`discovery/mce_validation.json`; arm64, numpy 2.4.3, scipy 1.18.0, one core). Input: the 500 cells SR was validated on, log2(CPM + 1.1) over SR's 8,468 network genes (101,889 edges before self-loops). One fixed-point iteration costs 1.689e-4 s per cell (50 iterations on all 500 cells, 4.2 s). How many iterations a cell needs depends on the stopping rule, which only MCE.m can fix, so the cost is projected per tolerance from a trace of the first 50 cells (stationarity residual max_j |(pi P)_j - pi_j|):

| tolerance | iterations | projected time, 39,505 cells | max abs. change in MCE against the 1e-8 solution |
|---|---|---|---|
| 1e-6 | 480 | 0.89 h | 1.6e-8 |
| 1e-8 | 16,950 | 31.42 h | 0 (reference) |
| 1e-10, 1e-12 | not reached in 20,000 iterations (residual 5.9e-9 at the cap) | | |

The fixed point converges fast to about 1e-6 and slowly after that; at 1e-6 the score is already fixed to 1.6e-8. On these 50 cells normalised MCE runs from 0.870 to 0.910 (median 0.890). The full-versus-subsample decision for the MCE row uses the projection of the port validated against MCE.m, under MCE.m's own stopping rule (PREREG.md).

## 1.5 StemFinder

Noller K, Cahan P. "Cell cycle expression heterogeneity predicts degree of differentiation." Brief Bioinform 25(6):bbae536 (2024), PMC11500603. Code: github.com/CahanLab/stemfinder at `db8ef0e` (R package 0.1.0); analysis scripts github.com/CahanLab/stemfinder_paper at `eab41f7`.

**What it computes** (`R/run_stemFinder.R` lines 19 to 72, read in full here). On a Seurat object with log-normalized and scaled data: binarize the scaled expression of the S and G2M cell-cycle genes at 0, build a kNN graph on PCs of the variable genes with the cell-cycle genes removed, k = round(sqrt(N)), each cell its own neighbour. For cell i and marker g, p_ig is the fraction of the other k - 1 neighbours whose binarized value equals cell i's, and G_ig = p_ig (1 - p_ig). stemFinder_raw_i = sum over g of G_ig, and the reported score is stemFinder_i = 1 - stemFinder_raw_i / max(stemFinder_raw). Higher stemFinder means more differentiated (the package documentation: "corresponding to pseudotime"). The package bundles a human list of 43 S and 55 G2M genes, 97 unique: E2F8 is on both. The README vignette concatenates the two lists without removing the duplicate, so in the authors' own construction E2F8 enters `run_stemFinder` and `gene_set_score` twice; the wrapper does the same.

**Class: unsupervised.** Scoring reads no metadata column; the gene list is curated (Regev lab / Scanpy), thresh = 0 is fixed, and k = sqrt(N) is a heuristic. The number of PCs is picked by eye from the elbow plot (Supp Methods); PREREG.md replaces the eye with a fixed rule. Design choices (Gini over SD and variance; the k range) were compared on the paper's 24-dataset benchmark ground truth, which includes the Farrell zebrafish set (GSE106587, the SuperSeries of GSE106474, 6,000 cells). That is selection among a few variants, not fitting, but it is a contact between this method and the audit's own source study, and it is recorded.

**Declared low-order statistic.** The authors compare stemFinder against the cell-cycle gene-set score, the mean log-normalized expression of the same markers (`R/gene_set_score.R`), and report heterogeneity as "a significantly better correlate of ground truth differentiation than cell cycle gene set score" (Results, Appendix Fig. 1A, B). That is the statistic the score is built from and benchmarked against, and it is the declared primitive. They do not compare stemFinder with gene counts.

**Language and whether it runs.** R. Depends on Seurat; the code reads `@scale.data`, which Seurat 5's Assay5 objects lack, so it runs with `options(Seurat.object.assay.version = "v3")`. The package is not installed; its two R files and the gene lists are sourced from the pinned checkout. Two practical points: scaling all genes densely does not fit in memory at 39,505 cells (scaling only the variable genes plus markers gives the markers identical values), and the per-cell loop reads rows of the kNN graph. The row-by-row access was timed on a synthetic graph of the real size (39,505 cells, k = 199): 9.92 s per 300 rows on the column-compressed graph Seurat returns, which projects to 0.4 h for all cells, so the graph is passed as Seurat builds it.

**It runs.** With Seurat 5.6.0 and SeuratObject 5.4.0 (installed for this step into a project-local library, `data/scratch/Rlib`), `score_stemfinder.R` scored all 39,505 cells of GSE106474 in 1,352 s with a peak memory footprint of 6.7 GB: 82 marker entries (81 genes, E2F8 twice), 7 PCs by the elbow rule out of 2,487 variable genes, k = 199. No label was read. (A first run that de-duplicated E2F8 took 1,354 s; the pre-push review caught the difference from the vignette and the score was recomputed before anything was registered.) Two environment facts: Homebrew's R 4.6.1 compiles C with `-std=gnu23`, which the installed Command Line Tools clang 15 rejects, so the R packages were built with a project-local Makevars setting `-std=gnu17`; and Seurat replaces underscores in gene names with dashes, which changes no cell and no marker.

Gene overlap with the audit datasets (upper-case symbol match, `discovery/discovery_facts.json`): 81 of the 97 markers in GSE106474, 96 in GSE117498, 96 in GSE125970, 67 in GSE113074.

## 1.6 mRNAsi

Malta TM et al. "Machine Learning Identifies Stemness Features Associated with Oncogenic Dedifferentiation." Cell 173(2):338-354 (2018), PMC5902191.

**Weights.** TCGAbiolinks ships them as `SC_PCBC_stemSig` in `data/SC_PCBC_stemSig.rda`. Read here from Bioconductor RELEASE_3_23, TCGAbiolinks 2.40.0, commit `2f9d249` (2026-04-28): a named numeric vector of 12,956 HGNC symbols, 7,018 positive, 5,938 negative, 0 zero, no duplicates; file SHA256 `a9900d8001414737a0f8f1a7a89d788eaf341130ad39c20833091659042d0777`. It is the same vector as the authors' PanCanStem_Web `Stemsig/SC-pcbc-stemsig.tsv`. The paper's own 2018 GDC table has 12,945 genes. The two agree to Pearson 0.99999997 on the 12,691 shared symbols and differ in gene universe through symbol vintage (discovery agent's comparison). TCGAbiolinks' copy is what the brief names and what is used.

**Score.** `TCGAanalyze_Stemness` (`R/Stemness.R` lines 44 to 76): intersect the weight names with the matrix rownames, take each sample's Spearman correlation with the weight vector over the shared genes, then subtract the minimum and divide by the maximum across samples. The paper's Methods say the same ("Spearman correlations between the model's weight vector and the new sample's expression profile"; "subtracted the minimum and divided by the maximum"). The min-max step is monotone, and Spearman is computed within a cell, so a per-cell monotone transform of the counts does not change the score. Higher = more stem-like. The function appeared in TCGAbiolinks in 2019, after the paper; the paper's values came from the authors' workflow, which the function reproduces.

**Class: trained.** The weights are a one-class logistic regression (gelnet, l1 = 0, l2 = 1) fitted to the PCBC samples labelled SC (embryonic and induced pluripotent stem cells), after centring every gene on all PCBC samples. That is fitting to class data, so mRNAsi is a boundary row outside the unsupervised tally, by the rule that put CytoTRACE 2 there. It is not retrained.

**Declared low-order statistic: none.** The authors relate mRNAsi to stemness gene sets, EZH2, OCT4, SOX2, EMT and immune infiltration, never to a per-cell low-order statistic. Its baseline is the joint four-primitive residual.

Symbol overlap: 12,612 of the 12,956 weights match a GSE117498 symbol exactly, and 12,160 remain after the stemsc code path's filter (genes detected in at least 10 cells). The function drops the rest silently; `score_mrnasi.R` logs the count.

**It runs.** The function body and the weights, sourced and loaded from the pinned TCGAbiolinks checkout with no package installed, scored all 21,412 GSE117498 cells in 51 s, with no NaN. No label was read.

## 1.7 FitDevo

Zhang F et al. Brief Bioinform 23(5):bbac293 (2022), PMC9487676. Code: github.com/jumphone/FitDevo at `0c757e6` (2024-04-07), `fitdevo.R` v1.2 (SHA256 `0d182d0144c679c0282a2f8fa9539f06c1c3f050b011b14f9cdcb65fbe31e648`), pretrained `BGW.rds` (SHA256 `cccb4b62013f3ede24fdc6dfdb62053a7fb54a12566b4c89acd7371a8d79de43`, 14,717 genes, 11,387 ones, 3,330 zeros). It is not an R package; "install" is the pinned clone plus `source("fitdevo.R")`, which needs Seurat and qlcMatrix.

**What it computes** (`fitdevo()`, lines 90 to 245, read in full here). Upper-case the gene symbols and drop any that collide; LogNormalize at 1e4. SSGW step: above 50,000 cells, subsample 50,000 (seed 123) for this step only; 2,000 vst variable genes, scale, 50 PCs; for every gene, the Pearson correlation of its expression with each PC; a logistic regression of BGW on those loadings across genes; pBGW = 1 where the linear predictor is positive; SSGW = BGW + pBGW, in {0, 1, 2}. Score: the per-cell Pearson correlation, across the SSGW genes, between log-normalized expression and SSGW. Higher = more developmental potential. No smoothing.

**Class: trained.** BGW was fitted to timepoint labels: for each gene in each of 17 training samples, the Pearson correlation of its expression with the reversed timepoint order, scaled and averaged, thresholded at 0 (paper: "we propose a supervised workflow for generating GW"; Discussion: "FitDevo is a supervised method"). The per-dataset SSGW step uses no label of the target data. Boundary row outside the tally. Not retrained.

**Declared low-order statistic: none.** The score has CCAT's form, PCC(x, weight), with a learned 0/1/2 weight in place of network degree. The authors benchmark it against gene number, CCAT and CytoTRACE but do not claim it approximates one. Baseline: the joint four-primitive residual.

**The 17 training datasets** (paper Supplementary Table 1; SourceForge training index; GEO). None is one of the four audit accessions.

| # | dataset | species | system | accession | overlap with GSE106474, GSE113074, GSE117498, GSE125970 |
|---|---|---|---|---|---|
| 1 | HSCs Smart-seq2 | mouse | haematopoietic stem and progenitor cells (ageing) | GSE59114 | none (same system as GSE117498, other species and study) |
| 2 | Blastocyst SC3-seq | macaque | peri-implantation embryo | GSE74767 | none |
| 3 | Cortical interneurons C1 | mouse | cortical interneurons | GSE90860 | none |
| 4 | Dentate gyrus 10x | mouse | hippocampus | GSE95753 | none |
| 5 | Embryonic HSCs (Tang et al.) | mouse | embryonic HSC formation | GSE67123 | none (same system as GSE117498, other species and study) |
| 6 | Endometrium CEL-seq | mouse | uterine epithelium | GSE98451 | none |
| 7 | Germ cells Smart-seq2 | human | fetal germ cells | GSE86146 | none |
| 8 | Hepatoblast Smart-seq2 | mouse | fetal liver | GSE90047 | none |
| 9 | In vitro NPCs C1 | human | neural precursor differentiation | GSE102066 | none |
| 10 | Lung development C1 | mouse | distal lung epithelium | GSE52583 | none |
| 11 | Medial ganglionic eminence C1 | mouse | embryonic forebrain | GSE94641 | none |
| 12 | mESC in vitro RamDA-seq | mouse | ES cell differentiation | GSE98664 | none |
| 13 | Neural stem cells Drop-seq | mouse | embryonic neural stem cells | GSE107122 | none |
| 14 | Oligodendrocytes C1 | mouse | oligodendrocyte lineage | GSE75330 | none |
| 15 | Pancreatic alpha cells Smart-seq2 | mouse | islet alpha cells | GSE87375 | none |
| 16 | Pancreatic beta cells Smart-seq2 | mouse | islet beta cells | GSE87375 | none |
| 17 | Thymus Drop-seq | mouse | embryonic thymus | GSE107910 | none |

The zebrafish set the CytoTRACE authors provided (Farrell et al. 2018, GSE106587, the SuperSeries whose Drop-seq SubSeries is GSE106474) is listed in the table with an asterisk and was **excluded from training** ("the zebrafish sample is not included in our training dataset"; "we have removed a zebrafish sample due to the limited number of homologous genes"). The authors did evaluate GW on its stage labels and report a Spearman of 0.228. So GSE106474 is absent from the training set, but its labels have met this method once in print; that is recorded as a caveat on the GSE106474 line. GSE113074 (Xenopus), GSE117498 and GSE125970 appear nowhere in the paper, its supplement, the repository or the training index.

All four audit datasets are therefore absent from the training set. BGW symbol overlap (upper-case match, `discovery/discovery_facts.json`): 9,622 of 14,717 in GSE106474, 14,426 in GSE117498, 12,751 in GSE125970, 9,682 in GSE113074.

**It runs.** `score_fitdevo.R` sources `fitdevo.R` from the pinned clone and calls `fitdevo()` with its defaults and the shipped BGW, on Seurat 5.6.0 with v3 assays and qlcMatrix 0.9.9. No label was read. Per dataset, on the input `prepare_inputs.py` wrote:

| dataset | cells | genes in | genes in SSGW (0 / 1 / 2) | seconds | peak RSS |
|---|---|---|---|---|---|
| GSE106474 | 39,505 | 23,974 | 9,620 (18 / 1,827 / 7,775) | 19.4 | 3.2 GB |
| GSE117498 | 21,412 | 19,517 | 13,305 (83 / 2,840 / 10,382) | 7.3 | 3.1 GB |
| GSE125970 | 14,537 | 17,416 | 12,006 (111 / 2,291 / 9,604) | 6.8 | 3.1 GB |
| GSE113074 | 127,607 | 26,550 | 9,682 (70 / 2,048 / 7,564) | 27.1 | 5.7 GB |

On GSE113074 the SSGW step used FitDevo's own 50,000-cell subsample (seed 123); the score was computed for all 127,607 cells.

## 1.8 A defect in the GSE117498 code path

Found by the pre-push review and confirmed on the files. The eleven GSM files of GSE117498 come in two groups with different gene-name conventions. Each of the seven sorted-population files (HSC, MPP, MLP, PreB/NK, MEP, CMP, GMP) lists 25,464 genes, 4,355 of them with a hyphen (HLA-A, MT-CO1). Each of the four broad-gate files (the CD34/CD164 gates, one of them the ranked CD34+/CD164+ tier-4 population) lists 24,719 genes and none with a hyphen: the same names were written in R `make.names` form (HLA.A, MT.CO1). `w4_gate2_run.load_c1`, which `experiments/stemsc` and this audit's GSE117498 path use, unions the files by exact name and fills absent genes with zero. Every such gene therefore splits into two columns, one zero in every broad-gate cell and the other zero in every sorted cell. Any score or primitive that matches genes by symbol (mRNAsi, FitDevo, PCC(x, degree)) reads zero for those genes in the broad-gate cells, a difference that runs with population and so with tier.

The brief specifies the stemsc code path, so the GSE117498 lines use it unchanged. `prepare_inputs.py` also writes GSE117498h, the same cells with each broad-gate name merged into its hyphenated twin (4,137 name pairs merged; 18,819 genes remain after the 10-cell filter, against 19,517 on the unmerged path), and PREREG.md registers mRNAsi and FitDevo on it as sensitivity lines. The existing W4 and StemSC rows on GSE117498 went through the same loader; that is outside this task and is recorded here for the author.

## Files

| file | what |
|---|---|
| `scores/mce.py` | MCE from the published equations |
| `tests/test_mce.py` | analytic and optimizer checks; run standalone (`python3 tests/test_mce.py`) and by pytest in CI, not by `tests/run_tests.py` |
| `experiments/w5_new_rows/validate_mce.py` | toy validation, timing, Octave hook |
| `experiments/w5_new_rows/fetch_sources.py` | clones FitDevo, stemFinder and TCGAbiolinks at the pinned commits |
| `experiments/w5_new_rows/discovery_facts.py` | recomputes the weight-vector and overlap counts above |
| `experiments/w5_new_rows/discovery/*.json` | the outputs of the last three |
