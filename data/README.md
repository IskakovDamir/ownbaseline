# Data

No expression data is stored in this repo. All datasets are public; fetch them
into a working directory and point the experiment scripts at them (see the
`VAULT`/`PREP` path constants at the top of each script).

## Single-cell datasets (GEO)

| Accession | Study | Use in the paper |
|---|---|---|
| GSE106474 | Farrell 2018, zebrafish embryogenesis (Drop-seq, 12 stages, 39,505 cells) | primary clean fine-ordinal decider |
| GSE113074 | Briggs 2018, *Xenopus tropicalis* embryogenesis (inDrop, 10 stages) | second clean fine-ordinal atlas (cross-species) |
| GSE117498 | sorted human hematopoiesis | depth-confound worked example (Fig 3) |
| GSE125970 | human intestine | depth-confound worked example (Fig 3) |
| GSE150949 | Watermelon, PC9 / osimertinib drug-tolerant persister | cross-cancer primitive transfer |
| GSE223003 | ReSisTrace, Kuramochi / olaparib | cross-cancer primitive transfer |

Download, e.g.:

    # 12-stage zebrafish UMI counts
    curl -O "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/GSE106474/suppl/GSE106474_UMICounts.txt.gz"
    # Xenopus annotated counts (large; per-cell Developmental_stage is in the header rows)
    curl -O "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE113nnn/GSE113074/suppl/GSE113074_Raw_combined.annotated_counts.tsv.gz"

The Xenopus file is ~170 MB and NCBI throttles per-IP; a byte-range segmented
download is in `experiments/fix3/code/download_briggs_segmented.py`.

## PPI network — STRING v12

The degree-correlation primitive PCC(x, degree) and the SCENT/CCAT scores use
the STRING v12 human network, high-confidence edges (combined_score >= 700),
largest connected component. Build it from the STRING downloads with:

    experiments/w4/scripts/build_human_string_v12_thr700_lcc.py

which reads `9606.protein.links.v12.0.txt.gz` and `9606.protein.info.v12.0.txt.gz`
from https://stringdb-downloads.org (see the script header for exact URLs) and
writes an `.npz` (genes, degree, adjacency). Cross-species datasets (zebrafish,
Xenopus) are matched to the human network by uppercased ortholog symbol — a
disclosed convention.
