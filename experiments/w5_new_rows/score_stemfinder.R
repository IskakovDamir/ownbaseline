# score_stemfinder.R -- stemFinder (Noller and Cahan, Brief Bioinform 2024,
# bbae536) with the authors' run_stemFinder() and gene_set_score() sourced
# unmodified from github.com/CahanLab/stemfinder at db8ef0e (package 0.1.0).
#
#   Rscript experiments/w5_new_rows/score_stemfinder.R GSE106474
#
# Pipeline, as pre-registered in PREREG.md (process_cyto_adata and the README
# vignette, with the human S/G2M list the package bundles):
#   gene-level steps of process_cyto_adata: genes detected in >= 5 cells,
#     mitochondrial genes (^MT-) and MALAT1 removed; the cell-level filters are
#     not applied because the audit's cells are fixed by the repository QC
#   NormalizeData LogNormalize 1e4; FindVariableFeatures vst 2500
#   markers = bundled human S + G2M genes present in the data (symbols are
#     upper case in all w5 datasets except GSE113074, matched upper-cased)
#   cell-cycle markers removed from the variable features
#   ScaleData on the variable features plus the markers. Seurat scales each
#     gene on its own, so the markers' scaled values are those a scale over
#     every gene would give; scaling all ~24,000 genes densely does not fit
#     in memory at 39,505 cells.
#   RunPCA, 50 components, seed 42; pcs = the elbow of the 50 standard
#     deviations, taken as the point farthest from the chord joining the
#     first and the last (both axes scaled to [0, 1]). The authors pick it by
#     eye; this is the pre-registered rule that replaces the eye.
#   k = round(sqrt(N)); FindNeighbors(dims = 1:pcs, k.param = k); the RNA_nn
#     graph includes each cell as its own neighbour, as run_stemFinder needs
#   run_stemFinder(thresh = 0, method = "gini"); gene_set_score(markers), the
#     authors' cell-cycle gene-set score, is the declared primitive

.argv  <- commandArgs(trailingOnly = FALSE)
.self  <- sub("^--file=", "", grep("^--file=", .argv, value = TRUE))[1]
.start <- if (is.na(.self)) getwd() else dirname(normalizePath(.self))
.repo  <- .start
while (!file.exists(file.path(.repo, "own_baseline", "paths.R")) &&
       dirname(.repo) != .repo) .repo <- dirname(.repo)
source(file.path(.repo, "own_baseline", "paths.R"))
source(file.path(.repo, "experiments", "w5_new_rows", "w5_io.R"))

args <- commandArgs(trailingOnly = TRUE)
dataset <- args[1]
stopifnot(!is.na(dataset))

options(Seurat.object.assay.version = "v3")
suppressMessages(library(Seurat))

src <- file.path(w5_root(), "src", "stemfinder")
source(file.path(src, "R", "run_stemFinder.R"))
source(file.path(src, "R", "gene_set_score.R"))
load(file.path(src, "data", "s_genes_human.rda"))
load(file.path(src, "data", "g2m_genes_human.rda"))

elbow <- function(sdev) {
  n <- length(sdev)
  x <- (seq_len(n) - 1) / (n - 1)
  y <- (sdev - min(sdev)) / (max(sdev) - min(sdev))
  # distance of each point to the line through (0, y1) and (1, yn)
  d <- abs((y[n] - y[1]) * x - (1 - 0) * y + 1 * y[1] - 0 * y[n]) /
       sqrt((y[n] - y[1])^2 + 1)
  which.max(d)
}

MAT <- w5_read_counts(dataset)
t0 <- Sys.time()
up <- toupper(rownames(MAT))
keep <- !grepl("^MT-", up) & up != "MALAT1"
MAT <- MAT[keep, ]
adata <- CreateSeuratObject(counts = MAT, min.cells = 5, min.features = 0,
                            project = dataset)
adata <- NormalizeData(adata, normalization.method = "LogNormalize",
                       scale.factor = 10000, verbose = FALSE)
adata <- FindVariableFeatures(adata, selection.method = "vst", nfeatures = 2500,
                              verbose = FALSE)

# the README vignette: cell_cycle_genes = c(s_genes, g2m_genes) restricted to
# the genes present, with no unique(). E2F8 is on both lists, so it enters
# run_stemFinder and gene_set_score twice, as in the authors' own construction.
cc <- c(s_genes_human, g2m_genes_human)
present <- rownames(adata)
match_up <- match(toupper(cc), toupper(present))
markers <- present[match_up[!is.na(match_up)]]
markers_unique <- unique(markers)
VariableFeatures(adata) <- setdiff(VariableFeatures(adata), markers_unique)
adata <- ScaleData(adata, features = union(VariableFeatures(adata), markers_unique),
                   verbose = FALSE)
adata <- RunPCA(adata, npcs = 50, features = VariableFeatures(adata),
                seed.use = 42, verbose = FALSE)
pcs <- elbow(adata@reductions$pca@stdev)
N <- ncol(adata)
k <- round(sqrt(N))
set.seed(42)
adata <- FindNeighbors(adata, dims = 1:pcs, k.param = k, verbose = FALSE)
knn <- adata@graphs$RNA_nn
self_ok <- all(diag(knn[seq_len(min(200, N)), seq_len(min(200, N))]) == 1)
stopifnot(self_ok)
adata <- run_stemFinder(adata, nn = knn, k = k, thresh = 0, markers = markers,
                        method = "gini")
adata <- gene_set_score(markers, adata)
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

df <- data.frame(cell = colnames(adata),
                 stemfinder = adata$stemFinder,
                 stemfinder_raw = adata$stemFinder_raw,
                 cc_gene_set_score = adata$gene.set.score)
stopifnot(identical(df$cell, colnames(MAT)))
w5_write_scores(df, dataset, "stemfinder")
info <- file.path(w5_root(), "scores", dataset, "stemfinder_info.txt")
writeLines(c(sprintf("seconds\t%.1f", secs),
             sprintf("n_cells\t%d", N),
             sprintf("n_genes_after_gene_steps\t%d", nrow(adata)),
             sprintf("n_markers\t%d", length(markers)),
             sprintf("n_markers_unique\t%d", length(markers_unique)),
             sprintf("n_markers_listed\t%d", length(cc)),
             sprintf("n_variable_features_used_for_pca\t%d", length(VariableFeatures(adata))),
             sprintf("pcs\t%d", pcs),
             sprintf("k\t%d", k),
             sprintf("seurat\t%s", as.character(packageVersion("Seurat"))),
             sprintf("seuratobject\t%s", as.character(packageVersion("SeuratObject"))),
             sprintf("r\t%s", R.version.string)), info)
cat(sprintf("stemFinder %s: %d cells, %d markers, pcs %d, k %d, %.0f s\n",
            dataset, N, length(markers), pcs, k, secs))
