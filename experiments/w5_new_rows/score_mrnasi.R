# score_mrnasi.R -- mRNAsi (Malta et al., Cell 2018) with the published PCBC
# one-class logistic regression weights as TCGAbiolinks ships them
# (SC_PCBC_stemSig, Bioconductor RELEASE_3_23, TCGAbiolinks 2.40.0, commit
# 2f9d249), scored by TCGAbiolinks' own TCGAanalyze_Stemness sourced from
# R/Stemness.R at that commit. No retraining.
#
#   Rscript experiments/w5_new_rows/score_mrnasi.R GSE117498
#
# The function intersects the weight names with the matrix rownames and drops
# the rest silently; the overlap is logged here. Spearman is computed within
# each cell, so the raw counts need no per-cell normalisation: any monotone
# per-cell transform gives the same score.

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

src <- file.path(w5_root(), "src", "TCGAbiolinks")
source(file.path(src, "R", "Stemness.R"))
load(file.path(src, "data", "SC_PCBC_stemSig.rda"))

MAT <- w5_read_counts(dataset)
common <- intersect(names(SC_PCBC_stemSig), rownames(MAT))
cat(sprintf("%s: %d genes x %d cells; %d of %d weights matched\n", dataset,
            nrow(MAT), ncol(MAT), length(common), length(SC_PCBC_stemSig)))
# the function densifies X[common, ]; doing the subset first on the sparse
# matrix keeps memory to the matched genes and changes nothing it computes
dataGE <- as.matrix(MAT[common, ])
t0 <- Sys.time()
s <- TCGAanalyze_Stemness(stemSig = SC_PCBC_stemSig, dataGE = dataGE,
                          colname.score = "stemness_score")
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))
stopifnot(identical(s$Sample, colnames(MAT)))
w5_write_scores(data.frame(cell = s$Sample, mrnasi = s$stemness_score), dataset, "mrnasi")
info <- file.path(w5_root(), "scores", dataset, "mrnasi_info.txt")
writeLines(c(sprintf("seconds\t%.1f", secs),
             sprintf("n_cells\t%d", ncol(MAT)),
             sprintf("n_weights\t%d", length(SC_PCBC_stemSig)),
             sprintf("n_weights_matched\t%d", length(common)),
             sprintf("n_nan\t%d", sum(!is.finite(s$stemness_score))),
             sprintf("r\t%s", R.version.string)), info)
cat(sprintf("mRNAsi %s: %d cells, %.0f s\n", dataset, ncol(MAT), secs))
