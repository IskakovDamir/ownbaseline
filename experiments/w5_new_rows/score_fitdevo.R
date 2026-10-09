# score_fitdevo.R -- FitDevo developmental potential (Zhang et al., Brief
# Bioinform 2022, bbac293) on one w5 input dataset, with the shipped BGW and
# the authors' fitdevo() from fitdevo.R v1.2 (github.com/jumphone/FitDevo at
# 0c757e6), sourced unmodified. No retraining: BGW.rds is the pretrained
# weight vector, and the per-dataset SSGW step uses no label.
#
#   Rscript experiments/w5_new_rows/score_fitdevo.R GSE117498
#
# The one environment setting: fitdevo.R reads pbmc@assays$RNA@data, which a
# Seurat 5 Assay5 object does not have, so assays are created as v3 Assay
# objects. That changes where Seurat keeps the matrices, not what is computed.

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
suppressMessages({library(Seurat); library(qlcMatrix)})

src <- file.path(w5_root(), "src", "FitDevo")
source(file.path(src, "fitdevo.R"))
BGW <- readRDS(file.path(src, "BGW.rds"))

MAT <- w5_read_counts(dataset)
cat(sprintf("%s: %d genes x %d cells\n", dataset, nrow(MAT), ncol(MAT)))
t0 <- Sys.time()
out <- fitdevo(MAT, BGW, NORM = TRUE, PCNUM = 50, VARGENE = 2000,
               tooLargeLimit = 50000, SEED = 123, SS = TRUE, DETAIL = TRUE)
secs <- as.numeric(difftime(Sys.time(), t0, units = "secs"))

gw <- out$USED_GW
df <- data.frame(cell = colnames(MAT), fitdevo_dp = out$DP)
w5_write_scores(df, dataset, "fitdevo")
info <- file.path(w5_root(), "scores", dataset, "fitdevo_info.txt")
writeLines(c(
  sprintf("seconds\t%.1f", secs),
  sprintf("n_cells\t%d", ncol(MAT)),
  sprintf("n_genes_input\t%d", nrow(MAT)),
  sprintf("n_used_gw\t%d", length(gw)),
  sprintf("ssgw_0\t%d", sum(gw == 0)),
  sprintf("ssgw_1\t%d", sum(gw == 1)),
  sprintf("ssgw_2\t%d", sum(gw == 2)),
  sprintf("seurat\t%s", as.character(packageVersion("Seurat"))),
  sprintf("seuratobject\t%s", as.character(packageVersion("SeuratObject"))),
  sprintf("qlcmatrix\t%s", as.character(packageVersion("qlcMatrix"))),
  sprintf("r\t%s", R.version.string)), info)
cat(sprintf("FitDevo %s: %d cells, %d genes in SSGW, %.0f s\n",
            dataset, ncol(MAT), length(gw), secs))
