#!/usr/bin/env Rscript
# Run the REAL SLICE package (xu-lab/SLICE) scEntropy on the prepared subsample.
# Score = per-cell single-cell entropy (sc@entropy) from SLICE::getEntropy.
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({ library(Matrix); library(SLICE) })

# --- repo-relative I/O roots -------------------------------------------------
# Was an ephemeral agent-session scratchpad under /private/tmp, so this script
# only ran on one machine on one day. Override with $OWNBASELINE_SCRATCH /
# $OWNBASELINE_DATA_ROOT; defaults are <repo>/data/scratch and <repo>/data/runs.
.argv  <- commandArgs(trailingOnly = FALSE)
.self  <- sub("^--file=", "", grep("^--file=", .argv, value = TRUE))[1]
.start <- if (is.na(.self)) getwd() else dirname(normalizePath(.self))
.repo  <- .start
while (!file.exists(file.path(.repo, "own_baseline", "paths.R")) &&
       dirname(.repo) != .repo) .repo <- dirname(.repo)
source(file.path(.repo, "own_baseline", "paths.R"))
SCR <- file.path(ob_scratch_root(), "fix3_sub")
args <- commandArgs(trailingOnly = TRUE)
B.num <- if (length(args) >= 1) as.integer(args[1]) else 100L
out_txt <- if (length(args) >= 2) args[2] else file.path(SCR, "slice_score.txt")

# --- real SLICE gene kappa-similarity matrix (human), from data/hs_kappasim.rda ---
e <- new.env(); load(file.path(dirname(SCR), "hs_kappasim.rda"), envir = e)
km <- get(ls(e)[1], envir = e)                 # object 'kh', 14280 x 14280
cat("km dim:", paste(dim(km), collapse = "x"), "\n")

# --- prepared subsample (genes x cells, raw UMI) ---
M <- readMM(file.path(SCR, "sub_genes_cells.mtx"))
feat <- readLines(file.path(SCR, "sub_features.tsv"))
idx <- scan(file.path(SCR, "sub_cell_idx.txt"), quiet = TRUE)
mat <- as.matrix(M)
storage.mode(mat) <- "double"
rownames(mat) <- feat
colnames(mat) <- paste0("c", idx)
cat("expr matrix:", paste(dim(mat), collapse = "x"),
    " genes overlapping km:", sum(rownames(mat) %in% rownames(km)), "\n")

# slice S4 class requires exprmatrix as data.frame and cellidentity as factor
sc <- construct(exprmatrix = as.data.frame(mat),
                cellidentity = factor(rep("cell", ncol(mat))))
t0 <- Sys.time()
sc <- getEntropy(sc, km = km,
                 calculation = "bootstrap",
                 B.num       = B.num,
                 B.size      = 1000,
                 exp.cutoff  = 1,
                 clustering.k = floor(sqrt(1000 / 2)),   # = 22, as in demo/FB.R
                 random.seed = 201602)
el <- as.numeric(Sys.time() - t0, units = "secs")
cat(sprintf("getEntropy B.num=%d elapsed=%.1fs\n", B.num, el))

sco <- as.numeric(sc@entropy)
cat("score n:", length(sco), " range:", sprintf("%.4f..%.4f", min(sco), max(sco)),
    " finite:", all(is.finite(sco)), "\n")
writeLines(sprintf("%.10g", sco), out_txt)
cat("wrote", out_txt, "\n")
