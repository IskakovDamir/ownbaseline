#!/usr/bin/env Rscript
# Run the REAL dpath package (gongx030/dpath) on the prepared subsample.
# Score = per-cell metagene entropy (dpath's "differentiation potential"):
#   H_i = entropy2( t(dp$V) )[i]   where dp$V is the K x N metagene-coefficient
#   matrix from weighted-Poisson NMF, combined across repeats by reorder.dpath.
# entropy2(x) = -rowSums(x * log(x + eps)) is dpath's own metagene-entropy fn.
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressMessages({ library(Matrix); library(dpath) })

SCR <- "/private/tmp/claude-501/-Users-damir-damir-research-vault/f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad/fix3_sub"
args <- commandArgs(trailingOnly = TRUE)
repeat.mf <- if (length(args) >= 1) as.integer(args[1]) else 20L
ngene     <- if (length(args) >= 2) as.integer(args[2]) else 2000L
out_txt   <- if (length(args) >= 3) args[3] else file.path(SCR, "dpath_score.txt")

M <- readMM(file.path(SCR, "sub_genes_cells.mtx"))     # genes x cells, raw UMI
feat <- readLines(file.path(SCR, "sub_features.tsv"))
idx  <- scan(file.path(SCR, "sub_cell_idx.txt"), quiet = TRUE)
mat  <- as.matrix(M); storage.mode(mat) <- "double"
rownames(mat) <- feat; colnames(mat) <- paste0("c", idx)

# informative gene set for wp-NMF (dpath's intended 'subset.gene' usage):
# drop genes expressed in <3 cells, keep top-`ngene` by log1p variance.
expressed <- rowSums(mat > 0) >= 3
mat <- mat[expressed, , drop = FALSE]
v   <- apply(log1p(mat), 1, var)
sel <- names(sort(v, decreasing = TRUE))[seq_len(min(ngene, nrow(mat)))]
X   <- mat[sel, , drop = FALSE]
cat("wp-NMF input X:", paste(dim(X), collapse = "x"), "\n")

# dpath()'s input guard `if (class(X) != 'matrix')` is length-2 on R>=4.2
# (class(matrix) == c('matrix','array')). X genuinely IS a numeric matrix;
# set an explicit class attr so the guard is length-1 TRUE. No package code changed.
# (class(X)<-"matrix" is normalized by R back to c("matrix","array"); attr() sticks.)
attr(X, "class") <- "matrix"

set.seed(42)
t0 <- Sys.time()
dp <- dpath(X, K = 5, lambda0 = 0.1, max.iter = 500,
            repeat.mf = repeat.mf, mc.cores = 4)
dp <- reorder.dpath(dp)
el <- as.numeric(Sys.time() - t0, units = "secs")
cat(sprintf("dpath repeat.mf=%d ngene=%d elapsed=%.1fs\n", repeat.mf, ngene, el))

stopifnot(!is.null(dp$V), ncol(dp$V) == ncol(X))
H <- as.numeric(dpath:::entropy2(t(dp$V)))     # per-cell metagene entropy
cat("V dim:", paste(dim(dp$V), collapse = "x"),
    " colSums(V) approx 1?", round(mean(colSums(dp$V)), 4), "\n")
cat("score n:", length(H), " range:", sprintf("%.4f..%.4f", min(H), max(H)),
    " finite:", all(is.finite(H)), "\n")
writeLines(sprintf("%.10g", H), out_txt)
cat("wrote", out_txt, "\n")
