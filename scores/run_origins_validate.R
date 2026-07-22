# Track 4 — REAL ORIGINS::activity on the 500-cell subsample (verbatim package).
# Reference for validating the vectorized x^T A x used at full scale.
suppressMessages({library(Matrix); library(ORIGINS)})
IO <- "/private/tmp/claude-501/-Users-damir-damir-research-vault/f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad/origins_io"

genes <- readLines(file.path(IO, "origins_genes.txt"))
expr <- as.matrix(readMM(file.path(IO, "sub_expr_cpm.mtx")))   # genes x cells (CPM)
rownames(expr) <- genes
colnames(expr) <- paste0("c", seq_len(ncol(expr)))
cat("expr", dim(expr), "\n")

data(differentiation_edges)
act <- activity(expr, differentiation_edges)                   # real ORIGINS, native network

write.csv(data.frame(cell = seq_along(act), origins_R = as.vector(act)),
          file.path(IO, "origins_R_subsample.csv"), row.names = FALSE)
cat(sprintf("REAL ORIGINS done: n=%d range [%.4f, %.4f]\n", length(act), min(act), max(act)))
