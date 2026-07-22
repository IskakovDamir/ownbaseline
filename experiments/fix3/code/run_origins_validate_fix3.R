# FIX 3 — REAL ORIGINS::activity on a 500-cell Briggs subsample (re-validates the
# vectorized x^T A x used at full scale; same as the zebrafish run_origins_validate.R).
suppressMessages({library(Matrix); library(ORIGINS)})
IO <- "/private/tmp/claude-501/-Users-damir-damir-research-vault/f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad/fix3_origins_io"
genes <- readLines(file.path(IO, "origins_genes.txt"))
expr <- as.matrix(readMM(file.path(IO, "sub_expr_cpm.mtx"))); rownames(expr) <- genes
colnames(expr) <- paste0("c", seq_len(ncol(expr)))
data(differentiation_edges)
act <- activity(expr, differentiation_edges)
write.csv(data.frame(cell = seq_along(act), origins_R = as.vector(act)),
          file.path(IO, "origins_R_subsample.csv"), row.names = FALSE)
cat(sprintf("REAL ORIGINS (Briggs) done: n=%d range [%.4f,%.4f]\n", length(act), min(act), max(act)))
