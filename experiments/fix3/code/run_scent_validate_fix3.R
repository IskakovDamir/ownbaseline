# FIX 3 — REAL SCENT (verbatim aet21/SCENT source) on a 500-cell Briggs subsample,
# to independently re-validate the vectorized SR/CCAT used at full scale (same as the
# zebrafish decider's run_scent_validate.R, pointed at the fix3 IO dir).
suppressMessages({library(Matrix); library(igraph); library(parallel)})
IO  <- "/private/tmp/claude-501/-Users-damir-damir-research-vault/f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad/fix3_scent_io"
SRC <- "/private/tmp/claude-501/-Users-damir-damir-research-vault/f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad/scent_src"
source(file.path(SRC, "CompCCAT.R")); source(file.path(SRC, "DoIntegPPI.R")); source(file.path(SRC, "CompSRana.R"))

mc_genes <- readLines(file.path(IO, "mc_genes.txt"))
adj <- as.matrix(readMM(file.path(IO, "adjMC.mtx"))); rownames(adj) <- mc_genes; colnames(adj) <- mc_genes; diag(adj) <- 0
expr <- as.matrix(readMM(file.path(IO, "sub_expr_libnorm.mtx"))); rownames(expr) <- mc_genes
colnames(expr) <- paste0("c", seq_len(ncol(expr)))
cat("adj", dim(adj), " expr", dim(expr), " max(expr)=", max(expr), "\n")

ccat <- CompCCAT(exp.m = expr, ppiA.m = adj)
integ <- DoIntegPPI(exp.m = expr, ppiA.m = adj)
srobj <- CompSRana(integ, local = FALSE, mc.cores = 2)
write.csv(data.frame(cell = seq_along(ccat), ccat_R = as.vector(ccat), sr_R = as.vector(srobj$SR)),
          file.path(IO, "scent_R_subsample.csv"), row.names = FALSE)
cat(sprintf("REAL SCENT (Briggs) done: n=%d SR[%.4f,%.4f] CCAT[%.4f,%.4f]\n",
            length(ccat), min(srobj$SR), max(srobj$SR), min(ccat), max(ccat)))
