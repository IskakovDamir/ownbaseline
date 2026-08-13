# Track 2 Gate 2 — REAL SCENT on a 500-cell subsample (verbatim aet21/SCENT source).
# Establishes the reference SR/CCAT that the vectorized Python reimpl must match
# before Python is applied at full 39,505-cell scale.
suppressMessages({library(Matrix); library(igraph); library(parallel)})

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
IO   <- file.path(ob_scratch_root(), "scent_io")
SRC  <- file.path(ob_scratch_root(), "scent_src")
source(file.path(SRC, "CompCCAT.R"))
source(file.path(SRC, "DoIntegPPI.R"))
source(file.path(SRC, "CompSRana.R"))

mc_genes <- readLines(file.path(IO, "mc_genes.txt"))
adj <- as.matrix(readMM(file.path(IO, "adjMC.mtx")))          # dense base matrix (SCENT convention)
rownames(adj) <- mc_genes; colnames(adj) <- mc_genes
diag(adj) <- 0

expr <- as.matrix(readMM(file.path(IO, "sub_expr_libnorm.mtx")))  # genes x cells, linear CPM
rownames(expr) <- mc_genes
colnames(expr) <- paste0("c", seq_len(ncol(expr)))
cat("adj", dim(adj), " expr", dim(expr), " max(expr)=", max(expr), "\n")

# ---- REAL SCENT CCAT (log2(x+1) inside, Pearson expr vs degree) ----
ccat <- CompCCAT(exp.m = expr, ppiA.m = adj)

# ---- REAL SCENT SR (DoIntegPPI log2(x+1.1) -> CompSRana entropy rate) ----
integ <- DoIntegPPI(exp.m = expr, ppiA.m = adj)
maxSR <- CompMaxSR(integ)
srobj <- CompSRana(integ, local = FALSE, mc.cores = 2)
sr <- srobj$SR                                                # normalized (/maxSR)

out <- data.frame(cell = seq_len(ncol(expr)), ccat_R = as.vector(ccat), sr_R = as.vector(sr))
write.csv(out, file.path(IO, "scent_R_subsample.csv"), row.names = FALSE)
cat(sprintf("REAL SCENT done: n=%d  maxSR=%.4f\n", nrow(out), maxSR))
cat(sprintf("  SR   range [%.4f, %.4f] mean %.4f\n", min(sr), max(sr), mean(sr)))
cat(sprintf("  CCAT range [%.4f, %.4f] mean %.4f\n", min(ccat), max(ccat), mean(ccat)))
