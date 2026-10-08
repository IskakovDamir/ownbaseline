# FIX 3 — REAL ORIGINS::activity on a 500-cell Briggs subsample (re-validates the
# vectorized x^T A x used at full scale; same as the zebrafish run_origins_validate.R).
suppressMessages({library(Matrix); library(ORIGINS)})
# --- repo-relative I/O roots -------------------------------------------------
# Was an ephemeral scratch directory under /private/tmp, so this script
# only ran on one machine on one day. Override with $OWNBASELINE_SCRATCH /
# $OWNBASELINE_DATA_ROOT; defaults are <repo>/data/scratch and <repo>/data/runs.
.argv  <- commandArgs(trailingOnly = FALSE)
.self  <- sub("^--file=", "", grep("^--file=", .argv, value = TRUE))[1]
.start <- if (is.na(.self)) getwd() else dirname(normalizePath(.self))
.repo  <- .start
while (!file.exists(file.path(.repo, "own_baseline", "paths.R")) &&
       dirname(.repo) != .repo) .repo <- dirname(.repo)
source(file.path(.repo, "own_baseline", "paths.R"))
IO <- file.path(ob_scratch_root(), "fix3_origins_io")
genes <- readLines(file.path(IO, "origins_genes.txt"))
expr <- as.matrix(readMM(file.path(IO, "sub_expr_cpm.mtx"))); rownames(expr) <- genes
colnames(expr) <- paste0("c", seq_len(ncol(expr)))
data(differentiation_edges)
act <- activity(expr, differentiation_edges)
write.csv(data.frame(cell = seq_along(act), origins_R = as.vector(act)),
          file.path(IO, "origins_R_subsample.csv"), row.names = FALSE)
cat(sprintf("REAL ORIGINS (Briggs) done: n=%d range [%.4f,%.4f]\n", length(act), min(act), max(act)))
