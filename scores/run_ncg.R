#!/usr/bin/env Rscript
# Run the REAL NCG potency model (Ni et al. 2021; github.com/Xinzhe-Ni/NCG)
# per-cell on the prepared zebrafish subsample.
#
# Pipeline (all from the repo's own Function/*.R, unmodified):
#   Processdata(counts, PPI)  -> quantile-normalize + log2 expr; ENTREZ->SYMBOL on PPI
#   DoIntegPPI(exp, adj)      -> common genes + maximally connected subnetwork
#   CompECC(adjMC)            -> edge clustering coefficient matrix
#   CompNCG(ECC, expMC, km)   -> per-cell NCG score (normalized by max over cells run)
#
# Resources (repo-bundled, from Data/Data.zip, sha256-verified against the LFS pointer):
#   hprdAsigH-13Jun12.Rd  -> hprdAsigH.m  (8434x8434 PPI, ENTREZ ids)
#   hs_km.Rda             -> km           (13908x13908 GO kappa-similarity, SYMBOLs)
#
# Expression: shared 3000-cell subsample (numpy default_rng(42)), genes x cells raw UMI.

.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
options(repos = c(CRAN = "https://cloud.r-project.org"), Ncpus = 4)
suppressMessages(library(Matrix))

REPO <- "/private/tmp/claude-501/-Users-damir-damir-research-vault/f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad/NCG"
SCR  <- "/private/tmp/claude-501/-Users-damir-damir-research-vault/f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad"
FUN  <- file.path(REPO, "Function")
DAT  <- file.path(REPO, "Data")

flush_cat <- function(...) { cat(...); flush.console() }

# --- real repo functions, unmodified ---
source(file.path(FUN, "Processdata.R"))
source(file.path(FUN, "DoIntegPPI.R"))
source(file.path(FUN, "CompECC.R"))
source(file.path(FUN, "CompNCG.R"))

# --- repo-bundled resources ---
e1 <- new.env(); load(file.path(DAT, "hprdAsigH-13Jun12.Rd"), envir = e1)
PPI <- get("hprdAsigH.m", envir = e1)
e2 <- new.env(); load(file.path(DAT, "hs_km.Rda"), envir = e2)
km  <- get("km", envir = e2)
flush_cat(sprintf("PPI hprdAsigH.m: %dx%d (ENTREZ)\n", nrow(PPI), ncol(PPI)))
flush_cat(sprintf("km hs_km: %dx%d (SYMBOL)\n", nrow(km), ncol(km)))

# --- prepared subsample expression (genes x cells, raw UMI) ---
M    <- as.matrix(readMM(file.path(SCR, "sub_genes_cells.mtx")))
feat <- readLines(file.path(SCR, "sub_features.tsv"))
idx  <- scan(file.path(SCR, "sub_cell_idx.txt"), quiet = TRUE)
stopifnot(nrow(M) == length(feat), ncol(M) == length(idx))
storage.mode(M) <- "double"
colnames(M) <- paste0("c", idx)
# counts as Processdata expects: col 1 = gene symbol, remaining = cells
counts <- data.frame(gene = feat, M, check.names = FALSE, stringsAsFactors = FALSE)
flush_cat(sprintf("expression subsample: %d genes x %d cells (raw UMI)\n", nrow(M), ncol(M)))

# --- (1) Processdata: quantile-norm + log2 + PPI ENTREZ->SYMBOL ---
t0 <- Sys.time()
data <- Processdata(counts, PPI)
flush_cat(sprintf("[Processdata] exp %dx%d | adj %dx%d | %.1fs\n",
                  nrow(data$exp), ncol(data$exp), nrow(data$adj), ncol(data$adj),
                  as.numeric(Sys.time() - t0, units = "secs")))
flush_cat(sprintf("  PPI symbols non-NA: %d / %d\n",
                  sum(!is.na(rownames(data$adj))), nrow(data$adj)))
flush_cat(sprintf("  expr-genes in PPI(symbol): %d ; expr-genes in km: %d\n",
                  length(intersect(rownames(data$exp), rownames(data$adj))),
                  length(intersect(rownames(data$exp), rownames(km)))))

# --- (2) DoIntegPPI: max connected subnetwork ---
t0 <- Sys.time()
int.o <- DoIntegPPI(data$exp, data$adj)
flush_cat(sprintf("[DoIntegPPI] expMC %dx%d | adjMC %dx%d | %.1fs\n",
                  nrow(int.o$expMC), ncol(int.o$expMC), nrow(int.o$adjMC), ncol(int.o$adjMC),
                  as.numeric(Sys.time() - t0, units = "secs")))

# --- (3) CompECC: edge clustering coefficient (O(n^2) loop, cell-independent) ---
t0 <- Sys.time()
ECC <- CompECC(int.o$adjMC)
flush_cat(sprintf("[CompECC] ECC %dx%d | %.1fs\n",
                  nrow(ECC), ncol(ECC), as.numeric(Sys.time() - t0, units = "secs")))

# --- (4) CompNCG: per-cell NCG score ---
common1 <- intersect(rownames(km), rownames(ECC))
common  <- intersect(common1, rownames(int.o$expMC))
flush_cat(sprintf("[CompNCG] genes shared ECC+km+expr: %d\n", length(common)))
t0 <- Sys.time()
NCG <- CompNCG(ECC, int.o$expMC, km)
flush_cat(sprintf("[CompNCG] n=%d | range %.6f..%.6f | %.1fs\n",
                  length(NCG), min(NCG), max(NCG),
                  as.numeric(Sys.time() - t0, units = "secs")))

stopifnot(length(NCG) == ncol(M), all(is.finite(NCG)))
out_txt <- file.path(SCR, "ncg_score.txt")
writeLines(sprintf("%.10g", NCG), out_txt)
flush_cat(sprintf("wrote %s (n=%d)\n", out_txt, length(NCG)))
flush_cat("DONE\n")
