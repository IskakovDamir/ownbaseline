# w5_io.R -- shared I/O for the w5 score wrappers.
#
# Reads the genes x cells count matrix prepare_inputs.py wrote (raw little-
# endian CSC arrays, no text parsing) and resolves the data roots exactly as
# own_baseline/paths.R does. Source after the paths.R bootstrap.

# Seurat and qlcMatrix live in a project-local library under the scratch
# root (installed by the command in REPORT.md), ahead of the site library
.libPaths(c(file.path(ob_scratch_root(), "Rlib"), .libPaths()))
suppressMessages(library(Matrix))

w5_root <- function() file.path(ob_data_root(), "w5_new_rows")

w5_read_counts <- function(dataset) {
  d <- file.path(w5_root(), "inputs", dataset)
  dims <- jsonlite_free_read(file.path(d, "dims.json"))
  nnz <- dims[["nnz"]]
  i <- readBin(file.path(d, "csc_i.bin"), "integer", n = nnz, size = 4, endian = "little")
  p <- readBin(file.path(d, "csc_p.bin"), "integer", n = dims[["n_cells"]] + 1,
               size = 4, endian = "little")
  x <- readBin(file.path(d, "csc_x.bin"), "double", n = nnz, size = 8, endian = "little")
  genes <- readLines(file.path(d, "genes.txt"))
  cells <- readLines(file.path(d, "cells.txt"))
  m <- new("dgCMatrix", i = i, p = p, x = x,
           Dim = c(as.integer(dims[["n_genes"]]), as.integer(dims[["n_cells"]])),
           Dimnames = list(genes, cells))
  stopifnot(length(genes) == nrow(m), length(cells) == ncol(m))
  m
}

# dims.json is three integers; parsed without a JSON package so the wrapper
# needs nothing beyond what the scored method needs
jsonlite_free_read <- function(path) {
  txt <- paste(readLines(path, warn = FALSE), collapse = "")
  keys <- regmatches(txt, gregexpr('"[a-z_]+"\\s*:\\s*[0-9]+', txt))[[1]]
  out <- list()
  for (k in keys) {
    kv <- strsplit(k, ":")[[1]]
    out[[gsub('[" ]', "", kv[1])]] <- as.numeric(trimws(kv[2]))
  }
  out
}

w5_write_scores <- function(df, dataset, row) {
  dir <- file.path(w5_root(), "scores", dataset)
  dir.create(dir, recursive = TRUE, showWarnings = FALSE)
  f <- file.path(dir, paste0(row, ".csv"))
  write.csv(df, f, row.names = FALSE)
  cat(sprintf("wrote %s (%d cells)\n", f, nrow(df)))
  f
}
