# paths.R -- the R half of own_baseline/paths.py.
#
# The R score wrappers used to hard-code an ephemeral agent-session scratchpad
# under /private/tmp. They now resolve the same two roots the Python side uses:
#
#   OWNBASELINE_DATA_ROOT   run inputs and outputs   default <repo>/data/runs
#   OWNBASELINE_SCRATCH     R <-> Python handoff     default <repo>/data/scratch
#
# Source it from a script with the bootstrap block that each wrapper carries:
#
#   .argv  <- commandArgs(trailingOnly = FALSE)
#   .self  <- sub("^--file=", "", grep("^--file=", .argv, value = TRUE))[1]
#   .start <- if (is.na(.self)) getwd() else dirname(normalizePath(.self))
#   .repo  <- .start
#   while (!file.exists(file.path(.repo, "own_baseline", "paths.R")) &&
#          dirname(.repo) != .repo) .repo <- dirname(.repo)
#   source(file.path(.repo, "own_baseline", "paths.R"))
#
# after which ob_data_root() and ob_scratch_root() are available.
#
# Nothing here affects a computation. It resolves locations only.

ob_repo_root <- function() .repo

ob_data_root <- function() {
  v <- Sys.getenv("OWNBASELINE_DATA_ROOT")
  if (nzchar(v)) path.expand(v) else file.path(.repo, "data", "runs")
}

ob_scratch_root <- function() {
  v <- Sys.getenv("OWNBASELINE_SCRATCH")
  if (nzchar(v)) path.expand(v) else file.path(.repo, "data", "scratch")
}

ob_ensure <- function(...) {
  for (p in c(...)) if (!dir.exists(p)) dir.create(p, recursive = TRUE)
  invisible(NULL)
}
