#!/usr/bin/env python3
"""Build a reproducible 3000-cell subsample (seed 42) shared by SLICE and dpath.

Writes, into the scratch root:
  sub_genes_cells.mtx   MatrixMarket, genes x cells, raw UMI counts (subsample)
  sub_features.tsv      gene symbols (row i of the mtx), full 23974
  sub_cell_idx.txt      0-based indices into cells.tsv canonical order (length 3000, sorted)
The genes x cells orientation and gene rownames match what SLICE::scEntropy and
dpath::dpath expect.
"""
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# scratch directory, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root, scratch_root  # noqa: E402
PREP = str(data_root() / "track2/data/prepared")
OUT = str(scratch_root())
N_SUB = 3000
SEED = 42

X = sp.load_npz(f"{PREP}/X_cells_genes.npz").tocsr()  # cells x genes
n_cells, n_genes = X.shape
print("full X:", X.shape, "nnz", X.nnz)

with open(f"{PREP}/features.tsv") as f:
    features = [ln.strip() for ln in f if ln.strip() != ""]
assert len(features) == n_genes, (len(features), n_genes)

rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(n_cells, size=N_SUB, replace=False))
print("subsample:", len(idx), "cells; first5", idx[:5].tolist())

Xsub = X[idx, :]                # cells x genes (subsample)
G = Xsub.T.tocoo()             # genes x cells
G = sp.csc_matrix(G)
print("genes x cells subsample:", G.shape, "nnz", G.nnz)

sio.mmwrite(f"{OUT}/sub_genes_cells.mtx", G, field="integer", symmetry="general")
with open(f"{OUT}/sub_features.tsv", "w") as f:
    f.write("\n".join(features) + "\n")
np.savetxt(f"{OUT}/sub_cell_idx.txt", idx, fmt="%d")
# sanity: per-cell library size of subsample (for later cross-check)
libsize = np.asarray(Xsub.sum(axis=1)).ravel()
print("libsize range:", int(libsize.min()), int(libsize.max()))
print("DONE")
