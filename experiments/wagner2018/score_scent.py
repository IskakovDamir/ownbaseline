#!/usr/bin/env python3
"""
score_scent.py: SR and CCAT on the Wagner cells, through run_scent_full's own
functions. The R cross-check of run_scent_full.main is left out; the same code
matched the SCENT R functions on GSE106474. Reads no label.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments" / "track2" / "code"))
import run_scent_full as rs  # noqa: E402


def main():
    A, degree, mc_col, target_sum, _ = rs.load_net()
    max_sr = rs.pm._max_entropy_rate(A)
    X = sp.load_npz(rs.PREP / "X_cells_genes.npz").tocsr()
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    ccat, sr = rs.compute_scores(X, lib, np.arange(X.shape[0]), A, degree, mc_col, target_sum, max_sr)
    rs.OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(rs.OUT, ccat=ccat, sr=sr, maxSR=max_sr)
    print(f"SR and CCAT on {X.shape[0]} cells, network {A.shape[0]} genes, NaN {int(np.isnan(sr).sum())}, {int(np.isnan(ccat).sum())}")


if __name__ == "__main__":
    main()
