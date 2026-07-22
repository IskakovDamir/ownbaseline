#!/usr/bin/env python3
"""Assemble a Track-4 score npz from an R score text file + the subsample cell_idx.
Usage: write_npz.py <score_txt> <arrayname> <out_npz>
Writes: array `<arrayname>` (per-cell score, subsample order) and `cell_idx`
(int32 indices into cells.tsv canonical order, same order as the scores).
"""
import sys
import numpy as np

SCR = "/private/tmp/claude-501/-Users-damir-damir-research-vault/f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad"
score_txt, arrayname, out_npz = sys.argv[1], sys.argv[2], sys.argv[3]

score = np.loadtxt(score_txt, dtype=np.float64)
cell_idx = np.loadtxt(f"{SCR}/sub_cell_idx.txt", dtype=np.int64).astype(np.int32)
assert score.shape[0] == cell_idx.shape[0], (score.shape, cell_idx.shape)
assert np.all(np.isfinite(score)), "non-finite scores present"

np.savez(out_npz, **{arrayname: score, "cell_idx": cell_idx})
print(f"wrote {out_npz}: {arrayname} n={score.shape[0]} "
      f"range=[{score.min():.4f},{score.max():.4f}] cell_idx n={cell_idx.shape[0]}")
