#!/usr/bin/env python3
"""Assemble a Track-4 score npz from an R score text file + the subsample cell_idx.
Usage: write_npz.py <score_txt> <arrayname> <out_npz>
Writes: array `<arrayname>` (per-cell score, subsample order) and `cell_idx`
(int32 indices into cells.tsv canonical order, same order as the scores).
"""
import sys
import numpy as np

# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# scratch directory, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import scratch_root  # noqa: E402
SCR = str(scratch_root())
score_txt, arrayname, out_npz = sys.argv[1], sys.argv[2], sys.argv[3]

score = np.loadtxt(score_txt, dtype=np.float64)
cell_idx = np.loadtxt(f"{SCR}/sub_cell_idx.txt", dtype=np.int64).astype(np.int32)
assert score.shape[0] == cell_idx.shape[0], (score.shape, cell_idx.shape)
assert np.all(np.isfinite(score)), "non-finite scores present"

np.savez(out_npz, **{arrayname: score, "cell_idx": cell_idx})
print(f"wrote {out_npz}: {arrayname} n={score.shape[0]} "
      f"range=[{score.min():.4f},{score.max():.4f}] cell_idx n={cell_idx.shape[0]}")
