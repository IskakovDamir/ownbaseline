#!/usr/bin/env python3
"""
score_mce.py — MCE (scores/mce.py, the validated port of MCE.m) on all
39,505 cells of GSE106474, on SR's network object and SR's input, as
PREREG.md's MCE amendment registers it. Reads no label.

    python3 experiments/w5_new_rows/score_mce.py [--procs 8]

Input per cell: log2(CPM + 1.1) over the network genes of
<scratch>/scent_io/adjMC.npz (the object run_scent_full.py computes SR on),
CPM target sum = median library size, exactly as validate_mce.py feeds both
the port and MCE.m. Every cell stops by MCE.m's own rule, so how cells are
chunked across processes changes no value (tests/test_mce.py).

Writes $OWNBASELINE_DATA_ROOT/w5_new_rows/scores/GSE106474/mce.csv (all cells
in the prepared input's order) and mce_info.txt.
"""
from __future__ import annotations

import argparse
import csv
import platform
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scores"))
sys.path.insert(0, str(REPO / "experiments" / "track2" / "code"))
from own_baseline.paths import data_root  # noqa: E402
from mce import mce, MCE_M_EPSILON, MCE_M_MAX_ITER  # noqa: E402

CHUNK = 500
_STATE = {}


def _init():
    from run_scent_full import load_net
    A, degree, mc_col, target_sum, _ = load_net()
    X = sp.load_npz(data_root() / "track2/data/prepared/X_cells_genes.npz").tocsr()
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    _STATE.update(A=A, mc_col=mc_col, target_sum=target_sum, X=X, lib=lib)


def _block(s0):
    st = _STATE
    idx = np.arange(s0, min(s0 + CHUNK, st["X"].shape[0]))
    counts = st["X"][idx][:, st["mc_col"]].toarray().astype(np.float64)
    tot = st["lib"][idx][:, None]
    Xn = np.log2(counts / np.where(tot > 0, tot, 1.0) * st["target_sum"] + 1.1)
    vals, info = mce(Xn, st["A"], return_info=True, chunk=CHUNK)
    return s0, vals, info["iterations"], info["err"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--procs", type=int, default=8)
    args = ap.parse_args()
    d = data_root() / "w5_new_rows" / "inputs" / "GSE106474"
    cells = d.joinpath("cells.txt").read_text().split("\n")[:-1]
    n = sp.load_npz(data_root() / "track2/data/prepared/X_cells_genes.npz").shape[0]
    if n != len(cells):
        raise SystemExit("prepared matrix and input cell list differ")
    out = np.full(n, np.nan)
    steps = np.zeros(n, dtype=np.int64)
    errs = np.full(n, np.nan)
    t0 = time.perf_counter()
    with Pool(args.procs, initializer=_init) as pool:
        for k, (s0, vals, it, err) in enumerate(pool.imap_unordered(_block, range(0, n, CHUNK)), 1):
            out[s0:s0 + len(vals)] = vals
            steps[s0:s0 + len(vals)] = it
            errs[s0:s0 + len(vals)] = err
            if k % 10 == 0:
                print(f"[mce] {k} blocks, {time.perf_counter() - t0:.0f} s", flush=True)
    secs = time.perf_counter() - t0
    dest = data_root() / "w5_new_rows" / "scores" / "GSE106474"
    dest.mkdir(parents=True, exist_ok=True)
    with open(dest / "mce.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "mce"])
        for c, v in zip(cells, out):
            w.writerow([c, repr(float(v))])
    (dest / "mce_info.txt").write_text("\n".join([
        f"seconds\t{secs:.1f}", f"procs\t{args.procs}", f"n_cells\t{n}",
        f"rule\tmce_m (epsilon {MCE_M_EPSILON:g}, max_iter {MCE_M_MAX_ITER})",
        f"steps_min\t{int(steps.min())}", f"steps_median\t{float(np.median(steps)):g}",
        f"steps_max\t{int(steps.max())}", f"err_max\t{float(np.nanmax(errs)):.3e}",
        f"n_nan\t{int(np.isnan(out).sum())}",
        f"numpy\t{np.__version__}", f"scipy\t{scipy.__version__}",
        f"python\t{platform.python_version()}"]) + "\n")
    print(f"MCE GSE106474: {n} cells in {secs:.0f} s on {args.procs} processes, steps "
          f"{int(steps.min())}-{int(steps.max())}, NaN {int(np.isnan(out).sum())}")
    print(f"wrote {dest / 'mce.csv'}")


if __name__ == "__main__":
    main()
