#!/usr/bin/env python3
"""
validate_mce.py — step 1.4 of the w5 new-row audit: check scores/mce.py, time
it, and compare it with the authors' MCE.m in GNU Octave when that file is
present.

Reads no stage label. Inputs are the 500-cell SCENT validation subsample
(seed 42, <scratch>/scent_io/adjMC.npz: the network object SR uses and its
cell indices) and the post-QC count matrix of GSE106474.

Three parts, each recorded in the output JSON:

  toy        analytic answers on small networks (static-entropy limit,
             complete-graph limit, maximum at pi proportional to degree) and
             a generic constrained optimizer on a random graph
  timing     wall time of scores/mce.py on the 500 cells, the input SR uses
             (log2(CPM + 1.1) over the network genes), projected linearly to
             39,505 cells
  reference  MCE.m run by Octave on the same 500 cells, Spearman and max
             absolute deviation against the port; or the reason it did not run

    python3 experiments/w5_new_rows/validate_mce.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scores"))
sys.path.insert(0, str(REPO / "experiments" / "track2" / "code"))
sys.path.insert(0, str(REPO / "tests"))

from own_baseline.paths import data_root, scratch_root  # noqa: E402
from mce import mce, mce_one, with_self_loops  # noqa: E402

OUT = HERE / "discovery" / "mce_validation.json"
REF = REPO / "reference" / "MCE.m"
N_FULL = 39505


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _H(p):
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def toy():
    """Analytic and optimizer checks; returns value pairs and deviations."""
    import test_mce as T
    rng = np.random.default_rng(T.SEED)
    rows = []

    n = 7
    pi = T._pi(n, T.SEED)
    got = mce_one(pi, sp.csr_matrix((n, n)), normalize=False)
    rows.append({"case": "self-loops only (P = I): MCE = H(pi)", "n_nodes": n,
                 "expected": _H(pi), "got": got, "abs_dev": abs(got - _H(pi))})

    n = 6
    pi = T._pi(n, T.SEED + 1)
    A = sp.csr_matrix(np.ones((n, n)) - np.eye(n))
    got = mce_one(pi, A, normalize=False)
    rows.append({"case": "complete graph: MCE = 2 H(pi)", "n_nodes": n,
                 "expected": 2 * _H(pi), "got": got,
                 "abs_dev": abs(got - 2 * _H(pi))})

    rng = np.random.default_rng(T.SEED + 2)
    n = 30
    M = np.triu((rng.random((n, n)) < 0.15).astype(float), 1)
    for i in range(n - 1):
        M[i, i + 1] = 1.0
    A = sp.csr_matrix(M + M.T)
    d = np.asarray(with_self_loops(A).sum(axis=1)).ravel()
    got = mce_one(d, A, normalize=True)
    rows.append({"case": "pi proportional to self-loop degree: normalised MCE = 1",
                 "n_nodes": n, "expected": 1.0, "got": got,
                 "abs_dev": abs(got - 1.0)})

    # the optimizer comparison, rerun here so its numbers are recorded
    from scipy.optimize import minimize
    rng = np.random.default_rng(T.SEED + 3)
    n = 6
    M = np.triu((rng.random((n, n)) < 0.4).astype(float), 1)
    for i in range(n - 1):
        M[i, i + 1] = 1.0
    A = sp.csr_matrix(M + M.T)
    AL = with_self_loops(A).toarray()
    edges = np.argwhere(AL > 0)
    pi = T._pi(n, T.SEED + 4)
    cons = []
    for i in range(n):
        r = np.where(edges[:, 0] == i)[0]
        cons.append({"type": "eq", "fun": lambda q, r=r, i=i: q[r].sum() - pi[i]})
    for i in range(n - 1):
        c = np.where(edges[:, 1] == i)[0]
        cons.append({"type": "eq", "fun": lambda q, c=c, i=i: q[c].sum() - pi[i]})
    q0 = np.array([pi[i] / AL[i].sum() for i, _ in edges])
    res = minimize(lambda q: float((np.clip(q, 1e-300, None)
                                    * np.log(np.clip(q, 1e-300, None))).sum()),
                   q0, jac=lambda q: np.log(np.clip(q, 1e-300, None)) + 1.0,
                   constraints=cons, method="SLSQP", bounds=[(0, 1)] * len(edges),
                   options={"ftol": 1e-15, "maxiter": 2000})
    got = mce_one(pi, A, normalize=False)
    rows.append({"case": "random 6-node graph vs SLSQP on Eq 3", "n_nodes": n,
                 "expected": float(-res.fun), "got": got,
                 "abs_dev": abs(got - float(-res.fun)),
                 "optimizer_success": bool(res.success)})
    return rows


def subsample_input():
    """The 500 cells and the network SR was validated on, in SR's input units."""
    from run_scent_full import load_net
    A, degree, mc_col, target_sum, sub_idx = load_net()
    X = sp.load_npz(data_root() / "track2/data/prepared/X_cells_genes.npz").tocsr()
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    counts = X[sub_idx][:, mc_col].toarray().astype(np.float64)
    tot = lib[sub_idx][:, None]
    cpm = counts / np.where(tot > 0, tot, 1.0) * target_sum
    Xn = np.log2(cpm + 1.1)          # DoIntegPPI convention, as SR
    return A, Xn, sub_idx, target_sum


TOLS = (1e-6, 1e-8, 1e-10, 1e-12)
N_TRACE = 50          # cells whose convergence is traced to the tightest tol
TRACE_CAP = 20000     # iteration cap for the trace
N_ITER_TIMED = 50     # iterations timed on the full 500-cell block


def timing(A, Xn):
    """
    The stopping rule of MCE.m is not known, so the cost is measured in two
    parts and projected per tolerance: seconds per iteration per cell on the
    full 500-cell block, and iterations to each tolerance on the first 50
    cells. The projection assumes 500-cell chunks iterate as long as their
    slowest cell, which the 50-cell trace can underestimate.
    """
    from mce import _scale
    AL = with_self_loops(A)
    Pi = Xn / Xn.sum(axis=1, keepdims=True)

    t0 = time.perf_counter()
    _scale(Pi, AL, tol=0.0, max_iter=N_ITER_TIMED, check_every=N_ITER_TIMED + 1)
    secs = time.perf_counter() - t0
    per_iter_cell = secs / N_ITER_TIMED / Xn.shape[0]

    hist = []
    t1 = time.perf_counter()
    _scale(Pi[:N_TRACE], AL, tol=TOLS[-1], max_iter=TRACE_CAP, check_every=10,
           history=hist)
    trace_secs = time.perf_counter() - t1
    reached = {}
    for tol in TOLS:
        hit = next((it for it, err in hist if err < tol), None)
        reached[f"{tol:g}"] = hit

    vals = {}
    for tol in TOLS:
        if reached[f"{tol:g}"] is not None:
            vals[f"{tol:g}"] = mce(Xn[:N_TRACE], A, tol=tol, max_iter=TRACE_CAP)
    ref_key = next((f"{t:g}" for t in reversed(TOLS) if f"{t:g}" in vals), None)
    dev_vs_tightest = {k: float(np.max(np.abs(v - vals[ref_key])))
                       for k, v in vals.items()} if ref_key else {}

    proj = {}
    for k, it in reached.items():
        if it is not None:
            s = it * per_iter_cell * N_FULL
            proj[k] = {"iterations": it, "projected_hours_39505": s / 3600.0,
                       "under_8_hours": bool(s < 8 * 3600)}
    return {
        "n_cells_timed": int(Xn.shape[0]), "n_network_genes": int(A.shape[0]),
        "n_edges_without_loops": int(A.nnz // 2),
        "iterations_timed": N_ITER_TIMED, "seconds_timed": secs,
        "seconds_per_iteration_per_cell": per_iter_cell,
        "trace_cells": N_TRACE, "trace_cap": TRACE_CAP,
        "trace_seconds": trace_secs,
        "trace_final": {"iteration": hist[-1][0], "residual": hist[-1][1]} if hist else None,
        "iterations_to_tol": reached,
        "mce_max_abs_dev_vs_tightest_reached_tol": {"reference_tol": ref_key,
                                                    **dev_vs_tightest},
        "projection_by_tol": proj,
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "numpy": np.__version__, "scipy": scipy.__version__,
        "python": platform.python_version(),
    }, (vals[ref_key] if ref_key else None)


def reference(A, Xn, port_vals):
    """Run MCE.m in Octave on the same cells, if both exist."""
    octave = shutil.which("octave-cli") or shutil.which("octave")
    if not REF.is_file():
        return {"status": "NOT RUN",
                "reason": "reference/MCE.m is absent: the supplement of Shi et "
                          "al. 2020 could not be downloaded in this session "
                          "(see DISCOVERY.md). Waiting for the author.",
                "octave": (subprocess.run([octave, "--version"], capture_output=True,
                                          text=True).stdout.splitlines()[0]
                           if octave else None)}
    if octave is None:
        return {"status": "NOT RUN", "reason": "GNU Octave is not installed",
                "mce_m_sha256": sha256(REF)}
    # MCE.m's calling convention is only known once the file is read; the
    # driver below is filled in at that point (amendment recorded in PREREG.md)
    return {"status": "NOT RUN",
            "reason": "MCE.m is present but no Octave driver has been written "
                      "for its calling convention yet",
            "mce_m_sha256": sha256(REF)}


def main():
    if "--reference-only" in sys.argv:
        # refresh only the MCE.m comparison block; the toy checks and the timing
        # in the existing file are kept as they were measured. Once MCE.m is in
        # place the Octave comparison needs the full run, not this.
        out = json.loads(OUT.read_text())
        if REF.is_file():
            raise SystemExit("MCE.m is present: run the full validation")
        out["reference"] = reference(None, None, None)
        OUT.write_text(json.dumps(out, indent=2) + "\n")
        print(f"[reference] {out['reference']['status']}: {out['reference']['reason']}")
        print(f"wrote {OUT}")
        return
    out = {"script": "experiments/w5_new_rows/validate_mce.py",
           "implementation": "scores/mce.py (from the published equations; "
                             "not yet a port of MCE.m)",
           "toy": toy()}
    for r in out["toy"]:
        print(f"[toy] {r['case']}: expected {r['expected']:.12g} got "
              f"{r['got']:.12g} |dev| {r['abs_dev']:.2e}", flush=True)
    A, Xn, sub_idx, target_sum = subsample_input()
    t, vals = timing(A, Xn)
    t["input"] = ("log2(CPM + 1.1) over the network genes of SR's STRING v12 "
                  f"object, CPM target sum = median library size ({target_sum:g})")
    if vals is not None:
        t["mce_summary_trace_cells"] = {"min": float(np.min(vals)),
                                        "median": float(np.median(vals)),
                                        "max": float(np.max(vals)),
                                        "n_nan": int(np.isnan(vals).sum())}
    out["timing"] = t
    out["reference"] = reference(A, Xn, vals)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"[timing] {t['seconds_per_iteration_per_cell']:.3e} s per iteration "
          f"per cell ({t['n_cells_timed']} cells, {t['iterations_timed']} "
          f"iterations); trace on {t['trace_cells']} cells ended at iteration "
          f"{t['trace_final']['iteration']} residual {t['trace_final']['residual']:.1e}")
    for k, it in t["iterations_to_tol"].items():
        p = t["projection_by_tol"].get(k)
        print(f"[timing]   tol {k}: iterations {it}"
              + (f", projected {p['projected_hours_39505']:.2f} h for {N_FULL:,} "
                 f"cells, max |MCE - tightest| {t['mce_max_abs_dev_vs_tightest_reached_tol'].get(k, float('nan')):.1e}"
                 if p else ", not reached within the cap"))
    print(f"[reference] {out['reference']['status']}: {out['reference'].get('reason', '')}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
