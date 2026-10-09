#!/usr/bin/env python3
"""
validate_mce.py — step 1.4 of the w5 new-row audit: check scores/mce.py
against analytic answers and against the authors' MCE.m run in GNU Octave,
and time it under MCE.m's own stopping rule.

Reads no stage label. Inputs are the 500-cell SCENT validation subsample
(seed 42, <scratch>/scent_io/adjMC.npz: the network object SR uses and its
cell indices) and the post-QC count matrix of GSE106474, in SR's units,
log2(CPM + 1.1) over the network genes.

Parts, each recorded in discovery/mce_validation.json:

  toy          analytic answers on small networks and a generic constrained
               optimizer (rule="residual", the converged fixed point)
  reference    reference/MCE.m run by octave-cli on the 500 cells; Spearman,
               Pearson and maximum absolute deviation against the port
               (rule="mce_m"), and whether each cell took the same number of
               steps
  timing       wall time of the port on the 500 cells under MCE.m's rule, one
               core, projected linearly to 39,505 cells
  convergence  how far MCE.m's 1e-2 stopping rule leaves the score from the
               fixed point solved to a stationarity residual of 1e-6
  provisional  the per-tolerance trace measured before MCE.m was available,
               carried over unchanged from the earlier file

    python3 experiments/w5_new_rows/validate_mce.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.io as sio
import scipy.sparse as sp
from scipy.stats import pearsonr, spearmanr

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scores"))
sys.path.insert(0, str(REPO / "experiments" / "track2" / "code"))
sys.path.insert(0, str(REPO / "tests"))

from own_baseline.paths import data_root  # noqa: E402
from mce import mce, mce_one, with_self_loops, MCE_M_EPSILON, MCE_M_MAX_ITER  # noqa: E402

OUT = HERE / "discovery" / "mce_validation.json"
REF = REPO / "reference" / "MCE.m"
WORK = data_root() / "w5_new_rows" / "mce_octave"
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
    """Analytic and optimizer checks of the fixed point, solved to convergence."""
    import test_mce as T
    rows = []
    kw = {"rule": "residual"}

    n = 7
    pi = T._pi(n, T.SEED)
    got = mce_one(pi, sp.csr_matrix((n, n)), normalize=False, **kw)
    rows.append({"case": "self-loops only (P = I): MCE = H(pi)", "n_nodes": n,
                 "expected": _H(pi), "got": got, "abs_dev": abs(got - _H(pi))})

    n = 6
    pi = T._pi(n, T.SEED + 1)
    A = sp.csr_matrix(np.ones((n, n)) - np.eye(n))
    got = mce_one(pi, A, normalize=False, **kw)
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
    got = mce_one(d, A, normalize=True, **kw)
    rows.append({"case": "pi proportional to self-loop degree: normalised MCE = 1",
                 "n_nodes": n, "expected": 1.0, "got": got,
                 "abs_dev": abs(got - 1.0)})

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
    got = mce_one(pi, A, normalize=False, **kw)
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


def run_octave(A, Xn):
    """reference/MCE.m on the same cells, through octave-cli."""
    octave = shutil.which("octave-cli") or shutil.which("octave")
    if not REF.is_file():
        return {"status": "NOT RUN", "reason": "reference/MCE.m is absent"}, None, None
    if octave is None:
        return {"status": "NOT RUN", "reason": "GNU Octave is not installed",
                "mce_m_sha256": sha256(REF)}, None, None
    WORK.mkdir(parents=True, exist_ok=True)
    mat = WORK / "mce_input.mat"
    sio.savemat(mat, {"data": np.ascontiguousarray(Xn.T),          # genes x cells
                      "net": sp.csc_matrix(A, dtype=np.float64)}, do_compression=False)
    out = WORK / "mce_octave_ge.txt"
    log = WORK / "mce_octave.log"
    script = (f"addpath('{REF.parent}'); load('{mat}'); t0 = tic; "
              f"ge = MCE(data, net); secs = toc(t0); "
              f"fid = fopen('{out}', 'w'); fprintf(fid, '%.17g\\n', ge); fclose(fid); "
              f"printf('OCTAVE_SECONDS %.3f\\n', secs);")
    t0 = time.perf_counter()
    r = subprocess.run([octave, "--no-gui", "--quiet", "--eval", script],
                       capture_output=True, text=True)
    wall = time.perf_counter() - t0
    log.write_text(r.stdout + r.stderr)
    if r.returncode != 0:
        return {"status": "FAILED", "reason": r.stderr.strip()[-2000:],
                "mce_m_sha256": sha256(REF)}, None, None
    ge = np.loadtxt(out)
    steps = np.array([int(m) for m in re.findall(r"Sample \d+/\d+: (\d+) iterations", r.stdout)])
    secs = float(re.search(r"OCTAVE_SECONDS ([0-9.]+)", r.stdout).group(1))
    version = subprocess.run([octave, "--version"], capture_output=True,
                             text=True).stdout.splitlines()[0]
    return {"status": "RUN", "octave": version, "mce_m_sha256": sha256(REF),
            "seconds_inside_octave": secs, "seconds_wall": wall,
            "n_cells": int(len(ge))}, ge, steps


def main():
    out = {"script": "experiments/w5_new_rows/validate_mce.py",
           "implementation": "scores/mce.py, a port of MCE.m (rule='mce_m'); "
                             "rule='residual' solves the same fixed point to convergence",
           "mce_m_sha256": sha256(REF) if REF.is_file() else None,
           "toy": toy()}
    for r in out["toy"]:
        print(f"[toy] {r['case']}: expected {r['expected']:.12g} got "
              f"{r['got']:.12g} |dev| {r['abs_dev']:.2e}", flush=True)
    if OUT.is_file():
        old = json.loads(OUT.read_text())
        if "timing" in old and "projection_by_tol" in old["timing"]:
            out["provisional"] = {"note": "measured on the equations-based solver before "
                                          "MCE.m was available; superseded by 'timing'",
                                  **old["timing"]}
        elif "provisional" in old:
            out["provisional"] = old["provisional"]

    A, Xn, sub_idx, target_sum = subsample_input()
    out["input"] = ("the 500-cell SCENT validation subsample (seed 42), "
                    f"log2(CPM + 1.1) over the {A.shape[0]:,} network genes of SR's "
                    f"STRING v12 object, CPM target sum = median library size ({target_sum:g})")

    t0 = time.perf_counter()
    port, info = mce(Xn, A, return_info=True)          # rule = MCE.m's
    secs = time.perf_counter() - t0
    per_cell = secs / Xn.shape[0]
    out["timing"] = {
        "rule": "mce_m", "epsilon": MCE_M_EPSILON, "max_iter": MCE_M_MAX_ITER,
        "n_cells": int(Xn.shape[0]), "n_network_genes": int(A.shape[0]),
        "n_edges_without_loops": int(A.nnz // 2), "log_d": info["log_d"],
        "steps_min": int(info["iterations"].min()),
        "steps_median": float(np.median(info["iterations"])),
        "steps_max": int(info["iterations"].max()),
        "seconds": secs, "seconds_per_cell": per_cell,
        "projected_hours_39505_single_core": per_cell * N_FULL / 3600.0,
        "under_8_hours": bool(per_cell * N_FULL < 8 * 3600),
        "machine": f"{platform.machine()} {platform.system()} {platform.release()}",
        "numpy": np.__version__, "scipy": scipy.__version__,
        "python": platform.python_version(),
    }
    print(f"[timing] {Xn.shape[0]} cells in {secs:.1f} s, steps "
          f"{out['timing']['steps_min']}-{out['timing']['steps_max']} (median "
          f"{out['timing']['steps_median']:g}); projected "
          f"{out['timing']['projected_hours_39505_single_core']:.2f} h for {N_FULL:,} cells",
          flush=True)

    ref, ge, steps = run_octave(A, Xn)
    if ge is not None:
        ref.update({
            "spearman_port_vs_mce_m": float(spearmanr(port, ge).statistic),
            "pearson_port_vs_mce_m": float(pearsonr(port, ge)[0]),
            "max_abs_dev": float(np.max(np.abs(port - ge))),
            "steps_identical": bool(len(steps) == len(port)
                                    and np.array_equal(steps, info["iterations"])),
            "n_steps_parsed": int(len(steps)),
        })
        print(f"[reference] MCE.m in {ref['octave']}: Spearman "
              f"{ref['spearman_port_vs_mce_m']:.6f}, max |dev| {ref['max_abs_dev']:.3e}, "
              f"steps identical {ref['steps_identical']}, "
              f"{ref['seconds_inside_octave']:.1f} s in Octave", flush=True)
    else:
        print(f"[reference] {ref['status']}: {ref['reason']}", flush=True)
    out["reference"] = ref

    conv, cinfo = mce(Xn, A, rule="residual", tol=1e-6, return_info=True)
    out["convergence"] = {
        "converged_rule": "residual, tol 1e-6 (the 1e-6 solution was within 1.6e-8 of "
                          "the 1e-8 one in the provisional trace)",
        "converged_steps_max": int(cinfo["iterations"].max()),
        "max_abs_dev_mce_m_vs_converged": float(np.max(np.abs(port - conv))),
        "median_abs_dev": float(np.median(np.abs(port - conv))),
        "spearman_mce_m_vs_converged": float(spearmanr(port, conv).statistic),
        "port_summary": {"min": float(np.min(port)), "median": float(np.median(port)),
                         "max": float(np.max(port))},
    }
    print(f"[convergence] MCE.m rule vs converged: max |dev| "
          f"{out['convergence']['max_abs_dev_mce_m_vs_converged']:.3e}, Spearman "
          f"{out['convergence']['spearman_mce_m_vs_converged']:.6f}", flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
