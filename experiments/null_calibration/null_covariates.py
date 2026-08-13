#!/usr/bin/env python3
"""
Supplementary null: does the bias grow with the NUMBER of covariates removed?

Declared as an addition to the pre-registered grid (see the results file,
"Departures"). The pre-registered grid residualizes on one primitive. The
manuscript also reports:

  - a depth-robustness pass on 2 covariates (primitive + log10 library size)
  - a joint pass on 4 covariates (gene count, PCC(x,degree), Shannon, log lib)

If the null bias grows with the number of covariates, those two passes have a
different null from the single-primitive rows, and cannot be read against the
single-primitive null cells. This script measures that, and nothing else.

Null B geometry throughout: the covariates jointly drive the ordinal, and the
score is conditionally independent of the ordinal given the covariates, so the
true conditional skill is exactly 0 at every k.

    python3 experiments/null_calibration/null_covariates.py --out cov.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, weightedtau

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.conditional_skill import align, rank_resid_multi  # noqa: E402

N_GRID = [3000, 39505]
K_GRID = [1, 2, 4]
RHO_GRID = [0.0, 0.5, 0.9, 0.93, 0.998]
N_SEEDS = 200
LEVELS = 12
STRENGTH = 0.55
BASE = 20260814
KERNELS = ("weightedtau_rankTrue", "weightedtau_rankFalse", "kendalltau")


def _lp(r):
    return 2.0 * np.sin(np.pi * r / 6.0)


def run_cell(task):
    n, k, rho, n_seeds, cid = task
    vals = {kk: np.empty(n_seeds) for kk in KERNELS}
    t0 = time.time()
    for j in range(n_seeds):
        rng = np.random.default_rng([BASE, 7, cid, j])
        # k covariates, mildly correlated with one another
        base = rng.standard_normal(n)
        covs = [0.6 * base + 0.8 * rng.standard_normal(n) for _ in range(k)]
        primitive = covs[0]
        # the covariates jointly drive the ordinal
        drive = np.mean(covs, axis=0)
        drive = (drive - drive.mean()) / (drive.std() + 1e-12)
        rp = _lp(STRENGTH)
        latent = rp * drive + np.sqrt(max(0.0, 1 - rp ** 2)) * rng.standard_normal(n)
        q = np.quantile(latent, np.linspace(0, 1, LEVELS + 1)[1:-1])
        ordinal = np.searchsorted(q, latent).astype(float)
        # score tracks the primitive; its own noise is independent of the ordinal
        if rho >= 1.0:
            score = np.exp(primitive)
        else:
            r = _lp(rho)
            score = r * primitive + np.sqrt(max(0.0, 1 - r ** 2)) * rng.standard_normal(n)
        s = align(score, ordinal)
        resid = rank_resid_multi(s, covs)
        vals["weightedtau_rankTrue"][j] = weightedtau(resid, ordinal).statistic
        vals["weightedtau_rankFalse"][j] = weightedtau(resid, ordinal, rank=False).statistic
        vals["kendalltau"][j] = kendalltau(resid, ordinal).statistic

    rows = []
    for kk in KERNELS:
        v = vals[kk][np.isfinite(vals[kk])]
        mcse = float(np.std(v, ddof=1) / np.sqrt(len(v)))
        rows.append({
            "n": n, "k_covariates": k, "rho_target": rho, "kernel": kk,
            "n_seeds": n_seeds, "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)), "mcse": mcse,
            "p2.5": float(np.percentile(v, 2.5)),
            "p97.5": float(np.percentile(v, 97.5)),
            "mean_is_zero": int(abs(float(np.mean(v))) <= 2 * mcse),
            "seconds": round(time.time() - t0, 1),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="null_covariates.csv")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    tasks, cid = [], 0
    for n in N_GRID:
        for k in K_GRID:
            for rho in RHO_GRID:
                tasks.append((n, k, rho, args.seeds, cid))
                cid += 1
    tasks.sort(key=lambda t: -t[0])
    print(f"[cov] {len(tasks)} cells x {args.seeds} seeds", flush=True)

    rows = []
    t0 = time.time()
    with Pool(args.procs) as pool:
        for i, cell in enumerate(pool.imap_unordered(run_cell, tasks), 1):
            rows.extend(cell)
            if i % 5 == 0 or i == len(tasks):
                print(f"[cov] {i}/{len(tasks)} ({time.time()-t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: (r["n"], r["k_covariates"], r["rho_target"], r["kernel"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[cov] wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
