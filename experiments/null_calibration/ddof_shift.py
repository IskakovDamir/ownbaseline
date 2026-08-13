#!/usr/bin/env python3
"""
Quantify the ddof defect: how much does the cov/var slope inflation move a
reported statistic, as a function of n?

    rank_resid_multi   beta = lstsq([rank(p), 1], rank(s))          <- unaffected
    rank_residual      beta = np.cov(r_s, r_p)[0,1] / np.var(r_p)   <- affected

np.cov defaults to ddof=1 and np.var to ddof=0, so the second slope is the
first times exactly n/(n-1). This script measures the resulting shift in the
reported tau, under both kernels, across n and across the score-primitive
rank correlation.

It measures. It does not fix anything.

    python3 experiments/null_calibration/ddof_shift.py --out ddof.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr, weightedtau

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.conditional_skill import align, rank_resid_multi  # noqa: E402

N_GRID = [50, 100, 200, 500, 1000, 3000, 5000, 10000, 39505, 127607]
RHO_GRID = [0.0, 0.45, 0.48, 0.5, 0.9, 0.93, 0.932, 0.998]
N_SEEDS = 200
LEVELS = 12
BASE = 20260814


def rank_residual_covvar(score, primitive):
    """Verbatim copy of the affected implementation. Do not tidy it."""
    r_s = rankdata(score)
    r_p = rankdata(primitive)
    beta = np.cov(r_s, r_p)[0, 1] / np.var(r_p) if np.var(r_p) > 0 else 0.0
    return r_s - beta * r_p


def rank_residual_covvar_ddof0(score, primitive):
    """The same, with the ddof mismatch removed. For measurement only."""
    r_s = rankdata(score)
    r_p = rankdata(primitive)
    v = np.var(r_p)
    beta = np.cov(r_s, r_p, ddof=0)[0, 1] / v if v > 0 else 0.0
    return r_s - beta * r_p


def _latent_pearson(rho_s):
    return 2.0 * np.sin(np.pi * rho_s / 6.0)


def draw(n, rho, rng, strength=0.55):
    """Realistic geometry: primitive drives the ordinal, score tracks the primitive."""
    primitive = rng.standard_normal(n)
    if rho >= 1.0:
        score = np.exp(primitive)
    else:
        rp = _latent_pearson(rho)
        score = rp * primitive + np.sqrt(max(0.0, 1 - rp ** 2)) * rng.standard_normal(n)
    rpo = _latent_pearson(strength)
    latent = rpo * primitive + np.sqrt(max(0.0, 1 - rpo ** 2)) * rng.standard_normal(n)
    q = np.quantile(latent, np.linspace(0, 1, LEVELS + 1)[1:-1])
    ordinal = np.searchsorted(q, latent).astype(float)
    # a real score also carries signal beyond the primitive, or there would be
    # nothing to shift; add a component independent of the primitive
    score = score + 0.6 * (ordinal - ordinal.mean()) / (ordinal.std() + 1e-12)
    return score, primitive, ordinal


def taus(resid, ordinal):
    return (float(weightedtau(resid, ordinal).statistic),
            float(kendalltau(resid, ordinal).statistic))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ddof_shift.csv")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    args = ap.parse_args()

    rows = []
    for n in N_GRID:
        for rho in RHO_GRID:
            d_w, d_k, d_lstsq_w, d_lstsq_k, rho_real = [], [], [], [], []
            for j in range(args.seeds):
                rng = np.random.default_rng([BASE, n, int(rho * 1000), j])
                s, p, o = draw(n, rho, rng)
                sa = align(s, o)
                r_bad = rank_residual_covvar(sa, p)
                r_fix = rank_residual_covvar_ddof0(sa, p)
                r_ls = rank_resid_multi(sa, [p])
                bw, bk = taus(r_bad, o)
                fw, fk = taus(r_fix, o)
                lw, lk = taus(r_ls, o)
                d_w.append(bw - fw)
                d_k.append(bk - fk)
                d_lstsq_w.append(bw - lw)
                d_lstsq_k.append(bk - lk)
                rho_real.append(spearmanr(s, p).statistic)
            for kern, d, dl in (("weightedtau", d_w, d_lstsq_w),
                                ("kendalltau", d_k, d_lstsq_k)):
                d = np.asarray(d)
                dl = np.asarray(dl)
                rows.append({
                    "n": n, "rho_target": rho, "kernel": kern,
                    "n_seeds": args.seeds,
                    "slope_inflation": n / (n - 1),
                    "mean_shift_vs_ddof0": float(np.mean(d)),
                    "mean_abs_shift_vs_ddof0": float(np.mean(np.abs(d))),
                    "max_abs_shift_vs_ddof0": float(np.max(np.abs(d))),
                    "p97.5_abs_shift_vs_ddof0": float(np.percentile(np.abs(d), 97.5)),
                    "mean_abs_shift_vs_lstsq": float(np.mean(np.abs(dl))),
                    "max_abs_shift_vs_lstsq": float(np.max(np.abs(dl))),
                    "rho_realized_mean": float(np.mean(rho_real)),
                })
            print(f"  n={n:7d} rho={rho:.3f}  "
                  f"|shift| wtau mean {np.mean(np.abs(d_w)):.2e} max {np.max(np.abs(d_w)):.2e} | "
                  f"ktau mean {np.mean(np.abs(d_k)):.2e} max {np.max(np.abs(d_k)):.2e}",
                  flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
