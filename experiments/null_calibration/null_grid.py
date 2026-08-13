#!/usr/bin/env python3
"""
Null calibration of the conditional-skill estimator.

Pre-registered at 04-experiments/2026-08-14-null-calibration-PREREG.md, vault
commit 18aa63a, BEFORE this script was run.

What is measured: the shipped estimator, unmodified.

    s_aligned = align(score, ordinal)                     # if align is on
    resid     = rank_resid_multi(s_aligned, [primitive])
    tau       = kernel(resid, ordinal)

with `align` and `rank_resid_multi` imported from own_baseline.conditional_skill
exactly as committed. Nothing here corrects, debiases or re-specifies the
estimator. The three kernels are evaluated on the same residual and reported
separately; they are never averaged.

Two nulls, both with true conditional skill exactly 0:

  A  ordinal is independent of both score and primitive.
     Isolates the kernel's own behaviour on an uninformative residual.

  B  ordinal is a monotone function of the primitive plus noise; the score
     depends on the primitive only and is conditionally independent of the
     ordinal given the primitive. The residualization has real work to do.
     This is the case the manuscript's arbiter actually faces.

Usage:
    python3 experiments/null_calibration/null_grid.py --out grid.csv
    python3 experiments/null_calibration/null_grid.py --smoke      # tiny grid
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
from scipy.stats import kendalltau, rankdata, spearmanr, weightedtau

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.conditional_skill import align, rank_resid_multi  # noqa: E402

# ---------------------------------------------------------------- grid (locked)
N_GRID = [500, 3000, 10000, 39505, 127607]
RHO_GRID = [0.0, 0.5, 0.9, 0.93, 0.998, 1.0]      # score-primitive RANK corr
ALIGN_GRID = [True, False]
# Null B: how strongly the primitive explains the ordinal. Spans the range the
# real primitives occupy against the 12-stage ordinal (gene count 0.454,
# PCC(x,degree) 0.516, Shannon 0.629 -- absolute Spearman, reproduced run).
NULLB_STRENGTH = [0.45, 0.55, 0.63]
N_SEEDS = 200
ORDINAL_LEVELS = 12
BASE_ENTROPY = 20260814

KERNELS = ("weightedtau_rankTrue", "weightedtau_rankFalse", "kendalltau")


def _latent_pearson(target_spearman: float) -> float:
    """Gaussian copula: rho_pearson that yields the target Spearman."""
    return 2.0 * np.sin(np.pi * target_spearman / 6.0)


def _ordinal_from_latent(latent: np.ndarray, levels: int) -> np.ndarray:
    """Equal-occupancy binning of a latent variable into ordered levels."""
    q = np.quantile(latent, np.linspace(0, 1, levels + 1)[1:-1])
    return np.searchsorted(q, latent).astype(np.float64)


def draw(null: str, n: int, rho: float, levels: int, strength: float,
         rng: np.random.Generator):
    """One replicate. Returns (score, primitive, ordinal)."""
    primitive = rng.standard_normal(n)

    if rho == 1.0:
        # exact identity in rank space: a strictly monotone map
        score = np.exp(primitive)
    else:
        rp = _latent_pearson(rho)
        score = rp * primitive + np.sqrt(max(0.0, 1.0 - rp ** 2)) * rng.standard_normal(n)

    if null == "A":
        # ordinal independent of everything
        ordinal = _ordinal_from_latent(rng.standard_normal(n), levels)
    elif null == "B":
        # ordinal driven by the primitive; score conditionally independent of
        # the ordinal given the primitive (its noise is a separate draw)
        rp_po = _latent_pearson(strength)
        latent = rp_po * primitive + np.sqrt(max(0.0, 1.0 - rp_po ** 2)) * rng.standard_normal(n)
        ordinal = _ordinal_from_latent(latent, levels)
    else:
        raise ValueError(null)
    return score, primitive, ordinal


def estimate(score, primitive, ordinal, do_align: bool) -> dict:
    """The shipped estimator, then all three kernels on the same residual."""
    s = align(score, ordinal) if do_align else score
    resid = rank_resid_multi(s, [primitive])
    m = np.isfinite(resid) & np.isfinite(ordinal)
    r, o = resid[m], ordinal[m]
    return {
        "weightedtau_rankTrue": float(weightedtau(r, o).statistic),
        "weightedtau_rankFalse": float(weightedtau(r, o, rank=False).statistic),
        "kendalltau": float(kendalltau(r, o).statistic),
    }


def run_cell(task):
    """One grid cell: N_SEEDS replicates. Returns a summary row per kernel."""
    (null, n, rho, do_align, strength, levels, n_seeds, cell_id) = task
    vals = {k: np.empty(n_seeds) for k in KERNELS}
    rho_real = np.empty(n_seeds)
    rho_po = np.empty(n_seeds)
    t0 = time.time()
    for j in range(n_seeds):
        rng = np.random.default_rng([BASE_ENTROPY, cell_id, j])
        score, primitive, ordinal = draw(null, n, rho, levels, strength, rng)
        out = estimate(score, primitive, ordinal, do_align)
        for k in KERNELS:
            vals[k][j] = out[k]
        rho_real[j] = spearmanr(score, primitive).statistic
        rho_po[j] = spearmanr(primitive, ordinal).statistic

    rows = []
    for k in KERNELS:
        v = vals[k]
        finite = v[np.isfinite(v)]
        mean = float(np.mean(finite)) if len(finite) else float("nan")
        sd = float(np.std(finite, ddof=1)) if len(finite) > 1 else float("nan")
        mcse = sd / np.sqrt(len(finite)) if len(finite) > 1 else float("nan")
        rows.append({
            "null": null, "n": n, "rho_target": rho, "align": int(do_align),
            "nullB_strength": strength if null == "B" else "",
            "levels": levels, "kernel": k, "n_seeds": n_seeds,
            "n_finite": int(len(finite)),
            "mean": mean, "sd": sd, "mcse": mcse,
            "p2.5": float(np.percentile(finite, 2.5)) if len(finite) else float("nan"),
            "p97.5": float(np.percentile(finite, 97.5)) if len(finite) else float("nan"),
            "min": float(np.min(finite)) if len(finite) else float("nan"),
            "max": float(np.max(finite)) if len(finite) else float("nan"),
            "mean_is_zero": int(abs(mean) <= 2 * mcse) if np.isfinite(mcse) else "",
            "rho_realized_mean": float(np.mean(rho_real)),
            "rho_prim_ordinal_mean": float(np.mean(rho_po)),
            "seconds": round(time.time() - t0, 1),
            "cell_id": cell_id,
        })
    return rows, vals


def build_tasks(n_seeds, smoke=False):
    n_grid = [500, 3000] if smoke else N_GRID
    rho_grid = [0.0, 0.998, 1.0] if smoke else RHO_GRID
    strengths = [0.55] if smoke else NULLB_STRENGTH
    tasks, cid = [], 0
    for null in ("A", "B"):
        for n in n_grid:
            for rho in rho_grid:
                for al in ALIGN_GRID:
                    for st in (strengths if null == "B" else [0.0]):
                        tasks.append((null, n, rho, al, st, ORDINAL_LEVELS,
                                      n_seeds, cid))
                        cid += 1
    # secondary: ordinal level sweep at n = 3000
    if not smoke:
        for null in ("A", "B"):
            for levels in (2, 4, 50):          # 12 already covered above
                for rho in rho_grid:
                    for al in ALIGN_GRID:
                        st = 0.55 if null == "B" else 0.0
                        tasks.append((null, 3000, rho, al, st, levels,
                                      n_seeds, cid))
                        cid += 1
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="null_grid.csv")
    ap.add_argument("--raw", default=None, help="optional .npz of every replicate")
    ap.add_argument("--seeds", type=int, default=N_SEEDS)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    tasks = build_tasks(args.seeds if not args.smoke else 20, smoke=args.smoke)
    # longest cells first, so the tail does not straggle
    tasks.sort(key=lambda t: -t[1])
    print(f"[null-grid] {len(tasks)} cells x {tasks[0][6]} seeds "
          f"= {sum(t[6] for t in tasks):,} replicates on {args.procs} procs",
          flush=True)

    t0 = time.time()
    rows, raw = [], {}
    with Pool(args.procs) as pool:
        for i, (cell_rows, vals) in enumerate(
                pool.imap_unordered(run_cell, tasks), 1):
            rows.extend(cell_rows)
            if args.raw:
                cid = cell_rows[0]["cell_id"]
                for k in KERNELS:
                    raw[f"{cid}_{k}"] = vals[k]
            if i % 20 == 0 or i == len(tasks):
                print(f"[null-grid] {i}/{len(tasks)} cells "
                      f"({time.time()-t0:.0f}s)", flush=True)

    rows.sort(key=lambda r: (r["null"], r["levels"], r["n"], r["rho_target"],
                             -r["align"], str(r["nullB_strength"]), r["kernel"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[null-grid] wrote {out} ({len(rows)} rows) in {time.time()-t0:.0f}s")

    if args.raw:
        np.savez_compressed(args.raw, **raw)
        print(f"[null-grid] wrote {args.raw} ({len(raw)} arrays)")


if __name__ == "__main__":
    main()
