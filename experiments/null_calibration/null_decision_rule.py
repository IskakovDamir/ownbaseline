#!/usr/bin/env python3
"""
Does the manuscript's DECISION RULE fire under the null?

The pre-registered grid measures the sampling distribution of the point
estimate. The manuscript does not decide on a point estimate. It decides on the
conjunction stated in `conditional_skill_report`'s docstring:

    "call a score substantive only when all three hold -- the two kernels
     agree, the interval excludes zero, and it survives a depth control
     (add log10 library size as a further primitive)."

So "is the estimator biased?" and "does the published rule mistake that bias
for a finding?" are different questions, and only the second one decides
whether the manuscript's verdicts stand. This script measures the second, by
running the shipped `_unit` -- shipped resample counts, shipped bootstrap seed,
shipped verdict vocabulary -- on data whose true conditional skill is exactly
zero, and counting how often each leg of the conjunction fires.

DECLARED AS A DEPARTURE (an addition). It alters no pre-registered cell. It is
a companion to `null_bootstrap.py`, which measures the single-covariate leg
only; this file measures all three legs of the conjunction on the same
replicate, which is what the rule actually requires.

NULL GEOMETRY. Null B as `null_grid.draw` builds it -- the primitive drives the
12-level ordinal, the score tracks the primitive at the target rank
correlation, and the score's own noise is a separate draw -- plus a depth
covariate that also drives the ordinal and is likewise independent of the
score's noise. True conditional skill is exactly 0 against the primitive alone
AND against {primitive, depth} jointly, so every firing counted here is a false
positive.

    python3 experiments/null_calibration/null_decision_rule.py --out dec.csv
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

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.conditional_skill import _unit, align  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from null_grid import _latent_pearson, _ordinal_from_latent  # noqa: E402

RHO_GRID = [0.5, 0.9, 0.998]
STRENGTH = 0.55          # primitive -> ordinal, as in the pre-registered grid
DEPTH_STRENGTH = 0.40    # depth covariate's weight in the ordinal's driver
DRIVE_STRENGTH = 0.60    # rank correlation of that driver with the ordinal
LEVELS = 12
BASE = 20260814
N_BOOT = (300, 1000)     # the shipped published counts
KERNELS = ("scipy_weightedtau", "kang_wdm_taub")


def draw_with_depth(n, rho, rng):
    """Null B plus a depth covariate. True conditional skill is 0 at k=1 and k=2."""
    primitive = rng.standard_normal(n)
    depth = 0.5 * primitive + np.sqrt(1 - 0.25) * rng.standard_normal(n)

    if rho >= 1.0:
        score = np.exp(primitive)
    else:
        rp = _latent_pearson(rho)
        score = rp * primitive + np.sqrt(max(0.0, 1 - rp ** 2)) * rng.standard_normal(n)

    drive = STRENGTH * primitive + DEPTH_STRENGTH * depth
    drive = (drive - drive.mean()) / (drive.std() + 1e-12)
    rp = _latent_pearson(DRIVE_STRENGTH)
    latent = rp * drive + np.sqrt(max(0.0, 1 - rp ** 2)) * rng.standard_normal(n)
    ordinal = _ordinal_from_latent(latent, LEVELS)
    return score, primitive, depth, ordinal


def run_cell(task):
    n, rho, reps, cid = task
    rows = []
    t0 = time.time()
    for j in range(reps):
        rng = np.random.default_rng([BASE, 13, cid, j])
        score, primitive, depth, ordinal = draw_with_depth(n, rho, rng)
        s = align(score, ordinal)
        _, k1 = _unit((f"k1 n={n} rho={rho} rep={j}", s, [primitive],
                       ordinal, N_BOOT))
        _, k2 = _unit((f"k2 n={n} rho={rho} rep={j}", s, [primitive, depth],
                       ordinal, N_BOOT))
        row = {"n": n, "rho_target": rho, "rep": j}
        for tag, res in (("k1", k1), ("k2", k2)):
            for kn in KERNELS:
                e = res[kn]
                row[f"{tag}_{kn}_tau"] = e["tau"]
                row[f"{tag}_{kn}_lo"] = e["CI95"][0]
                row[f"{tag}_{kn}_hi"] = e["CI95"][1]
                row[f"{tag}_{kn}_verdict"] = e["verdict"]
                row[f"{tag}_{kn}_excl0"] = int(e["CI95"][0] > 0 or e["CI95"][1] < 0)
                row[f"{tag}_{kn}_excl0_pos"] = int(e["CI95"][0] > 0)
            row[f"{tag}_kernels_agree"] = int(res["kernels_agree"])
            row[f"{tag}_combined"] = res["combined_verdict"]
        # the manuscript's conjunction, all three legs on the same replicate
        row["rule_fires"] = int(
            row["k1_kernels_agree"]
            and row["k1_scipy_weightedtau_excl0_pos"]
            and row["k1_kang_wdm_taub_excl0_pos"]
            and row["k2_scipy_weightedtau_excl0_pos"]
            and row["k2_kang_wdm_taub_excl0_pos"])
        rows.append(row)
    print(f"[dec] n={n} rho={rho} {reps} reps in {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="null_decision_rule.csv")
    ap.add_argument("--reps-3000", type=int, default=200)
    ap.add_argument("--reps-39505", type=int, default=40)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    tasks, cid = [], 0
    for n, reps in ((3000, args.reps_3000), (39505, args.reps_39505)):
        for rho in RHO_GRID:
            tasks.append((n, rho, reps, cid))
            cid += 1
    tasks.sort(key=lambda t: -t[0])
    print(f"[dec] {len(tasks)} cells, bootstrap {N_BOOT} per fit, "
          f"2 fits per replicate", flush=True)

    t0 = time.time()
    rows = []
    with Pool(args.procs) as pool:
        for cell in pool.imap_unordered(run_cell, tasks):
            rows.extend(cell)

    rows.sort(key=lambda r: (r["n"], r["rho_target"], r["rep"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[dec] wrote {out} ({len(rows)} rows) in {time.time()-t0:.0f}s")

    print(f"\n{'n':>7} {'rho':>6} {'reps':>5} {'taub CI excl 0':>15} "
          f"{'taub CI > 0':>12} {'wtau CI > 0':>12} {'kernels agree':>14} "
          f"{'FULL RULE FIRES':>16}")
    for n in (3000, 39505):
        for rho in RHO_GRID:
            rs = [r for r in rows if r["n"] == n and r["rho_target"] == rho]
            if not rs:
                continue
            f_ = lambda k: sum(r[k] for r in rs) / len(rs)
            print(f"{n:7d} {rho:6.3f} {len(rs):5d} "
                  f"{f_('k1_kang_wdm_taub_excl0'):15.1%} "
                  f"{f_('k1_kang_wdm_taub_excl0_pos'):12.1%} "
                  f"{f_('k1_scipy_weightedtau_excl0_pos'):12.1%} "
                  f"{f_('k1_kernels_agree'):14.1%} "
                  f"{f_('rule_fires'):16.1%}")


if __name__ == "__main__":
    main()
