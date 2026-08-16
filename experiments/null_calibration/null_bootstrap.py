#!/usr/bin/env python3
"""
Does the shipped decision rule fire under the null?

The pre-registered grid measures the sampling distribution of the point
estimate. The manuscript does not decide on the point estimate: it decides on
`_unit`'s bootstrap interval, with the verdict vocabulary

    CI includes 0 -> TAUTOLOG ; CI strictly > 0 -> ADDS-BEYOND

so the question "does the interval exclude zero under the null?" is not read
off the grid without an assumption about the bootstrap standard error. This
script measures it directly, by calling the shipped `_unit` on null data.

DECLARED AS A DEPARTURE (an addition). No pre-registered cell is altered or
dropped. The estimator, the residual, the resample count, the fixed bootstrap
seed and the verdict function are the shipped ones, imported and called; this
file constructs null data and counts verdicts, and does nothing else.

Null B geometry, exactly as null_grid.py draws it: the primitive drives the
12-level ordinal at Spearman ~0.55, the score tracks the primitive at the
target rank correlation, and the score's own noise is a separate draw, so the
true conditional skill is exactly 0.

    python3 experiments/null_calibration/null_bootstrap.py --out boot.csv
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
from null_grid import draw  # noqa: E402  (the same null construction, unmodified)

N_GRID = [3000, 39505]
RHO_GRID = [0.5, 0.9, 0.998]
N_REPS = 100                     # this arm only; the pre-registered grid is 200
STRENGTH = 0.55
LEVELS = 12
BASE = 20260814
N_BOOT = (300, 1000)             # the shipped published counts


def run_cell(task):
    n, rho, reps, cid = task
    rows = []
    t0 = time.time()
    for j in range(reps):
        rng = np.random.default_rng([BASE, 11, cid, j])
        score, primitive, ordinal = draw("B", n, rho, LEVELS, STRENGTH, rng)
        s = align(score, ordinal)
        _, out = _unit((f"null n={n} rho={rho} rep={j}", s, [primitive],
                        ordinal, N_BOOT))
        for kname in ("scipy_weightedtau", "kang_wdm_taub"):
            e = out[kname]
            rows.append({
                "n": n, "rho_target": rho, "rep": j, "kernel": kname,
                "tau": e["tau"], "ci_lo": e["CI95"][0], "ci_hi": e["CI95"][1],
                "verdict": e["verdict"],
                "excludes_zero": int(e["CI95"][0] > 0 or e["CI95"][1] < 0),
                "excludes_zero_positive": int(e["CI95"][0] > 0),
                "n_boot": e["n_boot"],
            })
    print(f"[boot] n={n} rho={rho} done in {time.time()-t0:.0f}s", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="null_bootstrap.csv")
    ap.add_argument("--reps", type=int, default=N_REPS)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    tasks, cid = [], 0
    for n in N_GRID:
        for rho in RHO_GRID:
            tasks.append((n, rho, args.reps, cid))
            cid += 1
    tasks.sort(key=lambda t: -t[0])
    print(f"[boot] {len(tasks)} cells x {args.reps} reps, "
          f"bootstrap {N_BOOT} per rep", flush=True)

    t0 = time.time()
    rows = []
    with Pool(args.procs) as pool:
        for cell in pool.imap_unordered(run_cell, tasks):
            rows.extend(cell)

    rows.sort(key=lambda r: (r["n"], r["rho_target"], r["kernel"], r["rep"]))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[boot] wrote {out} ({len(rows)} rows) in {time.time()-t0:.0f}s")

    # summary to stdout; the CSV is the record
    print(f"\n{'n':>7} {'rho':>6} {'kernel':20} {'reps':>5} "
          f"{'CI excl 0':>10} {'CI excl 0 (+)':>14} {'ADDS-BEYOND':>12}")
    for n in N_GRID:
        for rho in RHO_GRID:
            for k in ("kang_wdm_taub", "scipy_weightedtau"):
                rs = [r for r in rows if r["n"] == n and r["rho_target"] == rho
                      and r["kernel"] == k]
                if not rs:
                    continue
                e = sum(r["excludes_zero"] for r in rs) / len(rs)
                p = sum(r["excludes_zero_positive"] for r in rs) / len(rs)
                a = sum(r["verdict"] == "ADDS-BEYOND" for r in rs) / len(rs)
                print(f"{n:7d} {rho:6.3f} {k:20} {len(rs):5d} "
                      f"{e:10.1%} {p:14.1%} {a:12.1%}")


if __name__ == "__main__":
    main()
