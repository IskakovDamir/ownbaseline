#!/usr/bin/env python3
"""
Run named cells of the pre-registered grid, with the pre-registered seeding.

WHY THIS EXISTS. The 2026-08-16 driver built the full 312-cell task list and
worked through it longest-first; it was stopped after 36 of the 48 n = 127,607
cells. The remaining sample sizes were completed later with `null_grid.py --n`,
which rebuilds a SHORTER task list and therefore renumbers `cell_id`. Since the
seed is `default_rng([BASE_ENTROPY, cell_id, seed_index])`, a `--n` run cannot
reproduce the seed stream a full-grid cell would have used, and `--n 127607`
would give the 12 unfinished cells different seeds from their 36 completed
siblings.

This driver takes cell ids from the FULL 312-cell task list -- the same list the
2026-08-16 driver used -- so the cells it runs carry exactly the seeds they
would have carried had that run not been interrupted.

It imports `build_tasks` and `run_cell` from `null_grid` unmodified and changes
nothing about the estimator, the nulls, the grid or the summary statistics.

    python3 experiments/null_calibration/run_missing_cells.py \
        --cells 228-239 --out results/null_grid_n127607_tail.csv \
        --raw results/null_grid_n127607_tail.npz
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from null_grid import build_tasks, run_cell  # noqa: E402  (unmodified)


def parse_cells(spec: str) -> set[int]:
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-")
            out.update(range(int(a), int(b) + 1))
        elif part:
            out.add(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True,
                    help="cell ids from the FULL 312-cell task list, e.g. 228-239")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    wanted = parse_cells(args.cells)
    tasks = [t for t in build_tasks(args.seeds) if t[7] in wanted]
    found = {t[7] for t in tasks}
    if found != wanted:
        raise SystemExit(f"cell ids not in the full task list: {sorted(wanted - found)}")
    tasks.sort(key=lambda t: -t[1])

    print(f"[cells] {len(tasks)} cells x {args.seeds} seeds on {args.procs} procs",
          flush=True)
    for t in tasks:
        print(f"[cells]   id={t[7]} null={t[0]} n={t[1]} rho={t[2]} "
              f"align={int(t[3])} strength={t[4]} levels={t[5]}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    raw: dict[str, np.ndarray] = {}
    t0 = time.time()
    with open(out, "w", newline="") as f:
        writer = None
        with Pool(args.procs) as pool:
            for i, (cell_rows, vals) in enumerate(
                    pool.imap_unordered(run_cell, tasks), 1):
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
                    writer.writeheader()
                writer.writerows(cell_rows)
                f.flush()
                os.fsync(f.fileno())
                cid = cell_rows[0]["cell_id"]
                for k, v in vals.items():
                    raw[f"{cid}_{k}"] = v
                print(f"[cells] {i}/{len(tasks)} (cell {cid}, "
                      f"{time.time()-t0:.0f}s)", flush=True)

    if args.raw:
        np.savez_compressed(args.raw, **raw)
    print(f"[cells] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
