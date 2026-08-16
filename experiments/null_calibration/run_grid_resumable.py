#!/usr/bin/env python3
"""
Durable driver for the pre-registered null grid.

WHY THIS EXISTS. `null_grid.py` accumulates every cell in the parent process
and writes its CSV once, after the whole pool drains. The 2026-08-16 19:03 run
was interrupted and left zero rows on disk despite ~25 minutes of compute. This
driver runs the identical measurement and appends each cell to the CSV the
moment that cell finishes, so an interruption costs at most one cell.

WHAT IT DOES NOT DO. It does not touch the estimator, the null constructions,
the grid, the seeds or the summary statistics. `build_tasks` and `run_cell` are
imported from `null_grid` unmodified and called with the pre-registered
arguments. Seeding is `default_rng([BASE_ENTROPY, cell_id, seed_index])`, which
is a pure function of the cell, so --resume reproduces skipped cells exactly;
this was verified by re-running a completed cell and comparing raw replicate
arrays (identical).

    python3 experiments/null_calibration/run_grid_resumable.py --out grid.csv
    python3 experiments/null_calibration/run_grid_resumable.py --out grid.csv --resume
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


def completed_cell_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    with open(path, newline="") as f:
        return {int(r["cell_id"]) for r in csv.DictReader(f)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="null_grid.csv")
    ap.add_argument("--raw", default=None, help=".npz of every replicate")
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tasks = build_tasks(args.seeds)
    done = completed_cell_ids(out) if args.resume else set()
    if done and not args.resume:
        raise SystemExit(f"{out} exists; pass --resume or remove it")
    todo = [t for t in tasks if t[7] not in done]
    todo.sort(key=lambda t: -t[1])          # longest cells first

    print(f"[grid] {len(tasks)} cells total, {len(done)} already on disk, "
          f"{len(todo)} to run, {args.seeds} seeds each, {args.procs} procs",
          flush=True)
    if not todo:
        print("[grid] nothing to do")
        return

    raw: dict[str, np.ndarray] = {}
    if args.raw and Path(args.raw).is_file() and args.resume:
        raw = dict(np.load(args.raw))

    t0 = time.time()
    header_written = out.is_file() and out.stat().st_size > 0
    with open(out, "a", newline="") as f:
        writer = None
        with Pool(args.procs) as pool:
            for i, (cell_rows, vals) in enumerate(
                    pool.imap_unordered(run_cell, todo), 1):
                if writer is None:
                    writer = csv.DictWriter(f, fieldnames=list(cell_rows[0].keys()))
                    if not header_written:
                        writer.writeheader()
                writer.writerows(cell_rows)
                f.flush()
                os.fsync(f.fileno())        # survive a kill, not just an exit
                cid = cell_rows[0]["cell_id"]
                for k, v in vals.items():
                    raw[f"{cid}_{k}"] = v
                if args.raw and (i % 25 == 0 or i == len(todo)):
                    np.savez_compressed(args.raw, **raw)
                print(f"[grid] {i}/{len(todo)} cells "
                      f"(cell {cid}, n={cell_rows[0]['n']}, "
                      f"{time.time()-t0:.0f}s)", flush=True)

    if args.raw:
        np.savez_compressed(args.raw, **raw)
    print(f"[grid] done in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
