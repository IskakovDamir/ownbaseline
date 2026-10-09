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

DESIGN POINTS OUTSIDE THE GRID (added 2026-10-09 for experiments/w5_new_rows).
A value is read against the null at its own n, rho, number of covariates and
number of ordinal levels. When the shipped grid has no cell there, --design
runs one, with the same generators and the same seeding scheme:

    python3 experiments/null_calibration/run_missing_cells.py \
        --design n=12354,rho=0.41,levels=4,k=1 --out <csv>

  k = 1   null_grid.run_cell, Null B, align on, at each of the grid's three
          coupling strengths (0.45, 0.55, 0.63); the floor is the maximum of
          the three 97.5th percentiles, as own_baseline.floors reads the grid
  k > 1   null_covariates.run_cell at its single coupling strength 0.55 with
          its ordinal level count set to `levels`

A design cell's id is a hash of its parameters, offset past the 312 grid ids,
so its seed stream is the same on every run and never coincides with a grid
cell's: default_rng([BASE_ENTROPY, id, seed]) for k = 1 (null_grid's stream)
and default_rng([BASE, 7, id, seed]) for k > 1 (null_covariates' stream).
The CSV is written to a .partial file and renamed only when every cell has
finished, so an interrupted run never leaves a partial file at --out.
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


NULLB_STRENGTH = (0.45, 0.55, 0.63)      # null_grid.NULLB_STRENGTH
COV_STRENGTH = 0.55                      # null_covariates.STRENGTH
DESIGN_FIELDS = ["design", "design_id", "null", "n", "rho_target", "levels",
                 "k_covariates", "align", "nullB_strength", "kernel", "n_seeds",
                 "n_finite", "mean", "sd", "mcse", "p2.5", "p97.5", "min", "max",
                 "seconds"]


def parse_design(spec: str) -> dict:
    d = dict(kv.split("=") for kv in spec.split(","))
    return {"n": int(d["n"]), "rho": round(float(d["rho"]), 4),
            "levels": int(d.get("levels", 12)), "k": int(d.get("k", 1))}


def design_id(null, n, rho, levels, strength, k) -> int:
    import hashlib
    key = f"{null}|{n}|{rho:.4f}|{levels}|{strength:.2f}|{k}"
    return 1000 + int(hashlib.sha256(key.encode()).hexdigest()[:7], 16)


def run_design(task):
    """One design cell. Returns rows in DESIGN_FIELDS form."""
    spec, n, rho, levels, k, strength, seeds, cid = task
    if k == 1:
        rows, _ = run_cell(("B", n, rho, True, strength, levels, seeds, cid))
    else:
        import null_covariates as nc            # unmodified; two module constants set
        nc.LEVELS, nc.STRENGTH = levels, strength
        rows = nc.run_cell((n, k, rho, seeds, cid))
    out = []
    for r in rows:
        out.append({"design": spec, "design_id": cid, "null": "B", "n": n,
                    "rho_target": rho, "levels": levels, "k_covariates": k,
                    "align": 1, "nullB_strength": strength,
                    **{f: r.get(f, "") for f in DESIGN_FIELDS[9:]}})
    return out


def main_design(args):
    tasks = []
    for spec in args.design:
        d = parse_design(spec)
        strengths = NULLB_STRENGTH if d["k"] == 1 else (COV_STRENGTH,)
        for st in strengths:
            cid = design_id("B", d["n"], d["rho"], d["levels"], st, d["k"])
            tasks.append((spec, d["n"], d["rho"], d["levels"], d["k"], st,
                          args.seeds, cid))
    print(f"[design] {len(tasks)} cells x {args.seeds} seeds on {args.procs} procs",
          flush=True)
    for t in tasks:
        print(f"[design]   id={t[7]} n={t[1]} rho={t[2]} levels={t[3]} k={t[4]} "
              f"strength={t[5]}", flush=True)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    partial = out.with_name(out.name + ".partial")
    t0 = time.time()
    with open(partial, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=DESIGN_FIELDS)
        w.writeheader()
        with Pool(args.procs) as pool:
            for i, rows in enumerate(pool.imap_unordered(run_design, tasks), 1):
                w.writerows(rows)
                f.flush()
                os.fsync(f.fileno())
                print(f"[design] {i}/{len(tasks)} (id {rows[0]['design_id']}, "
                      f"{time.time()-t0:.0f}s)", flush=True)
    os.replace(partial, out)
    print(f"[design] done in {time.time()-t0:.0f}s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default=None,
                    help="cell ids from the FULL 312-cell task list, e.g. 228-239")
    ap.add_argument("--design", action="append", default=None,
                    help="a design point outside the grid, n=..,rho=..,levels=..,k=..; "
                         "repeatable")
    ap.add_argument("--out", required=True)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--seeds", type=int, default=200)
    ap.add_argument("--procs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    args = ap.parse_args()

    if args.design:
        return main_design(args)
    if not args.cells:
        raise SystemExit("give --cells or --design")
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
