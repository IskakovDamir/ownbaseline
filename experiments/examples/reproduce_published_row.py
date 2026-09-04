#!/usr/bin/env python3
"""
Reproduce one published row from the command line, and check it against the
frozen run record rather than against a number typed into this file.

The row: CytoTRACE v1 residualized on gene count, against the twelve-stage
zebrafish ordinal of GSE106474. The manuscript reports tau_b = +0.6531 with a
95% bootstrap interval of [+0.6479, +0.6579] at n = 39,505, against its own
measured null floor of +0.0247.

Two of the three numbers come out of artefacts in this repository. The published
value and its interval are read from data/run_record/track2/, which is the run
the figures were drawn from. The floor is recomputed from the shipped null grid.
Only the recomputation needs the prepared atlas, which is not in the repository:
point OWN_BASELINE_DATA at a data root holding
track2/data/prepared/primitives.npz and track2/results/cytotrace_scores.npz, or
run with --floors-only to check the parts that need no data.

    python3 experiments/examples/reproduce_published_row.py
    python3 experiments/examples/reproduce_published_row.py --floors-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from own_baseline import floors                                    # noqa: E402
from own_baseline.cli import conditional_skill, residual_scale     # noqa: E402
from own_baseline.conditional_skill import align                   # noqa: E402
from own_baseline.paths import data_root                           # noqa: E402

RECORD = REPO / "data/run_record/track2/gate2_conditional_skill.json"
ROW = "CytoTRACE_v1 | gene_count"
PUBLISHED_TAU = 0.6531
PUBLISHED_CI = (0.6479, 0.6579)
PUBLISHED_FLOOR = 0.0247
PUBLISHED_N = 39505
PUBLISHED_RHO = 0.4844


def _dig(d, path):
    for k in path.split("/"):
        d = d[k]
    return d


def check(label, got, want, tol):
    ok = abs(got - want) <= tol
    print(f"  {'OK ' if ok else 'FAIL'}  {label:44s} {got:+.4f}  "
          f"published {want:+.4f}  |delta| {abs(got - want):.2e}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors-only", action="store_true")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()
    ok = True

    print("\n1. the floor, recomputed from the shipped null grid")
    f = floors.floor(PUBLISHED_N, PUBLISHED_RHO, "taub", 1, 12)
    print(f"     cell: {f['cell']}")
    ok &= check("floor at n=39,505, rho=0.484", f["floor"], PUBLISHED_FLOOR, 5e-5)

    print("\n2. the published value, read from the frozen run record")
    if not RECORD.is_file():
        print(f"  SKIP  {RECORD} is missing")
    else:
        rec = json.loads(RECORD.read_text())
        tau = _dig(rec, f"conditional_skill/{ROW}/kang_wdm_taub/tau")
        lo, hi = _dig(rec, f"conditional_skill/{ROW}/kang_wdm_taub/CI95")
        ok &= check("tau_b in the run record", tau, PUBLISHED_TAU, 5e-5)
        ok &= check("interval, lower", lo, PUBLISHED_CI[0], 5e-5)
        ok &= check("interval, upper", hi, PUBLISHED_CI[1], 5e-5)
        print(f"  ..    ratio to its own floor: {tau / f['floor']:.1f}x  "
              f"(the manuscript says 26.4)")

    if args.floors_only:
        print("\n--floors-only: skipping the recomputation from the atlas.")
        return 0 if ok else 1

    print("\n3. recomputed from the prepared atlas")
    root = data_root()
    prim_p = root / "track2/data/prepared/primitives.npz"
    score_p = root / "track2/results/cytotrace_scores.npz"
    if not (prim_p.is_file() and score_p.is_file()):
        print(f"  SKIP  no prepared atlas under {root}")
        print( "        set OWN_BASELINE_DATA, or pass --floors-only")
        return 0 if ok else 1

    P = np.load(prim_p)
    ordinal = np.asarray(P["rank_kimmel"], float)
    primitive = np.asarray(P["gene_count"], float)
    score = np.asarray(np.load(score_p)["ct_v1"], float)
    s = align(score, ordinal)

    ratio, _ = residual_scale(s, [primitive])
    print(f"  ..    residual scale {ratio:.3g} times the arithmetic noise floor "
          f"(debris sits at 1)")

    cs = conditional_skill(s, ordinal, [primitive], "taub", args.boot, 42)
    ok &= check("tau_b recomputed", cs["tau"], PUBLISHED_TAU, 5e-5)
    ok &= check("interval, lower", cs["CI95"][0], PUBLISHED_CI[0], 5e-5)
    ok &= check("interval, upper", cs["CI95"][1], PUBLISHED_CI[1], 5e-5)

    print(f"\n{'REPRODUCES' if ok else 'DOES NOT REPRODUCE'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
