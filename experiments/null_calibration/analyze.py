#!/usr/bin/env python3
"""
Read the null grid and answer the pre-registered questions.

    python3 experiments/null_calibration/analyze.py \
        --grid 04-experiments/2026-08-14-null-calibration-grid.csv \
        --cov  04-experiments/2026-08-14-null-covariates.csv

Prints, in order:
  1. P1-P5 verdicts.
  2. Null mean by kernel and n, both nulls.
  3. Flatness in n.
  4. The deliverable: every value the manuscript reports, against its own null.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict

# ---------------------------------------------------------------------------
# What the manuscript reports. Values and score-primitive rank correlations are
# from the run reproduced on 2026-08-13 (data/runs/track2/results/
# gate2_conditional_skill.json), except where marked, which come from the
# vault notes because the score could not be recomputed without R.
# ---------------------------------------------------------------------------
PAPER = [
    # label,                       n,      rho,    k, tau_b,   wtau,    source
    ("CytoTRACE_v1 | gene_count",  39505, 0.4844, 1, +0.6531, +0.7901, "reproduced"),
    ("SR | PCC(x,degree)",         39505, 0.9305, 1, +0.4300, +0.7366, "reproduced"),
    ("SR | CCAT",                  39505, 0.9267, 1, +0.4678, +0.7522, "reproduced"),
    ("CCAT | PCC(x,degree)",       39505, 0.9982, 1, -0.2209, +0.1716, "reproduced"),
    ("ORIGINS | PCC(x,degree)",    39505, 0.4339, 1, +0.4019, +0.5272, "vault note"),
    ("ORIGINS | gene_count",       39505, 0.4339, 1, +0.4546, +0.4772, "vault note"),
    ("NCG | PCC(x,degree)",         3000, 0.9000, 1, +0.4015, +0.7200, "vault note, GO connectome"),
    ("dpath | Shannon_H",           3000, 0.4769, 1, +0.1202, +0.1914, "vault note, subsample"),
    ("SLICE | Shannon_H",           3000, 0.9317, 1, -0.1132, -0.0522, "vault note, subsample"),
    # depth robustness: 2 covariates
    ("CytoTRACE_v1 | gc + loglib", 39505, 0.4844, 2, +0.4574, +0.7398, "reproduced"),
    ("SR | PCC + loglib",          39505, 0.9305, 2, +0.2340, +0.6575, "reproduced"),
    ("SR | CCAT + loglib",         39505, 0.9267, 2, +0.2533, +0.6618, "reproduced"),
    ("CCAT | PCC + loglib",        39505, 0.9982, 2, -0.0610, +0.3837, "reproduced"),
    # joint-4
    ("CytoTRACE_v1 | joint-4",     39505, 0.4844, 4, +0.2880, +0.7000, "vault note"),
    ("SR | joint-4",               39505, 0.9305, 4, +0.1830, +0.6560, "vault note"),
    ("ORIGINS | joint-4",          39505, 0.4339, 4, +0.1560, +0.1550, "vault note"),
    ("dpath | joint-4",             3000, 0.4769, 4, +0.1340, +0.2410, "vault note"),
]

RHO_GRID = [0.0, 0.5, 0.9, 0.93, 0.998, 1.0]
N_GRID = [500, 3000, 10000, 39505, 127607]
K_TO_KERNEL = {"tau_b": "kendalltau", "wtau": "weightedtau_rankTrue"}


def load(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in ("mean", "sd", "mcse", "p2.5", "p97.5", "rho_target",
                  "rho_realized_mean", "rho_prim_ordinal_mean", "min", "max"):
            if k in r and r[k] not in ("", None):
                r[k] = float(r[k])
        for k in ("n", "n_seeds", "align", "levels", "k_covariates"):
            if k in r and r[k] not in ("", None):
                r[k] = int(r[k])
    return rows


def nearest(v, grid):
    return min(grid, key=lambda g: abs(g - v))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", required=True)
    ap.add_argument("--cov", default=None)
    args = ap.parse_args()

    G = load(args.grid)
    primary = [r for r in G if r["levels"] == 12]

    def cell(null, n, rho, kernel, align_on=1, strength=None):
        out = [r for r in primary
               if r["null"] == null and r["n"] == n and r["kernel"] == kernel
               and abs(r["rho_target"] - rho) < 1e-9 and r["align"] == align_on
               and (strength is None or str(r["nullB_strength"]) == str(strength))]
        return out

    print("=" * 78)
    print("1. PRE-REGISTERED PREDICTIONS")
    print("=" * 78)

    # P1 / P2 / P4: null mean by kernel, Null A, align on, averaged over rho<1
    for kern, pred in (("weightedtau_rankTrue", "P1  bias ~ +0.3, flat in n"),
                       ("kendalltau", "P2  no bias"),
                       ("weightedtau_rankFalse", "P4  no bias")):
        print(f"\n  {pred}   [{kern}]")
        print(f"    {'n':>8} {'mean':>9} {'sd':>8} {'mcse':>8} {'2.5%':>9} {'97.5%':>9}  zero?")
        means = []
        for n in N_GRID:
            rs = [r for r in primary if r["null"] == "A" and r["n"] == n
                  and r["kernel"] == kern and r["align"] == 1
                  and r["rho_target"] < 1.0]
            if not rs:
                continue
            m = sum(r["mean"] for r in rs) / len(rs)
            sd = sum(r["sd"] for r in rs) / len(rs)
            mc = sum(r["mcse"] for r in rs) / len(rs)
            lo = min(r["p2.5"] for r in rs)
            hi = max(r["p97.5"] for r in rs)
            means.append(m)
            print(f"    {n:8d} {m:+9.4f} {sd:8.4f} {mc:8.4f} {lo:+9.4f} {hi:+9.4f}"
                  f"  {'yes' if abs(m) <= 2*mc else 'NO'}")
        if means:
            print(f"    range across n: {min(means):+.4f} .. {max(means):+.4f}"
                  f"   ratio max/min = {max(means)/min(means):.2f}" if min(means) != 0
                  else "")

    # P5: align on vs off
    print("\n  P5  align contributes to the bias")
    print(f"    {'kernel':22} {'align=on':>10} {'align=off':>10} {'difference':>11}")
    for kern in ("weightedtau_rankTrue", "weightedtau_rankFalse", "kendalltau"):
        on = [r["mean"] for r in primary if r["null"] == "A" and r["kernel"] == kern
              and r["align"] == 1 and r["rho_target"] < 1.0]
        off = [r["mean"] for r in primary if r["null"] == "A" and r["kernel"] == kern
               and r["align"] == 0 and r["rho_target"] < 1.0]
        if on and off:
            a, b = sum(on)/len(on), sum(off)/len(off)
            print(f"    {kern:22} {a:+10.4f} {b:+10.4f} {a-b:+11.4f}")

    # P3: Null B floor
    print("\n  P3  residual bias floor ~ +0.03, flat in n   [Null B, kendalltau, align on]")
    print(f"    {'n':>8} {'mean':>9} {'sd':>8} {'mcse':>8} {'2.5%':>9} {'97.5%':>9}  zero?")
    for n in N_GRID:
        rs = [r for r in primary if r["null"] == "B" and r["n"] == n
              and r["kernel"] == "kendalltau" and r["align"] == 1
              and r["rho_target"] < 1.0]
        if not rs:
            continue
        m = sum(r["mean"] for r in rs)/len(rs)
        sd = sum(r["sd"] for r in rs)/len(rs)
        mc = sum(r["mcse"] for r in rs)/len(rs)
        print(f"    {n:8d} {m:+9.4f} {sd:8.4f} {mc:8.4f} "
              f"{min(r['p2.5'] for r in rs):+9.4f} {max(r['p97.5'] for r in rs):+9.4f}"
              f"  {'yes' if abs(m) <= 2*mc else 'NO'}")

    print("\n" + "=" * 78)
    print("2. NULL MEAN BY rho  (Null A and B, align on, n = 39505)")
    print("=" * 78)
    for null in ("A", "B"):
        print(f"\n  Null {null}")
        print(f"    {'rho':>6} " + "".join(f"{k[:18]:>20}" for k in
                                           ("weightedtau_rankTrue", "weightedtau_rankFalse", "kendalltau")))
        for rho in RHO_GRID:
            line = f"    {rho:6.3f} "
            for kern in ("weightedtau_rankTrue", "weightedtau_rankFalse", "kendalltau"):
                rs = cell(null, 39505, rho, kern)
                if rs:
                    m = sum(r["mean"] for r in rs)/len(rs)
                    p = max(r["p97.5"] for r in rs)
                    line += f"{m:+9.4f}/{p:+9.4f}"
                else:
                    line += f"{'--':>20}"
            print(line)
    print("    (cells are mean / 97.5th percentile)")

    print("\n" + "=" * 78)
    print("3. THE DELIVERABLE — every reported value against its own null")
    print("=" * 78)

    C = load(args.cov) if args.cov else []

    def null_for(n, rho, k, kernel):
        """Null cell at this value's own n, rho and covariate count."""
        rg = nearest(rho, RHO_GRID)
        ng = nearest(n, N_GRID)
        if k == 1:
            rs = cell("B", ng, rg, kernel)
            if rs:
                m = sum(r["mean"] for r in rs)/len(rs)
                return m, max(r["p97.5"] for r in rs), f"NullB n={ng} rho={rg}"
        else:
            rs = [r for r in C if r["n"] == nearest(n, [3000, 39505])
                  and r["k_covariates"] == k and r["kernel"] == kernel
                  and abs(r["rho_target"] - nearest(rho, [0.0, 0.5, 0.9, 0.93, 0.998])) < 1e-9]
            if rs:
                return (rs[0]["mean"], rs[0]["p97.5"],
                        f"cov n={rs[0]['n']} k={k} rho={rs[0]['rho_target']}")
        return None, None, "NO NULL CELL"

    for which, kernel in (("tau_b", "kendalltau"), ("wtau", "weightedtau_rankTrue")):
        print(f"\n  --- {which}  ({kernel}) ---")
        print(f"  {'row':30} {'n':>7} {'rho':>6} {'k':>2} {'value':>8} "
              f"{'null mean':>10} {'null 97.5%':>11}  verdict")
        for label, n, rho, k, tb, wt, src in PAPER:
            val = tb if which == "tau_b" else wt
            nm, np97, cellname = null_for(n, rho, k, kernel)
            if nm is None:
                print(f"  {label:30} {n:7d} {rho:6.3f} {k:2d} {val:+8.4f} "
                      f"{'--':>10} {'--':>11}  NO NULL CELL")
                continue
            if val > np97:
                v = "ABOVE its null"
            elif val > nm:
                v = "above null MEAN only -- NOT above"
            else:
                v = "AT OR BELOW its null"
            print(f"  {label:30} {n:7d} {rho:6.3f} {k:2d} {val:+8.4f} "
                  f"{nm:+10.4f} {np97:+11.4f}  {v}")

    if C:
        print("\n" + "=" * 78)
        print("4. DOES THE BIAS GROW WITH THE NUMBER OF COVARIATES REMOVED?")
        print("=" * 78)
        for n in sorted({r["n"] for r in C}):
            print(f"\n  n = {n}")
            print(f"    {'kernel':22} " + "".join(f"{'k='+str(k):>12}" for k in (1, 2, 4)))
            for kern in ("weightedtau_rankTrue", "weightedtau_rankFalse", "kendalltau"):
                line = f"    {kern:22} "
                for k in (1, 2, 4):
                    rs = [r for r in C if r["n"] == n and r["k_covariates"] == k
                          and r["kernel"] == kern and r["rho_target"] < 1.0]
                    line += f"{sum(r['mean'] for r in rs)/len(rs):+12.4f}" if rs else f"{'--':>12}"
                print(line)


if __name__ == "__main__":
    main()
