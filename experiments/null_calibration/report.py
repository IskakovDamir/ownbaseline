#!/usr/bin/env python3
"""
Read the null-calibration grid and print every table the pre-registration asks
for, in the pre-registration's order.

    python3 experiments/null_calibration/report.py --results results/ --out report.txt

Sections:
  0  provenance and completeness of the grid (which file each cell came from)
  1  per-cell table, every cell, with the "is the null mean zero" answer
  2  the deliverable table: every manuscript value against its own null cell
  3  P1 .. P5
  4  section 5, the ddof deliverable
  5  the decision rule under the null
  6  the rho = 1.0 regime

Nothing here recomputes an estimate. It reads the CSVs the runners wrote.
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_values import NOT_PLACEABLE, PAPER  # noqa: E402

RHO_GRID = [0.0, 0.5, 0.9, 0.93, 0.998, 1.0]
N_GRID = [500, 3000, 10000, 39505, 127607]
KERNELS = ("weightedtau_rankTrue", "weightedtau_rankFalse", "kendalltau")
K_OF = {"wtau": "weightedtau_rankTrue", "tau_b": "kendalltau"}

# Which file is the canonical source for which part of the grid. The level
# sweep at n = 3000 exists in three copies (see section 0); null_grid_n3000.csv
# is the canonical one and the other two are reported as replications.
SOURCES = [
    ("null_grid.csv", "2026-08-16 full-grid driver (interrupted)"),
    ("null_grid_n127607_tail.csv", "2026-08-28 tail of the interrupted run"),
    ("null_grid_n3000.csv", "2026-08-27 --n 3000"),
    ("null_grid_n39505.csv", "2026-08-27 --n 39505"),
    ("null_grid_n500_n10000.csv", "2026-08-27 --n 500,10000"),
]

FLOAT_COLS = ("mean", "sd", "mcse", "p2.5", "p97.5", "min", "max",
              "rho_target", "rho_realized_mean", "rho_prim_ordinal_mean",
              "nullB_strength", "seconds", "tau", "shift_wtau_ddof1_minus_ddof0",
              "shift_taub_ddof1_minus_ddof0", "rho_score_first_covariate",
              "synth_mean_abs_shift_wtau", "synth_mean_abs_shift_taub",
              "synth_max_abs_shift_wtau", "synth_max_abs_shift_taub",
              "tau_wtau_lstsq", "tau_taub_lstsq", "slope_inflation_if_applied")
INT_COLS = ("n", "n_seeds", "n_finite", "align", "levels", "cell_id",
            "mean_is_zero", "k_covariates", "rep", "ddof_mismatch_expressible")


def load(path):
    if not Path(path).is_file():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        for k in FLOAT_COLS:
            if k in r and r[k] not in ("", None):
                r[k] = float(r[k])
        for k in INT_COLS:
            if k in r and r[k] not in ("", None):
                r[k] = int(r[k])
        for k, v in list(r.items()):
            if k.endswith(("_tau", "_lo", "_hi", "_excl0", "_excl0_pos",
                           "_agree", "rule_fires")) and v not in ("", None):
                try:
                    r[k] = float(v)
                except ValueError:
                    pass
    return rows


def nearest(v, grid):
    return min(grid, key=lambda g: abs(g - v))


def cellkey(r):
    return (r["null"], r["n"], r["rho_target"], r["align"],
            "" if r["nullB_strength"] == "" else r["nullB_strength"],
            r["levels"], r["kernel"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/null_calibration/results")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    R = Path(args.results)

    sink = open(args.out, "w") if args.out else sys.stdout

    def P(*a):
        print(*a, file=sink)

    def H(title):
        P("\n" + "=" * 100)
        P(title)
        P("=" * 100)

    # ---------------------------------------------------------------- load
    per_file = {}
    for fname, when in SOURCES:
        rows = load(R / fname)
        for r in rows:
            r["_file"] = fname
        per_file[fname] = rows

    H("0. PROVENANCE AND COMPLETENESS")
    P(f"{'file':32} {'rows':>6} {'cells':>6}  n values")
    for fname, when in SOURCES:
        rows = per_file[fname]
        if not rows:
            P(f"{fname:32} {'--':>6} {'--':>6}  ABSENT")
            continue
        ns = sorted({r["n"] for r in rows})
        P(f"{fname:32} {len(rows):6d} {len({r['cell_id'] for r in rows}):6d}  "
          f"{ns}   [{when}]")

    # canonical selection
    canonical, dupes = {}, defaultdict(list)
    order = ["null_grid.csv", "null_grid_n127607_tail.csv", "null_grid_n3000.csv",
             "null_grid_n39505.csv", "null_grid_n500_n10000.csv"]
    for fname in order:
        for r in per_file.get(fname, []):
            k = cellkey(r)
            if k in canonical:
                dupes[k].append(r)
            else:
                canonical[k] = r
    G = list(canonical.values())

    want = []
    for null in ("A", "B"):
        for n in N_GRID:
            for rho in RHO_GRID:
                for al in (1, 0):
                    for st in ([0.45, 0.55, 0.63] if null == "B" else [""]):
                        for kern in KERNELS:
                            want.append((null, n, rho, al, st, 12, kern))
    for null in ("A", "B"):
        for lv in (2, 4, 50):
            for rho in RHO_GRID:
                for al in (1, 0):
                    st = 0.55 if null == "B" else ""
                    for kern in KERNELS:
                        want.append((null, 3000, rho, al, st, lv, kern))
    missing = [w for w in want if w not in canonical]
    P(f"\npre-registered kernel-rows: {len(want)}   present: {len(want)-len(missing)}"
      f"   missing: {len(missing)}")
    for m in missing:
        P(f"   MISSING {m}")

    P(f"\nconfigurations measured more than once: {len(dupes)} "
      f"(kernel-rows), i.e. {len(dupes)//3} cells")
    ident = sum(1 for k, v in dupes.items()
                for r in v if abs(r["mean"] - canonical[k]["mean"]) < 1e-12)
    diff = sum(len(v) for v in dupes.values()) - ident
    P(f"   of the repeat measurements, {ident} are bit-identical to the canonical "
      f"copy (same cell_id -> same seeds)")
    P(f"   and {diff} are independent replicates (different cell_id -> different seeds)")
    if dupes:
        P("\n   independent replication check, kendalltau rows only "
          "(canonical mean vs independent-replicate mean):")
        P(f"   {'null':>4} {'lev':>4} {'rho':>6} {'al':>3} {'canon':>9} "
          f"{'replicate':>10} {'diff':>9} {'mcse':>8}")
        shown = 0
        for k, v in sorted(dupes.items(), key=lambda x: (x[0][5], x[0][0], x[0][2])):
            if k[6] != "kendalltau":
                continue
            for r in v:
                if abs(r["mean"] - canonical[k]["mean"]) < 1e-12:
                    continue
                c = canonical[k]
                P(f"   {k[0]:>4} {k[5]:>4} {k[2]:6.3f} {k[3]:>3} {c['mean']:+9.4f} "
                  f"{r['mean']:+10.4f} {r['mean']-c['mean']:+9.4f} {c['mcse']:8.4f}")
                shown += 1
        P(f"   ({shown} rows)")

    seeds = sorted({r["n_seeds"] for r in G})
    fin = sorted({r["n_finite"] for r in G})
    P(f"\nseeds per cell: {seeds}   realized finite replicates per cell: "
      f"min {min(fin)} max {max(fin)}")
    nonfull = [r for r in G if r["n_finite"] < r["n_seeds"]]
    P(f"cells with at least one non-finite replicate: {len(nonfull)} of {len(G)}")
    byk = defaultdict(int)
    for r in nonfull:
        byk[(r["kernel"], r["rho_target"])] += 1
    for (kern, rho), c in sorted(byk.items()):
        P(f"   {kern:24} rho={rho:5.3f}  {c} cells")

    # ------------------------------------------------------- 1. per-cell table
    H("1. PER-CELL TABLE  (every cell; 'zero?' = does 0 lie within mean +/- 2*MCSE)")
    P(f"{'null':>4} {'lev':>4} {'n':>7} {'rho':>6} {'al':>3} {'str':>5} "
      f"{'kernel':22} {'seeds':>6} {'fin':>5} {'mean':>9} {'sd':>8} {'mcse':>8} "
      f"{'2.5%':>9} {'97.5%':>9} {'rho_real':>9} {'zero?':>6}")
    for r in sorted(G, key=lambda r: (r["null"], r["levels"], r["n"],
                                      r["rho_target"], -r["align"],
                                      str(r["nullB_strength"]), r["kernel"])):
        st = "" if r["nullB_strength"] == "" else f"{r['nullB_strength']:.2f}"
        P(f"{r['null']:>4} {r['levels']:4d} {r['n']:7d} {r['rho_target']:6.3f} "
          f"{r['align']:3d} {st:>5} {r['kernel']:22} {r['n_seeds']:6d} "
          f"{r['n_finite']:5d} {r['mean']:+9.4f} {r['sd']:8.4f} {r['mcse']:8.4f} "
          f"{r['p2.5']:+9.4f} {r['p97.5']:+9.4f} {r['rho_realized_mean']:+9.4f} "
          f"{'yes' if r['mean_is_zero'] else 'NO':>6}")

    prim = [r for r in G if r["levels"] == 12]

    def cells(null, n, rho, kern, align_on=1):
        return [r for r in prim if r["null"] == null and r["n"] == n
                and r["kernel"] == kern and abs(r["rho_target"] - rho) < 1e-9
                and r["align"] == align_on]

    # ------------------------------------------------- 2. the deliverable table
    H("2. THE DELIVERABLE TABLE — every manuscript value against ITS OWN null")
    C = load(R / "null_covariates.csv")
    COV_RHO = [0.0, 0.5, 0.9, 0.93, 0.998]

    def null_for(n, rho, k, kern):
        ng = nearest(n, N_GRID)
        if k == 1:
            rg = nearest(abs(rho), RHO_GRID)
            rs = cells("B", ng, rg, kern)
            if not rs:
                return None
            mean = sum(r["mean"] for r in rs) / len(rs)
            p975 = max(r["p97.5"] for r in rs)
            return mean, p975, f"Null B, n={ng}, rho={rg}", len(rs)
        ncov = nearest(n, [3000, 39505])
        rg = nearest(abs(rho), COV_RHO)
        rs = [r for r in C if r["n"] == ncov and r["k_covariates"] == k
              and r["kernel"] == kern and abs(r["rho_target"] - rg) < 1e-9]
        if not rs:
            return None
        return (rs[0]["mean"], rs[0]["p97.5"],
                f"covariate arm, n={ncov}, k={k}, rho={rg}", len(rs))

    for which in ("tau_b", "wtau"):
        kern = K_OF[which]
        P(f"\n--- {which}   [{kern}] ---")
        P(f"  {'row':30} {'atlas':10} {'n':>7} {'rho':>7} {'k':>2} {'value':>9} "
          f"{'null mean':>10} {'null 97.5%':>11}  {'null cell used':34} verdict")
        for label, atlas, n, rho, k, wt, tb, prov in PAPER:
            val = tb if which == "tau_b" else wt
            got = null_for(n, rho, k, kern)
            if got is None:
                P(f"  {label:30} {atlas:10} {n:7d} {rho:+7.4f} {k:2d} {val:+9.4f} "
                  f"{'--':>10} {'--':>11}  {'NO NULL CELL':34} not placeable")
                continue
            nm, p975, name, ncells = got
            if val <= 0:
                v = "not above (value is negative; the threshold is positive)"
            elif val > p975:
                v = "ABOVE ITS NULL"
            elif val > nm:
                v = "above the null mean only -- NOT above its null"
            else:
                v = "at or below its null mean"
            P(f"  {label:30} {atlas:10} {n:7d} {rho:+7.4f} {k:2d} {val:+9.4f} "
              f"{nm:+10.4f} {p975:+11.4f}  {name:34} {v}")

    P("\n  Reported by the manuscript but NOT placeable against this grid:")
    for row in NOT_PLACEABLE:
        P(f"   - {row[0]}  [{row[1]}]\n       {row[-1]}")

    # ------------------------------------------------------------ 3. P1 .. P5
    H("3. PRE-REGISTERED PREDICTIONS P1 .. P5")

    def sweep(null, kern, label, align_on=1, exclude_rho1=True):
        P(f"\n  {label}   [{kern}, Null {null}, align "
          f"{'on' if align_on else 'off'}"
          f"{', rho < 1 only' if exclude_rho1 else ''}]")
        P(f"    {'n':>8} {'cells':>6} {'mean':>9} {'sd':>8} {'mcse':>8} "
          f"{'min 2.5%':>9} {'max 97.5%':>10}  zero?")
        means = {}
        for n in N_GRID:
            rs = [r for r in prim if r["null"] == null and r["n"] == n
                  and r["kernel"] == kern and r["align"] == align_on
                  and (r["rho_target"] < 1.0 or not exclude_rho1)]
            if not rs:
                continue
            m = sum(r["mean"] for r in rs) / len(rs)
            sd = sum(r["sd"] for r in rs) / len(rs)
            mc = (sum(r["mcse"] ** 2 for r in rs) ** 0.5) / len(rs)
            means[n] = m
            P(f"    {n:8d} {len(rs):6d} {m:+9.4f} {sd:8.4f} {mc:8.4f} "
              f"{min(r['p2.5'] for r in rs):+9.4f} "
              f"{max(r['p97.5'] for r in rs):+10.4f}  "
              f"{'yes' if abs(m) <= 2*mc else 'NO'}")
        if means:
            lo, hi = min(means.values()), max(means.values())
            P(f"    across n: {lo:+.4f} .. {hi:+.4f}   "
              f"ratio max/min = {hi/lo:.2f}" if lo else "")
        return means

    P("\nP1  weightedtau(rank=True) carries a positive null bias near +0.3 that "
      "does NOT shrink with n.")
    a1 = sweep("A", "weightedtau_rankTrue", "P1 under Null A")
    b1 = sweep("B", "weightedtau_rankTrue", "P1 under Null B")

    P("\nP2  Kendall tau_b's null mean is within Monte Carlo error of 0 at every n.")
    sweep("A", "kendalltau", "P2 under Null A")
    sweep("B", "kendalltau", "P2 under Null B  (P2 was stated for the kernel, "
                             "not for this null; shown for completeness)")

    P("\nP3  Under Null B the residual carries a positive floor near +0.03, flat "
      "in n, with sd falling as n grows.")
    b3 = sweep("B", "kendalltau", "P3, Null B, tau_b")
    P("\n    sd by n, Null B, tau_b, align on, rho < 1 (P3's second half):")
    for n in N_GRID:
        rs = [r for r in prim if r["null"] == "B" and r["n"] == n
              and r["kernel"] == "kendalltau" and r["align"] == 1
              and r["rho_target"] < 1.0]
        if rs:
            P(f"      n={n:7d}  sd = {sum(r['sd'] for r in rs)/len(rs):.4f}   "
              f"fraction of cells whose 2.5th percentile is > 0: "
              f"{sum(1 for r in rs if r['p2.5'] > 0)}/{len(rs)}")

    P("\nP4  weightedtau(rank=False) has a null mean near 0.")
    sweep("A", "weightedtau_rankFalse", "P4 under Null A")
    sweep("B", "weightedtau_rankFalse", "P4 under Null B")

    P("\nP5  null mean with align ON exceeds null mean with align OFF, under "
      "every kernel.")
    for null in ("A", "B"):
        P(f"\n    Null {null}   (rho < 1 only)")
        P(f"    {'kernel':22} {'n':>8} {'align on':>10} {'align off':>10} "
          f"{'difference':>11} {'2*mcse(diff)':>13}  P5 holds?")
        for kern in KERNELS:
            for n in N_GRID:
                on = [r for r in prim if r["null"] == null and r["n"] == n
                      and r["kernel"] == kern and r["align"] == 1
                      and r["rho_target"] < 1.0]
                off = [r for r in prim if r["null"] == null and r["n"] == n
                       and r["kernel"] == kern and r["align"] == 0
                       and r["rho_target"] < 1.0]
                if not (on and off):
                    continue
                a = sum(r["mean"] for r in on) / len(on)
                b = sum(r["mean"] for r in off) / len(off)
                se = ((sum(r["mcse"] ** 2 for r in on) / len(on) ** 2)
                      + (sum(r["mcse"] ** 2 for r in off) / len(off) ** 2)) ** 0.5
                P(f"    {kern:22} {n:8d} {a:+10.4f} {b:+10.4f} {a-b:+11.4f} "
                  f"{2*se:13.4f}  {'yes' if a - b > 2*se else 'NO'}")

    # ----------------------------------------------------- 4. ddof deliverable
    H("4. PRE-REGISTRATION SECTION 5 — THE ddof DELIVERABLE, PER ROW")
    D = load(R / "ddof_section5.csv")
    P(f"  {'atlas':10} {'row':24} {'n':>6} {'k':>2} {'rho':>8} {'basis':14} "
      f"{'|shift| wtau':>13} {'|shift| tau_b':>14}  implementation / note")
    for r in D:
        if r["k_covariates"] == 1:
            sw = abs(r["shift_wtau_ddof1_minus_ddof0"])
            sk = abs(r["shift_taub_ddof1_minus_ddof0"])
            P(f"  {r['atlas']:10} {r['label']:24} {r['n']:6d} "
              f"{r['k_covariates']:2d} {r['rho_score_first_covariate']:+8.4f} "
              f"{r['basis']:14} {sw:13.2e} {sk:14.2e}  {r['residual_implementation']}")
        else:
            P(f"  {r['atlas']:10} {r['label']:24} {r['n']:6d} "
              f"{r['k_covariates']:2d} {'--':>8} {r['basis']:14} "
              f"{'n/a':>13} {'n/a':>14}  {r.get('note','')}")
    P("\n  matched-synthetic comparison at the same n and rho (200 seeds), for "
      "the k=1 rows:")
    P(f"  {'atlas':10} {'row':24} {'synth mean |d| wtau':>21} "
      f"{'synth max':>10} {'synth mean |d| tau_b':>21} {'synth max':>10}")
    for r in D:
        if r["k_covariates"] != 1:
            continue
        P(f"  {r['atlas']:10} {r['label']:24} {r['synth_mean_abs_shift_wtau']:21.2e} "
          f"{r['synth_max_abs_shift_wtau']:10.2e} "
          f"{r['synth_mean_abs_shift_taub']:21.2e} "
          f"{r['synth_max_abs_shift_taub']:10.2e}")

    # ------------------------------------------------ 5. the rule under the null
    H("5. THE SHIPPED DECISION RULE UNDER THE NULL")
    B = load(R / "null_decision_rule.csv")
    if not B:
        P("  null_decision_rule.csv absent — not run.")
    else:
        P(f"  {'n':>7} {'rho':>6} {'reps':>5} {'tau_b CI excl 0':>16} "
          f"{'tau_b CI > 0':>13} {'wtau CI > 0':>12} {'kernels agree':>14} "
          f"{'k=2 tau_b CI > 0':>17} {'FULL RULE FIRES':>16}")
        for n in (3000, 39505):
            for rho in (0.5, 0.9, 0.998):
                rs = [r for r in B if r["n"] == n
                      and abs(r["rho_target"] - rho) < 1e-9]
                if not rs:
                    continue
                fr = lambda k: sum(r[k] for r in rs) / len(rs)
                P(f"  {n:7d} {rho:6.3f} {len(rs):5d} "
                  f"{fr('k1_kang_wdm_taub_excl0'):15.1%} "
                  f"{fr('k1_kang_wdm_taub_excl0_pos'):12.1%} "
                  f"{fr('k1_scipy_weightedtau_excl0_pos'):11.1%} "
                  f"{fr('k1_kernels_agree'):13.1%} "
                  f"{fr('k2_kang_wdm_taub_excl0_pos'):16.1%} "
                  f"{fr('rule_fires'):15.1%}")
        P("\n  verdict vocabulary actually emitted, k = 1, tau_b:")
        vs = defaultdict(int)
        for r in B:
            vs[(r["n"], r["k1_kang_wdm_taub_verdict"])] += 1
        for (n, v), c in sorted(vs.items()):
            P(f"    n={n:6d}  {v:20} {c}")

    # ------------------------------------------------------------- 6. rho = 1.0
    H("6. THE rho = 1.0 REGIME  (rank(score) == rank(primitive) identically)")
    for kern in KERNELS:
        P(f"\n  {kern}, Null B, align on, all three strengths")
        P(f"    {'n':>8} {'str':>5} {'mean':>9} {'sd':>8} {'mcse':>8} "
          f"{'2.5%':>9} {'97.5%':>9} {'min':>9} {'max':>9} {'fin':>5}")
        for n in N_GRID:
            for r in sorted([r for r in prim if r["null"] == "B" and r["n"] == n
                             and r["kernel"] == kern and r["align"] == 1
                             and r["rho_target"] == 1.0],
                            key=lambda r: r["nullB_strength"]):
                P(f"    {n:8d} {r['nullB_strength']:5.2f} {r['mean']:+9.4f} "
                  f"{r['sd']:8.4f} {r['mcse']:8.4f} {r['p2.5']:+9.4f} "
                  f"{r['p97.5']:+9.4f} {r['min']:+9.4f} {r['max']:+9.4f} "
                  f"{r['n_finite']:5d}")
        rs = [r for r in prim if r["null"] == "B" and r["kernel"] == kern
              and r["align"] == 1 and r["rho_target"] == 1.0]
        if rs:
            ms = [r["mean"] for r in rs]
            P(f"    across all rho=1.0 cells: min {min(ms):+.4f} max {max(ms):+.4f}"
              f"   sign changes: {'yes' if min(ms) < 0 < max(ms) else 'no'}")
        rs0 = [r for r in prim if r["null"] == "A" and r["kernel"] == kern
               and r["align"] == 1 and r["rho_target"] == 1.0]
        if rs0:
            P("    Null A, align on, rho = 1.0:  "
              + "  ".join(f"n={r['n']}: {r['mean']:+.4f}" for r in
                          sorted(rs0, key=lambda r: r["n"])))

    if sink is not sys.stdout:
        sink.close()
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
