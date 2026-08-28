#!/usr/bin/env python3
"""
Section 5 of the null-calibration results file: how often does the shipped
DECISION RULE fire when the truth is exactly zero?

Reads `results/null_decision_rule.csv` (written by `null_decision_rule.py`) and
prints the per-cell firing rates the results file quotes. Nothing here
re-estimates anything; it counts the flags the decider already wrote.

The rule the manuscript ships, from `conditional_skill_report`'s docstring:
call a score substantive only when the two kernels agree, the interval excludes
zero, and it survives a depth control. The narrow leg the pre-registration's P3
predicts on -- "the 95% bootstrap interval lies strictly above zero" -- is
reported first and per kernel, because that is the sentence section 5 owes.

    python3 experiments/null_calibration/decision_rule_report.py
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CSV = HERE / "results" / "null_decision_rule.csv"

RHOS = [0.5, 0.9, 0.998]
NS = [3000, 39505]


def wilson(k, n, z=1.96):
    """Wilson 95% interval for a proportion; correct at k = n, unlike mean +- 2se."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def pct(k, n):
    lo, hi = wilson(k, n)
    return f"{k/n:6.1%}  [{lo:5.1%}, {hi:5.1%}]"


def main():
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else CSV
    if not path.is_file():
        sys.exit(f"missing {path}: the decision-rule arm has not finished")
    with open(path) as f:
        rows = [{k: v for k, v in r.items()} for r in csv.DictReader(f)]
    for r in rows:
        r["n"] = int(r["n"])
        r["rho_target"] = float(r["rho_target"])
        for k, v in list(r.items()):
            if k.endswith(("_tau", "_lo", "_hi")):
                r[k] = float(v)
            elif k.endswith(("excl0", "excl0_pos", "kernels_agree", "fires")):
                r[k] = int(v)

    def cell(n, rho):
        return [r for r in rows if r["n"] == n and r["rho_target"] == rho]

    def rate(rs, key):
        return sum(r[key] for r in rs), len(rs)

    print("=" * 100)
    print("5. THE DECISION RULE UNDER THE NULL")
    print("=" * 100)
    print(f"\nsource: {path}   rows: {len(rows)}")
    print("cells:", sorted({(r['n'], r['rho_target']) for r in rows}))

    # ---- the narrow leg, per kernel, k = 1 -------------------------------
    print("\n\n5.1  'the 95% bootstrap interval lies strictly above zero', k = 1")
    print(f"\n{'n':>7} {'rho':>6} {'reps':>5}   {'tau_b CI > 0':>26}   "
          f"{'wtau CI > 0':>26}   {'tau_b CI excludes 0':>26}")
    for n in NS:
        for rho in RHOS:
            rs = cell(n, rho)
            if not rs:
                continue
            k1, m = rate(rs, "k1_kang_wdm_taub_excl0_pos")
            k2, _ = rate(rs, "k1_scipy_weightedtau_excl0_pos")
            k3, _ = rate(rs, "k1_kang_wdm_taub_excl0")
            print(f"{n:7d} {rho:6.3f} {m:5d}   {pct(k1, m):>26}   "
                  f"{pct(k2, m):>26}   {pct(k3, m):>26}")

    # ---- the same leg after the depth control, k = 2 ---------------------
    print("\n\n5.2  the same leg after the depth control (k = 2: primitive + depth)")
    print(f"\n{'n':>7} {'rho':>6} {'reps':>5}   {'tau_b CI > 0':>26}   {'wtau CI > 0':>26}")
    for n in NS:
        for rho in RHOS:
            rs = cell(n, rho)
            if not rs:
                continue
            k1, m = rate(rs, "k2_kang_wdm_taub_excl0_pos")
            k2, _ = rate(rs, "k2_scipy_weightedtau_excl0_pos")
            print(f"{n:7d} {rho:6.3f} {m:5d}   {pct(k1, m):>26}   {pct(k2, m):>26}")

    # ---- the full conjunction --------------------------------------------
    print("\n\n5.3  the manuscript's three-part conjunction, all legs on the same replicate")
    print(f"\n{'n':>7} {'rho':>6} {'reps':>5}   {'kernels agree (k=1)':>26}   "
          f"{'FULL RULE FIRES':>26}")
    for n in NS:
        for rho in RHOS:
            rs = cell(n, rho)
            if not rs:
                continue
            a, m = rate(rs, "k1_kernels_agree")
            f_, _ = rate(rs, "rule_fires")
            print(f"{n:7d} {rho:6.3f} {m:5d}   {pct(a, m):>26}   {pct(f_, m):>26}")

    # ---- why: the point estimate against the interval width --------------
    print("\n\n5.4  why it fires: the null point estimate against the interval half-width")
    print(f"\n{'n':>7} {'rho':>6} {'kernel':>10} {'mean tau':>10} {'mean lo':>10} "
          f"{'mean hi':>10} {'half-width':>11} {'mean tau / hw':>14}")
    for n in NS:
        for rho in RHOS:
            rs = cell(n, rho)
            if not rs:
                continue
            for tag, kn in (("tau_b", "kang_wdm_taub"),
                            ("wtau", "scipy_weightedtau")):
                t = sum(r[f"k1_{kn}_tau"] for r in rs) / len(rs)
                lo = sum(r[f"k1_{kn}_lo"] for r in rs) / len(rs)
                hi = sum(r[f"k1_{kn}_hi"] for r in rs) / len(rs)
                hw = (hi - lo) / 2
                print(f"{n:7d} {rho:6.3f} {tag:>10} {t:+10.4f} {lo:+10.4f} "
                      f"{hi:+10.4f} {hw:11.4f} {t/hw if hw else float('nan'):14.2f}")

    # ---- verdict vocabulary ----------------------------------------------
    print("\n\n5.5  the shipped verdict vocabulary, k = 1, counts per cell")
    vocab = sorted({r["k1_combined"] for r in rows})
    print(f"\n{'n':>7} {'rho':>6} {'reps':>5}  " +
          "  ".join(f"{v:>28}" for v in vocab))
    for n in NS:
        for rho in RHOS:
            rs = cell(n, rho)
            if not rs:
                continue
            counts = [sum(1 for r in rs if r["k1_combined"] == v) for v in vocab]
            print(f"{n:7d} {rho:6.3f} {len(rs):5d}  " +
                  "  ".join(f"{c:>28d}" for c in counts))

    # ---- headline numbers the prose quotes --------------------------------
    print("\n\n5.6  the numbers the prose quotes")
    for n in NS:
        rs = [r for r in rows if r["n"] == n]
        k, m = rate(rs, "k1_kang_wdm_taub_excl0_pos")
        f_, _ = rate(rs, "rule_fires")
        print(f"  n = {n:6d}:  tau_b CI strictly above zero in {k}/{m} = {k/m:.1%} "
              f"of replicates; full conjunction fires {f_}/{m} = {f_/m:.1%}")
    allr = rows
    k, m = rate(allr, "k1_kang_wdm_taub_excl0_pos")
    print(f"  pooled over both n and all rho: {k}/{m} = {k/m:.1%}")
    worst = min((sum(r["k1_kang_wdm_taub_excl0_pos"] for r in cell(n, rho)) / len(cell(n, rho)),
                 n, rho) for n in NS for rho in RHOS if cell(n, rho))
    print(f"  lowest single-cell tau_b firing rate: {worst[0]:.1%} at n = {worst[1]}, rho = {worst[2]}")


if __name__ == "__main__":
    main()
