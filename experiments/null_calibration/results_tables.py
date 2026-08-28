#!/usr/bin/env python3
"""
Emit the markdown tables that go into 04-experiments/2026-08-14-null-calibration.md.

Every number in the results file that is not prose comes from here or from
report.py, so no figure in the write-up is transcribed by hand.

    python3 experiments/null_calibration/results_tables.py --which deliverable
    python3 experiments/null_calibration/results_tables.py --which floors
    python3 experiments/null_calibration/results_tables.py --which p1 | p3 | p5 | rho1 | ddof | rule
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paper_values import PAPER  # noqa: E402
from report import (K_OF, KERNELS, N_GRID, RHO_GRID, SOURCES,  # noqa: E402
                    cellkey, load, nearest)

COV_RHO = [0.0, 0.5, 0.9, 0.93, 0.998]


def grid(R):
    canonical = {}
    for fname, _ in SOURCES:
        for r in load(R / fname):
            canonical.setdefault(cellkey(r), r)
    return list(canonical.values())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="experiments/null_calibration/results")
    ap.add_argument("--which", required=True)
    a = ap.parse_args()
    R = Path(a.results)
    G = grid(R)
    prim = [r for r in G if r["levels"] == 12]

    def cells(null, n, rho, kern, al=1):
        return [r for r in prim if r["null"] == null and r["n"] == n
                and r["kernel"] == kern and abs(r["rho_target"] - rho) < 1e-9
                and r["align"] == al]

    def block(null, kern, al=1, exclude_rho1=True):
        out = {}
        for n in N_GRID:
            rs = [r for r in prim if r["null"] == null and r["n"] == n
                  and r["kernel"] == kern and r["align"] == al
                  and (r["rho_target"] < 1.0 or not exclude_rho1)]
            m = sum(r["mean"] for r in rs) / len(rs)
            sd = sum(r["sd"] for r in rs) / len(rs)
            mc = (sum(r["mcse"] ** 2 for r in rs) ** 0.5) / len(rs)
            out[n] = (m, sd, mc, len(rs))
        return out

    if a.which == "deliverable":
        C = load(R / "null_covariates.csv")

        def null_for(n, rho, k, kern):
            ng = nearest(n, N_GRID)
            if k == 1:
                rg = nearest(abs(rho), RHO_GRID)
                rs = cells("B", ng, rg, kern)
                return (sum(r["mean"] for r in rs) / len(rs),
                        max(r["p97.5"] for r in rs),
                        f"Null B, n={ng:,}, rho={rg}")
            ncov = nearest(n, [3000, 39505])
            rg = nearest(abs(rho), COV_RHO)
            rs = [r for r in C if r["n"] == ncov and r["k_covariates"] == k
                  and r["kernel"] == kern and abs(r["rho_target"] - rg) < 1e-9]
            tag = f"covariate arm, n={ncov:,}, k={k}, rho={rg}"
            if ncov != n:
                tag += f" *(no n={n:,} covariate cell)*"
            return rs[0]["mean"], rs[0]["p97.5"], tag
        for which in ("tau_b", "wtau"):
            kern = K_OF[which]
            print(f"\n#### {which} (`{kern}`)\n")
            print("| row | atlas | n | rho | k | value | null mean | null 97.5th | "
                  "null cell used | above its null? |")
            print("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
            for label, atlas, n, rho, k, wt, tb, prov in PAPER:
                val = tb if which == "tau_b" else wt
                nm, p975, name = null_for(n, rho, k, kern)
                if val <= 0:
                    v = "**no** — the value is negative and the threshold is positive"
                elif val > p975:
                    v = "**yes**"
                elif val > nm:
                    v = "**no** — above the null mean only"
                else:
                    v = "**no** — at or below the null mean"
                print(f"| {label} | {atlas} | {n:,} | {rho:+.4f} | {k} | "
                      f"{val:+.4f} | {nm:+.4f} | {p975:+.4f} | {name} | {v} |")

    elif a.which == "floors":
        for kern in ("kendalltau", "weightedtau_rankTrue"):
            for stat, key in (("mean", "mean"), ("97.5th percentile", "p97.5")):
                print(f"\n**Null B, align on, `{kern}` — {stat}** "
                      f"(mean averaged over the three strengths; percentile is the "
                      f"largest of the three)\n")
                print("| rho | " + " | ".join(f"n = {n:,}" for n in N_GRID) + " |")
                print("|---:|" + "---:|" * len(N_GRID))
                for rho in RHO_GRID[:-1]:
                    vals = []
                    for n in N_GRID:
                        rs = cells("B", n, rho, kern)
                        v = (sum(r[key] for r in rs) / len(rs) if key == "mean"
                             else max(r[key] for r in rs))
                        vals.append(f"{v:+.4f}")
                    print(f"| {rho} | " + " | ".join(vals) + " |")

    elif a.which in ("p1", "p2", "p3", "p4"):
        kern = {"p1": "weightedtau_rankTrue", "p2": "kendalltau",
                "p3": "kendalltau", "p4": "weightedtau_rankFalse"}[a.which]
        print(f"\n| n | Null A mean | Null A sd | Null A MCSE | zero? | "
              f"Null B mean | Null B sd | Null B MCSE | zero? |")
        print("|---:|---:|---:|---:|---|---:|---:|---:|---|")
        A, B = block("A", kern), block("B", kern)
        for n in N_GRID:
            am, asd, amc, _ = A[n]
            bm, bsd, bmc, _ = B[n]
            print(f"| {n:,} | {am:+.4f} | {asd:.4f} | {amc:.4f} | "
                  f"{'yes' if abs(am) <= 2*amc else '**no**'} | "
                  f"{bm:+.4f} | {bsd:.4f} | {bmc:.4f} | "
                  f"{'yes' if abs(bm) <= 2*bmc else '**no**'} |")
        for tag, blk in (("A", A), ("B", B)):
            ms = [blk[n][0] for n in N_GRID]
            print(f"\nNull {tag}: {min(ms):+.4f} .. {max(ms):+.4f}, "
                  f"ratio max/min = {max(ms)/min(ms):.2f}, "
                  f"monotone in n = {all(ms[i] < ms[i+1] for i in range(4))}")

    elif a.which == "p5":
        print("\n| kernel | null | n | align on | align off | difference | "
              "2 × MCSE(diff) | align raises the mean? |")
        print("|---|---|---:|---:|---:|---:|---:|---|")
        for kern in KERNELS:
            for null in ("A", "B"):
                for n in N_GRID:
                    on = [r for r in prim if r["null"] == null and r["n"] == n
                          and r["kernel"] == kern and r["align"] == 1
                          and r["rho_target"] < 1.0]
                    off = [r for r in prim if r["null"] == null and r["n"] == n
                           and r["kernel"] == kern and r["align"] == 0
                           and r["rho_target"] < 1.0]
                    x = sum(r["mean"] for r in on) / len(on)
                    y = sum(r["mean"] for r in off) / len(off)
                    se = ((sum(r["mcse"] ** 2 for r in on) / len(on) ** 2)
                          + (sum(r["mcse"] ** 2 for r in off) / len(off) ** 2)) ** 0.5
                    print(f"| `{kern}` | {null} | {n:,} | {x:+.4f} | {y:+.4f} | "
                          f"{x-y:+.4f} | {2*se:.4f} | "
                          f"{'**yes**' if x-y > 2*se else 'no'} |")

    elif a.which == "rho1":
        for kern in ("kendalltau", "weightedtau_rankTrue", "weightedtau_rankFalse"):
            print(f"\n**`{kern}`, Null B, align on, rho = 1.0**\n")
            print("| n | strength | mean | sd | MCSE | 2.5th | 97.5th | finite |")
            print("|---:|---:|---:|---:|---:|---:|---:|---:|")
            for n in N_GRID:
                for r in sorted(cells("B", n, 1.0, kern),
                                key=lambda r: r["nullB_strength"]):
                    print(f"| {n:,} | {r['nullB_strength']:.2f} | {r['mean']:+.4f} | "
                          f"{r['sd']:.4f} | {r['mcse']:.4f} | {r['p2.5']:+.4f} | "
                          f"{r['p97.5']:+.4f} | {r['n_finite']} |")
            ms = [r["mean"] for r in
                  [c for n in N_GRID for c in cells("B", n, 1.0, kern)]]
            print(f"\nrange {min(ms):+.4f} .. {max(ms):+.4f}; sign changes: "
                  f"{'yes' if min(ms) < 0 < max(ms) else 'no'}; "
                  f"largest/smallest |mean| = "
                  f"{max(abs(m) for m in ms)/min(abs(m) for m in ms):.0f}x")

    elif a.which == "covk":
        C = load(R / "null_covariates.csv")
        print("\n| n | k | `kendalltau` mean | `weightedtau(rank=True)` mean | "
              "`weightedtau(rank=False)` mean |")
        print("|---:|---:|---:|---:|---:|")
        for n in (3000, 39505):
            for k in (1, 2, 4):
                v = []
                for kern in ("kendalltau", "weightedtau_rankTrue",
                             "weightedtau_rankFalse"):
                    rs = [r for r in C if r["n"] == n and r["k_covariates"] == k
                          and r["kernel"] == kern]
                    v.append(f"{sum(r['mean'] for r in rs)/len(rs):+.4f}")
                print(f"| {n:,} | {k} | " + " | ".join(v) + " |")

    elif a.which == "ddof":
        D = load(R / "ddof_section5.csv")
        print("\n| atlas | row | n | k | rho | shift computed on | "
              "residual implementation the run used | ddof mismatch expressible? | "
              "\\|shift\\| weightedtau | \\|shift\\| tau_b |")
        print("|---|---|---:|---:|---:|---|---|---|---:|---:|")
        for r in D:
            if r["k_covariates"] == 1:
                print(f"| {r['atlas']} | {r['label']} | {r['n']:,} | "
                      f"{r['k_covariates']} | {r['rho_score_first_covariate']:+.4f} | "
                      f"**real vectors** | `{r['residual_implementation']}` | yes | "
                      f"{abs(r['shift_wtau_ddof1_minus_ddof0']):.2e} | "
                      f"{abs(r['shift_taub_ddof1_minus_ddof0']):.2e} |")
            else:
                print(f"| {r['atlas']} | {r['label']} | {r['n']:,} | "
                      f"{r['k_covariates']} | — | **real vectors** | "
                      f"`{r['residual_implementation']}` | "
                      f"no (single-covariate form only) | n/a | n/a |")
        print("\n**Matched-synthetic comparison** at the same n and the same "
              "score-primitive correlation, 200 seeds, for the k = 1 rows:\n")
        print("| atlas | row | synthetic mean \\|shift\\| wtau | max | "
              "synthetic mean \\|shift\\| tau_b | max |")
        print("|---|---|---:|---:|---:|---:|")
        for r in D:
            if r["k_covariates"] != 1:
                continue
            print(f"| {r['atlas']} | {r['label']} | "
                  f"{r['synth_mean_abs_shift_wtau']:.2e} | "
                  f"{r['synth_max_abs_shift_wtau']:.2e} | "
                  f"{r['synth_mean_abs_shift_taub']:.2e} | "
                  f"{r['synth_max_abs_shift_taub']:.2e} |")

    elif a.which == "rule":
        B = load(R / "null_decision_rule.csv")
        if not B:
            print("null_decision_rule.csv absent")
            return
        print("\n| n | rho | replicates | tau_b CI excludes 0 | tau_b CI > 0 | "
              "wtau CI > 0 | kernels agree | k=2 tau_b CI > 0 | "
              "**full conjunction fires** |")
        print("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for n in (3000, 39505):
            for rho in (0.5, 0.9, 0.998):
                rs = [r for r in B if r["n"] == n
                      and abs(r["rho_target"] - rho) < 1e-9]
                if not rs:
                    continue
                f = lambda k: sum(r[k] for r in rs) / len(rs)
                print(f"| {n:,} | {rho} | {len(rs)} | "
                      f"{f('k1_kang_wdm_taub_excl0'):.0%} | "
                      f"{f('k1_kang_wdm_taub_excl0_pos'):.0%} | "
                      f"{f('k1_scipy_weightedtau_excl0_pos'):.0%} | "
                      f"{f('k1_kernels_agree'):.0%} | "
                      f"{f('k2_kang_wdm_taub_excl0_pos'):.0%} | "
                      f"**{f('rule_fires'):.0%}** |")
    else:
        raise SystemExit(f"unknown --which {a.which}")


if __name__ == "__main__":
    main()
