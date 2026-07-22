"""
wdm_coverage_14rows.py
======================
PI-gate 2026-07-17, mandate 3.

Current wdm validation covers ONE dataset (Retinal neurons 10x) against Kang's
S12. Kang's S12 has 13 Test-cohort rows (mandate said 14; audit shows 13 non-
median rows in Test cohort — one row appears absent, likely reflecting Kang's
own exclusions). We validate against ALL Test-cohort S12 rows.

For each Test row and each of the 5 methods (CT2, CT1, CCAT, SR, StemID),
compute |Δτ| between our wdm impl and Kang's published S12 value. Report:
  - max |Δτ| per method across all rows
  - worst-case row × method
  - flag if any exceeds 0.02

Kang S12 uses RELATIVE ORDER labels (S4) as target, with absolute-order weighting.
Per Kang Methods and dossier §2, we use S4 relative-order labels and Kang's absolute-
order weighting (Standardized phenotype × relative-order level).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import weightedtau

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from wdm_tau import wdm_tau_b_bit, kang_absolute_weights   # noqa: E402

SI = HERE / "kang2025_SI"
OUT_PATH = HERE / "wdm_coverage.json"

METHODS = ["CytoTRACE 2", "CytoTRACE 1", "SCENT (CCAT)", "SCENT (SR)", "StemID"]

# Column-name map: S12 label -> S11 column name
S12_TO_S11 = {
    "CytoTRACE 2":  "CytoTRACE 2 potency score",
    "CytoTRACE 1":  "CytoTRACE 1",
    "SCENT (CCAT)": "SCENT (CCAT)",
    "SCENT (SR)":   "SCENT (SR)",
    "StemID":       "StemID",
}


def load_tables():
    S4  = pd.read_csv(SI / "Table_S4.csv",  header=6).dropna(how="all").dropna(axis=1, how="all")
    S11 = pd.read_csv(SI / "Table_S11.csv", header=4).dropna(how="all").dropna(axis=1, how="all")
    S12 = pd.read_csv(SI / "Table_S12.csv", header=4).dropna(how="all").dropna(axis=1, how="all")
    return S4, S11, S12


def relative_order_map(S4, dataset_name):
    """S4 gives relative-order integer levels per standardized phenotype per dataset."""
    # S4 columns: check
    ds_col = None
    for c in S4.columns:
        if "dataset" in c.lower():
            ds_col = c
            break
    ph_col = None
    for c in S4.columns:
        if "phenotype" in c.lower() and "standardized" in c.lower():
            ph_col = c
            break
    ro_col = None
    for c in S4.columns:
        if "relative order" in c.lower() or "relative_order" in c.lower():
            ro_col = c
            break
    if ro_col is None:
        # Try any column with 'order' or 'level'
        for c in S4.columns:
            if "order" in c.lower() and "level" in c.lower():
                ro_col = c
                break
    sub = S4[S4[ds_col].astype(str).str.strip() == dataset_name].copy()
    m = {}
    for _, r in sub.iterrows():
        m[str(r[ph_col]).strip()] = r[ro_col]
    return m


def compute_wdm_for(S11, S4, dataset_name):
    """Return dict of method -> our wdm-tau vs S4 relative-order target, using Kang weights."""
    cells = S11[S11["Dataset"] == dataset_name].copy()
    if len(cells) == 0:
        return None

    # Map phenotype -> relative-order level using S4
    ro_map = relative_order_map(S4, dataset_name)
    if not ro_map:
        return None

    cells["ro_level"] = cells["Standardized phenotypea"].astype(str).str.strip().map(ro_map)
    cells["ro_level"] = pd.to_numeric(cells["ro_level"], errors="coerce")
    cells = cells[cells["ro_level"].notna()].copy()
    if len(cells) < 3:
        return None

    for col in S12_TO_S11.values():
        cells[col] = pd.to_numeric(cells[col], errors="coerce")

    # Kang S5 absolute-order weighting: 1 / (n_cells_per_phenotype * n_phenotypes_per_level)
    # Applied on RELATIVE order levels (per S12 target)
    w_kang = kang_absolute_weights(cells["Standardized phenotypea"], cells["ro_level"])
    target = -cells["ro_level"].values  # more potent = higher; target = -level

    out = {"n_cells": int(len(cells)), "n_ro_levels": int(cells["ro_level"].nunique())}
    for label, col in S12_TO_S11.items():
        x = cells[col].values
        tau = wdm_tau_b_bit(x, target, w_kang)
        out[label] = float(tau) if not (isinstance(tau, float) and math.isnan(tau)) else None
    return out


def main():
    print("=" * 80)
    print("PI-gate mandate 3: wdm emulation vs Kang S12 across all Test-cohort rows")
    print("=" * 80)

    S4, S11, S12 = load_tables()

    # Test-cohort rows in S12
    S12_test = S12[S12["Cohort"] == "Test"].copy().reset_index(drop=True)
    print(f"\nS12 Test-cohort rows: {len(S12_test)}")

    per_row = []
    for _, r in S12_test.iterrows():
        ds = str(r["Dataset"]).strip()
        # try to resolve to S11 name (S11 may have footnote suffix)
        s11_match = ds
        if s11_match not in S11["Dataset"].unique():
            # try footnote-strip variants like Session 2
            candidates = [x for x in S11["Dataset"].unique() if str(x).startswith(ds)]
            if candidates:
                s11_match = candidates[0]
            else:
                candidates = [x for x in S11["Dataset"].unique() if ds.startswith(str(x))]
                if candidates:
                    s11_match = candidates[0]
                else:
                    print(f"  UNRESOLVED in S11: {ds}")
                    per_row.append({
                        "dataset": ds, "s11_match": None, "status": "unresolved",
                        "kang_published": {m: float(r[m]) if pd.notna(r[m]) else None for m in METHODS},
                    })
                    continue

        wdm_res = compute_wdm_for(S11, S4, s11_match)
        if wdm_res is None:
            print(f"  NO S4 relative-order mapping: {ds} (s11={s11_match})")
            per_row.append({
                "dataset": ds, "s11_match": s11_match, "status": "no_S4_map",
                "kang_published": {m: float(r[m]) if pd.notna(r[m]) else None for m in METHODS},
            })
            continue

        deltas = {}
        for m in METHODS:
            ours = wdm_res.get(m)
            kang = float(r[m]) if pd.notna(r[m]) else None
            if ours is None or kang is None:
                deltas[m] = None
            else:
                deltas[m] = abs(ours - kang)

        print(f"\n  {ds}  (s11={s11_match}, n_cells={wdm_res['n_cells']}, n_ro_levels={wdm_res['n_ro_levels']})")
        for m in METHODS:
            ours = wdm_res.get(m); kang = float(r[m]) if pd.notna(r[m]) else None
            d = deltas[m]
            ours_s = f"{ours:+.4f}" if ours is not None else "n/a"
            kang_s = f"{kang:+.4f}" if kang is not None else "n/a"
            d_s = f"{d:.4f}" if d is not None else "n/a"
            flag = " *" if (d is not None and d > 0.02) else ""
            print(f"    {m:15s} ours={ours_s}  kang={kang_s}  |Δ|={d_s}{flag}")

        per_row.append({
            "dataset": ds,
            "s11_match": s11_match,
            "n_cells": wdm_res["n_cells"],
            "n_ro_levels": wdm_res["n_ro_levels"],
            "ours":           {m: wdm_res.get(m) for m in METHODS},
            "kang_published": {m: (float(r[m]) if pd.notna(r[m]) else None) for m in METHODS},
            "abs_delta":      deltas,
        })

    # Aggregate per-method max |Δ|
    print("\n" + "=" * 80)
    print("PER-METHOD SUMMARY")
    print("=" * 80)
    summary = {}
    for m in METHODS:
        vs = [(row["dataset"], row["abs_delta"][m]) for row in per_row
              if row.get("abs_delta") and row["abs_delta"].get(m) is not None]
        if not vs:
            summary[m] = {"n": 0, "max": None, "worst_row": None}
            continue
        max_d = max(vs, key=lambda x: x[1])
        summary[m] = {
            "n": len(vs),
            "max_abs_delta": float(max_d[1]),
            "worst_row": max_d[0],
            "median_abs_delta": float(np.median([v[1] for v in vs])),
        }
        print(f"  {m:15s}  n={len(vs):2d}  max|Δ|={max_d[1]:.4f} (worst: {max_d[0]})  "
              f"median|Δ|={summary[m]['median_abs_delta']:.4f}")

    all_deltas = [d for row in per_row if row.get("abs_delta")
                  for m, d in row["abs_delta"].items() if d is not None]
    if all_deltas:
        overall_max = max(all_deltas)
        overall_med = float(np.median(all_deltas))
        print(f"\n  Overall (all {len(all_deltas)} row×method cells): "
              f"max|Δ|={overall_max:.4f}  median|Δ|={overall_med:.4f}")
        n_over_002 = sum(1 for d in all_deltas if d > 0.02)
        print(f"  Cells exceeding 0.02: {n_over_002} / {len(all_deltas)} ({n_over_002/len(all_deltas)*100:.1f}%)")

    result = {
        "n_rows_tested": len(per_row),
        "n_rows_S12_test_cohort": len(S12_test),
        "per_row": per_row,
        "per_method_summary": summary,
        "overall_max_abs_delta": float(overall_max) if all_deltas else None,
        "overall_median_abs_delta": float(overall_med) if all_deltas else None,
    }
    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWritten: {OUT_PATH}")


if __name__ == "__main__":
    main()
