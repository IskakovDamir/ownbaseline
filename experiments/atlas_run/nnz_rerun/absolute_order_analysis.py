"""
absolute_order_analysis.py
===========================
Session 4 Task B: verify Kang S13 numbers verbatim; test PI's "scaffold-anchoring
mechanism" reading of the absolute-order row via cross-dataset variance
decomposition on Kang S11.

Aggregation definitions (from Kang 2025 Methods, S13 legend):

  Absolute order:  pool all cells across all Test datasets (14 rows) into one
                   vector, compute one weighted-τ against absolute-potency labels.
                   Numerator: cross-dataset comparability of the score's absolute scale.

  Median relative order: per-dataset τ against per-dataset relative-order (1..N),
                         then median across 13 datasets.
                         Numerator: within-dataset ordering skill only.

  Our H1: per-dataset τ against broad-potency ordinal (1..6), then median across
          23 datasets. (Own-baseline Δ subtracted.)
          Numerator: within-dataset ordering skill, own-baseline.

PI's mechanistic reading (to be TESTED):
  CT1 is anchored to a WITHIN-dataset quantity (gene-count rank), so its scores
  are not comparable across datasets and it collapses on pooled absolute order.
  SR/CCAT are anchored to an EXTERNAL FIXED scaffold (PPI graph / degree vector),
  identical across every dataset, so their scores are automatically cross-dataset
  comparable and they "transfer".

Empirical test on Kang S11:
  For each score S ∈ {CT1, SR, CCAT, StemID, CT2}:
    - Per-dataset mean of S (23 rows H1 subset — hold Test cohort separate).
    - Per-dataset std of S.
    - Cross-dataset variance of per-dataset MEANS = σ²(between).
    - Median within-dataset std/mean ratio.

  If scaffold-anchoring hypothesis holds:
    σ²(between-datasets, SR/CCAT) < σ²(between-datasets, CT1)
    (SR/CCAT means cluster tightly across datasets because they're all
     evaluations against the same PPI scaffold; CT1 means vary wildly because
     gene counts depend on QC / platform / cell type mix).

  If scaffold-anchoring hypothesis fails:
    Between-dataset variances would be roughly equal across scores.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).parent
SI = HERE.parent / "kang2025_SI"


def main():
    S13 = pd.read_csv(SI / "Table_S13.csv", header=5).dropna(how="all").dropna(axis=1, how="all")
    S11 = pd.read_csv(SI / "Table_S11.csv", header=4).dropna(how="all").dropna(axis=1, how="all")

    print("=" * 80)
    print("Task B — Kang S13 verbatim verification + absolute-order mechanism test")
    print("=" * 80)

    # ---- B1. Verbatim quote of S13 rows for CT1, SR, CCAT, StemID ----
    print("\nB1. S13 method rows (verbatim from Table_S13.csv):\n")
    S13.columns = [c.strip() for c in S13.columns]
    # Find the method-name column
    print("Columns:", list(S13.columns))
    method_col = [c for c in S13.columns if "Comput" in c or "trategie" in c or "gene set" in c][0]
    core_col = [c for c in S13.columns if "Core" in c][0]
    abs_test_col = [c for c in S13.columns if "Absolute" in c or "n = 14" in c][0]
    med_test_col = [c for c in S13.columns if "Median" in c or "n = 13" in c][0]
    print(f"Using: method='{method_col}', core='{core_col}', abs_test='{abs_test_col}', med_test='{med_test_col}'")

    METHODS = ["CytoTRACE 2", "CytoTRACE 1", "SCENT (CCAT)", "SCENT (SR)", "StemID"]
    rows_out = {}
    for m in METHODS:
        r = S13[S13[method_col].astype(str).str.strip() == m]
        if len(r) == 0:
            print(f"  MISSING: {m}")
            continue
        rec = r.iloc[0]
        core = str(rec[core_col]).strip()
        try:
            abs_v = float(rec[abs_test_col])
        except Exception:
            abs_v = None
        try:
            med_v = float(rec[med_test_col])
        except Exception:
            med_v = None
        print(f"  {m:15s} core='{core:20s}' abs(Test n=14)={abs_v:+.5f}  med_rel(Test n=13)={med_v:+.5f}")
        rows_out[m] = {"core_method": core, "abs_order_test_n14": abs_v, "med_rel_test_n13": med_v}

    # ---- B2. Aggregation prose (embedded in main dossier) ----
    print("\nB2. Aggregation definitions embedded in output JSON.\n")

    # ---- B3. Cross-dataset variance decomposition on Kang S11 ----
    print("\nB3. Cross-dataset variance decomposition (H1 subset, 23 rows):\n")

    # Load the 23 rows list from per_dataset_results.json
    with open(HERE.parent / "per_dataset_results.json") as f:
        pdr = json.load(f)
    h1_names = [r["s11_dataset_name"] for r in pdr["per_dataset"] if r.get("s11_dataset_name")]
    print(f"  H1 subset: {len(h1_names)} rows")

    S11_sub = S11[S11["Dataset"].isin(h1_names)].copy()
    print(f"  Kang S11 subset: {len(S11_sub)} cells across {S11_sub['Dataset'].nunique()} datasets")

    score_cols = {
        "CytoTRACE 1 (gene counts)": "CytoTRACE 1",
        "SCENT (SR)": "SCENT (SR)",
        "SCENT (CCAT)": "SCENT (CCAT)",
        "StemID": "StemID",
        "CytoTRACE 2 potency score": "CytoTRACE 2 potency score",
    }

    print(f"\n  {'Score':30s} | between-ds σ² | between-ds σ | median within-ds σ | median CV | between/within")
    print(f"  {'-'*30} + ------------- + ----------- + ------------------ + --------- + --------------")

    per_score = {}
    for label, col in score_cols.items():
        S11_sub[col] = pd.to_numeric(S11_sub[col], errors="coerce")
        per_ds = S11_sub.groupby("Dataset")[col].agg(["mean", "std", "count"]).dropna()
        # Between-dataset variance = variance of per-dataset means
        between_var = float(per_ds["mean"].var(ddof=1))
        between_std = float(math.sqrt(between_var))
        # Median within-dataset std
        within_std = float(per_ds["std"].median())
        # Median coefficient of variation (std/mean) per dataset — use absolute mean to avoid sign flip
        # For scores that can be negative (CT2, SR, CCAT), CV is not meaningful; use std / |mean| instead
        cv_per_ds = (per_ds["std"] / per_ds["mean"].abs()).replace([np.inf, -np.inf], np.nan).dropna()
        median_cv = float(cv_per_ds.median()) if len(cv_per_ds) > 0 else float("nan")
        # Ratio between/within — higher = more between-dataset heterogeneity vs within-ds
        b_over_w = between_std / within_std if within_std > 0 else float("nan")
        per_score[label] = {
            "between_var": between_var,
            "between_std": between_std,
            "median_within_std": within_std,
            "median_cv": median_cv,
            "between_over_within": b_over_w,
        }
        print(f"  {label:30s} | {between_var:12.6f} | {between_std:10.4f} | {within_std:17.4f} | {median_cv:8.3f} | {b_over_w:10.3f}")

    # PI's prediction: SR/CCAT should have SMALLER between-dataset variance than CT1
    ct1 = per_score.get("CytoTRACE 1 (gene counts)")
    sr = per_score.get("SCENT (SR)")
    ccat = per_score.get("SCENT (CCAT)")
    stemid = per_score.get("StemID")

    verdict = {}
    if ct1 is not None:
        verdict["between_var_SR_lt_CT1"] = (sr["between_var"] < ct1["between_var"]) if sr else None
        verdict["between_var_CCAT_lt_CT1"] = (ccat["between_var"] < ct1["between_var"]) if ccat else None
        verdict["between_std_SR_over_CT1"] = sr["between_std"] / ct1["between_std"] if sr and ct1["between_std"] > 0 else None
        verdict["between_std_CCAT_over_CT1"] = ccat["between_std"] / ct1["between_std"] if ccat and ct1["between_std"] > 0 else None

    print(f"\n  PI's prediction test:")
    print(f"    SR between-var < CT1 between-var? {verdict.get('between_var_SR_lt_CT1')}")
    print(f"    CCAT between-var < CT1 between-var? {verdict.get('between_var_CCAT_lt_CT1')}")
    print(f"    SR/CT1 between-std ratio: {verdict.get('between_std_SR_over_CT1'):.3f}")
    print(f"    CCAT/CT1 between-std ratio: {verdict.get('between_std_CCAT_over_CT1'):.3f}")

    # Interpretation
    if (verdict.get("between_var_SR_lt_CT1") and verdict.get("between_var_CCAT_lt_CT1")):
        interp = ("SR AND CCAT both have smaller cross-dataset variance in their per-dataset means "
                  "than CT1 does. This is consistent with the PI's reading: SR/CCAT are anchored "
                  "to the same external scaffold and their scores are on a stable cross-dataset "
                  "scale, while CT1's scale depends on the dataset's gene-count distribution. "
                  "Kang's absolute-order row (SR/CCAT beat CT1 8x) is then a mechanistic "
                  "consequence of scaffold-anchoring, not conserved biology.")
    else:
        interp = ("PI's mechanistic reading NOT clearly supported by variance decomposition. "
                  "Between-dataset variances are close or SR/CCAT are higher-variance than CT1. "
                  "The scaffold-anchoring mechanism may still be part of the story but is not the "
                  "cleanest explanation of the absolute-order gap. Requires refinement.")
    print(f"\n  Interpretation:\n    {interp}\n")

    out = {
        "s13_verbatim": rows_out,
        "aggregation_definitions": {
            "absolute_order": "Pool cells across all datasets (14 Test rows); ONE weighted-τ against absolute-potency labels. Tests cross-dataset comparability of score scale.",
            "median_relative_order_kang": "Per-dataset τ against per-dataset relative-order (1..N); median across 13 Test datasets. Tests within-dataset ordering skill only, per-dataset comparability of RANK.",
            "our_H1": "Per-dataset τ against broad-potency ordinal (1..6); median across 23 datasets (own-baseline Δ subtracted). Tests within-dataset ordering skill, own-baseline framing.",
        },
        "cross_dataset_variance_test": per_score,
        "pi_prediction_verdict": verdict,
        "interpretation": interp,
    }
    with open(HERE / "cross_dataset_variance_test.json", "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"Written: {HERE / 'cross_dataset_variance_test.json'}")


if __name__ == "__main__":
    main()
