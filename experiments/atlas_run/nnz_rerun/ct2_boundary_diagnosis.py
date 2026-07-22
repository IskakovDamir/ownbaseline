"""
ct2_boundary_diagnosis.py
=========================
II.7 close-out Task A1 (2026-07-17): CT2's median Δ against RAW nnz baseline on
the 4 raw-nnz atlas rows for which Session 4 obtained per-cell nnz.

Purpose: separate the baseline-switch effect (raw nnz vs CT1) from the sample-size
effect (n=3 II.3′ vs n=23 atlas) in the CT2 median-Δ shrinkage.

Inputs:
  - atlas_run/per_dataset_results.json  → CT2 τ per row (both kernels), CT1 τ (as "gene_counts")
  - atlas_run/nnz_rerun/h1_raw_nnz_results.json → per-row τ(nnz) both kernels for the 4 rows

Method: for each of the 4 rows, compute Δ_CT2 = τ(CT2) − τ(nnz) (raw nnz baseline)
under scipy and Kang kernels. Report median across n=4. Compare vs:
  (a) II.3′ n=3 value +0.175 (raw-nnz baseline, hypothesis-generating)
  (b) II.7 n=23 CT1-atlas-proxy value +0.042 scipy / +0.088 Kang

Determination:
  If A1's n=4 raw-nnz Δ ≈ +0.175 → baseline switch (CT1) drove the shrinkage
                                   at atlas scale (n-effect small).
  If A1's n=4 raw-nnz Δ ≈ +0.042/+0.088 → n-effect drove the shrinkage.
  If in between → both effects contribute.

Author: researcher-explorer II.7 close-out. No SSOT / FULL-DRAFT writes.
"""
from __future__ import annotations

import json
from pathlib import Path
from statistics import median

HERE = Path(__file__).parent
ATLAS_PDR = HERE.parent / "per_dataset_results.json"
RAW_NNZ = HERE / "h1_raw_nnz_results.json"

OUT = HERE / "ct2_boundary_diagnosis.json"

# Target rows: the 4 for which raw nnz was obtained in Session 4.
# Atlas row names (dataset_name field in per_dataset_results.json) MUST match
# h1_raw_nnz_results.json per_row_results keys.
TARGETS = [
    "HSC development (Smart-seq2)",       # E-MTAB-9067
    "Mouse embryo 1 (Tang et al.)",        # GSE45719
    "Intestine (Drop-seq)",                # GSE92332 Drop-seq
    "BM-MNC (CITE-seq)e",                  # GSE128639
]

II3_RAW_NNZ = 0.17461     # II.3′ n=3 raw-nnz median Δ (Bastidas-Ponce/Stoeckius/Paul15)
II7_CT1_SCIPY = 0.042     # II.7 n=23 CT1-atlas-proxy median Δ scipy
II7_CT1_KANG = 0.088      # II.7 n=23 CT1-atlas-proxy median Δ Kang


def main():
    atlas = json.load(open(ATLAS_PDR))
    per = {r["dataset_name"]: r for r in atlas["per_dataset"]}
    raw_nnz = json.load(open(RAW_NNZ))
    per_raw = raw_nnz["per_row_results"]

    rows = []
    for name in TARGETS:
        assert name in per, f"{name} missing from per_dataset_results.json"
        assert name in per_raw, f"{name} missing from h1_raw_nnz_results.json"
        row_atlas = per[name]
        row_raw = per_raw[name]

        tau_ct2_scipy = row_atlas["tau_scipy"]["CytoTRACE 2 potency score"]
        tau_ct2_kang = row_atlas["tau_kang"]["CytoTRACE 2 potency score"]
        # CT1 = the "gene_counts" column in atlas_run.py, which is Kang's CT1 column.
        tau_ct1_scipy = row_atlas["tau_scipy"]["gene_counts"]
        tau_ct1_kang = row_atlas["tau_kang"]["gene_counts"]

        # Raw nnz τ from Session 4 (h1_raw_nnz_results.json)
        tau_nnz_scipy = row_raw["tau_nnz_scipy"]
        tau_nnz_kang = row_raw["tau_nnz_kang"]

        # Δ CT2 vs raw nnz (the A1 quantity)
        delta_ct2_nnz_scipy = tau_ct2_scipy - tau_nnz_scipy
        delta_ct2_nnz_kang = tau_ct2_kang - tau_nnz_kang
        # Δ CT2 vs CT1 (for comparison; matches atlas-scale CT2 headline)
        delta_ct2_ct1_scipy = tau_ct2_scipy - tau_ct1_scipy
        delta_ct2_ct1_kang = tau_ct2_kang - tau_ct1_kang
        # Δ CT1 vs raw nnz (H1's baseline gap on these rows)
        delta_ct1_nnz_scipy = tau_ct1_scipy - tau_nnz_scipy
        delta_ct1_nnz_kang = tau_ct1_kang - tau_nnz_kang

        rows.append({
            "dataset": name,
            "tau_ct2_scipy": tau_ct2_scipy,
            "tau_ct2_kang": tau_ct2_kang,
            "tau_ct1_scipy": tau_ct1_scipy,
            "tau_ct1_kang": tau_ct1_kang,
            "tau_nnz_scipy": tau_nnz_scipy,
            "tau_nnz_kang": tau_nnz_kang,
            "delta_ct2_vs_nnz_scipy": delta_ct2_nnz_scipy,
            "delta_ct2_vs_nnz_kang": delta_ct2_nnz_kang,
            "delta_ct2_vs_ct1_scipy": delta_ct2_ct1_scipy,
            "delta_ct2_vs_ct1_kang": delta_ct2_ct1_kang,
            "delta_ct1_vs_nnz_scipy": delta_ct1_nnz_scipy,
            "delta_ct1_vs_nnz_kang": delta_ct1_nnz_kang,
        })

    # Aggregate summary
    med_ct2_nnz_scipy = median([r["delta_ct2_vs_nnz_scipy"] for r in rows])
    med_ct2_nnz_kang = median([r["delta_ct2_vs_nnz_kang"] for r in rows])
    med_ct2_ct1_scipy = median([r["delta_ct2_vs_ct1_scipy"] for r in rows])
    med_ct2_ct1_kang = median([r["delta_ct2_vs_ct1_kang"] for r in rows])
    med_ct1_nnz_scipy = median([r["delta_ct1_vs_nnz_scipy"] for r in rows])
    med_ct1_nnz_kang = median([r["delta_ct1_vs_nnz_kang"] for r in rows])

    out = {
        "task": "II.7 close-out A1 — CT2 boundary diagnosis on 4 raw-nnz atlas rows",
        "targets_n": len(rows),
        "per_row": rows,
        "medians_across_4_rows": {
            "delta_ct2_vs_raw_nnz_scipy": med_ct2_nnz_scipy,
            "delta_ct2_vs_raw_nnz_kang": med_ct2_nnz_kang,
            "delta_ct2_vs_ct1_scipy": med_ct2_ct1_scipy,
            "delta_ct2_vs_ct1_kang": med_ct2_ct1_kang,
            "delta_ct1_vs_raw_nnz_scipy": med_ct1_nnz_scipy,
            "delta_ct1_vs_raw_nnz_kang": med_ct1_nnz_kang,
        },
        "comparators": {
            "ii3_n3_raw_nnz_median_delta": II3_RAW_NNZ,
            "ii7_n23_ct1_proxy_scipy": II7_CT1_SCIPY,
            "ii7_n23_ct1_proxy_kang": II7_CT1_KANG,
        },
        "decision_rule_ii3prereg": {
            "PASS": "delta <= 0.05",
            "INCONCLUSIVE": "0.05 < delta <= 0.10",
            "FAIL": "delta > 0.10",
        },
        "note": (
            "A1 diagnostic: does the CT2-vs-raw-nnz Δ on n=4 atlas rows match the "
            "II.3′ n=3 value (+0.175, hypothesis-generating), the II.7 n=23 CT1-proxy value "
            "(+0.042 scipy / +0.088 Kang), or something in between? "
            "If close to II.3′ → the atlas shrinkage was baseline-driven. "
            "If close to II.7 → the n=3 result was hypothesis-generating noise. "
            "Reported at face value; interpretation belongs in the dossier."
        ),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
