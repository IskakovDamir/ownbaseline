"""
pi_gate_recompute.py
====================
PI-gate 2026-07-17: independently recompute CT2 median Δ vs gene_counts on the
23 H1 rows, under both kernels, from the live per_dataset_results.json.

Purpose: resolve the same-document contradiction in the dossier (§4 vs §8):
  §4 lines 108-109: scipy +0.042, Kang +0.088
  §8 lines 198-199: scipy +0.144, Kang +0.207

Report authoritative values and root-cause the discrepancy. Also verify H1 numbers.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
JSON_PATH = HERE / "per_dataset_results.json"


def compute_delta_stats(per_ds, score_tag, kernel, verbose=False):
    """Δ_d(S) = τ(S, GT_d) − τ(gene_counts, GT_d); returns median, IQR."""
    deltas = []
    rows_used = []
    for r in per_ds:
        tk = r.get("tau_" + kernel, {})
        base = tk.get("gene_counts")
        val = tk.get(score_tag)
        if base is None or val is None or (isinstance(base, float) and math.isnan(base)) or \
           (isinstance(val, float) and math.isnan(val)):
            continue
        d = val - base
        deltas.append(d)
        rows_used.append((r["dataset_name"], base, val, d))
    a = np.array(deltas)
    med = float(np.median(a))
    lo = float(np.percentile(a, 25))
    hi = float(np.percentile(a, 75))
    if verbose:
        print(f"\n  === {score_tag} vs gene_counts under {kernel} kernel (n={len(a)}) ===")
        print(f"  {'dataset':45s} {'tau_gc':>8s} {'tau_S':>8s} {'delta':>8s}")
        for name, b, v, d in sorted(rows_used, key=lambda x: x[3]):
            print(f"  {name:45s} {b:+.4f} {v:+.4f} {d:+.4f}")
        print(f"  median Δ = {med:+.4f}    IQR = [{lo:+.4f}, {hi:+.4f}]")
    return {"n": len(a), "median": med, "iqr_lo": lo, "iqr_hi": hi, "deltas": deltas}


def median_of_medians(per_ds, score_tag, kernel):
    """Alternative statistic: (median across d of τ_S) − (median across d of τ_gc).
    This is Δ-of-medians, NOT median-of-Δ; used to check whether §8's numbers came
    from this alternative aggregation."""
    xs, ys = [], []
    for r in per_ds:
        tk = r.get("tau_" + kernel, {})
        base = tk.get("gene_counts")
        val = tk.get(score_tag)
        if base is None or val is None:
            continue
        if math.isnan(base) or math.isnan(val):
            continue
        xs.append(val)
        ys.append(base)
    if not xs:
        return None
    return float(np.median(xs)) - float(np.median(ys))


def main():
    with open(JSON_PATH) as f:
        blob = json.load(f)
    per_ds = blob["per_dataset"]
    print(f"Loaded {len(per_ds)} datasets from {JSON_PATH.name}")

    # Verify H1 (should match dossier §4 numbers for SR, CCAT, StemID)
    print("\n" + "=" * 80)
    print("H1 primary verification — median Δ per unsupervised score vs gene_counts")
    print("=" * 80)
    for tag in ["SCENT (SR)", "SCENT (CCAT)", "StemID"]:
        for kernel in ["scipy", "kang"]:
            r = compute_delta_stats(per_ds, tag, kernel)
            print(f"  {tag:15s} [{kernel:5s}]  n={r['n']:2d}  median={r['median']:+.4f}  "
                  f"IQR=[{r['iqr_lo']:+.4f}, {r['iqr_hi']:+.4f}]")

    # THE contradiction: CT2 median Δ vs gene_counts
    print("\n" + "=" * 80)
    print("CT2 (supervised contrast): median Δ vs gene_counts — RESOLVE CONTRADICTION")
    print("=" * 80)
    for kernel in ["scipy", "kang"]:
        r = compute_delta_stats(per_ds, "CytoTRACE 2 potency score", kernel, verbose=True)
        print(f"  AUTHORITATIVE: CT2 median Δ [{kernel}] = {r['median']:+.4f}  "
              f"IQR = [{r['iqr_lo']:+.4f}, {r['iqr_hi']:+.4f}]  (n={r['n']})")

    # Cross-check: could §8's numbers come from median-of-medians (Δ-of-medians)?
    print("\n" + "=" * 80)
    print("Cross-check: (median τ_CT2) − (median τ_gc) — different statistic")
    print("=" * 80)
    for kernel in ["scipy", "kang"]:
        d = median_of_medians(per_ds, "CytoTRACE 2 potency score", kernel)
        print(f"  {kernel}: (median CT2) − (median gc) = {d:+.4f}")

    # Compare against dossier's two conflicting claims:
    print("\n" + "=" * 80)
    print("DOSSIER CONTRADICTION AUDIT")
    print("=" * 80)
    print("  §4 lines 108-109 claim:  scipy = +0.042 [IQR -0.001, +0.277];  Kang = +0.088 [IQR -0.002, +0.266]")
    print("  §8 lines 198-199 claim:  scipy = +0.144 [IQR +0.075, +0.244];  Kang = +0.207 [IQR +0.108, +0.298]")
    print("  See authoritative values above.")


if __name__ == "__main__":
    main()
