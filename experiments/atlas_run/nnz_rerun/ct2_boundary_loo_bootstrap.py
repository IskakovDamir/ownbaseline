"""
A3 LOO stability + bootstrap 95 % CI on n=4 raw-nnz CT2 boundary.

II.7 close-out FRAMING-CORRECTION cycle (2026-07-17).

Purpose: apply the II.3′ prereg §3 decision rule (PASS ≤ 0.05, INCONCLUSIVE
0.05–0.10, FAIL > 0.10) to the n=4 raw-nnz atlas subset (dossier §S6 A1).
Report LOO drop-one stability per kernel and dataset-level bootstrap 95 % CI
on the median with seed=42, B=10 000. Reported at face value.

Input: `ct2_boundary_diagnosis.json` in this directory (deltas per row).
Output: `ct2_boundary_loo_bootstrap.json` next to it.

No new data, no re-run of atlas or raw-nnz; only lookups + arithmetic.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def verdict(delta: float) -> str:
    if delta > 0.10:
        return "FAIL"
    if delta > 0.05:
        return "INCONCLUSIVE"
    return "PASS"


def main() -> None:
    here = Path(__file__).parent
    j = json.loads((here / "ct2_boundary_diagnosis.json").read_text())

    rows = j["per_row"]
    labels = [r["dataset"] for r in rows]
    scipy_vals = np.array([r["delta_ct2_vs_nnz_scipy"] for r in rows])
    kang_vals = np.array([r["delta_ct2_vs_nnz_kang"] for r in rows])

    full_scipy = float(np.median(scipy_vals))
    full_kang = float(np.median(kang_vals))

    rng = np.random.default_rng(42)
    B = 10_000

    def boot_median_ci(vals: np.ndarray) -> tuple[float, float]:
        idx = rng.integers(0, len(vals), size=(B, len(vals)))
        meds = np.median(vals[idx], axis=1)
        return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))

    s_lo, s_hi = boot_median_ci(scipy_vals)
    k_lo, k_hi = boot_median_ci(kang_vals)

    loo = []
    for i, lab in enumerate(labels):
        s = float(np.median(np.delete(scipy_vals, i)))
        k = float(np.median(np.delete(kang_vals, i)))
        loo.append(
            {
                "dropped_dataset": lab,
                "n3_median_scipy": s,
                "n3_median_kang": k,
                "n3_verdict_scipy": verdict(s),
                "n3_verdict_kang": verdict(k),
            }
        )

    out = {
        "task": "A3 LOO stability + bootstrap CI on n=4 raw-nnz CT2 boundary",
        "cycle": "II.7 close-out FRAMING-CORRECTION (2026-07-17)",
        "n": 4,
        "labels": labels,
        "delta_scipy": scipy_vals.tolist(),
        "delta_kang": kang_vals.tolist(),
        "full_median_scipy": full_scipy,
        "full_median_kang": full_kang,
        "full_verdict_scipy": verdict(full_scipy),
        "full_verdict_kang": verdict(full_kang),
        "loo": loo,
        "bootstrap_seed": 42,
        "bootstrap_B": B,
        "bootstrap_ci_scipy": [s_lo, s_hi],
        "bootstrap_ci_kang": [k_lo, k_hi],
        "decision_rule_ii3prereg": {
            "PASS": "<=0.05",
            "INCONCLUSIVE": "0.05-0.10",
            "FAIL": ">0.10",
        },
        "notes": (
            "n=4 median bootstrap of medians is coarse by construction "
            "(only 6 distinct combinations at the median); wide CIs are expected. "
            "Both CIs entirely to the right of zero; lower bounds touch the II.3′ "
            "PASS band but the point estimate + all 4 LOO n=3 medians are FAIL. "
            "Reported at face value."
        ),
    }

    outpath = here / "ct2_boundary_loo_bootstrap.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"Wrote: {outpath}")
    print(f"Full n=4 medians: scipy {full_scipy:+.4f} ({verdict(full_scipy)}), "
          f"Kang {full_kang:+.4f} ({verdict(full_kang)})")
    print(f"LOO n=3 verdicts: scipy {[r['n3_verdict_scipy'] for r in loo]}, "
          f"Kang {[r['n3_verdict_kang'] for r in loo]}")
    print(f"Bootstrap 95% CI: scipy [{s_lo:+.4f}, {s_hi:+.4f}], "
          f"Kang [{k_lo:+.4f}, {k_hi:+.4f}]")


if __name__ == "__main__":
    main()
