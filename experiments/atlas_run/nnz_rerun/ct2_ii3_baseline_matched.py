"""
ct2_ii3_baseline_matched.py
===========================
II.7 close-out Task A2 (2026-07-17, DECISIVE diagnostic):
Recompute CT2's Δ on the II.3′ 3 datasets using the CT1-atlas-proxy baseline
(same n=3 as II.3′, but with the new baseline that II.7 shipped).

Rationale
---------
II.3′ n=3 raw-nnz median Δ = +0.175 → A-FAIL per prereg §3 (>0.10).
II.7 n=23 CT1-proxy median Δ = +0.042 scipy / +0.088 Kang → PASS/INCONCLUSIVE.

The FULL-DRAFT states +0.18 "outside the fixed-scaffold family" — contradicted
by the II.7 result at n=23. A1 (ct2_boundary_diagnosis.py) established that on
the 4 raw-nnz atlas rows, CT2-vs-raw-nnz Δ is +0.389 scipy / +0.440 Kang — MUCH
larger than II.3′ or II.7. This suggests the four rows are NOT representative of
either the II.3′ 3-dataset probe or the 23-row atlas.

The DECISIVE separation: does changing ONLY the baseline (raw nnz → CT1-proxy) on
the II.3′ 3 datasets shrink Δ from +0.175 to ~+0.088 (baseline-driven) or leave
it near +0.175 (n-driven, meaning II.7 n=23 shrinkage came from more datasets)?

Method
------
Pancreas + Cord blood:
  Both II.3′ datasets are SUBSETS of Kang atlas datasets:
    - Pancreas (10x): 2850 II.3′ cells fully contained in Kang's 11401 cells.
    - Cord blood (CITE-seq): all 2308 II.3′ cells contained in Kang's 7690 cells.
  For each II.3′ cell, look up its CT2 and CT1 scores in Kang S11.
  Compute τ(CT2, GT_ii3prereg) and τ(CT1, GT_ii3prereg) using the II.3′ potency map
  (NOT Kang's broad-potency ordinal — the II.3′ map is what the +0.175 was
  computed with, so this isolates the baseline effect from any GT-map effect).

Paul15:
  NOT in Kang S11 (Paul et al. 2015 hematopoiesis not in Kang's 33 GT datasets).
  We do NOT have CT1 for Paul15 without running the CytoTRACE v1 pipeline
  (out-of-scope for this session). Report as unavailable; n=2 baseline-matched.

Decision rule (II.3′ prereg §3, verbatim)
------------------------------------------
- Δ_med ≤ 0.05          → PASS ("ordering low-order")
- 0.05 < Δ_med ≤ 0.10   → INCONCLUSIVE
- Δ_med > 0.10          → A-FAIL

Reproducibility
---------------
- Vignette 1 CT2 CSV downloaded from digitalcytometry/cytotrace2 main branch.
- Vignette 2 RDS: Seurat obj with cell IDs, cached under $OWNBASELINE_SCRATCH/ct2_probe/data/.
- Kang S11: local vault CSV.
- II.3′ potency maps: read from ct2_probe results JSONs.

Author: researcher-explorer II.7 close-out. No SSOT / FULL-DRAFT writes.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path
from statistics import median

import numpy as np
from scipy.stats import weightedtau

HERE = Path(__file__).parent
KANG_S11 = HERE.parent / "kang2025_SI" / "Table_S11.csv"
# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# agent-session scratchpad, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root, scratch_root  # noqa: E402
CT2_PROBE_RESULTS = data_root() / "ct2_probe/results"
VIGNETTE1_LOCAL = scratch_root() / "vignette1.csv"
VIGNETTE2_RDS = scratch_root() / "ct2_probe/data/Vignette2_CytoTRACE2_results.rds"

OUT = HERE / "ct2_ii3_baseline_matched.json"


# --- Kang S11 lookup helpers ---
def load_kang_s11(dataset_name: str) -> dict:
    """Return {cell_id: {"CT2": float, "CT1": float, "std_phenotype": str}}."""
    rec = {}
    with open(KANG_S11) as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < 5:
                continue
            if len(row) < 12 or row[2] != dataset_name:
                continue
            cell_id = row[1]
            try:
                ct2 = float(row[7])
                ct1 = float(row[8])
            except (ValueError, IndexError):
                continue
            rec[cell_id] = {"CT2": ct2, "CT1": ct1, "std_phenotype": row[4]}
    return rec


def wtau(pred: np.ndarray, gt: np.ndarray) -> float:
    """scipy.stats.weightedtau (hyperbolic rank weigher, prereg §2 default)."""
    m = ~(np.isnan(pred) | np.isnan(gt))
    return float(weightedtau(pred[m], gt[m]).statistic)


# --- II.3′ pancreas ---
def get_pancreas():
    """Return list of (cell_id_kang_format, ct2_kang, ct1_kang, gt_ii3prereg)."""
    # Load Kang Pancreas (10x)
    kang = load_kang_s11("Pancreas (10x)")
    # V1 cells: reformat underscore → dot
    with open(VIGNETTE1_LOCAL) as f:
        reader = csv.reader(f)
        header = next(reader)
        v1_ids_raw = [row[0] for row in reader]
    v1_ids_norm = [x.replace("_", ".") for x in v1_ids_raw]

    # II.3′ pancreas potency map (from ct2_probe/results/result_pancreas.json):
    # Multipotent → 3, Endocrine progenitor/committed/immature → 1, Alpha/Beta/Delta/Epsilon → 0
    # Kang std phenotypes in S11 for Pancreas (10x):
    #   "Multipotent pancreatic progenitor", "Endocrine progenitor",
    #   "Endocrine committed precursor cell", "Immature endocrine cell",
    #   "Alpha cell", "Beta cell", "Delta cell", "Epsilon cell"
    pot_map = {
        "Multipotent pancreatic progenitor": 3,
        "Endocrine progenitor": 1,
        "Endocrine committed precursor cell": 1,
        "Immature endocrine cell": 1,
        "Alpha cell": 0,
        "Beta cell": 0,
        "Delta cell": 0,
        "Epsilon cell": 0,
    }
    rows = []
    n_missing_kang = 0
    n_missing_phen = 0
    for cid in v1_ids_norm:
        if cid not in kang:
            n_missing_kang += 1
            continue
        pheno = kang[cid]["std_phenotype"]
        if pheno not in pot_map:
            n_missing_phen += 1
            continue
        rows.append({
            "cell_id": cid,
            "CT2": kang[cid]["CT2"],
            "CT1": kang[cid]["CT1"],
            "gt": pot_map[pheno],
            "std_phenotype": pheno,
        })
    return rows, {
        "n_v1_input": len(v1_ids_norm),
        "n_matched_kang_s11": len(rows),
        "n_missing_kang": n_missing_kang,
        "n_missing_phen": n_missing_phen,
    }


# --- II.3′ cord blood ---
def get_cordblood():
    kang = load_kang_s11("Cord blood (CITE-seq)")

    # Read cord blood cell IDs from the Vignette2 Seurat RDS via II.3′ venv.
    # Called this via $CT2_PROBE_PYTHON (an interpreter with rdata);
    # embed the extraction inline here as a subprocess call for reproducibility.
    import subprocess
    script = """
import rdata, json, warnings
warnings.filterwarnings('ignore')
p = rdata.parser.parse_file(RDS_PATH_PLACEHOLDER)
obj = rdata.conversion.convert(p)
cells = obj.assays['RNA'].cells['dim_0'].values.tolist()
# We also need CT2 scores per cell — extract from meta.data
# For Seurat objects with CytoTRACE2 pre-computed, scores are in meta.data columns.
# Use the ct2_probe/results/result_cordblood.json for I.3'.
print(json.dumps({'cells': cells}))
""".replace("RDS_PATH_PLACEHOLDER", repr(str(VIGNETTE2_RDS)))
    result = subprocess.run(
        [os.environ.get("CT2_PROBE_PYTHON", sys.executable), "-c", script],
        capture_output=True, text=True, timeout=60,
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr, file=sys.stderr)
        raise RuntimeError("Failed to extract cord blood cell IDs from RDS")
    parsed = json.loads(result.stdout.strip().split("\n")[-1])
    v2_ids = parsed["cells"]

    # II.3′ cord blood potency map (from result_cordblood.json):
    # HSPC → 2, all mature → 0
    # NOTE: Kang re-standardized phenotypes finer than the II.3′ source used;
    # II.3′ used "Hematopoietic stem and progenitor" (atlas standardized_phenotype);
    # Kang splits mature into Classical monocyte / Non-classical monocyte / Naive T cell / T cell / etc.
    # We map Kang's standardized_phenotype using the same II.3′ potency-level convention
    # (HSPC ordinal → 2; every mature phenotype → 0). This preserves the II.3′ binary
    # potency structure (HSPCs vs mature) while using Kang's finer std_phenotype.
    pot_map = {
        "Hematopoietic progenitor": 2,      # Kang's re-standardization of HSPC
        "T cell": 0,
        "Naive T cell": 0,
        "B cell": 0,
        "NK cell": 0,
        "Classical monocyte": 0,
        "Non-classical monocyte": 0,
        "Dendritic cell": 0,
        "Megakaryocyte": 0,
        "Erythrocyte": 0,
    }
    rows = []
    n_missing_kang = 0
    n_missing_phen = 0
    for cid in v2_ids:
        if cid not in kang:
            n_missing_kang += 1
            continue
        pheno = kang[cid]["std_phenotype"]
        if pheno not in pot_map:
            n_missing_phen += 1
            continue
        rows.append({
            "cell_id": cid,
            "CT2": kang[cid]["CT2"],
            "CT1": kang[cid]["CT1"],
            "gt": pot_map[pheno],
            "std_phenotype": pheno,
        })
    return rows, {
        "n_v2_input": len(v2_ids),
        "n_matched_kang_s11": len(rows),
        "n_missing_kang": n_missing_kang,
        "n_missing_phen": n_missing_phen,
    }


def compute_delta(rows, dataset_name: str) -> dict:
    if not rows:
        return {"dataset": dataset_name, "error": "no rows"}
    ct2 = np.array([r["CT2"] for r in rows], dtype=np.float64)
    ct1 = np.array([r["CT1"] for r in rows], dtype=np.float64)
    gt = np.array([r["gt"] for r in rows], dtype=np.float64)
    tau_ct2 = wtau(ct2, gt)
    tau_ct1 = wtau(ct1, gt)
    return {
        "dataset": dataset_name,
        "n_cells": len(rows),
        "tau_ct2_scipy": tau_ct2,
        "tau_ct1_scipy": tau_ct1,
        "delta_ct2_minus_ct1_scipy": tau_ct2 - tau_ct1,
    }


def main():
    print("=" * 60)
    print("A2 — DECISIVE: II.3′ 3 datasets under CT1-proxy baseline")
    print("=" * 60)

    # Pancreas
    print("\n--- Pancreas (Vignette 1, Bastidas-Ponce 2019) ---")
    pan_rows, pan_meta = get_pancreas()
    print(f"  Match stats: {pan_meta}")
    pan_result = compute_delta(pan_rows, "Pancreas (Bastidas-Ponce)")
    print(f"  {pan_result}")

    # Cord blood
    print("\n--- Cord blood (Vignette 2, Stoeckius 2017) ---")
    cb_rows, cb_meta = get_cordblood()
    print(f"  Match stats: {cb_meta}")
    cb_result = compute_delta(cb_rows, "Cord blood (Stoeckius)")
    print(f"  {cb_result}")

    # Paul15 note
    paul15_note = {
        "dataset": "Paul15 (Paul et al. 2015)",
        "status": "not_in_kang_s11",
        "reason": (
            "Paul15 mouse hematopoiesis (Drop-seq) not among Kang's 33 ground-truth datasets. "
            "CT1 not available without re-running the CytoTRACE v1 pipeline. Baseline-matched Δ "
            "cannot be computed this session. Reported honestly as a data-availability constraint."
        ),
        "ii3prereg_raw_nnz_delta": 0.17461,
    }
    print(f"\n--- Paul15 ---\n  {paul15_note}")

    # Aggregate: n=2 baseline-matched median (pancreas + cord blood)
    baseline_matched_deltas = []
    if "delta_ct2_minus_ct1_scipy" in pan_result:
        baseline_matched_deltas.append(pan_result["delta_ct2_minus_ct1_scipy"])
    if "delta_ct2_minus_ct1_scipy" in cb_result:
        baseline_matched_deltas.append(cb_result["delta_ct2_minus_ct1_scipy"])
    med_baseline_matched = median(baseline_matched_deltas) if baseline_matched_deltas else None

    # For each of pancreas + cord blood, the II.3′ raw-nnz delta was:
    # Pancreas: +0.716; Cord blood: +0.149; Paul15: +0.175
    # The pair median (Pancreas + Cord blood) under raw nnz:
    ii3_pair_median_raw_nnz = median([0.71607, 0.14899])
    ii3_full_median_raw_nnz = median([0.71607, 0.14899, 0.17461])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline-matched Δ (CT1-proxy baseline), n=2 (Pancreas + Cord blood):")
    print(f"  Pancreas Δ = {pan_result.get('delta_ct2_minus_ct1_scipy', 'n/a')}")
    print(f"  Cord blood Δ = {cb_result.get('delta_ct2_minus_ct1_scipy', 'n/a')}")
    print(f"  Median (n=2) = {med_baseline_matched}")
    print()
    print(f"Same 2 datasets under raw-nnz baseline (II.3′):")
    print(f"  Pancreas Δ = +0.71607")
    print(f"  Cord blood Δ = +0.14899")
    print(f"  Median (n=2) = {ii3_pair_median_raw_nnz}")
    print()
    print(f"II.3′ full n=3 raw-nnz median: {ii3_full_median_raw_nnz}")
    print(f"II.7 n=23 CT1-proxy median: scipy +0.042 / Kang +0.088")

    verdict_line = None
    if med_baseline_matched is not None:
        if med_baseline_matched <= 0.05:
            verdict_line = "PASS (n=2 baseline-matched)"
        elif med_baseline_matched <= 0.10:
            verdict_line = "INCONCLUSIVE (n=2 baseline-matched)"
        else:
            verdict_line = "A-FAIL (n=2 baseline-matched)"

    out = {
        "task": "II.7 close-out A2 — DECISIVE baseline-matched Δ on II.3′ 3 datasets",
        "per_dataset": {
            "pancreas": {"stats": pan_meta, "result": pan_result},
            "cord_blood": {"stats": cb_meta, "result": cb_result},
            "paul15": paul15_note,
        },
        "aggregates": {
            "baseline_matched_median_n2": med_baseline_matched,
            "baseline_matched_verdict_ii3prereg": verdict_line,
            "ii3_pair_raw_nnz_median_n2": ii3_pair_median_raw_nnz,
            "ii3_full_raw_nnz_median_n3": ii3_full_median_raw_nnz,
            "ii7_n23_ct1_proxy_scipy": 0.042,
            "ii7_n23_ct1_proxy_kang": 0.088,
        },
        "diagnostic_interpretation": (
            "Compare baseline-matched median (n=2) to (a) ii3_pair_raw_nnz_median_n2 "
            "(the same 2 datasets under raw nnz) and (b) ii7_n23_ct1_proxy_kang. "
            "If baseline-matched median is close to the II.3′ pair median → baseline "
            "switch does NOT explain the shrinkage (n-effect dominates). "
            "If baseline-matched median is close to the II.7 atlas median → baseline "
            "switch explains most of the shrinkage. "
            "Paul15 excluded because not in Kang S11 (constraint stated honestly)."
        ),
    }

    OUT.write_text(json.dumps(out, indent=2))
    print("\nWritten:", OUT)


if __name__ == "__main__":
    main()
