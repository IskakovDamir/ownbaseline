"""
CytoTRACE 2 own-baseline probe (Question A, single dataset — mouse pancreas).

Purpose
-------
Cheap feasibility check for the pre-registered probe in
04-experiments/2026-07-12-hod3-cytotrace2-ownbaseline-probe-prereg.md.

Compares within-dataset weighted Kendall tau of two potency predictors
against ground-truth phenotype-derived potency ordinal:

    tau_gc  = weightedtau(gene_counts, ground_truth)
    tau_ct2 = weightedtau(CytoTRACE2_Score, ground_truth)
    Delta   = tau_ct2 - tau_gc

Pre-registered decision rule (SINGLE dataset -> data point, not verdict):
    Delta <= 0.05  -> A-PASS-direction (within-dataset ordering is low-order)
    Delta  > 0.10  -> A-FAIL-direction (CT2 catches beyond-breadth structure)
    0.05 < Delta <= 0.10 -> inconclusive

Data (public):
    - Expression + phenotype: Google Drive id 1TYdQsMoDIJjoeuiTD5EO_kZgNJUyfRY2
      (Pancreas_10x_downsampled.rds, from CytoTRACE 2 Vignette 1; originally
      Bastidas-Ponce et al. 2019, mouse pancreas 10x).
    - CT2 scores: https://raw.githubusercontent.com/digitalcytometry/cytotrace2/main/Vignette1_CytoTRACE2_results.csv

Ground-truth potency ordinal (per prereg prompt, pancreas biology):
    Multipotent pancreatic progenitor            -> 3
    Endocrine progenitor / committed precursor
    cell / Immature endocrine cell               -> 1
    Alpha / Beta / Delta / Epsilon cell          -> 0

Reproducibility
---------------
    python -m venv venv && source venv/bin/activate
    pip install scipy numpy pandas pyreadr rdata gdown requests
    python probe_ct2_ownbaseline_pancreas.py

Deterministic (no stochastic step); SEED kept for future bootstrap extensions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import weightedtau

SEED = 20260713
DRIVE_ID = "1TYdQsMoDIJjoeuiTD5EO_kZgNJUyfRY2"
DRIVE_URL = f"https://drive.google.com/uc?id={DRIVE_ID}"
CT2_CSV_URL = (
    "https://raw.githubusercontent.com/digitalcytometry/cytotrace2/main/"
    "Vignette1_CytoTRACE2_results.csv"
)

# Ground-truth potency map (per prereg prompt; higher = more potent)
POTENCY_MAP = {
    "Multipotent pancreatic progenitor": 3,
    "Endocrine progenitor": 1,
    "Endocrine committed precursor cell": 1,
    "Immature endocrine cell": 1,
    "Alpha cell": 0,
    "Beta cell": 0,
    "Delta cell": 0,
    "Epsilon cell": 0,
}


def sha256_head(path: Path, nbytes: int = 4 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(nbytes))
    return h.hexdigest()


def download_rds(dest: Path) -> Path:
    if dest.exists():
        return dest
    import gdown  # local import so pure-analysis reruns don't require it

    dest.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(DRIVE_URL, str(dest), quiet=False)
    return dest


def load_rds(path: Path):
    """Read the RDS list -> (expression DataFrame, annotation DataFrame).

    The .rds is a named R list with two elements (expression_data, annotation).
    pyreadr's list handling was silent-empty on this file; `rdata` (pure Py)
    parses it correctly.
    """
    import rdata

    parsed = rdata.parser.parse_file(str(path))
    d = rdata.conversion.convert(parsed)
    expr = d["expression_data"]  # 27998 genes x 2850 cells
    ann = d["annotation"]  # 2850 x 1 (column 'phenotype')
    return expr, ann


def compute_gene_counts(expr: pd.DataFrame) -> pd.Series:
    """Number of nonzero genes per cell (CytoTRACE v1 primitive, H0)."""
    vals = expr.to_numpy()
    counts = (vals > 0).sum(axis=0)  # sum over genes -> per cell
    return pd.Series(counts, index=expr.columns.astype(str), name="gene_counts")


def load_ct2_scores() -> pd.DataFrame:
    df = pd.read_csv(CT2_CSV_URL, index_col=0)
    df.index = df.index.astype(str)
    return df


def map_potency(ann: pd.DataFrame) -> pd.Series:
    col = ann.columns[0]  # 'phenotype'
    labels = ann[col].astype(str)
    unmapped = sorted(set(labels) - set(POTENCY_MAP))
    if unmapped:
        raise RuntimeError(f"Unmapped phenotype labels: {unmapped}")
    potency = labels.map(POTENCY_MAP).astype(int)
    potency.index = ann.index.astype(str)
    potency.name = "ground_truth_potency"
    return potency


def main(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rds_path = download_rds(out_dir / "Pancreas_10x_downsampled.rds")
    file_sha = sha256_head(rds_path)

    expr, ann = load_rds(rds_path)
    ct2 = load_ct2_scores()
    potency = map_potency(ann)
    gene_counts = compute_gene_counts(expr)

    # Align on cell IDs across all three sources
    cells = (
        set(expr.columns.astype(str))
        & set(ct2.index)
        & set(potency.index)
    )
    cells = sorted(cells)
    n_cells = len(cells)
    if n_cells == 0:
        raise RuntimeError("No cells overlap between expr/CT2/annotation")

    ct2_score = ct2.loc[cells, "CytoTRACE2_Score"].astype(float).to_numpy()
    gc = gene_counts.loc[cells].astype(float).to_numpy()
    gt = potency.loc[cells].astype(int).to_numpy()

    # Sanity: coarse sign checks (mean predictor per ground-truth stratum,
    # ascending; expect roughly monotone with potency for both).
    strata = {}
    for k in sorted(np.unique(gt)):
        mask = gt == k
        strata[int(k)] = {
            "n": int(mask.sum()),
            "mean_ct2": float(ct2_score[mask].mean()),
            "mean_gene_counts": float(gc[mask].mean()),
        }

    # Primary: weighted Kendall tau, hyperbolic weighting per scipy default.
    tau_ct2, p_ct2 = weightedtau(ct2_score, gt)
    tau_gc, p_gc = weightedtau(gc, gt)
    delta = float(tau_ct2 - tau_gc)

    # Sign-direction sanity: both should be > 0 (higher predictor -> higher
    # potency). weightedtau reports positive-correlation direction.
    tau_ct2_sign_ok = tau_ct2 > 0
    tau_gc_sign_ok = tau_gc > 0

    # Decision rule (single dataset, single data point).
    if delta <= 0.05:
        verdict = "A-PASS-direction"
        rationale = (
            "Delta <= 0.05: within-dataset CT2 ordering does not appreciably "
            "exceed the gene_counts baseline; consistent with the hypothesis "
            "that within-dataset potency ordering is low-order (H0-like)."
        )
    elif delta > 0.10:
        verdict = "A-FAIL-direction"
        rationale = (
            "Delta > 0.10: within-dataset CT2 ordering exceeds gene_counts "
            "baseline by a pre-specified margin; CT2 catches structure beyond "
            "gene-count breadth. Pre-registered failed prediction; recorded."
        )
    else:
        verdict = "inconclusive"
        rationale = (
            "0.05 < Delta <= 0.10: pre-registered gray zone; single-dataset "
            "point cannot resolve. Requires the full multi-dataset probe."
        )

    result = {
        "seed": SEED,
        "data": {
            "rds_url": DRIVE_URL,
            "rds_local_path": str(rds_path),
            "rds_sha256_first_4MB": file_sha,
            "ct2_csv_url": CT2_CSV_URL,
        },
        "cell_counts": {
            "expr_cells": int(expr.shape[1]),
            "ct2_cells": int(ct2.shape[0]),
            "ann_cells": int(ann.shape[0]),
            "overlap_used": n_cells,
        },
        "phenotype_counts": ann.iloc[:, 0].value_counts().to_dict(),
        "potency_map": POTENCY_MAP,
        "potency_strata": strata,
        "tau_ct2_vs_ground_truth": float(tau_ct2),
        "tau_gene_counts_vs_ground_truth": float(tau_gc),
        "delta_tau_ct2_minus_gc": delta,
        "tau_ct2_p_hyperbolic": float(p_ct2),
        "tau_gc_p_hyperbolic": float(p_gc),
        "tau_ct2_positive_direction": bool(tau_ct2_sign_ok),
        "tau_gc_positive_direction": bool(tau_gc_sign_ok),
        "prereg_decision_rule": {
            "A-PASS-direction": "delta <= 0.05",
            "inconclusive": "0.05 < delta <= 0.10",
            "A-FAIL-direction": "delta > 0.10",
        },
        "verdict_single_dataset": verdict,
        "rationale": rationale,
        "single_dataset_caveat": (
            "One dataset = one data point. This is a feasibility+harness "
            "check, NOT the full Question A verdict, which requires the "
            "multi-dataset median as pre-registered."
        ),
    }

    out_file = out_dir / "result.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    # Human-readable summary
    print("=" * 70)
    print("CT2 own-baseline probe -- mouse pancreas (Bastidas-Ponce et al. 2019)")
    print("=" * 70)
    print(f"cells used (overlap):       {n_cells}")
    print("phenotype distribution:")
    for lbl, c in ann.iloc[:, 0].value_counts().items():
        print(f"    {lbl:<38s} n={c}  potency={POTENCY_MAP[lbl]}")
    print()
    print("mean predictor per potency stratum (ascending in potency):")
    for k in sorted(strata):
        s = strata[k]
        print(
            f"    potency={k} (n={s['n']:>4d}):  "
            f"mean CT2={s['mean_ct2']:.4f}   "
            f"mean gene_counts={s['mean_gene_counts']:.1f}"
        )
    print()
    print(f"weighted Kendall tau (CT2_Score vs ground truth) : {tau_ct2:+.5f}")
    print(f"weighted Kendall tau (gene_counts vs ground truth): {tau_gc:+.5f}")
    print(f"Delta = tau(CT2) - tau(gc)                        : {delta:+.5f}")
    print()
    print(f"sign direction CT2 positive? {tau_ct2_sign_ok}")
    print(f"sign direction gene_counts positive? {tau_gc_sign_ok}")
    print()
    print(f"pre-registered decision (SINGLE dataset): {verdict}")
    print(f"    {rationale}")
    print()
    print(f"artifact -> {out_file}")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default="/tmp/ct2_probe/data",
        help="Working directory for data + result.json",
    )
    args = ap.parse_args()
    main(Path(args.out))
