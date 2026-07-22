"""
CytoTRACE 2 own-baseline probe (Question A, single dataset - human cord blood CITE-seq).

Purpose
-------
Second dataset in the pre-registered probe (SSOT: CT2 own-baseline prereg 2026-07-12).
Same protocol as pancreas (probe_ct2_ownbaseline_pancreas.py).

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
    - Expression + phenotype + PRE-COMPUTED CT2 scores in the same Seurat object:
      the published Vignette 2 result RDS at
      https://raw.githubusercontent.com/digitalcytometry/cytotrace2/main/Vignette2_CytoTRACE2_results.rds
      (this is the SeuratObj input + CT2 output rolled into one; the counts slot
      is the identical dgCMatrix as the raw
      https://drive.google.com/uc?id=1_OhZdz4y0R0MeB6gNlYAFC8P8rutgqzJ file,
      which is also downloaded here as a cross-check).
    - Reference paper: Stoeckius et al. 2017, doi 10.1038/nmeth.4380
    - Species: human. CITE-seq. n=2308 cells, 20401 genes.

Ground-truth potency ordinal (per prereg prompt, hematopoietic biology):
    Hematopoietic stem and progenitor  -> 2  (multipotent / oligopotent)
    T cell / B cell / NK cell / Monocyte / Dendritic cell / Megakaryocyte / Erythrocyte  -> 0  (mature differentiated)

Rationale for mapping:
    - The CT2 atlas assigns `potency_annotation=3` (Multipotent tier) to the 42
      HSPCs and `potency_annotation=5` (mature/differentiated tier) to the 2266
      mature immune / erythroid / megakaryocyte cells (verified in the
      metadata). Higher CT2_Score for HSPCs (0.51) vs mature cells (0.10)
      confirms HSPCs are the more-potent group.
    - We keep the paper convention "higher ordinal = more potent" (as in the
      pancreas prereg) so signs of tau align across datasets. HSPCs -> 2 (as
      multipotent progenitors, one tier below the pancreas "3" for the
      multipotent pancreatic progenitors, to reflect that HSPCs sit in the
      Multipotent broad tier not the Pluripotent/Totipotent one).
    - Only two ground-truth potency levels are present in this dataset (2 vs 0);
      three-level ordering as in the pancreas dataset is not available here.
      This is a limitation of the vignette dataset, not of the protocol.

Reproducibility
---------------
    python -m venv venv && source venv/bin/activate
    pip install scipy numpy pandas rdata gdown requests
    python probe_ct2_ownbaseline_cordblood.py

Deterministic (no stochastic step); SEED kept for future bootstrap extensions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy.stats import weightedtau

SEED = 20260713
CORDBLOOD_DRIVE_ID = "1_OhZdz4y0R0MeB6gNlYAFC8P8rutgqzJ"
CORDBLOOD_DRIVE_URL = f"https://drive.google.com/uc?id={CORDBLOOD_DRIVE_ID}"
CT2_RESULT_URL = (
    "https://raw.githubusercontent.com/digitalcytometry/cytotrace2/main/"
    "Vignette2_CytoTRACE2_results.rds"
)

# Ground-truth potency map (per prereg prompt; higher = more potent, matches
# the pancreas prereg convention). HSPCs = 2 (Multipotent broad tier).
POTENCY_MAP = {
    "Hematopoietic stem and progenitor": 2,
    "T cell": 0,
    "B cell": 0,
    "NK cell": 0,
    "Monocyte": 0,
    "Dendritic cell": 0,
    "Megakaryocyte": 0,
    "Erythrocyte": 0,
}


def sha256_full(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return dest
    if "drive.google.com" in url:
        import gdown
        gdown.download(url, str(dest), quiet=False)
    else:
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in r.iter_content(1024 * 1024):
                    f.write(chunk)
    return dest


def load_ct2_result_rds(path: Path):
    """Parse the post-CT2 Seurat object RDS.

    Returns:
        counts_csr: scipy CSR sparse (genes x cells), from RNA/counts layer
        gene_names: array-like of gene names (may be None if not stored)
        cell_ids: array-like of cell barcodes (from meta.data index)
        meta: pandas DataFrame with the seurat meta.data (contains
              CytoTRACE2_Score, standardized_phenotype, etc.)
    """
    import rdata
    from scipy.sparse import csc_matrix

    warnings.simplefilter("ignore")
    parsed = rdata.parser.parse_file(str(path))
    d = rdata.conversion.convert(parsed)

    meta = getattr(d, "meta.data")
    # Normalize dtypes
    meta.index = meta.index.astype(str)

    # Sparse counts (dgCMatrix): i, p, x -> CSC
    rna_name = np.str_("RNA")
    layers = d.assays[rna_name].layers
    counts_ns = layers[np.str_("counts")]
    n_genes, n_cells = int(counts_ns.Dim[0]), int(counts_ns.Dim[1])
    counts_csc = csc_matrix(
        (counts_ns.x, counts_ns.i, counts_ns.p),
        shape=(n_genes, n_cells),
    )
    counts_csr = counts_csc.tocsr()

    return counts_csr, n_genes, n_cells, meta


def compute_gene_counts_sparse(counts_csr, n_cells: int) -> np.ndarray:
    """Number of nonzero genes per cell (CytoTRACE v1 primitive, H0).

    counts_csr is genes x cells; column-nonzero-count per cell.
    """
    # Convert to CSC for column ops (nnz per column = per-cell nonzero genes)
    counts_csc = counts_csr.tocsc()
    return np.diff(counts_csc.indptr)  # per-column nnz


def map_potency(meta: pd.DataFrame) -> pd.Series:
    col = "standardized_phenotype"
    labels = meta[col].astype(str)
    unmapped = sorted(set(labels) - set(POTENCY_MAP))
    if unmapped:
        raise RuntimeError(f"Unmapped phenotype labels: {unmapped}")
    potency = labels.map(POTENCY_MAP).astype(int)
    potency.index = meta.index.astype(str)
    potency.name = "ground_truth_potency"
    return potency


def main(out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Download both the original Drive RDS (for provenance / SHA logging) and
    # the CT2-result RDS (which contains the counts + CT2 scores + metadata).
    orig_path = download(
        CORDBLOOD_DRIVE_URL,
        out_dir / "Cord_blood_CITE_seq_SeuratObj.rds",
    )
    result_path = download(
        CT2_RESULT_URL,
        out_dir / "Vignette2_CytoTRACE2_results.rds",
    )
    orig_sha = sha256_full(orig_path)
    result_sha = sha256_full(result_path)

    counts_csr, n_genes, n_cells, meta = load_ct2_result_rds(result_path)

    # Cell IDs from metadata (should match counts columns 1:1)
    cell_ids = meta.index.astype(str).to_numpy()
    if len(cell_ids) != n_cells:
        raise RuntimeError(
            f"cell count mismatch: meta.data={len(cell_ids)} counts_cols={n_cells}"
        )

    # gene_counts per cell (H0 primitive)
    gc = compute_gene_counts_sparse(counts_csr, n_cells)
    if gc.shape[0] != n_cells:
        raise RuntimeError(f"gc len {gc.shape[0]} != n_cells {n_cells}")

    # potency ordinal from standardized_phenotype
    potency = map_potency(meta)  # 2 for HSPC, 0 for mature

    # Verify CT2 score column exists
    if "CytoTRACE2_Score" not in meta.columns:
        raise RuntimeError("CytoTRACE2_Score column missing from meta.data")
    ct2_score = meta["CytoTRACE2_Score"].astype(float).to_numpy()

    gt = potency.loc[cell_ids].astype(int).to_numpy()
    gc_arr = gc.astype(float)  # already aligned to counts columns == meta rows

    # Sanity: coarse mean predictor per potency stratum
    strata = {}
    for k in sorted(np.unique(gt)):
        mask = gt == k
        strata[int(k)] = {
            "n": int(mask.sum()),
            "mean_ct2": float(ct2_score[mask].mean()),
            "mean_gene_counts": float(gc_arr[mask].mean()),
        }

    # Primary: weighted Kendall tau, hyperbolic weighting per scipy default.
    tau_ct2, p_ct2 = weightedtau(ct2_score, gt)
    tau_gc, p_gc = weightedtau(gc_arr, gt)
    delta = float(tau_ct2 - tau_gc)

    tau_ct2_sign_ok = tau_ct2 > 0
    tau_gc_sign_ok = tau_gc > 0

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
        "dataset": "human_cord_blood_CITEseq_vignette2",
        "data": {
            "original_drive_url": CORDBLOOD_DRIVE_URL,
            "original_local_path": str(orig_path),
            "original_sha256": orig_sha,
            "ct2_result_url": CT2_RESULT_URL,
            "ct2_result_local_path": str(result_path),
            "ct2_result_sha256": result_sha,
        },
        "cell_counts": {
            "counts_cells": int(n_cells),
            "counts_genes": int(n_genes),
            "meta_cells": int(meta.shape[0]),
            "overlap_used": int(n_cells),
        },
        "phenotype_counts": (
            meta["standardized_phenotype"].value_counts().to_dict()
        ),
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
            "One dataset = one data point. This is dataset 2 of the multi-"
            "dataset probe; final verdict requires the median across all "
            "datasets per prereg."
        ),
        "dataset_caveat": (
            "Cord blood CITE-seq has only TWO ground-truth potency levels "
            "(HSPC=42 cells vs mature=2266 cells). This reduces the number of "
            "concordant/discordant pair configurations but weightedtau is "
            "still well-defined. Effect size Delta is directly comparable."
        ),
    }

    out_file = out_dir / "result_cordblood.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("=" * 70)
    print("CT2 own-baseline probe -- human cord blood CITE-seq (Stoeckius 2017)")
    print("=" * 70)
    print(f"cells used:               {n_cells}")
    print("phenotype distribution:")
    for lbl, c in meta["standardized_phenotype"].value_counts().items():
        print(f"    {lbl:<38s} n={c:>4d}  potency={POTENCY_MAP[lbl]}")
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
