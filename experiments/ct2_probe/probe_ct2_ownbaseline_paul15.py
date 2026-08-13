"""
CytoTRACE 2 own-baseline probe (Question A, dataset 3 - Paul et al. 2015 mouse
hematopoiesis, Drop-seq).

Purpose
-------
Third dataset in the pre-registered multi-dataset probe. Same protocol as
pancreas + cord blood. Complements the two vignette datasets with a
Drop-seq dataset spanning committed progenitors -> mature hematopoietic
lineages, providing a graded potency ordinal.

Pre-registered decision rule (SINGLE dataset -> data point):
    Delta = tau(CT2) - tau(gc)
    Delta <= 0.05 -> A-PASS-direction
    Delta  > 0.10 -> A-FAIL-direction
    else          -> inconclusive

Data (public):
    - Paul et al. 2015 hematopoiesis, mouse, Drop-seq
      (via scanpy.datasets.paul15() which fetches the pre-processed
      2730-cell x 3451-gene raw-count matrix from the scanpy S3 mirror
      of the original PAGA/Paul2015 anndata: h5 file "paul15.h5"
      https://ndownloader.figshare.com/files/25022578 originally curated by
      Wolf, Angerer & Theis for the Scanpy paper).
    - Reference paper: Paul F. et al. 2015, doi 10.1016/j.cell.2015.11.013
    - Cluster labels: paul15_clusters column of adata.obs (19 clusters in
      the 1-19 numbering scheme of the Paul et al Cell 2015 supplement).

Ground-truth potency ordinal (PRE-REGISTERED before looking at results):
    Cluster names contain a lineage stage prefix per Paul et al. 2015
    Fig 1D annotation. Higher = more potent:
      * 7MEP, 9GMP, 10GMP -> potency 2  (oligopotent progenitors: 3 clusters,
        MEP = Megakaryocyte-Erythroid Progenitor, GMP = Granulocyte-Monocyte
        Progenitor; each capable of >=2 downstream lineages)
      * 1Ery, 11DC        -> potency 1  (lineage-restricted / unipotent
        commitment: earliest erythroid cluster 1Ery has been reported as
        pro-erythroblast-like commitment; 11DC is a small early-DC cluster)
      * 2Ery, 3Ery, 4Ery, 5Ery, 6Ery, 8Mk, 12Baso, 13Baso, 14Mo, 15Mo,
        16Neu, 17Neu, 18Eos, 19Lymph -> potency 0  (differentiated /
        terminally committed erythroblasts, megakaryocytes, basophils,
        monocytes, neutrophils, eosinophils, lymphoid progeny)

Rationale for mapping:
    Paul et al. 2015 explicitly identified clusters 7, 9, 10 as bipotent
    progenitor cell states (see Fig 1D / Table S1 of the original paper).
    Other clusters are lineage-committed with a maturation gradient.
    A conservative 3-level ordinal (2, 1, 0) preserves the biology without
    over-fitting a linear pseudotime. The mapping is fixed here BEFORE
    computing any tau.

Feature-coverage caveat (flagged BEFORE running):
    scanpy's paul15() ships the pre-filtered 3451-variable-gene matrix; the
    CT2 model was trained on ~14271 features. Only ~2953 of the model
    features overlap with the paul15 feature set (~21%). This constrains
    CT2's prediction quality here vs its native performance on
    full-transcriptome inputs. The Delta comparison is still fair
    (gene_counts also uses only the same 3451 features), and CT2 still
    produces a per-cell score - but generalizability to CT2's native
    performance on this dataset should be interpreted with the coverage
    caveat noted.

Reproducibility
---------------
    (venv2, python 3.13)
    pip install cytotrace2-py 'setuptools<81' scanpy openpyxl
    python run_ct2_paul15.py       # produces $OWNBASELINE_SCRATCH/ct2_probe/paul15/ct2_result.csv
    python probe_ct2_ownbaseline_paul15.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import weightedtau

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import scratch_root  # noqa: E402


SEED = 20260713

# PRE-REGISTERED ground-truth potency map (fixed before computing tau)
POTENCY_MAP = {
    # Oligopotent progenitors (multi-lineage capacity)
    "7MEP": 2,
    "9GMP": 2,
    "10GMP": 2,
    # Earliest / partial-committed lineage cells (unipotent commitment)
    "1Ery": 1,
    "11DC": 1,
    # Differentiated / terminally committed lineages
    "2Ery": 0,
    "3Ery": 0,
    "4Ery": 0,
    "5Ery": 0,
    "6Ery": 0,
    "8Mk": 0,
    "12Baso": 0,
    "13Baso": 0,
    "14Mo": 0,
    "15Mo": 0,
    "16Neu": 0,
    "17Neu": 0,
    "18Eos": 0,
    "19Lymph": 0,
}


def sha256_full(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_paul15_inputs(work_dir: Path):
    """(re)generate paul15_counts.txt + annotation from scanpy.
    Returns (counts_txt_path, annotation_csv_path, adata_obs).
    """
    import scanpy as sc
    sc.settings.verbosity = 0
    adata = sc.datasets.paul15()
    X_int = adata.X.astype(np.int32).T  # genes x cells
    df = pd.DataFrame(X_int, index=adata.var_names, columns=adata.obs_names)
    df.index.name = ""
    work_dir.mkdir(parents=True, exist_ok=True)
    counts_path = work_dir / "paul15_counts.txt"
    ann_path = work_dir / "paul15_annotation.csv"
    if not counts_path.exists():
        df.to_csv(counts_path, sep="\t")
    if not ann_path.exists():
        adata.obs.to_csv(ann_path)
    return counts_path, ann_path, adata.obs, df


def run_ct2_if_needed(counts_txt: Path, out_dir: Path) -> Path:
    """Run CytoTRACE 2 if output not present. Returns per-cell result CSV path."""
    result_csv = out_dir.parent / "ct2_result.csv"
    if result_csv.exists():
        return result_csv
    from cytotrace2_py.cytotrace2_py import cytotrace2
    out_dir.mkdir(parents=True, exist_ok=True)
    result = cytotrace2(
        str(counts_txt),
        species="mouse",
        seed=14,
        output_dir=str(out_dir),
        disable_plotting=True,
        disable_verbose=False,
        max_cores=1,
        disable_parallelization=True,
    )
    result.to_csv(result_csv)
    return result_csv


def map_potency(paul_clusters: pd.Series) -> pd.Series:
    labels = paul_clusters.astype(str)
    unmapped = sorted(set(labels) - set(POTENCY_MAP))
    if unmapped:
        raise RuntimeError(f"Unmapped Paul15 cluster labels: {unmapped}")
    return labels.map(POTENCY_MAP).astype(int).rename("ground_truth_potency")


def main(work_dir: Path) -> dict:
    work_dir.mkdir(parents=True, exist_ok=True)

    counts_txt, ann_csv, obs, counts_df = prepare_paul15_inputs(work_dir)
    counts_sha = sha256_full(counts_txt)

    ct2_out_dir = work_dir / "ct2_out"
    ct2_result_csv = run_ct2_if_needed(counts_txt, ct2_out_dir)

    ct2 = pd.read_csv(ct2_result_csv, index_col=0)
    ct2.index = ct2.index.astype(str)

    # Cell IDs must match
    cell_ids = obs.index.astype(str)
    if not (ct2.index == cell_ids).all():
        # Take overlap and record
        common = sorted(set(ct2.index) & set(cell_ids))
    else:
        common = list(cell_ids)

    ct2_score = ct2.loc[common, "CytoTRACE2_Score"].astype(float).to_numpy()

    # gene_counts per cell (H0 primitive): nonzero-gene count from the
    # exact input matrix (genes x cells).
    # counts_df is genes x cells; per-cell nnz = (col > 0).sum()
    counts_vals = counts_df[common].to_numpy()
    gc_arr = (counts_vals > 0).sum(axis=0).astype(float)

    potency = map_potency(obs.loc[common, "paul15_clusters"])
    gt = potency.to_numpy()

    # Sanity strata
    strata = {}
    for k in sorted(np.unique(gt)):
        mask = gt == k
        strata[int(k)] = {
            "n": int(mask.sum()),
            "mean_ct2": float(ct2_score[mask].mean()),
            "mean_gene_counts": float(gc_arr[mask].mean()),
        }

    # Primary: weighted Kendall tau
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
            "point cannot resolve."
        )

    # Also compute standard Kendall + Spearman as sanity (as done for pancreas)
    from scipy.stats import kendalltau, spearmanr
    std_tau_ct2 = kendalltau(ct2_score, gt).statistic
    std_tau_gc = kendalltau(gc_arr, gt).statistic
    spear_ct2 = spearmanr(ct2_score, gt).statistic
    spear_gc = spearmanr(gc_arr, gt).statistic

    result = {
        "seed": SEED,
        "dataset": "paul15_mouse_hematopoiesis_dropseq",
        "data": {
            "source": "scanpy.datasets.paul15()",
            "reference": "Paul F. et al. 2015, doi 10.1016/j.cell.2015.11.013",
            "counts_local_path": str(counts_txt),
            "counts_sha256": counts_sha,
            "ct2_result_path": str(ct2_result_csv),
            "ct2_script": str(Path(__file__).parent / "probe_ct2_ownbaseline_paul15.py"),
        },
        "cell_counts": {
            "n_cells": int(len(common)),
            "n_genes_input": int(counts_df.shape[0]),
            "n_ct2_model_features_present": 2953,  # from CT2 verbose log
        },
        "phenotype_counts": obs.loc[common, "paul15_clusters"].astype(str).value_counts().to_dict(),
        "potency_map": POTENCY_MAP,
        "potency_strata": strata,
        "tau_ct2_vs_ground_truth": float(tau_ct2),
        "tau_gene_counts_vs_ground_truth": float(tau_gc),
        "delta_tau_ct2_minus_gc": delta,
        "tau_ct2_p_hyperbolic": float(p_ct2),
        "tau_gc_p_hyperbolic": float(p_gc),
        "tau_ct2_positive_direction": bool(tau_ct2_sign_ok),
        "tau_gc_positive_direction": bool(tau_gc_sign_ok),
        "sanity_standard_kendall_ct2": float(std_tau_ct2),
        "sanity_standard_kendall_gc": float(std_tau_gc),
        "sanity_spearman_ct2": float(spear_ct2),
        "sanity_spearman_gc": float(spear_gc),
        "prereg_decision_rule": {
            "A-PASS-direction": "delta <= 0.05",
            "inconclusive": "0.05 < delta <= 0.10",
            "A-FAIL-direction": "delta > 0.10",
        },
        "verdict_single_dataset": verdict,
        "rationale": rationale,
        "feature_coverage_caveat": (
            "Paul15 ships the pre-filtered 3451-variable-gene matrix; CT2 was "
            "trained on ~14271 features. CT2 log reports 2953 model features "
            "present (~21% coverage). Delta comparison is fair (gene_counts "
            "computed on same input), but CT2's native performance would be "
            "higher on full-transcriptome data. Flagged as pre-registered."
        ),
    }

    out_file = work_dir / "result_paul15.json"
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print("=" * 70)
    print("CT2 own-baseline probe -- Paul15 mouse hematopoiesis (Drop-seq)")
    print("=" * 70)
    print(f"cells used:               {len(common)}")
    print("cluster distribution (top 10):")
    for lbl, c in obs.loc[common, "paul15_clusters"].astype(str).value_counts().head(19).items():
        print(f"    {lbl:<12s} n={c:>4d}  potency={POTENCY_MAP[lbl]}")
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
    print(f"standard Kendall tau (sanity): CT2={std_tau_ct2:+.4f} gc={std_tau_gc:+.4f}")
    print(f"Spearman rho          (sanity): CT2={spear_ct2:+.4f} gc={spear_gc:+.4f}")
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
    ap.add_argument("--work", default=str(scratch_root() / "ct2_probe" / "paul15"))
    args = ap.parse_args()
    main(Path(args.work))
