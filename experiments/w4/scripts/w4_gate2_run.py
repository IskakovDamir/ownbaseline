"""
w4_gate2_run.py — W4 GATE 2: contamination-clean own-baseline runs on the two
locked independent atlases (C1 GSE117498 human hematopoietic progenitors; C2
GSE125970 human intestinal epithelium).

Follows the locked prereg 2026-07-18-w4-independent-atlas-prereg.md and the
locked atlas ordinals from 2026-07-18-w4-gate1-atlas-lock.md — BYTE-IDENTICAL,
no modification allowed at GATE 2.

Design:
  - Preprocessing per prereg §4:
      library-size lognorm (log1p(CPM/median)), gene filter >= N cells
      (N locked per atlas below).
  - Scaffold: STRING v12.0 human, threshold 700, LCC =
      $OWNBASELINE_DATA_ROOT/w4/scaffolds/human_string_v12_thr700_lcc.npz
    intersected per-atlas with expressed genes at analysis time
    (mirrors the mouse pipeline convention in scaffolds/).
  - Scores: CytoTRACE v1 (via potency_metrics.cytotrace_proxy → gene_counts;
    CT v1 primitive = |supp(x)| = raw gene count).
             SR (SCENT) via potency_metrics.scent_sr → primitive = <x, degree>
             CCAT via potency_metrics.ccat → primitive = Pearson(x, degree)
  - Skill: weighted Kendall τ_wt via scipy.stats.weightedtau
             (tie-handling: scipy averages over both lexicographic orderings;
              same kernel applied to score AND primitive on same cells).
             AUROC for top-vs-bottom binary split.
  - Own-baseline:
      marginal Δ = τ_wt(score, ord) - τ_wt(own primitive, ord)
      conditional skill = τ_wt(residual(score | primitive), ord)
                          where residual is the score minus its rank-based
                          fit on the primitive (Spearman-residual; standard
                          partial rank association).
  - Bootstrap CIs (per-cell resample, n=1000, seed=42) for skill values and Δ.

Outputs:
  results/c1_gse117498_results.json
  results/c2_gse125970_results.json
  results/summary.json

Contamination discipline:
  Ordinal loaded here BYTE-IDENTICAL from the GATE 1 lock; unmapped labels are
  marked UNRANKED and excluded from skill computation.

Reproducibility:
  seed = 42 fixed. Numbers only from real runs.
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import weightedtau, spearmanr, rankdata
from sklearn.metrics import roc_auc_score

# Import score implementations from existing tautology-diagnostic package
HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent.parent  # tautology-diagnostic/
sys.path.insert(0, str(PKG_ROOT))
import potency_metrics as pm  # noqa: E402


# ----------------------------------------------------------------------------- #
#  Constants (locked)
# ----------------------------------------------------------------------------- #
SEED = 42
N_BOOT = 1000

# Gene filter: gene expressed in >= N cells. Locked per atlas below.
C1_GENE_FILTER_MIN_CELLS = 10   # ~0.05% of C1 sorted set (small-population aware)
C2_GENE_FILTER_MIN_CELLS = 10   # matches C1 for cross-atlas consistency

# Locked threshold band (prereg §2)
DELTA_TAUTOLOG = 0.05
DELTA_SCORE_BEATS = 0.10
DELTA_SIGN_FLIP = -0.10

# STRING scaffold
STRING_NPZ = HERE.parent / "scaffolds" / "human_string_v12_thr700_lcc.npz"


# =============================================================================
#  C1 — GSE117498 (Pellin 2019) loader + ordinal
# =============================================================================
# Ordinal BYTE-IDENTICAL from GATE 1 lock §4:
#   HSC=5, MPP=4, CD34+/CD164+ broad gate=4, MLP=3, CMP=3, GMP=2, MEP=2, PreB/NK=2
# Sorted-population labels in Pellin's GSMs mapped to the ordinal:
#   GSM3305359 HSC             -> HSC                       -> 5
#   GSM3305360 MPP             -> MPP                       -> 4
#   GSM3305361 MLP             -> MLP                       -> 3
#   GSM3305362 PreBNK          -> PreB/NK                   -> 2
#   GSM3305363 MEP             -> MEP                       -> 2
#   GSM3305364 CMP             -> CMP                       -> 3
#   GSM3305365 GMP             -> GMP                       -> 2
#   GSM3305366 LinNegCD34PosCD164Pos -> CD34+/CD164+ broad gate -> 4
#   GSM3305367 LinNegCD34NegCD164high -> UNRANKED (CD34-neg, not in locked ordinal)
#   GSM3305368 LinNegCD34lowCD164high -> UNRANKED (CD34-low, not in locked ordinal)
#   GSM3305369 LinNegCD34NegCD164low  -> UNRANKED (CD34-neg, not in locked ordinal)
# The three UNRANKED files are downstream mature/rare subsets outside the
# broad-CD34+/CD164+ gate the ordinal labels; per contamination discipline,
# they are LOADED for depth reporting but EXCLUDED from skill computation.
C1_POP_TO_RANK = {
    "HSC": 5,
    "MPP": 4,
    "MLP": 3,
    "PreBNK": 2,
    "MEP": 2,
    "CMP": 3,
    "GMP": 2,
    "LinNegCD34PosCD164Pos": 4,
    "LinNegCD34NegCD164high": None,
    "LinNegCD34lowCD164high": None,
    "LinNegCD34NegCD164low": None,
}

C1_POP_LABEL = {
    "HSC": "HSC",
    "MPP": "MPP",
    "MLP": "MLP",
    "PreBNK": "PreB/NK",
    "MEP": "MEP",
    "CMP": "CMP",
    "GMP": "GMP",
    "LinNegCD34PosCD164Pos": "CD34+/CD164+ broad gate",
    "LinNegCD34NegCD164high": "CD34-/CD164-high (unranked)",
    "LinNegCD34lowCD164high": "CD34-low/CD164-high (unranked)",
    "LinNegCD34NegCD164low": "CD34-/CD164-low (unranked)",
}

C1_GSM_FILES = {
    "GSM3305359_HSC.raw_counts.tsv.gz": "HSC",
    "GSM3305360_MPP.raw_counts.tsv.gz": "MPP",
    "GSM3305361_MLP.raw_counts.tsv.gz": "MLP",
    "GSM3305362_PreBNK.raw_counts.tsv.gz": "PreBNK",
    "GSM3305363_MEP.raw_counts.tsv.gz": "MEP",
    "GSM3305364_CMP.raw_counts.tsv.gz": "CMP",
    "GSM3305365_GMP.raw_counts.tsv.gz": "GMP",
    "GSM3305366_LinNegCD34PosCD164Pos.raw_counts.tsv.gz": "LinNegCD34PosCD164Pos",
    "GSM3305367_LinNegCD34NegCD164high.raw_counts.tsv.gz": "LinNegCD34NegCD164high",
    "GSM3305368_LinNegCD34lowCD164high.raw_counts.tsv.gz": "LinNegCD34lowCD164high",
    "GSM3305369_LinNegCD34NegCD164low.raw_counts.tsv.gz": "LinNegCD34NegCD164low",
}


def load_c1(data_dir):
    """
    Load Pellin GSE117498 as a per-cell counts matrix (cells x genes) with
    associated population labels and ranks.

    Two file groups have different gene indices — union them across files and
    fill missing genes with 0 (a raw-count TSV omitting a gene is equivalent
    to zero expression per Pellin's release format).
    """
    print("[C1] loading population TSVs...")
    per_pop_frames = {}
    all_genes = set()
    for fname, pop in C1_GSM_FILES.items():
        p = Path(data_dir) / fname
        # first row is "Barcode\t<cellid>..."; second row is "Library\t1\t1..." — skip
        df = pd.read_csv(p, sep="\t", compression="gzip", index_col=0)
        # drop the "Library" row that ships as the first data row
        if "Library" in df.index:
            df = df.drop(index="Library")
        df.index.name = "gene"
        per_pop_frames[pop] = df
        all_genes.update(df.index.tolist())
        print(f"  {pop:26s} : {df.shape[1]:>6} cells x {df.shape[0]:>6} genes")

    gene_list = sorted(all_genes)
    gene_idx = {g: i for i, g in enumerate(gene_list)}
    n_genes = len(gene_list)
    print(f"[C1] union genes across all populations: {n_genes:,}")

    # Assemble one big cells x genes float32 matrix, cell metadata
    total_cells = sum(f.shape[1] for f in per_pop_frames.values())
    print(f"[C1] total cells: {total_cells:,}")
    X = np.zeros((total_cells, n_genes), dtype=np.float32)
    cell_ids = []
    pop_labels = []
    ranks = []
    row = 0
    for pop, df in per_pop_frames.items():
        # gene alignment: use full gene index
        pos = np.array([gene_idx[g] for g in df.index], dtype=np.int64)
        sub = df.to_numpy().astype(np.float32).T  # cells x local_genes
        X[row:row + sub.shape[0], pos] = sub
        for cid in df.columns:
            cell_ids.append(f"{pop}::{cid}")
            pop_labels.append(pop)
            ranks.append(C1_POP_TO_RANK[pop])
        row += sub.shape[0]
    ranks_arr = np.array([r if r is not None else np.nan for r in ranks], dtype=np.float64)
    return X, gene_list, np.array(cell_ids), np.array(pop_labels), ranks_arr


# =============================================================================
#  C2 — GSE125970 (Wang 2020) loader + ordinal
# =============================================================================
# Ordinal BYTE-IDENTICAL from GATE 1 lock §4:
#   Stem cells=4, TA=3, Progenitor (unresolved)=3,
#   Enterocyte=1, Goblet=1, Enteroendocrine=1, Paneth-like=1, Tuft=1
# (Wang metadata labels observed: Stem Cell, TA, Progenitor, Enterocyte,
#  Goblet, Enteriendocrine [sic — Wang typo], Paneth-like. No Tuft label
#  in Wang's metadata; face-value — Tuft in ordinal but 0 cells, no bias.)
C2_LABEL_TO_RANK = {
    "Stem Cell": 4,
    "TA": 3,
    "Progenitor": 3,
    "Enterocyte": 1,
    "Goblet": 1,
    "Enteriendocrine": 1,   # Wang metadata spelling — enteroendocrine
    "Paneth-like": 1,
    "Tuft": 1,
}


def load_c2(data_dir):
    """
    Load Wang GSE125970 raw UMI matrix (genes x cells → cells x genes) and
    cell metadata (Sample_ID, CellType).
    """
    print("[C2] loading raw UMI matrix (this reads ~19k genes x 14.5k cells)...")
    matrix_path = Path(data_dir) / "GSE125970_raw_UMIcounts.txt.gz"
    df = pd.read_csv(matrix_path, sep="\t", compression="gzip", index_col=0)
    print(f"[C2] matrix shape: {df.shape[0]:,} genes x {df.shape[1]:,} cells")
    # cells x genes
    X = df.to_numpy().astype(np.float32).T
    gene_list = df.index.tolist()
    cell_ids = np.array(df.columns.tolist())

    # metadata
    meta_path = Path(data_dir) / "GSE125970_cell_info.txt.gz"
    meta = pd.read_csv(meta_path, sep="\t", compression="gzip")
    # UniqueCell_ID matches df.columns
    meta = meta.set_index("UniqueCell_ID")
    # align to X
    meta_aligned = meta.reindex(cell_ids)
    labels = meta_aligned["CellType"].to_numpy()
    n_missing = pd.isna(labels).sum()
    if n_missing:
        print(f"[C2] WARNING: {n_missing} cells without CellType metadata")
    ranks = np.array([C2_LABEL_TO_RANK.get(l, np.nan) for l in labels], dtype=np.float64)
    return X, gene_list, cell_ids, labels, ranks


# =============================================================================
#  Preprocessing + score pipeline (both atlases)
# =============================================================================
def preprocess_and_filter(X, gene_list, min_cells):
    """
    Per prereg §4:
      - gene filter: expressed in >= min_cells cells
      - library-size lognorm: log1p(CPM/median). Actually implemented in
        potency_metrics.library_normalize which uses target_sum = median of
        raw totals; then we log1p externally. But the score implementations
        already handle normalization internally; we pass RAW counts to them.

    For the primitives:
      - gene_count = raw nnz per cell (matches CT v1 primitive |supp(x)|)
    """
    # gene filter on RAW counts
    gene_ncells = (X > 0).sum(axis=0)
    keep_gene = gene_ncells >= min_cells
    print(f"    gene filter: kept {int(keep_gene.sum()):,} / {len(gene_list):,} genes (>= {min_cells} cells)")
    Xf = X[:, keep_gene]
    gene_list_f = [g for g, k in zip(gene_list, keep_gene) if k]
    return Xf, gene_list_f


def load_string_scaffold(gene_list):
    """
    Load pre-built human STRING v12 LCC and intersect with atlas gene list.
    Returns (A_intersect, degree, col_idx) where col_idx are positions in
    gene_list of scaffold genes present in the atlas.
    """
    z = np.load(STRING_NPZ, allow_pickle=True)
    A = sp.csr_matrix(
        (z["data"].astype(np.float64), z["indices"], z["indptr"]),
        shape=tuple(z["shape"]),
    )
    scaffold_genes = list(z["genes"])
    print(f"  STRING LCC (free): {A.shape[0]:,} nodes")

    # intersect with atlas
    gene_pos = {g: i for i, g in enumerate(gene_list)}
    scaffold_pos = {g: i for i, g in enumerate(scaffold_genes)}
    common = [g for g in scaffold_genes if g in gene_pos]
    print(f"  STRING ∩ atlas: {len(common):,} genes present in both")

    # Rows/cols in scaffold matrix for common
    sub_idx = np.array([scaffold_pos[g] for g in common], dtype=np.int64)
    A_sub = A[sub_idx][:, sub_idx]

    # Take LCC of intersected subgraph (some genes will drop out)
    ncomp, labels = connected_components(A_sub, directed=False)
    if ncomp > 1:
        sizes = np.bincount(labels)
        big = int(np.argmax(sizes))
        sel = np.where(labels == big)[0]
        A_sub = A_sub[sel][:, sel]
        common = [common[i] for i in sel]
        print(f"  STRING ∩ atlas LCC: {A_sub.shape[0]:,} nodes")

    # col_idx = positions in atlas gene_list of the intersected scaffold genes
    col_idx = np.array([gene_pos[g] for g in common], dtype=np.int64)
    A_sub = A_sub.tocsr()
    A_sub.setdiag(0)
    A_sub.eliminate_zeros()
    A_sub.data = A_sub.data.astype(np.float64)
    A_sub = (A_sub > 0).astype(np.float64).tocsr()
    degree = np.asarray(A_sub.sum(axis=1), dtype=np.float64).ravel()
    return A_sub, degree, col_idx


# =============================================================================
#  Score computation
# =============================================================================
def compute_scores_and_primitives(X, gene_list, A_scaffold, degree, col_idx):
    """
    Compute for each cell:
      CT_score      : proxy CytoTRACE GCS (geometric mean of top-200 GC-correlated genes)
      CT_primitive  : |supp(x)| = raw gene count (nnz)
      SR_score      : SCENT SR on scaffold A
      SR_primitive  : <x, degree> — mean-field of SR, unnormalized inner product
      CCAT_score    : CCAT — Pearson(log1p(rel(x)), degree)
      CCAT_primitive: identical form as score (CCAT itself = its primitive by definition
                      in prereg §1 / theory Prop 1 — the "primitive of CCAT" is
                      T = PCC(x, degree). CCAT IS the primitive. So Δ_CCAT = 0
                      by construction unless we distinguish log vs linear.
                      Per prereg: primitive = "PCC(x, degree)" — we treat it as
                      Pearson(rel(x), degree) WITHOUT log — the raw, non-log
                      form — while the CCAT score is with log1p.)

    All primitives are computed on the same RAW-counts matrix restricted (for
    graph-based ones) to the same set of scaffold genes.
    """
    from potency_metrics import (
        library_normalize, _xlogx, _get_X, cytotrace_proxy,
        _max_entropy_rate, _sr_from_matrix, _ccat_from_matrix,
    )

    # AnnData shim
    import anndata as ad
    adata = ad.AnnData(X=X)
    adata.var_names = [str(g) for g in gene_list]
    adata.obs_names = [f"c{i}" for i in range(X.shape[0])]

    print("  CytoTRACE proxy (GCS on top-200 GC-correlated genes) ...")
    cyto = cytotrace_proxy(adata)
    CT_score = cyto["gcs"]
    CT_primitive = cyto["gene_counts"]  # |supp(x)|

    print("  SCENT SR on human STRING LCC intersection ...")
    SR_score = pm.scent_sr(adata, A_scaffold, col_idx)

    # SR primitive: <x, degree>  (mean-field limit of SR)
    Xrel = library_normalize(X, target_sum=None)
    Xn = Xrel[:, col_idx]
    SR_primitive = Xn @ degree

    print("  CCAT on human STRING LCC intersection ...")
    CCAT_score = pm.ccat(adata, col_idx, degree, use_log=True)
    # CCAT primitive: same functional form (PCC), but without log1p — the raw
    # low-order statistic Pearson(rel(x), degree)
    CCAT_primitive = _ccat_from_matrix(Xn, degree)

    return {
        "CT_score": CT_score,
        "CT_primitive": CT_primitive,
        "SR_score": SR_score,
        "SR_primitive": SR_primitive,
        "CCAT_score": CCAT_score,
        "CCAT_primitive": CCAT_primitive,
    }


# =============================================================================
#  Skill + own-baseline + bootstrap
# =============================================================================
def _wtau(x, y):
    """Weighted Kendall tau with tie-handling via scipy averaging (see scipy docs)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    r = weightedtau(x[mask], y[mask])
    return float(getattr(r, "statistic", None) or r.correlation)


def _rank_residual(score, primitive):
    """
    Rank-residual of score against primitive (Spearman-style partial):
      1) rank both;
      2) fit a simple linear regression of rank(score) on rank(primitive);
      3) residual = rank(score) - fitted rank(score).
    This is the standard partial rank association construction.
    """
    x = np.asarray(score, dtype=np.float64)
    y = np.asarray(primitive, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    # linear fit
    cov = np.cov(rx, ry, ddof=0)[0, 1]
    var_y = np.var(ry)
    if var_y == 0:
        return None, mask
    beta = cov / var_y
    resid = rx - beta * ry
    return resid, mask


def _auroc_topbot(score, rank_arr, top_val, bot_vals):
    """AUROC for top-tier vs one-or-more bottom-tier values."""
    y = np.full(len(rank_arr), np.nan)
    y[rank_arr == top_val] = 1
    for b in bot_vals:
        y[rank_arr == b] = 0
    m = np.isfinite(y) & np.isfinite(score)
    if m.sum() < 3 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m].astype(int), score[m]))


def bootstrap_ci(fn, arrays, n_boot=N_BOOT, seed=SEED, ci=0.95):
    """Non-parametric percentile bootstrap CI for a scalar statistic over cells."""
    rng = np.random.default_rng(seed)
    N = len(arrays[0])
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        vals[b] = fn(*(a[idx] for a in arrays))
    lo = float(np.nanpercentile(vals, 100 * (1 - ci) / 2))
    hi = float(np.nanpercentile(vals, 100 * (1 - (1 - ci) / 2)))
    med = float(np.nanmedian(vals))
    return {"median": med, "lo": lo, "hi": hi}


def own_baseline_stats(score, primitive, rank_arr, top_val, bot_vals,
                       n_boot=N_BOOT, seed=SEED):
    """
    Full own-baseline stats for one score × one atlas:
      - skill_score  = τ_wt(score, ord)
      - skill_prim   = τ_wt(primitive, ord)
      - delta        = skill_score - skill_prim
      - cond_skill   = τ_wt(residual(score | primitive), ord)
      - AUROC_score  = AUROC(score, top-vs-bottom binary)
      - AUROC_prim   = AUROC(primitive, top-vs-bottom binary)
      - bootstrap 95% CIs for skill_score, skill_prim, delta, cond_skill
    """
    # Restrict to ranked cells only (drop NaN in rank_arr — UNRANKED cells)
    m = np.isfinite(rank_arr) & np.isfinite(score) & np.isfinite(primitive)
    s = score[m]; p = primitive[m]; r = rank_arr[m]
    n = int(m.sum())

    skill_score = _wtau(s, r)
    skill_prim = _wtau(p, r)
    delta = skill_score - skill_prim

    resid, _ = _rank_residual(s, p)
    if resid is None:
        cond_skill = float("nan")
    else:
        cond_skill = _wtau(resid, r)

    aur_score = _auroc_topbot(s, r, top_val, bot_vals)
    aur_prim = _auroc_topbot(p, r, top_val, bot_vals)

    # Bootstraps
    ci_score = bootstrap_ci(lambda a, b: _wtau(a, b), (s, r), n_boot=n_boot, seed=seed)
    ci_prim  = bootstrap_ci(lambda a, b: _wtau(a, b), (p, r), n_boot=n_boot, seed=seed + 1)
    ci_delta = bootstrap_ci(
        lambda a, b, c: _wtau(a, c) - _wtau(b, c),
        (s, p, r), n_boot=n_boot, seed=seed + 2,
    )
    def _cond_boot(a, b, c):
        resid_b, _m = _rank_residual(a, b)
        return _wtau(resid_b, c) if resid_b is not None else float("nan")
    ci_cond = bootstrap_ci(_cond_boot, (s, p, r), n_boot=n_boot, seed=seed + 3)

    return {
        "n_ranked_cells": n,
        "skill_score": skill_score,
        "skill_score_CI95": [ci_score["lo"], ci_score["hi"]],
        "skill_primitive": skill_prim,
        "skill_primitive_CI95": [ci_prim["lo"], ci_prim["hi"]],
        "marginal_delta": delta,
        "marginal_delta_CI95": [ci_delta["lo"], ci_delta["hi"]],
        "conditional_skill": cond_skill,
        "conditional_skill_CI95": [ci_cond["lo"], ci_cond["hi"]],
        "AUROC_score_top_vs_bot": aur_score,
        "AUROC_primitive_top_vs_bot": aur_prim,
    }


def verdict_from_delta(delta, cond_skill_ci_lo, cond_skill_ci_hi):
    """
    Locked thresholds (prereg §2):
      |Δ| <= 0.05          -> TAUTOLOG
      Δ > 0.10             -> SCORE-BEATS-PRIMITIVE (needs conditional CI to exclude 0)
      0.05 < Δ <= 0.10     -> inconclusive
      Δ < -0.10            -> sign-flipped scaffold
    """
    if not np.isfinite(delta):
        return "UNDEFINED"
    if abs(delta) <= DELTA_TAUTOLOG:
        return "TAUTOLOG"
    if delta > DELTA_SCORE_BEATS:
        # require conditional CI to exclude 0 to fire
        if np.isfinite(cond_skill_ci_lo) and np.isfinite(cond_skill_ci_hi):
            if cond_skill_ci_lo > 0 or cond_skill_ci_hi < 0:
                return "SCORE-BEATS-PRIMITIVE"
        return "inconclusive (Δ > 0.10 but conditional CI includes 0)"
    if delta < DELTA_SIGN_FLIP:
        return "SIGN-FLIPPED SCAFFOLD"
    return "inconclusive"


# =============================================================================
#  Per-atlas orchestration
# =============================================================================
def per_pop_depth(X, labels, unique_labels):
    """Median library size + median gene count per population."""
    out = {}
    lib = X.sum(axis=1)
    nnz = (X > 0).sum(axis=1)
    for u in unique_labels:
        m = (labels == u)
        if m.sum() == 0:
            out[str(u)] = {"n_cells": 0, "median_library_size": None, "median_gene_count": None}
            continue
        out[str(u)] = {
            "n_cells": int(m.sum()),
            "median_library_size": float(np.median(lib[m])),
            "median_gene_count": float(np.median(nnz[m])),
        }
    return out


def run_atlas(atlas_name, X, gene_list, cell_ids, labels, ranks, min_cells,
              rank_to_label_map, out_path):
    """Full pipeline: preprocess, scaffold, scores, own-baseline, dump JSON."""
    t0 = time.time()
    print(f"\n=== {atlas_name} ===")
    print(f"  cells (loaded): {X.shape[0]:,}")
    print(f"  genes (loaded): {X.shape[1]:,}")

    # per-population depth (CAVEAT 2)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    depth = per_pop_depth(X, labels, unique_labels)

    # gene filter
    print(f"  applying gene filter (min_cells={min_cells}) ...")
    Xf, gene_list_f = preprocess_and_filter(X, gene_list, min_cells)

    # scaffold
    print("  loading STRING v12 human LCC scaffold ...")
    A_scaffold, degree, col_idx = load_string_scaffold(gene_list_f)

    # scores + primitives
    print("  computing scores and primitives ...")
    S = compute_scores_and_primitives(Xf, gene_list_f, A_scaffold, degree, col_idx)

    # per-tier n (CAVEAT 1)
    tier_ns = {}
    for r in sorted(set(int(v) for v in ranks if np.isfinite(v))):
        m = ranks == r
        tier_ns[str(int(r))] = {
            "n_cells": int(m.sum()),
            "labels_included": sorted(set(labels[m].tolist())),
        }
    unranked_n = int(np.sum(~np.isfinite(ranks)))
    tier_ns["UNRANKED"] = {
        "n_cells": unranked_n,
        "labels_included": sorted(set(labels[~np.isfinite(ranks)].tolist())),
    }

    # Own-baseline for each score
    finite_ranks = ranks[np.isfinite(ranks)]
    top_val = int(np.max(finite_ranks))
    bot_val = int(np.min(finite_ranks))
    bot_vals = [bot_val]

    per_score_results = {}
    for stub in ["CT", "SR", "CCAT"]:
        print(f"  own-baseline: {stub} ...")
        stats = own_baseline_stats(
            score=S[f"{stub}_score"],
            primitive=S[f"{stub}_primitive"],
            rank_arr=ranks,
            top_val=top_val,
            bot_vals=bot_vals,
        )
        v = verdict_from_delta(
            stats["marginal_delta"],
            stats["conditional_skill_CI95"][0],
            stats["conditional_skill_CI95"][1],
        )
        stats["verdict"] = v
        per_score_results[stub] = stats

    result = {
        "atlas": atlas_name,
        "n_cells_total": int(X.shape[0]),
        "n_cells_ranked": int(np.sum(np.isfinite(ranks))),
        "n_cells_unranked": int(np.sum(~np.isfinite(ranks))),
        "n_genes_loaded": int(X.shape[1]),
        "n_genes_after_filter": int(Xf.shape[1]),
        "gene_filter_min_cells": min_cells,
        "scaffold": {
            "source": "STRING v12.0 human, threshold >=700, LCC free = 15,882 nodes",
            "intersect_atlas_LCC_nodes": int(A_scaffold.shape[0]),
        },
        "kernel": "scipy.stats.weightedtau (tie-handled via lexicographic averaging)",
        "per_tier_n": tier_ns,
        "per_population_depth": depth,
        "per_score": per_score_results,
        "top_tier": top_val,
        "bottom_tier": bot_val,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  written {out_path}")
    return result


# =============================================================================
#  Main
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-only", action="store_true")
    parser.add_argument("--c2-only", action="store_true")
    args = parser.parse_args()

    out_dir = HERE.parent / "results"
    out_dir.mkdir(exist_ok=True, parents=True)

    summary = {}

    if not args.c2_only:
        X, genes, cell_ids, labels, ranks = load_c1(
            HERE.parent / "data" / "c1_gse117498"
        )
        res_c1 = run_atlas(
            "C1_GSE117498",
            X, genes, cell_ids, labels, ranks,
            min_cells=C1_GENE_FILTER_MIN_CELLS,
            rank_to_label_map=C1_POP_TO_RANK,
            out_path=out_dir / "c1_gse117498_results.json",
        )
        summary["C1_GSE117498"] = {
            k: {
                "skill_score": v["skill_score"],
                "skill_primitive": v["skill_primitive"],
                "marginal_delta": v["marginal_delta"],
                "marginal_delta_CI95": v["marginal_delta_CI95"],
                "conditional_skill": v["conditional_skill"],
                "conditional_skill_CI95": v["conditional_skill_CI95"],
                "AUROC_score_top_vs_bot": v["AUROC_score_top_vs_bot"],
                "AUROC_primitive_top_vs_bot": v["AUROC_primitive_top_vs_bot"],
                "verdict": v["verdict"],
            }
            for k, v in res_c1["per_score"].items()
        }

    if not args.c1_only:
        X, genes, cell_ids, labels, ranks = load_c2(
            HERE.parent / "data" / "c2_gse125970"
        )
        res_c2 = run_atlas(
            "C2_GSE125970",
            X, genes, cell_ids, labels, ranks,
            min_cells=C2_GENE_FILTER_MIN_CELLS,
            rank_to_label_map=C2_LABEL_TO_RANK,
            out_path=out_dir / "c2_gse125970_results.json",
        )
        summary["C2_GSE125970"] = {
            k: {
                "skill_score": v["skill_score"],
                "skill_primitive": v["skill_primitive"],
                "marginal_delta": v["marginal_delta"],
                "marginal_delta_CI95": v["marginal_delta_CI95"],
                "conditional_skill": v["conditional_skill"],
                "conditional_skill_CI95": v["conditional_skill_CI95"],
                "AUROC_score_top_vs_bot": v["AUROC_score_top_vs_bot"],
                "AUROC_primitive_top_vs_bot": v["AUROC_primitive_top_vs_bot"],
                "verdict": v["verdict"],
            }
            for k, v in res_c2["per_score"].items()
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSUMMARY written: {out_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
