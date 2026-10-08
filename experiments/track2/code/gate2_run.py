"""
Track 2 Gate 2 — clean fine-ordinal own-baseline on Farrell 2018 zebrafish
embryogenesis (GSE106474 Drop-seq, 12 stages, 39,505 cells).

Runs (via existing potency_metrics + Track 4 ORIGINS reimpl):
  CT_proxy, SR, CCAT, ORIGINS (diff-BP), plus primitives (gene_count,
  PCC(x, degree), Shannon entropy).

Reports per-score conditional skill vs declared §3a primitive, both kernels
(scipy weightedtau + Kang wdm-uniform), bootstrap 95% CI seed=42, on the
12-stage ordinal.

DECIDER read: (a) kernels agree per score? (b) residual persists or vanishes
for SR/CCAT/CT?

Zebrafish gene symbols → human STRING v12 LCC intersection is by direct
symbol match (uppercase). Paralog forms (e.g. A2ML, ABCA1A/B) drop out;
disclosed. This is the standard cross-species convention for scaffold-based
scores.
"""
from __future__ import annotations
import sys, json, time, gzip, csv
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import weightedtau, spearmanr, rankdata

# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# scratch directory, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root, repo_root  # noqa: E402
CODE = repo_root()   # this repository (code)
VAULT = data_root()  # run inputs and outputs (data)
sys.path.insert(0, str(CODE / "own_baseline"))
sys.path.insert(0, str(CODE / "own_baseline" / "atlas_run"))
sys.path.insert(0, str(CODE / "experiments" / "w4" / "scripts"))
sys.path.insert(0, str(CODE / "experiments" / "track4" / "code"))
import potency_metrics as pm
from wdm_tau import wdm_tau_b_bit

UMI_FILE = VAULT / "track2/data/GSE106474_UMICounts.txt.gz"
STRING = VAULT / "w4/scaffolds/human_string_v12_thr700_lcc.npz"
DIFFBP = VAULT / "track4/data/diffbp_genes_in_string.txt"
OUT = VAULT / "track2/results/gate2_scoring.json"

SEED = 42
N_BOOT = 500

# Zebrafish developmental stages in ordered hpf progression
# Ordering per standard zebrafish staging (Kimmel 1995) — HIGH < OBLONG < DOME
#   < 30 < 50 < SHIELD < 60 < 75 < 90 < BUD < 3S < 6S
STAGE_ORDER = {
    "ZFHIGH": 1,   # ~3.3 hpf
    "ZFOBLONG": 2, # ~3.7
    "ZFDOME": 3,   # ~4.3
    "ZF30": 4,     # 30% epiboly ~4.7
    "ZF50": 5,     # 50% epiboly ~5.3
    "ZFS": 6,      # shield ~6.0
    "ZF60": 7,     # 60% epiboly ~6.7
    "ZF75": 8,     # 75% epiboly ~8.0
    "ZF90": 9,     # 90% epiboly ~9.0
    "ZFB": 10,     # bud ~10
    "ZF3S": 11,    # 3-somite ~11
    "ZF6S": 12,    # 6-somite ~12
}
STAGE_HPF = {
    "ZFHIGH": 3.3, "ZFOBLONG": 3.7, "ZFDOME": 4.3, "ZF30": 4.7,
    "ZF50": 5.3, "ZFS": 6.0, "ZF60": 6.7, "ZF75": 8.0, "ZF90": 9.0,
    "ZFB": 10.0, "ZF3S": 11.0, "ZF6S": 12.0,
}


def load_matrix():
    """Load Farrell UMI counts as (genes, cells) sparse matrix + gene list + cell list + stage int per cell."""
    print(f"Loading {UMI_FILE} ...")
    t0 = time.time()
    with gzip.open(UMI_FILE, "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        cell_ids = header  # first token is empty or gene column label; adjust
    # Determine if first header entry is empty (matrix convention)
    # Read column names — first row: cell IDs (39,504 cols after "" or gene col)
    if len(cell_ids) == 39505:
        # first token is "" — cell IDs start from index 1
        cell_ids = cell_ids[1:]
    # Load full matrix via pandas (dense; ~1 GB memory for 23974 × 39504 int)
    # Use dtype uint16 to save memory
    print("  Reading via pandas (may take a couple minutes)...")
    df = pd.read_csv(UMI_FILE, sep="\t", index_col=0, compression="gzip")
    # Convert to float32 after index (gene names) is set as row index
    print(f"  df: {df.shape}, took {time.time()-t0:.1f}s")
    genes = df.index.tolist()
    cells = df.columns.tolist()
    X = sp.csr_matrix(df.values.T.astype(np.float32))  # (cells, genes) sparse
    del df
    return X, genes, cells


def build_ordinal(cells):
    """Assign integer stage rank to each cell from its ID prefix."""
    stage_int = np.zeros(len(cells), dtype=np.float32)
    stage_str = []
    for i, c in enumerate(cells):
        pfx = c.split("_")[0]
        if pfx not in STAGE_ORDER:
            stage_int[i] = np.nan
            stage_str.append(None)
        else:
            stage_int[i] = STAGE_ORDER[pfx]
            stage_str.append(pfx)
    return stage_int, stage_str


def qc(X, min_genes_per_cell=200):
    """Cells with < min_genes_per_cell detected genes are dropped."""
    nnz = np.asarray((X > 0).sum(axis=1)).ravel()
    keep = nnz >= min_genes_per_cell
    print(f"  QC: kept {keep.sum()}/{len(keep)} cells with >= {min_genes_per_cell} genes")
    return keep


def load_string():
    d = np.load(STRING, allow_pickle=True)
    adj = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    genes = list(d["genes"])
    degree = np.asarray(d["degree"], dtype=np.float64)
    return adj, genes, degree


def compute_primitives(X, gene_list, string_genes, string_deg):
    """Per-cell primitives:
      - gene_count |supp(x)|
      - PCC(rel(x), degree) on STRING intersection
      - Shannon entropy H(x/Σx) over ALL genes
    """
    print("  primitives: gene_count, PCC(x,deg) on STRING, Shannon entropy...")
    nnz = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float64)
    # Intersect atlas with STRING
    string_idx = {g: i for i, g in enumerate(string_genes)}
    col_atlas = []
    col_string = []
    for i, g in enumerate(gene_list):
        if g in string_idx:
            col_atlas.append(i)
            col_string.append(string_idx[g])
    print(f"    atlas ∩ STRING: {len(col_atlas)} genes")
    col_atlas = np.array(col_atlas)
    col_string = np.array(col_string)
    deg_sub = string_deg[col_string]

    # PCC(rel(x), degree) per cell — use potency_metrics._ccat_from_matrix
    Xrel = pm.library_normalize(X, target_sum=None)
    Xrel_sub = Xrel[:, col_atlas]
    pcc = pm._ccat_from_matrix(Xrel_sub, deg_sub)

    # Shannon entropy H(x/Σx) — sparse safe
    row_sums = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    entropy = np.zeros(X.shape[0], dtype=np.float64)
    Xf = X.tocsr()
    for k in range(X.shape[0]):
        s = row_sums[k]
        if s <= 0:
            entropy[k] = np.nan; continue
        row = Xf.getrow(k)
        vals = row.data.astype(np.float64) / s
        vals = vals[vals > 0]
        entropy[k] = -np.sum(vals * np.log(vals))

    return nnz, pcc, entropy, col_atlas, col_string, deg_sub


def compute_scores(X, gene_list, adj_full, string_genes, string_deg, col_atlas, col_string, deg_sub):
    """Scores: CT_proxy, SR, CCAT, ORIGINS_diffbp."""
    print("  scores: CT_proxy, SR, CCAT, ORIGINS_diffbp...")
    import anndata as ad
    a = ad.AnnData(X=X)
    a.var_names = [str(g) for g in gene_list]
    a.obs_names = [f"c{i}" for i in range(X.shape[0])]
    # CT_proxy (GCS on top-200 GC-correlated genes)
    cyto = pm.cytotrace_proxy(a)
    ct_score = cyto["gcs"]
    # SR on STRING intersection subscaffold (need to build A_sub on col_string indexes)
    A_sub = adj_full[col_string][:, col_string]  # ordered by col_atlas positions
    # SR + CCAT — potency_metrics APIs expect col_idx into gene_list, A on that basis
    sr_score = pm.scent_sr(a, A_sub, col_atlas)
    ccat_score = pm.ccat(a, col_atlas, deg_sub, use_log=True)
    # ORIGINS on diff-BP subscaffold
    diffbp = set(open(DIFFBP).read().strip().split("\n"))
    diffbp_in_atlas = [g for g in gene_list if g in diffbp]
    diffbp_in_string = [g for g in diffbp_in_atlas if g in set(string_genes)]
    print(f"    diff-BP ∩ atlas ∩ STRING: {len(diffbp_in_string)} genes")
    string_idx = {g: i for i, g in enumerate(string_genes)}
    atlas_idx = {g: i for i, g in enumerate(gene_list)}
    diff_string_idx = [string_idx[g] for g in diffbp_in_string]
    diff_atlas_idx = [atlas_idx[g] for g in diffbp_in_string]
    A_diff = adj_full[diff_string_idx][:, diff_string_idx]
    X_diff = X[:, diff_atlas_idx]
    if not sp.issparse(X_diff):
        X_diff = sp.csr_matrix(X_diff)
    origins = np.zeros(X.shape[0], dtype=np.float64)
    for k in range(X.shape[0]):
        x = X_diff.getrow(k).toarray().ravel().astype(np.float64)
        origins[k] = float(x @ (A_diff @ x))
    return ct_score, sr_score, ccat_score, origins


# ===================== skills =====================
def scipy_wtau(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return float("nan")
    r = weightedtau(x[m], y[m])
    return float(getattr(r, "statistic", None) or r.correlation)


def kang_wtau(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return float("nan")
    w = np.ones(int(m.sum()), dtype=np.float64)
    return float(wdm_tau_b_bit(x[m], y[m], w))


def rank_residual(score, primitive):
    m = np.isfinite(score) & np.isfinite(primitive)
    if m.sum() < 20:
        return np.full_like(score, np.nan, dtype=np.float64)
    r_s = rankdata(score[m])
    r_p = rankdata(primitive[m])
    var_p = np.var(r_p)
    beta = np.cov(r_s, r_p, ddof=0)[0, 1] / var_p if var_p > 0 else 0.0
    out = np.full_like(score, np.nan, dtype=np.float64)
    out[m] = r_s - beta * r_p
    return out


def bootstrap_ci(fn, arrays, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    mask = np.ones(len(arrays[0]), dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    idx = np.where(mask)[0]
    if len(idx) < 20:
        return float("nan"), (float("nan"), float("nan"))
    real = fn(*[a[idx] for a in arrays])
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots[i] = fn(*[a[s] for a in arrays])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(real), (float(lo), float(hi))


def verdict(ci_lo, ci_hi):
    if not (np.isfinite(ci_lo) and np.isfinite(ci_hi)):
        return "INCONCLUSIVE"
    if ci_lo <= 0 <= ci_hi:
        return "TAUTOLOG (CI includes 0)"
    if ci_lo > 0:
        return "ADDS-BEYOND (CI strictly > 0)"
    if ci_hi < 0:
        return "SIGN-FLIPPED (CI strictly < 0)"
    return "INCONCLUSIVE"


def cond_skill_both_kernels(score, primitive, ordinal):
    resid = rank_residual(score, primitive)
    tau_scipy, ci_scipy = bootstrap_ci(scipy_wtau, [resid, ordinal])
    tau_kang, ci_kang = bootstrap_ci(kang_wtau, [resid, ordinal])
    v_scipy = verdict(*ci_scipy)
    v_kang = verdict(*ci_kang)
    combined = v_scipy if v_scipy == v_kang else "INCONCLUSIVE-kernel-disagree"
    return {
        "scipy_wtau": {"tau": tau_scipy, "CI95": list(ci_scipy), "verdict": v_scipy},
        "kang_wdm_uniform": {"tau": tau_kang, "CI95": list(ci_kang), "verdict": v_kang},
        "combined_verdict": combined,
    }


def per_stage_depth(X, stage_str, stage_order):
    nnz = np.asarray((X > 0).sum(axis=1)).ravel()
    libsize = np.asarray(X.sum(axis=1)).ravel()
    out = {}
    stage_arr = np.array(stage_str)
    for stage, rank in stage_order.items():
        mask = stage_arr == stage
        if mask.sum() == 0: continue
        out[stage] = {
            "rank": int(rank),
            "n_cells": int(mask.sum()),
            "median_gene_count": float(np.median(nnz[mask])),
            "median_library_size": float(np.median(libsize[mask])),
        }
    return out


def main():
    print("=== Track 2 Gate 2 scoring on Farrell 2018 GSE106474 ===\n")
    X, genes, cells = load_matrix()
    print(f"  loaded: {X.shape[0]} cells × {X.shape[1]} genes")
    stage_int, stage_str = build_ordinal(cells)
    print(f"  stages assigned: {(~np.isnan(stage_int)).sum()}/{len(stage_int)} cells")

    keep = qc(X, min_genes_per_cell=200)
    X = X[keep, :]
    stage_int = stage_int[keep]
    stage_str = [s for s, k in zip(stage_str, keep) if k]
    cells = [c for c, k in zip(cells, keep) if k]

    print("\nLoading STRING v12 human LCC...")
    adj_full, string_genes, string_deg = load_string()
    print(f"  STRING: {len(string_genes)} nodes, {adj_full.nnz} edges, mean deg {string_deg.mean():.1f}")

    print("\nComputing primitives...")
    nnz, pcc, entropy, col_atlas, col_string, deg_sub = compute_primitives(X, genes, string_genes, string_deg)

    print("\nComputing scores...")
    ct_score, sr_score, ccat_score, origins = compute_scores(X, genes, adj_full, string_genes, string_deg, col_atlas, col_string, deg_sub)

    print("\nComputing conditional skills against 12-stage ordinal...")
    # Per §3a declared primitives
    results = {
        "CT (proxy) | gene_count":        cond_skill_both_kernels(ct_score, nnz, stage_int),
        "SR | PCC(x, degree)":            cond_skill_both_kernels(sr_score, pcc, stage_int),
        "CCAT | PCC(x, degree)":          cond_skill_both_kernels(ccat_score, pcc, stage_int),
        "ORIGINS_diffbp | gene_count":    cond_skill_both_kernels(origins, nnz, stage_int),
        "ORIGINS_diffbp | PCC(x, deg)":   cond_skill_both_kernels(origins, pcc, stage_int),
    }
    for name, r in results.items():
        print(f"  {name}:")
        print(f"    scipy: tau={r['scipy_wtau']['tau']:+.4f} CI={r['scipy_wtau']['CI95']} → {r['scipy_wtau']['verdict']}")
        print(f"    kang : tau={r['kang_wdm_uniform']['tau']:+.4f} CI={r['kang_wdm_uniform']['CI95']} → {r['kang_wdm_uniform']['verdict']}")
        print(f"    combined: {r['combined_verdict']}")

    # Marginal skills (score vs ordinal directly)
    print("\nMarginal skills (score vs 12-stage ordinal):")
    marginal = {}
    for name, arr in [("CT_proxy", ct_score), ("SR", sr_score), ("CCAT", ccat_score),
                      ("ORIGINS_diffbp", origins), ("gene_count", nnz), ("PCC(x,degree)", pcc),
                      ("Shannon_entropy", entropy)]:
        t_sc, ci_sc = bootstrap_ci(scipy_wtau, [arr, stage_int])
        t_kg, ci_kg = bootstrap_ci(kang_wtau, [arr, stage_int])
        marginal[name] = {
            "scipy": {"tau": t_sc, "CI95": list(ci_sc)},
            "kang": {"tau": t_kg, "CI95": list(ci_kg)},
        }
        print(f"  {name:20s}  scipy tau={t_sc:+.4f}  kang tau={t_kg:+.4f}")

    # Cross-correlations
    xcorr = {
        "CT_vs_gene_count": float(spearmanr(ct_score, nnz).statistic),
        "SR_vs_PCC": float(spearmanr(sr_score, pcc).statistic),
        "CCAT_vs_PCC": float(spearmanr(ccat_score, pcc).statistic),
        "ORIGINS_vs_gene_count": float(spearmanr(origins, nnz).statistic),
        "ORIGINS_vs_PCC": float(spearmanr(origins, pcc).statistic),
    }
    print("\nSpearman cross-correlations:")
    for k, v in xcorr.items():
        print(f"  {k}: {v:+.4f}")

    # Per-stage depth
    print("\nPer-stage depth disclosure:")
    depth = per_stage_depth(X, stage_str, STAGE_ORDER)
    for s in sorted(depth, key=lambda x: depth[x]["rank"]):
        d = depth[s]
        print(f"  {s:10s} rank={d['rank']:2d} n={d['n_cells']:5d} med_genes={d['median_gene_count']:.0f} med_lib={d['median_library_size']:.0f}")

    out = {
        "dataset": "GSE106474 Farrell 2018 zebrafish (Drop-seq, 12 stages)",
        "n_cells_after_qc": int(X.shape[0]),
        "n_genes": int(X.shape[1]),
        "n_ranked": int((~np.isnan(stage_int)).sum()),
        "n_genes_atlas_x_string": int(len(col_atlas)),
        "n_genes_diffbp_x_atlas_x_string": None,  # filled from origins call
        "conditional_skills": results,
        "marginal_skills": marginal,
        "spearman_cross_correlations": xcorr,
        "per_stage_depth": depth,
        "notes": {
            "stage_ordering": "HIGH < OBLONG < DOME < 30 < 50 < SHIELD < 60 < 75 < 90 < BUD < 3S < 6S (Kimmel 1995 hpf).",
            "cross_species_scaffold": "STRING v12 HUMAN LCC used for SR/CCAT/ORIGINS. Zebrafish gene symbols matched by exact string; paralogs (A2ML, ABCA1A/B etc.) drop out. Standard cross-species scaffold convention.",
            "kang_kernel_weights": "uniform per-cell (no Kang absolute-order weights available for this non-Kang atlas — reduces to plain Kendall tau-b).",
            "seed": SEED,
            "n_boot": N_BOOT,
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
