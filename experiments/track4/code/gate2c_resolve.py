"""
Track 4 Gate 2c — resolve ORIGINS + SR primitive on C1/C2.

ORIGINS re-test:
  (a) Rebuild on Senra 2022's actual scaffold — the CELL DIFFERENTIATION (GO:0030154)
      biological-process subnetwork. This session's proxy: STRING v12 human LCC
      restricted to GO:0030154 members (n = 3,196 genes with STRING presence).
      Senra used Pathway Commons PPI (11,582 proteins, 191,072 edges) — the
      proxy is a smaller intersection but the CLOSEST documented equivalent
      that keeps the scaffold building block (STRING v12) constant with the
      rest of the pipeline. Deviation disclosed.
  (b) Compute conditional skill of ORIGINS vs gene count |supp(x)| — the
      empirically-observed correlate (Spearman 0.81 / 0.90 in Gate 2b).
  (c) Also report vs PCC(x, degree) for comparison (§3a candidate primitive).
  BOTH KERNELS.

SR re-test:
  Compute conditional skill of SR vs PCC(x, degree) — the §3a-locked primitive
  (NOT <x, degree> which was the Gate 2b substitute). Confirm TAUTOLOG consistent
  with authors' R²=0.96 admission (Teschendorff 2017 Supp Fig 7B).
  BOTH KERNELS.
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import weightedtau, spearmanr, rankdata

VAULT = Path("/Users/damir/damir-research-vault/06-code/tautology-diagnostic")
sys.path.insert(0, str(VAULT / "w4/scripts"))
sys.path.insert(0, str(VAULT))
sys.path.insert(0, str(VAULT / "atlas_run"))
import w4_gate2_run as w4
from wdm_tau import wdm_tau_b_bit, kang_absolute_weights  # noqa: E402
import potency_metrics as pm

SEED = 42
N_BOOT = 1000
DIFFBP_LIST = Path("/Users/damir/damir-research-vault/06-code/tautology-diagnostic/track4/data/diffbp_genes_in_string.txt")
OUT = Path("/Users/damir/damir-research-vault/06-code/tautology-diagnostic/track4/results/L2c_resolve.json")


def scipy_wtau(x, y):
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return float('nan')
    r = weightedtau(x[m], y[m])
    return float(getattr(r, "statistic", None) or r.correlation)


def kang_wtau(x, y):
    """Kang wdm kernel — dense O(n^2) implementation from wdm_tau_b_bit.

    C1/C2 external atlases lack Kang's standardized-phenotype + potency-level metadata
    needed for the absolute-order weight scheme (`kang_absolute_weights` requires
    both). We use UNIFORM per-cell weights on these atlases, which reduces
    wdm_tau_b_bit to plain Kendall tau-b (weighted by pair, not by rank). This is
    the honest per-atlas kernel for §3b's "both kernels" clause on these two
    external atlases; it is disclosed as such.
    """
    x = np.asarray(x, dtype=np.float64); y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3: return float('nan')
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
    resid_masked = r_s - beta * r_p
    out = np.full_like(score, np.nan, dtype=np.float64)
    out[m] = resid_masked
    return out


def bootstrap_ci(fn, arrays, n_boot=N_BOOT, seed=SEED, ci=0.95):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    mask = np.ones(n, dtype=bool)
    for a in arrays:
        mask &= np.isfinite(a)
    idx = np.where(mask)[0]
    if len(idx) < 20:
        return float('nan'), (float('nan'), float('nan'))
    real = fn(*[a[idx] for a in arrays])
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots[i] = fn(*[a[s] for a in arrays])
    lo, hi = np.quantile(boots, [(1-ci)/2, 1-(1-ci)/2])
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
    return {
        "scipy_wtau": {
            "tau": tau_scipy,
            "CI95": list(ci_scipy),
            "verdict": verdict(*ci_scipy),
        },
        "kang_wdm": {
            "tau": tau_kang,
            "CI95": list(ci_kang),
            "verdict": verdict(*ci_kang),
        },
    }


def build_diffbp_subscaffold(A_full, string_genes_all, diffbp_genes_in_string):
    """Restrict A_full to diff-BP genes; return A_sub, sub_gene_list, sub_degree."""
    string_idx = {g: i for i, g in enumerate(string_genes_all)}
    idx = np.array([string_idx[g] for g in diffbp_genes_in_string if g in string_idx])
    A_sub = A_full[idx][:, idx]
    sub_genes = [string_genes_all[i] for i in idx]
    sub_degree = np.asarray(A_sub.sum(axis=1)).ravel().astype(np.float64)
    return A_sub, sub_genes, sub_degree, idx


def compute_origins_and_primitives_diffbp(X_pre, gene_list_pre, A_full, string_genes_all, diffbp_genes):
    """On the diff-BP subscaffold:
      - ORIGINS = x^T A_diff x where x is per-cell expression restricted to diff-BP genes
      - PCC(x, degree) primitive against the diff-BP subscaffold degree
      - gene_count |supp(x)| — scaffold-independent (raw count over ALL genes,
        NOT restricted to diff-BP; this is the operational primitive papers use)
    """
    A_sub, sub_genes, sub_deg, _idx_in_full = build_diffbp_subscaffold(A_full, string_genes_all, diffbp_genes)
    print(f"  diff-BP subscaffold: {A_sub.shape}, nnz={A_sub.nnz}, deg mean={sub_deg.mean():.2f}")
    # Map atlas gene list → position in diff-BP subscaffold
    gene_pos = {g: i for i, g in enumerate(gene_list_pre)}
    sub_col_idx = np.array([gene_pos[g] for g in sub_genes if g in gene_pos])
    kept_sub_genes = [g for g in sub_genes if g in gene_pos]
    keep_str_pos = np.array([sub_genes.index(g) for g in kept_sub_genes])
    A_sub_kept = A_sub[keep_str_pos][:, keep_str_pos]
    sub_deg_kept = sub_deg[keep_str_pos]
    print(f"  atlas ∩ diff-BP: {len(sub_col_idx)} genes")
    # Slice X to sub_col_idx
    Xs = X_pre[:, sub_col_idx]
    if not sp.issparse(Xs):
        Xs = sp.csr_matrix(Xs)
    n_cells = Xs.shape[0]
    # ORIGINS per cell: raw expression as in Senra (paper uses normalized 0-1 counts; we use raw here consistent with Gate 2b; disclose)
    origins = np.empty(n_cells, dtype=np.float64)
    for k in range(n_cells):
        x = Xs[k].toarray().ravel().astype(np.float64)
        origins[k] = float(x @ (A_sub_kept @ x))
    # PCC(x, degree) on diff-BP subscaffold — Pearson(rel(x), degree)
    from potency_metrics import library_normalize, _ccat_from_matrix
    Xrel_sub = library_normalize(Xs, target_sum=None)
    # For PCC computation, use Pearson(rel(x), sub_degree) per cell
    pcc_diffbp = _ccat_from_matrix(Xrel_sub, sub_deg_kept)
    return origins, pcc_diffbp, sub_col_idx


def run_atlas_2c(atlas_name, X, gene_list, cell_ids, labels, ranks, min_cells):
    print(f"\n=== {atlas_name} ===")
    X_pre, gene_list_pre = w4.preprocess_and_filter(X, gene_list, min_cells)
    # Load full STRING for both full-LCC primitives + diff-BP subscaffold
    A_full_scaf, degree_full, col_idx_full = w4.load_string_scaffold(gene_list_pre)
    # Load STRING v12 genes for diff-BP building
    d = np.load(VAULT / "w4/scaffolds/human_string_v12_thr700_lcc.npz", allow_pickle=True)
    A_full_all = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    string_genes_all = list(d["genes"])
    diffbp_genes = open(DIFFBP_LIST).read().strip().split('\n')

    # Metrics from w4 (CT/SR/CCAT scores + their primitives on full-LCC)
    m = w4.compute_scores_and_primitives(X_pre, gene_list_pre, A_full_scaf, degree_full, col_idx_full)
    ct_gc = np.asarray(m["CT_primitive"], dtype=np.float64)  # gene_count
    sr_score = np.asarray(m["SR_score"], dtype=np.float64)
    ccat_pcc = np.asarray(m["CCAT_primitive"], dtype=np.float64)  # PCC(rel(x), degree) — §3a-locked SR primitive too

    # ORIGINS + PCC on diff-BP subscaffold
    print("  Building diff-BP ORIGINS...")
    origins_diffbp, pcc_diffbp, _ = compute_origins_and_primitives_diffbp(
        X_pre, gene_list_pre, A_full_all, string_genes_all, diffbp_genes)

    ranks_arr = np.array([r if r is not None else np.nan for r in ranks], dtype=np.float64)

    # ORIGINS conditional skill vs gene_count (empirical correlate)
    print("  ORIGINS_diffBP | gene_count ...")
    ck_o_gc = cond_skill_both_kernels(origins_diffbp, ct_gc, ranks_arr)
    print(f"    scipy: {ck_o_gc['scipy_wtau']['tau']:+.4f} CI={ck_o_gc['scipy_wtau']['CI95']} → {ck_o_gc['scipy_wtau']['verdict']}")
    print(f"    kang : {ck_o_gc['kang_wdm']['tau']:+.4f} CI={ck_o_gc['kang_wdm']['CI95']} → {ck_o_gc['kang_wdm']['verdict']}")

    # ORIGINS conditional skill vs PCC(x, degree) — diff-BP subscaffold degree
    print("  ORIGINS_diffBP | PCC(x, diffBP_degree) ...")
    ck_o_pcc_diff = cond_skill_both_kernels(origins_diffbp, pcc_diffbp, ranks_arr)
    print(f"    scipy: {ck_o_pcc_diff['scipy_wtau']['tau']:+.4f} CI={ck_o_pcc_diff['scipy_wtau']['CI95']} → {ck_o_pcc_diff['scipy_wtau']['verdict']}")
    print(f"    kang : {ck_o_pcc_diff['kang_wdm']['tau']:+.4f} CI={ck_o_pcc_diff['kang_wdm']['CI95']} → {ck_o_pcc_diff['kang_wdm']['verdict']}")

    # ORIGINS conditional skill vs PCC(x, full-LCC degree) — Gate 2b comparison
    print("  ORIGINS_diffBP | PCC(x, full-STRING-degree) ...")
    ck_o_pcc_full = cond_skill_both_kernels(origins_diffbp, ccat_pcc, ranks_arr)
    print(f"    scipy: {ck_o_pcc_full['scipy_wtau']['tau']:+.4f} CI={ck_o_pcc_full['scipy_wtau']['CI95']} → {ck_o_pcc_full['scipy_wtau']['verdict']}")
    print(f"    kang : {ck_o_pcc_full['kang_wdm']['tau']:+.4f} CI={ck_o_pcc_full['kang_wdm']['CI95']} → {ck_o_pcc_full['kang_wdm']['verdict']}")

    # Cross-correlations for transparency
    rho_o_gc = float(spearmanr(origins_diffbp, ct_gc, nan_policy='omit').statistic)
    rho_o_pcc_diff = float(spearmanr(origins_diffbp, pcc_diffbp, nan_policy='omit').statistic)
    rho_o_pcc_full = float(spearmanr(origins_diffbp, ccat_pcc, nan_policy='omit').statistic)

    # SR conditional skill vs PCC(x, full-LCC degree) — §3a-locked primitive
    print("  SR | PCC(x, degree) [§3a-locked] ...")
    ck_sr_pcc = cond_skill_both_kernels(sr_score, ccat_pcc, ranks_arr)
    print(f"    scipy: {ck_sr_pcc['scipy_wtau']['tau']:+.4f} CI={ck_sr_pcc['scipy_wtau']['CI95']} → {ck_sr_pcc['scipy_wtau']['verdict']}")
    print(f"    kang : {ck_sr_pcc['kang_wdm']['tau']:+.4f} CI={ck_sr_pcc['kang_wdm']['CI95']} → {ck_sr_pcc['kang_wdm']['verdict']}")

    # Cross-correlations for SR
    rho_sr_pcc = float(spearmanr(sr_score, ccat_pcc, nan_policy='omit').statistic)
    rho_sr_gc = float(spearmanr(sr_score, ct_gc, nan_policy='omit').statistic)

    return {
        "atlas": atlas_name,
        "n_cells": int(X_pre.shape[0]),
        "n_ranked": int(np.isfinite(ranks_arr).sum()),
        "origins_diffbp": {
            "cross_correlations_spearman": {
                "vs_gene_count": rho_o_gc,
                "vs_PCC_diffbp_degree": rho_o_pcc_diff,
                "vs_PCC_full_string_degree": rho_o_pcc_full,
            },
            "conditional_skill_vs_gene_count": ck_o_gc,
            "conditional_skill_vs_PCC_diffbp_degree": ck_o_pcc_diff,
            "conditional_skill_vs_PCC_full_string_degree": ck_o_pcc_full,
        },
        "sr": {
            "cross_correlations_spearman": {
                "vs_PCC_x_degree": rho_sr_pcc,
                "vs_gene_count": rho_sr_gc,
            },
            "conditional_skill_vs_PCC_x_degree": ck_sr_pcc,
        },
        "notes": {
            "scaffold_deviation": "diff-BP subscaffold = STRING v12 LCC restricted to GO:0030154 members (proxy, n=3,196 ∩ atlas). Senra 2022 used Pathway Commons PPI restricted to GO:0030154 (11,582 proteins). Proxy keeps STRING v12 constant with rest of pipeline.",
            "SR_primitive_choice": "PCC(rel(x), degree) — §3a-locked; NOT the <x, degree> mean-field substitute used in Gate 2b w4 pipeline.",
        },
    }


def main():
    out = {}
    print("Loading C1 (GSE117498 hemat) ...")
    data_dir_c1 = str(VAULT / "w4/data/c1_gse117498")
    X_c1, gene_list_c1, cell_ids_c1, labels_c1, ranks_c1 = w4.load_c1(data_dir_c1)
    out["C1_GSE117498"] = run_atlas_2c("C1_GSE117498", X_c1, gene_list_c1, cell_ids_c1, labels_c1, ranks_c1, min_cells=3)

    print("\nLoading C2 (GSE125970 intestinal) ...")
    data_dir_c2 = str(VAULT / "w4/data/c2_gse125970")
    X_c2, gene_list_c2, cell_ids_c2, labels_c2, ranks_c2 = w4.load_c2(data_dir_c2)
    out["C2_GSE125970"] = run_atlas_2c("C2_GSE125970", X_c2, gene_list_c2, cell_ids_c2, labels_c2, ranks_c2, min_cells=3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
