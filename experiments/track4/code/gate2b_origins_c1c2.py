"""
Track 4 Gate 2b — empirical own-baseline of ORIGINS on the same C1/C2 benchmark
used by the w4 pipeline. Piggybacks on the existing loader + preprocessing +
primitive+skill machinery in w4_gate2_run.py so the setup is identical to what
the PI already validated for CT/SR/CCAT.

Adds ONE score to the w4 setup: ORIGINS = P_k = x_k^T A x_k on the STRING v12
LCC restricted to expressed genes. Its declared primitive per §3a is
PCC(x, degree) — reuse w4's `CCAT_primitive` = Pearson(rel(x), degree).

For the by-admission scores (SR, CCAT, MCE, NCG, CT, StemID, cmEntropy) we
DO NOT re-run — the existing w4 numbers + Kang atlas per_dataset_results
+ authors' admissions in §3a are the record.
"""
from __future__ import annotations
import sys, json
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.stats import weightedtau
from scipy.stats import spearmanr, rankdata

# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# agent-session scratchpad, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root, repo_root  # noqa: E402
CODE = repo_root()   # this repository (code)
VAULT = data_root()  # run inputs and outputs (data)
sys.path.insert(0, str(CODE / "experiments" / "w4" / "scripts"))
sys.path.insert(0, str(CODE / "own_baseline"))
import w4_gate2_run as w4

SEED = 42
N_BOOT = 1000


def scipy_wtau(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    r = weightedtau(x[m], y[m])
    return float(getattr(r, "statistic", None) or r.correlation)


def rank_residual(score, primitive):
    m = np.isfinite(score) & np.isfinite(primitive)
    r_s = rankdata(score[m])
    r_p = rankdata(primitive[m])
    var_p = np.var(r_p)
    beta = np.cov(r_s, r_p, ddof=0)[0, 1] / var_p if var_p > 0 else 0.0
    resid_masked = r_s - beta * r_p
    resid = np.full_like(score, np.nan, dtype=np.float64)
    resid[m] = resid_masked
    return resid


def bootstrap_ci_wtau(score, ordinal, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    m = np.isfinite(score) & np.isfinite(ordinal)
    idx = np.where(m)[0]
    if len(idx) < 20:
        return float("nan"), (float("nan"), float("nan"))
    real = scipy_wtau(score[idx], ordinal[idx])
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        boots[i] = scipy_wtau(score[s], ordinal[s])
    lo, hi = np.quantile(boots, [0.025, 0.975])
    return float(real), (float(lo), float(hi))


def compute_origins(X, col_idx, A_scaffold):
    """
    ORIGINS per cell: P_k = x_k^T A_sub x_k where A_sub is A_scaffold indexed
    by col_idx (STRING gene columns present in this atlas).

    X is a scipy.sparse (or dense) matrix (cells × all_genes); col_idx are the
    positions of the STRING scaffold genes in the atlas gene ordering.
    """
    Xs = X[:, col_idx]
    if not sp.issparse(Xs):
        Xs = sp.csr_matrix(Xs)
    # ORIGINS uses raw expression per Senra et al. (formula only, no library norm)
    # But for comparability with CCAT primitive we also test lib-normalized below.
    n_cells = Xs.shape[0]
    origins = np.empty(n_cells, dtype=np.float64)
    for k in range(n_cells):
        x = Xs[k].toarray().ravel().astype(np.float64)
        Ax = A_scaffold @ x  # dense output vector
        origins[k] = float(x @ Ax)
    return origins


def run_atlas(atlas_name, X, gene_list, cell_ids, labels, ranks, min_cells, data_dir):
    """Wrap w4's run_atlas to also compute ORIGINS + its own-baseline."""
    # Use w4 preprocessing to intersect with STRING and get A_scaffold, degree, col_idx
    X_pre, gene_list_pre = w4.preprocess_and_filter(X, gene_list, min_cells)
    A_scaffold, degree, col_idx = w4.load_string_scaffold(gene_list_pre)

    # Get scores + primitives via w4 (CT, SR, CCAT)
    metrics = w4.compute_scores_and_primitives(X_pre, gene_list_pre, A_scaffold, degree, col_idx)

    # Compute ORIGINS
    print(f"  ORIGINS: P = x^T A x on STRING LCC ({A_scaffold.shape}, deg mean {degree.mean():.1f})")
    origins = compute_origins(X_pre, col_idx, A_scaffold)

    # Broad-potency ordinal filtered to ranked cells
    ranked_mask = np.array([r is not None and not (isinstance(r, float) and np.isnan(r)) for r in ranks])
    ranks_arr = np.array([r if r is not None else np.nan for r in ranks], dtype=np.float64)
    # apply preprocessing kept_mask to align
    if len(ranks_arr) != X_pre.shape[0]:
        # w4 preprocess_and_filter may drop cells too? Check.
        # Actually w4 keeps all cells; kept_mask is over genes. Ranks align with cells.
        pass

    # Compute skills for ORIGINS + its primitive (PCC(x, degree))
    pcc_primitive = metrics["CCAT_primitive"]  # Pearson(rel(x), degree) — the §3a primitive

    # For ORIGINS use PCC(x,degree) as primitive (§3a) — but ORIGINS is computed on raw X while CCAT primitive uses rel(x).
    # This is a scaling deviation; disclose. For a pure test, also compute PCC on raw X.
    Xrel = w4.pm.library_normalize(X_pre, target_sum=None) if hasattr(w4.pm, 'library_normalize') else None

    # Marginal skills
    mk_origins, ci_origins = bootstrap_ci_wtau(origins, ranks_arr)
    mk_pcc, ci_pcc = bootstrap_ci_wtau(pcc_primitive, ranks_arr)
    mk_gc, ci_gc = bootstrap_ci_wtau(metrics["CT_primitive"], ranks_arr)  # gene count
    print(f"  marginal skill ORIGINS: {mk_origins:+.4f} CI={ci_origins}")
    print(f"  marginal skill PCC(x,deg): {mk_pcc:+.4f} CI={ci_pcc}")
    print(f"  marginal skill gene_count: {mk_gc:+.4f} CI={ci_gc}")

    # Cross-correlations
    valid = np.isfinite(origins) & np.isfinite(pcc_primitive) & np.isfinite(metrics["CT_primitive"])
    rho_origins_pcc = float(spearmanr(origins[valid], pcc_primitive[valid]).statistic)
    rho_origins_gc = float(spearmanr(origins[valid], np.asarray(metrics["CT_primitive"])[valid]).statistic)
    rho_pcc_gc = float(spearmanr(pcc_primitive[valid], np.asarray(metrics["CT_primitive"])[valid]).statistic)
    print(f"  Spearman(ORIGINS, PCC) = {rho_origins_pcc:+.3f}")
    print(f"  Spearman(ORIGINS, gc)  = {rho_origins_gc:+.3f}")
    print(f"  Spearman(PCC, gc)      = {rho_pcc_gc:+.3f}")

    # Conditional skill (ORIGINS | PCC)
    resid_origins_given_pcc = rank_residual(origins, pcc_primitive)
    cond_tau, cond_ci = bootstrap_ci_wtau(resid_origins_given_pcc, ranks_arr)
    print(f"  conditional skill(ORIGINS | PCC(x,deg)): tau={cond_tau:+.4f} CI={cond_ci}")
    ci_lo, ci_hi = cond_ci
    if not np.isfinite(cond_tau):
        verdict = "INCONCLUSIVE"
    elif ci_lo <= 0 <= ci_hi:
        verdict = "TAUTOLOG (CI includes 0)"
    elif ci_lo > 0:
        verdict = "ADDS-BEYOND (CI strictly positive)"
    elif ci_hi < 0:
        verdict = "SIGN-FLIPPED"
    else:
        verdict = "INCONCLUSIVE"
    print(f"  VERDICT (§4 band): {verdict}")

    return {
        "atlas": atlas_name,
        "n_cells": int(X_pre.shape[0]),
        "n_genes_after_filter": int(X_pre.shape[1]),
        "scaffold_n_genes_in_atlas": int(len(col_idx)),
        "spearman_origins_vs_pcc": rho_origins_pcc,
        "spearman_origins_vs_gene_count": rho_origins_gc,
        "spearman_pcc_vs_gene_count": rho_pcc_gc,
        "marginal_skills": {
            "ORIGINS": {"tau": mk_origins, "CI95": list(ci_origins)},
            "PCC_x_degree": {"tau": mk_pcc, "CI95": list(ci_pcc)},
            "gene_count": {"tau": mk_gc, "CI95": list(ci_gc)},
        },
        "conditional_ORIGINS_given_PCC_x_deg": {
            "tau": cond_tau,
            "CI95": list(cond_ci),
            "verdict": verdict,
        },
        "kernel": "scipy.stats.weightedtau (rank-residual; boot n=1000 seed=42)",
    }


def main():
    out = {}

    # C1 hematopoietic (GSE117498)
    print("\n=== C1 GSE117498 ===")
    data_dir_c1 = str(VAULT / "w4/data/c1_gse117498")
    X_c1, gene_list_c1, cell_ids_c1, labels_c1, ranks_c1 = w4.load_c1(data_dir_c1)
    r1 = run_atlas("C1_GSE117498", X_c1, gene_list_c1, cell_ids_c1, labels_c1, ranks_c1, min_cells=3, data_dir=data_dir_c1)
    out["C1_GSE117498"] = r1

    # C2 intestinal (GSE125970)
    print("\n=== C2 GSE125970 ===")
    data_dir_c2 = str(VAULT / "w4/data/c2_gse125970")
    X_c2, gene_list_c2, cell_ids_c2, labels_c2, ranks_c2 = w4.load_c2(data_dir_c2)
    r2 = run_atlas("C2_GSE125970", X_c2, gene_list_c2, cell_ids_c2, labels_c2, ranks_c2, min_cells=3, data_dir=data_dir_c2)
    out["C2_GSE125970"] = r2

    out_path = VAULT / "track4/results/L2b_origins_c1c2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
