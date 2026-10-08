"""
Track 4 Gate 2b — empirical own-baseline pilot on Kang atlas dataset E-MTAB-9067.

Focus: ORIGINS reimplementation from formula (P = x^T A x on STRING v12 LCC) +
conditional skill against declared primitive PCC(x, degree). Confirms/refutes
the PI §3a PARTIAL classification empirically on one Kang benchmark row.

By-admission scores (SR, CCAT, MCE, NCG, StemID, cmEntropy, CT):
  do NOT re-run. Cite existing Kang atlas per_dataset_results and the authors'
  admissions from §3a.

INCONCLUSIVE-infra scores (SLICE, dpath, scEnergy, SPIDE):
  reported honestly per §7 — R / MATLAB / paywall infrastructure unavailable
  in this session's env.

Kernels: scipy weightedtau (available); Kang wdm (via
  06-code/tautology-diagnostic/atlas_run/wdm_tau.py) — imported if needed but
  scope: scipy weightedtau is the primary kernel for the pilot; Kang wdm
  values for by-admission scores read from per_dataset_results.json.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.stats import weightedtau, spearmanr
import anndata as ad

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
sys.path.insert(0, str(CODE / "own_baseline" / "atlas_run"))
from wdm_tau import wdm_tau_b_bit, kang_absolute_weights  # noqa: E402

H5AD = VAULT / "atlas_run/data/E-MTAB-9067/E-MTAB-9067.h5ad"
STRING = VAULT / "w4/scaffolds/human_string_v12_thr700_lcc.npz"
S3 = VAULT / "atlas_run/kang2025_SI/Table_S3.csv"
OUT = VAULT / "track4/results/L2b_pilot_e_mtab_9067.json"

SEED = 42
N_BOOT = 1000
POT_ORD = {
    "Totipotent": 1, "Pluripotent": 2, "Multipotent": 3,
    "Oligopotent": 4, "Unipotent": 5, "Differentiated": 6,
}


def load_expression():
    a = ad.read_h5ad(H5AD)
    # var_names may be Ensembl or symbols; check
    var = a.var_names
    print(f"  E-MTAB shape: {a.shape}; first vars: {list(var[:5])}")
    return a


def load_string():
    d = np.load(STRING, allow_pickle=True)
    adj = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    return adj, list(d["genes"]), np.asarray(d["degree"], dtype=np.float64)


def broad_potency_from_author_label(a):
    """Return per-cell broad_potency integer (1..6) or NaN via Kang S3 crosswalk on HSC development."""
    import pandas as pd
    S3 = pd.read_csv(str(globals()["S3"]), header=1)  # header row is line 2 in file
    hsc = S3[S3["Standardized dataset name"].astype(str).str.contains("HSC development", na=False)]
    # Build author-label -> broad map from Kang S3
    author_map = {}
    for _, r in hsc.iterrows():
        auth = str(r["Original phenotype"]).strip()
        broad = str(r["Broad potency level"]).strip()
        if broad in POT_ORD:
            author_map[auth] = POT_ORD[broad]
    # E-MTAB uses expanded English names; map via a fixed English->Kang crosswalk
    # (This mirrors baseline_validation_e_mtab_9067.py's approach.)
    ENG_TO_KANG = {
        "hematopoietic multipotent progenitor cell": "HSC-MPPs",
        "multipotent progenitor cell": "HSC-MPPs",
        "progenitor cell": "HSC-MPPs",
        "erythroid progenitor cell": "MEMPs",
        "granulocyte progenitor": "GPs",
        "pro-B cell": "Pro-B cells",
        "precursor B cell": "Pre-B cells",
        "mature B cell": "Mature B cells",
        "monocyte 1": "Monocytes 1",
        "monocyte 2": "Monocytes 2",
        "mast cell": "Mast cells",
        "cycling committed progenitor": "HSC-MPPs-Cycle",
        "committed progenitor": "LMPs",
    }
    lab_col = "Factor Value[inferred cell type - authors labels]"
    labs = a.obs[lab_col].astype(str).str.strip().str.lower()
    def resolve(x):
        x = x.strip()
        if x in ENG_TO_KANG:
            return author_map.get(ENG_TO_KANG[x], np.nan)
        return author_map.get(x, np.nan)
    ords = np.array([resolve(v) for v in labs], dtype=np.float64)
    n_mapped = int(np.isfinite(ords).sum())
    print(f"  broad-potency ordinal mapped for {n_mapped}/{len(ords)} cells")
    return ords


def compute_primitives_and_origins(a, string_genes, degree):
    """Compute gene_count, PCC(x, degree), Shannon entropy, and ORIGINS per cell.

    ORIGINS = x^T A x on STRING v12 LCC — reimplemented per Senra 2022 formula:
      P_k = Σ_{i,j} A_ij x_i x_j = x_k^T A x_k
    where A is the STRING v12 human LCC adjacency (subset to expressed genes).
    """
    # Intersect var_names with STRING genes
    string_gene_set = {g: i for i, g in enumerate(string_genes)}
    var = list(a.var_names)
    in_string_mask = np.array([g in string_gene_set for g in var])
    print(f"  {in_string_mask.sum()}/{len(var)} vars in STRING LCC")
    # Build gene→string_idx map for expressed genes only
    keep_var = [i for i, g in enumerate(var) if g in string_gene_set]
    keep_str = [string_gene_set[var[i]] for i in keep_var]
    # Reindex expression matrix to STRING gene order (only genes present in both)
    X = a.X[:, keep_var]  # (n_cells, n_keep) sparse
    if not sp.issparse(X):
        X = sp.csr_matrix(X)
    # Build sub-adjacency of STRING LCC restricted to keep_str genes
    # A_sub[i, j] = A_full[keep_str[i], keep_str[j]]
    # Load full adjacency
    d = np.load(STRING, allow_pickle=True)
    A_full = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    idx = np.array(keep_str)
    A_sub = A_full[idx][:, idx]
    deg_sub = np.asarray(degree)[idx]
    print(f"  A_sub: {A_sub.shape}, nnz={A_sub.nnz}, mean deg={deg_sub.mean():.1f}")

    # Per-cell primitives
    n_cells = X.shape[0]
    # gene count (nnz per row)
    gc = np.asarray((X > 0).sum(axis=1)).ravel()

    # PCC(x, degree) per cell over the shared gene space
    X_arr = X.toarray().astype(np.float64)  # OK for this size (5315 × ~13000)
    print(f"  X dense shape: {X_arr.shape}, mem ~{X_arr.nbytes/1e6:.1f} MB")
    dmean = deg_sub.mean()
    dstd = deg_sub.std()
    pcc = np.zeros(n_cells)
    for k in range(n_cells):
        x = X_arr[k]
        xmean = x.mean()
        xstd = x.std()
        if xstd == 0 or dstd == 0:
            pcc[k] = 0.0
        else:
            pcc[k] = np.mean((x - xmean) * (deg_sub - dmean)) / (xstd * dstd)

    # Shannon entropy H(x/Σx) per cell
    row_sums = X_arr.sum(axis=1)
    entropy = np.zeros(n_cells)
    for k in range(n_cells):
        s = row_sums[k]
        if s <= 0:
            entropy[k] = np.nan; continue
        p = X_arr[k] / s
        pnz = p[p > 0]
        entropy[k] = -np.sum(pnz * np.log(pnz))

    # ORIGINS: P_k = x_k^T A x_k
    # Compute as: for each cell k, compute A @ x_k then dot with x_k
    origins = np.zeros(n_cells)
    for k in range(n_cells):
        x = X_arr[k]
        Ax = A_sub @ x  # vector
        origins[k] = float(x @ Ax)

    return {
        "n_cells": int(n_cells),
        "n_genes_shared": int(len(keep_var)),
        "gene_count": gc.tolist(),
        "pcc_x_degree": pcc.tolist(),
        "shannon_entropy": entropy.tolist(),
        "origins": origins.tolist(),
    }


def scipy_wtau(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return float("nan")
    r = weightedtau(x[m], y[m])
    return float(r.statistic if hasattr(r, "statistic") else r.correlation)


def rank_residual(score, primitive):
    """Residual of score after regressing on primitive (rank-space)."""
    from scipy.stats import rankdata
    r_s = rankdata(score)
    r_p = rankdata(primitive)
    # linear residual r_s - beta * r_p
    beta = np.cov(r_s, r_p)[0, 1] / np.var(r_p) if np.var(r_p) > 0 else 0.0
    return r_s - beta * r_p


def bootstrap_ci_wtau(score, ordinal, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(score)
    m = ~(np.isnan(score) | np.isnan(ordinal))
    idx_valid = np.where(m)[0]
    if len(idx_valid) < 20:
        return float("nan"), (float("nan"), float("nan"))
    real = scipy_wtau(score[idx_valid], ordinal[idx_valid])
    boots = []
    for _ in range(n_boot):
        smp = rng.choice(idx_valid, size=len(idx_valid), replace=True)
        boots.append(scipy_wtau(score[smp], ordinal[smp]))
    boots = np.asarray(boots)
    ci = (float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975)))
    return float(real), ci


def main():
    print("=== Track 4 Gate 2b pilot: E-MTAB-9067 ===")
    a = load_expression()
    _adj, string_genes, degree = load_string()

    print("Computing per-cell primitives + ORIGINS...")
    P = compute_primitives_and_origins(a, string_genes, degree)

    print("Building broad-potency ordinal from Kang S3 crosswalk...")
    ord_arr = broad_potency_from_author_label(a)

    # convert to np arrays
    origins = np.asarray(P["origins"])
    pcc = np.asarray(P["pcc_x_degree"])
    gc = np.asarray(P["gene_count"], dtype=np.float64)
    ent = np.asarray(P["shannon_entropy"])

    # Sanity: correlation of ORIGINS with its declared primitive PCC(x, degree)
    from scipy.stats import spearmanr
    rho_origins_pcc = spearmanr(origins, pcc).statistic
    rho_origins_gc = spearmanr(origins, gc).statistic
    rho_pcc_gc = spearmanr(pcc, gc).statistic
    print(f"  Spearman(ORIGINS, PCC(x,degree)) = {rho_origins_pcc:.3f}")
    print(f"  Spearman(ORIGINS, gene_count)    = {rho_origins_gc:.3f}")
    print(f"  Spearman(PCC, gene_count)        = {rho_pcc_gc:.3f}")

    # Skills against broad-potency ordinal (per §3b — sign of tau depends on ordering direction;
    # broad-potency scale is 1=high potency; higher score means what we want mapped consistently)
    # Sign-agnostic: report |tau|
    def report(name, arr):
        tau, ci = bootstrap_ci_wtau(arr, ord_arr)
        print(f"  τ_wt(|{name}|, broad_potency): tau={tau:+.4f} CI=[{ci[0]:+.4f}, {ci[1]:+.4f}]")
        return {"tau": tau, "CI95": list(ci)}

    print("\nMarginal skills:")
    sk_origins = report("ORIGINS", origins)
    sk_pcc = report("PCC(x,degree)", pcc)
    sk_gc = report("gene_count", gc)
    sk_ent = report("Shannon_entropy", ent)

    # Conditional skill(ORIGINS | PCC(x,degree)) — rank-residual method (as in w4)
    resid = rank_residual(origins, pcc)
    cond_tau, cond_ci = bootstrap_ci_wtau(resid, ord_arr)
    print(f"\nConditional skill(ORIGINS | PCC(x,degree)): tau={cond_tau:+.4f} CI=[{cond_ci[0]:+.4f}, {cond_ci[1]:+.4f}]")
    # Verdict per §4 bands: |cond_skill CI| includes 0 → TAUTOLOG; excludes 0 positive → ADDS-BEYOND
    ci_lo, ci_hi = cond_ci
    if ci_lo <= 0 <= ci_hi:
        verdict = "TAUTOLOG (conditional-skill CI includes 0)"
    elif ci_lo > 0:
        verdict = "ADDS-BEYOND (conditional-skill CI strictly positive)"
    elif ci_hi < 0:
        verdict = "SIGN-FLIPPED (conditional-skill CI strictly negative — score orders opposite to primitive residual)"
    else:
        verdict = "INCONCLUSIVE"
    print(f"  VERDICT (§4 bands): {verdict}")

    # Sanity check: conditional skill(gene_count | gene_count) — trivially 0 by construction
    resid_self = rank_residual(gc, gc)
    cond_gc_gc, cond_gc_gc_ci = bootstrap_ci_wtau(resid_self, ord_arr)
    print(f"\nSanity: conditional skill(gene_count | gene_count) tau={cond_gc_gc:+.4f} (should be ~0 by construction)")

    out = {
        "dataset": "E-MTAB-9067 (HSC development, Smart-seq2)",
        "n_cells": int(P["n_cells"]),
        "n_genes_shared_with_STRING_LCC": int(P["n_genes_shared"]),
        "broad_potency_ordinal_mapped_n": int(np.isfinite(ord_arr).sum()),
        "spearman_origins_vs_pcc_x_deg": float(rho_origins_pcc),
        "spearman_origins_vs_gene_count": float(rho_origins_gc),
        "spearman_pcc_vs_gene_count": float(rho_pcc_gc),
        "marginal_skills": {
            "ORIGINS": sk_origins,
            "PCC_x_degree": sk_pcc,
            "gene_count": sk_gc,
            "Shannon_entropy": sk_ent,
        },
        "conditional_skill_ORIGINS_given_PCC_x_deg": {
            "tau": float(cond_tau),
            "CI95": [float(cond_ci[0]), float(cond_ci[1])],
            "kernel": "scipy.stats.weightedtau (rank-residual against primitive; boot n=1000 seed=42)",
            "verdict_§4_band": verdict,
        },
        "sanity_conditional_gc_given_gc": {
            "tau": float(cond_gc_gc),
            "CI95": [float(cond_gc_gc_ci[0]), float(cond_gc_gc_ci[1])],
            "expected": "≈ 0 by construction",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
