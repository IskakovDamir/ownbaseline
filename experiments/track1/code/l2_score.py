"""
Track-1 Gate-2 L2 scoring pipeline (Venet-style).

For a given (row, signature, cohort):
  1. Load Xena-format expression matrix (log2(x+1) TPM/RSEM).
  2. Compute per-sample signature score S_i = mean over sig genes of z-scored expression.
  3. Cox OS ~ S  (Model A).
  4. Cox OS ~ S + total_library_size  (Model B, depth-adjustment prong).
     total_library_size = sum of raw counts inferred from log2(x+1) → 2^(x)-1 → sum → log2.
     For Xena HiSeqV2 (RSEM normalized log2(x+1)), sum-of-2^x is a normalized-library-size
     proxy; disclose as a normalized-depth covariate not a raw-count depth.
  5. Venet null: draw B random gene sets of same size, compute Cox HR (Model A) for each,
     report the real signature's percentile in the null distribution.
  6. Write JSON result.

RULE 2 provenance: signature files are the frozen JSON loaded by path; cohort files were
fetched from UCSC Xena hub in the same session (see fetch trail in scoring-results.md).
"""
import argparse, json, sys, os, warnings, hashlib
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

from lifelines import CoxPHFitter

DATA_DIR = "/Users/damir/damir-research-vault/06-code/tautology-diagnostic/track1/data"
RES_DIR = "/Users/damir/damir-research-vault/06-code/tautology-diagnostic/track1/results"

def load_xena_expr(path_gz):
    df = pd.read_csv(path_gz, sep="\t", index_col=0, compression="gzip")
    df.index.name = "gene"
    return df  # rows=genes, cols=samples; values = log2(x+1)

def load_survival(path_tsv):
    df = pd.read_csv(path_tsv, sep="\t")
    return df

def z_score_rows(expr):
    # z-score each gene across samples
    mu = expr.mean(axis=1)
    sd = expr.std(axis=1).replace(0, np.nan)
    return expr.subtract(mu, axis=0).divide(sd, axis=0)

def sig_score(zexpr, sig_genes):
    # per-sample mean of z-scored expression of signature genes
    present = [g for g in sig_genes if g in zexpr.index]
    n_present = len(present); n_total = len(set(sig_genes))
    if not present:
        return None, {"n_present":0, "n_total":n_total}
    S = zexpr.loc[present].mean(axis=0)
    return S, {"n_present":n_present, "n_total":n_total, "present_genes":present}

def library_size_proxy(expr):
    # expr = log2(x+1). Sum of un-logged expression per sample, then log2.
    x = np.power(2, expr) - 1
    lib = x.sum(axis=0)
    return np.log2(lib + 1)

def fit_cox(df_surv, extra_col=None):
    if extra_col is None:
        cols = ["OS.time","OS","S"]
    else:
        cols = ["OS.time","OS","S", extra_col]
    d = df_surv[cols].dropna()
    if len(d) < 20 or d["OS"].sum() < 5:
        return None
    cph = CoxPHFitter()
    cph.fit(d, duration_col="OS.time", event_col="OS")
    hr = float(np.exp(cph.params_["S"]))
    ci_low, ci_high = [float(x) for x in np.exp(cph.confidence_intervals_.loc["S"].values)]
    coef = float(cph.params_["S"])
    p = float(cph.summary.loc["S","p"])
    return {"n":int(len(d)), "events":int(d["OS"].sum()), "coef_S":coef,
            "HR_S":hr, "CI95":[ci_low, ci_high], "p_S":p}

def venet_null(zexpr, df_surv, sig_size, B, seed):
    """Venet null with library-size covariate in EVERY Cox fit (AMENDMENT 2b).

    Each draw fits OS ~ S_random + lib_log2 so the reported coef is the random signature's
    Cox coefficient CONDITIONED on log total library size. This centers the null near 0 and
    makes the null magnitude comparison consistent with the depth-adjustment prong (Model B),
    which also conditions on lib_log2.
    """
    rng = np.random.default_rng(seed)
    gene_pool = list(zexpr.index)
    # Reuse OS.time/OS/lib_log2 in df_surv; overwrite S column per draw
    coefs = []
    for b in range(B):
        picks = rng.choice(len(gene_pool), size=sig_size, replace=False)
        genes = [gene_pool[i] for i in picks]
        S = zexpr.loc[genes].mean(axis=0)
        tmp = df_surv.copy()
        tmp["S"] = S.reindex(tmp["sample"]).values
        d = tmp[["OS.time","OS","S","lib_log2"]].dropna()
        if len(d) < 20 or d["OS"].sum() < 5:
            coefs.append(np.nan); continue
        try:
            cph = CoxPHFitter()
            cph.fit(d, duration_col="OS.time", event_col="OS")
            coefs.append(float(cph.params_["S"]))
        except Exception:
            coefs.append(np.nan)
    return np.array(coefs)

def score_row(row_id, expr_path, surv_path, sig_path, cohort_name, out_json, B=1000, seed=42):
    print(f"[L2 {row_id}] cohort={cohort_name}")
    expr = load_xena_expr(expr_path)
    print(f"  expr: {expr.shape[0]} genes x {expr.shape[1]} samples")
    surv = load_survival(surv_path)
    # Restrict to tumor samples (any -01, -02, -03, -05, -06 suffix; drop -11 normal)
    surv = surv[~surv["sample"].str.endswith("-11")]
    # Match samples
    shared = list(set(expr.columns) & set(surv["sample"]))
    print(f"  shared samples (primary tumor, OS): {len(shared)}")
    expr = expr[shared]
    zexpr = z_score_rows(expr)
    sig = json.load(open(sig_path))
    print(f"  signature: {sig['n_unique_symbols']} genes from {sig['source']}")
    S, sinfo = sig_score(zexpr, sig["genes"])
    print(f"  signature genes present in expression: {sinfo['n_present']}/{sinfo['n_total']}")
    # Build survival dataframe with S and library-size
    lib = library_size_proxy(expr)
    df = surv[["sample","OS","OS.time"]].copy()
    df = df.set_index("sample").reindex(shared).reset_index().rename(columns={"index":"sample"})
    df["S"] = S.values
    df["lib_log2"] = lib.reindex(shared).values
    # Model A: OS ~ S
    modelA = fit_cox(df)
    # Model B: OS ~ S + lib_log2
    modelB = fit_cox(df, extra_col="lib_log2")
    # Correlation between S and library size
    corr_S_lib = float(stats.spearmanr(df["S"].dropna(), df["lib_log2"].dropna()).statistic)
    # Venet null (AMENDMENT 2b): S+lib_log2 Cox for every random draw; compare to
    # real signature's S+lib_log2 Cox coefficient (modelB), so both real and null are
    # measured "beyond library size". This centres the null near 0 and makes the
    # Venet prong consistent with the depth-adjustment prong.
    null_coefs = venet_null(zexpr, df, sinfo["n_present"], B=B, seed=seed)
    n_valid = int(np.isfinite(null_coefs).sum())
    real_coef = modelB["coef_S"] if modelB else None  # coef beyond library size
    percentile = None; two_sided_p = None
    if real_coef is not None and n_valid > 0:
        finite = null_coefs[np.isfinite(null_coefs)]
        # percentile of real coefficient (signed)
        percentile = float((finite <= real_coef).mean())
        # two-sided empirical p (|null|>=|real|) — beats-Venet if this is < 0.05
        two_sided_p = float((np.abs(finite) >= abs(real_coef)).mean())
    result = {
        "row": row_id,
        "cohort": cohort_name,
        "signature_source": sig["source"],
        "signature_fetch_url": sig["fetch_url"],
        "signature_fetch_date": sig["fetch_date"],
        "signature_n_unique_symbols": sig["n_unique_symbols"],
        "signature_n_present_in_expression": sinfo["n_present"],
        "expr_source_url": os.path.basename(expr_path),
        "expr_shape_samples": int(expr.shape[1]),
        "expr_shape_genes": int(expr.shape[0]),
        "surv_source_url": os.path.basename(surv_path),
        "surv_samples_matched": len(shared),
        "model_A_S_only": modelA,
        "model_B_S_plus_lib": modelB,
        "spearman_S_lib": corr_S_lib,
        "venet_null": {
            "spec": "AMENDMENT 2b: every Cox fit (real + all B random) conditions on lib_log2",
            "B_requested": B,
            "B_valid": n_valid,
            "seed": seed,
            "real_coef_S_adjusted_for_lib": real_coef,
            "null_coef_percentile_of_real": percentile,
            "empirical_two_sided_p_vs_null": two_sided_p,
            "null_coef_mean": float(np.nanmean(null_coefs)),
            "null_coef_std": float(np.nanstd(null_coefs)),
        }
    }
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
    return result

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--row", required=True)
    p.add_argument("--expr", required=True)
    p.add_argument("--surv", required=True)
    p.add_argument("--sig", required=True)
    p.add_argument("--cohort", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    r = score_row(args.row, args.expr, args.surv, args.sig, args.cohort, args.out, args.B, args.seed)
    print(json.dumps({k:v for k,v in r.items() if not isinstance(v, dict)}, indent=2))
