"""
L2 EXPANSION scoring pipeline — ssGSEA + Cox + library-size-adjusted Venet null.

PI convention (locked 2026-07-20):
  Signature = the gene list the paper APPLIES to the bulk cohort (published supp table).
  Parse verbatim, freeze, no reselection.
  Score = ssGSEA enrichment of the signature per bulk sample (uniform across all rows +
          all Venet draws).
  Cox: Model A (OS ~ score); Model B (OS ~ score + log total library size).
  Depth prong: does score survive Model B at p < 0.05?
  Venet: B=1000 random gene sets of the SAME size, each ssGSEA-scored, each fit Cox
          WITH library-size covariate (per AMENDMENT 2b); seed=42.
  Venet prong: real |coef| below 95th percentile of null → FAILS (CLINICAL-ARTIFACT).
  Verdict per §5b: CLINICAL-ARTIFACT (either prong fails) / CLINICAL-SURVIVES (both pass) /
                    INCONCLUSIVE.

ssGSEA implementation follows Barbie et al. 2009 (weighted running-sum with alpha=0.25).
"""
import argparse, json, os, sys, warnings, time
import numpy as np
import pandas as pd
from scipy import stats
from lifelines import CoxPHFitter

warnings.filterwarnings("ignore")


def load_xena_expr(path_gz):
    df = pd.read_csv(path_gz, sep="\t", index_col=0, compression="gzip")
    df.index.name = "gene"
    return df  # rows=genes, cols=samples; values = log2(x+1)


def load_survival(path_tsv):
    return pd.read_csv(path_tsv, sep="\t")


def library_size_proxy(expr):
    x = np.power(2, expr) - 1
    lib = x.sum(axis=0)
    return np.log2(lib + 1)


def precompute_ssgsea_ranks(expr, alpha=0.25):
    """Precompute per-sample ordering + weights for fast ssGSEA."""
    exp_val = expr.values.astype(np.float64)  # (n_g, n_s)
    # sort descending per column
    order = np.argsort(-exp_val, axis=0)  # (n_g, n_s)
    # sorted expression per column
    sorted_expr = np.take_along_axis(exp_val, order, axis=0)  # (n_g, n_s)
    # weights (|x|^alpha) per rank position
    weights = np.power(np.abs(sorted_expr), alpha)  # (n_g, n_s)
    return order, weights


def ssgsea_scores(order, weights, sig_mask):
    """Compute ssGSEA ES per sample for one signature.

    order[i, j]     = index of gene at rank i in sample j (0 = top-ranked)
    weights[i, j]   = |sorted_expr|^alpha at rank i in sample j
    sig_mask[g]     = True if gene g is in the signature (indexed over all genes)

    Returns: es (n_s,) — Barbie ssGSEA enrichment per sample.
    """
    n_g, n_s = order.shape
    n_sig = int(sig_mask.sum())
    if n_sig == 0 or n_sig == n_g:
        return np.full(n_s, np.nan)
    n_bg = n_g - n_sig

    # sorted_sig[i, j] = sig_mask[order[i, j]]
    sorted_sig = sig_mask[order]  # (n_g, n_s), bool

    hit_w = sorted_sig * weights  # zeros where not in sig
    norm_hit = hit_w.sum(axis=0)  # (n_s,)

    valid = norm_hit > 0
    norm_hit_safe = np.where(valid, norm_hit, 1.0)

    cum_hit = np.cumsum(hit_w, axis=0) / norm_hit_safe[None, :]  # (n_g, n_s)
    cum_miss = np.cumsum((~sorted_sig).astype(np.float64), axis=0) / n_bg  # (n_g, n_s)

    es = (cum_hit - cum_miss).sum(axis=0)  # (n_s,) — Barbie ssGSEA (integral of running diff)
    es[~valid] = np.nan
    return es


def fit_cox(df, cols):
    d = df[cols].dropna()
    if len(d) < 20 or d["OS"].sum() < 5:
        return None
    cph = CoxPHFitter()
    cph.fit(d, duration_col="OS.time", event_col="OS")
    hr = float(np.exp(cph.params_["S"]))
    ci_low, ci_high = [float(x) for x in np.exp(cph.confidence_intervals_.loc["S"].values)]
    return {
        "n": int(len(d)),
        "events": int(d["OS"].sum()),
        "coef_S": float(cph.params_["S"]),
        "HR_S": hr,
        "CI95": [ci_low, ci_high],
        "p_S": float(cph.summary.loc["S", "p"]),
    }


def venet_null(order, weights, gene_index, df_surv, sig_size, B, seed):
    """Venet null: random signatures of the SAME size, each ssGSEA-scored,
    each fit Cox WITH lib_log2 covariate (per AMENDMENT 2b)."""
    rng = np.random.default_rng(seed)
    n_g = order.shape[0]
    coefs = np.full(B, np.nan)
    t0 = time.time()
    for b in range(B):
        picks = rng.choice(n_g, size=sig_size, replace=False)
        mask = np.zeros(n_g, dtype=bool)
        mask[picks] = True
        es = ssgsea_scores(order, weights, mask)
        tmp = df_surv.copy()
        tmp["S"] = pd.Series(es, index=gene_index).reindex(tmp["sample"]).values
        # Actually: es indexed by sample position, need mapping
        # Simpler: assume df_surv["sample"] already in same order as expr columns → assign by index
        tmp["S"] = es  # es is indexed by sample position (0..n_s-1), matching gene_index order
        d = tmp[["OS.time", "OS", "S", "lib_log2"]].dropna()
        if len(d) < 20 or d["OS"].sum() < 5:
            continue
        try:
            cph = CoxPHFitter()
            cph.fit(d, duration_col="OS.time", event_col="OS")
            coefs[b] = float(cph.params_["S"])
        except Exception:
            pass
        if b > 0 and b % 100 == 0:
            print(f"    venet: {b}/{B} ({time.time()-t0:.1f}s)")
    return coefs


def score_row(row_id, expr_path, surv_path, sig_path, cohort_name, endpoint_note,
              out_json, B=1000, seed=42, alpha=0.25):
    print(f"[L2 {row_id}] cohort={cohort_name}")
    expr = load_xena_expr(expr_path)
    print(f"  expr: {expr.shape[0]} genes x {expr.shape[1]} samples")
    surv = load_survival(surv_path)
    surv = surv[~surv["sample"].str.endswith("-11")]  # drop normals
    shared = sorted(set(expr.columns) & set(surv["sample"]))
    expr = expr[shared]
    print(f"  shared samples: {len(shared)}")

    sig = json.load(open(sig_path))
    sig_genes = sig["genes"]
    sig_set = set(sig_genes)
    gene_idx = list(expr.index)
    sig_mask = np.array([g in sig_set for g in gene_idx])
    n_present = int(sig_mask.sum())
    print(f"  signature n present/total = {n_present}/{sig['n_unique_symbols']}")

    # Precompute per-sample ranks + weights
    t0 = time.time()
    print(f"  precomputing ssGSEA ranks/weights...")
    order, weights = precompute_ssgsea_ranks(expr, alpha=alpha)
    print(f"    done in {time.time()-t0:.1f}s")

    # Real ssGSEA scores
    S_real = ssgsea_scores(order, weights, sig_mask)
    lib = library_size_proxy(expr).reindex(shared).values

    df = pd.DataFrame({
        "sample": shared,
        "S": S_real,
        "lib_log2": lib,
    })
    surv_by_sample = surv.set_index("sample")
    df["OS"] = df["sample"].map(surv_by_sample["OS"])
    df["OS.time"] = df["sample"].map(surv_by_sample["OS.time"])

    modelA = fit_cox(df, ["OS.time", "OS", "S"])
    modelB = fit_cox(df, ["OS.time", "OS", "S", "lib_log2"])
    corr_S_lib = float(stats.spearmanr(df["S"].dropna(), df["lib_log2"].dropna()).statistic)

    # Venet null with library-size covariate
    print(f"  venet: B={B} random {n_present}-gene signatures, seed={seed}")
    t0 = time.time()
    null_coefs = venet_null(order, weights, shared, df, n_present, B=B, seed=seed)
    print(f"    venet done in {time.time()-t0:.1f}s")

    n_valid = int(np.isfinite(null_coefs).sum())
    real_coef = modelB["coef_S"] if modelB else None
    percentile = None
    two_sided_p = None
    q95_abs_null = None
    if real_coef is not None and n_valid > 0:
        finite = null_coefs[np.isfinite(null_coefs)]
        percentile = float((finite <= real_coef).mean())
        two_sided_p = float((np.abs(finite) >= abs(real_coef)).mean())
        q95_abs_null = float(np.quantile(np.abs(finite), 0.95))

    fails_adj = (modelB is None) or (modelB["p_S"] > 0.05)
    beats_null_2sided = (two_sided_p is not None) and (two_sided_p < 0.05)
    beats_null_magnitude_95 = (real_coef is not None and q95_abs_null is not None) and (abs(real_coef) >= q95_abs_null)
    # §5b: real |coef| below 95th percentile of null → FAILS
    fails_venet = not beats_null_magnitude_95

    if fails_adj or fails_venet:
        verdict = "CLINICAL-ARTIFACT"
    elif not fails_adj and not fails_venet:
        verdict = "CLINICAL-SURVIVES"
    else:
        verdict = "INCONCLUSIVE"

    # Borderline flag if either prong is near threshold
    borderline_flags = []
    if modelB and abs(modelB["p_S"] - 0.05) < 0.02:
        borderline_flags.append(f"Model B p_S={modelB['p_S']:.3f} near 0.05")
    if q95_abs_null is not None and real_coef is not None and abs(abs(real_coef) - q95_abs_null) / max(q95_abs_null, 1e-6) < 0.10:
        borderline_flags.append(f"|real coef|={abs(real_coef):.3f} within 10% of null-|coef| 95th pct={q95_abs_null:.3f}")

    result = {
        "row": row_id,
        "cohort": cohort_name,
        "endpoint": "overall survival (uniform primary per PI convention)",
        "endpoint_paper_note": endpoint_note,
        "signature_source": sig["source"],
        "signature_fetch_url": sig["fetch_url"],
        "signature_fetch_date": sig["fetch_date"],
        "signature_n_unique_symbols": sig["n_unique_symbols"],
        "signature_n_present_in_expression": n_present,
        "signature_class": sig.get("signature_class", "unknown"),
        "expr_source_file": os.path.basename(expr_path),
        "expr_shape_samples": int(expr.shape[1]),
        "expr_shape_genes": int(expr.shape[0]),
        "surv_source_file": os.path.basename(surv_path),
        "surv_samples_matched": len(shared),
        "ssgsea_alpha": alpha,
        "model_A_S_only": modelA,
        "model_B_S_plus_lib": modelB,
        "spearman_S_lib": corr_S_lib,
        "venet_null": {
            "spec": "AMENDMENT 2b: every Cox fit (real + all B random) conditions on lib_log2; scoring = ssGSEA uniform across real + null",
            "B_requested": B,
            "B_valid": n_valid,
            "seed": seed,
            "real_coef_S_adjusted_for_lib": real_coef,
            "null_coef_percentile_of_real": percentile,
            "empirical_two_sided_p_vs_null": two_sided_p,
            "null_coef_mean": float(np.nanmean(null_coefs)),
            "null_coef_std": float(np.nanstd(null_coefs)),
            "null_abs_coef_q95": q95_abs_null,
        },
        "depth_prong_verdict": ("FAILS" if fails_adj else "PASSES"),
        "venet_prong_verdict": ("FAILS" if fails_venet else "PASSES"),
        "verdict": verdict,
        "borderline_flags": borderline_flags,
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
    p.add_argument("--endpoint-note", default="OS matches paper")
    p.add_argument("--out", required=True)
    p.add_argument("--B", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--alpha", type=float, default=0.25)
    args = p.parse_args()
    r = score_row(args.row, args.expr, args.surv, args.sig, args.cohort,
                  args.endpoint_note, args.out, args.B, args.seed, args.alpha)
    print(json.dumps({k: v for k, v in r.items() if not isinstance(v, dict)}, indent=2))
