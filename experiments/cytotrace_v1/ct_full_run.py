"""
ct_full_run.py — full CytoTRACE v1 own-baseline on W4 C1 + C2 atlases.

Follows the LOCKED pre-registration in the task mandate:
  - Same cells as W4 (byte-identical loaders + gene filter + ordinal via
    import from w4/scripts/w4_gate2_run.py)
  - Same weightedtau kernel (default settings)
  - Same bootstrap (1000 resamples, seed 42)
  - Same verdict bands (imported from w4_gate2_run.verdict_from_delta)
  - Additional Kang wdm weighted kernel (imported from atlas_run/wdm_tau.py)
    as a second-kernel sanity check per prereg

Score: cytotrace_v1_full() from cytotrace_full.py (full 4-step algorithm)
Primitive: raw gene count |supp(x)| (same as W4 CT primitive)

`--variant` selects the CT v1 similarity-matrix cleaning order:
  - "rfaithful" (default): R-faithful cleaning order (2026-07-19 fidelity
    fix). Writes to results/rfaithful/.
  - "rowstoch_pyorder": legacy 2026-07-19 first-run order. Writes to
    results/ (top-level, matches prior file layout).

Outputs (rfaithful, DEFAULT):
  cytotrace_v1/results/rfaithful/c1_gse117498_rfaithful.json
  cytotrace_v1/results/rfaithful/c2_gse125970_rfaithful.json
  cytotrace_v1/results/rfaithful/summary.json

Outputs (rowstoch_pyorder, LEGACY):
  cytotrace_v1/results/c1_gse117498_ctfull.json
  cytotrace_v1/results/c2_gse125970_ctfull.json
  cytotrace_v1/results/summary.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import anndata as ad
from scipy.stats import weightedtau, rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent  # tautology-diagnostic/
W4_SCRIPTS = PKG_ROOT / "w4" / "scripts"
W4_DATA = PKG_ROOT / "w4" / "data"
W4_RESULTS = PKG_ROOT / "w4" / "results"
ATLAS_RUN = PKG_ROOT / "atlas_run"

# Add paths so we can reuse W4 loaders and Kang wdm
sys.path.insert(0, str(PKG_ROOT))
sys.path.insert(0, str(W4_SCRIPTS))
sys.path.insert(0, str(ATLAS_RUN))
sys.path.insert(0, str(HERE))

from w4_gate2_run import (  # noqa: E402
    load_c1, load_c2,
    C1_POP_TO_RANK, C2_LABEL_TO_RANK,
    preprocess_and_filter, per_pop_depth,
    bootstrap_ci, verdict_from_delta,
    C1_GENE_FILTER_MIN_CELLS, C2_GENE_FILTER_MIN_CELLS,
    SEED, N_BOOT,
    DELTA_TAUTOLOG, DELTA_SCORE_BEATS, DELTA_SIGN_FLIP,
)
from wdm_tau import wdm_tau_b_bit  # noqa: E402
from cytotrace_full import cytotrace_v1_full  # noqa: E402


# ---------------------------------------------------------------------------
#  Kernel wrappers
# ---------------------------------------------------------------------------
def _wtau_scipy(x, y):
    """scipy weighted Kendall tau — byte-identical to W4 kernel."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    r = weightedtau(x[mask], y[mask])
    return float(getattr(r, "statistic", None) or r.correlation)


def _wtau_kang(x, y):
    """
    Kang wdm weighted Kendall tau-b, unit weights.

    Uses the same wdm_tau_b_bit implementation the atlas_run uses to reproduce
    Kang 2025 Table S13. We pass unit weights (all 1's) since this run does
    not have Kang's std_phenotype nested-weighting structure. The result is
    then a plain wdm-style tau-b, differing from scipy weightedtau in tie
    handling (wdm uses concordant/discordant/x-tie/y-tie counting; scipy uses
    hyperbolic-weight averaging over both lexicographic orderings).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    w = np.ones(int(mask.sum()), dtype=np.float64)
    return float(wdm_tau_b_bit(x[mask], y[mask], w))


def _rank_residual(score, primitive):
    """Standard rank-linear-residual (matches W4 helper)."""
    x = np.asarray(score, dtype=np.float64)
    y = np.asarray(primitive, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    rx = rankdata(x[mask])
    ry = rankdata(y[mask])
    cov = np.cov(rx, ry, ddof=0)[0, 1]
    var_y = np.var(ry)
    if var_y == 0:
        return None, mask
    beta = cov / var_y
    resid = rx - beta * ry
    return resid, mask


def _auroc_topbot(score, rank_arr, top_val, bot_vals):
    y = np.full(len(rank_arr), np.nan)
    y[rank_arr == top_val] = 1
    for b in bot_vals:
        y[rank_arr == b] = 0
    m = np.isfinite(y) & np.isfinite(score)
    if m.sum() < 3 or len(np.unique(y[m])) < 2:
        return float("nan")
    return float(roc_auc_score(y[m].astype(int), score[m]))


def own_baseline_two_kernels(score, primitive, rank_arr, top_val, bot_vals,
                             n_boot=N_BOOT, seed=SEED):
    """
    Full own-baseline stats under BOTH the scipy weightedtau kernel
    and the Kang wdm kernel. Uses the same seed for both.

    Returns dict with per-kernel skill_score, skill_primitive, marginal_delta,
    conditional_skill, and their CIs; also AUROC top-vs-bot (kernel-independent).
    """
    m = np.isfinite(rank_arr) & np.isfinite(score) & np.isfinite(primitive)
    s = score[m]; p = primitive[m]; r = rank_arr[m]
    n = int(m.sum())

    out = {"n_ranked_cells": n}

    for kernel_name, kernel_fn in [("scipy", _wtau_scipy), ("kang", _wtau_kang)]:
        skill_score = kernel_fn(s, r)
        skill_prim = kernel_fn(p, r)
        delta = skill_score - skill_prim

        resid, _ = _rank_residual(s, p)
        cond_skill = kernel_fn(resid, r) if resid is not None else float("nan")

        # Bootstrap CIs (same seed as W4 for scipy path)
        ci_score = bootstrap_ci(kernel_fn, (s, r), n_boot=n_boot, seed=seed)
        ci_prim = bootstrap_ci(kernel_fn, (p, r), n_boot=n_boot, seed=seed + 1)
        ci_delta = bootstrap_ci(
            lambda a, b, c: kernel_fn(a, c) - kernel_fn(b, c),
            (s, p, r), n_boot=n_boot, seed=seed + 2,
        )
        def _cond_boot(a, b, c, _kf=kernel_fn):
            resid_b, _mm = _rank_residual(a, b)
            return _kf(resid_b, c) if resid_b is not None else float("nan")
        ci_cond = bootstrap_ci(_cond_boot, (s, p, r), n_boot=n_boot, seed=seed + 3)

        out[kernel_name] = {
            "skill_score": skill_score,
            "skill_score_CI95": [ci_score["lo"], ci_score["hi"]],
            "skill_primitive": skill_prim,
            "skill_primitive_CI95": [ci_prim["lo"], ci_prim["hi"]],
            "marginal_delta": delta,
            "marginal_delta_CI95": [ci_delta["lo"], ci_delta["hi"]],
            "conditional_skill": cond_skill,
            "conditional_skill_CI95": [ci_cond["lo"], ci_cond["hi"]],
        }

    # AUROC top-vs-bot (kernel-independent)
    out["AUROC_score_top_vs_bot"] = _auroc_topbot(s, r, top_val, bot_vals)
    out["AUROC_primitive_top_vs_bot"] = _auroc_topbot(p, r, top_val, bot_vals)

    # Verdicts per kernel using W4's locked verdict_from_delta
    for kernel_name in ["scipy", "kang"]:
        d = out[kernel_name]["marginal_delta"]
        cd_lo = out[kernel_name]["conditional_skill_CI95"][0]
        cd_hi = out[kernel_name]["conditional_skill_CI95"][1]
        out[kernel_name]["verdict"] = verdict_from_delta(d, cd_lo, cd_hi)

    return out


# ---------------------------------------------------------------------------
#  Atlas orchestration
# ---------------------------------------------------------------------------
def run_atlas(atlas_name, X, gene_list, cell_ids, labels, ranks, min_cells,
              out_path, variant="rfaithful", verbose_ct=True):
    t0 = time.time()
    print(f"\n=== {atlas_name} ===")
    print(f"  cells (loaded): {X.shape[0]:,}")
    print(f"  genes (loaded): {X.shape[1]:,}")

    unique_labels = list(dict.fromkeys(labels.tolist()))
    depth = per_pop_depth(X, labels, unique_labels)

    print(f"  applying gene filter (min_cells={min_cells}) ...")
    Xf, gene_list_f = preprocess_and_filter(X, gene_list, min_cells)
    print(f"  post-filter genes: {len(gene_list_f):,}")

    # per-tier n
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

    # Build AnnData shim for cytotrace_v1_full
    adata = ad.AnnData(X=Xf)
    adata.var_names = [str(g) for g in gene_list_f]
    adata.obs_names = [f"c{i}" for i in range(Xf.shape[0])]

    print(f"  computing full CytoTRACE v1 (variant={variant!r}, "
          "4-step: GC + GCS + NNLS + diffusion) ...")
    ct = cytotrace_v1_full(adata, variant=variant, verbose=verbose_ct)
    ct_score = ct["ct_v1"]
    ct_primitive = ct["gene_counts"]
    print(f"  diffusion iterations: {ct['diffusion_iters']}")

    finite_ranks = ranks[np.isfinite(ranks)]
    top_val = int(np.max(finite_ranks))
    bot_val = int(np.min(finite_ranks))
    bot_vals = [bot_val]

    print("  own-baseline (two-kernel): full CT v1 vs raw gene count ...")
    stats = own_baseline_two_kernels(
        score=ct_score, primitive=ct_primitive,
        rank_arr=ranks, top_val=top_val, bot_vals=bot_vals,
    )

    # Load GCS-proxy numbers from W4 results for side-by-side comparison
    proxy_json_path = W4_RESULTS / (
        "c1_gse117498_results.json" if "C1" in atlas_name else "c2_gse125970_results.json"
    )
    proxy_stats = None
    if proxy_json_path.exists():
        with open(proxy_json_path) as f:
            w4res = json.load(f)
        pct = w4res["per_score"]["CT"]
        proxy_stats = {
            "skill_score": pct["skill_score"],
            "skill_score_CI95": pct["skill_score_CI95"],
            "skill_primitive": pct["skill_primitive"],
            "skill_primitive_CI95": pct["skill_primitive_CI95"],
            "marginal_delta": pct["marginal_delta"],
            "marginal_delta_CI95": pct["marginal_delta_CI95"],
            "conditional_skill": pct["conditional_skill"],
            "conditional_skill_CI95": pct["conditional_skill_CI95"],
            "AUROC_score_top_vs_bot": pct["AUROC_score_top_vs_bot"],
            "AUROC_primitive_top_vs_bot": pct["AUROC_primitive_top_vs_bot"],
            "verdict": pct["verdict"],
            "source": str(proxy_json_path),
        }

    result = {
        "atlas": atlas_name,
        "score": "full CytoTRACE v1 (GC + GCS + NNLS + diffusion, per Gulati 2020)",
        "primitive": "|supp(x)| = raw gene count",
        "n_cells_total": int(X.shape[0]),
        "n_cells_ranked": int(np.sum(np.isfinite(ranks))),
        "n_cells_unranked": int(np.sum(~np.isfinite(ranks))),
        "n_genes_loaded": int(X.shape[1]),
        "n_genes_after_filter": int(Xf.shape[1]),
        "gene_filter_min_cells": min_cells,
        "ct_full_params": {
            "variant": variant,
            "mvg_top": 1000,
            "gcs_top": 200,
            "alpha": 0.9,
            "max_iter": 10000,
            "tol": 1e-6,
            "diffusion_iters_used": ct["diffusion_iters"],
        },
        "kernel_notes": {
            "scipy": "scipy.stats.weightedtau default (hyperbolic weigher, lexicographic averaging)",
            "kang": "wdm-style tau-b via wdm_tau_b_bit (unit weights); matches Kang 2025 protocol",
        },
        "per_tier_n": tier_ns,
        "per_population_depth": depth,
        "full_ct_v1": stats,
        "gcs_proxy_reference (from w4 results)": proxy_stats,
        "top_tier": top_val,
        "bottom_tier": bot_val,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    out_path.parent.mkdir(exist_ok=True, parents=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  written {out_path}")

    # Concise verdict summary
    print(f"\n  === {atlas_name} full CT v1 verdict ===")
    for k in ["scipy", "kang"]:
        st = stats[k]
        d = st["marginal_delta"]
        cs = st["conditional_skill"]
        cs_lo, cs_hi = st["conditional_skill_CI95"]
        print(f"    [{k}] skill_score={st['skill_score']:+.4f} "
              f"skill_prim={st['skill_primitive']:+.4f} "
              f"Δ={d:+.4f} "
              f"cond={cs:+.4f} [{cs_lo:+.4f}, {cs_hi:+.4f}] "
              f"verdict={st['verdict']}")
    print(f"    AUROC(score, top-vs-bot)={stats['AUROC_score_top_vs_bot']:.4f} "
          f"AUROC(prim, top-vs-bot)={stats['AUROC_primitive_top_vs_bot']:.4f}")

    if proxy_stats:
        print(f"\n  === GCS-proxy comparison (from w4/results) ===")
        print(f"    proxy Δ = {proxy_stats['marginal_delta']:+.4f} | "
              f"full CT v1 Δ (scipy) = {stats['scipy']['marginal_delta']:+.4f}")
        print(f"    proxy cond = {proxy_stats['conditional_skill']:+.4f} | "
              f"full CT v1 cond (scipy) = {stats['scipy']['conditional_skill']:+.4f}")

    return result


# ---------------------------------------------------------------------------
#  Sanity check on C2 (see mandate Task 3)
# ---------------------------------------------------------------------------
def sanity_check_c2(X, gene_list, cell_ids, labels, ranks, min_cells,
                    variant="rfaithful"):
    """
    Coarse sanity check on C2 intestine: full CT v1 should put Stem cells
    higher than differentiated cells (AUROC(ct, Stem-vs-differentiated) > 0.5).
    """
    print(f"\n=== SANITY CHECK on C2 (intestine, variant={variant!r}): "
          "Stem-cells-higher ===")
    Xf, gene_list_f = preprocess_and_filter(X, gene_list, min_cells)
    adata = ad.AnnData(X=Xf)
    adata.var_names = [str(g) for g in gene_list_f]
    adata.obs_names = [f"c{i}" for i in range(Xf.shape[0])]
    ct = cytotrace_v1_full(adata, variant=variant, verbose=False)
    ct_score = ct["ct_v1"]

    top_val = 4  # Stem cells
    bot_val = 1  # differentiated (Enterocyte + Goblet + ...)
    auc = _auroc_topbot(ct_score, ranks, top_val, [bot_val])
    print(f"  AUROC(ct_v1, Stem-vs-differentiated) = {auc:.4f}")
    if auc > 0.5:
        print("  PASS sanity: full CT v1 puts Stem cells higher.")
    else:
        print("  FAIL sanity: full CT v1 does NOT put Stem cells higher — STOP.")
    return auc


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-only", action="store_true")
    parser.add_argument("--c2-only", action="store_true")
    parser.add_argument("--sanity-only", action="store_true",
                        help="Only run C2 sanity check, no full pipeline")
    parser.add_argument("--variant", choices=["rfaithful", "rowstoch_pyorder"],
                        default="rfaithful",
                        help="CT v1 similarity-cleaning variant. "
                             "'rfaithful' = R-faithful order (2026-07-19 fix, default). "
                             "'rowstoch_pyorder' = legacy first-run order.")
    args = parser.parse_args()

    if args.variant == "rfaithful":
        out_dir = HERE / "results" / "rfaithful"
        c1_out_name = "c1_gse117498_rfaithful.json"
        c2_out_name = "c2_gse125970_rfaithful.json"
    else:
        out_dir = HERE / "results"
        c1_out_name = "c1_gse117498_ctfull.json"
        c2_out_name = "c2_gse125970_ctfull.json"
    out_dir.mkdir(exist_ok=True, parents=True)

    summary = {"variant": args.variant}

    # SANITY CHECK first (per mandate Task 3): C2 biology-direction check
    print("Loading C2 for sanity check ...")
    X_c2, genes_c2, cell_ids_c2, labels_c2, ranks_c2 = load_c2(W4_DATA / "c2_gse125970")
    sanity_auc = sanity_check_c2(X_c2, genes_c2, cell_ids_c2, labels_c2, ranks_c2,
                                  min_cells=C2_GENE_FILTER_MIN_CELLS,
                                  variant=args.variant)
    if sanity_auc <= 0.5:
        print("SANITY CHECK FAILED. Aborting main run per mandate Task 3.")
        sys.exit(1)

    if args.sanity_only:
        print(f"\nSanity-only mode (variant={args.variant!r}): "
              f"AUROC(ct_v1, Stem-vs-diff) = {sanity_auc:.4f}. Done.")
        return

    if not args.c2_only:
        X, genes, cell_ids, labels, ranks = load_c1(W4_DATA / "c1_gse117498")
        res_c1 = run_atlas(
            "C1_GSE117498",
            X, genes, cell_ids, labels, ranks,
            min_cells=C1_GENE_FILTER_MIN_CELLS,
            out_path=out_dir / c1_out_name,
            variant=args.variant,
        )
        summary["C1_GSE117498"] = {
            "full_ct_v1_scipy": {
                "skill_score": res_c1["full_ct_v1"]["scipy"]["skill_score"],
                "skill_primitive": res_c1["full_ct_v1"]["scipy"]["skill_primitive"],
                "marginal_delta": res_c1["full_ct_v1"]["scipy"]["marginal_delta"],
                "marginal_delta_CI95": res_c1["full_ct_v1"]["scipy"]["marginal_delta_CI95"],
                "conditional_skill": res_c1["full_ct_v1"]["scipy"]["conditional_skill"],
                "conditional_skill_CI95": res_c1["full_ct_v1"]["scipy"]["conditional_skill_CI95"],
                "verdict": res_c1["full_ct_v1"]["scipy"]["verdict"],
            },
            "full_ct_v1_kang": {
                "skill_score": res_c1["full_ct_v1"]["kang"]["skill_score"],
                "skill_primitive": res_c1["full_ct_v1"]["kang"]["skill_primitive"],
                "marginal_delta": res_c1["full_ct_v1"]["kang"]["marginal_delta"],
                "marginal_delta_CI95": res_c1["full_ct_v1"]["kang"]["marginal_delta_CI95"],
                "conditional_skill": res_c1["full_ct_v1"]["kang"]["conditional_skill"],
                "conditional_skill_CI95": res_c1["full_ct_v1"]["kang"]["conditional_skill_CI95"],
                "verdict": res_c1["full_ct_v1"]["kang"]["verdict"],
            },
            "AUROC_score_top_vs_bot": res_c1["full_ct_v1"]["AUROC_score_top_vs_bot"],
            "AUROC_primitive_top_vs_bot": res_c1["full_ct_v1"]["AUROC_primitive_top_vs_bot"],
            "gcs_proxy_reference": res_c1["gcs_proxy_reference (from w4 results)"],
        }

    if not args.c1_only:
        # Reuse the already-loaded C2 data
        res_c2 = run_atlas(
            "C2_GSE125970",
            X_c2, genes_c2, cell_ids_c2, labels_c2, ranks_c2,
            min_cells=C2_GENE_FILTER_MIN_CELLS,
            out_path=out_dir / c2_out_name,
            variant=args.variant,
        )
        summary["C2_GSE125970"] = {
            "full_ct_v1_scipy": {
                "skill_score": res_c2["full_ct_v1"]["scipy"]["skill_score"],
                "skill_primitive": res_c2["full_ct_v1"]["scipy"]["skill_primitive"],
                "marginal_delta": res_c2["full_ct_v1"]["scipy"]["marginal_delta"],
                "marginal_delta_CI95": res_c2["full_ct_v1"]["scipy"]["marginal_delta_CI95"],
                "conditional_skill": res_c2["full_ct_v1"]["scipy"]["conditional_skill"],
                "conditional_skill_CI95": res_c2["full_ct_v1"]["scipy"]["conditional_skill_CI95"],
                "verdict": res_c2["full_ct_v1"]["scipy"]["verdict"],
            },
            "full_ct_v1_kang": {
                "skill_score": res_c2["full_ct_v1"]["kang"]["skill_score"],
                "skill_primitive": res_c2["full_ct_v1"]["kang"]["skill_primitive"],
                "marginal_delta": res_c2["full_ct_v1"]["kang"]["marginal_delta"],
                "marginal_delta_CI95": res_c2["full_ct_v1"]["kang"]["marginal_delta_CI95"],
                "conditional_skill": res_c2["full_ct_v1"]["kang"]["conditional_skill"],
                "conditional_skill_CI95": res_c2["full_ct_v1"]["kang"]["conditional_skill_CI95"],
                "verdict": res_c2["full_ct_v1"]["kang"]["verdict"],
            },
            "AUROC_score_top_vs_bot": res_c2["full_ct_v1"]["AUROC_score_top_vs_bot"],
            "AUROC_primitive_top_vs_bot": res_c2["full_ct_v1"]["AUROC_primitive_top_vs_bot"],
            "gcs_proxy_reference": res_c2["gcs_proxy_reference (from w4 results)"],
        }

    summary["sanity_check_c2_AUROC_stem_vs_diff"] = sanity_auc
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSUMMARY written: {out_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
