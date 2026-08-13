"""
stemsc_run.py — GATE 2 own-baseline analysis for StemSC on the two locked
W4 atlases (C1 GSE117498 human hematopoietic progenitors; C2 GSE125970
human intestinal epithelium).

Follows the LOCKED prereg
    04-experiments/2026-07-18-exploration-prereg-stemsc-own-baseline.md
byte-identically. Mirrors the W4 GATE 2 protocol in
    06-code/tautology-diagnostic/w4/scripts/w4_gate2_run.py
with three substitutions:
    (1) score          : StemSC   (instead of CT / SR / CCAT)
    (2) primitive      : REO-primitive  (raw k / n_present)
    (3) no PPI scaffold: StemSC is scaffold-free by construction, so no
                        STRING v12 load and no per-atlas graph intersection.

Ordinals, per-tier tie handling, per-population depth reporting, `min_cells=10`
gene filter, `scipy.stats.weightedtau` kernel, 1000-sample bootstrap with
seed=42, and locked verdict bands are BYTE-IDENTICAL to W4 GATE 2.

Outputs:
    stemsc/results/c1_gse117498_stemsc.json
    stemsc/results/c2_gse125970_stemsc.json
    stemsc/results/summary.json
"""
from __future__ import annotations
import argparse
import gzip
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import weightedtau, rankdata
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
PKG_ROOT = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(PKG_ROOT))
from own_baseline.paths import data_root  # noqa: E402
W4_SCRIPTS = PKG_ROOT / "experiments" / "w4" / "scripts"
W4_DATA = data_root() / "w4" / "data"
STEMSC_DIR = HERE

# Re-use the C1/C2 loaders from the W4 script byte-identically. Adding the
# W4 scripts dir to sys.path lets us import w4_gate2_run.load_c1 and load_c2
# without touching the W4 source.
sys.path.insert(0, str(W4_SCRIPTS))
sys.path.insert(0, str(STEMSC_DIR))
sys.path.append(str(PKG_ROOT / "own_baseline"))

from w4_gate2_run import (  # noqa: E402
    load_c1, load_c2,
    C1_POP_TO_RANK, C2_LABEL_TO_RANK,
    preprocess_and_filter, per_pop_depth,
    bootstrap_ci, verdict_from_delta,
    C1_GENE_FILTER_MIN_CELLS, C2_GENE_FILTER_MIN_CELLS,
    SEED, N_BOOT,
    DELTA_TAUTOLOG, DELTA_SCORE_BEATS, DELTA_SIGN_FLIP,
)
from stemsc import load_ref_pairs, stemsc_from_matrix  # noqa: E402


# =============================================================================
#  Symbol → Entrez mapping (needed because C1/C2 use HGNC symbols, pair set
#  uses Entrez IDs).
# =============================================================================
GENE_INFO = HERE / "mapping" / "Homo_sapiens.gene_info.gz"


def load_symbol_to_entrez():
    """Build the symbol → Entrez lookup from NCBI Homo_sapiens.gene_info.
    Primary Symbol column wins; Synonyms are used as fallback (first-writer)."""
    df = pd.read_csv(GENE_INFO, sep="\t", compression="gzip", dtype=str,
                     low_memory=False)
    sym2e = {}
    for _, row in df.iterrows():
        e = row["GeneID"]
        sym2e[row["Symbol"]] = e
        syns = row.get("Synonyms")
        if isinstance(syns, str) and syns != "-":
            for s in syns.split("|"):
                sym2e.setdefault(s, e)
    return sym2e


def map_gene_list_symbols_to_entrez(gene_symbols, sym2entrez):
    """
    For an ordered list of gene symbols (as in the atlas expression matrix
    columns), return a same-length list of Entrez IDs (as strings), with None
    for symbols that could not be mapped.

    Note: this is a per-column mapping, so it preserves the atlas's column
    order. StemSC needs Entrez IDs to match against the reference pair set,
    but does NOT need every column mapped — the R code (and our port) filters
    the pair set to those pairs whose BOTH genes are in the target's ID set,
    then computes k per cell over the surviving pairs.
    """
    return [sym2entrez.get(s) for s in gene_symbols]


# =============================================================================
#  Skill + own-baseline utilities (byte-identical to W4)
# =============================================================================
def _wtau(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return float("nan")
    r = weightedtau(x[mask], y[mask])
    return float(getattr(r, "statistic", None) or r.correlation)


def _rank_residual(score, primitive):
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


def own_baseline_stats(score, primitive, rank_arr, top_val, bot_vals,
                       n_boot=N_BOOT, seed=SEED):
    m = np.isfinite(rank_arr) & np.isfinite(score) & np.isfinite(primitive)
    s = score[m]; p = primitive[m]; r = rank_arr[m]
    n = int(m.sum())

    skill_score = _wtau(s, r)
    skill_prim = _wtau(p, r)
    delta = skill_score - skill_prim

    resid, _ = _rank_residual(s, p)
    cond_skill = _wtau(resid, r) if resid is not None else float("nan")

    aur_score = _auroc_topbot(s, r, top_val, bot_vals)
    aur_prim = _auroc_topbot(p, r, top_val, bot_vals)

    ci_score = bootstrap_ci(lambda a, b: _wtau(a, b), (s, r), n_boot=n_boot, seed=seed)
    ci_prim = bootstrap_ci(lambda a, b: _wtau(a, b), (p, r), n_boot=n_boot, seed=seed + 1)
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


# =============================================================================
#  Atlas orchestration
# =============================================================================
def run_atlas(atlas_name, X, gene_list_symbols, cell_ids, labels, ranks,
              min_cells, out_path, sym2entrez, ref_pairs):
    t0 = time.time()
    print(f"\n=== {atlas_name} ===")
    print(f"  cells (loaded)   : {X.shape[0]:,}")
    print(f"  genes (loaded)   : {X.shape[1]:,}")

    # Per-population depth (CAVEAT 2)
    unique_labels = list(dict.fromkeys(labels.tolist()))
    depth = per_pop_depth(X, labels, unique_labels)

    # Gene filter (same as W4)
    print(f"  applying gene filter (min_cells={min_cells}) ...")
    Xf, gene_list_f = preprocess_and_filter(X, gene_list_symbols, min_cells)
    print(f"  post-filter genes: {len(gene_list_f):,}")

    # Symbol -> Entrez mapping on the filtered gene list
    entrez_ids = map_gene_list_symbols_to_entrez(gene_list_f, sym2entrez)
    n_mapped = sum(1 for e in entrez_ids if e is not None)
    print(f"  symbol -> Entrez mapped: {n_mapped:,} / {len(gene_list_f):,}")

    # Replace unmapped symbols with a sentinel Entrez ID that will never appear
    # in the reference pair set. Using a unique-per-column placeholder ensures
    # any pair (which is defined over Entrez IDs) still gets a well-defined
    # "not present" verdict, and StemSC filters those pairs out cell-agnostic.
    entrez_ids_final = []
    for i, e in enumerate(entrez_ids):
        entrez_ids_final.append(e if e is not None else f"UNMAPPED_{i}")

    # Compute StemSC score + REO-primitive
    print("  computing StemSC score + REO-primitive ...")
    stemsc_score, reo_prim, info = stemsc_from_matrix(Xf, entrez_ids_final, ref_pairs)
    print(f"  reference pairs surviving: {info['n_pairs_present']:,} / "
          f"{info['n_pairs_total']:,} "
          f"({info['n_pairs_present']/info['n_pairs_total']:.1%})")
    print(f"  reference genes present  : {info['genes_ref_present']:,} / "
          f"{info['genes_ref_total']:,}")

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

    # Own-baseline stats (mirrors W4)
    finite_ranks = ranks[np.isfinite(ranks)]
    top_val = int(np.max(finite_ranks))
    bot_val = int(np.min(finite_ranks))
    bot_vals = [bot_val]

    print("  own-baseline: StemSC ...")
    stats = own_baseline_stats(
        score=stemsc_score,
        primitive=reo_prim,
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

    result = {
        "atlas": atlas_name,
        "score": "StemSC",
        "primitive": "REO-primitive (k / n_present)",
        "n_cells_total": int(X.shape[0]),
        "n_cells_ranked": int(np.sum(np.isfinite(ranks))),
        "n_cells_unranked": int(np.sum(~np.isfinite(ranks))),
        "n_genes_loaded": int(X.shape[1]),
        "n_genes_after_filter": int(Xf.shape[1]),
        "n_genes_mapped_to_entrez": n_mapped,
        "gene_filter_min_cells": min_cells,
        "ref_pairs_total": info["n_pairs_total"],
        "ref_pairs_present_in_atlas": info["n_pairs_present"],
        "ref_pairs_present_pct": info["n_pairs_present"] / info["n_pairs_total"],
        "ref_genes_total": info["genes_ref_total"],
        "ref_genes_present": info["genes_ref_present"],
        "kernel": "scipy.stats.weightedtau (tie-handled via lexicographic averaging)",
        "per_tier_n": tier_ns,
        "per_population_depth": depth,
        "per_score": {"StemSC": stats},
        "top_tier": top_val,
        "bottom_tier": bot_val,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"  written {out_path}")

    # Print concise verdict summary
    d = stats["marginal_delta"]
    cd_lo, cd_hi = stats["conditional_skill_CI95"]
    print(f"\n  === {atlas_name} StemSC verdict ===")
    print(f"    skill(StemSC)         = {stats['skill_score']:+.4f}  "
          f"CI [{stats['skill_score_CI95'][0]:+.4f}, {stats['skill_score_CI95'][1]:+.4f}]")
    print(f"    skill(REO-primitive)  = {stats['skill_primitive']:+.4f}  "
          f"CI [{stats['skill_primitive_CI95'][0]:+.4f}, {stats['skill_primitive_CI95'][1]:+.4f}]")
    print(f"    marginal Δ            = {d:+.4f}  "
          f"CI [{stats['marginal_delta_CI95'][0]:+.4f}, {stats['marginal_delta_CI95'][1]:+.4f}]")
    print(f"    conditional skill     = {stats['conditional_skill']:+.4f}  "
          f"CI [{cd_lo:+.4f}, {cd_hi:+.4f}]")
    print(f"    AUROC(score, top-vs-bot)   = {stats['AUROC_score_top_vs_bot']:.4f}")
    print(f"    AUROC(prim, top-vs-bot)    = {stats['AUROC_primitive_top_vs_bot']:.4f}")
    print(f"    VERDICT               = {v}")

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1-only", action="store_true")
    parser.add_argument("--c2-only", action="store_true")
    args = parser.parse_args()

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True, parents=True)

    print("Loading symbol->Entrez mapping (NCBI gene_info) ...")
    sym2entrez = load_symbol_to_entrez()
    print(f"  {len(sym2entrez):,} symbol -> Entrez entries (primary + synonyms)")

    print("Loading reference pair set ...")
    ref_pairs = load_ref_pairs()
    print(f"  {len(ref_pairs):,} reference pairs")

    summary = {}

    if not args.c2_only:
        X, genes, cell_ids, labels, ranks = load_c1(W4_DATA / "c1_gse117498")
        res_c1 = run_atlas(
            "C1_GSE117498",
            X, genes, cell_ids, labels, ranks,
            min_cells=C1_GENE_FILTER_MIN_CELLS,
            out_path=out_dir / "c1_gse117498_stemsc.json",
            sym2entrez=sym2entrez,
            ref_pairs=ref_pairs,
        )
        v = res_c1["per_score"]["StemSC"]
        summary["C1_GSE117498"] = {
            "StemSC": {
                "skill_score": v["skill_score"],
                "skill_primitive": v["skill_primitive"],
                "marginal_delta": v["marginal_delta"],
                "marginal_delta_CI95": v["marginal_delta_CI95"],
                "conditional_skill": v["conditional_skill"],
                "conditional_skill_CI95": v["conditional_skill_CI95"],
                "AUROC_score_top_vs_bot": v["AUROC_score_top_vs_bot"],
                "AUROC_primitive_top_vs_bot": v["AUROC_primitive_top_vs_bot"],
                "verdict": v["verdict"],
                "ref_pairs_present_pct": res_c1["ref_pairs_present_pct"],
            }
        }

    if not args.c1_only:
        X, genes, cell_ids, labels, ranks = load_c2(W4_DATA / "c2_gse125970")
        res_c2 = run_atlas(
            "C2_GSE125970",
            X, genes, cell_ids, labels, ranks,
            min_cells=C2_GENE_FILTER_MIN_CELLS,
            out_path=out_dir / "c2_gse125970_stemsc.json",
            sym2entrez=sym2entrez,
            ref_pairs=ref_pairs,
        )
        v = res_c2["per_score"]["StemSC"]
        summary["C2_GSE125970"] = {
            "StemSC": {
                "skill_score": v["skill_score"],
                "skill_primitive": v["skill_primitive"],
                "marginal_delta": v["marginal_delta"],
                "marginal_delta_CI95": v["marginal_delta_CI95"],
                "conditional_skill": v["conditional_skill"],
                "conditional_skill_CI95": v["conditional_skill_CI95"],
                "AUROC_score_top_vs_bot": v["AUROC_score_top_vs_bot"],
                "AUROC_primitive_top_vs_bot": v["AUROC_primitive_top_vs_bot"],
                "verdict": v["verdict"],
                "ref_pairs_present_pct": res_c2["ref_pairs_present_pct"],
            }
        }

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSUMMARY written: {out_dir / 'summary.json'}")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
