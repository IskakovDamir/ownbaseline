#!/usr/bin/env python3
"""
measure_split.py — what the GSE117498 gene-name split does to the W4 rows
(CytoTRACE proxy, SR, CCAT) and to StemSC, the results that went through
w4_gate2_run.load_c1 before the split was found.

The seven sorted-population files name genes in HGNC form (HLA-A), the four
broad-gate files in R make.names form (HLA.A); load_c1 unions by exact name.
"split" is load_c1 as the published runs used it. "merged" is the same cells
with the names merged by the rule of
experiments/w5_new_rows/prepare_inputs.ds_gse117498h, which is called here
directly: its unfiltered merged matrix is taken as it enters the 10-cell gene
filter, so the rule is reused and not copied. Both conditions then go through
w4_gate2_run.run_atlas and stemsc_run.run_atlas unchanged, at both kernels for
W4 (tau_b is the run record's, weighted tau the manuscript-era copy's) and at
weighted tau for StemSC (the kernel of its run record).

    python3 experiments/w4_name_split/measure_split.py facts
    python3 experiments/w4_name_split/measure_split.py loader
    python3 experiments/w4_name_split/measure_split.py w4 split
    python3 experiments/w4_name_split/measure_split.py w4 merged
    python3 experiments/w4_name_split/measure_split.py stemsc split  --stemsc-ref DIR
    python3 experiments/w4_name_split/measure_split.py stemsc merged --stemsc-ref DIR
    python3 experiments/w4_name_split/measure_split.py compare [--w4-weighted FILE]

DIR holds StemSC's ref_pairs.tsv and mapping/Homo_sapiens.gene_info.gz, which
the repository does not ship (default: experiments/stemsc, where stemsc.py
looks). --w4-weighted is the weighted-tau W4 C1 JSON of the original run, for
the reproduction check of that kernel; without it the check is skipped.

Per-condition outputs go to $OWNBASELINE_DATA_ROOT/w4_name_split/; compare
writes experiments/w4_name_split/split_vs_merged.json and prints the tables
NOTE.md carries. No tracked result file is read for writing.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "w4" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "stemsc"))
sys.path.insert(0, str(REPO / "experiments" / "w5_new_rows"))
from own_baseline.paths import data_root  # noqa: E402
import w4_gate2_run as w4  # noqa: E402

OUT = data_root() / "w4_name_split"
RECORD = REPO / "data" / "run_record"
SUMMARY = HERE / "split_vs_merged.json"
C1_DIR = data_root() / "w4" / "data" / "c1_gse117498"
KERNELS = ("kendalltau", "weightedtau")
# the C1 run record the published numbers came from, replaced on 2026-10-09
SPLIT_RECORD_REV = "c26b588"
SPLIT_RECORD_SHA256 = "1fc503e5ab8fc45ed93b4e63d4b3bf06efa841fb5dabe7cdba9ba8a333f6d6ec"
RANKED_POPS = ["HSC", "MPP", "LinNegCD34PosCD164Pos", "MLP", "CMP", "PreBNK", "MEP", "GMP"]


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ----------------------------------------------------------------------------- #
#  facts: the gene names of the eleven files, no expression value read
# ----------------------------------------------------------------------------- #
def file_gene_names():
    per_file = {}
    for fname, pop in w4.C1_GSM_FILES.items():
        with gzip.open(C1_DIR / fname, "rt") as fh:
            fh.readline()
            per_file[fname] = [line.split("\t", 1)[0] for line in fh if
                               not line.startswith("Library\t")]
    return per_file


def pairs_from(sorted_genes, broad_genes):
    """ds_gse117498h's rule, restated only to count it; the merge itself is its."""
    pairs = {}
    for x in sorted_genes:
        if "-" in x:
            y = x.replace("-", ".")
            if y in broad_genes and y not in sorted_genes:
                pairs[y] = x
    return pairs


def cmd_facts(_args):
    per_file = file_gene_names()
    files = []
    sorted_genes, broad_genes = set(), set()
    for fname, names in per_file.items():
        pop = w4.C1_GSM_FILES[fname]
        broad = pop.startswith("LinNeg")
        (broad_genes if broad else sorted_genes).update(names)
        files.append({"file": fname, "group": "broad-gate" if broad else "sorted",
                      "n_genes": len(names), "n_unique": len(set(names)),
                      "n_hyphen": sum("-" in g for g in names),
                      "n_dot": sum("." in g for g in names)})
    pairs = pairs_from(sorted_genes, broad_genes)
    broad_only = broad_genes - sorted_genes
    sorted_only = sorted_genes - broad_genes
    hyph_unpaired = sorted(x for x in sorted_genes if "-" in x and x not in set(pairs.values()))
    z = np.load(w4.STRING_NPZ, allow_pickle=True)
    string_genes = {str(g) for g in z["genes"]}
    facts = {
        "files": files,
        "n_sorted_union": len(sorted_genes), "n_broad_union": len(broad_genes),
        "n_union": len(sorted_genes | broad_genes),
        "n_common_names": len(sorted_genes & broad_genes),
        "n_pairs": len(pairs),
        "n_broad_only": len(broad_only),
        "n_broad_only_not_paired": len(broad_only - set(pairs)),
        "n_sorted_only": len(sorted_only),
        "n_sorted_hyphen_not_paired": len(hyph_unpaired),
        "sorted_hyphen_not_paired_dot_form_is_sorted_name": sum(
            x.replace("-", ".") in sorted_genes for x in hyph_unpaired),
        "sorted_hyphen_not_paired_dot_form_absent_from_broad": sum(
            x.replace("-", ".") not in broad_genes for x in hyph_unpaired),
        "n_paired_names_in_string_v12_lcc": len(set(pairs.values()) & string_genes),
        "n_string_v12_lcc_genes": len(string_genes),
        "examples": sorted(pairs.items())[:5],
        "pairs": pairs,
        "mt_pairs": sorted(x for x in pairs.values() if x.startswith("MT-")),
        "hla_pairs": sorted(x for x in pairs.values() if x.startswith("HLA-")),
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps({k: v for k, v in facts.items() if k not in ("files", "pairs")}, indent=2))
    for f in files:
        print(f"  {f['file']:52s} {f['group']:10s} {f['n_genes']:>6,} genes "
              f"{f['n_hyphen']:>5,} hyphen {f['n_dot']:>5,} dot")


# ----------------------------------------------------------------------------- #
#  the two conditions, unfiltered, as run_atlas expects them
# ----------------------------------------------------------------------------- #
@contextlib.contextmanager
def patched(module, name, fn):
    orig = getattr(module, name)
    setattr(module, name, fn(orig))
    try:
        yield
    finally:
        setattr(module, name, orig)


def load(cond):
    if cond == "split":
        X, genes, cell_ids, labels, ranks = w4.load_c1(C1_DIR)
        return X, list(genes), cell_ids, labels, ranks, {}
    import prepare_inputs as pi
    got = {}

    def grab(orig):
        def f(X, gene_list, min_cells):
            got["X"], got["genes"] = X, list(gene_list)
            return orig(X, gene_list, min_cells)
        return f

    with patched(w4, "preprocess_and_filter", grab):
        _Xf, genes_f, cell_ids, prim, meta = pi.ds_gse117498h()
    del _Xf
    cell_ids = np.array(cell_ids)
    labels = np.array([c.split("::", 1)[0] for c in cell_ids])
    ranks = np.asarray(prim["ordinal"], dtype=np.float64)
    info = {"n_gene_pairs_merged": meta["n_gene_pairs_merged"],
            "n_genes_after_filter_ds_gse117498h": len(genes_f)}
    w5 = data_root() / "w5_new_rows" / "inputs" / "GSE117498h" / "genes.txt"
    if w5.is_file():
        info["genes_after_filter_equal_w5_inputs_GSE117498h"] = (
            w5.read_text().split("\n")[:-1] == [str(g) for g in genes_f])
    return got["X"], got["genes"], cell_ids, labels, ranks, info


def per_pop_medians(arrays, labels):
    return {pop: {k: float(np.nanmedian(v[labels == pop])) for k, v in arrays.items()}
            for pop in RANKED_POPS}


def cmd_loader(_args):
    """load_c1(merge_names=True) against the merged matrix ds_gse117498h builds."""
    Xm, genes_m, cells_m, labels_m, ranks_m, _info = load("merged")
    X, genes, cells, labels, ranks = w4.load_c1(C1_DIR, merge_names=True)
    res = {"genes_equal": list(genes) == list(genes_m),
           "cells_equal": bool(np.array_equal(cells, cells_m)),
           "labels_equal": bool(np.array_equal(labels, labels_m)),
           "ranks_equal": bool(np.array_equal(ranks, ranks_m, equal_nan=True)),
           "dtype": [str(X.dtype), str(Xm.dtype)],
           "matrix_equal": bool(X.shape == Xm.shape and np.array_equal(X, Xm)),
           "n_genes": len(genes)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "loader_check.json").write_text(json.dumps(res, indent=2) + "\n")
    print(json.dumps(res, indent=2))
    if not all(v for k, v in res.items() if k.endswith("_equal")):
        raise SystemExit("load_c1(merge_names=True) differs from ds_gse117498h")


# ----------------------------------------------------------------------------- #
#  W4 rows
# ----------------------------------------------------------------------------- #
def cmd_w4(args):
    import potency_metrics as pm
    X, genes, cell_ids, labels, ranks, info = load(args.cond)
    cache = {}

    def keep_ct(orig):
        def f(adata, *a, **k):
            r = orig(adata, *a, **k)
            cache["ct_top"] = [str(adata.var_names[i]) for i in r["gcs_gene_idx"]]
            return r
        return f

    def keep_scores(orig):
        def f(Xf, gene_list, A, degree, col_idx):
            if "S" not in cache:
                with patched(pm, "cytotrace_proxy", keep_ct):
                    cache["S"] = orig(Xf, gene_list, A, degree, col_idx)
                cache["genes_f"] = list(gene_list)
                cache["scaffold"] = [gene_list[i] for i in col_idx]
            return cache["S"]
        return f

    OUT.mkdir(parents=True, exist_ok=True)
    old = os.environ.get("OWNBASELINE_KERNEL")
    try:
        with patched(w4, "compute_scores_and_primitives", keep_scores):
            for kern in KERNELS:
                os.environ["OWNBASELINE_KERNEL"] = kern
                res = w4.run_atlas("C1_GSE117498", X, genes, cell_ids, labels, ranks,
                                   min_cells=w4.C1_GENE_FILTER_MIN_CELLS,
                                   rank_to_label_map=w4.C1_POP_TO_RANK,
                                   out_path=OUT / f"w4_{args.cond}_{kern}.json")
    finally:
        if old is None:
            os.environ.pop("OWNBASELINE_KERNEL", None)
        else:
            os.environ["OWNBASELINE_KERNEL"] = old
    S = cache["S"]
    extra = dict(info, condition=args.cond,
                 per_population_median=per_pop_medians(S, labels),
                 ct_top200=cache["ct_top"], scaffold_genes=cache["scaffold"],
                 n_genes_after_filter=len(cache["genes_f"]))
    (OUT / f"w4_{args.cond}_extra.json").write_text(json.dumps(extra, indent=1) + "\n")
    np.savez(OUT / f"w4_{args.cond}_cells.npz", labels=labels, ranks=ranks,
             **{k: np.asarray(v) for k, v in S.items()})
    print(f"wrote {OUT}/w4_{args.cond}_*.json, n_genes_after_filter "
          f"{res['n_genes_after_filter']:,}, info {info}")


# ----------------------------------------------------------------------------- #
#  StemSC
# ----------------------------------------------------------------------------- #
def cmd_stemsc(args):
    import stemsc_run as sr
    from stemsc import load_ref_pairs
    ref_dir = Path(args.stemsc_ref)
    pairs_tsv = ref_dir / "ref_pairs.tsv"
    gene_info = ref_dir / "mapping" / "Homo_sapiens.gene_info.gz"
    sr.GENE_INFO = gene_info
    sym2e = sr.load_symbol_to_entrez()
    ref_pairs = load_ref_pairs(pairs_tsv)
    X, genes, cell_ids, labels, ranks, info = load(args.cond)
    cache = {}

    def keep(orig):
        def f(Xf, ids, pairs):
            score, prim, inf = orig(Xf, ids, pairs)
            cache.update(score=score, prim=prim, ids=list(ids))
            return score, prim, inf
        return f

    OUT.mkdir(parents=True, exist_ok=True)
    with patched(sr, "stemsc_from_matrix", keep):
        res = sr.run_atlas("C1_GSE117498", X, genes, cell_ids, labels, ranks,
                           min_cells=w4.C1_GENE_FILTER_MIN_CELLS,
                           out_path=OUT / f"stemsc_{args.cond}.json",
                           sym2entrez=sym2e, ref_pairs=ref_pairs)
    ref_genes = set(ref_pairs["geneA_entrez"]) | set(ref_pairs["geneB_entrez"])
    # reference genes reached only through a hyphenated symbol (the ones the split
    # leaves at zero in broad-gate cells)
    hyph = {sym2e.get(g) for g in genes if "-" in g} & ref_genes
    extra = dict(info, condition=args.cond,
                 sha256_ref_pairs=sha256(pairs_tsv), sha256_gene_info=sha256(gene_info),
                 n_ref_genes_with_hyphenated_symbol_in_atlas=len(hyph),
                 per_population_median=per_pop_medians(
                     {"StemSC": cache["score"], "REO_primitive": cache["prim"]}, labels))
    (OUT / f"stemsc_{args.cond}_extra.json").write_text(json.dumps(extra, indent=1) + "\n")
    np.savez(OUT / f"stemsc_{args.cond}_cells.npz", labels=labels, ranks=ranks,
             StemSC=cache["score"], REO_primitive=cache["prim"])
    print(f"wrote {OUT}/stemsc_{args.cond}*.json; pairs present "
          f"{res['ref_pairs_present_in_atlas']:,}, info {info}")


# ----------------------------------------------------------------------------- #
#  compare
# ----------------------------------------------------------------------------- #
STAT_FIELDS = ["skill_score", "skill_primitive", "marginal_delta", "conditional_skill",
               "AUROC_score_top_vs_bot", "AUROC_primitive_top_vs_bot"]
CI_FIELDS = ["skill_score_CI95", "skill_primitive_CI95", "marginal_delta_CI95",
             "conditional_skill_CI95"]


def max_abs_diff(a, b, skip=("elapsed_seconds",)):
    """Largest numeric disagreement between two JSON trees, and any non-numeric one."""
    worst, other = 0.0, []

    def walk(x, y, path):
        nonlocal worst
        if isinstance(x, dict) and isinstance(y, dict):
            for k in set(x) | set(y):
                if k in skip:
                    continue
                if k not in x or k not in y:
                    other.append(f"{path}/{k} missing on one side")
                    continue
                walk(x[k], y[k], f"{path}/{k}")
        elif isinstance(x, list) and isinstance(y, list) and len(x) == len(y):
            for i, (u, v) in enumerate(zip(x, y)):
                walk(u, v, f"{path}[{i}]")
        elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
            if np.isnan(x) or np.isnan(y):
                if not (np.isnan(x) and np.isnan(y)):
                    other.append(f"{path}: {x!r} != {y!r}")
            else:
                worst = max(worst, abs(float(x) - float(y)))
        elif x != y:
            other.append(f"{path}: {x!r} != {y!r}")
    walk(a, b, "")
    return worst, other


def stat_rows(old, new):
    rows = {}
    for score, o in old.items():
        n = new[score]
        r = {}
        for f in STAT_FIELDS:
            r[f] = {"split": o[f], "merged": n[f], "change": n[f] - o[f]}
        for f in CI_FIELDS:
            r[f] = {"split": o[f], "merged": n[f]}
        r["verdict"] = {"split": o["verdict"], "merged": n["verdict"]}
        rows[score] = r
    return rows


def fmt(v, d=4):
    return f"{v:+.{d}f}"


def print_table(title, rows):
    print(f"\n### {title}\n")
    print("| score | quantity | split (published path) | merged | change |")
    print("|---|---|---|---|---|")
    for score, r in rows.items():
        for f in STAT_FIELDS:
            x = r[f]
            cf = f + "_CI95"
            if cf in r:
                ci_o = r[cf]["split"]; ci_n = r[cf]["merged"]
                print(f"| {score} | {f} | {fmt(x['split'])} [{fmt(ci_o[0])}, {fmt(ci_o[1])}] | "
                      f"{fmt(x['merged'])} [{fmt(ci_n[0])}, {fmt(ci_n[1])}] | {fmt(x['change'])} |")
            else:
                print(f"| {score} | {f} | {x['split']:.4f} | {x['merged']:.4f} | {fmt(x['change'])} |")
        print(f"| {score} | verdict | {r['verdict']['split']} | {r['verdict']['merged']} | "
              f"{'same' if r['verdict']['split'] == r['verdict']['merged'] else 'CHANGED'} |")


def cmd_compare(args):
    L = lambda p: json.loads(Path(p).read_text())  # noqa: E731
    out = {"facts": L(OUT / "facts.json"), "reproduction": {}, "w4": {}, "stemsc": {}}

    # reproduction: the split recomputation against what the published runs wrote
    # (the C1 record before the 2026-10-09 correction, read from git history)
    import subprocess
    old = subprocess.run(["git", "-C", str(REPO), "show", f"{SPLIT_RECORD_REV}:"
                          "data/run_record/w4/c1_gse117498_results.json"],
                         capture_output=True, check=True).stdout
    if hashlib.sha256(old).hexdigest() != SPLIT_RECORD_SHA256:
        raise SystemExit(f"{SPLIT_RECORD_REV}: not the pre-correction C1 record")
    d, other = max_abs_diff(json.loads(old), L(OUT / "w4_split_kendalltau.json"))
    out["reproduction"]["w4_tau_b_vs_run_record_before_correction"] = {
        "record_sha256": SPLIT_RECORD_SHA256, "max_abs_diff": d, "non_numeric": other}
    cur = RECORD / "w4" / "c1_gse117498_results.json"
    d, other = max_abs_diff(L(cur), L(OUT / "w4_merged_kendalltau.json"))
    out["reproduction"]["w4_merged_tau_b_vs_run_record_now"] = {
        "record_sha256": sha256(cur), "max_abs_diff": d, "non_numeric": other}
    if args.w4_weighted:
        d, other = max_abs_diff(L(args.w4_weighted), L(OUT / "w4_split_weightedtau.json"))
        out["reproduction"]["w4_weighted_vs_original_run"] = {
            "file_sha256": sha256(args.w4_weighted), "max_abs_diff": d, "non_numeric": other}
    rec_ss = L(RECORD / "stemsc" / "summary.json")["C1_GSE117498"]["StemSC"]
    new_ss = L(OUT / "stemsc_split.json")
    mine = dict(new_ss["per_score"]["StemSC"], ref_pairs_present_pct=new_ss["ref_pairs_present_pct"])
    d, other = max_abs_diff(rec_ss, {k: mine[k] for k in rec_ss})
    out["reproduction"]["stemsc_vs_run_record"] = {"max_abs_diff": d, "non_numeric": other}

    for kern in KERNELS:
        s = L(OUT / f"w4_split_{kern}.json"); m = L(OUT / f"w4_merged_{kern}.json")
        out["w4"][kern] = {
            "stats": stat_rows(s["per_score"], m["per_score"]),
            "n_genes_after_filter": [s["n_genes_after_filter"], m["n_genes_after_filter"]],
            "n_genes_loaded": [s["n_genes_loaded"], m["n_genes_loaded"]],
            "string_atlas_lcc_nodes": [s["scaffold"]["intersect_atlas_LCC_nodes"],
                                       m["scaffold"]["intersect_atlas_LCC_nodes"]],
            "n_cells_ranked": [s["n_cells_ranked"], m["n_cells_ranked"]],
            "per_population_depth_max_abs_diff": max_abs_diff(
                s["per_population_depth"], m["per_population_depth"])[0],
        }
    es, em = L(OUT / "w4_split_extra.json"), L(OUT / "w4_merged_extra.json")
    # CT gene-set overlap with each broad-gate twin read as its HGNC name
    pairs = out["facts"].pop("pairs")
    paired_x = set(pairs.values())
    top_s = [pairs.get(g, g) for g in es["ct_top200"]]
    top_m = list(em["ct_top200"])
    out["w4"]["ct_top200"] = {
        "overlap_after_name_normalisation": len(set(top_s) & set(top_m)),
        "split_top200_hgnc_paired_names": sum(g in paired_x for g in es["ct_top200"]),
        "split_top200_dot_names": sum(g in pairs for g in es["ct_top200"]),
        "merged_top200_paired_names": sum(g in paired_x for g in top_m),
        "only_split": sorted(set(top_s) - set(top_m)),
        "only_merged": sorted(set(top_m) - set(top_s)),
    }
    out["w4"]["scaffold"] = {
        "split_paired_hgnc_names": sum(g in paired_x for g in es["scaffold_genes"]),
        "split_dot_names": sum(g in pairs for g in es["scaffold_genes"]),
        "merged_paired_hgnc_names": sum(g in paired_x for g in em["scaffold_genes"]),
    }
    out["w4"]["per_population_median"] = {"split": es["per_population_median"],
                                          "merged": em["per_population_median"]}
    out["w4"]["merged_info"] = {k: v for k, v in em.items() if k.startswith("n_gene")
                                or k.startswith("genes_after")}

    ss, sm = L(OUT / "stemsc_split.json"), L(OUT / "stemsc_merged.json")
    xs, xm = L(OUT / "stemsc_split_extra.json"), L(OUT / "stemsc_merged_extra.json")
    out["stemsc"] = {
        "stats": stat_rows(ss["per_score"], sm["per_score"]),
        "kernel": ss["kernel"],
        "ref_pairs_present": [ss["ref_pairs_present_in_atlas"], sm["ref_pairs_present_in_atlas"]],
        "ref_genes_present": [ss["ref_genes_present"], sm["ref_genes_present"]],
        "n_genes_mapped_to_entrez": [ss["n_genes_mapped_to_entrez"], sm["n_genes_mapped_to_entrez"]],
        "n_ref_genes_with_hyphenated_symbol_in_atlas": [
            xs["n_ref_genes_with_hyphenated_symbol_in_atlas"],
            xm["n_ref_genes_with_hyphenated_symbol_in_atlas"]],
        "sha256_ref_pairs": xs["sha256_ref_pairs"], "sha256_gene_info": xs["sha256_gene_info"],
        "per_population_median": {"split": xs["per_population_median"],
                                  "merged": xm["per_population_median"]},
    }
    SUMMARY.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {SUMMARY}")

    print("\n## reproduction\n")
    print(json.dumps(out["reproduction"], indent=2))
    for kern in KERNELS:
        print_table(f"W4, {kern}", out["w4"][kern]["stats"])
        print({k: v for k, v in out["w4"][kern].items() if k != "stats"})
    print_table("StemSC, weightedtau", out["stemsc"]["stats"])
    print({k: v for k, v in out["stemsc"].items() if k not in ("stats", "per_population_median")})
    print("\nct_top200", {k: v for k, v in out["w4"]["ct_top200"].items()
                          if not k.startswith("only")})
    print("scaffold", out["w4"]["scaffold"])
    print("\nper-population medians (split -> merged)")
    for block, src in (("w4", out["w4"]["per_population_median"]),
                       ("stemsc", out["stemsc"]["per_population_median"])):
        for pop in RANKED_POPS:
            a, b = src["split"][pop], src["merged"][pop]
            print(f"  {block:6s} {pop:24s} " + "  ".join(
                f"{k} {a[k]:.4g}->{b[k]:.4g}" for k in a))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("facts")
    sub.add_parser("loader")
    p = sub.add_parser("w4"); p.add_argument("cond", choices=["split", "merged"])
    p = sub.add_parser("stemsc"); p.add_argument("cond", choices=["split", "merged"])
    p.add_argument("--stemsc-ref", default=str(REPO / "experiments" / "stemsc"))
    p = sub.add_parser("compare")
    p.add_argument("--w4-weighted", default=None)
    args = ap.parse_args()
    {"facts": cmd_facts, "loader": cmd_loader, "w4": cmd_w4, "stemsc": cmd_stemsc,
     "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    main()
