#!/usr/bin/env python3
"""
sorted_depth.py: is the reversed gene-count ordering of the sorted atlases a matter of depth,
and does the degree correlation keep its direction at equal depth?

    python3 experiments/specificity/sorted_depth.py blind      # reads no label
    python3 experiments/specificity/sorted_depth.py analyse    # after PREREG_depth_2.md is committed

Plan: PREREG_depth_2.md. `blind` draws every cell of the two sorted atlases down to a common
depth and recounts detected genes and the gene-count signature, and recomputes PCC(x, degree)
on the three staged series at 1,000 counts per cell. `analyse` refuses to run unless the plan,
this file and blind_manifest_depth_2.json are committed and the files match the manifest.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (REPO, SERIES, WORK, load_cells, load_degree, load_genes, load_matrix, lognorm, num,  # noqa: E402
                    scaffold_columns, sha256)
from controls import gene_count_signature, pcc_degree, thin_counts  # noqa: E402

ATLASES = ("GSE117498h", "GSE125970")
DEPTHS = (1000, 500, 2000)
TOP_BOTTOM = {"GSE117498h": (5, 2), "GSE125970": (4, 1)}
MANIFEST = HERE / "blind_manifest_depth_2.json"


def auroc(pos, neg):
    """P(a positive scores above a negative), ties counted half."""
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def blind():
    WORK.mkdir(parents=True, exist_ok=True)
    out = {}
    for ai, geo in enumerate(ATLASES):
        x = load_matrix(geo)
        rec = {}
        for depth in DEPTHS:
            keep, xt = thin_counts(x, depth, (40, ai, depth))
            rec[f"keep_{depth}"] = keep
            rec[f"genes_{depth}"] = np.asarray((xt > 0).sum(axis=1)).ravel().astype(np.float64)
            if depth == 1000:
                xn, _ = lognorm(xt)
                rec["gcs_1000"], _ = gene_count_signature(xn, rec["genes_1000"])
                xk = x.tocsr()[keep].astype(np.float64)                  # the kept cells at raw depth
                xn, _ = lognorm(xk)
                gck = np.asarray((xk > 0).sum(axis=1)).ravel().astype(np.float64)
                rec["gcs_kept_raw"], _ = gene_count_signature(xn, gck)
        xall = x.astype(np.float64)
        xn, _ = lognorm(xall)
        rec["genes_raw"] = np.asarray((xall > 0).sum(axis=1)).ravel().astype(np.float64)
        rec["gcs_raw"], _ = gene_count_signature(xn, rec["genes_raw"])
        path = WORK / f"{geo}_depth2.npz"
        np.savez(path, **rec)
        out[path.name] = {"sha256": sha256(path), "shapes": {k: list(v.shape) for k, v in rec.items()}}
        print(path.name, out[path.name]["sha256"], {d: int(len(rec[f"keep_{d}"])) for d in DEPTHS}, flush=True)
    degree = load_degree()
    for geo in SERIES:
        x = load_matrix(geo)
        keep, xt = thin_counts(x, 1000, (4, SERIES.index(geo), 1000))     # the draw of controls.py thin
        cols, deg = scaffold_columns(geo, load_genes(geo), degree)
        xn, _ = lognorm(xt)
        rec = {"keep_1000": keep, "pcc_1000": pcc_degree(xn[:, cols].tocsr(), deg[:, None])[:, 0]}
        path = WORK / f"{geo}_pcc_depth2.npz"
        np.savez(path, **rec)
        out[path.name] = {"sha256": sha256(path), "shapes": {k: list(v.shape) for k, v in rec.items()}}
        print(path.name, out[path.name]["sha256"], int(len(keep)), flush=True)
    MANIFEST.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")


def locked():
    rel = HERE.relative_to(REPO).as_posix()
    run = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True, check=True).stdout  # noqa: E731
    dirty = [ln for ln in run("status", "--porcelain", "--", rel).splitlines() if "/results/" not in ln]
    if dirty:
        raise SystemExit("uncommitted changes under experiments/specificity:\n" + "\n".join(dirty))
    committed = run("show", f"HEAD:{rel}/{MANIFEST.name}")
    if committed != MANIFEST.read_text():
        raise SystemExit(f"{MANIFEST.name} differs from the committed one")
    m = json.loads(committed)
    for name, rec in m.items():
        if sha256(WORK / name) != rec["sha256"]:
            raise SystemExit(f"{name} does not match the committed manifest")
    return run("rev-parse", "HEAD").strip()


def analyse():
    head = locked()
    res = {"locked_at_commit": head, "atlases": {}, "staged": {}}
    for geo in ATLASES:
        c = load_cells(geo)
        o, lib = num(c["ordinal"]), num(c["library_size"])
        label = np.array(c["label"])
        z = np.load(WORK / f"{geo}_depth2.npz")
        top, bottom = TOP_BOTTOM[geo]
        ranked = np.isfinite(o)

        def read(values, cells):
            oo = o[cells]
            ok = np.isfinite(oo)
            return {"auroc_top_vs_bottom": auroc(values[oo == top], values[oo == bottom]),
                    "tau_b_all_ranked": float(kendalltau(values[ok], oo[ok]).statistic),
                    "top_cells": int((oo == top).sum()), "bottom_cells": int((oo == bottom).sum()), "ranked_cells": int(ok.sum())}

        every = np.arange(len(o))
        a = {"deposit_detected_genes": read(num(c["detected_genes"]), every),
             "raw": {"detected_genes": read(z["genes_raw"], every), "gene_count_signature": read(z["gcs_raw"], every)},
             "kept_cells_raw_depth": {"detected_genes": read(z["genes_raw"][z["keep_1000"]], z["keep_1000"]),
                                      "gene_count_signature": read(z["gcs_kept_raw"], z["keep_1000"])},
             "thinned": {}, "by_population": {}}
        for d in DEPTHS:
            a["thinned"][d] = {"detected_genes": read(z[f"genes_{d}"], z[f"keep_{d}"])}
        a["thinned"][1000]["gene_count_signature"] = read(z["gcs_1000"], z["keep_1000"])
        kept = np.zeros(len(o), bool)
        kept[z["keep_1000"]] = True
        thin_genes = np.full(len(o), np.nan)
        thin_genes[z["keep_1000"]] = z["genes_1000"]
        for pop in sorted(set(label[ranked])):
            m = label == pop
            a["by_population"][pop] = {"ordinal": float(o[m][0]), "cells": int(m.sum()), "cells_with_1000_counts": int((m & kept).sum()),
                                       "median_library_size": float(np.median(lib[m])),
                                       "median_detected_genes": float(np.median(z["genes_raw"][m])),
                                       "median_detected_genes_at_1000": float(np.nanmedian(thin_genes[m])) if (m & kept).any() else None}
        res["atlases"][geo] = a
        print(geo, json.dumps({k: v for k, v in a.items() if k != "by_population"}), flush=True)
        print(geo, json.dumps(a["by_population"]), flush=True)
    for geo in SERIES:
        c = load_cells(geo)
        o, pcc = num(c["ordinal"]), num(c["pcc_x_degree"])
        z = np.load(WORK / f"{geo}_pcc_depth2.npz")
        k = z["keep_1000"]
        res["staged"][geo] = {"cells": int(len(o)), "cells_kept": int(len(k)),
                              "spearman_pcc_with_stage_raw": float(spearmanr(pcc, o).statistic),
                              "spearman_pcc_with_stage_raw_kept_cells": float(spearmanr(pcc[k], o[k]).statistic),
                              "spearman_pcc_with_stage_at_1000": float(spearmanr(z["pcc_1000"], o[k]).statistic),
                              "spearman_pcc_raw_with_pcc_at_1000": float(spearmanr(pcc[k], z["pcc_1000"]).statistic)}
        print(geo, json.dumps(res["staged"][geo]), flush=True)
    t = res["atlases"]
    res["predictions"] = {
        "T1": {"held": t["GSE117498h"]["thinned"][1000]["detected_genes"]["auroc_top_vs_bottom"] > 0.5,
               "observed": t["GSE117498h"]["thinned"][1000]["detected_genes"]["auroc_top_vs_bottom"]},
        "T2": {"held": t["GSE125970"]["thinned"][1000]["detected_genes"]["auroc_top_vs_bottom"] > 0.5,
               "observed": t["GSE125970"]["thinned"][1000]["detected_genes"]["auroc_top_vs_bottom"]},
        "T3": {"held": all(res["staged"][g]["spearman_pcc_with_stage_at_1000"] >= 0.1 for g in SERIES),
               "observed": {g: res["staged"][g]["spearman_pcc_with_stage_at_1000"] for g in SERIES}}}
    (HERE / "results" / "depth_2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["predictions"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["blind", "analyse"])
    a = ap.parse_args()
    blind() if a.mode == "blind" else analyse()
