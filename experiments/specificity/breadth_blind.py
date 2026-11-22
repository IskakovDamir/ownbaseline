#!/usr/bin/env python3
"""
breadth_blind.py: the label-blind stage of the specificity plan on a series outside the deposit.

    python3 experiments/specificity/breadth_blind.py scores     GSE169210 --procs 8
    python3 experiments/specificity/breadth_blind.py signatures GSE169210
    python3 experiments/specificity/breadth_blind.py thin       GSE169210 --procs 8
    python3 experiments/specificity/breadth_blind.py manifest   GSE169210

The series is laid out by its own preparation script under data/runs/breadth/<series>/:

    root/track2/data/prepared/   X_cells_genes.npz, genes.txt, primitives.npz    (no label)
    root/track2/results/         scent_scores.npz, written by wagner2018/score_scent.py
    labels/labels.tsv            cell, sample, batch, stage: never opened here
    blind/                       what this script writes

Every statistic is computed by the functions of controls.py, unchanged: A1 (1,000 random
200-gene sets), A2 (1,000 degree permutations), thinning to 1,000, 500 and 2,000 counts, and
CytoTRACE v1 through the chunked pipeline that reproduced the deposit's values exactly. The
only label-related act is `manifest`, which hashes labels.tsv as bytes so that the labels
the analysis reads are the ones that existed at the lock.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import REPO, lognorm, load_degree, rng_for, sha256  # noqa: E402
from controls import (cytotrace_pipeline, gene_count_signature, gene_set_scores, pcc_degree,  # noqa: E402
                      thin_counts)

# seed index of each series: the deposit's three series hold 0 to 2 in common.SERIES
SID = {"GSE169210": 3, "GSE223922": 4, "GSE223922b": 5}        # b: the second, disjoint draw of GSE223922
M_A1, G_A1, M_A2, MIN_CELLS = 1000, 200, 1000, 10
THIN_PRIMARY, THIN_SENS = 1000, (500, 2000)
MANIFEST = HERE / "blind_manifest_breadth.json"
RESULTS = HERE / "results"


def paths(geo):
    run = REPO / "data" / "runs" / "breadth" / geo
    return {"run": run, "prep": run / "root" / "track2" / "data" / "prepared",
            "scent": run / "root" / "track2" / "results" / "scent_scores.npz",
            "mce": run / "root" / "w5_new_rows" / "scores" / "GSE106474" / "mce.csv",
            "labels": run / "labels" / "labels.tsv", "blind": run / "blind"}


def load_counts(geo):
    p = paths(geo)["prep"]
    return sp.load_npz(p / "X_cells_genes.npz").tocsr().astype(np.float32), (p / "genes.txt").read_text().split("\n")


def record(mode, geo, meta):
    RESULTS.mkdir(exist_ok=True)
    f = RESULTS / f"breadth_blind_{mode}.json"
    prior = json.loads(f.read_text()) if f.is_file() else {}
    prior[geo] = meta
    f.write_text(json.dumps(prior, indent=1))
    print(json.dumps(meta), flush=True)


def scores(geo, procs):
    """CytoTRACE v1 and its gene-count signature; SR, CCAT and MCE are collected from their own runners."""
    t0 = time.time()
    p = paths(geo)
    x, _ = load_counts(geo)
    x = x.astype(np.float64)
    xn, _ = lognorm(x)
    gc = np.asarray((x > 0).sum(axis=1)).ravel().astype(np.float64)
    gcs, top = gene_count_signature(xn, gc)
    out = {"cytotrace_gene_count_signature": gcs, "gcs_genes": top,
           "cytotrace_v1": cytotrace_pipeline(xn, gcs[:, None], procs, geo)[:, 0]}
    z = np.load(p["scent"])
    out["scent_sr"], out["scent_ccat"] = z["sr"], z["ccat"]
    if p["mce"].is_file():
        import csv
        with open(p["mce"]) as fh:
            rows = list(csv.DictReader(fh))
        out["mce"] = np.array([float(r["mce"]) for r in rows])
    for k, v in out.items():
        if k != "gcs_genes" and (len(v) != x.shape[0] or not np.all(np.isfinite(v))):
            raise SystemExit(f"{k}: {len(v)} values for {x.shape[0]} cells, or a non-finite value")
    path = p["blind"] / "scores.npz"
    np.savez(path, **out)
    record("scores", geo, {"series": geo, "cells": int(x.shape[0]), "scores": sorted(k for k in out if k != "gcs_genes"),
                           "seconds": round(time.time() - t0, 1), "sha256": sha256(path)})


def signatures(geo):
    t0 = time.time()
    p = paths(geo)
    x, genes = load_counts(geo)
    xn, target = lognorm(x)
    universe = np.flatnonzero(np.asarray((x > 0).sum(axis=0)).ravel() >= MIN_CELLS)
    sid = SID[geo]
    out = {"universe": universe}
    out["a1_g200"], out["a1_g200_sets"] = gene_set_scores(xn, universe, G_A1, M_A1, (1, sid, G_A1))
    degree = load_degree()
    cols = np.array([j for j, g in enumerate(genes) if g in degree])
    deg = np.array([degree[genes[j]] for j in cols])
    xs = xn[:, cols].tocsr()
    real = pcc_degree(xs, deg[:, None])[:, 0]
    stored = np.load(p["prep"] / "primitives.npz")["pcc_string_v12"]
    perms = np.stack([rng_for(2, sid, j).permutation(deg) for j in range(M_A2)], axis=1)
    out["a2"] = pcc_degree(xs, perms).astype(np.float32)
    out["pcc_real_recomputed"] = real
    path = p["blind"] / "signatures.npz"
    np.savez(path, **out)
    record("signatures", geo, {"series": geo, "cells": int(x.shape[0]), "genes": int(x.shape[1]), "universe": int(len(universe)),
                               "target_library_size": target, "scaffold_genes": int(len(cols)),
                               "max_abs_diff_pcc_vs_primitive": float(np.max(np.abs(real - stored))),
                               "seconds": round(time.time() - t0, 1), "sha256": sha256(path)})


def thin(geo, procs):
    t0 = time.time()
    p = paths(geo)
    x, _ = load_counts(geo)
    sid = SID[geo]
    out, kept = {}, {}
    for depth in (THIN_PRIMARY, *THIN_SENS):
        keep, xt = thin_counts(x, depth, (4, sid, depth))
        out[f"keep_{depth}"] = keep
        out[f"genes_{depth}"] = np.asarray((xt > 0).sum(axis=1)).ravel().astype(np.float64)
        kept[depth] = int(len(keep))
        if depth == THIN_PRIMARY:
            xn, _ = lognorm(xt)
            gcs, _ = gene_count_signature(xn, out[f"genes_{depth}"])
            out["gcs_thinned"] = gcs
            out["cytotrace_thinned"] = cytotrace_pipeline(xn, gcs[:, None], procs, f"{geo}_thin")[:, 0]
            xk = x.tocsr()[keep].astype(np.float64)      # the same cells at raw depth
            xn, _ = lognorm(xk)
            gck = np.asarray((xk > 0).sum(axis=1)).ravel().astype(np.float64)
            gcs, _ = gene_count_signature(xn, gck)
            out["gcs_kept_raw"] = gcs
            out["cytotrace_kept_raw"] = cytotrace_pipeline(xn, gcs[:, None], procs, f"{geo}_kept")[:, 0]
    path = p["blind"] / "thinned.npz"
    np.savez(path, **out)
    record("thin", geo, {"series": geo, "cells": int(x.shape[0]), "kept": kept,
                         "seconds": round(time.time() - t0, 1), "sha256": sha256(path)})


def manifest(geo):
    p = paths(geo)
    out = json.loads(MANIFEST.read_text()) if MANIFEST.is_file() else {}
    out = {k: v for k, v in out.items() if not k.startswith(geo + "/")}
    files = {"primitives.npz": p["prep"] / "primitives.npz", "scores.npz": p["blind"] / "scores.npz",
             "signatures.npz": p["blind"] / "signatures.npz", "thinned.npz": p["blind"] / "thinned.npz"}
    for name, path in files.items():
        z = np.load(path)
        shapes = {k: list(z[k].shape) for k in z.files}
        if name == "signatures.npz" and (shapes["a1_g200"][1], shapes["a2"][1]) != (M_A1, M_A2):
            raise SystemExit("signatures.npz does not hold the registered number of controls")
        out[f"{geo}/{name}"] = {"sha256": sha256(path), "bytes": path.stat().st_size, "shapes": shapes}
    lab = p["labels"]                                       # hashed as bytes; its content is not parsed here
    out[f"{geo}/labels.tsv"] = {"sha256": sha256(lab), "bytes": lab.stat().st_size, "shapes": {}}
    MANIFEST.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for k in sorted(out):
        print(out[k]["sha256"], k)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["scores", "signatures", "thin", "manifest"])
    ap.add_argument("series", choices=sorted(SID))
    ap.add_argument("--procs", type=int, default=8)
    a = ap.parse_args()
    paths(a.series)["blind"].mkdir(parents=True, exist_ok=True)
    {"scores": lambda: scores(a.series, a.procs), "signatures": lambda: signatures(a.series),
     "thin": lambda: thin(a.series, a.procs), "manifest": lambda: manifest(a.series)}[a.mode]()


if __name__ == "__main__":
    main()
