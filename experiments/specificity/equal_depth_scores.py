#!/usr/bin/env python3
"""
equal_depth_scores.py: signalling entropy, CCAT and MCE recomputed at 1,000 counts per cell.

    python3 experiments/specificity/equal_depth_scores.py blind      # reads no label
    python3 experiments/specificity/equal_depth_scores.py analyse    # after PREREG_depth_3.md is committed

Plan: PREREG_depth_3.md. `blind` rebuilds the interaction network of each staged series from the
deposit's STRING scaffold, checks on 10,000 cells at raw depth that the score code returns the
deposit's own SR and CCAT (and MCE on 300 cells), then scores the matrix that controls.py's
thinning draw produces. `analyse` refuses to run unless the plan, this file and the manifest are
committed unchanged and the score files match the manifest.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import gzip
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DEPOSIT, REPO, SERIES, WORK, load_cells, load_genes, load_matrix, num, rng_for, sha256, table  # noqa: E402
from controls import thin_counts  # noqa: E402
from stats import conditional_skill  # noqa: E402

sys.path.insert(0, str(REPO / "scores"))
import mce as mce_port  # noqa: E402

from own_baseline import potency_metrics as pm  # noqa: E402
from own_baseline.conditional_null import cs_value, m_for, permute_within, strata_1d  # noqa: E402

SCORE_FILE = {"GSE106474": "GSE106474_scores", "GSE112294": "GSE112294_scores", "GSE113074": "GSE113074_audit_scores"}
MANIFEST = HERE / "blind_manifest_depth_3.json"
OWN = ("PREREG_depth_3.md", "equal_depth_scores.py", "blind_manifest_depth_3.json")
CHUNK = 2000
MCE_SERIES = ("GSE106474", "GSE112294")


def network(geo, genes):
    """The largest connected component of the scaffold on the genes of the series, as SCENT takes it."""
    upper = geo in ("GSE113074", "GSE112294")
    col = {}
    for j, g in enumerate(genes):
        col.setdefault(g.upper() if upper else g, j)
    with gzip.open(DEPOSIT / "scaffold" / "string_v12_human_thr700_lcc_edges.tsv.gz", "rt") as fh:
        fh.readline()
        edges = [ln.rstrip("\n").split("\t")[:2] for ln in fh]
    edges = [(a, b) for a, b in edges if a in col and b in col and a != b]
    names = sorted({g for e in edges for g in e})
    idx = {g: i for i, g in enumerate(names)}
    r = np.array([idx[a] for a, _ in edges])
    c = np.array([idx[b] for _, b in edges])
    a = sp.coo_matrix((np.ones(len(edges)), (r, c)), shape=(len(names), len(names)))
    a = ((a + a.T) > 0).astype(np.float64).tocsr()
    ncomp, labels = connected_components(a, directed=False)
    sel = np.flatnonzero(labels == np.argmax(np.bincount(labels)))
    a = a[sel][:, sel].tocsr()
    cols = np.array([col[names[i]] for i in sel])
    return a, cols, np.asarray(a.sum(axis=1)).ravel(), pm._max_entropy_rate(a)


def scent(x, cells, a, cols, degree, max_sr):
    """SR and CCAT of the given cells, with the transforms of the audit's runner."""
    lib = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    target = float(np.median(lib[lib > 0]))
    sr, ccat = np.empty(len(cells)), np.empty(len(cells))
    for s0 in range(0, len(cells), CHUNK):
        i = cells[s0:s0 + CHUNK]
        cpm = x[i][:, cols].toarray().astype(np.float64) / np.where(lib[i] > 0, lib[i], 1.0)[:, None] * target
        ccat[s0:s0 + CHUNK] = pm._ccat_from_matrix(np.log2(cpm + 1.0), degree)
        sr[s0:s0 + CHUNK] = pm._sr_from_matrix(np.log2(cpm + 1.1), a, max_sr, chunk=CHUNK)
    return sr, ccat


_NET = {}


def _mce_init(a):
    _NET["a"] = a


def _mce_block(xn):
    return mce_port.mce(xn, _NET["a"])


def mce_scores(x, cells, a, cols, procs=5):
    """MCE on signalling entropy's input, log2(x + 1.1) of counts scaled to the median library size."""
    lib = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    target = float(np.median(lib[lib > 0]))

    def blocks():
        for s0 in range(0, len(cells), 250):
            i = cells[s0:s0 + 250]
            yield np.log2(x[i][:, cols].toarray().astype(np.float64) / np.where(lib[i] > 0, lib[i], 1.0)[:, None] * target + 1.1)

    with Pool(procs, initializer=_mce_init, initargs=(a,)) as pool:
        return np.concatenate(list(pool.imap(_mce_block, blocks(), chunksize=1)))


def blind():
    record = {}
    for geo in SERIES:
        t0 = time.time()
        sid = SERIES.index(geo)
        x = load_matrix(geo).tocsr()
        a, cols, degree, max_sr = network(geo, load_genes(geo))
        stored = table(DEPOSIT / "scores" / f"{SCORE_FILE[geo]}.csv.gz")
        check = np.sort(rng_for(50, sid).permutation(x.shape[0])[:10_000])
        sr, ccat = scent(x, check, a, cols, degree, max_sr)
        fid = {"network_genes": int(len(cols)),
               "sr_spearman_vs_deposit": float(spearmanr(sr, num(stored["scent_sr"])[check]).statistic),
               "sr_max_abs_diff": float(np.max(np.abs(sr - num(stored["scent_sr"])[check]))),
               "ccat_spearman_vs_deposit": float(spearmanr(ccat, num(stored["scent_ccat"])[check]).statistic),
               "ccat_max_abs_diff": float(np.max(np.abs(ccat - num(stored["scent_ccat"])[check])))}
        keep, xt = thin_counts(x, 1000, (4, sid, 1000))            # the draw of controls.py thin
        out = {"keep_1000": keep}
        out["sr_1000"], out["ccat_1000"] = scent(xt.tocsr(), np.arange(len(keep)), a, cols, degree, max_sr)
        if geo in MCE_SERIES:
            sub = check[:300]
            m = mce_scores(x, sub, a, cols)
            fid["mce_spearman_vs_deposit"] = float(spearmanr(m, num(stored["mce"])[sub]).statistic)
            fid["mce_max_abs_diff"] = float(np.max(np.abs(m - num(stored["mce"])[sub])))
            out["mce_1000"] = mce_scores(xt.tocsr(), np.arange(len(keep)), a, cols)
        path = WORK / f"{geo}_scores_depth3.npz"
        np.savez(path, **out)
        record[path.name] = {"sha256": sha256(path), "shapes": {k: list(v.shape) for k, v in out.items()},
                             "fidelity_at_raw_depth": fid, "seconds": round(time.time() - t0, 1)}
        print(geo, json.dumps(record[path.name]["fidelity_at_raw_depth"]), record[path.name]["seconds"], flush=True)
        MANIFEST.write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")


def locked():
    rel = HERE.relative_to(REPO).as_posix()
    paths = [f"{rel}/{p}" for p in OWN]
    if subprocess.run(["git", "-C", str(REPO), "diff", "--quiet", "HEAD", "--", *paths]).returncode != 0:
        raise SystemExit("the plan, this script or the manifest differ from the commit")
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "--error-unmatch", *paths], capture_output=True)
    if out.returncode != 0:
        raise SystemExit("the plan, this script and the manifest must be committed first")
    m = json.loads(MANIFEST.read_text())
    for name, rec in m.items():
        if sha256(WORK / name) != rec["sha256"]:
            raise SystemExit(f"{name} does not match the committed manifest")
    return subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def analyse():
    res = {"locked_at_commit": locked(), "series": {}}
    for geo in SERIES:
        sid = SERIES.index(geo)
        o = num(load_cells(geo)["ordinal"])
        stored = table(DEPOSIT / "scores" / f"{SCORE_FILE[geo]}.csv.gz")
        z = np.load(WORK / f"{geo}_scores_depth3.npz")
        pz = np.load(WORK / f"{geo}_pcc_depth2.npz")
        k = z["keep_1000"]
        assert np.array_equal(k, pz["keep_1000"])
        p, ok = pz["pcc_1000"], o[k]
        labels = strata_1d(p, m_for(len(k)), rng_for(51, sid))
        rows = {}
        for name, key, col in (("SR", "sr_1000", "scent_sr"), ("CCAT", "ccat_1000", "scent_ccat"), ("MCE", "mce_1000", "mce")):
            if key not in z.files:
                continue
            s = z[key]
            cs = conditional_skill(s, [p], ok)
            d = np.array([cs_value(permute_within(s, labels, rng_for(52, sid, i)), [p], labels, ok, "linear") for i in range(200)])
            rows[name] = {"cells": int(len(k)),
                          "spearman_with_stage_raw": float(spearmanr(num(stored[col]), o).statistic),
                          "spearman_with_stage_at_1000": float(spearmanr(s, ok).statistic),
                          "spearman_with_primitive_at_1000": float(spearmanr(s, p).statistic),
                          "spearman_raw_score_with_score_at_1000": float(spearmanr(num(stored[col])[k], s).statistic),
                          "conditional_skill_at_1000": cs, "permutation_floor": float(np.quantile(d, 0.975)),
                          "clears": bool(cs > np.quantile(d, 0.975))}
            print(geo, name, json.dumps(rows[name]), flush=True)
        res["series"][geo] = rows
    s = res["series"]
    res["predictions"] = {
        "V1": {"held": all(s[g]["SR"]["spearman_with_stage_at_1000"] >= 0.1 for g in SERIES),
               "observed": {g: s[g]["SR"]["spearman_with_stage_at_1000"] for g in SERIES}},
        "V2": {"held": all(s[g]["SR"]["clears"] for g in SERIES),
               "observed": {g: s[g]["SR"]["conditional_skill_at_1000"] for g in SERIES}},
        "V3": {"held": all(s[g]["CCAT"]["conditional_skill_at_1000"] < 0 for g in SERIES),
               "observed": {g: s[g]["CCAT"]["conditional_skill_at_1000"] for g in SERIES}}}
    if all("MCE" in s[g] for g in MCE_SERIES):
        res["predictions"]["V4"] = {"held": all(s[g]["MCE"]["conditional_skill_at_1000"] < 0 for g in MCE_SERIES),
                                    "observed": {g: s[g]["MCE"]["conditional_skill_at_1000"] for g in MCE_SERIES}}
    (HERE / "results" / "depth_3.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res["predictions"]), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["blind", "analyse"])
    blind() if ap.parse_args().mode == "blind" else analyse()
