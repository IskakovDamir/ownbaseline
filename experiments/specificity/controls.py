#!/usr/bin/env python3
"""
controls.py: negative-control scores, computed without reading a stage label.

    python3 experiments/specificity/controls.py signatures   # families A1 and A2
    python3 experiments/specificity/controls.py cytotrace    # family A3 and the pipeline check
    python3 experiments/specificity/controls.py thin         # depth-equalised gene counts and CytoTRACE

A negative control is a per-cell statistic nobody would call a potency score.

  A1  the mean log-normalised expression of a random gene set (g = 200 primary; 20 and
      2,000 as sensitivity), drawn uniformly from genes detected in at least 10 cells;
  A2  PCC(x, degree) with the degree vector permuted among the scaffold genes, the
      degree-correlation primitive on a network that carries no information;
  A3  CytoTRACE v1 with its 200 gene-count-correlated genes replaced by 200 random genes,
      every later step (similarity matrix, non-negative least squares, diffusion, chunking
      seed) unchanged.

`thin` draws every cell's counts down to a common depth without replacement and recounts
detected genes; at the primary depth it also reruns CytoTRACE v1 on the thinned matrix, and on
the same cells at raw depth.

Each mode writes an .npz under the scratch root. manifest.py then records the SHA256 and the
array shapes of every file in blind_manifest.json, which is committed with PREREG.md before the
analysis stage reads a label.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker: the pool is the parallelism

import argparse
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (REPO, SERIES, WORK, load_cells, load_degree, load_genes, load_matrix, lognorm,  # noqa: E402
                    num, rng_for, scaffold_columns, sha256, table, DEPOSIT)

sys.path.insert(0, str(REPO / "scores"))
from cytotrace_full import (DIFF_ALPHA, DIFF_MAX_ITER, DIFF_TOL, GCS_TOP_N, MVG_TOP_N,  # noqa: E402
                            SUBSAMPLE_SEED, SUBSAMPLE_SIZE, diffusion_smooth, most_variable_genes,
                            nnls_regression, rank_and_scale_01, similarity_matrix_cleaned)

M_PRIMARY, G_PRIMARY = 1000, 200
SENS = ((20, 200), (2000, 200))          # (genes per set, number of sets)
M_DEGREE = 1000
B_CYTOTRACE = 100
MIN_CELLS = 10
THIN_PRIMARY, THIN_SENS = 1000, (500, 2000)


def gene_set_scores(xn, universe, g, m, key):
    """Mean log-normalised expression over m random sets of g genes. Returns (n x m float32, m x g indices)."""
    sets = np.stack([rng_for(*key, j).choice(universe, g, replace=False) for j in range(m)])
    w = sp.csr_matrix((np.full(m * g, 1.0 / g), (sets.ravel(), np.repeat(np.arange(m), g))),
                      shape=(xn.shape[1], m))
    return np.asarray((xn @ w).todense(), dtype=np.float32), sets


def pcc_degree(xs, deg):
    """Per cell, the Pearson correlation of expression with each column of deg (genes x m)."""
    m = xs.shape[1]
    dc = deg - deg.mean(axis=0, keepdims=True)
    numer = xs @ dc
    sx = np.asarray(xs.sum(axis=1)).ravel()
    sxx = np.asarray(xs.multiply(xs).sum(axis=1)).ravel()
    den_x = np.sqrt(np.maximum(sxx - sx * sx / m, 0.0))
    den = den_x[:, None] * np.sqrt((dc * dc).sum(axis=0))[None, :]
    return np.where(den > 0, numer / np.where(den > 0, den, 1.0), 0.0)


def signatures(geo):
    t0 = time.time()
    x = load_matrix(geo)
    genes = load_genes(geo)
    xn, target = lognorm(x)
    detected_in = np.asarray((x > 0).sum(axis=0)).ravel()
    universe = np.flatnonzero(detected_in >= MIN_CELLS)
    sid = SERIES.index(geo)
    out = {"universe": universe}
    out["a1_g200"], out["a1_g200_sets"] = gene_set_scores(xn, universe, G_PRIMARY, M_PRIMARY, (1, sid, G_PRIMARY))
    for g, m in SENS:
        out[f"a1_g{g}"], out[f"a1_g{g}_sets"] = gene_set_scores(xn, universe, g, m, (1, sid, g))
    cols, deg = scaffold_columns(geo, genes, load_degree())
    xs = xn[:, cols].tocsr()
    real = pcc_degree(xs, deg[:, None])[:, 0]
    stored = num(load_cells(geo)["pcc_x_degree"])
    check = float(np.max(np.abs(real - stored)))
    perms = np.stack([rng_for(2, sid, j).permutation(deg) for j in range(M_DEGREE)], axis=1)
    out["a2"] = pcc_degree(xs, perms).astype(np.float32)
    out["pcc_real_recomputed"] = real
    path = WORK / f"{geo}_signatures.npz"
    np.savez(path, **out)
    meta = {"series": geo, "cells": int(x.shape[0]), "genes": int(x.shape[1]), "universe": int(len(universe)),
            "target_library_size": target, "scaffold_genes": int(len(cols)),
            "max_abs_diff_pcc_vs_deposit": check, "seconds": round(time.time() - t0, 1),
            "file": path.name, "sha256": sha256(path)}
    print(json.dumps(meta), flush=True)
    return meta


# ---------------------------------------------------------------------------------------
# A3: CytoTRACE v1 with the gene-count signature swapped for a random 200-gene signature
# ---------------------------------------------------------------------------------------
def _ct_chunk(args):
    idx, xsub, vals = args
    logn = np.asarray(xsub.todense(), dtype=np.float64)
    mvg = most_variable_genes(logn, top_n=MVG_TOP_N)
    d = similarity_matrix_cleaned(logn[:, mvg], variant="rfaithful")
    del logn                                   # the regression loop below needs only d
    out = np.empty(vals.shape)
    for j in range(vals.shape[1]):
        nn = nnls_regression(d, vals[:, j])
        out[:, j], _ = diffusion_smooth(d, nn, alpha=DIFF_ALPHA, max_iter=DIFF_MAX_ITER, tol=DIFF_TOL)
    return idx, out


def cytotrace_pipeline(xn, vals, procs, tag=None):
    """Steps 3 and 4 of CytoTRACE v1 for each column of vals, chunked exactly as the audit's runner."""
    n = xn.shape[0]
    perm = np.random.default_rng(SUBSAMPLE_SEED).permutation(n)
    chunks = np.array_split(perm, max(1, int(round(n / SUBSAMPLE_SIZE))))
    xn = xn.tocsr()
    diffused = np.full((n, vals.shape[1]), np.nan)
    jobs = ((idx, xn[idx, :], vals[idx]) for idx in chunks)     # each worker gets only its own chunk
    with Pool(procs) as pool:
        for done, (idx, out) in enumerate(pool.imap_unordered(_ct_chunk, jobs, chunksize=1), 1):
            diffused[idx] = out
            if done % 10 == 0:
                print(f"  {tag}: chunk {done}/{len(chunks)}", file=sys.stderr, flush=True)
    return np.column_stack([rank_and_scale_01(diffused[:, j]) for j in range(vals.shape[1])])


def gene_count_signature(xn, gc):
    """The 200 genes most correlated with the gene count, and exp of their mean log expression."""
    n = xn.shape[0]
    sum_x = np.asarray(xn.sum(axis=0)).ravel()
    sum_x2 = np.asarray(xn.multiply(xn).sum(axis=0)).ravel()
    sum_xy = np.asarray(xn.T @ gc).ravel()
    sgc, sgc2 = gc.sum(), (gc * gc).sum()
    numer = n * sum_xy - sum_x * sgc
    den = np.sqrt((n * sum_x2 - sum_x ** 2) * (n * sgc2 - sgc ** 2))
    r = np.where(den > 0, numer / np.where(den > 0, den, 1.0), 0.0)
    top = np.argsort(r)[-GCS_TOP_N:]
    return np.exp(np.asarray(xn[:, top].todense()).mean(axis=1)).ravel(), top


def cytotrace(geo, procs, b):
    t0 = time.time()
    x = load_matrix(geo).astype(np.float64)
    xn, _ = lognorm(x)
    gc = np.asarray((x > 0).sum(axis=1)).ravel().astype(np.float64)
    gcs, top = gene_count_signature(xn, gc)
    universe = np.flatnonzero(np.asarray((x > 0).sum(axis=0)).ravel() >= MIN_CELLS)
    sid = SERIES.index(geo)
    sets = np.stack([rng_for(3, sid, j).choice(universe, GCS_TOP_N, replace=False) for j in range(b)])
    raw = np.column_stack([np.exp(np.asarray(xn[:, s].todense()).mean(axis=1)).ravel() for s in sets])
    vals = np.column_stack([gcs, raw])
    del x
    scored = cytotrace_pipeline(xn, vals, procs, geo)
    stored = num(table(DEPOSIT / "scores" / (f"{geo}_audit_scores.csv.gz" if geo == "GSE113074" else f"{geo}_scores.csv.gz"))["cytotrace_v1"])
    from scipy.stats import spearmanr
    agree = float(spearmanr(scored[:, 0], stored).statistic)
    path = WORK / f"{geo}_cytotrace_controls.npz"
    np.savez(path, ct_real_rerun=scored[:, 0], gcs_rerun=gcs, a3=scored[:, 1:].astype(np.float32),
             a3_unsmoothed=raw.astype(np.float32), a3_sets=sets, gcs_genes=top)
    meta = {"series": geo, "controls": b, "spearman_rerun_vs_deposit_cytotrace": agree,
            "max_abs_diff_rerun_vs_deposit": float(np.max(np.abs(scored[:, 0] - stored))),
            "seconds": round(time.time() - t0, 1), "file": path.name, "sha256": sha256(path)}
    print(json.dumps(meta), flush=True)
    return meta


# ---------------------------------------------------------------------------------------
# Depth equalisation
# ---------------------------------------------------------------------------------------
def thin_counts(x, depth, key):
    """Each cell with at least `depth` counts is drawn down to exactly `depth`, without replacement."""
    x = x.tocsr()
    lib = np.asarray(x.sum(axis=1)).ravel()
    keep = np.flatnonzero(lib >= depth)
    rng = rng_for(*key)
    indptr = [0]
    indices, data = [], []
    for c in keep:
        seg = slice(x.indptr[c], x.indptr[c + 1])
        counts = x.data[seg].astype(np.int64)
        drawn = rng.multivariate_hypergeometric(counts, depth)
        nz = drawn > 0
        indices.append(x.indices[seg][nz])
        data.append(drawn[nz])
        indptr.append(indptr[-1] + int(nz.sum()))
    thinned = sp.csr_matrix((np.concatenate(data).astype(np.float64), np.concatenate(indices), np.array(indptr)),
                            shape=(len(keep), x.shape[1]))
    return keep, thinned


def thin(geo, procs):
    t0 = time.time()
    x = load_matrix(geo)
    sid = SERIES.index(geo)
    out, meta = {}, {"series": geo, "cells": int(x.shape[0]), "kept": {}}
    for depth in (THIN_PRIMARY, *THIN_SENS):
        keep, xt = thin_counts(x, depth, (4, sid, depth))
        out[f"keep_{depth}"] = keep
        out[f"genes_{depth}"] = np.asarray((xt > 0).sum(axis=1)).ravel().astype(np.float64)
        meta["kept"][depth] = int(len(keep))
        if depth == THIN_PRIMARY and geo != "GSE113074":
            xn, _ = lognorm(xt)
            gcs, _ = gene_count_signature(xn, out[f"genes_{depth}"])
            out["gcs_thinned"] = gcs
            out["cytotrace_thinned"] = cytotrace_pipeline(xn, gcs[:, None], procs, f"{geo}_thin")[:, 0]
            # the same cells at raw depth, so that dropping the shallow cells is not mistaken for thinning
            xk = x.tocsr()[keep].astype(np.float64)
            xn, _ = lognorm(xk)
            gck = np.asarray((xk > 0).sum(axis=1)).ravel().astype(np.float64)
            gcs, _ = gene_count_signature(xn, gck)
            out["gcs_kept_raw"] = gcs
            out["cytotrace_kept_raw"] = cytotrace_pipeline(xn, gcs[:, None], procs, f"{geo}_kept")[:, 0]
    path = WORK / f"{geo}_thinned.npz"
    np.savez(path, **out)
    meta.update({"seconds": round(time.time() - t0, 1), "file": path.name, "sha256": sha256(path)})
    print(json.dumps(meta), flush=True)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["signatures", "cytotrace", "thin"])
    ap.add_argument("--series", default=",".join(SERIES))
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--controls", type=int, default=B_CYTOTRACE)
    a = ap.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    metas = []
    for geo in a.series.split(","):
        if a.mode == "signatures":
            metas.append(signatures(geo))
        elif a.mode == "cytotrace":
            metas.append(cytotrace(geo, a.procs, a.controls))
        else:
            metas.append(thin(geo, a.procs))
    record = HERE_RESULTS / f"blind_{a.mode}.json"
    prior = json.loads(record.read_text()) if record.is_file() else {}
    prior.update({m["series"]: m for m in metas})
    record.write_text(json.dumps(prior, indent=1))


HERE_RESULTS = Path(__file__).resolve().parent / "results"

if __name__ == "__main__":
    main()
