"""
common.py: loaders and small helpers shared by the specificity experiments.

Everything here reads the data deposit (data/deposit/ownbaseline-data-v1) and the
package's own estimator. Nothing here reads a stage label except `load_cells`, which
returns the cell table. controls.py never touches its `ordinal`, `label` or `hpf` columns;
window.py reads `hpf` to select cells and nothing else.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from own_baseline.paths import scratch_root  # noqa: E402

DEPOSIT = REPO / "data" / "deposit" / "ownbaseline-data-v1"
WORK = scratch_root() / "specificity"
SEED = 20261122
SERIES = ("GSE106474", "GSE112294", "GSE113074")
JOINT = ("detected_genes", "pcc_x_degree", "shannon_entropy", "log10_library_size")


def rng_for(*key):
    return np.random.default_rng([SEED, *key])


def sha256(path, block=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def table(path):
    """A csv.gz as a dict of string columns, in file order."""
    with gzip.open(path, "rt", newline="") as fh:
        r = csv.reader(fh)
        head = next(r)
        rows = list(r)
    return {k: [row[i] for row in rows] for i, k in enumerate(head)}


def num(col):
    return np.array([float(x) if x != "" else np.nan for x in col])


def load_cells(geo):
    return table(DEPOSIT / "datasets" / geo / "cells.csv.gz")


def load_genes(geo):
    with gzip.open(DEPOSIT / "datasets" / geo / "features.tsv.gz", "rt") as fh:
        return [line.split("\t")[0] for line in fh.read().splitlines()]


def load_matrix(geo):
    """Counts as CSR, cells x genes, float32. Cached under the scratch root."""
    WORK.mkdir(parents=True, exist_ok=True)
    cache = WORK / f"{geo}_counts_csr.npz"
    if cache.is_file():
        return sp.load_npz(cache)
    from scipy.io import mmread
    m = mmread(str(DEPOSIT / "datasets" / geo / "matrix.mtx.gz"))  # genes x cells
    x = sp.csr_matrix(m.T, dtype=np.float32)
    x.sum_duplicates()
    sp.save_npz(cache, x, compressed=False)
    return x


def lognorm(x):
    """log1p of counts scaled to the median library size, zeros kept sparse. Returns (matrix, target)."""
    lib = np.asarray(x.sum(axis=1)).ravel().astype(np.float64)
    target = float(np.median(lib[lib > 0]))
    inv = np.where(lib > 0, target / lib, 0.0)
    out = x.astype(np.float64).tocsr(copy=True)
    out.data = np.log1p(out.data * np.repeat(inv, np.diff(out.indptr)))
    return out, target


def load_degree():
    """STRING v12 degree per gene symbol, as shipped in the deposit."""
    with gzip.open(DEPOSIT / "scaffold" / "string_v12_human_thr700_lcc_degree.tsv.gz", "rt") as fh:
        head = fh.readline().rstrip("\n").split("\t")
        rows = [line.rstrip("\n").split("\t") for line in fh]
    gi = head.index("gene") if "gene" in head else 0
    di = head.index("degree") if "degree" in head else 1
    return {r[gi]: float(r[di]) for r in rows}


def scaffold_columns(geo, genes, degree):
    """Column indices of the genes on the scaffold and their degrees, matched as the audit matched them."""
    upper = geo in ("GSE113074", "GSE112294")
    idx, deg, seen = [], [], set()
    for j, g in enumerate(genes):
        key = g.upper() if upper else g
        if key in degree and key not in seen:
            seen.add(key)
            idx.append(j)
            deg.append(degree[key])
    return np.array(idx), np.array(deg)
