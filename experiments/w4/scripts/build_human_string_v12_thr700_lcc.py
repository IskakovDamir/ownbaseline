"""
build_human_string_v12_thr700_lcc.py
=====================================
Build STRING v12.0 human (Homo sapiens, taxon 9606) high-confidence PPI scaffold,
threshold combined_score >= 700, largest connected component (LCC), and dump to
an .npz artifact for the W4 GATE 2 runs on independent atlases.

Identical logic to build_mouse_string_v12_thr700_lcc.py — different taxon.
No ortholog transfer. Human-specialized.

Inputs:
  9606.protein.info.v12.0.txt.gz
  9606.protein.links.v12.0.txt.gz
  (both from stringdb-downloads.org, fetched 2026-07-18)

Output:
  scaffolds/human_string_v12_thr700_lcc.npz
"""
from __future__ import annotations
import argparse
import os
import sys
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

# --- repo-relative I/O roots -------------------------------------------------
# The --out default was an absolute path under the author's home directory.
# Override with $OWNBASELINE_DATA_ROOT; default is <repo>/data/runs.
# See own_baseline/paths.py.
_sys_path_anchor = next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                        if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(_sys_path_anchor))
from own_baseline.paths import data_root  # noqa: E402



def load_string_edges_symbols(links_path, info_path, score_threshold=700):
    info = pd.read_csv(info_path, sep="\t", compression="infer",
                       usecols=[0, 1], names=["sid", "symbol"], header=0)
    id2sym = dict(zip(info["sid"], info["symbol"]))
    links = pd.read_csv(links_path, sep=" ", compression="infer",
                        usecols=["protein1", "protein2", "combined_score"])
    links = links[links["combined_score"] >= score_threshold]
    a = links["protein1"].map(id2sym)
    b = links["protein2"].map(id2sym)
    keep = a.notna() & b.notna() & (a != b)
    a_arr = a[keep].to_numpy()
    b_arr = b[keep].to_numpy()
    swap = a_arr > b_arr
    lo = np.where(swap, b_arr, a_arr)
    hi = np.where(swap, a_arr, b_arr)
    de = pd.DataFrame({"lo": lo, "hi": hi}).drop_duplicates()
    return de.to_numpy()


def build_lcc(edges):
    uniq = np.unique(edges)
    g2i = {g: i for i, g in enumerate(uniq)}
    n = len(uniq)
    rows = np.array([g2i[e0] for e0, e1 in edges], dtype=np.int64)
    cols = np.array([g2i[e1] for e0, e1 in edges], dtype=np.int64)
    data = np.ones(len(edges), dtype=np.float64)
    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    A = A + A.T
    A = (A > 0).astype(np.float64).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    n_full = int(n)
    m_full = int(A.nnz // 2)

    ncomp, labels = connected_components(A, directed=False)
    if ncomp > 1:
        sizes = np.bincount(labels)
        big = int(np.argmax(sizes))
        sel = np.where(labels == big)[0]
        A = A[sel][:, sel]
        genes = list(uniq[sel])
    else:
        genes = list(uniq)

    A = A.tocsr()
    degree = np.asarray(A.sum(axis=1), dtype=np.int64).ravel()
    return A, genes, degree, n_full, m_full


def save_npz(path, A, genes, degree):
    np.savez_compressed(
        path,
        data=A.data.astype(np.int8),
        indices=A.indices,
        indptr=A.indptr,
        shape=np.array(A.shape, dtype=np.int64),
        genes=np.array(genes, dtype=object),
        degree=degree,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--links", default="/tmp/string_human_dl/9606.protein.links.v12.0.txt.gz")
    parser.add_argument("--info", default="/tmp/string_human_dl/9606.protein.info.v12.0.txt.gz")
    parser.add_argument("--out", default=str(
        data_root() / "w4/scaffolds/human_string_v12_thr700_lcc.npz"))
    parser.add_argument("--threshold", type=int, default=700)
    args = parser.parse_args()

    print(f"[1/3] parsing STRING (thr>={args.threshold})")
    edges = load_string_edges_symbols(args.links, args.info, score_threshold=args.threshold)
    print(f"      canonicalized undirected edges: {len(edges):,}")

    print("[2/3] building adjacency + LCC")
    A, genes, degree, n_full, m_full = build_lcc(edges)
    print(f"      full high-conf : {n_full:,} nodes / {m_full:,} edges")
    print(f"      LCC            : {A.shape[0]:,} nodes / {A.nnz//2:,} edges")
    print(f"      mean deg = {degree.mean():.2f} · median = {int(np.median(degree))} · max = {int(degree.max())}")

    print(f"[3/3] writing {args.out}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    save_npz(args.out, A, genes, degree)
    top = np.argsort(degree)[-10:][::-1]
    print("top-10 hubs by degree:")
    for i in top:
        print(f"  {genes[i]:12s}  k={int(degree[i])}")
    print("done.")


if __name__ == "__main__":
    main()
