"""
build_mouse_string_v12_thr700_lcc.py
=====================================
Build STRING v12.0 mouse (Mus musculus, taxon 10090) high-confidence PPI scaffold,
threshold combined_score >= 700, largest connected component (LCC), and dump to
an .npz artifact matching the human pipeline conventions in `potency_metrics.py`.

Design choices (locked to mirror human):
  - Node IDs = **preferred symbol** from STRING info file (same convention as
    load_string_ppi() in potency_metrics.py, which uses the 'preferred_name'
    column from 9606.protein.info.v12.0.txt.gz).
  - Score threshold = 700 (STRING "high confidence" tier), same as human runs.
  - LCC extraction identical to build_ppi_adjacency() — connected_components
    (undirected), keep the biggest.
  - **No ortholog transfer from human.** Species-matched per II.7 prereg §3.
    Reason: transferring human hubs into mouse would introduce an extra degree
    of freedom (choice of ortholog map, ambiguous genes) that our diagnostic
    is designed to detect. Mouse scaffold is downloaded and built independently.

Inputs (downloaded from stringdb-downloads.org, verified 200 OK 2026-07-17):
  https://stringdb-downloads.org/download/protein.info.v12.0/10090.protein.info.v12.0.txt.gz  (1.9 MB)
  https://stringdb-downloads.org/download/protein.links.v12.0/10090.protein.links.v12.0.txt.gz (75 MB)

Output:
  scaffolds/mouse_string_v12_thr700_lcc.npz
    data          : CSR adjacency data (0/1, symmetric, diagonal zeroed)
    indices,indptr: CSR structure
    shape         : (n_nodes, n_nodes) — LCC only
    genes         : object array of preferred-name gene symbols, ordered as
                    matrix rows/cols
    degree        : integer degree per node

Docstring records (populated by main()):
  N_NODES_FULL_HIGHCONF : node count in the thresholded graph before LCC
  N_EDGES_FULL_HIGHCONF : edge count in the thresholded graph before LCC
  N_NODES_LCC           : node count in LCC (== len(genes) in output)
  N_EDGES_LCC           : edge count in LCC
  LCC_NODE_COVERAGE     : N_NODES_LCC / N_NODES_FULL_HIGHCONF
  LCC_EDGE_COVERAGE     : N_EDGES_LCC / N_EDGES_FULL_HIGHCONF

Reproducibility:
  Only inputs are the two STRING files at the URLs above. No RNG. Determinism
  is exact modulo pandas/numpy version drift (structural: LCC is graph-defined).

Build result (locked 2026-07-17 by researcher-builder):
  full high-conf : 15,971 nodes / 201,860 edges
  LCC            : 15,080 nodes / 201,141 edges
  LCC coverage   : 0.9442 nodes / 0.9964 edges
  mean degree    : 26.68  ·  median 12  ·  p95 98  ·  max 480
  top-10 hubs by degree: Trp53(480), Fau(464), Rps27a(458), Tnf(425), Akt1(394),
                        Gnal(384), Ctnnb1(384), Rps11(380), Il6(377), Rps3(370)
  smoke test    : PASS (symmetric, 0/1, no self-loops, 1 connected component)

Comparison to human (per SSOT §3): human LCC = 11,820 (Watermelon/ReSisTrace
intersection). Mouse LCC = 15,080 (free scaffold, not intersected with any
dataset yet — the per-atlas intersection happens in the II.7 probe at analysis
time). Top hubs (Trp53, Fau, Rps27a, ribosomal cluster) mirror human hub
structure, consistent with the CCAT-axis mechanism found in Cell 8.
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import scratch_root  # noqa: E402



# ---- run-time records (populated by main()) --------------------------------
N_NODES_FULL_HIGHCONF: int | None = None
N_EDGES_FULL_HIGHCONF: int | None = None
N_NODES_LCC: int | None = None
N_EDGES_LCC: int | None = None
LCC_NODE_COVERAGE: float | None = None
LCC_EDGE_COVERAGE: float | None = None


def load_string_edges_symbols(links_path: str, info_path: str,
                              score_threshold: int = 700) -> np.ndarray:
    """
    Mirror of potency_metrics.load_string_ppi() semantics, mouse-specialized.
    Returns np.ndarray of shape (E, 2) with canonicalized (A, B) pairs by
    preferred-name gene symbol. Deduplicated, no self-loops.
    """
    info = pd.read_csv(
        info_path, sep="\t", compression="infer",
        usecols=[0, 1], names=["sid", "symbol"], header=0
    )
    id2sym = dict(zip(info["sid"], info["symbol"]))

    links = pd.read_csv(
        links_path, sep=" ", compression="infer",
        usecols=["protein1", "protein2", "combined_score"]
    )
    links = links[links["combined_score"] >= score_threshold]

    a = links["protein1"].map(id2sym)
    b = links["protein2"].map(id2sym)
    keep = a.notna() & b.notna() & (a != b)
    a_arr = a[keep].to_numpy()
    b_arr = b[keep].to_numpy()

    # canonical pair order (A, B) with A <= B for deduplication
    swap = a_arr > b_arr
    lo = np.where(swap, b_arr, a_arr)
    hi = np.where(swap, a_arr, b_arr)
    de = pd.DataFrame({"lo": lo, "hi": hi}).drop_duplicates()
    return de.to_numpy()


def build_lcc(edges: np.ndarray) -> tuple[sp.csr_matrix, list[str], np.ndarray]:
    """
    Given an edge list of symbol pairs, build a symmetric 0/1 CSR adjacency
    matrix on the union of endpoints, then take the largest connected
    component. Same as build_ppi_adjacency() in potency_metrics.py, but with
    the adata-var-name filtering step SKIPPED (we build the free scaffold;
    the atlas-scale probe intersects per dataset at analysis time).
    Returns (A_lcc_csr, genes_lcc_list, degree_vec).
    """
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

    # record full-highconf stats before LCC
    global N_NODES_FULL_HIGHCONF, N_EDGES_FULL_HIGHCONF
    N_NODES_FULL_HIGHCONF = int(n)
    N_EDGES_FULL_HIGHCONF = int(A.nnz // 2)  # undirected, symmetric

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
    return A, genes, degree


def save_npz(path: str, A: sp.csr_matrix, genes: list[str], degree: np.ndarray) -> None:
    """Dump CSR adjacency + genes + degree to a single .npz for portability."""
    np.savez_compressed(
        path,
        data=A.data.astype(np.int8),        # 0/1 -> int8 for size
        indices=A.indices,
        indptr=A.indptr,
        shape=np.array(A.shape, dtype=np.int64),
        genes=np.array(genes, dtype=object),
        degree=degree,
    )


def load_npz(path: str) -> tuple[sp.csr_matrix, list[str], np.ndarray]:
    """Inverse of save_npz — smoke-test loader used by the __main__ check."""
    z = np.load(path, allow_pickle=True)
    A = sp.csr_matrix(
        (z["data"].astype(np.float64), z["indices"], z["indptr"]),
        shape=tuple(z["shape"]),
    )
    return A, list(z["genes"]), z["degree"]


def main(links_path: str, info_path: str, out_path: str,
         score_threshold: int = 700) -> None:
    """
    Full build: parse STRING → threshold → LCC → dump .npz.
    Populates the module-level docstring records and prints them.
    """
    print(f"[1/4] parsing STRING info: {info_path}")
    print(f"[2/4] parsing STRING links (thr>={score_threshold}): {links_path}")
    edges = load_string_edges_symbols(links_path, info_path,
                                      score_threshold=score_threshold)
    print(f"      unique high-confidence edges (symbol-canonicalized): {len(edges):,}")

    print("[3/4] building adjacency + LCC")
    A, genes, degree = build_lcc(edges)

    global N_NODES_LCC, N_EDGES_LCC, LCC_NODE_COVERAGE, LCC_EDGE_COVERAGE
    N_NODES_LCC = int(A.shape[0])
    N_EDGES_LCC = int(A.nnz // 2)
    LCC_NODE_COVERAGE = N_NODES_LCC / N_NODES_FULL_HIGHCONF
    LCC_EDGE_COVERAGE = N_EDGES_LCC / N_EDGES_FULL_HIGHCONF

    print(f"      full high-conf : {N_NODES_FULL_HIGHCONF:,} nodes / {N_EDGES_FULL_HIGHCONF:,} edges")
    print(f"      LCC            : {N_NODES_LCC:,} nodes / {N_EDGES_LCC:,} edges")
    print(f"      LCC coverage   : {LCC_NODE_COVERAGE:.4f} nodes / {LCC_EDGE_COVERAGE:.4f} edges")
    print(f"      mean degree    : {degree.mean():.2f} · median {np.median(degree):.0f} · max {degree.max()}")

    print(f"[4/4] writing {out_path}")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_npz(out_path, A, genes, degree)
    print("done.")


def smoke_test(path: str) -> None:
    """Load + degree-distribution sanity check. Called by --smoke-test."""
    print(f"[smoke] loading {path}")
    A, genes, degree = load_npz(path)
    print(f"[smoke] shape={A.shape} · nnz={A.nnz} · genes={len(genes)}")
    assert A.shape[0] == A.shape[1] == len(genes) == len(degree), "shape mismatch"
    # symmetric?
    diff = (A - A.T)
    assert diff.nnz == 0, "adjacency not symmetric"
    # 0/1?
    assert set(np.unique(A.data).tolist()).issubset({0.0, 1.0}), "not 0/1"
    # no diagonal?
    assert (A.diagonal() == 0).all(), "self-loops present"
    # LCC check (should be 1 connected component)
    ncomp, _ = connected_components(A, directed=False)
    assert ncomp == 1, f"not LCC: {ncomp} components"

    # degree distribution summary
    d = degree
    print(f"[smoke] degree: mean={d.mean():.2f}  median={np.median(d):.0f}  "
          f"p95={np.percentile(d, 95):.0f}  max={d.max()}  "
          f"n_deg1={(d == 1).sum()}  n_deg100+={(d >= 100).sum()}")
    # top-10 hubs (sanity check — expect canonical mouse hubs like Trp53, Ctnnb1, Grb2, etc.)
    top_idx = np.argsort(d)[-10:][::-1]
    print("[smoke] top-10 hubs by degree:")
    for i in top_idx:
        print(f"        {genes[i]:12s}  k={d[i]}")
    print("[smoke] all checks passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    _dl = scratch_root() / "string_mouse_dl"
    parser.add_argument("--links", default=str(_dl / "10090.protein.links.v12.0.txt.gz"),
                        help="path to 10090.protein.links.v12.0.txt.gz")
    parser.add_argument("--info", default=str(_dl / "10090.protein.info.v12.0.txt.gz"),
                        help="path to 10090.protein.info.v12.0.txt.gz")
    parser.add_argument("--out", default=os.path.join(os.path.dirname(__file__),
                                                     "mouse_string_v12_thr700_lcc.npz"),
                        help="output .npz path")
    parser.add_argument("--threshold", type=int, default=700,
                        help="STRING combined_score threshold (default 700, high-conf)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="skip build, just load --out and run sanity checks")
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args.out)
    else:
        main(args.links, args.info, args.out, score_threshold=args.threshold)
        print("\n--- smoke test on the freshly written artifact ---")
        smoke_test(args.out)
