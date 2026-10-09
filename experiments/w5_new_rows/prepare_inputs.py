#!/usr/bin/env python3
"""
prepare_inputs.py — the expression, the four primitives and the ordinal of
each dataset a w5 row runs on, written once so the R score wrappers and the
Python decider read the same cells in the same order.

    python3 experiments/w5_new_rows/prepare_inputs.py [GSE106474 GSE117498 GSE125970 GSE113074]

For each dataset, under $OWNBASELINE_DATA_ROOT/w5_new_rows/inputs/<GSE>/:

  csc_i.bin, csc_p.bin, csc_x.bin, dims.json
        genes x cells raw counts, column-compressed, little-endian int32
        (i, p) and float64 (x), so R builds a dgCMatrix without parsing text
  genes.txt, cells.txt
  primitives.npz
        gene_count     detected genes per cell
        pcc_string_v12 Pearson(log1p(CPM), STRING v12 degree) over the genes in
                       both the dataset and the thr700 LCC scaffold, CPM target
                       sum = median library size
        shannon_H      Shannon entropy (natural log) of raw counts, all genes
        log10_lib      log10 of the library size
        ordinal        the dataset's existing ordinal, NaN where unranked
        potency_sign   +1 if higher ordinal = more potent, -1 if later stage
These are the definitions of experiments/track2/code/prep_data.py, which is
what experiments/fix3/code/prep_briggs.py and the joint residual of
experiments/track4/code/fix2_joint_primitive.py used.

Cells and genes per dataset follow the code path the audit already uses:
  GSE106474  track2 prepared matrix: 39,505 cells after the repo's QC, all genes;
             primitives from prepared/primitives.npz, ordinal rank_kimmel
  GSE117498  experiments/stemsc path: w4 load_c1, genes in >= 10 cells, all
             cells (the three unranked populations carry NaN ordinal)
  GSE125970  w4 load_c2, genes in >= 10 cells
  GSE113074  fix3 prepared matrix (Briggs 2018), primitives from its
             primitives.npz, ordinal rank_ordinal
  GSE117498h GSE117498 with the broad-gate files' make.names gene names merged
             into their hyphenated twins (the registered sensitivity input;
             see ds_gse117498h)
No score is computed here and no score meets a label.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "w4" / "scripts"))
from own_baseline.paths import data_root  # noqa: E402

OUT = data_root() / "w5_new_rows" / "inputs"
STRING = data_root() / "w4/scaffolds/human_string_v12_thr700_lcc.npz"
CHUNK = 4000


def write_csc(dest, X_cells_genes, genes, cells):
    """X is cells x genes; written as genes x cells CSC = cells x genes CSR."""
    dest.mkdir(parents=True, exist_ok=True)
    G = sp.csr_matrix(X_cells_genes)          # rows = cells -> CSC of the transpose
    G.sort_indices()
    G.indices.astype("<i4").tofile(dest / "csc_i.bin")
    G.indptr.astype("<i4").tofile(dest / "csc_p.bin")
    G.data.astype("<f8").tofile(dest / "csc_x.bin")
    (dest / "dims.json").write_text(json.dumps(
        {"n_genes": int(G.shape[1]), "n_cells": int(G.shape[0]), "nnz": int(G.nnz)}))
    (dest / "genes.txt").write_text("\n".join(map(str, genes)) + "\n")
    (dest / "cells.txt").write_text("\n".join(map(str, cells)) + "\n")


def primitives(X, genes):
    """prep_data.py's three primitives plus log10 library size, cells x genes X."""
    X = sp.csr_matrix(X, dtype=np.float64)
    n = X.shape[0]
    lib = np.asarray(X.sum(axis=1)).ravel()
    gc = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float64)
    shan = np.empty(n)
    for c in range(n):
        seg = X.data[X.indptr[c]:X.indptr[c + 1]]
        s = seg.sum()
        if s <= 0:
            shan[c] = np.nan
            continue
        p = seg / s
        shan[c] = -np.sum(p * np.log(p))
    d = np.load(STRING, allow_pickle=True)
    sidx = {str(g): i for i, g in enumerate(d["genes"])}
    deg = np.asarray(d["degree"], dtype=np.float64)
    ca, cs = [], []
    for i, g in enumerate(genes):
        j = sidx.get(str(g))
        if j is not None:
            ca.append(i)
            cs.append(j)
    ca, cs = np.array(ca), np.array(cs)
    k = deg[cs]
    kc = k - k.mean()
    kn = np.sqrt((kc ** 2).sum())
    target = float(np.median(lib[lib > 0]))
    Xs = X[:, ca].tocsr()
    pcc = np.empty(n)
    for s0 in range(0, n, CHUNK):
        s1 = min(s0 + CHUNK, n)
        b = Xs[s0:s1].toarray()
        tot = np.where(lib[s0:s1] > 0, lib[s0:s1], 1.0)[:, None]
        Xn = np.log1p(b / tot * target)
        Rc = Xn - Xn.mean(axis=1, keepdims=True)
        num = Rc @ kc
        den = np.sqrt((Rc ** 2).sum(axis=1)) * kn
        o = np.zeros(s1 - s0)
        ok = den > 0
        o[ok] = num[ok] / den[ok]
        pcc[s0:s1] = o
    meta = {"n_atlas_x_string": int(len(ca)), "target_sum_cpm": target}
    return gc, pcc, shan, np.log10(lib), meta


def ds_gse106474():
    P = data_root() / "track2/data/prepared"
    X = sp.load_npz(P / "X_cells_genes.npz").tocsr()
    genes = P.joinpath("genes.txt").read_text().split("\n")
    rows = list(csv.DictReader(open(P / "cells.tsv"), delimiter="\t"))
    cells = [r["cell_id"] for r in rows]
    p = np.load(P / "primitives.npz")
    lib = np.array([float(r["lib_size"]) for r in rows])
    prim = dict(gene_count=p["gene_count"], pcc_string_v12=p["pcc_string_v12"],
                shannon_H=p["shannon_H"], log10_lib=np.log10(lib),
                ordinal=p["rank_kimmel"].astype(float), potency_sign=-1.0)
    return X, genes, cells, prim, {"source": "track2/data/prepared (repo prep_data.py)",
                                   "ordinal": "rank_kimmel (1 = high stage ... 12 = 6-somite)"}


def _w4(loader, sub, min_cells):
    import w4_gate2_run as w4
    X, genes, cell_ids, labels, ranks = loader(data_root() / "w4" / "data" / sub)
    Xf, genes_f = w4.preprocess_and_filter(X, genes, min_cells)
    Xf = sp.csr_matrix(Xf)
    gc, pcc, shan, loglib, meta = primitives(Xf, genes_f)
    prim = dict(gene_count=gc, pcc_string_v12=pcc, shannon_H=shan, log10_lib=loglib,
                ordinal=np.asarray(ranks, dtype=float), potency_sign=1.0)
    return Xf, genes_f, list(cell_ids), prim, meta


def ds_gse117498():
    import w4_gate2_run as w4
    X, genes, cells, prim, meta = _w4(w4.load_c1, "c1_gse117498", w4.C1_GENE_FILTER_MIN_CELLS)
    meta.update({"source": "w4 load_c1 (GSE117498 GSM raw counts), genes in >= 10 cells",
                 "ordinal": "C1_POP_TO_RANK: HSC 5, MPP 4, CD34+/CD164+ 4, MLP 3, CMP 3, "
                            "GMP 2, MEP 2, PreB/NK 2; three CD34-/low populations unranked"})
    return X, genes, cells, prim, meta


def ds_gse117498h():
    """
    GSE117498 with its gene names harmonized: the registered sensitivity input.

    The seven sorted-population files name genes in HGNC form (HLA-A); the four
    broad-gate files carry R make.names forms of the same names (HLA.A). load_c1
    unions by exact name, so every such gene splits into two columns, one zero
    in every broad-gate cell and the other zero in every sorted cell. Here a
    broad-gate name Y is merged into the sorted-file name X when X contains a
    hyphen, X with every hyphen replaced by a dot equals Y, and Y is not itself
    a sorted-file name; the two columns are summed. Everything else is the
    stemsc code path (load_c1, genes in >= 10 cells after the merge).
    """
    import gzip
    import w4_gate2_run as w4
    d = data_root() / "w4" / "data" / "c1_gse117498"
    sorted_genes, broad_genes = set(), set()
    for fname, pop in w4.C1_GSM_FILES.items():
        with gzip.open(d / fname, "rt") as fh:
            fh.readline()
            names = {line.split("\t", 1)[0] for line in fh} - {"Library"}
        (sorted_genes if not pop.startswith("LinNeg") else broad_genes).update(names)
    pairs = {}
    for x in sorted_genes:
        if "-" in x:
            y = x.replace("-", ".")
            if y in broad_genes and y not in sorted_genes:
                pairs[y] = x
    X, genes, cell_ids, labels, ranks = w4.load_c1(d)
    X = sp.csc_matrix(X)
    gidx = {g: i for i, g in enumerate(genes)}
    keep = np.ones(len(genes), dtype=bool)
    add_rows, add_cols, add_vals = [], [], []
    for y, x in pairs.items():
        iy, ix = gidx[y], gidx[x]
        col = X[:, iy].tocoo()
        add_rows.append(col.row)
        add_cols.append(np.full(col.nnz, ix))
        add_vals.append(col.data)
        keep[iy] = False
    if pairs:
        extra = sp.csc_matrix((np.concatenate(add_vals),
                               (np.concatenate(add_rows), np.concatenate(add_cols))),
                              shape=X.shape)
        X = X + extra
    X = X[:, np.where(keep)[0]].astype(np.float32).toarray()   # dense, as load_c1 returns
    genes_m = [g for g, k in zip(genes, keep) if k]
    Xf, genes_f = w4.preprocess_and_filter(X, genes_m, w4.C1_GENE_FILTER_MIN_CELLS)
    Xf = sp.csr_matrix(Xf)
    gc, pcc, shan, loglib, meta = primitives(Xf, genes_f)
    prim = dict(gene_count=gc, pcc_string_v12=pcc, shannon_H=shan, log10_lib=loglib,
                ordinal=np.asarray(ranks, dtype=float), potency_sign=1.0)
    meta.update({"source": "w4 load_c1 (GSE117498), broad-gate make.names gene names "
                           "merged into their hyphenated twins, genes in >= 10 cells",
                 "n_gene_pairs_merged": len(pairs),
                 "n_sorted_file_genes": len(sorted_genes),
                 "n_broad_gate_file_genes": len(broad_genes),
                 "ordinal": "as GSE117498"})
    return Xf, genes_f, list(cell_ids), prim, meta


def ds_gse125970():
    import w4_gate2_run as w4
    X, genes, cells, prim, meta = _w4(w4.load_c2, "c2_gse125970", w4.C2_GENE_FILTER_MIN_CELLS)
    meta.update({"source": "w4 load_c2 (GSE125970 raw UMI), genes in >= 10 cells",
                 "ordinal": "C2_LABEL_TO_RANK: stem 4, TA 3, progenitor 3, differentiated 1"})
    return X, genes, cells, prim, meta


def ds_gse113074():
    P = data_root() / "fix3/data/prepared"
    X = sp.load_npz(P / "X_cells_genes.npz").tocsr()
    genes = P.joinpath("genes.txt").read_text().split("\n")
    rows = list(csv.DictReader(open(P / "cells.tsv"), delimiter="\t"))
    cells = [r["cell_id"] for r in rows]
    p = np.load(P / "primitives.npz")
    lib = np.array([float(r["lib_size"]) for r in rows])
    prim = dict(gene_count=p["gene_count"], pcc_string_v12=p["pcc_string_v12"],
                shannon_H=p["shannon_H"], log10_lib=np.log10(lib),
                ordinal=p["rank_ordinal"].astype(float), potency_sign=-1.0)
    return X, genes, cells, prim, {"source": "fix3/data/prepared (repo prep_briggs.py)",
                                   "ordinal": "rank_ordinal (1 = stage 8 ... 10 = stage 22)"}


DATASETS = {"GSE106474": ds_gse106474, "GSE117498": ds_gse117498,
            "GSE125970": ds_gse125970, "GSE113074": ds_gse113074,
            "GSE117498h": ds_gse117498h}


def main():
    wanted = sys.argv[1:] or list(DATASETS)
    for name in wanted:
        X, genes, cells, prim, meta = DATASETS[name]()
        genes = [g for g in genes]
        if len(genes) != X.shape[1] or len(cells) != X.shape[0]:
            raise SystemExit(f"{name}: shape {X.shape} vs {len(cells)} cells, {len(genes)} genes")
        dest = OUT / name
        write_csc(dest, X, genes, cells)
        np.savez(dest / "primitives.npz", **{k: np.asarray(v) for k, v in prim.items()})
        o = prim["ordinal"]
        meta.update({"n_cells": int(X.shape[0]), "n_genes": int(X.shape[1]),
                     "nnz": int(sp.csr_matrix(X).nnz),
                     "n_cells_ranked": int(np.isfinite(o).sum()),
                     "n_ordinal_levels": int(len(np.unique(o[np.isfinite(o)])))})
        (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        print(f"{name}: {meta['n_cells']:,} cells x {meta['n_genes']:,} genes, "
              f"{meta['n_cells_ranked']:,} ranked over {meta['n_ordinal_levels']} levels "
              f"-> {dest}", flush=True)


if __name__ == "__main__":
    main()
