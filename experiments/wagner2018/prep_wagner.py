#!/usr/bin/env python3
"""
prep_wagner.py: Wagner et al. 2018 (GSE112294) laid out for the audit's own code.

    OWNBASELINE_DATA_ROOT=data/runs/wagner2018/root \\
    OWNBASELINE_SCRATCH=data/runs/wagner2018/scratch \\
    python3 experiments/wagner2018/prep_wagner.py data/runs/wagner2018/raw

A staged series independent of GSE106474: another laboratory, another platform
(inDrops), embryos collected at 4, 6, 8, 10, 14, 18 and 24 hours post
fertilisation. The seven GSM*_NNhpf.csv.gz tables are raw UMI counts, genes in
rows. The two TracerSeq libraries are left out: they are lineage-tracing
experiments and not part of the staged series.

The output is the layout experiments/track2/code reads, so run_cytotrace.py,
prep_scent_inputs.py, run_scent_full.py and experiments/w5_new_rows/score_mce.py
run on these cells unchanged. QC keeps cells with at least 200 detected genes, as
prep_data.py does. Symbols are upper-cased to match the human STRING scaffold, the
convention the audit discloses for the other non-human series; symbols that collide
after upper-casing are summed. Reads no score.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "experiments" / "w4" / "scripts"))
sys.path.insert(0, str(REPO / "experiments" / "w5_new_rows"))
from own_baseline.paths import data_root  # noqa: E402

STAGES = [("GSM3067189_04hpf", 4), ("GSM3067190_06hpf", 6), ("GSM3067191_08hpf", 8), ("GSM3067192_10hpf", 10),
          ("GSM3067193_14hpf", 14), ("GSM3067194_18hpf", 18), ("GSM3067195_24hpf", 24)]
MIN_GENES = 200
CHUNK_GENES = 2000


def load(path):
    genes, mats = [], []
    cells = None
    for ch in pd.read_csv(path, index_col=0, compression="gzip", chunksize=CHUNK_GENES):
        cells = ch.columns.tolist() if cells is None else cells
        genes.extend(ch.index.astype(str).tolist())
        mats.append(sp.csr_matrix(ch.values.astype(np.int32)))
    return sp.vstack(mats).tocsr().T.tocsr(), genes, cells


def main():
    raw = Path(sys.argv[1])
    root = data_root()
    out = root / "track2/data/prepared"
    out.mkdir(parents=True, exist_ok=True)
    Xs, cell_ids, stage, ranks, hpf, genes0 = [], [], [], [], [], None
    for rank, (stem, h) in enumerate(STAGES, 1):
        X, genes, cells = load(raw / f"{stem}.csv.gz")
        genes0 = genes0 or genes
        if genes != genes0:
            raise SystemExit(f"{stem}: gene list differs from the first table")
        Xs.append(X)
        cell_ids += [f"{h:02d}hpf_{c}" for c in cells]
        stage += [f"{h:02d}hpf"] * len(cells)
        ranks += [rank] * len(cells)
        hpf += [h] * len(cells)
        print(f"{stem}: {X.shape[0]} cells, {X.nnz} non-zero", flush=True)
    X = sp.vstack(Xs).tocsr()
    upper = np.array([g.upper() for g in genes0])
    genes, inverse = np.unique(upper, return_inverse=True)
    merge = sp.csr_matrix((np.ones(len(upper), dtype=np.int32), (np.arange(len(upper)), inverse)),
                          shape=(len(upper), len(genes)))
    X = (X @ merge).tocsr()
    X.eliminate_zeros()
    detected = np.asarray((X > 0).sum(axis=1)).ravel()
    keep = detected >= MIN_GENES
    print(f"QC: kept {int(keep.sum())} of {len(keep)} cells; {len(upper)} symbols, {len(genes)} after upper-casing", flush=True)
    X = X[keep].tocsr()
    cell_ids = np.array(cell_ids)[keep]
    stage = np.array(stage)[keep]
    ranks = np.array(ranks, dtype=np.float64)[keep]
    hpf = np.array(hpf, dtype=np.float64)[keep]
    if len(set(cell_ids)) != len(cell_ids):
        raise SystemExit("cell ids are not unique")
    sp.save_npz(out / "X_cells_genes.npz", X.astype(np.int32))
    (out / "genes.txt").write_text("\n".join(map(str, genes)))

    import prepare_inputs as pi  # noqa: E402
    gc, pcc, shan, loglib, meta = pi.primitives(X, genes)
    lib = np.asarray(X.sum(axis=1)).ravel()
    with open(out / "cells.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cell_id", "stage", "rank_ordinal", "hpf", "lib_size", "gene_count"])
        for row in zip(cell_ids, stage, ranks.astype(int), hpf.astype(int), lib, gc.astype(int)):
            w.writerow(row)
    np.savez(out / "primitives.npz", gene_count=gc, pcc_string_v12=pcc, shannon_H=shan, log10_lib=loglib, rank_kimmel=ranks,
             ordinal=ranks, potency_sign=-1.0)
    dest = root / "w5_new_rows" / "inputs" / "GSE106474"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cells.txt").write_text("\n".join(cell_ids) + "\n")
    (out / "meta.json").write_text(json.dumps({
        "source": "GSE112294 GSM3067189-GSM3067195 raw counts", "n_cells": int(X.shape[0]), "n_genes": int(X.shape[1]),
        "nnz": int(X.nnz), "ordinal": "1 = 4 hpf ... 7 = 24 hpf", "n_ordinal_levels": 7,
        "n_cells_by_stage": {str(k): int(v) for k, v in zip(*np.unique(stage, return_counts=True))}, **meta}, indent=1))
    print(json.dumps(json.loads((out / "meta.json").read_text()), indent=1), flush=True)


if __name__ == "__main__":
    main()
