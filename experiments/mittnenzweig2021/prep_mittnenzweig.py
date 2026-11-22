#!/usr/bin/env python3
"""
prep_mittnenzweig.py: Mittnenzweig et al. 2021 (GSE169210) laid out for the audit's own code.

    python3 experiments/mittnenzweig2021/prep_mittnenzweig.py data/runs/breadth/GSE169210

Mouse gastrulation, MARS-seq, 153 wild-type embryos processed one by one, each cell's embryo
recorded at the sort. Input, as GEO serves it:

    raw/GSE169210_RAW.tar                                  one UMI table per 384-well plate, genes in rows
    meta/GSE169210_wt_{bl6,icr}_single_cell_metadata.txt.gz  cell, plate, embryo, developmental_time, metacell

Cells kept: wells of an embryo that carries a developmental time and that the authors assigned
to a metacell (their own quality filter; 33,700 wells), then at least 200 detected genes, as
prep_data.py does. The chimera and tetraploid plates are not part of the staged series.

Gene names are mm9 symbols, 2,163 of them several aliases joined by ';'. Each is mapped to one
upper-cased symbol to meet the human STRING scaffold, the convention the audit discloses for
its other non-human series: the first alias on the scaffold, otherwise the first alias. Names
that collide after mapping are summed.

Output, under the run directory:

    root/track2/data/prepared/   X_cells_genes.npz, genes.txt, cells.tsv, primitives.npz, meta.json
    labels/labels.tsv            cell_id, sample (embryo), batch (sort session), ordinal, strain, plate

The label table is written here and compared with nothing. The prepared directory holds no
label, so the scoring and control code cannot read one. Reads no score.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MIN_GENES = 200


def read_meta(meta_dir):
    """well id -> (embryo, developmental time, strain, plate) for the wells of the staged series."""
    wells = {}
    for strain in ("bl6", "icr"):
        with gzip.open(meta_dir / f"GSE169210_wt_{strain}_single_cell_metadata.txt.gz", "rt", newline="") as fh:
            for r in csv.DictReader(fh, delimiter="\t"):
                if r["developmental_time"] == "NA" or r["metacell_id"] == "NA" or r["embryo_id"] == "empty":
                    continue
                if not r["cell_id"].startswith(r["plate"] + "."):
                    raise SystemExit(f"cell id {r['cell_id']} does not start with its plate")
                well = r["cell_id"][len(r["plate"]) + 1:]              # plate names hold dots (E7.5), so no split
                if well in wells:
                    raise SystemExit(f"well {well} is listed twice")
                wells[well] = (r["embryo_id"], float(r["developmental_time"]), strain, r["plate"])
    return wells


def main():
    run = Path(sys.argv[1]).resolve()
    root = run / "root"
    os.environ["OWNBASELINE_DATA_ROOT"] = str(root)          # prepare_inputs reads the scaffold from the data root
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "experiments" / "w5_new_rows"))
    sys.path.insert(0, str(REPO / "experiments" / "specificity"))
    scaffold = root / "w4" / "scaffolds" / "human_string_v12_thr700_lcc.npz"
    scaffold.parent.mkdir(parents=True, exist_ok=True)
    if not scaffold.is_file():
        shutil.copyfile(REPO / "data" / "runs" / "w4" / "scaffolds" / scaffold.name, scaffold)
    out = root / "track2" / "data" / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    (run / "labels").mkdir(exist_ok=True)

    wells = read_meta(run / "meta")
    mats, ids, genes0, plates = [], [], None, 0
    with tarfile.open(run / "raw" / "GSE169210_RAW.tar", "r|") as tf:
        for m in tf:
            if not m.name.endswith(".txt.gz"):
                continue
            head = gzip.open(io.BytesIO(tf.extractfile(m).read()), "rt")
            cols = head.readline().rstrip("\n").split("\t")
            keep = [j for j, c in enumerate(cols) if c in wells]
            if not keep:
                continue
            df = pd.read_csv(head, sep="\t", header=None, index_col=0)
            if df.shape[1] != len(cols):
                raise SystemExit(f"{m.name}: {df.shape[1]} columns for {len(cols)} wells")
            genes = df.index.astype(str).tolist()
            genes0 = genes0 or genes
            if genes != genes0:
                raise SystemExit(f"{m.name}: gene list differs from the first plate")
            mats.append(sp.csr_matrix(df.values[:, keep].astype(np.int32).T))
            ids += [cols[j] for j in keep]
            plates += 1
    X = sp.vstack(mats).tocsr()
    missing = set(wells) - set(ids)
    print(f"{plates} plates, {X.shape[0]} of {len(wells)} listed wells found, {len(missing)} missing", flush=True)

    from common import load_degree  # noqa: E402
    on_scaffold = load_degree()
    symbol = []
    for g in genes0:
        alias = [a.upper() for a in g.split(";")]
        symbol.append(next((a for a in alias if a in on_scaffold), alias[0]))
    genes, inverse = np.unique(np.array(symbol), return_inverse=True)
    merge = sp.csr_matrix((np.ones(len(symbol), dtype=np.int32), (np.arange(len(symbol)), inverse)), shape=(len(symbol), len(genes)))
    X = (X @ merge).tocsr()
    X.eliminate_zeros()
    detected = np.asarray((X > 0).sum(axis=1)).ravel()
    keep = detected >= MIN_GENES
    print(f"QC: kept {int(keep.sum())} of {len(keep)} cells; {len(symbol)} names, {len(genes)} symbols after mapping", flush=True)
    X = X[keep].tocsr()
    ids = np.array(ids)[keep]
    if len(set(ids)) != len(ids):
        raise SystemExit("cell ids are not unique")
    sp.save_npz(out / "X_cells_genes.npz", X.astype(np.int32))
    (out / "genes.txt").write_text("\n".join(map(str, genes)))

    import prepare_inputs as pi  # noqa: E402
    gc, pcc, shan, loglib, meta = pi.primitives(X, genes)
    np.savez(out / "primitives.npz", gene_count=gc, pcc_string_v12=pcc, shannon_H=shan, log10_lib=loglib)
    lib = np.asarray(X.sum(axis=1)).ravel()
    with open(out / "cells.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cell_id", "lib_size", "gene_count"])
        w.writerows(zip(ids, lib, gc.astype(int)))
    dest = root / "w5_new_rows" / "inputs" / "GSE106474"       # where score_mce.py looks for the cell order
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "cells.txt").write_text("\n".join(ids) + "\n")

    with open(run / "labels" / "labels.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cell_id", "sample", "batch", "ordinal", "strain", "plate"])
        for c in ids:
            embryo, t, strain, plate = wells[c]
            w.writerow([c, embryo, embryo.split("_")[0], t, strain, plate])
    embryos = {wells[c][0] for c in ids}
    (out / "meta.json").write_text(json.dumps({
        "source": "GSE169210 RAW.tar plate tables and the two wild-type metadata tables", "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]), "nnz": int(X.nnz), "plates": plates, "embryos": len(embryos),
        "sort_sessions": len({e.split("_")[0] for e in embryos}), "wells_listed": len(wells), "wells_missing": len(missing),
        "median_library_size": float(np.median(lib)), "cells_with_1000_counts": int((lib >= 1000).sum()), **meta}, indent=1))
    print((out / "meta.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
