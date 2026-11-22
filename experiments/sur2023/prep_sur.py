#!/usr/bin/env python3
"""
prep_sur.py: Sur et al. 2023 (GSE223922), 14 to 120 hours post fertilisation, laid out for the audit's own code.

    python3 experiments/sur2023/prep_sur.py data/runs/breadth/GSE223922

Zebrafish, 10x, whole embryos and larvae collected at 53 times. In 26 lanes embryos of four or six
collection times were hashed with MULTI-seq and loaded together; 32 lanes hold one collection time.
The 38,731 Drop-seq cells of 3 to 12 hpf are the cells of Farrell et al. 2018 remapped, that is
GSE106474, a series the audit already holds; they are left out.

Input, as GEO serves it, under raw/: GSE223922_Sur2023_counts.mtx.gz (genes x cells),
_counts_rows_genes.txt.gz, _counts_cols_cells.txt.gz, _metadata.tsv.gz.

The atlas has 450,955 10x cells, more than this machine holds, so a subsample is fixed here from the
metadata table alone, before the count matrix is opened: from each lane-by-time unit, PER_UNIT cells
drawn uniformly without replacement among the unit's cells with at least 200 detected genes (the
table's nGene), every cell of a unit that has fewer. The draw reads the collection time to form the
units and reads no expression value. Symbols are upper-cased to meet the human STRING scaffold, as
for the audit's other zebrafish series; symbols that collide are summed.

Output, under the run directory:

    root/track2/data/prepared/   X_cells_genes.npz, genes.txt, cells.tsv, primitives.npz, meta.json
    labels/labels.tsv            cell_id, sample (lane|time), batch (lane), batch2 (experiment), ordinal (hpf)

The prepared directory holds no label. Reads no score.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
MIN_GENES, PER_UNIT, SID, CHUNK = 200, 220, 4, 20_000_000
STEM = "GSE223922_Sur2023_"


def main():
    run = Path(sys.argv[1]).resolve()
    draw = int(sys.argv[2]) if len(sys.argv) > 2 else 1          # 2: a second draw, disjoint from the first
    raw, root = (Path(sys.argv[3]).resolve() if len(sys.argv) > 3 else run / "raw"), run / "root"
    os.environ["OWNBASELINE_DATA_ROOT"] = str(root)
    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "experiments" / "w5_new_rows"))
    sys.path.insert(0, str(REPO / "experiments" / "specificity"))
    from common import rng_for  # noqa: E402
    scaffold = root / "w4" / "scaffolds" / "human_string_v12_thr700_lcc.npz"
    scaffold.parent.mkdir(parents=True, exist_ok=True)
    if not scaffold.is_file():
        shutil.copyfile(REPO / "data" / "runs" / "w4" / "scaffolds" / scaffold.name, scaffold)
    out = root / "track2" / "data" / "prepared"
    out.mkdir(parents=True, exist_ok=True)
    (run / "labels").mkdir(exist_ok=True)

    # ---- the subsample, from the metadata table alone
    units = {}
    with gzip.open(raw / f"{STEM}metadata.tsv.gz", "rt", newline="") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            if r["method"] == "10X" and int(r["nGene"]) >= MIN_GENES:
                units.setdefault((r["sample"], int(r["stage.integer"])), []).append(r["cell"])
    label = {}
    for ui, key in enumerate(sorted(units)):
        cells = sorted(units[key])
        take = cells if len(cells) <= PER_UNIT else [cells[i] for i in np.sort(rng_for(20, SID, ui).choice(len(cells), PER_UNIT, replace=False))]
        if draw == 2:                                            # what the first draw left, under a new seed
            first = set(take)
            rest = [c for c in cells if c not in first]
            take = rest if len(rest) <= PER_UNIT else [rest[i] for i in np.sort(rng_for(21, SID, ui).choice(len(rest), PER_UNIT, replace=False))]
        for c in take:
            label[c] = key
    with gzip.open(raw / f"{STEM}counts_cols_cells.txt.gz", "rt") as fh:
        cols = fh.read().split()
    with gzip.open(raw / f"{STEM}counts_rows_genes.txt.gz", "rt") as fh:
        genes0 = fh.read().split("\n")
    genes0 = genes0[:-1] if genes0[-1] == "" else genes0
    lut = np.full(len(cols) + 1, -1, dtype=np.int32)             # 1-based matrix column -> row of the subsample
    ids = [c for c in cols if c in label]
    lut[[j + 1 for j, c in enumerate(cols) if c in label]] = np.arange(len(ids), dtype=np.int32)
    print(f"{len(units)} units, {len(ids)} of {sum(map(len, units.values()))} eligible cells drawn", flush=True)

    # ---- the count matrix, streamed; only the drawn cells are kept
    with gzip.open(raw / f"{STEM}counts.mtx.gz", "rt") as fh:
        fh.readline()
        n_genes, n_cells, nnz = map(int, fh.readline().split())
    if (n_genes, n_cells) != (len(genes0), len(cols)):
        raise SystemExit("matrix shape does not match the gene and cell lists")
    gi, ci, vv, seen = [], [], [], 0
    for ch in pd.read_csv(raw / f"{STEM}counts.mtx.gz", sep=" ", header=None, skiprows=2, names=["g", "c", "v"],
                          dtype=np.int32, chunksize=CHUNK, compression="gzip"):
        seen += len(ch)
        row = lut[ch["c"].to_numpy()]
        m = row >= 0
        gi.append(ch["g"].to_numpy()[m] - 1)
        ci.append(row[m])
        vv.append(ch["v"].to_numpy()[m])
        print(f"  {seen / nnz:.0%} of the matrix read", flush=True)
    if seen != nnz:
        raise SystemExit(f"{seen} entries read, the header announces {nnz}")
    X = sp.csr_matrix((np.concatenate(vv), (np.concatenate(ci), np.concatenate(gi))), shape=(len(ids), n_genes), dtype=np.int32)
    del gi, ci, vv

    upper = np.array([g.upper() for g in genes0])
    genes, inverse = np.unique(upper, return_inverse=True)
    merge = sp.csr_matrix((np.ones(len(upper), dtype=np.int32), (np.arange(len(upper)), inverse)), shape=(len(upper), len(genes)))
    X = (X @ merge).tocsr()
    X.eliminate_zeros()
    detected = np.asarray((X > 0).sum(axis=1)).ravel()
    keep = detected >= MIN_GENES
    print(f"QC: kept {int(keep.sum())} of {len(keep)} cells; {len(upper)} symbols, {len(genes)} after upper-casing", flush=True)
    X = X[keep].tocsr()
    ids = np.array(ids)[keep]
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
        w.writerow(["cell_id", "sample", "batch", "batch2", "ordinal"])
        for c in ids:
            lane, hpf = label[c]
            w.writerow([c, f"{lane}|{hpf}", lane, lane.split("-")[0], hpf])
    kept_units = {label[c] for c in ids}
    (out / "meta.json").write_text(json.dumps({
        "source": "GSE223922 counts.mtx.gz, 10x cells only, subsample fixed from the metadata table", "n_cells": int(X.shape[0]),
        "n_genes": int(X.shape[1]), "nnz": int(X.nnz), "units": len(kept_units), "lanes": len({u[0] for u in kept_units}),
        "collection_times": len({u[1] for u in kept_units}), "cells_per_unit": PER_UNIT, "draw": draw,
        "median_library_size": float(np.median(lib)), "cells_with_1000_counts": int((lib >= 1000).sum()), **meta}, indent=1))
    print((out / "meta.json").read_text(), flush=True)


if __name__ == "__main__":
    main()
