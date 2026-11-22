#!/usr/bin/env python3
"""
window.py: CytoTRACE v1 recomputed on the hours the two zebrafish series share.

    python3 experiments/specificity/window.py --procs 6

GSE106474 runs from 3.3 to 12 hours post fertilisation and GSE112294 from 4 to 24. The
window they share is 4 to 12 hours: dome to the 6-somite stage in the first, and the 4, 6,
8 and 10 hour collections in the second. For each series this script keeps the cells of
that window, at raw depth and after every cell is drawn down to 1,000 counts, and reruns
the gene count, the gene-count signature and the full CytoTRACE v1 pipeline on them.

The hours are read only to select cells. No statistic is compared with a stage here; the
analysis stage does that after PREREG.md is locked.
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import SERIES, WORK, load_cells, load_matrix, lognorm, num, sha256  # noqa: E402
from controls import THIN_PRIMARY, cytotrace_pipeline, gene_count_signature, thin_counts  # noqa: E402

WINDOW = (4.0, 12.0)
ZEBRAFISH = ("GSE106474", "GSE112294")


def run(x, procs, tag):
    xn, _ = lognorm(x)
    gc = np.asarray((x > 0).sum(axis=1)).ravel().astype(np.float64)
    gcs, _ = gene_count_signature(xn, gc)
    return gc, gcs, cytotrace_pipeline(xn, gcs[:, None], procs, tag)[:, 0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--procs", type=int, default=6)
    a = ap.parse_args()
    record = {}
    for geo in ZEBRAFISH:
        t0 = time.time()
        x = load_matrix(geo).tocsr()
        hpf = num(load_cells(geo)["hpf"])
        inside = (hpf >= WINDOW[0]) & (hpf <= WINDOW[1])
        out = {"window_cells": np.flatnonzero(inside)}
        out["genes_window"], out["gcs_window"], out["cytotrace_window"] = run(x[inside], procs=a.procs, tag=f"{geo} window")
        keep, xt = thin_counts(x, THIN_PRIMARY, (4, SERIES.index(geo), THIN_PRIMARY))    # the draw controls.py thin makes
        sel = inside[keep]
        out["window_thinned_cells"] = keep[sel]
        out["genes_window_thinned"], out["gcs_window_thinned"], out["cytotrace_window_thinned"] = run(
            xt[sel], procs=a.procs, tag=f"{geo} window thinned")
        path = WORK / f"{geo}_window.npz"
        np.savez(path, **out)
        record[geo] = {"series": geo, "window_hpf": list(WINDOW), "cells_in_window": int(inside.sum()),
                       "cells_in_window_thinned": int(sel.sum()), "seconds": round(time.time() - t0, 1),
                       "file": path.name, "sha256": sha256(path)}
        print(json.dumps(record[geo]), flush=True)
    (HERE / "results" / "blind_window.json").write_text(json.dumps(record, indent=1))


if __name__ == "__main__":
    main()
