#!/usr/bin/env python3
"""
floor_extend.py: the permutation floor of PREREG_breadth_3.md on the first 200 random gene sets.

    python3 experiments/specificity/floor_extend.py --procs 8

Same strata, draws and random streams as analyse.py and breadth_analyse.py, so the first 50 floors
of each series return unchanged. Writes results/floor_extended.json.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse import chunks, perm_floor_block  # noqa: E402
from breadth_analyse import read_labels  # noqa: E402
from breadth_blind import SID, paths  # noqa: E402
from common import JOINT, SERIES, WORK, load_cells, num, rng_for  # noqa: E402
from own_baseline.conditional_null import m_for, strata_1d  # noqa: E402

N_SETS, N_DRAWS = 200, 200
ORDER = ("GSE169210", "GSE223922", "GSE223922b", "GSE106474", "GSE112294")


def load(geo):
    if geo in SID:
        p = paths(geo)
        o = read_labels(p["labels"])["ordinal"].astype(float)
        prim = np.load(p["prep"] / "primitives.npz")
        frame = {"detected_genes": prim["gene_count"], "pcc_x_degree": prim["pcc_string_v12"],
                 "shannon_entropy": prim["shannon_H"], "log10_library_size": prim["log10_lib"]}
        return SID[geo], o, frame, np.load(p["blind"] / "signatures.npz")["a1_g200"][:, :N_SETS], "breadth"
    cells = load_cells(geo)
    return SERIES.index(geo), num(cells["ordinal"]), {k: num(cells[k]) for k in JOINT}, np.load(WORK / f"{geo}_signatures.npz")["a1_g200"][:, :N_SETS], "specificity"


def wilson(k, n, z=1.959964):
    ph = k / n; d = 1 + z * z / n; c = ph + z * z / (2 * n); h = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--procs", type=int, default=8); a = ap.parse_args()
    out_file = HERE / "results" / "floor_extended.json"
    out = json.loads(out_file.read_text()) if out_file.is_file() else {}
    with Pool(a.procs) as pool:
        for geo in ORDER:
            if geo in out:
                continue
            sid, o, frame, sub, kind = load(geo)
            cs_all = np.load(HERE / "results" / f"{kind}_{geo}_control_values.npz")
            m, res = m_for(len(o)), {}
            for pi, q in enumerate(JOINT):
                labels = strata_1d(frame[q], m, rng_for(5, sid, pi))
                jobs = [(sub[:, sl], frame[q], o, labels, (6, sid, pi, sl.start), N_DRAWS) for sl in chunks(sub.shape[1], 2)]
                fl = np.concatenate(pool.map(perm_floor_block, jobs, chunksize=1))
                cs = cs_all[f"full|a1_g200|{q}"][:N_SETS]
                k = int((cs > fl).sum())
                res[q] = {"sets": N_SETS, "above": k, "share": k / N_SETS, "wilson95": wilson(k, N_SETS),
                          "share_first_50": float((cs[:50] > fl[:50]).mean()), "median_floor": float(np.median(fl))}
                print(geo, q, res[q], flush=True)
            out[geo] = res
            out_file.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
