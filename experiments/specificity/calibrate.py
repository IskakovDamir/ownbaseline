#!/usr/bin/env python3
"""
calibrate.py: what the added statistics do when the answer is known. Reads no real score.

    python3 experiments/specificity/calibrate.py cluster       # the sample test, zero true skill and planted trends
    python3 experiments/specificity/calibrate.py concordance   # the stratified concordance under two nulls

`cluster` keeps the design of GSE106474 (which sample holds how many cells, at which stage, in
which sequencing batch; no expression value and no score) and simulates everything else. A score
is a function of its primitive plus noise plus an effect shared by the cells of a sample and drawn
independently of the stage, so its true conditional skill is zero. The function is a straight line
at rank correlation 0.5 or 0.9, a cube or a square. It then asks how often each rule calls the
score skilled: the registered floor, the permutation floor over cells, and the trend test over
samples on the stratum-mean residual (the one that decides) and on the least-squares residual.
A planted trend over stages measures the power of the sample test.

The least-squares residual is there because an adversarial review of this plan found, before any
label was read, that the sample test on it rejects far too often when no sample effect is
present: a straight line on ranks leaves a function of the primitive in the residual, sample
means of the primitive trend with stage, and Kendall's S over 28 means ignores how small that
trend is. The stratum-mean residual removes it.

`concordance` repeats the two cases of experiments/conditional_null (coupled, curved) for the
stratified concordance.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker: the pool is the parallelism

import argparse
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import load_cells, num, rng_for  # noqa: E402
from stats import concordance_test, conditional_skill, residual, sample_level_test  # noqa: E402

from own_baseline.conditional_null import cs_value, m_for, permute_within, strata_1d  # noqa: E402
from own_baseline.floors import floor as grid_floor  # noqa: E402

# shape, rank correlation of the straight line, sample sd, batch sd, planted trend, also measure the permutation floor
CASES = ([("linear", 0.5, sd, 0.0, 0.0, sd in (0.0, 0.1, 0.5)) for sd in (0.0, 0.05, 0.1, 0.2, 0.5)]
         + [("linear", 0.9, sd, 0.0, 0.0, False) for sd in (0.0, 0.1, 0.5)]
         + [("cube", 0.0, sd, 0.0, 0.0, False) for sd in (0.0, 0.1)]
         + [("square", 0.0, sd, 0.0, 0.0, sd == 0.0) for sd in (0.0, 0.1)]
         + [("linear", 0.5, 0.1, 0.3, 0.0, False)]
         + [("linear", 0.5, 0.1, 0.0, dl, False) for dl in (0.05, 0.1, 0.2)])
SHAPES = ("linear", "cube", "square")
DESIGN = None


def design():
    c = load_cells("GSE106474")
    sample = np.array([b.rsplit("_", 1)[0] for b in c["barcode"]])
    batch = np.array([s.split("_")[2][:3] for s in sample])
    return sample, batch, num(c["ordinal"])


def _init():
    global DESIGN
    DESIGN = design()


def cluster_case(args):
    ci, rep, with_perm, n_draws = args
    shape, rho, sigma, batch_sd, delta, _ = CASES[ci]
    sample, batch, o = DESIGN
    rng = rng_for(20, ci, rep)
    n = len(o)
    z = (o - o.mean()) / o.std()
    g = 2 * np.sin(np.pi * 0.55 / 6)
    p = g * z + np.sqrt(1 - g * g) * rng.standard_normal(n)       # the primitive tracks stage, as the real ones do
    ids, inv = np.unique(sample, return_inverse=True)
    bids, binv = np.unique(batch, return_inverse=True)
    effect = sigma * rng.standard_normal(len(ids))[inv] + batch_sd * rng.standard_normal(len(bids))[binv]
    if shape == "linear":
        a = 2 * np.sin(np.pi * rho / 6)
        base = a * p + np.sqrt(1 - a * a) * rng.standard_normal(n)
    else:
        base = (p ** 3 if shape == "cube" else p ** 2) + rng.standard_normal(n)
    s = base + effect + delta * z
    cs = conditional_skill(s, [p], o)
    fl = grid_floor(n, abs(float(spearmanr(s, p).statistic)), kernel="taub", k_covariates=1, levels=12)["floor"]
    labels = strata_1d(p, m_for(n), rng_for(23, ci, rep))
    rs, rl = residual(s, [p], o, labels, "stratum_mean"), residual(s, [p], o, labels, "linear")
    out = {"case": ci, "cs": cs, "clears_grid": bool(cs > fl),
           "p_stratum_mean": sample_level_test(rs, sample, o, batch, n_perm=5000, rng=rng_for(21, ci, rep))["p_two_sided"],
           "p_stratum_mean_no_batch": sample_level_test(rs, sample, o, None, n_perm=5000, rng=rng_for(22, ci, rep))["p_two_sided"],
           "p_linear": sample_level_test(rl, sample, o, batch, n_perm=5000, rng=rng_for(25, ci, rep))["p_two_sided"]}
    if with_perm:
        d = np.array([cs_value(permute_within(s, labels, rng_for(24, ci, rep, i)), [p], labels, o, "linear") for i in range(n_draws)])
        out["clears_perm"] = bool(cs > np.quantile(d, 0.975))
    return out


def cluster(out_dir, procs, reps, perm_reps, n_draws):
    tasks = [(ci, r, c[5] and r < perm_reps, n_draws) for ci, c in enumerate(CASES) for r in range(reps)]
    with Pool(procs, initializer=_init) as pool:
        res = pool.map(cluster_case, tasks, chunksize=2)
    rows = []
    for ci, (shape, rho, sigma, batch_sd, delta, _) in enumerate(CASES):
        g = [r for r in res if r["case"] == ci]
        gp = [r for r in g if "clears_perm" in r]
        rows.append({"shape": shape, "rho": rho if shape == "linear" else None, "sample_sd": sigma, "batch_sd": batch_sd,
                     "planted_trend": delta, "reps": len(g),
                     "mean_abs_cs": float(np.mean([abs(r["cs"]) for r in g])),
                     "clears_registered_floor": float(np.mean([r["clears_grid"] for r in g])),
                     "clears_permutation_floor": float(np.mean([r["clears_perm"] for r in gp])) if gp else None,
                     "perm_reps": len(gp),
                     "sample_test_stratum_mean": float(np.mean([r["p_stratum_mean"] < 0.05 for r in g])),
                     "sample_test_stratum_mean_no_batch_strata": float(np.mean([r["p_stratum_mean_no_batch"] < 0.05 for r in g])),
                     "sample_test_least_squares": float(np.mean([r["p_linear"] < 0.05 for r in g]))})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_cluster.json").write_text(json.dumps(
        {"design": "GSE106474: samples, batches and cells per stage; everything else simulated",
         "rejection": "share of replicates with p < 0.05, two-sided", "rows": rows}, indent=1))
    for r in rows:
        print(json.dumps(r), flush=True)


def concordance_case(args):
    case, n, rho, rep, n_draws = args
    rng = rng_for(30, rep, int(rho * 1000), 0 if case == "coupled" else 1, n)
    a, g = 2 * np.sin(np.pi * rho / 6), 2 * np.sin(np.pi * 0.55 / 6)
    p = rng.standard_normal(n)
    curve = (p * p - 1) / np.sqrt(2) if case == "curved" else p
    latent = g * curve + np.sqrt(1 - g * g) * rng.standard_normal(n)
    o = (rankdata(latent, method="ordinal") - 1) * 12 // n + 1.0
    s = p + curve + (1 - rho) * rng.standard_normal(n) if case == "curved" else a * p + np.sqrt(1 - a * a) * rng.standard_normal(n)
    labels = strata_1d(p, m_for(n), rng_for(31, rep))
    t = concordance_test(s, o, labels, rng_for, (32, rep), n_draws)
    return {"case": case, "n": n, "rho": rho, "value": t["value"], "clears": t["clears"], "z": t["z"], "null_mean": t["null_mean"],
            "null_sd": t["null_sd"]}


def concordance(out_dir, procs, reps, n_draws):
    tasks = [(c, 3000, r, i, n_draws) for c in ("coupled", "curved") for r in (0.5, 0.9) for i in range(reps)]
    tasks += [(c, 39505, r, i, n_draws // 2) for c in ("coupled", "curved") for r in (0.5, 0.9) for i in range(20)]
    with Pool(procs) as pool:
        res = pool.map(concordance_case, tasks, chunksize=1)
    rows = []
    for key in sorted({(r["case"], r["n"], r["rho"]) for r in res}):
        g = [r for r in res if (r["case"], r["n"], r["rho"]) == key]
        rows.append({"case": key[0], "n": key[1], "rho": key[2], "reps": len(g),
                     "mean_value": float(np.mean([r["value"] for r in g])),
                     "false_clear": float(np.mean([r["clears"] for r in g])),
                     "mean_null_mean_in_sd": float(np.mean([r["null_mean"] / r["null_sd"] for r in g])),
                     "mean_z": float(np.mean([r["z"] for r in g]))})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "calibration_concordance.json").write_text(json.dumps({"rows": rows}, indent=1))
    for r in rows:
        print(json.dumps(r), flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["cluster", "concordance"])
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--perm-reps", type=int, default=40)
    ap.add_argument("--n-draws", type=int, default=200)
    a = ap.parse_args()
    if a.mode == "cluster":
        cluster(Path(a.out), a.procs, a.reps, a.perm_reps, a.n_draws)
    else:
        concordance(Path(a.out), a.procs, a.reps, a.n_draws)


if __name__ == "__main__":
    main()
