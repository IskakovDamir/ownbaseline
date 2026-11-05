#!/usr/bin/env python3
"""
conditional_null.py: floors measured on the real cells, without the simulator.

    python3 experiments/conditional_null/conditional_null.py synthetic
    python3 experiments/conditional_null/conditional_null.py real --deposit data/deposit/ownbaseline-data-v1

The audit's floor is read from a simulated null (Appendix D). Two objections go
to it: the floor is only as good as the simulator, and a least-squares fit on
ranks can leave a nonlinear trace of the primitive that a kernel then scores.
This script answers both from the real vectors. Under "the score carries nothing
beyond its primitive", score and ordinal are independent given the primitive, so
permuting the score among cells with the same primitive value leaves the joint
law unchanged (Berrett et al. 2020, JRSS B 82:175). A draw keeps the real
primitive, the real ordinal, their real dependence, and the real score's
distribution and nonlinear relation to the primitive at each primitive level.
The estimator is then applied to the draw exactly as to the real score. PREREG.md
in this directory is the plan.

What it computes, per row:
  at k = 1, conditional skill and its permutation floor under three residualizers:
  the registered least-squares fit (own_baseline.conditional_skill), an isotonic fit,
  and the stratum mean, which follows a non-monotone trace as well;
  the generalised covariance measure at k = 1 and at k = 4, which regresses score and
  ordinal on the primitives by cross-fitted boosting and needs no strata;
  the same under the authors' polarity, with alignment off;
  leave-one-stage-out conditional skill;
  the same statistic inside each sequencing batch of GSE106474;
  a bootstrap interval over the 28 Drop-seq samples instead of over cells.

Series: GSE106474, the audit's ordinal; GSE113074, the Xenopus replicate; GSE112294
(Wagner et al. 2018), an independent zebrafish series prepared by
experiments/wagner2018. The sample bootstrap runs where the barcode names a sample.

The synthetic mode reads no real data. It measures how often the shipped floor
and the permutation floor call a score skilled when its true conditional skill
is zero, under the coupled null and when score and ordinal share a curved function of the primitive.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from own_baseline.conditional_null import (  # noqa: E402
    SEED, cs_value, gcm, m_for, permute_within, rng_for, strata_1d)
from own_baseline.conditional_skill import align, rank_resid_multi  # noqa: E402
from own_baseline.floors import floor as grid_floor  # noqa: E402
from own_baseline.paths import data_root  # noqa: E402

N_DRAWS = 200
N_BOOT = 1000
SYNTH_M = (10, 25, 50, 100)
SYNTH_CASES = ["coupled", "curved", "coupled_k4", "curved_k4"]
JOINT = ["detected_genes", "pcc_x_degree", "shannon_entropy", "log10_library_size"]
# name, score column, declared primitive, sign of the authors' polarity on the score, joint row?
ROWS_FULL = [
    ("CytoTRACE", "cytotrace_v1", "detected_genes", 1, True),
    ("SR", "scent_sr", "pcc_x_degree", 1, True),
    ("ORIGINS", "origins", "pcc_x_degree", 1, True),
    ("StemFinder", "stemfinder", "stemfinder_cell_cycle_score", -1, True),
    ("CCAT", "scent_ccat", "pcc_x_degree", 1, False),
    ("MCE", "mce", "pcc_x_degree", 1, True),
]
ROWS_SUB = [
    ("NCG", "ncg", "pcc_x_degree", 1, False),
    ("dpath", "dpath", "shannon_entropy", 1, True),
    ("SLICE", "slice", "shannon_entropy", 1, False),
]
TRAINED = [("mRNAsi", "mrnasi", "GSE117498"), ("mRNAsi", "mrnasi", "GSE117498h"),
           ("FitDevo", "fitdevo", "GSE106474"), ("FitDevo", "fitdevo", "GSE117498"),
           ("FitDevo", "fitdevo", "GSE117498h"), ("FitDevo", "fitdevo", "GSE125970"),
           ("FitDevo", "fitdevo", "GSE113074")]


def taub(x, y):
    return float(kendalltau(x, y).statistic)


def one_row(task):
    name, s, covs, o, m, which_set, polarity, pol_o, labels, key, n_draws = task
    t0 = time.time()
    out = {"row": name, "n": len(s), "k": len(covs), "m": m, "strata": int(labels.max()) + 1,
           "registered": {}, "authors": {}}
    draws = {w: np.empty(n_draws) for w in which_set}
    draws_a = {w: np.empty(n_draws) for w in which_set}
    for i in range(n_draws):
        d = permute_within(s, labels, rng_for(*key, i))
        for w in which_set:
            draws[w][i] = cs_value(d, covs, labels, o, w)
            if polarity is not None:
                draws_a[w][i] = cs_value(d, covs, labels, pol_o, w, polarity)
    for w in which_set:
        v = cs_value(s, covs, labels, o, w)
        q = float(np.quantile(draws[w], 0.975))
        out["registered"][w] = {"value": v, "floor_perm": q, "null_mean": float(draws[w].mean()), "clears": bool(v > q)}
        if polarity is not None:
            v = cs_value(s, covs, labels, pol_o, w, polarity)
            q = float(np.quantile(draws_a[w], 0.975))
            out["authors"][w] = {"value": v, "floor_perm": q, "clears": bool(v > q)}
    out["seconds"] = round(time.time() - t0, 1)
    return out


def gcm_task(args):
    name, s, covs, o, sign, key = args
    out = {"row": name, "n": len(s), "k": len(covs), "cs_linear": cs_value(s, covs, None, o, "linear"),
           "registered": gcm(s, o, covs, key=key)}
    if sign is not None:
        out["authors"] = gcm(s, -o, covs, key=key, polarity=sign)
    return out


def leave_one_stage_out(args):
    name, s, covs, o = args
    vals = {}
    for stage in np.unique(o):
        keep = o != stage
        vals[int(stage)] = taub(rank_resid_multi(align(s[keep], o[keep]), [c[keep] for c in covs]), o[keep])
    return {"row": name, "by_stage_left_out": vals, "min": min(vals.values()), "max": max(vals.values())}


def sample_bootstrap(args):
    name, s, covs, o, sample, n_boot = args
    ids = np.unique(sample)
    where = {x: np.flatnonzero(sample == x) for x in ids}
    b = np.empty(n_boot)
    for j in range(n_boot):
        idx = np.concatenate([where[x] for x in rng_for(9, j).choice(ids, len(ids))])
        b[j] = taub(rank_resid_multi(align(s[idx], o[idx]), [c[idx] for c in covs]), o[idx])
    lo, hi = np.nanquantile(b, [0.025, 0.975])
    return {"row": name, "samples": len(ids), "CI95": [float(lo), float(hi)]}


def load(deposit):
    d = Path(deposit)
    cells = {a: pd.read_csv(d / "datasets" / a / "cells.csv.gz") for a in
             ["GSE106474", "GSE117498", "GSE117498h", "GSE125970", "GSE113074"]}
    scores = {a: pd.read_csv(d / "scores" / f"{a}_scores.csv.gz") for a in cells}
    sub = pd.read_csv(d / "scores" / "GSE106474_subsample3000_scores.csv.gz")
    return cells, scores, sub


def series_tasks(series, sid, frame, ss, o, rows, n_draws, n_boot, sample):
    """One block of rows that share cells: floors at three stratum sizes, and the joint row."""
    tasks, lo, boot, gc = [], [], [], []
    for i, (name, col, prim, sign, has_joint) in enumerate(rows):
        s = ss[col].to_numpy(float)
        p = (ss if prim.startswith("stemfinder") else frame)[prim].to_numpy(float)
        label, mp = f"{series} {name}", m_for(len(s))
        for m in (mp, mp // 2, mp * 2):
            which = ("linear", "isotonic", "stratum_mean") if m == mp else ("linear", "stratum_mean")
            labels = strata_1d(p, m, rng_for(1, sid, i, m))
            tasks.append((label, s, [p], o, m, which, float(sign), -o, labels, (2, sid, i, m), n_draws))
        lo.append((label, s, [p], o))
        if sample is not None:
            boot.append((label, s, [p], o, sample, n_boot))
        gc.append((label, s, [p], o, float(sign), (3, sid, i)))
        if has_joint:
            gc.append((f"{label} joint", s, [frame[c].to_numpy(float) for c in JOINT], o, float(sign), (4, sid, i)))
    return tasks, lo, boot, gc


def xenopus(cells, xdir):
    """The score vectors of the Xenopus runs, in the order of the prepared matrix."""
    z = cells["GSE113074"]
    ct, sc, og = (np.load(xdir / f"{n}_scores.npz") for n in ("cytotrace", "scent", "origins"))
    dp, sl = (np.load(xdir / f"{n}_scores.npz") for n in ("dpath", "slice"))
    if not np.array_equal(ct["gene_counts"], z["detected_genes"].to_numpy(float)):
        raise SystemExit("Xenopus score vectors are not in the deposit's cell order")
    if not np.array_equal(dp["cell_idx"], sl["cell_idx"]):
        raise SystemExit("Xenopus dpath and SLICE sit on different subsamples")
    full = pd.DataFrame({"cytotrace_v1": ct["ct_v1"], "scent_sr": sc["sr"], "scent_ccat": sc["ccat"], "origins": og["origins"]})
    sub = pd.DataFrame({"cell_index": dp["cell_idx"].astype(int), "dpath": dp["dpath"], "slice": sl["slice"]})
    return z, full, sub


def wagner(root):
    """The independent staged series, from the layout prep_wagner.py wrote."""
    d = root / "track2/data/prepared"
    p = np.load(d / "primitives.npz")
    c = pd.read_csv(d / "cells.tsv", sep="\t")
    frame = pd.DataFrame({"detected_genes": p["gene_count"], "pcc_x_degree": p["pcc_string_v12"],
                          "shannon_entropy": p["shannon_H"], "log10_library_size": p["log10_lib"]})
    ct, sc = (np.load(root / f"track2/results/{n}_scores.npz") for n in ("cytotrace", "scent"))
    mce = pd.read_csv(root / "w5_new_rows/scores/GSE106474/mce.csv")
    if list(mce["cell"]) != list(c["cell_id"]):
        raise SystemExit("Wagner MCE is not in the prepared cell order")
    scores = pd.DataFrame({"cytotrace_v1": ct["ct_v1"], "scent_sr": sc["sr"], "scent_ccat": sc["ccat"], "mce": mce["mce"]})
    return frame, scores, p["ordinal"].astype(float), c["cell_id"].str.split("_").str[1].to_numpy()


def real(deposit, xdir, wroot, out_dir, procs, n_draws, n_boot):
    cells, scores, sub = load(deposit)
    z, sc = cells["GSE106474"], scores["GSE106474"]
    o = z["ordinal"].to_numpy(float)
    sample = z["barcode"].str.rsplit("_", n=1).str[0].to_numpy()
    zi = z.iloc[sub["cell_index"]].reset_index(drop=True)
    blocks = [series_tasks("GSE106474", 0, z, sc, o, ROWS_FULL, n_draws, n_boot, sample),
              series_tasks("GSE106474", 1, zi, sub, o[sub["cell_index"]], ROWS_SUB, n_draws, n_boot, sample[sub["cell_index"]])]
    # stage and sequencing batch are partly confounded on GSE106474 (DS5 holds seven stages,
    # DS2 to DS4 the other five), so the registered statistic is repeated inside each batch
    batch = pd.Series(sample).str.extract(r"_(DS\d)")[0].to_numpy()
    within = []
    for bi, b in enumerate(np.unique(batch)):
        for r, (rows, frame, ss, oo, bb) in enumerate([(ROWS_FULL, z, sc, o, batch),
                                                       (ROWS_SUB, zi, sub, o[sub["cell_index"]], batch[sub["cell_index"]])]):
            keep = bb == b
            for i, (name, col, prim, _, _) in enumerate(rows):
                s = ss[col].to_numpy(float)[keep]
                p = (ss if prim.startswith("stemfinder") else frame)[prim].to_numpy(float)[keep]
                labels = strata_1d(p, m_for(len(s)), rng_for(6, bi, r, i))
                within.append((f"GSE106474 {b} {name}", s, [p], oo[keep], m_for(len(s)), ("linear", "stratum_mean"),
                               None, None, labels, (7, bi, r, i), n_draws))
    xf, xs, xsub = xenopus(cells, xdir)
    xo = xf["ordinal"].to_numpy(float)
    xi = xf.iloc[xsub["cell_index"]].reset_index(drop=True)
    blocks += [series_tasks("GSE113074", 2, xf, xs, xo, [r for r in ROWS_FULL if r[1] in xs.columns], n_draws, n_boot, None),
               series_tasks("GSE113074", 3, xi, xsub, xo[xsub["cell_index"]], [r for r in ROWS_SUB if r[1] in xsub.columns], n_draws, n_boot, None)]
    wf, ws, wo, wsample = wagner(wroot)
    blocks.append(series_tasks("GSE112294", 4, wf, ws, wo, [r for r in ROWS_FULL if r[1] in ws.columns], n_draws, n_boot, wsample))
    tasks = [t for b in blocks for t in b[0]] + within
    lo = [t for b in blocks for t in b[1]]
    boot = [t for b in blocks for t in b[2]]
    gc = [t for b in blocks for t in b[3]]
    for t, (name, col, acc) in enumerate(TRAINED):
        c, s = cells[acc], scores[acc][col].to_numpy(float)
        keep = c["ordinal"].notna().to_numpy()
        covs = [c[k].to_numpy(float)[keep] for k in JOINT]
        gc.append((f"{acc} {name} joint", s[keep], covs, c["ordinal"].to_numpy(float)[keep], None, (5, t)))
    print(f"{len(tasks)} floor tasks, {len(gc)} GCM, {len(lo)} leave-one-stage-out, {len(boot)} sample bootstraps", flush=True)
    with Pool(procs) as pool:
        floors = []
        for f in pool.imap(one_row, tasks, chunksize=1):
            floors.append(f)
            print(f["row"], f"m={f['m']}", f"{f['seconds']}s", json.dumps(f["registered"]), flush=True)
        gcms = pool.map(gcm_task, gc, chunksize=1)
        loso = pool.map(leave_one_stage_out, lo, chunksize=1)
        clusters = pool.map(sample_bootstrap, boot, chunksize=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "conditional_null.json").write_text(json.dumps(
        {"n_draws": n_draws, "n_boot": n_boot, "seed": SEED, "floors": floors, "gcm": gcms, "leave_one_stage_out": loso,
         "sample_bootstrap": clusters}, indent=1))


def synth_case(args):
    case, n, rho, c, rep_id, n_draws = args
    rng = rng_for(5, rep_id, int(rho * 1000), SYNTH_CASES.index(case))
    a, g = 2 * np.sin(np.pi * rho / 6), 2 * np.sin(np.pi * c / 6)
    if case.endswith("k4"):
        ps = 0.6 * rng.standard_normal(n) + 0.8 * rng.standard_normal((4, n))
        z = ps.mean(axis=0)
        z = (z - z.mean()) / z.std()
    else:
        ps = rng.standard_normal((1, n))
        z = ps[0]
    p = ps[0]
    # every case keeps s independent of o given the covariates, so the true conditional skill
    # is zero; in the curved cases ordinal and score share a curved function of the covariates,
    # which a straight line on ranks cannot remove
    curve = (z * z - 1) / np.sqrt(2) if case.startswith("curved") else z
    latent = g * curve + np.sqrt(1 - g * g) * rng.standard_normal(n)
    o = (rankdata(latent, method="ordinal") - 1) * 12 // n + 1.0
    if case.startswith("curved"):
        s = p + curve + (1 - rho) * rng.standard_normal(n)
    else:
        s = a * p + np.sqrt(1 - a * a) * rng.standard_normal(n)
    covs = list(ps)
    k = len(covs)
    axis = p
    rho_hat = abs(float(spearmanr(s, p).statistic))
    fl = grid_floor(n, rho_hat, kernel="taub", k_covariates=k, levels=12)
    fl = fl["floor"] if isinstance(fl, dict) else fl
    out = []
    for m in SYNTH_M:
        labels = strata_1d(axis, m, rng_for(6, rep_id, m))
        for which in ("linear", "stratum_mean"):
            v = cs_value(s, covs, labels, o, which)
            d = np.array([cs_value(permute_within(s, labels, rng_for(7, rep_id, m, i)), covs, labels, o, which)
                          for i in range(n_draws)])
            out.append({"n": n, "case": case, "rho": rho, "rho_hat": rho_hat, "m": m, "residual": which,
                        "value": v, "grid": fl, "perm": float(np.quantile(d, 0.975))})
    return out


def synthetic(out_dir, procs, reps, n_draws, n, cases):
    tasks = [(case, n, rho, 0.55, r, n_draws) for case in cases for rho in (0.5, 0.9) for r in range(reps)]
    with Pool(procs) as pool:
        res = [r for rows in pool.map(synth_case, tasks, chunksize=1) for r in rows]
    rows = []
    for (case, rho, m, which), g in pd.DataFrame(res).groupby(["case", "rho", "m", "residual"]):
        rows.append({"case": case, "rho_param": rho, "m": m, "residual": which,
                     "rho_hat_mean": float(g["rho_hat"].mean()), "reps": len(g), "mean_value": float(g["value"].mean()),
                     "false_clear_grid": float((g["value"] > g["grid"]).mean()),
                     "false_clear_perm": float((g["value"] > g["perm"]).mean())})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"synthetic_{'k4' if cases[0].endswith('k4') else 'k1'}_n{n}.json").write_text(json.dumps({"n": n, "n_draws": n_draws, "rows": rows}, indent=1))
    print(pd.DataFrame(rows).to_string(index=False))


def gcm_case(args):
    case, n, rho, rep_id = args
    rng = rng_for(5, rep_id, int(rho * 1000), SYNTH_CASES.index(case))
    a, g = 2 * np.sin(np.pi * rho / 6), 2 * np.sin(np.pi * 0.55 / 6)
    if case.endswith("k4"):
        ps = 0.6 * rng.standard_normal(n) + 0.8 * rng.standard_normal((4, n))
        z = ps.mean(axis=0)
        z = (z - z.mean()) / z.std()
    else:
        ps = rng.standard_normal((1, n))
        z = ps[0]
    p = ps[0]
    curve = (z * z - 1) / np.sqrt(2) if case.startswith("curved") else z
    latent = g * curve + np.sqrt(1 - g * g) * rng.standard_normal(n)
    o = (rankdata(latent, method="ordinal") - 1) * 12 // n + 1.0
    s = p + curve + (1 - rho) * rng.standard_normal(n) if case.startswith("curved") else a * p + np.sqrt(1 - a * a) * rng.standard_normal(n)
    r = gcm(s, o, list(ps), key=(rep_id,))
    return {"case": case, "rho": rho, **r}


def synthetic_gcm(out_dir, procs, reps, n):
    tasks = [(case, n, rho, r) for case in SYNTH_CASES for rho in (0.5, 0.9) for r in range(reps)]
    with Pool(procs) as pool:
        res = pd.DataFrame(pool.map(gcm_case, tasks, chunksize=2))
    rows = [{"case": c, "rho_param": r, "reps": len(g), "mean_T": float(g["T"].mean()), "sd_T": float(g["T"].std()),
             "false_clear": float(g["clears"].mean())} for (c, r), g in res.groupby(["case", "rho"])]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"synthetic_gcm_n{n}.json").write_text(json.dumps({"n": n, "rows": rows}, indent=1))
    print(pd.DataFrame(rows).to_string(index=False))


def power_case(args):
    delta, n, rep_id, n_draws = args
    rng = rng_for(10, rep_id, int(delta * 1000), n)
    p, b = rng.standard_normal(n), rng.standard_normal(n)
    o = (rankdata(0.6 * p + 0.8 * b, method="ordinal") - 1) * 12 // n + 1.0
    s = p + delta * b + rng.standard_normal(n)
    labels = strata_1d(p, m_for(n), rng_for(11, rep_id))
    v = cs_value(s, [p], labels, o, "linear")
    fl = grid_floor(n, abs(float(spearmanr(s, p).statistic)), kernel="taub", k_covariates=1, levels=12)["floor"]
    d = np.array([cs_value(permute_within(s, labels, rng_for(12, rep_id, i)), [p], labels, o, "linear")
                  for i in range(n_draws)])
    return {"delta": delta, "value": v, "grid": fl, "perm": float(np.quantile(d, 0.975))}


def power(out_dir, procs, reps, n_draws, n):
    """A planted positive control: how much skill beyond the primitive each floor detects."""
    tasks = [(delta, n, r, n_draws) for delta in (0.0, 0.02, 0.05, 0.1, 0.2, 0.5) for r in range(reps)]
    with Pool(procs) as pool:
        res = pool.map(power_case, tasks, chunksize=1)
    rows = [{"delta": d, "reps": len(g), "mean_cs": float(g["value"].mean()),
             "clears_grid": float((g["value"] > g["grid"]).mean()),
             "clears_perm": float((g["value"] > g["perm"]).mean())}
            for d, g in pd.DataFrame(res).groupby("delta")]
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"power_n{n}.json").write_text(json.dumps({"n": n, "n_draws": n_draws, "rows": rows}, indent=1))
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["synthetic", "gcm", "power", "real"])
    ap.add_argument("--deposit", default=str(REPO / "data/deposit/ownbaseline-data-v1"))
    ap.add_argument("--xenopus", default=str(data_root() / "fix3/results"))
    ap.add_argument("--wagner", default=str(data_root() / "wagner2018/root"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--procs", type=int, default=max((os.cpu_count() or 2) - 2, 1))
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--n", type=int, default=3000, help="cells per synthetic replicate")
    ap.add_argument("--cases", default="coupled,curved")
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    a = ap.parse_args()
    if a.mode == "synthetic":
        synthetic(Path(a.out), a.procs, a.reps, a.n_draws, a.n, a.cases.split(","))
    elif a.mode == "gcm":
        synthetic_gcm(Path(a.out), a.procs, a.reps, a.n)
    elif a.mode == "power":
        power(Path(a.out), a.procs, a.reps, a.n_draws, a.n)
    else:
        real(a.deposit, Path(a.xenopus), Path(a.wagner), Path(a.out), a.procs, a.n_draws, a.n_boot)


if __name__ == "__main__":
    main()
