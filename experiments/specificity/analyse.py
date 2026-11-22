#!/usr/bin/env python3
"""
analyse.py: the stage of the specificity plan that reads stage labels.

    python3 experiments/specificity/analyse.py --procs 6

Run only after PREREG.md is committed. It refuses to start unless

  - PREREG.md, blind_manifest.json and every .py file of this directory are committed and
    unchanged in the working tree, and
  - every control file it opens is listed in the committed blind_manifest.json with the
    SHA256 and the shapes recorded there,

so the controls it reads are the ones that existed, with those hashes, in the commit that
locked the plan. `--dry-run` skips those checks, shuffles the stage labels between cells before
anything is computed, and writes to results/dryrun.

Part A  negative controls: conditional skill of every control on each primitive and on the
        four jointly, how many controls clear the registered floor and the permutation
        floor, and where each audited score's increment sits among the controls' increments.
Part B  the trend over samples for every audited row.
Part C  sequencing depth and direction: gene count and CytoTRACE at equal depth and on the
        hours the two zebrafish series share.
Part D  the stratified concordance of every audited row.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per worker: the pool is the parallelism

import argparse
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import DEPOSIT, JOINT, REPO, SERIES, WORK, load_cells, num, rng_for, sha256, table  # noqa: E402
from stats import concordance_test, conditional_skill, residual, sample_level_test  # noqa: E402

from own_baseline.conditional_null import crossfit, cs_value, m_for, permute_within, strata_1d  # noqa: E402
from own_baseline.conditional_skill import align, kang_taub, rank_resid_multi  # noqa: E402
from own_baseline.floors import floor as grid_floor  # noqa: E402
from own_baseline.floors import grid as floor_grid  # noqa: E402

N_DRAWS = 200
N_PERM_SAMPLES = 100_000
PERM_SUBSET = {"GSE106474": 50, "GSE112294": 50, "GSE113074": 20}
QS = (0.025, 0.25, 0.5, 0.75, 0.975)
# name, score column, declared primitive
FULL = {"GSE106474": [("CytoTRACE", "cytotrace_v1", "detected_genes"), ("SR", "scent_sr", "pcc_x_degree"),
                      ("ORIGINS", "origins", "pcc_x_degree"), ("StemFinder", "stemfinder", "stemfinder_cell_cycle_score"),
                      ("CCAT", "scent_ccat", "pcc_x_degree"), ("MCE", "mce", "pcc_x_degree"),
                      ("GCS", "cytotrace_gene_count_signature", "detected_genes")],
        "GSE112294": [("CytoTRACE", "cytotrace_v1", "detected_genes"), ("SR", "scent_sr", "pcc_x_degree"),
                      ("CCAT", "scent_ccat", "pcc_x_degree"), ("MCE", "mce", "pcc_x_degree"),
                      ("GCS", "cytotrace_gene_count_signature", "detected_genes")],
        "GSE113074": [("CytoTRACE", "cytotrace_v1", "detected_genes"), ("SR", "scent_sr", "pcc_x_degree"),
                      ("ORIGINS", "origins", "pcc_x_degree"), ("CCAT", "scent_ccat", "pcc_x_degree"),
                      ("GCS", "cytotrace_gene_count_signature", "detected_genes")]}
SUB = {"GSE106474": [("NCG", "ncg", "pcc_x_degree"), ("dpath", "dpath", "shannon_entropy"), ("SLICE", "slice", "shannon_entropy")],
       "GSE113074": [("dpath", "dpath", "shannon_entropy"), ("SLICE", "slice", "shannon_entropy")]}
SCORE_FILE = {"GSE106474": "GSE106474_scores", "GSE112294": "GSE112294_scores", "GSE113074": "GSE113074_audit_scores"}
SUB_FILE = {"GSE106474": "GSE106474_subsample3000_scores", "GSE113074": "GSE113074_subsample3000_scores"}
FAMILIES = ("a1_g200", "a1_g20", "a1_g2000", "a2")
MANIFEST = HERE / "blind_manifest.json"
_LOCK = {"manifest": None, "head": None}


# ---------------------------------------------------------------------------------------
# The lock
# ---------------------------------------------------------------------------------------
def check_lock():
    """The plan, the manifest and the code must be the committed ones. Returns the commit and the manifest."""
    rel = HERE.relative_to(REPO).as_posix()
    run = lambda *args: subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout  # noqa: E731
    dirty = [ln for ln in run("status", "--porcelain", "--", rel).splitlines() if "/results/" not in ln]
    if dirty:
        raise SystemExit("uncommitted changes under experiments/specificity; commit the plan and the code first:\n" + "\n".join(dirty))
    plan = (HERE / "PREREG.md").read_text()
    if "BLIND_FILES_TABLE" in plan or "CALIBRATION_TABLES" in plan:
        raise SystemExit("PREREG.md still carries a placeholder")
    committed = run("show", f"HEAD:{rel}/blind_manifest.json")
    if committed != MANIFEST.read_text():
        raise SystemExit("blind_manifest.json differs from the committed one")
    return run("rev-parse", "HEAD").strip(), json.loads(committed)


def open_blind(name, dry_run):
    """np.load of a control file, after its hash and shapes are checked against the committed manifest."""
    path = WORK / name
    if not dry_run:
        m = _LOCK["manifest"]
        if name not in m:
            raise SystemExit(f"{name} is not in the committed manifest")
        if sha256(path) != m[name]["sha256"]:
            raise SystemExit(f"{name} does not match the hash recorded before the labels were read")
    z = np.load(path)
    if not dry_run:
        for key, shape in _LOCK["manifest"][name]["shapes"].items():
            if list(z[key].shape) != shape:
                raise SystemExit(f"{name}:{key} has shape {z[key].shape}, the manifest records {shape}")
    return z


# ---------------------------------------------------------------------------------------
# Workers
# ---------------------------------------------------------------------------------------
def cs_block(args):
    """Conditional skill, marginal tau_b and score-covariate correlations for each column of a block."""
    block, covs, o, want_marginal = args
    k = block.shape[1]
    cs, tau, rmax = np.empty(k), np.full(k, np.nan), np.empty(k)
    for j in range(k):
        s = block[:, j].astype(np.float64)
        cs[j] = conditional_skill(s, covs, o)
        rmax[j] = max(abs(float(spearmanr(s, c).statistic)) for c in covs)
        if want_marginal:
            tau[j] = kang_taub(s, o)
    return cs, tau, rmax


def perm_floor_block(args):
    """Permutation floor of the registered statistic for each column of a block, shared strata."""
    block, p, o, labels, key, n_draws = args
    out = np.empty(block.shape[1])
    for j in range(block.shape[1]):
        s = block[:, j].astype(np.float64)
        d = np.array([cs_value(permute_within(s, labels, rng_for(*key, j, i)), [p], labels, o, "linear") for i in range(n_draws)])
        out[j] = np.quantile(d, 0.975)
    return out


def chunks(n, size):
    return [slice(a, min(a + size, n)) for a in range(0, n, size)]


def summarise(v):
    return {"n": int(len(v)), "mean": float(np.mean(v)), "quantiles": {str(q): float(np.quantile(v, q)) for q in QS},
            "fraction_positive": float(np.mean(v > 0))}


def floors_for(n, r, k):
    """
    Two registered floors for a control at score-covariate correlation r: the nearest grid point, the
    lookup the audit uses, and the larger of the two grid points that bracket r. A control seldom sits
    on a grid point, and the nearest one can lie far below it, where the floor is lower; the bracketed
    floor is the harder of the two to clear. Returns (nearest, bracketed, distant).
    """
    near = grid_floor(n, r, kernel="taub", k_covariates=k, levels=12)
    if near.get("floor") is None:
        return np.nan, np.nan, 1.0
    rhos = floor_grid()[k]["rho"]
    lo = max([g for g in rhos if g <= r], default=rhos[0])
    hi = min([g for g in rhos if g >= r], default=rhos[-1])
    both = [grid_floor(n, g, kernel="taub", k_covariates=k, levels=12)["floor"] for g in {lo, hi}]
    return near["floor"], max(both), float(bool(near["distant"]))


def control_distributions(pool, geo, mats, o, tag, families, conds, registered_floors, size=25):
    """Every control of every family on every conditioning set. Returns (summary, raw values)."""
    out, raw = {}, {}
    n = len(o)
    first = next(iter(conds))
    for fam in families:
        mat = mats[fam]
        out[fam] = {}
        for cname, covs in conds.items():
            jobs = [(mat[:, sl], covs, o, cname == first) for sl in chunks(mat.shape[1], size)]
            res = pool.map(cs_block, jobs, chunksize=1)
            cs = np.concatenate([r[0] for r in res])
            rmax = np.concatenate([r[2] for r in res])
            d = {"conditional_skill": summarise(cs), "max_abs_spearman_with_covariates": summarise(rmax)}
            if cname == first:
                tau = np.concatenate([r[1] for r in res])
                out[fam]["marginal_abs_tau"] = summarise(np.abs(tau))
                raw[f"{tag}|{fam}|marginal_tau"] = tau
            if registered_floors:
                fl = np.array([floors_for(n, r, len(covs)) for r in rmax], dtype=float)
                ok = np.isfinite(fl[:, 0])
                near, brk, far = fl[ok, 0], fl[ok, 1], fl[ok, 2] > 0.5
                d["controls_without_a_grid_floor"] = int((~ok).sum())
                d["controls_distant_from_grid"] = int(far.sum())
                d["fraction_above_registered_floor_nearest"] = float(np.mean(cs[ok] > near)) if ok.any() else None
                d["fraction_above_registered_floor_bracketed"] = float(np.mean(cs[ok] > brk)) if ok.any() else None
                d["fraction_above_registered_floor_nearest_not_distant"] = (
                    float(np.mean(cs[ok][~far] > near[~far])) if (~far).any() else None)
            out[fam][cname] = d
            raw[f"{tag}|{fam}|{cname}"] = cs
            raw[f"{tag}|{fam}|{cname}|rho"] = rmax
            print(geo, tag, fam, cname, json.dumps(d["conditional_skill"]["quantiles"]), flush=True)
    return out, raw


def percentile(value, dist):
    return float(np.mean(dist < value))


def boosted_residual(s, covs, o, key):
    """Rank of the aligned score minus its cross-fitted boosted prediction from the covariates' ranks."""
    y = rankdata(align(s, o))
    x = np.column_stack([rankdata(c) for c in covs])
    folds = rng_for(*key).integers(0, 5, len(y))
    return y - crossfit(y, x, folds)


def series(pool, geo, n_draws, n_perm, dry_run=False, max_controls=None):
    t0 = time.time()
    sid = SERIES.index(geo)
    cells = load_cells(geo)
    o = num(cells["ordinal"])
    if dry_run:      # a smoke test of the code path: the stage labels are shuffled between cells first
        o = rng_for(99, sid).permutation(o)
    n = len(o)
    scores = table(DEPOSIT / "scores" / f"{SCORE_FILE[geo]}.csv.gz")
    frame = {k: num(cells[k]) for k in JOINT}
    if "stemfinder_cell_cycle_score" in scores:
        frame["stemfinder_cell_cycle_score"] = num(scores["stemfinder_cell_cycle_score"])
    sig = open_blind(f"{geo}_signatures.npz", dry_run)
    mats = {f: (sig[f] if max_controls is None else sig[f][:, :max_controls]) for f in FAMILIES}
    conds = {p: [frame[p]] for p in JOINT}
    conds["joint"] = [frame[p] for p in JOINT]
    if "stemfinder_cell_cycle_score" in frame:
        conds["stemfinder_cell_cycle_score"] = [frame["stemfinder_cell_cycle_score"]]
    out = {"series": geo, "n": n, "levels": int(len(np.unique(o))), "dry_run": bool(dry_run), "locked_at_commit": _LOCK["head"]}

    # ---- Part A: the control distributions on all cells
    out["controls"], raw = control_distributions(pool, geo, mats, o, "full", FAMILIES, conds, geo == "GSE106474")

    # permutation floors for the first controls of the primary family, one primitive at a time
    m = m_for(n)
    n_sub = PERM_SUBSET[geo] if max_controls is None else min(PERM_SUBSET[geo], 4)
    out["controls_permutation_floor"] = {"m": m, "controls": n_sub, "n_draws": n_draws, "by_primitive": {}}
    sub = mats["a1_g200"][:, :n_sub]
    for pi, p in enumerate(JOINT):
        labels = strata_1d(frame[p], m, rng_for(5, sid, pi))
        jobs = [(sub[:, sl], frame[p], o, labels, (6, sid, pi, sl.start), n_draws) for sl in chunks(sub.shape[1], 2)]
        fl = np.concatenate(pool.map(perm_floor_block, jobs, chunksize=1))
        cs = raw[f"full|a1_g200|{p}"][:n_sub]
        out["controls_permutation_floor"]["by_primitive"][p] = {
            "fraction_above": float(np.mean(cs > fl)), "floor_quantiles": {str(q): float(np.quantile(fl, q)) for q in (0.025, 0.5, 0.975)}}
        print(geo, "perm floor", p, out["controls_permutation_floor"]["by_primitive"][p], flush=True)

    # ---- the audited rows
    sample = batch = None
    if geo == "GSE106474":
        sample = np.array([b.rsplit("_", 1)[0] for b in cells["barcode"]])
        batch = np.array([s.split("_")[2][:3] for s in sample])
    elif geo == "GSE112294":
        sample = np.array(cells["sample"])
    rows = {}
    blocks = [("full", FULL[geo], frame, {c: num(scores[c]) for _, c, _ in FULL[geo]}, o, sample, batch)]
    if geo in SUB:
        st = table(DEPOSIT / "scores" / f"{SUB_FILE[geo]}.csv.gz")
        ci = num(st["cell_index"]).astype(int)
        sframe = {k: v[ci] for k, v in frame.items()}
        smats = {f: mats[f][ci] for f in ("a1_g200", "a2")}
        sconds = {p: [sframe[p]] for p in ("pcc_x_degree", "shannon_entropy")}
        sconds["joint"] = [sframe[p] for p in JOINT]
        out["controls_subsample"], sraw = control_distributions(pool, geo, smats, o[ci], "sub", ("a1_g200", "a2"),
                                                               sconds, geo == "GSE106474", size=100)
        raw.update(sraw)
        blocks.append(("sub", SUB[geo], sframe, {c: num(st[c]) for _, c, _ in SUB[geo]}, o[ci],
                       None if sample is None else sample[ci], None if batch is None else batch[ci]))
    for bi, (tag, defs, fr, sc, oo, smp, bat) in enumerate(blocks):
        for ri, (name, col, prim) in enumerate(defs):
            s, p = sc[col], fr[prim]
            cs = conditional_skill(s, [p], oo)
            row = {"n": int(len(oo)), "primitive": prim, "conditional_skill": cs, "marginal_tau": kang_taub(s, oo),
                   "spearman_with_stage": float(spearmanr(s, oo).statistic),
                   "spearman_with_primitive": float(spearmanr(s, p).statistic)}
            joint = [fr[q] for q in JOINT]
            row["conditional_skill_joint"] = conditional_skill(s, joint, oo)
            row["max_abs_spearman_with_joint_covariates"] = max(abs(float(spearmanr(s, c).statistic)) for c in joint)
            # where the row's increment sits among the controls' increments over the same covariates, on the same cells
            row["vs_controls"] = {}
            for fam in (FAMILIES if tag == "full" else ("a1_g200", "a2")):
                d1, dj = raw[f"{tag}|{fam}|{prim}"], raw[f"{tag}|{fam}|joint"]
                dm = np.abs(raw[f"{tag}|{fam}|marginal_tau"])
                row["vs_controls"][fam] = {
                    "percentile": percentile(cs, d1), "control_p975": float(np.quantile(d1, 0.975)),
                    "exceeds_p975": bool(cs > np.quantile(d1, 0.975)),
                    "control_median_abs_spearman_with_primitive": float(np.median(raw[f"{tag}|{fam}|{prim}|rho"])),
                    "joint_percentile": percentile(row["conditional_skill_joint"], dj),
                    "joint_control_p975": float(np.quantile(dj, 0.975)),
                    "joint_exceeds_p975": bool(row["conditional_skill_joint"] > np.quantile(dj, 0.975)),
                    "control_median_max_abs_spearman_with_joint": float(np.median(raw[f"{tag}|{fam}|joint|rho"])),
                    "marginal_percentile": percentile(abs(row["marginal_tau"]), dm),
                    "fraction_of_controls_with_larger_abs_marginal_tau": float(np.mean(dm > abs(row["marginal_tau"])))}
            # ---- Part B: the trend over samples. The stratum-mean residual decides; the least-squares one is sensitivity.
            if smp is not None:
                labels = strata_1d(p, m_for(len(oo)), rng_for(7, sid, bi, ri))
                for which, key in (("stratum_mean", "sample_test"), ("linear", "sample_test_linear")):
                    r = residual(s, [p], oo, labels, which)
                    row[key] = sample_level_test(r, smp, oo, bat, n_perm=n_perm, rng=rng_for(8, sid, bi, ri))
                    if bat is not None:
                        row[f"{key}_no_batch_strata"] = sample_level_test(r, smp, oo, None, n_perm=n_perm, rng=rng_for(9, sid, bi, ri))
                rj = boosted_residual(s, joint, oo, (10, sid, bi, ri))
                row["sample_test_joint_boosted"] = sample_level_test(rj, smp, oo, bat, n_perm=n_perm, rng=rng_for(13, sid, bi, ri))
            # ---- Part D: the stratified concordance
            labels = strata_1d(p, m_for(len(oo)), rng_for(11, sid, bi, ri))
            row["concordance"] = concordance_test(s, oo, labels, rng_for, (12, sid, bi, ri), n_draws)
            rows[name] = row
            print(geo, name, json.dumps({k: v for k, v in row.items() if k != "vs_controls"}), flush=True)
    out["rows"] = rows

    # ---- A3: CytoTRACE with a random gene set, and the same gene sets without smoothing
    if geo != "GSE113074":
        ct = open_blind(f"{geo}_cytotrace_controls.npz", dry_run)
        gc = frame["detected_genes"]
        a3 = {}
        for key in ("a3", "a3_unsmoothed"):
            mat = ct[key] if max_controls is None else ct[key][:, :max_controls]
            res = pool.map(cs_block, [(mat[:, sl], [gc], o, True) for sl in chunks(mat.shape[1], 10)], chunksize=1)
            cs, tau = np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])
            resj = pool.map(cs_block, [(mat[:, sl], [frame[q] for q in JOINT], o, False) for sl in chunks(mat.shape[1], 10)], chunksize=1)
            csj = np.concatenate([r[0] for r in resj])
            raw[f"full|{key}|detected_genes"], raw[f"full|{key}|joint"], raw[f"full|{key}|marginal_tau"] = cs, csj, tau
            raw[f"full|{key}|detected_genes|rho"] = np.concatenate([r[2] for r in res])
            a3[key] = {"detected_genes": summarise(cs), "joint": summarise(csj), "marginal_abs_tau": summarise(np.abs(tau)),
                       "marginal_tau_fraction_positive": float(np.mean(tau > 0)),
                       "abs_spearman_with_gene_count": summarise(raw[f"full|{key}|detected_genes|rho"])}
        for name, ref in (("CytoTRACE", "a3"), ("GCS", "a3_unsmoothed")):
            r = rows[name]
            d1, dj, dm = raw[f"full|{ref}|detected_genes"], raw[f"full|{ref}|joint"], np.abs(raw[f"full|{ref}|marginal_tau"])
            r["vs_controls"][ref] = {"percentile": percentile(r["conditional_skill"], d1),
                                     "control_p975": float(np.quantile(d1, 0.975)),
                                     "exceeds_p975": bool(r["conditional_skill"] > np.quantile(d1, 0.975)),
                                     "joint_percentile": percentile(r["conditional_skill_joint"], dj),
                                     "joint_control_p975": float(np.quantile(dj, 0.975)),
                                     "joint_exceeds_p975": bool(r["conditional_skill_joint"] > np.quantile(dj, 0.975)),
                                     "marginal_percentile": percentile(abs(r["marginal_tau"]), dm)}
        out["cytotrace_ablation"] = a3
        print(geo, "A3", json.dumps(a3), flush=True)

    # ---- Part C: depth and direction
    lib = 10 ** frame["log10_library_size"]
    gc = frame["detected_genes"]
    depth = {"spearman_library_size_with_stage": float(spearmanr(lib, o).statistic),
             "spearman_detected_genes_with_stage": float(spearmanr(gc, o).statistic),
             "spearman_detected_genes_with_stage_given_library_size": float(
                 spearmanr(rank_resid_multi(gc, [frame["log10_library_size"]]), o).statistic),
             "per_stage": {int(st): {"cells": int((o == st).sum()), "median_library_size": float(np.median(lib[o == st])),
                                     "median_detected_genes": float(np.median(gc[o == st]))} for st in np.unique(o)},
             "thinned": {}}
    th = open_blind(f"{geo}_thinned.npz", dry_run)
    for d in (1000, 500, 2000):
        keep = th[f"keep_{d}"]
        depth["thinned"][d] = {"cells_kept": int(len(keep)),
                               "cells_kept_per_stage": {int(st): int((o[keep] == st).sum()) for st in np.unique(o)},
                               "spearman_detected_genes_with_stage": float(spearmanr(th[f"genes_{d}"], o[keep]).statistic),
                               "median_detected_genes_per_stage": {int(st): float(np.median(th[f"genes_{d}"][o[keep] == st]))
                                                                   for st in np.unique(o[keep])}}
    if "cytotrace_thinned" in th:
        keep = th["keep_1000"]
        t = depth["thinned"][1000]
        t["spearman_cytotrace_with_stage"] = float(spearmanr(th["cytotrace_thinned"], o[keep]).statistic)
        t["spearman_gene_count_signature_with_stage"] = float(spearmanr(th["gcs_thinned"], o[keep]).statistic)
        t["cytotrace_conditional_skill_on_thinned_gene_count"] = conditional_skill(th["cytotrace_thinned"], [th["genes_1000"]], o[keep])
        # the same cells at raw depth: what dropping the shallow cells does without the thinning
        depth["kept_cells_raw_depth"] = {
            "cells": int(len(keep)),
            "spearman_detected_genes_with_stage": float(spearmanr(gc[keep], o[keep]).statistic),
            "spearman_gene_count_signature_with_stage": float(spearmanr(th["gcs_kept_raw"], o[keep]).statistic),
            "spearman_cytotrace_with_stage": float(spearmanr(th["cytotrace_kept_raw"], o[keep]).statistic)}
    if geo != "GSE113074":
        w = open_blind(f"{geo}_window.npz", dry_run)
        depth["shared_window"] = {}
        for tag, cells_key in (("window", "window_cells"), ("window_thinned", "window_thinned_cells")):
            ow = o[w[cells_key]]
            depth["shared_window"][tag] = {
                "cells": int(len(ow)), "cells_per_stage": {int(st): int((ow == st).sum()) for st in np.unique(ow)},
                "spearman_detected_genes_with_stage": float(spearmanr(w[f"genes_{tag}"], ow).statistic),
                "spearman_gene_count_signature_with_stage": float(spearmanr(w[f"gcs_{tag}"], ow).statistic),
                "spearman_cytotrace_with_stage": float(spearmanr(w[f"cytotrace_{tag}"], ow).statistic)}
        ow = o[w["window_cells"]]      # the stored scores on the same window cells, beside the reruns
        depth["shared_window"]["stored_scores_on_window_cells"] = {
            "spearman_cytotrace_with_stage": float(spearmanr(num(scores["cytotrace_v1"])[w["window_cells"]], ow).statistic),
            "spearman_gene_count_signature_with_stage": float(
                spearmanr(num(scores["cytotrace_gene_count_signature"])[w["window_cells"]], ow).statistic)}
    out["depth"] = depth
    print(geo, "depth", json.dumps({k: v for k, v in depth.items() if k != "per_stage"}), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out, raw


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", default=",".join(SERIES))
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-perm", type=int, default=N_PERM_SAMPLES)
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--max-controls", type=int, default=None, help="dry run only: truncate every control family")
    ap.add_argument("--dry-run", action="store_true", help="shuffle the stage labels between cells; writes to <out>/dryrun")
    a = ap.parse_args()
    if a.max_controls is not None and not a.dry_run:
        raise SystemExit("--max-controls is for --dry-run only")
    if not a.dry_run:
        _LOCK["head"], _LOCK["manifest"] = check_lock()
    out_dir = Path(a.out) / "dryrun" if a.dry_run else Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as pool:
        for geo in a.series.split(","):
            res, raw = series(pool, geo, a.n_draws, a.n_perm, a.dry_run, a.max_controls)
            (out_dir / f"specificity_{geo}.json").write_text(json.dumps(res, indent=1))
            np.savez(out_dir / f"specificity_{geo}_control_values.npz", **raw)


if __name__ == "__main__":
    main()
