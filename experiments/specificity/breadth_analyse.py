#!/usr/bin/env python3
"""
breadth_analyse.py: the stage of PREREG_breadth.md that reads stage labels.

    python3 experiments/specificity/breadth_analyse.py GSE169210 --procs 6

Run only after PREREG_breadth.md and blind_manifest_breadth.json are committed. It refuses to
start on a dirty tree under experiments/specificity, and it opens a blind file or the label
table only if its SHA256 is the one in the committed manifest. `--dry-run` skips the lock,
shuffles the stage labels between cells before anything is computed and writes to
results/dryrun.

The statistics are those of analyse.py, through its own functions: the control distributions
(A1, 1,000 random 200-gene sets; A2, 1,000 degree permutations) on each primitive and on the
four jointly, the permutation floor, each audited row against the controls, the trend over
the replicated unit, and depth. Parts not carried over are named in the plan.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import csv
import json
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from analyse import chunks, control_distributions, cs_block, perm_floor_block, percentile  # noqa: E402,F401
from breadth_blind import MANIFEST, SID, paths  # noqa: E402
from common import JOINT, REPO, rng_for, sha256  # noqa: E402
from stats import conditional_skill, residual  # noqa: E402

from own_baseline.conditional_null import m_for, strata_1d  # noqa: E402
from own_baseline.conditional_skill import kang_taub, rank_resid_multi  # noqa: E402

N_DRAWS, N_PERM, PERM_CONTROLS = 200, 100_000, 50
FAMILIES = ("a1_g200", "a2")
PLAN = HERE / "PREREG_breadth.md"
NAMES = {"GSE169210": "mouse, Mittnenzweig 2021", "GSE223922": "zebrafish 14 to 120 hpf, Sur 2023"}
PRIMITIVE = {"detected_genes": "gene count", "pcc_x_degree": "PCC(x, degree)", "shannon_entropy": "Shannon entropy",
             "log10_library_size": "log library size", "joint": "all four"}
# PREREG_breadth.md, the same for both series: is the joint increment above the 97.5th percentile of the
# random gene sets, and is the residual trend supported over the replicated unit
EXPECT_JOINT = {"CytoTRACE": True, "SR": True, "CCAT": False, "MCE": False}
EXPECT_UNIT = {"CytoTRACE": True, "SR": True, "CCAT": True, "MCE": True}
ROWS = [("CytoTRACE", "cytotrace_v1", "detected_genes"), ("SR", "scent_sr", "pcc_x_degree"),
        ("CCAT", "scent_ccat", "pcc_x_degree"), ("MCE", "mce", "pcc_x_degree"),
        ("GCS", "cytotrace_gene_count_signature", "detected_genes")]
_LOCK = {"manifest": None, "head": None}


def check_lock():
    rel = HERE.relative_to(REPO).as_posix()
    run = lambda *args: subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout  # noqa: E731
    # the plan, the manifest and every file this analysis runs through; other work shares the directory, so it is not all of it
    mine = [f"{rel}/{f}" for f in (PLAN.name, MANIFEST.name, "breadth_analyse.py", "breadth_blind.py", "analyse.py", "common.py",
                                   "controls.py", "stats.py")]
    mine += ["experiments/mittnenzweig2021/prep_mittnenzweig.py", "experiments/sur2023/prep_sur.py", "own_baseline"]
    dirty = run("status", "--porcelain", "--", *mine).splitlines()
    if dirty:
        raise SystemExit("uncommitted changes in the plan or its code; commit them first:\n" + "\n".join(dirty))
    run("show", f"HEAD:{rel}/{PLAN.name}")
    committed = run("show", f"HEAD:{rel}/{MANIFEST.name}")
    if committed != MANIFEST.read_text():
        raise SystemExit(f"{MANIFEST.name} differs from the committed one")
    return run("rev-parse", "HEAD").strip(), json.loads(committed)


def locked(geo, name, path, dry_run):
    """The path, after its hash is checked against the committed manifest."""
    if not dry_run:
        m = _LOCK["manifest"].get(f"{geo}/{name}")
        if m is None or sha256(path) != m["sha256"]:
            raise SystemExit(f"{geo}/{name} is not the file hashed before the labels were read")
    return path


def read_labels(path):
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def kendall_s(r, o):
    return float((np.sign(r[:, None] - r[None, :]) * np.sign(o[:, None] - o[None, :])).sum()) / 2.0


def sample_trend_test(resid, sample, stage, batch=None, *, n_perm=N_PERM, rng=None, min_cells=5):
    """
    stats.sample_level_test with the permutation block sized to the group, so that a group of 150
    units fits in memory. Same statistic, same null: Kendall's S between the per-sample mean residual
    and the sample's stage, summed over batches, stage labels permuted between the samples of a batch.
    """
    ids, inv, counts = np.unique(sample, return_inverse=True, return_counts=True)
    means = np.bincount(inv, resid) / counts
    first = np.array([np.flatnonzero(inv == j)[0] for j in range(len(ids))])
    st = np.asarray(stage, dtype=float)[first]
    bt = np.zeros(len(ids), dtype=int) if batch is None else np.unique(np.asarray(batch)[first], return_inverse=True)[1]
    keep = counts >= min_cells
    means, st, bt = means[keep], st[keep], bt[keep]
    groups = [np.flatnonzero(bt == b) for b in np.unique(bt)]
    groups = [g for g in groups if len(np.unique(st[g])) > 1]
    s_obs = sum(kendall_s(means[g], st[g]) for g in groups)
    pairs = sum(float(np.abs(np.sign(st[g][:, None] - st[g][None, :])).sum()) / 2.0 for g in groups)
    s_null = np.zeros(n_perm)
    for g in groups:
        a = np.sign(means[g][:, None] - means[g][None, :]).astype(np.float32)
        step = max(1, min(10_000, int(2e7 // len(g) ** 2)))
        for lo in range(0, n_perm, step):
            hi = min(lo + step, n_perm)
            op = st[g][np.argsort(rng.random((hi - lo, len(g))), axis=1)].astype(np.float32)
            b = np.sign(op[:, :, None] - op[:, None, :])
            s_null[lo:hi] += np.einsum("ij,kij->k", a, b) / 2.0
    p = float((1 + np.sum(np.abs(s_null) >= abs(s_obs) - 1e-9)) / (n_perm + 1))
    return {"S": s_obs, "tau": s_obs / pairs if pairs else float("nan"), "p_two_sided": p,
            "samples": int(keep.sum()), "samples_dropped": int((~keep).sum()),
            "batches": len(groups), "samples_in_tested_batches": int(sum(len(g) for g in groups)),
            "pairs": pairs, "n_perm": n_perm}


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def series(pool, geo, n_draws, n_perm, dry_run=False, max_controls=None):
    t0 = time.time()
    sid, p = SID[geo], paths(geo)
    lab = read_labels(locked(geo, "labels.tsv", p["labels"], dry_run))
    o = lab["ordinal"].astype(float)
    if dry_run:
        o = rng_for(99, sid).permutation(o)
    sample = lab["sample"]
    batch = lab["batch"] if len(np.unique(lab["batch"])) > 1 else None
    n = len(o)
    prim = np.load(locked(geo, "primitives.npz", p["prep"] / "primitives.npz", dry_run))
    frame = {"detected_genes": prim["gene_count"], "pcc_x_degree": prim["pcc_string_v12"],
             "shannon_entropy": prim["shannon_H"], "log10_library_size": prim["log10_lib"]}
    sig = np.load(locked(geo, "signatures.npz", p["blind"] / "signatures.npz", dry_run))
    sc = np.load(locked(geo, "scores.npz", p["blind"] / "scores.npz", dry_run))
    th = np.load(locked(geo, "thinned.npz", p["blind"] / "thinned.npz", dry_run))
    if not (len(frame["detected_genes"]) == sig["a1_g200"].shape[0] == len(sc["cytotrace_v1"]) == n):
        raise SystemExit("labels, primitives, controls and scores do not hold the same cells")
    mats = {f: (sig[f] if max_controls is None else sig[f][:, :max_controls]) for f in FAMILIES}
    conds = {q: [frame[q]] for q in JOINT}
    conds["joint"] = [frame[q] for q in JOINT]
    out = {"series": geo, "n": n, "levels": int(len(np.unique(o))), "samples": int(len(np.unique(sample))),
           "batches": 1 if batch is None else int(len(np.unique(batch))), "dry_run": bool(dry_run), "locked_at_commit": _LOCK["head"]}

    # ---- the control distributions, with the audit's grid floor read at 12 levels and the nearest n
    out["controls"], raw = control_distributions(pool, geo, mats, o, "full", FAMILIES, conds, True)

    # ---- permutation floors: the first controls of A1 on each primitive
    m = m_for(n)
    n_sub = PERM_CONTROLS if max_controls is None else min(PERM_CONTROLS, 4)
    out["controls_permutation_floor"] = {"m": m, "controls": n_sub, "n_draws": n_draws, "by_primitive": {}}
    sub = mats["a1_g200"][:, :n_sub]
    strata = {}
    for pi, q in enumerate(JOINT):
        strata[q] = strata_1d(frame[q], m, rng_for(5, sid, pi))
        jobs = [(sub[:, sl], frame[q], o, strata[q], (6, sid, pi, sl.start), n_draws) for sl in chunks(sub.shape[1], 2)]
        fl = np.concatenate(pool.map(perm_floor_block, jobs, chunksize=1))
        cs = raw[f"full|a1_g200|{q}"][:n_sub]
        out["controls_permutation_floor"]["by_primitive"][q] = {
            "fraction_above": float(np.mean(cs > fl)), "floor_quantiles": {str(x): float(np.quantile(fl, x)) for x in (0.025, 0.5, 0.975)}}
        print(geo, "perm floor", q, out["controls_permutation_floor"]["by_primitive"][q], flush=True)

    # ---- the audited rows
    rows = {}
    joint = [frame[q] for q in JOINT]
    for ri, (name, col, pname) in enumerate(ROWS):
        if col not in sc.files:
            continue
        s, pv = sc[col].astype(np.float64), frame[pname]
        cs = conditional_skill(s, [pv], o)
        fl = float(perm_floor_block((s[:, None], pv, o, strata[pname], (14, sid, ri), n_draws))[0])
        row = {"n": n, "primitive": pname, "conditional_skill": cs, "permutation_floor": fl, "clears_permutation_floor": bool(cs > fl),
               "marginal_tau": kang_taub(s, o), "spearman_with_stage": rho(s, o), "spearman_with_primitive": rho(s, pv),
               "conditional_skill_joint": conditional_skill(s, joint, o),
               "max_abs_spearman_with_joint_covariates": max(abs(rho(s, c)) for c in joint), "vs_controls": {}}
        for fam in FAMILIES:
            d1, dj = raw[f"full|{fam}|{pname}"], raw[f"full|{fam}|joint"]
            dm = np.abs(raw[f"full|{fam}|marginal_tau"])
            row["vs_controls"][fam] = {
                "percentile": percentile(cs, d1), "control_p975": float(np.quantile(d1, 0.975)),
                "exceeds_p975": bool(cs > np.quantile(d1, 0.975)),
                "control_median_abs_spearman_with_primitive": float(np.median(raw[f"full|{fam}|{pname}|rho"])),
                "joint_percentile": percentile(row["conditional_skill_joint"], dj),
                "joint_control_p975": float(np.quantile(dj, 0.975)),
                "joint_exceeds_p975": bool(row["conditional_skill_joint"] > np.quantile(dj, 0.975)),
                "fraction_of_controls_with_larger_abs_marginal_tau": float(np.mean(dm > abs(row["marginal_tau"])))}
        # the trend over the replicated unit: the stratum-mean residual decides, least squares is sensitivity
        labels = strata_1d(pv, m, rng_for(7, sid, ri))
        for which, key in (("stratum_mean", "sample_test"), ("linear", "sample_test_linear")):
            r = residual(s, [pv], o, labels, which)
            row[key] = sample_trend_test(r, sample, o, batch, n_perm=n_perm, rng=rng_for(8, sid, ri))
            if batch is not None:
                row[f"{key}_no_batch_strata"] = sample_trend_test(r, sample, o, None, n_perm=n_perm, rng=rng_for(9, sid, ri))
            if "batch2" in lab:
                row[f"{key}_batch2"] = sample_trend_test(r, sample, o, lab["batch2"], n_perm=n_perm, rng=rng_for(15, sid, ri))
        rows[name] = row
        print(geo, name, json.dumps({k: v for k, v in row.items() if k != "vs_controls"}), flush=True)
    out["rows"] = rows

    # ---- depth
    lib, gc = 10 ** frame["log10_library_size"], frame["detected_genes"]
    depth = {"spearman_library_size_with_stage": rho(lib, o), "spearman_detected_genes_with_stage": rho(gc, o),
             "spearman_detected_genes_with_stage_given_library_size": rho(rank_resid_multi(gc, [frame["log10_library_size"]]), o),
             "spearman_gene_count_signature_with_stage": rho(sc["cytotrace_gene_count_signature"], o),
             "spearman_cytotrace_with_stage": rho(sc["cytotrace_v1"], o),
             "per_stage": {str(st): {"cells": int((o == st).sum()), "median_library_size": float(np.median(lib[o == st])),
                                     "median_detected_genes": float(np.median(gc[o == st]))} for st in np.unique(o)},
             "thinned": {}}
    for d in (1000, 500, 2000):
        keep = th[f"keep_{d}"]
        depth["thinned"][d] = {"cells_kept": int(len(keep)), "samples_kept": int(len(np.unique(sample[keep]))),
                               "cells_kept_per_stage": {str(st): int((o[keep] == st).sum()) for st in np.unique(o)},
                               "spearman_detected_genes_with_stage": rho(th[f"genes_{d}"], o[keep])}
    keep, t = th["keep_1000"], depth["thinned"][1000]
    t["spearman_cytotrace_with_stage"] = rho(th["cytotrace_thinned"], o[keep])
    t["spearman_gene_count_signature_with_stage"] = rho(th["gcs_thinned"], o[keep])
    t["cytotrace_conditional_skill_on_thinned_gene_count"] = conditional_skill(th["cytotrace_thinned"], [th["genes_1000"]], o[keep])
    depth["kept_cells_raw_depth"] = {"cells": int(len(keep)), "spearman_detected_genes_with_stage": rho(gc[keep], o[keep]),
                                     "spearman_gene_count_signature_with_stage": rho(th["gcs_kept_raw"], o[keep]),
                                     "spearman_cytotrace_with_stage": rho(th["cytotrace_kept_raw"], o[keep])}
    # the same four correlations over the replicated unit: per-sample medians against the sample's stage
    ids, inv = np.unique(sample, return_inverse=True)
    so = np.array([o[inv == j][0] for j in range(len(ids))])
    med = lambda v, cells=None: np.array([np.median(v[inv == j]) for j in range(len(ids))])  # noqa: E731
    depth["over_samples"] = {"samples": int(len(ids)), "spearman_median_library_size_with_stage": rho(med(lib), so),
                             "spearman_median_detected_genes_with_stage": rho(med(gc), so),
                             "spearman_median_cytotrace_with_stage": rho(med(sc["cytotrace_v1"]), so)}
    kid, kinv = np.unique(sample[keep], return_inverse=True)
    ko = np.array([o[keep][kinv == j][0] for j in range(len(kid))])
    kmed = lambda v: np.array([np.median(v[kinv == j]) for j in range(len(kid))])  # noqa: E731
    depth["over_samples"]["thinned_1000"] = {"samples": int(len(kid)),
                                             "spearman_median_detected_genes_with_stage": rho(kmed(th["genes_1000"]), ko),
                                             "spearman_median_cytotrace_with_stage": rho(kmed(th["cytotrace_thinned"]), ko)}
    out["depth"] = depth
    print(geo, "depth", json.dumps({k: v for k, v in depth.items() if k != "per_stage"}), flush=True)
    out["seconds"] = round(time.time() - t0, 1)
    return out, raw


def supported(row, key="sample_test"):
    t = row[key]
    return bool(t["p_two_sided"] < 0.05 and np.sign(t["S"]) == np.sign(row["conditional_skill"]))


def predictions(d):
    """The thirteen registered rows of PREREG_breadth.md for one series: (id, statement, outcome, observed)."""
    tag = {"GSE169210": "M", "GSE223922": "Z"}[d["series"]]
    rows, dep = d["rows"], d["depth"]
    pf = d["controls_permutation_floor"]["by_primitive"]
    shares = [pf[q]["fraction_above"] for q in JOINT]
    out = [(f"{tag}1", "more than half of the first 50 random gene sets are above the permutation floor on each primitive",
            all(x > 0.5 for x in shares), ", ".join(f"{PRIMITIVE[q]} {x:.0%}" for q, x in zip(JOINT, shares)))]
    for i, name in enumerate(("CytoTRACE", "SR", "CCAT", "MCE"), 2):
        want = EXPECT_JOINT[name]
        text = f"{name}: joint increment {'above' if want else 'not above'} the 97.5th percentile of the random gene sets"
        if name not in rows:
            out.append((f"{tag}{i}", text, None, "not scored"))
            continue
        v = rows[name]["vs_controls"]["a1_g200"]
        out.append((f"{tag}{i}", text, v["joint_exceeds_p975"] == want,
                    f"joint CS {rows[name]['conditional_skill_joint']:+.3f}, controls' 97.5th {v['joint_control_p975']:+.3f}"))
    for i, name in enumerate(("CytoTRACE", "SR", "CCAT", "MCE"), 6):
        text = f"{name}: residual trend supported over the replicated unit"
        if name not in rows:
            out.append((f"{tag}{i}", text, None, "not scored"))
            continue
        t = rows[name]["sample_test"]
        out.append((f"{tag}{i}", text, supported(rows[name]) == EXPECT_UNIT[name],
                    f"tau {t['tau']:+.3f}, p {t['p_two_sided']:.4f}, {t['samples_in_tested_batches']} units in {t['batches']} strata, "
                    f"CS {rows[name]['conditional_skill']:+.3f}"))
    t = dep["thinned"]["1000"] if "1000" in dep["thinned"] else dep["thinned"][1000]
    for i, (text, v) in enumerate((("gene count falls with stage at raw depth", dep["spearman_detected_genes_with_stage"]),
                                   ("gene count falls with stage at 1,000 counts", t["spearman_detected_genes_with_stage"]),
                                   ("CytoTRACE falls with stage at raw depth", dep["spearman_cytotrace_with_stage"]),
                                   ("CytoTRACE falls with stage at 1,000 counts", t["spearman_cytotrace_with_stage"])), 10):
        out.append((f"{tag}{i}", f"{text}, Spearman -0.1 or below", bool(v <= -0.1), f"Spearman {v:+.3f}"))
    return out


def report():
    """REPORT_breadth.md and predictions_breadth.json from every breadth_<series>.json in results/."""
    res = HERE / "results"
    L = ["# Breadth: two further staged series, results", "",
         "Generated by `breadth_analyse.py` from `breadth_<series>.json`. Plan: `PREREG_breadth.md`.", ""]
    allp = {}
    for geo in sorted(SID):
        f = res / f"breadth_{geo}.json"
        if not f.is_file():
            continue
        d = json.loads(f.read_text())
        rows, dep, ctl = d["rows"], d["depth"], d["controls"]["a1_g200"]
        L += [f"## {geo} ({NAMES[geo]})", "",
              f"{d['n']:,} cells, {d['samples']} replicated units, {d['batches']} strata, {d['levels']} distinct stage values. "
              f"Locked at commit `{d['locked_at_commit'][:7]}`.", "",
              "### A. How often an arbitrary gene set clears the floor", "",
              "Permutation floor: the first 50 of the 1,000 random 200-gene sets, 200 draws each; this is the reading the plan registers. "
              "Grid: all 1,000 sets against the audit's simulated floor, bracketed reading, which was built for 12 ordinal levels and is an extrapolation here.", "",
              "| covariates removed | above permutation floor | above grid floor (extrapolated) | median CS | 2.5th | 97.5th | median abs. Spearman with covariates |", "|---|---|---|---|---|---|---|"]
        for q in (*JOINT, "joint"):
            c = ctl[q]
            pfv = d["controls_permutation_floor"]["by_primitive"].get(q, {}).get("fraction_above")
            g = c.get("fraction_above_registered_floor_bracketed")
            qs = c["conditional_skill"]["quantiles"]
            L.append(f"| {PRIMITIVE[q]} | {'n/a' if pfv is None else f'{pfv:.1%}'} | {'n/a' if g is None else f'{g:.1%}'} | {qs['0.5']:+.3f} | {qs['0.025']:+.3f} | "
                     f"{qs['0.975']:+.3f} | {c['max_abs_spearman_with_covariates']['quantiles']['0.5']:.2f} |")
        L += ["", "### A. Each audited row against arbitrary gene sets", "",
              "| row | primitive | rho, row / controls | CS | permutation floor | clears | A1 97.5th | exceeds | joint CS | joint A1 97.5th | exceeds | controls with larger marginal tau | exceeds A2, primitive / joint |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        yn = lambda b: "yes" if b else "no"  # noqa: E731
        for name, r in rows.items():
            v, a2 = r["vs_controls"]["a1_g200"], r["vs_controls"]["a2"]
            L.append(f"| {name} | {PRIMITIVE[r['primitive']]} | {abs(r['spearman_with_primitive']):.2f} / {v['control_median_abs_spearman_with_primitive']:.2f} | "
                     f"{r['conditional_skill']:+.3f} | {r['permutation_floor']:+.3f} | {yn(r['clears_permutation_floor'])} | {v['control_p975']:+.3f} | {yn(v['exceeds_p975'])} | "
                     f"{r['conditional_skill_joint']:+.3f} | {v['joint_control_p975']:+.3f} | {yn(v['joint_exceeds_p975'])} | "
                     f"{v['fraction_of_controls_with_larger_abs_marginal_tau']:.1%} | {yn(a2['exceeds_p975'])} / {yn(a2['joint_exceeds_p975'])} |")
        L += ["", "### B. The trend over the replicated unit", "",
              "Kendall's tau between the per-unit mean of the stratum-mean residual and the unit's stage, stage labels permuted between the units of a stratum. "
              "Least squares: the residual of Eq. (2), anti-conservative in the calibration of the first plan.", "",
              "| row | units tested | strata | tau | p | sign agrees with CS | supported | no strata p | least-squares p | second stratification p |", "|---|---|---|---|---|---|---|---|---|---|"]
        for name, r in rows.items():
            t = r["sample_test"]
            g = lambda k: f"{r[k]['p_two_sided']:.4f}" if k in r else "n/a"  # noqa: E731
            L.append(f"| {name} | {t['samples_in_tested_batches']} of {t['samples']} | {t['batches']} | {t['tau']:+.3f} | {t['p_two_sided']:.4f} | "
                     f"{yn(np.sign(t['S']) == np.sign(r['conditional_skill']))} | {yn(supported(r))} | {g('sample_test_no_batch_strata')} | {g('sample_test_linear')} | {g('sample_test_batch2')} |")
        th, kr, ov = dep["thinned"], dep["kept_cells_raw_depth"], dep["over_samples"]
        L += ["", "### C. Depth", "", "Spearman correlation with stage, early to late.", "",
              f"Library size {dep['spearman_library_size_with_stage']:+.3f}; gene count {dep['spearman_detected_genes_with_stage']:+.3f}; "
              f"gene count given library size {dep['spearman_detected_genes_with_stage_given_library_size']:+.3f}.", "",
              "| depth | cells kept | units kept | gene count | gene-count signature | CytoTRACE |", "|---|---|---|---|---|---|",
              f"| raw | {d['n']:,} | {d['samples']} | {dep['spearman_detected_genes_with_stage']:+.3f} | {dep['spearman_gene_count_signature_with_stage']:+.3f} | {dep['spearman_cytotrace_with_stage']:+.3f} |",
              f"| 1,000 counts | {th['1000']['cells_kept']:,} | {th['1000']['samples_kept']} | {th['1000']['spearman_detected_genes_with_stage']:+.3f} | "
              f"{th['1000']['spearman_gene_count_signature_with_stage']:+.3f} | {th['1000']['spearman_cytotrace_with_stage']:+.3f} |",
              f"| raw, cells with 1,000 counts | {kr['cells']:,} | {th['1000']['samples_kept']} | {kr['spearman_detected_genes_with_stage']:+.3f} | "
              f"{kr['spearman_gene_count_signature_with_stage']:+.3f} | {kr['spearman_cytotrace_with_stage']:+.3f} |"]
        for k in ("500", "2000"):
            L.append(f"| {int(k):,} counts | {th[k]['cells_kept']:,} | {th[k]['samples_kept']} | {th[k]['spearman_detected_genes_with_stage']:+.3f} | n/a | n/a |")
        L += ["", f"Over the {ov['samples']} units, per-unit medians against the unit's stage: library size {ov['spearman_median_library_size_with_stage']:+.3f}, "
              f"gene count {ov['spearman_median_detected_genes_with_stage']:+.3f}, CytoTRACE {ov['spearman_median_cytotrace_with_stage']:+.3f}; "
              f"at 1,000 counts ({ov['thinned_1000']['samples']} units) gene count {ov['thinned_1000']['spearman_median_detected_genes_with_stage']:+.3f}, "
              f"CytoTRACE {ov['thinned_1000']['spearman_median_cytotrace_with_stage']:+.3f}.", "",
              "### The registered predictions", "", "| prediction | registered | outcome | observed |", "|---|---|---|---|"]
        allp[geo] = []
        for pid, text, held, obs in predictions(d):
            outcome = "void" if held is None else ("held" if held else "failed")
            L.append(f"| {pid} | {text} | {outcome} | {obs} |")
            allp[geo].append({"id": pid, "prediction": text, "outcome": outcome, "observed": obs})
        L.append("")
    (res / "REPORT_breadth.md").write_text("\n".join(L))
    (res / "predictions_breadth.json").write_text(json.dumps(allp, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("series", choices=[*sorted(SID), "report"])
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--n-draws", type=int, default=N_DRAWS)
    ap.add_argument("--n-perm", type=int, default=N_PERM)
    ap.add_argument("--max-controls", type=int, default=None, help="dry run only: truncate every control family")
    ap.add_argument("--dry-run", action="store_true", help="shuffle the stage labels between cells; writes to results/dryrun")
    a = ap.parse_args()
    if a.series == "report":
        return report()
    if a.max_controls is not None and not a.dry_run:
        raise SystemExit("--max-controls is for --dry-run only")
    if not a.dry_run:
        _LOCK["head"], _LOCK["manifest"] = check_lock()
    out_dir = HERE / "results" / "dryrun" if a.dry_run else HERE / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    with Pool(a.procs) as pool:
        res, raw = series(pool, a.series, a.n_draws, a.n_perm, a.dry_run, a.max_controls)
    (out_dir / f"breadth_{a.series}.json").write_text(json.dumps(res, indent=1))
    np.savez(out_dir / f"breadth_{a.series}_control_values.npz", **raw)
    if not a.dry_run:
        report()


if __name__ == "__main__":
    main()
