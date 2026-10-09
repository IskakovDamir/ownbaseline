#!/usr/bin/env python3
"""
run_rows.py — the w5 new rows, exactly as experiments/w5_new_rows/PREREG.md
registers them.

    python3 experiments/w5_new_rows/run_rows.py plan      # no label is compared
    python3 experiments/w5_new_rows/run_rows.py nulls     # missing design points
    python3 experiments/w5_new_rows/run_rows.py values    # the association step

plan    For every value the PREREG names: the cells used (ranked, every input
        finite, the registered cell set), n, the score-primitive Spearman rho,
        the SHA256 of every input file, and the floor lookup
        `python3 -m own_baseline.cli floors --n N --rho R --covariates K
        --levels L --json` for tau_b, plus own_baseline.floors.floor for the
        weighted kernels. A design point is MISSING when the lookup gives no
        floor, or marks the grid point as distant, or as permissive in n. For an
        ordinal with L != 12 levels the 12-level floor is looked up too, as a
        reported sensitivity. The ordinal is used only to know which cells are
        ranked and how many levels it has; no score is compared with it.
nulls   Each missing design point not already complete in
        $OWNBASELINE_DATA_ROOT/w5_new_rows/nulls/design_points.csv is run with
        experiments/null_calibration/run_missing_cells.py --design (200 seeds)
        and merged in.
values  Runs only if PREREG.md and this file are tracked and identical to
        origin/main, the plan's input hashes, n and rho still hold, and every
        missing design point is complete. Then conditional skill with the
        decider's _unit (rank residual; Kendall tau_b 1,000 and weightedtau 300
        bootstrap resamples; seed 42), the joint four-primitive residual, the
        floors, the direction check. Per-row JSON to
        $OWNBASELINE_DATA_ROOT/w5_new_rows/results/, a summary to
        experiments/w5_new_rows/results/summary.json.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(REPO))
from own_baseline.paths import data_root  # noqa: E402
from own_baseline.conditional_skill import (  # noqa: E402
    _unit, align, kang_taub, scipy_wtau, SEED, N_BOOT_WT, N_BOOT_KT)
from own_baseline import floors as F  # noqa: E402
from own_baseline.cli import residual_scale, DEBRIS_RATIO  # noqa: E402

W5 = data_root() / "w5_new_rows"
INPUTS = W5 / "inputs"
SCORES = W5 / "scores"
RESULTS = W5 / "results"
NULLS = W5 / "nulls"
DESIGN_CSV = NULLS / "design_points.csv"
PLAN = W5 / "plan.json"
SUMMARY = HERE / "results" / "summary.json"
PREREG_REL = "experiments/w5_new_rows/PREREG.md"
SELF_REL = "experiments/w5_new_rows/run_rows.py"
FOUR = ["gene_count", "pcc_string_v12", "shannon_H", "log10_lib"]
FOUR_LABEL = {"gene_count": "gene count", "pcc_string_v12": "PCC(x, degree)",
              "shannon_H": "Shannon entropy", "log10_lib": "log10 library size"}
N_SEEDS = 200
STRENGTHS = {1: (0.45, 0.55, 0.63), 4: (0.55,)}
KERNELS = ("kendalltau", "weightedtau_rankTrue", "weightedtau_rankFalse")

# MCE's cell set is fixed by the amendment PREREG.md requires before its row
# runs: "full" (all 39,505) or "subsample" (make_subsample.py, seed 42).
# Amendment 1 (2026-10-09): the port under MCE.m's stopping rule projects to
# 0.58 h on one core for 39,505 cells, under 8 hours, so "full".
MCE_CELLS = "full"

SPLIT = ("GSE117498 through the stemsc code path: the four broad-gate files name "
         "genes in R make.names form (HLA.A) and the seven sorted files in HGNC form "
         "(HLA-A); load_c1 unions by exact name, so in broad-gate cells (tier 4 among "
         "them) those genes read zero. The GSE117498h line is the registered "
         "sensitivity with the names merged.")
ROWS = [
    {"row": "MCE", "dataset": "GSE106474", "class": "unsupervised",
     "file": "mce.csv", "col": "mce", "primitive": "pcc_string_v12", "polarity": 1,
     "cells": MCE_CELLS, "caveats": []},
    {"row": "StemFinder", "dataset": "GSE106474", "class": "unsupervised",
     "file": "stemfinder.csv", "col": "stemfinder",
     "primitive": ("stemfinder.csv", "cc_gene_set_score"), "polarity": -1, "cells": "full",
     "caveats": ["nothing is fitted, but the authors chose among method variants (Gini, "
                 "SD, variance; the k range) on a benchmark that includes the Farrell "
                 "zebrafish set GSE106587, the SuperSeries of GSE106474"]},
    {"row": "mRNAsi", "dataset": "GSE117498", "class": "trained",
     "file": "mrnasi.csv", "col": "mrnasi", "primitive": None, "polarity": 1,
     "cells": "full", "caveats": [SPLIT]},
    {"row": "mRNAsi", "dataset": "GSE117498h", "class": "trained",
     "file": "mrnasi.csv", "col": "mrnasi", "primitive": None, "polarity": 1,
     "cells": "full", "sensitivity_of": "mRNAsi_GSE117498",
     "caveats": ["registered sensitivity: gene names merged (prepare_inputs.ds_gse117498h)"]},
    {"row": "FitDevo", "dataset": "GSE106474", "class": "trained",
     "file": "fitdevo.csv", "col": "fitdevo_dp", "primitive": None, "polarity": 1,
     "cells": "full",
     "caveats": ["not in the training set, but the authors evaluated their weights on "
                 "this study's stage labels once (Spearman 0.228, GSE106587)",
                 "zebrafish symbols matched to human by upper case; the authors left "
                 "zebrafish out of training for too few homologous genes"]},
    {"row": "FitDevo", "dataset": "GSE117498", "class": "trained",
     "file": "fitdevo.csv", "col": "fitdevo_dp", "primitive": None, "polarity": 1,
     "cells": "full", "caveats": [SPLIT]},
    {"row": "FitDevo", "dataset": "GSE117498h", "class": "trained",
     "file": "fitdevo.csv", "col": "fitdevo_dp", "primitive": None, "polarity": 1,
     "cells": "full", "sensitivity_of": "FitDevo_GSE117498",
     "caveats": ["registered sensitivity: gene names merged (prepare_inputs.ds_gse117498h)"]},
    {"row": "FitDevo", "dataset": "GSE125970", "class": "trained",
     "file": "fitdevo.csv", "col": "fitdevo_dp", "primitive": None, "polarity": 1,
     "cells": "full", "caveats": []},
    {"row": "FitDevo", "dataset": "GSE113074", "class": "trained",
     "file": "fitdevo.csv", "col": "fitdevo_dp", "primitive": None, "polarity": 1,
     "cells": "full",
     "caveats": ["Xenopus symbols matched to human by upper case; the homology concern "
                 "the authors raised for zebrafish applies here too"]},
]


def key(r):
    return f"{r['row']}_{r['dataset']}"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_col(path, col, cells):
    with open(path) as fh:
        rows = list(csv.DictReader(fh))
    if [r["cell"] for r in rows] != cells:
        raise SystemExit(f"{path}: cell order differs from the prepared input")
    return np.array([float(r[col]) if r[col] not in ("", "NA", "NaN") else np.nan
                     for r in rows])


def subsample_index(n_cells):
    """experiments/track4/code/make_subsample.py's 3,000 cells, same recipe."""
    rng = np.random.default_rng(42)
    return np.sort(rng.choice(n_cells, size=3000, replace=False))


def load(r):
    """Score, primitives and ordinal of one row, or None with the reason."""
    d = INPUTS / r["dataset"]
    cells = d.joinpath("cells.txt").read_text().split("\n")[:-1]
    sf = SCORES / r["dataset"] / r["file"]
    if not sf.is_file():
        why = (f"waiting for the author: MCE.m is not in reference/, so no validated "
               f"port has written {sf.relative_to(data_root())}" if r["row"] == "MCE"
               else f"no score file {sf.relative_to(data_root())}")
        return None, why
    files = {str(sf.relative_to(data_root())): sha256(sf),
             str((d / "primitives.npz").relative_to(data_root())): sha256(d / "primitives.npz")}
    score = read_col(sf, r["col"], cells)
    p = np.load(d / "primitives.npz")
    prims = {k: p[k].astype(float) for k in FOUR}
    declared = None
    if isinstance(r["primitive"], tuple):
        pf = SCORES / r["dataset"] / r["primitive"][0]
        files[str(pf.relative_to(data_root()))] = sha256(pf)
        declared = ("cell-cycle gene-set score", read_col(pf, r["primitive"][1], cells))
    elif r["primitive"]:
        declared = (FOUR_LABEL[r["primitive"]], prims[r["primitive"]])
    o = p["ordinal"].astype(float)
    return {"score": score, "prims": prims, "declared": declared, "ordinal": o,
            "potency_sign": float(p["potency_sign"]), "files": files}, None


def cell_mask(r, x):
    """The registered cell set; refuses any other pattern of finite scores."""
    ranked = np.isfinite(x["ordinal"])
    inputs_ok = ranked.copy()
    for v in x["prims"].values():
        inputs_ok &= np.isfinite(v)
    if x["declared"] is not None:
        inputs_ok &= np.isfinite(x["declared"][1])
    finite = np.isfinite(x["score"])
    if r["cells"] == "full":
        missing = int((inputs_ok & ~finite).sum())
        if missing:
            raise SystemExit(f"{key(r)}: {missing} ranked cells have no finite score; "
                             f"the registered cell set is all ranked cells")
        return inputs_ok
    if r["cells"] == "subsample":
        want = np.zeros(len(finite), dtype=bool)
        want[subsample_index(len(finite))] = True
        if not np.array_equal(finite, want):
            raise SystemExit(f"{key(r)}: finite scores are not exactly the 3,000-cell "
                             f"make_subsample.py index")
        return inputs_ok & want
    raise SystemExit(f"{key(r)}: no cell set registered (PREREG.md Amendments)")


def cli_floor(n, rho, k, levels):
    cmd = [sys.executable, "-m", "own_baseline.cli", "floors", "--n", str(n),
           "--rho", f"{abs(rho):.4f}", "--kernel", "taub", "--covariates", str(k),
           "--levels", str(levels), "--json"]
    out = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True)
    try:
        res = json.loads(out.stdout)
    except json.JSONDecodeError:
        res = {"floor": None, "reason": out.stdout.strip() or out.stderr.strip()}
    cmd_txt = ("python3 -m own_baseline.cli floors --n {} --rho {:.4f} --kernel taub "
               "--covariates {} --levels {} --json").format(n, abs(rho), k, levels)
    return res, cmd_txt


def lookup(n, rho, k, levels):
    """Shipped-grid floors for the three kernels and whether the point is missing."""
    taub, cmd = cli_floor(n, rho, k, levels)
    wt = F.floor(n, rho, "weightedtau_rankTrue", k, levels)
    wf = F.floor(n, rho, "weightedtau_rankFalse", k, levels)
    missing = (taub.get("floor") is None or bool(taub.get("distant"))
               or taub.get("n_side") == "permissive")
    return {"n": n, "rho": float(rho), "k": k, "levels": levels, "cli": cmd,
            "taub": taub, "weighted_rankTrue": wt, "weighted_rankFalse_cli": wf,
            "missing": bool(missing),
            "design": f"n={n},rho={round(abs(rho), 3)},levels={levels},k={k}"}


def measure(r, x):
    m = cell_mask(r, x)
    s = x["score"][m]
    n = int(m.sum())
    levels = int(len(np.unique(x["ordinal"][m])))
    rho_four = {k: float(spearmanr(v[m], s).statistic) for k, v in x["prims"].items()}
    rho_declared = (float(spearmanr(x["declared"][1][m], s).statistic)
                    if x["declared"] is not None else None)
    return m, n, levels, rho_four, rho_declared


def phase_plan():
    plan = {"written": time.strftime("%Y-%m-%d %H:%M:%S %z"), "rows": {}}
    for r in ROWS:
        x, why = load(r)
        if x is None:
            plan["rows"][key(r)] = {"status": "NOT RUN", "reason": why}
            print(f"{key(r):22s} NOT RUN: {why}")
            continue
        m, n, levels, rho_four, rho_decl = measure(r, x)
        entry = {"n": n, "levels": levels, "cells": r["cells"], "files": x["files"],
                 "rho_four": rho_four, "rho_declared": rho_decl, "values": {}}
        if x["declared"] is not None:
            entry["values"]["declared"] = {"primitive": x["declared"][0],
                                           **lookup(n, rho_decl, 1, levels)}
        # the paper placed its joint rows at the declared primitive's rho; a row
        # with no declared primitive takes the largest single |rho| (cmd_check)
        rho_j = abs(rho_decl) if rho_decl is not None else max(abs(v) for v in rho_four.values())
        entry["values"]["joint4"] = {
            "primitive": "joint: " + ", ".join(FOUR_LABEL[k] for k in FOUR),
            "rho_rule": ("declared primitive's |rho| (the paper's joint rows)"
                         if rho_decl is not None else
                         "largest single-primitive |rho| (ownbaseline check)"),
            **lookup(n, rho_j, 4, levels)}
        for v in entry["values"].values():
            v["L12"] = lookup(n, v["rho"], v["k"], 12) if levels != 12 else None
        plan["rows"][key(r)] = entry
        for name, v in entry["values"].items():
            t = v["taub"]
            extra = (f" | L=12 sensitivity -> {'MISSING ' + v['L12']['design'] if v['L12']['missing'] else 'grid'}"
                     if v["L12"] else "")
            print(f"{key(r):22s} {name:8s} n={n:,} L={levels} k={v['k']} rho={v['rho']:+.4f} "
                  f"floor={t.get('floor')} -> "
                  f"{'MISSING ' + v['design'] if v['missing'] else 'grid'}{extra}")
    PLAN.parent.mkdir(parents=True, exist_ok=True)
    PLAN.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"wrote {PLAN} (sha256 {sha256(PLAN)})")


def needed_designs(plan):
    out = set()
    for e in plan["rows"].values():
        for v in e.get("values", {}).values():
            if v["missing"]:
                out.add(v["design"])
            if v.get("L12") and v["L12"]["missing"]:
                out.add(v["L12"]["design"])
    return sorted(out)


def design_rows():
    return list(csv.DictReader(open(DESIGN_CSV))) if DESIGN_CSV.is_file() else []


def complete(design, rows):
    k = int(dict(kv.split("=") for kv in design.split(","))["k"])
    have = {(round(float(r["nullB_strength"]), 2), r["kernel"]) for r in rows
            if r["design"] == design and int(r["n_seeds"]) == N_SEEDS}
    return all((s, kern) in have for s in STRENGTHS[1 if k == 1 else 4] for kern in KERNELS)


def phase_nulls():
    plan = json.loads(PLAN.read_text())
    rows = design_rows()
    todo = [d for d in needed_designs(plan) if not complete(d, rows)]
    if not todo:
        print("every missing design point is already complete")
        return
    NULLS.mkdir(parents=True, exist_ok=True)
    new = NULLS / "design_points_new.csv"
    cmd = [sys.executable, str(REPO / "experiments/null_calibration/run_missing_cells.py"),
           "--seeds", str(N_SEEDS), "--out", str(new)]
    for d in todo:
        cmd += ["--design", d]
    print("$ " + " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=REPO).returncode:
        raise SystemExit("run_missing_cells.py failed")
    merged = [r for r in rows if r["design"] not in todo] + list(csv.DictReader(open(new)))
    tmp = DESIGN_CSV.with_name(DESIGN_CSV.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(merged[0].keys()))
        w.writeheader()
        w.writerows(merged)
    os.replace(tmp, DESIGN_CSV)
    new.unlink()
    print(f"merged {len(todo)} design points into {DESIGN_CSV}")


def design_floor(design, kernel, rows):
    """Max over coupling strengths of the 97.5th percentile at a design point."""
    sel = [r for r in rows if r["design"] == design and r["kernel"] == kernel]
    return {"floor": max(float(r["p97.5"]) for r in sel),
            "null_mean": float(np.mean([float(r["mean"]) for r in sel])),
            "strengths": sorted({float(r["nullB_strength"]) for r in sel}),
            "n_seeds": sum(int(r["n_seeds"]) for r in sel),
            "source": f"run_missing_cells.py --design {design}"}


def git(*args, timeout=120):
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True,
                          timeout=timeout)


def require_prereg_pushed():
    f = git("fetch", "--quiet", "origin")
    fetched = f.returncode == 0
    for rel in (PREREG_REL, SELF_REL):
        if git("ls-files", "--error-unmatch", rel).returncode:
            raise SystemExit(f"{rel} is not tracked; the values phase does not run")
        if git("cat-file", "-e", f"origin/main:{rel}").returncode:
            raise SystemExit(f"{rel} is not on origin/main; the values phase does not run")
        if git("diff", "--quiet", "origin/main", "--", rel).returncode:
            raise SystemExit(f"{rel} differs from origin/main; push it first")
    text = (REPO / PREREG_REL).read_text()
    if "PLACEHOLDER" in text or "STEMFINDER_RUNTIME" in text:
        raise SystemExit("PREREG.md still carries a placeholder")
    log = git("log", "--diff-filter=A", "--format=%h %ci", "origin/main", "--",
              PREREG_REL).stdout.strip().splitlines()
    later = git("log", "--format=%h %ci %s", "origin/main", "--",
                PREREG_REL).stdout.strip().splitlines()
    first = log[-1] if log else None
    return {"prereg_commit": first, "prereg_commits_newest_first": later,
            "origin_fetched_now": fetched}


def verify_plan(plan):
    """The plan the PREREG table was made from must still describe the data."""
    for r in ROWS:
        pe = plan["rows"].get(key(r))
        if pe is None:
            raise SystemExit(f"{key(r)} is not in plan.json; rerun plan before pushing")
        x, why = load(r)
        if x is None or "values" not in pe:
            if (x is None) != ("values" not in pe):
                raise SystemExit(f"{key(r)}: score availability changed since the plan")
            continue
        if x["files"] != pe["files"]:
            raise SystemExit(f"{key(r)}: an input file changed since the plan: "
                             f"{sorted(set(x['files'].items()) ^ set(pe['files'].items()))}")
        m, n, levels, rho_four, rho_decl = measure(r, x)
        if (n, levels) != (pe["n"], pe["levels"]):
            raise SystemExit(f"{key(r)}: n or levels changed since the plan")
        for k, v in rho_four.items():
            if abs(v - pe["rho_four"][k]) > 1e-12:
                raise SystemExit(f"{key(r)}: rho with {k} changed since the plan")
        if rho_decl is not None and abs(rho_decl - pe["rho_declared"]) > 1e-12:
            raise SystemExit(f"{key(r)}: declared rho changed since the plan")


def direction(x, m, r):
    """Primitive and score directions against the potency-oriented ordinal."""
    pot = x["potency_sign"] * x["ordinal"][m]
    out = {"ordinal_orientation": "higher = more potent" if x["potency_sign"] > 0
           else "higher = later stage, so potency = -ordinal"}
    prims = {FOUR_LABEL[k]: v for k, v in x["prims"].items()}
    if x["declared"] is not None and x["declared"][0] not in prims:
        prims[x["declared"][0]] = x["declared"][1]
    out["primitives"] = {}
    for name, v in prims.items():
        rho = float(spearmanr(v[m], pot).statistic)
        out["primitives"][name] = {"spearman_vs_potency": rho,
                                   "orders_below_chance": bool(rho < 0)}
    s_rho = float(spearmanr(x["score"][m], pot).statistic)
    out["score"] = {"spearman_vs_potency": s_rho,
                    "authors_polarity": "higher = more potent" if r["polarity"] > 0
                    else "higher = more differentiated",
                    "runs_as_authors_state": bool(np.sign(s_rho) == r["polarity"])}
    return out


def floors_for(v, rows):
    if v["missing"]:
        return {"taub": design_floor(v["design"], "kendalltau", rows),
                "weighted": design_floor(v["design"], "weightedtau_rankTrue", rows),
                "weighted_rankFalse_cli": design_floor(v["design"], "weightedtau_rankFalse", rows)}
    return {"taub": {"floor": v["taub"]["floor"], "source": v["cli"], "cell": v["taub"]["cell"]},
            "weighted": {"floor": v["weighted_rankTrue"]["floor"],
                         "source": "own_baseline.floors.floor(kernel='weightedtau_rankTrue')",
                         "cell": v["weighted_rankTrue"]["cell"]},
            "weighted_rankFalse_cli": {"floor": v["weighted_rankFalse_cli"]["floor"],
                                       "cell": v["weighted_rankFalse_cli"]["cell"]}}


def phase_values():
    gate = require_prereg_pushed()
    plan = json.loads(PLAN.read_text())
    verify_plan(plan)
    rows_csv = design_rows()
    incomplete = [d for d in needed_designs(plan) if not complete(d, rows_csv)]
    if incomplete:
        raise SystemExit(f"design points not computed yet: {incomplete}; run nulls first")
    RESULTS.mkdir(parents=True, exist_ok=True)
    summary = {**gate, "plan_sha256": sha256(PLAN), "seed": SEED,
               "n_boot": {"kang_wdm_taub": N_BOOT_KT, "scipy_weightedtau": N_BOOT_WT},
               "rows": {}}
    jobs, ctx = [], {}
    for r in ROWS:
        k = key(r)
        if "values" not in plan["rows"][k]:
            summary["rows"][k] = {**{f: r[f] for f in ("row", "dataset", "class")},
                                  **plan["rows"][k]}
            continue
        x, _ = load(r)
        m = cell_mask(r, x)
        o = x["ordinal"][m]
        s = align(x["score"][m], o)
        ctx[k] = (r, x, m, o, s)
        if x["declared"] is not None:
            jobs.append((f"{k}|declared", s, [x["declared"][1][m]], o))
        jobs.append((f"{k}|joint4", s, [x["prims"][p][m] for p in FOUR], o))
    print(f"[values] {len(jobs)} bootstrap units", flush=True)
    t0 = time.time()
    with Pool(min(8, len(jobs))) as pool:
        res = dict(pool.map(_unit, jobs))
    print(f"[values] bootstrap done in {time.time()-t0:.0f}s", flush=True)

    for k, (r, x, m, o, s) in ctx.items():
        pe = plan["rows"][k]
        row = {"row": r["row"], "dataset": r["dataset"], "class": r["class"],
               "sensitivity_of": r.get("sensitivity_of"), "caveats": r["caveats"],
               "cells": r["cells"], "n": int(m.sum()), "levels": pe["levels"],
               "input_sha256": pe["files"], "prereg_commit": gate["prereg_commit"],
               "score_sign_flipped_to_align": bool(np.any(s != x["score"][m])),
               "marginal": {"kang_wdm_taub": kang_taub(x["score"][m], o),
                            "scipy_weightedtau": scipy_wtau(x["score"][m], o),
                            "spearman_vs_ordinal": float(spearmanr(x["score"][m], o).statistic)},
               "direction_check": direction(x, m, r), "values": {}}
        for name, v in pe["values"].items():
            cs = res[f"{k}|{name}"]
            covs = ([x["declared"][1][m]] if name == "declared"
                    else [x["prims"][p][m] for p in FOUR])
            ratio, _ = residual_scale(s, covs)
            readable = ratio >= DEBRIS_RATIO
            fl = floors_for(v, rows_csv)
            fl12 = floors_for(v["L12"], rows_csv) if v.get("L12") else None
            if not readable:   # the CLI's refusal: no value is read under either kernel
                cs = {kern: {"tau": None, "CI95": [None, None], "n_boot": 0}
                      for kern in ("scipy_weightedtau", "kang_wdm_taub")}
            tb, fb = cs["kang_wdm_taub"]["tau"], fl["taub"]["floor"]
            wt, fw = cs["scipy_weightedtau"]["tau"], fl["weighted"]["floor"]
            row["values"][name] = {
                "primitive": v["primitive"], "k_covariates": v["k"],
                "rho_score_primitive": v["rho"], "rho_rule": v.get("rho_rule"),
                "residual_scale": ratio, "conditional_skill": cs,
                "floors": fl, "floors_L12_sensitivity": fl12,
                "verdict_taub": ("NOT READ (residual is rounding debris)" if not readable
                                 else "ABOVE FLOOR" if tb > fb else "NOT ABOVE FLOOR"),
                "weighted_vs_floor (reported, not a verdict)": (
                    "NOT READ" if not readable
                    else "above" if wt > fw else "not above"),
                "taub_vs_L12_floor (reported, not a verdict)": (
                    None if fl12 is None or not readable
                    else "above" if tb > fl12["taub"]["floor"] else "not above"),
            }
        decisive = "declared" if "declared" in row["values"] else "joint4"
        row["decisive_value"] = decisive
        row["verdict_taub"] = row["values"][decisive]["verdict_taub"]
        (RESULTS / f"{k}.json").write_text(json.dumps(row, indent=2) + "\n")
        d, j = row["values"][decisive], row["values"]["joint4"]
        summary["rows"][k] = {
            "row": r["row"], "dataset": r["dataset"], "class": r["class"],
            "sensitivity_of": r.get("sensitivity_of"), "caveats": r["caveats"],
            "n": row["n"], "levels": row["levels"], "decisive_value": decisive,
            "primitive": d["primitive"], "rho": d["rho_score_primitive"],
            "marginal_taub": row["marginal"]["kang_wdm_taub"],
            "conditional_taub": d["conditional_skill"]["kang_wdm_taub"]["tau"],
            "conditional_taub_CI95": d["conditional_skill"]["kang_wdm_taub"]["CI95"],
            "floor_taub": d["floors"]["taub"]["floor"],
            "floor_taub_source": d["floors"]["taub"]["source"],
            "floor_taub_L12": (d["floors_L12_sensitivity"] or {}).get("taub", {}).get("floor"),
            "verdict_taub": row["verdict_taub"],
            "weighted": d["conditional_skill"]["scipy_weightedtau"]["tau"],
            "weighted_CI95": d["conditional_skill"]["scipy_weightedtau"]["CI95"],
            "floor_weighted": d["floors"]["weighted"]["floor"],
            "weighted_vs_floor": d["weighted_vs_floor (reported, not a verdict)"],
            "joint4_rho": j["rho_score_primitive"],
            "joint4_taub": j["conditional_skill"]["kang_wdm_taub"]["tau"],
            "joint4_taub_CI95": j["conditional_skill"]["kang_wdm_taub"]["CI95"],
            "joint4_floor_taub": j["floors"]["taub"]["floor"],
            "joint4_verdict_taub": j["verdict_taub"],
            "direction_check": row["direction_check"],
        }
        print(f"{k:22s} n={row['n']:,} {decisive}: tau_b "
              f"{summary['rows'][k]['conditional_taub']} floor "
              f"{summary['rows'][k]['floor_taub']} -> {row['verdict_taub']}", flush=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"wrote {SUMMARY}")


if __name__ == "__main__":
    phase = sys.argv[1] if len(sys.argv) > 1 else ""
    {"plan": phase_plan, "nulls": phase_nulls, "values": phase_values}.get(
        phase, lambda: sys.exit(__doc__))()
