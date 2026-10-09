#!/usr/bin/env python3
"""
make_report.py — write experiments/w5_new_rows/REPORT.md from the run outputs,
so no number in it is typed by hand.

    python3 experiments/w5_new_rows/make_report.py

Reads experiments/w5_new_rows/results/summary.json,
discovery/{mce_validation,discovery_facts}.json,
$OWNBASELINE_DATA_ROOT/w5_new_rows/{plan,sources}.json, the per-row JSON, the
design-point CSV and the score wrappers' info files. The P1 sentence and the
blocked-row text follow the MCE row's state in the summary.
"""
from __future__ import annotations

import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(REPO))
from own_baseline.paths import data_root  # noqa: E402

W5 = data_root() / "w5_new_rows"
RES = HERE / "results"
OUT = HERE / "REPORT.md"
FOUR_LABEL = {"gene_count": "gene count", "pcc_string_v12": "PCC(x, degree)",
              "shannon_H": "Shannon entropy", "log10_lib": "log10 library size"}
ORDER = ["MCE_GSE106474", "StemFinder_GSE106474", "mRNAsi_GSE117498",
         "mRNAsi_GSE117498h", "FitDevo_GSE106474", "FitDevo_GSE117498",
         "FitDevo_GSE117498h", "FitDevo_GSE125970", "FitDevo_GSE113074"]
POTENCY_SIGN = {"GSE106474": -1, "GSE113074": -1, "GSE117498": 1, "GSE117498h": 1,
                "GSE125970": 1}


def j(p):
    return json.loads(Path(p).read_text())


def info(dataset, row):
    f = W5 / "scores" / dataset / f"{row}_info.txt"
    if not f.is_file():
        return {}
    return dict(line.split("\t", 1) for line in f.read_text().splitlines() if "\t" in line)


def f4(x):
    return "-" if x is None else f"{x:+.4f}"


def ci(c):
    return "-" if not c or c[0] is None else f"[{c[0]:+.4f}, {c[1]:+.4f}]"


def direction_text(dc):
    below = [k for k, v in dc["primitives"].items() if v["orders_below_chance"]]
    s = dc["score"]
    txt = ("score runs as its authors state" if s["runs_as_authors_state"]
           else "score runs AGAINST its authors' stated direction")
    txt += f" (Spearman vs potency {s['spearman_vs_potency']:+.3f})"
    txt += "; primitives below chance: " + (", ".join(below) if below else "none")
    return txt


def residual_direction(k, r, row_json):
    """
    Direction of the decisive residual relative to the authors' polarity, from
    registered numbers only. rank_resid_multi with an intercept and Kendall tau_b
    are both odd in the score's sign, so the raw score's residual tau_b is the
    aligned value, negated when align() negated the score.
    """
    cs = row_json["values"][row_json["decisive_value"]]["conditional_skill"]["kang_wdm_taub"]
    if cs["tau"] is None:
        return None
    flip = -1.0 if row_json["score_sign_flipped_to_align"] else 1.0
    raw = flip * cs["tau"]
    lo, hi = sorted((flip * cs["CI95"][0], flip * cs["CI95"][1]))
    pol = 1 if row_json["direction_check"]["score"]["authors_polarity"].startswith("higher = more potent") else -1
    rel = pol * POTENCY_SIGN[r["dataset"]]          # raw residual vs ordinal -> vs authors' claim
    return {"raw": raw, "raw_ci": (lo, hi), "rel": rel * raw, "rel_ci": sorted((rel * lo, rel * hi)),
            "flipped": bool(row_json["score_sign_flipped_to_align"]),
            "marginal_spearman": row_json["marginal"]["spearman_vs_ordinal"]}


def main():
    summ = j(RES / "summary.json")
    val = j(HERE / "discovery" / "mce_validation.json")
    facts = j(HERE / "discovery" / "discovery_facts.json")
    src = j(W5 / "sources.json")
    plan = j(W5 / "plan.json")
    rows = summ["rows"]
    rowj = {k: j(W5 / "results" / f"{k}.json") for k in ORDER
            if (W5 / "results" / f"{k}.json").is_file()}
    mce = rows.get("MCE_GSE106474", {})
    mce_ran = "n" in mce

    L = ["# W5 new rows: report\n"]
    if mce_ran:
        held = mce["verdict_taub"] == "ABOVE FLOOR"
        L.append(f"**P1 is {'confirmed' if held else 'refuted'}**: MCE's tau_b conditional "
                 f"skill on PCC(x, degree) is {f4(mce['conditional_taub'])} against a floor of "
                 f"{f4(mce['floor_taub'])} at n = {mce['n']:,}, so it is "
                 f"{mce['verdict_taub']}, as the registered entropy-rate branch "
                 f"{'predicted' if held else 'did not predict'}.\n")
    else:
        L.append("**P1 is not testable in this session**: the MCE row needs MCE.m, which "
                 "Qingyang Wang reports is in the supplement of Shi et al. 2020 (not "
                 "verified here), and OUP serves that supplement only through a signed "
                 "link behind a human-verification page that was not attempted; the "
                 "dataset, primitive and prediction are registered (entropy-rate branch: "
                 "MCE keeps tau_b conditional skill above its floor on PCC(x, degree), "
                 "switching to the static-entropy branch if MCE.m computes only H(pi)), "
                 "and MCE.m's SHA256, the port's settings and the cell set are to be fixed "
                 "by dated amendment before the row runs.\n")
    amend = summ.get("prereg_commits_newest_first", [])[:-1]
    L.append(f"Pre-registration: `PREREG.md`, first added in commit {summ['prereg_commit']} "
             "and pushed before any new score met a label"
             + (f"; later commits to it: {'; '.join(amend)}" if amend else
                "; no amendment since") + f". Seed {summ['seed']}; bootstrap "
             f"{summ['n_boot']['kang_wdm_taub']} resamples for tau_b and "
             f"{summ['n_boot']['scipy_weightedtau']} for weighted tau; plan.json SHA256 "
             f"{summ['plan_sha256']}.\n")

    L.append("## Results\n")
    L.append("Conditional skill is Kendall tau_b of the aligned score's rank residual on "
             "the declared primitive (the joint four-primitive residual for rows that "
             "declare none), read against the 97.5th percentile of the estimator's measured "
             "null at the value's n, rho, number of covariates and number of ordinal levels. "
             "Grid floors come from the nearest grid cell (the Floors table names it); "
             "design floors were computed at the value's own rho. For a joint value the "
             "rho is the declared primitive's |rho| when the row has one and otherwise the "
             "largest single-primitive |rho|, named in the column. Brackets after a floor "
             "give the 12-level floor where the ordinal has another level count. Marginal "
             "tau_b is the raw score against the raw ordinal (GSE106474 and GSE113074 run "
             "early to late stage, GSE117498 and GSE125970 low to high potency), so its "
             "sign mixes the two orientations; the direction check is potency-oriented. "
             "Weighted tau (rank=True) with its 95% interval is compared with the same "
             "cell's rank=True column and is reported, not a verdict. Numbered notes are "
             "caveats, under the table.\n")
    hdr = ("| score | class | dataset | n | rho with primitive | marginal tau_b | "
           "conditional tau_b [95% CI] | floor | verdict under tau_b | "
           "weighted tau [95% CI] / floor | joint four-primitive residual tau_b [95% CI] / floor | "
           "direction check |")
    L.append(hdr)
    L.append("|" + "---|" * 12)
    notes = []
    for k in ORDER:
        r = rows[k]
        if "n" not in r:
            L.append(f"| {r['row']} | {r['class']} (in tally) | {r['dataset']} | - | - | - | - | "
                     f"- | not run: {r.get('reason', '')} | - | - | - |")
            continue
        refs = []
        for c in r.get("caveats", []):
            if c not in notes:
                notes.append(c)
            refs.append(str(notes.index(c) + 1))
        tag = f" [{', '.join(refs)}]" if refs else ""
        clss = r["class"] + (" (in tally)" if r["class"] == "unsupervised"
                             else " (boundary, outside tally)")
        if r.get("sensitivity_of"):
            clss += ", sensitivity line"
        if r["decisive_value"] == "declared":
            rho_txt = f"{r['rho']:+.3f} with {r['primitive']}"
        else:
            rf = plan["rows"][k]["rho_four"]
            top = max(rf, key=lambda q: abs(rf[q]))
            rho_txt = f"{r['rho']:+.3f} (largest single: {FOUR_LABEL[top]})"
        fl12 = f" (12 levels: {f4(r['floor_taub_L12'])})" if r.get("floor_taub_L12") is not None else ""
        L.append(
            f"| {r['row']}{tag} | {clss} | {r['dataset']} ({r['levels']} levels) | {r['n']:,} | "
            f"{rho_txt} | {f4(r['marginal_taub'])} | "
            f"{f4(r['conditional_taub'])} {ci(r['conditional_taub_CI95'])} | "
            f"{f4(r['floor_taub'])}{fl12} | {r['verdict_taub']} | "
            f"{f4(r['weighted'])} {ci(r['weighted_CI95'])} / {f4(r['floor_weighted'])} "
            f"({r['weighted_vs_floor']}) | "
            f"{f4(r['joint4_taub'])} {ci(r['joint4_taub_CI95'])} / "
            f"{f4(r['joint4_floor_taub'])} ({r['joint4_verdict_taub']}) | "
            f"{direction_text(r['direction_check'])} |")
    L.append("")
    for i, c in enumerate(notes, 1):
        L.append(f"{i}. {c}")
    L.append("")
    L.append("StemFinder's joint value sits at its declared primitive's rho; its weighted "
             "joint value and every other number not in the table are in the per-row JSON "
             "(`$OWNBASELINE_DATA_ROOT/w5_new_rows/results/<row>_<dataset>.json`).\n")

    L.append("### Reading the table\n")
    L.append("The verdicts are the registered rule's. The notes are derived from the same "
             "registered numbers and change none of them. The estimator is sign-agnostic: it "
             "points each score the way its marginal Spearman correlation with the ordinal "
             "runs, then residualizes. Because the rank residual and tau_b are both odd in "
             "the score's sign, the raw score's residual is the aligned value, negated when "
             "the score was negated. Read against the authors' stated polarity:\n")
    dirs = {}
    for k in ORDER:
        r = rows[k]
        if "n" not in r or k not in rowj:
            continue
        d = residual_direction(k, r, rowj[k])
        dirs[k] = d
        lo, hi = d["rel_ci"]
        if lo > 0:
            way = "in the direction its authors state"
        elif hi < 0:
            way = "against the direction its authors state"
        else:
            way = "in no direction its interval can separate from zero"
        L.append(f"- {r['row']} on {r['dataset']}: the raw score's residual beyond "
                 f"{'its declared primitive' if r['decisive_value'] == 'declared' else 'the four primitives'} "
                 f"orders cells {way} (tau_b {d['rel']:+.4f} [{lo:+.4f}, {hi:+.4f}] with "
                 f"the authors' polarity counted positive); the marginal Spearman with the "
                 f"ordinal is {d['marginal_spearman']:+.3f}, so the score was "
                 f"{'negated' if d['flipped'] else 'kept as it is'} before residualizing, "
                 f"and the registered verdict is {r['verdict_taub']}.")
    L.append("")
    for k in ORDER:
        r = rows[k]
        b = rows.get(r.get("sensitivity_of") or "", {})
        if r.get("sensitivity_of") and "n" in r and "n" in b:
            db, dr = dirs[r["sensitivity_of"]], dirs[k]
            if b["verdict_taub"] == r["verdict_taub"]:
                L.append(f"- {r['row']}: the GSE117498 line and its name-harmonized "
                         f"sensitivity GSE117498h agree ({b['verdict_taub']}, conditional "
                         f"tau_b {f4(b['conditional_taub'])} and {f4(r['conditional_taub'])}); "
                         "the gene-name split does not move the verdict.")
            else:
                L.append(f"- {r['row']}: the verdict differs between GSE117498 "
                         f"({b['verdict_taub']}) and the name-harmonized GSE117498h "
                         f"({r['verdict_taub']}), but what the score carries beyond the "
                         f"primitives does not change direction: relative to the authors' "
                         f"polarity the residual is {db['rel']:+.4f} and {dr['rel']:+.4f}. "
                         f"The split moves the marginal Spearman from "
                         f"{db['marginal_spearman']:+.3f} to {dr['marginal_spearman']:+.3f}, "
                         "which flips the alignment, and the registered rule reads only an "
                         "aligned residual above its floor. The decisive verdict therefore "
                         "rests on the sign of a marginal correlation close to zero.")
    L.append("")
    tally = [k for k in ORDER if rows[k].get("class") == "unsupervised"]
    L.append("Audit tally, unsupervised rows only: " + "; ".join(
        f"{rows[k]['row']} {rows[k].get('verdict_taub', 'not run')}" for k in tally)
        + ". mRNAsi and FitDevo are trained and stay outside it.\n")

    L.append("### Floors\n")
    L.append("| value | n | k | levels | rho | floor source |")
    L.append("|---|---|---|---|---|---|")
    for k in ORDER:
        e = plan["rows"].get(k, {})
        for name, v in e.get("values", {}).items():
            for vv, lab in ((v, ""), (v.get("L12"), " (12-level sensitivity)")):
                if not vv:
                    continue
                srcf = (f"design point `{vv['design']}` (run_missing_cells.py, 200 seeds)"
                        if vv["missing"] else f"shipped grid: {vv['taub'].get('cell')}")
                L.append(f"| {k} {name}{lab} | {vv['n']:,} | {vv['k']} | {vv['levels']} | "
                         f"{vv['rho']:+.4f} | {srcf} |")
    L.append("")
    dp = list(csv.DictReader(open(W5 / "nulls" / "design_points.csv")))
    n_designs = len({r["design"] for r in dp})
    L.append(f"`design_points.csv` holds {n_designs} design points ({len(dp)} rows), all "
             "computed before the PREREG commit in two passes of `run_rows.py nulls`: the "
             "first plan's points, then the points the reviewed plan added (the GSE117498h "
             "lines and the 12-level sensitivities). Weighted-tau floors as `ownbaseline "
             "check --kernel weighted` would read them (the rank=False column) are in each "
             "per-row JSON under `floors.weighted_rankFalse_cli`.\n")

    L.append("## Blocked rows\n")
    if not mce_ran:
        L.append("- **MCE**: not run. Qingyang Wang reports the authors' MATLAB file in "
                 "`bby093_supp.zip`, the supplement on the article page "
                 "https://academic.oup.com/bib/article/21/1/248/5115275#supplementary-data "
                 "(that the zip holds MCE.m is not verified). The CDN link is signed per page "
                 "view and the page sits behind an interactive Cloudflare check, which was not "
                 "attempted, so neither a script nor the built-in browser could fetch it. What "
                 "the author has to provide: the zip saved into `reference/`, unzipped, and "
                 "MCE.m copied to `reference/MCE.m` (commands in DISCOVERY.md 1.2). Then the "
                 "steps under MCE in PREREG.md run in order.\n")
    else:
        L.append("- none\n")

    L.append("## MCE port validation\n")
    L.append("`scores/mce.py` implements the published equations (Eqs 3 to 8). Against "
             "analytic answers (`discovery/mce_validation.json`):\n")
    L.append("| case | expected | got | abs. deviation |")
    L.append("|---|---|---|---|")
    for t in val["toy"]:
        L.append(f"| {t['case']} | {t['expected']:.12g} | {t['got']:.12g} | {t['abs_dev']:.1e} |")
    t = val["timing"]
    L.append("")
    L.append(f"Against MCE.m in Octave: **{val['reference']['status']}**. "
             f"{val['reference']['reason']} Spearman and maximum absolute deviation will "
             "be reported here when it runs.\n")
    L.append(f"Timing: {t['seconds_per_iteration_per_cell']:.3e} s per iteration per cell "
             f"on {t['n_cells_timed']} cells over {t['n_network_genes']:,} network genes "
             "(one core). Iterations to each stopping tolerance come from a convergence "
             f"trace of {t['trace_cells']} cells, which can underestimate the slowest cell "
             "of a 500-cell chunk. Projection to 39,505 cells: " + "; ".join(
                 f"tolerance {k}: {v['iterations']:,} iterations, {v['projected_hours_39505']:.2f} h"
                 for k, v in t["projection_by_tol"].items())
             + f". Tighter tolerances were not reached in {t['trace_cap']:,} iterations. "
             "Which applies is MCE.m's stopping rule.\n")

    L.append("## Commands\n")
    L.append("From the repository root, in this order, with `$OWNBASELINE_DATA_ROOT` and "
             "`$OWNBASELINE_SCRATCH` unset or at their defaults (`data/runs`, `data/scratch`): "
             "the R library path below is written for the default scratch root.\n")
    L.append("Inputs read from the data root and not made here: "
             "`track2/data/prepared/` (GSE106474 after the repository's QC, from "
             "`python3 reproduce.py --stages fetch-string scaffold fetch-zebrafish prep`), "
             "`w4/scaffolds/human_string_v12_thr700_lcc.npz` (stage `scaffold`), "
             "`<scratch>/scent_io/adjMC.npz` (`experiments/track2/code/prep_scent_inputs.py`, "
             "SR's network, used by the MCE timing), `w4/data/c1_gse117498/` and "
             "`w4/data/c2_gse125970/` (GEO GSE117498 and GSE125970), and "
             "`fix3/data/prepared/` (`experiments/fix3/code/prep_briggs.py` on GSE113074). "
             "`data/README.md` has the accessions.\n")
    L.append("```")
    L += [
        "python3 tests/run_tests.py",
        "python3 tests/test_mce.py",
        "python3 experiments/w5_new_rows/fetch_sources.py",
        "python3 experiments/w5_new_rows/discovery_facts.py",
        "python3 experiments/w5_new_rows/validate_mce.py",
        "python3 experiments/w5_new_rows/prepare_inputs.py",
        "printf 'CC=clang -std=gnu17\\nCC17=clang -std=gnu17\\nCC23=clang -std=gnu17\\n' > data/scratch/Makevars.w5",
        "R_MAKEVARS_USER=$PWD/data/scratch/Makevars.w5 Rscript --vanilla -e 'lib <- \"data/scratch/Rlib\"; "
        ".libPaths(c(lib, .libPaths())); install.packages(c(\"Seurat\", \"qlcMatrix\"), lib = lib, "
        "repos = \"https://cloud.r-project.org\", Ncpus = 8)'",
        "Rscript --vanilla experiments/w5_new_rows/score_stemfinder.R GSE106474",
        "Rscript --vanilla experiments/w5_new_rows/score_mrnasi.R GSE117498",
        "Rscript --vanilla experiments/w5_new_rows/score_mrnasi.R GSE117498h",
        "Rscript --vanilla experiments/w5_new_rows/score_fitdevo.R GSE106474",
        "Rscript --vanilla experiments/w5_new_rows/score_fitdevo.R GSE117498",
        "Rscript --vanilla experiments/w5_new_rows/score_fitdevo.R GSE117498h",
        "Rscript --vanilla experiments/w5_new_rows/score_fitdevo.R GSE125970",
        "Rscript --vanilla experiments/w5_new_rows/score_fitdevo.R GSE113074",
        "python3 experiments/w5_new_rows/run_rows.py plan",
        "python3 experiments/w5_new_rows/run_rows.py nulls",
        "python3 experiments/w5_new_rows/check_design_extension.py",
        "python3 experiments/w5_new_rows/run_rows.py values",
        "python3 experiments/w5_new_rows/make_report.py",
    ]
    L.append("```\n")
    L.append("Per row: StemFinder needs `prepare_inputs.py`, `score_stemfinder.R GSE106474`, "
             "then `run_rows.py plan`, `nulls`, `values`; mRNAsi needs `score_mrnasi.R` on "
             "GSE117498 and GSE117498h; FitDevo needs `score_fitdevo.R` on the five inputs; "
             "every row is read by the same `run_rows.py values`. Homebrew's R 4.6.1 asks "
             "for `-std=gnu23`, which the Command Line Tools clang 15 rejects; the Makevars "
             "line works around it for this install only.\n")

    L.append("## Software and provenance\n")
    sf = info("GSE106474", "stemfinder")
    fd = info("GSE117498", "fitdevo") or info("GSE106474", "fitdevo")
    import numpy, scipy, sklearn  # noqa: E401
    octv = subprocess.run(["octave-cli", "--version"], capture_output=True,
                          text=True).stdout.splitlines()[0]
    L.append("| item | value |")
    L.append("|---|---|")
    L.append(f"| Python, numpy, scipy, scikit-learn | {platform.python_version()}, "
             f"{numpy.__version__}, {scipy.__version__}, {sklearn.__version__} |")
    L.append(f"| R | {sf.get('r', fd.get('r', '-'))} |")
    L.append(f"| Seurat, SeuratObject, qlcMatrix | {sf.get('seurat', fd.get('seurat', '-'))}, "
             f"{sf.get('seuratobject', fd.get('seuratobject', '-'))}, {fd.get('qlcmatrix', '-')} |")
    L.append(f"| GNU Octave | {octv} |")
    m = src["MCE.m"]
    L.append("| SHA256 of MCE.m | " + (m["sha256"] if isinstance(m, dict) else
                                       "not available: MCE.m not obtained (waiting for the author)")
             + " |")
    for name, label in [("FitDevo", "FitDevo commit"), ("stemfinder", "stemFinder commit"),
                        ("TCGAbiolinks", "TCGAbiolinks")]:
        e = src["sources"][name]
        ver = f" (version {e['package_version']})" if e.get("package_version") else ""
        L.append(f"| {label} | {e['commit']}{ver}, {e['commit_date']} |")
    tb = src["sources"]["TCGAbiolinks"]["files"]["data/SC_PCBC_stemSig.rda"]
    L.append(f"| SC_PCBC_stemSig.rda SHA256 | {tb['sha256']} ({facts['w_n']:,} weights) |")
    bg = src["sources"]["FitDevo"]["files"]["BGW.rds"]
    L.append(f"| FitDevo BGW.rds SHA256 | {bg['sha256']} ({facts['bgw_n']:,} genes) |")
    L.append("")
    L.append("Per-score run records (runtime, genes matched, parameters chosen by rule):\n")
    for ds, row in [("GSE106474", "stemfinder"), ("GSE117498", "mrnasi"),
                    ("GSE117498h", "mrnasi"), ("GSE106474", "fitdevo"),
                    ("GSE117498", "fitdevo"), ("GSE117498h", "fitdevo"),
                    ("GSE125970", "fitdevo"), ("GSE113074", "fitdevo")]:
        d = info(ds, row)
        if d:
            keep = {k: v for k, v in d.items() if k not in ("r", "seurat", "seuratobject", "qlcmatrix")}
            L.append(f"- {row} on {ds}: " + ", ".join(f"{k} {v}" for k, v in keep.items()))
    L.append("")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
