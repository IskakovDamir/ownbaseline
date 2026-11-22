#!/usr/bin/env python3
"""
make_tex.py: the scorecard, the LaTeX tables and the number macros of the revised paper.

    python3 experiments/specificity/make_tex.py --out <manuscript folder>

Reads results/specificity_<series>.json (this plan), experiments/conditional_null/results/
conditional_null.json (the permutation floors, the authors' reading and the within-batch rows
of the earlier plan), results/calibration_*.json and the shipped floor table. Writes

    <out>/generated_tables.tex     the tables of the new appendix and the main-text scorecard
    <out>/generated_numbers.tex    one macro per number quoted in the text
    results/SCORECARD.md           the scorecard as markdown
    results/numbers.json           the same numbers, keyed as the macros are

No number is typed in: each one is read from a results file and formatted here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from own_baseline.floors import floor as grid_floor  # noqa: E402

SERIES = ("GSE106474", "GSE112294", "GSE113074")
ORDER = ("CytoTRACE", "SR", "ORIGINS", "NCG", "dpath", "StemFinder", "SLICE", "CCAT", "MCE")
PRIM = {"detected_genes": "gene count", "pcc_x_degree": "PCC", "shannon_entropy": "Shannon", "stemfinder_cell_cycle_score": "cell cycle"}
LETTERS = {"GSE106474": "F", "GSE112294": "W", "GSE113074": "X"}


def sgn(x, d=3):
    return f"$-${abs(x):.{d}f}" if x < 0 else f"+{x:.{d}f}"


def txt(x, d=3):
    """A value for running text: a minus sign when negative, no plus sign."""
    return f"$-${abs(x):.{d}f}" if x < 0 else f"{x:.{d}f}"


def pc(x, d=1):
    return "--" if x is None else f"{100 * x:.{d}f}"


def yes(b):
    return "yes" if b else "no"


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        adj[i] = running
    return adj


def pval(p):
    return "$<$0.0001" if p < 0.0001 else f"{p:.4f}"


def load():
    d = {g: json.loads((HERE / "results" / f"specificity_{g}.json").read_text()) for g in SERIES}
    for g in SERIES:
        if d[g].get("dry_run"):
            raise SystemExit(f"specificity_{g}.json is a dry run")
    old = json.loads((REPO / "experiments" / "conditional_null" / "results" / "conditional_null.json").read_text())
    perm, batch = {}, {}
    for f in old["floors"]:
        parts = f["row"].split()
        if len(parts) == 2 and f["m"] in (100, 25):
            perm[(parts[0], parts[1])] = f
        elif len(parts) == 3:
            batch.setdefault((parts[0], parts[2]), []).append(f["registered"]["linear"]["clears"])
    cal = {}
    for name in ("cluster", "concordance"):
        path = HERE / "results" / f"calibration_{name}.json"
        cal[name] = json.loads(path.read_text())["rows"] if path.is_file() else []
    path = HERE / "results" / "depth_2.json"
    cal["depth_2"] = json.loads(path.read_text()) if path.is_file() else None
    path = HERE / "results" / "depth_3.json"
    cal["depth_3"] = json.loads(path.read_text()) if path.is_file() else None
    return d, perm, batch, cal


def supported(r):
    t = r["sample_test"]
    return t["p_two_sided"] < 0.05 and t["S"] != 0 and (t["S"] > 0) == (r["conditional_skill"] > 0)


def scorecard(d, perm, batch):
    z = d["GSE106474"]
    adj = dict(zip(ORDER, holm([z["rows"][n]["sample_test"]["p_two_sided"] for n in ORDER])))
    rows = []
    for n in ORDER:
        r = z["rows"][n]
        fl = grid_floor(r["n"], abs(r["spearman_with_primitive"]), kernel="taub", k_covariates=1, levels=12)["floor"]
        pf = perm[("GSE106474", n)]
        t = r["sample_test"]
        a = r["vs_controls"]["a1_g200"]
        rows.append({
            "score": n, "primitive": PRIM[r["primitive"]], "n": r["n"], "rho": abs(r["spearman_with_primitive"]),
            "cs": r["conditional_skill"], "floor": fl, "clears_registered": r["conditional_skill"] > fl,
            "clears_authors": pf["authors"]["linear"]["clears"],
            "clears_permutation": pf["registered"]["linear"]["clears"],
            "batches": sum(batch[("GSE106474", n)]), "batches_of": len(batch[("GSE106474", n)]),
            "sample_tau": t["tau"], "sample_p": t["p_two_sided"], "sample_holm": adj[n], "sample_supported": supported(r),
            "control_percentile": a["percentile"], "control_p975": a["control_p975"], "exceeds": a["exceeds_p975"],
            "control_rho": a["control_median_abs_spearman_with_primitive"],
            "joint": r["conditional_skill_joint"], "joint_p975": a["joint_control_p975"], "joint_exceeds": a["joint_exceeds_p975"],
            "joint_percentile": a["joint_percentile"], "marginal_larger": a["fraction_of_controls_with_larger_abs_marginal_tau"],
            "exceeds_a2": r["vs_controls"]["a2"]["exceeds_p975"],
            "concordance": r["concordance"]["value"], "concordance_clears": r["concordance"]["clears"]})
    return rows


def write_scorecard_md(rows):
    L = ["# Scorecard, GSE106474", "",
         "CS is the registered conditional skill on the declared primitive. Percentiles are among 1,000 random 200-gene sets "
         "residualised on the same covariates on the same cells.", "",
         "| score | primitive | CS | registered floor | clears | authors' direction | permutation floor | batches | "
         "sample tau | sample p | supported | controls with larger marginal tau | increment percentile | exceeds | "
         "joint CS | joint percentile | joint exceeds |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['score']} | {r['primitive']} | {r['cs']:+.3f} | {r['floor']:.4f} | {yes(r['clears_registered'])} | "
                 f"{yes(r['clears_authors'])} | {yes(r['clears_permutation'])} | {r['batches']}/{r['batches_of']} | "
                 f"{r['sample_tau']:+.3f} | {r['sample_p']:.4f} | {yes(r['sample_supported'])} | {pc(r['marginal_larger'])}% | "
                 f"{pc(r['control_percentile'])}% | {yes(r['exceeds'])} | {r['joint']:+.3f} | {pc(r['joint_percentile'])}% | {yes(r['joint_exceeds'])} |")
    (HERE / "results" / "SCORECARD.md").write_text("\n".join(L) + "\n")


def tables(d, rows, cal):
    T = []
    # ---- main-text scorecard
    T += [r"\newcommand{\scorecardtable}{%", r"\begin{tabular}{llrrccccrrcrc}", r"\toprule",
          r"score & primitive & CS & floor & \shortstack{clears\\floor} & \shortstack{authors'\\direction} & "
          r"\shortstack{every\\batch} & \shortstack{sample\\level} & \shortstack{sample\\$p$} & "
          r"\shortstack{increment,\\percentile} & \shortstack{above\\controls} & \shortstack{joint,\\percentile} & \shortstack{above\\controls} \\", r"\midrule"]
    for r in rows:
        T.append(f"{r['score']} & {r['primitive']} & {sgn(r['cs'])} & {r['floor']:.4f} & {yes(r['clears_registered'])} & "
                 f"{yes(r['clears_authors'])} & {r['batches']}/{r['batches_of']} & {yes(r['sample_supported'])} & {pval(r['sample_p'])} & "
                 f"{pc(r['control_percentile'])} & {yes(r['exceeds'])} & {pc(r['joint_percentile'])} & {yes(r['joint_exceeds'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}}", ""]

    # ---- how often a control clears
    z = d["GSE106474"]
    T += [r"\newcommand{\controlfloortable}{%", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
          r"covariates removed & \shortstack{bracketed\\floor, \%} & \shortstack{nearest\\floor, \%} & \shortstack{nearest, not\\distant, \%} & "
          r"\shortstack{permutation\\floor, \%} & median & 2.5th & 97.5th & \shortstack{median\\$|\rho|$} \\", r"\midrule"]
    for c, label in (("detected_genes", "gene count"), ("pcc_x_degree", "PCC(x, degree)"), ("shannon_entropy", "Shannon entropy"),
                     ("log10_library_size", "log library size"), ("joint", "all four")):
        e = z["controls"]["a1_g200"][c]
        q = e["conditional_skill"]["quantiles"]
        pf = z["controls_permutation_floor"]["by_primitive"].get(c)
        T.append(f"{label} & {pc(e['fraction_above_registered_floor_bracketed'])} & {pc(e['fraction_above_registered_floor_nearest'])} & "
                 f"{pc(e['fraction_above_registered_floor_nearest_not_distant'])} & {pc(pf['fraction_above'], 0) if pf else '--'} & "
                 f"{sgn(q['0.5'])} & {sgn(q['0.025'])} & {sgn(q['0.975'])} & {e['max_abs_spearman_with_covariates']['quantiles']['0.5']:.2f} \\\\")
    T += [r"\bottomrule", r"\end{tabular}}", ""]

    # ---- every row against the control families, all series
    T += [r"\newcommand{\controlrowstable}{%", r"\begin{tabular}{llrrrrrcrrcccr}", r"\toprule",
          r"series & score & $n$ & \shortstack{$|\rho|$, score\\/ controls} & CS & \shortstack{A1\\97.5th} & \shortstack{per-\\centile} & "
          r"\shortstack{ex-\\ceeds} & \shortstack{joint\\CS} & \shortstack{joint A1\\97.5th} & \shortstack{ex-\\ceeds} & "
          r"\shortstack{$g{=}20$ /\\$2{,}000$} & A2 & \shortstack{larger marginal\\$\tau_b$, \%} \\", r"\midrule"]
    for g in SERIES:
        for n, r in d[g]["rows"].items():
            v = r["vs_controls"]
            a = v["a1_g200"]
            sens = " / ".join(yes(v[k]["exceeds_p975"]) for k in ("a1_g20", "a1_g2000") if k in v) or "--"
            T.append(f"{g} & {n} & {r['n']:,} & {abs(r['spearman_with_primitive']):.2f} / {a['control_median_abs_spearman_with_primitive']:.2f} & "
                     f"{sgn(r['conditional_skill'])} & {sgn(a['control_p975'])} & {pc(a['percentile'])} & "
                     f"{yes(a['exceeds_p975'])} & {sgn(r['conditional_skill_joint'])} & {sgn(a['joint_control_p975'])} & "
                     f"{yes(a['joint_exceeds_p975'])} & {sens} & {yes(v['a2']['exceeds_p975'])} & "
                     f"{pc(a['fraction_of_controls_with_larger_abs_marginal_tau'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}}", ""]

    # ---- CytoTRACE two by two
    T += [r"\newcommand{\cytotracetable}{%", r"\begin{tabular}{lllrrrrr}", r"\toprule",
          r"series & genes & smoothing & \shortstack{value or\\median} & 2.5th & 97.5th & \shortstack{joint value\\or median} & \shortstack{joint\\97.5th} \\", r"\midrule"]
    for g in ("GSE106474", "GSE112294"):
        ab, rows_g = d[g]["cytotrace_ablation"], d[g]["rows"]
        for key, row, sm in (("a3_unsmoothed", "GCS", "off"), ("a3", "CytoTRACE", "on")):
            q, qj = ab[key]["detected_genes"]["quantiles"], ab[key]["joint"]["quantiles"]
            T.append(f"{g} & by gene count & {sm} & {sgn(rows_g[row]['conditional_skill'])} & & & {sgn(rows_g[row]['conditional_skill_joint'])} & \\\\")
            T.append(f"{g} & random & {sm} & {sgn(q['0.5'])} & {sgn(q['0.025'])} & {sgn(q['0.975'])} & {sgn(qj['0.5'])} & {sgn(qj['0.975'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}}", ""]

    # ---- the trend over samples
    T += [r"\newcommand{\sampletable}{%", r"\begin{tabular}{llrrrrcrrr}", r"\toprule",
          r"series & score & samples & $\tau$ & $p$ & Holm $p$ & supported & \shortstack{no batch\\strata $p$} & \shortstack{least-squares\\residual $p$} & joint $p$ \\", r"\midrule"]
    for g in ("GSE106474", "GSE112294"):
        rr = {n: r for n, r in d[g]["rows"].items() if "sample_test" in r and n != "GCS"}
        adj = dict(zip(rr, holm([r["sample_test"]["p_two_sided"] for r in rr.values()])))
        for n, r in rr.items():
            t = r["sample_test"]
            un = r.get("sample_test_no_batch_strata", {}).get("p_two_sided")
            T.append(f"{g} & {n} & {t['samples']} & {sgn(t['tau'])} & {pval(t['p_two_sided'])} & {pval(adj[n])} & "
                     f"{yes(supported(r))} & {'--' if un is None else pval(un)} & {pval(r['sample_test_linear']['p_two_sided'])} & "
                     f"{pval(r['sample_test_joint_boosted']['p_two_sided'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}}", ""]

    # ---- depth and the window
    T += [r"\newcommand{\depthtable}{%", r"\begin{tabular}{lllrrrr}", r"\toprule",
          r"series & cells & depth & cells kept & gene count & \shortstack{gene-count\\signature} & CytoTRACE \\", r"\midrule"]
    for g in SERIES:
        dp, rows_g = d[g]["depth"], d[g]["rows"]
        T.append(f"{g} & all hours & raw & {d[g]['n']:,} & {sgn(dp['spearman_detected_genes_with_stage'])} & "
                 f"{sgn(rows_g['GCS']['spearman_with_stage'])} & {sgn(rows_g['CytoTRACE']['spearman_with_stage'])} \\\\")
        k = dp.get("kept_cells_raw_depth")
        if k:
            T.append(f"{g} & cells with 1,000 counts & raw & {k['cells']:,} & {sgn(k['spearman_detected_genes_with_stage'])} & "
                     f"{sgn(k['spearman_gene_count_signature_with_stage'])} & {sgn(k['spearman_cytotrace_with_stage'])} \\\\")
        for depth in ("1000", "500", "2000"):
            t = dp["thinned"].get(depth)
            if t:
                gcs, ct = t.get("spearman_gene_count_signature_with_stage"), t.get("spearman_cytotrace_with_stage")
                T.append(f"{g} & all hours & {int(depth):,} & {t['cells_kept']:,} & {sgn(t['spearman_detected_genes_with_stage'])} & "
                         f"{'--' if gcs is None else sgn(gcs)} & {'--' if ct is None else sgn(ct)} \\\\")
        for tag, label in (("window", "raw"), ("window_thinned", "1,000")):
            w = dp.get("shared_window", {}).get(tag)
            if w:
                T.append(f"{g} & 4 to 12 h & {label} & {w['cells']:,} & {sgn(w['spearman_detected_genes_with_stage'])} & "
                         f"{sgn(w['spearman_gene_count_signature_with_stage'])} & {sgn(w['spearman_cytotrace_with_stage'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}}", ""]

    # ---- stratified concordance
    T += [r"\newcommand{\concordancetable}{%", r"\begin{tabular}{llrrrrc}", r"\toprule",
          r"series & score & value & null mean & null 97.5th & $z$ & above \\", r"\midrule"]
    for g in SERIES:
        for n, r in d[g]["rows"].items():
            c = r["concordance"]
            T.append(f"{g} & {n} & {sgn(c['value'])} & {sgn(c['null_mean'], 4)} & {sgn(c['null_p975'], 4)} & {c['z']:.1f} & {yes(c['clears'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}}", ""]

    # ---- calibration of the sample test
    T += [r"\newcommand{\calibrationtable}{%", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
          r"score given primitive & \shortstack{sample\\sd} & \shortstack{batch\\sd} & \shortstack{planted\\trend} & "
          r"\shortstack{registered\\floor, \%} & \shortstack{permutation\\floor, \%} & \shortstack{sample test,\\\%} & "
          r"\shortstack{no batch\\strata, \%} & \shortstack{least-squares\\residual, \%} \\", r"\midrule"]
    for r in cal["cluster"]:
        shape = f"line, $\\rho = {r['rho']}$" if r["shape"] == "linear" else r["shape"]
        T.append(f"{shape} & {r['sample_sd']:g} & {r['batch_sd']:g} & {r['planted_trend']:g} & {pc(r['clears_registered_floor'])} & "
                 f"{pc(r['clears_permutation_floor'])} & {pc(r['sample_test_stratum_mean'])} & "
                 f"{pc(r['sample_test_stratum_mean_no_batch_strata'])} & {pc(r['sample_test_least_squares'])} \\\\")
    T += [r"\bottomrule", r"\end{tabular}}", ""]
    d2 = cal["depth_2"]
    if d2:
        T += [r"\newcommand{\depthtwotable}{%", r"\begin{tabular}{llrrrrrr}", r"\toprule",
              r"atlas & statistic & \shortstack{AUROC,\\raw} & \shortstack{AUROC,\\kept cells} & \shortstack{AUROC,\\1,000} & "
              r"\shortstack{AUROC,\\500 / 2,000} & \shortstack{$\tau_b$,\\raw} & \shortstack{$\tau_b$,\\1,000} \\", r"\midrule"]
        for geo in ("GSE117498h", "GSE125970"):
            a = d2["atlases"][geo]
            th = a["thinned"]
            T.append(f"{geo} & detected genes & {a['raw']['detected_genes']['auroc_top_vs_bottom']:.3f} & "
                     f"{a['kept_cells_raw_depth']['detected_genes']['auroc_top_vs_bottom']:.3f} & {th['1000']['detected_genes']['auroc_top_vs_bottom']:.3f} & "
                     f"{th['500']['detected_genes']['auroc_top_vs_bottom']:.3f} / {th['2000']['detected_genes']['auroc_top_vs_bottom']:.3f} & "
                     f"{sgn(a['raw']['detected_genes']['tau_b_all_ranked'])} & {sgn(th['1000']['detected_genes']['tau_b_all_ranked'])} \\\\")
            T.append(f"{geo} & gene-count signature & {a['raw']['gene_count_signature']['auroc_top_vs_bottom']:.3f} & "
                     f"{a['kept_cells_raw_depth']['gene_count_signature']['auroc_top_vs_bottom']:.3f} & "
                     f"{th['1000']['gene_count_signature']['auroc_top_vs_bottom']:.3f} & -- & "
                     f"{sgn(a['raw']['gene_count_signature']['tau_b_all_ranked'])} & {sgn(th['1000']['gene_count_signature']['tau_b_all_ranked'])} \\\\")
        T += [r"\bottomrule", r"\end{tabular}}", ""]
        T += [r"\newcommand{\depthtwopoptable}{%", r"\begin{tabular}{llrrrrr}", r"\toprule",
              r"atlas & population & tier & cells & \shortstack{median\\library size} & \shortstack{median genes,\\raw} & \shortstack{median genes,\\1,000} \\", r"\midrule"]
        for geo in ("GSE117498h", "GSE125970"):
            for pop, v in sorted(d2["atlases"][geo]["by_population"].items(), key=lambda kv: (-kv[1]["ordinal"], kv[0])):
                at = "--" if v["median_detected_genes_at_1000"] is None else f"{v['median_detected_genes_at_1000']:,.0f}"
                T.append(f"{geo} & {pop.replace('_', ' ')} & {v['ordinal']:.0f} & {v['cells']:,} & {v['median_library_size']:,.0f} & "
                         f"{v['median_detected_genes']:,.0f} & {at} \\\\")
        T += [r"\bottomrule", r"\end{tabular}}", ""]
        T += [r"\newcommand{\depthtwopcctable}{%", r"\begin{tabular}{lrrrrr}", r"\toprule",
              r"series & cells & cells kept & \shortstack{PCC with stage,\\raw} & \shortstack{PCC with stage,\\1,000} & \shortstack{raw against\\1,000} \\", r"\midrule"]
        for geo in SERIES:
            v = d2["staged"][geo]
            T.append(f"{geo} & {v['cells']:,} & {v['cells_kept']:,} & {sgn(v['spearman_pcc_with_stage_raw'])} & "
                     f"{sgn(v['spearman_pcc_with_stage_at_1000'])} & {v['spearman_pcc_raw_with_pcc_at_1000']:.2f} \\\\")
        T += [r"\bottomrule", r"\end{tabular}}", ""]
    d3 = cal["depth_3"]
    if d3:
        T += [r"\newcommand{\depththreetable}{%", r"\begin{tabular}{llrrrrrrc}", r"\toprule",
              r"series & score & cells & \shortstack{with stage,\\raw} & \shortstack{with stage,\\1,000} & \shortstack{raw against\\1,000} & "
              r"\shortstack{CS on PCC,\\1,000} & \shortstack{permutation\\floor} & clears \\", r"\midrule"]
        for g in SERIES:
            for n, r in d3["series"][g].items():
                T.append(f"{g} & {n} & {r['cells']:,} & {sgn(r['spearman_with_stage_raw'])} & {sgn(r['spearman_with_stage_at_1000'])} & "
                         f"{r['spearman_raw_score_with_score_at_1000']:.2f} & {sgn(r['conditional_skill_at_1000'])} & "
                         f"{r['permutation_floor']:.4f} & {yes(r['clears'])} \\\\")
        T += [r"\bottomrule", r"\end{tabular}}", ""]
    return "\n".join(T) + "\n"


def numbers(d, rows, cal):
    """Macros for the numbers the text quotes. Names are letters only: series letter F, W or X, then a tag."""
    N = {}
    for g in SERIES:
        s = LETTERS[g]
        z = d[g]
        for c, tag in (("detected_genes", "Gc"), ("pcc_x_degree", "Pcc"), ("shannon_entropy", "Sh"), ("log10_library_size", "Lib"), ("joint", "Joint")):
            e = z["controls"]["a1_g200"][c]
            q = e["conditional_skill"]["quantiles"]
            N[f"ctl{s}{tag}Med"], N[f"ctl{s}{tag}Hi"], N[f"ctl{s}{tag}Lo"] = txt(q["0.5"]), txt(q["0.975"]), txt(q["0.025"])
            N[f"ctl{s}{tag}Rho"] = f"{e['max_abs_spearman_with_covariates']['quantiles']['0.5']:.2f}"
            if e.get("fraction_above_registered_floor_bracketed") is not None:
                N[f"ctl{s}{tag}AboveFloor"] = pc(e["fraction_above_registered_floor_bracketed"])
                N[f"ctl{s}{tag}AboveNearest"] = pc(e["fraction_above_registered_floor_nearest"])
                N[f"ctl{s}{tag}AboveNotDistant"] = pc(e["fraction_above_registered_floor_nearest_not_distant"])
            pf = z["controls_permutation_floor"]["by_primitive"].get(c)
            if pf:
                N[f"ctl{s}{tag}AbovePerm"] = pc(pf["fraction_above"], 0)
        N[f"ctl{s}PermN"] = str(z["controls_permutation_floor"]["controls"])
        qm = z["controls"]["a1_g200"]["marginal_abs_tau"]["quantiles"]
        N[f"ctl{s}MargMed"], N[f"ctl{s}MargHi"] = f"{qm['0.5']:.3f}", f"{qm['0.975']:.3f}"
        for n, r in z["rows"].items():
            a = r["vs_controls"]["a1_g200"]
            N[f"row{s}{n}Cs"] = txt(r["conditional_skill"])
            N[f"row{s}{n}Rho"] = f"{abs(r['spearman_with_primitive']):.2f}"
            N[f"row{s}{n}ConcPct"] = pc((1 + r["concordance"]["value"]) / 2, 0)
            N[f"row{s}{n}Marg"] = txt(r["marginal_tau"])
            N[f"row{s}{n}Pct"] = pc(a["percentile"])
            N[f"row{s}{n}NotBelow"] = str(round((1 - a["percentile"]) * 1000))
            N[f"row{s}{n}Hi"] = txt(a["control_p975"])
            N[f"row{s}{n}Joint"] = txt(r["conditional_skill_joint"])
            N[f"row{s}{n}JointHi"] = txt(a["joint_control_p975"])
            N[f"row{s}{n}JointPct"] = pc(a["joint_percentile"])
            N[f"row{s}{n}MargLarger"] = pc(a["fraction_of_controls_with_larger_abs_marginal_tau"])
            N[f"row{s}{n}Conc"] = txt(r["concordance"]["value"])
            if "sample_test" in r:
                N[f"row{s}{n}SampleTau"] = txt(r["sample_test"]["tau"])
                N[f"row{s}{n}SampleP"] = pval(r["sample_test"]["p_two_sided"]).replace("$<$", "below ")
                N[f"row{s}{n}SampleN"] = str(r["sample_test"]["samples"])
        if "cytotrace_ablation" in z:
            for key, tag in (("a3", "Smooth"), ("a3_unsmoothed", "Raw")):
                q = z["cytotrace_ablation"][key]["detected_genes"]["quantiles"]
                qj = z["cytotrace_ablation"][key]["joint"]["quantiles"]
                N[f"abl{s}{tag}Med"], N[f"abl{s}{tag}Hi"], N[f"abl{s}{tag}Lo"] = txt(q["0.5"]), txt(q["0.975"]), txt(q["0.025"])
                N[f"abl{s}{tag}JointMed"], N[f"abl{s}{tag}JointHi"] = txt(qj["0.5"]), txt(qj["0.975"])
            N[f"abl{s}CtPct"] = pc(z["rows"]["CytoTRACE"]["vs_controls"]["a3"]["percentile"], 0)
            N[f"abl{s}CtJointPct"] = pc(z["rows"]["CytoTRACE"]["vs_controls"]["a3"]["joint_percentile"], 0)
            N[f"abl{s}GcsPct"] = pc(z["rows"]["GCS"]["vs_controls"]["a3_unsmoothed"]["percentile"], 0)
        dp = z["depth"]
        N[f"dep{s}Lib"] = txt(dp["spearman_library_size_with_stage"])
        N[f"dep{s}Gc"] = txt(dp["spearman_detected_genes_with_stage"])
        N[f"dep{s}GcGivenLib"] = txt(dp["spearman_detected_genes_with_stage_given_library_size"])
        N[f"dep{s}Ct"] = txt(z["rows"]["CytoTRACE"]["spearman_with_stage"])
        libs = [v["median_library_size"] for _, v in sorted(dp["per_stage"].items(), key=lambda kv: int(kv[0]))]
        N[f"dep{s}LibFirst"], N[f"dep{s}LibLast"] = f"{libs[0]:,.0f}", f"{libs[-1]:,.0f}"
        N[f"dep{s}LibRestLo"], N[f"dep{s}LibRestHi"] = f"{min(libs[1:]):,.0f}", f"{max(libs[1:]):,.0f}"
        t = dp["thinned"].get("1000")
        if t:
            N[f"dep{s}ThinGc"] = txt(t["spearman_detected_genes_with_stage"])
            N[f"dep{s}ThinKept"] = f"{t['cells_kept']:,}"
            if "spearman_cytotrace_with_stage" in t:
                N[f"dep{s}ThinCs"] = txt(t["cytotrace_conditional_skill_on_thinned_gene_count"])
                N[f"dep{s}ThinCt"] = txt(t["spearman_cytotrace_with_stage"])
                N[f"dep{s}ThinGcs"] = txt(t["spearman_gene_count_signature_with_stage"])
        k = dp.get("kept_cells_raw_depth")
        if k:
            N[f"dep{s}KeptCt"] = txt(k["spearman_cytotrace_with_stage"])
            N[f"dep{s}KeptGc"] = txt(k["spearman_detected_genes_with_stage"])
        for tag, name in (("window", "Win"), ("window_thinned", "WinThin")):
            w = dp.get("shared_window", {}).get(tag)
            if w:
                N[f"dep{s}{name}Ct"] = txt(w["spearman_cytotrace_with_stage"])
                N[f"dep{s}{name}Gc"] = txt(w["spearman_detected_genes_with_stage"])
                N[f"dep{s}{name}Gcs"] = txt(w["spearman_gene_count_signature_with_stage"])
                N[f"dep{s}{name}Cells"] = f"{w['cells']:,}"
    clearing = [r for r in rows if r["clears_registered"]]
    N["nClear"] = str(len(clearing))
    N["nClearExceed"] = str(sum(r["exceeds"] for r in clearing))
    N["nClearExceedJoint"] = str(sum(r["joint_exceeds"] for r in clearing))
    N["nClearSample"] = str(sum(r["sample_supported"] for r in clearing))
    words = ["none", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
    hz = d["GSE106474"]["rows"]
    hp = dict(zip(ORDER, holm([hz[n]["sample_test"]["p_two_sided"] for n in ORDER])))
    for n in ORDER:
        N[f"rowF{n}SampleHolm"] = pval(hp[n]).replace("$<$", "below ")
    for g, sl in LETTERS.items():
        th = d[g]["depth"]["thinned"]["1000"]
        N[f"dep{sl}ThinLost"] = f"{d[g]['n'] - th['cells_kept']:,}"
        N[f"dep{sl}Cells"] = f"{d[g]['n']:,}"
    for key in ("nClear", "nClearExceed", "nClearExceedJoint", "nClearSample"):
        N[key + "Word"] = words[int(N[key])]
    N["nClearNotSampleWord"] = words[int(N["nClear"]) - int(N["nClearSample"])]
    z = d["GSE106474"]["controls"]["a1_g200"]
    N["ctlFloorGap"] = pc(max(abs(z[c]["fraction_above_registered_floor_bracketed"] - z[c]["fraction_above_registered_floor_nearest"])
                              for c in ("detected_genes", "pcc_x_degree", "shannon_entropy", "log10_library_size", "joint")))
    for r in cal["cluster"]:
        tag = {"linear": "Line", "cube": "Cube", "square": "Square"}[r["shape"]]
        if r["shape"] == "linear":
            tag += {0.5: "Half", 0.9: "Nine"}[r["rho"]]
        key = f"cal{tag}Sd{str(r['sample_sd']).replace('.', 'p')}B{str(r['batch_sd']).replace('.', 'p')}T{str(r['planted_trend']).replace('.', 'p')}"
        N[key + "Sample"] = pc(r["sample_test_stratum_mean"])
        N[key + "Ls"] = pc(r["sample_test_least_squares"])
        N[key + "Floor"] = pc(r["clears_registered_floor"])
        if r["clears_permutation_floor"] is not None:
            N[key + "Perm"] = pc(r["clears_permutation_floor"])
    zero = [r for r in cal["cluster"] if r["planted_trend"] == 0]
    if zero:
        N["calNullSampleMin"] = pc(min(r["sample_test_stratum_mean"] for r in zero))
        N["calNullSampleMax"] = pc(max(r["sample_test_stratum_mean"] for r in zero))
        N["calNullLsMax"] = pc(max(r["sample_test_least_squares"] for r in zero))
        N["calNullNoBatchMax"] = pc(max(r["sample_test_stratum_mean_no_batch_strata"] for r in zero))
        N["calNullFloorMin"] = pc(min(r["clears_registered_floor"] for r in zero))
        N["calNullFloorMax"] = pc(max(r["clears_registered_floor"] for r in zero))
        perm = [r["clears_permutation_floor"] for r in zero if r["clears_permutation_floor"] is not None]
        N["calNullPermMin"], N["calNullPermMax"] = pc(min(perm)), pc(max(perm))
        N["calReps"] = str(zero[0]["reps"])
        for r in cal["cluster"]:
            if r["planted_trend"] > 0:
                N["calPower" + {0.05: "Low", 0.1: "Mid", 0.2: "High"}[r["planted_trend"]]] = pc(r["sample_test_stratum_mean"])
    if cal["concordance"]:
        N["calConcMin"] = pc(min(r["false_clear"] for r in cal["concordance"]))
        N["calConcMax"] = pc(max(r["false_clear"] for r in cal["concordance"]))
    d2 = cal["depth_2"]
    if d2:
        for geo, tag in (("GSE117498h", "Blood"), ("GSE125970", "Gut")):
            a = d2["atlases"][geo]
            N[f"dtwo{tag}Raw"] = f"{a['deposit_detected_genes']['auroc_top_vs_bottom']:.3f}"
            N[f"dtwo{tag}Kept"] = f"{a['kept_cells_raw_depth']['detected_genes']['auroc_top_vs_bottom']:.3f}"
            N[f"dtwo{tag}Thin"] = f"{a['thinned']['1000']['detected_genes']['auroc_top_vs_bottom']:.3f}"
            N[f"dtwo{tag}ThinLow"] = f"{a['thinned']['500']['detected_genes']['auroc_top_vs_bottom']:.3f}"
            N[f"dtwo{tag}ThinHigh"] = f"{a['thinned']['2000']['detected_genes']['auroc_top_vs_bottom']:.3f}"
            N[f"dtwo{tag}TauRaw"] = txt(a["deposit_detected_genes"]["tau_b_all_ranked"])
            N[f"dtwo{tag}TauThin"] = txt(a["thinned"]["1000"]["detected_genes"]["tau_b_all_ranked"])
            N[f"dtwo{tag}GcsRaw"] = f"{a['raw']['gene_count_signature']['auroc_top_vs_bottom']:.3f}"
            N[f"dtwo{tag}GcsThin"] = f"{a['thinned']['1000']['gene_count_signature']['auroc_top_vs_bottom']:.3f}"
            N[f"dtwo{tag}GcsTauRaw"] = txt(a["raw"]["gene_count_signature"]["tau_b_all_ranked"])
            N[f"dtwo{tag}GcsTauThin"] = txt(a["thinned"]["1000"]["gene_count_signature"]["tau_b_all_ranked"])
            t = a["thinned"]["1000"]["detected_genes"]
            N[f"dtwo{tag}TopKept"], N[f"dtwo{tag}BottomKept"] = f"{t['top_cells']:,}", f"{t['bottom_cells']:,}"
            N[f"dtwo{tag}TopAll"] = f"{a['deposit_detected_genes']['top_cells']:,}"
            N[f"dtwo{tag}BottomAll"] = f"{a['deposit_detected_genes']['bottom_cells']:,}"
            for pop, v in a["by_population"].items():
                key = "".join(ch for ch in pop if ch.isalpha())
                N[f"dtwo{tag}Lib{key}"] = f"{v['median_library_size']:,.0f}"
                if v["median_detected_genes_at_1000"] is not None:
                    N[f"dtwo{tag}Thin{key}"] = f"{v['median_detected_genes_at_1000']:,.0f}"
        for geo, s in LETTERS.items():
            v = d2["staged"][geo]
            N[f"dtwoPcc{s}Raw"] = txt(v["spearman_pcc_with_stage_raw"])
            N[f"dtwoPcc{s}Thin"] = txt(v["spearman_pcc_with_stage_at_1000"])
            N[f"dtwoPcc{s}Agree"] = f"{v['spearman_pcc_raw_with_pcc_at_1000']:.2f}"
            N[f"dtwoPcc{s}KeptRaw"] = txt(v["spearman_pcc_with_stage_raw_kept_cells"])
            N[f"dtwoPcc{s}LostPct"] = pc(1 - v["cells_kept"] / v["cells"])
    d3 = cal["depth_3"]
    if d3:
        for g, sl in LETTERS.items():
            for n, r in d3["series"][g].items():
                N[f"dthree{sl}{n}Stage"] = txt(r["spearman_with_stage_at_1000"])
                N[f"dthree{sl}{n}Cs"] = txt(r["conditional_skill_at_1000"])
                N[f"dthree{sl}{n}Floor"] = f"{r['permutation_floor']:.3f}"
                N[f"dthree{sl}{n}Agree"] = f"{r['spearman_raw_score_with_score_at_1000']:.2f}"
    digits = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}
    safe = {}
    for k, v in N.items():
        name = "".join(digits.get(ch, ch) for ch in k if ch.isalnum())
        if name in safe:
            raise SystemExit(f"macro name collision: {name}")
        safe[name] = v
    return "\n".join(f"\\newcommand{{\\{k}}}{{{v}}}" for k, v in sorted(safe.items())) + "\n", safe


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    d, perm, batch, cal = load()
    rows = scorecard(d, perm, batch)
    write_scorecard_md(rows)
    out = Path(a.out)
    (out / "generated_tables.tex").write_text(tables(d, rows, cal))
    tex, raw = numbers(d, rows, cal)
    (out / "generated_numbers.tex").write_text(tex)
    (HERE / "results" / "numbers.json").write_text(json.dumps(raw, indent=1, sort_keys=True))
    print((HERE / "results" / "SCORECARD.md").read_text())


if __name__ == "__main__":
    main()
