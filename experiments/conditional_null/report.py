#!/usr/bin/env python3
"""
report.py: the tables and the marginal correlations of experiments/conditional_null.

    python3 experiments/conditional_null/report.py

Reads results/conditional_null.json and the score tables, writes results/summary.json
and results/tables.tex, the rows printed in preprint Appendix J. Nothing is typed by
hand: every number in those tables comes from the run.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from own_baseline.paths import data_root  # noqa: E402

RES = HERE / "results"
BATCHES = ("DS2", "DS3", "DS4", "DS5")
REGISTERED_FLOOR = {"CytoTRACE": 0.0247, "SR": 0.0311, "ORIGINS": 0.0247, "NCG": 0.0411, "dpath": 0.0411,
                    "StemFinder": 0.0247, "SLICE": 0.0479, "CCAT": 0.0115, "MCE": 0.0325}
ORDER = ["CytoTRACE", "SR", "ORIGINS", "NCG", "dpath", "StemFinder", "SLICE", "CCAT", "MCE"]
JOINT_FLOOR_106474 = {"CytoTRACE": 0.0171, "SR": 0.0227, "ORIGINS": 0.0171, "dpath": 0.0325, "StemFinder": 0.0171, "MCE": 0.0233}
PRIMITIVE = {"CytoTRACE": "gene count", "SR": "PCC(x, degree)", "ORIGINS": "PCC(x, degree)", "CCAT": "PCC(x, degree)",
             "MCE": "PCC(x, degree)", "dpath": "Shannon entropy", "SLICE": "Shannon entropy"}


def num(v, d=3):
    s = f"{abs(v):.{d}f}"
    return f"$-${s}" if v < 0 and float(s) != 0 else f"+{s}"


def fl4(v, d=4):
    s = f"{abs(v):.{d}f}"
    return f"$-${s}" if v < 0 else s


def design_floors():
    best = {}
    for f in ("wagner_design_points.csv", "wagner_design_points_mce.csv"):
        for r in csv.DictReader(open(RES / f)):
            if r["kernel"] == "kendalltau":
                k = (float(r["rho_target"]), int(r["k_covariates"]))
                best[k] = max(best.get(k, 0), float(r["p97.5"]))
    return best


def marginals(deposit, xdir, wroot):
    out = {}
    z = pd.read_csv(deposit / "datasets/GSE106474/cells.csv.gz")
    s = pd.read_csv(deposit / "scores/GSE106474_scores.csv.gz")
    o = z["ordinal"].to_numpy(float)
    rho = lambda v, oo: float(spearmanr(v, oo).statistic)  # noqa: E731
    out["GSE106474"] = {"gene count": rho(z["detected_genes"], o), "PCC(x, degree)": rho(z["pcc_x_degree"], o),
                        "Shannon entropy": rho(z["shannon_entropy"], o), "CytoTRACE": rho(s["cytotrace_v1"], o),
                        "gene-count signature": rho(s["cytotrace_gene_count_signature"], o), "SR": rho(s["scent_sr"], o),
                        "CCAT": rho(s["scent_ccat"], o), "MCE": rho(s["mce"], o), "ORIGINS": rho(s["origins"], o)}
    x = pd.read_csv(deposit / "datasets/GSE113074/cells.csv.gz")
    ox = x["ordinal"].to_numpy(float)
    ct, sc, og = (np.load(xdir / f"{n}_scores.npz") for n in ("cytotrace", "scent", "origins"))
    out["GSE113074"] = {"gene count": rho(x["detected_genes"], ox), "PCC(x, degree)": rho(x["pcc_x_degree"], ox),
                        "Shannon entropy": rho(x["shannon_entropy"], ox), "CytoTRACE": rho(ct["ct_v1"], ox),
                        "gene-count signature": rho(ct["gcs"], ox), "SR": rho(sc["sr"], ox), "CCAT": rho(sc["ccat"], ox),
                        "ORIGINS": rho(og["origins"], ox)}
    p = np.load(wroot / "track2/data/prepared/primitives.npz")
    ow = p["ordinal"]
    ct, sc = (np.load(wroot / f"track2/results/{n}_scores.npz") for n in ("cytotrace", "scent"))
    m = pd.read_csv(wroot / "w5_new_rows/scores/GSE106474/mce.csv")["mce"].to_numpy()
    out["GSE112294"] = {"gene count": rho(p["gene_count"], ow), "PCC(x, degree)": rho(p["pcc_string_v12"], ow),
                        "Shannon entropy": rho(p["shannon_H"], ow), "CytoTRACE": rho(ct["ct_v1"], ow),
                        "gene-count signature": rho(ct["gcs"], ow), "SR": rho(sc["sr"], ow), "CCAT": rho(sc["ccat"], ow),
                        "MCE": rho(m, ow)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default=str(REPO / "data/deposit/ownbaseline-data-v1"))
    ap.add_argument("--xenopus", default=str(data_root() / "fix3/results"))
    ap.add_argument("--wagner", default=str(data_root() / "wagner2018/root"))
    a = ap.parse_args()
    d = json.load(open(RES / "conditional_null.json"))
    floors = {(f["row"], f["m"]): f for f in d["floors"]}
    primary = {f["row"]: f for f in d["floors"] if "isotonic" in f["registered"]}
    gcm = {g["row"]: g for g in d["gcm"]}
    loso = {r["row"]: r for r in d["leave_one_stage_out"]}
    boot = {r["row"]: r for r in d["sample_bootstrap"]}
    batch = {f["row"]: f for f in d["floors"] if " DS" in f["row"]}
    lines = ["% Table A6: GSE106474"]
    for name in ORDER:
        f = primary[f"GSE106474 {name}"]
        r, g = f["registered"], gcm[f"GSE106474 {name}"]["registered"]
        above = sum(batch[f"GSE106474 {b} {name}"]["registered"]["linear"]["clears"] for b in BATCHES)
        ci, lo = boot[f"GSE106474 {name}"]["CI95"], loso[f"GSE106474 {name}"]
        row = "SR" if name == "SR" else name
        lines.append(" & ".join([row, num(r["linear"]["value"]), f"{REGISTERED_FLOOR[name]:.4f}", fl4(r["linear"]["floor_perm"]),
                                 f"{num(r['stratum_mean']['value'])} / {fl4(r['stratum_mean']['floor_perm'])}",
                                 num(r["isotonic"]["value"]), f"{g['T']:.1f}".replace("-", "$-$"), f"{above}/4",
                                 f"[{num(ci[0])}, {num(ci[1])}]", f"{num(lo['min'])} to {num(lo['max'])}"]) + r" \\")
    lines.append("% Table A7: batches")
    for name in ORDER:
        cells = []
        for b in BATCHES:
            x = batch[f"GSE106474 {b} {name}"]["registered"]["linear"]
            cells.append(f"{num(x['value'])} / {fl4(x['floor_perm'], 3)}")
        lines.append(" & ".join([name, *cells]) + r" \\")
    lines.append("% Table A8: the other two series")
    df = design_floors()
    marg = marginals(Path(a.deposit), Path(a.xenopus), Path(a.wagner))
    summary = {"marginals": marg, "rows": []}
    series = [("GSE112294", ["CytoTRACE", "SR", "CCAT", "MCE"], 7), ("GSE113074", ["CytoTRACE", "SR", "ORIGINS", "CCAT", "dpath", "SLICE"], 10)]
    for acc, names, L in series:
        for name in names:
            f = primary[f"{acc} {name}"]
            r, auth = f["registered"], f["authors"]
            j = gcm.get(f"{acc} {name} joint")
            rho = None
            if acc == "GSE112294":
                rho = {"CytoTRACE": 0.172, "SR": 0.923, "CCAT": 0.998, "MCE": 0.89}[name]
                fl = df[(rho, 1)]
                jfl = df.get((rho, 4))
            else:
                fl, jfl = None, None
            summary["rows"].append({"series": acc, "row": name, "n": f["n"], "levels": L, "rho": rho, "cs": r["linear"]["value"], "floor": fl,
                                    "perm": r["linear"]["floor_perm"], "stratum": r["stratum_mean"], "authors": auth["linear"],
                                    "joint": j and {"cs": j["cs_linear"], "T": j["registered"]["T"], "floor": jfl}})
            lines.append(" & ".join([acc, name, PRIMITIVE[name], f"{f['n']:,}", str(L), num(r["linear"]["value"]),
                                     f"{fl:.4f}" if fl else "--", fl4(r["linear"]["floor_perm"]),
                                     f"{num(r['stratum_mean']['value'])} / {fl4(r['stratum_mean']['floor_perm'])}",
                                     f"{num(auth['linear']['value'])} / {fl4(auth['linear']['floor_perm'], 3)}",
                                     (f"{num(j['cs_linear'])}, " + f"{j['registered']['T']:.1f}".replace("-", "$-$") if j else "--")]) + r" \\")
    lines.append("% Table all: every other value of the k = 1 rows")
    for acc in ("GSE106474", "GSE112294", "GSE113074"):
        for name in ORDER:
            f = primary.get(f"{acc} {name}")
            if f is None:
                continue
            r, auth = f["registered"], f["authors"]["linear"]
            g, lo = gcm[f"{acc} {name}"]["registered"], loso[f"{acc} {name}"]
            lines.append(" & ".join([acc, name, f"{num(r['isotonic']['value'])} / {fl4(r['isotonic']['floor_perm'])}",
                                     f"{g['T']:.1f}".replace("-", "$-$"), num(g["partial_rank_correlation"]),
                                     f"{num(auth['value'])} / {fl4(auth['floor_perm'], 3)}",
                                     f"{num(lo['min'])} to {num(lo['max'])}"]) + r" \\")
    (RES / "tables.tex").write_text("\n".join(lines) + "\n")
    (RES / "summary.json").write_text(json.dumps(summary, indent=1))
    print("\n".join(lines))
    print(json.dumps(marg, indent=0)[:1500])


if __name__ == "__main__":
    main()
