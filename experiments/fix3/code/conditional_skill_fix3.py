"""
FIX 3 GATE 2 — conditional-skill test on the Briggs 10-stage ordinal, mirroring the
zebrafish decider EXACTLY (imports conditional_skill.py's kernel/residual functions,
same seed 42, same N_BOOT, both kernels). Also (a) re-validates the Python SR/CCAT/ORIGINS
subsample scores against the real SCENT/ORIGINS R CSVs, (b) writes the per-score table
+ marginals for the generalization read.
Output: fix3/results/gate2_conditional_skill.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.stats import spearmanr

VAULT = Path("/Users/damir/damir-research-vault/06-code/tautology-diagnostic")
sys.path.insert(0, str(VAULT / "atlas_run"))
sys.path.insert(0, str(VAULT / "track2/code"))
from conditional_skill import (scipy_wtau, kang_taub, rank_resid_multi, _verdict, align,
                               _unit, SEED, N_BOOT_WT, N_BOOT_KT)
from wdm_tau import wdm_tau_b_bit

PREP = VAULT / "fix3/data/prepared"
RES = VAULT / "fix3/results"
SC = Path("/private/tmp/claude-501/-Users-damir-damir-research-vault/f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad")
OUT = RES / "gate2_conditional_skill.json"


def validate():
    """Python subsample vs real SCENT/ORIGINS R CSVs (Spearman)."""
    import csv as _csv
    v = {}
    sm = SC / "fix3_scent_io"
    if (sm / "scent_R_subsample.csv").exists() and (sm / "meta.npz").exists():
        m = np.load(sm / "meta.npz"); rows = list(_csv.DictReader(open(sm / "scent_R_subsample.csv")))
        cR = np.array([float(r["ccat_R"]) for r in rows]); sR = np.array([float(r["sr_R"]) for r in rows])
        v["CCAT"] = float(spearmanr(cR, m["sub_ccat"]).statistic)
        v["SR"] = float(spearmanr(sR, m["sub_sr"]).statistic)
    om = SC / "fix3_origins_io"
    if (om / "origins_R_subsample.csv").exists() and (om / "meta.npz").exists():
        m = np.load(om / "meta.npz"); rows = list(_csv.DictReader(open(om / "origins_R_subsample.csv")))
        oR = np.array([float(r["origins_R"]) for r in rows])
        v["ORIGINS"] = float(spearmanr(oR, m["sub_raw"]).statistic)
    return v


def main():
    p = np.load(PREP / "primitives.npz")
    prims = {"gene_count": p["gene_count"], "PCC(x,degree)": p["pcc_string_v12"], "Shannon_H": p["shannon_H"]}
    rk = p["rank_ordinal"].astype(float)
    import csv as _csv
    loglib = np.log10(np.array([float(r["lib_size"]) for r in
                                _csv.DictReader(open(PREP / "cells.tsv"), delimiter="\t")]))

    full = dict(prims); prov = {}
    full["CytoTRACE_v1"] = np.load(RES / "cytotrace_scores.npz")["ct_v1"]
    if (RES / "scent_scores.npz").exists():
        sc = np.load(RES / "scent_scores.npz"); full["SR"], full["CCAT"] = sc["sr"], sc["ccat"]
    if (RES / "origins_scores.npz").exists():
        full["ORIGINS"] = np.load(RES / "origins_scores.npz")["origins"]
    subs = {}
    for nm, fn, key in [("SLICE", "slice_scores.npz", "slice"), ("dpath", "dpath_scores.npz", "dpath")]:
        if (RES / fn).exists():
            d = np.load(RES / fn); subs[nm] = (d[key], d["cell_idx"])

    pairs = [("CytoTRACE_v1", "gene_count"), ("SR", "PCC(x,degree)"), ("SR", "CCAT"),
             ("CCAT", "PCC(x,degree)"), ("ORIGINS", "gene_count"), ("ORIGINS", "PCC(x,degree)"),
             ("SLICE", "Shannon_H"), ("dpath", "Shannon_H")]
    tasks, meta = [], []
    for sn, pn in pairs:
        if sn in full and pn in full:
            tasks.append((f"{sn} | {pn}", align(full[sn], rk), [full[pn]], rk)); meta.append(("cond", f"{sn} | {pn}"))
        elif sn in subs and pn in full:
            sv, ci = subs[sn]
            tasks.append((f"{sn} | {pn} [n={len(ci)}]", align(sv, rk[ci]), [full[pn][ci]], rk[ci]))
            meta.append(("cond", f"{sn} | {pn}"))
        else:
            meta.append(("infra", f"{sn} | {pn}"))
    # depth robustness
    robust = [("CytoTRACE_v1", "gene_count"), ("SR", "PCC(x,degree)"), ("SR", "CCAT"),
              ("CCAT", "PCC(x,degree)"), ("ORIGINS", "gene_count"), ("ORIGINS", "PCC(x,degree)")]
    for sn, pn in robust:
        if sn in full and pn in full:
            tasks.append((f"{sn} | {pn} + log10(lib)", align(full[sn], rk), [full[pn], loglib], rk))
            meta.append(("robust", f"{sn} | {pn} + log10(lib)"))

    print(f"running {len(tasks)} bootstrap units ...")
    with Pool(6) as pool:
        results = dict(pool.map(_unit, tasks))

    out = {"atlas": "Briggs 2018 Xenopus GSE113074 (10-stage Nieuwkoop-Faber ordinal)",
           "n_cells_full": int(len(rk)), "seed": SEED,
           "n_boot": {"scipy_weightedtau": N_BOOT_WT, "kang_wdm_taub": N_BOOT_KT},
           "reimpl_validation_spearman_vs_real_R": validate(),
           "conditional_skill": {}, "depth_robustness_cond_given_primitive_AND_loglib": {},
           "marginal_skill": {}, "spearman_score_vs_primitive": {}}
    for kind, label in meta:
        if kind == "infra":
            out["conditional_skill"][label] = {"status": "INCONCLUSIVE-infra"}
        elif kind == "cond":
            key = [t[0] for t in tasks if t[0].startswith(label)][0]
            out["conditional_skill"][label] = results[key]
    for label in [m[1] for m in meta if m[0] == "robust"]:
        out["depth_robustness_cond_given_primitive_AND_loglib"][label] = results[label]
    for sn, pn in pairs:
        if sn in full and pn in full:
            out["spearman_score_vs_primitive"][f"{sn} | {pn}"] = float(spearmanr(full[sn], full[pn]).statistic)
        elif sn in subs and pn in full:
            sv, ci = subs[sn]
            out["spearman_score_vs_primitive"][f"{sn} | {pn}"] = float(spearmanr(sv, full[pn][ci]).statistic)
    for nm, v in full.items():
        out["marginal_skill"][nm] = {"scipy_weightedtau": scipy_wtau(v, rk), "kang_wdm_taub": kang_taub(v, rk),
                                     "spearman_vs_ordinal": float(spearmanr(v, rk).statistic)}
    for nm, (sv, ci) in subs.items():
        out["marginal_skill"][nm] = {"scipy_weightedtau": scipy_wtau(sv, rk[ci]), "kang_wdm_taub": kang_taub(sv, rk[ci]),
                                     "spearman_vs_ordinal": float(spearmanr(sv, rk[ci]).statistic), "n": int(len(ci))}
    # K2 equivalence spot-check
    s = align(full["CytoTRACE_v1"], rk); resid = rank_resid_multi(s, [full["gene_count"]])
    out["K2_equivalence_check"] = {"kendalltau": kang_taub(resid, rk),
                                   "wdm_tau_b_bit_uniform": float(wdm_tau_b_bit(resid, rk, np.ones(len(rk)))),
                                   "abs_diff": abs(kang_taub(resid, rk) - float(wdm_tau_b_bit(resid, rk, np.ones(len(rk)))))}
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}\n")
    print("reimpl validation vs real R:", out["reimpl_validation_spearman_vs_real_R"])
    print("\n=== CONDITIONAL SKILL (Briggs 10-stage) ===")
    for k, r in out["conditional_skill"].items():
        if "status" in r:
            print(f"  {k:26s} {r['status']}"); continue
        a, b = r["scipy_weightedtau"], r["kang_wdm_taub"]
        print(f"  {k:26s} scipy {a['tau']:+.3f}[{a['CI95'][0]:+.3f},{a['CI95'][1]:+.3f}] "
              f"kang {b['tau']:+.3f}[{b['CI95'][0]:+.3f},{b['CI95'][1]:+.3f}] => {r['combined_verdict']}")


if __name__ == "__main__":
    main()
