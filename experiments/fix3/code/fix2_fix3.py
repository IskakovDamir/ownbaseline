"""
FIX 3 GATE 2 — FIX 2 replicate on Briggs: joint-primitive residualization of the 4
substantive scores (CytoTRACE, SR, ORIGINS, dpath) on ALL 4 low-order primitives
jointly (gene count + PCC + Shannon + log10 lib), both kernels, bootstrap CI seed 42.
Same pre-registered rule as the zebrafish FIX 2: both-kernels-ADDS-BEYOND → SURVIVES-ALL
else COMPOUND-LOW-ORDER. Imports the exact decider functions.
Output: fix3/results/fix2_joint_primitive.json
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
from conditional_skill import align, _unit, SEED, N_BOOT_WT, N_BOOT_KT

PREP = VAULT / "fix3/data/prepared"
RES = VAULT / "fix3/results"
OUT = RES / "fix2_joint_primitive.json"


def main():
    p = np.load(PREP / "primitives.npz")
    gc, pcc, shan = p["gene_count"], p["pcc_string_v12"], p["shannon_H"]
    rk = p["rank_ordinal"].astype(float)
    import csv as _csv
    loglib = np.log10(np.array([float(r["lib_size"]) for r in
                                _csv.DictReader(open(PREP / "cells.tsv"), delimiter="\t")]))
    dec = json.load(open(RES / "gate2_conditional_skill.json"))["conditional_skill"]
    full = {"CytoTRACE_v1": (np.load(RES / "cytotrace_scores.npz")["ct_v1"], "CytoTRACE_v1 | gene_count"),
            "SR": (np.load(RES / "scent_scores.npz")["sr"], "SR | PCC(x,degree)"),
            "ORIGINS": (np.load(RES / "origins_scores.npz")["origins"], "ORIGINS | PCC(x,degree)")}
    dp = np.load(RES / "dpath_scores.npz"); dpath_v, dci = dp["dpath"], dp["cell_idx"]

    tasks, meta = [], []
    for nm, (v, dk) in full.items():
        covs = [gc, pcc, shan, loglib]; assert len(covs) == 4
        tasks.append((nm, align(v, rk), covs, rk)); meta.append((nm, dk, len(rk)))
    covs = [gc[dci], pcc[dci], shan[dci], loglib[dci]]; assert len(covs) == 4
    tasks.append(("dpath", align(dpath_v, rk[dci]), covs, rk[dci])); meta.append(("dpath", "dpath | Shannon_H", len(dci)))

    with Pool(4) as pool:
        results = dict(pool.map(_unit, tasks))

    out = {"design": {"joint_primitives": ["gene_count", "PCC(x,degree)", "Shannon_H", "log10(lib)"],
                      "n_covariates": 4, "seed": SEED,
                      "n_boot": {"scipy_weightedtau": N_BOOT_WT, "kang_wdm_taub": N_BOOT_KT}},
           "per_score": {}}
    print("\n=== FIX 2 (Briggs): single vs joint-4-primitive ===")
    for nm, dk, n in meta:
        j = results[nm]; single = dec[dk]
        both_pos = (j["scipy_weightedtau"]["verdict"] == "ADDS-BEYOND" and
                    j["kang_wdm_taub"]["verdict"] == "ADDS-BEYOND")
        row = {"n": int(n),
               "single_primitive": {"primitive": dk.split("|")[1].strip(),
                                    "scipy_weightedtau": single["scipy_weightedtau"]["tau"],
                                    "kang_wdm_taub": single["kang_wdm_taub"]["tau"], "verdict": single["combined_verdict"]},
               "joint_4_primitive": {"scipy_weightedtau": j["scipy_weightedtau"], "kang_wdm_taub": j["kang_wdm_taub"],
                                     "kernels_agree": j["kernels_agree"]},
               "joint_verdict": "SURVIVES-ALL-LOW-ORDER" if both_pos else "COMPOUND-LOW-ORDER"}
        out["per_score"][nm] = row
        a, b = j["scipy_weightedtau"], j["kang_wdm_taub"]
        print(f"  {nm:13s} n={n:6d} single scipy {single['scipy_weightedtau']['tau']:+.3f} "
              f"kang {single['kang_wdm_taub']['tau']:+.3f} --> JOINT scipy {a['tau']:+.3f}"
              f"[{a['CI95'][0]:+.3f},{a['CI95'][1]:+.3f}] kang {b['tau']:+.3f}"
              f"[{b['CI95'][0]:+.3f},{b['CI95'][1]:+.3f}] => {row['joint_verdict']}")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
