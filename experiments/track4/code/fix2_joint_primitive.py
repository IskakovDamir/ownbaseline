"""
Paper A FIX 2 — JOINT-primitive residualization (the Tier-1 probe).

For each of the 4 substantive scores (CytoTRACE, SR, ORIGINS, dpath), residualize
the score's RANK jointly on ALL FOUR low-order primitives at once:
    gene count |supp(x)| , PCC(x,degree) STRING v12 , Shannon H(x/Σx) , log10(library size)
via multiple OLS on ranks (rank_resid_multi — the SAME function the decider used),
then conditional skill of the residual vs the 12-stage Kimmel ordinal, BOTH kernels
(scipy weightedtau + Kendall τ-b == Kang wdm uniform), bootstrap 95% CI seed 42.

Pre-registered interpretation (fixed BEFORE running; report whichever the data shows):
  CI excludes 0 on the positive side under BOTH kernels → SURVIVES-ALL-LOW-ORDER
    (real ordering signal beyond every low-order statistic jointly).
  CI includes 0 (either kernel) / kernels disagree → COMPOUND-LOW-ORDER
    (the score is a combination of low-order statistics, not beyond them).

dpath ran on a 3,000-cell seed-42 subsample; its row uses primitives+ordinal
restricted to that cell_idx (n disclosed).

Output: track4/results/fix2_joint_primitive.json
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
# reuse the DECIDER's exact functions (same kernels, same bootstrap, same seed)
from conditional_skill import (scipy_wtau, kang_taub, rank_resid_multi, _verdict,
                               align, _unit, SEED, N_BOOT_WT, N_BOOT_KT)

PREP = VAULT / "track2/data/prepared"
RES = VAULT / "track2/results"
OUT = VAULT / "track4/results/fix2_joint_primitive.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    p = np.load(PREP / "primitives.npz")
    gc, pcc, shan = p["gene_count"], p["pcc_string_v12"], p["shannon_H"]
    rk = p["rank_kimmel"].astype(float)
    import csv as _csv
    loglib = np.log10(np.array([float(r["lib_size"]) for r in
                                _csv.DictReader(open(PREP / "cells.tsv"), delimiter="\t")]))

    # the FOUR joint low-order covariates (order fixed)
    PRIM_NAMES = ["gene_count", "PCC(x,degree)", "Shannon_H", "log10(lib)"]

    # substantive scores + their single-primitive decider baseline (for the table)
    dec = json.load(open(RES / "gate2_conditional_skill.json"))["conditional_skill"]
    scores_full = {
        "CytoTRACE_v1": (np.load(RES / "cytotrace_scores.npz")["ct_v1"], "CytoTRACE_v1 | gene_count"),
        "SR":           (np.load(RES / "scent_scores.npz")["sr"],       "SR | PCC(x,degree)"),
        "ORIGINS":      (np.load(RES / "origins_scores.npz")["origins"],"ORIGINS | PCC(x,degree)"),
    }
    dp = np.load(RES / "dpath_scores.npz"); dpath_v, dci = dp["dpath"], dp["cell_idx"]

    tasks, meta = [], []
    # full-cell scores
    for nm, (v, deckey) in scores_full.items():
        s = align(v, rk)
        covs = [gc, pcc, shan, loglib]            # <-- FOUR primitives, jointly
        assert len(covs) == 4
        tasks.append((nm, s, covs, rk)); meta.append((nm, deckey, len(rk)))
    # dpath on its subsample
    s = align(dpath_v, rk[dci])
    covs = [gc[dci], pcc[dci], shan[dci], loglib[dci]]
    assert len(covs) == 4
    tasks.append(("dpath", s, covs, rk[dci])); meta.append(("dpath", "dpath | Shannon_H", len(dci)))

    print(f"running {len(tasks)} joint-residual units (4 primitives each) in parallel ...")
    with Pool(4) as pool:
        results = dict(pool.map(_unit, tasks))

    out = {"design": {"joint_primitives": PRIM_NAMES, "n_covariates": 4,
                      "method": "rank(score) sign-aligned, then multiple OLS on rank(gene_count),"
                                " rank(PCC), rank(Shannon), rank(log10 lib) jointly (+intercept);"
                                " conditional skill = weighted-tau(residual, 12-stage ordinal)",
                      "kernels": ["scipy.stats.weightedtau", "Kendall tau-b == Kang wdm uniform"],
                      "seed": SEED, "n_boot": {"scipy_weightedtau": N_BOOT_WT, "kang_wdm_taub": N_BOOT_KT}},
           "per_score": {}}

    print("\n=== SINGLE-primitive (decider) vs JOINT-4-primitive conditional skill ===")
    for nm, deckey, n in meta:
        j = results[nm]
        single = dec[deckey]
        row = {
            "n": int(n),
            "single_primitive": {"primitive": deckey.split("|")[1].strip(),
                                 "scipy_weightedtau": single["scipy_weightedtau"]["tau"],
                                 "kang_wdm_taub": single["kang_wdm_taub"]["tau"],
                                 "verdict": single["combined_verdict"]},
            "joint_4_primitive": {"scipy_weightedtau": j["scipy_weightedtau"],
                                  "kang_wdm_taub": j["kang_wdm_taub"],
                                  "kernels_agree": j["kernels_agree"]},
        }
        # pre-registered joint verdict
        both_pos = (j["scipy_weightedtau"]["verdict"] == "ADDS-BEYOND" and
                    j["kang_wdm_taub"]["verdict"] == "ADDS-BEYOND")
        row["joint_verdict"] = "SURVIVES-ALL-LOW-ORDER" if both_pos else "COMPOUND-LOW-ORDER"
        out["per_score"][nm] = row
        a, b = j["scipy_weightedtau"], j["kang_wdm_taub"]
        print(f"  {nm:13s} n={n:5d} | single scipy {single['scipy_weightedtau']['tau']:+.3f} "
              f"kang {single['kang_wdm_taub']['tau']:+.3f} --> JOINT scipy {a['tau']:+.3f}"
              f"[{a['CI95'][0]:+.3f},{a['CI95'][1]:+.3f}] kang {b['tau']:+.3f}"
              f"[{b['CI95'][0]:+.3f},{b['CI95'][1]:+.3f}] => {row['joint_verdict']}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
