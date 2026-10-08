"""
Track 2 Gate 2 — Phase C: sign-agnostic conditional-skill test on the 12-stage
Farrell ordinal, BOTH kernels, bootstrap 95% CI seed 42.

Kernels:
  (K1) scipy.stats.weightedtau     — hyperbolic-weighted tau (top-weighted). ~490 ms/call
       at n=39.5k, so its bootstrap uses N_BOOT_WT resamples (parallelized).
  (K2) Kang wdm, uniform weights   — == Kendall tau-b on this NON-Kang atlas (no Kang
       absolute-order weights exist); computed via scipy.stats.kendalltau (C, ~26 ms),
       spot-validated == atlas_run.wdm_tau.wdm_tau_b_bit(·, ones) to <1e-9. N_BOOT_KT.

Sign-agnostic: each score is sign-aligned to the ordinal (negate if Spearman<0) BEFORE
rank-residualizing on its §3a primitive, so a positive residual-vs-ordinal association
reads as ADDS-BEYOND regardless of the score's polarity convention.
Conditional skill = weighted-tau( rank(score_aligned) − OLS-fit on rank(primitive[s]), ordinal ).
Bootstrap recomputes the residual within each resample. Verdict per §4:
  CI includes 0 → TAUTOLOG ; CI strictly >0 → ADDS-BEYOND ; CI strictly <0 → SIGN-FLIPPED ;
  kernels disagree → INCONCLUSIVE-kernel-disagree.

SLICE and dpath ran on a fixed 3,000-cell seed-42 subsample (their native algorithms
are infeasible at 39.5k); their rows are computed on that subsample (n disclosed).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
from multiprocessing import Pool
import numpy as np
from scipy.stats import weightedtau, kendalltau, rankdata, spearmanr

# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# scratch directory, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root, repo_root  # noqa: E402
CODE = repo_root()   # this repository (code)
VAULT = data_root()  # run inputs and outputs (data)
sys.path.insert(0, str(CODE / "own_baseline" / "atlas_run"))
from wdm_tau import wdm_tau_b_bit

PREP = VAULT / "track2/data/prepared"
RES = VAULT / "track2/results"
OUT = RES / "gate2_conditional_skill.json"
SEED = 42
N_BOOT_WT = 300      # weightedtau bootstrap (expensive kernel)
N_BOOT_KT = 1000     # kendalltau bootstrap


def scipy_wtau(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(getattr(weightedtau(x[m], y[m]), "statistic", np.nan))


def kang_taub(x, y):
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 3:
        return float("nan")
    return float(kendalltau(x[m], y[m]).statistic)


def rank_resid_multi(s, covs):
    if not covs:
        return s
    y = rankdata(s)
    Xd = np.column_stack([rankdata(c) for c in covs] + [np.ones(len(y))])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    return y - Xd @ beta


def _verdict(lo, hi):
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "INCONCLUSIVE"
    if lo <= 0 <= hi:
        return "TAUTOLOG"
    if lo > 0:
        return "ADDS-BEYOND"
    if hi < 0:
        return "SIGN-FLIPPED"
    return "INCONCLUSIVE"


def _unit(task):
    """task = (label, s_aligned, cov_list, o). Returns (label, both-kernel result)."""
    label, s, covs, o = task
    out = {}
    for kname, kfun, nb in [("scipy_weightedtau", scipy_wtau, N_BOOT_WT),
                            ("kang_wdm_taub", kang_taub, N_BOOT_KT)]:
        real = kfun(rank_resid_multi(s, covs), o)
        rng = np.random.default_rng(SEED)
        n = len(o)
        b = np.empty(nb)
        for j in range(nb):
            i = rng.integers(0, n, n)
            b[j] = kfun(rank_resid_multi(s[i], [c[i] for c in covs]), o[i])
        lo, hi = np.nanquantile(b, [0.025, 0.975])
        out[kname] = {"tau": float(real), "CI95": [float(lo), float(hi)],
                      "n_boot": nb, "verdict": _verdict(lo, hi)}
    ag = out["scipy_weightedtau"]["verdict"] == out["kang_wdm_taub"]["verdict"]
    out["kernels_agree"] = ag
    out["combined_verdict"] = out["scipy_weightedtau"]["verdict"] if ag else "INCONCLUSIVE-kernel-disagree"
    return label, out


def align(score, ordinal):
    return (-1.0 if spearmanr(score, ordinal, nan_policy="omit").statistic < 0 else 1.0) * score


def main():
    p = np.load(PREP / "primitives.npz")
    prims = {"gene_count": p["gene_count"], "PCC(x,degree)": p["pcc_string_v12"],
             "Shannon_H": p["shannon_H"]}
    rk, rg = p["rank_kimmel"].astype(float), p["rank_gate1alt"].astype(float)
    import csv as _csv
    loglib = np.log10(np.array([float(r["lib_size"]) for r in
                                _csv.DictReader(open(PREP / "cells.tsv"), delimiter="\t")]))

    # full-length scores
    full = dict(prims)
    prov = {}
    full["CytoTRACE_v1"] = np.load(RES / "cytotrace_scores.npz")["ct_v1"]
    prov["CytoTRACE_v1"] = "full R-faithful port (MVG→GCS→NNLS→Markov diffusion), verbatim Gulati 2020 core fns"
    if (RES / "scent_scores.npz").exists():
        sc = np.load(RES / "scent_scores.npz")
        full["SR"], full["CCAT"] = sc["sr"], sc["ccat"]
        prov["SR"] = prov["CCAT"] = "vectorized verbatim-SCENT algorithm; == SCENT R on 500-cell subsample (Spearman 1.000, max|Δ|<1e-13)"
    if (RES / "origins_scores.npz").exists():
        full["ORIGINS"] = np.load(RES / "origins_scores.npz")["origins"]
        prov["ORIGINS"] = "vectorized x^T A x on ORIGINS native diff-network; == ORIGINS::activity R on 500-cell subsample (Spearman 1.000)"

    # subsample scores (SLICE/dpath): (score_vec, cell_idx)
    subs = {}
    for nm, fn, key, note in [("SLICE", "slice_scores.npz", "slice", "real SLICE::getEntropy (GO kappa-sim, bootstrap scEntropy)"),
                              ("dpath", "dpath_scores.npz", "dpath", "real dpath wp-NMF metagene entropy (K=5), entropy2(t(V))")]:
        f = RES / fn
        if f.exists():
            d = np.load(f)
            subs[nm] = (d[key], d["cell_idx"]); prov[nm] = note

    pairs = [("CytoTRACE_v1", "gene_count"), ("SR", "PCC(x,degree)"), ("SR", "CCAT"),
             ("CCAT", "PCC(x,degree)"), ("ORIGINS", "gene_count"), ("ORIGINS", "PCC(x,degree)"),
             ("SLICE", "Shannon_H"), ("dpath", "Shannon_H")]

    # ---- build bootstrap tasks (primary conditional pairs + depth-robustness) ----
    tasks, meta = [], []
    for sn, pn in pairs:
        if sn in full and pn in full:
            s = align(full[sn], rk)
            tasks.append((f"{sn} | {pn}", s, [full[pn]], rk)); meta.append(("cond", f"{sn} | {pn}"))
        elif sn in subs and pn in full:
            sv, ci = subs[sn]
            s = align(sv, rk[ci])
            tasks.append((f"{sn} | {pn} [n={len(ci)} subsample]", s, [full[pn][ci]], rk[ci]))
            meta.append(("cond", f"{sn} | {pn}"))
        else:
            meta.append(("infra", f"{sn} | {pn}"))
    robust = [("CytoTRACE_v1", "gene_count"), ("SR", "PCC(x,degree)"),
              ("SR", "CCAT"), ("CCAT", "PCC(x,degree)"),
              ("ORIGINS", "gene_count"), ("ORIGINS", "PCC(x,degree)")]
    for sn, pn in robust:
        if sn in full and pn in full:
            s = align(full[sn], rk)
            tasks.append((f"{sn} | {pn} + log10(lib)", s, [full[pn], loglib], rk))
            meta.append(("robust", f"{sn} | {pn} + log10(lib)"))

    print(f"running {len(tasks)} bootstrap units in parallel ...")
    with Pool(8) as pool:
        results = dict(pool.map(_unit, tasks))

    out = {"n_cells_full": int(len(rk)), "seed": SEED,
           "n_boot": {"scipy_weightedtau": N_BOOT_WT, "kang_wdm_taub": N_BOOT_KT},
           "kernels": ["scipy.stats.weightedtau",
                       "Kang wdm uniform == Kendall tau-b (scipy.stats.kendalltau)"],
           "score_provenance": prov, "conditional_skill": {},
           "depth_robustness_cond_given_primitive_AND_loglib": {},
           "conditional_skill_shield_sensitivity_pointest": {},
           "marginal_skill": {}, "spearman_score_vs_primitive": {}}

    for kind, label in meta:
        if kind == "infra":
            out["conditional_skill"][label] = {"status": "INCONCLUSIVE-infra (real implementation not available)"}
        elif kind == "cond":
            key = [t[0] for t in tasks if t[0].startswith(label)][0]
            out["conditional_skill"][label] = {**results[key], "unit": key}
    for label in [m[1] for m in meta if m[0] == "robust"]:
        out["depth_robustness_cond_given_primitive_AND_loglib"][label] = results[label]

    # spearman(score, primitive)
    for sn, pn in pairs:
        if sn in full and pn in full:
            out["spearman_score_vs_primitive"][f"{sn} | {pn}"] = float(spearmanr(full[sn], full[pn]).statistic)
        elif sn in subs and pn in full:
            sv, ci = subs[sn]
            out["spearman_score_vs_primitive"][f"{sn} | {pn}"] = float(spearmanr(sv, full[pn][ci]).statistic)

    # marginal skills (point estimates both kernels + Spearman); shield point-estimates
    for nm, v in full.items():
        s = align(v, rk)
        out["marginal_skill"][nm] = {"scipy_weightedtau": scipy_wtau(v, rk),
                                     "kang_wdm_taub": kang_taub(v, rk),
                                     "spearman_vs_ordinal": float(spearmanr(v, rk).statistic)}
    for nm, (sv, ci) in subs.items():
        out["marginal_skill"][nm] = {"scipy_weightedtau": scipy_wtau(sv, rk[ci]),
                                     "kang_wdm_taub": kang_taub(sv, rk[ci]),
                                     "spearman_vs_ordinal": float(spearmanr(sv, rk[ci]).statistic),
                                     "n": int(len(ci))}
    # shield sensitivity: point-estimate conditional skill on gate1alt ordinal
    for sn, pn in pairs:
        if sn in full and pn in full:
            s = align(full[sn], rg)
            r = rank_resid_multi(s, [full[pn]])
            out["conditional_skill_shield_sensitivity_pointest"][f"{sn} | {pn}"] = {
                "scipy_weightedtau": scipy_wtau(r, rg), "kang_wdm_taub": kang_taub(r, rg)}
        elif sn in subs and pn in full:
            sv, ci = subs[sn]
            s = align(sv, rg[ci]); r = rank_resid_multi(s, [full[pn][ci]])
            out["conditional_skill_shield_sensitivity_pointest"][f"{sn} | {pn}"] = {
                "scipy_weightedtau": scipy_wtau(r, rg[ci]), "kang_wdm_taub": kang_taub(r, rg[ci])}

    # K2 equivalence spot-check on the full CT residual
    s = align(full["CytoTRACE_v1"], rk); resid = rank_resid_multi(s, [full["gene_count"]])
    out["K2_equivalence_check"] = {"kendalltau": kang_taub(resid, rk),
                                   "wdm_tau_b_bit_uniform": float(wdm_tau_b_bit(resid, rk, np.ones(len(rk)))),
                                   "abs_diff": abs(kang_taub(resid, rk) - float(wdm_tau_b_bit(resid, rk, np.ones(len(rk)))))}

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}\n")
    print("=== CONDITIONAL SKILL (primary Kimmel ordinal) ===")
    for k, r in out["conditional_skill"].items():
        if "status" in r:
            print(f"  {k:26s} {r['status']}"); continue
        a, b = r["scipy_weightedtau"], r["kang_wdm_taub"]
        print(f"  {k:26s} scipy τ={a['tau']:+.4f} CI[{a['CI95'][0]:+.3f},{a['CI95'][1]:+.3f}] {a['verdict']:11s}"
              f" | kang τ={b['tau']:+.4f} CI[{b['CI95'][0]:+.3f},{b['CI95'][1]:+.3f}] {b['verdict']:11s} => {r['combined_verdict']}")
    print("\n=== DEPTH ROBUSTNESS (+log10 lib) ===")
    for k, r in out["depth_robustness_cond_given_primitive_AND_loglib"].items():
        print(f"  {k:34s} => {r['combined_verdict']}")
    print(f"\nK2 equivalence |Δ|={out['K2_equivalence_check']['abs_diff']:.2e}")


if __name__ == "__main__":
    main()
