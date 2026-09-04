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

# repo config: wdm_tau lives in this package's atlas_run/ (was an absolute vault path)
sys.path.insert(0, str(Path(__file__).resolve().parent / "atlas_run"))
from wdm_tau import wdm_tau_b_bit

# experiment I/O roots (only used when this file is run as the track-2 decider script,
# not on import); point at the vault where the runs happened.
# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# agent-session scratchpad, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root  # noqa: E402
VAULT = data_root()
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
    """
    task = (label, s_aligned, cov_list, o). Returns (label, both-kernel result).

    The 4-tuple form is what the track-2 decider run passed, and Pool.map still
    passes it. A 5-tuple (…, n_boot) overrides the resample counts; omitting it
    keeps (N_BOOT_WT, N_BOOT_KT) and therefore the published numbers.
    """
    if len(task) == 5:
        label, s, covs, o, n_boot = task
    else:
        label, s, covs, o = task
        n_boot = (N_BOOT_WT, N_BOOT_KT)
    out = {}
    for kname, kfun, nb in [("scipy_weightedtau", scipy_wtau, n_boot[0]),
                            ("kang_wdm_taub", kang_taub, n_boot[1])]:
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


# ============================================================================
# PRACTITIONER API
# ============================================================================
# Everything above is the track-2 decider exactly as it ran. The function below
# is a wrapper: it reuses align(), rank_resid_multi(), scipy_wtau(), kang_taub()
# and _unit() unchanged, and adds no estimator of its own. It exists because
# what the paper describes -- conditional skill against a score's own primitive
# -- was reachable only by running this file as a script against one specific
# prepared dataset. run_own_baseline() in own_baseline.py, the function a reader
# would actually find, computes something weaker: a marginal gap against raw
# gene counts with no residualization.
#
# NOTE ON THE OTHER RESIDUAL IMPLEMENTATION. This repository contains two.
# This one, rank_resid_multi() above, is the track-2 estimator: it rank-
# transforms score and covariates, fits by least squares WITH an intercept, and
# accepts any number of covariates. The other appears under TWO names, so a
# grep for one of them misses half the call sites: rank_residual() in
# experiments/track2/code/gate2_run.py and experiments/track4/code/
# {gate2b_ownbaseline, gate2b_origins_c1c2, gate2c_resolve}.py, and
# _rank_residual() -- leading underscore -- in
# experiments/w4/scripts/w4_gate2_run.py, experiments/cytotrace_v1/ct_full_run.py
# and experiments/stemsc/stemsc_run.py. It differs in three ways:
#
#   1. Its slope is np.cov(r_s, r_p)[0, 1] / np.var(r_p). np.cov defaults to
#      ddof=1 and np.var to ddof=0, so its slope is the least-squares slope
#      times n/(n-1) -- 1.000025 at n = 39,505, but 1.0204 at n = 50.
#   2. It fits no intercept. Rank-based tau is invariant to a constant offset,
#      so this difference alone does not move a number.
#   3. It takes exactly one covariate. The joint residualization on four
#      primitives at once is only expressible with rank_resid_multi().
#
#   Measured consequence of (1): |tau_A - tau_B| reaches 3.7e-2 at n = 106 and
#   falls to 6.6e-07 at n = 39,505. It is negligible at atlas scale and is not
#   negligible on the 3,000-cell subsamples or smaller.
#
#   The two are NOT merged here, and this wrapper uses only rank_resid_multi().
#   Deciding which is correct, and re-running whatever depends on the other, is
#   a numerical decision for the authors, not a refactor.

def _as_primitive_dict(primitives):
    """Accept a dict, a single vector, or a sequence of vectors."""
    if isinstance(primitives, dict):
        return {str(k): np.asarray(v, dtype=float) for k, v in primitives.items()}
    arr = np.asarray(primitives, dtype=float)
    if arr.ndim == 1:
        return {"primitive": arr}
    return {f"primitive_{i}": np.asarray(v, dtype=float) for i, v in enumerate(arr)}


def conditional_skill_report(
    score,
    ordinal,
    primitives,
    *,
    score_name="score",
    n_boot=None,
    joint=True,
    verbose=True,
):
    """
    Does `score` order cells by `ordinal` beyond what `primitives` already do?

    This is the test the paper reports. It answers a different question from
    run_own_baseline(), which compares two marginal correlations and can call a
    score substantive when it is the primitive plus noise.

    Parameters
    ----------
    score : (n,) array
        Per-cell score. Polarity does not matter; it is sign-aligned to the
        ordinal before residualization, so a score where "low = more potent"
        needs no manual negation.
    ordinal : (n,) array
        Score-independent potency ordinal, higher = more potent. It must be
        fixed independently of expression -- a developmental stage assigned by
        microscopy before dissociation, not a pseudotime.
    primitives : dict[str, array] | array | sequence of arrays
        The low-order statistic(s) the score is claimed to approximate. Use the
        one the score's own authors name. own_baseline.potency_metrics computes
        the three used in the paper: gene count, Pearson(x, PPI degree),
        Shannon entropy.
    n_boot : (int, int), optional
        Bootstrap resamples for (weightedtau, kendalltau). Defaults to the
        published (300, 1000). Lower them for a quick look; the intervals widen.
    joint : bool
        Also residualize on every primitive at once. Only meaningful with more
        than one primitive.

    Returns
    -------
    dict with, for each primitive name and for "JOINT":

        marginal_skill      tau(score, ordinal) and tau(primitive, ordinal),
                            both kernels -- the two numbers the field usually
                            reports.
        marginal_gap        tau(score) - tau(primitive), both kernels. This is
                            the quantity run_own_baseline() calls Delta.
        direction_check     the sign of the primitive's own association with
                            the ordinal, and a flag when it is negative. A
                            primitive that orders cells the wrong way makes the
                            marginal gap meaningless: beating a reversed
                            baseline is a sign correction, not biology. In
                            sorted haematopoietic progenitors the gene-count
                            primitive sits at AUROC 0.378, and CytoTRACE
                            "beats" it by 0.141.
        conditional_skill   tau(residual, ordinal) with a 95% bootstrap
                            interval and a verdict, under both kernels, plus
                            whether the kernels agree. This is the answer.

    The verdict vocabulary is the decider's, unchanged:
        TAUTOLOG      interval contains 0 -- no skill beyond the primitive
        ADDS-BEYOND   interval strictly positive
        SIGN-FLIPPED  interval strictly negative
        INCONCLUSIVE-kernel-disagree

    Reading it: compare the value against the estimator's own measured null
    floor at that value's own sample size, kernel and score-primitive rank
    correlation. `own_baseline.floors.floor(n, rho, kernel, k_covariates)`
    returns it from the shipped 200-seed grid, and the `ownbaseline` CLI prints
    it beside every row.

    Do NOT use the older rule, which asked for the two kernels to agree, the
    interval to exclude zero, and the value to survive a depth control. It was
    retired because it was measured on synthetic data whose true conditional
    skill is exactly zero and it fired in 40 of 40 replicates at the cells where
    CytoTRACE, signalling entropy and ORIGINS sit. A bootstrap interval narrows
    as one over the square root of n while the floor is close to flat in n, so
    at atlas scale an interval excluding zero is not evidence of anything: at
    n = 39,505 the tau_b floor sits three to four interval half-widths above
    zero. The kernel-agreement leg could not do its job either, because
    weightedtau returned a positive verdict on all 720 null replicates run.

    One more thing the interval will not tell you. When rank(score) equals
    rank(primitive) the OLS fit is exact and this function returns the rounding
    error of the subtraction, on the order of eps times the largest rank. That
    debris is monotone in the primitive rank, so where the primitive predicts
    the ordinal it inherits that association and both kernels score it, reaching
    0.87 in magnitude on a twelve-level staged ordinal, with a sign that changes
    with the seed and a size that changes with the machine: across sixteen
    cells, aarch64 with numpy 2.2 puts all of them between 0.71 and 0.87, and
    x86_64 with numpy 2.4 puts fourteen there and collapses two toward zero. Check the residual's
    magnitude against eps * n before reading any value near rho = 1;
    `own_baseline.cli.residual_scale` does this and the CLI refuses to report.

    Rows with a non-finite score, ordinal or primitive are dropped jointly
    before anything is computed; `n` in the result is what remained.
    """
    score = np.asarray(score, dtype=float)
    ordinal = np.asarray(ordinal, dtype=float)
    prims = _as_primitive_dict(primitives)

    n_in = len(score)
    for name, v in prims.items():
        if len(v) != n_in:
            raise ValueError(
                f"primitive {name!r} has length {len(v)}, score has {n_in}")
    if len(ordinal) != n_in:
        raise ValueError(
            f"ordinal has length {len(ordinal)}, score has {n_in}")

    mask = np.isfinite(score) & np.isfinite(ordinal)
    for v in prims.values():
        mask &= np.isfinite(v)
    if mask.sum() < 3:
        raise ValueError(
            f"only {int(mask.sum())} rows are finite across score, ordinal and "
            f"all primitives; need at least 3")
    s, o = score[mask], ordinal[mask]
    prims = {k: v[mask] for k, v in prims.items()}

    s_aligned = align(s, o)
    flipped = bool(np.any(s_aligned != s))

    tau_score = {"scipy_weightedtau": scipy_wtau(s_aligned, o),
                 "kang_wdm_taub": kang_taub(s_aligned, o)}

    units, out = [], {}
    for name, p in prims.items():
        units.append((name, s_aligned, [p], o))
    if joint and len(prims) > 1:
        units.append(("JOINT", s_aligned, list(prims.values()), o))

    if n_boot is not None:
        units = [(*u, tuple(n_boot)) for u in units]

    results = dict(_unit(u) for u in units)

    for name in list(prims) + (["JOINT"] if joint and len(prims) > 1 else []):
        if name == "JOINT":
            tau_prim = {k: None for k in tau_score}
            direction = {"note": "no single direction for a joint fit"}
        else:
            p = prims[name]
            tau_prim = {"scipy_weightedtau": scipy_wtau(p, o),
                        "kang_wdm_taub": kang_taub(p, o)}
            rho = float(spearmanr(p, o, nan_policy="omit").statistic)
            below = rho < 0
            direction = {
                "spearman_primitive_vs_ordinal": rho,
                "primitive_orders_below_chance": below,
                "note": (
                    "the primitive orders cells OPPOSITE to the ordinal, so a "
                    "positive marginal gap is a sign correction, not evidence "
                    "of added biology; read the conditional skill instead"
                ) if below else "primitive orders in the same direction as the ordinal",
            }
        out[name] = {
            "marginal_skill": {f"tau({score_name})": tau_score,
                               f"tau({name})": tau_prim},
            "marginal_gap": {
                k: (None if tau_prim[k] is None else tau_score[k] - tau_prim[k])
                for k in tau_score},
            "direction_check": direction,
            "conditional_skill": results[name],
        }

    report = {
        "score_name": score_name,
        "n_input": int(n_in),
        "n_used": int(mask.sum()),
        "n_dropped_nonfinite": int(n_in - mask.sum()),
        "score_sign_flipped_to_align": flipped,
        "seed": SEED,
        "kernels": ["scipy.stats.weightedtau",
                    "Kang wdm uniform == Kendall tau-b (scipy.stats.kendalltau)"],
        "residual": "rank_resid_multi (track-2 estimator: rank, OLS with "
                    "intercept, any number of covariates)",
        "by_primitive": out,
    }

    if verbose:
        _print_report(report)
    return report


def _print_report(r):
    dropped = r["n_dropped_nonfinite"]
    tail = f" (dropped {dropped} non-finite)" if dropped else ""
    print(f"[conditional_skill] score={r['score_name']} n={r['n_used']}{tail}")
    if r["score_sign_flipped_to_align"]:
        print("[conditional_skill] score was negated to align with the ordinal")
    for name, d in r["by_primitive"].items():
        print(f"\n  vs primitive: {name}")
        dc = d["direction_check"]
        if dc.get("primitive_orders_below_chance"):
            print(f"    !! DIRECTION: primitive orders BELOW chance "
                  f"(Spearman {dc['spearman_primitive_vs_ordinal']:+.3f}). "
                  f"The marginal gap below is a sign correction.")
        for k in ("scipy_weightedtau", "kang_wdm_taub"):
            gap = d["marginal_gap"][k]
            cs = d["conditional_skill"][k]
            gs = "     n/a" if gap is None else f"{gap:+8.4f}"
            print(f"    {k:18s} marginal gap {gs}   "
                  f"conditional skill {cs['tau']:+.4f} "
                  f"CI[{cs['CI95'][0]:+.4f},{cs['CI95'][1]:+.4f}]  {cs['verdict']}")
        print(f"    => {d['conditional_skill']['combined_verdict']}")


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
            out["conditional_skill"][label] = {"status": "INCONCLUSIVE-infra (real implementation not available in-session)"}
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
