"""
h6_partial_rho.py
=================
H6 (difficulty-controlled separation): partial-rho analysis per prereg
2026-07-17-hod6-sign-rule-and-separation-prereg.md.

Computes Spearman partial correlation
    partial-rho(tau_SR_d, tau_gc_d | tau_CT2_d)
    partial-rho(tau_CCAT_d, tau_gc_d | tau_CT2_d)
under both kernels (scipy tau values + Kang wdm tau values) on:
  - Primary: n = 23 (full atlas)
  - Robustness: n = 7 (Test cohort only; CT2 did not train on these)

Implementation of partial-rho (standard):
  1. Rank-transform tau_score, tau_gc, tau_CT2 (Spearman == Pearson on ranks).
  2. Residualize tau_score_ranks on tau_CT2_ranks via OLS (intercept).
  3. Residualize tau_gc_ranks on tau_CT2_ranks via OLS (intercept).
  4. Pearson-correlate the residuals -> partial-rho.

Confidence intervals: 10 000 non-parametric row-bootstrap samples,
percentile method (2.5%/97.5%). Seed 42.

p-values: two-sided permutation test on partial-rho, 10 000 permutations of
tau_score labels (holding tau_gc and tau_CT2 fixed), seed 42.

Locked decision rule (both kernels required):
  PASS: partial-rho >= 0.40 under BOTH kernels
  FAIL: partial-rho <  0.20 under EITHER kernel
  else: inconclusive
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

DATA_PATH = Path(__file__).parent / "per_dataset_results.json"
OUT_PATH = Path(__file__).parent / "h6_partial_rho_results.json"

SEED = 42
N_BOOT = 10_000
N_PERM = 10_000


def _rank(v: np.ndarray) -> np.ndarray:
    """Average-tie Spearman ranks."""
    return stats.rankdata(v, method="average")


def _residualize(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """OLS residuals of y on [1, x]."""
    X = np.column_stack([np.ones_like(x), x])
    # Solve X @ beta = y in least-squares sense
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ beta


def partial_spearman(score: np.ndarray, gc: np.ndarray, ct2: np.ndarray) -> float:
    """Spearman partial correlation: partial-rho(score, gc | ct2)."""
    rs = _rank(score)
    rg = _rank(gc)
    rc = _rank(ct2)
    e_sg = _residualize(rs, rc)
    e_gg = _residualize(rg, rc)
    # Pearson on residuals
    if np.std(e_sg) == 0 or np.std(e_gg) == 0:
        return float("nan")
    return float(np.corrcoef(e_sg, e_gg)[0, 1])


def bootstrap_ci(score, gc, ct2, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Percentile bootstrap CI over rows."""
    rng = np.random.default_rng(seed)
    n = len(score)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        # Guard against degenerate resamples with fewer than 3 unique rows
        # (Spearman residualization still runs; only fails if constant vector.)
        s_b = score[idx]
        g_b = gc[idx]
        c_b = ct2[idx]
        try:
            vals[b] = partial_spearman(s_b, g_b, c_b)
        except Exception:
            vals[b] = np.nan
    vals = vals[~np.isnan(vals)]
    lo = float(np.percentile(vals, 100 * alpha / 2))
    hi = float(np.percentile(vals, 100 * (1 - alpha / 2)))
    return lo, hi, vals


def permutation_pvalue(score, gc, ct2, observed, n_perm=N_PERM, seed=SEED):
    """
    Two-sided permutation p-value: permute score labels, holding (gc, ct2) fixed,
    recompute partial-rho, count |perm| >= |observed|. Adds +1 to numerator and
    denominator (standard convention).
    """
    rng = np.random.default_rng(seed + 1)  # different stream from bootstrap
    n = len(score)
    perm_vals = np.empty(n_perm)
    for p in range(n_perm):
        idx = rng.permutation(n)
        perm_vals[p] = partial_spearman(score[idx], gc, ct2)
    perm_vals = perm_vals[~np.isnan(perm_vals)]
    count = int(np.sum(np.abs(perm_vals) >= abs(observed)))
    p = (count + 1) / (len(perm_vals) + 1)
    return float(p)


def analyze(rows, kernel_key, score_key, gc_key="gene_counts",
            ct2_key="CytoTRACE 2 potency score"):
    """Run partial-rho for one (kernel, score) combination."""
    score = np.array([r[kernel_key][score_key] for r in rows], dtype=np.float64)
    gc = np.array([r[kernel_key][gc_key] for r in rows], dtype=np.float64)
    ct2 = np.array([r[kernel_key][ct2_key] for r in rows], dtype=np.float64)

    obs = partial_spearman(score, gc, ct2)
    ci_lo, ci_hi, _ = bootstrap_ci(score, gc, ct2)
    p = permutation_pvalue(score, gc, ct2, obs)

    return {
        "n": len(rows),
        "partial_rho": obs,
        "ci95_lo": ci_lo,
        "ci95_hi": ci_hi,
        "p_two_sided_perm": p,
    }


def verdict(rho_scipy_sr, rho_kang_sr, rho_scipy_ccat, rho_kang_ccat):
    """Apply locked decision rule per prereg §Правило решения."""
    def one(rho_s, rho_k, name):
        if rho_s >= 0.40 and rho_k >= 0.40:
            return f"{name}: PASS (both kernels >= 0.40)"
        if rho_s < 0.20 or rho_k < 0.20:
            return f"{name}: FAIL (at least one kernel < 0.20)"
        return f"{name}: INCONCLUSIVE (in (0.20, 0.40) interval under at least one kernel)"
    sr_v = one(rho_scipy_sr, rho_kang_sr, "SR")
    ccat_v = one(rho_scipy_ccat, rho_kang_ccat, "CCAT")
    return sr_v, ccat_v


def main():
    with open(DATA_PATH) as f:
        d = json.load(f)
    rows_all = d["per_dataset"]
    rows_test = [r for r in rows_all if r["cohort"] == "Test"]

    assert len(rows_all) == 23, f"expected 23 atlas rows, got {len(rows_all)}"
    assert len(rows_test) == 7, f"expected 7 Test rows, got {len(rows_test)}"

    results = {"n23": {}, "n7_test": {}}

    for label, rows in (("n23", rows_all), ("n7_test", rows_test)):
        for kernel in ("tau_scipy", "tau_kang"):
            for score in ("SCENT (SR)", "SCENT (CCAT)"):
                key = f"{kernel}__{score}"
                results[label][key] = analyze(rows, kernel, score)

    # Verdict
    sr_scipy = results["n23"]["tau_scipy__SCENT (SR)"]["partial_rho"]
    sr_kang = results["n23"]["tau_kang__SCENT (SR)"]["partial_rho"]
    ccat_scipy = results["n23"]["tau_scipy__SCENT (CCAT)"]["partial_rho"]
    ccat_kang = results["n23"]["tau_kang__SCENT (CCAT)"]["partial_rho"]
    sr_v, ccat_v = verdict(sr_scipy, sr_kang, ccat_scipy, ccat_kang)
    results["verdict_locked_rule"] = {
        "SR": sr_v,
        "CCAT": ccat_v,
        "thresholds": "PASS both>=0.40 | FAIL either<0.20 | else inconclusive",
    }

    # Test cohort label — n=7 is underpowered; report as-is
    results["n7_test_note"] = (
        "n=7 partial correlation is severely underpowered; per prereg, "
        "this is reported as robustness, NEVER as confirmation of n=23."
    )

    # Metadata
    results["meta"] = {
        "seed": SEED,
        "n_bootstrap": N_BOOT,
        "n_permutations": N_PERM,
        "data_source": str(DATA_PATH.name),
        "atlas_generated_at": d["provenance"]["generated_at"],
        "wdm_tau_hash": d["provenance"]["wdm_tau_hash"],
        "script_hash_upstream": d["provenance"]["script_hash"],
        "method": "Spearman partial-rho: rank both variables, residualize on ranked ct2 via OLS (intercept), Pearson of residuals",
    }

    OUT_PATH.write_text(json.dumps(results, indent=2, default=str))
    print(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
