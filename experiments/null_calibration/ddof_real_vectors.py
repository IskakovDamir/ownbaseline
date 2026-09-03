#!/usr/bin/env python3
"""
The ddof deliverable, on the runs' OWN score vectors.

Pre-registration section 5 asks for the shift in the reported statistic for
every run recorded in 04-experiments/ with n < 5000, and says: "Where the run's
own score vectors are recoverable from 06-code/tautology-diagnostic, the shift
is computed on the real vectors instead, and that is stated per row."

ddof_shift.py does the matched-synthetic-data half. This file does the real-
vector half. It measures three residual constructions on the same aligned score
and the same primitive, and reports the tau each produces under both kernels:

    lstsq    rank_resid_multi(s, [p])                     the shipped estimator
    ddof1    beta = np.cov(r_s, r_p)[0,1] / np.var(r_p)   the defective form
    ddof0    beta = np.cov(r_s, r_p, ddof=0)[0,1] / var   what six of the seven
                                                          call sites actually do

Both cov/var forms are transcribed verbatim from the call sites; they are not
tidied and not corrected. Nothing here is written back into any run.

Score vectors live in the research vault, not in this repository (the released
package ships the two full-atlas score families only). Point
$TAUTOLOGY_DIAGNOSTIC at a checkout of 06-code/tautology-diagnostic to run it.

    python3 experiments/null_calibration/ddof_real_vectors.py --out real.csv
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, rankdata, spearmanr, weightedtau

sys.path.insert(0, str(next(p for p in Path(__file__).resolve().parents
                            if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.conditional_skill import align, rank_resid_multi  # noqa: E402
from own_baseline.paths import data_root  # noqa: E402

# Where the frozen per-cell score vectors live. Set $TAUTOLOGY_DIAGNOSTIC to the
# directory holding track2/, fix3/ and w4/. Defaults to the repository's own data
# root (see own_baseline/paths.py), so this module imports on any machine.
DIAG = Path(os.environ.get("TAUTOLOGY_DIAGNOSTIC", data_root()))

def _rel(path):
    """Provenance without the machine. Records which vector was read, relative to
    $TAUTOLOGY_DIAGNOSTIC, so the row identifies its source on any checkout and
    carries no absolute home path."""
    try:
        return str(Path(path).resolve().relative_to(Path(DIAG).resolve()))
    except ValueError:
        return Path(path).name



def rank_residual_ddof1(score, primitive):
    """Verbatim from track4/code/gate2b_ownbaseline.py. Do not tidy it."""
    r_s = rankdata(score)
    r_p = rankdata(primitive)
    beta = np.cov(r_s, r_p)[0, 1] / np.var(r_p) if np.var(r_p) > 0 else 0.0
    return r_s - beta * r_p


def rank_residual_ddof0(score, primitive):
    """Verbatim from track2/code/gate2_run.py (and five siblings)."""
    r_s = rankdata(score)
    r_p = rankdata(primitive)
    var_p = np.var(r_p)
    beta = np.cov(r_s, r_p, ddof=0)[0, 1] / var_p if var_p > 0 else 0.0
    return r_s - beta * r_p


def taus(resid, ordinal):
    return (float(weightedtau(resid, ordinal).statistic),
            float(kendalltau(resid, ordinal).statistic))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ddof_real_vectors.csv")
    args = ap.parse_args()

    res_repo = data_root() / "track2/results"
    res_vault = DIAG / "track2/results"

    # The primitives were looked for under data_root() only, so the docstring's
    # instruction -- point $TAUTOLOGY_DIAGNOSTIC at a checkout of
    # 06-code/tautology-diagnostic -- did not on its own make this runnable.
    # Search both roots, in the same order the score vectors are searched.
    prep = next((r / "track2/data/prepared" for r in (DIAG, data_root())
                 if (r / "track2/data/prepared/primitives.npz").is_file()), None)
    if prep is None:
        raise SystemExit(
            "primitives.npz not found under $TAUTOLOGY_DIAGNOSTIC or "
            "$OWNBASELINE_DATA_ROOT at track2/data/prepared/")

    p = np.load(prep / "primitives.npz")
    prims = {"gene_count": p["gene_count"],
             "PCC(x,degree)": p["pcc_string_v12"],
             "Shannon_H": p["shannon_H"]}
    ordinal_full = p["rank_kimmel"].astype(float)

    def load(fname, key):
        for root in (res_vault, res_repo):
            f = root / fname
            if f.exists():
                d = np.load(f)
                idx = d["cell_idx"] if "cell_idx" in d.files else None
                return d[key], idx, str(f)
        return None, None, ""

    # (label, npz, key, primitive)  -- the n<5000 rows first
    PAIRS = [
        ("SLICE | Shannon_H",         "slice_scores.npz",     "slice",  "Shannon_H"),
        ("dpath | Shannon_H",         "dpath_scores.npz",     "dpath",  "Shannon_H"),
        ("NCG | PCC(x,degree)",       "ncg_scores.npz",       "ncg",    "PCC(x,degree)"),
        ("CytoTRACE_v1 | gene_count", "cytotrace_scores.npz", "ct_v1",  "gene_count"),
        ("SR | PCC(x,degree)",        "scent_scores.npz",     "sr",     "PCC(x,degree)"),
        ("CCAT | PCC(x,degree)",      "scent_scores.npz",     "ccat",   "PCC(x,degree)"),
        ("ORIGINS | gene_count",      "origins_scores.npz",   "origins", "gene_count"),
        ("ORIGINS | PCC(x,degree)",   "origins_scores.npz",   "origins", "PCC(x,degree)"),
    ]

    rows = []
    for label, fname, key, pname in PAIRS:
        score, idx, src = load(fname, key)
        if score is None:
            print(f"  {label:28s} SKIPPED (no {fname} under {res_vault} or {res_repo})")
            continue
        prim = prims[pname]
        if idx is not None:
            prim, o = prim[idx], ordinal_full[idx]
        else:
            o = ordinal_full
        n = len(score)

        s = align(score, o)
        r_ls = rank_resid_multi(s, [prim])
        r_d1 = rank_residual_ddof1(s, prim)
        r_d0 = rank_residual_ddof0(s, prim)

        w_ls, k_ls = taus(r_ls, o)
        w_d1, k_d1 = taus(r_d1, o)
        w_d0, k_d0 = taus(r_d0, o)

        rho = float(spearmanr(score, prim).statistic)
        for kern, ls, d1, d0 in (("weightedtau", w_ls, w_d1, w_d0),
                                 ("kendalltau", k_ls, k_d1, k_d0)):
            rows.append({
                "label": label, "n": n, "in_section5_scope": int(n < 5000),
                "rho_score_primitive": rho, "kernel": kern,
                "slope_inflation": n / (n - 1),
                "tau_lstsq": ls, "tau_covvar_ddof1": d1, "tau_covvar_ddof0": d0,
                "shift_ddof1_minus_ddof0": d1 - d0,
                "shift_ddof1_minus_lstsq": d1 - ls,
                "shift_ddof0_minus_lstsq": d0 - ls,
                "source_npz": _rel(src),
            })
        print(f"  {label:28s} n={n:6d} rho={rho:+.4f}  "
              f"wtau lstsq {w_ls:+.6f} ddof1 {w_d1:+.6f} (Δ {w_d1-w_ls:+.2e})  "
              f"ktau lstsq {k_ls:+.6f} ddof1 {k_d1:+.6f} (Δ {k_d1-k_ls:+.2e})")

    if not rows:
        print("no vectors found; nothing written")
        return
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
