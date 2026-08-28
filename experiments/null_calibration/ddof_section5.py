#!/usr/bin/env python3
"""
Pre-registration section 5, completed: the ddof deliverable, row by row.

WHAT SECTION 5 ASKS. Identify every run recorded in `04-experiments/` with
n < 5000, and for each quantify the shift in the reported statistic caused by
the cov/var slope's ddof mismatch -- on the run's own score vectors where those
are recoverable from `06-code/tautology-diagnostic`, on matched synthetic data
at that n and that score-primitive correlation otherwise -- and state per row
which of the two was used.

WHAT THE PREMISE TURNS OUT TO BE. Section 5 states that
`rank_residual` / `_rank_residual` computes its slope as
`np.cov(r_s, r_p)[0,1] / np.var(r_p)`, ddof=1 over ddof=0. Enumerating the call
sites shows that ONE of the seven does; the other six pass `ddof=0` to
`np.cov` explicitly. This script records which implementation produced each
row, so the shift is attributed rather than assumed. It measures. It corrects
nothing and writes nothing back into any run.

The three residual constructions, all transcribed verbatim from their call
sites and deliberately not tidied:

    lstsq   rank_resid_multi(s, covs)                     own_baseline/conditional_skill.py
    ddof1   beta = np.cov(r_s, r_p)[0,1] / np.var(r_p)    track4/code/gate2b_ownbaseline.py
    ddof0   beta = np.cov(r_s, r_p, ddof=0)[0,1] / var    track2/code/gate2_run.py + 5 siblings

Score vectors live in the research vault, not in this repository. Point
$TAUTOLOGY_DIAGNOSTIC at a checkout of `06-code/tautology-diagnostic`.

    python3 experiments/null_calibration/ddof_section5.py --out ddof_section5.csv
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ddof_shift import (rank_residual_covvar,          # noqa: E402  ddof1, verbatim
                        rank_residual_covvar_ddof0,    # noqa: E402  ddof0, verbatim
                        draw as draw_matched)          # noqa: E402

DIAG = Path(os.environ.get(
    "TAUTOLOGY_DIAGNOSTIC",
    Path.home() / "damir-research-vault/06-code/tautology-diagnostic"))
N_SYNTH_SEEDS = 200
BASE = 20260814

# ---------------------------------------------------------------------------
# Every run recorded in 04-experiments/ with n < 5000. Established by reading
# the notes, not inferred: the only sub-5000 conditional-skill runs in the
# vault are the 3,000-cell seed-42 subsamples on the two atlases. The w1 arms
# (n = 27 .. 1,229) are Cox C-index comparisons with no rank residualization
# at all, and the track4 / w4 / StemSC atlases are n = 12,354 and n = 14,537;
# both groups are listed in NOT_IN_SCOPE below with the reason.
#
# atlas, label, score npz, key, covariate names, k, the note that records it
ROWS = [
    ("zebrafish", "SLICE | Shannon_H", "track2", "slice_scores.npz", "slice",
     ["Shannon_H"], 1, "2026-07-20-track2-gate2-scoring-results.md"),
    ("zebrafish", "dpath | Shannon_H", "track2", "dpath_scores.npz", "dpath",
     ["Shannon_H"], 1, "2026-07-20-track2-gate2-scoring-results.md"),
    ("zebrafish", "NCG | PCC(x,degree)", "track2", "ncg_scores.npz", "ncg",
     ["PCC(x,degree)"], 1, "2026-07-21-fix12-mce-ncg-and-joint-primitive.md"),
    ("zebrafish", "NCG | PCC + loglib", "track2", "ncg_scores.npz", "ncg",
     ["PCC(x,degree)", "log10_lib"], 2,
     "2026-07-21-fix12-mce-ncg-and-joint-primitive.md"),
    ("zebrafish", "dpath | joint-4", "track2", "dpath_scores.npz", "dpath",
     ["gene_count", "PCC(x,degree)", "Shannon_H", "log10_lib"], 4,
     "2026-07-21-fix12-mce-ncg-and-joint-primitive.md"),
    ("Xenopus", "SLICE | Shannon_H", "fix3", "slice_scores.npz", "slice",
     ["Shannon_H"], 1, "2026-07-21-fix3-gate2-briggs-replicate.md"),
    ("Xenopus", "dpath | Shannon_H", "fix3", "dpath_scores.npz", "dpath",
     ["Shannon_H"], 1, "2026-07-21-fix3-gate2-briggs-replicate.md"),
    ("Xenopus", "dpath | joint-4", "fix3", "dpath_scores.npz", "dpath",
     ["gene_count", "PCC(x,degree)", "Shannon_H", "log10_lib"], 4,
     "2026-07-21-fix3-gate2-briggs-replicate.md"),
]

NOT_IN_SCOPE = [
    ("w1 T1-T5 bulk survival arms", "27 .. 1,229",
     "Cox 5-fold-CV C-index against a library-size baseline. No rank "
     "residualization is performed, so no slope exists to be inflated. "
     "2026-07-18-w1-ownbaseline-results.md"),
    ("track4 Gate 2b/2c, w4 Gate 2, StemSC (C1 GSE117498)", "12,354",
     "above the n < 5000 threshold. 2026-07-18-w4-ownbaseline-results.md, "
     "2026-07-18-stemsc-ownbaseline-results.md, "
     "2026-07-20-track4-gate2b-ownbaseline-results.md"),
    ("track4 Gate 2b/2c, w4 Gate 2, StemSC (C2 GSE125970)", "14,537",
     "above the n < 5000 threshold. Same notes."),
    ("track4 Gate 2b pilot on E-MTAB-9067", "not recorded",
     "the only call site carrying the ddof mismatch "
     "(track4/code/gate2b_ownbaseline.py). Its output "
     "track4/results/L2b_pilot_e_mtab_9067.json is not on disk and no "
     "conditional-skill number from it appears in 04-experiments/: the "
     "Gate 2b note's CT/SR/CCAT rows come from the w4 run and its ORIGINS "
     "rows from gate2b_origins_c1c2.py, both of which pass ddof=0."),
    ("Kang 23-dataset atlas probe", "up to 5,315",
     "marginal tau against the broad-potency ordinal, not a residual. "
     "2026-07-17-hod5-atlas-scale-dossier.md"),
]


def taus(resid, ordinal):
    return (float(weightedtau(resid, ordinal).statistic),
            float(kendalltau(resid, ordinal).statistic))


def load_atlas(which):
    """Primitives, ordinal and log library size for one atlas."""
    base = DIAG / which
    prep = base / ("data/prepared")
    p = np.load(prep / "primitives.npz")
    prims = {"gene_count": p["gene_count"],
             "PCC(x,degree)": p["pcc_string_v12"],
             "Shannon_H": p["shannon_H"]}
    ordinal = (p["rank_kimmel"] if "rank_kimmel" in p.files
               else p["rank_ordinal"]).astype(float)
    with open(prep / "cells.tsv") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    prims["log10_lib"] = np.log10(np.array([float(r["lib_size"]) for r in rows]))
    return prims, ordinal, base / "results"


def matched_synthetic(n, rho, seeds=N_SYNTH_SEEDS):
    """The shift the same row would show on matched synthetic data at (n, rho)."""
    dw, dk = [], []
    for j in range(seeds):
        rng = np.random.default_rng([BASE, 5, n, int(round(abs(rho) * 1000)), j])
        s, p, o = draw_matched(n, abs(rho), rng)
        sa = align(s, o)
        w1, k1 = taus(rank_residual_covvar(sa, p), o)
        w0, k0 = taus(rank_residual_covvar_ddof0(sa, p), o)
        dw.append(w1 - w0)
        dk.append(k1 - k0)
    return (float(np.mean(np.abs(dw))), float(np.max(np.abs(dw))),
            float(np.mean(np.abs(dk))), float(np.max(np.abs(dk))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="ddof_section5.csv")
    ap.add_argument("--synth-seeds", type=int, default=N_SYNTH_SEEDS)
    args = ap.parse_args()

    cache = {}
    out_rows = []
    for atlas, label, which, fname, key, covnames, k, note in ROWS:
        if which not in cache:
            cache[which] = load_atlas(which)
        prims, ordinal_full, resdir = cache[which]

        f = resdir / fname
        if not f.is_file():
            print(f"  {atlas:10s} {label:24s} SKIPPED (no {f})")
            continue
        d = np.load(f)
        score = d[key]
        idx = d["cell_idx"] if "cell_idx" in d.files else None
        covs = [prims[c] for c in covnames]
        if idx is not None:
            covs = [c[idx] for c in covs]
            o = ordinal_full[idx]
        else:
            o = ordinal_full
        n = len(score)

        s = align(score, o)
        w_ls, k_ls = taus(rank_resid_multi(s, covs), o)
        rho = float(spearmanr(score, covs[0]).statistic)

        row = {
            "atlas": atlas, "label": label, "n": n, "k_covariates": k,
            "rho_score_first_covariate": rho,
            "recorded_in": note,
            "residual_implementation": "rank_resid_multi (lstsq, intercept)",
            "ddof_mismatch_expressible": int(k == 1),
            "basis": "real vectors",
            "source_npz": str(f),
            "slope_inflation_if_applied": n / (n - 1),
            "tau_wtau_lstsq": w_ls, "tau_taub_lstsq": k_ls,
        }
        if k == 1:
            p0 = covs[0]
            w_d1, k_d1 = taus(rank_residual_covvar(s, p0), o)
            w_d0, k_d0 = taus(rank_residual_covvar_ddof0(s, p0), o)
            row.update({
                "tau_wtau_ddof1": w_d1, "tau_wtau_ddof0": w_d0,
                "tau_taub_ddof1": k_d1, "tau_taub_ddof0": k_d0,
                "shift_wtau_ddof1_minus_ddof0": w_d1 - w_d0,
                "shift_taub_ddof1_minus_ddof0": k_d1 - k_d0,
                "shift_wtau_ddof1_minus_lstsq": w_d1 - w_ls,
                "shift_taub_ddof1_minus_lstsq": k_d1 - k_ls,
            })
            mw, xw, mk, xk = matched_synthetic(n, rho, args.synth_seeds)
            row.update({
                "synth_mean_abs_shift_wtau": mw, "synth_max_abs_shift_wtau": xw,
                "synth_mean_abs_shift_taub": mk, "synth_max_abs_shift_taub": xk,
                "synth_seeds": args.synth_seeds,
            })
            print(f"  {atlas:10s} {label:24s} n={n:5d} rho={rho:+.4f} REAL  "
                  f"wtau d1-d0 {w_d1-w_d0:+.2e}  taub d1-d0 {k_d1-k_d0:+.2e}  "
                  f"(synth mean |d| {mw:.2e} / {mk:.2e})")
        else:
            row["note"] = ("the cov/var form takes exactly one covariate; a "
                           f"k={k} residual is only expressible as rank_resid_multi, "
                           "so the ddof mismatch cannot arise on this row")
            print(f"  {atlas:10s} {label:24s} n={n:5d} k={k}  "
                  f"REAL  ddof mismatch not expressible")
        out_rows.append(row)

    fields = sorted({k for r in out_rows for k in r})
    fields = (["atlas", "label", "n", "k_covariates", "basis"]
              + [f for f in fields if f not in
                 ("atlas", "label", "n", "k_covariates", "basis")])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out} ({len(out_rows)} rows)")
    print("\nNOT IN SCOPE (listed so the row set is exhaustive rather than truncated):")
    for name, n, why in NOT_IN_SCOPE:
        print(f"  {name}  n = {n}\n      {why}")


if __name__ == "__main__":
    main()
