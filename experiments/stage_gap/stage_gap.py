#!/usr/bin/env python3
"""
stage_gap.py: the stage-gap and per-transition decomposition of the zebrafish
result (preprint Appendix B, Table A2), recomputed from the score vectors.

    python3 experiments/stage_gap/stage_gap.py              # 1,000 resamples
    python3 experiments/stage_gap/stage_gap.py --n-boot 0   # point values only

The audit reports one Kendall tau_b per score against the twelve-stage ordinal.
That statistic pools every cross-stage pair. This script splits it, on the same
residuals as the audit: own_baseline.conditional_skill.align, then
rank_resid_multi on the score's declared primitive, as in
experiments/w6_direction/direction_check.py.

What it computes, per row:
  g_k          concordance restricted to cell pairs whose stage ranks differ by
               exactly k (k = 1..11): the concordant-minus-discordant count over
               those pairs, divided by sqrt(N * (N - T_r)), where N is the number
               of pairs in the restriction and T_r the pairs among them tied on
               the residual. No pair in the restriction is tied on the stage.
  transitions  the same statistic for each of the eleven pairs of adjacent
               stages separately, labelled by hours post fertilisation.
  ALL          the same statistic over every cross-stage pair (all k pooled).
  retention    g_1 / ALL, with a cell-level bootstrap interval: seed 42,
               indices drawn as in conditional_skill._unit, g_1 and ALL
               recomputed on each resample of the audit residual. The residual
               is not refit within a resample, which is how the printed
               intervals were made (run of 2026-09-04); --refit refits it.
  tau_b check  the same counts with the pairs tied within a stage put back
               into the denominator; this must equal scipy.stats.kendalltau and
               the frozen value in figures/fig2_values.json.

Rows: the seven scores placed on the ordinal before the stage-gap analysis was
run (2026-09-04). MCE and StemFinder were added later and are not decomposed,
as the preprint says. NCG, dpath and SLICE are on the shared seed-42 3,000-cell
subsample, as in the audit.

Inputs (under $OWNBASELINE_DATA_ROOT, default data/runs; git-ignored):
  track2/data/prepared/primitives.npz
  track2/results/{cytotrace,scent,origins,ncg,dpath,slice}_scores.npz
reproduce.py writes primitives.npz, cytotrace_scores.npz and scent_scores.npz.
The ORIGINS, NCG, dpath and SLICE vectors need their R packages
(reproduce.py --ledger); no stage here writes them.

Outputs (committed; this directory's results/):
  stage_gap.json     everything below, plus input sha256 and the check result
  gap_profile.csv    Table A2: one row per score, g_1..g_11
  transitions.csv    one row per score, the eleven adjacent transitions
  retention.csv      ALL, g_1, retention and its bootstrap interval
The script compares its rounded output with the values printed in the
preprint (PAPER below) and exits 1 if any differs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from own_baseline.conditional_skill import align, rank_resid_multi  # noqa: E402
from own_baseline.paths import data_root  # noqa: E402

SEED = 42
FIG2_VALUES = REPO / "figures" / "fig2_values.json"
ROWS = [  # name as in fig2_values.json, score file, key, primitive key, primitive name
    ("CytoTRACE v1", "cytotrace", "ct_v1", "gene_count", "gene count"),
    ("SCENT SR", "scent", "sr", "pcc_string_v12", "PCC(x, degree)"),
    ("ORIGINS", "origins", "origins", "pcc_string_v12", "PCC(x, degree)"),
    ("NCG", "ncg", "ncg", "pcc_string_v12", "PCC(x, degree)"),
    ("dpath", "dpath", "dpath", "shannon_H", "Shannon entropy"),
    ("SLICE", "slice", "slice", "shannon_H", "Shannon entropy"),
    ("CCAT", "scent", "ccat", "pcc_string_v12", "PCC(x, degree)"),
]

# Values as printed in the preprint: Table A2 (three decimals) and the numbers
# in the Appendix B text, transcribed from its source. The run's own output is
# compared with them at the precision they were printed.
PAPER = {
    "table_a2": {
        "CytoTRACE v1": [+0.318, +0.575, +0.755, +0.896, +0.964, +0.982, +0.990, +0.997, +0.998, +0.999, +0.999],
        "SCENT SR": [+0.241, +0.331, +0.415, +0.544, +0.700, +0.694, +0.802, +0.796, +0.841, +0.866, +0.853],
        "ORIGINS": [+0.248, +0.318, +0.397, +0.449, +0.614, +0.659, +0.764, +0.822, +0.893, +0.964, +0.953],
        "NCG": [+0.224, +0.373, +0.499, +0.510, +0.604, +0.446, +0.638, +0.439, +0.417, +0.454, +0.537],
        "dpath": [+0.191, +0.276, +0.061, -0.014, +0.075, -0.058, +0.322, +0.015, -0.255, -0.973, -0.975],
        "SLICE": [+0.048, -0.001, -0.108, -0.149, -0.261, -0.474, -0.427, -0.577, -0.391, -0.079, +0.013],
        "CCAT": [-0.094, -0.141, -0.288, -0.252, -0.406, -0.320, -0.463, -0.320, -0.343, -0.351, -0.363],
    },
    "hpf": [3.3, 3.7, 4.3, 4.7, 5.3, 6.0, 6.7, 8.0, 9.0, 10.0, 11.0, 12.0],
    # CytoTRACE's negative transitions, keyed by (from hpf, to hpf)
    "cytotrace_negative_transitions": {(3.3, 3.7): -0.163, (3.7, 4.3): -0.149, (5.3, 6.0): -0.252},
    "negative_transition_counts": {"CytoTRACE v1": 3, "SCENT SR": 6, "ORIGINS": 3},
    "origins_smallest_reversal": 0.002,
    "all_cross_stage": {"CytoTRACE v1": +0.700, "SCENT SR": +0.461},
    "retention": {"CytoTRACE v1": 0.455, "SCENT SR": 0.523},
    "retention_ci95": {"CytoTRACE v1": [0.448, 0.461], "SCENT SR": [0.511, 0.535]},
    "subsample_earliest_stage_cells": 28,
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def block_counts(ra_sorted: np.ndarray, rb: np.ndarray):
    """Pairs (a in earlier stage, b in later stage): (nc - nd, pairs tied on r)."""
    lo = np.searchsorted(ra_sorted, rb, side="left")    # a strictly below b
    hi = np.searchsorted(ra_sorted, rb, side="right")
    above = len(ra_sorted) - hi                         # a strictly above b
    return float(lo.sum() - above.sum()), float((hi - lo).sum())


def pair_tables(r: np.ndarray, o: np.ndarray, levels: np.ndarray):
    """(nc - nd), N and T_r for every ordered pair of stages i < j."""
    L = len(levels)
    groups = [np.sort(r[o == lv]) for lv in levels]
    size = np.array([len(g) for g in groups], dtype=float)
    S = np.zeros((L, L))
    T = np.zeros((L, L))
    N = np.outer(size, size)
    for i in range(L):
        for j in range(i + 1, L):
            S[i, j], T[i, j] = block_counts(groups[i], groups[j])
    return S, N, T, size


def conc(S, N, T, mask) -> float:
    s, n, t = S[mask].sum(), N[mask].sum(), T[mask].sum()
    return float(s / np.sqrt(n * (n - t))) if n > t else float("nan")


def gap_mask(L: int, k: int) -> np.ndarray:
    m = np.zeros((L, L), dtype=bool)
    for i in range(L - k):
        m[i, i + k] = True
    return m


def cross_mask(L: int) -> np.ndarray:
    return np.triu(np.ones((L, L), dtype=bool), 1)


def decompose(r: np.ndarray, o: np.ndarray, levels: np.ndarray) -> dict:
    L = len(levels)
    S, N, T, size = pair_tables(r, o, levels)
    gaps = [conc(S, N, T, gap_mask(L, k)) for k in range(1, L)]
    trans = []
    for i in range(L - 1):
        m = np.zeros((L, L), dtype=bool)
        m[i, i + 1] = True
        trans.append(conc(S, N, T, m))
    allx = conc(S, N, T, cross_mask(L))
    # tau_b from the same counts: put the within-stage pairs back
    n = len(r)
    n0 = n * (n - 1) / 2.0
    n1 = float((size * (size - 1) / 2.0).sum())               # tied on stage
    _, cnt = np.unique(r, return_counts=True)
    n2 = float((cnt * (cnt - 1) / 2.0).sum())                 # tied on residual
    taub = float(S[cross_mask(L)].sum() / np.sqrt((n0 - n1) * (n0 - n2)))
    return {"gaps": gaps, "transitions": trans, "all_cross_stage": allx,
            "taub_same_routine": taub, "stage_sizes": size.astype(int).tolist(),
            "cross_pairs": float(N[cross_mask(L)].sum()), "within_stage_pairs": n1}


def adj_and_all(r: np.ndarray, o: np.ndarray, levels: np.ndarray):
    L = len(levels)
    S, N, T, _ = pair_tables(r, o, levels)
    return conc(S, N, T, gap_mask(L, 1)), conc(S, N, T, cross_mask(L))


def bootstrap_retention(r, s_aligned, p, o, levels, n_boot: int, refit: bool):
    """seed 42 and the index draw of conditional_skill._unit."""
    rng = np.random.default_rng(SEED)
    n = len(o)
    out = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        rb = rank_resid_multi(s_aligned[i], [p[i]]) if refit else r[i]
        adj, allx = adj_and_all(rb, o[i], levels)
        out[b] = adj / allx
    lo, hi = np.nanquantile(out, [0.025, 0.975])
    return float(lo), float(hi)


def check_against_paper(rows_out: dict, hpf: list, sub_first: int) -> list:
    """Every printed value against this run, at the precision it was printed."""
    bad = []

    def same(label, got, want, nd=3):
        if round(got, nd) != round(want, nd):
            bad.append(f"{label}: run {got:+.{nd + 2}f}, paper {want:+.{nd}f}")

    if [round(h, 1) for h in hpf] != PAPER["hpf"]:
        bad.append(f"stage to hpf map: run {hpf}, paper {PAPER['hpf']}")
    for nm, want in PAPER["table_a2"].items():
        for k, (g, w) in enumerate(zip(rows_out[nm]["gaps"], want), start=1):
            same(f"Table A2 {nm} g{k}", g, w)
    ct = rows_out["CytoTRACE v1"]["transitions"]
    labels = [(hpf[i], hpf[i + 1]) for i in range(len(hpf) - 1)]
    neg_ct = {labels[i]: v for i, v in enumerate(ct) if v < 0}
    if set(neg_ct) != set(PAPER["cytotrace_negative_transitions"]):
        bad.append(f"CytoTRACE negative transitions: run {sorted(neg_ct)}, "
                   f"paper {sorted(PAPER['cytotrace_negative_transitions'])}")
    for lab, w in PAPER["cytotrace_negative_transitions"].items():
        if lab in neg_ct:
            same(f"CytoTRACE transition {lab[0]:g} to {lab[1]:g}", neg_ct[lab], w)
    for nm, w in PAPER["negative_transition_counts"].items():
        got = sum(v < 0 for v in rows_out[nm]["transitions"])
        if got != w:
            bad.append(f"{nm} negative transitions: run {got}, paper {w}")
    org_neg = [v for v in rows_out["ORIGINS"]["transitions"] if v < 0]
    same("ORIGINS smallest reversal", min(abs(v) for v in org_neg),
         PAPER["origins_smallest_reversal"])
    for nm, w in PAPER["all_cross_stage"].items():
        same(f"{nm} all cross-stage pairs", rows_out[nm]["all_cross_stage"], w)
    for nm, w in PAPER["retention"].items():
        same(f"{nm} retention", rows_out[nm]["retention"], w)
    for nm, (wlo, whi) in PAPER["retention_ci95"].items():
        ci = rows_out[nm].get("retention_ci95")
        if ci is None:  # --n-boot 0: nothing to compare
            continue
        same(f"{nm} retention 2.5%", ci[0], wlo)
        same(f"{nm} retention 97.5%", ci[1], whi)
    if sub_first != PAPER["subsample_earliest_stage_cells"]:
        bad.append(f"subsample earliest stage: run {sub_first} cells, "
                   f"paper {PAPER['subsample_earliest_stage_cells']}")
    return bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n-boot", type=int, default=1000,
                    help="bootstrap resamples for the retention interval (0 skips it)")
    ap.add_argument("--refit", action="store_true",
                    help="refit the residual within each resample (not how the printed "
                         "intervals were made, so the check is skipped for them)")
    ap.add_argument("--out", type=Path, default=HERE / "results",
                    help="output directory (default: the committed results/ next to this script)")
    args = ap.parse_args(argv)

    root = data_root()
    prim_f = root / "track2/data/prepared/primitives.npz"
    res = root / "track2/results"
    files = {f: res / f"{f}_scores.npz" for f in ("cytotrace", "scent", "origins", "ncg", "dpath", "slice")}
    missing = [str(p) for p in [prim_f, *files.values()] if not p.exists()]
    if missing:
        print("missing inputs under the data root:\n  " + "\n  ".join(missing), file=sys.stderr)
        print("see the module docstring and `python3 reproduce.py --ledger`", file=sys.stderr)
        return 1

    P = np.load(prim_f)
    stage = P["rank_kimmel"].astype(float)
    hpf_all = P["hpf"].astype(float)
    levels = np.unique(stage)
    hpf = []
    for lv in levels:
        h = np.unique(hpf_all[stage == lv])
        assert len(h) == 1, f"stage {lv} carries more than one hpf value: {h}"
        hpf.append(float(h[0]))

    idx42 = np.sort(np.random.default_rng(42).choice(len(stage), size=3000, replace=False))
    frozen = {r["score"]: r for r in json.loads(FIG2_VALUES.read_text())["rows"]
              if r["kind"] == "measured"}

    rows_out = {}
    for nm, f, key, pkey, pname in ROWS:
        d = np.load(files[f])
        s = d[key].astype(float)
        if "cell_idx" in d.files:
            ci = d["cell_idx"].astype(int)
            assert np.array_equal(ci, idx42), f"{nm} is not on the seed-42 subsample"
        else:
            assert len(s) == len(stage), nm
            ci = np.arange(len(stage))
        o, p = stage[ci], P[pkey].astype(float)[ci]
        lv = np.unique(o)
        assert np.array_equal(lv, levels), f"{nm}: a stage is empty on its cells"
        sa = align(s, o)
        r = rank_resid_multi(sa, [p])
        dec = decompose(r, o, levels)
        kt = float(kendalltau(r, o).statistic)
        assert abs(dec["taub_same_routine"] - kt) < 1e-12, f"{nm}: {dec['taub_same_routine']} vs {kt}"
        assert abs(kt - frozen[nm]["tau"]) < 1e-12, f"{nm}: {kt} vs frozen {frozen[nm]['tau']}"
        row = {"score": nm, "primitive": pname, "n": int(len(ci)),
               "cells": "seed-42 3,000-cell subsample" if len(ci) < len(stage) else "all",
               "align_sign": 1.0 if np.array_equal(sa, s) else -1.0,
               "taub_scipy": kt, "taub_frozen_fig2_values": frozen[nm]["tau"], **dec}
        row["retention"] = dec["gaps"][0] / dec["all_cross_stage"]
        if args.n_boot > 0:
            t0 = time.time()
            ci95 = list(bootstrap_retention(r, sa, p, o, levels, args.n_boot, args.refit))
            row["retention_ci95_refit" if args.refit else "retention_ci95"] = ci95
            row["retention_n_boot"] = args.n_boot
            print(f"  {nm}: {args.n_boot} resamples in {time.time() - t0:.0f} s", file=sys.stderr)
        rows_out[nm] = row

    sub_first = rows_out["NCG"]["stage_sizes"][0]
    bad = check_against_paper(rows_out, hpf, sub_first)

    trans_labels = [f"{hpf[i]:g}>{hpf[i + 1]:g}" for i in range(len(hpf) - 1)]
    summary = {
        "written_by": "experiments/stage_gap/stage_gap.py",
        "what": "stage-gap (Table A2) and per-transition (Appendix B) concordance of each "
                "row's audit residual on the GSE106474 twelve-stage ordinal",
        "estimator": "own_baseline.conditional_skill.align then rank_resid_multi on the "
                     "declared primitive; concordance (nc - nd) / sqrt(N (N - T_r)) over "
                     "the restricted cross-stage pairs",
        "ordinal": "rank_kimmel, stored early (1) to late (12)",
        "stage_rank_to_hpf": {int(lv): h for lv, h in zip(levels, hpf)},
        "transition_labels_hpf": trans_labels,
        "bootstrap": {"n_boot": args.n_boot, "seed": SEED, "refit_within_resample": args.refit},
        "inputs_sha256": {str(p.relative_to(root)): sha(p) for p in [prim_f, *files.values()]},
        "fig2_values_sha256": sha(FIG2_VALUES),
        "check_against_preprint": {"mismatches": bad, "pass": not bad},
        "rows": list(rows_out.values()),
    }
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "stage_gap.json").write_text(json.dumps(summary, indent=1) + "\n")
    with open(out / "gap_profile.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "primitive", "n"] + [f"g{k}" for k in range(1, len(levels))])
        for r in rows_out.values():
            w.writerow([r["score"], r["primitive"], r["n"]] + [f"{x:+.4f}" for x in r["gaps"]])
    with open(out / "transitions.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "n"] + [f"hpf {t}" for t in trans_labels] + ["negative"])
        for r in rows_out.values():
            w.writerow([r["score"], r["n"]] + [f"{x:+.4f}" for x in r["transitions"]]
                       + [sum(x < 0 for x in r["transitions"])])
    with open(out / "retention.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["score", "n", "all_cross_stage", "g1", "retention", "ci95_lo", "ci95_hi",
                    "n_boot", "taub"])
        for r in rows_out.values():
            ci = r.get("retention_ci95") or r.get("retention_ci95_refit") or [float("nan")] * 2
            w.writerow([r["score"], r["n"], f"{r['all_cross_stage']:+.4f}", f"{r['gaps'][0]:+.4f}",
                        f"{r['retention']:.4f}", f"{ci[0]:.4f}", f"{ci[1]:.4f}",
                        r.get("retention_n_boot", 0), f"{r['taub_scipy']:+.4f}"])

    print("stage rank -> hpf:", ", ".join(f"{int(lv)}:{h:g}" for lv, h in zip(levels, hpf)))
    print(f"\n{'Table A2':<14}" + "".join(f"{'g' + str(k):>8}" for k in range(1, len(levels))))
    for r in rows_out.values():
        print(f"{r['score']:<14}" + "".join(f"{x:>+8.3f}" for x in r["gaps"]))
    print(f"\n{'transitions':<14}" + "".join(f"{t:>9}" for t in trans_labels))
    for r in rows_out.values():
        print(f"{r['score']:<14}" + "".join(f"{x:>+9.3f}" for x in r["transitions"]))
    print(f"\n{'row':<14}{'n':>7}{'tau_b':>9}{'ALL':>9}{'g1':>9}{'retention':>11}  95% interval")
    for r in rows_out.values():
        ci = r.get("retention_ci95") or r.get("retention_ci95_refit")
        cis = f"[{ci[0]:.3f}, {ci[1]:.3f}]" if ci else "not run"
        print(f"{r['score']:<14}{r['n']:>7}{r['taub_scipy']:>+9.4f}{r['all_cross_stage']:>+9.4f}"
              f"{r['gaps'][0]:>+9.4f}{r['retention']:>11.4f}  {cis}")
    print(f"\nsubsample stage sizes: {rows_out['NCG']['stage_sizes']}")
    print(f"wrote {out}/stage_gap.json, gap_profile.csv, transitions.csv, retention.csv")
    if bad:
        print("\nDIFFERS FROM THE PREPRINT:\n  " + "\n  ".join(bad))
        return 1
    print("\nevery value checked equals the preprint's Table A2 and Appendix B")
    return 0


if __name__ == "__main__":
    sys.exit(main())
