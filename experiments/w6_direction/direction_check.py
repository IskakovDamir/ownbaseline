#!/usr/bin/env python3
"""
direction_check.py: the direction check of section 2, applied to every score
placed on the zebrafish ordinal, and the conditional skill read with each
score's own authors' polarity counted positive.

    python3 experiments/w6_direction/direction_check.py

POST HOC, AND LABELLED SO. Written 2026-10-12, after every registered value
was known. It changes no registered verdict. It exists because the registered
estimator takes each score's sign from the score's own marginal correlation
with the ordinal (own_baseline.conditional_skill.align), while published
benchmarks of these scores fix the sign in advance from the method's
definition (Gulati 2020, Kang 2025, Teschendorff 2017 and 2021, Shi 2020,
Noller and Cahan 2024, Zhang 2022) and count a reversed ordering as a failure.
The two readings are reported side by side, for every row, with every verdict
that differs between them; see DEVIATIONS.md.

What it computes, per row, on the cells the paper used:
  - rho_S and tau_b of the raw score with the stage ordinal (stored early to
    late), so the sign the alignment step took;
  - the authors' polarity c (+1: high = more potent, -1: high = more
    differentiated) with its source, and whether the marginal ordering runs
    with it (the direction check);
  - CS, the registered conditional tau_b (recomputed and checked against the
    frozen value), and CS_auth = tau_b(R(c s) - X beta, -stage), the same
    residual read with the authors' polarity counted positive;
  - the registered floor (Null B, align on) and, for CS_auth, the align-off
    floor of the same grid cell, since a sign fixed in advance is not chosen
    from the data;
  - the null's 2.5th percentile at the cell, for the record.
Joint four-primitive values are recomputed the same way; no align-off null
exists for k > 1, so CS_auth there is set beside the registered k = 4 floor.

Inputs (all under $OWNBASELINE_DATA_ROOT, default data/runs; git-ignored):
  track2/data/prepared/{primitives.npz, cells.tsv}
  track2/results/{cytotrace,scent,origins,ncg,dpath,slice}_scores.npz
  w5_new_rows/inputs/GSE106474/cells.txt
  w5_new_rows/scores/GSE106474/{mce,stemfinder}.csv
Output: experiments/w6_direction/results/direction.json (copied to
data/run_record/w6/direction.json by hand, with its sha256 in that README).
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from own_baseline.conditional_skill import align, kang_taub, rank_resid_multi  # noqa: E402
from own_baseline.paths import data_root  # noqa: E402

NULL_TABLE = REPO / "own_baseline" / "data" / "null_floors.csv.gz"
FIG2_VALUES = REPO / "figures" / "fig2_values.json"
JOINT_FROZEN = {  # registered joint values: (artefact, how to read it)
    "CytoTRACE v1": 0.2881, "SCENT SR": 0.1829, "ORIGINS": 0.1564, "dpath": 0.1341,
}

# Authors' polarity, +1 = high means more potent. Sources are the authors' own
# words unless marked; NCG's direction is inferred from its construction.
POLARITY = {
    "CytoTRACE v1": (+1, "Gulati et al. 2020: '0 = more differentiated; 1 = less differentiated' "
                         "(quoted at scores/cytotrace_full.py:384-385)"),
    "SCENT SR":     (+1, "Teschendorff and Enver 2017; Teschendorff et al. 2021 s2.2, SR "
                         "'correlates with cell potency'"),
    "ORIGINS":      (+1, "Senra et al. 2022: root set at 'the cells with the highest "
                         "differentiation PPI network activity'"),
    "NCG":          (+1, "inferred: Ni et al. 2022 state no direction; Function/CompNCG.R is an "
                         "expression-weighted centrality sum with no sign flip"),
    "dpath":        (+1, "Gong et al. 2017: 'cells with high metagene entropy have higher "
                         "differentiation potential'"),
    "StemFinder":   (-1, "Noller and Cahan 2024: higher stemFinder = more differentiated "
                         "(experiments/w5_new_rows/DISCOVERY.md:128)"),
    "SLICE":        (+1, "Guo et al. 2017: higher entropy, more differentiation potential"),
    "CCAT":         (+1, "Teschendorff et al. 2021 s2.1-2.2: a positively correlated proxy to SR"),
    "MCE":          (+1, "Shi et al. 2020: cells higher on MCE are more potent "
                         "(experiments/w5_new_rows/DISCOVERY.md:56)"),
}


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def null_rows():
    with gzip.open(NULL_TABLE, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def cell(rows, n, rho, align_on, k=1, levels=12):
    """The grid cell own_baseline.floors reads: nearest n, nearest |rho|."""
    pool = [r for r in rows if r["null"] == "B" and r["kernel"] == "kendalltau"
            and r["align"] == ("1" if align_on else "0") and r["levels"] == str(levels)
            and int(r["k_covariates"]) == k]
    if not pool:
        return None
    ng = min({int(r["n"]) for r in pool}, key=lambda g: abs(g - n))
    rg = min({float(r["rho_target"]) for r in pool}, key=lambda g: abs(g - abs(rho)))
    sel = [r for r in pool if int(r["n"]) == ng and abs(float(r["rho_target"]) - rg) < 1e-9]
    return {"n": ng, "rho": rg, "align": align_on, "k": k,
            "p97_5_max": max(float(r["p97.5"]) for r in sel),
            "p2_5_min": min(float(r["p2.5"]) for r in sel),
            "strengths": sorted({r["nullB_strength"] for r in sel})}


def main():
    root = data_root()
    P = np.load(root / "track2/data/prepared/primitives.npz")
    stage = P["rank_kimmel"].astype(float)
    cells_tsv = list(csv.DictReader(open(root / "track2/data/prepared/cells.tsv"), delimiter="\t"))
    cells_w5 = (root / "w5_new_rows/inputs/GSE106474/cells.txt").read_text().split("\n")[:-1]
    assert [r["cell_id"] for r in cells_tsv] == cells_w5, "cell order differs between track2 and w5"
    loglib = np.log10(np.array([float(r["lib_size"]) for r in cells_tsv]))

    def csv_col(rel, col):
        rows = list(csv.DictReader(open(root / rel)))
        assert [r["cell"] for r in rows] == cells_w5, rel
        return np.array([float(r[col]) for r in rows])

    res = root / "track2/results"
    ct = np.load(res / "cytotrace_scores.npz")
    sc = np.load(res / "scent_scores.npz")
    sub = {nm: np.load(res / f"{f}_scores.npz") for nm, f in
           (("ORIGINS", "origins"), ("NCG", "ncg"), ("dpath", "dpath"), ("SLICE", "slice"))}
    idx42 = np.sort(np.random.default_rng(42).choice(len(stage), size=3000, replace=False))
    for nm in ("NCG", "dpath", "SLICE"):
        assert np.array_equal(sub[nm]["cell_idx"].astype(int), idx42), f"{nm} is not on the seed-42 subsample"
    mce = csv_col("w5_new_rows/scores/GSE106474/mce.csv", "mce")
    stf = csv_col("w5_new_rows/scores/GSE106474/stemfinder.csv", "stemfinder")
    ccs = csv_col("w5_new_rows/scores/GSE106474/stemfinder.csv", "cc_gene_set_score")

    ALL = np.arange(len(stage))
    pcc, gc, sh = P["pcc_string_v12"], P["gene_count"], P["shannon_H"]
    spec = [  # name as in fig2_values.json, score, primitive, primitive name, cells
        ("CytoTRACE v1", ct["ct_v1"], gc, "gene count", ALL),
        ("SCENT SR", sc["sr"], pcc, "PCC(x, degree)", ALL),
        ("ORIGINS", sub["ORIGINS"]["origins"], pcc, "PCC(x, degree)", ALL),
        ("NCG", sub["NCG"]["ncg"], pcc, "PCC(x, degree)", idx42),
        ("dpath", sub["dpath"]["dpath"], sh, "Shannon entropy", idx42),
        ("StemFinder", stf, ccs, "cell-cycle gene-set score", ALL),
        ("SLICE", sub["SLICE"]["slice"], sh, "Shannon entropy", idx42),
        ("CCAT", sc["ccat"], pcc, "PCC(x, degree)", ALL),
        ("MCE", mce, pcc, "PCC(x, degree)", ALL),
    ]
    fig2 = {r["score"]: r for r in json.loads(FIG2_VALUES.read_text())["rows"] if r["kind"] == "measured"}
    NULL = null_rows()
    four = [gc, pcc, sh, loglib]
    out_rows, flips = [], []
    for nm, s_full, p_full, pname, ci in spec:
        s = s_full if len(s_full) == len(ci) else s_full[ci]
        o, p = stage[ci], p_full[ci]
        c, src = POLARITY["SCENT SR" if nm == "SCENT SR" else nm]
        rs, kt = float(spearmanr(s, o).statistic), float(kendalltau(s, o).statistic)
        a_sign = -1.0 if rs < 0 else 1.0                       # align()'s choice
        cs = kang_taub(rank_resid_multi(align(s, o), [p]), o)
        cs_auth = kang_taub(rank_resid_multi(c * s, [p]), -o)
        frozen = fig2[nm]
        assert abs(cs - frozen["tau"]) < 1e-12, f"{nm}: {cs} != frozen {frozen['tau']}"
        assert abs(cs_auth - (-c * a_sign * cs)) < 1e-12, nm
        with_authors = bool(np.sign(spearmanr(c * s, -o).statistic) > 0)
        rho_sp = float(spearmanr(s, p).statistic)
        on, off = cell(NULL, len(ci), rho_sp, True), cell(NULL, len(ci), rho_sp, False)
        assert abs(on["p97_5_max"] - frozen["null_p97_5"]) < 1e-12, nm
        lo, hi = frozen["ci95"]
        ci_auth = sorted([-c * a_sign * lo, -c * a_sign * hi])
        row = {
            "score": nm, "n": int(len(ci)), "primitive": pname,
            "rhoS_score_stage": rs, "taub_score_stage": kt,
            "rhoS_primitive_stage": float(spearmanr(p, o).statistic),
            "rho_score_primitive": rho_sp,
            "authors_polarity": c, "polarity_source": src,
            "marginal_runs_with_authors": with_authors,
            "align_sign": a_sign,
            "cs_registered": cs, "cs_registered_CI95": [lo, hi],
            "floor_registered": on["p97_5_max"], "null_cell_align_on": on,
            "clears_registered": bool(cs > on["p97_5_max"]),
            "cs_authors_polarity": cs_auth, "cs_authors_polarity_CI95": ci_auth,
            "floor_align_off": off["p97_5_max"], "null_cell_align_off": off,
            "clears_authors_polarity": bool(cs_auth > off["p97_5_max"]),
            "below_null_p2_5_align_on": bool(cs < on["p2_5_min"]),
        }
        if row["clears_registered"] != row["clears_authors_polarity"]:
            flips.append(nm)
        # joint four-primitive residual, where the paper reports one
        if nm in JOINT_FROZEN or nm in ("StemFinder", "MCE"):
            covs = [v[ci] for v in four]
            j = kang_taub(rank_resid_multi(align(s, o), covs), o)
            j_auth = kang_taub(rank_resid_multi(c * s, covs), -o)
            j_floor = cell(NULL, len(ci), abs(rho_sp), True, k=4)
            row["joint4"] = {"cs_registered": j, "cs_authors_polarity": j_auth,
                             "floor_registered_k4": j_floor["p97_5_max"] if j_floor else None,
                             "null_cell_k4": j_floor}
            if nm in JOINT_FROZEN:
                assert abs(round(j, 4) - JOINT_FROZEN[nm]) < 1e-4, f"{nm} joint {j}"
        out_rows.append(row)

    reg = [r["score"] for r in out_rows if r["clears_registered"]]
    aut = [r["score"] for r in out_rows if r["clears_authors_polarity"]]
    summary = {
        "written_by": "experiments/w6_direction/direction_check.py",
        "status": "post hoc sensitivity analysis, 2026-10-12; registered verdicts unchanged",
        "ordinal": "GSE106474 stage rank, stored early (1) to late (12); earlier = more potent",
        "inputs_sha256": {str(p.relative_to(root)): sha(p) for p in [
            root / "track2/data/prepared/primitives.npz", root / "track2/data/prepared/cells.tsv",
            *(res / f"{f}_scores.npz" for f in ("cytotrace", "scent", "origins", "ncg", "dpath", "slice")),
            root / "w5_new_rows/scores/GSE106474/mce.csv",
            root / "w5_new_rows/scores/GSE106474/stemfinder.csv"]},
        "null_table_sha256": sha(NULL_TABLE),
        "clear_registered": reg, "clear_authors_polarity": aut,
        "clear_both": [s for s in reg if s in aut],
        "verdict_differs": flips,
        "runs_against_authors": [r["score"] for r in out_rows if not r["marginal_runs_with_authors"]],
        "rows": out_rows,
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "direction.json").write_text(json.dumps(summary, indent=1) + "\n")

    print(f"{'score':<13}{'n':>7}{'rhoS(s,stage)':>15}{'with?':>8}{'CS reg':>9}{'floor':>8}"
          f"{'CS auth':>9}{'floor off':>10}  registered / authors")
    for r in out_rows:
        print(f"{r['score']:<13}{r['n']:>7}{r['rhoS_score_stage']:>+15.4f}"
              f"{'with' if r['marginal_runs_with_authors'] else 'AGAINST':>8}"
              f"{r['cs_registered']:>+9.4f}{r['floor_registered']:>8.4f}"
              f"{r['cs_authors_polarity']:>+9.4f}{r['floor_align_off']:>10.4f}  "
              f"{'clears' if r['clears_registered'] else '-':>6} / "
              f"{'clears' if r['clears_authors_polarity'] else '-'}")
        if "joint4" in r:
            j = r["joint4"]
            print(f"{'  joint-4':<13}{'':>7}{'':>15}{'':>8}{j['cs_registered']:>+9.4f}"
                  f"{j['floor_registered_k4']:>8.4f}{j['cs_authors_polarity']:>+9.4f}")
    print("clear registered:", reg)
    print("clear authors' polarity:", aut)
    print("clear both:", summary["clear_both"], " verdict differs:", flips)
    print("marginal runs against authors:", summary["runs_against_authors"])
    print("wrote", HERE / "results" / "direction.json")


if __name__ == "__main__":
    main()
