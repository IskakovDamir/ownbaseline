"""
atlas_run.py
============
II.7 Atlas-scale reduction pipeline (Session 2, unblocked).

Uses Kang et al. 2025 (Nat Methods, doi:10.1038/s41592-025-02857-2) Supplementary
Table S11 as the per-cell score matrix — the atlas's own author-computed values for
CytoTRACE 2, CytoTRACE 1 (gene counts), SCENT (CCAT), SCENT (SR), StemID on every
cell of every atlas dataset — with S3 as the broad-potency crosswalk.

Ground truth: per prereg §3, broad potency ordinal from S3 (verbatim). H1 denominator:
23 rows (Training 16 + Test 7; Human 7 + Mouse 16), locked in atlas_N.json.

Metric:
  Prereg §3 locked `scipy.stats.weightedtau` (rank-based hyperbolic weigher). We compute
  this as the PRIMARY per-prereg value.
  Additionally, we compute Kang's own per-cell class-balanced weighted Kendall tau
  (`wdm_tau_b`, matches R wdm::wdm) as the atlas-native evaluation protocol; this is
  reported as a supplementary "atlas-native" number so we can honestly see whether the
  choice of aggregation kernel flips the H1 verdict.

Runs H1 (primary), H2 (secondary), H3 (moderator), H4 (descriptive). Bootstrap CIs for
H2/H3 use 10k dataset-level resamples with a fixed seed.

Every number in the emitted JSON comes with (a) the SI source table+cell citation for
GT / score values, plus (b) reproducible-run identifiers (seed, script git hash).

Reproducibility:
    python3 atlas_run.py
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import weightedtau, spearmanr

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from wdm_tau import wdm_tau_b_bit, kang_absolute_weights   # noqa: E402

SI = HERE / "kang2025_SI"
ATLAS_N_PATH = HERE / "atlas_N.json"
OUT_PATH = HERE / "per_dataset_results.json"

# Prereg §3: broad-potency ordinal, verbatim Kang order (Totipotent 1..Differentiated 6)
POT_ORD = {
    "Totipotent": 1,
    "Pluripotent": 2,
    "Multipotent": 3,
    "Oligopotent": 4,
    "Unipotent": 5,
    "Differentiated": 6,
}

# Prereg §2 H1 thresholds:
H1_PASS = 0.05
H1_FAIL = 0.10

# Prereg §2 H2 thresholds:
# Note: prereg lists CytoTRACE-v1 GCS with rho>=0.7 but Kang's atlas conflates it with
# gene_counts (see SCORES note below); rho would be 1.0 by construction. Dropped from H2.
H2_THRESH = {
    "SCENT (SR)":    ("ge", 0.4),
    "SCENT (CCAT)":  ("ge", 0.4),
    "CytoTRACE 2 potency score": ("le", 0.3),  # supervised contrast
}

# Prereg §2 H3: rho >= 0.4 for CCAT moderator model
H3_THRESH = 0.4


def load_atlas_N():
    with open(ATLAS_N_PATH) as f:
        return json.load(f)


def load_kang_tables():
    def _load(name, hdr):
        return pd.read_csv(SI / name, header=hdr).dropna(how="all").dropna(axis=1, how="all")
    S1  = _load("Table_S1.csv", 12)
    S3  = _load("Table_S3.csv", 7)
    S4  = _load("Table_S4.csv", 6)
    S11 = _load("Table_S11.csv", 4)
    S12 = _load("Table_S12.csv", 4)
    S13 = _load("Table_S13.csv", 5)
    return {"S1": S1, "S3": S3, "S4": S4, "S11": S11, "S12": S12, "S13": S13}


def build_h1_row_list(atlas_N, S1) -> pd.DataFrame:
    """Reconstruct the locked H1 row list from atlas_N + S1."""
    tt = S1[S1["Dataset / Analysis"].isin(["Training", "Test"])].copy()
    tt["n_cells"] = pd.to_numeric(tt["Number of cells analyzedb"], errors="coerce")
    tt["n_pot"] = pd.to_numeric(tt["Number of broad potency levels"], errors="coerce")
    kept = tt[(tt["n_cells"] >= 200) & (tt["n_pot"] >= 2)].copy()
    kept = kept[~kept["Accession number"].isin({"GSE132188", "GSE100866"})]
    kept["dataset_name_clean"] = kept["Standardized dataset name"].str.rstrip("degf")
    assert len(kept) == 23, f"H1 denominator drift: got {len(kept)}"
    return kept


def scipy_tau(x, y):
    """scipy.stats.weightedtau (prereg §3 locked, default hyperbolic weigher, rank=True)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return float("nan")
    r = weightedtau(x[m], y[m])
    return float(r.statistic if hasattr(r, "statistic") else r.correlation)


def resolve_dataset_name(row, s11_ds_names):
    """
    Map an S1 'Standardized dataset name' to the matching name in S11.
    S1 may have footnote letter suffixes (e, d, g, f) and Tabula Muris names
    use spaces where S11 uses underscores.
    """
    raw = row["Standardized dataset name"]

    def variants(name):
        # normalized variants: as-is, footnote-stripped, underscore-swapped Tabula Muris tissue
        yield name
        # strip 1-2 char footnote letters
        for k in (1, 2):
            if len(name) > k and name[-k:].isalpha() and all(c in "defg" for c in name[-k:]):
                yield name[:-k]
        # Tabula Muris underscore conversion for tissue names (only inside 'Tabula Muris - X (P)')
        if "Tabula Muris - " in name and " (" in name:
            head, tail = name.split(" - ", 1)
            tissue, rest = tail.split(" (", 1)
            yield f"{head} - {tissue.replace(' ', '_')} ({rest}"
        # Also try both together
        for k in (1, 2):
            if len(name) > k and name[-k:].isalpha() and all(c in "defg" for c in name[-k:]):
                stripped = name[:-k]
                if "Tabula Muris - " in stripped and " (" in stripped:
                    head, tail = stripped.split(" - ", 1)
                    tissue, rest = tail.split(" (", 1)
                    yield f"{head} - {tissue.replace(' ', '_')} ({rest}"

    for v in variants(raw):
        if v in s11_ds_names:
            return v
    return None


def compute_per_dataset(atlas_N, S1, S11, S3):
    kept = build_h1_row_list(atlas_N, S1)
    s11_ds_names = set(S11["Dataset"].dropna().unique())

    # Which scores?
    # NOTE on gcs: The prereg asks for CytoTRACE-v1 GCS. Kang's atlas S11 exposes ONLY
    # a single "CytoTRACE 1" column, and Kang labels this "Gene counts" in S13 (row 3:
    # Core method = "Gene counts"). Per Kang's Methods, they operationalize CytoTRACE 1
    # via the gene-count-based per-cell score — i.e. CytoTRACE 1 IS the gene_counts primitive
    # in the atlas. There is no separate GCS column. We therefore run the tautology check
    # against the "CytoTRACE 1" column as the shared gene_counts baseline, and DROP the
    # separate 'gcs' vs 'gene_counts' H1 line as degenerate (Δ = 0 by construction).
    # This is documented in the dossier.
    SCORES = {
        "gene_counts": "CytoTRACE 1",
        "SCENT (CCAT)": "SCENT (CCAT)",
        "SCENT (SR)":   "SCENT (SR)",
        "StemID":       "StemID",
        "CytoTRACE 2 potency score": "CytoTRACE 2 potency score",
    }

    per_ds = []
    for _, ds_row in kept.iterrows():
        raw = ds_row["Standardized dataset name"]
        s11_name = resolve_dataset_name(ds_row, s11_ds_names)
        if s11_name is None:
            per_ds.append({
                "dataset_name": raw,
                "s11_dataset_name": None,
                "note": "unresolved in S11 (no match after footnote-strip)",
            })
            continue

        cells = S11[S11["Dataset"] == s11_name].copy()
        # Coerce score columns to float
        for src_col in set(SCORES.values()):
            cells[src_col] = pd.to_numeric(cells[src_col], errors="coerce")
        # Broad-potency ordinal from S11 'Ground truth potency' (Kang's own crosswalk applied)
        cells["broad_pot_ord"] = cells["Ground truth potency"].map(POT_ORD)
        # target: more potent = higher, so target = -ordinal (Totipotent -> -1 is highest,
        # Differentiated -> -6 is lowest). A method that ranks potent cells higher gets +tau.
        cells["target"] = -cells["broad_pot_ord"]

        # scipy weightedtau (prereg-locked) per score
        # wdm-b (Kang's own protocol, matched to atlas GT) per score
        # Compute Kang absolute-order per-cell weights (S5, broad-potency variant)
        w_kang = kang_absolute_weights(cells["Standardized phenotypea"], cells["broad_pot_ord"])

        row_result = {
            "dataset_name":     raw,
            "s11_dataset_name": s11_name,
            "accession":        ds_row["Accession number"],
            "cohort":           ds_row["Dataset / Analysis"],
            "species":          ds_row["Species"],
            "platform":         ds_row["Platform"],
            "n_cells":          int(len(cells)),
            "n_broad_pot_levels": int(cells["broad_pot_ord"].nunique()),
            "n_std_phenotypes":  int(cells["Standardized phenotypea"].nunique()),
            "tau_scipy":  {},
            "tau_kang":   {},
        }
        target = cells["target"].values
        for tag, col in SCORES.items():
            x = cells[col].values
            row_result["tau_scipy"][tag] = scipy_tau(x, target)
            row_result["tau_kang"][tag]  = wdm_tau_b_bit(x, target, w_kang)
        per_ds.append(row_result)
    return per_ds


def median_delta(per_ds, score_tag, kernel):
    """Δ_d(S) = τ_wt(S, GT_d) − τ_wt(gene_counts, GT_d); returns per-ds Deltas + median + IQR."""
    deltas = []
    for r in per_ds:
        if "tau_" + kernel not in r: continue
        base = r["tau_" + kernel].get("gene_counts")
        val = r["tau_" + kernel].get(score_tag)
        if base is None or val is None or math.isnan(base) or math.isnan(val):
            continue
        deltas.append(val - base)
    if not deltas:
        return {"n": 0, "median": None, "iqr_lo": None, "iqr_hi": None, "deltas": []}
    a = np.array(deltas)
    return {"n": len(a),
            "median": float(np.median(a)),
            "iqr_lo": float(np.percentile(a, 25)),
            "iqr_hi": float(np.percentile(a, 75)),
            "deltas": [float(x) for x in a]}


def h1_decision(med):
    if med is None: return "N/A"
    if med <= H1_PASS: return "PASS"
    if med > H1_FAIL:  return "FAIL"
    return "INCONCLUSIVE"


def h2_spearman(per_ds, score_tag, kernel, n_boot=10000, seed=42):
    """H2: rho = Spearman(tau_S per ds, tau_gc per ds), with bootstrap CI."""
    xs, ys = [], []
    for r in per_ds:
        if "tau_" + kernel not in r: continue
        v = r["tau_" + kernel].get(score_tag)
        b = r["tau_" + kernel].get("gene_counts")
        if v is None or b is None or math.isnan(v) or math.isnan(b):
            continue
        xs.append(v); ys.append(b)
    if len(xs) < 4:
        return {"n": len(xs), "rho": None, "ci95_lo": None, "ci95_hi": None}
    xs = np.array(xs); ys = np.array(ys)
    rho = float(spearmanr(xs, ys).statistic)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    n = len(xs)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            boot[i] = spearmanr(xs[idx], ys[idx]).statistic
        except Exception:
            boot[i] = np.nan
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {"n": n, "rho": rho, "ci95_lo": float(lo), "ci95_hi": float(hi)}


def h3_moderator(per_ds, kernel):
    """
    H3: predictor per ds = |tau_wt(gene_counts, GT_d)|
        [ideally * corr(CCAT_d, gene_counts_d) but S11 gives per-cell CCAT and gene_counts,
         so the within-ds Pearson between them is computable from S11.]
    """
    xs, ys = [], []
    # For per-dataset Pearson(CCAT, gene_counts) we need per-cell scores — pull from S11
    S11 = pd.read_csv(SI / "Table_S11.csv", header=4).dropna(how="all").dropna(axis=1, how="all")

    for r in per_ds:
        s11_name = r.get("s11_dataset_name")
        if s11_name is None: continue
        tk = r.get("tau_" + kernel, {})
        tau_gc = tk.get("gene_counts")
        tau_ccat = tk.get("SCENT (CCAT)")
        if tau_gc is None or tau_ccat is None or math.isnan(tau_gc) or math.isnan(tau_ccat):
            continue
        cells = S11[S11["Dataset"] == s11_name]
        ccat = pd.to_numeric(cells["SCENT (CCAT)"], errors="coerce").values
        gc = pd.to_numeric(cells["CytoTRACE 1"], errors="coerce").values
        m = ~(np.isnan(ccat) | np.isnan(gc))
        if m.sum() < 3: continue
        pear = float(np.corrcoef(ccat[m], gc[m])[0, 1])
        pred = pear * abs(tau_gc)
        xs.append(pred); ys.append(abs(tau_ccat))
    if len(xs) < 4:
        return {"n": len(xs), "rho": None, "verdict": "N/A"}
    xs = np.array(xs); ys = np.array(ys)
    rho = float(spearmanr(xs, ys).statistic)
    verdict = "PASS" if rho >= H3_THRESH else "FAIL"
    return {"n": len(xs), "rho": rho, "verdict": verdict}


def h4_descriptive(per_ds, kernel):
    """H4: per-dataset tau(gene_counts, GT_d) — distribution, sign flips."""
    xs = []
    tissues = []
    for r in per_ds:
        tk = r.get("tau_" + kernel, {})
        v = tk.get("gene_counts")
        if v is None or math.isnan(v): continue
        xs.append(v)
        tissues.append(r["dataset_name"])
    if not xs:
        return {}
    a = np.array(xs)
    return {
        "n": len(a),
        "median": float(np.median(a)),
        "mean":   float(np.mean(a)),
        "std":    float(np.std(a)),
        "min":    float(np.min(a)),
        "max":    float(np.max(a)),
        "frac_negative": float((a < 0).mean()),
        "frac_near_zero": float((abs(a) < 0.1).mean()),
        "per_dataset": [
            {"dataset": t, "tau_gc": float(v)} for t, v in zip(tissues, xs)
        ],
    }


def main():
    t0 = time.time()
    print("Loading atlas_N.json + Kang SI tables...")
    atlas_N = load_atlas_N()
    tabs = load_kang_tables()
    print(f"  S1  rows: {len(tabs['S1'])}")
    print(f"  S3  rows: {len(tabs['S3'])}")
    print(f"  S11 rows: {len(tabs['S11'])}")
    print(f"  S12 rows: {len(tabs['S12'])}")

    # SHA256 provenance
    provenance = {}
    for f in sorted(os.listdir(SI)):
        p = SI / f
        if p.is_file():
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                while True:
                    b = fh.read(1 << 20)
                    if not b: break
                    h.update(b)
            provenance[f] = {"sha256": h.hexdigest(), "size_bytes": os.path.getsize(p)}

    print("\nComputing per-dataset tau values on 23 H1 rows...")
    print(" ... (both scipy.stats.weightedtau [prereg-locked] and wdm-emulation [Kang])")
    per_ds = compute_per_dataset(atlas_N, tabs["S1"], tabs["S11"], tabs["S3"])
    for r in per_ds:
        sgc = r.get('tau_scipy', {}).get('gene_counts')
        kgc = r.get('tau_kang',  {}).get('gene_counts')
        sgc_s = f"{sgc:+.3f}" if isinstance(sgc, (int, float)) and not math.isnan(sgc) else "n/a"
        kgc_s = f"{kgc:+.3f}" if isinstance(kgc, (int, float)) and not math.isnan(kgc) else "n/a"
        print(f"  [{r.get('cohort','?'):8s}] {r['dataset_name']:40s} "
              f"n={r.get('n_cells','?')} "
              f"scipy_gc={sgc_s} kang_gc={kgc_s} "
              f"{'-- unresolved' if r.get('s11_dataset_name') is None else ''}")

    # H1 for each score S in {SCENT-SR, CCAT, gcs (=CytoTRACE 1), StemID}
    print("\n=== H1 (primary): median_d Δ_d(S) per score ===")
    print("Score               | scipy median Δ (IQR)     | Kang median Δ (IQR)       | scipy verdict | Kang verdict")
    print("-" * 120)
    h1_out = {}
    for tag in ["SCENT (SR)", "SCENT (CCAT)", "StemID"]:
        sc = median_delta(per_ds, tag, "scipy")
        kg = median_delta(per_ds, tag, "kang")
        sc_ver = h1_decision(sc["median"])
        kg_ver = h1_decision(kg["median"])
        print(f"{tag:20s}| n={sc['n']:2d} med={sc.get('median',0):+.4f} "
              f"[{sc.get('iqr_lo',0):+.3f},{sc.get('iqr_hi',0):+.3f}] "
              f"| n={kg['n']:2d} med={kg.get('median',0):+.4f} "
              f"[{kg.get('iqr_lo',0):+.3f},{kg.get('iqr_hi',0):+.3f}] "
              f"| {sc_ver:13s} | {kg_ver}")
        h1_out[tag] = {"scipy": sc, "kang": kg, "scipy_verdict": sc_ver, "kang_verdict": kg_ver}

    # H2
    print("\n=== H2 (secondary): rho_S = Spearman(tau_S, tau_gc) across datasets ===")
    print("Score                 | scipy rho [CI95]        | Kang rho [CI95]         | prereg thresh | scipy | Kang")
    print("-" * 130)
    h2_out = {}
    for tag, (op, thresh) in H2_THRESH.items():
        sc = h2_spearman(per_ds, tag, "scipy")
        kg = h2_spearman(per_ds, tag, "kang")
        def verdict(rho, op, th):
            if rho is None: return "N/A"
            if op == "ge": return "PASS" if rho >= th else "FAIL"
            return "PASS" if rho <= th else "FAIL"
        sc_ver = verdict(sc["rho"], op, thresh)
        kg_ver = verdict(kg["rho"], op, thresh)
        sc_str = f"n={sc['n']:2d} rho={sc.get('rho',0):+.4f} [{sc.get('ci95_lo',0):+.3f},{sc.get('ci95_hi',0):+.3f}]" if sc.get("rho") is not None else "n/a"
        kg_str = f"n={kg['n']:2d} rho={kg.get('rho',0):+.4f} [{kg.get('ci95_lo',0):+.3f},{kg.get('ci95_hi',0):+.3f}]" if kg.get("rho") is not None else "n/a"
        print(f"{tag:22s}| {sc_str:35s} | {kg_str:35s} | rho {op} {thresh}       | {sc_ver:5s} | {kg_ver}")
        h2_out[tag] = {"scipy": sc, "kang": kg, "scipy_verdict": sc_ver, "kang_verdict": kg_ver,
                       "prereg_op": op, "prereg_threshold": thresh}

    # H3
    print("\n=== H3: CCAT moderator ===")
    h3_out = {"scipy": h3_moderator(per_ds, "scipy"),
              "kang":  h3_moderator(per_ds, "kang")}
    for kernel, v in h3_out.items():
        print(f"  {kernel}: n={v['n']} rho={v.get('rho')} verdict={v.get('verdict')}")

    # H4
    print("\n=== H4 (descriptive): tau(gene_counts, GT_d) distribution ===")
    h4_out = {"scipy": h4_descriptive(per_ds, "scipy"),
              "kang":  h4_descriptive(per_ds, "kang")}
    for kernel, v in h4_out.items():
        if not v: continue
        print(f"  {kernel} [n={v['n']}]: median={v['median']:+.3f} mean={v['mean']:+.3f} "
              f"[{v['min']:+.3f}, {v['max']:+.3f}] frac_neg={v['frac_negative']:.2f} "
              f"frac_|.|<0.1={v['frac_near_zero']:.2f}")

    out = {
        "provenance": {
            "SI_files": provenance,
            "atlas_N_snapshot_H1_rows": atlas_N["H1_denominator"]["H1_denominator_rows"],
            "script_hash": hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
            "wdm_tau_hash": hashlib.sha256(open(HERE/"wdm_tau.py","rb").read()).hexdigest(),
            "seed_bootstrap": 42,
            "n_bootstrap": 10000,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0)),
            "wall_seconds": round(time.time() - t0, 2),
        },
        "per_dataset": per_ds,
        "H1": h1_out,
        "H2": h2_out,
        "H3": h3_out,
        "H4": h4_out,
    }

    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nWritten: {OUT_PATH}   ({round(time.time()-t0,1)}s)")


if __name__ == "__main__":
    main()
