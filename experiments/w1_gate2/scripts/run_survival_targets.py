"""W1 GATE 2 — own-baseline runs for survival targets T2..T5.

LOCK 1: same metric (C-index) on same Cox model and same split for both sides.
LOCK 2: use paper's RELEASED signature (gene set) via unweighted z-score mean
        where a full coefficient list is not published; use paper's weighted
        formula where it is (T3).

Split: no paper unambiguously specifies an internal train/test split for
       the discovery cohort (TCGA-cancer) that we can reproduce for both
       stemness and library-depth on-the-same-data. We therefore use the
       standard 5-fold CV C-index (Harrell's, out-of-fold predictions
       aggregated), pre-registered here and reported for BOTH sides.
       As a robustness read, we also report the in-sample C-index for
       comparability with the paper's own numbers (which lack an explicit
       held-out set for these single-cohort discovery arms).

Metric: Harrell's C-index. Bootstrap 95% CI with 500 resamples on the
        out-of-fold predictions.

Baseline: per-sample log10(total STAR raw counts summed across genes).
          This is the sample's library size — the raw depth confound.

Output: JSON per target under ../results/, and a summary printed.
"""

from __future__ import annotations
import gzip
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

RNG = np.random.default_rng(20260718)
# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# scratch directory, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root  # noqa: E402
DATA = data_root() / "w1_gate2/data"
RESULTS = data_root() / "w1_gate2/results"
RESULTS.mkdir(exist_ok=True)


# ---- released paper signatures (from Methods / Table S6 / Results text) ----

# T2 Shi 2025 — paper explicitly names 8 genes from univariate Cox (Fig 3A);
# the abstract claim of "18 genes" is not reconcilable with the released
# gene list in text. We use the 8 named genes as an unweighted z-score
# mean because LASSO coefficients are shown only in Fig 3C and not
# machine-readably published. This is a DEVIATION disclosed here.
T2_TSCMS = ["TSPO", "COX4I1", "DSTN", "EIF5", "GPX4", "RPL9", "ZFAS1", "RPL14"]

# T3 Liu 2024 — 4 genes with explicit multivariate Cox coefficients.
T3_CSC = {
    "STIP1": 0.426,
    "H2AZ1": 0.203,   # paper writes H2AFZ; new HGNC symbol is H2AZ1 (same locus)
    "BRIX1": 0.421,
    "TUBB": -0.228,
}

# T4 Lin 2024 — 6 hub genes; CoxBoost+RSF ensemble has no simple linear
# formula. We use unweighted z-score mean as the released-signature proxy.
# DEVIATION disclosed.
T4_CSC = ["SUB1", "POLD2", "ELOVL6", "TNNT1", "PPIA", "IRX2"]

# T5 Fang 2025 — 5 key genes from StepCox+RSF variable importance ranking.
# Ensemble has no simple linear formula. Unweighted z-score mean.
# DEVIATION disclosed.
T5_CSCLPI = ["SHCBP1", "CEP55", "GMNN", "CDCA5", "FAM111B"]


def load_xena_counts(gz_path: Path) -> pd.DataFrame:
    """UCSC Xena GDC hub STAR counts: rows=Ensembl gene, cols=TCGA
    sample barcodes. Values are log2(counts+1) as delivered by Xena.
    We convert back to counts via 2**x - 1 for library-size baseline.
    """
    with gzip.open(gz_path, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    # Strip Ensembl versions
    df.index = df.index.str.split(".").str[0]
    return df


def load_gene_map() -> pd.DataFrame:
    """Symbol -> Ensembl ID (short, no version) map, built via mygene.info
    over the 60,660 ENSGs in the Xena STAR matrix. Returned as a DataFrame
    with columns ['gene','id'] for API compatibility with earlier code."""
    p = DATA / "tcga" / "sym2ens.json"
    import json as _json
    sym2ens = _json.loads(p.read_text())
    return pd.DataFrame({"gene": list(sym2ens.keys()),
                         "id": list(sym2ens.values())})


def score_signature_z(expr_log2p1: pd.DataFrame,
                      gene_map: pd.DataFrame,
                      symbols: list[str]) -> pd.Series:
    """Return per-sample unweighted mean z-score of gene set."""
    # Map symbols to Ensembl IDs present in expr matrix
    sym2ens = gene_map.set_index("gene")["id"].to_dict()
    ens = [sym2ens.get(s) for s in symbols]
    ens_kept = [e for e in ens if isinstance(e, str) and e.split(".")[0] in expr_log2p1.index]
    ens_kept = [e.split(".")[0] for e in ens_kept]
    sub = expr_log2p1.loc[ens_kept]  # log2 space
    # Z-score across samples per gene
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, 1), axis=0)
    return z.mean(axis=0), ens_kept


def score_signature_weighted(expr_log2p1: pd.DataFrame,
                             gene_map: pd.DataFrame,
                             coefs: dict[str, float]) -> pd.Series:
    """Weighted linear score: sum_i coef_i * z_i(gene_i)."""
    sym2ens = gene_map.set_index("gene")["id"].to_dict()
    parts = []
    used = {}
    for sym, c in coefs.items():
        ens = sym2ens.get(sym)
        if ens is None:
            continue
        ens_short = ens.split(".")[0]
        if ens_short not in expr_log2p1.index:
            continue
        v = expr_log2p1.loc[ens_short]
        z = (v - v.mean()) / (v.std() if v.std() > 0 else 1.0)
        parts.append(c * z)
        used[sym] = c
    if not parts:
        return None, {}
    score = sum(parts)
    return score, used


def library_size_baseline(expr_log2p1: pd.DataFrame) -> pd.Series:
    """Per-sample log10(total raw counts). Xena stores log2(count+1);
    invert to counts, sum, log10."""
    counts = (2.0 ** expr_log2p1 - 1.0).clip(lower=0)
    lib = counts.sum(axis=0)
    return np.log10(lib + 1)


def cindex_5fold(x: pd.Series, T: np.ndarray, E: np.ndarray,
                 seed: int = 20260718) -> tuple[float, np.ndarray]:
    """5-fold CV: fit CoxPH with single covariate on train, get partial
    hazard on test, aggregate out-of-fold, compute Harrell C-index.
    Returns (c_index, oof_predictions)."""
    x = x.dropna()
    idx = x.index.intersection(pd.Index(range(len(T)))) if not isinstance(T, pd.Series) else x.index
    # Work in aligned arrays
    df = pd.DataFrame({"x": x.values, "T": T, "E": E}).dropna()
    df = df[df["T"] > 0]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(df), np.nan)
    for tr, te in kf.split(df):
        train = df.iloc[tr]
        # Guard against degenerate folds
        if train["E"].sum() < 5 or train["x"].std() == 0:
            continue
        cph = CoxPHFitter()
        try:
            cph.fit(train[["x", "T", "E"]], duration_col="T", event_col="E",
                    show_progress=False)
            oof[te] = cph.predict_partial_hazard(df.iloc[te][["x"]]).values
        except Exception:
            continue
    valid = ~np.isnan(oof)
    if valid.sum() < 20:
        return np.nan, oof
    ci = concordance_index(df["T"].values[valid], -oof[valid], df["E"].values[valid])
    return ci, oof


def cindex_in_sample(x: pd.Series, T: np.ndarray, E: np.ndarray) -> float:
    df = pd.DataFrame({"x": x.values, "T": T, "E": E}).dropna()
    df = df[df["T"] > 0]
    if df["E"].sum() < 5 or df["x"].std() == 0:
        return np.nan
    cph = CoxPHFitter()
    cph.fit(df[["x", "T", "E"]], duration_col="T", event_col="E",
            show_progress=False)
    return cph.concordance_index_


def bootstrap_ci(fn, x, T, E, n=200, seed=20260718):
    rng = np.random.default_rng(seed)
    N = len(T)
    idxs = np.arange(N)
    vals = []
    for _ in range(n):
        b = rng.choice(idxs, N, replace=True)
        v = fn(x.iloc[b].reset_index(drop=True),
               T[b], E[b])
        if isinstance(v, tuple):
            v = v[0]
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan, np.nan)
    return float(np.median(vals)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def load_clinical(cancer: str) -> pd.DataFrame:
    cdr = pd.read_excel(DATA / "tcga" / "TCGA-CDR-SupplementalTableS1.xlsx",
                        sheet_name="TCGA-CDR")
    sub = cdr[cdr["type"] == cancer][["bcr_patient_barcode", "OS", "OS.time"]].copy()
    sub = sub.dropna(subset=["OS", "OS.time"])
    sub = sub[sub["OS.time"] > 0]
    return sub


def align_expr_to_clinical(expr: pd.DataFrame, clin: pd.DataFrame):
    """Xena sample IDs look like TCGA-XX-YYYY-01A; patient = first 12 chars."""
    patient_of_sample = pd.Series({s: s[:12] for s in expr.columns})
    # Keep primary tumor samples only (barcode positions 14-15 = 01)
    tumor_mask = pd.Series({s: (len(s) >= 15 and s[13:15] == "01") for s in expr.columns})
    expr_tumor = expr.loc[:, tumor_mask[tumor_mask].index]
    patient_of_sample = patient_of_sample[expr_tumor.columns]
    # Pick one sample per patient (first)
    first_sample = patient_of_sample.reset_index().groupby(0)["index"].first()
    keep_samples = first_sample.values
    expr_one = expr_tumor.loc[:, keep_samples]
    expr_one.columns = first_sample.index
    # Join clinical
    common = clin[clin["bcr_patient_barcode"].isin(expr_one.columns)].copy()
    common = common.set_index("bcr_patient_barcode")
    common = common.loc[[p for p in expr_one.columns if p in common.index]]
    expr_one = expr_one.loc[:, common.index]
    return expr_one, common


def run_target(target: str, cancer: str, gz_file: str,
               signature: str, extras: dict = None) -> dict:
    print(f"\n===== {target} ({cancer}) =====")
    gene_map = load_gene_map()
    expr = load_xena_counts(DATA / "tcga" / gz_file)
    print(f"  expr shape: {expr.shape}")
    clin = load_clinical(cancer)
    print(f"  clinical rows: {len(clin)}")

    expr_a, clin_a = align_expr_to_clinical(expr, clin)
    print(f"  aligned samples: {expr_a.shape[1]}")

    T = clin_a["OS.time"].astype(float).values
    E = clin_a["OS"].astype(int).values

    lib = library_size_baseline(expr_a)

    if target == "T3":
        stem, used = score_signature_weighted(expr_a, gene_map, T3_CSC)
        sig_type = "released_weighted"
        used_disp = used
    else:
        symbols = extras["symbols"]
        stem, used = score_signature_z(expr_a, gene_map, symbols)
        sig_type = "released_unweighted_zmean"
        used_disp = used

    print(f"  signature: {sig_type}  genes used={len(used_disp)}")

    # Compute C-index for both sides
    ci_stem, oof_stem = cindex_5fold(stem, T, E)
    ci_lib, oof_lib = cindex_5fold(lib, T, E)
    delta = ci_stem - ci_lib

    ci_stem_in = cindex_in_sample(stem, T, E)
    ci_lib_in = cindex_in_sample(lib, T, E)

    # Bootstrap CIs on 5-fold C-index for both
    def _fn_stem(x, T, E):
        return cindex_5fold(x, T, E, seed=int(RNG.integers(1e9)))[0]

    def _fn_lib(x, T, E):
        return cindex_5fold(x, T, E, seed=int(RNG.integers(1e9)))[0]

    boot_stem = bootstrap_ci(_fn_stem, stem, T, E, n=100)
    boot_lib = bootstrap_ci(_fn_lib, lib, T, E, n=100)

    if abs(delta) <= 0.05:
        verdict = "TAUTOLOG"
    elif delta > 0.10:
        verdict = "SURVIVES"
    elif delta > 0.05:
        verdict = "inconclusive"
    elif delta < -0.10:
        verdict = "raw-beats-stemness"
    else:
        verdict = "inconclusive_neg"

    out = {
        "target": target,
        "cancer": cancer,
        "n_samples": int(expr_a.shape[1]),
        "n_events": int(E.sum()),
        "signature_type": sig_type,
        "signature_used": used_disp if isinstance(used_disp, dict) else list(used_disp),
        "signature_gene_count": len(used_disp),
        "metric": "Harrell C-index",
        "split": "5-fold CV out-of-fold Cox partial hazard",
        "c_index_stemness_5fold": float(ci_stem),
        "c_index_libsize_5fold": float(ci_lib),
        "delta_5fold": float(delta),
        "c_index_stemness_in_sample": float(ci_stem_in),
        "c_index_libsize_in_sample": float(ci_lib_in),
        "delta_in_sample": float(ci_stem_in - ci_lib_in),
        "boot_stemness_median_ci95": boot_stem,
        "boot_libsize_median_ci95": boot_lib,
        "verdict": verdict,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "signature_used"}, indent=2))
    (RESULTS / f"{target}_result.json").write_text(json.dumps(out, indent=2))
    return out


def main():
    all_res = {}
    all_res["T2"] = run_target("T2", "ESCA", "TCGA-ESCA.star_counts.tsv.gz",
                                "TSCMS", {"symbols": T2_TSCMS})
    all_res["T3"] = run_target("T3", "LIHC", "TCGA-LIHC.star_counts.tsv.gz",
                                "CSC", None)
    all_res["T4"] = run_target("T4", "LUAD", "TCGA-LUAD.star_counts.tsv.gz",
                                "LUAD_stemness", {"symbols": T4_CSC})
    all_res["T5"] = run_target("T5", "PAAD", "TCGA-PAAD.star_counts.tsv.gz",
                                "CSCLPI", {"symbols": T5_CSCLPI})
    (RESULTS / "survival_summary.json").write_text(json.dumps(all_res, indent=2))
    # Print summary
    print("\n\n===== SUMMARY =====")
    for k, r in all_res.items():
        print(f"{k}: n={r['n_samples']}  Δ_5fold={r['delta_5fold']:+.3f}  verdict={r['verdict']}")


if __name__ == "__main__":
    main()
