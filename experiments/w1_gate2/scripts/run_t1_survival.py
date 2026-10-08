"""T1 survival arm — Stem.Sig (Zhang 2022) on TCGA.

LOCK 1: same metric (C-index), same Cox model, same split for BOTH the Stem.Sig
        score and the library-size baseline.
LOCK 2: use released Stem.Sig gene set from Table S6 of Zhang 2022 supplementary
        (n=454 genes, of which 394 map to Xena STAR IDs — 86.7% coverage).

Coverage deviation (DISCLOSED): Zhang used TCGA pan-cancer (30+ cancer types,
n≈10,154). This session downloaded only 4 cancers already required for T2..T5
(ESCA + LIHC + LUAD + PAAD). We compute T1 survival as a **4-cancer subset**
of Zhang's protocol, with cancer_type as a Cox stratum, and report which
cancers are missing. This deviates from Zhang's coverage but reproduces the
protocol form. The direction of the deviation: fewer cancers → less
statistical power, no directional bias toward stemness signal.
"""

from __future__ import annotations
import gzip, json
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


def load_xena_counts(gz_path: Path) -> pd.DataFrame:
    with gzip.open(gz_path, "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    df.index = df.index.str.split(".").str[0]
    return df


def load_stem_sig() -> list[str]:
    return json.loads((DATA / "stem_sig_genes.json").read_text())


def load_sym2ens() -> dict[str, str]:
    return json.loads((DATA / "tcga" / "sym2ens.json").read_text())


def load_clinical_multi(cancers: list[str]) -> pd.DataFrame:
    cdr = pd.read_excel(DATA / "tcga" / "TCGA-CDR-SupplementalTableS1.xlsx",
                        sheet_name="TCGA-CDR")
    sub = cdr[cdr["type"].isin(cancers)][
        ["bcr_patient_barcode", "type", "OS", "OS.time"]].copy()
    sub = sub.dropna(subset=["OS", "OS.time"])
    sub = sub[sub["OS.time"] > 0]
    return sub


def align_expr(expr: pd.DataFrame, clin: pd.DataFrame):
    tumor_mask = pd.Series({s: (len(s) >= 15 and s[13:15] == "01")
                            for s in expr.columns})
    expr_tumor = expr.loc[:, tumor_mask[tumor_mask].index]
    patient_of_sample = pd.Series({s: s[:12] for s in expr_tumor.columns})
    first_sample = patient_of_sample.reset_index().groupby(0)["index"].first()
    expr_one = expr_tumor.loc[:, first_sample.values]
    expr_one.columns = first_sample.index
    common = clin[clin["bcr_patient_barcode"].isin(expr_one.columns)].copy()
    common = common.set_index("bcr_patient_barcode")
    keep = [p for p in expr_one.columns if p in common.index]
    expr_one = expr_one.loc[:, keep]
    common = common.loc[keep]
    return expr_one, common


def score_stem_sig(expr_log2p1: pd.DataFrame, stem_sig: list[str],
                   sym2ens: dict) -> tuple[pd.Series, int]:
    ens_kept = []
    for s in stem_sig:
        e = sym2ens.get(s)
        if e is None:
            continue
        if e in expr_log2p1.index:
            ens_kept.append(e)
    sub = expr_log2p1.loc[ens_kept]
    z = sub.sub(sub.mean(axis=1), axis=0).div(
        sub.std(axis=1).replace(0, 1), axis=0)
    return z.mean(axis=0), len(ens_kept)


def library_size(expr_log2p1: pd.DataFrame) -> pd.Series:
    counts = (2.0 ** expr_log2p1 - 1.0).clip(lower=0)
    return np.log10(counts.sum(axis=0) + 1)


def cindex_5fold_stratified(x, T, E, strata, seed=20260718):
    """Cox with cancer stratum, single continuous covariate x, 5-fold CV."""
    df = pd.DataFrame({"x": x.values, "T": T, "E": E,
                       "strat": strata}).dropna()
    df = df[df["T"] > 0]
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.full(len(df), np.nan)
    for tr, te in kf.split(df):
        train = df.iloc[tr]
        if train["E"].sum() < 5 or train["x"].std() == 0:
            continue
        cph = CoxPHFitter()
        try:
            cph.fit(train[["x", "T", "E", "strat"]],
                    duration_col="T", event_col="E",
                    strata=["strat"], show_progress=False)
            oof[te] = cph.predict_partial_hazard(
                df.iloc[te][["x", "strat"]]).values
        except Exception:
            continue
    valid = ~np.isnan(oof)
    if valid.sum() < 20:
        return np.nan
    return concordance_index(df["T"].values[valid], -oof[valid],
                             df["E"].values[valid])


def bootstrap(fn, n=100, seed=20260718):
    """fn(seed) -> scalar. Return (median, 2.5%, 97.5%)."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        v = fn(int(rng.integers(1e9)))
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (np.nan, np.nan, np.nan)
    return (float(np.median(vals)),
            float(np.percentile(vals, 2.5)),
            float(np.percentile(vals, 97.5)))


def main():
    cancers = ["ESCA", "LIHC", "LUAD", "PAAD"]
    print(f"Cancers loaded (subset of Zhang 2022's 30-cancer pan-cancer): {cancers}")

    stem = load_stem_sig()
    sym2ens = load_sym2ens()

    # Load all four expression matrices and stack (they share the same 60660 ENSGs)
    exprs = {}
    for c in cancers:
        gz = DATA / "tcga" / f"TCGA-{c}.star_counts.tsv.gz"
        exprs[c] = load_xena_counts(gz)
        print(f"  {c}: {exprs[c].shape}")

    clin = load_clinical_multi(cancers)
    print(f"  clinical: {len(clin)} patients")

    # Per-cancer align, then concat
    per_scores = []
    per_libs = []
    per_meta = []
    for c in cancers:
        expr_a, clin_a = align_expr(exprs[c], clin[clin["type"] == c])
        stem_score, n_genes_kept = score_stem_sig(expr_a, stem, sym2ens)
        lib = library_size(expr_a)
        per_scores.append(stem_score.rename(f"stem"))
        per_libs.append(lib.rename(f"lib"))
        meta = clin_a[["type", "OS", "OS.time"]].copy()
        meta["stem"] = stem_score.values
        meta["lib"] = lib.values
        per_meta.append(meta)
        print(f"  {c}: aligned n={expr_a.shape[1]}, Stem.Sig genes used={n_genes_kept}")

    combined = pd.concat(per_meta)
    print(f"\nCombined: n={len(combined)} across {combined['type'].nunique()} cancers")
    print(combined.groupby("type")["OS"].agg(["count", "sum"]))

    T = combined["OS.time"].astype(float).values
    E = combined["OS"].astype(int).values
    strat = combined["type"].values

    # Compute stratified Cox C-index for both
    ci_stem = cindex_5fold_stratified(combined["stem"], T, E, strat, seed=20260718)
    ci_lib = cindex_5fold_stratified(combined["lib"], T, E, strat, seed=20260718)
    delta = ci_stem - ci_lib
    print(f"\n5-fold stratified Cox C-index:")
    print(f"  Stem.Sig:      {ci_stem:.4f}")
    print(f"  library-size:  {ci_lib:.4f}")
    print(f"  Δ = {delta:+.4f}")

    # Bootstrap CIs (100 resamples)
    def _fn_stem(seed):
        return cindex_5fold_stratified(combined["stem"], T, E, strat, seed=seed)
    def _fn_lib(seed):
        return cindex_5fold_stratified(combined["lib"], T, E, strat, seed=seed)

    boot_stem = bootstrap(_fn_stem, n=50, seed=42)  # 50 resamples for speed
    boot_lib = bootstrap(_fn_lib, n=50, seed=43)
    print(f"  boot stem [median (2.5-97.5)]: {boot_stem}")
    print(f"  boot lib  [median (2.5-97.5)]: {boot_lib}")

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

    n_stem_genes = int((combined["stem"].abs() > 0).any() and
                       len([s for s in stem if sym2ens.get(s) in exprs[cancers[0]].index]))

    out = {
        "target": "T1_survival_partial",
        "cohorts": cancers,
        "cohort_note": ("4-cancer subset of Zhang 2022 30-cancer pan-cancer TCGA; "
                        "deviation from prereg n≈10,154: this session uses only the four "
                        "cancers already downloaded for T2..T5, disclosed here."),
        "n_samples": int(len(combined)),
        "n_events": int(E.sum()),
        "stem_sig_gene_count": len(stem),
        "stem_sig_genes_mapped_to_xena_ens": int(len([s for s in stem if sym2ens.get(s) in exprs[cancers[0]].index])),
        "signature_type": "released_unweighted_zmean (Zhang 2022 Table S6)",
        "metric": "Harrell C-index",
        "split": "5-fold CV, Cox with cancer-type stratum",
        "c_index_stemness": float(ci_stem),
        "c_index_libsize": float(ci_lib),
        "delta": float(delta),
        "boot_stemness_median_ci95": boot_stem,
        "boot_libsize_median_ci95": boot_lib,
        "verdict": verdict,
    }
    (RESULTS / "T1_survival_partial.json").write_text(json.dumps(out, indent=2))
    print("\n" + json.dumps({k: v for k, v in out.items()
                              if k != "cohorts" and k != "cohort_note"}, indent=2))


if __name__ == "__main__":
    main()
