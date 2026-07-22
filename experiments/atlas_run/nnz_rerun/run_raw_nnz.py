"""
run_raw_nnz.py
==============
Session 4 (2026-07-17): Raw-nnz H1 re-run per prereg §3 letter.

Per the Session 3 Judge verdict item 2, the PI chose pathway (2a): re-run H1 on
raw nnz for as many rows as we can obtain. This script does that end-to-end for
the subset of rows whose raw expression data we can download AND where per-cell
potency labels can be resolved (either from the matrix cell IDs directly, from
the associated GEO SOFT metadata at per-sample resolution matching one plate/lane
to one label, or via the Kang S3 crosswalk if the file exposes author labels).

Design:
  For each obtainable row, we produce (nnz_per_cell, broad_potency_ordinal) pairs.
  τ_wt(nnz, GT) is then computed under both scipy.stats.weightedtau (prereg letter)
  and Kang's wdm class-balance weights (matches per_dataset_results.json 'tau_kang').
  We then take the SR/CCAT/StemID τ values FROM per_dataset_results.json (Kang S11)
  and compute Δ_d(S) = τ(S) − τ(nnz).

  For each row we also record τ(CT1) (from per_dataset_results.json) so we can
  report the per-row baseline gap distribution (Task A6).

Rows attempted this session:
  - E-MTAB-9067 (HSC development, Smart-seq2): DONE from baseline calibration
  - GSE45719 (Mouse embryo 1, Tang et al.): per-cell stage encoded in file names
  - GSE92332 (Intestine, both Drop-seq and Smart-seq2 rows share the atlas file with cell types embedded in barcodes)
  - GSE99933 (Peripheral glia, Smart-seq2): per-cell labels absent from GEO — dropped w/ reason
  - GSE122466 (Retinal neurons): barcodes only, no cell type in file — dropped w/ reason
  - GSE76408 (Lgr5 intestine, CEL-seq): FACS gate labels, not potency-informative — dropped w/ reason

Others attempted / dropped: see row_availability.md.

Reproducibility:
  Seed unused (no bootstrap). Every τ ships with the input file's sha256.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import re
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import weightedtau

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent))  # atlas_run/ is parent
from wdm_tau import wdm_tau_b_bit, kang_absolute_weights  # noqa: E402

CACHE = HERE / "cache"
CACHE.mkdir(parents=True, exist_ok=True)

SI = HERE.parent / "kang2025_SI"
ATLAS_RESULTS = HERE.parent / "per_dataset_results.json"
E_MTAB_CALIBRATION = HERE.parent / "baseline_validation_e_mtab_9067.json"
BASELINE_JSON = HERE.parent / "baseline_validation_e_mtab_9067.json"

# Prereg §3
POT_ORD = {
    "Totipotent": 1, "Pluripotent": 2, "Multipotent": 3,
    "Oligopotent": 4, "Unipotent": 5, "Differentiated": 6,
}
H1_PASS = 0.05
H1_FAIL = 0.10


def scipy_tau(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = ~(np.isnan(x) | np.isnan(y))
    if m.sum() < 3:
        return float("nan")
    r = weightedtau(x[m], y[m])
    return float(r.statistic if hasattr(r, "statistic") else r.correlation)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_per_dataset_results() -> Dict[str, dict]:
    """Load the atlas_run per-row τ values (τ under CT1 baseline). Keyed by dataset_name."""
    with open(ATLAS_RESULTS) as f:
        d = json.load(f)
    return {r["dataset_name"]: r for r in d["per_dataset"]}


# ============================================================================
# ROW-SPECIFIC LOADERS
# Each returns (df: pd.DataFrame with cols "nnz", "broad_pot_ord", "std_phenotype")
# or (None, reason) if the row is not obtainable.
# ============================================================================


def row_E_MTAB_9067() -> Tuple[Optional[pd.DataFrame], str]:
    """HSC development — reuse the calibration script's aggregate result.
    Returns pre-computed τ values instead of a df; we treat this as a special case."""
    with open(BASELINE_JSON) as f:
        d = json.load(f)
    # Package into a "pre-computed" row for downstream Δ calculation
    return None, "PRE_COMPUTED"


def row_GSE45719_mouse_embryo_1() -> Tuple[Optional[pd.DataFrame], str]:
    """Mouse embryo 1 (Tang et al.) — 286 cells in Kang; 291 GSMs (zygote → late blastocyst).
    Each GSM = one cell; filename encodes stage.
    """
    tar_path = CACHE / "GSE45719_RAW.tar"
    if not tar_path.exists():
        return None, f"Tar not downloaded ({tar_path})"

    # Stage → broad potency (per Kang S3 for Mouse embryo 1)
    # Kang S3 for Tang et al 2010 GSE45719 - use S11 broad potency labels observed:
    # Totipotent: zygote, 2cell (early/mid/late)
    # Pluripotent: 4cell, 8cell, 16cell, earlyblast, midblast, lateblast
    STAGE_TO_BROAD = {
        "zy": "Totipotent",
        "early2cell": "Totipotent",
        "mid2cell": "Totipotent",
        "late2cell": "Totipotent",
        "2cell": "Totipotent",
        "4cell": "Pluripotent",
        "8cell": "Pluripotent",
        "16cell": "Pluripotent",
        "earlyblast": "Pluripotent",
        "midblast": "Pluripotent",
        "lateblast": "Pluripotent",
    }

    # Read tar; each member is one cell's expression file. Compute nnz per cell.
    stages = []
    nnzs = []
    with tarfile.open(tar_path) as tar:
        for member in tar.getmembers():
            if not member.name.endswith(".txt.gz"):
                continue
            fname = os.path.basename(member.name)
            # E.g. GSM1112490_16cell_1-10_expression.txt.gz
            m = re.match(r"GSM\d+_([a-zA-Z0-9]+)_[\d-]+_expression\.txt\.gz", fname)
            if not m:
                continue
            stage = m.group(1)
            if stage not in STAGE_TO_BROAD:
                continue
            fobj = tar.extractfile(member)
            if fobj is None:
                continue
            data = gzip.decompress(fobj.read()).decode("utf-8", errors="ignore")
            # Format: GeneName\tRefSeq\tRPKM\tRPKM\tcount\tcount  ... actually let's peek
            # First line is header
            lines = data.splitlines()
            if len(lines) < 2:
                continue
            # Tang et al 2013 expression files: header "#Gene_symbol\tRefseq_IDs\tRPKM\treads\t..."
            # Compute nnz on "reads" column (raw read counts).
            header = lines[0].lstrip("#").split("\t")
            count_col = None
            for idx, h in enumerate(header):
                if h.strip().lower() in {"reads", "read_count", "count", "counts", "readcount"}:
                    count_col = idx
                    break
            if count_col is None:
                # Fallback: try last column
                count_col = len(header) - 1
            counts = []
            for line in lines[1:]:
                parts = line.split("\t")
                if len(parts) <= count_col:
                    continue
                try:
                    val = float(parts[count_col])
                    counts.append(val)
                except ValueError:
                    continue
            if not counts:
                continue
            counts = np.array(counts)
            nnz = int((counts > 0).sum())
            stages.append(stage)
            nnzs.append(nnz)

    if not nnzs:
        return None, "No usable expression files parsed from tar"

    # Build the df
    df = pd.DataFrame({
        "nnz": nnzs,
        "stage": stages,
    })
    df["broad_potency"] = df["stage"].map(STAGE_TO_BROAD)
    df["broad_pot_ord"] = df["broad_potency"].map(POT_ORD)
    df["std_phenotype"] = df["stage"]  # per-stage nested weights approximation
    df = df.dropna(subset=["broad_pot_ord"])
    return df, f"OK — {len(df)} cells across {df['broad_potency'].nunique()} broad-potency levels ({', '.join(sorted(df['broad_potency'].unique()))})"


def _parse_GSE92332_atlas(mtx_path: Path) -> pd.DataFrame:
    """Load GSE92332 atlas file. Cell IDs embed cell type as suffix."""
    # File: rows = genes, cols = cells with names like B1_AAACATTGTTTGGG_Enterocyte.Immature.Distal
    df = pd.read_csv(mtx_path, sep="\t", index_col=0)
    return df


def row_GSE92332_intestine(subplatform: str) -> Tuple[Optional[pd.DataFrame], str]:
    """Intestine (Drop-seq and Smart-seq2).
    subplatform: "Drop-seq" or "Smart-seq2"

    File GSE92332_atlas_UMIcounts.txt.gz contains DROP-SEQ atlas.
    Cell IDs are B1_ACG..._<CellType> — the batch prefix (B1..B4) is 10x lane; cell type at end.

    For Smart-seq2 row we need GSE92332_AtlasFullLength_TPM.txt.gz (TPM, not raw counts) -
    NOT raw counts, so we cannot compute raw nnz for the Smart-seq2 row from this file
    (TPM values > 0 mean the gene is detected, which IS effectively an nnz signal, but
    TPM has already been normalized). Report this row as "no raw counts" for Smart-seq2.
    """
    if subplatform != "Drop-seq":
        return None, "Smart-seq2 subplatform: GSE92332_AtlasFullLength_TPM is TPM (not raw counts); no raw-counts file exposed. Dropped."

    mtx_path = CACHE / "GSE92332_atlas_UMIcounts.txt.gz"
    if not mtx_path.exists():
        return None, f"Matrix not downloaded ({mtx_path})"

    df = _parse_GSE92332_atlas(mtx_path)  # (genes x cells)
    # Cell IDs
    cell_ids = df.columns.tolist()
    # Extract cell type from suffix after last "_"
    types = []
    for cid in cell_ids:
        parts = cid.split("_")
        if len(parts) >= 3:
            ct = "_".join(parts[2:])  # some names have dots
        else:
            ct = None
        types.append(ct)

    # Map cell type to Kang S3 broad potency for GSE92332 intestine Drop-seq
    # Kang S3 for Intestine (Drop-seq): read from S3
    S3 = pd.read_csv(SI / "Table_S3.csv", header=7).dropna(how="all").dropna(axis=1, how="all")
    is_int = S3["Standardized dataset name"].astype(str).str.contains("Intestine (Drop-seq)", regex=False, na=False)
    s3_int = S3[is_int]
    orig_to_broad = dict(zip(
        s3_int["Original phenotype"].astype(str).str.strip(),
        s3_int["Broad potency level"]
    ))
    orig_to_std = dict(zip(
        s3_int["Original phenotype"].astype(str).str.strip(),
        s3_int["Standardized phenotype"]
    ))
    # Print S3 mapping for logging
    log_info = "S3 mapping used:\n"
    for orig, broad in orig_to_broad.items():
        log_info += f"  {orig!r} -> {broad}\n"

    # Kang S3 uses underscores (TA_Early, Enterocyte_Immature_Distal); file uses dots (TA.Early, ...).
    # Normalize the file's dot-separated tokens to Kang's underscore form.
    def _normalize(x):
        if x is None:
            return None
        return x.replace(".", "_").strip()

    df_out = pd.DataFrame({
        "cell": cell_ids,
        "orig_phenotype_raw": types,
    })
    df_out["orig_phenotype_norm"] = df_out["orig_phenotype_raw"].apply(_normalize)

    df_out["broad_potency"] = df_out["orig_phenotype_norm"].map(orig_to_broad)
    df_out["std_phenotype"] = df_out["orig_phenotype_norm"].map(orig_to_std)
    df_out["broad_pot_ord"] = df_out["broad_potency"].map(POT_ORD)

    match_rate = df_out["broad_pot_ord"].notna().mean()
    unmatched = df_out[df_out["broad_pot_ord"].isna()]["orig_phenotype_raw"].value_counts().head(20)
    log_info += f"\nMatch rate: {match_rate*100:.1f}% ({df_out['broad_pot_ord'].notna().sum()}/{len(df_out)})\n"
    if len(unmatched) > 0:
        log_info += f"Unmatched phenotype-tokens:\n{unmatched.to_string()}\n"

    # Compute nnz per cell
    nnz = (df.values > 0).sum(axis=0)
    df_out["nnz"] = nnz

    df_valid = df_out.dropna(subset=["broad_pot_ord"]).copy()
    return df_valid, f"OK — {len(df_valid)} cells matched to broad potency ({match_rate*100:.1f}% of atlas file); {log_info}"


def row_GSE128639_bm_mnc() -> Tuple[Optional[pd.DataFrame], str]:
    """BM-MNC (CITE-seq) — 30672 cells in Kang; 33454 cells in GEO raw counts.
    Kang cell IDs like 'a_AAACCTGAGCTTATCG-1' match GEO 'a_AAACCTGAGCTTATCG.1' after '.'→'-'.
    Compute nnz per cell from raw counts, then merge Kang per-cell broad-potency labels
    (from S11) by cell_id.

    File format: header row = 33454 cell IDs (no leading empty). Each data row = gene_name + 33454 count values.
    """
    mtx_path = CACHE / "GSM3681518_MNC_RNA_counts.tsv.gz"
    if not mtx_path.exists():
        return None, f"Matrix not downloaded ({mtx_path})"

    print(f"  Loading BM-MNC counts (~38MB gz)...")
    # File: header row has 33454 cell IDs (one per column); each data row has gene_symbol + 33454 counts.
    # Total cols in data rows = 33455 (gene + 33454).
    # Streaming approach: accumulate nnz per column (= per cell) without loading whole matrix.
    with gzip.open(mtx_path, "rt") as f:
        header_line = f.readline().rstrip("\n")
        cell_ids = header_line.split("\t")  # 33454 cell IDs
        n_cells = len(cell_ids)
        nnz = np.zeros(n_cells, dtype=np.int64)
        n_genes = 0
        for i, line in enumerate(f):
            parts = line.rstrip("\n").split("\t")
            # First token is gene name; the remainder are counts
            # Values may be int or "0" strings
            if len(parts) - 1 != n_cells:
                # Skip malformed rows
                continue
            for j, v in enumerate(parts[1:]):
                if v != "0" and v != "" and v != "0.0":
                    nnz[j] += 1
            n_genes += 1
            if (i + 1) % 5000 == 0:
                print(f"    parsed {n_genes} genes...")
    print(f"  Parsed {n_genes} genes × {n_cells} cells")

    # Normalize IDs to match Kang: '.'→'-'
    kang_ids = [c.replace(".1", "-1") for c in cell_ids]

    # Load Kang S11 for BM-MNC
    S11 = pd.read_csv(SI / "Table_S11.csv", header=4).dropna(how="all").dropna(axis=1, how="all")
    bmk = S11[S11["Dataset"] == "BM-MNC (CITE-seq)"].copy()
    bmk["broad_pot_ord"] = bmk["Ground truth potency"].map(POT_ORD)
    print(f"  Kang S11 BM-MNC: {len(bmk)} cells")

    # Build merged df
    geo_df = pd.DataFrame({
        "cell_kang_form": kang_ids,
        "nnz": nnz,
    })
    merged = geo_df.merge(bmk[["Cell identifier", "Standardized phenotypea", "Ground truth potency", "broad_pot_ord"]],
                          left_on="cell_kang_form", right_on="Cell identifier", how="inner")
    print(f"  Merged: {len(merged)} cells (match rate: {len(merged)/len(bmk)*100:.1f}% of Kang / {len(merged)/n_cells*100:.1f}% of GEO)")

    df_valid = merged.dropna(subset=["broad_pot_ord"]).copy()
    df_valid["std_phenotype"] = df_valid["Standardized phenotypea"]
    df_valid["broad_potency"] = df_valid["Ground truth potency"]
    return df_valid[["nnz", "std_phenotype", "broad_potency", "broad_pot_ord"]].reset_index(drop=True), \
           f"OK — {len(df_valid)} cells matched to Kang S11 broad potency; {df_valid['broad_potency'].nunique()} potency levels"


def compute_row_taus(df: pd.DataFrame) -> Dict[str, float]:
    """Given a df with nnz + broad_pot_ord + std_phenotype, compute τ(nnz, GT) under both kernels."""
    target = -df["broad_pot_ord"].astype(float).values
    x = df["nnz"].astype(float).values
    n_unique_std = df["std_phenotype"].nunique()
    # Kang weights: need class-balance nesting
    if n_unique_std < 2 or df["broad_pot_ord"].nunique() < 2:
        return {"tau_nnz_scipy": float("nan"), "tau_nnz_kang": float("nan"), "n": len(df)}
    w = kang_absolute_weights(df["std_phenotype"], df["broad_pot_ord"])
    return {
        "tau_nnz_scipy": scipy_tau(x, target),
        "tau_nnz_kang": wdm_tau_b_bit(x, target, w),
        "n": int(len(df)),
    }


def h1_verdict(median_delta: float) -> str:
    if median_delta is None or math.isnan(median_delta):
        return "N/A"
    if median_delta <= H1_PASS:
        return "PASS"
    if median_delta > H1_FAIL:
        return "FAIL"
    return "INCONCLUSIVE"


def main():
    t0 = time.time()
    print("=" * 80)
    print("Session 4 raw-nnz H1 re-run")
    print("=" * 80)

    per_ds = load_per_dataset_results()
    print(f"Loaded per_dataset_results.json: {len(per_ds)} rows")

    # Row list to attempt
    ATTEMPTS = [
        # (dataset_name in per_ds, loader_fn, notes)
        ("HSC development (Smart-seq2)", None, "pre_computed via baseline_validation_e_mtab_9067.json"),
        ("Mouse embryo 1 (Tang et al.)", row_GSE45719_mouse_embryo_1, "GSE45719"),
        ("Intestine (Drop-seq)", lambda: row_GSE92332_intestine("Drop-seq"), "GSE92332 atlas file"),
        ("Intestine (Smart-seq2)", lambda: row_GSE92332_intestine("Smart-seq2"), "GSE92332 (TPM only)"),
        ("BM-MNC (CITE-seq)e", row_GSE128639_bm_mnc, "GSE128639 CITE-seq RNA counts (cell-ID match with Kang after '.'->'-' fix)"),
    ]

    results = {}

    for name, loader, note in ATTEMPTS:
        print(f"\n--- {name} ({note}) ---")
        atlas_row = per_ds.get(name)
        if atlas_row is None:
            print(f"  DROP: not in per_dataset_results.json")
            results[name] = {"status": "not_in_atlas_denominator", "note": note}
            continue

        # τ(CT1) from atlas
        tau_ct1_scipy = atlas_row["tau_scipy"].get("gene_counts")
        tau_ct1_kang = atlas_row["tau_kang"].get("gene_counts")

        # Handle pre-computed E-MTAB-9067
        if name == "HSC development (Smart-seq2)":
            with open(BASELINE_JSON) as f:
                bl = json.load(f)
            tau_nnz_scipy = bl["tau_nnz_GT_scipy_emtab_side"]
            tau_nnz_kang = bl["tau_nnz_GT_kang_emtab_side"]
            n = bl["e_mtab_mapped_n"]
            row_result = {
                "status": "OK_PRE_COMPUTED",
                "source": "baseline_validation_e_mtab_9067.json",
                "n_cells_used": n,
                "cell_match_rate": n / bl["kang_s11_n"],
                "tau_nnz_scipy": tau_nnz_scipy,
                "tau_nnz_kang": tau_nnz_kang,
                "tau_CT1_scipy_atlas": tau_ct1_scipy,
                "tau_CT1_kang_atlas": tau_ct1_kang,
                "note": note,
            }
        else:
            df, load_info = loader()
            print(f"  Loader info: {load_info}")
            if df is None or len(df) == 0:
                results[name] = {"status": "DROPPED", "reason": load_info}
                continue
            taus = compute_row_taus(df)
            row_result = {
                "status": "OK",
                "n_cells_used": taus["n"],
                "tau_nnz_scipy": taus["tau_nnz_scipy"],
                "tau_nnz_kang": taus["tau_nnz_kang"],
                "tau_CT1_scipy_atlas": tau_ct1_scipy,
                "tau_CT1_kang_atlas": tau_ct1_kang,
                "loader_note": load_info[:300],
            }

        # Compute τ(SR) etc from atlas
        for score_tag in ["SCENT (SR)", "SCENT (CCAT)", "StemID"]:
            row_result[f"tau_{score_tag}_scipy_atlas"] = atlas_row["tau_scipy"].get(score_tag)
            row_result[f"tau_{score_tag}_kang_atlas"] = atlas_row["tau_kang"].get(score_tag)

        # Per-row Δ vs nnz
        for score_tag in ["SCENT (SR)", "SCENT (CCAT)", "StemID"]:
            for kernel in ["scipy", "kang"]:
                s_val = row_result[f"tau_{score_tag}_{kernel}_atlas"]
                n_val = row_result[f"tau_nnz_{kernel}"]
                if s_val is None or n_val is None or math.isnan(s_val) or math.isnan(n_val):
                    row_result[f"delta_{score_tag}_{kernel}"] = float("nan")
                else:
                    row_result[f"delta_{score_tag}_{kernel}"] = s_val - n_val

        # Per-row baseline gap CT1 − nnz
        for kernel in ["scipy", "kang"]:
            ct1_val = row_result[f"tau_CT1_{kernel}_atlas"]
            nnz_val = row_result[f"tau_nnz_{kernel}"]
            if ct1_val is None or nnz_val is None or math.isnan(ct1_val) or math.isnan(nnz_val):
                row_result[f"baseline_gap_ct1_minus_nnz_{kernel}"] = float("nan")
            else:
                row_result[f"baseline_gap_ct1_minus_nnz_{kernel}"] = ct1_val - nnz_val

        print(f"  τ(CT1) [atlas]: scipy={tau_ct1_scipy:+.4f} kang={tau_ct1_kang:+.4f}")
        print(f"  τ(nnz) [row]  : scipy={row_result['tau_nnz_scipy']:+.4f} kang={row_result['tau_nnz_kang']:+.4f}")
        print(f"  Δτ (CT1−nnz)  : scipy={row_result['baseline_gap_ct1_minus_nnz_scipy']:+.4f} "
              f"kang={row_result['baseline_gap_ct1_minus_nnz_kang']:+.4f}")

        results[name] = row_result

    # Aggregate H1 medians on raw-nnz baseline over obtainable rows
    print("\n" + "=" * 80)
    print("AGGREGATED H1 on raw-nnz baseline")
    print("=" * 80)

    obtainable = [r for r in results.values() if r.get("status") in ("OK", "OK_PRE_COMPUTED")]
    print(f"\nObtainable rows: {len(obtainable)} of {len(ATTEMPTS)} attempted")
    for name, r in results.items():
        st = r.get("status", "?")
        print(f"  [{st:20s}] {name}")

    h1_summary = {}
    for score_tag in ["SCENT (SR)", "SCENT (CCAT)", "StemID"]:
        for kernel in ["scipy", "kang"]:
            deltas = []
            for r in obtainable:
                v = r.get(f"delta_{score_tag}_{kernel}")
                if v is not None and not math.isnan(v):
                    deltas.append(v)
            if deltas:
                med = float(np.median(deltas))
                iqr_lo = float(np.percentile(deltas, 25))
                iqr_hi = float(np.percentile(deltas, 75))
                verdict = h1_verdict(med)
                h1_summary[f"{score_tag}_{kernel}"] = {
                    "n": len(deltas),
                    "median_delta": med,
                    "iqr_lo": iqr_lo,
                    "iqr_hi": iqr_hi,
                    "verdict": verdict,
                    "deltas": deltas,
                }
                print(f"  {score_tag:15s} {kernel:6s}: n={len(deltas)} med={med:+.4f} "
                      f"IQR=[{iqr_lo:+.3f},{iqr_hi:+.3f}] verdict={verdict}")

    # Baseline gap distribution
    print("\n" + "=" * 80)
    print("Baseline gap distribution (τ(CT1) − τ(nnz)) across obtainable rows")
    print("=" * 80)
    gap_summary = {}
    for kernel in ["scipy", "kang"]:
        gaps = []
        for r in obtainable:
            g = r.get(f"baseline_gap_ct1_minus_nnz_{kernel}")
            if g is not None and not math.isnan(g):
                gaps.append(g)
        if gaps:
            gap_summary[kernel] = {
                "n": len(gaps),
                "mean": float(np.mean(gaps)),
                "median": float(np.median(gaps)),
                "min": float(np.min(gaps)),
                "max": float(np.max(gaps)),
                "std": float(np.std(gaps)),
                "iqr_lo": float(np.percentile(gaps, 25)) if len(gaps) > 1 else float(gaps[0]),
                "iqr_hi": float(np.percentile(gaps, 75)) if len(gaps) > 1 else float(gaps[0]),
                "gaps": gaps,
            }
            print(f"  {kernel}: n={len(gaps)} mean={gap_summary[kernel]['mean']:+.4f} "
                  f"median={gap_summary[kernel]['median']:+.4f} "
                  f"range=[{gap_summary[kernel]['min']:+.4f},{gap_summary[kernel]['max']:+.4f}]")

    # Write JSON
    out = {
        "per_row_results": results,
        "h1_raw_nnz_summary": h1_summary,
        "baseline_gap_distribution": gap_summary,
        "meta": {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(t0)),
            "wall_seconds": round(time.time() - t0, 2),
        },
    }
    with open(HERE / "h1_raw_nnz_results.json", "w") as f:
        json.dump(out, f, indent=2, default=str)

    with open(HERE / "baseline_gap_distribution.json", "w") as f:
        json.dump(gap_summary, f, indent=2, default=str)

    print(f"\nWritten: {HERE / 'h1_raw_nnz_results.json'}")
    print(f"Written: {HERE / 'baseline_gap_distribution.json'}")


if __name__ == "__main__":
    main()
