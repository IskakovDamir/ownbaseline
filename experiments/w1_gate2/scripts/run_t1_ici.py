"""T1 ICI arm — Stem.Sig (Zhang 2022) → response prediction.

LOCK 1: same metric (AUROC), same features (per-sample score), same
        split (per-cohort AUROC) for both Stem.Sig and library-size baseline.
LOCK 2: use released Stem.Sig gene set from Zhang 2022 Table S6.

Cohort coverage this session:
  - Hugo 2016 GSE78220 (SKCM anti-PD-1)   ← attempted here
  - Riaz 2017 GSE91061 (SKCM anti-PD-1)   ← attempted here (Pre only)
  - Mariathasan 2018 IMvigor210          ← EGA controlled access, NOT-COMPUTABLE
  - Gide 2019, Braun 2020, Liu 2019, Van Allen 2015, Kim 2018,
    Zhao 2019, Snyder 2017                ← scattered across dbGaP/EGA/GEO,
                                            not downloaded this session,
                                            reported as NOT-COMPUTABLE

Metric: AUROC of the score on RECIST-based responder label (CR+PR = R,
        SD+PD = NR). AUROC computed independently per cohort (Zhang used
        train/test split on pooled cohorts + LOO; per-cohort AUROC is the
        canonical single-cohort read).
"""

from __future__ import annotations
import gzip, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

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
DATA = data_root() / "w1_gate2/data"
RESULTS = data_root() / "w1_gate2/results"


def stem_sig() -> list[str]:
    return json.loads((DATA / "stem_sig_genes.json").read_text())


# -------- Hugo 2016 (GSE78220) ---------------------------------

def hugo_load():
    """FPKM matrix + response labels from GEO series metadata."""
    fpkm = pd.read_excel(DATA / "ici" / "GSE78220_PatientFPKM.xlsx",
                         sheet_name="FPKM")
    fpkm = fpkm.set_index("Gene")
    # Keep baseline only (Pt16 is OnTx — drop, per pre-treatment ICI protocol)
    baseline_cols = [c for c in fpkm.columns if c.endswith(".baseline")]
    fpkm = fpkm[baseline_cols]
    fpkm.columns = [c.replace(".baseline", "") for c in fpkm.columns]

    # Labels from series matrix (Complete/Partial Response = R, PD = NR)
    labels_map = {
        # Extracted from series matrix Sample_source_name_ch1
        "Pt1": "PD", "Pt2": "PR", "Pt4": "PR", "Pt5": "PR", "Pt6": "PR",
        "Pt7": "PD", "Pt8": "CR", "Pt9": "CR", "Pt10": "PD", "Pt12": "PD",
        "Pt13": "CR", "Pt14": "PD", "Pt15": "PR", "Pt16": "PD", "Pt19": "PR",
        "Pt20": "PD", "Pt22": "PD", "Pt23": "PD", "Pt25": "PD", "Pt27A": "CR",
        "Pt27B": "CR", "Pt28": "PR", "Pt29": "PD", "Pt31": "PD", "Pt32": "PD",
        "Pt35": "PR", "Pt37": "PR", "Pt38": "PR",
    }
    resp = {p: (1 if labels_map[p] in ("CR", "PR") else 0)
            for p in fpkm.columns}
    return fpkm, pd.Series(resp)


def hugo_run():
    fpkm, resp = hugo_load()
    stem = stem_sig()

    # log2(FPKM+1) then z-score per gene
    expr = np.log2(fpkm.astype(float) + 1.0)
    genes_present = [g for g in stem if g in expr.index]
    print(f"  Hugo Stem.Sig genes in matrix: {len(genes_present)}/{len(stem)}")
    sub = expr.loc[genes_present]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, 1), axis=0)
    stem_score = z.mean(axis=0)
    lib = np.log10(fpkm.sum(axis=0) + 1)  # per-sample total FPKM as depth proxy

    # AUROC
    resp_aligned = resp.loc[stem_score.index]
    auc_stem = roc_auc_score(resp_aligned, stem_score.values)
    auc_lib = roc_auc_score(resp_aligned, lib.values)
    # Zhang's Stem.Sig on responders should have LOWER stemness (paper claims
    # stemness ↑ → ICI resistance ↑). So we may need to invert. Report both.
    auc_stem_inv = roc_auc_score(resp_aligned, -stem_score.values)
    auc_lib_inv = roc_auc_score(resp_aligned, -lib.values)
    print(f"  Hugo n={len(resp_aligned)}, responders={int(resp_aligned.sum())}")
    print(f"  Stem.Sig AUC={auc_stem:.3f} (inverted: {auc_stem_inv:.3f})")
    print(f"  library AUC={auc_lib:.3f} (inverted: {auc_lib_inv:.3f})")

    # Use max of AUC and inverse-AUC as effective skill (paper direction is
    # published as stemness→resistance, so responder=1 should have LOWER score;
    # AUROC(-score, y) captures that).
    return {
        "cohort": "Hugo_GSE78220",
        "n": int(len(resp_aligned)),
        "responders": int(resp_aligned.sum()),
        "stem_genes_in_matrix": len(genes_present),
        "auc_stemsig_direct": float(auc_stem),
        "auc_stemsig_inverted_matches_paper": float(auc_stem_inv),
        "auc_libsize_direct": float(auc_lib),
        "auc_libsize_inverted": float(auc_lib_inv),
    }


# -------- Riaz 2017 (GSE91061) ---------------------------------

def riaz_load():
    """Raw counts (Entrez IDs) + response labels via series matrix meta.
    Series matrix stores response codes per GSM — we parse GEO SOFT."""
    with gzip.open(DATA / "ici" / "GSE91061_raw.csv.gz", "rt") as f:
        counts = pd.read_csv(f, index_col=0)
    counts.index = counts.index.astype(str)
    # Keep Pre-treatment samples only
    pre_cols = [c for c in counts.columns if "_Pre_" in c]
    counts_pre = counts[pre_cols]
    # Patient ID = Pt<N>
    counts_pre.columns = [c.split("_")[0] for c in counts_pre.columns]

    # Response: parse series matrix. Riaz encodes both visit
    # ("visit (pre or on treatment)") and "response" per sample.
    # Sample_title is patient_visit_barcode, e.g. "Pt1_Pre_AD101148-6".
    # We match on sample title stem (patient_Pre) to the counts columns.
    with gzip.open(DATA / "ici" / "GSE91061_matrix.txt.gz", "rt") as f:
        smraw = f.read()
    lines = smraw.split("\n")
    titles = []
    visits = []
    responses = []
    for L in lines:
        if L.startswith('!Sample_title'):
            titles = [x.strip('"') for x in L.split("\t")[1:]]
        if L.startswith('!Sample_characteristics_ch1'):
            fields = [x.strip('"') for x in L.split("\t")[1:]]
            if not fields:
                continue
            head = fields[0].split(": ", 1)[0].lower() if ": " in fields[0] else ""
            if "visit" in head:
                visits = [x.split(": ", 1)[-1] for x in fields]
            elif head == "response":
                responses = [x.split(": ", 1)[-1] for x in fields]
    # Build title -> response map, but restricted to Pre-treatment visits
    title_to_resp = {}
    for t, v, r in zip(titles, visits, responses):
        if v.strip() == "Pre":
            # Patient ID (before first underscore) is the counts-column key
            title_to_resp[t.split("_")[0]] = r
    return counts_pre, title_to_resp


def riaz_run():
    counts, p2r = riaz_load()
    # Load Entrez -> symbol via mygene (cached across sessions)
    import requests, time, json as _json
    cache_path = DATA / "ici" / "entrez2sym.json"
    if cache_path.exists():
        sym_for_entrez = _json.loads(cache_path.read_text())
    else:
        sym_for_entrez = {}
        entrez_ids = counts.index.tolist()
        for i in range(0, len(entrez_ids), 500):
            chunk = entrez_ids[i:i+500]
            for attempt in range(3):
                try:
                    r = requests.post("https://mygene.info/v3/query",
                        data={"q": ",".join(chunk), "scopes": "entrezgene",
                              "fields": "symbol", "species": "human"},
                        timeout=120)
                    if r.status_code == 200:
                        break
                except Exception:
                    time.sleep(3)
            if r.status_code != 200:
                continue
            for h in r.json():
                s = h.get("symbol")
                q = h.get("query")
                if s and q and "notfound" not in h:
                    sym_for_entrez.setdefault(str(q), s)
            time.sleep(0.5)
        cache_path.write_text(_json.dumps(sym_for_entrez))
    # Map counts index to symbol
    idx_sym = pd.Series({k: sym_for_entrez.get(k) for k in counts.index})
    counts2 = counts.copy()
    counts2["_sym"] = idx_sym.values
    counts2 = counts2.dropna(subset=["_sym"])
    # Collapse duplicate symbols by sum
    counts2 = counts2.groupby("_sym").sum()

    stem = stem_sig()
    genes_present = [g for g in stem if g in counts2.index]
    print(f"  Riaz Stem.Sig genes in matrix: {len(genes_present)}/{len(stem)}")

    # Normalize: log2(CPM+1); then z-score
    lib_raw = counts2.sum(axis=0)
    cpm = counts2.div(lib_raw / 1e6, axis=1)
    log_cpm = np.log2(cpm + 1)
    sub = log_cpm.loc[genes_present]
    z = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0, 1), axis=0)
    stem_score = z.mean(axis=0)
    lib_baseline = np.log10(lib_raw + 1)

    # Build responder labels. Riaz encodes as PRCR (combined), SD, PD, UNK.
    resp = {}
    for c in stem_score.index:
        r = p2r.get(c)
        if r in ("CR", "PR", "PRCR"):
            resp[c] = 1
        elif r in ("PD", "SD"):
            resp[c] = 0
    resp_s = pd.Series(resp)
    common = stem_score.index.intersection(resp_s.index)
    print(f"  Riaz common samples with label: {len(common)}, responders={int(resp_s.loc[common].sum())}")

    if len(common) < 10:
        return {"cohort": "Riaz_GSE91061", "error": "too few labeled samples"}

    y = resp_s.loc[common].values
    ss = stem_score.loc[common].values
    lb = lib_baseline.loc[common].values

    auc_stem = roc_auc_score(y, ss)
    auc_stem_inv = roc_auc_score(y, -ss)
    auc_lib = roc_auc_score(y, lb)
    auc_lib_inv = roc_auc_score(y, -lb)

    print(f"  Riaz n={len(common)}, responders={int(y.sum())}")
    print(f"  Stem.Sig AUC={auc_stem:.3f} (inverted: {auc_stem_inv:.3f})")
    print(f"  library AUC={auc_lib:.3f} (inverted: {auc_lib_inv:.3f})")

    return {
        "cohort": "Riaz_GSE91061",
        "n": int(len(common)),
        "responders": int(y.sum()),
        "stem_genes_in_matrix": len(genes_present),
        "auc_stemsig_direct": float(auc_stem),
        "auc_stemsig_inverted_matches_paper": float(auc_stem_inv),
        "auc_libsize_direct": float(auc_lib),
        "auc_libsize_inverted": float(auc_lib_inv),
    }


def main():
    print("===== T1 ICI arm =====")
    out = {}
    print("\n--- Hugo 2016 (GSE78220) ---")
    try:
        out["hugo"] = hugo_run()
    except Exception as e:
        out["hugo"] = {"error": str(e)}
        print(f"  Hugo failed: {e}")

    print("\n--- Riaz 2017 (GSE91061) ---")
    try:
        out["riaz"] = riaz_run()
    except Exception as e:
        out["riaz"] = {"error": str(e)}
        print(f"  Riaz failed: {e}")

    print("\n--- Others (NOT-COMPUTABLE this session) ---")
    others = ["Mariathasan_IMvigor210_EGAS00001004386",
              "Gide_2019_ENA_PRJEB23709",
              "Braun_2020_dbGaP",
              "Liu_2019_dbGaP",
              "Van_Allen_2015_dbGaP",
              "Kim_2018_ENA",
              "Zhao_2019_dbGaP",
              "Snyder_2017"]
    for o in others:
        print(f"  {o}: NOT-COMPUTABLE (controlled access or scattered; not downloaded this session)")
    out["not_computable"] = others

    # AUROC skill delta (paper direction is stem_inv/lib_inv)
    for k in ("hugo", "riaz"):
        r = out.get(k, {})
        if "auc_stemsig_inverted_matches_paper" in r:
            delta = r["auc_stemsig_inverted_matches_paper"] - r["auc_libsize_inverted"]
            r["delta_stem_minus_lib_paper_direction"] = float(delta)
            if abs(delta) <= 0.05:
                r["verdict"] = "TAUTOLOG"
            elif delta > 0.10:
                r["verdict"] = "SURVIVES"
            elif delta > 0.05:
                r["verdict"] = "inconclusive"
            elif delta < -0.10:
                r["verdict"] = "raw-beats-stemness"
            else:
                r["verdict"] = "inconclusive_neg"

    (RESULTS / "T1_ici_partial.json").write_text(json.dumps(out, indent=2))
    print("\n===== SUMMARY =====")
    for k in ("hugo", "riaz"):
        r = out.get(k, {})
        if "delta_stem_minus_lib_paper_direction" in r:
            print(f"  {k}: n={r['n']}  Δ={r['delta_stem_minus_lib_paper_direction']:+.3f}  verdict={r['verdict']}")


if __name__ == "__main__":
    main()
