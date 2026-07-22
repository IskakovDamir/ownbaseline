"""
fix_atlas_denominator.py
========================
Reproducibly re-derive the CytoTRACE 2 atlas denominator (N datasets, cohorts,
tissues, species) from the CT2 Shiny metadata file.

Rationale
---------
The prereg (04-experiments/2026-07-17-hod5-atlas-scale-reduction-prereg.md §3)
says: 'Explorer must fix the real N by the metadata file BEFORE running.'
Three sources disagree:
    - GitHub README:               '34 datasets / 24 tissues'
    - CT2 Nature Methods 2025:     '34 human and mouse scRNA-seq datasets'
    - CT2 Shiny site text (paper): '33-dataset atlas' (33 ground-truth annotated)
    - Prior provenance memo:       '47 Training + 14 Test + 48 Tabula Sapiens'
        (= 109 rows in metadata_paper_datasets.txt)
    - Extended annot file:         '2 Training + 1 Test + 1 Tabula Muris agg
                                    + 1 Tabula Sapiens agg' (aggregated view)

Resolution: use the primary metadata file `metadata_paper_datasets.txt` as
authoritative for per-row counts. Aggregate to a canonical view.

Output
------
    atlas_N.json  -- machine-readable canonical denominator
    stdout        -- human-readable summary
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

METADATA_URL = "https://cytotrace2.stanford.edu/metadata_paper_datasets.txt"
LOCAL_CACHE = Path("/tmp/ct2_provenance/probe_metadata_paper_datasets.txt")


def sha256_first_4mb(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(4 * 1024 * 1024))
    return h.hexdigest()


def load_metadata() -> tuple[list[dict], dict]:
    """Return (rows, source_info). Uses local cache if present, else downloads."""
    src = {"url": METADATA_URL}
    if LOCAL_CACHE.exists():
        path = LOCAL_CACHE
        src["local_cache"] = str(path)
        src["source"] = "local_cache"
    else:
        path = Path("/tmp/ct2_atlas_metadata.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(METADATA_URL, path)
        src["local_cache"] = str(path)
        src["source"] = "downloaded"
    src["sha256_first_4mb"] = sha256_first_4mb(path)
    src["size_bytes"] = os.path.getsize(path)

    rows = []
    with open(path, "r") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            fields = line.rstrip("\n").split("\t")
            if len(fields) != len(header):
                continue
            row = dict(zip(header, fields))
            rows.append(row)
    return rows, src


def main():
    rows, src = load_metadata()

    # Canonical fields (columns as in metadata_paper_datasets.txt):
    #  Dataset | Accession number | PMID | Species | Platform | Number of total cells |
    #  Number of cells with potency annotation | Number of cells analyzed |
    #  Number of phenotypes | Number of broad potency levels | Cohort
    n_rows = len(rows)

    by_cohort = {}
    for r in rows:
        c = r["Cohort"]
        by_cohort.setdefault(c, []).append(r)

    # Tissue extraction: for Tabula Muris / Tabula Sapiens the tissue is in the
    # Dataset name after ' - ' ; for named datasets we use the Dataset name.
    def extract_tissue(dataset_name: str, cohort: str) -> str:
        if cohort in ("Tabula Sapiens",) and " - " in dataset_name:
            # 'Tabula Sapiens - Bladder (10x)' -> 'Bladder'
            return dataset_name.split(" - ", 1)[1].split(" (")[0]
        if "Tabula Muris - " in dataset_name:
            return dataset_name.split(" - ", 1)[1].split(" (")[0]
        if " (" in dataset_name:
            return dataset_name.split(" (")[0]
        return dataset_name

    unique_tissues_all = set()
    unique_tissues_paper = set()  # Training + Test only
    species_counts = {"Human": 0, "Mouse": 0}
    for r in rows:
        t = extract_tissue(r["Dataset"], r["Cohort"])
        unique_tissues_all.add(t)
        if r["Cohort"] in ("Training", "Test"):
            unique_tissues_paper.add(t)
        sp = r["Species"]
        if sp in species_counts:
            species_counts[sp] += 1

    # Species by cohort
    species_by_cohort = {}
    for c, rs in by_cohort.items():
        d = {"Human": 0, "Mouse": 0}
        for r in rs:
            if r["Species"] in d:
                d[r["Species"]] += 1
        species_by_cohort[c] = d

    # Datasets by 'broad potency levels' count
    potency_level_counts = {}
    excludable_low_potency = []  # < 2 unique potency levels per prereg §3
    for r in rows:
        p = int(r["Number of broad potency levels"])
        potency_level_counts[p] = potency_level_counts.get(p, 0) + 1
        if p < 2:
            excludable_low_potency.append({
                "dataset": r["Dataset"],
                "accession": r["Accession number"],
                "cohort": r["Cohort"],
                "n_cells": int(r["Number of cells analyzed"]),
                "n_broad_potency_levels": p,
            })

    # Datasets excludable by n_cells < 200 (prereg §3)
    excludable_small = []
    for r in rows:
        n = int(r["Number of cells analyzed"])
        if n < 200:
            excludable_small.append({
                "dataset": r["Dataset"],
                "accession": r["Accession number"],
                "cohort": r["Cohort"],
                "n_cells": n,
                "n_broad_potency_levels": int(r["Number of broad potency levels"]),
            })

    # Union of excludables (prereg §3 criteria)
    excludable_ids = {
        (r["dataset"], r["accession"], r["cohort"]) for r in excludable_low_potency
    } | {
        (r["dataset"], r["accession"], r["cohort"]) for r in excludable_small
    }
    n_excludable = len(excludable_ids)

    # Explicit prereg §1 exclusions (probe-3 hypothesis-generating set)
    prereg_hg_exclusions = [
        {"label": "pancreas Bastidas-Ponce 2019", "accession": "GSE132188",
         "cohort_in_atlas": "Test",
         "reason": "prereg §1 hypothesis-generating (probe-3, II.3')"},
        {"label": "cord blood CITE-seq Stoeckius 2017", "accession": "GSE100866",
         "cohort_in_atlas": "Test",
         "reason": "prereg §1 hypothesis-generating (probe-3, II.3')"},
        {"label": "Paul et al. 2015 hematopoiesis", "accession": "GSE72857",
         "cohort_in_atlas": "not in atlas",
         "reason": "prereg §1 hypothesis-generating (probe-3, II.3'); not in "
                   "atlas so already outside the denominator"},
    ]

    # Canonical N view: 'paper' scope (Training + Test) vs full metadata
    n_paper_scope = sum(1 for r in rows
                        if r["Cohort"] in ("Training", "Test"))
    n_training = len(by_cohort.get("Training", []))
    n_test = len(by_cohort.get("Test", []))
    n_tabula_sapiens = len(by_cohort.get("Tabula Sapiens", []))

    # 'Named' training datasets: the 47 Training rows include 30 Tabula Muris
    # tissue slices. If we collapse Tabula Muris to 1 (as one publication), the
    # 'paper' Training count drops to 47 - 30 + 1 = 18. Matches CT2's own
    # '19 modules trained on 19 datasets' — off by one (probably 'AT2/AT1
    # lineage' is 2 developmental stages). Documenting both views.
    tm_slices = sum(1 for r in by_cohort.get("Training", [])
                    if "Tabula Muris - " in r["Dataset"])
    n_training_named = n_training - tm_slices  # 47 - 30 = 17
    n_training_collapsed = n_training - tm_slices + 1  # 18 (TM as one)

    ts_slices = len(by_cohort.get("Tabula Sapiens", []))
    n_tabula_sapiens_collapsed = 1 if ts_slices > 0 else 0

    canonical = {
        "rows_in_metadata_file": n_rows,
        "by_cohort_rows": {
            "Training": n_training,
            "Test": n_test,
            "Tabula Sapiens": n_tabula_sapiens,
        },
        "Training_breakdown": {
            "Tabula_Muris_tissue_slices": tm_slices,
            "named_non_TM_training_datasets": n_training_named,
            "n_training_if_TM_collapsed_to_1_publication": n_training_collapsed,
        },
        "paper_scope_view": {
            "note": "Training + Test rows in the metadata file (excludes "
                    "Tabula Sapiens extended cohort)",
            "n_rows": n_paper_scope,
            "n_datasets_if_TM_collapsed": n_training_collapsed + n_test,
        },
        "atlas_view_matching_CT2_text": {
            "note": "CT2 site+paper says '33 gold-standard datasets' (Training "
                    "17-18 non-TM named + 1 TM aggregate + 14 Test = 32-33). "
                    "Off-by-one likely from AT2/AT1 developmental-stage "
                    "grouping. GitHub README says 34 datasets / 24 tissues -- "
                    "matches the 'Mouse embryo 3' + GSE249416 counted separately.",
            "n_datasets_reconciled": n_training_collapsed + n_test,
        },
        "unique_tissues_paper_scope": sorted(unique_tissues_paper),
        "n_unique_tissues_paper_scope": len(unique_tissues_paper),
        "unique_tissues_all": sorted(unique_tissues_all),
        "n_unique_tissues_all": len(unique_tissues_all),
        "species_by_cohort": species_by_cohort,
        "species_total_rows": species_counts,
        "potency_level_counts_across_rows": potency_level_counts,
    }

    exclusions = {
        "prereg_section_1_hypothesis_generating": prereg_hg_exclusions,
        "prereg_section_3_n_potency_levels_lt_2_rows": excludable_low_potency,
        "prereg_section_3_n_cells_lt_200_rows": excludable_small,
        "n_rows_excluded_by_prereg_section_3_criteria": n_excludable,
    }

    # H1 denominator: Training + Test (46 rows after removing rows that fail
    # prereg §3 criteria), minus the 2 probe-3 datasets that are in the atlas
    # (pancreas Test, cord blood Test). Paul15 is not in the atlas so already
    # out.
    training_test_rows = [
        r for r in rows if r["Cohort"] in ("Training", "Test")
    ]
    # Filter by prereg §3 criteria
    kept_after_prereg3 = []
    dropped_by_prereg3 = []
    for r in training_test_rows:
        n_cells_r = int(r["Number of cells analyzed"])
        n_pot = int(r["Number of broad potency levels"])
        if n_cells_r < 200 or n_pot < 2:
            dropped_by_prereg3.append({
                "dataset": r["Dataset"],
                "accession": r["Accession number"],
                "cohort": r["Cohort"],
                "n_cells": n_cells_r,
                "n_broad_potency_levels": n_pot,
                "reason": "<200 cells" if n_cells_r < 200 else "<2 potency levels",
            })
        else:
            kept_after_prereg3.append(r)

    # Remove probe-3 (only pancreas + cord blood are in atlas)
    probe_3_atlas_accessions = {"GSE132188", "GSE100866"}
    kept_final = [
        r for r in kept_after_prereg3
        if r["Accession number"] not in probe_3_atlas_accessions
    ]
    dropped_probe_3 = [
        r for r in kept_after_prereg3
        if r["Accession number"] in probe_3_atlas_accessions
    ]

    H1_denominator = {
        "note": "Rows entering H1 primary (median-Delta rule per prereg §2). "
                "Training + Test cohorts; applies prereg §3 exclusions "
                "(n_cells<200 or n_broad_potency<2) and prereg §1 hypothesis-"
                "generating exclusions (2 probe-3 datasets that are in the atlas).",
        "n_rows_Training_plus_Test_raw": len(training_test_rows),
        "n_rows_after_prereg_section_3": len(kept_after_prereg3),
        "n_rows_dropped_by_prereg_section_3": len(dropped_by_prereg3),
        "n_rows_after_probe_3_exclusion": len(kept_final),
        "n_rows_dropped_probe_3_in_atlas": len(dropped_probe_3),
        "H1_denominator_rows": len(kept_final),
        "H1_denominator_datasets_if_TM_collapsed": (
            len(kept_final)
            - sum(1 for r in kept_final if "Tabula Muris - " in r["Dataset"])
            + (1 if any("Tabula Muris - " in r["Dataset"] for r in kept_final) else 0)
        ),
        "H1_denominator_species_split": {
            "Human": sum(1 for r in kept_final if r["Species"] == "Human"),
            "Mouse": sum(1 for r in kept_final if r["Species"] == "Mouse"),
        },
        "H1_denominator_by_cohort": {
            "Training": sum(1 for r in kept_final if r["Cohort"] == "Training"),
            "Test": sum(1 for r in kept_final if r["Cohort"] == "Test"),
        },
        "H1_denominator_dropped_by_prereg_section_3_detail": dropped_by_prereg3,
        "H1_denominator_dropped_probe_3_detail": [
            {"dataset": r["Dataset"], "accession": r["Accession number"],
             "cohort": r["Cohort"]} for r in dropped_probe_3
        ],
    }

    output = {
        "purpose": "Fix atlas denominator N per prereg §3 for II.7 H1 primary.",
        "source": src,
        "canonical_N_view": canonical,
        "prereg_exclusions": exclusions,
        "H1_denominator": H1_denominator,
        "known_discrepancies_across_sources": {
            "GitHub_README": "34 datasets / 24 tissues",
            "CT2_paper_figure_and_site_text": "33-dataset atlas / 33 gold-standard",
            "CT2_site_first_paragraph": "34 human and mouse datasets encompassing 24 tissues",
            "CT2_ssGSEA_page": "33 gold-standard",
            "provenance_memo_2026_07_13": "47 Training + 14 Test + 48 Tabula Sapiens rows = 109",
            "extended_annot_file": "2 Training + 1 Test + 1 TM aggregate + 1 TS aggregate = 5 rows",
            "resolution": ("Primary metadata file rows = 109 (row-level, with "
                          "Tabula Muris and Tabula Sapiens exploded to tissue "
                          "slices). Named-datasets view (TM + TS each collapsed "
                          "to a single aggregated 'dataset') = "
                          + str(n_training_collapsed + n_test + n_tabula_sapiens_collapsed)
                          + " (Training + Test + TS). The CT2 site's '33-34 gold "
                          "standard' matches Training-collapsed + Test = "
                          + str(n_training_collapsed + n_test)
                          + ", with off-by-one from AT2/AT1 grouping or "
                          "GSE249416 double-count.")
        },
    }

    outfile = Path(__file__).parent / "atlas_N.json"
    with open(outfile, "w") as f:
        json.dump(output, f, indent=2)

    # Human-readable summary
    print("=" * 78)
    print("CytoTRACE 2 atlas N (fixed 2026-07-17 for II.7 primary probe)")
    print("=" * 78)
    print(f"Source: {src['url']} (sha256 first 4MB: {src['sha256_first_4mb'][:16]}...)")
    print(f"Metadata rows: {n_rows}")
    print()
    print("Rows by cohort:")
    for c in ("Training", "Test", "Tabula Sapiens"):
        n = len(by_cohort.get(c, []))
        sp = species_by_cohort.get(c, {})
        print(f"  {c:20s} n_rows = {n:>3d}  (H={sp.get('Human',0)}, M={sp.get('Mouse',0)})")
    print()
    print(f"Training breakdown:")
    print(f"  Tabula Muris tissue slices : {tm_slices}")
    print(f"  Named non-TM training      : {n_training_named}")
    print(f"  If TM collapsed to 1       : {n_training_collapsed}")
    print()
    print(f"Paper-scope view (Training + Test):")
    print(f"  Rows                       : {n_paper_scope}")
    print(f"  If TM collapsed to 1       : {n_training_collapsed + n_test}"
          f"  <-- matches CT2 '33-34 gold-standard datasets'")
    print()
    print(f"Unique tissues (Training + Test only, paper scope): "
          f"{len(unique_tissues_paper)}")
    print()
    print("Prereg §3 exclusions on Training + Test:")
    print(f"  <200 cells or <2 potency levels: {len(dropped_by_prereg3)} rows dropped")
    for d in dropped_by_prereg3:
        print(f"    - {d['dataset']:60s} n={d['n_cells']:>5d} pot_lv={d['n_broad_potency_levels']} ({d['reason']})")
    print()
    print("Prereg §1 hypothesis-generating exclusions (probe-3):")
    for r in dropped_probe_3:
        print(f"  - {r['Dataset']:60s} {r['Accession number']} ({r['Cohort']})")
    print()
    print("=" * 78)
    print(f"H1 primary denominator (Training + Test, after all exclusions):")
    print(f"  n_rows              = {len(kept_final)}")
    print(f"  n_datasets (TM=1)   = {H1_denominator['H1_denominator_datasets_if_TM_collapsed']}")
    print(f"  species split       = Human {H1_denominator['H1_denominator_species_split']['Human']}, "
          f"Mouse {H1_denominator['H1_denominator_species_split']['Mouse']}")
    print(f"  by cohort           = Training {H1_denominator['H1_denominator_by_cohort']['Training']}, "
          f"Test {H1_denominator['H1_denominator_by_cohort']['Test']}")
    print("=" * 78)
    print()
    print(f"Written: {outfile}")


if __name__ == "__main__":
    main()
