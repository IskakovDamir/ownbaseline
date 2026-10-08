"""
FIX 3 GATE 2 — Phase A prep on Briggs 2018 Xenopus (GSE113074).
Mirrors the zebrafish decider prep (track2/code/prep_data.py): stream-load to sparse,
build the 10-stage Nieuwkoop-Faber ordinal (score-independent morphology), QC,
Xenopus symbols → uppercase human orthologs on STRING v12 LCC (same disclosed
cross-species convention as the zebrafish run), primitives, per-stage depth.

Data format (GSE113074_Raw_combined.annotated_counts.tsv.gz):
  rows 1-9 = per-cell metadata (col1=label, cols2..=per-cell value); CELLS ARE COLUMNS.
    row7 Developmental_stage ∈ {Stage_8,10,11,12,13,14,16,18,20,22} (the 10-tier ordinal).
  rows 10+ = gene rows (col1=Xenopus gene symbol, cols2..=integer UMI counts).
"""
from __future__ import annotations
import sys, json, gzip, time, warnings, csv
from pathlib import Path
import numpy as np
import scipy.sparse as sp

warnings.filterwarnings("ignore")
# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# scratch directory, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root, repo_root  # noqa: E402
CODE = repo_root()   # this repository (code)
VAULT = data_root()  # run inputs and outputs (data)
sys.path.insert(0, str(CODE / "own_baseline"))
SRC = VAULT / "fix3/data/GSE113074_Raw_combined.annotated_counts.tsv.gz"
STRING = VAULT / "w4/scaffolds/human_string_v12_thr700_lcc.npz"
OUT = VAULT / "fix3/data/prepared"
OUT.mkdir(parents=True, exist_ok=True)

MIN_GENES = 200
CHUNK_CELLS = 4000
# Nieuwkoop-Faber stage → integer rank (early = high potency = rank 1)
STAGE_RANK = {"Stage_8": 1, "Stage_10": 2, "Stage_11": 3, "Stage_12": 4, "Stage_13": 5,
              "Stage_14": 6, "Stage_16": 7, "Stage_18": 8, "Stage_20": 9, "Stage_22": 10}


def main():
    t0 = time.time()
    print("Streaming Briggs annotated_counts ...")
    meta = {}
    genes, data_rows, indices_rows = [], [], []  # CSR-by-gene
    with gzip.open(SRC, "rt") as f:
        for li, line in enumerate(f):
            tab = line.find("\t")
            label = line[:tab]
            if li < 9:  # metadata rows
                meta[label] = np.array(line.rstrip("\n").split("\t")[1:])
                continue
            # gene row
            vals = np.fromstring(line[tab + 1:], dtype=np.int64, sep="\t")
            nz = np.nonzero(vals)[0]
            genes.append(label)
            indices_rows.append(nz.astype(np.int32))
            data_rows.append(vals[nz].astype(np.int32))
            if len(genes) % 5000 == 0:
                print(f"  {len(genes)} genes, {time.time()-t0:.0f}s")
    n_cells = len(meta["Developmental_stage"])
    n_genes = len(genes)
    print(f"  parsed {n_genes} genes x {n_cells} cells, {time.time()-t0:.0f}s")

    indptr = np.zeros(n_genes + 1, dtype=np.int64)
    for i, idx in enumerate(indices_rows):
        indptr[i + 1] = indptr[i] + len(idx)
    Xgc = sp.csr_matrix((np.concatenate(data_rows), np.concatenate(indices_rows), indptr),
                        shape=(n_genes, n_cells))
    X = Xgc.T.tocsr()                      # cells x genes
    del Xgc, data_rows, indices_rows
    genes = np.array(genes)

    # metadata disclosure
    stage_str = meta["Developmental_stage"]
    print("\nDevelopmental_stage counts:", dict(zip(*np.unique(stage_str, return_counts=True))))
    print("Replicate_name uniq:", np.unique(meta["Replicate_name"]).tolist())
    print("Library_name uniq (first 20):", np.unique(meta["Library_name"])[:20].tolist())
    # dissected neural-plate-border cells: the NPB_dissection replicate is a
    # targeted anatomical dissection (all at Stage_11), NOT whole-embryo → exclude per §2(3).
    # (The dissection label is in Replicate_name, not Library_name.)
    rep = meta["Replicate_name"]
    dissect = np.array(["dissect" in r.lower() or "npb" in r.lower() for r in rep])
    print(f"  dissected (non-whole-embryo) cells excluded: {dissect.sum()} "
          f"(replicates: {np.unique(rep[dissect]).tolist() if dissect.sum() else 'none'})")

    rank = np.array([STAGE_RANK.get(s, np.nan) for s in stage_str], dtype=float)

    gene_count_all = np.asarray((X > 0).sum(axis=1)).ravel()
    keep = (gene_count_all >= MIN_GENES) & np.isfinite(rank) & (~dissect)
    print(f"\nQC min_genes={MIN_GENES}: kept {keep.sum()}/{len(keep)} "
          f"(dropped {(gene_count_all < MIN_GENES).sum()} low-gene, "
          f"{(~np.isfinite(rank)).sum()} unstaged, {dissect.sum()} dissected)")
    X = X[keep].tocsr()
    rank = rank[keep]; stage_str = stage_str[keep]
    n_cells = X.shape[0]
    barcode = meta["Barcode_name"][keep] if "Barcode_name" in meta else np.arange(n_cells).astype(str)

    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    gene_count = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float64)

    # ---- primitives ----
    print("\nprimitives: Shannon H, PCC(x,degree) STRING v12 ...")
    shannon = np.empty(n_cells)
    ip, dt = X.indptr, X.data.astype(np.float64)
    for c in range(n_cells):
        seg = dt[ip[c]:ip[c+1]]; s = seg.sum()
        shannon[c] = (-(seg/s * np.log(seg/s)).sum()) if s > 0 else np.nan

    d = np.load(STRING, allow_pickle=True)
    sgenes = np.array([str(g) for g in d["genes"]]); sdeg = np.asarray(d["degree"], float)
    sidx = {g: i for i, g in enumerate(sgenes)}
    genes_up = np.array([g.upper() for g in genes])       # Xenopus → uppercase human ortholog
    col_atlas, col_string = [], []
    seen = set()
    for i, g in enumerate(genes_up):
        j = sidx.get(g)
        if j is not None and g not in seen:               # first occurrence per symbol
            col_atlas.append(i); col_string.append(j); seen.add(g)
    col_atlas = np.array(col_atlas); col_string = np.array(col_string)
    deg_sub = sdeg[col_string]
    print(f"  atlas ∩ STRING v12 (uppercase ortholog match) = {len(col_atlas)} genes "
          f"(of {n_genes}; deg mean {deg_sub.mean():.1f})")

    target_sum = float(np.median(lib[lib > 0]))
    Xsub = X[:, col_atlas].tocsr()
    kc = deg_sub - deg_sub.mean(); kc_norm = np.sqrt((kc**2).sum())
    pcc = np.empty(n_cells)
    for s0 in range(0, n_cells, CHUNK_CELLS):
        s1 = min(s0 + CHUNK_CELLS, n_cells)
        block = Xsub[s0:s1].toarray().astype(np.float64)
        tot = lib[s0:s1][:, None]; tot = np.where(tot > 0, tot, 1.0)
        Xn = np.log1p(block / tot * target_sum)
        Rc = Xn - Xn.mean(axis=1, keepdims=True)
        den = np.sqrt((Rc**2).sum(axis=1)) * kc_norm
        out = np.zeros(s1 - s0); ok = den > 0
        out[ok] = (Rc @ kc)[ok] / den[ok]
        pcc[s0:s1] = out
    print(f"  PCC mean {np.nanmean(pcc):+.4f}")

    # ---- depth per stage ----
    depth = {}
    for s, rk in STAGE_RANK.items():
        m = stage_str == s
        if m.sum() == 0: continue
        depth[s] = {"rank": rk, "n_cells": int(m.sum()),
                    "median_gene_count": float(np.median(gene_count[m])),
                    "median_library_size": float(np.median(lib[m]))}

    # ---- save (same keys as zebrafish prep) ----
    sp.save_npz(OUT / "X_cells_genes.npz", X)
    (OUT / "features.tsv").write_text("\n".join(map(str, genes)))
    (OUT / "genes.txt").write_text("\n".join(map(str, genes)))
    np.savez(OUT / "primitives.npz", gene_count=gene_count, pcc_string_v12=pcc,
             shannon_H=shannon, rank_ordinal=rank)
    np.savez(OUT / "string_cols.npz", col_atlas=col_atlas, col_string=col_string, deg_sub=deg_sub)
    with open(OUT / "cells.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cell_id", "stage", "rank_ordinal", "lib_size", "gene_count"])
        for i in range(n_cells):
            w.writerow([barcode[i], stage_str[i], int(rank[i]), int(lib[i]), int(gene_count[i])])
    with open(OUT / "depth_per_stage.json", "w") as f:
        json.dump({"min_genes_qc": MIN_GENES, "n_cells_post_qc": n_cells, "n_genes": int(n_genes),
                   "n_atlas_x_string": int(len(col_atlas)), "target_sum_cpm": target_sum,
                   "per_stage": depth}, f, indent=2)

    print("\n=== DEPTH PER STAGE (Briggs 10-tier) ===")
    for s in sorted(depth, key=lambda x: depth[x]["rank"]):
        z = depth[s]
        print(f"  r{z['rank']:2d} {s:9s} n={z['n_cells']:6d} med_genes={z['median_gene_count']:6.0f} "
              f"med_lib={z['median_library_size']:7.0f}")
    print(f"\nprepared → {OUT}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
