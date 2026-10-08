"""
Track 2 Gate 2 — Phase A: data preparation on Farrell 2018 zebrafish embryogenesis.

ACCESSION RECONCILIATION (disclosed):
  Gate-1 atlas lock named GSE106587, but the downloaded GSE106587_RAW.tar holds
  only 2 samples (GSM3084402_10xWT6S, GSM3084403_10xMZoep6S) — both 6-somite,
  10x Genomics — and CANNOT form a 12-stage ordinal. The actual 12-stage Drop-seq
  atlas is GSE106474_UMICounts.txt.gz (Farrell 2018; the whole-embryo Drop-seq
  UMI table, cell IDs prefixed with stage code). We use GSE106474 and disclose
  the accession correction (GSE106587 is a 10x subseries of the same study).
  [FETCH: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE106474 · file on disk 2026-07-21]

STAGE ORDERING (disclosed):
  Biological Kimmel 1995 hpf order is used as PRIMARY:
    HIGH(1,3.3) < OBLONG(2,3.7) < DOME(3,4.3) < 30%(4,4.7) < 50%(5,5.3)
      < SHIELD(6,6.0) < 60%(7,6.7) < 75%(8,8.0) < 90%(9,9.0) < BUD(10,10)
      < 3-somite(11,11) < 6-somite(12,12).
  The Gate-1 lock had placed SHIELD at rank 10 (between bud and 3-somite) but
  flagged that itself as anomalous ("per GEO listing"). Shield stage is ~6 hpf,
  developmentally between 50% and 60% epiboly, so rank 6 is biologically correct.
  We keep the Gate-1 alt ordering (shield=10) as a pre-committed SENSITIVITY check.

Outputs under track2/data/prepared/:
  X_cells_genes.npz         scipy CSR (cells x genes), post-QC, int32 counts
  genes.txt                 gene symbols (columns of X)
  cells.tsv                 cell_id, stage_str, rank_kimmel, rank_gate1alt, hpf, lib_size, gene_count
  primitives.npz            gene_count, pcc_string_v12, shannon_H  (post-QC cell order)
  string_cols.npz          atlas col idx + string idx + degree for the STRING intersection
  depth_per_stage.json      per-stage n, median lib size, median gene count
  matrix.mtx / barcodes.tsv / features.tsv   MatrixMarket for R consumers (post-QC, genes x cells)
"""
from __future__ import annotations
import sys, json, gzip, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

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
import potency_metrics as pm

UMI = VAULT / "track2/data/GSE106474_UMICounts.txt.gz"
STRING = VAULT / "w4/scaffolds/human_string_v12_thr700_lcc.npz"
OUT = VAULT / "track2/data/prepared"
OUT.mkdir(parents=True, exist_ok=True)

MIN_GENES = 200          # QC: drop cells with fewer detected genes
CHUNK_GENES = 2000       # rows (genes) per pandas read chunk
CHUNK_CELLS = 4000       # cells per PCC chunk (bounds dense memory)

# Kimmel 1995 hpf ordering (PRIMARY). shield=6.
RANK_KIMMEL = {
    "ZFHIGH": 1, "ZFOBLONG": 2, "ZFDOME": 3, "ZF30": 4, "ZF50": 5, "ZFS": 6,
    "ZF60": 7, "ZF75": 8, "ZF90": 9, "ZFB": 10, "ZF3S": 11, "ZF6S": 12,
}
# Gate-1 lock ordering (shield=10) — sensitivity only.
RANK_GATE1ALT = {
    "ZFHIGH": 1, "ZFOBLONG": 2, "ZFDOME": 3, "ZF30": 4, "ZF50": 5, "ZF60": 6,
    "ZF75": 7, "ZF90": 8, "ZFB": 9, "ZFS": 10, "ZF3S": 11, "ZF6S": 12,
}
HPF = {
    "ZFHIGH": 3.3, "ZFOBLONG": 3.7, "ZFDOME": 4.3, "ZF30": 4.7, "ZF50": 5.3,
    "ZFS": 6.0, "ZF60": 6.7, "ZF75": 8.0, "ZF90": 9.0, "ZFB": 10.0,
    "ZF3S": 11.0, "ZF6S": 12.0,
}


def load_sparse():
    """Stream-load the wide UMI table to CSR (genes x cells) without dense blowup."""
    import pandas as pd
    print(f"Loading {UMI.name} in gene-chunks of {CHUNK_GENES} ...")
    t0 = time.time()
    genes, mats, cells = [], [], None
    reader = pd.read_csv(UMI, sep="\t", index_col=0, compression="gzip",
                         chunksize=CHUNK_GENES)
    for i, ch in enumerate(reader):
        if cells is None:
            cells = ch.columns.tolist()
        genes.extend(ch.index.tolist())
        mats.append(sp.csr_matrix(ch.values.astype(np.int32)))
        if (i + 1) % 4 == 0:
            print(f"  chunk {i+1}: {len(genes)} genes so far, {time.time()-t0:.0f}s")
    Xgc = sp.vstack(mats).tocsr()          # (genes, cells)
    print(f"  loaded genes x cells = {Xgc.shape}, nnz={Xgc.nnz}, {time.time()-t0:.0f}s")
    return Xgc, genes, cells


def main():
    Xgc, genes, cells = load_sparse()
    genes = np.array(genes)
    cells = np.array(cells)
    n_genes, n_cells = Xgc.shape
    assert n_cells == len(cells) and n_genes == len(genes)

    # Cells x genes
    X = Xgc.T.tocsr()                      # (cells, genes) int32
    del Xgc

    # Stage assignment from cell-ID prefix
    prefixes = np.array([c.split("_")[0] for c in cells])
    uniq_pfx, counts = np.unique(prefixes, return_counts=True)
    print("\nStage prefixes found:", dict(zip(uniq_pfx.tolist(), counts.tolist())))
    unknown = [p for p in uniq_pfx if p not in RANK_KIMMEL]
    if unknown:
        print("  WARNING unknown prefixes (excluded from ordinal):", unknown)

    rank_k = np.array([RANK_KIMMEL.get(p, np.nan) for p in prefixes], dtype=float)
    rank_g = np.array([RANK_GATE1ALT.get(p, np.nan) for p in prefixes], dtype=float)
    hpf = np.array([HPF.get(p, np.nan) for p in prefixes], dtype=float)

    # QC
    gene_count_all = np.asarray((X > 0).sum(axis=1)).ravel()
    keep = (gene_count_all >= MIN_GENES) & np.isfinite(rank_k)
    print(f"\nQC: min_genes={MIN_GENES} -> kept {keep.sum()}/{len(keep)} cells "
          f"(dropped {(~keep).sum()}: {(gene_count_all<MIN_GENES).sum()} low-gene, "
          f"{(~np.isfinite(rank_k)).sum()} unranked)")

    X = X[keep].tocsr()
    cells = cells[keep]; prefixes = prefixes[keep]
    rank_k = rank_k[keep]; rank_g = rank_g[keep]; hpf = hpf[keep]
    n_cells = X.shape[0]

    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    gene_count = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float64)

    # ---- primitives ----
    print("\nPrimitives:")
    # (1) gene_count = gene_count (already)
    # (2) Shannon H(x/Sx) natural log, over ALL genes (raw counts)
    print("  Shannon H ...")
    shannon = np.empty(n_cells)
    indptr, data = X.indptr, X.data.astype(np.float64)
    for c in range(n_cells):
        seg = data[indptr[c]:indptr[c+1]]
        s = seg.sum()
        if s <= 0:
            shannon[c] = np.nan; continue
        p = seg / s
        shannon[c] = -np.sum(p * np.log(p))

    # (3) PCC(x, degree) STRING v12 : log1p(CPM_full) then Pearson with degree,
    #     over atlas ∩ STRING genes, chunked over cells to bound memory.
    print("  PCC(x, degree) STRING v12 ...")
    d = np.load(STRING, allow_pickle=True)
    string_genes = np.array([str(g) for g in d["genes"]])
    string_deg = np.asarray(d["degree"], dtype=np.float64)
    sidx = {g: i for i, g in enumerate(string_genes)}
    col_atlas, col_string = [], []
    for i, g in enumerate(genes):
        j = sidx.get(str(g))
        if j is not None:
            col_atlas.append(i); col_string.append(j)
    col_atlas = np.array(col_atlas); col_string = np.array(col_string)
    deg_sub = string_deg[col_string]
    print(f"    atlas ∩ STRING v12 = {len(col_atlas)} genes "
          f"(deg mean {deg_sub.mean():.1f})")

    target_sum = float(np.median(lib[lib > 0]))
    Xsub = X[:, col_atlas].tocsr()         # cells x n_string genes (sparse)
    kc = deg_sub - deg_sub.mean()
    kc_norm = np.sqrt((kc**2).sum())
    pcc = np.empty(n_cells)
    for s0 in range(0, n_cells, CHUNK_CELLS):
        s1 = min(s0 + CHUNK_CELLS, n_cells)
        block = Xsub[s0:s1].toarray().astype(np.float64)     # (c, G)
        tot = lib[s0:s1][:, None]
        tot = np.where(tot > 0, tot, 1.0)
        rel = block / tot * target_sum
        Xn = np.log1p(rel)
        Rc = Xn - Xn.mean(axis=1, keepdims=True)
        num = Rc @ kc
        den = np.sqrt((Rc**2).sum(axis=1)) * kc_norm
        out = np.zeros(s1 - s0)
        ok = den > 0
        out[ok] = num[ok] / den[ok]
        pcc[s0:s1] = out
    print(f"    PCC done: mean {np.nanmean(pcc):+.4f}")

    # ---- depth per stage ----
    depth = {}
    for p in RANK_KIMMEL:
        m = prefixes == p
        if m.sum() == 0:
            continue
        depth[p] = {
            "rank_kimmel": RANK_KIMMEL[p], "rank_gate1alt": RANK_GATE1ALT[p],
            "hpf": HPF[p], "n_cells": int(m.sum()),
            "median_gene_count": float(np.median(gene_count[m])),
            "median_library_size": float(np.median(lib[m])),
        }

    # ---- save ----
    sp.save_npz(OUT / "X_cells_genes.npz", X)
    (OUT / "genes.txt").write_text("\n".join(map(str, genes)))
    import csv
    with open(OUT / "cells.tsv", "w", newline="") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow(["cell_id", "stage", "rank_kimmel", "rank_gate1alt", "hpf",
                    "lib_size", "gene_count"])
        for i in range(n_cells):
            w.writerow([cells[i], prefixes[i], int(rank_k[i]), int(rank_g[i]),
                        hpf[i], int(lib[i]), int(gene_count[i])])
    np.savez(OUT / "primitives.npz", gene_count=gene_count, pcc_string_v12=pcc,
             shannon_H=shannon, rank_kimmel=rank_k, rank_gate1alt=rank_g, hpf=hpf)
    np.savez(OUT / "string_cols.npz", col_atlas=col_atlas, col_string=col_string,
             deg_sub=deg_sub)
    with open(OUT / "depth_per_stage.json", "w") as f:
        json.dump({"min_genes_qc": MIN_GENES, "n_cells_post_qc": n_cells,
                   "n_genes": int(len(genes)), "target_sum_cpm": target_sum,
                   "n_atlas_x_string": int(len(col_atlas)),
                   "per_stage": depth}, f, indent=2)

    # MatrixMarket (genes x cells) for R consumers — post-QC
    print("\nWriting MatrixMarket for R ...")
    sio.mmwrite(str(OUT / "matrix.mtx"), X.T.tocsr().astype(np.int32))
    (OUT / "features.tsv").write_text("\n".join(map(str, genes)))
    (OUT / "barcodes.tsv").write_text("\n".join(map(str, cells)))

    print("\n=== DEPTH PER STAGE (Kimmel order) ===")
    for p in sorted(depth, key=lambda x: depth[x]["rank_kimmel"]):
        z = depth[p]
        print(f"  r{z['rank_kimmel']:2d} {p:10s} n={z['n_cells']:5d} "
              f"med_genes={z['median_gene_count']:6.0f} med_lib={z['median_library_size']:7.0f}")
    print(f"\nPrepared inputs written to {OUT}")


if __name__ == "__main__":
    main()
