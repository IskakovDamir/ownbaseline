"""
Track 2 Gate 2 — CytoTRACE v1 (full R-faithful port) on Farrell GSE106474.

Memory-safe driver: reuses the VERBATIM core functions from
cytotrace_v1/cytotrace_full.py (most_variable_genes, similarity_matrix_cleaned,
nnls_regression, diffusion_smooth, rank_and_scale_01 — each sourced verbatim
from Gulati 2020 R code, see that file's [FETCH] anchors), but keeps the
log-normalized matrix SPARSE (log1p preserves zeros) and densifies only the
per-chunk (≤~1000 cells) MVG submatrix. This replicates cytotrace_v1_full(
enable_fast=True, variant="rfaithful") byte-faithfully without the full-dense
39505×23974 (=7.6 GB) blow-up that the reference driver would trigger on 16 GB.

CytoTRACE's §3a-declared primitive = gene count |supp(x)| = H0.

Output: track2/results/cytotrace_scores.npz  (ct_v1, gcs, gene_counts, diffused)
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np
import scipy.sparse as sp

VAULT = Path("/Users/damir/damir-research-vault/06-code/tautology-diagnostic")
sys.path.insert(0, str(VAULT / "cytotrace_v1"))
sys.path.insert(0, str(VAULT))
from cytotrace_full import (
    most_variable_genes, similarity_matrix_cleaned, nnls_regression,
    diffusion_smooth, rank_and_scale_01,
    GCS_TOP_N, MVG_TOP_N, DIFF_ALPHA, DIFF_MAX_ITER, DIFF_TOL,
    SUBSAMPLE_SIZE, SUBSAMPLE_SEED,
)

PREP = VAULT / "track2/data/prepared"
OUT = VAULT / "track2/results/cytotrace_scores.npz"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    print("Loading prepared matrix ...")
    X = sp.load_npz(PREP / "X_cells_genes.npz").tocsr().astype(np.float64)  # cells x genes
    n_cells, n_genes = X.shape
    lib = np.asarray(X.sum(axis=1)).ravel()
    gc = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float64)  # gene counts (primitive)
    target_sum = float(np.median(lib[lib > 0]))
    print(f"  {n_cells} cells x {n_genes} genes, target_sum(CPM median)={target_sum:.1f}")

    # sparse log-normalized: logn = log1p(counts / lib * target_sum), zeros preserved
    inv = np.where(lib > 0, target_sum / lib, 0.0)
    Xn = X.copy()
    per_nnz_inv = np.repeat(inv, np.diff(Xn.indptr))
    Xn.data = np.log1p(Xn.data * per_nnz_inv)
    del X

    # ---- GCS gene selection: Pearson(logn[:,g], gc) over cells (sparse, incl. zeros) ----
    sum_x = np.asarray(Xn.sum(axis=0)).ravel()
    sum_x2 = np.asarray(Xn.multiply(Xn).sum(axis=0)).ravel()
    sum_xy = np.asarray(Xn.T @ gc).ravel()          # sum_c logn[c,g]*gc[c]
    sgc, sgc2 = gc.sum(), (gc * gc).sum()
    num = n_cells * sum_xy - sum_x * sgc
    den = np.sqrt((n_cells * sum_x2 - sum_x**2) * (n_cells * sgc2 - sgc**2))
    r = np.where(den > 0, num / den, 0.0)
    top = np.argsort(r)[-GCS_TOP_N:]                # top-200 genes correlated with GC
    gcs = np.exp(np.asarray(Xn[:, top].todense()).mean(axis=1)).ravel()  # exp(mean log) — matches proxy
    print(f"  GCS from top-{GCS_TOP_N} GC-correlated genes; gcs range "
          f"[{gcs.min():.3f}, {gcs.max():.3f}]")

    # ---- enableFast chunking: identical to cytotrace_v1_full ----
    rng = np.random.default_rng(SUBSAMPLE_SEED)      # 42
    chunk_count = max(1, int(round(n_cells / SUBSAMPLE_SIZE)))
    perm = rng.permutation(n_cells)
    chunks = np.array_split(perm, chunk_count)
    print(f"  chunking into {len(chunks)} chunks of ~{SUBSAMPLE_SIZE} (seed {SUBSAMPLE_SEED})")

    diffused = np.full(n_cells, np.nan)
    for ci, idx in enumerate(chunks):
        idx = np.asarray(idx)
        logn_chunk = np.asarray(Xn[idx, :].todense())          # (c, G) dense
        mvg = most_variable_genes(logn_chunk, top_n=MVG_TOP_N)
        D_stoch = similarity_matrix_cleaned(logn_chunk[:, mvg], variant="rfaithful")
        nn = nnls_regression(D_stoch, gcs[idx])
        diff, n_iter = diffusion_smooth(D_stoch, nn, alpha=DIFF_ALPHA,
                                        max_iter=DIFF_MAX_ITER, tol=DIFF_TOL)
        diffused[idx] = diff
        if (ci + 1) % 5 == 0 or ci == len(chunks) - 1:
            print(f"    chunk {ci+1}/{len(chunks)} done (iters={n_iter}, "
                  f"{time.time()-t0:.0f}s)")

    ct_v1 = rank_and_scale_01(diffused)
    np.savez(OUT, ct_v1=ct_v1, gcs=gcs, gene_counts=gc, diffused=diffused,
             gcs_top_gene_idx=top)
    from scipy.stats import spearmanr
    p = np.load(PREP / "primitives.npz")
    print(f"\n  CT v1 vs rank_kimmel: rho={spearmanr(ct_v1, p['rank_kimmel']).statistic:+.4f}")
    print(f"  CT v1 vs gene_count : rho={spearmanr(ct_v1, gc).statistic:+.4f}")
    print(f"  Wrote {OUT}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
