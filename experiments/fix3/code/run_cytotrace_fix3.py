"""
FIX 3 GATE 2 — CytoTRACE v1 (full R-faithful port) on Briggs Xenopus GSE113074.
IDENTICAL algorithm to the zebrafish decider (run_cytotrace.py) — imports the same
verbatim Gulati-2020 core functions; only the prep/results paths differ (fix3/).
Memory-safe driver (sparse logn; densify only per enableFast chunk). Primitive = gene count.
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

PREP = VAULT / "fix3/data/prepared"
OUT = VAULT / "fix3/results/cytotrace_scores.npz"
OUT.parent.mkdir(parents=True, exist_ok=True)


def main():
    t0 = time.time()
    X = sp.load_npz(PREP / "X_cells_genes.npz").tocsr().astype(np.float64)
    n_cells, n_genes = X.shape
    lib = np.asarray(X.sum(axis=1)).ravel()
    gc = np.asarray((X > 0).sum(axis=1)).ravel().astype(np.float64)
    target_sum = float(np.median(lib[lib > 0]))
    print(f"  {n_cells} cells x {n_genes} genes, target_sum={target_sum:.1f}")

    inv = np.where(lib > 0, target_sum / lib, 0.0)
    Xn = X.copy()
    Xn.data = np.log1p(Xn.data * np.repeat(inv, np.diff(Xn.indptr)))
    del X

    sum_x = np.asarray(Xn.sum(axis=0)).ravel()
    sum_x2 = np.asarray(Xn.multiply(Xn).sum(axis=0)).ravel()
    sum_xy = np.asarray(Xn.T @ gc).ravel()
    sgc, sgc2 = gc.sum(), (gc * gc).sum()
    num = n_cells * sum_xy - sum_x * sgc
    den = np.sqrt((n_cells * sum_x2 - sum_x**2) * (n_cells * sgc2 - sgc**2))
    r = np.where(den > 0, num / den, 0.0)
    top = np.argsort(r)[-GCS_TOP_N:]
    gcs = np.exp(np.asarray(Xn[:, top].todense()).mean(axis=1)).ravel()

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    chunk_count = max(1, int(round(n_cells / SUBSAMPLE_SIZE)))
    perm = rng.permutation(n_cells)
    chunks = np.array_split(perm, chunk_count)
    print(f"  {len(chunks)} enableFast chunks of ~{SUBSAMPLE_SIZE} (seed {SUBSAMPLE_SEED})")

    diffused = np.full(n_cells, np.nan)
    for ci, idx in enumerate(chunks):
        idx = np.asarray(idx)
        logn_chunk = np.asarray(Xn[idx, :].todense())
        mvg = most_variable_genes(logn_chunk, top_n=MVG_TOP_N)
        D_stoch = similarity_matrix_cleaned(logn_chunk[:, mvg], variant="rfaithful")
        nn = nnls_regression(D_stoch, gcs[idx])
        diff, _ = diffusion_smooth(D_stoch, nn, alpha=DIFF_ALPHA,
                                   max_iter=DIFF_MAX_ITER, tol=DIFF_TOL)
        diffused[idx] = diff

    ct_v1 = rank_and_scale_01(diffused)
    np.savez(OUT, ct_v1=ct_v1, gcs=gcs, gene_counts=gc, diffused=diffused)
    from scipy.stats import spearmanr
    p = np.load(PREP / "primitives.npz")
    print(f"  CT v1 vs ordinal rho={spearmanr(ct_v1, p['rank_ordinal']).statistic:+.4f}"
          f"  vs gene_count rho={spearmanr(ct_v1, gc).statistic:+.4f}  ({time.time()-t0:.0f}s)")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
