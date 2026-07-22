"""
cytotrace_full.py — full CytoTRACE v1 implementation (Gulati et al. 2020 Science).

=============================================================================
R-FAITHFUL FIDELITY-FIX REVISION · 2026-07-19 (EXPLORER)
=============================================================================
This revision implements a `variant` switch on the pipeline:

  variant="rfaithful"  (DEFAULT — the R-faithful implementation)
  variant="rowstoch_pyorder"  (LEGACY — the 2026-07-19 first-run implementation
                                kept in code for reproducibility of the prior
                                dossier's JSON outputs; NOT recommended for
                                new runs)

The R source `similarity_matrix_cleaned` function was re-fetched in-session on
2026-07-19 to verify the exact operation order:

    > "similarity_matrix_cleaned <- function(similarity_matrix){
    >   D <- similarity_matrix
    >   cutoff <- mean(as.vector(D))
    >   diag(D) <- 0;
    >   D[which(D < 0)] <- 0;
    >   D[which(D <= cutoff)] <- 0;
    >   Ds <- D
    >   D <- D / rowSums(D);
    >   D[which(rowSums(Ds)==0),] <- 0
    >   return(D)
    > }"
    [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]

Four side-by-side fixes vs the prior (legacy) implementation:

  FIX 1 — cutoff timing.  R: `cutoff <- mean(as.vector(D))` is computed on the
          RAW similarity matrix (with the Pearson self-similarity = 1 on the
          diagonal and any negative correlations intact). Prior Python:
          computed cutoff AFTER zeroing negatives + diagonal (mean over cleaned
          entries). FIX: compute cutoff on the raw D, then clean.
          Direction of bias vs prior: R's cutoff is higher because the
          diagonal 1s pull the mean up (and negatives pull the mean down;
          typically diagonal dominates).

  FIX 2 — diagonal-zeroing timing.  R: `diag(D) <- 0` comes AFTER the cutoff
          is computed but BEFORE the cutoff mask. Prior Python: zeroed the
          diagonal AFTER negatives and BEFORE the mean was computed.
          FIX: match R's order — compute cutoff on raw, then zero diagonal,
          then zero negatives, then apply cutoff mask.

  FIX 3 — NNLS input.  R passes the row-normalized D returned by
          `similarity_matrix_cleaned` to `nnls::nnls`:
          > "out <- nnls::nnls(similarity_matrix_cleaned,score)"
          [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]
          Prior Python ALREADY passes the row-normalized D_stoch to NNLS.
          NO CHANGE required — behaviour was already correct. The task
          mandate's premise that R passes the un-normalized matrix to NNLS
          is contradicted by the re-fetched R source (the row-normalization
          line `D <- D / rowSums(D)` is INSIDE `similarity_matrix_cleaned`,
          before the return; both NNLS and diffusion receive that returned
          row-normalized D).

  FIX 4 — diffusion multiplication.  R multiplies by the row-normalized D
          returned from `similarity_matrix_cleaned`:
          > "v_curr <- ALPHA * (similarity_matrix_cleaned %*% v_curr) +
          >           (1 - ALPHA) * vals;"
          [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]
          Prior Python ALREADY multiplies by D_stoch (row-stochastic).
          NO CHANGE required — behaviour was already correct.

Net: the actual R-faithful deviation from the prior implementation is
FIX 1 + FIX 2 (cleaning-order). FIX 3 + FIX 4 required no change; the
task mandate's premise about R's NNLS/diffusion input was based on a
misreading of the R source. Every parameter (ALPHA=0.9, tol=1e-6,
max_iter=10000, GCS_TOP_N=200, MVG_TOP_N=1000, SUBSAMPLE_SIZE=1000,
SUBSAMPLE_SEED=42, MVG_MIN_EXPR_FRAC=0.05) is FROZEN — this is a
fidelity fix, NOT a tuning change.

=============================================================================
ORIGINAL DOCSTRING (from 2026-07-19 first run)
=============================================================================

This is a faithful port of the 4-step algorithm exposed in the R package
`gunsagargulati/CytoTRACE` (master, file `R/CytoTRACE.R`), as fetched
in-session:
  [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]

The four steps are:
  (1) Gene counts     — apply(mat > 0, 2, sum)
  (2) GCS             — top-200 genes correlated with counts, arithmetic mean
                        [note: R code uses `apply(mat2[...],2,mean)`, i.e.
                         arithmetic mean of log-normalized expression across
                         the top-200 genes — same rank as geometric mean of
                         raw expression but computed on log-scale]
  (3) NNLS            — nnls::nnls(similarity_matrix_cleaned, score)
                        followed by score_regressed = D %*% coef
  (4) Markov diffusion — for i in 1:10000 { v = ALPHA*(D %*% v) + (1-ALPHA)*init;
                                             if mean|v - v_prev| <= 1e-6 break }
                        with ALPHA = 0.9

Steps 1 and 2 exist already in `potency_metrics.cytotrace_proxy()`; we reuse
them via import. Steps 3 and 4 are new here.

Key parameters (from the R source, verbatim):
  - top variable genes for similarity: 1000
  - top gene-count-correlated genes for GCS: 200
  - similarity threshold: mean(D) (cutoff on Pearson gene-gene correlation)
  - diffusion damping ALPHA: 0.9
  - diffusion max iterations: 10000
  - diffusion convergence tolerance: 1e-6
  - subsamplesize: 1000 (default; enableFast=TRUE mode).  When cells exceed
    subsamplesize the R code partitions cells randomly into ceil(N/1000)
    chunks and runs the full 4-step pipeline INDEPENDENTLY per chunk,
    then concatenates the per-chunk scores.  Quoted:
        > "chunk <- round(ncol(mat)/size)
        >  subsamples <- split(1:ncol(mat), sample(factor(1:ncol(mat) %% chunk)))"
    [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]
    We match this behaviour: at C > subsamplesize we run the 4-step pipeline
    per chunk with a fixed seed and concatenate.

Note on similarity graph: the R code builds a **cell-cell similarity matrix**
by taking Pearson correlation between cells over the top-1000 most-variable
genes (mvg). The line `similarity_matrix_cleaned(HiClimR::fastCor(mvg(mat2)))`
computes gene-vs-gene fastCor if mvg returns a genes-x-cells matrix. Reading
`mvg()` more carefully:
    mvg <- function(matn) { A <- matn; ...; A_filt <- A[disp >= last_disp,]; return(A_filt) }
returns a **genes x cells** submatrix (top-1000 most-variable genes). Then
`fastCor(mvg(mat2))` in HiClimR by default correlates **columns**, so we
end up with a **cell x cell** correlation matrix. That is the intended
similarity matrix.  See:
  [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]

Author: EXPLORER agent, 2026-07-19
"""

from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.sparse as sp

# Reuse steps 1-2 from potency_metrics
HERE = Path(__file__).resolve().parent
PKG_ROOT = HERE.parent  # tautology-diagnostic/
sys.path.insert(0, str(PKG_ROOT))
import potency_metrics as pm  # noqa: E402
from potency_metrics import (  # noqa: E402
    _rows_dense, library_normalize, cytotrace_proxy, _pearson_cols_vs_vec,
)


# ---------------------------------------------------------------------------
#  Locked parameter defaults (from R source, see file docstring)
#  [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]
# ---------------------------------------------------------------------------
MVG_TOP_N = 1000            # top variable genes for cell-cell similarity
GCS_TOP_N = 200             # genes for GCS (top-200 most correlated with counts)
DIFF_ALPHA = 0.9            # diffusion damping factor
DIFF_MAX_ITER = 10000       # diffusion max iterations
DIFF_TOL = 1e-6             # diffusion convergence tolerance (mean |Δ|)
MVG_MIN_EXPR_FRAC = 0.05    # gene must be expressed in >= 5% of cells (mvg filter)
SUBSAMPLE_SIZE = 1000       # enableFast=TRUE default chunk size for NNLS+diffusion
SUBSAMPLE_SEED = 42         # locked seed for reproducible chunking


# ---------------------------------------------------------------------------
#  Step 3-4 helpers (new)
# ---------------------------------------------------------------------------
def most_variable_genes(X_cells_by_genes: np.ndarray,
                        top_n: int = MVG_TOP_N,
                        min_expr_frac: float = MVG_MIN_EXPR_FRAC
                        ) -> np.ndarray:
    """
    Top-N most variable genes by dispersion (var / mean).

    Quoted from Gulati R code `mvg()`:
        > "n_expr <- rowSums(A > 0); A_filt <- A[n_expr >= 0.05 * ncol(A),];
        >  vars <- apply(A_filt, 1, var); means <- apply(A_filt, 1, mean);
        >  disp <- vars / means; last_disp <- tail(sort(disp), 1000)[1];
        >  A_filt <- A_filt[disp >= last_disp,]"
    [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]

    Input here: X is cells x genes (numpy convention).  The R code operates on
    genes x cells; we transpose the reasoning to columns.

    Returns index array (column positions in X) of the selected top-N genes.
    """
    X = np.asarray(X_cells_by_genes)
    n_cells = X.shape[0]
    # column-wise: gene = column
    n_expr = (X > 0).sum(axis=0)
    keep = n_expr >= (min_expr_frac * n_cells)
    if keep.sum() < top_n:
        # Cannot filter down to top_n — take all that survive the expression filter
        # (rare on real atlases with n_cells >> 100)
        candidate_idx = np.where(keep)[0]
        return candidate_idx
    X_filt = X[:, keep]
    means = X_filt.mean(axis=0)
    vars_ = X_filt.var(axis=0)
    safe_means = np.where(means > 0, means, 1.0)
    disp = vars_ / safe_means
    disp[means <= 0] = -np.inf  # drop zero-mean genes
    # top_n by dispersion
    k = min(top_n, len(disp))
    top_local = np.argsort(disp)[-k:]
    # map back to original column indices
    orig_idx = np.where(keep)[0][top_local]
    return orig_idx


def similarity_matrix_cleaned(X_cells_by_topgenes: np.ndarray,
                              variant: str = "rfaithful") -> np.ndarray:
    """
    Cell-cell similarity matrix.

    R source (verbatim, re-fetched in-session 2026-07-19):
        > "similarity_matrix_cleaned <- function(similarity_matrix){
        >   D <- similarity_matrix
        >   cutoff <- mean(as.vector(D))
        >   diag(D) <- 0;
        >   D[which(D < 0)] <- 0;
        >   D[which(D <= cutoff)] <- 0;
        >   Ds <- D
        >   D <- D / rowSums(D);
        >   D[which(rowSums(Ds)==0),] <- 0
        >   return(D)
        > }"
        > "D <- similarity_matrix_cleaned(HiClimR::fastCor(mvg(mat2)))"
    [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]

    Two variants:

    variant="rfaithful" (DEFAULT — R-faithful, 2026-07-19 fidelity fix):
      1. Compute Pearson correlation matrix D between every pair of cells
         over the top-1000-most-variable genes.
      2. cutoff = mean(D_raw) — computed on RAW D (with self-similarity 1
         on the diagonal and negative entries intact) BEFORE any cleaning.
      3. diag(D) = 0.
      4. D[D < 0] = 0.
      5. D[D <= cutoff] = 0.
      6. row-normalize to a Markov transition matrix (returned; NOT the
         un-row-normalized matrix).
      7. rows with all-zero pre-normalization sums are left as zero rows.

    variant="rowstoch_pyorder" (LEGACY — 2026-07-19 first-run):
      1. Compute Pearson correlation matrix D.
      2. D[D < 0] = 0.
      3. diag(D) = 0.
      4. cutoff = mean(D_cleaned) — computed AFTER negatives + diagonal
         are zeroed.
      5. D[D <= cutoff] = 0.
      6. (row-normalization then applied in a separate helper, matching
         final Markov behaviour.)

      This LEGACY variant is preserved so the prior dossier's JSON outputs
      can be reproduced bit-identically; it is NOT recommended for new runs.

    Note: In both variants the returned matrix under variant="rfaithful"
    is row-normalized (matches R behaviour), while under variant="rowstoch_pyorder"
    the returned matrix is NOT row-normalized — the caller applies
    `row_normalize_stochastic` separately (matches the prior Python
    two-step behaviour).
    """
    X = np.asarray(X_cells_by_topgenes, dtype=np.float32)
    Xc = X - X.mean(axis=1, keepdims=True)
    row_norm = np.sqrt((Xc * Xc).sum(axis=1))
    row_norm_safe = np.where(row_norm > 0, row_norm, 1.0)
    Xn = Xc / row_norm_safe[:, None]           # unit row norm
    D = Xn @ Xn.T                              # C x C, float32 — RAW Pearson

    if variant == "rfaithful":
        # FIX 1 + FIX 2 (2026-07-19 R-faithful revision):
        # R computes cutoff on RAW D (diag=1, negatives intact) BEFORE any
        # zeroing, then zeros diagonal, then zeros negatives, then applies
        # cutoff mask, then row-normalizes.
        cutoff = float(D.mean())              # mean over RAW D (incl. 1s on diag)
        np.fill_diagonal(D, 0.0)              # diag(D) <- 0
        D[D < 0] = 0.0                        # D[D<0] <- 0
        D[D <= cutoff] = 0.0                  # D[D<=cutoff] <- 0
        # Row-normalize inside this function (matches R which does it
        # INSIDE similarity_matrix_cleaned before the return statement).
        Ds_rowsum = D.sum(axis=1)
        rs = Ds_rowsum.reshape(-1, 1)
        safe = np.where(rs > 0, rs, 1.0)
        D = D / safe
        D[Ds_rowsum == 0, :] = 0.0
        return D
    elif variant == "rowstoch_pyorder":
        # LEGACY 2026-07-19 first-run behaviour (do NOT row-normalize here;
        # caller applies row_normalize_stochastic separately).
        D[D < 0] = 0.0
        np.fill_diagonal(D, 0.0)
        cutoff = float(D.mean())
        D[D <= cutoff] = 0.0
        return D
    else:
        raise ValueError(f"Unknown variant {variant!r}. "
                         "Use 'rfaithful' (default) or 'rowstoch_pyorder' (legacy).")


def row_normalize_stochastic(D: np.ndarray) -> np.ndarray:
    """
    Row-normalize D to a Markov transition matrix.

    Quoted from Gulati R code:
        > "D <- D / rowSums(D); D[which(rowSums(Ds)==0),] <- 0"
    [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]

    Rows with zero sum are left zero (no outbound transitions).
    """
    D = D.astype(np.float32, copy=True)
    rs = D.sum(axis=1, keepdims=True)
    safe = np.where(rs > 0, rs, 1.0)
    D_stoch = D / safe
    D_stoch[rs.ravel() == 0, :] = 0.0
    return D_stoch


def nnls_regression(D_stoch: np.ndarray, gcs: np.ndarray) -> np.ndarray:
    """
    NNLS regression of GCS onto cell-cell similarity basis.

    Quoted from Gulati R code:
        > "out <- nnls::nnls(similarity_matrix_cleaned, score)"
        > "score_regressed <- similarity_matrix_cleaned %*% out$x"
    [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]

    Uses scipy.optimize.nnls.  For a C x C system this is O(C^3) worst-case
    which is slow at C=14k; scipy's Lawson–Hanson solves iteratively.  We
    accept the runtime — this is Gulati's spec.
    """
    from scipy.optimize import nnls
    coef, _resid = nnls(D_stoch.astype(np.float64), gcs.astype(np.float64), maxiter=None)
    score_regressed = D_stoch @ coef.astype(np.float32)
    return score_regressed.astype(np.float64)


def diffusion_smooth(D_stoch: np.ndarray, init_score: np.ndarray,
                     alpha: float = DIFF_ALPHA,
                     max_iter: int = DIFF_MAX_ITER,
                     tol: float = DIFF_TOL,
                     verbose: bool = False,
                     ) -> tuple[np.ndarray, int]:
    """
    Personalized-PageRank-style diffusion.

    Quoted from Gulati R code:
        > "for(i in 1:10000) {
        >    v_prev <- rep(v_curr);
        >    v_curr <- ALPHA * (similarity_matrix_cleaned %*% v_curr) +
        >              (1 - ALPHA) * vals;
        >    diff <- mean(abs(v_curr - v_prev));
        >    if(diff <= 1e-6) { break; }
        >  }"
    [FETCH: https://raw.githubusercontent.com/gunsagargulati/CytoTRACE/master/R/CytoTRACE.R · 2026-07-19]

    ALPHA = 0.9, tol = 1e-6, max_iter = 10000.
    init_score is the NNLS-regressed score (or the raw GCS if no NNLS step
    ran); vals is the same init_score used as the personalization vector.

    Returns (final v, number of iterations).
    """
    D_stoch = D_stoch.astype(np.float32)
    vals = init_score.astype(np.float32)
    v_curr = vals.copy()
    n_iter = 0
    for i in range(max_iter):
        v_prev = v_curr
        v_curr = alpha * (D_stoch @ v_curr) + (1.0 - alpha) * vals
        diff = float(np.mean(np.abs(v_curr - v_prev)))
        n_iter = i + 1
        if diff <= tol:
            if verbose:
                print(f"  diffusion converged at iter {n_iter} (Δ={diff:.2e})")
            break
    return v_curr.astype(np.float64), n_iter


def rank_and_scale_01(x: np.ndarray) -> np.ndarray:
    """
    Rank + linear-scale to [0, 1]. This is Gulati's final step: the
    diffused value is monotonically transformed to [0, 1] by rank.

    Quoted from Gulati R landing page:
        > "final values ranked and scaled to [0, 1] (0 = more
        > differentiated; 1 = less differentiated)"
    [FETCH: https://cytotrace.stanford.edu/ · 2026-07-19]

    Uses standard rank (average ties) then divides by (n-1) so that
    min -> 0, max -> 1.
    """
    from scipy.stats import rankdata
    r = rankdata(x, method="average")
    n = len(r)
    if n <= 1:
        return np.zeros_like(r, dtype=np.float64)
    return (r - 1.0) / (n - 1.0)


# ---------------------------------------------------------------------------
#  Full CT v1 pipeline
# ---------------------------------------------------------------------------
def cytotrace_v1_full(adata,
                      layer: Optional[str] = None,
                      n_top: int = GCS_TOP_N,
                      target_sum: Optional[float] = None,
                      mvg_top: int = MVG_TOP_N,
                      alpha: float = DIFF_ALPHA,
                      max_iter: int = DIFF_MAX_ITER,
                      tol: float = DIFF_TOL,
                      subsamplesize: int = SUBSAMPLE_SIZE,
                      subsample_seed: int = SUBSAMPLE_SEED,
                      enable_fast: bool = True,
                      run_nnls: bool = True,
                      run_diffusion: bool = True,
                      variant: str = "rfaithful",
                      verbose: bool = False,
                      ) -> dict:
    """
    Full CytoTRACE v1 per Gulati 2020, 4-step pipeline.

    Steps (all sourced verbatim; see the individual function docstrings):
      1. Gene counts + library-lognormalize   (from potency_metrics.cytotrace_proxy)
      2. GCS = mean of log1p(rel) over top-200 genes with Pearson(gene, GC) > 0
         (from potency_metrics.cytotrace_proxy — note: R uses arithmetic mean
          on log-transformed values, which is monotone to geometric mean of
          raw expression; we use the same log-arithmetic-mean here so the
          rank is identical to the R version)
      3. NNLS regression of GCS on the cell-cell similarity matrix
      4. Personalized-PageRank-style diffusion (ALPHA=0.9, tol=1e-6, max 10000 iter)

    variant: str = "rfaithful" (default) or "rowstoch_pyorder"
      - "rfaithful": R-faithful cleaning order (2026-07-19 fidelity fix) —
         cutoff on raw D, then diag=0, then neg=0, then cutoff mask, then
         row-normalize INSIDE similarity_matrix_cleaned. This matches the
         R source verbatim.
      - "rowstoch_pyorder": legacy 2026-07-19 first-run cleaning order —
         negatives=0, diag=0, cutoff from cleaned mean, cutoff mask, then
         row-normalize as a separate step. Kept for reproducibility of the
         prior dossier's JSON outputs only; NOT recommended for new runs.

    Returns dict with:
      'gene_counts'  : raw gene count per cell (step 1)
      'gcs'          : GCS per cell (step 2)
      'nnls_score'   : NNLS-regressed score (step 3 output; = gcs if run_nnls=False)
      'diffused'     : diffusion-smoothed score (step 4 output; = nnls_score if
                       run_diffusion=False)
      'ct_v1'        : final [0, 1] CT v1 score = rank_and_scale_01(diffused)
      'similarity'   : cell-cell similarity matrix D (before row normalization)
                       — returned only if verbose=True to save memory
      'D_stoch'      : row-normalized transition matrix (Markov)
                       — returned only if verbose=True
      'diffusion_iters' : number of diffusion iterations executed

    Note: for atlases larger than ~15k cells this needs O(C^2) memory
    (~1 GB float32 at 15k cells). Consider batching or PCA prefilter
    for >30k cell atlases; Gulati's original R code has similar memory
    scaling by design.
    """
    if verbose:
        print("[CT v1] step 1-2: gene counts + GCS (reused from cytotrace_proxy)")
    proxy = cytotrace_proxy(adata, layer=layer, n_top=n_top, target_sum=target_sum)
    gc = proxy["gene_counts"]
    gcs = proxy["gcs"]

    # cells x genes matrix in library-normalized log space (matches proxy inputs)
    X = pm._get_X(adata, layer)
    Xd = _rows_dense(X)
    rel = library_normalize(Xd, target_sum=target_sum)
    logn = np.log1p(rel)                       # C x G, float64 (matches proxy)
    n_cells = logn.shape[0]

    # Decide chunking (enableFast=TRUE, subsamplesize)
    if enable_fast and n_cells > subsamplesize:
        # Match R behaviour: split cells into ceil(N/subsamplesize) chunks
        rng = np.random.default_rng(subsample_seed)
        chunk_count = max(1, int(round(n_cells / subsamplesize)))
        # random permutation → chunk_count roughly-equal groups
        perm = rng.permutation(n_cells)
        chunks = np.array_split(perm, chunk_count)
        if verbose:
            print(f"[CT v1] chunking {n_cells} cells into {chunk_count} chunks "
                  f"of ~{subsamplesize} (enableFast=TRUE, seed={subsample_seed})")
    else:
        chunks = [np.arange(n_cells)]
        if verbose:
            print(f"[CT v1] running as a single chunk of {n_cells} cells "
                  f"(enable_fast={enable_fast}, subsamplesize={subsamplesize})")

    diffused = np.full(n_cells, np.nan, dtype=np.float64)
    nnls_scores = np.full(n_cells, np.nan, dtype=np.float64)
    n_iter_total = 0
    for ci, cell_idx in enumerate(chunks):
        cell_idx = np.asarray(cell_idx)
        if verbose:
            print(f"  chunk {ci + 1}/{len(chunks)}: {len(cell_idx)} cells")
        # MVG on chunk
        logn_chunk = logn[cell_idx, :]
        mvg_idx = most_variable_genes(logn_chunk, top_n=mvg_top)
        if verbose:
            print(f"    selected {len(mvg_idx)} MVGs; computing cell-cell similarity...")
        D = similarity_matrix_cleaned(logn_chunk[:, mvg_idx], variant=variant)
        if variant == "rfaithful":
            # rfaithful similarity_matrix_cleaned already row-normalizes inside
            # (matches R's function-internal `D <- D / rowSums(D)`), so `D` is
            # ALREADY the row-stochastic matrix that goes to NNLS + diffusion.
            D_stoch = D
        else:
            # legacy rowstoch_pyorder: caller row-normalizes as a separate step
            # (matches prior 2026-07-19 first-run behaviour byte-identically).
            D_stoch = row_normalize_stochastic(D)
            del D
        gcs_chunk = gcs[cell_idx]
        if run_nnls:
            nnls_chunk = nnls_regression(D_stoch, gcs_chunk)
        else:
            nnls_chunk = gcs_chunk.copy()
        nnls_scores[cell_idx] = nnls_chunk
        if run_diffusion:
            diff_chunk, n_iter = diffusion_smooth(
                D_stoch, nnls_chunk, alpha=alpha, max_iter=max_iter, tol=tol,
                verbose=verbose,
            )
        else:
            diff_chunk = nnls_chunk.copy()
            n_iter = 0
        diffused[cell_idx] = diff_chunk
        n_iter_total += n_iter
        del D_stoch

    # Final rank+scale across ALL cells (matches R behavior — the R code
    # ranks after concatenating chunk outputs).
    ct_v1 = rank_and_scale_01(diffused)

    out = {
        "gene_counts": gc,
        "gcs": gcs,
        "nnls_score": nnls_scores,
        "diffused": diffused,
        "ct_v1": ct_v1,
        "diffusion_iters": n_iter_total,
        "n_chunks": len(chunks),
    }
    return out


# ---------------------------------------------------------------------------
#  Self-test on synthetic data (matches potency_metrics.py's __main__ style)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import anndata as ad
    from scipy.stats import spearmanr

    print("=" * 70)
    print("Self-test: full CT v1 on synthetic potency-gradient AnnData")
    print("=" * 70)

    rng = np.random.default_rng(42)
    n_cells, n_genes = 400, 600
    t = rng.uniform(0, 1, n_cells)             # latent potency

    # Same synthetic-counts construction as potency_metrics.__main__:
    base_rate = 2.0
    # scale-free-ish degree vector for hub-driven bias
    deg_target = (rng.zipf(2.0, n_genes)).clip(1, 60).astype(float)
    deg_norm = (deg_target - deg_target.mean()) / (deg_target.std() + 1e-9)

    counts = np.zeros((n_cells, n_genes))
    for c in range(n_cells):
        active_p = 0.2 + 0.6 * t[c]
        active = rng.random(n_genes) < active_p
        rate = base_rate * (1 + 0.8 * t[c] * deg_norm)
        rate = np.clip(rate, 0.05, None)
        counts[c] = rng.poisson(rate) * active
    counts = counts.astype(np.float64)

    adata = ad.AnnData(X=counts)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs_names = [f"cell{i}" for i in range(n_cells)]

    print("\n=== running cytotrace_v1_full ===")
    result = cytotrace_v1_full(adata, mvg_top=200, verbose=True)

    # (a) all values in [0, 1]
    ct = result["ct_v1"]
    print(f"\nct_v1 range: [{ct.min():.4f}, {ct.max():.4f}]")
    assert (ct >= 0).all() and (ct <= 1).all(), "ct_v1 out of [0, 1]"
    print("PASS: ct_v1 in [0, 1]")

    # (b) correlates with true potency at Spearman > 0.3
    rho = float(spearmanr(ct, t).statistic)
    print(f"Spearman(ct_v1, true potency) = {rho:+.4f}")
    assert rho > 0.3, f"ct_v1 not correlated with potency (rho={rho:.3f})"
    print("PASS: Spearman(ct_v1, t) > 0.3")

    # (c) ct_v1 differs from raw gene counts
    gc = result["gene_counts"]
    rho_gc = float(spearmanr(ct, gc).statistic)
    print(f"Spearman(ct_v1, gene_counts) = {rho_gc:+.4f}")
    assert rho_gc < 1.0, "ct_v1 identical rank to gene counts"
    print("PASS: Spearman(ct_v1, gene_counts) < 1.0 (differs from raw GC)")

    # Bonus: diagnostic direction
    rho_gc_pot = float(spearmanr(gc, t).statistic)
    rho_gcs_pot = float(spearmanr(result["gcs"], t).statistic)
    print(f"\nDiagnostic ordering vs potency (synthetic):")
    print(f"  Spearman(gene_counts, t) = {rho_gc_pot:+.4f}")
    print(f"  Spearman(gcs,         t) = {rho_gcs_pot:+.4f}")
    print(f"  Spearman(ct_v1,       t) = {rho:+.4f}")
    print(f"  diffusion iters converged in: {result['diffusion_iters']}")
    print("\nSelf-test PASSED.")
