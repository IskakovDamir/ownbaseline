"""
mce.py — Markov chain entropy (MCE), Shi, Teschendorff, Chen, Chen and Li,
Briefings in Bioinformatics 21(1):248-261 (2020), doi 10.1093/bib/bby093.

STATUS (2026-10-09). Written from the published Methods, not yet from MCE.m.
The authors' MATLAB file sits in the article's supplement (bby093_supp.zip),
which this session could not download: OUP serves it only behind a signed CDN
link and a human-verification page. experiments/w5_new_rows/DISCOVERY.md
records the URL and what has to be placed in reference/. Until MCE.m is there
and validate_mce.py has compared the two in Octave, this is the equations of
the paper, checked against analytic answers on toy networks, and nothing more.
Defaults that MCE.m may set differently (input transform, starting point,
tolerance, iteration cap) are arguments here, so the port can adopt the
authors' values without changing the algorithm.

What it computes, per cell (Methods, sections "MCE", "Computing MCE for each
sample", "Normalization of MCE"):

    network     A = 0/1 adjacency of the interaction network WITH a self-loop
                on every node, A_ii = 1. SCENT's SR uses the same network with
                A_ii = 0; the self-loop is the one structural difference
                (Wang et al., Cell Systems, notation section).
    stationary  pi = x / sum(x) over the network genes
    chain       the transition matrix P supported on A with pi invariant that
                maximises the entropy of an edge (pi_i P_ij), Eq 3
    solution    P_ij = alpha_i beta_j A_ij; fixed point (Eq 6)
                    alpha = 1 ./ (A beta)
                    beta  = pi ./ (B' alpha),   B_ij = pi_i A_ij
    score       MCE = -sum_i pi_i log(alpha_i beta_i pi_i)
                    = H(pi) + h(P)       (node entropy + entropy rate, Eq 2)
    normalised  MCE / log(d), d = number of nonzero entries of A, self-loops
                included (Eqs 7 and 8); 1 exactly when pi_i is proportional
                to the self-loop degree d_i

So MCE is the entropy of the stationary EDGE distribution: the static entropy
of the expression distribution plus the entropy rate of the maximum-entropy
chain that has it as its stationary law. SR is an entropy rate alone, of the
mass-action chain P_ij ~ A_ij x_j.

The network object is the one SR uses (A without self-loops, as built by
experiments/track2/code/prep_scent_inputs.py); mce() adds the identity itself,
so the two scores share one scaffold and differ only in the score.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["mce", "mce_one", "with_self_loops", "max_mce"]


def with_self_loops(A):
    """A as a 0/1 CSR matrix with every diagonal entry set to 1."""
    A = sp.csr_matrix(A, dtype=np.float64)
    A = (A + A.T) > 0
    A = sp.csr_matrix(A, dtype=np.float64)
    A.setdiag(1.0)
    A.eliminate_zeros()
    return A


def max_mce(A_loops):
    """log(d), d = nnz of the self-looped adjacency (Eq 7)."""
    return float(np.log(A_loops.nnz))


def _scale(Pi, AL, tol, max_iter, check_every, history=None):
    """
    Fixed-point iteration of Eq 6 for a block of cells.

    Pi : (c, G) rows are stationary distributions (each sums to 1)
    AL : (G, G) symmetric 0/1 adjacency with self-loops
    history : optional list; (iteration, residual) is appended at each check

    Returns alpha, beta (c, G), the iteration count and the final constraint
    error max_j |beta_j (B' alpha)_j - pi_j|, which is the stationarity
    residual of P = diag(alpha) A diag(beta).
    """
    pos = Pi > 0
    beta = np.where(pos, 1.0, 0.0)
    alpha = np.zeros_like(Pi)
    err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        Ab = np.asarray(beta @ AL)                     # (A beta)_i, A symmetric
        alpha = np.divide(1.0, Ab, out=np.zeros_like(Ab), where=Ab > 0)
        Ba = np.asarray((Pi * alpha) @ AL)             # (B' alpha)_j
        beta_new = np.divide(Pi, Ba, out=np.zeros_like(Ba), where=Ba > 0)
        if it % check_every == 0 or it == max_iter:
            # column constraint after the alpha update is exact for the row
            # sums; measure the stationarity residual with the new beta's
            # partner alpha
            Ab2 = np.asarray(beta_new @ AL)
            alpha2 = np.divide(1.0, Ab2, out=np.zeros_like(Ab2), where=Ab2 > 0)
            Ba2 = np.asarray((Pi * alpha2) @ AL)
            err = float(np.max(np.abs(beta_new * Ba2 - Pi)))
            if history is not None:
                history.append((it, err))
            beta = beta_new
            alpha = alpha2
            if err < tol:
                break
        else:
            beta = beta_new
    return alpha, beta, it, err


def mce(X, A, *, normalize=True, tol=1e-12, max_iter=20000, check_every=10,
        chunk=500, return_info=False):
    """
    Per-cell Markov chain entropy.

    Parameters
    ----------
    X : (cells, network genes) array, non-negative. The expression the chain
        is built on; pi is each row divided by its sum. No transform is applied
        here: pass the matrix the caller has decided on (SR's run uses
        log2(CPM + 1.1), SCENT's DoIntegPPI convention).
    A : (G, G) adjacency of the network, with or without self-loops; the
        diagonal is set to 1 here, as in the paper.
    normalize : divide by log(d) (Eqs 7 and 8).
    tol : stop when the stationarity residual max_j |(pi P)_j - pi_j| is
        below this. The rows of P sum to 1 by construction at every step.
    max_iter, check_every, chunk : iteration cap, how often the residual is
        measured, and how many cells are solved at once.

    Returns
    -------
    (cells,) array; with return_info also a dict of per-cell iteration counts
    and residuals. A cell whose row sums to 0 gets NaN.
    """
    X = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float64)
    if np.any(X < 0):
        raise ValueError("MCE needs non-negative expression")
    AL = with_self_loops(A)
    if AL.shape[0] != X.shape[1]:
        raise ValueError(f"A has {AL.shape[0]} nodes, X has {X.shape[1]} genes")
    logd = max_mce(AL)
    n = X.shape[0]
    out = np.full(n, np.nan)
    iters = np.zeros(n, dtype=np.int64)
    resid = np.full(n, np.nan)
    for s in range(0, n, chunk):
        Xc = X[s:s + chunk]
        tot = Xc.sum(axis=1, keepdims=True)
        ok = tot[:, 0] > 0
        Pi = np.divide(Xc, tot, out=np.zeros_like(Xc), where=tot > 0)
        alpha, beta, it, err = _scale(Pi, AL, tol, max_iter, check_every)
        prod = alpha * beta * Pi
        term = np.zeros_like(Pi)
        m = Pi > 0
        term[m] = Pi[m] * np.log(prod[m])
        val = -term.sum(axis=1)
        if normalize:
            val = val / logd
        val[~ok] = np.nan
        out[s:s + chunk] = val
        iters[s:s + chunk] = it
        resid[s:s + chunk] = err
    if return_info:
        return out, {"iterations": iters, "residual": resid, "log_d": logd,
                     "tol": tol, "max_iter": max_iter}
    return out


def mce_one(x, A, **kw):
    """MCE of a single expression vector."""
    return float(mce(np.asarray(x, dtype=np.float64)[None, :], A, **kw)[0])
