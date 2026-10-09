"""
mce.py — Markov chain entropy (MCE), Shi, Teschendorff, Chen, Chen and Li,
Briefings in Bioinformatics 21(1):248-261 (2020), doi 10.1093/bib/bby093.

A port of the authors' MCE.m (supplement bby093_supp.zip; SHA256
c8ad11f2cdadc979cc303a21785f65eba77a41dfe565ace2beffaeae265fdce2), validated
against MCE.m run in GNU Octave by experiments/w5_new_rows/validate_mce.py.
The default rule="mce_m" reproduces MCE.m line by line:

    network     net with its diagonal set to 1 (MCE.m line 19); SCENT's SR
                uses the same network with A_ii = 0
    per sample  p0 = x / sum(x)                            (line 26)
                B_ij = p0_i net_ij                         (line 27)
    iteration   lambda0 = p0, theta0 = 1                   (lines 64-65)
                repeat: lambda = 1 ./ (A theta0)
                        theta  = p0 ./ (B' lambda)
                        err = max |[lambda - lambda0; theta - theta0]|
                        stop when err < 1e-2 or after more than 1e6 steps
                (lines 62-77); each sample stops on its own
    score       -dot(p0, log(lambda .* theta))             (line 30, the
                entropy rate h(P) of P_ij = lambda_i theta_j net_ij)
                - sum p0 log p0 over p0 > 0               (line 32, H(p0))
                divided by log(nnz(net))                   (lines 36-37)

MCE.m applies no transform to the expression it is given; it only divides by
the sum. A zero entry makes 0 * log(0) = NaN at line 30, so it needs strictly
positive input. The audit gives it SR's input, log2(CPM + 1.1) over SR's
network genes (PREREG.md, Amendments).

rule="residual" is the same fixed point solved to a stationarity residual
(max_j |(p0 P)_j - p0_j| < tol, checked every check_every steps), which the
analytic tests in tests/test_mce.py use, and which validate_mce.py uses to say
how far MCE.m's 1e-2 stopping rule leaves the score from the converged one.

The network object is the one SR uses (A without self-loops, as built by
experiments/track2/code/prep_scent_inputs.py); mce() adds the identity itself,
so the two scores share one scaffold and differ only in the score.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["mce", "mce_one", "with_self_loops", "max_mce"]

MCE_M_EPSILON = 1e-2      # MCE.m line 62
MCE_M_MAX_ITER = 10 ** 6  # MCE.m line 62


def with_self_loops(A):
    """A as a 0/1 CSR matrix with every diagonal entry set to 1."""
    A = sp.csr_matrix(A, dtype=np.float64)
    A = (A + A.T) > 0
    A = sp.csr_matrix(A, dtype=np.float64)
    A.setdiag(1.0)
    A.eliminate_zeros()
    return A


def max_mce(A_loops):
    """log(d), d = nnz of the self-looped adjacency (Eq 7; MCE.m line 36)."""
    return float(np.log(A_loops.nnz))


def _mce_m_iteration(Pi, AL, epsilon, max_iter):
    """
    MCE.m's fun_iteration for a block of cells, each cell stopping on its own.

    Returns lambda, theta (c, G) as MCE.m returns them (lambda from the
    previous theta, theta from that lambda), the per-cell step count and the
    last err.
    """
    c = Pi.shape[0]
    lam0 = Pi.copy()
    th0 = np.ones_like(Pi)
    n_it = np.zeros(c, dtype=np.int64)
    err_last = np.full(c, np.nan)
    active = np.arange(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        while active.size:
            P = Pi[active]
            lam = 1.0 / np.asarray(th0[active] @ AL)        # 1 ./ (A theta0), A symmetric
            th = P / np.asarray((P * lam) @ AL)             # p0 ./ (B' lambda)
            err = np.maximum(np.abs(lam - lam0[active]).max(axis=1),
                             np.abs(th - th0[active]).max(axis=1))
            n_it[active] += 1
            lam0[active] = lam
            th0[active] = th
            err_last[active] = err
            stop = (err < epsilon) | (n_it[active] > max_iter)
            active = active[~stop]
    return lam0, th0, n_it, err_last


def _residual_iteration(Pi, AL, tol, max_iter, check_every, history=None):
    """
    The same fixed point solved to a stationarity residual.

    Returns alpha, beta (c, G) with rows of P = diag(alpha) A diag(beta)
    summing to 1 exactly, the iteration count and the final residual
    max_j |beta_j (B' alpha)_j - pi_j|. history, if a list, collects
    (iteration, residual) at every check.
    """
    pos = Pi > 0
    beta = np.where(pos, 1.0, 0.0)
    alpha = np.zeros_like(Pi)
    err = np.inf
    it = 0
    for it in range(1, max_iter + 1):
        Ab = np.asarray(beta @ AL)
        alpha = np.divide(1.0, Ab, out=np.zeros_like(Ab), where=Ab > 0)
        Ba = np.asarray((Pi * alpha) @ AL)
        beta_new = np.divide(Pi, Ba, out=np.zeros_like(Ba), where=Ba > 0)
        if it % check_every == 0 or it == max_iter:
            Ab2 = np.asarray(beta_new @ AL)
            alpha2 = np.divide(1.0, Ab2, out=np.zeros_like(Ab2), where=Ab2 > 0)
            Ba2 = np.asarray((Pi * alpha2) @ AL)
            err = float(np.max(np.abs(beta_new * Ba2 - Pi)))
            if history is not None:
                history.append((it, err))
            beta, alpha = beta_new, alpha2
            if err < tol:
                break
        else:
            beta = beta_new
    return alpha, beta, it, err


def mce(X, A, *, rule="mce_m", normalize=True, epsilon=MCE_M_EPSILON,
        max_iter=None, tol=1e-12, check_every=10, chunk=500, return_info=False):
    """
    Per-cell Markov chain entropy.

    Parameters
    ----------
    X : (cells, network genes) array, non-negative. The expression the chain
        is built on; p0 is each row divided by its sum. No transform is
        applied here, as in MCE.m.
    A : (G, G) adjacency of the network, with or without self-loops; the
        diagonal is set to 1 here, as MCE.m does.
    rule : "mce_m" (MCE.m's iteration and stopping rule, the default) or
        "residual" (the same fixed point solved to a stationarity residual).
    epsilon, max_iter : MCE.m's stopping threshold on the change in
        (lambda, theta) and its step cap (1e-2 and 1e6) under rule="mce_m";
        under rule="residual", max_iter caps the iterations (default 20,000)
        and tol is the residual threshold.
    normalize : divide by log(nnz(A with self-loops)), MCE.m lines 36-37.

    Returns
    -------
    (cells,) array; with return_info also a dict of per-cell iteration counts
    and the last err (mce_m) or residual (residual).
    """
    X = np.asarray(X.toarray() if sp.issparse(X) else X, dtype=np.float64)
    if np.any(X < 0):
        raise ValueError("MCE needs non-negative expression")
    AL = with_self_loops(A)
    if AL.shape[0] != X.shape[1]:
        raise ValueError(f"A has {AL.shape[0]} nodes, X has {X.shape[1]} genes")
    if rule not in ("mce_m", "residual"):
        raise ValueError(f"unknown rule {rule!r}")
    if max_iter is None:
        max_iter = MCE_M_MAX_ITER if rule == "mce_m" else 20000
    logd = max_mce(AL)
    n = X.shape[0]
    out = np.full(n, np.nan)
    iters = np.zeros(n, dtype=np.int64)
    errs = np.full(n, np.nan)
    for s in range(0, n, chunk):
        Xc = X[s:s + chunk]
        tot = Xc.sum(axis=1, keepdims=True)
        ok = tot[:, 0] > 0
        Pi = np.divide(Xc, tot, out=np.zeros_like(Xc), where=tot > 0)
        if rule == "mce_m":
            lam, th, it, err = _mce_m_iteration(Pi, AL, epsilon, max_iter)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = -(Pi * np.log(lam * th)).sum(axis=1)          # line 30
            m = Pi > 0
            node = -np.where(m, Pi * np.log(np.where(m, Pi, 1.0)), 0.0).sum(axis=1)  # line 32
            val = rate + node
        else:
            alpha, beta, it, err = _residual_iteration(Pi, AL, tol, max_iter, check_every)
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
        errs[s:s + chunk] = err
    if return_info:
        return out, {"rule": rule, "iterations": iters, "err": errs, "log_d": logd,
                     "epsilon": epsilon if rule == "mce_m" else None,
                     "tol": tol if rule == "residual" else None,
                     "max_iter": max_iter}
    return out


def mce_one(x, A, **kw):
    """MCE of a single expression vector."""
    return float(mce(np.asarray(x, dtype=np.float64)[None, :], A, **kw)[0])
