"""
Tests of scores/mce.py against answers known in advance.

The analytic tests solve the fixed point to convergence (rule="residual"):
each expected value is a closed form for a network whose maximum-entropy chain
can be written down by hand, or the optimum of the same problem solved by a
generic constrained optimizer that shares no code with the fixed point. The
MCE.m tests check the default rule against a line-by-line transcription of
the authors' MCE.m, run one sample at a time as MCE.m runs. None of them
asserts the current output.

    python3 tests/test_mce.py          # standalone, same harness as run_tests.py
    pytest tests/test_mce.py           # or collected with the rest by pytest

It is kept out of run_tests.py so the estimator suite's count, which the
README states, stays what it is.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scores"))
from mce import mce, mce_one, with_self_loops  # noqa: E402

SEED = 20261009


def _H(p):
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def _pi(n, seed):
    x = np.random.default_rng(seed).gamma(2.0, 1.0, n)
    return x / x.sum()


def test_mce_with_only_self_loops_is_the_static_entropy_of_pi():
    """
    No edges: every node can only return to itself, so P = I, the entropy rate
    is 0 and the edge distribution is pi on the diagonal. MCE = H(pi).
    """
    n = 7
    pi = _pi(n, SEED)
    A = sp.csr_matrix((n, n))
    got = mce_one(pi, A, normalize=False, rule="residual")
    assert abs(got - _H(pi)) < 1e-10, (got, _H(pi))


def test_mce_on_the_complete_graph_is_twice_the_entropy_of_pi():
    """
    Complete graph with self-loops: every coupling of pi with itself is
    allowed, and the maximum-entropy one is the independent coupling
    Q = pi pi^T, so P_ij = pi_j and MCE = 2 H(pi).
    """
    n = 6
    pi = _pi(n, SEED + 1)
    A = sp.csr_matrix(np.ones((n, n)) - np.eye(n))
    got = mce_one(pi, A, normalize=False, rule="residual")
    assert abs(got - 2 * _H(pi)) < 1e-10, (got, 2 * _H(pi))


def test_normalised_mce_is_one_when_pi_is_proportional_to_self_loop_degree():
    """
    Eqs 7 and 8: with pi_i proportional to d_i (degree counting the self-loop)
    the uniform walk P_ij = 1/d_i keeps pi invariant, every edge carries mass
    1/d, and MCE reaches its maximum log(d). Normalised, exactly 1.
    """
    rng = np.random.default_rng(SEED + 2)
    n = 30
    M = np.triu((rng.random((n, n)) < 0.15).astype(float), 1)
    for i in range(n - 1):            # a path guarantees one component
        M[i, i + 1] = 1.0
    A = sp.csr_matrix(M + M.T)
    d = np.asarray(with_self_loops(A).sum(axis=1)).ravel()
    got = mce_one(d, A, normalize=True, rule="residual")
    assert abs(got - 1.0) < 1e-10, got


def test_mce_matches_a_generic_constrained_optimizer_on_a_random_network():
    """
    The same problem, Eq 3, solved directly: maximise the entropy of
    Q_ij = pi_i P_ij over the edges of A (self-loops included) subject to
    row sums of Q equal to pi and column sums equal to pi. SLSQP knows
    nothing about the alpha-beta factorisation.
    """
    from scipy.optimize import minimize

    rng = np.random.default_rng(SEED + 3)
    n = 6
    M = np.triu((rng.random((n, n)) < 0.4).astype(float), 1)
    for i in range(n - 1):
        M[i, i + 1] = 1.0
    A = sp.csr_matrix(M + M.T)
    AL = with_self_loops(A).toarray()
    edges = np.argwhere(AL > 0)
    pi = _pi(n, SEED + 4)

    def negH(q):
        q = np.clip(q, 1e-300, None)
        return float((q * np.log(q)).sum())

    def grad(q):
        return np.log(np.clip(q, 1e-300, None)) + 1.0

    # row sums and column sums both total 1, so one column constraint is
    # implied by the others; SLSQP needs a full-rank set
    cons = []
    for i in range(n):
        rows = np.where(edges[:, 0] == i)[0]
        cons.append({"type": "eq", "fun": lambda q, r=rows, i=i: q[r].sum() - pi[i]})
    for i in range(n - 1):
        cols = np.where(edges[:, 1] == i)[0]
        cons.append({"type": "eq", "fun": lambda q, c=cols, i=i: q[c].sum() - pi[i]})
    q0 = np.array([pi[i] / AL[i].sum() for i, _ in edges])
    res = minimize(negH, q0, jac=grad, constraints=cons, method="SLSQP",
                   bounds=[(0, 1)] * len(edges),
                   options={"ftol": 1e-15, "maxiter": 2000})
    assert res.success, res.message
    want = -res.fun
    got = mce_one(pi, A, normalize=False, rule="residual")
    assert abs(got - want) < 1e-6, (got, want)


def test_mce_chain_is_stochastic_with_pi_invariant_and_rows_are_independent():
    """
    The returned residual is the stationarity error of the fitted chain, and
    solving cells together or one at a time gives the same numbers.
    """
    rng = np.random.default_rng(SEED + 5)
    n = 40
    M = np.triu((rng.random((n, n)) < 0.1).astype(float), 1)
    for i in range(n - 1):
        M[i, i + 1] = 1.0
    A = sp.csr_matrix(M + M.T)
    X = rng.gamma(1.5, 1.0, (9, n))
    together, info = mce(X, A, return_info=True, chunk=9, rule="residual")
    alone = np.array([mce_one(x, A, rule="residual") for x in X])
    assert np.all(info["err"] < 1e-12), info["err"]
    assert np.max(np.abs(together - alone)) < 1e-10
    assert np.all((together > 0) & (together <= 1 + 1e-12))


def _mce_m_literal(data, net):
    """
    MCE.m transcribed line by line (SHA256 c8ad11f2...), one sample at a time,
    dense, as the authors' loop runs: data is genes x samples.
    """
    p, n = data.shape
    net = net.copy()
    np.fill_diagonal(net, 1.0)                                   # line 19
    nx = np.count_nonzero(net)                                   # line 21
    ge = np.zeros(n)
    steps = np.zeros(n, dtype=int)
    for i in range(n):
        p0 = data[:, i] / data[:, i].sum()                       # line 26
        B = net * p0[:, None]                                    # line 27
        lambda0, theta0, n_it = p0.copy(), np.ones(p), 0         # lines 63-65
        while True:
            n_it += 1
            lam = 1.0 / (net @ theta0)                           # line 69
            theta = p0 / (B.T @ lam)                             # line 72
            err = np.max(np.abs(np.concatenate([lam - lambda0, theta - theta0])))
            stop = (err < 1e-2) or (n_it > 10 ** 6)              # lines 62, 75
            lambda0, theta0 = lam, theta
            if stop:
                break
        ge[i] = -np.dot(p0, np.log(lambda0 * theta0))            # line 30
        ge[i] -= np.sum(p0[p0 > 0] * np.log(p0[p0 > 0]))         # line 32
        steps[i] = n_it
    return ge / np.log(nx), steps                                # lines 36-37


def test_default_rule_reproduces_a_literal_transcription_of_mce_m():
    """
    The vectorized port, with every cell stopping on its own, gives the same
    scores and the same step counts as MCE.m's per-sample loop.
    """
    rng = np.random.default_rng(SEED + 6)
    n = 60
    M = np.triu((rng.random((n, n)) < 0.08).astype(float), 1)
    for i in range(n - 1):
        M[i, i + 1] = 1.0
    A = M + M.T
    X = rng.gamma(1.2, 1.0, (17, n)) + 0.05                      # cells x genes, > 0
    want, want_steps = _mce_m_literal(X.T, A)
    got, info = mce(X, sp.csr_matrix(A), return_info=True, chunk=5)
    assert np.array_equal(info["iterations"], want_steps), (info["iterations"], want_steps)
    assert np.max(np.abs(got - want)) < 1e-12, np.max(np.abs(got - want))


def test_mce_m_rule_stops_each_cell_on_its_own_and_lands_near_the_fixed_point():
    """
    Cells that converge at different speeds stop at different steps, and the
    1e-2 rule leaves the score close to, but not at, the converged value.
    """
    rng = np.random.default_rng(SEED + 7)
    n = 50
    M = np.triu((rng.random((n, n)) < 0.1).astype(float), 1)
    for i in range(n - 1):
        M[i, i + 1] = 1.0
    A = sp.csr_matrix(M + M.T)
    X = np.vstack([rng.gamma(0.3, 1.0, n) + 1e-3, np.ones(n), rng.gamma(5.0, 1.0, n)])
    got, info = mce(X, A, return_info=True)
    conv = mce(X, A, rule="residual", tol=1e-13)
    assert len(set(info["iterations"].tolist())) > 1, info["iterations"]
    assert np.all(info["err"] < 1e-2)
    assert np.max(np.abs(got - conv)) < 1e-2, np.abs(got - conv)


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import _harness
    sys.exit(_harness.run([sys.modules[__name__]]))
