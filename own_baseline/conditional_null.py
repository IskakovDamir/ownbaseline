"""
A floor measured on your own cells, without the simulator.

The shipped floor is read from a simulated null, so it covers a score that is its
primitive plus noise and nothing else. A score and an ordinal that share a curved
function of the primitive clear it with zero true conditional skill, because a
straight line on ranks leaves the curve in the residual. In the calibration of
experiments/conditional_null that happened in every replicate.

The null here is built from the data. If the score carries nothing beyond its
primitive, score and ordinal are independent given the primitive, so permuting the
score among cells with nearly the same primitive value leaves the joint law
unchanged (Berrett et al. 2020, JRSS B 82:175). A draw keeps the real primitive, the
real ordinal, their dependence, and the score's own relation to the primitive,
curve included. The estimator then runs on the draw as it runs on the real score.

    from own_baseline.conditional_null import permutation_floor
    r = permutation_floor(score, ordinal, primitive)
    r["value"], r["floor"], r["clears"]

The ordinal is never used to form strata. With more than one covariate strata are too
wide to hold, in calibration or in principle, so gcm() answers there: score and ordinal
are both regressed on the covariates by cross-fitted boosting and the covariance of the
two residuals is tested.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from .conditional_skill import align, kang_taub, rank_resid_multi

SEED = 42


def rng_for(*key):
    return np.random.default_rng([SEED, *key])


def m_for(n):
    """Cells per stratum. 25 and 100 held the false-clear rate near nominal in the calibration."""
    return 25 if n <= 3000 else 100


def strata_1d(p, m, rng):
    """Consecutive strata of m cells along p; ties broken at random, the last stratum takes the rest."""
    n = len(p)
    shuffled = rng.permutation(n)
    order = shuffled[np.argsort(p[shuffled], kind="stable")]
    labels = np.empty(n, dtype=np.int64)
    labels[order] = np.minimum(np.arange(n) // m, max(n // m - 1, 0))
    return labels


def crossfit(y, X, folds):
    pred = np.empty(len(y))
    for f in np.unique(folds):
        fit = HistGradientBoostingRegressor(max_iter=200, random_state=SEED).fit(X[folds != f], y[folds != f])
        pred[folds == f] = fit.predict(X[folds == f])
    return pred


def gcm(score, ordinal, primitives, *, key=(0,), polarity=None):
    """
    The generalised covariance measure on ranks (Shah and Peters 2020, Ann. Statist. 48:1514).

    Both the score's rank and the ordinal's rank are regressed on the covariates by
    cross-fitted gradient boosting, so a curve shared by score and ordinal leaves both.
    T is standard normal under conditional independence when the two fits converge. The
    score is aligned first, Eq. (1), unless polarity fixes its sign and with it the
    orientation of the ordinal. Returns T, the partial rank correlation and clears. The
    threshold is 3.5 and not 1.96: at k = 4 and large n the boosted fits leave a little of a
    curve behind and T drifts upward, to a mean of 1.45 in the worst calibration case.
    """
    s = np.asarray(score, dtype=float)
    o = np.asarray(ordinal, dtype=float)
    covs = primitives if isinstance(primitives, (list, tuple)) else [primitives]
    X = np.column_stack([rankdata(np.asarray(c, dtype=float)) for c in covs])
    aligned = align(s, o) if polarity is None else polarity * s
    folds = rng_for(4, *key).integers(0, 5, len(s))
    rs = rankdata(aligned) - crossfit(rankdata(aligned), X, folds)
    ro = rankdata(o) - crossfit(rankdata(o), X, folds)
    prod = rs * ro
    t = float(np.sqrt(len(s)) * prod.mean() / prod.std())
    r = float(np.corrcoef(rs, ro)[0, 1])
    return {"T": t, "partial_rank_correlation": r, "clears": bool(t > 3.5)}


def permute_within(s, labels, rng):
    base = np.argsort(labels, kind="stable")
    drawn = np.lexsort((rng.random(len(s)), labels))
    out = np.empty_like(s)
    out[base] = s[drawn]
    return out


def resid_isotonic(s, p):
    y, x = rankdata(s), rankdata(p)
    rising = spearmanr(x, y).statistic >= 0
    fit = IsotonicRegression(increasing=bool(rising), out_of_bounds="clip").fit(x, y)
    return y - fit.predict(x)


def resid_stratum_mean(s, labels):
    y = rankdata(s)
    return y - (np.bincount(labels, y) / np.bincount(labels))[labels]


def residuals(s, covs, labels, which):
    if which == "linear":
        return rank_resid_multi(s, covs)
    if which == "isotonic":
        return resid_isotonic(s, covs[0])
    return resid_stratum_mean(s, labels)


def cs_value(s, covs, labels, o, which, polarity=None):
    """Conditional skill. polarity None is the registered reading, the sign taken from the data."""
    if polarity is None:
        return kang_taub(residuals(align(s, o), covs, labels, which), o)
    return kang_taub(residuals(polarity * s, covs, labels, which), o)


def permutation_floor(score, ordinal, primitives, *, residual="linear", m=None, n_draws=200, key=(0,)):
    """
    Conditional skill against a floor measured on these cells.

    Parameters
    ----------
    score, ordinal : (n,) arrays, finite
    primitives : (n,) array, or a one-element sequence
    residual : "linear", the registered fit; "stratum_mean", which also removes a
        non-monotone trace; "isotonic"
    m : cells per stratum, default m_for(n)
    n_draws : draws of the null
    key : seed stream

    Returns a dict with value, floor (the 97.5th percentile of the draws), clears,
    null_mean, m and strata.
    """
    s = np.asarray(score, dtype=float)
    o = np.asarray(ordinal, dtype=float)
    first = np.asarray(primitives[0] if isinstance(primitives, (list, tuple)) else primitives, dtype=float)
    covs = [np.asarray(c, dtype=float) for c in primitives] if isinstance(primitives, (list, tuple)) else [first]
    if len(covs) > 1:
        raise ValueError("strata were calibrated for one primitive; use gcm() for several")
    m = m or m_for(len(s))
    labels = strata_1d(covs[0], m, rng_for(2, *key))
    v = cs_value(s, covs, labels, o, residual)
    d = np.array([cs_value(permute_within(s, labels, rng_for(3, *key, i)), covs, labels, o, residual)
                  for i in range(n_draws)])
    q = float(np.quantile(d, 0.975))
    return {"value": v, "floor": q, "clears": bool(v > q), "null_mean": float(d.mean()), "m": m,
            "strata": int(labels.max()) + 1, "n_draws": n_draws, "residual": residual}
