"""
stats.py: the three statistics the specificity plan adds to the audit.

  conditional_skill     Eqs. (1) to (3) of the paper, through the package's own functions.
  stratified_concordance  Kendall's tau restricted to pairs of cells in the same stratum of
                        the primitive (Korn 1984). Its null is centred on zero by
                        construction, so it needs no floor table: the score is permuted
                        within strata and the statistic recomputed.
  sample_level_test     the replicated unit is the sample, not the cell. The residual of
                        Eq. (2) is averaged within each sample and tested for a trend over
                        the samples' stages, by permuting stage labels between samples
                        inside each sequencing batch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from own_baseline.conditional_null import permute_within, resid_stratum_mean  # noqa: E402
from own_baseline.conditional_skill import align, kang_taub, rank_resid_multi  # noqa: E402


def conditional_skill(s, covs, o):
    """Eq. (3): tau_b between the ordinal and the rank residual of the aligned score."""
    return kang_taub(rank_resid_multi(align(s, o), covs), o)


def _pair_sums(s, o):
    """For blocks (K, m): sum over pairs of sign(ds) * sign(do), and the number of pairs with do != 0."""
    ds = np.sign(s[:, :, None] - s[:, None, :])
    do = np.sign(o[:, :, None] - o[:, None, :])
    return float((ds * do).sum()) / 2.0, float(np.abs(do).sum()) / 2.0


def stratified_concordance(s, o, labels):
    """
    Kendall's tau over pairs of cells that share a stratum and differ in the ordinal:
    (concordant - discordant) / (pairs that differ in the ordinal). Pairs tied in the score
    count zero. Returns (value, number of such pairs).
    """
    order = np.argsort(labels, kind="stable")
    lab, ss, oo = labels[order], s[order], o[order]
    cuts = np.flatnonzero(np.diff(lab)) + 1
    starts = np.concatenate([[0], cuts])
    sizes = np.diff(np.concatenate([starts, [len(lab)]]))
    num = den = 0.0
    for m in np.unique(sizes):
        which = starts[sizes == m]
        idx = which[:, None] + np.arange(m)[None, :]
        a, b = _pair_sums(ss[idx], oo[idx])
        num += a
        den += b
    return (num / den if den > 0 else float("nan")), den


def concordance_test(s, o, labels, rng_for, key, n_draws=200):
    """The stratified concordance of the aligned score and its within-stratum permutation null."""
    sa = align(s, o)
    value, pairs = stratified_concordance(sa, o, labels)
    draws = np.empty(n_draws)
    for i in range(n_draws):
        d = permute_within(s, labels, rng_for(*key, i))
        draws[i] = stratified_concordance(align(d, o), o, labels)[0]
    sd = float(draws.std(ddof=1))
    return {"value": value, "pairs": pairs, "null_mean": float(draws.mean()), "null_sd": sd,
            "null_p975": float(np.quantile(draws, 0.975)), "z": float((value - draws.mean()) / sd) if sd > 0 else float("nan"),
            "clears": bool(value > np.quantile(draws, 0.975)), "n_draws": n_draws}


def _kendall_s(r, o):
    return float((np.sign(r[:, None] - r[None, :]) * np.sign(o[:, None] - o[None, :])).sum()) / 2.0


def sample_level_test(resid, sample, stage, batch=None, *, n_perm=100_000, rng=None, min_cells=5):
    """
    Trend of the per-sample mean residual over the samples' stages.

    resid   (n,) per-cell residual of Eq. (2), any sign convention
    sample  (n,) sample of each cell
    stage   (n,) ordinal of each cell, constant within a sample
    batch   (n,) sequencing batch of each cell, constant within a sample, or None

    The statistic is Kendall's S between the sample means and the sample stages, summed
    over batches. The null permutes stage labels between the samples of a batch, so a
    difference between batches cannot produce a trend and the number of independent units
    is the number of samples. Two-sided Monte Carlo p-value with the +1 correction.
    """
    rng = rng or np.random.default_rng(0)
    ids, inv, counts = np.unique(sample, return_inverse=True, return_counts=True)
    means = np.bincount(inv, resid) / counts
    first = np.array([np.flatnonzero(inv == j)[0] for j in range(len(ids))])
    st = np.asarray(stage, dtype=float)[first]
    bt = np.zeros(len(ids), dtype=int) if batch is None else np.unique(np.asarray(batch)[first], return_inverse=True)[1]
    keep = counts >= min_cells
    means, st, bt = means[keep], st[keep], bt[keep]
    groups = [np.flatnonzero(bt == b) for b in np.unique(bt)]
    groups = [g for g in groups if len(np.unique(st[g])) > 1]
    s_obs = sum(_kendall_s(means[g], st[g]) for g in groups)
    pairs = sum(float((np.abs(np.sign(st[g][:, None] - st[g][None, :]))).sum()) / 2.0 for g in groups)
    s_null = np.zeros(n_perm)
    for g in groups:
        a = np.sign(means[g][:, None] - means[g][None, :])
        for lo in range(0, n_perm, 10_000):
            hi = min(lo + 10_000, n_perm)
            op = st[g][np.argsort(rng.random((hi - lo, len(g))), axis=1)]
            b = np.sign(op[:, :, None] - op[:, None, :])
            s_null[lo:hi] += (a[None, :, :] * b).sum(axis=(1, 2)) / 2.0
    p = float((1 + np.sum(np.abs(s_null) >= abs(s_obs) - 1e-12)) / (n_perm + 1))
    return {"S": s_obs, "tau": s_obs / pairs if pairs else float("nan"), "p_two_sided": p,
            "samples": int(keep.sum()), "samples_dropped": int((~keep).sum()),
            "batches": len(groups), "pairs": pairs, "n_perm": n_perm}


def residual(s, covs, o, labels=None, which="linear"):
    """The residual of Eq. (2) on the aligned score, or its stratum-mean variant."""
    sa = align(s, o)
    return rank_resid_multi(sa, covs) if which == "linear" else resid_stratum_mean(sa, labels)


def rho(a, b):
    return float(spearmanr(a, b).statistic)


def ranks(a):
    return rankdata(a)
