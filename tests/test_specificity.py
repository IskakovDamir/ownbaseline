"""
Tests of the statistics added by experiments/specificity, against answers known in advance:
brute-force pair counts, identities that hold by construction, and planted cases.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "specificity"))

from common import rng_for  # noqa: E402
from controls import gene_set_scores, pcc_degree, thin_counts  # noqa: E402
from stats import concordance_test, residual, sample_level_test, stratified_concordance  # noqa: E402

from own_baseline.conditional_null import strata_1d  # noqa: E402

SEED = 20261122


def _brute(s, o, labels):
    num = den = 0
    for lab in np.unique(labels):
        i = np.flatnonzero(labels == lab)
        for a in range(len(i)):
            for b in range(a + 1, len(i)):
                num += np.sign(s[i[a]] - s[i[b]]) * np.sign(o[i[a]] - o[i[b]])
                den += o[i[a]] != o[i[b]]
    return num / den, den


def test_stratified_concordance_equals_the_brute_force_pair_count():
    rng = np.random.default_rng(SEED)
    n = 437                                    # not a multiple of the stratum size: the last stratum is larger
    s, o = rng.standard_normal(n), rng.integers(1, 8, n).astype(float)
    labels = strata_1d(rng.standard_normal(n), 50, rng)
    value, pairs = stratified_concordance(s, o, labels)
    expect, expect_pairs = _brute(s, o, labels)
    assert abs(value - expect) < 1e-12 and pairs == expect_pairs


def test_stratified_concordance_is_one_for_a_score_monotone_in_the_ordinal_within_strata():
    rng = np.random.default_rng(SEED + 1)
    n = 600
    o = rng.integers(1, 13, n).astype(float)
    labels = np.repeat(np.arange(6), 100)
    s = o + 100.0 * labels + 0.01 * rng.random(n)   # ordered by stage inside a stratum, by stratum across
    assert stratified_concordance(s, o, labels)[0] == 1.0


def test_concordance_null_is_centred_when_the_score_carries_nothing_beyond_the_primitive():
    rng = np.random.default_rng(SEED + 2)
    n = 4000
    p = rng.standard_normal(n)
    o = np.clip(np.round(6 + 2 * p + rng.standard_normal(n)), 1, 12)
    s = p ** 3 + 0.5 * rng.standard_normal(n)       # a curved function of the primitive plus noise
    t = concordance_test(s, o, strata_1d(p, 25, rng), rng_for, (901,), n_draws=100)
    assert abs(t["null_mean"]) < 0.01
    assert abs(t["z"]) < 4.0


def test_concordance_detects_a_planted_signal_beyond_the_primitive():
    rng = np.random.default_rng(SEED + 3)
    n = 4000
    p, b = rng.standard_normal(n), rng.standard_normal(n)
    o = np.clip(np.round(6 + 1.5 * p + 1.5 * b), 1, 12)
    s = p + 0.5 * b + rng.standard_normal(n)
    t = concordance_test(s, o, strata_1d(p, 25, rng), rng_for, (902,), n_draws=100)
    assert t["clears"] and t["z"] > 6.0


def _samples(rng, per_sample=40):
    stage = np.array([1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 6, 2, 4, 6, 1, 3, 5], dtype=float)
    batch = np.array([0] * 6 + [1] * 6 + [2] * 6)
    sample = np.repeat(np.arange(len(stage)), per_sample)
    return sample, stage[sample], batch[sample]


def test_sample_test_keeps_its_level_when_sample_effects_are_unrelated_to_stage():
    hits = 0
    for k in range(200):
        rng = np.random.default_rng([SEED, k])
        sample, stage, batch = _samples(rng)
        r = rng.standard_normal(len(sample)) + rng.standard_normal(18)[sample]   # a large effect per sample
        hits += sample_level_test(r, sample, stage, batch, n_perm=999, rng=rng)["p_two_sided"] < 0.05
    assert hits <= 20                              # nominal 10 of 200; 20 is far in the binomial tail


def test_sample_test_finds_a_trend_that_holds_inside_every_batch():
    rng = np.random.default_rng(SEED + 4)
    sample, stage, batch = _samples(rng)
    r = 0.3 * stage + 5.0 * batch + 0.1 * rng.standard_normal(len(sample))      # a batch offset must not matter
    t = sample_level_test(r, sample, stage, batch, n_perm=20_000, rng=rng)
    assert t["tau"] == 1.0 and t["p_two_sided"] < 0.001 and t["batches"] == 3


def test_sample_test_ignores_a_difference_between_batches():
    rng = np.random.default_rng(SEED + 5)
    stage = np.arange(1, 13, dtype=float)             # six early stages in one batch, six late in the other
    batch = np.repeat([0, 1], 6)
    sample = np.repeat(np.arange(12), 30)
    within = np.array([0.1, -0.3, 0.2, -0.1, 0.3, -0.2, -0.2, 0.3, -0.1, 0.2, -0.3, 0.1])   # no trend inside either
    r = 10.0 * batch[sample] + within[sample]         # the late batch sits far above the early one
    strat = sample_level_test(r, sample, stage[sample], batch[sample], n_perm=5000, rng=rng)
    pooled = sample_level_test(r, sample, stage[sample], None, n_perm=5000, rng=rng)
    assert abs(strat["S"]) <= 2 and strat["p_two_sided"] > 0.5
    assert pooled["p_two_sided"] < 0.05


def test_thinning_draws_every_kept_cell_down_to_the_same_depth_without_inventing_counts():
    rng = np.random.default_rng(SEED + 6)
    x = sp.csr_matrix(rng.poisson(0.4, size=(60, 500)).astype(np.float32))
    lib = np.asarray(x.sum(axis=1)).ravel()
    depth = int(np.median(lib))
    keep, xt = thin_counts(x, depth, (77,))
    assert set(keep) == set(np.flatnonzero(lib >= depth))
    assert np.all(np.asarray(xt.sum(axis=1)).ravel() == depth)
    assert (xt.toarray() <= x[keep].toarray()).all()


def test_degree_correlation_matches_numpy_row_by_row():
    rng = np.random.default_rng(SEED + 7)
    x = sp.csr_matrix(rng.poisson(0.5, size=(20, 300)).astype(np.float64))
    deg = rng.integers(1, 200, size=(300, 3)).astype(float)
    got = pcc_degree(x, deg)
    dense = x.toarray()
    for i in range(20):
        for j in range(3):
            assert abs(got[i, j] - np.corrcoef(dense[i], deg[:, j])[0, 1]) < 1e-10


def test_gene_set_scores_are_the_mean_over_the_drawn_genes():
    rng = np.random.default_rng(SEED + 8)
    x = sp.csr_matrix(rng.random((15, 80)) * (rng.random((15, 80)) < 0.3))
    scores, sets = gene_set_scores(x, np.arange(80), 10, 5, (78,))
    dense = x.toarray()
    for j in range(5):
        assert len(set(sets[j])) == 10
        assert np.allclose(scores[:, j], dense[:, sets[j]].mean(axis=1), atol=1e-6)


def test_sample_test_needs_the_stratum_mean_residual_when_the_score_is_curved_in_its_primitive():
    """
    Found by the adversarial review of experiments/specificity/PREREG.md before any label was read.
    The score is a square of its primitive plus noise, so it carries nothing beyond the primitive,
    and no sample differs from another except through the primitive. A straight line on ranks leaves
    the curve in the residual, the sample means inherit it, and the trend test rejects. The
    stratum-mean residual removes the curve. The design copies the audit's: which stages share a
    sequencing batch, and fewer cells in the earliest samples.
    """
    rng = np.random.default_rng(SEED + 9)
    stages = {0: [1, 1, 2, 2, 3, 4, 4, 6, 11, 12, 12], 1: [5, 7, 8, 9, 10, 10], 2: [5, 7, 8, 9, 10], 3: [5, 5, 7, 8, 9, 10]}
    stage_of = np.array([st for b in stages for st in stages[b]], dtype=float)
    batch_of = np.array([b for b in stages for _ in stages[b]])
    size_of = np.where(stage_of <= 2, 200, np.where(stage_of <= 4, 800, 1600)).astype(int)
    sample = np.repeat(np.arange(len(stage_of)), size_of)
    o, batch = stage_of[sample], batch_of[sample]
    z = (o - o.mean()) / o.std()
    p = 0.55 * z + np.sqrt(1 - 0.55 ** 2) * rng.standard_normal(len(o))
    s = p ** 2 + rng.standard_normal(len(o))
    labels = strata_1d(p, 100, rng)
    least_squares = sample_level_test(residual(s, [p], o, labels, "linear"), sample, o, batch, n_perm=4000, rng=rng)
    stratum_mean = sample_level_test(residual(s, [p], o, labels, "stratum_mean"), sample, o, batch, n_perm=4000, rng=rng)
    assert least_squares["p_two_sided"] < 0.1
    assert stratum_mean["p_two_sided"] > 0.2
