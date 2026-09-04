"""
Tests of the estimator against cases whose answer is known in advance.

None of these assert the current output. Every expected value is either an
identity that must hold by construction, a closed form derived independently of
this repository, or an invariant stated in the paper.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, rankdata, weightedtau

from own_baseline import (
    align,
    conditional_skill_report,
    kang_taub,
    nnz_per_cell,
    rank_resid_multi,
    run_own_baseline,
    scipy_wtau,
)
from own_baseline.atlas_run.wdm_tau import wdm_tau_b, wdm_tau_b_bit
from own_baseline.potency_metrics import (
    _ccat_from_matrix,
    library_normalize,
    stemid_entropy,
)

from _harness import known_defect

SEED = 20260813


def _ordinal(n, levels=6, seed=SEED):
    return np.floor(levels * np.random.default_rng(seed).random(n))


# ===========================================================================
# 1. A score that is an exact monotone function of its primitive has no
#    conditional skill. This is the defining property of the estimator: if it
#    does not hold, "adds beyond the primitive" does not mean what it says.
# ===========================================================================

@known_defect(
    "F-2",
    "rank_resid_multi returns floating-point rounding debris (1e-12, a few dozen "
    "distinct values) instead of an exact zero when rank(score) == "
    "rank(primitive). The debris is monotone in the primitive rank, so what a "
    "kernel returns depends on whether the primitive predicts the ordinal. THIS "
    "fixture draws the ordinal independently of it, and there only weightedtau "
    "misbehaves, reaching +0.51 where 0 is the only correct answer, while tau_b "
    "stays near 0.02. That is the narrow case. Where the primitive predicts the "
    "ordinal, which is Null B and the case the audit faces, the debris inherits "
    "that association and BOTH kernels score it, reaching 0.87 in magnitude with "
    "a sign that changes with the seed. The size depends on the machine, because "
    "the debris is the rounding error of a LAPACK fit: aarch64 with numpy 2.2 "
    "puts all sixteen measured cells between 0.71 and 0.87, x86_64 with numpy 2.4 "
    "puts fourteen there and collapses two toward zero. Pinned in tests/test_cli.py::"
    "test_debris_is_scored_by_both_kernels_when_the_primitive_predicts_the_ordinal. "
    "NOT FIXED -- fixing it changes what the estimator computes near rho = 1, and "
    "CCAT sits at rho = 0.998. Guarded instead: own_baseline.cli.residual_scale "
    "returns max|resid| / (eps * n), real residuals sit at 1e14 to 1e15 on that "
    "ratio and debris at 1, and `ownbaseline check` refuses to report below 1e6.",
)
def test_exact_monotone_function_of_primitive_has_zero_conditional_skill():
    rng = np.random.default_rng(SEED)
    n = 3000
    primitive = rng.standard_normal(n)
    ordinal = _ordinal(n)
    for transform in (np.exp, lambda v: v ** 3, lambda v: 10 * v + 7, rankdata):
        score = np.asarray(transform(primitive), dtype=float)
        assert np.array_equal(rankdata(score), rankdata(primitive)), \
            "test setup: transform must be rank-preserving"
        resid = rank_resid_multi(align(score, ordinal), [primitive])
        for kernel, tau in (("weightedtau", scipy_wtau(resid, ordinal)),
                            ("kendalltau", kang_taub(resid, ordinal))):
            assert abs(tau) < 1e-6, (
                f"{kernel}: score is an exact monotone function of the "
                f"primitive, so conditional skill must be 0, got {tau:+.6f}")


def test_residual_of_exact_monotone_function_is_numerically_negligible():
    """The residual itself IS ~0; it is the tau of it that misbehaves (F-2)."""
    rng = np.random.default_rng(SEED)
    primitive = rng.standard_normal(2000)
    resid = rank_resid_multi(np.exp(primitive), [primitive])
    assert np.abs(resid).max() < 1e-9, (
        f"residual should be float noise, got max |r| = {np.abs(resid).max():.3e}")


# ===========================================================================
# 2. Orientation invariance: a primitive at rank correlation -r must be
#    removed exactly as strongly as one at +r. The paper leans on this --
#    "rank residualization is orientation-invariant" -- to argue that a
#    beaten-but-reversed baseline does not survive as a margin.
# ===========================================================================

def test_residualization_is_invariant_to_the_sign_of_the_primitive():
    rng = np.random.default_rng(SEED)
    n = 1500
    primitive = rng.standard_normal(n)
    score = 0.7 * primitive + 0.7 * rng.standard_normal(n)
    r_pos = rank_resid_multi(score, [primitive])
    r_neg = rank_resid_multi(score, [-primitive])
    assert np.allclose(r_pos, r_neg, atol=1e-8), (
        "residualizing on p and on -p must give the same residual; "
        f"max difference {np.abs(r_pos - r_neg).max():.3e}")


def test_conditional_skill_is_invariant_to_the_sign_of_the_primitive():
    rng = np.random.default_rng(SEED)
    n = 1200
    primitive = rng.standard_normal(n)
    ordinal = _ordinal(n)
    score = 0.6 * primitive + 0.8 * rng.standard_normal(n)
    a = conditional_skill_report(score, ordinal, {"p": primitive},
                                 n_boot=(30, 60), verbose=False)
    b = conditional_skill_report(score, ordinal, {"p": -primitive},
                                 n_boot=(30, 60), verbose=False)
    for kernel in ("scipy_weightedtau", "kang_wdm_taub"):
        ta = a["by_primitive"]["p"]["conditional_skill"][kernel]["tau"]
        tb = b["by_primitive"]["p"]["conditional_skill"][kernel]["tau"]
        assert abs(ta - tb) < 1e-8, (
            f"{kernel}: flipping the primitive's sign changed conditional "
            f"skill from {ta:+.6f} to {tb:+.6f}")


def test_an_affine_rescaling_of_the_primitive_changes_nothing():
    rng = np.random.default_rng(SEED)
    n = 800
    primitive = rng.standard_normal(n)
    score = 0.5 * primitive + rng.standard_normal(n)
    base = rank_resid_multi(score, [primitive])
    for a, b in ((3.0, 0.0), (-2.0, 11.0), (0.001, -4.0)):
        assert np.allclose(base, rank_resid_multi(score, [a * primitive + b]),
                           atol=1e-8), f"failed for {a} * p + {b}"


# ===========================================================================
# 3. Known permutations give known taus. Closed forms derived from the
#    definitions, not read off this implementation.
# ===========================================================================

def test_kendall_tau_b_matches_the_inversion_count_closed_form():
    """For a permutation with no ties: tau_b = 1 - 4 * inversions / (n(n-1))."""
    cases = {
        6: [[0, 1, 2, 3, 4, 5], [1, 0, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0],
            [0, 2, 1, 4, 3, 5], [2, 0, 1, 5, 3, 4]],
        9: [[0, 1, 2, 3, 4, 5, 6, 7, 8], [8, 7, 6, 5, 4, 3, 2, 1, 0],
            [1, 0, 3, 2, 5, 4, 7, 6, 8]],
    }
    for n, perms in cases.items():
        x = np.arange(n, dtype=float)
        for perm in perms:
            y = np.asarray(perm, dtype=float)
            inversions = sum(1 for i in range(n) for j in range(i + 1, n)
                             if y[i] > y[j])
            expected = 1.0 - 4.0 * inversions / (n * (n - 1))
            got = wdm_tau_b_bit(x, y, np.ones(n))
            assert abs(got - expected) < 1e-12, (
                f"n={n} perm={perm}: {inversions} inversions imply tau_b = "
                f"{expected:+.12f}, wdm_tau_b_bit gave {got:+.12f}")
            assert abs(wdm_tau_b(x, y, np.ones(n)) - expected) < 1e-12, \
                "dense wdm_tau_b disagrees with the closed form"
            assert abs(kang_taub(x, y) - expected) < 1e-12, \
                "kang_taub disagrees with the closed form"


def test_weighted_tau_of_the_identity_and_reversal_permutations():
    """Any weighting scheme must give exactly +1 and -1 for these two."""
    for n in (5, 20, 200):
        x = np.arange(n, dtype=float)
        assert abs(weightedtau(x, x).statistic - 1.0) < 1e-12
        assert abs(scipy_wtau(x, x) - 1.0) < 1e-12, "identity must be +1"
        assert abs(scipy_wtau(x, x[::-1].copy()) + 1.0) < 1e-12, \
            "reversal must be -1"


def test_weighted_tau_is_top_weighted():
    """
    The defining property of the hyperbolic weigher: an inversion among the
    highest ranks costs more than the same inversion among the lowest. A
    plain tau-b cannot tell the two apart; this is why the paper runs both.
    """
    n = 40
    x = np.arange(n, dtype=float)

    def swap_at(i):
        y = x.copy()
        y[i], y[i + 1] = y[i + 1], y[i]
        return y

    top, bottom = scipy_wtau(x, swap_at(n - 2)), scipy_wtau(x, swap_at(0))
    assert top < bottom, (
        f"swapping the top pair ({top:+.6f}) must cost more than swapping the "
        f"bottom pair ({bottom:+.6f})")
    # tau-b, by contrast, must be indifferent: one inversion either way.
    assert abs(kang_taub(x, swap_at(n - 2)) - kang_taub(x, swap_at(0))) < 1e-12


def test_wdm_tau_b_equals_kendalltau_under_uniform_weights():
    rng = np.random.default_rng(SEED)
    for n in (50, 300):
        x = rng.standard_normal(n)
        y = x + 0.5 * rng.standard_normal(n)
        expected = kendalltau(x, y).statistic
        assert abs(wdm_tau_b(x, y, np.ones(n)) - expected) < 1e-10
        assert abs(wdm_tau_b_bit(x, y, np.ones(n)) - expected) < 1e-10


def test_wdm_tau_b_dense_and_fenwick_agree_under_weights_and_ties():
    rng = np.random.default_rng(SEED)
    n = 400
    w = rng.uniform(0.1, 2.0, n)
    for x, y in ((rng.standard_normal(n), np.floor(4 * rng.random(n))),
                 (np.floor(6 * rng.random(n)), np.floor(4 * rng.random(n))),
                 (np.floor(3 * rng.random(n)), np.floor(3 * rng.random(n)))):
        a, b = wdm_tau_b(x, y, w), wdm_tau_b_bit(x, y, w)
        assert abs(a - b) < 1e-9, f"dense {a:+.12f} vs Fenwick {b:+.12f}"


# ===========================================================================
# 4. Bootstrap intervals must contain the point estimate.
# ===========================================================================

def test_bootstrap_interval_contains_the_point_estimate():
    rng = np.random.default_rng(SEED)
    n = 700
    ordinal = _ordinal(n)
    primitive = rng.standard_normal(n)
    cases = {
        "independent": rng.standard_normal(n),
        "weakly_related": 0.3 * primitive + 1.0 * rng.standard_normal(n),
        "strongly_related": 0.9 * primitive + 0.2 * rng.standard_normal(n),
        "carries_the_ordinal": 0.4 * primitive + 1.5 * ordinal,
        "anti_aligned": -(0.4 * primitive + 1.5 * ordinal),
    }
    for name, score in cases.items():
        rep = conditional_skill_report(score, ordinal, {"p": primitive},
                                       n_boot=(200, 400), verbose=False)
        for kernel in ("scipy_weightedtau", "kang_wdm_taub"):
            r = rep["by_primitive"]["p"]["conditional_skill"][kernel]
            lo, hi = r["CI95"]
            assert lo <= r["tau"] <= hi, (
                f"{name}/{kernel}: point estimate {r['tau']:+.6f} outside its "
                f"own 95% interval [{lo:+.6f}, {hi:+.6f}]")
            assert lo <= hi, f"{name}/{kernel}: interval is inverted"


# ===========================================================================
# 5. The marginal-gap tool: identities that must hold exactly.
# ===========================================================================

def _toy_adata(n_cells=300, n_genes=400, seed=SEED):
    import anndata as ad
    rng = np.random.default_rng(seed)
    gt = rng.integers(0, 4, n_cells).astype(float)
    counts = np.zeros((n_cells, n_genes))
    for c in range(n_cells):
        counts[c] = rng.poisson(1.5) * (rng.random(n_genes) < 0.4 + 0.03 * gt[c])
    a = ad.AnnData(X=counts)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    a.obs["gt"] = gt
    return a, counts, gt


def test_score_equal_to_the_primitive_gives_exactly_zero_marginal_gap():
    adata, _, _ = _toy_adata()
    r = run_own_baseline(adata, score=nnz_per_cell, gt_ordinal="gt",
                         verbose=False)
    assert r["delta"] == 0.0, f"expected exactly 0.0, got {r['delta']}"
    assert r["verdict"] == "REDUCES-TO-PRIMITIVE"
    assert r["tau_score"] == r["tau_nnz"]


def test_a_score_that_is_the_ordinal_adds_beyond_the_primitive():
    adata, _, gt = _toy_adata()
    r = run_own_baseline(adata, score=lambda _a: gt, gt_ordinal="gt",
                         verbose=False)
    assert r["verdict"] == "ADDS-BEYOND-PRIMITIVE", r
    assert r["delta"] > 0.10
    assert r["tau_score"] == 1.0, (
        f"a score identical to the ordinal must have tau 1.0, got {r['tau_score']}")


def test_verdict_band_edges_use_the_documented_thresholds():
    from own_baseline import ADDS_DELTA, REDUCES_DELTA
    assert REDUCES_DELTA == 0.05
    assert ADDS_DELTA == 0.10


def test_nnz_counts_nonzero_genes_exactly_and_agrees_across_sparse_and_dense():
    import scipy.sparse as sp
    rng = np.random.default_rng(SEED)
    X = (rng.random((60, 90)) < 0.3) * rng.integers(1, 9, (60, 90))
    expected = (X > 0).sum(axis=1).astype(float)
    assert np.array_equal(nnz_per_cell(X), expected)
    assert np.array_equal(nnz_per_cell(sp.csr_matrix(X)), expected)
    # a row of all zeros must be 0, a full row must be n_genes
    X[0] = 0
    X[1] = 5
    assert nnz_per_cell(X)[0] == 0
    assert nnz_per_cell(X)[1] == X.shape[1]


# ===========================================================================
# 6. Primitives: closed forms.
# ===========================================================================

def test_shannon_entropy_of_a_uniform_cell_is_log2_of_the_support():
    import anndata as ad
    n_genes = 64
    X = np.zeros((3, n_genes))
    X[0, :] = 1.0            # uniform over 64 genes -> log2(64) = 6
    X[1, :8] = 1.0           # uniform over 8        -> log2(8)  = 3
    X[2, 0] = 1.0            # a point mass          -> 0
    a = ad.AnnData(X=X)
    a.var_names = [f"g{i}" for i in range(n_genes)]
    h = stemid_entropy(a, base=2.0)
    assert abs(h[0] - 6.0) < 1e-12, h[0]
    assert abs(h[1] - 3.0) < 1e-12, h[1]
    assert abs(h[2] - 0.0) < 1e-12, h[2]


def test_ccat_of_an_expression_vector_proportional_to_degree_is_one():
    degree = np.array([1.0, 2, 3, 4, 5, 8, 13, 21])
    X = np.vstack([3.0 * degree, -2.0 * degree + 7.0, np.ones_like(degree)])
    got = _ccat_from_matrix(X, degree)
    assert abs(got[0] - 1.0) < 1e-12, got[0]
    assert abs(got[1] + 1.0) < 1e-12, got[1]
    assert got[2] == 0.0, "a constant expression vector has no correlation"


def test_library_normalize_makes_every_row_sum_to_the_target():
    rng = np.random.default_rng(SEED)
    X = rng.integers(0, 20, (40, 30)).astype(float)
    out = library_normalize(X, target_sum=10_000.0)
    assert np.allclose(out.sum(axis=1), 10_000.0)


# ===========================================================================
# 7. The report wrapper: alignment, direction check, masking, validation.
# ===========================================================================

def test_a_score_with_reversed_polarity_is_aligned_not_penalised():
    rng = np.random.default_rng(SEED)
    n = 900
    ordinal = _ordinal(n)
    primitive = rng.standard_normal(n)
    score = 0.3 * primitive + 1.2 * ordinal
    up = conditional_skill_report(score, ordinal, {"p": primitive},
                                  n_boot=(30, 60), verbose=False)
    down = conditional_skill_report(-score, ordinal, {"p": primitive},
                                    n_boot=(30, 60), verbose=False)
    assert down["score_sign_flipped_to_align"] is True
    assert up["score_sign_flipped_to_align"] is False
    for kernel in ("scipy_weightedtau", "kang_wdm_taub"):
        a = up["by_primitive"]["p"]["conditional_skill"][kernel]["tau"]
        b = down["by_primitive"]["p"]["conditional_skill"][kernel]["tau"]
        assert abs(a - b) < 1e-9, (
            f"{kernel}: negating the score changed conditional skill "
            f"{a:+.6f} -> {b:+.6f}; the estimator is meant to be sign-agnostic")


def test_direction_check_flags_a_primitive_that_orders_below_chance():
    """
    The haematopoietic case: the primitive runs opposite to the ordinal, so a
    positive marginal gap is a sign correction. The report must say so.
    """
    rng = np.random.default_rng(SEED)
    n = 600
    ordinal = _ordinal(n)
    reversed_primitive = -ordinal + 0.3 * rng.standard_normal(n)
    score = ordinal + 0.2 * rng.standard_normal(n)
    rep = conditional_skill_report(score, ordinal, {"gene_count": reversed_primitive},
                                   n_boot=(30, 60), verbose=False)
    d = rep["by_primitive"]["gene_count"]["direction_check"]
    assert d["primitive_orders_below_chance"] is True
    assert d["spearman_primitive_vs_ordinal"] < 0
    gap = rep["by_primitive"]["gene_count"]["marginal_gap"]["scipy_weightedtau"]
    assert gap > 0, "test setup: this is the case where the naive gap looks good"


def test_non_finite_rows_are_dropped_jointly():
    rng = np.random.default_rng(SEED)
    n = 500
    ordinal = _ordinal(n)
    primitive = rng.standard_normal(n)
    score = 0.5 * primitive + rng.standard_normal(n)
    score[:7] = np.nan
    primitive[10:13] = np.inf
    ordinal[400] = np.nan
    rep = conditional_skill_report(score, ordinal, {"p": primitive},
                                   n_boot=(20, 20), verbose=False)
    assert rep["n_input"] == n
    assert rep["n_used"] == n - 11
    assert rep["n_dropped_nonfinite"] == 11


def test_mismatched_lengths_are_rejected():
    for kwargs in ({"score": np.zeros(10), "ordinal": np.zeros(9),
                    "primitives": {"p": np.zeros(10)}},
                   {"score": np.zeros(10), "ordinal": np.zeros(10),
                    "primitives": {"p": np.zeros(11)}}):
        try:
            conditional_skill_report(verbose=False, n_boot=(5, 5), **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"expected ValueError for {kwargs.keys()}")


def test_joint_residualization_removes_at_least_as_much_as_any_single_one():
    """
    Residualizing on {p1, p2} spans a superset of the space spanned by {p1}
    alone, so the joint residual cannot retain more rank-linear structure.
    """
    rng = np.random.default_rng(SEED)
    n = 1200
    p1, p2 = rng.standard_normal(n), rng.standard_normal(n)
    score = 0.6 * p1 + 0.5 * p2 + 0.4 * rng.standard_normal(n)
    r1 = rank_resid_multi(score, [p1])
    rj = rank_resid_multi(score, [p1, p2])
    assert np.sum(rj ** 2) <= np.sum(r1 ** 2) + 1e-6, (
        "the joint residual has larger sum of squares than the single-covariate "
        "residual, which is impossible for a least-squares fit on a superset")


def test_report_includes_both_kernels_and_a_combined_verdict():
    rng = np.random.default_rng(SEED)
    n = 400
    ordinal = _ordinal(n)
    p1, p2 = rng.standard_normal(n), rng.standard_normal(n)
    score = 0.4 * p1 + ordinal
    rep = conditional_skill_report(score, ordinal, {"a": p1, "b": p2},
                                   n_boot=(20, 40), verbose=False)
    assert set(rep["by_primitive"]) == {"a", "b", "JOINT"}
    for name, block in rep["by_primitive"].items():
        cs = block["conditional_skill"]
        assert {"scipy_weightedtau", "kang_wdm_taub"} <= set(cs)
        assert cs["combined_verdict"] in {
            "TAUTOLOG", "ADDS-BEYOND", "SIGN-FLIPPED", "INCONCLUSIVE",
            "INCONCLUSIVE-kernel-disagree"}
        assert "direction_check" in block and "marginal_gap" in block


# ===========================================================================
# 8. Paths resolve without reference to any one machine.
# ===========================================================================

def test_paths_default_inside_the_repository_and_honour_the_environment():
    import os
    from pathlib import Path

    from own_baseline.paths import data_root, repo_root, scratch_root

    root = repo_root()
    assert (root / "own_baseline" / "paths.py").is_file()
    saved = {k: os.environ.pop(k, None)
             for k in ("OWNBASELINE_DATA_ROOT", "OWNBASELINE_SCRATCH")}
    try:
        assert data_root() == root / "data" / "runs"
        assert scratch_root() == root / "data" / "scratch"
        os.environ["OWNBASELINE_DATA_ROOT"] = "/tmp/ob-test-root"
        assert data_root() == Path("/tmp/ob-test-root").resolve()
    finally:
        os.environ.pop("OWNBASELINE_DATA_ROOT", None)
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v


def test_no_source_file_hard_codes_an_absolute_home_path():
    """
    The guard is about what is committed, so it reads what git tracks rather
    than walking the directory. rglob() also swept up whatever happened to be
    sitting in the tree: a .venv, an egg-info from `pip install -e .`, a build/
    from a wheel. Several dependencies carry example home paths inside their
    own docstrings, so a virtualenv in the checkout failed this test on content
    nobody here wrote.
    """
    import subprocess

    import pytest

    from own_baseline.paths import repo_root
    root = repo_root()
    tracked = subprocess.run(["git", "-C", str(root), "ls-files", "-z"],
                             capture_output=True, text=True)
    if tracked.returncode != 0:
        pytest.skip("not a git checkout, so there is no tracked-file list to read")
    # Assembled from fragments so this file does not match its own guard.
    needles = ("/Us" + "ers/", "/priv" + "ate/tmp/", "/ho" + "me/")
    offenders = []
    for rel in sorted(f for f in tracked.stdout.split("\0") if f):
        path = root / rel
        if not path.is_file() or path.suffix not in {".py", ".R", ".md", ".txt"}:
            continue
        for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
            stripped = line.lstrip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue  # explanatory comments may name the old paths
            if any(nd in line for nd in needles):
                offenders.append(f"{path.relative_to(root)}:{i}: {line.strip()}")
    assert not offenders, "hard-coded absolute paths:\n  " + "\n  ".join(offenders)
