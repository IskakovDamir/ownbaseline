"""
wdm_tau.py
==========
Efficient weighted Kendall tau-b implementation that emulates
R `wdm::wdm(x, y, method="kendall", weights=w)` — the exact protocol
Kang et al. 2025 (Nat Methods, doi:10.1038/s41592-025-02857-2) uses to
evaluate CytoTRACE 2 and 8 benchmarking methods.

Formula (weighted tau-b, per-observation weights):

  Cw = sum_{i<j} w_i w_j [x_i<x_j][y_i<y_j] + [x_i>x_j][y_i>y_j]      (concordant)
  Dw = sum_{i<j} w_i w_j [x_i<x_j][y_i>y_j] + [x_i>x_j][y_i<y_j]      (discordant)
  Tx = sum_{i<j} w_i w_j [x_i==x_j]                                    (x-ties)
  Ty = sum_{i<j} w_i w_j [y_i==y_j]                                    (y-ties)

  numer = Cw - Dw
  denom = sqrt( (Cw + Dw + Tx) * (Cw + Dw + Ty) )
  tau_b = numer / denom

Algorithm: O(n^2) direct implementation, chunked to bound memory.

For n up to 30k this is 30k^2 = 9e8 pair operations. Naive: ~10s per score
if we chunk properly with numpy vector broadcast (2000-row chunks × 30k cols).

Reference: Vanderwerken et al. (2016); Nešlehová & Machler R:wdm package;
Newson (2002) Somers' D and weighted tau-b.

Validation: reproduces Kang Supplementary Table S12 relative-order tau
values to <0.02 absolute tolerance on 'Retinal neurons (10x)' (verified
2026-07-17 in atlas_run/metric_compat_probe log).
"""
from __future__ import annotations
import numpy as np


def wdm_tau_b(x, y, w, chunk: int = 1024) -> float:
    """
    Weighted Kendall tau-b (dense O(n^2)), matching R wdm::wdm.

    x, y : 1D arrays of scores (numeric), same length.
    w    : 1D array of nonnegative per-observation weights.
    chunk: row-chunk size for memory-bounded outer-product accumulation.

    Returns tau_b in [-1, 1] or NaN if fewer than 3 usable observations.

    Suitable for n <= ~5000 cells. For larger n, use wdm_tau_b_bit() below.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    m = ~(np.isnan(x) | np.isnan(y)) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    n = len(x)
    if n < 3:
        return float("nan")

    # Full-sum accumulators (over all i != j pairs, then divide by 2)
    num = 0.0    # sum_{i,j} w_i w_j * sign(x_i-x_j) * sign(y_i-y_j)
    T_x = 0.0    # sum_{i,j} w_i w_j * [sign(x_i-x_j) != 0]
    T_y = 0.0    # sum_{i,j} w_i w_j * [sign(y_i-y_j) != 0]

    for i0 in range(0, n, chunk):
        i1 = min(i0 + chunk, n)
        xi = x[i0:i1, None]
        yi = y[i0:i1, None]
        wi = w[i0:i1, None]
        dx = np.sign(xi - x[None, :]).astype(np.float64)  # (c, n)
        dy = np.sign(yi - y[None, :]).astype(np.float64)  # (c, n)
        ww = wi * w[None, :]                              # (c, n)
        num += float((ww * dx * dy).sum())
        T_x += float((ww * (dx != 0)).sum())
        T_y += float((ww * (dy != 0)).sum())

    # These sums double-count (i,j) and (j,i); divide by 2 for i<j
    num /= 2.0
    T_x /= 2.0
    T_y /= 2.0

    denom = np.sqrt(T_x * T_y)
    if denom <= 0:
        return float("nan")
    return float(num / denom)


class _BIT:
    """Fenwick tree over floats for weighted prefix sums."""
    __slots__ = ("n", "t")

    def __init__(self, n):
        self.n = n
        self.t = np.zeros(n + 1, dtype=np.float64)

    def update(self, i, v):
        i += 1
        while i <= self.n:
            self.t[i] += v
            i += i & (-i)

    def prefix(self, i):
        """Sum of positions [0..i]."""
        i += 1
        s = 0.0
        while i > 0:
            s += self.t[i]
            i -= i & (-i)
        return s

    def total(self):
        return self.prefix(self.n - 1)


def wdm_tau_b_bit(x, y, w) -> float:
    """
    Weighted Kendall tau-b via Fenwick tree — O(n log n) — matching R wdm::wdm.

    Uses the sort-by-x, count-concordant/discordant/ties-in-y-via-BIT algorithm,
    generalized with per-observation weights on the accumulators.

    Correctness: identical to wdm_tau_b() for the same inputs (validated in
    __main__ self-test).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    m = ~(np.isnan(x) | np.isnan(y)) & (w > 0)
    x, y, w = x[m], y[m], w[m]
    n = len(x)
    if n < 3:
        return float("nan")

    # Sort by x (stable sort so equal x values retain order)
    ord_x = np.argsort(x, kind="stable")
    x = x[ord_x]; y = y[ord_x]; w = w[ord_x]

    # Discretize y to ranks (with ties)
    uniq_y, y_idx = np.unique(y, return_inverse=True)
    K = len(uniq_y)

    # BITs on y-ranks:
    #   bit_w  : sum of weights of already-seen cells at y-rank
    #   bit_wy : (not needed for tau-b — but we do need x-tie handling)
    bit_w = _BIT(K)
    bit_w_by_y_tie_group = None  # not needed; y ties handled inline

    # Aggregate: iterate cells in x-ascending order. For each incoming cell i,
    # partition already-seen cells j (j has x_j <= x_i) into:
    #   - x_j < x_i (strict): call these "prior" pairs
    #   - x_j == x_i (tie on x): call these "x-tie" pairs; contribute to T_x
    # For prior pairs, further partition by y_j vs y_i:
    #   - y_j < y_i : concordant
    #   - y_j > y_i : discordant
    #   - y_j == y_i: y-tie (contribute to T_y, not to num)
    #
    # We process x-groups in ascending-x order.

    Cw = 0.0   # concordant weight
    Dw = 0.0   # discordant weight
    Txy = 0.0  # x-tied pairs
    Ty_only = 0.0  # y-tied pairs (where x is untied)

    # Group indices by equal x values (stable-sorted)
    i = 0
    prior_total_w = 0.0
    while i < n:
        j = i
        # find end of x-tie group (all cells with x == x[i])
        while j < n and x[j] == x[i]:
            j += 1
        # Pairs *within* the current x-tie group: both x-tied.
        # For each pair (a, b) in group with a != b: contributes to T_x (untied? no — tied on x)
        # Weighted count of such pairs = 0.5 * ( (sum w_grp)^2 - sum w_grp^2 )
        grp_w = w[i:j]
        gW = grp_w.sum()
        gW2 = (grp_w * grp_w).sum()
        # weighted pairs within group where x_i==x_j:
        # partition by y:
        for a in range(i, j):
            ya = y_idx[a]; wa = w[a]
            # look at other cells in group b != a
            # split by y_b vs y_a:
            # but this is O(group^2). For small groups this is fine.
            pass
        # Simpler: process the whole group in one shot using BIT-like sub-accumulator
        # For x-tie pairs, we don't care about C/D distinction; only whether y is tied too.
        # y-tie within group:
        #   pairs where y_a == y_b (both in group): contribute to (T_x AND T_y)? Actually
        #   pairs tied in both x and y are counted in both T_x and T_y (in tau-b denominator).
        # So we just need total-pair-weight, plus y-tie-only-in-group.
        # Total unordered pairs in group = 0.5 * (gW^2 - gW2)
        # Of those, y-tie pairs (where y_a == y_b in group):
        # aggregate by y within group:
        y_in_grp = y_idx[i:j]
        # sort by y
        ord_y = np.argsort(y_in_grp, kind="stable")
        yg = y_in_grp[ord_y]
        wg = grp_w[ord_y]
        yt = 0.0
        k0 = 0
        while k0 < len(yg):
            k1 = k0
            while k1 < len(yg) and yg[k1] == yg[k0]:
                k1 += 1
            sub = wg[k0:k1].sum()
            sub2 = (wg[k0:k1] * wg[k0:k1]).sum()
            yt += 0.5 * (sub * sub - sub2)  # y-tie pairs within group
            k0 = k1
        # Contribution:
        # Every within-group pair has x tied -> T_x += all
        Txy_group_pairs = 0.5 * (gW * gW - gW2)
        Txy += Txy_group_pairs
        # Every y-tie pair within group also contributes to T_y
        Ty_only += yt

        # Now count pairs between group and *prior* cells (i.e., cells k < i with x_k < x_i).
        # BIT holds prior cells' weights indexed by y-rank.
        prior_W = prior_total_w
        # For each cell 'a' in group:
        #   concordant_a = sum_{k prior} w_k * [y_k < y_a] * w_a
        #   discordant_a = sum_{k prior} w_k * [y_k > y_a] * w_a
        #   y_tie_a      = sum_{k prior} w_k * [y_k == y_a] * w_a
        #   sum of concord + discord + y_tie = prior_W * w_a
        # To get y_tie_a, we need weight of prior cells at exactly y_idx[a]:
        # BIT does prefix sums; we can query prefix(y-1) and prefix(y), diff = at y.
        for a in range(i, j):
            ya = y_idx[a]; wa = w[a]
            below = bit_w.prefix(ya - 1) if ya > 0 else 0.0
            at_and_below = bit_w.prefix(ya)
            at = at_and_below - below
            above = prior_W - at_and_below
            Cw += wa * below
            Dw += wa * above
            Ty_only += wa * at
        # Now add cells of this group to the BIT (they become prior for next groups)
        for a in range(i, j):
            bit_w.update(y_idx[a], w[a])
        prior_total_w += gW
        i = j

    # tau-b numerator: Cw - Dw
    # tau-b denominator: sqrt( (Cw + Dw + Ty_only) * (Cw + Dw + Txy) )
    # NOTE: our T_x/T_y here are pair-weights not counts of ties;
    #   * pairs where x is tied but y untied      = Txy_x_only
    #   * pairs where y is tied but x untied      = Ty_only
    #   * pairs tied in both                      = Txy_both (counted inside Txy_group_pairs)
    # tau-b (Kendall):
    #   n_x = # pairs untied in x    = Cw + Dw + Ty_only
    #   n_y = # pairs untied in y    = Cw + Dw + Txy - both  (but Txy already includes both)
    # Actually the standard tau-b definition:
    #   n_0 = total pairs
    #   n_1 = sum over groups of x-ties: 0.5 g(g-1)  =  Txy_group_pairs_summed  (weighted here)
    #   n_2 = sum over groups of y-ties: 0.5 h(h-1)  =  Ty_only_summed + within-x-group y-ties
    #   tau_b = (C - D) / sqrt( (n_0 - n_1)(n_0 - n_2) )
    # In our accounting:
    #   n_1 = Txy (all pairs tied on x, including double-tied)
    #   n_2 = (Ty_only accumulated) which already includes within-x-group y-ties
    n_0 = 0.5 * (w.sum() ** 2 - (w * w).sum())
    n_1 = Txy
    n_2 = Ty_only
    denom = np.sqrt((n_0 - n_1) * (n_0 - n_2))
    if denom <= 0:
        return float("nan")
    return float((Cw - Dw) / denom)


def kang_absolute_weights(std_phenotype, potency_level):
    """
    Kang S5 'Absolute order' per-cell weight (broad/granular potency ordinal target).

    'First balanced by number of cells per dataset assigned to a given phenotype,
     then balanced across phenotypes within a given broad/granular potency level'

    weight_i = 1 / ( n_cells_in_std_phenotype_i * n_std_phenotypes_at_potency_level_i )

    Note: Kang's exact spec balances by ORIGINAL author-supplied phenotype (S3 col E) first,
    then by STANDARDIZED phenotype (S3 col D). Because S11 gives only standardized phenotype
    per cell, this implementation approximates by using standardized phenotype for both nesting
    levels. Empirical validation against Kang S12 shows <0.02 absolute error on the tested
    dataset (Retinal neurons 10x). Report this approximation as such.
    """
    import pandas as pd
    ph = pd.Series(std_phenotype)
    lv = pd.Series(potency_level)
    ph_count = ph.value_counts().to_dict()
    lv_ph = ph.groupby(lv).nunique().to_dict()
    n = len(ph)
    w = np.zeros(n)
    for i in range(n):
        p = ph.iloc[i]
        l = lv.iloc[i]
        if pd.isna(l) or pd.isna(p):
            w[i] = 0.0
        else:
            w[i] = 1.0 / (ph_count[p] * lv_ph[l])
    return w


if __name__ == "__main__":
    from scipy.stats import kendalltau

    # Self-test 1: uniform weights + continuous data (no ties)
    rng = np.random.default_rng(0)
    n = 200
    x = rng.standard_normal(n)
    y = x + 0.5 * rng.standard_normal(n)
    w = np.ones(n)
    t_wdm = wdm_tau_b(x, y, w)
    t_bit = wdm_tau_b_bit(x, y, w)
    t_ref = kendalltau(x, y).statistic
    print(f"Self-test 1: uniform weights, no ties (n={n})")
    print(f"  wdm_tau_b     = {t_wdm:+.6f}")
    print(f"  wdm_tau_b_bit = {t_bit:+.6f}")
    print(f"  kendalltau    = {t_ref:+.6f}")
    assert abs(t_wdm - t_ref) < 1e-9
    assert abs(t_bit - t_ref) < 1e-9
    print("  OK — dense & BIT versions both equal kendalltau.\n")

    # Self-test 2: heavy y-ties (ordinal-target case, mimics broad potency GT)
    n = 500
    x = rng.standard_normal(n)
    y = rng.integers(0, 4, n).astype(float)   # 4 categories -> heavy ties
    w = rng.uniform(0.1, 2.0, n)
    t_wdm = wdm_tau_b(x, y, w)
    t_bit = wdm_tau_b_bit(x, y, w)
    print(f"Self-test 2: random weights + heavy y-ties (n={n})")
    print(f"  wdm_tau_b     = {t_wdm:+.6f}")
    print(f"  wdm_tau_b_bit = {t_bit:+.6f}")
    print(f"  |diff|        = {abs(t_wdm - t_bit):.2e}")
    assert abs(t_wdm - t_bit) < 1e-9, "dense and BIT must agree"
    print("  OK — dense & BIT agree under weights + y-ties.\n")

    # Self-test 3: heavy x-ties AND heavy y-ties
    n = 400
    x = rng.integers(0, 6, n).astype(float)   # 6 x-categories
    y = rng.integers(0, 4, n).astype(float)   # 4 y-categories
    w = rng.uniform(0.1, 2.0, n)
    t_wdm = wdm_tau_b(x, y, w)
    t_bit = wdm_tau_b_bit(x, y, w)
    print(f"Self-test 3: random weights + heavy x&y-ties (n={n})")
    print(f"  wdm_tau_b     = {t_wdm:+.6f}")
    print(f"  wdm_tau_b_bit = {t_bit:+.6f}")
    print(f"  |diff|        = {abs(t_wdm - t_bit):.2e}")
    assert abs(t_wdm - t_bit) < 1e-9, "dense and BIT must agree with both-ties"
    print("  OK — dense & BIT agree under weights + both-side ties.\n")

    print("All wdm_tau self-tests passed.")
