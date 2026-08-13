"""
own_baseline.py — the own-baseline diagnostic (two flavours).

Central claim
-------------
When a signal (an error model; a "potency" score) appears to transfer or to
order cells across contexts, that positive result is *expected* whenever the
signal factors through a low-order statistic of the data that two contexts
share by construction. Cross-context agreement is then NOT, on its own,
evidence of conserved structure.

Two diagnostics, same logic
---------------------------
(A) Error-model version (§4.3, §5 of the paper — the original Test A):
        Delta = AUROC_transfer(A -> B) - AUROC_own-entropy(B)
    Rationale: local label entropy estimates the Bayes error; the
    entropy->Bayes-error map is domain-invariant by Fano's inequality.
    See test_A() below.

(B) Potency-score version (§5 extension; §8.5 worked example — this file's
    new API):
        Delta = tau_wt(score, GT) - tau_wt(nnz, GT)
    where GT is a per-cell ground-truth potency ordinal and `nnz` is the
    raw per-cell gene count (nnz_g = #{g : x_{c,g} > 0}). Rationale
    (paper §5, algebra section):
        - CytoTRACE v1 orders cells by nnz => tau(CT v1, GT) ~= tau(nnz, GT)
          by construction (nnz IS the primitive).
        - SCENT SR reduces to mean-field corr(x, degree) (Teschendorff &
          Enver 2017 Nat Commun 8:15599); its rank is a low-order statistic
          driven by broad-of-expression.
        - CCAT = Pearson(x, degree) — an explicit low-order statistic
          (Teschendorff et al. 2021 Bioinformatics 37(11):1528).
    All three are conserved between any two datasets whose broad-of-
    expression profile is comparable — by construction, not biology.
    See run_own_baseline() below.

Decision rule (potency-score flavour)
-------------------------------------
    Delta <= 0.05        -> REDUCES-TO-PRIMITIVE
                            "no gain over the primitive baseline: the score is
                             not distinguishable from raw gene counts on this
                             dataset — a scaffold-anchored surrogate suffices
                             to explain what you see."
    0.05 < Delta <= 0.10 -> INCONCLUSIVE
                            "underpowered / borderline — collect more cells or
                             check baseline sensitivity across kernels."
    Delta > 0.10         -> ADDS-BEYOND-PRIMITIVE
                            "the score captures structure beyond raw gene
                             counts; proceed but check baseline sensitivity."

The verdict names say which way the evidence points. They are not a quality
judgement on the score and there is no pass/fail: ADDS-BEYOND-PRIMITIVE is the
outcome for most of the scores this repo audited.

Cite:
  - Fano (1961) via Cover & Thomas, Elements of Information Theory 2nd ed.
    (2006), Theorem 2.10.1.
  - Cover & Hart (1967) IEEE TIT 13(1):21-27 — kNN entropy consistency.
  - Kang et al. (2025) Nat Methods 22(11):2258-2263, doi
    10.1038/s41592-025-02857-2 — CytoTRACE 2 benchmark; source of Table S13
    Test-cohort median-relative-order CT1 = 0.512 vs SR = 0.194,
    CCAT = 0.271, StemID = 0.230 (the field's own numbers showing raw
    gene-counts already dominate the scaffold-anchored surrogates).
  - Banerji et al. (2015) PLoS Comput Biol, PMC4368751 — SR reported as
    conserved across breast and lung cancer.
  - See paper §5 (05-writing/paper1-FULL-DRAFT-v1.md) for the algebraic
    identities; the Session-6 dossier (04-experiments/2026-07-17-hod5-
    atlas-scale-dossier.md §S6) for the II.7 atlas numbers this tool
    reuses.

Practitioner-facing 5-line usage
--------------------------------
    from own_baseline import run_own_baseline
    from potency_metrics import ccat, build_ppi_adjacency, load_string_ppi

    edges = load_string_ppi("9606.protein.links.v12.0.txt.gz",
                            "9606.protein.info.v12.0.txt.gz",
                            score_threshold=700)
    result = run_own_baseline(adata, score="ccat", gt_ordinal="broad_pot",
                              ppi_edges=edges)
    print(result["takeaway"])
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Callable, Optional, Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

# scipy is a hard dep of the paper's stack; weightedtau is the prereg-locked
# kernel (atlas_run.py). We import inside functions to keep the error-model
# API (test_A) importable without scipy in edge cases.

__all__ = [
    # error-model version (unchanged)
    "own_entropy_baseline",
    "transfer_auc",
    "test_A",
    # potency-score version (new)
    "nnz_per_cell",
    "score_vs_gt_tau",
    "run_own_baseline",
]


def _auc_cv(X, y, cv: int = 5) -> float:
    """Cross-validated balanced-logistic AUROC (within-domain)."""
    X = np.asarray(X, float)
    y = np.asarray(y, int)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if len(np.unique(y)) < 2 or len(y) < 2 * cv:
        return float("nan")
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, proba))


def own_entropy_baseline(entropy_feat, y_err, cv: int = 5) -> float:
    """
    Within-target AUROC using the local label-entropy feature alone.

    This is the Bayes-error proxy: it is what the target already 'knows' about
    where its own classifier errs, from the same universal quantity that makes
    cross-domain transfer look successful.

    Parameters
    ----------
    entropy_feat : array (n_samples,)
        Per-sample local label entropy (e.g. kNN label entropy) in the target.
    y_err : array (n_samples,)
        Binary error labels in the target (1 = misclassified out-of-fold).
    """
    return _auc_cv(np.asarray(entropy_feat).reshape(-1, 1), y_err, cv=cv)


def transfer_auc(Xs, ys_err, Xt, yt_err) -> float:
    """
    AUROC of a source-trained error-model applied to the target domain.

    No cross-validation is needed: source and target are disjoint domains, so
    the model is trained on all of the source and evaluated on all of the
    target. Features must be gene-/coordinate-invariant so that the source
    model is applicable in the target without matching.

    Parameters
    ----------
    Xs, Xt : array (n_samples, n_features)
        Source / target error-model features (same feature columns).
    ys_err, yt_err : array (n_samples,)
        Binary error labels in source / target.
    """
    Xs = np.asarray(Xs, float)
    Xt = np.asarray(Xt, float)
    if Xs.ndim == 1:
        Xs, Xt = Xs.reshape(-1, 1), Xt.reshape(-1, 1)
    clf = LogisticRegression(max_iter=500, class_weight="balanced").fit(Xs, np.asarray(ys_err, int))
    return float(roc_auc_score(np.asarray(yt_err, int), clf.predict_proba(Xt)[:, 1]))


def test_A(Xs, ys_err, Xt, yt_err, entropy_feat_t, eps: float = 0.03, cv: int = 5) -> dict:
    """
    Run the diagnostic.

    Returns
    -------
    dict with:
        transfer_auc     : AUROC of the transferred (source-trained) error-model on the target
        own_entropy_auc  : within-target AUROC from local entropy alone
        delta            : transfer_auc - own_entropy_auc  (the informative quantity)
        verdict          : human-readable interpretation of delta vs eps

    Notes
    -----
    - `entropy_feat_t` should be the *most conservative* within-target baseline:
      local entropy alone. A within-target model on all structural features is at
      least as strong, so a transfer that fails to beat entropy-only also fails to
      beat the richer within-target model (an a-fortiori argument).
    - `eps` is the band for calling Delta ~ 0. Use a domain-appropriate value and
      fix it *before* running (pre-registration).
    """
    t = transfer_auc(Xs, ys_err, Xt, yt_err)
    o = own_entropy_baseline(entropy_feat_t, yt_err, cv=cv)
    d = t - o
    if np.isnan(d):
        verdict = "undefined (degenerate labels)"
    elif abs(d) <= eps:
        verdict = "TAUTOLOGICAL — no gain over own-entropy baseline (transfer is not evidence of conserved structure)"
    elif d > eps:
        verdict = "BEYOND-BAYES structure shared between domains"
    else:
        verdict = "transfer below own-baseline (worse than the target's own entropy)"
    return {
        "transfer_auc": None if np.isnan(t) else round(t, 4),
        "own_entropy_auc": None if np.isnan(o) else round(o, 4),
        "delta": None if np.isnan(d) else round(d, 4),
        "verdict": verdict,
    }


# ============================================================================
# POTENCY-SCORE VERSION (§8.5 practitioner tool)
# ============================================================================
#
# Decision-rule constants, pre-registered. The two thresholds 0.05 and 0.10 are
# unchanged from the atlas-run pre-registration, so verdicts here stay directly
# comparable to the atlas run's per_dataset_results.json. Only the NAMES of the
# bands changed (they used to be "PASS"/"FAIL", which read backwards).
REDUCES_DELTA = 0.05
ADDS_DELTA = 0.10

# Verdict labels. REDUCES-TO-PRIMITIVE and ADDS-BEYOND-PRIMITIVE describe the
# direction of the evidence; neither is a pass or a failure.
VERDICT_REDUCES = "REDUCES-TO-PRIMITIVE"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"
VERDICT_ADDS = "ADDS-BEYOND-PRIMITIVE"
VERDICT_UNDEFINED = "UNDEFINED"

# Algebraic identity messages (§5 of the paper).
# One line each, printed under PASS to explain WHY the score collapsed.
_ALGEBRA_HINT = {
    "cytotrace_v1": (
        "CytoTRACE v1 orders cells by gene counts (nnz); log(nnz) is the "
        "order-0 Rényi (Hartley) entropy H_0 of the expression vector. So "
        "tau(CT v1, GT) equals tau(nnz, GT) by construction — the "
        "scaffold is the gene-count ranking itself."
    ),
    "scent_sr": (
        "Signalling entropy rate reduces in the mean-field / degenerate "
        "limit to Pearson corr(x, degree) (Teschendorff & Enver 2017; "
        "see also CCAT). If it REDUCES-TO-PRIMITIVE here, the transferred "
        "signal is the shared PPI scaffold, not conserved potency biology. "
        "Note this repo's own audit found the opposite on a 12-stage staged "
        "ordinal: SR retains conditional skill beyond the degree correlation."
    ),
    "ccat": (
        "CCAT = Pearson(x, degree_vector) is an explicit low-order "
        "statistic of x anchored to a fixed scaffold (the PPI degree "
        "vector, shared between any two human single-cell datasets). "
        "If it REDUCES-TO-PRIMITIVE here, the 'conservation' is by "
        "construction. This is the one reduction the audit confirmed."
    ),
    "custom": (
        "Your custom score orders cells no better than raw nnz on this "
        "dataset. This is consistent with the score factoring through a "
        "low-order statistic shared with nnz (e.g. a monotone function of "
        "gene counts or of a fixed scaffold). See paper §5 for the class."
    ),
}


def nnz_per_cell(X_or_adata) -> np.ndarray:
    """
    Raw gene-count baseline: nnz_c = #{g : x_{c,g} > 0}.

    Accepts a scipy.sparse matrix, a dense numpy array (cells x genes), or
    an AnnData object (uses .X). This is the primitive against which every
    scaffold-anchored score is compared, per paper §5.

    NB: gene counts, not molecule counts. Kang 2025 confirms across n=23
    atlas rows that this raw primitive already dominates SCENT (SR),
    SCENT (CCAT), and StemID on the Test-cohort median relative order
    (S13: CT1=0.512 vs SR=0.194, CCAT=0.271, StemID=0.230).
    """
    try:
        import scipy.sparse as sp
        X = X_or_adata.X if hasattr(X_or_adata, "X") else X_or_adata
        if sp.issparse(X):
            # sparse: nnz per row (getnnz on axis=1) is O(nnz) and exact.
            nnz = np.asarray(X.getnnz(axis=1)).ravel().astype(np.float64)
        else:
            Xd = np.asarray(X)
            nnz = (Xd > 0).sum(axis=1).astype(np.float64)
    except ImportError:
        Xd = np.asarray(X_or_adata.X if hasattr(X_or_adata, "X") else X_or_adata)
        nnz = (Xd > 0).sum(axis=1).astype(np.float64)
    return nnz


def _resolve_gt(gt_ordinal, adata) -> np.ndarray:
    """Accept either an array of ordinals or an obs-column name."""
    if isinstance(gt_ordinal, str):
        if adata is None or not hasattr(adata, "obs"):
            raise ValueError(
                f"gt_ordinal={gt_ordinal!r} is a column name but no AnnData "
                f"was supplied."
            )
        if gt_ordinal not in adata.obs.columns:
            raise KeyError(
                f"gt_ordinal={gt_ordinal!r} not in adata.obs.columns"
            )
        return np.asarray(adata.obs[gt_ordinal].values, dtype=np.float64)
    return np.asarray(gt_ordinal, dtype=np.float64)


def score_vs_gt_tau(
    score: np.ndarray,
    gt: np.ndarray,
    kernel: str = "scipy",
    std_phenotype: Optional[np.ndarray] = None,
) -> float:
    """
    Weighted Kendall tau of a per-cell score against a ground-truth ordinal.

    Parameters
    ----------
    score : (n_cells,) — per-cell score (higher = more potent, by convention).
    gt    : (n_cells,) — per-cell potency ordinal (higher = more potent).
    kernel:
        "scipy" (default) — scipy.stats.weightedtau (rank-based hyperbolic
                            weigher). This is the prereg-locked kernel in
                            atlas_run.py.
        "kang"            — Kang class-balanced weighted Kendall tau-b
                            (matches R wdm::wdm(x, y, method="kendall",
                            weights=w)); requires std_phenotype so that
                            weights nest phenotype within potency level.
                            See atlas_run/wdm_tau.py.

    NB: this function does NOT negate the ordinal. Pass gt so that higher
    values mean MORE potent (which matches "higher score = more potent"
    for CT1/SR/CCAT). If your ordinal follows the Kang 1..6 convention
    (Totipotent=1, Differentiated=6), pass -ordinal so higher = more potent.
    """
    score = np.asarray(score, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    mask = ~(np.isnan(score) | np.isnan(gt))
    if mask.sum() < 3:
        return float("nan")
    s, g = score[mask], gt[mask]

    if kernel == "scipy":
        from scipy.stats import weightedtau
        r = weightedtau(s, g)
        return float(getattr(r, "statistic", None) or r.correlation)
    if kernel == "kang":
        if std_phenotype is None:
            raise ValueError(
                "kernel='kang' requires std_phenotype (per-cell standardized "
                "phenotype label) so weights can nest phenotype within "
                "potency level (Kang S5)."
            )
        # Local import so scipy is enough for the default path.
        _here = Path(__file__).parent
        sys.path.insert(0, str(_here / "atlas_run"))
        from wdm_tau import wdm_tau_b_bit, kang_absolute_weights  # type: ignore
        std_ph = np.asarray(std_phenotype)[mask]
        w = kang_absolute_weights(std_ph, g)
        return float(wdm_tau_b_bit(s, g, w))
    raise ValueError(f"unknown kernel {kernel!r} (want 'scipy' or 'kang')")


def _score_from_name(
    score_name: str,
    adata,
    ppi_edges,
) -> np.ndarray:
    """
    Compute a named potency score using potency_metrics.py.

    Supported names: 'cytotrace_v1' (returns gene_counts, i.e. the CT v1
    primitive per §5), 'scent_sr', 'ccat'.

    For 'scent_sr' and 'ccat' the caller MUST supply ppi_edges (edge list of
    gene symbols, e.g. from potency_metrics.load_string_ppi). This is a
    hard requirement: the "scaffold" is the PPI graph and randomizing it is
    the null the paper §5 discusses.
    """
    _here = Path(__file__).parent
    sys.path.insert(0, str(_here))
    import potency_metrics as pm  # type: ignore

    if score_name == "cytotrace_v1":
        # By paper §5: CT v1 orders cells by gene counts. Rank-equivalent
        # to nnz => tau(CT v1, GT) == tau(nnz, GT) exactly if we use nnz.
        # Returned as the raw primitive to make the identity visible.
        return pm.cytotrace_proxy(adata)["gene_counts"]

    if score_name == "scent_sr":
        if ppi_edges is None:
            raise ValueError(
                "score='scent_sr' requires ppi_edges (STRING edge list "
                "of gene symbols). See potency_metrics.load_string_ppi."
            )
        A, _genes, col_idx, _degree = pm.build_ppi_adjacency(
            ppi_edges, adata.var_names
        )
        return pm.scent_sr(adata, A, col_idx)

    if score_name == "ccat":
        if ppi_edges is None:
            raise ValueError(
                "score='ccat' requires ppi_edges (STRING edge list of "
                "gene symbols). See potency_metrics.load_string_ppi."
            )
        _A, _genes, col_idx, degree = pm.build_ppi_adjacency(
            ppi_edges, adata.var_names
        )
        return pm.ccat(adata, col_idx, degree)

    raise ValueError(
        f"unknown score name {score_name!r} (want 'scent_sr', 'ccat', "
        f"'cytotrace_v1', or a callable)"
    )


def run_own_baseline(
    adata,
    score: Union[str, Callable],
    gt_ordinal,
    kernel: str = "scipy",
    std_phenotype: Optional[Union[str, np.ndarray]] = None,
    ppi_edges=None,
    higher_gt_more_potent: bool = True,
    verbose: bool = True,
) -> dict:
    """
    Practitioner-facing own-baseline diagnostic for single-cell potency
    scores. Answers: "Is my score distinguishable from raw gene counts at
    ordering cells by potency on THIS dataset?"

    Parameters
    ----------
    adata : AnnData with .X = expression matrix (raw counts strongly
        preferred; nnz is well-defined on any nonneg matrix but its
        biological interpretation depends on raw counts).
    score : one of {"scent_sr", "ccat", "cytotrace_v1"} OR a callable
        `f(adata) -> np.ndarray[n_cells]` returning your score per cell.
    gt_ordinal : per-cell potency ordinal — array of length n_cells, or
        the name of an adata.obs column. Convention:
            higher_gt_more_potent=True (default) => higher = more potent
            higher_gt_more_potent=False          => e.g. Kang 1..6 (Toti=1)
        The tool internally negates if needed so tau is signed with the
        biological direction "more potent -> higher rank".
    kernel : "scipy" (default, prereg-locked in atlas_run) or "kang"
        (Kang class-balanced weights matching R wdm::wdm; needs
        std_phenotype).
    std_phenotype : required for kernel="kang". Array of per-cell
        standardized phenotype labels, or an adata.obs column name.
    ppi_edges : required for score in {"scent_sr", "ccat"}. Edge list of
        gene symbols; see potency_metrics.load_string_ppi.
    higher_gt_more_potent : if True, larger gt_ordinal is more potent
        (natural convention). If False (e.g. Kang 1..6), tool negates
        internally.
    verbose : print the takeaway sentence.

    Returns
    -------
    dict with:
        tau_score          : weighted tau of the score vs GT
        tau_nnz            : weighted tau of raw nnz vs GT (the baseline)
        delta              : tau_score - tau_nnz  (the informative quantity)
        verdict            : "REDUCES-TO-PRIMITIVE" / "INCONCLUSIVE" /
                             "ADDS-BEYOND-PRIMITIVE" / "UNDEFINED"
        algebra_hint       : if REDUCES-TO-PRIMITIVE, the identity explaining why
        takeaway           : one-sentence actionable message
        n_cells            : number of cells used (post-NaN mask)
        score_name         : label for the score

    Verdict decision rule (prereg-locked thresholds)
    ------------------------------------------------
        delta <= 0.05        -> REDUCES-TO-PRIMITIVE (not distinguishable
                                from nnz on this dataset)
        0.05 < delta <= 0.10 -> INCONCLUSIVE (underpowered)
        delta > 0.10         -> ADDS-BEYOND-PRIMITIVE (structure beyond nnz)

    Neither end is a pass or a failure; the label states which way the
    evidence points.
    """
    # 1) Resolve score.
    if isinstance(score, str):
        score_vec = _score_from_name(score, adata, ppi_edges)
        score_name = score
        hint_key = score if score in _ALGEBRA_HINT else "custom"
    elif callable(score):
        score_vec = np.asarray(score(adata), dtype=np.float64)
        score_name = getattr(score, "__name__", "custom_callable")
        hint_key = "custom"
    else:
        raise TypeError(
            "score must be a name string or a callable f(adata) -> array"
        )

    # 2) Resolve GT ordinal + orientation.
    gt = _resolve_gt(gt_ordinal, adata)
    if not higher_gt_more_potent:
        gt = -gt

    # 3) Resolve std_phenotype if Kang kernel.
    std_ph = None
    if kernel == "kang":
        if std_phenotype is None:
            raise ValueError(
                "kernel='kang' requires std_phenotype (array or "
                "adata.obs column name)."
            )
        if isinstance(std_phenotype, str):
            std_ph = np.asarray(adata.obs[std_phenotype].values)
        else:
            std_ph = np.asarray(std_phenotype)

    # 4) nnz baseline.
    nnz_vec = nnz_per_cell(adata)

    # 5) Compute both taus (same kernel, same cells, same weights).
    tau_score = score_vs_gt_tau(score_vec, gt, kernel=kernel,
                                std_phenotype=std_ph)
    tau_nnz = score_vs_gt_tau(nnz_vec, gt, kernel=kernel,
                              std_phenotype=std_ph)

    delta = tau_score - tau_nnz

    # 6) Verdict.
    if np.isnan(delta):
        verdict = VERDICT_UNDEFINED
        algebra = ""
        takeaway = (
            "Diagnostic undefined — check that ground-truth ordinal has "
            "at least 2 levels and that score/nnz have at least 3 non-NaN "
            "cells in common."
        )
    elif delta <= REDUCES_DELTA:
        verdict = VERDICT_REDUCES
        algebra = _ALGEBRA_HINT.get(hint_key, _ALGEBRA_HINT["custom"])
        takeaway = (
            f"Your {score_name} result is not distinguishable from raw "
            f"gene counts on this dataset (Delta = {delta:+.4f}, "
            f"REDUCES-TO-PRIMITIVE). Before publishing this as evidence of "
            f"conserved potency, re-run against a shuffle-scaffold or "
            f"per-cell nnz baseline. This marginal gap is the weaker of the "
            f"two tests in this repo; see conditional_skill_report() for the "
            f"residualized version."
        )
    elif delta <= ADDS_DELTA:
        verdict = VERDICT_INCONCLUSIVE
        algebra = ""
        takeaway = (
            f"Your {score_name} result is borderline vs raw gene counts "
            f"on this dataset (Delta = {delta:+.4f}, INCONCLUSIVE). "
            f"Collect more cells or check baseline sensitivity across "
            f"kernels (try kernel='kang' if you used 'scipy', and vice "
            f"versa) before reading this as conservation."
        )
    else:
        verdict = VERDICT_ADDS
        algebra = ""
        takeaway = (
            f"Your {score_name} adds structure beyond raw gene counts on "
            f"this dataset (Delta = {delta:+.4f}, ADDS-BEYOND-PRIMITIVE). "
            f"The potency claim survives the primitive; proceed but check "
            f"baseline sensitivity across kernels, and check the direction "
            f"of the baseline itself — a primitive that orders below chance "
            f"turns a beaten baseline into a sign correction."
        )

    result = {
        "score_name": score_name,
        "n_cells": int(np.sum(~(np.isnan(score_vec) | np.isnan(gt)))),
        "tau_score": None if np.isnan(tau_score) else round(tau_score, 4),
        "tau_nnz": None if np.isnan(tau_nnz) else round(tau_nnz, 4),
        "delta": None if np.isnan(delta) else round(delta, 4),
        "verdict": verdict,
        "kernel": kernel,
        "algebra_hint": algebra,
        "takeaway": takeaway,
    }

    if verbose:
        print(f"[own_baseline] score={score_name} kernel={kernel} "
              f"n={result['n_cells']}")
        print(f"[own_baseline] tau({score_name}, GT) = {tau_score:+.4f}")
        print(f"[own_baseline] tau(nnz, GT)        = {tau_nnz:+.4f}")
        print(f"[own_baseline] Delta               = {delta:+.4f}  "
              f"-> {verdict}")
        if algebra:
            print(f"[own_baseline] Why: {algebra}")
        print(f"[own_baseline] {takeaway}")

    return result


if __name__ == "__main__":
    # Minimal smoke test: synthetic REDUCES-by-construction (score == nnz)
    # and ADDS-by-construction (score correlates with the true ordinal
    # much better than nnz does). Full test suite in tests/.
    import anndata as ad

    rng = np.random.default_rng(0)
    n_cells, n_genes = 300, 400
    # ground-truth potency ordinal
    gt = rng.integers(0, 4, n_cells).astype(np.float64)
    # counts with a WEAK dependence on gt so nnz is a middling baseline —
    # leaves headroom for a genuinely stronger score to FAIL by construction.
    counts = np.zeros((n_cells, n_genes))
    for c in range(n_cells):
        active_p = 0.4 + 0.03 * gt[c]
        active = rng.random(n_genes) < active_p
        counts[c] = rng.poisson(1.5) * active
    adata = ad.AnnData(X=counts)
    adata.var_names = [f"g{i}" for i in range(n_genes)]
    adata.obs["gt"] = gt

    print("=== REDUCES case: custom score = nnz (Delta should be ~ 0) ===")
    _ = run_own_baseline(adata,
                        score=lambda ad_: nnz_per_cell(ad_),
                        gt_ordinal="gt",
                        higher_gt_more_potent=True)

    print("\n=== ADDS case: custom score = gt + tiny noise (Delta should be > 0.10) ===")
    _ = run_own_baseline(adata,
                        score=lambda ad_: adata.obs["gt"].values +
                                          0.05 * rng.standard_normal(n_cells),
                        gt_ordinal="gt",
                        higher_gt_more_potent=True)
