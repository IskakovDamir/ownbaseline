"""
own_baseline.py — does a score order cells better than its own primitive?

What this module does
---------------------
An unsupervised potency/stemness score is usually validated by correlating it
against a pseudotime, a marker panel, or a known hierarchy. A score that
restates sequencing depth passes that test, and so does a score carrying real
ordering information. The criterion does not separate them.

This module implements the cheap version of a criterion that does: compare the
score against the low-order statistic it is closest to — its *primitive* —
rather than against the gold standard alone.

(A) Error-model version (the original Test A):
        Delta = AUROC_transfer(A -> B) - AUROC_own-entropy(B)
    Rationale: local label entropy estimates the Bayes error; the
    entropy->Bayes-error map is domain-invariant by Fano's inequality.
    See test_A() below.

(B) Potency-score version — the marginal gap:
        Delta = tau_wt(score, GT) - tau_wt(nnz, GT)
    where GT is a per-cell ground-truth potency ordinal and `nnz` is the
    raw per-cell gene count (nnz_c = #{g : x_{c,g} > 0}).
    See run_own_baseline() below.

The primitives are not our choice. Each is the statistic the score's own
authors identify as what their score approximates:
    - CytoTRACE v1 orders cells by gene count (Gulati et al. 2020).
    - CCAT is *defined* as Pearson(x, PPI degree) (Teschendorff et al. 2021
      Bioinformatics 37(11):1528).
    - Teschendorff & Enver (2017) Nat Commun 8:15599 report that the same
      degree correlation approximates signalling entropy at R^2 = 0.96 — with
      an explicit caveat, in the same paragraph, that the approximation is
      empirical and not an equivalence.

Two warnings about this particular function
-------------------------------------------
1. The marginal gap is the WEAKER of the two tests in this repository. It does
   not residualize: it compares two marginal correlations. A score built from
   the primitive plus a little noise can show a positive Delta. The test the
   accompanying paper actually reports is conditional skill — the score rank
   residualized on the primitive rank, then scored against the ordinal under
   two kernels with a bootstrap interval. Use conditional_skill_report() in
   own_baseline.conditional_skill for that. Prefer it.

2. Check the direction of the baseline before reading the margin. In sorted
   haematopoietic progenitors the gene-count primitive orders cells BELOW
   chance (AUROC 0.378: HSC carry a median 916 detected genes, the GMP below
   them carry 1,404). A score that "beats" a reversed baseline has supplied a
   sign correction, not biology, and nothing in Delta shows this. In intestinal
   epithelium the same primitive reaches AUROC 0.798. The primitive did not
   change; only whether dissociation happened to put the deeper cells on top.

What the accompanying paper found
---------------------------------
This tool was built to demonstrate a strong hypothesis: that every unsupervised
potency score is a reparametrization of a fixed low-order statistic, so that
cross-context agreement between such scores is guaranteed by construction
rather than by biology. **That hypothesis was tested and refuted.**

On a 12-stage microscopy-staged zebrafish ordinal (GSE106474, 39,505 whole-
embryo cells), of the seven scores that could be placed on the ordinal, five
carry ordering skill beyond their own primitive, measured as Kendall tau_b
after rank residualization:

    CytoTRACE  0.653     SR (SCENT)  0.430     ORIGINS  0.402
    NCG        0.402 *   dpath       0.120
    CCAT       no measurable residual (rho = 0.998 with its primitive)
    SLICE      near zero (rho = 0.932 with its entropy primitive)
    * on a 3,000-cell subsample against its own GO-weighted connectome, so the
      agreement with the ORIGINS figure is a coincidence of rounding.

Residualizing on four low-order statistics at once — gene count, transcriptome-
connectome correlation, Shannon entropy, log library size — all four
substantive scores survive, with bootstrap intervals excluding zero under both
kernels: CytoTRACE 0.653 -> 0.288, SR 0.430 -> 0.183, ORIGINS 0.402 -> 0.156,
dpath 0.120 -> 0.134.

So the reduction is real for the identity and entropy constructions (CCAT is
its primitive by definition; StemID and cmEntropy equal a transcriptome
entropy by construction; SLICE tracks one it does not improve on) and FALSE
for the entropy-rate and diffusion constructions (signalling entropy,
CytoTRACE) at single-cell resolution on this ordinal.

Specifically: the claim that signalling entropy *reduces to* a mean-field
correlation with node degree, and that such scores are conserved across
datasets by construction rather than by biology, is NOT supported for SR. SR
sits at rho = 0.93 with the degree correlation and still retains conditional
skill. The claim stands for CCAT alone, where there is no gap between the
score and the statistic it would have to beat.

What the surviving residual consists of is open. A staged timecourse traces a
developmental manifold, and a score tracking that manifold keeps skill whether
or not it measures potency in any deeper sense.

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
    CCAT = 0.271, StemID = 0.230. Note these are measured against a proxy
    baseline, not against raw gene counts; see nnz_per_cell().
  - Banerji et al. (2015) PLoS Comput Biol, PMC4368751 — SR reported as
    conserved across breast and lung cancer.
  - Farrell et al. (2018) Science 360:eaar3131, GSE106474 — the 12-stage
    zebrafish ordinal the audit above is measured on.
  - Teschendorff & Enver (2017) Nat Commun 8:15599 — signalling entropy, and
    the R^2 = 0.96 degree-correlation approximation together with the caveat
    that it is empirical rather than an equivalence.
  - Gulati et al. (2020) Science 367:405 — CytoTRACE v1.
  - See README.md for the audit table and the reproduction entry point.

Practitioner-facing usage
-------------------------
    from own_baseline import run_own_baseline, load_string_ppi

    edges = load_string_ppi("9606.protein.links.v12.0.txt.gz",
                            "9606.protein.info.v12.0.txt.gz",
                            score_threshold=700)
    result = run_own_baseline(adata, score="ccat", gt_ordinal="broad_pot",
                              ppi_edges=edges)
    print(result["takeaway"])

For the residualized test the paper actually reports, use instead:

    from own_baseline import conditional_skill_report
    report = conditional_skill_report(score, ordinal, {"PCC(x,degree)": pcc})
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
# POTENCY-SCORE VERSION (the practitioner marginal-gap tool)
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

# Algebraic identity messages: why a score can collapse onto its primitive.
# One line each, printed under REDUCES-TO-PRIMITIVE to explain the collapse.
_ALGEBRA_HINT = {
    "cytotrace_v1": (
        "score='cytotrace_v1' returns the raw gene count itself, not "
        "CytoTRACE. Delta is therefore 0 by construction and carries no "
        "information: it is a self-test of the plumbing, not a measurement. "
        "The real CytoTRACE v1 adds an NNLS + Markov-diffusion smoothing "
        "step on top of the gene-count ranking, and in this repo's audit that "
        "full algorithm carries the LARGEST conditional skill beyond gene "
        "count of any score tested (tau_b 0.653). To measure it, run the "
        "R-faithful port in scores/cytotrace_full.py and pass the result as "
        "a callable."
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
        "gene counts or of a fixed scaffold). Confirm with the residualized "
        "test before concluding: see conditional_skill_report()."
    ),
}


def nnz_per_cell(X_or_adata) -> np.ndarray:
    """
    Raw gene-count baseline: nnz_c = #{g : x_{c,g} > 0}.

    Accepts a scipy.sparse matrix, a dense numpy array (cells x genes), or
    an AnnData object (uses .X). This is the primitive against which every
    scaffold-anchored score is compared.

    NB: gene counts, not molecule counts.

    A note on the atlas comparison, because an earlier version of this
    docstring got it wrong. Kang 2025 Table S13 reports Test-cohort median
    relative order CT1 = 0.512 against SR = 0.194, CCAT = 0.271 and
    StemID = 0.230. That is NOT a comparison against this primitive. Across
    those twenty-three datasets only published per-cell scores were
    available, so the baseline there is a proxy rather than raw gene counts,
    and the two comparisons are not the same quantity. The margin against
    raw gene counts at atlas scale is unmeasured.
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
        # NB: `getattr(r, "statistic", None) or r.correlation` would fall
        # through to the deprecated alias whenever the statistic is exactly
        # 0.0, which is falsy. Same value on scipy today, AttributeError the
        # day the alias is removed.
        return float(getattr(r, "statistic", np.nan))
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

    Supported names: 'scent_sr', 'ccat', and 'cytotrace_v1'.

    WARNING on 'cytotrace_v1': it returns the raw per-cell gene count, which
    is CytoTRACE v1's *primitive*, not CytoTRACE v1. Delta against nnz is then
    exactly 0 by construction. It is kept as a plumbing self-test and as a
    demonstration of the identity; it is not a measurement of CytoTRACE. The
    full algorithm (MVG -> GCS -> NNLS -> Markov diffusion) is in
    scores/cytotrace_full.py -- pass its output as a callable instead.

    For 'scent_sr' and 'ccat' the caller MUST supply ppi_edges (edge list of
    gene symbols, e.g. from potency_metrics.load_string_ppi). This is a
    hard requirement: the "scaffold" is the PPI graph and randomizing it is
    the scaffold-randomization null (see potency_metrics.scaffold_null).
    """
    _here = Path(__file__).parent
    sys.path.insert(0, str(_here))
    import potency_metrics as pm  # type: ignore

    if score_name == "cytotrace_v1":
        # Deliberately the PRIMITIVE, not the score: gene counts. Makes the
        # identity tau(nnz, GT) == tau(nnz, GT) visible as an exact 0. See the
        # WARNING above -- this is not the CytoTRACE algorithm.
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
