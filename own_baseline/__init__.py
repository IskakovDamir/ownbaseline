"""
own-baseline — does a single-cell potency score add anything beyond the
low-order statistic it is closest to?

Two tests, in increasing strength:

    run_own_baseline(adata, score, gt_ordinal)
        The marginal gap. tau(score, ordinal) - tau(gene counts, ordinal).
        One line, no residualization. Cheap, and weaker than it looks: a
        score that is its primitive plus noise can show a positive gap.

    conditional_skill_report(score, ordinal, primitives)
        The test the paper reports. Rank-residualize the score on its own
        primitive(s), then score the residual against the ordinal under two
        weighting kernels with a bootstrap interval. Also returns the
        marginal gap and a direction check, so the weaker number is visible
        next to the stronger one.

Both are documented in own_baseline/own_baseline.py and
own_baseline/conditional_skill.py.

Layout:
    own_baseline.py       marginal gap + the Test-A error-model flavour
    conditional_skill.py  the track-2 conditional-skill estimator
    potency_metrics.py    the primitives, plus SR / CCAT / CytoTRACE reimpls
    atlas_run/wdm_tau.py  Kang wdm weighted-Kendall kernel (== R wdm::wdm)
    paths.py              where this repo reads and writes
    synthetic.py          synthetic generators for the self-tests
"""
__version__ = "0.2.0"
# One place. The anonymised review copy blanks this line and nothing else.
__author__ = "Damir Iskakov"

from .own_baseline import (
    ADDS_DELTA,
    REDUCES_DELTA,
    VERDICT_ADDS,
    VERDICT_INCONCLUSIVE,
    VERDICT_REDUCES,
    VERDICT_UNDEFINED,
    nnz_per_cell,
    own_entropy_baseline,
    run_own_baseline,
    score_vs_gt_tau,
    test_A,
    transfer_auc,
)
from .conditional_skill import (
    align,
    conditional_skill_report,
    kang_taub,
    rank_resid_multi,
    scipy_wtau,
)
from .potency_metrics import (
    build_ppi_adjacency,
    ccat,
    compute_all_metrics,
    cytotrace_proxy,
    load_string_ppi,
    scaffold_null,
    scent_sr,
    stemid_entropy,
)
from .paths import data_root, repo_root, scratch_root

__all__ = [
    "__version__",
    "__author__",
    # marginal gap
    "run_own_baseline",
    "nnz_per_cell",
    "score_vs_gt_tau",
    "REDUCES_DELTA",
    "ADDS_DELTA",
    "VERDICT_REDUCES",
    "VERDICT_INCONCLUSIVE",
    "VERDICT_ADDS",
    "VERDICT_UNDEFINED",
    # conditional skill
    "conditional_skill_report",
    "rank_resid_multi",
    "align",
    "scipy_wtau",
    "kang_taub",
    # error-model flavour
    "test_A",
    "transfer_auc",
    "own_entropy_baseline",
    # primitives and scores
    "stemid_entropy",
    "cytotrace_proxy",
    "scent_sr",
    "ccat",
    "load_string_ppi",
    "build_ppi_adjacency",
    "compute_all_metrics",
    "scaffold_null",
    # paths
    "repo_root",
    "data_root",
    "scratch_root",
]
