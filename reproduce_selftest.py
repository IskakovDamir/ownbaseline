"""
Self-test for reproduce.py: run every computational stage on a small synthetic
atlas, so the whole chain can be verified in about a minute without
downloading 135 MB or waiting on a 39,505-cell run.

It exercises the same functions the real chain calls -- the CytoTRACE port,
SCENT SR and CCAT, the primitives, the conditional-skill estimator, the table
writer and the figure writer. It does NOT reproduce any paper number: the data
is synthetic. It answers "does the pipeline run", not "does it agree".

    python3 reproduce.py --self-test
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parent


def _synthetic_atlas(n_cells=400, n_genes=600, n_stages=12, seed=7):
    """A toy atlas with a real stage ordinal and a genuine potency gradient."""
    import numpy as np
    rng = np.random.default_rng(seed)
    stage = np.repeat(np.arange(1, n_stages + 1), n_cells // n_stages)
    stage = np.concatenate([stage, np.full(n_cells - len(stage), n_stages)])
    t = 1.0 - (stage - 1) / (n_stages - 1)          # 1 = most potent
    deg = rng.zipf(2.0, n_genes).clip(1, 40).astype(float)
    dn = (deg - deg.mean()) / (deg.std() + 1e-9)
    X = np.zeros((n_cells, n_genes))
    for c in range(n_cells):
        active = rng.random(n_genes) < (0.20 + 0.55 * t[c])
        rate = np.clip(2.0 * (1 + 0.8 * t[c] * dn), 0.05, None)
        X[c] = rng.poisson(rate) * active
    return X, stage, deg


def run_self_test() -> int:
    import numpy as np
    import scipy.sparse as sp

    sys.path.insert(0, str(REPO))
    sys.path.insert(0, str(REPO / "own_baseline"))
    sys.path.insert(0, str(REPO / "scores"))
    import potency_metrics as pm
    from own_baseline import conditional_skill_report

    print("[self-test] building a synthetic 12-stage atlas")
    X, stage, deg = _synthetic_atlas()
    n_genes = X.shape[1]
    genes = [f"g{i}" for i in range(n_genes)]

    # --- primitives, exactly as prep_data.py computes them ------------------
    print("[self-test] primitives")
    import anndata as ad
    adata = ad.AnnData(X=X)
    adata.var_names = genes
    gene_count = (X > 0).sum(axis=1).astype(float)
    shannon = pm.stemid_entropy(adata, base=2.0)
    Xn = np.log1p(pm.library_normalize(X))
    pcc = pm._ccat_from_matrix(Xn, deg)
    assert np.isfinite(gene_count).all() and np.isfinite(shannon).all()
    assert np.isfinite(pcc).all()
    print(f"           gene_count {gene_count.min():.0f}-{gene_count.max():.0f}, "
          f"Shannon {shannon.min():.2f}-{shannon.max():.2f}, "
          f"PCC {pcc.min():+.3f}..{pcc.max():+.3f}")

    # --- CytoTRACE v1, the real port ----------------------------------------
    print("[self-test] CytoTRACE v1 (MVG -> GCS -> NNLS -> diffusion)")
    from cytotrace_full import (DIFF_ALPHA, DIFF_MAX_ITER, DIFF_TOL,
                                GCS_TOP_N, MVG_TOP_N, diffusion_smooth,
                                most_variable_genes, nnls_regression,
                                rank_and_scale_01, similarity_matrix_cleaned)
    Xs = sp.csr_matrix(X)
    cts = _cytotrace_v1(Xs, most_variable_genes, similarity_matrix_cleaned,
                        nnls_regression, diffusion_smooth, rank_and_scale_01,
                        min(GCS_TOP_N, n_genes // 3), min(MVG_TOP_N, n_genes // 3),
                        DIFF_ALPHA, DIFF_MAX_ITER, DIFF_TOL)
    assert cts.shape == (X.shape[0],) and np.isfinite(cts).all()
    print(f"           CT v1 in [{cts.min():.3f}, {cts.max():.3f}]")

    # --- SCENT SR and CCAT ---------------------------------------------------
    print("[self-test] SCENT SR and CCAT")
    edges = _synthetic_ppi(genes, deg)
    A, _g, col_idx, degree = pm.build_ppi_adjacency(edges, np.asarray(genes))
    sr = pm.scent_sr(adata, A, col_idx)
    ccat = pm.ccat(adata, col_idx, degree)
    assert np.nanmin(sr) >= -1e-9 and np.nanmax(sr) <= 1 + 1e-6, "SR outside [0,1]"
    assert np.abs(ccat).max() <= 1 + 1e-9, "CCAT outside [-1,1]"
    print(f"           SR in [{np.nanmin(sr):.3f}, {np.nanmax(sr):.3f}], "
          f"CCAT in [{ccat.min():+.3f}, {ccat.max():+.3f}]")

    # --- conditional skill, both kernels ------------------------------------
    print("[self-test] conditional skill, both kernels, bootstrap CI")
    ordinal = -stage.astype(float)      # higher = more potent
    results = {}
    for name, score, prim_name, prim in (
            ("CytoTRACE_v1", cts, "gene_count", gene_count),
            ("SR", sr, "PCC(x,degree)", pcc),
            ("CCAT", ccat, "PCC(x,degree)", pcc)):
        rep = conditional_skill_report(score, ordinal, {prim_name: prim},
                                       score_name=name, n_boot=(60, 150),
                                       verbose=False)
        block = rep["by_primitive"][prim_name]
        results[f"{name} | {prim_name}"] = {
            "scipy_weightedtau": block["conditional_skill"]["scipy_weightedtau"],
            "kang_wdm_taub": block["conditional_skill"]["kang_wdm_taub"],
            "kernels_agree": block["conditional_skill"]["kernels_agree"],
            "combined_verdict": block["conditional_skill"]["combined_verdict"],
        }
        b = block["conditional_skill"]["kang_wdm_taub"]
        assert b["CI95"][0] <= b["tau"] <= b["CI95"][1], "CI excludes its own estimate"
        print(f"           {name:14s} tau_b {b['tau']:+.4f} "
              f"CI[{b['CI95'][0]:+.4f},{b['CI95'][1]:+.4f}]  "
              f"{block['conditional_skill']['combined_verdict']}")

    # --- table and figure writers, against a temporary data root ------------
    print("[self-test] table and figure writers")
    import reproduce
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "track2/results").mkdir(parents=True)
        (root / "track2/results/gate2_conditional_skill.json").write_text(
            json.dumps({"n_cells_full": int(X.shape[0]), "seed": 42,
                        "n_boot": {"scipy_weightedtau": 60, "kang_wdm_taub": 150},
                        "conditional_skill": results}, indent=2))
        reproduce.stage_table(root)
        reproduce.stage_figures(root)
    for f in ("table.md", "table.json", "figure2_reproduced.png"):
        p = REPO / "reproduction" / f
        assert p.exists() and p.stat().st_size > 0, f"{f} not written"
        print(f"           wrote reproduction/{f} ({p.stat().st_size} bytes)")

    print("\n[self-test] every stage ran. NOTE: synthetic data -- this verifies "
          "the pipeline, not any published number.")
    return 0


def _synthetic_ppi(genes, deg):
    import numpy as np
    rng = np.random.default_rng(11)
    order = np.argsort(-deg)
    edges = set()
    for rank, gi in enumerate(order[1:], start=1):
        for p in rng.choice(order[:rank], size=min(3, rank), replace=False):
            a, b = sorted((genes[gi], genes[int(p)]))
            edges.add((a, b))
    return np.array(sorted(edges))


def _cytotrace_v1(Xs, most_variable_genes, similarity_matrix_cleaned,
                  nnls_regression, diffusion_smooth, rank_and_scale_01,
                  gcs_top_n, mvg_top_n, alpha, max_iter, tol):
    """
    The CytoTRACE v1 chain, wired exactly as experiments/track2/code/
    run_cytotrace.py wires it: gene counts -> CPM -> log1p -> GCS from the
    top-N GC-correlated genes -> MVG -> cell-cell similarity -> NNLS ->
    Markov diffusion -> rank to [0, 1].
    """
    import numpy as np
    counts = np.asarray(Xs.todense(), dtype=np.float64)
    gc = (counts > 0).sum(axis=1).astype(np.float64)
    tot = counts.sum(axis=1, keepdims=True)
    logn = np.log1p(counts / np.where(tot > 0, tot, 1.0) * 1e6)

    r = _corr(logn, gc)
    top = np.argsort(r)[-gcs_top_n:]
    gcs = np.exp(logn[:, top].mean(axis=1))

    mvg = most_variable_genes(logn, top_n=mvg_top_n)
    D_stoch = similarity_matrix_cleaned(logn[:, mvg], variant="rfaithful")
    nn = nnls_regression(D_stoch, gcs)
    diffused, _n_iter = diffusion_smooth(D_stoch, nn, alpha=alpha,
                                         max_iter=max_iter, tol=tol)
    return np.asarray(rank_and_scale_01(diffused), dtype=np.float64).ravel()


def _corr(M, v):
    import numpy as np
    Mc = M - M.mean(0, keepdims=True)
    vc = v - v.mean()
    den = np.sqrt((Mc ** 2).sum(0)) * np.sqrt((vc ** 2).sum())
    out = np.zeros(M.shape[1])
    ok = den > 0
    out[ok] = (Mc.T @ vc)[ok] / den[ok]
    return out


