"""
Track 2 Gate 2 — SR + CCAT at full 39,505-cell scale, via a vectorized
implementation of the VERBATIM SCENT algorithm, VALIDATED against the SCENT R
package's CompSRanaPRL / CompCCAT (run in run_scent_validate.R) on a 500-cell
subsample. No substitution: same algorithm, proven numerically equivalent to the
reference R functions, then applied at scale (the R per-cell loop is infeasible
for 39.5k cells over an 8,468-node network).

Preprocessing matched to SCENT source exactly:
  CCAT : log2(CPM + 1)     then Pearson(expr, degree)          [CompCCAT.R]
  SR   : log2(CPM + 1.1)   then Σ_i π_i S_i / log(λ_max(A))     [DoIntegPPI + CompSRanaPRL]
CPM = counts / library_size * target_sum (target_sum = median library size).

Declared §3a primitive for BOTH SR and CCAT = PCC(x, degree) STRING v12.

Outputs: track2/results/scent_scores.npz (ccat, sr on full cell order) +
         track2/results/scent_validation.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp

# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# scratch directory, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root, repo_root, scratch_root  # noqa: E402
CODE = repo_root()   # this repository (code)
VAULT = data_root()  # run inputs and outputs (data)
sys.path.insert(0, str(CODE / "own_baseline"))
import potency_metrics as pm

PREP = VAULT / "track2/data/prepared"
IO = scratch_root() / "scent_io"
OUT = VAULT / "track2/results/scent_scores.npz"
VAL = VAULT / "track2/results/scent_validation.json"
CHUNK = 4000


def load_net():
    d = np.load(IO / "adjMC.npz")
    A = sp.csr_matrix((d["adjMC_data"], d["adjMC_indices"], d["adjMC_indptr"]),
                      shape=tuple(d["adjMC_shape"])).astype(np.float64)
    return (A, d["degree"].astype(np.float64), d["mc_atlas_col"],
            float(d["target_sum"]), d["sub_idx"])


def compute_scores(X, lib, cells_idx, A, degree, mc_col, target_sum, maxSR):
    """CCAT + SR for the given cell indices (chunked)."""
    n = len(cells_idx)
    ccat = np.empty(n); sr = np.empty(n)
    kc = degree - degree.mean()
    for s0 in range(0, n, CHUNK):
        s1 = min(s0 + CHUNK, n)
        idx = cells_idx[s0:s1]
        counts = X[idx][:, mc_col].toarray().astype(np.float64)     # (c, G)
        tot = lib[idx][:, None]
        cpm = counts / np.where(tot > 0, tot, 1.0) * target_sum
        # CCAT: Pearson(log2(cpm+1), degree)
        Xc = np.log2(cpm + 1.0)
        ccat[s0:s1] = pm._ccat_from_matrix(Xc, degree)
        # SR: log2(cpm+1.1) entropy rate
        Xn = np.log2(cpm + 1.1)
        sr[s0:s1] = pm._sr_from_matrix(Xn, A, maxSR, chunk=CHUNK)
    return ccat, sr


def main():
    A, degree, mc_col, target_sum, sub_idx = load_net()
    maxSR = pm._max_entropy_rate(A)
    print(f"network {A.shape[0]} genes, maxSR=log(lambda_max)={maxSR:.4f}")

    X = sp.load_npz(PREP / "X_cells_genes.npz").tocsr()
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    n_cells = X.shape[0]

    # ---- validation on the 500-cell subsample vs real SCENT R ----
    from scipy.stats import spearmanr, pearsonr
    import csv as _csv
    ccat_sub, sr_sub = compute_scores(X, lib, sub_idx, A, degree, mc_col, target_sum, maxSR)
    rrows = list(_csv.DictReader(open(IO / "scent_R_subsample.csv")))
    ccat_R = np.array([float(r["ccat_R"]) for r in rrows])
    sr_R = np.array([float(r["sr_R"]) for r in rrows])
    val = {
        "n_subsample": int(len(sub_idx)),
        "CCAT": {"spearman_R_vs_py": float(spearmanr(ccat_R, ccat_sub).statistic),
                 "pearson_R_vs_py": float(pearsonr(ccat_R, ccat_sub)[0]),
                 "max_abs_diff": float(np.max(np.abs(ccat_R - ccat_sub)))},
        "SR": {"spearman_R_vs_py": float(spearmanr(sr_R, sr_sub).statistic),
               "pearson_R_vs_py": float(pearsonr(sr_R, sr_sub)[0]),
               "max_abs_diff": float(np.max(np.abs(sr_R - sr_sub)))},
    }
    print("VALIDATION vs real SCENT R (500-cell subsample):")
    print(f"  CCAT: Spearman={val['CCAT']['spearman_R_vs_py']:.6f}  "
          f"max|Δ|={val['CCAT']['max_abs_diff']:.3e}")
    print(f"  SR  : Spearman={val['SR']['spearman_R_vs_py']:.6f}  "
          f"max|Δ|={val['SR']['max_abs_diff']:.3e}")
    with open(VAL, "w") as f:
        json.dump(val, f, indent=2)

    # ---- full scale ----
    print("Full-scale SR + CCAT on all cells ...")
    all_idx = np.arange(n_cells)
    ccat, sr = compute_scores(X, lib, all_idx, A, degree, mc_col, target_sum, maxSR)
    np.savez(OUT, ccat=ccat, sr=sr, maxSR=maxSR)
    p = np.load(PREP / "primitives.npz")
    print(f"  CCAT vs rank_kimmel rho={spearmanr(ccat, p['rank_kimmel']).statistic:+.4f}"
          f"  vs PCC rho={spearmanr(ccat, p['pcc_string_v12']).statistic:+.4f}")
    print(f"  SR   vs rank_kimmel rho={spearmanr(sr, p['rank_kimmel']).statistic:+.4f}"
          f"  vs PCC rho={spearmanr(sr, p['pcc_string_v12']).statistic:+.4f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
