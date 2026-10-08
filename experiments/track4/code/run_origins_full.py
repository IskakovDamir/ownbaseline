"""
Track 4 — ORIGINS at full 39,505-cell scale via vectorized x^T A x on ORIGINS'
native differentiation network, VALIDATED against real ORIGINS::activity (R) on
a 500-cell subsample (ranking-equivalent; ORIGINS' R double-loop is infeasible
at 39.5k cells). ORIGINS activity(cell) = Σ_i x_i (A x)_i, min-max normalized —
verbatim to the package's activity() (see run_origins_validate.R for the real run).

§3a-declared primitives for ORIGINS: report vs BOTH gene count AND PCC(x,degree).

Outputs: track2/results/origins_scores.npz (origins on full order) +
         track2/results/origins_validation.json
"""
from __future__ import annotations
import json
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
from own_baseline.paths import data_root, scratch_root  # noqa: E402
VAULT = data_root()  # run inputs and outputs (data)
PREP = VAULT / "track2/data/prepared"
IO = scratch_root() / "origins_io"
OUT = VAULT / "track2/results/origins_scores.npz"
VAL = VAULT / "track2/results/origins_validation.json"
CHUNK = 4000


def main():
    d = np.load(IO / "origins_net.npz")
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]),
                      shape=tuple(d["A_shape"]))
    atlas_col, target_sum, sub_idx = d["atlas_col"], float(d["target_sum"]), d["sub_idx"]

    X = sp.load_npz(PREP / "X_cells_genes.npz").tocsr()
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    n_cells = X.shape[0]

    def raw_activity(idx):
        out = np.empty(len(idx))
        for s0 in range(0, len(idx), CHUNK):
            s1 = min(s0 + CHUNK, len(idx))
            ii = idx[s0:s1]
            counts = X[ii][:, atlas_col].toarray().astype(np.float64)
            cpm = counts / np.where(lib[ii][:, None] > 0, lib[ii][:, None], 1.0) * target_sum
            D = cpm @ A                      # chunk × n (dense)
            out[s0:s1] = np.asarray(np.sum(cpm * D, axis=1)).ravel()
        return out

    # validation vs real ORIGINS R on subsample
    from scipy.stats import spearmanr
    import csv as _csv
    raw_sub = raw_activity(sub_idx)
    rrows = list(_csv.DictReader(open(IO / "origins_R_subsample.csv")))
    origins_R = np.array([float(r["origins_R"]) for r in rrows])
    val = {"n_subsample": int(len(sub_idx)),
           "spearman_R_vs_py": float(spearmanr(origins_R, raw_sub).statistic)}
    print(f"VALIDATION vs real ORIGINS R (subsample): Spearman={val['spearman_R_vs_py']:.6f}")
    with open(VAL, "w") as f:
        json.dump(val, f, indent=2)

    # full scale
    raw = raw_activity(np.arange(n_cells))
    origins = (raw - raw.min()) / (raw.max() - raw.min())   # min-max, matches activity()
    np.savez(OUT, origins=origins, raw=raw)
    p = np.load(PREP / "primitives.npz")
    print(f"  ORIGINS vs rank_kimmel rho={spearmanr(origins, p['rank_kimmel']).statistic:+.4f}"
          f"  vs gene_count rho={spearmanr(origins, p['gene_count']).statistic:+.4f}"
          f"  vs PCC rho={spearmanr(origins, p['pcc_string_v12']).statistic:+.4f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
