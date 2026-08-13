"""FIX 3 — Briggs 3000-cell seed-42 subsample shared by SLICE and dpath (mirrors
make_subsample.py; writes to a SEPARATE fix3_sub dir so the zebrafish subsample is intact)."""
import numpy as np, scipy.sparse as sp, scipy.io as sio
from pathlib import Path
# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# agent-session scratchpad, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root, scratch_root  # noqa: E402
PREP = data_root() / "fix3/data/prepared"
OUT = scratch_root() / "fix3_sub"
OUT.mkdir(parents=True, exist_ok=True)
N_SUB, SEED = 3000, 42
X = sp.load_npz(PREP / "X_cells_genes.npz").tocsr()
n_cells, n_genes = X.shape
feats = [l for l in (PREP / "features.tsv").read_text().split("\n") if l != ""]
assert len(feats) == n_genes, (len(feats), n_genes)
# SLICE's human GO kappa matrix uses uppercase human symbols → match Briggs uppercase orthologs
feats_up = [g.upper() for g in feats]
rng = np.random.default_rng(SEED)
idx = np.sort(rng.choice(n_cells, size=min(N_SUB, n_cells), replace=False))
G = sp.csc_matrix(X[idx, :].T.tocoo())               # genes x cells
sio.mmwrite(str(OUT / "sub_genes_cells.mtx"), G, field="integer", symmetry="general")
(OUT / "sub_features.tsv").write_text("\n".join(feats_up) + "\n")
np.savetxt(OUT / "sub_cell_idx.txt", idx, fmt="%d")
print(f"Briggs subsample: {len(idx)} cells x {n_genes} genes → {OUT}; first5 {idx[:5].tolist()}")
