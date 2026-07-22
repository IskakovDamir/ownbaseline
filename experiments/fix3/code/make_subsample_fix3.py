"""FIX 3 — Briggs 3000-cell seed-42 subsample shared by SLICE and dpath (mirrors
make_subsample.py; writes to a SEPARATE fix3_sub dir so the zebrafish subsample is intact)."""
import numpy as np, scipy.sparse as sp, scipy.io as sio
from pathlib import Path
PREP = Path("/Users/damir/damir-research-vault/06-code/tautology-diagnostic/fix3/data/prepared")
OUT = Path("/private/tmp/claude-501/-Users-damir-damir-research-vault/f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad/fix3_sub")
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
