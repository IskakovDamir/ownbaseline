"""
Build shared ORIGINS inputs so real ORIGINS::activity (R) and a vectorized
x^T A x (Python) run on the IDENTICAL differentiation network + expression.
Reuses the SAME 500-cell subsample as the SCENT validation (scent_io/adjMC.npz
sub_idx) for a consistent validation set.

Outputs (scratchpad/origins_io/):
  sub_expr_cpm.mtx / origins_genes.txt   CPM expression on (diff∩atlas) genes ×
                                          500 subsample cells (genes × cells).
  origins_net.npz                         symmetric adjacency + atlas col idx +
                                          target_sum for Python full-scale.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

VAULT = Path("/Users/damir/damir-research-vault/06-code/tautology-diagnostic")
PREP = VAULT / "track2/data/prepared"
SC = Path("/private/tmp/claude-501/-Users-damir-damir-research-vault/"
          "f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad")
IO = SC / "origins_io"
IO.mkdir(parents=True, exist_ok=True)


def main():
    edges = [l.split("\t") for l in (IO / "diff_edges.tsv").read_text().strip().split("\n")]
    genes = np.array(PREP.joinpath("genes.txt").read_text().split("\n"))
    gidx = {g: i for i, g in enumerate(genes)}

    diff_genes = sorted({e[0] for e in edges} | {e[1] for e in edges})
    common = [g for g in diff_genes if g in gidx]
    cpos = {g: i for i, g in enumerate(common)}
    print(f"diff genes {len(diff_genes)}, common with atlas {len(common)}")

    rows, cols = [], []
    for a, b in edges:
        if a in cpos and b in cpos:
            rows += [cpos[a], cpos[b]]; cols += [cpos[b], cpos[a]]  # symmetric
    n = len(common)
    A = sp.coo_matrix((np.ones(len(rows), np.float64), (rows, cols)), shape=(n, n))
    A = (A > 0).astype(np.float64).tocsr(); A.setdiag(0); A.eliminate_zeros()
    atlas_col = np.array([gidx[g] for g in common])
    print(f"adjacency {A.shape}, {A.nnz//2} undirected edges")

    X = sp.load_npz(PREP / "X_cells_genes.npz").tocsr()
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    target_sum = float(np.median(lib[lib > 0]))

    d = np.load(SC / "scent_io/adjMC.npz"); sub = d["sub_idx"]
    counts = X[sub][:, atlas_col].toarray().astype(np.float64)          # (500, n)
    cpm = counts / np.where(lib[sub][:, None] > 0, lib[sub][:, None], 1.0) * target_sum
    sio.mmwrite(str(IO / "sub_expr_cpm.mtx"), sp.csr_matrix(cpm.T))     # genes × cells
    (IO / "origins_genes.txt").write_text("\n".join(common))
    np.savez(IO / "origins_net.npz", A_data=A.data, A_indices=A.indices,
             A_indptr=A.indptr, A_shape=A.shape, atlas_col=atlas_col,
             target_sum=target_sum, sub_idx=sub)
    print(f"wrote ORIGINS IO to {IO}  (subsample N={len(sub)})")


if __name__ == "__main__":
    main()
