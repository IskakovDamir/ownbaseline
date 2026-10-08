"""
Track 2 Gate 2 — build shared SCENT inputs so the verbatim SCENT R functions
and the vectorized Python reimpl operate on the IDENTICAL network + expression.

Outputs (<scratch>/scent_io/):
  adjMC.mtx / mc_genes.txt      max-connected subnetwork of (atlas ∩ STRING v12),
                                symmetric 0/1 adjacency, genes as rownames.
  sub_expr_libnorm.mtx          (mc_genes × N_SUB) library-normalized (linear CPM,
                                NOT logged — R applies its own log2 per convention),
                                subsample of cells (seed 42).
  sub_cell_idx.txt              row indices (into post-QC cell order) of the subsample.
  adjMC.npz                     adjacency + degree + mc gene atlas-col idx for Python
                                full-scale SR/CCAT.
Network = the SAME LCC SCENT's DoIntegPPI would build, precomputed once here so R
and Python are apples-to-apples.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse.csgraph import connected_components

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
STRING = VAULT / "w4/scaffolds/human_string_v12_thr700_lcc.npz"
IO = scratch_root() / "scent_io"
IO.mkdir(parents=True, exist_ok=True)

N_SUB = 500          # subsample size for the R validation run
SEED = 42


def main():
    genes = np.array(PREP.joinpath("genes.txt").read_text().split("\n"))
    X = sp.load_npz(PREP / "X_cells_genes.npz").tocsr()      # cells x genes
    n_cells = X.shape[0]
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    target_sum = float(np.median(lib[lib > 0]))

    d = np.load(STRING, allow_pickle=True)
    sgenes = np.array([str(g) for g in d["genes"]])
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    sidx = {g: i for i, g in enumerate(sgenes)}

    # atlas ∩ STRING
    common_atlas, common_string = [], []
    for i, g in enumerate(genes):
        j = sidx.get(str(g))
        if j is not None:
            common_atlas.append(i); common_string.append(j)
    common_atlas = np.array(common_atlas); common_string = np.array(common_string)
    A_sub = A[common_string][:, common_string]
    A_sub = ((A_sub + A_sub.T) > 0).astype(np.int8)
    A_sub.setdiag(0); A_sub.eliminate_zeros()

    # max-connected component (SCENT DoIntegPPI behaviour)
    ncomp, labels = connected_components(A_sub, directed=False)
    big = np.argmax(np.bincount(labels))
    sel = np.where(labels == big)[0]
    adjMC = A_sub[sel][:, sel].tocsr()
    mc_atlas_col = common_atlas[sel]                 # atlas columns of the mc genes
    mc_genes = genes[mc_atlas_col]
    degree = np.asarray(adjMC.sum(axis=1)).ravel().astype(np.float64)
    print(f"atlas∩STRING={len(common_atlas)}  LCC(adjMC)={adjMC.shape[0]} genes, "
          f"{adjMC.nnz//2} edges, deg mean {degree.mean():.1f}")

    # subsample cells
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(n_cells, size=min(N_SUB, n_cells), replace=False))

    # library-normalized (linear CPM) expression on mc genes, genes x cells
    Xsub = X[sub][:, mc_atlas_col].toarray().astype(np.float64)  # (N_SUB, n_mc)
    libsub = lib[sub][:, None]
    cpm = Xsub / np.where(libsub > 0, libsub, 1.0) * target_sum  # (N_SUB, n_mc)
    expr_gc = cpm.T                                              # (n_mc, N_SUB) genes x cells

    # write for R
    sio.mmwrite(str(IO / "adjMC.mtx"), adjMC.astype(np.int8))
    (IO / "mc_genes.txt").write_text("\n".join(map(str, mc_genes)))
    sio.mmwrite(str(IO / "sub_expr_libnorm.mtx"), sp.csr_matrix(expr_gc))
    (IO / "sub_cell_idx.txt").write_text("\n".join(map(str, sub.tolist())))
    np.savez(IO / "adjMC.npz", adjMC_data=adjMC.data, adjMC_indices=adjMC.indices,
             adjMC_indptr=adjMC.indptr, adjMC_shape=adjMC.shape,
             degree=degree, mc_atlas_col=mc_atlas_col, target_sum=target_sum,
             sub_idx=sub)
    print(f"target_sum(CPM)={target_sum:.1f}  subsample N={len(sub)} (seed {SEED})")
    print(f"wrote SCENT IO to {IO}")


if __name__ == "__main__":
    main()
