"""
FIX 3 GATE 2 — SR + CCAT on Briggs Xenopus via the SAME vectorized verbatim-SCENT
algorithm validated == SCENT R to machine precision in the zebrafish decider
(run_scent_full.py). Builds the atlas∩STRING v12 LCC, computes SR (log2(CPM+1.1)
entropy rate) + CCAT (Pearson(log2(CPM+1), degree)) full-scale, and writes a
500-cell subsample (genes×cells CPM + adjMC) for an independent Briggs R re-validation
(run_scent_validate_fix3.R). Declared primitive for both = PCC(x,degree) STRING v12.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.io as sio
from scipy.sparse.csgraph import connected_components

# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# agent-session scratchpad, so the module only imported on one machine.
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

PREP = VAULT / "fix3/data/prepared"
STRING = VAULT / "w4/scaffolds/human_string_v12_thr700_lcc.npz"
IO = scratch_root() / "fix3_scent_io"
IO.mkdir(parents=True, exist_ok=True)
OUT = VAULT / "fix3/results/scent_scores.npz"
OUT.parent.mkdir(parents=True, exist_ok=True)
CHUNK = 4000
N_SUB = 500
SEED = 42


def main():
    genes = np.array(PREP.joinpath("genes.txt").read_text().split("\n"))
    genes_up = np.array([g.upper() for g in genes])
    X = sp.load_npz(PREP / "X_cells_genes.npz").tocsr()
    n_cells = X.shape[0]
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    target_sum = float(np.median(lib[lib > 0]))

    d = np.load(STRING, allow_pickle=True)
    sgenes = np.array([str(g) for g in d["genes"]])
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    sidx = {g: i for i, g in enumerate(sgenes)}
    ca, cs, seen = [], [], set()
    for i, g in enumerate(genes_up):
        j = sidx.get(g)
        if j is not None and g not in seen:
            ca.append(i); cs.append(j); seen.add(g)
    ca, cs = np.array(ca), np.array(cs)
    Asub = ((A[cs][:, cs] + A[cs][:, cs].T) > 0).astype(np.int8)
    Asub.setdiag(0); Asub.eliminate_zeros()
    ncomp, lab = connected_components(Asub, directed=False)
    big = np.argmax(np.bincount(lab)); sel = np.where(lab == big)[0]
    adjMC = Asub[sel][:, sel].tocsr().astype(np.float64)
    mc_col = ca[sel]
    degree = np.asarray(adjMC.sum(axis=1)).ravel()
    maxSR = pm._max_entropy_rate(adjMC)
    print(f"atlas∩STRING={len(ca)} LCC(adjMC)={adjMC.shape[0]} genes, {adjMC.nnz//2} edges, "
          f"maxSR={maxSR:.4f}")

    def scores(idx):
        n = len(idx); ccat = np.empty(n); sr = np.empty(n)
        for s0 in range(0, n, CHUNK):
            s1 = min(s0 + CHUNK, n); ii = idx[s0:s1]
            counts = X[ii][:, mc_col].toarray().astype(np.float64)
            tot = lib[ii][:, None]; cpm = counts / np.where(tot > 0, tot, 1.0) * target_sum
            ccat[s0:s1] = pm._ccat_from_matrix(np.log2(cpm + 1.0), degree)
            sr[s0:s1] = pm._sr_from_matrix(np.log2(cpm + 1.1), adjMC, maxSR, chunk=CHUNK)
        return ccat, sr

    # write 500-cell subsample for the independent Briggs R re-validation
    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(n_cells, size=min(N_SUB, n_cells), replace=False))
    cpm_sub = (X[sub][:, mc_col].toarray().astype(np.float64)
               / np.where(lib[sub][:, None] > 0, lib[sub][:, None], 1.0) * target_sum)
    sio.mmwrite(str(IO / "adjMC.mtx"), adjMC.astype(np.int8))
    sio.mmwrite(str(IO / "sub_expr_libnorm.mtx"), sp.csr_matrix(cpm_sub.T))
    (IO / "mc_genes.txt").write_text("\n".join(genes_up[mc_col]))
    np.savez(IO / "meta.npz", sub_idx=sub)

    ccat, sr = scores(np.arange(n_cells))
    np.savez(OUT, ccat=ccat, sr=sr, maxSR=maxSR)
    np.savez(IO / "meta.npz", sub_idx=sub, sub_ccat=ccat[sub], sub_sr=sr[sub])  # for R compare
    from scipy.stats import spearmanr
    p = np.load(PREP / "primitives.npz")
    print(f"  CCAT vs ordinal {spearmanr(ccat, p['rank_ordinal']).statistic:+.4f} vs PCC "
          f"{spearmanr(ccat, p['pcc_string_v12']).statistic:+.4f}")
    print(f"  SR   vs ordinal {spearmanr(sr, p['rank_ordinal']).statistic:+.4f} vs PCC "
          f"{spearmanr(sr, p['pcc_string_v12']).statistic:+.4f}")
    print(f"  wrote {OUT}; subsample+adjMC for R re-validation → {IO}")


if __name__ == "__main__":
    main()
