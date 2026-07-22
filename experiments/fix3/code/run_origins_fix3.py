"""
FIX 3 GATE 2 — ORIGINS on Briggs Xenopus via the vectorized x^T A x validated ==
ORIGINS::activity in the zebrafish decider. Uses ORIGINS' native differentiation_edges
network (already exported to scratchpad/origins_io/diff_edges.tsv). Writes a 500-cell
subsample for an independent Briggs R re-validation. Declared primitives: gene count AND
PCC(x,degree).
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import scipy.io as sio

VAULT = Path("/Users/damir/damir-research-vault/06-code/tautology-diagnostic")
PREP = VAULT / "fix3/data/prepared"
SC = Path("/private/tmp/claude-501/-Users-damir-damir-research-vault/"
          "f1408352-cdbb-4882-96f8-dc52eb89b3ad/scratchpad")
IO = SC / "fix3_origins_io"; IO.mkdir(parents=True, exist_ok=True)
OUT = VAULT / "fix3/results/origins_scores.npz"
CHUNK = 4000; N_SUB = 500; SEED = 42


def main():
    edges = [l.split("\t") for l in (SC / "origins_io/diff_edges.tsv").read_text().strip().split("\n")]
    genes = np.array(PREP.joinpath("genes.txt").read_text().split("\n"))
    genes_up = np.array([g.upper() for g in genes])
    gidx = {}
    for i, g in enumerate(genes_up):
        gidx.setdefault(g, i)                    # first occurrence per uppercase symbol
    diff_genes = sorted({e[0] for e in edges} | {e[1] for e in edges})
    common = [g for g in diff_genes if g in gidx]
    cpos = {g: i for i, g in enumerate(common)}
    rows, cols = [], []
    for a, b in edges:
        if a in cpos and b in cpos:
            rows += [cpos[a], cpos[b]]; cols += [cpos[b], cpos[a]]
    n = len(common)
    A = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    A = (A > 0).astype(np.float64).tocsr(); A.setdiag(0); A.eliminate_zeros()
    atlas_col = np.array([gidx[g] for g in common])
    print(f"diff∩Briggs common {len(common)} genes, {A.nnz//2} edges")

    X = sp.load_npz(PREP / "X_cells_genes.npz").tocsr()
    n_cells = X.shape[0]
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    target_sum = float(np.median(lib[lib > 0]))

    def raw_activity(idx):
        out = np.empty(len(idx))
        for s0 in range(0, len(idx), CHUNK):
            s1 = min(s0 + CHUNK, len(idx)); ii = idx[s0:s1]
            counts = X[ii][:, atlas_col].toarray().astype(np.float64)
            cpm = counts / np.where(lib[ii][:, None] > 0, lib[ii][:, None], 1.0) * target_sum
            D = cpm @ A
            out[s0:s1] = np.asarray(np.sum(cpm * D, axis=1)).ravel()
        return out

    rng = np.random.default_rng(SEED)
    sub = np.sort(rng.choice(n_cells, size=min(N_SUB, n_cells), replace=False))
    cpm_sub = (X[sub][:, atlas_col].toarray().astype(np.float64)
               / np.where(lib[sub][:, None] > 0, lib[sub][:, None], 1.0) * target_sum)
    sio.mmwrite(str(IO / "sub_expr_cpm.mtx"), sp.csr_matrix(cpm_sub.T))
    (IO / "origins_genes.txt").write_text("\n".join(common))

    raw = raw_activity(np.arange(n_cells))
    origins = (raw - raw.min()) / (raw.max() - raw.min())
    np.savez(OUT, origins=origins, raw=raw)
    np.savez(IO / "meta.npz", sub_idx=sub, sub_raw=raw[sub])
    from scipy.stats import spearmanr
    p = np.load(PREP / "primitives.npz")
    print(f"  ORIGINS vs ordinal {spearmanr(origins, p['rank_ordinal']).statistic:+.4f} "
          f"vs gene_count {spearmanr(origins, p['gene_count']).statistic:+.4f} "
          f"vs PCC {spearmanr(origins, p['pcc_string_v12']).statistic:+.4f}")
    print(f"  wrote {OUT}; subsample → {IO}")


if __name__ == "__main__":
    main()
