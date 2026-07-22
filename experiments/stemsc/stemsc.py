"""
stemsc.py — Python reference implementation of StemSC
    (Zhao et al. 2022, Stem Cell Research & Therapy 13:115;
     https://doi.org/10.1186/s13287-022-02803-5).

Byte-identical port of the R implementation:
    https://github.com/Zhao-Wenyuan/StemSC/blob/main/R/StemSC.R

The R implementation is:
    StemSC <- function(Exp) {
      data(gene_pairs)
      gid=rownames(Exp)
      pairs=pairs[pairs[,1]%in%gid&pairs[,2]%in%gid,]
      coms=Exp[match(pairs[,1],gid),,drop=F]-Exp[match(pairs[,2],gid),,drop=F]
      freq=colSums(coms>0)
      temp=Exp[match(pairs[,1],gid),,drop=F]+Exp[match(pairs[,2],gid),,drop=F]
      temp=colSums(temp==0)
      stemness = freq/(nrow(pairs)-temp)
      return(stemness)
    }

Score and primitive
-------------------
- StemSC score           = k / (n_present - n_zero_pairs)
    k             = # pairs with Exp[A] > Exp[B]
    n_present     = # reference pairs where BOTH genes appear in the target
                    dataset's gene index
    n_zero_pairs  = # pairs where BOTH Exp[A] == 0 and Exp[B] == 0
                    (informationless per Zhao 2022 R code)

- REO-primitive          = k / n_present    (the raw REO-fraction, before
                                             the "informationless" trim)

The theorem companion (paper1-theory-scaffold-characterization.md §5 point 4)
classifies StemSC as OUTSIDE the condemned class because its primitive is an
order-2 rank statistic: multiplying Exp by any positive scalar preserves
Exp[A] > Exp[B] and Exp[A] == 0 && Exp[B] == 0. So library-size normalization
does not change the score. This file's tests confirm both.

Reference gene pair set: `stemsc/ref_pairs.tsv` (extracted from the R package's
`data/gene_pairs.RData` — see the GATE 1 dossier for the extraction record).
Columns: geneA_entrez, geneB_entrez.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

REF_PAIRS_TSV = Path(__file__).parent / "ref_pairs.tsv"


def load_ref_pairs(path: Path = REF_PAIRS_TSV) -> pd.DataFrame:
    """Load the reference gene-pair set as a pandas DataFrame with columns
    ['geneA_entrez', 'geneB_entrez'] (both stored as strings).
    """
    df = pd.read_csv(path, sep="\t", dtype={"geneA_entrez": str, "geneB_entrez": str})
    return df


def _rows_dense(X):
    if sp.issparse(X):
        return np.asarray(X.todense(), dtype=np.float64)
    return np.asarray(X, dtype=np.float64)


def stemsc_from_matrix(
    X: np.ndarray,
    gene_ids: list[str],
    pairs: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Compute StemSC (score) and REO-primitive per cell.

    Parameters
    ----------
    X : (n_cells, n_genes) expression matrix (raw counts or RPKM — StemSC is
        library-size invariant; the paper explicitly states 'within-sample
        standardization does not change the results', so log/TPM/etc. all
        give the same score for the same cell as long as they are monotone
        per-cell).
    gene_ids : list of gene identifiers (Entrez IDs as strings) aligned with
        the columns of X.
    pairs : DataFrame with columns ['geneA_entrez', 'geneB_entrez'] (strings).

    Returns
    -------
    stemsc_score : (n_cells,) — the R-package-identical score = k / (n_present - n_zero_pairs)
    reo_primitive : (n_cells,) — the raw REO fraction = k / n_present
    info : dict with:
        n_pairs_total    : rows in `pairs` (16,893 for the shipped file)
        n_pairs_present  : # pairs where BOTH genes appear in gene_ids
        genes_ref_total  : # unique genes in `pairs`
        genes_ref_present: # of those present in gene_ids
        pair_zero_counts : (n_cells,) — per-cell "both-zero" pair count
    """
    Xd = _rows_dense(X)
    n_cells, n_genes = Xd.shape
    if n_genes != len(gene_ids):
        raise ValueError(f"gene_ids length {len(gene_ids)} != X columns {n_genes}")

    gene_ids_str = [str(g) for g in gene_ids]
    id2col = {g: i for i, g in enumerate(gene_ids_str)}
    genes_ref = set(pairs["geneA_entrez"].tolist()) | set(pairs["geneB_entrez"].tolist())
    genes_ref_present = [g for g in genes_ref if g in id2col]

    # Filter pairs to those whose BOTH genes are present in the target dataset.
    # This mirrors the R line:
    #   pairs = pairs[pairs[,1] %in% gid & pairs[,2] %in% gid, ]
    pair_A = pairs["geneA_entrez"].to_numpy()
    pair_B = pairs["geneB_entrez"].to_numpy()
    mask_present = np.array(
        [(a in id2col) and (b in id2col) for a, b in zip(pair_A, pair_B)],
        dtype=bool,
    )
    A_ids = pair_A[mask_present]
    B_ids = pair_B[mask_present]
    n_present = int(mask_present.sum())
    if n_present == 0:
        raise ValueError(
            "No reference pairs have BOTH genes present in the target "
            "dataset — check that gene_ids are Entrez IDs (as strings) "
            "matching the reference pair set."
        )

    A_cols = np.array([id2col[a] for a in A_ids], dtype=np.int64)
    B_cols = np.array([id2col[b] for b in B_ids], dtype=np.int64)

    # For each cell, compute:
    #   k       = # pairs where X[cell, A] > X[cell, B]
    #   n_zero  = # pairs where X[cell, A] == 0 AND X[cell, B] == 0
    # This is vectorized: XA = X[:, A_cols] is (n_cells, n_present).
    XA = Xd[:, A_cols]
    XB = Xd[:, B_cols]
    diff = XA - XB
    k = (diff > 0).sum(axis=1).astype(np.float64)
    n_zero = ((XA == 0.0) & (XB == 0.0)).sum(axis=1).astype(np.float64)

    denom_score = float(n_present) - n_zero  # (n_cells,)
    denom_score_safe = np.where(denom_score > 0, denom_score, 1.0)
    stemsc_score = k / denom_score_safe
    stemsc_score = np.where(denom_score > 0, stemsc_score, np.nan)

    # REO-primitive: raw fraction with the "informationless" trim removed
    # (denominator is a constant n_present per prereg definition).
    reo_primitive = k / float(n_present)

    info = {
        "n_pairs_total": int(len(pairs)),
        "n_pairs_present": n_present,
        "genes_ref_total": len(genes_ref),
        "genes_ref_present": len(genes_ref_present),
        "pair_zero_counts_median": float(np.median(n_zero)),
        "pair_zero_counts_max": float(n_zero.max()),
    }
    return stemsc_score, reo_primitive, info


def stemsc(
    adata,
    ref_pairs: Optional[pd.DataFrame] = None,
    layer: Optional[str] = None,
    gene_id_col: Optional[str] = None,
) -> np.ndarray:
    """AnnData-facing StemSC score.

    Parameters
    ----------
    adata : AnnData with .X = (n_cells, n_genes) expression matrix
    ref_pairs : reference pair DataFrame (default: load from `ref_pairs.tsv`)
    layer : optional adata.layers key
    gene_id_col : if adata.var has an Entrez-ID column, its name; otherwise
        adata.var_names is used directly.
    """
    if ref_pairs is None:
        ref_pairs = load_ref_pairs()
    X = adata.layers[layer] if layer is not None else adata.X
    if gene_id_col is not None:
        gene_ids = [str(g) for g in adata.var[gene_id_col].tolist()]
    else:
        gene_ids = [str(g) for g in adata.var_names]
    score, _prim, _info = stemsc_from_matrix(X, gene_ids, ref_pairs)
    return score


def reo_primitive(
    adata,
    ref_pairs: Optional[pd.DataFrame] = None,
    layer: Optional[str] = None,
    gene_id_col: Optional[str] = None,
) -> np.ndarray:
    """AnnData-facing REO primitive (raw fraction, before the informationless trim)."""
    if ref_pairs is None:
        ref_pairs = load_ref_pairs()
    X = adata.layers[layer] if layer is not None else adata.X
    if gene_id_col is not None:
        gene_ids = [str(g) for g in adata.var[gene_id_col].tolist()]
    else:
        gene_ids = [str(g) for g in adata.var_names]
    _score, prim, _info = stemsc_from_matrix(X, gene_ids, ref_pairs)
    return prim


def stemsc_both(
    adata,
    ref_pairs: Optional[pd.DataFrame] = None,
    layer: Optional[str] = None,
    gene_id_col: Optional[str] = None,
) -> dict:
    """Return dict {'score': ..., 'primitive': ..., 'info': ...}."""
    if ref_pairs is None:
        ref_pairs = load_ref_pairs()
    X = adata.layers[layer] if layer is not None else adata.X
    if gene_id_col is not None:
        gene_ids = [str(g) for g in adata.var[gene_id_col].tolist()]
    else:
        gene_ids = [str(g) for g in adata.var_names]
    score, prim, info = stemsc_from_matrix(X, gene_ids, ref_pairs)
    return {"score": score, "primitive": prim, "info": info}
