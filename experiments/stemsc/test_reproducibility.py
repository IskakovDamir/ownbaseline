"""
test_reproducibility.py — GATE 1 reproducibility check for StemSC.

Runs the Python StemSC implementation on Zhao et al.'s own example data
(GSE85066 H7 hESC RPKM matrix, shipped with the R package as data/example.RData)
and confirms it reproduces the ESC-ceiling behaviour reported in the paper
(median ESC StemSC = 0.990 on GSE85066).

Also runs library-size-invariance and REO-primitive sanity checks that verify
the theorem companion §5 point 4 classification (order-2 REO, library-size
invariant).

Data provenance:
  gene_pairs.RData [FETCH: https://raw.githubusercontent.com/Zhao-Wenyuan/StemSC/main/data/gene_pairs.RData · 2026-07-18]
  example.RData    [FETCH: https://raw.githubusercontent.com/Zhao-Wenyuan/StemSC/main/data/example.RData · 2026-07-18]

Reproducibility target (from Zhao 2022 paper text, quoted at GATE 1):
  "Median ESC StemSC values: 0.990 (GSE85066), 0.984 (GSE109979)"
  [FETCH: https://pmc.ncbi.nlm.nih.gov/articles/PMC8935746/ · 2026-07-18]
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import rdata  # type: ignore
from stemsc import load_ref_pairs, stemsc_from_matrix


def load_example():
    parsed = rdata.read_rda(str(HERE / "raw" / "example.RData"))
    Exp = parsed["Exp"]                       # (n_genes, n_cells) — R convention
    gene_ids = [str(int(g)) for g in Exp.index]  # Entrez IDs
    X = Exp.to_numpy().T                       # (n_cells, n_genes) — our convention
    cell_ids = [str(c) for c in Exp.columns]
    return X, gene_ids, cell_ids


def test_reproduces_esc_ceiling():
    pairs = load_ref_pairs()
    X, gene_ids, cell_ids = load_example()
    score, prim, info = stemsc_from_matrix(X, gene_ids, pairs)

    print(f"n_cells = {X.shape[0]}, n_genes = {X.shape[1]}")
    print(f"reference pairs total   : {info['n_pairs_total']}")
    print(f"reference pairs present : {info['n_pairs_present']}")
    print(f"reference genes total   : {info['genes_ref_total']}")
    print(f"reference genes present : {info['genes_ref_present']}")
    print(f"per-cell 'both-zero' pair count median = {info['pair_zero_counts_median']}, "
          f"max = {info['pair_zero_counts_max']}")
    print(f"StemSC score   : median = {np.nanmedian(score):.4f}, "
          f"min = {np.nanmin(score):.4f}, max = {np.nanmax(score):.4f}")
    print(f"REO-primitive : median = {np.nanmedian(prim):.4f}, "
          f"min = {np.nanmin(prim):.4f}, max = {np.nanmax(prim):.4f}")

    # Paper reports median ESC StemSC = 0.990 on GSE85066.
    # Allow a small tolerance (Zhao pipeline is deterministic; any difference
    # would reflect a genuine implementation discrepancy).
    target = 0.990
    tol = 0.01
    got = float(np.nanmedian(score))
    diff = abs(got - target)
    print(f"\nReproducibility target: median ESC StemSC = {target:.4f}")
    print(f"Observed:              median ESC StemSC = {got:.4f}")
    print(f"|diff| = {diff:.4f}  (tol = {tol:.4f})")
    assert diff <= tol, f"reproducibility FAIL: |median - {target}| = {diff:.4f} > {tol}"
    print("[PASS] Reproduces Zhao 2022's reported ESC-ceiling median (0.990 ± 0.01).")


def test_library_size_invariance():
    """Multiplying every cell by a positive scalar must not change the score
    (order-2 REO property). Verified against the R implementation semantics."""
    pairs = load_ref_pairs()
    X, gene_ids, _ = load_example()
    score_1, prim_1, _ = stemsc_from_matrix(X, gene_ids, pairs)

    # scale each cell by a random positive factor
    rng = np.random.default_rng(42)
    scales = rng.uniform(0.1, 10.0, size=X.shape[0])
    X_scaled = X * scales[:, None]
    score_2, prim_2, _ = stemsc_from_matrix(X_scaled, gene_ids, pairs)

    diff_score = np.nanmax(np.abs(score_1 - score_2))
    diff_prim = np.nanmax(np.abs(prim_1 - prim_2))
    print(f"\nlibrary-size invariance:")
    print(f"  max |Δ score| across cells = {diff_score:.3e}")
    print(f"  max |Δ REO-primitive| = {diff_prim:.3e}")
    assert diff_score < 1e-9, f"score NOT library-size invariant (max diff {diff_score:.3e})"
    assert diff_prim < 1e-9, f"primitive NOT library-size invariant (max diff {diff_prim:.3e})"
    print("[PASS] StemSC score AND REO-primitive are library-size invariant "
          "(order-2 REO — theorem classification confirmed).")


def test_log_transform_invariance():
    """log1p transform must not change the score (monotone per-cell → REO preserved)."""
    pairs = load_ref_pairs()
    X, gene_ids, _ = load_example()
    score_1, prim_1, _ = stemsc_from_matrix(X, gene_ids, pairs)

    X_log = np.log1p(X)
    score_2, prim_2, _ = stemsc_from_matrix(X_log, gene_ids, pairs)

    diff_score = np.nanmax(np.abs(score_1 - score_2))
    diff_prim = np.nanmax(np.abs(prim_1 - prim_2))
    print(f"\nlog1p invariance:")
    print(f"  max |Δ score| across cells = {diff_score:.3e}")
    print(f"  max |Δ REO-primitive| = {diff_prim:.3e}")
    assert diff_score < 1e-9, f"score NOT log1p-invariant"
    assert diff_prim < 1e-9, f"primitive NOT log1p-invariant"
    print("[PASS] log1p invariance holds (theorem §5 point 4 confirmed).")


if __name__ == "__main__":
    print("=" * 60)
    print("GATE 1 reproducibility check — StemSC on Zhao 2022 example data")
    print("=" * 60)
    test_reproduces_esc_ceiling()
    test_library_size_invariance()
    test_log_transform_invariance()
    print("\nAll GATE 1 reproducibility tests PASSED.")
