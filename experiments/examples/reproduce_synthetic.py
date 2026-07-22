"""
Reproduce the synthetic validation (paper Figure F2).

Runs the pre-registered align-sweep and prints the Delta table; if matplotlib is
available, also saves the two-panel figure (AUROC curves + the Delta diagnostic).

    python examples/reproduce_synthetic.py

Expected qualitative result (the pre-registered prediction):
    Delta ~ 0 at align = 0  (only the tautological / Bayes component is shared)
    Delta strictly increasing in align
    Delta > 0 at align = 1
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from synthetic import align_sweep

res = align_sweep()                       # {align: (mean_delta, mc_se)}
aligns = np.array(sorted(res))
delta = np.array([res[a][0] for a in aligns])
se = np.array([res[a][1] for a in aligns])

print(f"{'align':>6} {'mean_delta':>12} {'mc_se':>8}")
for a, d, s in zip(aligns, delta, se):
    print(f"{a:6.2f} {d:12.4f} {s:8.4f}")
print(f"\nalign=0 Delta = {delta[0]:+.4f}  (should be ~0: only the tautological component shared)")
print(f"align=1 Delta = {delta[-1]:+.4f}  (should be >0: genuine shared structure detected)")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    own = 0.726
    transfer = own + delta
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.4))
    a1.plot(aligns, transfer, "o-", color="#2c3e50", label="Transfer (A -> B)", lw=1.8)
    a1.plot(aligns, np.full_like(aligns, own), "s--", color="#16a085", label="Own-entropy (B)", lw=1.6)
    a1.axhline(0.5, ls=":", lw=1, color="#888", zorder=0)
    a1.text(1.0, 0.505, "chance", fontsize=8, color="#666", ha="right")
    a1.set_xlabel("align (fraction of structural axis shared across A, B)")
    a1.set_ylabel("AUROC")
    a1.set_ylim(0.48, 0.86)
    a1.set_title("Transfer > own-entropy only as real\nstructural axis becomes shared", fontsize=10.5)
    a1.legend(fontsize=9, loc="lower right", frameon=False)
    a1.spines[["top", "right"]].set_visible(False)

    a2.plot(aligns, delta, "o-", color="#c0392b", lw=2)
    a2.axhline(0, ls="--", lw=1, color="k", zorder=0)
    a2.axhspan(-0.005, 0.005, alpha=0.10, color="#888", zorder=0)
    # empirical anchor: align = 0 = real lung↔ovarian regime
    a2.scatter([0], [delta[0]], s=110, color="#c0392b", zorder=5)
    a2.annotate("align = 0: Δ ≈ 0\nEmpirical lung ↔ ovarian regime\n(only tautological Bayes/\nentropy component shared)",
                xy=(0.02, delta[0]), xytext=(0.10, 0.093),
                fontsize=8.5, color="#333",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    a2.scatter([1], [delta[-1]], s=110, color="#c0392b", zorder=5)
    a2.annotate(f"align = 1: Δ = {delta[-1]:+.3f}\nGenuine shared structure\n→ Δ recovers it (has power)",
                xy=(0.98, delta[-1]), xytext=(0.40, 0.030),
                fontsize=8.5, color="#333", ha="left",
                arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))
    a2.set_xlabel("align")
    a2.set_ylabel("Δ = transfer - own-entropy")
    a2.set_title("Δ reads 0 when only the Bayes component is shared,\nrises with real conserved structure", fontsize=10.5)
    a2.set_ylim(-0.015, 0.135)
    a2.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fig_F2_synthetic.png")
    fig.savefig(out, dpi=220, bbox_inches="tight")
    print(f"\nsaved figure: {out}")
except ImportError:
    print("\n(matplotlib not installed — skipped figure; table above is the result)")
