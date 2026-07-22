"""
Figure F1 — Test A: transfer error-model vs own-entropy baseline (paper §4.3).

Two systems (lung PC9, ovarian Kuramochi); target-wise Δ = transfer − own-entropy.
The tautology diagnostic: Δ ≈ 0 in both directions means the apparent transfer
is the shared entropy → Bayes-error component, not conserved biology.

Numbers are hardcoded from the paper's canonical pipeline run
(SSOT `05-writing/00-MASTER-idiosyncrasy-paper.md` §3 "Test A" block).
The CIs on the ovarian direction are cell-level (SSOT open item: sister-grouped
resampling requires a sisters vector on 9394 cells; the Δ ≈ 0 verdict is
unaffected). The lung direction CI is cluster-bootstrap-consistent.

Camera-ready fix (per figures.md flag): legend was overlapping the "chance"
line in the earlier draft — legend is now upper-left, chance line label
in-panel, and the two panels use distinct colors so the legend is unambiguous.
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --- Numbers from SSOT §3 (Test A block) ---
# {target: (transfer_auc_range, own_entropy_auc, delta, ci_lo, ci_hi)}
DATA = {
    "Ovarian target\n(lung-trained -> ovarian)": {
        "transfer_low": 0.763,   # ent+mix subset
        "transfer_high": 0.774,  # ALL 4 features
        "own_entropy": 0.770,
        "delta": -0.006,
        "ci_lo": -0.009,
        "ci_hi": -0.003,
        "note": "cell-level CI\n(sister-grouped\nblocked, camera-ready)",
    },
    "Lung target\n(ovarian-trained -> lung)": {
        "transfer_low": 0.701,
        "transfer_high": 0.707,
        "own_entropy": 0.705,
        "delta": +0.001,
        "ci_lo": -0.001,
        "ci_hi": +0.004,
        "note": "excludes 0: no",
    },
}


def plot_fig_F1(save_path: str):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.2, 4.2), gridspec_kw={"width_ratios": [1.35, 1]})

    targets = list(DATA.keys())
    x = np.arange(len(targets))
    w = 0.35

    # PANEL A: AUROC bars — transfer (mean over subsets) vs own-entropy
    transfer_means = [np.mean([DATA[t]["transfer_low"], DATA[t]["transfer_high"]]) for t in targets]
    transfer_lows = [DATA[t]["transfer_low"] for t in targets]
    transfer_highs = [DATA[t]["transfer_high"] for t in targets]
    own = [DATA[t]["own_entropy"] for t in targets]

    # transfer bar: mean, whiskers = subset range
    yerr = np.array([[m - lo for m, lo in zip(transfer_means, transfer_lows)],
                     [hi - m for m, hi in zip(transfer_means, transfer_highs)]])
    a1.bar(x - w/2, transfer_means, width=w, yerr=yerr,
           color="#2c3e50", ecolor="#2c3e50", capsize=4, label="Transferred error model")
    a1.bar(x + w/2, own, width=w, color="#16a085", label="Own-entropy baseline")

    a1.axhline(0.5, ls="--", lw=1.0, color="#888", zorder=0)
    a1.text(len(targets) - 0.5, 0.505, "chance", fontsize=8, color="#666",
            ha="right", va="bottom")

    a1.set_ylim(0.48, 0.82)
    a1.set_ylabel("AUROC")
    a1.set_xticks(x)
    a1.set_xticklabels(targets, fontsize=9)
    a1.set_title("Test A: transfer looks meaningful but does not\nexceed the target's own within-domain entropy", fontsize=10)
    # legend upper-left, tight — no overlap with chance line (fig.md flag fixed)
    a1.legend(fontsize=8, loc="upper left", frameon=False)
    a1.spines[["top", "right"]].set_visible(False)

    # PANEL B: Δ with CI, per target
    deltas = [DATA[t]["delta"] for t in targets]
    ci_los = [DATA[t]["ci_lo"] for t in targets]
    ci_his = [DATA[t]["ci_hi"] for t in targets]
    yerr_delta = np.array([[d - lo for d, lo in zip(deltas, ci_los)],
                           [hi - d for d, hi in zip(deltas, ci_his)]])
    colors = ["#c0392b", "#8e44ad"]
    a2.errorbar(x, deltas, yerr=yerr_delta, fmt="o", markersize=9,
                capsize=6, capthick=1.5, elinewidth=1.5,
                color="#c0392b", ecolor="#c0392b", label="Δ (transfer − own-entropy)")
    a2.axhline(0, ls="--", lw=1.2, color="k", zorder=0)

    # Annotate ±0.01 "in the noise" band as a light rectangle
    a2.axhspan(-0.01, 0.01, alpha=0.10, color="#888", zorder=0, label="±0.01 (in the noise)")

    a2.set_xticks(x)
    a2.set_xticklabels(targets, fontsize=9)
    a2.set_ylim(-0.014, 0.008)
    a2.set_ylabel("Δ = transfer − own-entropy")
    a2.set_title("Δ ≈ 0 in both directions\n→ tautological transfer, not conserved structure", fontsize=10)
    a2.legend(fontsize=8, loc="upper right", frameon=False)
    a2.spines[["top", "right"]].set_visible(False)

    # Annotate Δ values on the panel
    for xi, d in zip(x, deltas):
        sign = "+" if d > 0 else ""
        a2.annotate(f"Δ = {sign}{d:.3f}", (xi, d), xytext=(6, 6),
                    textcoords="offset points", fontsize=9, color="#c0392b")

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    plot_fig_F1(os.path.join(here, "fig_F1_testA.png"))
