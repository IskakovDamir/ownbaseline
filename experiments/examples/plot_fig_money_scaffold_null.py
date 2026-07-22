"""
Figure 3 (money) — scaffold-randomization is a misleading control (paper §5).

For each of three potency scores (SCENT SR, CCAT, CytoTRACE gene-counts H0):
  - real cross-cancer transfer (mean over the two directions)
  - target's own within-cancer baseline (mean over the two targets) — the arbiter
  - scaffold-null v1 (shared randomized scaffold across two systems)
  - scaffold-null v2 (independent randomized scaffolds)
  - chance = 0.5

Punchline (contribution v). SR gap over its scaffold-null (0.120) is TWICE
CCAT's (0.057), but the own-baseline shows both are equally redundant
(Δ ≈ 0). Naive "gap above scaffold-null → genuine" would rank SR the more
genuine of the two — exactly wrong. Own-baseline (Test A) is the reliable
arbiter; scaffold-randomization can be misleading.

Numbers hardcoded from SSOT `05-writing/00-MASTER-idiosyncrasy-paper.md` §3
"POSITIVE CONTROL — scaffold-anchored статистики" block.

Camera-ready fixes (per figures.md flag):
  - gap-label spacing (0.057 label was jammed against chance line) — labels
    now placed above their bars with per-bar padding
  - honest caption: y is not truncated below chance (we plot 0.40 → 0.62 so
    the SR-null-below-chance point is visible and unambiguous)
  - gene_counts as third group for contrast (Δ < 0 there — not even
    apparent transfer)
"""
from __future__ import annotations
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


# --- Numbers from SSOT §3 (POSITIVE CONTROL block) ---
# real transfer = mean over W->R and R->W (SSOT reports 0.557 as the pooled/decision-rule number for SR and CCAT)
# own_baseline_mean = mean of the two directions' own-baselines
# scaffold_null_v1 = shared random scaffold across two systems (± SE, SSOT)
# scaffold_null_v2 = independent random scaffolds (± SE, SSOT)
SCORES = {
    "SCENT SR":       {"real": 0.557, "own": 0.554, "null_v1": 0.437, "null_v1_se": 0.005,
                       "null_v2": 0.438, "null_v2_se": 0.003,
                       "delta_cell4_lo": 0.001, "delta_cell4_hi": 0.006},
    "CCAT":           {"real": 0.557, "own": 0.552, "null_v1": 0.500, "null_v1_se": 0.018,
                       "null_v2": 0.500, "null_v2_se": 0.020,
                       "delta_cell4_lo": 0.002, "delta_cell4_hi": 0.008},
    "CytoTRACE v1\n(gene_counts, H₀)": {"real": 0.434, "own": 0.564,  # mean of 0.409/0.458 and 0.590/0.538
                                       "null_v1": None, "null_v1_se": 0.0,
                                       "null_v2": None, "null_v2_se": 0.0,
                                       "delta_cell4_lo": -0.182, "delta_cell4_hi": -0.080},
}


def plot_money(save_path: str):
    fig, ax = plt.subplots(figsize=(11.0, 5.2))

    scores = list(SCORES.keys())
    x = np.arange(len(scores))
    w = 0.20  # narrow bars — 4 groups per score

    colors = {
        "real":    "#2c3e50",  # navy — actual transfer
        "own":     "#16a085",  # teal — the arbiter
        "null_v1": "#c0392b",  # red — scaffold-null v1 (shared random)
        "null_v2": "#e67e22",  # orange — scaffold-null v2 (independent random)
    }

    # Build 4 bars per score group
    reals = [SCORES[s]["real"] for s in scores]
    owns  = [SCORES[s]["own"]  for s in scores]
    n1s   = [SCORES[s]["null_v1"] if SCORES[s]["null_v1"] is not None else np.nan for s in scores]
    n2s   = [SCORES[s]["null_v2"] if SCORES[s]["null_v2"] is not None else np.nan for s in scores]
    n1_se = [SCORES[s]["null_v1_se"] for s in scores]
    n2_se = [SCORES[s]["null_v2_se"] for s in scores]

    ax.bar(x - 1.5*w, reals, width=w, color=colors["real"],  label="Real cross-cancer transfer")
    ax.bar(x - 0.5*w, owns,  width=w, color=colors["own"],   label="Target's own within-cancer baseline")
    b3 = ax.bar(x + 0.5*w, n1s,   width=w, yerr=n1_se, capsize=3,
                color=colors["null_v1"], ecolor=colors["null_v1"], label="Scaffold-null v1 (shared random)")
    b4 = ax.bar(x + 1.5*w, n2s,   width=w, yerr=n2_se, capsize=3,
                color=colors["null_v2"], ecolor=colors["null_v2"], label="Scaffold-null v2 (independent)")

    # Chance line
    ax.axhline(0.5, ls="--", lw=1.2, color="#666", zorder=0)
    ax.text(len(scores) - 0.4, 0.505, "chance", fontsize=9, color="#555",
            ha="right", va="bottom")

    # Δ ≈ 0 annotation between real & own bars for SR and CCAT
    for xi, sn in zip(x, scores):
        s = SCORES[sn]
        if abs(s["real"] - s["own"]) < 0.02 and s["real"] > 0.5:
            # Δ ≈ 0 — put a note between the two bars
            ymid = max(s["real"], s["own"]) + 0.008
            ax.annotate("Δ ≈ 0\n(own-baseline)", (xi - w, ymid), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8,
                        color="#333")

    # Gap labels: annotated in a callout at the RIGHT side of each score group,
    # BETWEEN real bar and null bar — with a vertical line/bracket so the
    # visual message "gap = real - null" is unambiguous (fig.md flag fixed).
    for xi, sn in zip(x, scores):
        s = SCORES[sn]
        if s["null_v1"] is not None:
            gap = s["real"] - s["null_v1"]
            # Position the label ON THE SIDE of the score group, at midheight,
            # NOT inside any bar
            y_mid = (s["real"] + s["null_v1"]) / 2
            x_side = xi + 1.5*w + 0.04  # to the right of the last bar
            ax.annotate(f"gap = {gap:+.3f}",
                        (x_side, y_mid),
                        xytext=(0, 0), textcoords="offset points",
                        ha="left", va="center", fontsize=8.5,
                        color=colors["null_v1"], fontstyle="italic",
                        fontweight="bold")
            # Small connector line indicating the gap span (visual bracket)
            ax.plot([x_side - 0.02, x_side - 0.02], [s["null_v1"], s["real"]],
                    color=colors["null_v1"], lw=1.0, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(scores, fontsize=10)
    ax.set_ylabel("Balanced AUROC")
    ax.set_ylim(0.38, 0.66)
    ax.set_xlim(-0.6, len(scores) - 0.05)
    ax.set_title(
        "Scaffold-randomization is a MISLEADING control; own-baseline is the arbiter\n"
        "SR's gap over scaffold-null (0.120) is 2× CCAT's (0.057), yet both give Δ ≈ 0 vs own-baseline",
        fontsize=10.5
    )
    # Legend OUTSIDE the plot area (upper right of the figure), so it never overlaps bars/labels
    ax.legend(fontsize=8.5, loc="upper right",
              bbox_to_anchor=(1.0, 1.0), frameon=False, ncol=1)
    ax.spines[["top", "right"]].set_visible(False)

    # Caption note in figure area (small, gray)
    ax.text(0.5, -0.14,
            "y-axis truncated at 0.38 to show SR-scaffold-null (0.437) below chance without stretching. "
            "gene_counts is included for contrast: it does not even give apparent transfer (real = 0.434, below chance) "
            "so scaffold-null is not defined for it in the same sense.",
            transform=ax.transAxes, ha="center", va="top",
            fontsize=8, color="#555", wrap=True)

    fig.tight_layout()
    fig.savefig(save_path, dpi=220, bbox_inches="tight")
    print(f"saved: {save_path}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    plot_money(os.path.join(here, "fig_money_scaffold_null_v2.png"))
