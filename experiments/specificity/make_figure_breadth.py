#!/usr/bin/env python3
"""
make_figure_breadth.py: the five staged series in one figure.

    python3 experiments/specificity/make_figure_breadth.py --out <dir>

(a) The conditional skill of 1,000 random 200-gene sets after four primitives are removed, per
    series, with the scores of the audit on the same cells. (b) CytoTRACE's Spearman correlation
    with stage at raw depth and with every cell drawn down to 1,000 counts.
Reads only result files under results/; computes nothing new.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent / "results"
SERIES = [("GSE106474", "specificity", "zebrafish\n3 to 12 h"), ("GSE112294", "specificity", "zebrafish\n4 to 24 h"),
          ("GSE113074", "specificity", "Xenopus\nten stages"), ("GSE223922", "breadth", "zebrafish\n14 to 120 h"),
          ("GSE169210", "breadth", "mouse\nE6.5 to E8.1")]
SCORES = [("CytoTRACE", "#1a7f37", "o"), ("SR", "#0969da", "s"), ("CCAT", "#bc4c00", "^"), ("MCE", "#8250df", "D")]
LABEL = {"SR": "signalling entropy"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 3.5), gridspec_kw={"width_ratios": [1.55, 1]})
    for i, (geo, kind, name) in enumerate(SERIES):
        d = json.loads((R / f"{kind}_{geo}.json").read_text())
        ctl = np.load(R / f"{kind}_{geo}_control_values.npz")["full|a1_g200|joint"]
        v = ax.violinplot(ctl, positions=[i], widths=0.8, showextrema=False)
        for b in v["bodies"]:
            b.set_facecolor("#b8bcc2"); b.set_edgecolor("none"); b.set_alpha(0.9)
        q = np.quantile(ctl, 0.975)
        ax.plot([i - 0.4, i + 0.4], [q, q], color="#24292f", lw=1.1, ls=(0, (3, 2)))
        for j, (s, col, mk) in enumerate(SCORES):
            if s in d["rows"]:
                ax.scatter(i - 0.27 + 0.18 * j, d["rows"][s]["conditional_skill_joint"], color=col, marker=mk, s=34, zorder=3,
                           edgecolor="white", linewidth=0.6, label=LABEL.get(s, s) if i == 0 else None)
        dp = d["depth"]
        raw = dp.get("spearman_cytotrace_with_stage", d["rows"]["CytoTRACE"]["spearman_with_stage"])
        thin = dp["thinned"].get("1000", {}).get("spearman_cytotrace_with_stage")
        bx.scatter(raw, i, facecolor="white", edgecolor="#1a7f37", s=46, zorder=3, label="raw depth" if i == 0 else None)
        if thin is not None:
            bx.annotate("", xy=(thin, i), xytext=(raw, i), arrowprops=dict(arrowstyle="-|>", color="#57606a", lw=1.0, shrinkA=4, shrinkB=4))
            bx.scatter(thin, i, color="#1a7f37", s=46, zorder=3, label="1,000 counts per cell" if i == 0 else None)
    ax.axhline(0, color="#57606a", lw=0.6)
    ax.set_xticks(range(len(SERIES))); ax.set_xticklabels([n for _, _, n in SERIES], fontsize=8)
    ax.set_ylabel("conditional skill, four primitives removed", fontsize=8.5)
    ax.set_title("(a) scores against 1,000 random 200-gene sets", fontsize=9.5, loc="left")
    ax.plot([], [], color="#24292f", lw=1.1, ls=(0, (3, 2)), label="random sets, 97.5th percentile")
    ax.set_ylim(-0.3, 0.47)
    ax.legend(fontsize=7.5, frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.2), columnspacing=1.0, handletextpad=0.3)
    bx.axvline(0, color="#57606a", lw=0.6)
    bx.set_yticks(range(len(SERIES))); bx.set_yticklabels([n.replace("\n", ", ") for _, _, n in SERIES], fontsize=8)
    bx.invert_yaxis(); bx.set_xlim(-1.05, 1.05)
    bx.set_xlabel("CytoTRACE, Spearman correlation with stage", fontsize=8.5)
    bx.set_title("(b) direction and sequencing depth", fontsize=9.5, loc="left")
    bx.legend(fontsize=7.5, frameon=False, loc="center right", bbox_to_anchor=(1.0, 0.5))
    for x in (ax, bx):
        x.spines[["top", "right"]].set_visible(False); x.tick_params(labelsize=8)
    fig.tight_layout()
    fig.savefig(Path(a.out) / "fig_five_series.pdf")


if __name__ == "__main__":
    main()
