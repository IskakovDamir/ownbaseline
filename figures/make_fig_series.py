#!/usr/bin/env python3
"""
make_fig_series.py: Figure 3 of the preprint, conditional skill and direction on the three
staged series. Reads experiments/conditional_null/results/{conditional_null,summary}.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import make_paper_a_figures as MPA  # noqa: E402

RES = REPO / "experiments/conditional_null/results"
SERIES = [("GSE106474", "o", "zebrafish, Drop-seq, audited"), ("GSE112294", "s", "zebrafish, inDrops, unseen"),
          ("GSE113074", "^", "Xenopus, inDrops")]
ROWS = ["CytoTRACE", "SR", "ORIGINS", "dpath", "SLICE", "CCAT", "MCE"]
MARG = ["gene count", "gene-count signature", "CytoTRACE", "PCC(x, degree)", "SR", "CCAT"]
PT = 7.5
OFFSET = {"GSE106474": 0.24, "GSE112294": 0.0, "GSE113074": -0.24}


def main():
    d = json.load(open(RES / "conditional_null.json"))
    s = json.load(open(RES / "summary.json"))
    primary = {f["row"]: f["registered"]["linear"] for f in d["floors"] if "isotonic" in f["registered"]}
    fig, (a, b) = plt.subplots(1, 2, figsize=(MPA.TEXT_W_IN, 2.5), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.42})
    for i, row in enumerate(ROWS):
        y = len(ROWS) - 1 - i
        for acc, marker, _ in SERIES:
            v = primary.get(f"{acc} {row}")
            if v is None:
                continue
            a.plot(v["value"], y + OFFSET[acc], marker, ms=4.6, mfc=MPA.CLEARS if v["clears"] else "white",
                   mec=MPA.CLEARS if v["clears"] else "#666666", mew=0.8, zorder=3)
    a.axvline(0, color="#555555", lw=0.6, zorder=1)
    a.set_yticks(range(len(ROWS)))
    a.set_yticklabels(ROWS[::-1], fontsize=PT)
    a.set_ylim(-0.6, len(ROWS) - 0.4)
    a.set_xlim(-0.6, 0.95)
    a.set_xlabel(r"conditional skill on the declared primitive ($\tau_b$)", fontsize=PT)
    a.set_title("(a) skill beyond the primitive", fontsize=PT + 0.5, loc="left", pad=3)
    for i, row in enumerate(MARG):
        y = len(MARG) - 1 - i
        for acc, marker, _ in SERIES:
            v = s["marginals"][acc].get(row)
            if v is not None:
                b.plot(v, y + OFFSET[acc], marker, ms=4.6, mfc="#444444", mec="#444444", zorder=3)
    b.axvline(0, color="#555555", lw=0.6, zorder=1)
    b.set_yticks(range(len(MARG)))
    b.set_yticklabels(MARG[::-1], fontsize=PT)
    b.set_ylim(-0.6, len(MARG) - 0.4)
    b.set_xlim(-1.05, 1.05)
    b.set_xlabel("Spearman with stage, early to late", fontsize=PT)
    b.set_title("(b) direction", fontsize=PT + 0.5, loc="left", pad=3)
    for ax in (a, b):
        ax.tick_params(labelsize=PT, length=2, width=0.5)
        ax.grid(axis="y", color="#e6e6e6", lw=0.5, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_linewidth(0.5)
    handles = [Line2D([], [], marker=m, ls="", ms=4.6, mfc="#444444", mec="#444444", label=f"{acc}, {t}") for acc, m, t in SERIES]
    handles.append(Line2D([], [], marker="o", ls="", ms=4.6, mfc=MPA.CLEARS, mec=MPA.CLEARS, label="above its permutation floor"))
    handles.append(Line2D([], [], marker="o", ls="", ms=4.6, mfc="white", mec="#666666", label="not above"))
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=PT - 0.5, frameon=False, bbox_to_anchor=(0.5, -0.2), handletextpad=0.2, columnspacing=1.2)
    fig.savefig(HERE / "fig_series.pdf", bbox_inches="tight")
    print("wrote", HERE / "fig_series.pdf")


if __name__ == "__main__":
    main()
