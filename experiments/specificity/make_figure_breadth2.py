#!/usr/bin/env python3
"""
make_figure_breadth2.py: two further figures from the result files; computes nothing new.

    python3 experiments/specificity/make_figure_breadth2.py --out <dir>

fig_floor_sweep.pdf   per series, the share of 1,000 random 200-gene sets whose conditional skill
                      exceeds a threshold, as the threshold is swept, one curve per primitive;
                      ticks mark the median permutation floor.
fig_survival.pdf      (a) random gene sets before and after CytoTRACE's smoothing, with CytoTRACE
                      and its gene-count signature; (b) which score passes which test on which series.
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = Path(__file__).resolve().parent / "results"
SERIES = [("GSE106474", "specificity", "zebrafish, 3 to 12 h"), ("GSE112294", "specificity", "zebrafish, 4 to 24 h"),
          ("GSE113074", "specificity", "Xenopus, ten stages"), ("GSE223922", "breadth", "zebrafish, 14 to 120 h"),
          ("GSE169210", "breadth", "mouse, E6.5 to E8.1")]
PRIM = [("detected_genes", "gene count", "#1a7f37"), ("pcc_x_degree", "PCC(x, degree)", "#0969da"),
        ("shannon_entropy", "Shannon entropy", "#bc4c00"), ("log10_library_size", "library size", "#8250df")]
SCORES = [("CytoTRACE", "CytoTRACE"), ("SR", "signalling entropy"), ("CCAT", "CCAT"), ("MCE", "MCE")]


def load(geo, kind):
    return json.loads((R / f"{kind}_{geo}.json").read_text()), np.load(R / f"{kind}_{geo}_control_values.npz")


def sweep(out):
    fig, axes = plt.subplots(1, 5, figsize=(11.2, 2.5), sharey=True)
    x = np.linspace(-0.1, 0.5, 241)
    for ax, (geo, kind, name) in zip(axes, SERIES):
        d, z = load(geo, kind)
        for key, lab, col in PRIM:
            cs = z[f"full|a1_g200|{key}"]
            ax.plot(x, [(cs > t).mean() for t in x], color=col, lw=1.3, label=lab)
            fl = d["controls_permutation_floor"]["by_primitive"][key]["floor_quantiles"]["0.5"]
            ax.plot([fl], [(cs > fl).mean()], marker="|", color=col, ms=9, mew=1.6)
        ax.axhline(0.025, color="#57606a", lw=0.7, ls=(0, (3, 2)))
        ax.set_title(name, fontsize=8.5); ax.set_xlim(-0.1, 0.5); ax.set_ylim(0, 1.02)
        ax.set_xlabel("threshold on conditional skill", fontsize=7.5); ax.tick_params(labelsize=7.5)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("share of random sets above", fontsize=8)
    axes[0].legend(fontsize=6.8, frameon=False, loc="upper right")
    fig.tight_layout(); fig.savefig(Path(out) / "fig_floor_sweep.pdf"); plt.close(fig)


def survival(out):
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(10.2, 3.2), gridspec_kw={"width_ratios": [1, 1.25]})
    pos = 0
    ticks = []
    for geo, name in (("GSE106474", "zebrafish\n3 to 12 h"), ("GSE112294", "zebrafish\n4 to 24 h")):
        d, z = load(geo, "specificity")
        for j, (key, col) in enumerate((("a3_unsmoothed", "#b8bcc2"), ("a3", "#6e7781"))):
            v = ax.violinplot(z[f"full|{key}|detected_genes"], positions=[pos + j], widths=0.8, showextrema=False)
            for b in v["bodies"]:
                b.set_facecolor(col); b.set_edgecolor("none"); b.set_alpha(0.95)
        ax.scatter(pos, d["rows"]["GCS"]["conditional_skill"], color="#1a7f37", marker="o", s=38, zorder=3, facecolor="white",
                   label="gene-count signature, unsmoothed" if pos == 0 else None)
        ax.scatter(pos + 1, d["rows"]["CytoTRACE"]["conditional_skill"], color="#1a7f37", marker="o", s=38, zorder=3,
                   label="CytoTRACE" if pos == 0 else None)
        ticks += [(pos, "before\nsmoothing"), (pos + 1, "after")]
        ax.text(pos + 0.5, 0.97, name, ha="center", va="top", fontsize=8, transform=ax.get_xaxis_transform())
        pos += 2.6
    ax.axhline(0, color="#57606a", lw=0.6)
    ax.set_xticks([t for t, _ in ticks]); ax.set_xticklabels([l for _, l in ticks], fontsize=7.5)
    ax.set_ylabel("conditional skill on gene count", fontsize=8.5); ax.set_ylim(-0.42, 1.08)
    ax.set_title("(a) 100 random gene sets through CytoTRACE's pipeline", fontsize=9.5, loc="left")
    ax.legend(fontsize=7.5, frameon=False, loc="lower center", ncol=2)
    for i, (geo, kind, name) in enumerate(SERIES):
        d, _ = load(geo, kind)
        for j, (s, _) in enumerate(SCORES):
            r = d["rows"].get(s)
            if r is None:
                bx.text(i, j, "not scored", ha="center", va="center", fontsize=6.5, color="#8c959f"); continue
            joint = r["vs_controls"]["a1_g200"]["joint_exceeds_p975"]
            bx.scatter(i - 0.17, j, s=95, marker="s", color="#1a7f37" if joint else "white", edgecolor="#1a7f37" if joint else "#8c959f", linewidth=1.1)
            t = r.get("sample_test")
            if t is None:
                bx.text(i + 0.17, j, "n/a", ha="center", va="center", fontsize=6.5, color="#8c959f")
            else:
                ok = t["p_two_sided"] < 0.05 and np.sign(t["S"]) == np.sign(r["conditional_skill"])
                bx.scatter(i + 0.17, j, s=95, marker="o", color="#0969da" if ok else "white", edgecolor="#0969da" if ok else "#8c959f", linewidth=1.1)
    bx.set_xticks(range(5)); bx.set_xticklabels([n.replace(", ", "\n") for _, _, n in SERIES], fontsize=7.5)
    bx.set_yticks(range(4)); bx.set_yticklabels([l for _, l in SCORES], fontsize=8.5); bx.invert_yaxis()
    bx.set_xlim(-0.6, 4.6); bx.set_ylim(3.6, -0.9)
    bx.scatter([], [], s=70, marker="s", color="#1a7f37", label="above random sets, four primitives removed")
    bx.scatter([], [], s=70, marker="o", color="#0969da", label="residual trend supported over units")
    bx.legend(fontsize=7.3, frameon=False, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02), columnspacing=1.0, handletextpad=0.2)
    bx.set_title("(b) which result survives where", fontsize=9.5, loc="left", pad=14)
    for x in (ax, bx):
        x.spines[["top", "right"]].set_visible(False)
    bx.spines[["left", "bottom"]].set_visible(False); bx.tick_params(length=0)
    fig.tight_layout(); fig.savefig(Path(out) / "fig_survival.pdf"); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sweep(a.out); survival(a.out)
