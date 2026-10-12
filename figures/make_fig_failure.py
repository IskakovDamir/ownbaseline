#!/usr/bin/env python3
"""
make_fig_failure.py: the failure case of section 1, as one full-width figure.

    python3 figures/make_fig_failure.py

Writes figures/fig1_failure.pdf (\\begin{figure*}, 6.50 in), plus
figures/fig1_failure_values.json and figures/fig1_failure_caption.txt.

(a) Simulation. Depth d (the primitive) and a biological position b are
    independent standard normals; the developmental ordinal is the
    equal-occupancy 12-level binning of z = 0.6 d + 0.8 b, so depth tracks the
    ordering, which is the case where a depth-restating score looks validated.
    Score A restates depth: A = d + 0.3 e_A. Score B carries ordering beyond
    depth: B = d + 0.5 b + s e_B, with the noise s chosen by bisection so that
    B's marginal Kendall tau_b with the ordinal equals A's. n = 3,000, the
    seed is fixed before anything is drawn. Every statistic is the paper's
    estimator: marginal tau_b, Spearman rho of score and primitive, and
    conditional skill (own_baseline.conditional_skill._unit, 1,000 bootstrap
    resamples, seed 42), each with its own floor (own_baseline.floors.floor).
(b, c) Real tissue, read from the run record by the same loader the earlier
    depth figure used (make_paper_a_figures._fig3_load): median detected genes
    in the top and the bottom population of sorted haematopoietic progenitors
    (GSE117498) and of intestinal epithelium (GSE125970), with the gene-count
    primitive's AUROC between the top and the bottom potency tier. Depth runs
    against potency in (b) and with it in (c).

Panels (b, c) replace the column panel this script drew on 2026-10-12 (CCAT
against signalling entropy on the zebrafish stages): both of those scores order
the stages opposite to their authors' polarity (experiments/w6_direction), so
a panel meant to show the failure case would itself depend on how a score's
direction is read.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from own_baseline.conditional_skill import _unit, align  # noqa: E402
from own_baseline import floors as F  # noqa: E402
import make_paper_a_figures as MPA  # noqa: E402

TEXT_W_IN = MPA.TEXT_W_IN
SEED = 20261012          # fixed before drawing: the date the figure was added
N, LEVELS = 3000, 12
CLEARS, FAILS, FLOORC, PRIMC = MPA.CLEARS, MPA.FAILS, MPA.FLOORC, "#1f4e79"


def binned(z, levels):
    q = np.quantile(z, np.linspace(0, 1, levels + 1)[1:-1])
    return np.searchsorted(q, z).astype(float)


def summarize(name, s, p, o):
    """Marginal tau_b, rho with the primitive, conditional tau_b with CI and floor."""
    sa = align(s, o)
    _, res = _unit((name, sa, [p], o, (300, 1000)))
    cs = res["kang_wdm_taub"]
    rho = float(spearmanr(s, p).statistic)
    fl = F.floor(len(s), rho, "kendalltau", 1, LEVELS)
    return {"marginal_taub": float(kendalltau(s, o).statistic), "rho_primitive": rho,
            "conditional_taub": cs["tau"], "conditional_CI95": cs["CI95"],
            "floor": fl["floor"], "floor_cell": fl["cell"],
            "clears": bool(cs["tau"] > fl["floor"])}


def toy():
    rng = np.random.default_rng(SEED)
    d, b = rng.standard_normal(N), rng.standard_normal(N)
    eA, eB = rng.standard_normal(N), rng.standard_normal(N)
    o = binned(0.6 * d + 0.8 * b, LEVELS)
    A = d + 0.3 * eA
    target = float(kendalltau(A, o).statistic)
    lo, hi = 0.0, 5.0                   # B's marginal tau_b falls as its noise grows
    for _ in range(60):
        s = (lo + hi) / 2
        t = float(kendalltau(d + 0.5 * b + s * eB, o).statistic)
        lo, hi = (s, hi) if t > target else (lo, s)
    s = (lo + hi) / 2
    B = d + 0.5 * b + s * eB
    return {"model": "d, b, e ~ N(0,1) iid; ordinal = 12 equal-occupancy bins of "
                     "0.6 d + 0.8 b; A = d + 0.3 e_A; B = d + 0.5 b + s e_B",
            "n": N, "seed": SEED, "noise_s_for_B": s,
            "primitive_marginal_taub": float(kendalltau(d, o).statistic),
            "A": summarize("A", A, d, o), "B": summarize("B", B, d, o)}


def tissue():
    panels, missing = MPA._fig3_load()
    if missing or len(panels) != 2:
        raise SystemExit(f"depth panels unavailable: {missing}")
    out = []
    for spec, g in panels:
        out.append({"title": spec["title"], "artefact": spec["art"], "sources": spec["sources"],
                    "hi": spec["hi"][0], "lo": spec["lo"][0], "color": spec["color"],
                    "hi_genes": g["hi_genes"], "lo_genes": g["lo_genes"],
                    "auroc_gene_count_top_vs_bottom": g["auroc"],
                    "ct_marginal_gap": g["delta"], "n_ranked": g["n"]})
    return out


def draw_toy(ax, pair, labels, prim_tau, pt):
    import matplotlib.transforms as mtrans
    groups = ["with ordinal", "with primitive", "beyond primitive"]
    keys = ["marginal_taub", "rho_primitive", "conditional_taub"]
    w = 0.36
    ticks, ticklabels = [], []
    for g, k in enumerate(keys):
        for j, (lab, rec) in enumerate(zip(labels, pair)):
            x = g + (j - 0.5) * w
            v = rec[k]
            color = CLEARS if rec["clears"] else FAILS
            ax.bar(x, v, width=w * 0.92, color=color, edgecolor="#333", lw=0.4, zorder=3)
            if k != "conditional_taub" and v > 0.25:
                ax.text(x, v * 0.3, f"{v:.2f}", ha="center", va="center", fontsize=pt,
                        color="white", zorder=4)
            elif k == "conditional_taub":
                # above the whisker and the floor, so neither strikes the digits
                top = max(v, rec["conditional_CI95"][1], rec["floor"])
                ax.text(x, top + 0.03, f"{v:+.3f}", ha="center", va="bottom", fontsize=pt,
                        zorder=7)
            else:
                ax.text(x, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=pt, zorder=4)
            if k == "conditional_taub":
                lo, hi = rec["conditional_CI95"]
                ax.plot([x, x], [lo, hi], color="#111", lw=0.9, zorder=5)
                ax.plot([x - w * 0.44, x + w * 0.44], [rec["floor"]] * 2, color=FLOORC,
                        lw=1.3, ls=(0, (2.0, 1.2)), zorder=6)
            ticks.append(x)
            ticklabels.append(lab)
    ax.plot([-0.40, 0.40], [prim_tau] * 2, color=PRIMC, lw=1.6, zorder=7)
    ax.axhline(0, color="#888", lw=0.7, zorder=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels(ticklabels, fontsize=pt)
    trans = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for g, name in enumerate(groups):
        ax.text(g, -0.115, name, transform=trans, ha="center", va="top", fontsize=pt)
    ax.tick_params(axis="y", labelsize=pt, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-0.1, 1.08)
    ax.set_ylabel(r"$\tau_b$ or $\rho$", fontsize=pt, labelpad=2)


def draw_tissue(ax, p, pt):
    vals = [p["hi_genes"], p["lo_genes"]]
    col = "#111"                        # neutral: green and red mean "clears" and "floor" in (a)
    ymax = max(vals) * 1.6
    ax.bar([0, 1], vals, 0.52, color=["#5f5f5f", "#c8c8c8"], edgecolor="#555", lw=0.5, zorder=3)
    for x, v in zip((0, 1), vals):
        ax.text(x, v + 0.02 * ymax, f"{v:,.0f}", ha="center", va="bottom", fontsize=pt, color="#333")
    lift = 0.17 * ymax
    ax.annotate("", xy=(1, vals[1] + lift), xytext=(0, vals[0] + lift),
                arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
    ax.text(0.5, 0.97, f"AUROC {p['auroc_gene_count_top_vs_bottom']:.3f}",
            transform=ax.transAxes, ha="center", va="top", fontsize=pt, color=col)
    ax.set_xticks([0, 1])
    short = {"HSC": "HSC", "GMP": "GMP", "Stem Cell": "stem", "Enterocyte": "enterocyte"}
    ax.set_xticklabels([short[p["hi"]], short[p["lo"]]], fontsize=pt)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.65, 1.65)
    ax.set_ylabel("median detected genes", fontsize=pt, labelpad=2)
    ax.tick_params(axis="y", labelsize=pt, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    t, tis = toy(), tissue()
    pt = 7.0
    W, H = TEXT_W_IN, 2.15
    fig = plt.figure(figsize=(W, H))
    bottom, height = 0.56 / H, 1.29 / H
    ax_a = fig.add_axes([0.42 / W, bottom, 2.85 / W, height])
    ax_b = fig.add_axes([3.86 / W, bottom, 1.05 / W, height])
    ax_c = fig.add_axes([5.42 / W, bottom, 1.05 / W, height])
    draw_toy(ax_a, [t["A"], t["B"]], ["A", "B"], t["primitive_marginal_taub"], pt)
    ax_a.set_title("(a) simulated: A restates depth, B adds to it", fontsize=pt + 0.5,
                   loc="left", pad=3)
    for ax, p, letter in ((ax_b, tis[0], "b"), (ax_c, tis[1], "c")):
        draw_tissue(ax, p, pt)
        name = "blood, sorted" if p["title"].startswith("Hemat") else "intestine"
        ax.set_title(f"({letter}) {name}", fontsize=pt + 0.5, loc="left", pad=3)
    handles = [mp.Patch(facecolor=CLEARS, edgecolor="#333", lw=0.4, label="clears its own floor"),
               mp.Patch(facecolor=FAILS, edgecolor="#333", lw=0.4, label="does not clear"),
               plt.Line2D([0], [0], color=FLOORC, lw=1.3, ls=(0, (2.0, 1.2)),
                          label="its own null floor"),
               plt.Line2D([0], [0], color="#111", lw=0.9, label="95% bootstrap CI"),
               plt.Line2D([0], [0], color=PRIMC, lw=1.6, label=r"depth's own $\tau_b$")]
    fig.legend(handles=handles, fontsize=pt, loc="lower left", ncol=5, frameon=False,
               bbox_to_anchor=(0.0, -0.01), handlelength=1.5, columnspacing=1.2,
               handletextpad=0.4)
    out = HERE / "fig1_failure.pdf"
    fig.savefig(out)
    plt.close(fig)

    vals = {"toy": t, "tissue": tis}
    (HERE / "fig1_failure_values.json").write_text(json.dumps(vals, indent=2) + "\n")
    A, B = t["A"], t["B"]
    hb, ic = tis
    cap = (f"(a) Simulated cells (n = {t['n']:,}) whose 12-level ordinal follows depth and an "
           f"independent position: A restates depth, B adds to it, both at tau_b "
           f"{A['marginal_taub']:+.3f} with the ordinal. Beyond depth A keeps "
           f"{A['conditional_taub']:+.3f} against a floor of {A['floor']:.4f} and B "
           f"{B['conditional_taub']:+.3f} against {B['floor']:.4f}. (b) {hb['hi']} "
           f"{hb['hi_genes']:,.1f} against {hb['lo']} {hb['lo_genes']:,.1f} median detected genes, "
           f"gene-count AUROC {hb['auroc_gene_count_top_vs_bottom']:.3f}. (c) {ic['hi']} "
           f"{ic['hi_genes']:,.1f} against {ic['lo']} {ic['lo_genes']:,.1f}, AUROC "
           f"{ic['auroc_gene_count_top_vs_bottom']:.3f}.")
    (HERE / "fig1_failure_caption.txt").write_text(cap + "\n")
    print(cap)
    print(f"toy: primitive tau_b {t['primitive_marginal_taub']:+.4f}, noise s {t['noise_s_for_B']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
