#!/usr/bin/env python3
"""
make_fig_failure.py: the failure case of section 1, as a figure: two scores
that the field's standard check (correlation with a gold standard) passes
alike, and the own-baseline test that tells them apart.

    python3 figures/make_fig_failure.py

Writes figures/fig1_failure.pdf (one column, 3.18 in, \\begin{figure}),
figures/fig1_failure_values.json and figures/fig1_failure_caption.txt.

(a) Simulation. Depth d (the primitive) and a biological position b are
    independent standard normals; the developmental ordinal is the
    equal-occupancy 12-level binning of z = 0.6 d + 0.8 b, so depth tracks the
    ordering, which is the case where a depth-restating score looks validated.
    Score A restates depth: A = d + 0.3 e_A. Score B carries ordering beyond
    depth: B = d + 0.5 b + s e_B, with the noise s chosen by bisection so that
    B's marginal Kendall tau_b with the ordinal equals A's. n = 3,000, the
    seed is fixed before anything is drawn.
(b) Zebrafish GSE106474, 39,505 cells, 12 stages, read from the run record:
    CCAT and signalling entropy (SR) against their shared primitive
    PCC(x, degree).

Every statistic is the paper's estimator: marginal tau_b of the score against
the ordinal, Spearman rho of score and primitive, and conditional skill, the
Kendall tau_b of the score's rank residual on the primitive
(own_baseline.conditional_skill._unit, 1,000 bootstrap resamples, seed 42).
Each conditional value is drawn with its own floor, the 97.5th percentile of
the estimator's measured null at its n and rho (own_baseline.floors.floor).
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
from own_baseline.conditional_skill import _unit, align  # noqa: E402
from own_baseline import floors as F  # noqa: E402

COLUMN_W_IN = 3.18
SEED = 20261012          # fixed before drawing: the date the figure was added
N, LEVELS = 3000, 12
CLEARS, FAILS, FLOORC, PRIMC = "#1a7f37", "#8a8a8a", "#b00000", "#1f4e79"
RUN_RECORD = REPO / "data" / "run_record" / "track2" / "gate2_conditional_skill.json"


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


def real():
    rec = json.loads(RUN_RECORD.read_text())
    out = {"source": "data/run_record/track2/gate2_conditional_skill.json",
           "n": rec["n_cells_full"],
           "primitive_marginal_taub": rec["marginal_skill"]["PCC(x,degree)"]["kang_wdm_taub"]}
    for name, key in (("CCAT", "CCAT"), ("SR", "SR")):
        cs = rec["conditional_skill"][f"{key} | PCC(x,degree)"]["kang_wdm_taub"]
        rho = rec["spearman_score_vs_primitive"][f"{key} | PCC(x,degree)"]
        fl = F.floor(rec["n_cells_full"], rho, "kendalltau", 1, LEVELS)
        out[name] = {"marginal_taub": rec["marginal_skill"][key]["kang_wdm_taub"],
                     "rho_primitive": rho, "conditional_taub": cs["tau"],
                     "conditional_CI95": cs["CI95"], "floor": fl["floor"],
                     "floor_cell": fl["cell"], "clears": bool(cs["tau"] > fl["floor"])}
    return out


def draw(ax, pair, labels, prim_tau, pt):
    """Three groups of two bars; each bar named on the axis, each group below."""
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
                ax.text(x, v * 0.3, f"{v:+.2f}", ha="center", va="center", fontsize=pt - 0.5,
                        color="white", zorder=4)
            else:
                ax.text(x, v + (0.03 if v >= 0 else -0.03), f"{v:+.2f}", ha="center",
                        va="bottom" if v >= 0 else "top", fontsize=pt - 0.5, zorder=4)
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
    ax.set_xticklabels(ticklabels, fontsize=pt - 0.5)
    trans = mtrans.blended_transform_factory(ax.transData, ax.transAxes)
    for g, name in enumerate(groups):
        ax.text(g, -0.17, name, transform=trans, ha="center", va="top", fontsize=pt - 0.5,
                linespacing=1.1)
    ax.tick_params(axis="y", labelsize=pt, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=1.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-0.55, 2.55)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    t, r = toy(), real()
    pt = 7.0
    fig, axes = plt.subplots(2, 1, figsize=(COLUMN_W_IN, 3.3))
    fig.subplots_adjust(left=0.135, right=0.985, top=0.93, bottom=0.225, hspace=0.72)
    draw(axes[0], [t["A"], t["B"]], ["A", "B"], t["primitive_marginal_taub"], pt)
    axes[0].set_ylim(-0.2, 1.15)
    axes[0].set_title("(a) simulated: A restates depth, B adds to it",
                      fontsize=pt + 0.5, loc="left", pad=3)
    draw(axes[1], [r["CCAT"], r["SR"]], ["CCAT", "SR"], r["primitive_marginal_taub"], pt)
    axes[1].set_ylim(-0.42, 1.15)
    axes[1].set_title("(b) zebrafish, 12 stages: CCAT and SR",
                      fontsize=pt + 0.5, loc="left", pad=3)
    for ax in axes:
        ax.set_ylabel(r"$\tau_b$ or $\rho$", fontsize=pt, labelpad=2)
    handles = [mp.Patch(facecolor=CLEARS, edgecolor="#333", lw=0.4,
                        label="clears its own floor"),
               mp.Patch(facecolor=FAILS, edgecolor="#333", lw=0.4,
                        label="does not clear"),
               plt.Line2D([0], [0], color=FLOORC, lw=1.3, ls=(0, (2.0, 1.2)),
                          label="its own null floor"),
               plt.Line2D([0], [0], color="#111", lw=0.9, label="95% bootstrap CI"),
               plt.Line2D([0], [0], color=PRIMC, lw=1.6,
                          label=r"primitive's $\tau_b$ with ordinal")]
    fig.legend(handles=handles, fontsize=pt, loc="lower left", ncol=2, frameon=False,
               bbox_to_anchor=(0.005, 0.0), handlelength=1.7, columnspacing=0.8,
               handletextpad=0.4, labelspacing=0.3)
    out = HERE / "fig1_failure.pdf"
    fig.savefig(out)
    plt.close(fig)

    vals = {"toy": t, "real": r}
    (HERE / "fig1_failure_values.json").write_text(json.dumps(vals, indent=2) + "\n")
    A, B, C, S = t["A"], t["B"], r["CCAT"], r["SR"]
    cap = (
        f"Two pairs of scores that a correlation check passes alike. (a) Simulated cells "
        f"(n = {t['n']:,}) whose 12-level ordinal follows depth: score A restates depth with "
        f"noise, score B adds the part of the ordering depth does not carry. Both correlate "
        f"with the ordinal at tau_b {A['marginal_taub']:+.3f} and {B['marginal_taub']:+.3f}. "
        f"After rank-residualizing on depth, A keeps {A['conditional_taub']:+.3f} against a "
        f"floor of {A['floor']:+.4f} and B keeps {B['conditional_taub']:+.3f} against "
        f"{B['floor']:+.4f}. (b) Zebrafish, {r['n']:,} cells: CCAT and signalling entropy (SR) "
        f"both correlate with the ordinal ({C['marginal_taub']:+.3f}, {S['marginal_taub']:+.3f}) "
        f"and with their primitive PCC(x, degree) (rho {C['rho_primitive']:.3f}, "
        f"{S['rho_primitive']:.3f}); beyond it CCAT leaves {C['conditional_taub']:+.3f} against "
        f"a floor of {C['floor']:+.4f} and SR {S['conditional_taub']:+.3f} against "
        f"{S['floor']:+.4f}. The blue tick is the primitive's own tau_b with the ordinal.")
    (HERE / "fig1_failure_caption.txt").write_text(cap + "\n")
    print(json.dumps({k: {kk: (round(vv, 4) if isinstance(vv, float) else vv)
                          for kk, vv in v.items() if kk in ("marginal_taub", "rho_primitive",
                                                            "conditional_taub", "floor", "clears")}
                      for k, v in (("A", A), ("B", B), ("CCAT", C), ("SR", S))}, indent=1))
    print(f"toy: primitive tau_b {t['primitive_marginal_taub']:+.4f}, noise s {t['noise_s_for_B']:.4f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
