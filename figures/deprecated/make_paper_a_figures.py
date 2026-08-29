"""Paper A figures — data-free, all numbers from the frozen benchmark + Track 2 decider."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mp
import numpy as np

C = {"deg": "#1a5fb4", "ent": "#9141ac", "cnt": "#c64600",
     "sub": "#1a7f37", "triv": "#8a8a8a", "pend": "#b08900", "unres": "#b00000"}


# ============ FIG 1 — reductions schematic ============
def make_fig1():
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, fc, tc, fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                     fc=fc, ec="#444", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, col):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                     arrowstyle="->,head_width=2.5,head_length=3", color=col, lw=1.0, mutation_scale=6))

    box(7.3, 4.4, 2.4, 1.0, "PRIMITIVE\nPearson(x, PPI degree)", "#eaf1fb", C["deg"], 7.4, True)
    box(7.3, 2.5, 2.4, 1.0, "PRIMITIVE\nShannon entropy H(x)", "#f3ecf7", C["ent"], 7.4, True)
    box(7.3, 0.6, 2.4, 1.0, "PRIMITIVE\ngene count = H0", "#fbeee6", C["cnt"], 7.4, True)

    ax.text(0.1, 5.6, "degree-correlation family", fontsize=7.6, color=C["deg"], fontweight="bold")
    for i, s in enumerate(["SR  (authors: R^2=0.96)", "CCAT  (= by definition)", "MCE  (authors admit)",
                           "NCG  (authors ~90%)", "ORIGINS  (+ Sx term)"]):
        yy = 5.15 - i * 0.42
        box(0.1, yy - 0.16, 3.4, 0.34, s, "#eaf1fb", C["deg"], 6.6)
        arrow(3.55, yy, 7.25, 4.9, C["deg"])

    ax.text(0.1, 2.95, "expression-entropy family", fontsize=7.6, color=C["ent"], fontweight="bold")
    for i, s in enumerate(["StemID", "cmEntropy (= by def)", "SLICE (fixed GO proj.)"]):
        yy = 2.5 - i * 0.42
        box(0.1, yy - 0.16, 3.4, 0.34, s, "#f3ecf7", C["ent"], 6.6)
        arrow(3.55, yy, 7.25, 3.0, C["ent"])

    ax.text(0.1, 0.98, "gene count", fontsize=7.6, color=C["cnt"], fontweight="bold")
    box(0.1, 0.4, 3.4, 0.34, "CytoTRACE (population)", "#fbeee6", C["cnt"], 6.6)
    arrow(3.55, 0.57, 7.25, 1.1, C["cnt"])

    ax.set_title('A field of "different" potency scores collapses to two primitives + gene count',
                 fontsize=8.6, pad=6)
    ax.text(5.0, -0.15, "own-baseline then tests whether each score adds ordering skill BEYOND its primitive (Fig 2, 4)",
            ha="center", fontsize=6.2, color="#555", style="italic")
    fig.savefig("fig1.png", dpi=150, bbox_inches="tight"); fig.savefig("fig1.pdf", bbox_inches="tight")
    plt.close(fig); print("fig1 ok")


# score: (name, pairwise rho, cond skill tau-b, verdict)
ROWS = [("CytoTRACE", 0.484, 0.653, "sub"), ("SR", 0.930, 0.430, "sub"),
        ("ORIGINS", 0.434, 0.402, "sub"), ("dpath", 0.477, 0.120, "sub"),
        ("CCAT", 0.998, -0.221, "triv"), ("SLICE", 0.932, -0.113, "triv"),
        ("StemID", 0.990, 0.000, "triv"), ("cmEntropy", 1.000, 0.000, "triv")]


# ============ FIG 2 — field benchmark spectrum ============
def make_fig2():
    bench = [("CytoTRACE", 0.653, "sub"), ("SR", 0.430, "sub"), ("ORIGINS", 0.402, "sub"),
             ("dpath", 0.120, "sub"), ("CCAT", -0.221, "triv"), ("SLICE", -0.113, "triv"),
             ("StemID", 0.0, "triv"), ("cmEntropy", 0.0, "triv"),
             ("MCE", None, "pend"), ("NCG", None, "pend"), ("SPIDE", None, "unres")]
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    y = np.arange(len(bench))[::-1]
    for yi, (name, sk, v) in zip(y, bench):
        if sk is None:
            ax.barh(yi, 0.02, color="none", edgecolor=C[v], hatch="///", linewidth=0.8)
            lbl = "own-baseline pending" if v == "pend" else "formula unresolved"
            ax.text(0.03, yi, lbl, va="center", fontsize=6.2, color=C[v], style="italic")
        else:
            ax.barh(yi, sk, color=C[v], edgecolor="#333", linewidth=0.5)
            ax.text(sk + (0.02 if sk >= 0 else -0.02), yi, f"{sk:+.2f}",
                    va="center", ha="left" if sk >= 0 else "right", fontsize=6.6)
    ax.set_yticks(y); ax.set_yticklabels([b[0] for b in bench], fontsize=7.4)
    ax.axvline(0, color="#888", lw=0.9)
    ax.set_xlabel("conditional skill beyond declared primitive (Kendall tau-b)")
    ax.set_title("Field audit on a clean fine ordinal (zebrafish, 12 stages)", fontsize=9)
    ax.set_xlim(-0.32, 0.80); ax.spines[["top", "right"]].set_visible(False)
    handles = [mp.Patch(color=C["sub"], label="substantive (adds real signal)"),
               mp.Patch(color=C["triv"], label="trivial (approx primitive)"),
               mp.Patch(facecolor="none", edgecolor=C["pend"], hatch="///", label="pending / unresolved")]
    ax.legend(handles=handles, fontsize=6.2, loc="lower right", frameon=False)
    fig.tight_layout(); fig.savefig("fig2.png", dpi=150); fig.savefig("fig2.pdf"); plt.close(fig)
    print("fig2 ok")


# ============ FIG 3 — depth confound ============
def make_fig3():
    fig = plt.figure(figsize=(4.9, 2.4))
    gs = fig.add_gridspec(1, 2, wspace=0.55, left=0.10, right=0.97, bottom=0.20, top=0.80)
    ax = fig.add_subplot(gs[0, 0])
    ax.text(-0.14, 1.06, "a", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.bar([0, 1], [916, 1404], 0.56, color=["#c8709a", "#b0b0b0"], edgecolor="#555", linewidth=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["HSC\n(potency high)", "GMP\n(potency low)"], fontsize=6.0)
    ax.set_ylabel("median detected genes"); ax.set_ylim(0, 1650)
    ax.annotate("", xy=(1, 1404), xytext=(0, 916), arrowprops=dict(arrowstyle="->", color="#b00000", lw=1.0))
    ax.text(0.5, 1150, "genes up as\npotency down", ha="center", fontsize=5.6, color="#b00000")
    ax.text(0.03, 0.97, "gene-count AUROC 0.378 (below chance)\nnaive 'score beats primitive':\nCT D = +0.141 reads as a WIN",
            transform=ax.transAxes, fontsize=5.3, va="top", color="#b00000")
    ax.set_title("Hematopoietic (sorted)", fontsize=7.2, pad=3)
    ax = fig.add_subplot(gs[0, 1])
    ax.text(-0.14, 1.06, "b", transform=ax.transAxes, fontsize=11, fontweight="bold")
    ax.bar([0, 1], [3263, 1862], 0.56, color=["#2f7f4f", "#b0b0b0"], edgecolor="#555", linewidth=0.5)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Stem\n(potency high)", "Enterocyte\n(potency low)"], fontsize=6.0)
    ax.set_ylabel("median detected genes"); ax.set_ylim(0, 3800)
    ax.annotate("", xy=(1, 1862), xytext=(0, 3263), arrowprops=dict(arrowstyle="->", color="#008000", lw=1.0))
    ax.text(0.5, 2700, "genes down as\npotency down", ha="center", fontsize=5.6, color="#008000")
    ax.text(0.03, 0.97, "gene-count AUROC 0.798\n(direction correct)", transform=ax.transAxes, fontsize=5.3, va="top", color="#008000")
    ax.set_title("Intestinal", fontsize=7.2, pad=3)
    fig.savefig("fig3.pdf"); fig.savefig("fig3.png"); plt.close(fig)
    print("fig3 ok")


# ============ FIG 4 — pairwise rho vs conditional skill ============
def make_fig4():
    fig, ax = plt.subplots(figsize=(4.4, 3.4))
    for name, rho, sk, v in ROWS:
        ax.scatter(rho, sk, s=70, color=C[v], edgecolor="#333", linewidth=0.6, zorder=3)
        ax.annotate(name, (rho, sk), textcoords="offset points",
                    xytext=(6, 6 if sk >= 0 else -12), fontsize=7.2)
    ax.axhline(0, color="#bbb", lw=0.8, ls="--", zorder=1)
    ax.add_patch(FancyArrowPatch((0.93, 0.43), (0.998, -0.22),
                 arrowstyle="-", color="#ccc", lw=0.8, mutation_scale=1))
    ax.text(0.60, 0.60, "high correlation,\nhigh skill", fontsize=6.4, color=C["sub"])
    ax.text(0.72, -0.30, "high correlation,\nno skill", fontsize=6.4, color=C["triv"])
    ax.set_xlabel("pairwise rho (score vs its primitive)")
    ax.set_ylabel("conditional skill beyond primitive (Kendall tau-b)")
    ax.set_title("Pairwise correlation != conditional skill", fontsize=9)
    ax.set_xlim(0.40, 1.03); ax.set_ylim(-0.40, 0.78)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig("fig4.png", dpi=150); fig.savefig("fig4.pdf"); plt.close(fig)
    print("fig4 ok")


if __name__ == "__main__":
    make_fig1(); make_fig2(); make_fig3(); make_fig4()
