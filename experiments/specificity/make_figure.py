#!/usr/bin/env python3
"""
make_figure.py: the negative-control figure, as a standalone pgfplots file.

    python3 experiments/specificity/make_figure.py --out <manuscript folder>/figures
    tectonic -X compile <manuscript folder>/figures/fig_controls.tex

Panel (a): each audited score's conditional skill on its declared primitive, on the band that
holds 95 percent of 1,000 random 200-gene sets residualised on the same primitive on the same
cells. Panel (b): the same after the four primitives are removed at once. Every coordinate is
read from results/specificity_GSE106474.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from own_baseline.floors import floor as grid_floor  # noqa: E402

ORDER = ("CytoTRACE", "SR", "ORIGINS", "NCG", "dpath", "StemFinder", "SLICE", "CCAT", "MCE")
SUB = ("NCG", "dpath", "SLICE")
PRIM = {"detected_genes": "gene count", "pcc_x_degree": "PCC(x, degree)", "shannon_entropy": "Shannon H",
        "stemfinder_cell_cycle_score": "cell-cycle score"}


def panel(z, joint, title, with_labels):
    n = len(ORDER)
    band, med, pts_yes, pts_no, floors = [], [], [], [], []
    for i, name in enumerate(ORDER):
        y = n - i
        r = z["rows"][name]
        fam = (z["controls_subsample"] if name in SUB else z["controls"])["a1_g200"]
        q = fam["joint" if joint else r["primitive"]]["conditional_skill"]["quantiles"]
        value = r["conditional_skill_joint"] if joint else r["conditional_skill"]
        a = r["vs_controls"]["a1_g200"]
        exceeds = a["joint_exceeds_p975"] if joint else a["exceeds_p975"]
        band.append(f"\\addplot[line width=5pt, black!22] coordinates {{({q['0.025']:.4f},{y}) ({q['0.975']:.4f},{y})}};")
        med.append(f"\\addplot[line width=0.9pt, black!60] coordinates {{({q['0.5']:.4f},{y - 0.24}) ({q['0.5']:.4f},{y + 0.24})}};")
        (pts_yes if exceeds else pts_no).append(f"({value:.4f},{y})")
        if not joint:
            fl = grid_floor(r["n"], abs(r["spearman_with_primitive"]), kernel="taub", k_covariates=1, levels=12)["floor"]
            floors.append(f"\\addplot[line width=0.8pt, red!75!black, densely dashed] coordinates {{({fl:.4f},{y - 0.34}) ({fl:.4f},{y + 0.34})}};")
    ticks = ",".join(str(n - i) for i in range(n))
    if with_labels:
        labels = ",".join("{" + f"{name} $|$ {PRIM[z['rows'][name]['primitive']]}" + "}" for name in ORDER)
    else:
        labels = ",".join("{}" for _ in ORDER)
    out = [f"\\begin{{axis}}[title={{{title}}}, width=6.4cm, height=6.3cm, xmin=-0.62, xmax=0.82, ymin=0.4, ymax={n + 0.6},",
           f"  ytick={{{ticks}}}, yticklabels={{{labels}}}, yticklabel style={{font=\\scriptsize}}, xticklabel style={{font=\\scriptsize}},",
           "  title style={font=\\footnotesize, yshift=-3pt}, xlabel style={font=\\scriptsize}, xtick={-0.5,0,0.5},",
           "  xlabel={conditional skill (Kendall $\\tau_b$)}, axis lines=left, tick align=outside, clip=false, xlabel style={font=\\scriptsize, yshift=4pt},",
           "  extra x ticks={0}, extra x tick labels={}, extra x tick style={grid=major, grid style={black!35}}]"]
    out += band + med + floors
    if pts_yes:
        out.append(f"\\addplot[only marks, mark=*, mark size=2.1pt, green!45!black] coordinates {{{' '.join(pts_yes)}}};")
    if pts_no:
        out.append(f"\\addplot[only marks, mark=*, mark size=2.1pt, black!70, fill=white] coordinates {{{' '.join(pts_no)}}};")
    out.append("\\end{axis}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dir", default=str(HERE / "results"))
    a = ap.parse_args()
    z = json.loads((Path(a.dir) / "specificity_GSE106474.json").read_text())
    tex = ["\\documentclass[border=2pt]{standalone}", "\\usepackage{pgfplots}", "\\pgfplotsset{compat=1.17}",
           "\\usepackage[T1]{fontenc}", "\\usepackage{lmodern}", "\\renewcommand{\\familydefault}{\\sfdefault}",
           "\\begin{document}", "\\begin{tikzpicture}",
           panel(z, False, "(a) beyond the declared primitive", True),
           "\\begin{scope}[xshift=6.1cm]", panel(z, True, "(b) beyond four primitives at once", False), "\\end{scope}",
           "\\begin{scope}[yshift=-1.55cm, xshift=-1.0cm]",
           "\\draw[line width=5pt, black!22] (0,0) -- (0.6,0); \\draw[line width=0.9pt, black!60] (0.3,-0.11) -- (0.3,0.11);",
           "\\node[anchor=west, font=\\scriptsize] at (0.7,0) {1,000 random 200-gene sets: central 95\\% and median};",
           "\\draw[line width=0.8pt, red!75!black, densely dashed] (7.6,-0.14) -- (7.6,0.14); \\node[anchor=west, font=\\scriptsize] at (7.75,0) {registered floor};",
           "\\fill[green!45!black] (0.3,-0.42) circle (2.1pt); \\node[anchor=west, font=\\scriptsize] at (0.7,-0.42) {score above that band};",
           "\\draw[black!70, fill=white] (4.3,-0.42) circle (2.1pt); \\node[anchor=west, font=\\scriptsize] at (4.5,-0.42) {score inside or below it};",
           "\\end{scope}", "\\end{tikzpicture}", "\\end{document}"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "fig_controls.tex").write_text("\n".join(tex) + "\n")
    print(out / "fig_controls.tex")


if __name__ == "__main__":
    main()
