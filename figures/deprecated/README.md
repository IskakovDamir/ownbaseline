# Deprecated figure script

`make_paper_a_figures.py` in this directory is the version that was in
`figures/` until 2026-08-29. It is kept, unmodified, for one reason:

**its literals are what the pre-2026-08 manuscript was traced against.**

Every number in it is hard-coded. Nothing in it reads a run artefact. The
traceability pass that checked the manuscript's text against the run record
covered the text and not the figure source, so these literals were never
checked, and three of them are wrong:

| literal in this script | what the run record says |
|---|---|
| `("StemID", 0.990, 0.000, "triv")` — drawn as a measured point in Fig 4, and at skill 0.000 in Fig 2 | no run produces either number. StemID equals the per-cell transcriptome-entropy component by construction and was never measured on the staged ordinal. |
| `("cmEntropy", 1.000, 0.000, "triv")` — same | same. `reproduction/table.md` records both as NOT-A-MEASUREMENT. |
| `("NCG", None, "pend")` — drawn as "own-baseline pending" | `track4/results/fix1_mce_ncg_skill.json` measured NCG against `PCC(x,degree)` on 2026-07-21 at Kendall tau_b **+0.4015** [+0.3824, +0.4212], n = 3,000. The headline count of five substantive scores depends on it. |

Two further defects, neither a wrong number:

- The caption promised 95% bootstrap intervals. The script has no `errorbar`
  call and draws none.
- `ROWS` mixes weighted tau and Kendall tau_b conventions in one table without
  saying which column is which; the replacement is tau_b throughout.

Do not run this file. It writes `fig1..fig4` into the current working
directory, which will overwrite the current figures if run from `figures/`.
The replacement at `figures/make_paper_a_figures.py` reads
`data/run_record/` and `experiments/null_calibration/results/` and names the
source of every value it draws.

Full comparison, literal by literal:
`04-experiments/2026-08-29-figure2-provenance.md` in the research vault.
