#!/usr/bin/env python3
"""
Paper A figures. Every plotted value is read from a run artefact.

    python3 figures/make_paper_a_figures.py            # write figures/fig{1..4}.{png,pdf}
    python3 figures/make_paper_a_figures.py --provenance  # + the provenance table
    python3 figures/make_paper_a_figures.py --crosscheck  # + compare to $OWNBASELINE_DATA_ROOT

WHY THIS FILE WAS REWRITTEN
---------------------------
The version at figures/deprecated/make_paper_a_figures.py held every number as a
literal. Three of them were wrong against the run record: StemID at rho 0.990
and cmEntropy at rho 1.000 were drawn as measured points although no run
produces them, NCG was drawn as "own-baseline pending" although the fix-1 run
measured it, and the caption promised bootstrap intervals the figure did not
draw. Nothing in this file is a measurement typed by hand. Each number carries
the artefact and the field it came from, and `--provenance` prints them.

THE DECISION RULE
-----------------
A value is above its null when it exceeds the 97.5th percentile of the
*estimator's own measured null* at that value's own n and its own
score-primitive rank correlation. Not when a bootstrap interval excludes zero:
the bootstrap measures sampling precision of the point estimate and says
nothing about where the estimator sits when the true conditional skill is zero.
The floors come from experiments/null_calibration/results/ and the selection
mirrors `null_for` in experiments/null_calibration/report.py exactly — Null B,
align on, the 12-level ordinal, nearest rho grid point to |rho|, nearest n grid
point, mean over the three Null B coupling strengths and the *largest* of their
three 97.5th percentiles.

Floors differ by n and by rho, so each score gets its own floor marker on its
own row. There is no single threshold line across this plot and drawing one
would be wrong.

KERNEL
------
Kendall tau_b throughout. Weighted tau appears nowhere in these figures: the
null calibration showed the weighted-tau floor is high enough to swallow several
weighted-tau values whole, so the two kernels are not interchangeable and the
manuscript's tau_b column is the one that is read here.

RULES THIS FILE ENFORCES
------------------------
* A score never measured on the staged ordinal is never plotted as a point.
  StemID and cmEntropy are by-construction cases and appear only in a separately
  labelled band that carries no value.
* A score whose value cannot be located in a run artefact is drawn as blocked,
  with the reason on the row. Never as zero and never dropped.
* Error bars are the bootstrap intervals from the run JSONs and are labelled as
  sampling precision, not as the verdict.
* Rows measured on the shared seed-42 3,000-cell subsample say so on the row,
  because their floors are roughly twice the 39,505-cell floors.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ORDINAL_ACCESSION = "GSE106474"      # the staged zebrafish atlas these figures audit

REPO = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
NULL_RESULTS = REPO / "experiments" / "null_calibration" / "results"


def run_record() -> Path:
    override = os.environ.get("OWNBASELINE_RUN_RECORD")
    return Path(override).expanduser().resolve() if override else REPO / "data" / "run_record"


# ---------------------------------------------------------------- artefacts

class Artefact:
    """One run JSON, loaded once, with its path and digest kept for the report."""

    _cache: dict[str, "Artefact"] = {}

    def __init__(self, rel: str):
        self.rel = rel
        self.path = run_record() / rel
        self.sha256 = None
        self.obj = None
        self.error = None
        if not self.path.is_file():
            self.error = f"artefact not found: {self.path}"
            return
        raw = self.path.read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        try:
            self.obj = json.loads(raw)
        except json.JSONDecodeError as exc:            # pragma: no cover
            self.error = f"artefact unreadable: {self.path}: {exc}"

    @classmethod
    def get(cls, rel: str) -> "Artefact":
        if rel not in cls._cache:
            cls._cache[rel] = cls(rel)
        return cls._cache[rel]

    def dig(self, field: str):
        """
        Fetch a value by slash-separated field path. Returns (value, error).
        A missing key or a null is an error, never a zero.
        """
        if self.obj is None:
            return None, self.error
        cur = self.obj
        for part in field.split("/"):
            if not isinstance(cur, dict) or part not in cur:
                return None, f"field absent: {self.rel}:{field} (stopped at {part!r})"
            cur = cur[part]
        if cur is None:
            return None, f"field is null in the run record: {self.rel}:{field}"
        return cur, None


def value_of(rel: str, field: str):
    return Artefact.get(rel).dig(field)


# ------------------------------------------------------------- null floors

GRID_FILES = ["null_grid.csv", "null_grid_n127607_tail.csv", "null_grid_n3000.csv",
              "null_grid_n39505.csv", "null_grid_n500_n10000.csv"]


def _nearest(v, grid):
    return min(grid, key=lambda g: abs(g - v))


class NullGrid:
    """
    The measured null of the shipped estimator. Cell selection is the one
    experiments/null_calibration/report.py uses for the deliverable table; this
    class is a reimplementation of that selection over the same CSVs, not a
    second opinion about it.
    """

    def __init__(self, results_dir: Path = NULL_RESULTS):
        self.dir = results_dir
        self.missing = [f for f in GRID_FILES if not (results_dir / f).is_file()]
        canonical: dict[tuple, dict] = {}
        for fname in GRID_FILES:                       # first file wins, as in report.py
            path = results_dir / fname
            if not path.is_file():
                continue
            with open(path) as fh:
                for r in csv.DictReader(fh):
                    key = (r["null"], int(r["n"]), float(r["rho_target"]),
                           int(r["align"]), r["nullB_strength"], int(r["levels"]),
                           r["kernel"])
                    if key not in canonical:
                        r["_file"] = fname
                        canonical[key] = r
        self.rows = [r for r in canonical.values() if int(r["levels"]) == 12]
        # the grid's own design points, read off the grid rather than restated
        self.n_grid = sorted({int(r["n"]) for r in self.rows})
        self.rho_grid = sorted({float(r["rho_target"]) for r in self.rows})

    def floor(self, n: int, rho: float, kernel: str = "kendalltau"):
        """
        Returns (mean, p97.5, cell description, source files, (n, rho) grid point)
        for the null cell this value is placed against, or Nones with a reason.
        """
        if self.missing:
            return None, None, f"null grid incomplete, missing {self.missing}", None, None
        ng, rg = _nearest(n, self.n_grid), _nearest(abs(rho), self.rho_grid)
        cells = [r for r in self.rows
                 if r["null"] == "B" and int(r["n"]) == ng and r["kernel"] == kernel
                 and abs(float(r["rho_target"]) - rg) < 1e-9 and int(r["align"]) == 1]
        if not cells:
            return None, None, f"no null cell at n={ng}, rho={rg}, kernel={kernel}", None, None
        mean = sum(float(c["mean"]) for c in cells) / len(cells)
        p975 = max(float(c["p97.5"]) for c in cells)
        desc = (f"Null B, align on, 12 levels, n={ng:,}, rho={rg}, {kernel}, "
                f"{len(cells)} coupling strengths, 200 seeds each")
        return mean, p975, desc, sorted({c["_file"] for c in cells}), (ng, rg)


# ------------------------------------------------------- what to plot, and why
#
# One row per score in the field audit. `field` paths address the run record.
# The `n` and `rho` a row reports are read from the artefact too, because they
# select the null cell and a typo in either would silently move the floor.

MEASURED = [
    dict(score="CytoTRACE v1", primitive="gene count",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/CytoTRACE_v1 | gene_count/kang_wdm_taub/tau",
         ci="conditional_skill/CytoTRACE_v1 | gene_count/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/CytoTRACE_v1 | gene_count",
         n="n_cells_full", subsample=False),
    dict(score="SCENT SR", primitive="PCC(x, degree)",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/SR | PCC(x,degree)/kang_wdm_taub/tau",
         ci="conditional_skill/SR | PCC(x,degree)/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/SR | PCC(x,degree)",
         n="n_cells_full", subsample=False),
    dict(score="ORIGINS", primitive="PCC(x, degree)",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/ORIGINS | PCC(x,degree)/kang_wdm_taub/tau",
         ci="conditional_skill/ORIGINS | PCC(x,degree)/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/ORIGINS | PCC(x,degree)",
         n="n_cells_full", subsample=False),
    dict(score="NCG", primitive="PCC(x, degree)",
         art="track4/fix1_mce_ncg_skill.json",
         tau="scores/NCG/conditional_skill_vs_PCC/kang_wdm_taub/tau",
         ci="scores/NCG/conditional_skill_vs_PCC/kang_wdm_taub/CI95",
         rho="scores/NCG/spearman_score_vs_PCC",
         n="scores/NCG/n", subsample=True),
    dict(score="dpath", primitive="Shannon H",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/dpath | Shannon_H/kang_wdm_taub/tau",
         ci="conditional_skill/dpath | Shannon_H/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/dpath | Shannon_H",
         n="marginal_skill/dpath/n", subsample=True),
    dict(score="SLICE", primitive="Shannon H",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/SLICE | Shannon_H/kang_wdm_taub/tau",
         ci="conditional_skill/SLICE | Shannon_H/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/SLICE | Shannon_H",
         n="marginal_skill/SLICE/n", subsample=True),
    dict(score="CCAT", primitive="PCC(x, degree)",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/CCAT | PCC(x,degree)/kang_wdm_taub/tau",
         ci="conditional_skill/CCAT | PCC(x,degree)/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/CCAT | PCC(x,degree)",
         n="n_cells_full", subsample=False),
]

# Rows whose block is itself recorded in a run artefact, and rows whose block is
# recorded only in the vault note named here. Neither is a measurement, so
# neither is a literal this file is forbidden to hold; both are drawn as blocked
# and print their reason.
BLOCKED = [
    dict(score="MCE", primitive="PCC(x, degree)",
         art="track4/fix1_mce_ncg_skill.json", status="scores/MCE/status",
         source="run record"),
    dict(score="SPIDE", primitive="unresolved",
         reason="formula paywalled (Zheng 2023); never obtained, so the declared "
                "primitive cannot be derived and no own-baseline can be defined",
         source="04-experiments/2026-07-21-frozen-benchmark-and-final-reductions.md, "
                "final-2-reductions section"),
    dict(score="scEnergy", primitive="no fixed primitive",
         reason="needs a MATLAB/Octave environment; no implementation ran. Its "
                "scaffold is a data-built co-expression graph, so it does not "
                "reduce to a fixed low-order statistic either",
         source="04-experiments/2026-07-21-frozen-benchmark-and-final-reductions.md, "
                "frozen-benchmark table"),
]

# Measured, but on a different ordinal. Showing it as blank on this axis would
# be as wrong as showing it as zero, so it gets its own row and says where its
# numbers live.
# Outside the class this figure audits. CytoTRACE 2 is supervised, so it is a
# boundary control rather than a member of the unsupervised field; it has a row
# because naming it in the caption and not drawing it is how the last set of
# omissions happened.
OUT_OF_CLASS = [
    dict(score="CytoTRACE 2", primitive="supervised",
         art="atlas_run/per_dataset_results.json", probe="per_dataset",
         score_key="CytoTRACE 2 potency score",
         reason="supervised model, outside the unsupervised class this figure audits"),
]

OFF_ORDINAL = [
    dict(score="StemSC", primitive="gene count",
         art="stemsc/summary.json", probe="C1_GSE117498/StemSC/conditional_skill",
         reason="measured on the sorted atlases (GSE117498, GSE125970) under a "
                "different estimator and a heavily-tied ordering, not comparable "
                "to this axis"),
]

# Not measurements. No point, no bar, no zero. They sit in their own band.
BY_CONSTRUCTION = [
    dict(score="StemID", primitive="Shannon H",
         reason="equals the per-cell transcriptome-entropy component by construction"),
    dict(score="cmEntropy", primitive="Shannon H",
         reason="equals H(x/sum x) by definition"),
]
BY_CONSTRUCTION_SOURCE = ("04-experiments/2026-07-21-frozen-benchmark-and-final-reductions.md "
                          "(reduction column) and reproduction/table.md (NOT-A-MEASUREMENT)")

# Published claims quoted as labels in Fig 1. These are other authors' numbers
# about their own scores, not measurements made in this repository, and Fig 1
# plots nothing. They are listed so --provenance can name their source.
# (score, short label used on Fig 1, full claim, source note)
LITERATURE_CLAIMS = [
    ("SR (SCENT)", "authors: R^2 = 0.96 vs PCC",
     "the SCENT authors report R^2 = 0.96 between SR and PCC(x, degree)",
     "04-experiments/2026-07-20-track4-gate2a-score-formulas.md"),
    ("MCE", "authors: approx PCC (their Fig 4D)",
     "the MCE authors report MCE approximately equal to PCC(x, degree), their Fig 4D",
     "04-experiments/2026-07-21-frozen-benchmark-and-final-reductions.md"),
    ("NCG", "authors: ~90% of variance",
     "the NCG authors report ~90% of NCG variance explained by the "
     "transcriptome-connectome PCC",
     "04-experiments/2026-07-21-frozen-benchmark-and-final-reductions.md"),
]

# Regression check, not plotted. The floors are re-derived above from the grid
# CSVs; this reads the deliverable table the null-calibration run itself printed
# (experiments/null_calibration/results/report.txt, the machine output that
# section 2.1 of 04-experiments/2026-08-14-null-calibration.md was transcribed
# from) and requires the two to agree. No floor is restated here as a constant.
REPORT_TXT = NULL_RESULTS / "report.txt"
_TAUB_ROW = re.compile(
    r"\s(?P<n>\d+)\s+(?P<rho>[+-]\d\.\d+)\s+(?P<k>\d+)\s+"
    r"(?P<value>[+-]\d\.\d+)\s+(?P<mean>[+-]\d\.\d+)\s+(?P<p975>[+-]\d\.\d+)\s+"
    r"(?P<cell>Null B|covariate arm)")


def published_floors():
    """
    (n, rho, k) -> 97.5th percentile, from the tau_b block of the run's own
    deliverable table. Returns {} if that file is not in the checkout.
    """
    if not REPORT_TXT.is_file():
        return {}
    txt = REPORT_TXT.read_text()
    try:
        block = txt.split("--- tau_b")[1].split("--- wtau")[0]
    except IndexError:                                  # pragma: no cover
        return {}
    out = {}
    for line in block.splitlines():
        m = _TAUB_ROW.search(line)
        if m:
            out[(int(m["n"]), round(float(m["rho"]), 4), int(m["k"]))] = float(m["p975"])
    return out


# ------------------------------------------------------------- row assembly

class Row:
    """A single Figure 2 row after its numbers have been fetched and placed."""

    def __init__(self, kind, score, primitive):
        self.kind = kind                  # measured | blocked | by-construction
        self.score = score
        self.primitive = primitive
        self.tau = self.lo = self.hi = self.rho = self.n = None
        self.floor = self.floor_mean = None
        self.floor_cell = self.floor_files = self.floor_at = None
        self.subsample = False
        self.reason = None                # why blocked, or why not a measurement
        self.sources = {}                 # field -> "path:field"
        self.artefact = None

    @property
    def clears(self):
        if self.tau is None or self.floor is None:
            return None
        return self.tau > self.floor

    @property
    def label(self):
        return f"{self.score}  |  {self.primitive}"


def build_rows(grid: NullGrid):
    rows, notes = [], []

    for spec in MEASURED:
        row = Row("measured", spec["score"], spec["primitive"])
        row.subsample = spec["subsample"]
        row.artefact = spec["art"]
        fetched, err = {}, None
        for key in ("tau", "ci", "rho", "n"):
            val, e = value_of(spec["art"], spec[key])
            row.sources[key] = f"{spec['art']}:{spec[key]}"
            if e and err is None:
                err = e
            fetched[key] = val
        if err is not None:
            row.kind = "blocked"
            row.reason = err
            rows.append(row)
            notes.append(f"{spec['score']}: BLOCKED — {err}")
            continue
        row.tau = float(fetched["tau"])
        row.lo, row.hi = (float(fetched["ci"][0]), float(fetched["ci"][1]))
        row.rho = float(fetched["rho"])
        row.n = int(fetched["n"])
        mean, p975, cell, files, at = grid.floor(row.n, row.rho)
        row.floor_mean, row.floor, row.floor_cell = mean, p975, cell
        row.floor_files, row.floor_at = files, at
        if p975 is None:
            notes.append(f"{spec['score']}: no null floor — {cell}")
        rows.append(row)

    for spec in BLOCKED:
        row = Row("blocked", spec["score"], spec["primitive"])
        if "status" in spec:
            row.artefact = spec["art"]
            status, err = value_of(spec["art"], spec["status"])
            row.sources["status"] = f"{spec['art']}:{spec['status']}"
            row.reason = err if err else str(status)
        else:
            row.reason = spec["reason"]
            row.sources["status"] = spec["source"]
        rows.append(row)

    for spec in OFF_ORDINAL:
        row = Row("off-ordinal", spec["score"], spec["primitive"])
        row.artefact = spec["art"]
        probe, err = value_of(spec["art"], spec["probe"])
        row.sources["status"] = f"{spec['art']}:{spec['probe']}"
        row.reason = (spec["reason"] if err is None
                      else f"{spec['reason']} — and that run is not in this checkout: {err}")
        rows.append(row)

    for spec in OUT_OF_CLASS:
        row = Row("out-of-class", spec["score"], spec["primitive"])
        row.artefact = spec["art"]
        probe, err = value_of(spec["art"], spec["probe"])
        row.sources["status"] = f"{spec['art']}:{spec['probe']}"
        if err is not None:
            row.reason = f"{spec['reason']} — and its runs are not in this checkout: {err}"
        else:
            runs = [d for d in probe if spec["score_key"] in d.get("tau_kang", {})]
            species = sorted({d.get("species") for d in runs if d.get("species")})
            here = any(d.get("accession") == ORDINAL_ACCESSION for d in runs)
            row.reason = (
                f"{spec['reason']}; measured on the {len(runs)}-dataset probe "
                f"({', '.join(species)}), "
                + (f"and ALSO on {ORDINAL_ACCESSION} — this row's premise is wrong, check it"
                   if here else
                   f"never on {ORDINAL_ACCESSION}"))
        rows.append(row)

    for spec in BY_CONSTRUCTION:
        row = Row("by-construction", spec["score"], spec["primitive"])
        row.reason = spec["reason"]
        row.sources["status"] = BY_CONSTRUCTION_SOURCE
        rows.append(row)

    measured = [r for r in rows if r.kind == "measured"]
    measured.sort(key=lambda r: -r.tau)
    ordered = (measured
               + [r for r in rows if r.kind == "blocked"]
               + [r for r in rows if r.kind == "off-ordinal"]
               + [r for r in rows if r.kind == "out-of-class"]
               + [r for r in rows if r.kind == "by-construction"])
    return ordered, notes


def check_floors(rows):
    """
    The floor this file derives must equal the floor the null-calibration run
    printed for the same (n, rho, k=1) cell. Returns (failures, n_checked).
    """
    pub = published_floors()
    bad, checked = [], 0
    for r in rows:
        if r.floor is None or r.rho is None:
            continue
        want = pub.get((r.n, round(abs(r.rho), 4), 1))
        if want is None:
            want = pub.get((r.n, round(r.rho, 4), 1))
        if want is None:
            continue
        checked += 1
        if abs(round(r.floor, 4) - want) > 5e-5:
            bad.append(f"{r.score}: floor {r.floor:+.4f} != "
                       f"{want:+.4f} in {REPORT_TXT.name} at n={r.n}, rho={r.rho:+.4f}")
    return bad, checked


# ----------------------------------------------------------------- caption

def subsample_penalty(rows):
    """
    How much smaller n costs a row, measured rather than asserted: each
    subsample row's floor over the floor at the same rho grid point at the
    largest n on the figure. Returns (lo, hi, n_ref) or None.
    """
    meas = [r for r in rows if r.kind == "measured" and r.floor_at]
    if not meas:
        return None
    n_ref = max(r.floor_at[0] for r in meas)
    ref = {r.floor_at[1]: r.floor for r in meas if r.floor_at[0] == n_ref}
    ratios = [r.floor / ref[r.floor_at[1]] for r in meas
              if r.subsample and r.floor_at[1] in ref]
    return (min(ratios), max(ratios), n_ref) if ratios else None


def caption(rows) -> str:
    """
    Built from the rows that were actually drawn, so the caption cannot drift
    away from the figure the way the deprecated one did.
    """
    meas = [r for r in rows if r.kind == "measured"]
    clear = [r for r in meas if r.clears]
    fail = [r for r in meas if r.clears is False]
    blocked = [r for r in rows if r.kind == "blocked"]
    off = [r for r in rows if r.kind == "off-ordinal"]
    out = [r for r in rows if r.kind == "out-of-class"]
    band = [r for r in rows if r.kind == "by-construction"]
    ns = sorted({r.n for r in meas}, reverse=True)
    sub = [r.score for r in meas if r.subsample]
    floors = [r.floor for r in meas if r.floor is not None]
    negative = [r.score for r in fail if r.tau is not None and r.tau <= 0]
    pen = subsample_penalty(rows)

    def names(rs):
        rs = [r.score if isinstance(r, Row) else r for r in rs]
        return ", ".join(rs[:-1]) + " and " + rs[-1] if len(rs) > 1 else (rs[0] if rs else "none")

    if not meas:
        return (
            "Figure 2 | Field audit on a clean fine ordinal. No score could be placed: every row "
            f"is blocked and prints its reason. Blocked: {names(blocked)}. "
            f"Measured on another ordinal: {names(off)}. "
            f"Outside the audited class: {names(out)}. "
            f"Not measurements, shown in their own band with no value: {names(band)}. "
            "Nothing on this figure is a number typed by hand, so with no run artefact there is "
            "nothing to draw.")

    floor_span = (f"({min(floors):+.4f} to {max(floors):+.4f} here) " if floors else "")
    return (
        f"Figure 2 | Field audit on a clean fine ordinal: {len(clear)} of {len(meas)} measured "
        f"scores carry ordering skill beyond their own declared primitive. "
        f"Conditional skill of each score after rank-residualizing on the primitive its authors "
        f"declare, against the 12-stage Kimmel ordinal of GSE106474 (zebrafish whole embryos), "
        f"Kendall tau_b, seed 42. "
        f"Bars are point estimates; black whiskers are 95% bootstrap intervals and measure "
        f"sampling precision of the point estimate, not the verdict. "
        f"The dashed marker on each bar is that score's own decision floor: the 97.5th percentile "
        f"of the estimator's measured null at that value's own n and its own score-primitive rank "
        f"correlation rho, from the 200-seed null grid. "
        f"Floors are per-row and not comparable across rows "
        f"{floor_span}because they rise as n falls and vary "
        f"with rho; there is no single threshold line. "
        + (f"{names(sub)} were measured on the shared seed-42 {min(ns):,}-cell subsample, which "
           f"raises their floors to {pen[0]:.2f}-{pen[1]:.2f} times the {pen[2]:,}-cell floor at "
           f"the same rho. " if pen else
           f"{names(sub)} were measured on a {min(ns):,}-cell subsample. ")
        + f"{names(clear)} exceed their own floors. "
        + (
        f"{names(fail)} do not")
        + (f"; {names(negative)} are negative, and a negative value does not exceed a positive "
           f"threshold — that is where the value sits, not a failed test" if negative else "")
        + f". {names(blocked)} are blocked and carry no value: the reason is printed on the row. "
        f"{names(off)} was measured, but on the sorted atlases rather than this ordinal, so it has "
        f"a row and no bar. "
        f"{names(out)} is supervised and so sits outside the unsupervised class audited here; it "
        f"has a row and no bar for the same reason. "
        f"{names(band)} are not measurements — each equals a transcriptome-entropy primitive by "
        f"construction, no run produces a number for them, and they appear only in the shaded "
        f"band at the foot of the plot, which carries no scale. "
        f"Values: {'; '.join(f'{r.score} {r.tau:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] vs floor {r.floor:+.4f} at n={r.n:,}, rho={r.rho:+.4f}' for r in meas)}."
    )


# ---------------------------------------------------------------- Figure 2

CLEARS = "#1a7f37"
FAILS = "#8a8a8a"
FLOORC = "#b00000"
BLOCKC = "#b08900"
BANDC = "#9141ac"
OFFC = "#3a6ea5"
OUTC = "#4a4a4a"


def make_fig2(rows):
    import textwrap

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    import numpy as np

    meas = [r for r in rows if r.kind == "measured"]
    n_rows = len(rows)
    has_sub = any(r.subsample for r in meas)

    # Margins are held constant in inches, not in figure fractions, so adding a
    # row does not push the legend around.
    H = 0.54 * n_rows + 2.6
    up = lambda inches: 1.0 - inches / H          # noqa: E731  (from the top)
    dn = lambda inches: inches / H                # noqa: E731  (from the bottom)
    fig = plt.figure(figsize=(10.2, H))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.0], wspace=0.03,
                          left=0.215, right=0.995, bottom=dn(1.50), top=up(1.30))
    ax = fig.add_subplot(gs[0, 0])
    axt = fig.add_subplot(gs[0, 1], sharey=ax)

    y = np.arange(n_rows)[::-1]
    lo_val = min([r.lo for r in meas] + [0.0])
    hi_val = max([r.hi for r in meas] + [0.0])
    xlo, xhi = min(-0.32, lo_val - 0.12), max(0.80, hi_val + 0.16)

    for yi, r in zip(y, rows):
        if r.kind == "by-construction":
            ax.add_patch(mp.Rectangle((xlo, yi - 0.5), xhi - xlo, 1.0,
                                      facecolor="#f2ecf7", edgecolor="none", zorder=0))
            continue
        if r.kind in ("blocked", "off-ordinal", "out-of-class"):
            c = {"blocked": BLOCKC, "off-ordinal": OFFC, "out-of-class": OUTC}[r.kind]
            lead = {"blocked": "blocked, no value drawn",
                    "off-ordinal": "not on this ordinal, no value drawn",
                    "out-of-class": "not in this figure's class, no value drawn"}[r.kind]
            hatch = {"blocked": "///", "off-ordinal": "\\\\", "out-of-class": "..."}[r.kind]
            ax.barh(yi, 0.016, height=0.62, color="none", edgecolor=c,
                    hatch=hatch, lw=0.9, zorder=3)
            ax.text(0.034, yi, textwrap.fill(f"{lead} — {r.reason}", 74), va="center",
                    fontsize=5.8, color=c, style="italic", linespacing=1.25)
            continue

        col = CLEARS if r.clears else FAILS
        ax.barh(yi, r.tau, height=0.62, color=col, edgecolor="#333", lw=0.5, zorder=3)
        ax.plot([r.lo, r.hi], [yi, yi], color="#111", lw=1.1, zorder=5)
        for e in (r.lo, r.hi):
            ax.plot([e, e], [yi - 0.13, yi + 0.13], color="#111", lw=1.1, zorder=5)
        if r.tau >= 0:
            ax.text(max(r.tau, r.hi) + 0.015, yi, f"{r.tau:+.3f}", va="center",
                    ha="left", fontsize=7.2, zorder=6)
        else:
            ax.text(min(r.tau, r.lo) - 0.015, yi, f"{r.tau:+.3f}", va="center",
                    ha="right", fontsize=7.2, zorder=6)

        if r.floor is not None:
            # this row's floor only: a dashed marker on the row band, never a
            # line across the plot, because floors differ by n and by rho.
            ax.plot([r.floor, r.floor], [yi - 0.37, yi + 0.37], color=FLOORC,
                    lw=1.6, ls=(0, (2.2, 1.4)), zorder=7, solid_capstyle="butt")
            if not r.clears:
                ax.annotate("", xy=(r.floor, yi), xytext=(max(r.tau, 0.0), yi),
                            arrowprops=dict(arrowstyle="->", color=FLOORC, lw=0.8,
                                            linestyle=(0, (1.6, 1.6)), shrinkA=0,
                                            shrinkB=2), zorder=6)

    ax.axvline(0, color="#888", lw=0.9, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels([r.label for r in rows], fontsize=7.4)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r.kind == "by-construction":
            tick.set_color(BANDC)
            tick.set_fontstyle("italic")
        elif r.kind == "blocked":
            tick.set_color(BLOCKC)
        elif r.kind == "off-ordinal":
            tick.set_color(OFFC)
        elif r.kind == "out-of-class":
            tick.set_color(OUTC)
    ax.set_ylim(-0.70, n_rows - 0.30)
    ax.set_xlim(xlo, xhi)
    ax.set_xlabel("conditional skill beyond the declared primitive   (Kendall tau_b)",
                  fontsize=8, labelpad=4)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=7)

    # ---- per-row column: n, this row's own floor, and the verdict in words
    axt.set_xlim(0, 1)
    axt.axis("off")
    head_y = n_rows - 0.42
    axt.text(0.03, head_y, "n", fontsize=6.6, fontweight="bold", ha="left")
    axt.text(0.40, head_y, "its own floor", fontsize=6.6, fontweight="bold", ha="right")
    axt.text(0.48, head_y, "clears it?", fontsize=6.6, fontweight="bold", ha="left")
    for yi, r in zip(y, rows):
        if r.kind == "measured":
            axt.text(0.03, yi, f"{r.n:,}" + (" \u2021" if r.subsample else ""),
                     fontsize=6.6, va="center", ha="left",
                     color=FLOORC if r.subsample else "#333")
            axt.text(0.40, yi, f"{r.floor:+.4f}" if r.floor is not None else "—",
                     fontsize=6.6, va="center", ha="right", color=FLOORC)
            verdict = "yes" if r.clears else ("no — negative" if r.tau <= 0 else "no")
            axt.text(0.48, yi, verdict, fontsize=6.9, va="center", ha="left",
                     color=CLEARS if r.clears else FAILS,
                     fontweight="bold" if r.clears else "normal")
        elif r.kind == "blocked":
            axt.text(0.03, yi, "—", fontsize=6.6, va="center", color=BLOCKC)
            axt.text(0.48, yi, "not measured", fontsize=6.9, va="center",
                     ha="left", color=BLOCKC, style="italic")
        elif r.kind == "off-ordinal":
            axt.text(0.03, yi, "—", fontsize=6.6, va="center", color=OFFC)
            axt.text(0.48, yi, "measured elsewhere", fontsize=6.9, va="center",
                     ha="left", color=OFFC, style="italic")
        elif r.kind == "out-of-class":
            axt.text(0.03, yi, "—", fontsize=6.6, va="center", color=OUTC)
            axt.text(0.48, yi, "outside the class", fontsize=6.9, va="center",
                     ha="left", color=OUTC, style="italic")
        else:
            axt.text(0.03, yi, "—", fontsize=6.6, va="center", color=BANDC)
            axt.text(0.48, yi, "not a measurement", fontsize=6.9, va="center",
                     ha="left", color=BANDC, style="italic")

    band = [r for r in rows if r.kind == "by-construction"]
    if band:
        y_band = [yi for yi, r in zip(y, rows) if r.kind == "by-construction"]
        ax.text(xlo + 0.014, min(y_band) - 0.44,
                "by construction \u2248 a transcriptome-entropy primitive: no run produces a "
                "number, so no value is plotted",
                fontsize=6.3, color=BANDC, style="italic", va="bottom")

    n_clear = sum(1 for r in meas if r.clears)
    fig.text(0.215, up(0.42),
             f"Field audit on a clean fine ordinal (zebrafish GSE106474, 12 stages)",
             fontsize=10.0, ha="left", va="top")
    n_blocked = len([r for r in rows if r.kind == "blocked"])
    n_off = len([r for r in rows if r.kind == "off-ordinal"])
    n_out = len([r for r in rows if r.kind == "out-of-class"])
    fig.text(0.215, up(0.80),
             f"{n_clear} of {len(meas)} measured scores exceed their own measured null floor\n"
             f"every audited score has a row: {n_blocked} blocked, "
             f"{n_off} measured on another ordinal, {n_out} outside the audited class, "
             f"{len(band)} never measured",
             fontsize=7.6, ha="left", va="top", color="#555", linespacing=1.5)

    pen = subsample_penalty(rows)
    if has_sub:
        n_sub = min(r.n for r in meas if r.subsample)
        detail = (f"{pen[0]:.2f}\u2013{pen[1]:.2f}\u00d7 the {pen[2]:,}-cell floor at the same rho"
                  if pen else "a higher floor")
        fig.text(0.215, dn(0.95),
                 f"\u2021 measured on the shared seed-42 {n_sub:,}-cell subsample; a smaller n "
                 f"raises the floor, so these rows face {detail}",
                 fontsize=6.2, color=FLOORC, style="italic", ha="left")

    handles = [
        mp.Patch(facecolor=CLEARS, edgecolor="#333", lw=0.5,
                 label="clears its own null floor"),
        plt.Line2D([0], [0], color=FLOORC, lw=1.6, ls=(0, (2.2, 1.4)),
                   label="that row's floor: 97.5th pct of this score's measured null, "
                         "at its own n and rho"),
        mp.Patch(facecolor=FAILS, edgecolor="#333", lw=0.5,
                 label="does not clear its own null floor"),
        plt.Line2D([0], [0], color="#111", lw=1.1,
                   label="95% bootstrap interval — sampling precision, not the verdict"),
        mp.Patch(facecolor="none", edgecolor=BLOCKC, hatch="///",
                 label="blocked — no run artefact; reason on the row"),
        mp.Patch(facecolor="none", edgecolor=OFFC, hatch="\\\\",
                 label="measured, but on another ordinal — not comparable here"),
        mp.Patch(facecolor="none", edgecolor=OUTC, hatch="...",
                 label="outside the unsupervised class this figure audits"),
        mp.Patch(facecolor="#f2ecf7", edgecolor="none",
                 label="by construction, never measured — no value"),
    ]
    fig.legend(handles=handles, fontsize=6.4, loc="lower left", frameon=False,
               bbox_to_anchor=(0.210, dn(0.10)), ncol=2, handlelength=2.6,
               columnspacing=1.6, labelspacing=0.55)

    fig.savefig(OUT / "fig2.png", dpi=200)
    fig.savefig(OUT / "fig2.pdf")
    plt.close(fig)
    print(f"fig2 ok -> {OUT/'fig2.png'}, {OUT/'fig2.pdf'}")



# ---------------------------------------------------------------- Figure 1
# A schematic. It plots no measurement: the only numbers on it are other
# authors' published claims about their own scores, and each is sourced in
# LITERATURE_CLAIMS above and printed by --provenance.

def make_fig1():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    claim = {name: short for name, short, _full, _src in LITERATURE_CLAIMS}
    deg = "#1a5fb4"; ent = "#9141ac"; cnt = "#c64600"

    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

    def box(x, y, w, h, text, fc, tc, fs=7.2, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fc, ec="#444", lw=0.7))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal")

    def arrow(x1, y1, x2, y2, col):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                     arrowstyle="->,head_width=2.5,head_length=3",
                                     color=col, lw=1.0, mutation_scale=6))

    box(7.15, 4.4, 2.75, 1.0, "PRIMITIVE\nPearson(x, PPI degree)", "#eaf1fb", deg, 7.2, True)
    box(7.15, 2.5, 2.75, 1.0, "PRIMITIVE\nShannon entropy H(x)", "#f3ecf7", ent, 7.2, True)
    box(7.15, 0.6, 2.75, 1.0, "PRIMITIVE\ngene count = H0", "#fbeee6", cnt, 7.2, True)

    ax.text(0.1, 5.6, "degree-correlation family", fontsize=7.6, color=deg, fontweight="bold")
    fam = [f"SR  ({claim['SR (SCENT)']})",
           "CCAT  (= by definition)",
           f"MCE  ({claim['MCE']})",
           f"NCG  ({claim['NCG']})",
           "ORIGINS  (+ Sx term)"]
    for i, s in enumerate(fam):
        yy = 5.15 - i * 0.42
        box(0.1, yy - 0.16, 4.6, 0.34, s, "#eaf1fb", deg, 6.0)
        arrow(4.75, yy, 7.10, 4.9, deg)

    ax.text(0.1, 2.95, "expression-entropy family", fontsize=7.6, color=ent, fontweight="bold")
    for i, s in enumerate(["StemID", "cmEntropy (= by def)", "SLICE (fixed GO proj.)"]):
        yy = 2.5 - i * 0.42
        box(0.1, yy - 0.16, 4.6, 0.34, s, "#f3ecf7", ent, 6.4)
        arrow(4.75, yy, 7.10, 3.0, ent)

    ax.text(0.1, 0.98, "gene count", fontsize=7.6, color=cnt, fontweight="bold")
    box(0.1, 0.4, 4.6, 0.34, "CytoTRACE (population)", "#fbeee6", cnt, 6.4)
    arrow(4.75, 0.57, 7.10, 1.1, cnt)

    ax.set_title('A field of "different" potency scores collapses to two primitives + gene count',
                 fontsize=8.6, pad=6)
    ax.text(5.0, -0.30,
            "parenthetical claims are the scores' own authors, not measurements here; "
            "own-baseline tests each score against its primitive in Fig 2 and Fig 4",
            ha="center", fontsize=5.9, color="#555", style="italic")
    fig.savefig(OUT / "fig1.png", dpi=200, bbox_inches="tight")
    fig.savefig(OUT / "fig1.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"fig1 ok -> {OUT/'fig1.png'}")


# ---------------------------------------------------------------- Figure 3
# The depth confound, read from the two w4 atlas runs. Kendall tau_b: the
# weighted-tau copy of the same run is not read by any figure here.

FIG3 = [
    dict(panel="a", title="Hematopoietic (sorted)", art="w4/c1_gse117498_results.json",
         hi=("HSC", "potency high"), lo=("GMP", "potency low"), color="#c8709a"),
    dict(panel="b", title="Intestinal", art="w4/c2_gse125970_results.json",
         hi=("Stem Cell", "potency high"), lo=("Enterocyte", "potency low"), color="#2f7f4f"),
]


def make_fig3():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels, missing = [], []
    for spec in FIG3:
        got = {}
        for key, field in (
                ("hi_genes", f"per_population_depth/{spec['hi'][0]}/median_gene_count"),
                ("lo_genes", f"per_population_depth/{spec['lo'][0]}/median_gene_count"),
                ("auroc", "per_score/CT/AUROC_primitive_top_vs_bot"),
                ("delta", "per_score/CT/marginal_delta"),
                ("kernel", "kernel"), ("n", "n_cells_ranked")):
            val, err = value_of(spec["art"], field)
            if err:
                missing.append(f"fig3 {spec['panel']}: {err}")
                got = None
                break
            got[key] = val
            spec.setdefault("sources", {})[key] = f"{spec['art']}:{field}"
        if got:
            panels.append((spec, got))

    if missing:
        for m in missing:
            print(f"  fig3 BLOCKED: {m}", file=sys.stderr)
    if not panels:
        print("fig3 skipped — no w4 artefacts in the run record", file=sys.stderr)
        return missing

    fig = plt.figure(figsize=(6.0, 2.9))
    gs = fig.add_gridspec(1, len(panels), wspace=0.42, left=0.10, right=0.98,
                          bottom=0.24, top=0.80)
    for i, (spec, g) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.text(-0.16, 1.08, spec["panel"], transform=ax.transAxes,
                fontsize=11, fontweight="bold")
        vals = [g["hi_genes"], g["lo_genes"]]
        ymax = max(vals) * 1.62
        ax.bar([0, 1], vals, 0.56, color=[spec["color"], "#b0b0b0"],
               edgecolor="#555", linewidth=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{spec['hi'][0]}\n({spec['hi'][1]})",
                            f"{spec['lo'][0]}\n({spec['lo'][1]})"], fontsize=6.0)
        ax.set_ylabel("median detected genes", fontsize=7)
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.62, 1.62)
        ax.tick_params(axis="y", labelsize=6.5)
        for v, x in zip(vals, (0, 1)):
            ax.text(x, v + 0.015 * ymax, f"{v:,.1f}", ha="center", va="bottom",
                    fontsize=6.0, color="#333")

        inverted = vals[1] > vals[0]
        col = "#b00000" if inverted else "#008000"
        lift = 0.105 * ymax
        ax.annotate("", xy=(1, vals[1] + lift), xytext=(0, vals[0] + lift),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
        ax.text(0.42 if inverted else 0.34, (vals[0] + vals[1]) / 2 + lift - 0.115 * ymax,
                "genes up as\npotency down" if inverted else "genes down as\npotency down",
                ha="center", va="top", fontsize=5.8, color=col, linespacing=1.3)
        note = (f"gene-count AUROC {g['auroc']:.3f}"
                + (" — below chance\nthe naive 'score beats primitive' test then\n"
                   f"reads CT marginal delta = {g['delta']:+.3f} as a WIN"
                   if inverted else " — direction correct"))
        ax.text(0.02, 0.985, note, transform=ax.transAxes, fontsize=5.4, va="top",
                color=col, linespacing=1.35)
        ax.set_title(f"{spec['title']}  (n = {g['n']:,})", fontsize=7.2, pad=3)

    fig.text(0.5, 0.035,
             "medians and AUROC read from the w4 run record; Kendall tau_b",
             ha="center", fontsize=5.6, color="#555", style="italic")
    fig.savefig(OUT / "fig3.png", dpi=200)
    fig.savefig(OUT / "fig3.pdf")
    plt.close(fig)
    print(f"fig3 ok -> {OUT/'fig3.png'}")
    return missing


# ---------------------------------------------------------------- Figure 4

def make_fig4(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    meas = [r for r in rows if r.kind == "measured"]
    band = [r for r in rows if r.kind == "by-construction"]
    if not meas:
        print("fig4 skipped — no measured row survived the run record", file=sys.stderr)
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for r in meas:
        col = CLEARS if r.clears else FAILS
        ax.scatter(abs(r.rho), r.tau, s=72, color=col, edgecolor="#333",
                   linewidth=0.6, zorder=4)
        ax.plot([abs(r.rho)] * 2, [r.lo, r.hi], color="#111", lw=0.9, zorder=3)
        if r.floor is not None:
            ax.plot([abs(r.rho) - 0.016, abs(r.rho) + 0.016], [r.floor] * 2,
                    color=FLOORC, lw=1.4, ls=(0, (2.0, 1.2)), zorder=5)
        ax.annotate(f"{r.score}", (abs(r.rho), r.tau), textcoords="offset points",
                    xytext=(7, 6 if r.tau >= 0 else -13), fontsize=7.0)

    ax.axhline(0, color="#bbb", lw=0.8, ls="--", zorder=1)

    ax.set_xlabel("|Spearman(score, its declared primitive)|", fontsize=8)
    ax.set_ylabel("conditional skill beyond the primitive (Kendall tau_b)", fontsize=8)
    ax.set_title("Pairwise correlation with the primitive does not predict conditional skill",
                 fontsize=8.6)
    ax.set_xlim(0.30, 1.11)
    ax.set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_ylim(min(r.lo for r in meas) - 0.10, max(r.hi for r in meas) + 0.13)
    if band:
        # placed past every measured rho so it cannot be read as a data location
        ax.axvspan(1.045, 1.105, color="#f2ecf7", zorder=0)
        ax.text(1.075, sum(ax.get_ylim()) / 2,
                ", ".join(r.score for r in band) + " — by construction, no measured value",
                fontsize=6.1, color=BANDC, ha="center", va="center", style="italic",
                rotation=90)
    ax.tick_params(labelsize=7)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [mp.Patch(color=CLEARS, label="clears its own null floor"),
               mp.Patch(color=FAILS, label="does not clear its own null floor"),
               plt.Line2D([0], [0], color=FLOORC, lw=1.4, ls=(0, (2.0, 1.2)),
                          label="that score's own null floor"),
               plt.Line2D([0], [0], color="#111", lw=0.9, label="95% bootstrap interval")]
    ax.legend(handles=handles, fontsize=5.9, loc="lower left", frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "fig4.png", dpi=200)
    fig.savefig(OUT / "fig4.pdf")
    plt.close(fig)
    print(f"fig4 ok -> {OUT/'fig4.png'}")


# ------------------------------------------------------------- provenance

def _md(s: str) -> str:
    """Markdown table cells cannot hold a bare pipe, and every row label has one."""
    return str(s).replace("|", "\\|")


def provenance(rows, grid) -> str:
    """Every plotted value, its source file and field, its floor, and the verdict."""
    L = []
    A = L.append
    A("| # | row | plotted value | source file : field | n | rho | own floor (97.5th pct) "
      "| floor source | clears? |")
    A("|---|---|---|---|---:|---:|---:|---|---|")
    i = 0
    for r in rows:
        i += 1
        if r.kind == "measured":
            files = ", ".join(r.floor_files or [])
            A(f"| {i} | `{_md(r.label)}` | `{r.tau:+.4f}` [{r.lo:+.4f}, {r.hi:+.4f}] "
              f"| `{_md(r.sources['tau'])}`<br>CI: `{_md(r.sources['ci'])}` "
              f"| {r.n:,}{' (subsample)' if r.subsample else ''} | {r.rho:+.4f} "
              f"| `{r.floor:+.4f}` | {r.floor_cell}<br>`experiments/null_calibration/results/"
              f"{{{files}}}` | **{'yes' if r.clears else 'no'}**"
              f"{'  (value is negative; the threshold is positive)' if r.tau <= 0 else ''} |")
        elif r.kind in ("blocked", "off-ordinal", "out-of-class"):
            # a row demoted from measured keeps the field path it tried to read
            src = r.sources.get("status") or r.sources.get("tau", "—")
            what = {"blocked": "*blocked, no value drawn*",
                    "off-ordinal": "*measured on another ordinal, no value drawn here*",
                    "out-of-class": "*outside the audited class, no value drawn here*"}[r.kind]
            verdict = {"blocked": "not measured", "off-ordinal": "measured elsewhere",
                       "out-of-class": "outside the class"}[r.kind]
            A(f"| {i} | `{_md(r.label)}` | {what} "
              f"| `{_md(src)}` | — | — | — | — | {verdict} |")
        else:
            A(f"| {i} | `{_md(r.label)}` | *no value drawn — band only* "
              f"| {_md(r.sources['status'])} | — | — | — | — | not a measurement |")
    A("")
    A("Blocked and by-construction reasons, verbatim:")
    A("")
    for r in rows:
        if r.kind in ("blocked", "off-ordinal", "out-of-class", "by-construction"):
            A(f"- **{r.score}** — {r.reason}")
    A("")
    A("Artefact digests:")
    A("")
    A("| artefact | sha256 |")
    A("|---|---|")
    for rel, art in sorted(Artefact._cache.items()):
        A(f"| `{_rel(art.path)}` | `{art.sha256 or 'ABSENT'}` |")
    A("")
    A("Literature claims quoted as labels in Figure 1 (not measurements, nothing plotted):")
    A("")
    for name, short, full, src in LITERATURE_CLAIMS:
        A(f"- **{name}** — label on Fig 1: \"{short}\"; in full: {full}  [{src}]")
    bad, checked = check_floors(rows)
    A("")
    A(f"Floor regression check against `{REPORT_TXT.relative_to(REPO)}` "
      f"(the deliverable table the null-calibration run printed, and the machine "
      f"output section 2.1 of `04-experiments/2026-08-14-null-calibration.md` was "
      f"transcribed from): "
      + (f"PASS, {checked} rows agree to 4 dp" if not bad and checked
         else ("SKIPPED, that file is not in this checkout" if not checked else "FAIL")))
    for b in bad:
        A(f"  - {b}")
    return "\n".join(L)


def _rel(path: Path) -> str:
    """Repo-relative when it is inside the repo, so the record is machine-neutral."""
    try:
        return str(Path(path).relative_to(REPO))
    except ValueError:
        return str(path)


def crosscheck(rows):
    """
    Re-read the same fields from $OWNBASELINE_DATA_ROOT, where a local re-run
    writes, and report the largest disagreement per row. A run that produced no
    value for a row is reported as such, not silently skipped.
    """
    sys.path.insert(0, str(REPO))
    from own_baseline.paths import data_root                       # noqa: E402
    root = data_root()
    lines = [f"crosscheck against {root}"]
    layout = {"track2/gate2_conditional_skill.json": "track2/results/gate2_conditional_skill.json",
              "track4/fix1_mce_ncg_skill.json": "track4/results/fix1_mce_ncg_skill.json",
              "w4/c1_gse117498_results.json": "w4/results/c1_gse117498_results.json",
              "w4/c2_gse125970_results.json": "w4/results/c2_gse125970_results.json"}
    for r in rows:
        if r.kind != "measured":
            continue
        rel, field = r.sources["tau"].split(":", 1)
        path = root / layout[rel]
        if not path.is_file():
            lines.append(f"  {r.score:14} no local re-run at {path}")
            continue
        cur = json.loads(path.read_text())
        for part in field.split("/"):
            cur = cur.get(part) if isinstance(cur, dict) else None
            if cur is None:
                break
        if cur is None:
            lines.append(f"  {r.score:14} present but not regenerated (field null/absent)")
        else:
            lines.append(f"  {r.score:14} record {r.tau:+.10f}  re-run {float(cur):+.10f}  "
                         f"|delta| {abs(float(cur) - r.tau):.2e}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--provenance", action="store_true",
                    help="print the value/source/floor table")
    ap.add_argument("--provenance-out", default=None,
                    help="also write the provenance table to this path")
    ap.add_argument("--crosscheck", action="store_true",
                    help="compare the record against $OWNBASELINE_DATA_ROOT")
    ap.add_argument("--only", default=None, help="comma-separated subset of 1,2,3,4")
    args = ap.parse_args()

    grid = NullGrid()
    if grid.missing:
        print(f"WARNING: null grid incomplete, missing {grid.missing}; "
              f"floors will be unavailable and every row will say so", file=sys.stderr)
    rows, notes = build_rows(grid)
    for n in notes:
        print(f"  NOTE: {n}", file=sys.stderr)

    bad, checked = check_floors(rows)
    for b in bad:
        print(f"  FLOOR CHECK FAILED: {b}", file=sys.stderr)
    if not bad and checked:
        print(f"  floor check: {checked} rows agree with {REPORT_TXT.name}", file=sys.stderr)

    want = set((args.only or "1,2,3,4").split(","))
    if "1" in want:
        make_fig1()
    if "2" in want:
        make_fig2(rows)
    if "3" in want:
        make_fig3()
    if "4" in want:
        make_fig4(rows)

    cap = caption(rows)
    (OUT / "fig2_caption.txt").write_text(cap + "\n")
    print(f"caption -> {OUT/'fig2_caption.txt'}")

    table = provenance(rows, grid)
    if args.provenance:
        print("\n" + table)
    if args.provenance_out:
        Path(args.provenance_out).write_text(table + "\n")
        print(f"provenance -> {args.provenance_out}")

    (OUT / "fig2_values.json").write_text(json.dumps({
        "kernel": "Kendall tau_b",
        "decision_rule": "value > 97.5th percentile of the estimator's measured null at "
                         "the value's own n and its own score-primitive rank correlation",
        "run_record": _rel(run_record()),
        "null_grid": _rel(NULL_RESULTS),
        "artefacts": {a.rel: a.sha256 for a in Artefact._cache.values()},
        "rows": [dict(kind=r.kind, score=r.score, primitive=r.primitive, tau=r.tau,
                      ci95=[r.lo, r.hi] if r.lo is not None else None, n=r.n, rho=r.rho,
                      null_mean=r.floor_mean, null_p97_5=r.floor, null_cell=r.floor_cell,
                      null_files=r.floor_files, clears_own_floor=r.clears,
                      subsample_3000=r.subsample, reason=r.reason, sources=r.sources)
                 for r in rows],
        "caption": cap,
    }, indent=2) + "\n")
    print(f"values -> {OUT/'fig2_values.json'}")

    if args.crosscheck:
        print("\n" + crosscheck(rows))

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
