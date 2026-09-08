#!/usr/bin/env python3
"""
Paper A figures. Every plotted value is read from a run artefact.

    python3 figures/make_paper_a_figures.py            # write the figures and captions
    python3 figures/make_paper_a_figures.py --provenance  # + the provenance table
    python3 figures/make_paper_a_figures.py --crosscheck  # + compare to $OWNBASELINE_DATA_ROOT

OUTPUTS
-------
    figures/fig2_body.pdf       3.18 in, \begin{figure}  — the 7 measured rows
    figures/fig2_appendix.pdf   6.50 in, \begin{figure*} — all 14 rows + side table
    figures/fig3.pdf            6.50 in, \begin{figure*} — the depth confound
    figures/fig1.{png,pdf}, figures/fig4.{png,pdf}
    figures/fig2_body_caption.txt, figures/fig2_appendix_caption.txt
    figures/fig2_values.json

PAGE GEOMETRY
-------------
ML4H 2026 is jmlr.cls with pmlr, twocolumn, 10pt: a 229.87749pt column and a
469.75499pt text block, i.e. 3.18 in and 6.50 in, over 10pt body text. The
retargeted figures are emitted at exactly those widths, so \\includegraphics
needs no scale factor and nothing on them is set below 7pt at final size. Both
are enforced rather than claimed: FontLedger.enforce() raises on type below the
floor, and each run reads the width back out of the written PDF's own MediaBox.

Figure 2 comes in two variants from one `rows` object. The body variant carries
only the scores measured on the ordinal, because a fourteen-row audit with a
reason on each row cannot be read at 3.18 in; the n / floor / verdict side table
it has no width for moves into its caption. The appendix variant is the full
audit and is where the rows that carry no value live. Figure 3 is a figure* for
a measured reason, printed by fig3_column_fit() on every run.

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
Kendall tau_b throughout. Nothing is plotted under weighted tau: the null
calibration showed the weighted-tau floor is high enough to swallow several
weighted-tau values whole, so the two kernels are not interchangeable and the
manuscript's tau_b column is the one that is read here.

That makes the figure's verdicts one-kernel verdicts, so the caption says which
of them the other kernel would not return. Any row clearing its tau_b floor but
not its weighted-tau floor is named in one sentence, with the count a both-kernel
rule would give. Both numbers are read from the wtau block of the same
report.txt the tau_b floors are checked against; if that block is missing, or a
row is absent from it, this file raises rather than writing a caption whose
verdict list is silently one-kernel.

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


# --------------------------------------------------------------- page geometry
#
# ML4H 2026 is jmlr.cls with the pmlr, twocolumn, 10pt options. That template's
# own \the\columnwidth and \the\textwidth are 229.87749pt and 469.75499pt in TeX
# points (1/72.27 in), i.e. 3.1808 in and 6.5000 in, over 10pt body text.
#
# Every retargeted figure is emitted at exactly the width it will occupy, so
# \includegraphics carries no scale factor and the point sizes requested below
# are the point sizes on the printed page. COLUMN_W_IN is 3.18 rather than the
# full 3.1808: 0.0008 in (0.06pt) inside the measured column, so a body figure
# cannot overfull it.
TEX_PT_PER_IN = 72.27
COLUMN_W_PT = 229.87749
TEXT_W_PT = 469.75499
COLUMN_W_IN = 3.18                     # \columnwidth, \begin{figure}
TEXT_W_IN = 6.50                       # \textwidth,   \begin{figure*}
MIN_PT = 7.0                           # nothing smaller survives next to 10pt body text


class TypographyTooSmall(RuntimeError):
    """
    A retargeted figure asked for type below MIN_PT at final size. Raised rather
    than warned: shrinking the font is the one way of fitting a figure into a
    column that costs the reader the figure.
    """


class FontLedger:
    """
    Every font size on a retargeted figure is requested through one of these, so
    the smallest size actually used is measured and enforced rather than claimed.
    The figure is emitted at its final width and LaTeX rescales nothing, so the
    number recorded here is the number on the page.
    """

    def __init__(self, name, floor_pt=MIN_PT):
        self.name = name
        self.floor_pt = floor_pt
        self.used = {}                 # pt -> {what it is used for}

    def __call__(self, pt, what):
        pt = round(float(pt), 2)
        self.used.setdefault(pt, set()).add(what)
        return pt

    @property
    def smallest(self):
        return min(self.used) if self.used else None

    def enforce(self):
        below = {pt: sorted(w) for pt, w in self.used.items()
                 if self.floor_pt is not None and pt < self.floor_pt - 1e-9}
        if below:
            raise TypographyTooSmall(f"{self.name}: {below} below the "
                                     f"{self.floor_pt:g}pt floor at final size")

    def report(self):
        s = self.smallest
        return (f"{self.name}: smallest type {s:.1f}pt at final size "
                f"({'; '.join(sorted(self.used[s]))})")


_MEDIABOX = re.compile(rb"/MediaBox\s*\[\s*([\d.eE+-]+)\s+([\d.eE+-]+)\s+"
                       rb"([\d.eE+-]+)\s+([\d.eE+-]+)\s*\]")


def pdf_size_inches(path):
    """
    (width, height) in inches read back out of the written PDF's own MediaBox,
    so the width reported is the width of the file rather than the width that
    was asked for. PDF user space is 1/72 in, which is what \\includegraphics
    assumes when no scale is given.
    """
    m = _MEDIABOX.search(Path(path).read_bytes())
    if not m:                                           # pragma: no cover
        raise RuntimeError(f"no /MediaBox in {path}: its width cannot be verified")
    x0, y0, x1, y1 = (float(g) for g in m.groups())
    return (x1 - x0) / 72.0, (y1 - y0) / 72.0


def report_width(path, target_in, ledger=None):
    """One line per emitted figure: the width of the file, against its target."""
    w, h = pdf_size_inches(path)
    flag = "ok" if abs(w - target_in) <= 0.005 else "MISMATCH"
    line = (f"  {Path(path).name:22} {w:.4f} x {h:.4f} in   "
            f"(target width {target_in:.2f} in) {flag}")
    if ledger is not None:
        line += f"\n  {'':22} {ledger.report()}"
    return line


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
         report_row="CytoTRACE_v1 | gene_count",
         ci="conditional_skill/CytoTRACE_v1 | gene_count/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/CytoTRACE_v1 | gene_count",
         n="n_cells_full", subsample=False),
    dict(score="SCENT SR", primitive="PCC(x, degree)",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/SR | PCC(x,degree)/kang_wdm_taub/tau",
         report_row="SR | PCC(x,degree)",
         ci="conditional_skill/SR | PCC(x,degree)/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/SR | PCC(x,degree)",
         n="n_cells_full", subsample=False),
    dict(score="ORIGINS", primitive="PCC(x, degree)",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/ORIGINS | PCC(x,degree)/kang_wdm_taub/tau",
         report_row="ORIGINS | PCC(x,degree)",
         ci="conditional_skill/ORIGINS | PCC(x,degree)/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/ORIGINS | PCC(x,degree)",
         n="n_cells_full", subsample=False),
    dict(score="NCG", primitive="PCC(x, degree)",
         art="track4/fix1_mce_ncg_skill.json",
         tau="scores/NCG/conditional_skill_vs_PCC/kang_wdm_taub/tau",
         report_row="NCG | PCC(x,degree)",
         ci="scores/NCG/conditional_skill_vs_PCC/kang_wdm_taub/CI95",
         rho="scores/NCG/spearman_score_vs_PCC",
         n="scores/NCG/n", subsample=True),
    dict(score="dpath", primitive="Shannon H",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/dpath | Shannon_H/kang_wdm_taub/tau",
         report_row="dpath | Shannon_H",
         ci="conditional_skill/dpath | Shannon_H/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/dpath | Shannon_H",
         n="marginal_skill/dpath/n", subsample=True),
    dict(score="SLICE", primitive="Shannon H",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/SLICE | Shannon_H/kang_wdm_taub/tau",
         report_row="SLICE | Shannon_H",
         ci="conditional_skill/SLICE | Shannon_H/kang_wdm_taub/CI95",
         rho="spearman_score_vs_primitive/SLICE | Shannon_H",
         n="marginal_skill/SLICE/n", subsample=True),
    dict(score="CCAT", primitive="PCC(x, degree)",
         art="track2/gate2_conditional_skill.json",
         tau="conditional_skill/CCAT | PCC(x,degree)/kang_wdm_taub/tau",
         report_row="CCAT | PCC(x,degree)",
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
         reason="source paper paywalled (Zheng 2023); a first-party implementation "
                "is public and was not run, so no own-baseline is defined here",
         source="04-experiments/2026-07-21-frozen-benchmark-and-final-reductions.md, "
                "final-2-reductions section"),
    dict(score="scEnergy", primitive="no fixed primitive",
         reason="needs MATLAB; its network step uses the Statistics toolbox and "
                "graph objects GNU Octave does not provide. Its scaffold is a "
                "data-built co-expression graph, so it does not reduce to a "
                "fixed low-order statistic either",
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


class WtauMissing(RuntimeError):
    """
    The weighted-tau qualifier could not be derived. Raised rather than swallowed:
    a caption that lists five clearing scores without saying one of them clears on
    a single kernel is exactly the incomplete claim this qualifier exists to fix.
    """


_WTAU_ROW = re.compile(
    r"^\s{2}(?P<row>\S.*?)\s{2,}(?P<atlas>\S+)\s+(?P<n>\d+)\s+(?P<rho>[+-]\d\.\d+)\s+"
    r"(?P<k>\d+)\s+(?P<value>[+-]\d\.\d+)\s+(?P<mean>[+-]\d\.\d+)\s+"
    r"(?P<p975>[+-]\d\.\d+)\s+(?P<cell>Null B|covariate arm)")


def wtau_table(atlas: str = "zebrafish", k: int = 1):
    """
    row label -> (value, null mean, null 97.5th) from the wtau block of the run's
    own deliverable table, for one atlas and one covariate count. The figure is
    tau_b, so this is read only to say which of its verdicts the other kernel
    would not return. Raises if the file or the block is not in the checkout,
    because the alternative is a caption that quietly overstates its own scope.
    """
    if not REPORT_TXT.is_file():
        raise WtauMissing(f"{REPORT_TXT} not in this checkout, so the weighted-tau "
                          f"floors the caption qualifier needs cannot be read")
    txt = REPORT_TXT.read_text()
    if "--- wtau" not in txt:
        raise WtauMissing(f"no '--- wtau' block in {REPORT_TXT.name}; the qualifier "
                          f"naming one-kernel verdicts cannot be derived")
    block = txt.split("--- wtau")[1].split("\n--- ")[0]
    out = {}
    for line in block.splitlines():
        m = _WTAU_ROW.match(line)
        if m and m["atlas"] == atlas and int(m["k"]) == k:
            out[m["row"]] = (float(m["value"]), float(m["mean"]), float(m["p975"]))
    if not out:
        raise WtauMissing(f"the wtau block of {REPORT_TXT.name} has no {atlas} k={k} "
                          f"rows; the caption qualifier cannot be derived")
    return out


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
        # the other kernel, read only to qualify this one's verdict, never plotted
        self.wtau = self.wtau_mean = self.wtau_floor = None
        self.floor_cell = self.floor_files = self.floor_at = None
        self.subsample = False
        self.reason = None                # why blocked, or why not a measurement
        self.sources = {}                 # field -> "path:field"
        self.artefact = None

    @property
    def clears_wtau(self):
        """Same rule, the other kernel. None when this row carries no weighted-tau pair."""
        if self.wtau is None or self.wtau_floor is None:
            return None
        return self.wtau > self.wtau_floor

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
    wtau = wtau_table()          # raises if the other kernel's table is not here

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
        # The other kernel's value and floor for the same row, so the caption can
        # say which tau_b verdicts a both-kernel rule would not return. Absent is
        # a hard error: see WtauMissing.
        label = spec["report_row"]
        if label not in wtau:
            raise WtauMissing(
                f"{spec['score']}: no row {label!r} in the wtau block of "
                f"{REPORT_TXT.name}. The caption cannot state whether this "
                f"verdict holds under both kernels, and must not omit the "
                f"question. Known rows: {sorted(wtau)}")
        row.wtau, row.wtau_mean, row.wtau_floor = wtau[label]
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


def caption(rows, variant="appendix") -> str:
    """
    Built from the rows that were actually drawn, so the caption cannot drift
    away from the figure the way the deprecated one did.

    variant="appendix"  the figure that carries every row: the caption this file
                        has always generated, relabelled, plus one sentence
                        saying which rows the body figure does not draw.
    variant="body"      the column-width figure: the same decision rule and the
                        same numbers for the measured rows, carrying the n, floor
                        and verdict that the dropped side table used to hold, and
                        naming the appendix figure the rest of the audit is in.
    """
    if variant not in ("body", "appendix"):                      # pragma: no cover
        raise ValueError(f"unknown caption variant {variant!r}")
    meas = [r for r in rows if r.kind == "measured"]
    clear = [r for r in meas if r.clears]
    fail = [r for r in meas if r.clears is False]
    # rows this figure calls clear that the other kernel would not: the figure is
    # tau_b, so its verdict list is a one-kernel list and has to say so.
    split = [r for r in clear if r.clears_wtau is False]
    blocked = [r for r in rows if r.kind == "blocked"]
    off = [r for r in rows if r.kind == "off-ordinal"]
    out = [r for r in rows if r.kind == "out-of-class"]
    band = [r for r in rows if r.kind == "by-construction"]
    valueless = blocked + off + out + band
    ns = sorted({r.n for r in meas}, reverse=True)
    sub = [r.score for r in meas if r.subsample]
    floors = [r.floor for r in meas if r.floor is not None]
    negative = [r.score for r in fail if r.tau is not None and r.tau <= 0]
    pen = subsample_penalty(rows)
    label = "Figure 2" if variant == "body" else APPENDIX_FIG

    def names(rs):
        rs = [r.score if isinstance(r, Row) else r for r in rs]
        return ", ".join(rs[:-1]) + " and " + rs[-1] if len(rs) > 1 else (rs[0] if rs else "none")

    if not meas:
        return (
            f"{label} | Field audit on a clean fine ordinal. No score could be placed: every row "
            f"is blocked and prints its reason. Blocked: {names(blocked)}. "
            f"Measured on another ordinal: {names(off)}. "
            f"Outside the audited class: {names(out)}. "
            f"Not measurements, shown in their own band with no value: {names(band)}. "
            "Nothing on this figure is a number typed by hand, so with no run artefact there is "
            "nothing to draw.")

    floor_span = (f"({min(floors):+.4f} to {max(floors):+.4f} here) " if floors else "")

    # ---- the part both variants share: the rule, the floors, the verdicts
    shared_rule = (
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
           f"{names(sub)} were measured on a {min(ns):,}-cell subsample. "))

    shared_verdicts = (
        f"{names(clear)} exceed their own floors. "
        + (f"{names(split)} {'clears' if len(split) == 1 else 'clear'} under tau_b only: "
           f"{'its' if len(split) == 1 else 'their'} weighted-tau "
           f"{'value' if len(split) == 1 else 'values'} of "
           f"{', '.join(f'{r.wtau:+.4f}' for r in split)} "
           f"{'sits' if len(split) == 1 else 'sit'} below that kernel's own floor of "
           f"{', '.join(f'{r.wtau_floor:+.4f}' for r in split)} at the same cell, so a rule "
           f"requiring both kernels would read {len(clear) - len(split)} of {len(meas)} here "
           f"rather than {len(clear)} of {len(meas)}. " if split else "")
        + f"{names(fail)} do not"
        + (f"; {names(negative)} are negative, and a negative value does not exceed a positive "
           f"threshold — that is where the value sits, not a failed test" if negative else "")
        + ". ")

    headline = (
        f"{label} | Field audit on a clean fine ordinal: {len(clear)} of {len(meas)} measured "
        f"scores carry ordering skill beyond their own declared primitive. ")

    if variant == "body":
        # The side table this variant has no column width for, printed per row.
        table = "; ".join(
            f"{r.score} {r.tau:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] vs its own floor "
            f"{r.floor:+.4f} at n={r.n:,}, rho={r.rho:+.4f}, "
            f"{'clears' if r.clears else 'does not clear'}"
            for r in meas)
        return (
            headline
            + f"Only the {len(meas)} scores measured on this ordinal are drawn here. The full "
              f"{len(rows)}-row audit, which adds the {len(valueless)} rows that carry no value "
              f"and prints the reason on each, is {APPENDIX_FIG}. "
            + shared_rule
            + shared_verdicts
            + f"The rows not drawn here are in {APPENDIX_FIG}: {names(blocked)} blocked with no "
              f"run artefact; {names(off)} measured, but on the sorted atlases rather than this "
              f"ordinal; {names(out)} supervised and so outside the unsupervised class audited "
              f"here; {names(band)} not measurements at all, each equal to a "
              f"transcriptome-entropy primitive by construction. "
            + f"Value, 95% interval, own floor, n, rho and verdict per row: {table}.")

    return (
        headline
        + f"This is the full audit behind Figure 2, which draws the {len(meas)} measured rows "
          f"only; the {len(valueless)} rows that carry no value are here, each with its reason. "
        + shared_rule
        + shared_verdicts
        + f"{names(blocked)} are blocked and carry no value: the reason is printed on the row. "
          f"{names(off)} was measured, but on the sorted atlases rather than this ordinal, so it "
          f"has a row and no bar. "
          f"{names(out)} is supervised and so sits outside the unsupervised class audited here; it "
          f"has a row and no bar for the same reason. "
          f"{names(band)} are not measurements — each equals a transcriptome-entropy primitive by "
          f"construction, no run produces a number for them, and they appear only in the shaded "
          f"band at the foot of the plot, which carries no scale. "
        + f"Values: {'; '.join(f'{r.score} {r.tau:+.4f} [{r.lo:+.4f}, {r.hi:+.4f}] vs floor {r.floor:+.4f} at n={r.n:,}, rho={r.rho:+.4f}' for r in meas)}.")


# ---------------------------------------------------------------- Figure 2
#
# Two variants of one audit, built from the same `rows`, so neither can carry a
# number the other does not and neither can be regenerated without the other.
#
#   fig2_body.pdf      COLUMN_W_IN wide, \begin{figure}. Only the rows measured
#                      on the ordinal. The n / floor / verdict side table is
#                      dropped for want of column width; its content is in this
#                      variant's caption and drawn in full in the appendix one.
#   fig2_appendix.pdf  TEXT_W_IN wide, \begin{figure*}. Every row, the side
#                      table, and the by-construction band. The rows that carry
#                      no value belong here, where there is room to print why.
#
# Both lay out in inches: one data unit on y is one inch down the axes, so a row
# is exactly as tall as its own wrapped text needs and no row is squeezed to fit
# another. Row heights therefore differ inside the appendix variant, which is
# what lets fourteen rows and their reasons fit on a page.
#
# Every block of running text is wrapped and every column is placed from a
# measurement of the text itself, taken on a throwaway figure of the target
# width before the real one is opened. A width guessed at 10.2 in is a width
# that silently runs off the page at 3.18 in.

CLEARS = "#1a7f37"
FAILS = "#8a8a8a"
FLOORC = "#b00000"
BLOCKC = "#b08900"
BANDC = "#9141ac"
OFFC = "#3a6ea5"
OUTC = "#4a4a4a"

APPENDIX_FIG = "Figure A1"          # what the body caption calls the fourteen-row audit

_KIND_COLOR = {"blocked": BLOCKC, "off-ordinal": OFFC, "out-of-class": OUTC,
               "by-construction": BANDC}
_KIND_LEAD = {"blocked": "blocked, no value drawn",
              "off-ordinal": "not on this ordinal, no value drawn",
              "out-of-class": "not in this figure's class, no value drawn"}
_KIND_HATCH = {"blocked": "///", "off-ordinal": "\\\\", "out-of-class": "..."}
_KIND_VERDICT = {"blocked": "not measured", "off-ordinal": "measured elsewhere",
                 "out-of-class": "outside the class", "by-construction": "not a measurement"}


class FloorMismatch(RuntimeError):
    """
    A floor about to be drawn disagrees with the floor the null-calibration run
    itself printed for the same cell. Both variants refuse to be written in that
    state: the floor markers are the argument, and a wrong one is worse than a
    missing figure.
    """


def _floors_checked(rows):
    """The existing cross-check, made a precondition of drawing either variant."""
    bad, checked = check_floors(rows)
    if bad:
        raise FloorMismatch("; ".join(bad))
    return checked


class _TextMetrics:
    """
    Text extents in inches, in the figure font, for layout arithmetic.

    Measured on a private canvas at DPI, not on the figure being laid out. At
    the default 100 dpi the Agg renderer rounds every glyph advance to a whole
    pixel and reports a string up to 7% narrower than the PDF backend will
    actually set it — which is the difference between a wrapped line that fits
    the page and one that runs off the right edge of it. At 1200 dpi the
    reported width agrees with the width in the written PDF to about 0.05%.
    """

    DPI = 1200

    def __init__(self):
        import matplotlib.pyplot as plt
        self._plt = plt
        self._fig = plt.figure(figsize=(1.0, 1.0), dpi=self.DPI)
        self._renderer = self._fig.canvas.get_renderer()
        self._probe = self._fig.text(0, 0, "", fontsize=10)

    def width(self, text, pt):
        widest = 0.0
        for line in str(text).split("\n"):
            self._probe.set_text(line)
            self._probe.set_fontsize(pt)
            widest = max(widest, self._probe.get_window_extent(
                renderer=self._renderer).width / self.DPI)
        return widest

    def wrap(self, text, width_in, pt):
        """Greedy wrap to width_in inches. Explicit newlines are kept as breaks."""
        out = []
        for para in str(text).split("\n"):
            cur = ""
            for word in para.split():
                trial = f"{cur} {word}".strip()
                if cur and self.width(trial, pt) > width_in:
                    out.append(cur)
                    cur = word
                else:
                    cur = trial
            out.append(cur)
        return out

    def close(self):
        self._plt.close(self._fig)


def _ylabel(r, dagger=False):
    """Two lines: one line of 'score  |  primitive' does not fit a column."""
    return f"{r.score}{' ‡' if dagger else ''}\n|  {r.primitive}"


def _xlim_for(bars, axes_w_in, label_w_in, pad_in=0.022):
    """
    x limits leaving exactly enough room for the outboard value labels, solved
    rather than guessed: the allowance is a fixed number of inches, so the data
    span it costs depends on the span, which this inverts in closed form.
    """
    lo_raw = min([r.lo for r in bars] + [0.0])
    hi_raw = max([r.hi for r in bars] + [0.0])
    left_in = (label_w_in + pad_in) if any(r.tau < 0 for r in bars) else pad_in
    right_in = (label_w_in + pad_in) if any(r.tau >= 0 for r in bars) else pad_in
    span = (hi_raw - lo_raw) / (1.0 - (left_in + right_in) / axes_w_in)
    xlo = lo_raw - left_in / axes_w_in * span
    return xlo, xlo + span


def _draw_bar_row(ax, r, y, bar_h, pt_value):
    """One measured row: the bar, its bootstrap interval, its own floor marker."""
    col = CLEARS if r.clears else FAILS
    ax.barh(y, r.tau, height=bar_h, color=col, edgecolor="#333", lw=0.5, zorder=3)
    ax.plot([r.lo, r.hi], [y, y], color="#111", lw=1.0, zorder=5)
    for e in (r.lo, r.hi):
        ax.plot([e, e], [y - bar_h * 0.30, y + bar_h * 0.30], color="#111", lw=1.0, zorder=5)
    if r.tau >= 0:
        ax.text(max(r.tau, r.hi) + 0.012, y, f"{r.tau:+.3f}", va="center",
                ha="left", fontsize=pt_value, zorder=6)
    else:
        ax.text(min(r.tau, r.lo) - 0.012, y, f"{r.tau:+.3f}", va="center",
                ha="right", fontsize=pt_value, zorder=6)
    if r.floor is not None:
        # this row's floor only, never a line across the plot: floors differ by
        # n and by rho, so one threshold line would be a different claim. It is
        # kept inside its own row band so it cannot be read as spanning rows.
        ax.plot([r.floor, r.floor], [y - bar_h * 0.72, y + bar_h * 0.72], color=FLOORC,
                lw=1.5, ls=(0, (2.0, 1.3)), zorder=7, solid_capstyle="butt")
        if not r.clears:
            ax.annotate("", xy=(r.floor, y), xytext=(max(r.tau, 0.0), y),
                        arrowprops=dict(arrowstyle="->", color=FLOORC, lw=0.8,
                                        linestyle=(0, (1.6, 1.6)), shrinkA=0,
                                        shrinkB=2), zorder=6)


TEXT_MARGIN_IN = 0.065      # left inset of every block of running text on a figure


def _line_in(pt, leading=1.20):
    return pt * leading / 72.0


# ------------------------------------------------------ Figure 2, body variant

def make_fig2_body(rows):
    """
    COLUMN_W_IN wide. Only the scores measured on the ordinal: a fourteen-row
    audit with its reasons cannot be read at 3.18 in, and the rows that carry no
    value are the ones that lose least by moving to the appendix figure, where
    there is room to print the reason each of them carries instead of a number.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    _floors_checked(rows)
    fs = FontLedger("fig2_body")
    meas = [r for r in rows if r.kind == "measured"]
    if not meas:
        print("fig2_body skipped — no measured row to draw", file=sys.stderr)
        return None

    pt_tick = fs(7.0, "y tick labels")
    pt_value = fs(7.0, "per-bar values")
    pt_xtick = fs(7.0, "x tick labels")
    pt_xlab = fs(7.0, "x axis label")
    pt_head = fs(8.5, "headline")
    pt_leg = fs(7.0, "legend")
    pt_foot = fs(7.0, "subsample footnote")
    fs.enforce()

    W = COLUMN_W_IN
    ROW_H, BAR_H = 0.285, 0.150
    RIGHT_IN = 0.05
    n_clear = sum(1 for r in meas if r.clears)
    pen = subsample_penalty(rows)
    ylabels = [_ylabel(r, r.subsample) for r in meas]
    head = f"{n_clear} of {len(meas)} measured scores clear their own null floor"
    xlab = "conditional skill beyond the declared primitive\n(Kendall tau_b)"
    foot = ""
    if any(r.subsample for r in meas):
        n_sub = min(r.n for r in meas if r.subsample)
        detail = (f"{pen[0]:.2f}–{pen[1]:.2f}× the {pen[2]:,}-cell floor at the same rho"
                  if pen else "a higher floor")
        foot = (f"‡ seed-42 {n_sub:,}-cell subsample; a smaller n raises the floor, so "
                f"these rows face {detail}")

    # ---- measure first, then choose the height the measurements imply
    pm = _TextMetrics()
    LEFT_IN = max(pm.width(t, pt_tick) for t in ylabels) + 0.07
    label_w = max(pm.width(f"{r.tau:+.3f}", pt_value) for r in meas)
    run_w = W - 2 * TEXT_MARGIN_IN
    head_lines = pm.wrap(head, run_w, pt_head)
    xlab_lines = pm.wrap(xlab, run_w, pt_xlab)
    foot_lines = pm.wrap(foot, run_w, pt_foot) if foot else []
    pm.close()

    HEAD_IN = 0.07 + len(head_lines) * _line_in(pt_head, 1.25)
    XAXIS_IN = 0.20 + len(xlab_lines) * _line_in(pt_xlab, 1.30)
    FOOT_IN = (0.06 + len(foot_lines) * _line_in(pt_foot, 1.25)) if foot_lines else 0.0
    LEG_IN = 2 * _line_in(pt_leg, 1.55) + 0.04          # two legend rows
    AX_H = ROW_H * len(meas)
    H = HEAD_IN + AX_H + XAXIS_IN + FOOT_IN + LEG_IN
    axes_w = W - LEFT_IN - RIGHT_IN

    fig = plt.figure(figsize=(W, H))
    xlo, xhi = _xlim_for(meas, axes_w, label_w)
    ax = fig.add_axes([LEFT_IN / W, (LEG_IN + FOOT_IN + XAXIS_IN) / H,
                       axes_w / W, AX_H / H])
    y = [AX_H - (i + 0.5) * ROW_H for i in range(len(meas))]
    for yi, r in zip(y, meas):
        _draw_bar_row(ax, r, yi, BAR_H, pt_value)

    ax.axvline(0, color="#888", lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=pt_tick, linespacing=1.15)
    ax.set_ylim(0, AX_H)
    ax.set_xlim(xlo, xhi)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=pt_xtick, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)

    # the x label is centred on the figure, not on the axes: the axes are pushed
    # right by the y labels, and a label centred on them runs off the column.
    xm = TEXT_MARGIN_IN / W
    fig.text(0.5, (LEG_IN + FOOT_IN + 0.035) / H, "\n".join(xlab_lines),
             fontsize=pt_xlab, ha="center", va="bottom", linespacing=1.30)
    fig.text(xm, 1.0 - 0.045 / H, "\n".join(head_lines),
             fontsize=pt_head, ha="left", va="top", linespacing=1.25)
    if foot_lines:
        fig.text(xm, (LEG_IN + 0.030) / H, "\n".join(foot_lines), fontsize=pt_foot,
                 color=FLOORC, style="italic", ha="left", va="bottom", linespacing=1.25)

    handles = [
        mp.Patch(facecolor=CLEARS, edgecolor="#333", lw=0.5, label="clears its own floor"),
        mp.Patch(facecolor=FAILS, edgecolor="#333", lw=0.5, label="does not clear"),
        plt.Line2D([0], [0], color=FLOORC, lw=1.5, ls=(0, (2.0, 1.3)),
                   label="that row's own null floor"),
        plt.Line2D([0], [0], color="#111", lw=1.0, label="95% bootstrap CI"),
    ]
    fig.legend(handles=handles, fontsize=pt_leg, loc="lower left", frameon=False,
               bbox_to_anchor=(0.006, 0.018 / H), ncol=2, handlelength=1.9,
               columnspacing=0.9, labelspacing=0.35, handletextpad=0.45,
               borderpad=0.0, borderaxespad=0.0)

    out = OUT / "fig2_body.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"fig2_body ok -> {out}")
    return fs


# -------------------------------------------------- Figure 2, appendix variant

def make_fig2_appendix(rows):
    """
    TEXT_W_IN wide, for \\begin{figure*}. Every row the audit considered, the
    per-row n / floor / verdict table, and the by-construction band. Rows are
    given individual heights so a three-line reason is not compressed into the
    height of a bar, and the table columns are placed from the measured width of
    their own longest entry.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt

    _floors_checked(rows)
    fs = FontLedger("fig2_appendix")
    meas = [r for r in rows if r.kind == "measured"]
    band = [r for r in rows if r.kind == "by-construction"]

    pt_tick = fs(7.0, "y tick labels")
    pt_value = fs(7.0, "per-bar values")
    pt_xtick = fs(7.0, "x tick labels")
    pt_xlab = fs(7.5, "x axis label")
    pt_reason = fs(7.0, "per-row reasons")
    pt_table = fs(7.0, "n / floor / verdict table")
    pt_head = fs(9.0, "headline")
    pt_sub = fs(7.5, "sub-headline")
    pt_leg = fs(7.0, "legend")
    pt_foot = fs(7.0, "subsample footnote")
    fs.enforce()

    W = TEXT_W_IN
    BAR_H, MEAS_ROW_H, BAND_ROW_H = 0.150, 0.250, 0.235
    RIGHT_IN, GAP_IN, STUB_IN = 0.04, 0.17, 0.055
    n_clear = sum(1 for r in meas if r.clears)
    n_blocked = len([r for r in rows if r.kind == "blocked"])
    n_off = len([r for r in rows if r.kind == "off-ordinal"])
    n_out = len([r for r in rows if r.kind == "out-of-class"])
    pen = subsample_penalty(rows)

    ylabels = [_ylabel(r) for r in rows]
    head = "Field audit on a clean fine ordinal (zebrafish GSE106474, 12 stages)"
    sub = (f"The full audit behind Figure 2, which draws the {len(meas)} measured rows "
           f"only. Every score considered has a row: {len(meas)} measured, {n_blocked} "
           f"blocked, {n_off} measured on another ordinal, {n_out} outside the audited "
           f"class, {len(band)} never measured.")
    xlab = "conditional skill beyond the declared primitive   (Kendall tau_b)"
    foot = ""
    if any(r.subsample for r in meas):
        n_sub = min(r.n for r in meas if r.subsample)
        detail = (f"{pen[0]:.2f}–{pen[1]:.2f}× the {pen[2]:,}-cell floor at the same rho"
                  if pen else "a higher floor")
        foot = (f"‡ measured on the shared seed-42 {n_sub:,}-cell subsample; a smaller n "
                f"raises the floor, so these rows face {detail}")

    n_txt = [f"{r.n:,}" + (" ‡" if r.subsample else "") if r.kind == "measured" else "—"
             for r in rows]
    floor_txt = [(f"{r.floor:+.4f}" if r.floor is not None else "—")
                 if r.kind == "measured" else "" for r in rows]
    verdict_txt = [("yes" if r.clears else ("no — negative" if r.tau <= 0 else "no"))
                   if r.kind == "measured" else _KIND_VERDICT[r.kind] for r in rows]

    # ---- measure first: the table columns, the y labels, and how many lines
    # each reason wraps to. Only then is the figure's height known.
    pm = _TextMetrics()
    LEFT_IN = max(pm.width(t, pt_tick) for t in ylabels) + 0.07
    C_N = 0.02
    C_FLOOR = C_N + max(pm.width(t, pt_table) for t in n_txt) + 0.12 \
        + max([pm.width(t, pt_table) for t in floor_txt if t] or [0.0])
    C_VERDICT = C_FLOOR + 0.09
    TABLE_IN = C_VERDICT + max(pm.width(t, pt_table) for t in verdict_txt) + 0.02
    TABLE_IN = max(TABLE_IN, pm.width("its own floor", pt_table) + C_N + 0.30)
    axes_w = W - LEFT_IN - TABLE_IN - GAP_IN - RIGHT_IN

    wrap_w = axes_w - STUB_IN - 0.06
    wrapped, heights = {}, []
    for r in rows:
        if r.kind == "measured":
            heights.append(MEAS_ROW_H)
            continue
        text = (r.reason if r.kind == "by-construction"
                else f"{_KIND_LEAD[r.kind]} — {r.reason}")
        lines = pm.wrap(text, wrap_w, pt_reason)
        wrapped[r.score] = lines
        base = BAND_ROW_H if r.kind == "by-construction" else MEAS_ROW_H
        heights.append(max(base, len(lines) * _line_in(pt_reason, 1.18) + 0.075))
    label_w = max(pm.width(f"{r.tau:+.3f}", pt_value) for r in meas) if meas else 0.30
    run_w = W - 2 * TEXT_MARGIN_IN
    head_lines = pm.wrap(head, run_w, pt_head)
    sub_lines = pm.wrap(sub, run_w, pt_sub)
    xlab_lines = pm.wrap(xlab, run_w, pt_xlab)
    foot_lines = pm.wrap(foot, run_w, pt_foot) if foot else []
    pm.close()

    HEAD_IN = (0.06 + len(head_lines) * _line_in(pt_head, 1.25)
               + 0.02 + len(sub_lines) * _line_in(pt_sub, 1.30) + 0.04)
    TBLHEAD_IN = _line_in(pt_table, 1.35) + 0.03
    XAXIS_IN = 0.20 + len(xlab_lines) * _line_in(pt_xlab, 1.30)
    FOOT_IN = (0.06 + len(foot_lines) * _line_in(pt_foot, 1.25)) if foot_lines else 0.0
    LEG_IN = 4 * _line_in(pt_leg, 1.55) + 0.04         # eight entries, two columns
    AX_H = sum(heights)
    H = HEAD_IN + TBLHEAD_IN + AX_H + XAXIS_IN + FOOT_IN + LEG_IN

    fig = plt.figure(figsize=(W, H))
    xlo, xhi = _xlim_for(meas, axes_w, label_w) if meas else (-0.32, 0.80)
    x_in = lambda inches: xlo + inches / axes_w * (xhi - xlo)    # noqa: E731
    w_in = lambda inches: inches / axes_w * (xhi - xlo)          # noqa: E731

    bottom = (LEG_IN + FOOT_IN + XAXIS_IN) / H
    ax = fig.add_axes([LEFT_IN / W, bottom, axes_w / W, AX_H / H])
    axt = fig.add_axes([(LEFT_IN + axes_w + GAP_IN) / W, bottom,
                        TABLE_IN / W, (AX_H + TBLHEAD_IN) / H])
    axt.set_xlim(0, TABLE_IN)
    axt.set_ylim(0, AX_H + TBLHEAD_IN)
    axt.axis("off")

    y, cursor = [], AX_H
    for h in heights:
        y.append(cursor - h / 2.0)
        cursor -= h

    for yi, h, r in zip(y, heights, rows):
        if r.kind == "by-construction":
            ax.add_patch(mp.Rectangle((xlo, yi - h / 2), xhi - xlo, h,
                                      facecolor="#f2ecf7", edgecolor="none", zorder=0))
            ax.text(x_in(0.06), yi, "\n".join(wrapped[r.score]), va="center",
                    fontsize=pt_reason, color=BANDC, style="italic", linespacing=1.18,
                    zorder=3)
        elif r.kind == "measured":
            _draw_bar_row(ax, r, yi, BAR_H, pt_value)
        else:
            # a marker, not a value: it sits at the left edge of the axes rather
            # than at zero, so it cannot be read as a bar of some magnitude.
            c = _KIND_COLOR[r.kind]
            ax.barh(yi, w_in(STUB_IN), left=xlo, height=BAR_H, color="none",
                    edgecolor=c, hatch=_KIND_HATCH[r.kind], lw=0.8, zorder=3)
            ax.text(x_in(STUB_IN + 0.05), yi, "\n".join(wrapped[r.score]), va="center",
                    fontsize=pt_reason, color=c, style="italic", linespacing=1.18,
                    zorder=3)

    ax.axvline(0, color="#888", lw=0.8, zorder=2)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=pt_tick, linespacing=1.15)
    for tick, r in zip(ax.get_yticklabels(), rows):
        if r.kind in _KIND_COLOR:
            tick.set_color(_KIND_COLOR[r.kind])
        if r.kind == "by-construction":
            tick.set_fontstyle("italic")
    ax.set_ylim(0, AX_H)
    ax.set_xlim(xlo, xhi)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", labelsize=pt_xtick, pad=1.5)
    ax.tick_params(axis="y", length=0, pad=2)

    # ---- per-row column: n, this row's own floor, the verdict in words
    for yi, r, nt, ft, vt in zip(y, rows, n_txt, floor_txt, verdict_txt):
        c = CLEARS if (r.kind == "measured" and r.clears) else (
            FAILS if r.kind == "measured" else _KIND_COLOR[r.kind])
        axt.text(C_N, yi, nt, fontsize=pt_table, va="center", ha="left",
                 color=(FLOORC if r.subsample else "#333") if r.kind == "measured" else c)
        if ft:
            axt.text(C_FLOOR, yi, ft, fontsize=pt_table, va="center", ha="right",
                     color=FLOORC)
        axt.text(C_VERDICT, yi, vt, fontsize=pt_table, va="center", ha="left", color=c,
                 style="normal" if r.kind == "measured" else "italic",
                 fontweight="bold" if (r.kind == "measured" and r.clears) else "normal")
    for xpos, txt, ha in ((C_N, "n", "left"), (C_FLOOR, "its own floor", "right"),
                          (C_VERDICT, "clears it?", "left")):
        axt.text(xpos, AX_H + 0.035, txt, fontsize=pt_table, fontweight="bold", ha=ha,
                 va="bottom")

    xm = TEXT_MARGIN_IN / W
    fig.text(xm, 1.0 - 0.045 / H, "\n".join(head_lines), fontsize=pt_head,
             ha="left", va="top", linespacing=1.25)
    fig.text(xm, 1.0 - (0.06 + len(head_lines) * _line_in(pt_head, 1.25) + 0.02) / H,
             "\n".join(sub_lines), fontsize=pt_sub, ha="left", va="top", color="#555",
             linespacing=1.30)
    fig.text(0.5, (LEG_IN + FOOT_IN + 0.035) / H, "\n".join(xlab_lines),
             fontsize=pt_xlab, ha="center", va="bottom", linespacing=1.30)
    if foot_lines:
        fig.text(xm, (LEG_IN + 0.030) / H, "\n".join(foot_lines), fontsize=pt_foot,
                 color=FLOORC, style="italic", ha="left", va="bottom", linespacing=1.25)

    handles = [
        mp.Patch(facecolor=CLEARS, edgecolor="#333", lw=0.5,
                 label="clears its own null floor"),
        mp.Patch(facecolor="none", edgecolor=BLOCKC, hatch="///",
                 label="blocked — no run artefact; reason on the row"),
        mp.Patch(facecolor=FAILS, edgecolor="#333", lw=0.5,
                 label="does not clear its own null floor"),
        mp.Patch(facecolor="none", edgecolor=OFFC, hatch="\\\\",
                 label="measured, but on another ordinal"),
        plt.Line2D([0], [0], color=FLOORC, lw=1.5, ls=(0, (2.0, 1.3)),
                   label="that row's floor: 97.5th pct of its own null, at its n and rho"),
        mp.Patch(facecolor="none", edgecolor=OUTC, hatch="...",
                 label="outside the unsupervised class audited here"),
        plt.Line2D([0], [0], color="#111", lw=1.0,
                   label="95% bootstrap interval — sampling precision, not the verdict"),
        mp.Patch(facecolor="#f2ecf7", edgecolor="none",
                 label="by construction, never measured — no value"),
    ]
    fig.legend(handles=handles, fontsize=pt_leg, loc="lower left", frameon=False,
               bbox_to_anchor=(0.006, 0.018 / H), ncol=2, handlelength=2.1,
               columnspacing=1.4, labelspacing=0.35, handletextpad=0.45,
               borderpad=0.0, borderaxespad=0.0)

    out = OUT / "fig2_appendix.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"fig2_appendix ok -> {out}")
    return fs



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
#
# Retargeted to TEXT_W_IN, i.e. \begin{figure*}, not to a column. The two panels
# are a matched pair — the same measurement on two atlases, and the argument is
# the contrast between them — so they are read side by side, and side by side at
# COLUMN_W_IN gives each panel about 1.06 in of plotting width. Its own title
# needs 2.0 in at 8pt and its two x tick labels need 1.40 in at 7pt, so the
# labels would collide with each other before the title even started to fit.
# fig3_column_fit() measures exactly that and prints it, so the choice is a
# number rather than a taste. Stacking the two panels in one column would fit,
# but costs about 3.6 in of column height in a four-page paper and turns a pair
# into a sequence. Nothing here is shrunk below MIN_PT to make it fit.

FIG3 = [
    dict(panel="a", title="Hematopoietic (sorted)", art="w4/c1_gse117498_results.json",
         hi=("HSC", "potency high"), lo=("GMP", "potency low"), color="#c8709a"),
    dict(panel="b", title="Intestinal", art="w4/c2_gse125970_results.json",
         hi=("Stem Cell", "potency high"), lo=("Enterocyte", "potency low"), color="#2f7f4f"),
]

# per-panel furniture, in inches, the same in either candidate layout
F3_YAXIS_IN = 0.50        # rotated y label + y tick labels
F3_GUTTER_IN = 0.12       # between a panel and the next panel's y axis


def _fig3_load():
    """(panels, missing) — every number on Figure 3, from the run record."""
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
    return panels, missing


def fig3_column_fit(panels, pt_title=8.0, pt_tick=MIN_PT):
    """
    Would the two panels fit side by side in one column without going below
    MIN_PT? Measured in the figure font, not assumed. Returns (fits, lines).
    """
    import matplotlib
    matplotlib.use("Agg")

    m = _TextMetrics()
    n = max(len(panels), 1)
    avail = (COLUMN_W_IN - n * F3_YAXIS_IN - (n - 1) * F3_GUTTER_IN) / n
    lines = [f"  fig3 side-by-side at {COLUMN_W_IN:.2f} in: {avail:.2f} in of plotting "
             f"width per panel"]
    fits = True
    for spec, g in panels:
        title = f"{spec['title']}  (n = {g['n']:,})"
        w_title = m.width(title, pt_title)
        w_ticks = sum(m.width(f"({lab})", pt_tick) for lab in (spec["hi"][1], spec["lo"][1]))
        fits &= (w_title <= avail and w_ticks <= avail)
        lines.append(f"    panel {spec['panel']}: title {w_title:.2f} in at {pt_title:g}pt, "
                     f"the two x tick labels {w_ticks:.2f} in at {pt_tick:g}pt "
                     f"-> {'fits' if w_title <= avail and w_ticks <= avail else 'does not fit'}")
    m.close()
    lines.append(f"    verdict: {'fits' if fits else 'does not fit'} in one column "
                 f"side by side at >= {MIN_PT:g}pt -> emitted at "
                 f"{COLUMN_W_IN if fits else TEXT_W_IN:.2f} in")
    return fits, lines


def make_fig3():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels, missing = _fig3_load()
    if missing:
        for x in missing:
            print(f"  fig3 BLOCKED: {x}", file=sys.stderr)
    if not panels:
        print("fig3 skipped — no w4 artefacts in the run record", file=sys.stderr)
        return missing, None, None

    fits, fit_lines = fig3_column_fit(panels)
    for line in fit_lines:
        print(line, file=sys.stderr)

    fs = FontLedger("fig3")
    pt_letter = fs(10.0, "panel letters")
    pt_title = fs(8.0, "panel titles")
    pt_ylab = fs(7.5, "y axis labels")
    pt_ytick = fs(7.0, "y tick labels")
    pt_xtick = fs(7.0, "x tick labels")
    pt_bar = fs(7.0, "bar values")
    pt_note = fs(7.0, "per-panel notes")
    pt_foot = fs(7.0, "source footnote")
    fs.enforce()

    W = TEXT_W_IN                       # \begin{figure*}: see fig3_column_fit above
    n = len(panels)
    TOP_IN, XTICK_IN, FOOT_IN = 0.38, 0.30, 0.20
    AX_H = 1.62
    H = TOP_IN + AX_H + XTICK_IN + FOOT_IN
    cell_w = W / n
    ax_w = cell_w - F3_YAXIS_IN - F3_GUTTER_IN

    fig = plt.figure(figsize=(W, H))
    m = _TextMetrics()
    for i, (spec, g) in enumerate(panels):
        left = (i * cell_w + F3_YAXIS_IN) / W
        ax = fig.add_axes([left, (XTICK_IN + FOOT_IN) / H, ax_w / W, AX_H / H])
        ax.text(-F3_YAXIS_IN / ax_w, 1.085, spec["panel"], transform=ax.transAxes,
                fontsize=pt_letter, fontweight="bold", va="bottom")
        vals = [g["hi_genes"], g["lo_genes"]]
        inverted = vals[1] > vals[0]
        col = "#b00000" if inverted else "#008000"
        # The direction phrase leads the note instead of labelling the arrow: on a
        # panel this size a label sitting on the arrow masks a bar's own value.
        note = ((f"genes up as potency down: gene-count AUROC {g['auroc']:.3f}, below "
                 f"chance. The naive 'score beats primitive' test then reads CT marginal "
                 f"delta = {g['delta']:+.3f} as a pass") if inverted else
                (f"genes down as potency down: gene-count AUROC {g['auroc']:.3f}, "
                 f"direction correct"))
        note_lines = m.wrap(note, ax_w * 0.97, pt_note)
        # headroom is solved, not guessed: the note block and the direction arrow
        # both have to sit above the taller bar without meeting each other.
        head_frac = (len(note_lines) * _line_in(pt_note, 1.25) + 0.10) / AX_H
        LIFT_FRAC = 0.12
        ymax = max(vals) / max(0.35, 1.0 - head_frac - LIFT_FRAC)
        ax.bar([0, 1], vals, 0.52, color=[spec["color"], "#b0b0b0"],
               edgecolor="#555", linewidth=0.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([f"{spec['hi'][0]}\n({spec['hi'][1]})",
                            f"{spec['lo'][0]}\n({spec['lo'][1]})"], fontsize=pt_xtick,
                           linespacing=1.2)
        ax.set_ylabel("median detected genes", fontsize=pt_ylab, labelpad=2)
        ax.set_ylim(0, ymax)
        ax.set_xlim(-0.70, 1.70)
        ax.tick_params(axis="y", labelsize=pt_ytick, pad=1.5)
        ax.tick_params(axis="x", length=0, pad=1.5)
        ax.spines[["top", "right"]].set_visible(False)
        for v, x in zip(vals, (0, 1)):
            ax.text(x, v + 0.015 * ymax, f"{v:,.1f}", ha="center", va="bottom",
                    fontsize=pt_bar, color="#333")

        lift = LIFT_FRAC * ymax
        ax.annotate("", xy=(1, vals[1] + lift), xytext=(0, vals[0] + lift),
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
        ax.text(0.015, 0.99, "\n".join(note_lines), transform=ax.transAxes,
                fontsize=pt_note, va="top", color=col, linespacing=1.25)
        ax.set_title(f"{spec['title']}  (n = {g['n']:,})", fontsize=pt_title, pad=3)

    fig.text(0.5, 0.035, "medians and AUROC read from the released run record; Kendall tau_b",
             ha="center", va="bottom", fontsize=pt_foot, color="#555", style="italic")
    m.close()
    out = OUT / "fig3.pdf"
    fig.savefig(out)
    plt.close(fig)
    print(f"fig3 ok -> {out}")
    return missing, fs, fit_lines


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
    # what was emitted at what width, so the widths can be checked rather than
    # trusted: (path, the width it was targeted at, its font ledger)
    emitted = []
    if "1" in want:
        make_fig1()
    if "2" in want:
        fs_body = make_fig2_body(rows)
        fs_app = make_fig2_appendix(rows)
        emitted.append((OUT / "fig2_body.pdf", COLUMN_W_IN, fs_body))
        emitted.append((OUT / "fig2_appendix.pdf", TEXT_W_IN, fs_app))
    if "3" in want:
        _, fs_fig3, _ = make_fig3()
        if fs_fig3 is not None:
            emitted.append((OUT / "fig3.pdf", TEXT_W_IN, fs_fig3))
    if "4" in want:
        make_fig4(rows)

    caps = {"fig2_body_caption.txt": caption(rows, "body"),
            "fig2_appendix_caption.txt": caption(rows, "appendix")}
    for name, text in caps.items():
        (OUT / name).write_text(text + "\n")
        print(f"caption -> {OUT/name}")

    if emitted:
        print("\nrendered size, read back from each PDF's own MediaBox:")
        for path, target, ledger in emitted:
            if path.is_file():
                print(report_width(path, target, ledger))

    table = provenance(rows, grid)
    if args.provenance:
        print("\n" + table)
    if args.provenance_out:
        Path(args.provenance_out).write_text(table + "\n")
        print(f"provenance -> {args.provenance_out}")

    (OUT / "fig2_values.json").write_text(json.dumps({
        "kernel": "Kendall tau_b",
        "wtau_note": "weighted tau (rank=True) is never plotted; its value and floor are "
                     "carried per row only so the caption can name the tau_b verdicts a "
                     "both-kernel rule would not return. Source: the wtau block of "
                     + REPORT_TXT.name,
        "decision_rule": "value > 97.5th percentile of the estimator's measured null at "
                         "the value's own n and its own score-primitive rank correlation",
        "run_record": _rel(run_record()),
        "null_grid": _rel(NULL_RESULTS),
        "artefacts": {a.rel: a.sha256 for a in Artefact._cache.values()},
        "rows": [dict(kind=r.kind, score=r.score, primitive=r.primitive, tau=r.tau,
                      ci95=[r.lo, r.hi] if r.lo is not None else None, n=r.n, rho=r.rho,
                      null_mean=r.floor_mean, null_p97_5=r.floor, null_cell=r.floor_cell,
                      null_files=r.floor_files, clears_own_floor=r.clears,
                      wtau=r.wtau, wtau_null_mean=r.wtau_mean, wtau_null_p97_5=r.wtau_floor,
                      clears_own_wtau_floor=r.clears_wtau,
                      subsample_3000=r.subsample, reason=r.reason, sources=r.sources)
                 for r in rows],
        "caption_body": caps["fig2_body_caption.txt"],
        "caption_appendix": caps["fig2_appendix_caption.txt"],
        "appendix_figure_label": APPENDIX_FIG,
        "target_widths_in": {"fig2_body.pdf": COLUMN_W_IN,
                             "fig2_appendix.pdf": TEXT_W_IN,
                             "fig3.pdf": TEXT_W_IN},
    }, indent=2) + "\n")
    print(f"values -> {OUT/'fig2_values.json'}")

    if args.crosscheck:
        print("\n" + crosscheck(rows))

    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
