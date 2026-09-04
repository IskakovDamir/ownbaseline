"""
The measured null floor of the conditional-skill estimator.

A floor is the 97.5th percentile of the estimator's own null distribution at a
given sample size, score-primitive rank correlation, kernel and covariate count.
It is not zero and it is not constant: it rises as n falls and it varies with
rho, so a value is read against its own cell and never against a single line
drawn across a figure.

The shipped table is `data/null_floors.csv.gz`, built from the 200-seed grid in
`experiments/null_calibration/`. Cells that were run in more than one batch have
their seeds pooled and the percentile recomputed on the pooled sample; where the
raw per-seed values were not kept, the largest replicate estimate is used,
because the smaller of two independent estimates of a decision threshold biases
toward declaring that a score clears it.

Cell selection reproduces `experiments/null_calibration/report.py`: Null B, align
on, twelve ordinal levels, nearest n and nearest |rho| among the grid's own design
points, and the maximum over the coupling strengths measured at that cell.
"""
from __future__ import annotations

import csv
import gzip
from functools import lru_cache
from pathlib import Path

TABLE = Path(__file__).with_name("data") / "null_floors.csv.gz"

KERNELS = {"taub": "kendalltau",
           "weighted": "weightedtau_rankFalse",
           "weighted-ranked": "weightedtau_rankTrue"}


@lru_cache(maxsize=1)
def _rows():
    if not TABLE.is_file():
        raise FileNotFoundError(
            f"the shipped null grid is missing at {TABLE}. Rebuild it with "
            f"experiments/null_calibration/consolidate.py, or pass --no-floors "
            f"to report conditional skill without a decision threshold.")
    with gzip.open(TABLE, "rt", newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


@lru_cache(maxsize=1)
def grid():
    """The design points the table actually holds, per covariate count."""
    rows = _rows()
    out = {}
    for k in sorted({int(r["k_covariates"]) for r in rows}):
        sub = [r for r in rows if int(r["k_covariates"]) == k]
        out[k] = {"n": sorted({int(r["n"]) for r in sub}),
                  "rho": sorted({float(r["rho_target"]) for r in sub}),
                  "levels": sorted({int(r["levels"]) for r in sub})}
    return out


def _nearest(v, points):
    return min(points, key=lambda g: abs(g - v))


def floor(n, rho, kernel="taub", k_covariates=1, levels=12):
    """
    The decision threshold for a conditional-skill value measured at (n, rho).

    Cell selection is the manuscript's: nearest measured n, nearest measured
    |rho|, maximum over the coupling strengths at that cell. Two things the
    manuscript never had to handle, because every value it reports sits on a
    design point, are handled here and labelled.

    n away from a design point. The floor falls monotonically with n across the
    whole measured range, 0.0741 at 500 down to 0.0228 at 127,607 for tau_b at
    rho = 0.5. So a grid point BELOW the query gives a floor that is too high,
    which is conservative: a score clearing it would clear the true one. A grid
    point ABOVE the query gives a floor that is too low, and a score can clear it
    without clearing the true one. The result says which case you are in.

    k_covariates away from a design point. The grid measured 1, 2 and 4, and the
    floor is not monotone in k: it mostly falls, and rises again at rho = 0 and
    rho = 0.998. There is no safe direction to round in, so an unmeasured k takes
    the maximum over the measured values that bracket it, which is conservative
    whichever way the true value moves.

    Never extrapolates: outside the grid's n or rho range it returns floor=None
    and names the cell that would have to be run.
    """
    kname = KERNELS.get(kernel, kernel)
    pool = [r for r in _rows()
            if r["null"] == "B" and int(r["align"]) == 1
            and r["kernel"] == kname and int(r["levels"]) == levels]
    if not pool:
        return {"floor": None, "reason": (
            f"the shipped grid has no cell at kernel={kname}, levels={levels}")}

    ks = sorted({int(r["k_covariates"]) for r in pool})
    if k_covariates in ks:
        k_used, bracket = [k_covariates], False
    else:
        below = [k for k in ks if k < k_covariates]
        above = [k for k in ks if k > k_covariates]
        if not (below and above):
            return {"floor": None, "reason": (
                f"the grid measured {k_covariates=} nowhere and does not bracket "
                f"it either; it holds {ks}. Run the covariate sweep at "
                f"k={k_covariates} rather than rounding to one of those.")}
        k_used, bracket = [below[-1], above[0]], True

    rows = [r for r in pool if int(r["k_covariates"]) in k_used]
    ns = sorted({int(r["n"]) for r in rows})
    rhos = sorted({float(r["rho_target"]) for r in rows})
    r_abs = abs(float(rho))
    if not (ns[0] <= n <= ns[-1]):
        return {"floor": None, "reason": (
            f"n={n:,} is outside the measured grid, which runs {ns[0]:,} to "
            f"{ns[-1]:,}. Extrapolating a floor would assert a number nobody "
            f"measured; run the grid at n={n:,} instead.")}
    if not (rhos[0] <= r_abs <= rhos[-1]):
        return {"floor": None, "reason": (
            f"|rho|={r_abs:.3f} is outside the measured grid, which runs "
            f"{rhos[0]:.3f} to {rhos[-1]:.3f}.")}

    ng, rg = _nearest(n, ns), _nearest(r_abs, rhos)
    cells = [r for r in rows if int(r["n"]) == ng
             and abs(float(r["rho_target"]) - rg) < 1e-9]
    p975 = max(float(c["p97.5"]) for c in cells)
    mean = sum(float(c["mean"]) for c in cells) / len(cells)
    seeds = sum(int(c["n_seeds"]) for c in cells)

    kdesc = (f"{k_used[0]} covariate" + ("s" if k_used[0] > 1 else "")
             if not bracket else
             f"k={k_covariates} bracketed by the measured {k_used[0]} and "
             f"{k_used[1]}, larger floor taken")
    if ng == n:
        n_side = "exact"
    elif ng < n:
        n_side = "conservative"      # the floor is too high, clearing it is safe
    else:
        n_side = "permissive"        # the floor is too low, clearing it proves less

    return {
        "floor": p975,
        "null_mean": mean,
        "cell": (f"Null B, align on, {levels} levels, n={ng:,}, rho={rg:g}, "
                 f"{kname}, {kdesc}, {len(cells)} coupling strength"
                 f"{'s' if len(cells) > 1 else ''}, {seeds:,} seeds"),
        "grid_n": ng, "grid_rho": rg,
        "n_offset": n - ng, "rho_offset": r_abs - rg,
        "n_side": n_side,
        "k_bracketed": bracket, "k_used": k_used,
        "distant": abs(n - ng) > 0.5 * ng or abs(r_abs - rg) > 0.15,
        "kernel": kname, "k_covariates": k_covariates, "levels": levels,
    }
