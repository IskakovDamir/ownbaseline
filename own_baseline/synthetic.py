"""
synthetic.py — pre-registered synthetic validation of the own-entropy diagnostic.

Design (ADEMP; Morris, White & Crowther 2019)
---------------------------------------------
Two error-model features per domain:
    f_ent : an entropy-type feature (the Bayes-error proxy)
    f_str : a structural feature (beyond-Bayes error structure)

Errors are Bernoulli with rate sigmoid(a * f_ent + b * f_str_eff), where the
entropy coefficient `a` is SHARED across domains (the tautological component),
and the structural axis is only partially shared:

    f_str_eff = align * f_str + sqrt(1 - align**2) * f_str_alt

with f_str_alt an independent draw. `align` in [0, 1] sets how much of the
structural error relationship is conserved between domains:
    align = 0  -> structural component domain-specific
    align = 1  -> structural component fully conserved

Estimand
--------
    Delta = AUROC_transfer(A -> B)  -  AUROC_own-entropy(B)

Pre-registered prediction
-------------------------
    Delta ~ 0 at align = 0  (only the tautological component is shared),
    Delta strictly increasing in align,
    Delta > 0 at align = 1.

If this holds, the diagnostic (a) reads 0 when only the tautology is shared and
(b) has power to detect genuine shared structure -> an observed Delta ~ 0 on
real data is evidence of domain-specific structure, not lack of sensitivity.

Note on coefficients
--------------------
`a`, `b`, `n`, and `seeds` set the absolute magnitude of Delta but not the
qualitative result. The values below are illustrative; the qualitative
behaviour (Delta ~ 0 at align = 0, rising with align) is the point. The default
b = 1.1 is chosen so align = 1 gives Delta ~ 0.107, matching the paper's
registered run.
"""
from __future__ import annotations
import numpy as np

from own_baseline import transfer_auc, own_entropy_baseline

__all__ = ["make_domain", "align_sweep"]


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def make_domain(n: int, a: float, b: float, align: float, rng: np.random.Generator) -> dict:
    """Draw one synthetic domain with the given entropy/structural coupling."""
    f_ent = rng.standard_normal(n)
    f_str = rng.standard_normal(n)
    f_alt = rng.standard_normal(n)
    f_str_eff = align * f_str + np.sqrt(max(0.0, 1.0 - align ** 2)) * f_alt
    p = _sigmoid(a * f_ent + b * f_str_eff)
    y_err = (rng.random(n) < p).astype(int)
    X = np.column_stack([f_ent, f_str])          # gene-invariant error-model features
    return {"f_ent": f_ent, "f_str": f_str, "X": X, "y": y_err}


def align_sweep(aligns=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
                n: int = 4000, a: float = 1.0, b: float = 1.1, seeds: int = 20) -> dict:
    """
    Run the sweep. Returns {align: (mean_delta, mc_se)} averaged over `seeds`.

    For each seed, domain A and domain B are drawn independently (different
    seeds) with the SAME (a, b, align), so only the shared `a` term and the
    aligned fraction of the structural axis carry cross-domain information.
    """
    out = {}
    for al in aligns:
        deltas = []
        for s in range(seeds):
            rngA = np.random.default_rng(s)
            rngB = np.random.default_rng(10_000 + s)
            A = make_domain(n, a, b, al, rngA)
            B = make_domain(n, a, b, al, rngB)
            t = transfer_auc(A["X"], A["y"], B["X"], B["y"])
            o = own_entropy_baseline(B["f_ent"], B["y"])
            deltas.append(t - o)
        deltas = np.asarray(deltas, float)
        out[al] = (float(deltas.mean()), float(deltas.std() / np.sqrt(seeds)))
    return out


if __name__ == "__main__":
    res = align_sweep()
    print(f"{'align':>6} {'mean_delta':>12} {'mc_se':>8}")
    for al, (m, se) in res.items():
        print(f"{al:6.2f} {m:12.4f} {se:8.4f}")
