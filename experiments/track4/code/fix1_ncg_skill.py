"""
Paper A FIX 1 — own-baseline conditional skill of NCG (and MCE if a real score
vector exists) vs its declared §3a primitive PCC(x,degree) on the zebrafish
12-stage ordinal, both kernels, bootstrap 95% CI seed 42, + log-lib robustness.
Reuses the decider's exact kernel/residual functions. Runs ONLY on score vectors
produced by a REAL implementation (ncg_scores.npz / mce_scores.npz); missing →
reported INCONCLUSIVE-infra by the results writeup (this script just skips).

Output: track4/results/fix1_mce_ncg_skill.json
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# scratch directory, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root, repo_root  # noqa: E402
CODE = repo_root()   # this repository (code)
VAULT = data_root()  # run inputs and outputs (data)
sys.path.insert(0, str(CODE / "own_baseline" / "atlas_run"))
sys.path.insert(0, str(CODE / "experiments" / "track2" / "code"))
from conditional_skill import scipy_wtau, kang_taub, rank_resid_multi, _verdict, align, SEED, N_BOOT_WT, N_BOOT_KT

PREP = VAULT / "track2/data/prepared"
RES = VAULT / "track2/results"
OUT = VAULT / "track4/results/fix1_mce_ncg_skill.json"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _boot(kfun, s, covs, o, nb):
    real = kfun(rank_resid_multi(s, covs), o)
    rng = np.random.default_rng(SEED)
    n = len(o)
    b = np.empty(nb)
    for j in range(nb):
        i = rng.integers(0, n, n)
        b[j] = kfun(rank_resid_multi(s[i], [c[i] for c in covs]), o[i])
    lo, hi = np.nanquantile(b, [0.025, 0.975])
    return {"tau": float(real), "CI95": [float(lo), float(hi)], "n_boot": nb, "verdict": _verdict(lo, hi)}


def both(s, covs, o):
    k1 = _boot(scipy_wtau, s, covs, o, N_BOOT_WT)
    k2 = _boot(kang_taub, s, covs, o, N_BOOT_KT)
    ag = k1["verdict"] == k2["verdict"]
    return {"scipy_weightedtau": k1, "kang_wdm_taub": k2, "kernels_agree": ag,
            "combined_verdict": k1["verdict"] if ag else "INCONCLUSIVE-kernel-disagree"}


def main():
    p = np.load(PREP / "primitives.npz")
    pcc, rk = p["pcc_string_v12"], p["rank_kimmel"].astype(float)
    import csv as _csv
    loglib = np.log10(np.array([float(r["lib_size"]) for r in
                                _csv.DictReader(open(PREP / "cells.tsv"), delimiter="\t")]))
    out = {"seed": SEED, "n_boot": {"scipy_weightedtau": N_BOOT_WT, "kang_wdm_taub": N_BOOT_KT},
           "declared_primitive": "PCC(x,degree) STRING v12 LCC", "scores": {}}

    for nm, fn, key in [("NCG", "ncg_scores.npz", "ncg"), ("MCE", "mce_scores.npz", "mce")]:
        f = RES / fn
        if not f.exists():
            out["scores"][nm] = {"status": "INCONCLUSIVE-infra (no real implementation ran)"}
            print(f"{nm}: INCONCLUSIVE-infra (no {fn})")
            continue
        d = np.load(f)
        v = d[key]
        ci = d["cell_idx"] if "cell_idx" in d.files else np.arange(len(rk))
        s = align(v, rk[ci])
        pcc_s, o, ll = pcc[ci], rk[ci], loglib[ci]
        single = both(s, [pcc_s], o)                    # NCG | PCC(x,degree)
        robust = both(s, [pcc_s, ll], o)                # + log-lib
        out["scores"][nm] = {
            "n": int(len(ci)),
            "spearman_score_vs_PCC": float(spearmanr(v, pcc_s).statistic),
            "spearman_score_vs_ordinal": float(spearmanr(v, o).statistic),
            "conditional_skill_vs_PCC": single,
            "conditional_skill_vs_PCC_plus_loglib": robust,
        }
        a, b = single["scipy_weightedtau"], single["kang_wdm_taub"]
        print(f"{nm}: n={len(ci)} rho(NCG,PCC)={out['scores'][nm]['spearman_score_vs_PCC']:+.3f} | "
              f"cond|PCC scipy {a['tau']:+.3f}{a['CI95']} kang {b['tau']:+.3f}{b['CI95']} "
              f"=> {single['combined_verdict']} | +loglib => {robust['combined_verdict']}")

    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
