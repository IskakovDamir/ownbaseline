"""
ownbaseline: does this potency score order cells beyond the statistic it is
closest to?

Four verbs.

    ownbaseline check       run the test on your data
    ownbaseline floors      the measured null floor at a given n, rho, kernel
    ownbaseline primitives  compute the low-order statistics from a matrix
    ownbaseline verify      re-run a receipt and report whether it reproduces

`check` wraps the conditional-skill estimator, not the marginal gap. The marginal
gap is `run_own_baseline` in the library: it compares two correlations without
residualizing, it returns a pass when a score is indistinguishable from gene
counts, and on sorted haematopoietic progenitors it returns the exact false
positive this tool exists to expose. The CLI does not offer it.

Three defaults exist to make the correct usage the easy one.

  * All four primitives are computed unless you name one. A user who does not
    know which statistic their score reduces to gets the full answer instead of
    a blank.
  * `--ordinal-source` is required and has no override. A pseudotime computed
    from the same expression matrix is not a valid ordinal for this test, and one
    required word prevents the most common way it would be misused.
  * Kendall tau_b is the kernel. The weighted kernel is available behind a flag,
    with a warning: its null mean is not zero, it rises with sample size from
    0.188 at n = 500 to 0.309 at n = 127,607, and it returned a positive verdict
    on 720 of 720 null replicates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from . import floors as _floors
from . import style as _s
from .conditional_skill import align, kang_taub, rank_resid_multi, scipy_wtau

__all__ = ["main"]

PRIMITIVE_NAMES = {
    "gene-count": "gene_count",
    "pcc-degree": "PCC(x,degree)",
    "entropy": "Shannon_H",
    "log-libsize": "log10_library_size",
}
NPZ_KEY = {"gene_count": "gene_count",
           "PCC(x,degree)": "pcc_degree",
           "Shannon_H": "shannon_H",
           "log10_library_size": "log10_library_size"}
KEY_NPZ = {v: k for k, v in NPZ_KEY.items()}

PSEUDOTIME_HINTS = ("dpt", "pseudotime", "latent_time", "palantir", "velocity",
                    "diffusion_time", "monocle")


# ------------------------------------------------------------------ small IO

def _die(msg, code=2):
    print(f"ownbaseline: {msg}", file=sys.stderr)
    raise SystemExit(code)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_array(a):
    b = np.ascontiguousarray(np.asarray(a, dtype=np.float64))
    return hashlib.sha256(b.tobytes()).hexdigest()


def _read_vector(spec, adata, what):
    """
    spec is one of:
        obs:name        a column of the AnnData .obs
        path.csv        one numeric column, optional header
        path.npy        a 1-D array
        pkg.mod:callable  a dotted path to something called with the AnnData
    """
    if spec.startswith("obs:"):
        if adata is None:
            _die(f"--{what} {spec!r} selects an .obs column, so a .h5ad file is "
                 f"needed as the first positional argument")
        col = spec[4:]
        if col not in adata.obs:
            near = ", ".join(sorted(adata.obs.columns)[:12])
            _die(f"no .obs column {col!r}. First few present: {near}")
        return np.asarray(adata.obs[col].values, dtype=float), spec

    p = Path(spec)
    if p.suffix == ".npy" and p.is_file():
        v = np.load(p)
        return np.asarray(v, dtype=float).ravel(), str(p.resolve())
    if p.is_file():
        rows = []
        with open(p) as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if not line:
                    continue
                tok = line.split(",")[0].strip()
                try:
                    rows.append(float(tok))
                except ValueError:
                    if i == 0:
                        continue           # a header line
                    _die(f"{p}: line {i + 1} is not a number: {tok!r}")
        return np.asarray(rows, dtype=float), str(p.resolve())

    if ":" in spec:                        # dotted path to a callable
        mod, _, fn = spec.partition(":")
        import importlib
        try:
            f = getattr(importlib.import_module(mod), fn)
        except Exception as e:             # noqa: BLE001
            _die(f"could not resolve --{what} {spec!r}: {e}")
        if adata is None:
            _die(f"--{what} {spec!r} is a callable, which is passed the AnnData; "
                 f"a .h5ad file is needed")
        return np.asarray(f(adata), dtype=float).ravel(), spec

    _die(f"--{what} {spec!r} is neither obs:<col>, an existing file, nor "
         f"module:callable")


def _load_adata(path):
    if path is None:
        return None
    try:
        import anndata
    except ImportError:
        _die("reading a .h5ad needs anndata. pip install 'own-baseline[full]', "
             "or pass --score and --ordinal as .csv or .npy files instead.")
    return anndata.read_h5ad(path)


# --------------------------------------------------------- direction check

def ordinal_auc(x, o):
    """
    Probability that a cell from a higher ordinal level scores above one from a
    lower level, ties counted half. Equals AUROC when the ordinal has two levels
    and generalises it to any number, so 0.5 is chance and a value below 0.5
    means the ranking runs the wrong way. Computed over stage blocks rather than
    over all n^2 pairs, so it is O(k^2 * n log n) in the number of levels.
    """
    x = np.asarray(x, float)
    o = np.asarray(o, float)
    m = np.isfinite(x) & np.isfinite(o)
    x, o = x[m], o[m]
    lv = np.unique(o)
    if len(lv) < 2:
        return float("nan")
    blocks = {v: np.sort(x[o == v]) for v in lv}
    num = den = 0.0
    for i, a in enumerate(lv):
        for b in lv[i + 1:]:
            xa, xb = blocks[a], blocks[b]
            na, nb = len(xa), len(xb)
            if na == 0 or nb == 0:
                continue
            lo = np.searchsorted(xa, xb, "left").sum()
            hi = np.searchsorted(xa, xb, "right").sum()
            num += lo + 0.5 * (hi - lo)     # xb above xa, ties half
            den += na * nb
    return float(num / den) if den else float("nan")


# ------------------------------------------------------------- the estimator

KERNEL_FN = {"taub": kang_taub, "weighted": scipy_wtau}


DEBRIS_RATIO = 1e6      # residual magnitudes below this times eps*n are float noise


def residual_scale(score_aligned, covs):
    """
    How far the residual sits above the arithmetic's own noise floor.

    When rank(score) equals rank(covariate), the OLS fit is exact and what
    `rank_resid_multi` returns is the rounding error of the subtraction, on the
    order of eps times the largest rank. That debris is not random: it is
    monotone in the covariate rank, so it inherits whatever association the
    covariate has with the ordinal, and a kernel scores it. Measured on a
    twelve-level staged ordinal that the covariate predicts, tau_b returns
    between 0.71 and 0.87 in magnitude on pure debris, with a sign that changes
    with the seed, and the weighted kernel between 0.62 and 0.93. On a fixture
    where the covariate is independent of the ordinal, tau_b stays near 0.02 and
    only the weighted kernel misbehaves, which is the case defect F-2 was first
    recorded on.

    Returns max|resid| divided by eps * n. Real residuals sit around 1e14 to
    1e15 on this ratio; debris sits at 1.
    """
    r = rank_resid_multi(score_aligned, covs)
    n = len(score_aligned)
    return float(np.abs(r).max() / (np.finfo(float).eps * n)), r


def conditional_skill(score_aligned, ordinal, covs, kernel, n_boot, seed):
    """
    One cell of the audit: residualize on `covs`, score the residual, bootstrap.

    Every arithmetic step is the library's own function, so this returns the
    same numbers as conditional_skill_report for the same kernel. It exists
    only so the CLI does not pay for a kernel it was not asked to report;
    tests/test_cli.py pins the two against each other.

    Adds one thing the library does not do: it refuses to return a value when
    the residual is arithmetic noise. See `residual_scale`.
    """
    ratio, resid = residual_scale(score_aligned, covs)
    if ratio < DEBRIS_RATIO:
        return {"tau": None, "CI95": [None, None], "n_boot": 0,
                "residual_scale": ratio,
                "refused": ("the residual is floating-point rounding debris: its "
                            "largest value is {:.1f} times the arithmetic noise "
                            "floor, where a real residual is around 1e14. The "
                            "score is a rank-preserving function of this "
                            "primitive, so its conditional skill is zero by "
                            "construction and any number a kernel returns here "
                            "is reading the rounding pattern.").format(ratio)}
    kfun = KERNEL_FN[kernel]
    tau = kfun(resid, ordinal)
    rng = np.random.default_rng(seed)
    n = len(ordinal)
    b = np.empty(n_boot)
    for j in range(n_boot):
        i = rng.integers(0, n, n)
        b[j] = kfun(rank_resid_multi(score_aligned[i], [c[i] for c in covs]),
                    ordinal[i])
    lo, hi = (np.nanquantile(b, [0.025, 0.975]) if n_boot
              else (float("nan"), float("nan")))
    return {"tau": float(tau), "CI95": [float(lo), float(hi)],
            "n_boot": n_boot, "residual_scale": ratio}


def _receipt(args, extra):
    def ver(mod):
        try:
            return __import__(mod).__version__
        except Exception:                                       # noqa: BLE001
            return None

    try:
        commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse",
             "--short", "HEAD"], capture_output=True, text=True,
            timeout=5).stdout.strip() or None
    except Exception:                                           # noqa: BLE001
        commit = None

    from . import __version__
    r = {"own_baseline_version": __version__,
         "git_commit": commit,
         "python": platform.python_version(),
         "numpy": ver("numpy"), "scipy": ver("scipy"),
         "seed": args.seed,
         "kernel": {"taub": "kendalltau (variant=b)",
                    "weighted": "scipy.stats.weightedtau (rank=False)"}[args.kernel],
         "estimator": "rank_resid_multi (rank, OLS with intercept)",
         "argv": sys.argv[1:],
         "primitives_from": getattr(args, "primitives", None),
         "scaffold": getattr(args, "scaffold", None) or getattr(args, "string_links", None),
         "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    r.update(extra)
    return r


# ------------------------------------------------------------------ scaffold

def load_scaffold(args, adata):
    """
    The interaction scaffold, as a dict with col_idx and degree, or None.

    Two routes. --scaffold takes a .npz already holding the column index into
    the expression matrix and the degree vector, which is what a prepared run
    carries; keys are read as col_idx or col_atlas, and degree or deg_sub.
    --string-links with --string-info builds one from a STRING release, at the
    confidence threshold --string-threshold, keeping the largest connected
    component. No interactome is bundled with the package: it is 100 MB and the
    release you pick changes every number downstream, so it is named on the
    command line and recorded in the receipt.
    """
    if getattr(args, "scaffold", None):
        z = np.load(args.scaffold)
        keys = set(z.files)
        ci = "col_idx" if "col_idx" in keys else ("col_atlas" if "col_atlas" in keys else None)
        dg = "degree" if "degree" in keys else ("deg_sub" if "deg_sub" in keys else None)
        if ci is None or dg is None:
            _die(f"--scaffold {args.scaffold}: expected a column index "
                 f"(col_idx or col_atlas) and a degree vector (degree or "
                 f"deg_sub), found {sorted(keys)}")
        col_idx = np.asarray(z[ci], dtype=int)
        degree = np.asarray(z[dg], dtype=float)
        if len(col_idx) != len(degree):
            _die(f"--scaffold {args.scaffold}: {ci} has {len(col_idx)} entries "
                 f"and {dg} has {len(degree)}")
        return {"col_idx": col_idx, "degree": degree,
                "source": f"{Path(args.scaffold).name} ({len(degree):,} genes)"}
    if getattr(args, "string_links", None) and getattr(args, "string_info", None):
        if adata is None:
            _die("--string-links needs the expression matrix to map gene names "
                 "onto columns; pass the .h5ad")
        from .potency_metrics import build_ppi_adjacency, load_string_ppi
        edges = load_string_ppi(args.string_links, args.string_info,
                                score_threshold=args.string_threshold)
        _A, _g, col_idx, degree = build_ppi_adjacency(edges, adata.var_names)
        return {"col_idx": col_idx, "degree": np.asarray(degree, float),
                "source": (f"STRING from {Path(args.string_links).name}, "
                           f"combined_score>={args.string_threshold}, largest "
                           f"connected component, {len(degree):,} genes")}
    return None


def scaffold_null(adata, ordinal, scaffold, kernel, n_perm, seed):
    """
    The degree-permutation control, run and reported with its own health check.

    Permuting which gene carries which degree keeps the degree distribution and
    breaks the correspondence between expression and connectivity, so it asks
    how much of a degree-anchored score's agreement with the ordinal survives a
    scaffold that carries no biology. It is a diagnostic and never a verdict:
    see the warning the CLI prints beside it.
    """
    from .potency_metrics import _ccat_from_matrix, _get_X, library_normalize
    kfun = KERNEL_FN[kernel]
    Xn = library_normalize(_get_X(adata, None))[:, scaffold["col_idx"]]
    Xn = np.log1p(Xn)
    degree = scaffold["degree"]
    real = float(kfun(align(_ccat_from_matrix(Xn, degree), ordinal), ordinal))
    rng = np.random.default_rng(seed)
    null = np.empty(n_perm)
    for i in range(n_perm):
        kp = degree[rng.permutation(len(degree))]
        null[i] = kfun(align(_ccat_from_matrix(Xn, kp), ordinal), ordinal)
    lo, hi = np.percentile(null, [2.5, 97.5])
    return {"metric": "PCC(x, degree), sign-aligned to the ordinal",
            "kernel": kernel, "n_perm": n_perm, "seed": seed,
            "real": real, "null_mean": float(null.mean()),
            "null_sd": float(null.std(ddof=1)),
            "null_CI95": [float(lo), float(hi)],
            "gap": real - float(null.mean()),
            "null_above_chance": bool(null.mean() > 0),
            "scaffold": scaffold["source"]}


# ------------------------------------------------------------------- check

def _resolve_primitives(args, adata, n):
    """Returns (dict name -> vector, list of (name, why it is absent))."""
    have, missing = {}, []
    wanted = args.primitive or list(PRIMITIVE_NAMES)

    # a file written by `ownbaseline primitives` supplies whatever it holds,
    # so the two verbs compose and nothing is recomputed twice
    from_file = {}
    if getattr(args, "primitives", None):
        z = np.load(args.primitives, allow_pickle=False)
        unknown = [k for k in z.files
                   if k not in KEY_NPZ and not k.startswith("_")]
        if unknown:
            _die(f"--primitives {args.primitives}: unrecognised arrays "
                 f"{unknown}. Expected any of {sorted(KEY_NPZ)}, as written by "
                 f"`ownbaseline primitives`.")
        for k in z.files:
            if k.startswith("_"):
                continue
            from_file[KEY_NPZ[k]] = np.asarray(z[k], dtype=float)
        if not from_file:
            _die(f"--primitives {args.primitives} holds no primitive arrays")

    for w in wanted:
        label = PRIMITIVE_NAMES[w]
        if label in from_file:
            have[label] = from_file[label]
            continue
        spec = getattr(args, w.replace("-", "_"), None)
        if spec:
            v, src = _read_vector(spec, adata, w)
            have[label] = v
            continue
        if adata is None:
            missing.append((label, "no expression matrix and no vector given; "
                                   f"pass --{w} <file>"))
            continue
        if w == "gene-count":
            from .own_baseline import nnz_per_cell
            have[label] = nnz_per_cell(adata)
        elif w == "entropy":
            from .potency_metrics import stemid_entropy
            have[label] = np.asarray(stemid_entropy(adata), dtype=float)
        elif w == "log-libsize":
            X = adata.X
            tot = (np.asarray(X.sum(axis=1)).ravel() if hasattr(X, "sum")
                   else np.asarray(X).sum(axis=1))
            have[label] = np.log10(np.asarray(tot, float) + 1.0)
        elif w == "pcc-degree":
            sc = load_scaffold(args, adata)
            if sc is None:
                missing.append((label,
                                "needs an interaction scaffold. Pass --scaffold "
                                "<npz> with a column index and a degree vector, "
                                "or --string-links and --string-info, or supply "
                                "the vector directly with --pcc-degree <file>. "
                                "No interactome is bundled: it is 100 MB and "
                                "which release you use changes the number."))
                continue
            from .potency_metrics import ccat
            have[label] = np.asarray(ccat(adata, sc["col_idx"], sc["degree"]),
                                     dtype=float)

    for k, v in list(have.items()):
        if len(v) != n:
            _die(f"primitive {k!r} has length {len(v)}, the score has {n}")
    return have, missing


def cmd_check(args):
    if args.ordinal_source == "derived":
        _die("--ordinal-source derived: a pseudotime, a trajectory position or "
             "any ordinal computed from the same expression matrix is not a "
             "valid ground truth for this test. The primitive predicts such an "
             "ordinal too, so a positive result would measure the shared "
             "dependence rather than the score. There is no flag to override "
             "this. Use an ordinal fixed before dissociation: a staged embryo, "
             "a sorted population, a timepoint.", 3)

    adata = _load_adata(args.data)
    score, score_src = _read_vector(args.score, adata, "score")
    ordinal, ord_src = _read_vector(args.ordinal, adata, "ordinal")
    if len(score) != len(ordinal):
        _die(f"score has {len(score)} rows, ordinal has {len(ordinal)}")

    low = args.score.lower()
    if any(h in low for h in PSEUDOTIME_HINTS) and not args.i_checked:
        _die(f"--score {args.score!r} names something that is usually computed "
             f"from expression. If this ordinal was fixed independently of the "
             f"score, pass --i-checked-this-is-not-derived. If it was not, the "
             f"test does not apply.", 3)

    prims, missing = _resolve_primitives(args, adata, len(score))
    if not prims:
        _die("no primitive could be computed. " +
             "; ".join(f"{k}: {why}" for k, why in missing))

    mask = np.isfinite(score) & np.isfinite(ordinal)
    for v in prims.values():
        mask &= np.isfinite(v)
    n_drop = int((~mask).sum())
    score, ordinal = score[mask], ordinal[mask]
    prims = {k: v[mask] for k, v in prims.items()}
    n = len(score)
    if n < 3:
        _die(f"only {n} rows are finite across score, ordinal and every primitive")

    levels = np.unique(ordinal)
    if len(levels) < 3:
        _die(f"the ordinal has {len(levels)} distinct levels. This estimator is "
             f"kernel-unstable on binary fate labels: on the drug-tolerance "
             f"datasets the two kernels disagree on the same data. Three levels "
             f"is the minimum this tool will run on.", 3)

    s = align(score, ordinal)
    flipped = bool(np.any(s != score))
    auc_score_raw = ordinal_auc(score, ordinal)

    from scipy.stats import spearmanr
    rows, order = {}, list(prims)
    for name, p in prims.items():
        rho = float(spearmanr(p, score, nan_policy="omit").statistic)
        cs = conditional_skill(s, ordinal, [p], args.kernel, args.boot, args.seed)
        fl = _floors.floor(n, rho, args.kernel, 1, 12) if not args.no_floors else {"floor": None, "reason": "--no-floors"}
        rows[name] = {"rho_score_primitive": rho,
                      "primitive_auc_vs_ordinal": ordinal_auc(p, ordinal),
                      "conditional_skill": cs, "null": fl}
    if len(prims) > 1:
        cs = conditional_skill(s, ordinal, list(prims.values()), args.kernel,
                               args.boot, args.seed)
        rho_j = max(abs(rows[k]["rho_score_primitive"]) for k in prims)
        fl = (_floors.floor(n, rho_j, args.kernel, len(prims), 12)
              if not args.no_floors else {"floor": None, "reason": "--no-floors"})
        rows["JOINT"] = {"rho_score_primitive": rho_j,
                         "rho_note": "largest single-primitive |rho|, used to "
                                     "select the null cell",
                         "primitive_auc_vs_ordinal": None,
                         "conditional_skill": cs, "null": fl}
        order.append("JOINT")

    report = {
        "score_name": args.score, "score_source": score_src,
        "ordinal_source_declared": args.ordinal_source,
        "ordinal_spec": args.ordinal, "ordinal_source": ord_src,
        "ordinal_levels": int(len(levels)),
        "n_input": int(len(mask)), "n_used": n, "n_dropped_nonfinite": n_drop,
        "score_sign_flipped_to_align": flipped,
        "direction_check": {
            "score_auc_vs_ordinal": ordinal_auc(s, ordinal),
            "score_auc_before_alignment": auc_score_raw,
            "score_runs_against_ordinal": bool(auc_score_raw < 0.5),
            "primitives_below_chance": [k for k in prims
                                        if ordinal_auc(prims[k], ordinal) < 0.5],
            "primitives_disagreeing_with_score": [
                k for k in prims
                if (ordinal_auc(prims[k], ordinal) - 0.5)
                * (auc_score_raw - 0.5) < 0],
        },
        "primitives_not_computed": [{"primitive": k, "reason": w}
                                    for k, w in missing],
        "by_primitive": rows, "order": order,
    }
    if getattr(args, "scaffold_null", 0):
        sc = load_scaffold(args, adata)
        if sc is None:
            print("\n  --scaffold-null needs a scaffold: pass --scaffold <npz> or "
                  "--string-links with --string-info.", file=sys.stderr)
        elif adata is None:
            print("\n  --scaffold-null needs the expression matrix; pass the .h5ad.",
                  file=sys.stderr)
        else:
            report["scaffold_null"] = scaffold_null(
                adata, ordinal, sc, args.kernel, int(args.scaffold_null), args.seed)

    report["receipt"] = _receipt(args, {
        "n": n, "n_boot": args.boot, "ordinal_levels": int(len(levels)),
        "sha256": {"score": _sha256_array(score),
                   "ordinal": _sha256_array(ordinal),
                   **{k: _sha256_array(v) for k, v in prims.items()}},
    })
    _print_check(report, args)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}")
    return report


def _print_check(r, args):
    from . import __author__, __version__
    d = r["direction_check"]
    drop = f", dropped {r['n_dropped_nonfinite']} non-finite" if r["n_dropped_nonfinite"] else ""
    print()
    print(_s.banner(__version__, __author__))
    print(_s.rule())
    print(f"  score    {r['score_name']}")
    print(f"  ordinal  {r['ordinal_spec']}  "
          + _s.dim(f"{r['ordinal_levels']} levels"))
    print(f"  n        {r['n_used']:,}{_s.dim(drop) if drop else ''}   "
          f"{_s.dim('seed ' + str(r['receipt']['seed']))}   "
          f"{_s.dim(r['receipt']['kernel'])}")
    print()

    print(_s.section("DIRECTION CHECK"))
    raw = d["score_auc_before_alignment"]
    print(f"    score      vs ordinal   AUC {raw:.3f}"
          + ("   points AGAINST the ordinal" if raw < 0.5 else
             "   points with the ordinal"))
    for name in r["order"]:
        if name == "JOINT":
            continue
        a = r["by_primitive"][name]["primitive_auc_vs_ordinal"]
        same = (a - 0.5) * (raw - 0.5) >= 0
        tag = ("points AGAINST the ordinal" if a < 0.5 else
               "points with the ordinal")
        tag += "" if same else "   DISAGREES WITH THE SCORE"
        print(f"    {name:24s} AUC {a:.3f}   {tag}")

    disagree = d.get("primitives_disagreeing_with_score") or []
    if raw < 0.5:
        print("\n    NOTE: the score ranks cells opposite to the ordinal as you encoded it,")
        print("    so it was sign-flipped before residualization. If your ordinal runs so")
        print("    that a higher value means more potent, then the score does not recover")
        print("    it and every margin below is computed on the reversed ordering. If your")
        print("    ordinal is a timeline, where a higher value means later, this is what a")
        print("    working potency score looks like.")
    if disagree:
        print(f"\n    WARNING: {', '.join(disagree)} order cells in the opposite direction")
        print("    to the score. A raw margin against such a baseline is mostly the sign")
        print("    difference between the two, and no gap shows this. Read the conditional")
        print("    skill, which is orientation-invariant.")
    if raw >= 0.5 and not disagree:
        print("    " + _s.dim("-> score and every primitive point the same way; "
                                "margins read as written."))

    for name in r["order"]:
        row = r["by_primitive"][name]
        cs, fl = row["conditional_skill"], row["null"]
        print()
        print(_s.section(f"vs {name}")
              + (_s.dim(f"    rho(score, primitive) = "
                        f"{row['rho_score_primitive']:+.3f}")
                 if name != "JOINT" else
                 _s.dim(f"    {len(r['order']) - 1} primitives at once")))
        if cs.get("refused"):
            print( "    NO VALUE. " + cs["refused"].split(": ")[0] + ".")
            print(f"    residual scale {cs['residual_scale']:.2f} against a noise floor of 1.0;")
            print( "    a real residual sits around 1e14. Conditional skill here is zero by")
            print( "    construction, and a kernel run on this returns the rounding pattern:")
            print( "    on a staged ordinal the primitive predicts, tau_b reaches 0.87 in")
            print( "    magnitude with a sign that changes with the seed.")
            continue
        lo, hi = cs["CI95"]
        print(f"    conditional skill  {cs['tau']:+.4f}   "
              f"CI95 [{lo:+.4f}, {hi:+.4f}]  ({cs['n_boot']} resamples)")
        if fl.get("floor") is None:
            print(f"    no floor: {fl.get('reason')}")
            print( "    -> NO VERDICT. A conditional skill without its own null floor is a")
            print( "       number, not a decision.")
            continue
        f = fl["floor"]
        print(f"    own floor          {f:+.4f}   {fl['cell']}")
        if fl.get("n_side") == "permissive":
            print("    " + _s.warn(f"WARNING: the nearest measured n is {fl['grid_n']:,}, above this run's"))
            print(f"    {r['n_used']:,}. The floor falls with n, so this threshold is LOWER than the")
            print( "    true one and clearing it proves less than it looks. Run the grid at this n,")
            print( "    or read the result as provisional.")
        elif fl.get("n_side") == "conservative":
            print(f"    NOTE: the nearest measured n is {fl['grid_n']:,}, below this run's "
                  f"{r['n_used']:,}.")
            print( "    The floor falls with n, so this threshold is higher than the true one and")
            print( "    clearing it is the safe direction to be wrong in.")
        if fl.get("k_bracketed"):
            print(f"    NOTE: no cell was measured at {len(r['order']) - 1} covariates. The floor is the")
            print(f"    larger of the measured {fl['k_used'][0]} and {fl['k_used'][1]}, because the floor is not")
            print( "    monotone in the covariate count and there is no safe direction to round in.")
        if abs(fl.get("rho_offset") or 0) > 0.15:
            print(f"    NOTE: rho is {fl['rho_offset']:+.2f} from the nearest measured {fl['grid_rho']:g}. The")
            print( "    floor is not monotone in rho, so this offset can run either way.")
        if (abs(row.get("rho_score_primitive") or 0) > 0.995
                and name != "JOINT" and cs.get("residual_scale", 1e18) < 1e12):
            print( "    CAVEAT (defect F-2): at this rank correlation the score is close to")
            print( "    an exact monotone function of its primitive, and the residual is")
            print( "    rounding debris on the order of 1e-13. Kendall tau_b returns about")
            print(f"    0.02 on such a fixture; the weighted kernel returns up to +0.51 on")
            print( "    the same debris, where zero is the only correct answer. Read a value")
            print( "    at this rho as a statement about the debris unless it is an order of")
            print( "    magnitude away from that scale.")
        if cs["tau"] > f:
            print("    " + _s.good(f"-> CLEARS its floor by {cs['tau'] / f:.1f}x"))
        elif cs["tau"] < 0:
            print( "    -> DOES NOT CLEAR. The residual is negative, and a negative value")
            print( "       does not exceed a positive threshold. That is where the value")
            print( "       sits; it is not a failed test.")
        else:
            print("    " + _s.bad(f"-> DOES NOT CLEAR ({cs['tau']:+.4f} against {f:+.4f})"))

    verdicts = {k: (r["by_primitive"][k]["conditional_skill"]["tau"],
                    r["by_primitive"][k]["null"].get("floor"))
                for k in r["order"]}
    refused = [k for k in r["order"]
               if r["by_primitive"][k]["conditional_skill"].get("refused")]
    scored = {k: v for k, v in verdicts.items()
              if v[0] is not None and v[1] is not None}
    print()
    print(_s.rule())
    if refused:
        print(f"  {', '.join(refused)}: no verdict, the score is a rank-preserving")
        print( "  function of that primitive and its conditional skill is zero by")
        print( "  construction.")
    if not scored:
        print("  NO VERDICT for the remaining primitives: no floor was available.")
    elif all(t > f for t, f in scored.values()):
        print("  " + _s.head("VERDICT") + "  " + _s.good("orders cells beyond every primitive tested here."))
    elif any(t > f for t, f in scored.values()):
        cleared = [k for k, (t, f) in scored.items() if t > f]
        print("  " + _s.head("VERDICT") + _s.warn(f"  mixed. Clears against "
              f"{', '.join(cleared)} and not against the rest."))
    else:
        print("  " + _s.head("VERDICT") + "  " + _s.bad("does not order cells beyond its primitive on this ordinal."))
    print("  Scope: this says the score orders cells beyond these statistics. It")
    print("  does not say what the residual is. A residual can be developmental")
    print("  position, manifold structure, or a fifth statistic nobody named.")

    sn = r.get("scaffold_null")
    if sn:
        print()
        print(_s.section(f"SCAFFOLD-RANDOMIZATION CONTROL")
              + _s.dim(f"   {sn['n_perm']} permutations of the degree vector"))
        print(f"    scaffold      {sn['scaffold']}")
        print(f"    real          {sn['real']:+.4f}")
        print(f"    permuted      {sn['null_mean']:+.4f}  sd {sn['null_sd']:.4f}  "
              f"95% [{sn['null_CI95'][0]:+.4f}, {sn['null_CI95'][1]:+.4f}]")
        print(f"    gap           {sn['gap']:+.4f}")
        print( "    This is a diagnostic and not a verdict. On the systems in the paper the")
        print( "    gap was 0.120 for signalling entropy against 0.057 for CCAT while both sat")
        print( "    at a conditional skill of about zero, so a rule reading a larger gap as more")
        print( "    genuine would have ranked them backwards. Two pre-registered predictions")
        print( "    about this control were falsified. Read the conditional skill above.")
        if not sn["null_above_chance"]:
            print(f"    The permuted mean is {sn['null_mean']:+.4f}, below chance, which leaves the gap")
            print( "    uninterpretable as a measure of how much biology the score carries.")

    if r["primitives_not_computed"]:
        print()
        print(_s.section("NOT COMPUTED, and therefore not controlled for"))
        for m in r["primitives_not_computed"]:
            print(f"    {m['primitive']}: {m['reason']}")

    if args.kernel == "weighted":
        print("\n  WARNING on the weighted kernel: its null mean is not zero and it")
        print("  rises with sample size, from 0.188 at n = 500 to 0.309 at n = 127,607.")
        print("  It returned a positive verdict on 720 of 720 null replicates. The")
        print("  floor above is its own measured floor, so the comparison is still")
        print("  valid, but the kernel cannot disagree with anything.")


# ------------------------------------------------------------------ floors

def cmd_floors(args):
    if args.list_grid:
        g = _floors.grid()
        print("measured design points in the shipped grid:")
        for k, v in g.items():
            print(f"  {k} covariate{'s' if k > 1 else ''}: "
                  f"n {v['n']}  rho {v['rho']}  levels {v['levels']}")
        return 0
    res = _floors.floor(args.n, args.rho, args.kernel, args.covariates, args.levels)
    if args.json:
        print(json.dumps(res, indent=2))
        return 0 if res.get("floor") is not None else 1
    if res.get("floor") is None:
        print(f"no floor: {res['reason']}")
        return 1
    print(f"floor      {res['floor']:+.4f}")
    print(f"null mean  {res['null_mean']:+.4f}")
    print(f"cell       {res['cell']}")
    if res["n_offset"] or res["rho_offset"]:
        print(f"offset     n {res['n_offset']:+,} from the grid point, "
              f"rho {res['rho_offset']:+.3f}")
    return 0


# -------------------------------------------------------------- primitives

def cmd_primitives(args):
    adata = _load_adata(args.data)
    if adata is None:
        _die("primitives needs a .h5ad")
    n = adata.n_obs
    have, missing = _resolve_primitives(args, adata, n)
    sc = load_scaffold(args, adata) if not getattr(args, "primitives", None) else None
    meta = {"n_cells": int(n), "source": str(args.data),
            "own_baseline_version": __import__("own_baseline").__version__,
            "scaffold": (sc or {}).get("source"),
            "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "sha256": {k: _sha256_array(v) for k, v in have.items()},
            "not_computed": {k: w for k, w in missing}}
    out = Path(args.out)
    np.savez(out, _meta=np.array(json.dumps(meta)),
             **{NPZ_KEY[k]: v for k, v in have.items()})
    print(f"wrote {out} with {len(have)} primitive"
          f"{'s' if len(have) != 1 else ''} over {n:,} cells:")
    for k, v in have.items():
        print(f"  {NPZ_KEY[k]:20s} median {np.nanmedian(v):.4g}  "
              f"sha256 {meta['sha256'][k][:16]}")
    for k, why in missing:
        print(f"  {NPZ_KEY[k]:20s} NOT COMPUTED: {why}")
    print(f"\nfeed it back with:  ownbaseline check --primitives {out} ...")
    return 0


# ------------------------------------------------------------------ verify

def cmd_verify(args):
    rep = json.loads(Path(args.receipt).read_text())
    rec = rep.get("receipt")
    if rec is None:
        _die(f"{args.receipt} carries no receipt block")
    from . import __version__
    print(f"receipt from {rec.get('utc')}\n")
    env_now = {"own_baseline_version": __version__,
               "python": platform.python_version()}
    for m in ("numpy", "scipy"):
        try:
            env_now[m] = __import__(m).__version__
        except Exception:                                       # noqa: BLE001
            env_now[m] = None
    drift = {k: (rec.get(k), v) for k, v in env_now.items() if rec.get(k) != v}
    if drift:
        print("  ENVIRONMENT DRIFT")
        for k, (was, now) in drift.items():
            print(f"    {k:24s} receipt {was}   here {now}")
    else:
        print("  environment matches the receipt")

    if not args.rerun:
        print("\n  hashes recorded in the receipt:")
        for k, v in (rec.get("sha256") or {}).items():
            print(f"    {k:24s} {v}")
        print("\n  pass --rerun with the same inputs to recompute and compare.")
        return 0 if not drift else 1

    if not (args.score and args.ordinal):
        _die("--rerun needs --score and --ordinal (and the same primitives) so "
             "the numbers can be recomputed from the inputs the receipt hashed")
    args.ordinal_source = "experimental"     # already gated at record time
    args.i_checked = True
    fresh = cmd_check(args)
    ok = True
    print("\n  COMPARISON WITH THE RECEIPT")
    for name, row in rep["by_primitive"].items():
        new = fresh["by_primitive"].get(name)
        if new is None:
            print(f"    {name:24s} MISSING in the re-run")
            ok = False
            continue
        d = new["conditional_skill"]["tau"] - row["conditional_skill"]["tau"]
        flag = "" if abs(d) < 1e-12 else "   DIFFERS"
        if flag:
            ok = False
        print(f"    {name:24s} {row['conditional_skill']['tau']:+.6f} -> "
              f"{new['conditional_skill']['tau']:+.6f}   delta {d:+.2e}{flag}")
    print(f"\n  {'REPRODUCES' if ok else 'DOES NOT REPRODUCE'}")
    return 0 if ok else 1


# -------------------------------------------------------------------- main

def _add_input_args(p, need_score=True, required=True):
    p.add_argument("data", nargs="?", help="a .h5ad, optional if the score, the "
                                          "ordinal and the primitives are files")
    if need_score:
        p.add_argument("--score", required=required,
                       help="obs:<col>, a .csv/.npy file, or module:callable")
        p.add_argument("--ordinal", required=required,
                       help="obs:<col>, a .csv/.npy file, or module:callable")
    for flag in PRIMITIVE_NAMES:
        p.add_argument(f"--{flag}", metavar="VEC",
                       help=f"supply {PRIMITIVE_NAMES[flag]} directly instead of "
                            f"computing it")
    p.add_argument("--primitives", metavar="NPZ",
                   help="an .npz written by `ownbaseline primitives`; whatever "
                        "it holds is used as-is and not recomputed")
    p.add_argument("--primitive", action="append", choices=list(PRIMITIVE_NAMES),
                   help="restrict to this primitive; repeatable. Default: all four.")
    p.add_argument("--scaffold", metavar="NPZ",
                   help="a prepared scaffold: an .npz holding a column index "
                        "(col_idx or col_atlas) and a degree vector (degree or "
                        "deg_sub). Cheaper than rebuilding one from STRING.")
    p.add_argument("--string-links", help="STRING protein.links.v12.0.txt.gz")
    p.add_argument("--string-info", help="STRING protein.info.v12.0.txt.gz")
    p.add_argument("--string-threshold", type=int, default=700,
                   help="STRING combined_score cutoff (default 700)")


def build_parser():
    ap = argparse.ArgumentParser(
        prog="ownbaseline",
        description="Does a single-cell potency score order cells beyond the "
                    "low-order statistic it is closest to?")
    from . import __version__
    ap.add_argument("--version", action="version",
                    version=f"ownbaseline {__version__}")
    sub = ap.add_subparsers(dest="cmd")

    c = sub.add_parser("check", help="run the test on your data")
    _add_input_args(c)
    c.add_argument("--ordinal-source", required=True,
                   choices=["experimental", "derived"],
                   help="experimental: the ordinal was fixed independently of "
                        "expression, by staging, sorting or a timepoint. "
                        "derived: it was computed from the same matrix, and the "
                        "test does not apply. There is no override.")
    c.add_argument("--kernel", default="taub", choices=["taub", "weighted"])
    c.add_argument("--boot", type=int, default=1000)
    c.add_argument("--seed", type=int, default=42)
    c.add_argument("--json", help="write the full report here")
    c.add_argument("--no-floors", action="store_true",
                   help="report conditional skill without a decision threshold")
    c.add_argument("--i-checked-this-is-not-derived", dest="i_checked",
                   action="store_true")
    c.add_argument("--scaffold-null", nargs="?", type=int, const=200, default=0,
                   metavar="N",
                   help="also run the degree-permutation control, N permutations "
                        "(default 200). Needs --scaffold or the STRING pair. It "
                        "is a diagnostic and not a verdict; read what it prints.")
    c.set_defaults(func=cmd_check)

    f = sub.add_parser("floors", help="the measured null floor at n, rho, kernel")
    f.add_argument("--n", type=int, default=0)
    f.add_argument("--rho", type=float, default=0.0)
    f.add_argument("--kernel", default="taub", choices=["taub", "weighted"])
    f.add_argument("--covariates", type=int, default=1)
    f.add_argument("--levels", type=int, default=12)
    f.add_argument("--json", action="store_true")
    f.add_argument("--list-grid", action="store_true",
                   help="print the design points the shipped grid holds")
    f.set_defaults(func=cmd_floors)

    p = sub.add_parser("primitives", help="compute the low-order statistics")
    _add_input_args(p, need_score=False)
    p.add_argument("--out", default="primitives.npz")
    p.set_defaults(func=cmd_primitives)

    v = sub.add_parser("verify", help="re-check a receipt against this environment")
    v.add_argument("receipt", help="the .json a previous check wrote")
    v.add_argument("--rerun", action="store_true",
                   help="recompute from the same inputs and compare the numbers")
    _add_input_args(v, required=False)
    v.add_argument("--kernel", default="taub", choices=["taub", "weighted"])
    v.add_argument("--boot", type=int, default=1000)
    v.add_argument("--seed", type=int, default=42)
    v.add_argument("--json")
    v.add_argument("--no-floors", action="store_true")
    v.set_defaults(func=cmd_verify)
    return ap


def _splash():
    from . import __author__, __version__
    print()
    print(_s.splash(__version__, __author__))
    print()
    print("  " + _s.head("check") + "        does this score order cells beyond "
          "the statistic it is closest to?")
    print("  " + _s.head("floors") + "       the estimator's measured null floor "
          "at a given n, rho and kernel")
    print("  " + _s.head("primitives") + "   compute the low-order statistics "
          "from an expression matrix")
    print("  " + _s.head("verify") + "       re-run a receipt and report whether "
          "it still reproduces")
    print()
    print("  " + _s.dim("ownbaseline check cells.h5ad --score obs:cytotrace "
                        "--ordinal obs:stage \\"))
    print("  " + _s.dim("                  --ordinal-source experimental"))
    print()
    print("  " + _s.dim("ownbaseline <verb> --help   ·   docs/interpreting.md "
                        "for what a verdict does not mean"))
    print()


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "cmd", None) is None:
        _splash()
        return 0
    rc = args.func(args)
    return 0 if rc is None or isinstance(rc, dict) else int(rc)


if __name__ == "__main__":
    raise SystemExit(main())
