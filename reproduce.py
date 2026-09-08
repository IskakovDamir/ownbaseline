#!/usr/bin/env python3
"""
reproduce.py — regenerate what this repository can regenerate, and say plainly
what it cannot.

    python3 reproduce.py --list          what each stage does and what it needs
    python3 reproduce.py --ledger        the honest coverage table, no work done
    python3 reproduce.py --self-test     run every stage on synthetic data (~1 min)
    python3 reproduce.py                 the real thing: download and run

Everything is written under $OWNBASELINE_DATA_ROOT (default <repo>/data/runs).
Set it to a disk with ~2 GB free. Nothing is written outside it except the
figures and table, which land in reproduction/.

WHAT THIS REPRODUCES, AND WHAT IT DOES NOT
------------------------------------------
Of the paper's audited scores, four are computable from this repository plus a
public download and nothing else: the three primitives (gene count,
Pearson(x, PPI degree), Shannon entropy), CytoTRACE v1, and SCENT's SR and
CCAT. Those rows run end to end here.

The rest do not, and the reasons differ. They are enumerated in LEDGER below
and printed by --ledger. A stage that cannot run says so and says why; none is
silently skipped.

The R cross-validation of the Python SR/CCAT/ORIGINS implementations is a
separate matter from computing them. The Python implementations agree with the
R packages on a 500-cell subsample (Spearman 1.000, max |delta| < 1e-13 for
SR/CCAT); that check needs the R packages and does not run here. This entry
point computes the Python scores and records that the cross-check was skipped.
"""
from __future__ import annotations

import argparse
import json
import ssl
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

from own_baseline.paths import data_root, ensure, scratch_root  # noqa: E402

PY = sys.executable

# --------------------------------------------------------------------------
# Downloads. Sizes are what the servers reported on 2026-08-13.
# --------------------------------------------------------------------------
DOWNLOADS = {
    "string_links": (
        "https://stringdb-downloads.org/download/protein.links.v12.0/"
        "9606.protein.links.v12.0.txt.gz", "string_human_dl",
        "9606.protein.links.v12.0.txt.gz", "~72 MB"),
    "string_info": (
        "https://stringdb-downloads.org/download/protein.info.v12.0/"
        "9606.protein.info.v12.0.txt.gz", "string_human_dl",
        "9606.protein.info.v12.0.txt.gz", "2 MB"),
    "zebrafish": (
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE106nnn/GSE106474/suppl/"
        "GSE106474_UMICounts.txt.gz", None,
        "track2/data/GSE106474_UMICounts.txt.gz", "61 MB"),
}

# --------------------------------------------------------------------------
# The coverage ledger. One row per headline claim in the paper.
# status: RUNS | BLOCKED | NOT-A-MEASUREMENT
# --------------------------------------------------------------------------
LEDGER = [
    ("gene count / PCC(x,degree) / Shannon primitives", "RUNS",
     "pure Python in own_baseline/potency_metrics.py; needs GSE106474 + STRING v12"),
    ("CytoTRACE v1  (tau_b 0.653 | gene count)", "RUNS",
     "R-faithful port in scores/cytotrace_full.py; MVG -> GCS -> NNLS -> Markov "
     "diffusion, no R needed"),
    ("SCENT SR      (tau_b 0.430 | PCC(x,degree))", "RUNS",
     "Python implementation in potency_metrics.scent_sr. The R cross-check "
     "against aet21/SCENT does NOT run here: it needs CompSRana.R, DoIntegPPI.R "
     "and CompCCAT.R copied into $OWNBASELINE_SCRATCH/scent_src, and no script "
     "in this repository fetches them."),
    ("SCENT CCAT    (rho 0.998 with its primitive)", "RUNS",
     "Python implementation in potency_metrics.ccat; same cross-check caveat"),
    ("ORIGINS       (tau_b 0.402 | PCC(x,degree))", "BLOCKED",
     "needs diff_edges.tsv, which is ORIGINS::differentiation_edges dumped to "
     "TSV by hand. No script here writes it, so this row cannot start."),
    ("SLICE         (rho 0.932, skill ~ 0, n=3,000)", "BLOCKED",
     "needs the SLICE R package (xu-lab/SLICE) and its bundled hs_kappasim.rda. "
     "Nothing here fetches or places that file."),
    ("dpath         (tau_b 0.120, n=3,000)", "BLOCKED",
     "needs the dpath R package (gongx030/dpath), GitHub-only, with a documented "
     "R >= 4.2 incompatibility patched at runtime in scores/run_dpath.R"),
    ("NCG           (tau_b 0.402, n=3,000, GO connectome)", "BLOCKED",
     "needs the NCG repo (Xinzhe-Ni/NCG) cloned WITH Git-LFS: a plain clone "
     "yields pointer files, not the HPRD and GO matrices"),
    ("joint-4 residuals (0.288 / 0.183 / 0.156 / 0.134)", "BLOCKED",
     "fix2_joint_primitive.py is here, two of its four score inputs are not: "
     "origins_scores.npz and dpath_scores.npz are absent for the reasons those "
     "two rows give, and no output JSON of this step ships. The four values in "
     "the paper, one of them in the abstract, therefore rest on a run record "
     "and not on anything this script can recompute."),
    ("MCE", "BLOCKED", "no public implementation exists (Shi 2020)"),
    ("SPIDE", "BLOCKED", "the source paper is paywalled (Zheng 2023); a first-party\n     Python implementation is public and was not run here"),
    ("scEnergy", "BLOCKED", "requires MATLAB; its network step uses the Statistics\n     toolbox and graph objects GNU Octave does not provide"),
    ("StemID, cmEntropy", "NOT-A-MEASUREMENT",
     "equal to a transcriptome-entropy primitive by construction. The paper "
     "declines to report a number for them and so does this script. "
     "figures/make_paper_a_figures.py used to plot them at rho 0.990 / 1.000 and "
     "skill 0.000 as though measured; since 2026-08-29 it reads its data and "
     "shows them in a by-construction band that carries no value."),
    ("StemSC", "NOT-A-MEASUREMENT", "measured on the sorted atlases, not on this ordinal"),
    ("Figure 3 / depth confound (AUROC 0.378 vs 0.798)", "RUNS-SEPARATELY",
     "experiments/w4/scripts/w4_gate2_run.py, on GSE117498 + GSE125970. Driven "
     "by --stages depth. Slower and larger than the zebrafish chain."),
]


def log(msg):
    print(f"[reproduce] {msg}", flush=True)


def _urlopen(url, timeout=60):
    try:
        return urllib.request.urlopen(url, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        if "CERTIFICATE_VERIFY_FAILED" not in str(exc):
            raise
        log("TLS verification failed (a intercepting proxy, most likely); "
            "retrying without verification")
        return urllib.request.urlopen(
            url, timeout=timeout, context=ssl._create_unverified_context())


def fetch(key, root):
    url, sub, rel, size = DOWNLOADS[key]
    dest = (scratch_root() / sub / rel) if sub else (root / rel)
    if dest.exists() and dest.stat().st_size > 0:
        log(f"{key}: already at {dest} ({dest.stat().st_size/1e6:.1f} MB)")
        return dest
    ensure(dest.parent)
    log(f"{key}: downloading {size} from {url}")
    t0 = time.time()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _urlopen(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)
            got += len(chunk)
            if total:
                print(f"\r    {got/1e6:7.1f} / {total/1e6:.1f} MB", end="", flush=True)
    print()
    tmp.rename(dest)
    log(f"{key}: {dest.stat().st_size/1e6:.1f} MB in {time.time()-t0:.0f}s -> {dest}")
    return dest


def run(script, *args, cwd=None):
    cmd = [PY, str(script), *map(str, args)]
    log("$ " + " ".join(cmd))
    r = subprocess.run(cmd, cwd=cwd or REPO)
    if r.returncode != 0:
        raise SystemExit(f"stage failed: {' '.join(cmd)}")


# --------------------------------------------------------------------------
# Stages
# --------------------------------------------------------------------------

def stage_fetch_string(root):
    fetch("string_links", root)
    fetch("string_info", root)


def stage_scaffold(root):
    out = root / "w4/scaffolds/human_string_v12_thr700_lcc.npz"
    if out.exists():
        log(f"scaffold: already at {out}")
        return
    ensure(out.parent)
    run(REPO / "experiments/w4/scripts/build_human_string_v12_thr700_lcc.py",
        "--out", out)


def stage_fetch_zebrafish(root):
    fetch("zebrafish", root)


def stage_prep(root):
    if (root / "track2/data/prepared/primitives.npz").exists():
        log("prep: prepared/ already present")
        return
    run(REPO / "experiments/track2/code/prep_data.py")


def stage_cytotrace(root):
    if (root / "track2/results/cytotrace_scores.npz").exists():
        log("cytotrace: already computed")
        return
    run(REPO / "experiments/track2/code/run_cytotrace.py")


def stage_scent(root):
    """
    SR and CCAT, Python implementation, without the R cross-check.

    run_scent_full.py computes the scores and then validates them against a
    SCENT R subsample CSV that this repository has no way to produce. The
    scores do not depend on that CSV, so we call the same compute function and
    record that the cross-check was skipped.
    """
    out = root / "track2/results/scent_scores.npz"
    if out.exists():
        log("scent: already computed")
        return
    run(REPO / "experiments/track2/code/prep_scent_inputs.py")

    import numpy as np
    import scipy.sparse as sp
    sys.path.insert(0, str(REPO / "experiments/track2/code"))
    sys.path.insert(0, str(REPO / "own_baseline"))
    import potency_metrics as pm
    from run_scent_full import compute_scores, load_net

    A, degree, mc_col, target_sum, _sub = load_net()
    maxSR = pm._max_entropy_rate(A)
    X = sp.load_npz(root / "track2/data/prepared/X_cells_genes.npz").tocsr()
    lib = np.asarray(X.sum(axis=1)).ravel().astype(np.float64)
    log(f"scent: {A.shape[0]} network genes, {X.shape[0]} cells, "
        f"maxSR={maxSR:.4f}")
    ccat, sr = compute_scores(X, lib, np.arange(X.shape[0]), A, degree, mc_col,
                              target_sum, maxSR)
    ensure(out.parent)
    np.savez(out, sr=sr, ccat=ccat)
    (root / "track2/results/scent_validation.json").write_text(json.dumps({
        "r_cross_check": "SKIPPED",
        "why": "needs aet21/SCENT R sources in $OWNBASELINE_SCRATCH/scent_src; "
               "no script in this repository fetches them",
        "published_cross_check": "Spearman 1.000, max |delta| < 1e-13 on a "
                                 "500-cell subsample",
    }, indent=2))
    log(f"scent: wrote {out} (R cross-check SKIPPED, recorded in "
        f"scent_validation.json)")


def stage_skill(root):
    run(REPO / "own_baseline/conditional_skill.py")


def stage_table(root):
    src = root / "track2/results/gate2_conditional_skill.json"
    if not src.exists():
        raise SystemExit(f"no results at {src}; run the earlier stages first")
    res = json.loads(src.read_text())
    out_dir = REPO / "reproduction"
    ensure(out_dir)

    lines = [
        "# Reproduced headline table",
        "",
        f"n_cells = {res.get('n_cells_full')}, seed = {res.get('seed')}, "
        f"bootstrap = {res.get('n_boot')}",
        "",
        "Conditional skill of each score against its own primitive, both "
        "kernels, 95% bootstrap interval.",
        "",
        "| score \\| primitive | weightedtau | 95% CI | tau_b | 95% CI | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for label, r in res.get("conditional_skill", {}).items():
        esc = label.replace("|", "\\|")   # the row labels contain a literal pipe
        if "status" in r:
            lines.append(f"| {esc} | — | — | — | — | {r['status']} |")
            continue
        a, b = r["scipy_weightedtau"], r["kang_wdm_taub"]
        lines.append(
            f"| {esc} | {a['tau']:+.4f} | "
            f"[{a['CI95'][0]:+.4f}, {a['CI95'][1]:+.4f}] | "
            f"{b['tau']:+.4f} | [{b['CI95'][0]:+.4f}, {b['CI95'][1]:+.4f}] | "
            f"{r['combined_verdict']} |")
    lines += ["", "## Coverage", ""]
    lines += ["| row | status | note |", "|---|---|---|"]
    for name, status, note in LEDGER:
        lines.append(f"| {name.replace(chr(124), chr(92) + chr(124))} | {status} | "
                     f"{note.replace(chr(124), chr(92) + chr(124))} |")
    lines += ["", "Generated by `reproduce.py`. The paper's own figures are in "
              "`figures/make_paper_a_figures.py`, which since 2026-08-29 reads "
              "every value from `data/run_record/` and the null-calibration grid "
              "rather than holding literals. This script recomputes the scores "
              "from the raw data instead, so the two are independent; "
              "`figures/fig2_values.json` records what the figure drew."]

    (out_dir / "table.md").write_text("\n".join(lines) + "\n")
    (out_dir / "table.json").write_text(json.dumps(
        {"results": res, "ledger": [
            {"row": n, "status": s, "note": t} for n, s, t in LEDGER]}, indent=2))
    log(f"table: wrote {out_dir/'table.md'} and {out_dir/'table.json'}")
    print("\n".join(lines[:20]))


def stage_figures(root):
    """
    Figure 2 from the values just computed, with the bootstrap intervals the
    paper's caption promises and figures/make_paper_a_figures.py does not draw.
    Rows that could not be run are drawn as hatched placeholders labelled with
    the reason, never as a zero.
    """
    src = root / "track2/results/gate2_conditional_skill.json"
    if not src.exists():
        raise SystemExit(f"no results at {src}; run the earlier stages first")
    res = json.loads(src.read_text())

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as mp
    import matplotlib.pyplot as plt
    import numpy as np

    out_dir = REPO / "reproduction"
    ensure(out_dir)

    rows = []
    for label, r in res.get("conditional_skill", {}).items():
        if "status" in r:
            rows.append((label, None, None, None, r["status"]))
        else:
            b = r["kang_wdm_taub"]
            rows.append((label, b["tau"], b["CI95"][0], b["CI95"][1],
                         r["combined_verdict"]))
    blocked = [(n, note) for n, s, note in LEDGER if s == "BLOCKED"]

    fig, ax = plt.subplots(figsize=(7.4, 0.42 * (len(rows) + len(blocked)) + 1.6))
    y = np.arange(len(rows) + len(blocked))[::-1]
    colors = {"ADDS-BEYOND": "#1a7f37", "TAUTOLOG": "#8a8a8a",
              "SIGN-FLIPPED": "#b00000",
              "INCONCLUSIVE-kernel-disagree": "#b08900"}
    for yi, (label, tau, lo, hi, verdict) in zip(y, rows):
        c = colors.get(verdict, "#8a8a8a")
        if tau is None:
            ax.barh(yi, 0.02, color="none", edgecolor=c, hatch="///", lw=0.8)
            continue
        ax.barh(yi, tau, color=c, edgecolor="#333", lw=0.5)
        ax.plot([lo, hi], [yi, yi], color="#111", lw=1.1)
        for e in (lo, hi):
            ax.plot([e, e], [yi - 0.16, yi + 0.16], color="#111", lw=1.1)
        ax.text(tau + (0.012 if tau >= 0 else -0.012), yi, f"{tau:+.3f}",
                va="center", ha="left" if tau >= 0 else "right", fontsize=6.4)
    for yi, (name, note) in zip(y[len(rows):], blocked):
        ax.barh(yi, 0.02, color="none", edgecolor="#b08900", hatch="///", lw=0.8)
        ax.text(0.03, yi, "not run: " + note.split(";")[0][:64], va="center",
                fontsize=5.8, color="#7a5c00", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows] + [b[0] for b in blocked], fontsize=6.6)
    ax.axvline(0, color="#888", lw=0.9)
    ax.set_xlabel("conditional skill beyond the declared primitive "
                  "(Kendall tau-b, 95% bootstrap interval)")
    ax.set_title(f"Reproduced from data: zebrafish 12-stage ordinal, "
                 f"n = {res.get('n_cells_full')}", fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(handles=[mp.Patch(color=v, label=k) for k, v in colors.items()]
              + [mp.Patch(facecolor="none", edgecolor="#b08900", hatch="///",
                          label="could not be run (see reproduction/table.md)")],
              fontsize=5.8, loc="lower right", frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "figure2_reproduced.png", dpi=170)
    fig.savefig(out_dir / "figure2_reproduced.pdf")
    plt.close(fig)
    log(f"figures: wrote {out_dir/'figure2_reproduced.png'}")
    log("figures: Figure 3 (the depth confound) comes from GSE117498 and "
        "GSE125970 via `--stages depth`, not from this chain")


def stage_depth(root):
    run(REPO / "experiments/w4/scripts/w4_gate2_run.py")


STAGES = [
    ("fetch-string", stage_fetch_string, "download STRING v12 human links + info (~74 MB)"),
    ("scaffold", stage_scaffold, "build the thr700 largest-connected-component scaffold"),
    ("fetch-zebrafish", stage_fetch_zebrafish, "download GSE106474 UMI counts (61 MB)"),
    ("prep", stage_prep, "QC, stage ordinal, and the three primitives"),
    ("cytotrace", stage_cytotrace, "CytoTRACE v1, full R-faithful port"),
    ("scent", stage_scent, "SCENT SR and CCAT, Python implementation"),
    ("skill", stage_skill, "conditional skill, both kernels, bootstrap CI"),
    ("table", stage_table, "write reproduction/table.md + table.json"),
    ("figures", stage_figures, "write reproduction/figure2_reproduced.{png,pdf}"),
    ("depth", stage_depth, "Figure 3: GSE117498 + GSE125970 depth confound (separate, large)"),
]
DEFAULT = [s for s, _, _ in STAGES if s != "depth"]


def print_ledger():
    width = max(len(n) for n, _, _ in LEDGER)
    print("\nWhat this repository can and cannot regenerate\n")
    for name, status, note in LEDGER:
        print(f"  {name:<{width}}  {status}")
        for i in range(0, len(note), 88):
            print(f"  {'':<{width}}    {note[i:i+88]}")
    print()


def main():
    ap = argparse.ArgumentParser(
        description="Regenerate the reproducible part of the paper's table and figures.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=None,
                    help=f"subset of: {', '.join(s for s, _, _ in STAGES)}")
    ap.add_argument("--list", action="store_true", help="list stages and exit")
    ap.add_argument("--ledger", action="store_true",
                    help="print the coverage table and exit")
    ap.add_argument("--self-test", action="store_true",
                    help="run the full chain on synthetic data (no download)")
    args = ap.parse_args()

    if args.list:
        for name, _, doc in STAGES:
            print(f"  {name:16s} {doc}")
        print(f"\n  default: {' '.join(DEFAULT)}")
        return 0
    if args.ledger:
        print_ledger()
        return 0
    if args.self_test:
        from reproduce_selftest import run_self_test
        return run_self_test()

    root = data_root()
    ensure(root)
    log(f"data root: {root}")
    log(f"scratch:   {scratch_root()}")
    chosen = args.stages or DEFAULT
    known = {name: fn for name, fn, _ in STAGES}
    for name in chosen:
        if name not in known:
            raise SystemExit(f"unknown stage {name!r}; see --list")
    for name in chosen:
        log(f"===== stage: {name} =====")
        known[name](root)
    print_ledger()
    return 0


if __name__ == "__main__":
    sys.exit(main())
