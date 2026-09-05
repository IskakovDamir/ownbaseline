"""
CLI-level fixtures. Each one encodes a claim the paper makes, so a change that
breaks the claim breaks a test rather than shipping quietly.

  F1  a score that is an exact monotone function of its primitive must not be
      reported as carrying skill, and must carry the F-2 rounding caveat
  F2  a score with real ordering skill must clear its own measured floor
  F3  the haematopoietic direction inversion must be named before any margin
  F4  the CLI's estimator must return exactly what the library returns
  F5  the published floors must come back out of the shipped grid
  F6  the guards must refuse rather than compute
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from own_baseline import floors
from own_baseline.cli import conditional_skill, main, ordinal_auc
from own_baseline.conditional_skill import (align, conditional_skill_report,
                                            rank_resid_multi)

SEED = 20260904


def _run(args, tmp):
    """Run the CLI in-process, return (exit code, report dict or None)."""
    out = tmp / "r.json"
    code = 0
    try:
        main([*args, "--json", str(out)])
    except SystemExit as e:
        code = int(e.code or 0)
    return code, (json.loads(out.read_text()) if out.is_file() else None)


def _vecs(tmp, **arrays):
    paths = {}
    for k, v in arrays.items():
        p = tmp / f"{k}.npy"
        np.save(p, np.asarray(v, dtype=float))
        paths[k] = str(p)
    return paths


def _staged(n, levels, rng):
    """A staged ordinal and a primitive that tracks it."""
    o = np.repeat(np.arange(levels, dtype=float), n // levels)
    o = np.concatenate([o, np.full(n - len(o), float(levels - 1))])
    prim = o + rng.normal(0, 1.2, n)
    return o, prim


# ------------------------------------------------------------------ F1

def test_exact_monotone_function_of_its_primitive_returns_no_value(tmp_path):
    """
    The score is a rank-preserving function of the primitive, so its conditional
    skill is zero by construction and the residual is arithmetic noise. The CLI
    must refuse rather than report, because on this fixture the kernels do NOT
    return zero: see test_debris_is_scored_by_both_kernels below.
    """
    rng = np.random.default_rng(SEED)
    n = 4000
    o, prim = _staged(n, 12, rng)
    score = np.exp(prim / 3.0)                 # strictly monotone in the primitive
    v = _vecs(tmp_path, score=score, ordinal=o, gc=prim)
    code, rep = _run(["check", "--score", v["score"], "--ordinal", v["ordinal"],
                      "--ordinal-source", "experimental",
                      "--primitive", "gene-count", "--gene-count", v["gc"],
                      "--boot", "60"], tmp_path)
    assert code == 0
    row = rep["by_primitive"]["gene_count"]
    assert abs(row["rho_score_primitive"]) > 0.995
    cs = row["conditional_skill"]
    assert cs["tau"] is None, f"a value was reported on debris: {cs['tau']}"
    assert cs["refused"]
    assert cs["residual_scale"] < 10


def test_exact_monotone_prints_no_value_and_says_why(tmp_path, capsys):
    rng = np.random.default_rng(SEED)
    o, prim = _staged(3000, 12, rng)
    v = _vecs(tmp_path, score=np.exp(prim / 3.0), ordinal=o, gc=prim)
    _run(["check", "--score", v["score"], "--ordinal", v["ordinal"],
          "--ordinal-source", "experimental", "--primitive", "gene-count",
          "--gene-count", v["gc"], "--boot", "40"], tmp_path)
    text = capsys.readouterr().out
    assert "NO VALUE" in text
    assert "rounding" in text


def _debris_sweep():
    """Sixteen cells of pure rounding debris: 4 seeds x 2 sample sizes x 2 couplings."""
    from scipy.stats import kendalltau, weightedtau
    rows = []
    for seed in (20260904, 1, 2, 3):
        for n in (4000, 39505):
            for coupling in (0.5, 1.0):
                rng = np.random.default_rng(seed)
                levels = 12
                o = np.repeat(np.arange(levels, dtype=float), n // levels)
                o = np.concatenate([o, np.full(n - len(o), float(levels - 1))])
                prim = coupling * o + rng.normal(0, 1.0, n)
                resid = rank_resid_multi(align(np.exp(prim / 3.0), o), [prim])
                rows.append((float(np.abs(resid).max()),
                             float(kendalltau(resid, o).statistic),
                             float(weightedtau(resid, o).statistic)))
    return rows


def test_debris_is_scored_by_both_kernels_when_the_primitive_predicts_the_ordinal():
    """
    Defect F-2, restated. It was recorded as a weightedtau problem on a fixture
    whose ordinal is independent of the primitive; there tau_b stays near 0.02.
    When the primitive predicts the ordinal, which is Null B and the case the
    audit faces, the debris inherits that association and BOTH kernels score it.

    The assertions are on the maximum rather than on every cell, because the
    debris is the rounding error of a LAPACK least-squares fit and its size
    depends on the machine. Measured: aarch64 with numpy 2.2 puts all sixteen
    cells between 0.71 and 0.87 under tau_b, while x86_64 with numpy 2.4 puts
    fourteen there and collapses two toward zero, the two where the residual
    itself lands two to three orders smaller. What is stable, and what the
    manuscript rests on, is that a score which is an exact reparametrisation of
    its primitive can be scored at 0.87.
    """
    rows = _debris_sweep()
    assert all(m < 1e-9 for m, _, _ in rows), "setup: every residual must be float noise"
    taub = [abs(t) for _, t, _ in rows]
    wtau = [abs(w) for _, _, w in rows]
    assert max(taub) > 0.8, f"tau_b no longer scores the debris: max {max(taub):.3f}"
    assert max(wtau) > 0.8, f"weightedtau no longer scores it: max {max(wtau):.3f}"
    assert sum(v > 0.6 for v in taub) >= 12, taub
    signs = {np.sign(t) for _, t, _ in rows}
    assert len(signs) > 1, "the sign should not be constant; the debris is not a measurement"


def test_the_debris_magnitude_is_a_property_of_the_machine():
    """
    The residual at rank identity spans orders of magnitude across cells, which
    is why the value a kernel returns on it cannot be quoted as a constant and
    why the guard is on the residual's size rather than on its tau.
    """
    mags = sorted(m for m, _, _ in _debris_sweep())
    assert mags[-1] / max(mags[0], 1e-300) > 100, mags


def test_debris_stays_near_zero_under_taub_when_the_ordinal_is_independent():
    """The original F-2 fixture, kept so the boundary between the two is explicit."""
    from scipy.stats import kendalltau, weightedtau
    rng = np.random.default_rng(20260813)
    n = 3000
    prim = rng.standard_normal(n)
    o = np.floor(6 * np.random.default_rng(20260813).random(n))
    resid = rank_resid_multi(align(np.exp(prim), o), [prim])
    assert abs(kendalltau(resid, o).statistic) < 0.05
    assert abs(weightedtau(resid, o).statistic) > 0.4


def test_ccat_residual_is_not_debris():
    """
    The paper's CCAT row sits at rho = 0.998, not at rank identity. Its residual
    is many orders above the arithmetic noise floor, so F-2 does not reach it.
    Pinned with a synthetic stand-in so the test runs without the atlas.
    """
    from own_baseline.cli import residual_scale
    rng = np.random.default_rng(11)
    n = 20000
    o = np.repeat(np.arange(12, dtype=float), n // 12)
    o = np.concatenate([o, np.full(n - len(o), 11.0)])
    prim = o + rng.normal(0, 1.0, n)
    score = prim + rng.normal(0, 0.06, n)          # rho about 0.998
    ratio, _ = residual_scale(align(score, o), [prim])
    assert ratio > 1e12, f"residual scale {ratio:.3g} is in the debris regime"


# ------------------------------------------------------------------ F2

def test_a_score_with_real_skill_clears_its_floor(tmp_path):
    rng = np.random.default_rng(SEED + 1)
    n = 4000
    o, prim = _staged(n, 12, rng)
    score = 0.4 * prim + 1.6 * o + rng.normal(0, 1.0, n)   # skill beyond prim
    v = _vecs(tmp_path, score=score, ordinal=o, gc=prim)
    code, rep = _run(["check", "--score", v["score"], "--ordinal", v["ordinal"],
                      "--ordinal-source", "experimental",
                      "--primitive", "gene-count", "--gene-count", v["gc"],
                      "--boot", "60"], tmp_path)
    assert code == 0
    row = rep["by_primitive"]["gene_count"]
    assert row["conditional_skill"]["tau"] > row["null"]["floor"]
    assert row["conditional_skill"]["CI95"][0] > 0


# ------------------------------------------------------------------ F3

def test_direction_inversion_is_named_before_any_margin(tmp_path, capsys):
    """
    The sorted-haematopoiesis shape: the ordinal is the potency hierarchy and
    both the score and the gene-count primitive rank cells against it. The tool
    must say so, and must say it above the numbers.
    """
    rng = np.random.default_rng(SEED + 2)
    n = 3000
    o = np.repeat(np.arange(4, dtype=float), n // 4)
    prim = -o + rng.normal(0, 1.0, n)          # depth runs against potency
    score = 0.9 * prim + rng.normal(0, 0.6, n)  # the score follows depth
    v = _vecs(tmp_path, score=score, ordinal=o, gc=prim)
    code, rep = _run(["check", "--score", v["score"], "--ordinal", v["ordinal"],
                      "--ordinal-source", "experimental",
                      "--primitive", "gene-count", "--gene-count", v["gc"],
                      "--boot", "40", "--no-floors"], tmp_path)
    assert code == 0
    d = rep["direction_check"]
    assert d["score_runs_against_ordinal"] is True
    assert d["score_auc_before_alignment"] < 0.5
    assert "gene_count" in d["primitives_below_chance"]

    text = capsys.readouterr().out
    assert "points AGAINST the ordinal" in text
    assert text.index("DIRECTION CHECK") < text.index("conditional skill"), \
        "the direction block must appear above the first margin"


# ------------------------------------------------------------------ F4

def test_cli_estimator_equals_the_library(tmp_path):
    rng = np.random.default_rng(SEED + 3)
    n = 1200
    o, prim = _staged(n, 12, rng)
    score = 0.5 * prim + 1.0 * o + rng.normal(0, 1.0, n)

    lib = conditional_skill_report(score, o, {"gene_count": prim},
                                   n_boot=(20, 120), joint=False, verbose=False)
    lib_row = lib["by_primitive"]["gene_count"]["conditional_skill"]["kang_wdm_taub"]

    mine = conditional_skill(align(score, o), o, [prim], "taub", 120, 42)
    assert mine["tau"] == pytest.approx(lib_row["tau"], abs=0, rel=0)
    assert mine["CI95"] == pytest.approx(lib_row["CI95"], abs=0, rel=0)


# ------------------------------------------------------------------ F5

@pytest.mark.parametrize("n,rho,expected", [
    (39505, 0.4844, 0.0247),     # CytoTRACE v1 | gene count
    (39505, 0.9305, 0.0311),     # SCENT SR | PCC(x, degree)
    (39505, 0.9982, 0.0115),     # CCAT | PCC(x, degree)
    (3000,  0.3778, 0.0411),     # NCG, on the 3,000-cell subsample
    (3000,  0.9317, 0.0479),     # SLICE, same subsample
])
def test_published_floors_come_out_of_the_shipped_grid(n, rho, expected):
    got = floors.floor(n, rho, "taub", 1, 12)["floor"]
    assert got == pytest.approx(expected, abs=5e-5)


def test_joint_four_covariate_floor():
    assert floors.floor(39505, 0.5, "taub", 4, 12)["floor"] == pytest.approx(0.0171, abs=5e-5)


def test_floors_refuse_to_extrapolate():
    r = floors.floor(120, 0.5, "taub", 1, 12)
    assert r["floor"] is None and "outside the measured grid" in r["reason"]


# ------------------------------------------------------------------ F6

def test_derived_ordinal_is_refused(tmp_path):
    v = _vecs(tmp_path, score=np.arange(300.0), ordinal=np.arange(300.0) % 5,
              gc=np.arange(300.0))
    code, _ = _run(["check", "--score", v["score"], "--ordinal", v["ordinal"],
                    "--ordinal-source", "derived", "--gene-count", v["gc"],
                    "--primitive", "gene-count"], tmp_path)
    assert code == 3


def test_binary_ordinal_is_refused(tmp_path):
    rng = np.random.default_rng(SEED + 4)
    o = rng.integers(0, 2, 500).astype(float)
    v = _vecs(tmp_path, score=rng.normal(size=500), ordinal=o,
              gc=rng.normal(size=500))
    code, _ = _run(["check", "--score", v["score"], "--ordinal", v["ordinal"],
                    "--ordinal-source", "experimental", "--gene-count", v["gc"],
                    "--primitive", "gene-count"], tmp_path)
    assert code == 3


def test_pseudotime_named_score_needs_an_explicit_acknowledgement(tmp_path):
    rng = np.random.default_rng(SEED + 5)
    n = 600
    o, prim = _staged(n, 12, rng)
    d = tmp_path / "dpt_pseudotime.npy"
    np.save(d, prim + rng.normal(0, 1, n))
    v = _vecs(tmp_path, ordinal=o, gc=prim)
    args = ["check", "--score", str(d), "--ordinal", v["ordinal"],
            "--ordinal-source", "experimental", "--gene-count", v["gc"],
            "--primitive", "gene-count", "--boot", "20"]
    code, _ = _run(args, tmp_path)
    assert code == 3
    code, rep = _run([*args, "--i-checked-this-is-not-derived"], tmp_path)
    assert code == 0 and rep is not None


# ------------------------------------------------------------------ misc

def test_ordinal_auc_matches_roc_auc_on_two_levels():
    rng = np.random.default_rng(7)
    o = rng.integers(0, 2, 3000).astype(float)
    x = o + rng.normal(0, 1, 3000)
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        pytest.skip("sklearn not installed")
    assert ordinal_auc(x, o) == pytest.approx(roc_auc_score(o, x), abs=1e-12)


def test_ordinal_auc_is_antisymmetric():
    rng = np.random.default_rng(8)
    o = rng.integers(0, 12, 2000).astype(float)
    x = o + rng.normal(0, 3, 2000)
    assert ordinal_auc(x, o) + ordinal_auc(-x, o) == pytest.approx(1.0, abs=1e-12)


def test_entry_point_is_installed_and_runs():
    r = subprocess.run([sys.executable, "-m", "own_baseline.cli",
                        "floors", "--n", "39505", "--rho", "0.4844"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "+0.0247" in r.stdout


# ------------------------------------------------------------------ F7
# floor selection away from a design point

def test_floor_below_the_query_n_is_labelled_conservative():
    """
    The floor falls monotonically with n, so a grid point below the query gives
    a threshold that is too high. Clearing it is the safe direction to be wrong
    in, and the result must say so rather than leaving the caller to work it out.
    """
    f = floors.floor(1500, 0.93, "taub", 1, 12)
    assert f["grid_n"] == 500 and f["n_side"] == "conservative"


def test_floor_above_the_query_n_is_labelled_permissive():
    f = floors.floor(2600, 0.5, "taub", 1, 12)
    assert f["grid_n"] == 3000 and f["n_side"] == "permissive"


def test_floor_on_a_design_point_is_labelled_exact():
    assert floors.floor(39505, 0.5, "taub", 1, 12)["n_side"] == "exact"


def test_the_floor_is_monotone_decreasing_in_n():
    """The claim the conservative/permissive labelling rests on."""
    vals = [floors.floor(n, 0.5, "taub", 1, 12)["floor"]
            for n in (500, 3000, 10000, 39505, 127607)]
    assert vals == sorted(vals, reverse=True), vals


def test_unmeasured_covariate_count_is_bracketed_not_rounded():
    """
    The floor is not monotone in the covariate count: it mostly falls and rises
    again at rho 0 and rho 0.998. So an unmeasured k takes the larger of the two
    measured values that bracket it.
    """
    f3 = floors.floor(39505, 0.5, "taub", 3, 12)
    f2 = floors.floor(39505, 0.5, "taub", 2, 12)
    f4 = floors.floor(39505, 0.5, "taub", 4, 12)
    assert f3["k_bracketed"] and f3["k_used"] == [2, 4]
    assert f3["floor"] == pytest.approx(max(f2["floor"], f4["floor"]))


def test_covariate_count_outside_the_bracket_is_refused():
    r = floors.floor(39505, 0.5, "taub", 9, 12)
    assert r["floor"] is None and "does not bracket" in r["reason"]


def test_the_floor_is_not_monotone_in_the_covariate_count():
    """Pins the reason bracketing exists rather than rounding down."""
    at = lambda k: floors.floor(39505, 0.998, "taub", k, 12)["floor"]
    assert not (at(1) >= at(2) >= at(4)), (at(1), at(2), at(4))


# ------------------------------------------------------------------ F8
# the h5ad path, which the README documents as the primary one

@pytest.fixture(scope="module")
def h5ad(tmp_path_factory):
    ad = pytest.importorskip("anndata")
    import pandas as pd
    import scipy.sparse as sp
    rng = np.random.default_rng(3)
    n_cells, n_genes, levels = 900, 200, 12
    stage = np.repeat(np.arange(levels, dtype=float), n_cells // levels)
    depth = np.clip(1000 - 55 * stage + rng.normal(0, 40, n_cells), 40, None)
    X = np.zeros((n_cells, n_genes))
    for i in range(n_cells):
        k = int(min(n_genes, max(5, depth[i] / 5)))
        X[i, rng.choice(n_genes, k, replace=False)] = rng.poisson(3, k) + 1
    obs = pd.DataFrame(
        {"stage": stage,
         "cytotrace": -(0.5 * depth / 100 + 1.4 * stage + rng.normal(0, 1, n_cells))},
        index=[f"c{i}" for i in range(n_cells)])
    p = tmp_path_factory.mktemp("h5ad") / "demo.h5ad"
    ad.AnnData(X=sp.csr_matrix(X), obs=obs,
               var=pd.DataFrame(index=[f"G{j}" for j in range(n_genes)])).write_h5ad(p)
    return str(p)


def test_h5ad_path_computes_three_primitives_from_the_matrix(h5ad, tmp_path):
    code, rep = _run([ "check", h5ad, "--score", "obs:cytotrace",
                       "--ordinal", "obs:stage", "--ordinal-source", "experimental",
                       "--boot", "30", "--no-floors"], tmp_path)
    assert code == 0
    assert set(rep["by_primitive"]) >= {"gene_count", "Shannon_H",
                                        "log10_library_size", "JOINT"}
    assert [m["primitive"] for m in rep["primitives_not_computed"]] == ["PCC(x,degree)"]
    assert "no interactome is bundled" in \
        rep["primitives_not_computed"][0]["reason"].lower()


def test_h5ad_missing_obs_column_names_what_is_there(h5ad, tmp_path):
    code, _ = _run(["check", h5ad, "--score", "obs:nope", "--ordinal", "obs:stage",
                    "--ordinal-source", "experimental"], tmp_path)
    assert code == 2


def test_scaffold_npz_is_accepted_under_either_key_spelling(h5ad, tmp_path):
    from own_baseline.cli import load_scaffold
    import argparse
    rng = np.random.default_rng(0)
    for ci, dg in (("col_idx", "degree"), ("col_atlas", "deg_sub")):
        p = tmp_path / f"{ci}.npz"
        np.savez(p, **{ci: np.arange(50), dg: rng.random(50) * 10})
        args = argparse.Namespace(scaffold=str(p), string_links=None, string_info=None)
        sc = load_scaffold(args, None)
        assert len(sc["col_idx"]) == 50 and len(sc["degree"]) == 50


def test_scaffold_with_mismatched_lengths_is_refused(tmp_path):
    from own_baseline.cli import load_scaffold
    import argparse
    p = tmp_path / "bad.npz"
    np.savez(p, col_idx=np.arange(50), degree=np.arange(10.0))
    args = argparse.Namespace(scaffold=str(p), string_links=None, string_info=None)
    with pytest.raises(SystemExit):
        load_scaffold(args, None)


# ------------------------------------------------------------------ F9
# the two verbs must compose

def test_primitives_then_check_round_trips(h5ad, tmp_path):
    """
    `primitives` writes an npz and `check --primitives` reads it straight back.
    Without this the first verb produces a file the second cannot use, which is
    a workflow that only looks complete.
    """
    out = tmp_path / "prim.npz"
    code = 0
    try:
        main(["primitives", h5ad, "--out", str(out)])
    except SystemExit as e:
        code = int(e.code or 0)
    assert code == 0 and out.is_file()

    z = np.load(out, allow_pickle=False)
    assert {"gene_count", "shannon_H", "log10_library_size"} <= set(z.files)
    meta = json.loads(str(z["_meta"]))
    assert meta["n_cells"] == 900
    assert set(meta["sha256"]) == {"gene_count", "Shannon_H", "log10_library_size"}
    assert "PCC(x,degree)" in meta["not_computed"]

    code, rep = _run(["check", h5ad, "--score", "obs:cytotrace",
                      "--ordinal", "obs:stage", "--ordinal-source", "experimental",
                      "--primitives", str(out), "--boot", "30", "--no-floors"],
                     tmp_path)
    assert code == 0
    assert set(rep["by_primitive"]) >= {"gene_count", "Shannon_H",
                                        "log10_library_size", "JOINT"}
    assert rep["receipt"]["primitives_from"] == str(out)


def test_check_recomputes_the_same_numbers_it_was_handed(h5ad, tmp_path):
    """A primitive read from a file must give the same answer as one computed."""
    out = tmp_path / "prim.npz"
    try:
        main(["primitives", h5ad, "--out", str(out)])
    except SystemExit:
        pass
    common = ["check", h5ad, "--score", "obs:cytotrace", "--ordinal", "obs:stage",
              "--ordinal-source", "experimental", "--boot", "20", "--no-floors"]
    _, a = _run(common, tmp_path)
    _, b = _run([*common, "--primitives", str(out)], tmp_path)
    for k in ("gene_count", "Shannon_H", "log10_library_size", "JOINT"):
        assert a["by_primitive"][k]["conditional_skill"]["tau"] == pytest.approx(
            b["by_primitive"][k]["conditional_skill"]["tau"], abs=0, rel=0), k


def test_primitives_npz_with_a_foreign_key_is_refused(h5ad, tmp_path):
    bad = tmp_path / "bad.npz"
    np.savez(bad, gene_count=np.ones(900), something_else=np.ones(900))
    code, _ = _run(["check", h5ad, "--score", "obs:cytotrace",
                    "--ordinal", "obs:stage", "--ordinal-source", "experimental",
                    "--primitives", str(bad)], tmp_path)
    assert code == 2


def test_version_flag():
    from own_baseline import __version__
    with pytest.raises(SystemExit) as e:
        main(["--version"])
    assert e.value.code == 0


def test_the_wheel_checker_rejects_a_wheel_without_the_floors_table(tmp_path):
    """
    tools/check_wheel.py is the only thing standing between a packaging mistake
    and a release where `ownbaseline floors` raises for every installed user.
    """
    import subprocess
    import zipfile
    whl = tmp_path / "fake-0.0.0-py3-none-any.whl"
    with zipfile.ZipFile(whl, "w") as z:
        z.writestr("own_baseline/cli.py", "")
        z.writestr("own_baseline/floors.py", "")
        z.writestr("own_baseline/conditional_skill.py", "")
        z.writestr("fake-0.0.0.dist-info/entry_points.txt",
                   "[console_scripts]\nownbaseline = own_baseline.cli:main\n")
    tool = Path(__file__).resolve().parents[1] / "tools/check_wheel.py"
    r = subprocess.run([sys.executable, str(tool), str(whl)],
                       capture_output=True, text=True)
    assert r.returncode != 0
    assert "null_floors" in (r.stdout + r.stderr)


# ----------------------------------------------------------------- F10
# terminal styling: it must never change what the output says

def _style_with(env, monkeypatch):
    import importlib
    for k in ("OWNBASELINE_COLOR", "NO_COLOR", "TERM", "COLORTERM"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    import own_baseline.style as st
    return importlib.reload(st)


def test_no_colour_when_stdout_is_not_a_terminal(monkeypatch):
    st = _style_with({"TERM": "xterm-256color"}, monkeypatch)
    assert st.DEPTH == 0
    assert st.accent("x") == "x" and st.head("VERDICT") == "VERDICT"


def test_no_color_environment_variable_is_honoured(monkeypatch):
    st = _style_with({"NO_COLOR": "1", "TERM": "xterm-256color"}, monkeypatch)
    assert st.DEPTH == 0


def test_truecolor_is_used_when_the_terminal_says_so(monkeypatch):
    st = _style_with({"OWNBASELINE_COLOR": "1", "COLORTERM": "truecolor"}, monkeypatch)
    assert st.DEPTH == 24
    assert "38;2;168;85;247" in st.accent("x")


def test_it_falls_back_to_256_and_then_to_eight(monkeypatch):
    st = _style_with({"OWNBASELINE_COLOR": "1", "TERM": "xterm-256color"}, monkeypatch)
    assert st.DEPTH == 8 and "38;5;141" in st.accent("x")


def test_styling_never_changes_the_text(monkeypatch, capsys, h5ad, tmp_path):
    """
    The same run with and without colour must differ only in escape sequences.
    A reader piping the output into a file loses the scannability and nothing
    else, so no number and no word may depend on whether colour is on.
    """
    import importlib
    import re

    import own_baseline.cli as cli
    import own_baseline.style as st

    args = ["check", h5ad, "--score", "obs:cytotrace", "--ordinal", "obs:stage",
            "--ordinal-source", "experimental", "--boot", "20", "--no-floors"]

    def run(env):
        for k in ("OWNBASELINE_COLOR", "NO_COLOR", "TERM", "COLORTERM"):
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        importlib.reload(st)
        importlib.reload(cli)
        capsys.readouterr()
        cli.main([*args, "--json", str(tmp_path / "same.json")])
        return capsys.readouterr().out

    plain = run({"OWNBASELINE_COLOR": "0"})
    coloured = run({"OWNBASELINE_COLOR": "1", "COLORTERM": "truecolor"})

    # leave the modules in the no-colour state for whatever runs next
    monkeypatch.delenv("COLORTERM", raising=False)
    monkeypatch.setenv("OWNBASELINE_COLOR", "0")
    importlib.reload(st)
    importlib.reload(cli)

    assert "\x1b[" in coloured and "\x1b[" not in plain
    assert re.sub(r"\x1b\[[0-9;]*m", "", coloured) == plain


def test_the_declared_version_matches_the_one_pip_resolves():
    """
    __version__ is a literal in __init__.py, and the version pip resolves is a
    different literal in pyproject.toml. 0.2.1 shipped with the first still at
    0.2.0, so every receipt that release wrote carries a version that was never
    released, and `ownbaseline --version` answered with it too. Nothing compared
    them, so CI stayed green through the whole thing.
    """
    import re

    import own_baseline
    pj = Path(own_baseline.__file__).resolve().parents[1] / "pyproject.toml"
    if not pj.is_file():
        pytest.skip("installed without the source tree")
    m = re.search(r'^version = "([^"]+)"', pj.read_text(), re.M)
    assert m, "pyproject.toml declares no version"
    assert own_baseline.__version__ == m.group(1), (
        f"__init__.py says {own_baseline.__version__}, "
        f"pyproject.toml says {m.group(1)}")


def test_the_author_line_lives_in_exactly_one_place():
    """
    The review copy blanks own_baseline.__author__ and nothing else, so the
    name must not be written out anywhere a second time.
    """
    import own_baseline
    root = Path(own_baseline.__file__).resolve().parents[1]
    author = own_baseline.__author__
    if not author:
        pytest.skip("this is the anonymised copy")
    hits = []
    for p in sorted(root.rglob("*.py")):
        if ".git" in p.parts or "__pycache__" in p.parts:
            continue
        for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
            if author in line:
                hits.append(f"{p.relative_to(root)}:{i}")
    assert hits == ["own_baseline/__init__.py:32"] or len(hits) == 1, hits


# ----------------------------------------------------------------- F11
# the CLI conventions, from clig.dev

def test_json_dash_puts_json_on_stdout_and_prose_on_stderr(h5ad, tmp_path, capsys):
    """
    `--json -` has to be pipeable. Anything human on stdout would break the
    reader on the other end of the pipe, so it goes to stderr instead.
    """
    code = 0
    try:
        main(["check", h5ad, "--score", "obs:cytotrace", "--ordinal", "obs:stage",
              "--ordinal-source", "experimental", "--boot", "20", "--no-floors",
              "--json", "-"])
    except SystemExit as e:
        code = int(e.code or 0)
    assert code == 0
    cap = capsys.readouterr()
    rep = json.loads(cap.out)                      # stdout must parse whole
    assert rep["n_used"] == 900
    assert "VERDICT" in cap.err                    # the prose went the other way


def test_quiet_keeps_the_numbers_and_drops_the_prose(h5ad, tmp_path, capsys):
    common = ["check", h5ad, "--score", "obs:cytotrace", "--ordinal", "obs:stage",
              "--ordinal-source", "experimental", "--boot", "20", "--no-floors"]
    main(common)
    loud = capsys.readouterr().out
    main([*common, "--quiet"])
    quiet = capsys.readouterr().out
    assert "conditional skill" in quiet and "VERDICT" in quiet
    assert "Scope:" in loud and "Scope:" not in quiet
    assert len(quiet.splitlines()) < len(loud.splitlines())


def test_quiet_and_no_color_are_accepted_before_and_after_the_verb(h5ad, tmp_path):
    """A global flag a user has to put in one particular place is a trap."""
    args = ["check", h5ad, "--score", "obs:cytotrace", "--ordinal", "obs:stage",
            "--ordinal-source", "experimental", "--boot", "10", "--no-floors"]
    for variant in ([args[0], "--quiet", *args[1:]], [*args, "--quiet"],
                    [args[0], "--no-color", *args[1:]], [*args, "--no-color"]):
        code = 0
        try:
            main(variant)
        except SystemExit as e:
            code = int(e.code or 0)
        assert code == 0, variant


def test_the_frame_is_dropped_rather_than_wrapped_on_a_narrow_terminal(monkeypatch):
    import importlib
    import own_baseline.style as st
    monkeypatch.setenv("OWNBASELINE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
    importlib.reload(st)
    wide = st.frame(["a line of some length here"])
    monkeypatch.setenv("COLUMNS", "12")
    narrow = st.frame(["a line of some length here"])
    assert len(wide) == 3 and "╭" in wide[0]
    assert narrow == ["a line of some length here"]
    monkeypatch.setenv("OWNBASELINE_COLOR", "0")
    importlib.reload(st)


def test_the_frame_width_ignores_escape_sequences(monkeypatch):
    import importlib
    import own_baseline.style as st
    monkeypatch.setenv("OWNBASELINE_COLOR", "1")
    monkeypatch.setenv("COLUMNS", "200")
    importlib.reload(st)
    plain = st.frame(["abcdefghij"])
    styled = st.frame([st.accent("abcdefghij")])
    assert st.printable(plain[0]) == st.printable(styled[0])
    monkeypatch.setenv("OWNBASELINE_COLOR", "0")
    importlib.reload(st)


def test_force_color_turns_colour_on(monkeypatch):
    import importlib
    import own_baseline.style as st
    for k in ("OWNBASELINE_COLOR", "NO_COLOR", "COLORTERM"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.setenv("TERM", "xterm-256color")
    importlib.reload(st)
    assert st.DEPTH > 0
    monkeypatch.setenv("OWNBASELINE_COLOR", "0")
    importlib.reload(st)


def test_a_mistyped_verb_names_the_nearest_real_one(capsys):
    with pytest.raises(SystemExit) as e:
        main(["chekc"])
    assert e.value.code == 2
    assert "Did you mean 'check'" in capsys.readouterr().err


def test_help_leads_with_examples_and_documents_the_exit_codes(capsys):
    with pytest.raises(SystemExit):
        main(["--help"])
    text = capsys.readouterr().out
    assert "examples" in text and "ownbaseline check cells.h5ad" in text
    assert "exit codes" in text
    for code in ("0  ", "2  ", "3  "):
        assert code in text
    assert "docs/interpreting.md" in text
