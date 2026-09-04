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


@pytest.mark.parametrize("seed", [20260904, 1, 2, 3])
@pytest.mark.parametrize("coupling", [0.5, 1.0])
def test_debris_is_scored_by_both_kernels_when_the_primitive_predicts_the_ordinal(
        seed, coupling):
    """
    Defect F-2, restated. It was recorded as a weighted-tau problem on a fixture
    whose ordinal is independent of the primitive; there tau_b stays near 0.02.
    When the primitive predicts the ordinal, which is Null B and the case the
    audit actually faces, the debris inherits that association and BOTH kernels
    score it. This test exists so the claim is pinned rather than remembered.
    """
    from scipy.stats import kendalltau, weightedtau
    rng = np.random.default_rng(seed)
    n, levels = 4000, 12
    o = np.repeat(np.arange(levels, dtype=float), n // levels)
    o = np.concatenate([o, np.full(n - len(o), float(levels - 1))])
    prim = coupling * o + rng.normal(0, 1.0, n)
    resid = rank_resid_multi(align(np.exp(prim / 3.0), o), [prim])

    assert np.abs(resid).max() < 1e-9, "setup: the residual must be float noise"
    assert abs(kendalltau(resid, o).statistic) > 0.6, \
        "tau_b no longer scores the debris; the paper's F-2 wording can be relaxed"
    assert abs(weightedtau(resid, o).statistic) > 0.5


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
