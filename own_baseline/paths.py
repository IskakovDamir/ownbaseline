"""
paths.py — where this repository reads and writes.

Every script here used to hard-code an absolute path under the author's home
directory (and, for the R handoff scripts, an ephemeral agent-session
scratchpad). Nothing outside that one machine could import them. This module
replaces those constants with two roots, each an environment variable with a
repository-relative default.

    OWNBASELINE_DATA_ROOT   run inputs and outputs   default <repo>/data/runs
    OWNBASELINE_SCRATCH     R <-> Python handoff     default <repo>/data/scratch

Both defaults are inside the repository and both are git-ignored, so a clone
works out of the box and nothing large is ever committed by accident.

The directory layout under OWNBASELINE_DATA_ROOT is the one the experiment
scripts already assumed, unchanged:

    <data root>/
      track1/{data,results}/          track2/{data/prepared,results}/
      track4/{data,results}/          fix3/{data/prepared,results}/
      w1_gate2/{data,results}/        w4/{scaffolds,results}/
      atlas_run/…                     ct2_probe/results/

Nothing in this module affects a computation. It resolves locations only.
"""
from __future__ import annotations

import os
from pathlib import Path

__all__ = ["repo_root", "data_root", "scratch_root", "ensure"]

_MARKER = Path("own_baseline") / "paths.py"


def repo_root(start: Path | str | None = None) -> Path:
    """
    Directory containing own_baseline/paths.py, found by walking upward.

    Works whether the caller is an installed package, a script run by path, or
    a file executed from an arbitrary working directory.
    """
    here = Path(start).resolve() if start is not None else Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / _MARKER).is_file():
            return parent
    # Fall back to this file's own grandparent: own_baseline/paths.py -> repo.
    return Path(__file__).resolve().parent.parent


def _root(env_var: str, default_rel: str) -> Path:
    override = os.environ.get(env_var)
    if override:
        return Path(override).expanduser().resolve()
    return repo_root() / default_rel


def data_root() -> Path:
    """
    Root for experiment inputs and outputs.

    $OWNBASELINE_DATA_ROOT if set, else <repo>/data/runs. This is what the
    experiment scripts used to call VAULT. No expression data ships with the
    repository; see data/README.md for the accessions and how to fetch them.
    """
    return _root("OWNBASELINE_DATA_ROOT", "data/runs")


def scratch_root() -> Path:
    """
    Root for intermediate files exchanged between the Python and R halves of a
    score run (matrix in, score CSV out).

    $OWNBASELINE_SCRATCH if set, else <repo>/data/scratch. The R scripts read
    the same environment variable, so setting it once lines both halves up.
    """
    return _root("OWNBASELINE_SCRATCH", "data/scratch")


def ensure(*paths: Path) -> None:
    """mkdir -p each argument. Convenience for run scripts."""
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"repo_root    = {repo_root()}")
    print(f"data_root    = {data_root()}"
          f"{'' if data_root().exists() else '   (does not exist yet)'}")
    print(f"scratch_root = {scratch_root()}"
          f"{'' if scratch_root().exists() else '   (does not exist yet)'}")
