#!/usr/bin/env python3
"""
Assert that a built wheel carries the null grid, not only the code.

The floors table is package data. If setuptools does not ship it, the library
imports fine, the test suite passes because it reads the source tree, and
`ownbaseline floors` raises FileNotFoundError for everyone who installed the
package. That failure is invisible to every other check in this repository,
which is why it has one of its own.

    python3 tools/check_wheel.py dist/*.whl
"""
from __future__ import annotations

import sys
import zipfile

REQUIRED = ("own_baseline/data/null_floors.csv.gz",
            "own_baseline/cli.py",
            "own_baseline/floors.py",
            "own_baseline/conditional_skill.py")


def main(argv):
    if len(argv) != 1:
        sys.exit(f"usage: {sys.argv[0]} <wheel>")
    whl = argv[0]
    names = set(zipfile.ZipFile(whl).namelist())
    missing = [n for n in REQUIRED if n not in names]
    if missing:
        sys.exit(f"{whl} is missing {missing}\n"
                 f"check [tool.setuptools.package-data] in pyproject.toml")
    ep = [n for n in names if n.endswith("entry_points.txt")]
    if not ep:
        sys.exit(f"{whl} declares no console script; check [project.scripts]")
    text = zipfile.ZipFile(whl).read(ep[0]).decode()
    if "ownbaseline" not in text:
        sys.exit(f"{whl} entry points do not mention ownbaseline:\n{text}")
    print(f"{whl}: {len(names)} entries, floors table present, "
          f"console script declared")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
