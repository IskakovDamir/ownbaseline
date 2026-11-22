#!/usr/bin/env python3
"""
manifest.py: record the control files before any label is read.

    python3 experiments/specificity/manifest.py

Writes blind_manifest.json beside PREREG.md: for every .npz the label-blind stage wrote, its
SHA256 and the shape of every array in it. The file is committed with the plan, and analyse.py
reads it from that commit, so a control file changed after the lock cannot be opened.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import SERIES, WORK, sha256  # noqa: E402

EXPECT = {"signatures": {"a1_g200": 1000, "a1_g20": 200, "a1_g2000": 200, "a2": 1000},
          "cytotrace_controls": {"a3": 100, "a3_unsmoothed": 100}}


def main():
    out = {}
    for geo in SERIES:
        for kind in ("signatures", "cytotrace_controls", "thinned", "window"):
            path = WORK / f"{geo}_{kind}.npz"
            if kind in ("cytotrace_controls", "window") and geo == "GSE113074":
                continue
            z = np.load(path)
            shapes = {k: list(z[k].shape) for k in z.files}
            for key, cols in EXPECT.get(kind, {}).items():
                if shapes[key][1] != cols:
                    raise SystemExit(f"{path.name}:{key} has {shapes[key][1]} controls, the plan registers {cols}")
            out[path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size, "shapes": shapes}
    (HERE / "blind_manifest.json").write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    for name, m in out.items():
        print(m["sha256"], name)


if __name__ == "__main__":
    main()
