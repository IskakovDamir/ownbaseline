#!/usr/bin/env python3
"""
check_design_extension.py — does run_missing_cells.py --design reproduce the
shipped null grid where the grid already has the cell?

Runs three grid cells through --design (200 seeds; the seed streams differ from
the grid's, so the two are independent estimates of the same percentile) and
compares the 97.5th percentiles with own_baseline.floors. Synthetic nulls only.

    python3 experiments/w5_new_rows/check_design_extension.py
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(REPO))
from own_baseline import floors as F  # noqa: E402
from own_baseline.paths import data_root  # noqa: E402

CELLS = [(3000, 0.5, 12, 1), (3000, 0.93, 4, 1), (3000, 0.5, 12, 4), (39505, 0.93, 12, 4)]
KERN = {"kendalltau": "kendalltau", "weightedtau_rankTrue": "weightedtau_rankTrue"}
OUT = HERE / "discovery" / "design_extension_check.json"


def main():
    csv_out = data_root() / "w5_new_rows" / "nulls" / "design_extension_check.csv"
    cmd = [sys.executable, str(REPO / "experiments/null_calibration/run_missing_cells.py"),
           "--seeds", "200", "--out", str(csv_out)]
    for n, rho, lv, k in CELLS:
        cmd += ["--design", f"n={n},rho={rho},levels={lv},k={k}"]
    subprocess.run(cmd, cwd=REPO, check=True)
    rows = list(csv.DictReader(open(csv_out)))
    res = []
    for n, rho, lv, k in CELLS:
        spec = f"n={n},rho={rho},levels={lv},k={k}"
        for kern in KERN:
            mine = max(float(r["p97.5"]) for r in rows
                       if r["design"] == spec and r["kernel"] == kern)
            grid = F.floor(n, rho, kern, k, lv)
            res.append({"cell": spec, "kernel": kern, "design_p97.5": mine,
                        "grid_p97.5": grid["floor"], "grid_cell": grid["cell"],
                        "difference": mine - grid["floor"]})
            print(f"{spec:32s} {kern:22s} design {mine:+.4f}  grid {grid['floor']:+.4f}  "
                  f"diff {mine - grid['floor']:+.4f}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
