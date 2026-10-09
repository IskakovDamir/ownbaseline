#!/usr/bin/env python3
"""
fetch_sources.py — place the third-party code and weights the w5 rows need,
at the commits recorded in DISCOVERY.md, under
$OWNBASELINE_DATA_ROOT/w5_new_rows/src/ (git-ignored). Prints what it found.

    python3 experiments/w5_new_rows/fetch_sources.py

Nothing here reads expression data or labels.

  FitDevo      github.com/jumphone/FitDevo           0c757e6 (fitdevo.R v1.2, BGW.rds)
  stemFinder   github.com/CahanLab/stemfinder         db8ef0e (R package 0.1.0)
  TCGAbiolinks git.bioconductor.org RELEASE_3_23      2f9d249 (2.40.0: R/Stemness.R,
                                                      data/SC_PCBC_stemSig.rda)
  MCE.m        not fetchable (OUP signed link behind a human check); the script
               reports whether reference/MCE.m has been placed by hand
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "own_baseline" / "paths.py").is_file())
sys.path.insert(0, str(REPO))
from own_baseline.paths import data_root  # noqa: E402

SRC = data_root() / "w5_new_rows" / "src"

SOURCES = {
    "FitDevo": ("https://github.com/jumphone/FitDevo", None,
                "0c757e609489867e83f32f6571ebcfce704c3124"),
    "stemfinder": ("https://github.com/CahanLab/stemfinder", None,
                   "db8ef0e1b4ad02d03165c734954bc9beb28e5900"),
    "TCGAbiolinks": ("https://git.bioconductor.org/packages/TCGAbiolinks",
                     "RELEASE_3_23", "2f9d2496241af1ad3950a23bfce7d434d1834516"),
}
FILES = {
    "FitDevo": ["fitdevo.R", "BGW.rds"],
    "stemfinder": ["R/run_stemFinder.R", "R/gene_set_score.R", "DESCRIPTION",
                   "data/s_genes_human.rda", "data/g2m_genes_human.rda"],
    "TCGAbiolinks": ["R/Stemness.R", "data/SC_PCBC_stemSig.rda", "DESCRIPTION"],
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git(*args, cwd=None):
    r = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout.strip()


def fetch(name, url, branch, sha):
    dest = SRC / name
    if not dest.is_dir():
        SRC.mkdir(parents=True, exist_ok=True)
        if branch:
            git("clone", "--depth", "50", "--branch", branch, url, str(dest))
        else:
            git("clone", url, str(dest))
    git("checkout", "--quiet", sha, cwd=dest)
    head = git("rev-parse", "HEAD", cwd=dest)
    if head != sha:
        raise SystemExit(f"{name}: HEAD {head} is not the recorded {sha}")
    return dest


def main():
    report = {"src_root": str(SRC), "sources": {}}
    for name, (url, branch, sha) in SOURCES.items():
        dest = fetch(name, url, branch, sha)
        entry = {"url": url, "branch": branch, "commit": sha,
                 "commit_date": git("show", "-s", "--format=%ci", "HEAD", cwd=dest),
                 "files": {}}
        for rel in FILES[name]:
            p = dest / rel
            entry["files"][rel] = ({"sha256": sha256(p), "bytes": p.stat().st_size}
                                   if p.is_file() else "MISSING")
        desc = dest / "DESCRIPTION"
        if desc.is_file():
            for line in desc.read_text().splitlines():
                if line.startswith("Version:"):
                    entry["package_version"] = line.split(":", 1)[1].strip()
        report["sources"][name] = entry
        print(f"{name:13s} {sha[:7]} {entry['commit_date']}  "
              f"version {entry.get('package_version', '-')}")
        for rel, v in entry["files"].items():
            print(f"    {rel:32s} " + (v if isinstance(v, str) else
                                       f"{v['bytes']:>9,} B  sha256 {v['sha256']}"))
    ref = REPO / "reference" / "MCE.m"
    report["MCE.m"] = ({"path": "reference/MCE.m", "sha256": sha256(ref)}
                       if ref.is_file() else "ABSENT: waiting for the author (DISCOVERY.md)")
    print(f"MCE.m         {report['MCE.m'] if isinstance(report['MCE.m'], str) else report['MCE.m']['sha256']}")
    out = data_root() / "w5_new_rows" / "sources.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
