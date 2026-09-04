#!/usr/bin/env python3
"""
Build the anonymised review copy of this repository as a zip.

Double-blind venues require the code to be anonymous too, links included. This
script produces that copy from the current working tree, so what a reviewer gets
matches what is on disk rather than the last commit.

Three edits, and it names none of them literally: the author string is read out
of own_baseline.__author__ and blanked wherever it appears, the clone line in the
README becomes an unzip line, and the repository URL leaves pyproject.toml. It
then greps the result for the author's own name, surname, e-mail and any home
path, and refuses to write the zip if anything survives. The refusal is the point
of the script; the edits are the easy part.

    python3 tools/make_review_copy.py --out review-copy.zip
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".cfg", ".yml", ".yaml", ".R",
                 ".r", ".json", ".sh", ".ini"}


def tracked_files(root: Path) -> list[str]:
    r = subprocess.run(["git", "-C", str(root), "ls-files"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("not a git checkout; this script copies what git tracks")
    return [f for f in r.stdout.splitlines() if f]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="review-copy.zip")
    ap.add_argument("--name", default="own-baseline",
                    help="the directory name inside the zip")
    ap.add_argument("--no-smoke", action="store_true",
                    help="skip running the copy after building it")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import own_baseline                                       # noqa: E402

    author = own_baseline.__author__
    surname = author.split()[-1] if author else ""
    forename = author.split()[0] if author else ""

    with tempfile.TemporaryDirectory() as td:
        stage = Path(td) / args.name
        for rel in tracked_files(root):
            src, dst = root / rel, stage / rel
            if not src.is_file():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

        # 1. the author string, wherever it appears
        for p in stage.rglob("*"):
            if not (p.is_file() and p.suffix in TEXT_SUFFIXES):
                continue
            t = p.read_text(errors="replace")
            if author and author in t:
                p.write_text(t.replace(author, ""))

        # 2. the clone line, and 3. the repository URL
        rd = stage / "README.md"
        if rd.is_file():
            t = rd.read_text()
            t = re.sub(r"git clone https://\S+",
                       f"unzip {Path(args.out).name}   "
                       f"# anonymised review copy", t)
            t = re.sub(r"cd [A-Za-z0-9._-]*potency[A-Za-z0-9._-]*",
                       f"cd {args.name}", t)
            rd.write_text(t)
        pj = stage / "pyproject.toml"
        if pj.is_file():
            pj.write_text(re.sub(r'\n\[project\.urls\]\nRepository = "[^"]*"\n',
                                 "\n", pj.read_text()))

        # the check that decides whether anything is written at all
        needles = [n for n in (author, surname, forename) if n]
        offenders = []
        for p in sorted(stage.rglob("*")):
            if not (p.is_file() and p.suffix in TEXT_SUFFIXES):
                continue
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                low = line.lower()
                hit = ([n for n in needles if n.lower() in low]
                       + [n for n in ("@gmail", "/Us" + "ers/", "/ho" + "me/")
                          if n.lower() in low]
                       + (["github.com/" + surname.lower()]
                          if surname and f"github.com/{surname.lower()}" in low
                          else []))
                if hit:
                    offenders.append(f"{p.relative_to(stage)}:{i}: {line.strip()[:90]}")
        if offenders:
            print("REFUSING to write the zip. These survived the edits:",
                  file=sys.stderr)
            for o in offenders[:40]:
                print("  " + o, file=sys.stderr)
            if len(offenders) > 40:
                print(f"  ... and {len(offenders) - 40} more", file=sys.stderr)
            return 1

        # the copy carries only what git tracks, so an uncommitted module is
        # simply absent from it and every import that needs it fails. Grepping
        # the copy would not notice. Running it does.
        if not args.no_smoke:
            checks = [
                [sys.executable, "-c",
                 "import own_baseline, own_baseline.cli, own_baseline.floors; "
                 "print(own_baseline.__version__)"],
                [sys.executable, "-m", "own_baseline.cli", "floors",
                 "--n", "39505", "--rho", "0.4844"],
                [sys.executable, "-m", "pytest", "tests/", "-q",
                 "--no-header", "-x"],
            ]
            for cmd in checks:
                r = subprocess.run(cmd, cwd=stage, capture_output=True, text=True)
                if r.returncode != 0:
                    print("REFUSING to write the zip. The copy does not run:",
                          file=sys.stderr)
                    print("  " + " ".join(cmd[1:]), file=sys.stderr)
                    tail = (r.stdout + r.stderr).strip().splitlines()[-15:]
                    for line in tail:
                        print("  " + line, file=sys.stderr)
                    return 1
            print("the copy imports, answers a published floor and passes its "
                  "own test suite")

        out = Path(args.out).resolve()
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(stage.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(stage.parent))
        n = len(zipfile.ZipFile(out).namelist())
        print(f"wrote {out}  ({n} files, {out.stat().st_size / 1e6:.1f} MB)")
        print("no author string, e-mail, repository URL or home path survives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
