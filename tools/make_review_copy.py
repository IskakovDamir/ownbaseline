#!/usr/bin/env python3
"""
Build the anonymised review copy of this repository as a zip.

Double-blind venues require the code to be anonymous too, links included. This
script produces that copy from the current working tree, so what a reviewer gets
matches what is on disk rather than the last commit.

The edits name none of them literally. The author string is read out of
own_baseline.__author__ and blanked case-insensitively wherever it appears, the
install commands stop naming a repository, the whole [project.urls] table leaves
pyproject.toml, and any remaining GitHub URL under an owner whose handle carries
the author's name becomes a placeholder. Third-party repositories survive, since
scrubbing a citation would be a different kind of damage. CI workflows are left
out of the copy: they carry the repository owner, the deployment environment and
the name of the PyPI project, and a reviewer needs none of it.

It then greps the result for the author's own name, surname, e-mail and any home
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

# A workflow file names the repository owner, the deployment environment and the
# project it publishes to. None of that is reviewable and all of it is a leak.
# This script is left out for the same reason: its list of things to grep for is
# a description of the author.
SKIP_PREFIXES = (".github/", "tools/make_review_copy.py")

WITHHELD = "[repository URL withheld for review]"

# The Install section of the public README names the PyPI distribution, whose
# project page carries the repository owner and the uploading account. The
# review copy gets this instead.
ANON_INSTALL = """## Install

Unpack the zip and install from it. Python 3.10 or newer:

```bash
pip install .
ownbaseline                         # the splash, and the four verbs
```

"""


def tracked_files(root: Path) -> list[str]:
    r = subprocess.run(["git", "-C", str(root), "ls-files"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        sys.exit("not a git checkout; this script copies what git tracks")
    return [f for f in r.stdout.splitlines()
            if f and not f.startswith(SKIP_PREFIXES)]


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
        copied = []
        for rel in tracked_files(root):
            src, dst = root / rel, stage / rel
            if not src.is_file():
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            copied.append(rel)

        # 1. the install commands. They have to keep working from the
        #    unzipped directory, so the URL becomes a dot and not a placeholder.
        rd = stage / "README.md"
        if rd.is_file():
            t = rd.read_text()
            t = re.sub(r"git clone https://\S+",
                       f"unzip {Path(args.out).name}   "
                       f"# anonymised review copy", t)
            t = re.sub(r'"git\+https://\S+?"', ".", t)
            t = re.sub(r"cd (?:[A-Za-z0-9._-]*potency[A-Za-z0-9._-]*|ownbaseline)",
                       f"cd {args.name}", t)

            # The package is on PyPI, and its project page carries the owner's
            # GitHub and the uploader's account name. An install line naming it
            # is one click from the author, so the whole Install section is
            # replaced with an install from the zip, and any install command
            # elsewhere in the README stops naming the distribution.
            i = t.find("## Install")
            j = t.find("That gives the diagnostic", i + 1)
            if i != -1 and j != -1:
                t = t[:i] + ANON_INSTALL + t[j:]
            else:
                print("note: the Install section did not match; check the copy",
                      file=sys.stderr)
            t = re.sub(r"((?:pip|pipx|uv tool)\s+install\s+[^\n]*?)\bown-baseline\b",
                       r"\1.", t)
            rd.write_text(t)

        # 2. the whole [project.urls] table. Removing one key of it left Issues
        #    behind, which is how this was found.
        pj = stage / "pyproject.toml"
        if pj.is_file():
            pj.write_text(re.sub(r"\n\[project\.urls\]\n[^\[]*?(?=\n\[|\Z)",
                                 "\n", pj.read_text()))

        # 3. the author string, and whatever still points at the author's own
        #    GitHub. Case-insensitively: a name shouted in a docstring is the
        #    same leak as the same name in a metadata field, and the check below
        #    is case-insensitive, so a case-sensitive edit here only fails later.
        #    Third-party repositories are left alone; they are citations.
        author_re = re.compile(re.escape(author), re.I) if author else None
        gh_re = re.compile(r"(?:https://)?github\.com/(?P<owner>[A-Za-z0-9._-]+)"
                           r"/[A-Za-z0-9._-]+(?:/[^\s\"\'`)\]]*)?")
        mine = [n.lower() for n in (surname, forename) if n]

        def _scrub(m):
            owner = m.group("owner").lower()
            return WITHHELD if any(n in owner for n in mine) else m.group(0)

        for p in stage.rglob("*"):
            if not (p.is_file() and p.suffix in TEXT_SUFFIXES):
                continue
            t = orig = p.read_text(errors="replace")
            if author_re:
                t = author_re.sub("", t)
            t = gh_re.sub(_scrub, t)
            if t != orig:
                p.write_text(t)

        # the check that decides whether anything is written at all
        needles = [n for n in (author, surname, forename) if n]
        offenders = []
        for p in sorted(stage.rglob("*")):
            if not (p.is_file() and p.suffix in TEXT_SUFFIXES):
                continue
            for i, line in enumerate(p.read_text(errors="replace").splitlines(), 1):
                low = line.lower()
                hit = ([n for n in needles if n.lower() in low]
                       + [n for n in ("@gm" + "ail", "/Us" + "ers/", "/ho" + "me/",
                                      "install own-base" + "line",
                                      "pypi.org/project/own-base" + "line")
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
                # the splash reads __author__, which this script has just
                # emptied. An author-derived monogram has to degrade to nothing
                # rather than raise, and only running it shows that.
                [sys.executable, "-m", "own_baseline.cli", "--no-color"],
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
        # Only what git tracked. The smoke run above leaves __pycache__ and
        # .pytest_cache in the stage, the grep ran before it and never saw them,
        # and a .pyc is not a text suffix so it would never be grepped at all.
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            for rel in copied:
                f = stage / rel
                if f.is_file():
                    z.write(f, Path(args.name) / rel)
        n = len(zipfile.ZipFile(out).namelist())
        print(f"wrote {out}  ({n} files, {out.stat().st_size / 1e6:.1f} MB)")
        print("no author string, e-mail, repository URL or home path survives")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
