"""Corruption-proof segmented download of GSE113074 annotated counts.
Each byte-range part goes to its own file, verified by EXACT size, retried
independently; then concatenated in order. Avoids the curl -C- mid-resume
misalignment that corrupted earlier attempts. (NCBI throttles per-IP so this
is not faster, but it is CORRECT.)"""
import os, time, urllib.request
BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE113nnn/GSE113074/suppl/GSE113074_Raw_combined.annotated_counts.tsv.gz"
# --- repo-relative I/O roots -------------------------------------------------
# These were absolute paths under the author's home directory and an ephemeral
# agent-session scratchpad, so the module only imported on one machine.
# Override with $OWNBASELINE_DATA_ROOT / $OWNBASELINE_SCRATCH; defaults are
# <repo>/data/runs and <repo>/data/scratch. See own_baseline/paths.py.
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(next(p for p in _Path(__file__).resolve().parents
                             if (p / "own_baseline" / "paths.py").is_file())))
from own_baseline.paths import data_root  # noqa: E402
F = str(data_root() / "fix3/data/GSE113074_Raw_combined.annotated_counts.tsv.gz")
SIZE = 177842456
PARTS = 24
ps = SIZE // PARTS

def fetch(i, s, e):
    exp = e - s + 1
    pf = f"{F}.part{i}"
    if os.path.exists(pf) and os.path.getsize(pf) == exp:
        return True
    for attempt in range(8):
        try:
            req = urllib.request.Request(BASE, headers={"Range": f"bytes={s}-{e}"})
            with urllib.request.urlopen(req, timeout=180) as r, open(pf, "wb") as out:
                out.write(r.read())
            if os.path.getsize(pf) == exp:
                return True
            print(f"  part{i} attempt{attempt}: got {os.path.getsize(pf)}/{exp}, retry")
        except Exception as ex:
            print(f"  part{i} attempt{attempt} err: {ex}")
        time.sleep(4)
    return os.path.getsize(pf) == exp if os.path.exists(pf) else False

t0 = time.time()
for i in range(PARTS):
    s = i * ps
    e = SIZE - 1 if i == PARTS - 1 else (i + 1) * ps - 1
    ok = fetch(i, s, e)
    print(f"part {i+1}/{PARTS} [{s}-{e}] size_ok={ok} ({time.time()-t0:.0f}s)")
    assert ok, f"part {i} failed"

with open(F, "wb") as out:
    for i in range(PARTS):
        with open(f"{F}.part{i}", "rb") as p:
            out.write(p.read())
sz = os.path.getsize(F)
print(f"concatenated: {sz} bytes (expect {SIZE}) match={sz==SIZE}")
assert sz == SIZE
for i in range(PARTS):
    os.remove(f"{F}.part{i}")
print("DONE — parts cleaned")
