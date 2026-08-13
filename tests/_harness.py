"""
Minimal test harness.

The tests are plain functions named test_* using plain assert, so pytest
collects them natively if it is installed. They also run with nothing but the
standard library via `python3 tests/run_tests.py`, so a reader can verify the
suite without installing a test runner.

@known_defect marks a test that SHOULD pass and currently does not. It is not a
way to hide a failure: the runner prints it in its own section with the finding
id, and if such a test ever starts passing the runner reports XPASS and exits
non-zero, so the marker cannot outlive the defect silently.
"""
from __future__ import annotations

import traceback

KNOWN_DEFECTS: dict[str, tuple[str, str]] = {}


def known_defect(finding_id: str, reason: str):
    """Mark a test as an expected failure that documents a real defect."""
    def deco(fn):
        fn._known_defect = (finding_id, reason)
        KNOWN_DEFECTS[fn.__name__] = (finding_id, reason)
        try:  # let pytest treat it the same way, strictly
            import pytest
            return pytest.mark.xfail(strict=True, reason=f"{finding_id}: {reason}")(fn)
        except ImportError:
            return fn
    return deco


def run(modules) -> int:
    passed, failed, xfailed, xpassed = [], [], [], []
    for mod in modules:
        for name in sorted(vars(mod)):
            if not name.startswith("test_"):
                continue
            fn = vars(mod)[name]
            if not callable(fn):
                continue
            label = f"{mod.__name__}.{name}"
            defect = getattr(fn, "_known_defect", None)
            try:
                fn()
            except Exception:  # noqa: BLE001
                if defect:
                    xfailed.append((label, defect, traceback.format_exc()))
                else:
                    failed.append((label, traceback.format_exc()))
            else:
                (xpassed if defect else passed).append(
                    (label, defect) if defect else label)

    print(f"\npassed  {len(passed)}")
    for label in passed:
        print(f"  .  {label}")

    if xfailed:
        print(f"\nknown defects, still failing  {len(xfailed)}")
        for label, (fid, reason), tb in xfailed:
            print(f"  x  {label}")
            print(f"       {fid}: {reason}")
            print(f"       {tb.strip().splitlines()[-1]}")

    if xpassed:
        print(f"\nXPASS -- marked as a known defect but PASSING  {len(xpassed)}")
        for label, (fid, reason) in xpassed:
            print(f"  !  {label}  ({fid})")
        print("     The defect appears to be fixed. Remove the marker and the")
        print("     corresponding finding, or the suite is lying about it.")

    if failed:
        print(f"\nFAILED  {len(failed)}")
        for label, tb in failed:
            print(f"  F  {label}")
            for line in tb.strip().splitlines():
                print(f"       {line}")

    print(f"\n{len(passed)} passed, {len(failed)} failed, "
          f"{len(xfailed)} known defects, {len(xpassed)} xpassed")
    return 1 if (failed or xpassed) else 0
