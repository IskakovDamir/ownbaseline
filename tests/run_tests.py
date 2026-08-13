#!/usr/bin/env python3
"""
Run the test suite with nothing but the standard library and the repo's own
runtime dependencies.

    python3 tests/run_tests.py

pytest also collects tests/ natively if you have it:

    python3 -m pytest tests/ -q
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import _harness  # noqa: E402
import test_estimator  # noqa: E402

if __name__ == "__main__":
    sys.exit(_harness.run([test_estimator]))
