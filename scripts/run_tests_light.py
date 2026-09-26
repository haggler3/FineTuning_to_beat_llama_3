"""Run the test suite with the heavy ML libraries stubbed out.

Keeps the suite runnable on a laptop with no GPU and no multi-gigabyte installs.
`pytest` works too once the package is installed; this script is the zero-install
path and is what CI runs.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "src")
TESTS_DIR = os.path.join(REPO_ROOT, "tests")

# Mock out heavy libraries before importing test files. The modules under test
# import these at module scope, but the logic the tests exercise does not touch
# them, so stubbing keeps the suite fast without weakening what is asserted.
HEAVY_MODULES = [
    "torch",
    "transformers",
    "datasets",
    "peft",
    "huggingface_hub",
    "evaluate",
    "sentence_transformers",
    "wandb",
    "trl",
]


def main():
    # The package uses 3.10+ syntax (notably zip(..., strict=True)); on 3.9 the
    # failures surface deep inside the code under test as confusing TypeErrors.
    # noqa UP036: ruff reads target-version and calls this dead, but the whole
    # point is to catch someone whose default `python` is older than the target.
    if sys.version_info < (3, 10):  # noqa: UP036
        print(
            "ERROR: Python 3.10+ is required, found "
            f"{sys.version_info.major}.{sys.version_info.minor}."
        )
        return 1

    for mod in HEAVY_MODULES:
        sys.modules[mod] = MagicMock()

    # src/ for the package under test, tests/ for the shared fakes module.
    sys.path.insert(0, SRC_DIR)
    sys.path.insert(0, TESTS_DIR)

    tests = unittest.defaultTestLoader.discover(TESTS_DIR, pattern="test_*.py")

    # Discovery silently yields an empty suite when the pattern matches nothing,
    # and an empty suite reports success. That once hid the fact that the whole
    # suite collected zero tests on case-sensitive filesystems.
    if tests.countTestCases() == 0:
        print("ERROR: no tests were discovered - refusing to report success.")
        return 1

    result = unittest.TextTestRunner(verbosity=2).run(tests)
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
