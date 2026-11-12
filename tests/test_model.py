"""Sanity checks for the kinetic debt model.

Run with `pytest tests/` or simply `python tests/test_model.py`.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from kinetic_debt_model import run_checks  # noqa: E402


def test_all_checks_pass():
    failed = [(name, detail) for name, ok, detail in run_checks(verbose=False) if not ok]
    assert not failed, f"Failed checks: {failed}"


if __name__ == "__main__":
    results = run_checks(verbose=True)
    sys.exit(0 if all(ok for _, ok, _ in results) else 1)
