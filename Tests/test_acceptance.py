"""Regression tests for the acceptance clauses of the task specification.

This is the pytest wrapper around the internal acceptance tool
acceptance_check.py (that tool is not released with the public repository).
What it guards is "the clauses required by the task specification are still
there" -- a different concern from regular pytest cases:
regular cases test specific behavior, this one tests whether the end-to-end
capability is complete.

Why wrap it at all
------------------
An earlier round found that `frames_from_result` returned an empty list:
`record_trajectory` defaults to False, so no trajectory is captured unless it
is passed explicitly.
pytest was fully green, yet clause R4.7 of the task specification,
"scheduling process visualization", had become an empty shell.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _load(name: str, rel: str):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_acceptance_clauses(capsys):
    """Every clause of the task specification can be verified in a real run.

    The slimmed-down public repository does not include
    tools/acceptance_check.py (an internal acceptance tool), so skip instead of
    failing here -- otherwise anyone who clones the repository sees a bogus red.
    """
    if not (ROOT / "tools" / "acceptance_check.py").exists():
        pytest.skip("the internal acceptance tool is not in the public repository")

    mod = _load("_acceptance", "tools/acceptance_check.py")

    # The module-level checks have already run; main() only prints and returns a code
    rc = mod.main()
    out = capsys.readouterr().out
    assert rc == 0, "clauses not satisfied:\n" + out
    assert len(mod.ok) >= 5, f"only {len(mod.ok)} clauses were checked; the list looks truncated"
