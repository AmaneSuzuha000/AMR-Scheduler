"""The data manifest must not let one unmeasured cell erase a whole column.

A route that did not complete has no accuracy, and records nan to say so. That is honest
per row, but statistics.fmean([0.15, nan]) is nan, so a single failed route turned the
column mean into nan and any later drift in the rows that *did* complete became invisible
-- exactly the kind of quiet loss of checking the manifest exists to prevent.
"""
from __future__ import annotations

import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import data_manifest                                       # noqa: E402


def _write(tmp_path, text):
    path = tmp_path / "sample.csv"
    path.write_text(text, encoding="utf-8")
    return str(path)


def test_non_finite_values_are_skipped_not_propagated(tmp_path):
    path = _write(tmp_path, "a,b\n1.0,0.2\n2.0,nan\n3.0,0.4\n")
    d = data_manifest.describe(path)
    assert d["means"]["a"] == pytest.approx(2.0)
    assert d["means"]["b"] == pytest.approx(0.3)
    assert math.isfinite(d["means"]["b"]), (
        "one nan must not turn the whole column mean into nan")


def test_infinity_is_skipped_too(tmp_path):
    path = _write(tmp_path, "a\n1.0\ninf\n3.0\n")
    assert data_manifest.describe(path)["means"]["a"] == pytest.approx(2.0)


def test_a_column_that_is_entirely_non_finite_is_omitted(tmp_path):
    """No finite value means no mean; reporting nan as a statistic would be a lie."""
    path = _write(tmp_path, "a,b\n1.0,nan\n2.0,nan\n")
    d = data_manifest.describe(path)
    assert "b" not in d["means"]
    assert d["means"]["a"] == pytest.approx(1.5)


def test_a_non_numeric_column_still_drops_out_entirely(tmp_path):
    """Text columns are not statistics; that behaviour must be unchanged."""
    path = _write(tmp_path, "name,value\nalpha,1.0\nbeta,3.0\n")
    d = data_manifest.describe(path)
    assert "name" not in d["means"]
    assert d["means"]["value"] == pytest.approx(2.0)


def test_the_committed_gazebo_results_are_numeric():
    """The recorded run must parse cleanly, so the manifest really is checking it."""
    path = os.path.join(ROOT, "results", "gazebo_routes.csv")
    if not os.path.exists(path):
        pytest.skip("no recorded Gazebo run in this checkout")
    d = data_manifest.describe(path)
    for col in ("mean_err_m", "max_err_m", "reached", "waypoints", "completed"):
        assert col in d["means"], (
            "%s produced no mean, so a drift in it would go unnoticed" % col)
