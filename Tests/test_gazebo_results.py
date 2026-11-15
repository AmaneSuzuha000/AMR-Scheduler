"""Gazebo route results must be recorded from the run, not transcribed from it.

The Gazebo accuracy numbers used to live only in GAZEBO.md, typed by hand. That is the
same failure mode as the handwritten routes.json: correct on the day, and unchecked
afterwards. The verification script now writes results/gazebo_routes.csv, which
tools/data_manifest.py records alongside the other 22 CSVs, so a drift shows up in
--check like any other.

These tests cover the part that can be checked without a simulator: that the parser reads
the runner output faithfully, and that the CSV shape is the one the manifest expects.
"""
from __future__ import annotations

import csv
import math
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools", "live_bridge"))

import verify_scheduler_to_gazebo as vg                  # noqa: E402


SAMPLE = """\
route D->W3  nodes=D -> R0 -> R1 -> R2 -> W3
    wp0 (-1.00,0.80) reached dist=0.124
    wp1 (0.00,0.00) reached dist=0.128
  PASS 5/5  mean_err=0.172 max_err=0.435
route R0->B1  nodes=R0 -> R1 -> R4 -> B1
    wp0 (0.00,0.00) reached dist=0.000
  FAIL 2/4  stalled at wp2 (1.60,1.60) pose=(2.533,0.803) dist=1.227
routes failed: 1/2
"""


def test_parser_reads_pass_and_fail_rows():
    rows = vg.parse_route_lines(SAMPLE)
    assert [r["route"] for r in rows] == ["D->W3", "R0->B1"]
    ok, bad = rows
    assert (ok["reached"], ok["waypoints"], ok["completed"]) == (5, 5, 1)
    assert ok["mean_err_m"] == pytest.approx(0.172)
    assert ok["max_err_m"] == pytest.approx(0.435)
    assert (bad["reached"], bad["waypoints"], bad["completed"]) == (2, 4, 0)
    # A failed route reports no accuracy, and must say so numerically rather than with an
    # empty cell that would make the whole column unparseable for the manifest.
    assert math.isnan(bad["mean_err_m"])


def test_parser_keeps_route_order():
    """The CSV must read in the order the routes were driven."""
    rows = vg.parse_route_lines(SAMPLE)
    assert rows[0]["route"] == "D->W3"


def test_a_run_with_no_printed_results_yields_no_rows():
    """Silence must not be recorded as success.

    If the runner stops printing per-route lines the results file would otherwise be empty
    and look like a clean run with nothing to report.
    """
    assert vg.parse_route_lines("") == []
    assert vg.parse_route_lines("some unrelated output\n") == []


def test_written_csv_has_the_columns_the_manifest_expects(tmp_path):
    rows = vg.parse_route_lines(SAMPLE)
    out = tmp_path / "gazebo_routes.csv"
    vg.write_results(rows, 12.5, str(out))
    with open(out, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == vg.RESULT_COLUMNS
        written = list(reader)
    assert len(written) == 2
    # Numeric columns must be numeric, or data_manifest silently skips them when it
    # computes means and the manifest would not notice a drift in them.
    for col in ("waypoints", "reached", "mean_err_m", "max_err_m", "completed",
                "wall_clock_s"):
        for r in written:
            float(r[col])
    # The failed route must still carry a numeric "not measured" rather than a blank.
    assert written[1]["completed"] == "0"
    assert math.isnan(float(written[1]["mean_err_m"]))


def test_wall_clock_is_not_part_of_the_fingerprint():
    """Wall clock varies by machine load, so the manifest must ignore it."""
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    import data_manifest
    assert "wall_clock_s" in data_manifest.NONDET


def test_the_recorded_results_match_the_committed_csv():
    """The CSV in results/ must describe complete routes, whatever else it says."""
    path = os.path.join(ROOT, "results", "gazebo_routes.csv")
    if not os.path.exists(path):
        pytest.skip("no recorded Gazebo run in this checkout")
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "an empty results file is not evidence of anything"
    for r in rows:
        assert int(r["completed"]) == 1, (
            "%s did not complete but is recorded as a result" % r["route"])
        assert int(r["reached"]) == int(r["waypoints"])
        assert float(r["mean_err_m"]) > 0.0, (
            "a zero mean error would mean the vehicle did not move")
