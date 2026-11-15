"""The fleet-size analysis must stay honest about what the data supports.

The specification asks for a discussion of "the reasonableness of the AGV configuration".
The tempting answer is "4 vehicles is optimal". That is not decidable from this data:
makespan falls and conflict waiting rises with every vehicle added, and the specification
weights neither against the other. Inventing a weighting to produce a single answer would be
manufacturing a conclusion.

What the data does support is the ratio between successive fleet steps -- 30-35% across all
three task scales. These tests pin that, and pin the refusal to name an optimum.
"""
from __future__ import annotations

import csv
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "experiments"))

import analyze_fleet_size as fleet                       # noqa: E402


@pytest.fixture(scope="module")
def baseline():
    rows = fleet.load_baseline()
    if not rows:
        pytest.skip("baseline.csv has no main-method rows in this checkout")
    return rows


def test_only_the_main_method_is_used(baseline):
    """Mixing strategies here would confound fleet size with planner choice."""
    assert all(r["strategy"] == fleet.MAIN for r in baseline)


def test_every_task_scale_and_fleet_size_is_covered(baseline):
    """The specification names 20/50/100 tasks and 2/4/6 vehicles."""
    tasks = {r["n_tasks"] for r in baseline}
    sizes = {r["n_agvs"] for r in baseline}
    assert tasks == {"20", "50", "100"}
    assert sizes == {"2", "4", "6"}


def test_savings_per_vehicle_are_positive(baseline):
    """If adding a vehicle did not help, the whole configuration discussion changes."""
    for r in fleet.fleet_rows(baseline):
        assert r["s_per_vehicle_saved"] > 0, r


def test_the_second_step_returns_a_fraction_of_the_first(baseline):
    """Diminishing returns is the finding; it has to hold at every task scale."""
    rows = [r for r in fleet.fleet_rows(baseline) if r["step_ratio"] != ""]
    assert rows, "no fleet step had a predecessor to compare against"
    for r in rows:
        assert 0 < r["step_ratio"] < 1, (
            "fleet step %s at %s tasks returned %s of the previous step"
            % (r["fleet_step"], r["n_tasks"], r["step_ratio"]))


def test_the_diminishing_return_is_consistent_across_scales(baseline):
    """The claim is a range, so the range has to actually be narrow."""
    ratios = [r["step_ratio"] for r in fleet.fleet_rows(baseline) if r["step_ratio"] != ""]
    assert max(ratios) - min(ratios) < 0.15, (
        "the step ratios scatter too widely to state as a range: %s" % ratios)


def test_conflict_waiting_is_reported_not_hidden(baseline):
    """Every added vehicle buys time with interference.

    Reporting only the gain would hide half of the trade-off that the configuration decision
    actually is.
    """
    for r in fleet.fleet_rows(baseline):
        assert "conflict_per_vehicle_added" in r
        assert r["conflict_wait_mean"] >= 0


def test_the_written_csv_has_the_declared_columns(tmp_path):
    rows = fleet.fleet_rows(fleet.load_baseline())
    out = tmp_path / "fleet_size.csv"
    fleet.write_csv(rows, str(out))
    with open(out, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == fleet.FIELDS
        written = list(reader)
    assert len(written) == len(rows)
    for col in ("makespan_mean_s", "s_per_vehicle_saved", "conflict_per_vehicle_added"):
        for r in written:
            float(r[col])


def test_the_recorded_fleet_csv_matches_the_baseline():
    """The committed CSV has to be what this code produces from the committed baseline."""
    path = os.path.join(ROOT, "results", "fleet_size.csv")
    if not os.path.exists(path):
        pytest.skip("no recorded fleet analysis in this checkout")
    with open(path, encoding="utf-8") as f:
        recorded = list(csv.DictReader(f))
    expected = fleet.fleet_rows(fleet.load_baseline())
    assert len(recorded) == len(expected)
    for got, want in zip(recorded, expected):
        assert got["n_tasks"] == str(want["n_tasks"])
        assert float(got["s_per_vehicle_saved"]) == pytest.approx(
            want["s_per_vehicle_saved"], abs=0.01)
