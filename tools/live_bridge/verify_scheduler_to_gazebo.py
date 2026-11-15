#!/usr/bin/env python3
"""End-to-end check: scheduler network -> route file -> Gazebo -> measured accuracy.

Each stage already has its own tests. What this adds is the joins between them, which is
where the chain can break while every part still passes: the exporter could emit the right
shape with the wrong scale, the follower could read the file and drive somewhere else, the
pose service could report the vehicle is elsewhere than it is.

Run without a simulator it verifies the file stage and reports that the rest needs Gazebo.
Run with one it also drives every route and checks the accuracy against the same tolerance
the follower uses.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
# tools/live_bridge -> repository root
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

from amr_sched.network import build_demo_network          # noqa: E402
import gz_export_routes as ex                              # noqa: E402


def stage_export(tmpdir):
    """Network -> route document, with the geometry derived from the network."""
    routes = ex.build_routes(build_demo_network())
    path = os.path.join(tmpdir, "routes.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(routes, f)
    print("[1/3] exported %d routes" % len(routes))

    pos = ex.scaled_positions(build_demo_network())
    assert pos[ex.ORIGIN_NODE] == (0.0, 0.0), "origin node is not at the world origin"
    # A wrong scale moves every waypoint while leaving the document shape intact.
    assert abs(pos["R1"][0] - pos["R0"][0]) == 1.6, "aisle pitch does not scale to 1.6 m"
    for name, spec in routes.items():
        assert len(spec["waypoints"]) == len(spec["nodes"]), name
    print("      geometry checks out: origin at (0,0), aisle pitch 1.6 m")
    return path, routes


def stage_plan_vs_file(routes):
    """The file must still describe the network it came from."""
    pos = ex.scaled_positions(build_demo_network())
    for name, spec in routes.items():
        expected = [list(pos[n]) for n in spec["nodes"]]
        for got, want in zip(spec["waypoints"], expected):
            assert math.hypot(got[0] - want[0], got[1] - want[1]) < 1e-9, name
    print("[2/3] every waypoint equals its node position in the network")


#: One row per route, in the same shape as every other results CSV so that
#: tools/data_manifest.py records it without any special case.
RESULT_COLUMNS = ["route", "waypoints", "reached", "mean_err_m", "max_err_m",
                  "completed", "wall_clock_s"]

#: A failed route has no accuracy to report. Leaving the cell empty is honest but makes the
#: column non-numeric, and tools/data_manifest.py skips any column it cannot parse entirely
#: -- so one failed route would silently remove mean_err_m from the manifest means and a
#: later drift in that column would go unnoticed. NaN keeps the column numeric and still
#: says "not measured", which is what it is.
NOT_MEASURED = "nan"


def parse_route_lines(out):
    """Pull per-route results out of the runner's human-readable output.

    The runner prints for humans; this reads it back so the numbers can be recorded. Parsing
    the printed form rather than adding a second machine format keeps one source of truth --
    if the runner stops printing what happened, the results file goes empty and that is a
    visible failure rather than a silently correct-looking zero.
    """
    rows = {}
    order = []
    current = None
    for line in out.splitlines():
        m = re.match(r"route (\S+)\s+nodes=", line)
        if m:
            current = m.group(1)
            rows[current] = {"route": current, "waypoints": 0, "reached": 0,
                             "mean_err_m": "", "max_err_m": "", "completed": 0}
            order.append(current)
            continue
        if current is None:
            continue
        m = re.search(r"PASS (\d+)/(\d+)\s+mean_err=([0-9.]+) max_err=([0-9.]+)", line)
        if m:
            rows[current].update({
                "reached": int(m.group(1)), "waypoints": int(m.group(2)),
                "mean_err_m": float(m.group(3)), "max_err_m": float(m.group(4)),
                "completed": 1 if int(m.group(1)) == int(m.group(2)) else 0,
            })
            continue
        m = re.search(r"FAIL (\d+)/(\d+)", line)
        if m:
            rows[current].update({"reached": int(m.group(1)),
                                  "waypoints": int(m.group(2)), "completed": 0,
                                  "mean_err_m": float("nan"),
                                  "max_err_m": float("nan")})
    return [rows[k] for k in order]


def write_results(rows, elapsed, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_COLUMNS)
        w.writeheader()
        for r in rows:
            r = dict(r)
            r["wall_clock_s"] = round(elapsed, 3)
            w.writerow(r)
    # relpath raises on Windows when the path is on another drive than the cwd, which is
    # what happens when a caller passes a temp path. Report the absolute path instead of
    # letting a logging line fail the run.
    try:
        shown = os.path.relpath(out_path)
    except ValueError:
        shown = out_path
    print("      wrote %s (%d routes)" % (shown, len(rows)))


def stage_drive(path, timeout, results_csv=None):
    """Drive the file in Gazebo, check the accuracy, and record what happened."""
    import time
    runner = os.path.join(HERE, "gz_follow_routes.py")
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-u", runner, "--routes", path],
        capture_output=True, text=True, timeout=timeout,
    )
    elapsed = time.monotonic() - t0
    out = proc.stdout + proc.stderr
    print("[3/3] driving routes in Gazebo")
    for line in out.splitlines():
        if line.startswith("route ") or "PASS" in line or "FAIL" in line:
            print("      " + line.strip())

    rows = parse_route_lines(out)
    if results_csv:
        write_results(rows, elapsed, results_csv)

    # Requiring the parsed rows to agree with the summary line means a change in either one
    # surfaces here instead of leaving a results file that quietly disagrees with the run.
    if "routes failed: 0/" not in out:
        print("FAIL: at least one route did not complete")
        return False
    if not rows or not all(r["completed"] for r in rows):
        print("FAIL: printed PASS/FAIL lines do not describe a complete run")
        return False
    print("      all routes completed")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--skip-drive", action="store_true")
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "gazebo_routes.csv"),
                    help="where to record per-route results")
    args = ap.parse_args(argv)

    with tempfile.TemporaryDirectory() as tmp:
        path, routes = stage_export(tmp)
        stage_plan_vs_file(routes)
        if args.skip_drive:
            print("[3/3] skipped (--skip-drive); driving needs Gazebo and ROS 2")
            return 0
        ok = stage_drive(path, args.timeout, args.results)

    print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
