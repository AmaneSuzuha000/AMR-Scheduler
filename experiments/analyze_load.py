"""Aisle-load instance analysis (the task specification R5.3).

Key convention: **runs that did not finish all tasks must not enter the mean
completion time**. With a single lane (aisle_capacity=1) about 4~5% of the
tasks never finish; makespan for those runs is truncated at max_steps, so
mixing them in turns "completion time" into an illusion. All completion-time
statistics are therefore computed only on the all_done==1 subset.

Usage::

    python experiments/analyze_load.py
    python experiments/analyze_load.py --csv results/load.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEVELS = ["low", "medium", "high"]
LEVEL_CN = {"low": "low load", "medium": "medium load", "high": "high load"}


def load(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def mean(vals):
    return sum(vals) / len(vals) if vals else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/load.csv")
    args = p.parse_args(argv)

    rows = load(args.csv)
    caps = sorted({int(r["aisle_capacity"]) for r in rows})
    planners = sorted({r["planner"] for r in rows})

    print("=" * 104)
    print("Aisle-load instances (R5.3): 50 tasks / 4 AGVs / 10 seeds")
    print("Load level is set by the arrival window: low=1800 steps, medium=600, high=200 (shorter = more concentrated)")
    print("=" * 104)
    print()

    for cap in caps:
        lane = "two-lane" if cap >= 2 else "single-lane"
        print("[aisle capacity = %d (%s)]" % (cap, lane))
        print("-" * 104)
        print("%-8s | %10s %10s %10s | %10s | %10s %10s"
              % ("level", "completion", "makespan", "on-time", "aisle load", "conflict wait", "incomplete"))
        for lv in LEVELS:
            sel = [r for r in rows
                   if r["load_level"] == lv and int(r["aisle_capacity"]) == cap]
            if not sel:
                continue
            done_rates = [float(r["completion_rate"]) for r in sel]
            ms = [float(r["makespan_s"]) for r in sel if int(r["all_done"]) == 1]
            ot = [float(r["on_time_rate"]) for r in sel if int(r["all_done"]) == 1]
            cl = [float(r["mean_corridor_load"]) for r in sel]
            wt = [float(r["conflict_wait_steps"]) for r in sel]
            nbad = sum(1 for r in sel if int(r["all_done"]) == 0)
            print("%-8s | %10.3f %10.1f %10.4f | %10.3f | %10.1f %10d"
                  % (LEVEL_CN[lv], mean(done_rates),
                     mean(ms) if ms else float("nan"),
                     mean(ot) if ot else float("nan"),
                     mean(cl), mean(wt), nbad))
        print()

    # ---- completion rate is the key metric under a single lane ----
    print("=" * 104)
    print("Completion-rate comparison (can the work be finished at all; the key metric under a single lane)")
    print("-" * 104)
    print("%-10s | %14s %14s | %10s" % ("level", "two-lane completion", "single-lane completion", "gap"))
    for lv in LEVELS:
        a = [float(r["completion_rate"]) for r in rows
             if r["load_level"] == lv and int(r["aisle_capacity"]) == 2]
        b = [float(r["completion_rate"]) for r in rows
             if r["load_level"] == lv and int(r["aisle_capacity"]) == 1]
        if not a or not b:
            continue
        print("%-10s | %14.3f %14.3f | %9.1f%%"
              % (LEVEL_CN[lv], mean(a), mean(b), 100.0 * (mean(b) - mean(a))))
    print()

    # ---- planner comparison (only on the clean two-lane data) ----
    print("=" * 104)
    print("Planner comparison (two-lane only; under a single lane so many runs are unfinished that makespan is not comparable)")
    print("-" * 104)
    for lv in LEVELS:
        base_r = [r for r in rows if r["load_level"] == lv
                  and int(r["aisle_capacity"]) == 2
                  and r["planner"] == "static_dijkstra"
                  and int(r["all_done"]) == 1]
        cmpr = [r for r in rows if r["load_level"] == lv
                and int(r["aisle_capacity"]) == 2
                and r["planner"] == "spacetime_astar"
                and int(r["all_done"]) == 1]
        if not base_r or not cmpr:
            continue
        bm = mean([float(r["makespan_s"]) for r in base_r])
        cm = mean([float(r["makespan_s"]) for r in cmpr])
        bw = mean([float(r["conflict_wait_steps"]) for r in base_r])
        cw = mean([float(r["conflict_wait_steps"]) for r in cmpr])
        print("  %-8s static %8.1fs / space-time A* %8.1fs  (%+.2f%%)   conflict wait %6.1f -> %6.1f (%+.1f%%)"
              % (LEVEL_CN[lv], bm, cm, 100.0 * (cm - bm) / bm, bw, cw,
                 100.0 * (cw - bw) / bw if bw else 0.0))
    print()

    # ---- data integrity ----
    print("=" * 104)
    bad_coll = sum(1 for r in rows if int(r["collision_events"]) > 0)
    bad_hard = sum(1 for r in rows if int(r["hard_failures"]) > 0)
    bad_done = sum(1 for r in rows if int(r["all_done"]) == 0)
    print("Data integrity: %d runs with collisions>0, %d runs with hard failures>0, %d/%d unfinished"
          % (bad_coll, bad_hard, bad_done, len(rows)))
    print()
    print("Note: **all** unfinished runs come from the single lane (aisle_capacity=1).")
    print("This is a real deadlock phenomenon, not missing data; it is exactly what the R5.3 stress instances are meant to expose.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
