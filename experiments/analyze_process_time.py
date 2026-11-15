"""Analyze the sensitivity to workstation process time."""

from __future__ import annotations

import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from collections import defaultdict


def main() -> int:
    rows = list(csv.DictReader(
        open("results/process_time.csv", encoding="utf-8")))
    pts = sorted({float(r["process_time"]) for r in rows})

    print("Workstation process-time sensitivity (90 runs per level = 2 planners x 9 cells x 5 seeds)")
    print("=" * 84)
    print("{:>8} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "proc(s)", "makespan", "on-time", "util", "conflicts", "completion"))
    print("-" * 84)
    base_ms = None
    for pt in pts:
        sub = [r for r in rows if float(r["process_time"]) == pt]
        ms = statistics.mean([float(r["makespan_s"]) for r in sub])
        ot = statistics.mean([float(r["on_time_rate"]) for r in sub])
        ut = statistics.mean([float(r["agv_utilization"]) for r in sub])
        ce = statistics.mean([int(r["conflict_wait_events"]) for r in sub])
        done = sum(int(r["tasks_completed"]) for r in sub)
        need = sum(int(r["n_tasks"]) for r in sub)
        if base_ms is None:
            base_ms = ms
        print("{:>8.0f} {:>10.0f} {:>10.3f} {:>10.3f} {:>10.1f} {:>9.1f}%".format(
            pt, ms, ot, ut, ce, 100.0 * done / need))

    print()
    print("Change relative to process time = 0")
    print("-" * 84)
    z = [r for r in rows if float(r["process_time"]) == 0.0]
    ms0 = statistics.mean([float(r["makespan_s"]) for r in z])
    ot0 = statistics.mean([float(r["on_time_rate"]) for r in z])
    ut0 = statistics.mean([float(r["agv_utilization"]) for r in z])
    ce0 = statistics.mean([int(r["conflict_wait_events"]) for r in z])
    for pt in pts[1:]:
        sub = [r for r in rows if float(r["process_time"]) == pt]
        ms = statistics.mean([float(r["makespan_s"]) for r in sub])
        ot = statistics.mean([float(r["on_time_rate"]) for r in sub])
        ut = statistics.mean([float(r["agv_utilization"]) for r in sub])
        ce = statistics.mean([int(r["conflict_wait_events"]) for r in sub])
        print("  process={:3.0f}s  makespan {:+.1f}%  on-time {:.3f}->{:.3f}  "
              "util {:.3f}->{:.3f}  conflicts {:+.1f}%"
              .format(pt, (ms - ms0) / ms0 * 100, ot0, ot, ut0, ut,
                      (ce - ce0) / ce0 * 100 if ce0 else 0))

    print()
    print("Broken down by task scale (completion-time change vs process time = 0, %)")
    print("-" * 84)
    print("{:>8}".format("proc(s)"), end="")
    for n in (20, 50, 100):
        print("{:>14}".format("%d tasks" % n), end="")
    print()
    for pt in pts:
        print("{:>8.0f}".format(pt), end="")
        for n in (20, 50, 100):
            sub = [r for r in rows
                   if float(r["process_time"]) == pt and int(r["n_tasks"]) == n]
            sub0 = [r for r in rows
                    if float(r["process_time"]) == 0.0 and int(r["n_tasks"]) == n]
            if not sub or not sub0:
                print("{:>14}".format("-"), end="")
                continue
            m = statistics.mean([float(r["makespan_s"]) for r in sub])
            m0 = statistics.mean([float(r["makespan_s"]) for r in sub0])
            print("{:>13.1f}%".format((m - m0) / m0 * 100), end="")
        print()

    print()
    print("Completion-rate check")
    print("-" * 84)
    for pt in pts:
        sub = [r for r in rows if float(r["process_time"]) == pt]
        bad = [r for r in sub if int(r["tasks_completed"]) != int(r["n_tasks"])]
        print("  process={:3.0f}s  {}/{} runs unfinished".format(pt, len(bad), len(sub)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())