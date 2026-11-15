"""Analyze the multi-layout results: is the strategy ranking stable across layouts."""

from __future__ import annotations

import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_reschedule import exact_wilcoxon_p

from collections import defaultdict


def load():
    rows = list(csv.DictReader(
        open("results/layout_portability.csv", encoding="utf-8")))
    d = defaultdict(dict)
    for r in rows:
        d[(r["layout"], r["n_tasks"], r["n_agvs"], r["seed"])][r["strategy"]] = r
    return rows, d


def main() -> int:
    rows, d = load()
    layouts = ["ring", "grid", "hub"]
    strategies = ["baseline_nearest", "baseline_fifo",
                  "dijkstra+greedy", "spacetime+greedy"]

    print("Multi-layout validation: median completion time (s)")
    print("=" * 88)
    print("{:<20}".format("strategy"), end="")
    for lay in layouts:
        print("{:>18}".format(lay), end="")
    print("{:>16}".format("consistency"))
    print("-" * 88)

    rankings = {}
    for lay in layouts:
        med = {}
        for s in strategies:
            vals = [float(r["makespan_s"]) for r in rows
                    if r["layout"] == lay and r["strategy"] == s
                    and int(r["tasks_completed"]) > 0]
            med[s] = statistics.median(vals) if vals else float("inf")
        rankings[lay] = sorted(med, key=lambda k: med[k])

    for s in strategies:
        print("{:<20}".format(s), end="")
        for lay in layouts:
            vals = [float(r["makespan_s"]) for r in rows
                    if r["layout"] == lay and r["strategy"] == s
                    and int(r["tasks_completed"]) > 0]
            m = statistics.median(vals) if vals else float("nan")
            print("{:>18.0f}".format(m), end="")
        print("{:>16}".format(""))

    print()
    print("Ranking per layout (completion time, best to worst):")
    for lay in layouts:
        print("  %-5s: %s" % (lay, " < ".join(rankings[lay])))

    identical = len({tuple(rankings[l]) for l in layouts}) == 1
    print()
    if identical:
        print("=> the ranking is **identical** across all three layouts; the conclusion does not hinge on one layout")
    else:
        print("=> the rankings differ; state which conclusions are layout-dependent")

    # main method vs space-time A*, paired test per layout
    print()
    print("Paired test: dijkstra+greedy vs spacetime+greedy (completion time)")
    print("-" * 88)
    for lay in layouts:
        xs, ys = [], []
        for k, v in d.items():
            if k[0] != lay:
                continue
            if "dijkstra+greedy" in v and "spacetime+greedy" in v:
                a = float(v["dijkstra+greedy"]["makespan_s"])
                b = float(v["spacetime+greedy"]["makespan_s"])
                if int(v["dijkstra+greedy"]["tasks_completed"]) > 0 and \
                   int(v["spacetime+greedy"]["tasks_completed"]) > 0:
                    xs.append(a)
                    ys.append(b)
        if not xs:
            continue
        better = sum(1 for x, y in zip(xs, ys) if x < y)
        worse = sum(1 for x, y in zip(xs, ys) if x > y)
        p = exact_wilcoxon_p([x - y for x, y in zip(xs, ys)])
        delta = (statistics.mean(xs) - statistics.mean(ys)) / statistics.mean(ys) * 100
        print("  %-5s n=%2d  %2dbetter/%2dworse  p=%.4f  mean diff=%+.1f%%"
              % (lay, len(xs), better, worse, p, delta))

    # completion rate (are there runs that never finish)
    print()
    print("Completion-rate check (are any tasks lost)")
    print("-" * 88)
    for lay in layouts:
        sub = [r for r in rows if r["layout"] == lay]
        incomplete = [r for r in sub
                      if int(r["tasks_completed"]) != int(r["n_tasks"])]
        print("  %-5s %3d runs in total, %d incomplete" % (lay, len(sub), len(incomplete)))
        if incomplete:
            for r in incomplete[:3]:
                print("      %s tasks/%s AGVs/seed%s: %s/%s"
                      % (r["n_tasks"], r["n_agvs"], r["seed"],
                         r["tasks_completed"], r["n_tasks"]))

    # conflict waiting: event counts across layouts
    print()
    print("Conflict-wait events (mean per strategy)")
    print("-" * 88)
    for lay in layouts:
        print("  %-5s" % lay, end="")
        for s in strategies:
            vals = [int(r["conflict_wait_events"]) for r in rows
                    if r["layout"] == lay and r["strategy"] == s]
            print("  %s=%.1f" % (s.split("+")[0][:9], statistics.mean(vals)),
                  end="")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())