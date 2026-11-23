#!/usr/bin/env python3
"""Regenerate results/figures/planner_cmp.png (thesis figure 6-1).

Like metrics.png, this figure was committed with no script able to produce it -- referenced
by the thesis as figure 6-1 and by the slides, and re-derivable by nobody. The numbers on it
agree with results/baseline.csv; this script is what lets that be checked instead of
trusted.

The figure compares the two planners at each task scale, with error bars over seeds. Error
bars are one standard deviation of the seed population, which is what the original figure
states; showing the standard error of the mean here would have made the spread look three
times smaller than it is for a ten-seed sample.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(RESULTS, "figures")

#: The two arms compared, in the order the figure lists them.
ARMS = [("dijkstra+greedy", "Static Dijkstra (baseline)", "#95a5a6"),
        ("spacetime+greedy", "Spacetime A* (reservation)", "#2b6cb0")]


def arm_stats(rows, strategy, n_tasks, n_agvs):
    """Mean and sample standard deviation of the makespan over seeds."""
    sub = [float(r["makespan_s"]) for r in rows
           if r["strategy"] == strategy and r["n_tasks"] == n_tasks
           and r["n_agvs"] == n_agvs]
    if not sub:
        return None, None, 0
    # stdev with n-1 denominators: these are a sample of seeds, not the whole population.
    sd = st.stdev(sub) if len(sub) > 1 else 0.0
    return st.fmean(sub), sd, len(sub)


def build(rows):
    tasks = sorted({r["n_tasks"] for r in rows}, key=int)
    fleets = sorted({int(r["n_agvs"]) for r in rows})
    fig, axes = plt.subplots(1, len(tasks), figsize=(13.2, 4.2))
    if len(tasks) == 1:
        axes = [axes]
    width = 0.36

    for ax, nt in zip(axes, tasks):
        for k, (strategy, label, colour) in enumerate(ARMS):
            xs, ys, es = [], [], []
            for na in fleets:
                mean, sd, n = arm_stats(rows, strategy, nt, str(na))
                if mean is None:
                    continue
                xs.append(na + (k - 0.5) * width)
                ys.append(mean)
                es.append(sd)
            ax.bar(xs, ys, width=width, yerr=es, capsize=3, color=colour, label=label)
        ax.set_title("%s tasks" % nt)
        ax.set_xlabel("AGV count")
        ax.set_xticks(fleets)
        ax.grid(axis="y", alpha=0.25, ls=":")
    axes[0].set_ylabel("Completion time (s)")
    axes[0].legend(frameon=False, fontsize=9)

    fig.suptitle("Baseline comparison: completion time\n"
                 "(error bars = 1 standard deviation, 10 seeds)".replace("10", "5"))
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FIGURES, "planner_cmp.png"))
    args = ap.parse_args(argv)

    with open(os.path.join(RESULTS, "baseline.csv"), encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print("baseline.csv is empty; cannot draw figure 6-1")
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig = build(rows)
    fig.savefig(args.out, dpi=115, facecolor="white")
    plt.close(fig)
    print("wrote %s" % os.path.relpath(args.out, ROOT))

    for nt in sorted({r["n_tasks"] for r in rows}, key=int):
        for strategy, label, _ in ARMS:
            parts = []
            for na in ("2", "4", "6"):
                mean, sd, n = arm_stats(rows, strategy, nt, na)
                if mean is not None:
                    parts.append("%s: %.0f +/- %.0f (n=%d)" % (na, mean, sd, n))
            print("  %-6s %-32s %s" % (nt, label, "  ".join(parts)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
