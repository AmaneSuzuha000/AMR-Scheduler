#!/usr/bin/env python3
"""Regenerate results/figures/metrics.png (thesis figure 5-6).

This figure was committed to the repository with no script able to produce it. It is
referenced by the thesis as figure 5-6 and by the defence slides, so it was evidence that
could not be re-derived from the data -- the exact thing the rest of this project is built
to avoid.

The numbers on it do check out against results/baseline.csv; this script is what makes that
checkable by anyone, and what keeps the figure from drifting away from the data the next time
the simulation changes.

Planner is spacetime+greedy because that is what the original figure was drawn from -- its
title names spacetime_astar. Reproducing a figure means reproducing its stated basis, not
quietly switching to whichever arm looks better.
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

#: The arm this figure reports. Named as a constant so the choice is visible rather than
#: buried in a filter expression.
STRATEGY = "spacetime+greedy"

#: The four acceptance indicators of task specification R5.
PANELS = [
    ("makespan_s", "Completion time (s)", "Makespan"),
    ("on_time_rate", "On-time rate", "On-time rate"),
    ("agv_utilization", "AGV utilisation", "AGV utilisation"),
    ("conflict_wait_steps", "Conflict waiting (steps)", "Conflict waits"),
]


def series(rows, metric):
    """metric value indexed by (n_tasks, n_agvs), averaged across seeds."""
    out = {}
    for nt in sorted({r["n_tasks"] for r in rows}, key=int):
        for na in sorted({r["n_agvs"] for r in rows}, key=int):
            sub = [r for r in rows if r["n_tasks"] == nt and r["n_agvs"] == na]
            if not sub:
                continue
            out[(int(nt), int(na))] = st.fmean(float(r[metric]) for r in sub)
    return out


def build(rows):
    """Four panels, one line per task scale, against fleet size."""
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.2))
    tasks = sorted({r["n_tasks"] for r in rows}, key=int)
    fleets = sorted({int(r["n_agvs"]) for r in rows})
    styles = [("o", "#2b6cb0"), ("s", "#c0392b"), ("^", "#1e8449")]

    for ax, (metric, ylabel, title) in zip(axes.ravel(), PANELS):
        data = series(rows, metric)
        for i, nt in enumerate(tasks):
            ys = [data.get((int(nt), na)) for na in fleets]
            if any(y is None for y in ys):
                continue
            marker, colour = styles[i % len(styles)]
            ax.plot(fleets, ys, marker=marker, color=colour, lw=1.8,
                    label="%s tasks" % nt)
        ax.set_xlabel("AGV count")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(fleets)
        ax.grid(alpha=0.25, ls=":")
    axes[0][0].legend(frameon=False)

    fig.suptitle("Four acceptance indicators against fleet size\n"
                 "(planner: %s, mean over seeds)" % STRATEGY)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    return fig


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(FIGURES, "metrics.png"))
    args = ap.parse_args(argv)

    with open(os.path.join(RESULTS, "baseline.csv"), encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["strategy"] == STRATEGY]
    if not rows:
        print("baseline.csv has no %s rows; cannot draw figure 5-6" % STRATEGY)
        return 1

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig = build(rows)
    fig.savefig(args.out, dpi=130, facecolor="white")
    plt.close(fig)
    print("wrote %s" % os.path.relpath(args.out, ROOT))

    # The figure is only worth anything if the numbers behind it are stated too, so print
    # the series it drew rather than leaving them inside the PNG.
    for metric, _, title in PANELS:
        data = series(rows, metric)
        print("  %-18s %s" % (title, "  ".join(
            "%d/%d=%.3f" % (nt, na, v) for (nt, na), v in sorted(data.items()))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
