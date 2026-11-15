"""Metric plots built from the experiment CSVs."""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt

from amr_sched.viz import style

METRIC_LABELS = {
    "makespan_s": "Completion time (s)",
    "on_time_rate": "On-time rate",
    "agv_utilization": "AGV utilization",
    "conflict_wait_steps": "Conflict wait (steps)",
}


def _load(csv_path: str) -> tuple[list[dict], list[str], list[str]]:
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    for r in rows:
        for k in ("n_tasks", "n_agvs", "seed"):
            r[k] = int(r[k])
        for k in METRIC_LABELS:
            if k in r:
                r[k] = float(r[k])
    tasks = sorted({r["n_tasks"] for r in rows})
    planners = sorted({r["planner"] for r in rows})
    return rows, tasks, planners


def plot_metric_grid(csv_path: str,
                     out_path: str | None = None,
                     title: str = "Acceptance metrics vs fleet size"):
    """2x2 panels: each metric against fleet size, one line per task scale."""
    style.setup(8.5)
    rows, tasks, planners = _load(csv_path)
    planner = "spacetime_astar" if "spacetime_astar" in planners else planners[0]
    rows = [r for r in rows if r["planner"] == planner]

    agvs = sorted({r["n_agvs"] for r in rows})
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.2))
    markers = ["o", "s", "^", "D"]
    colors = ["#2f6fb3", "#d64545", "#2e8b57", "#d99b1c"]

    for ax, (metric, label) in zip(axes.ravel(), METRIC_LABELS.items()):
        for i, nt in enumerate(tasks):
            by_agv = defaultdict(list)
            for r in rows:
                if r["n_tasks"] == nt:
                    by_agv[r["n_agvs"]].append(r[metric])
            xs = [a for a in agvs if a in by_agv]
            ys = [sum(by_agv[a]) / len(by_agv[a]) for a in xs]
            ax.plot(xs, ys, marker=markers[i % len(markers)],
                    color=colors[i % len(colors)], label="%d tasks" % nt,
                    linewidth=1.4, markersize=5)
        ax.set_xlabel("Number of AGVs")
        ax.set_ylabel(label)
        ax.set_xticks(agvs)
        ax.grid(linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)

    axes[0][0].legend(frameon=False, fontsize=8)
    fig.suptitle("%s\n(planner: %s, mean across seeds)" % (title, planner), fontsize=10)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig


def plot_planner_comparison(csv_path: str,
                            out_path: str | None = None,
                            baseline: str = "static_dijkstra",
                            compare: str = "spacetime_astar"):
    """Compare two planners on completion time by fleet size."""
    style.setup(8.5)
    rows, tasks, _ = _load(csv_path)
    agvs = sorted({r["n_agvs"] for r in rows})

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.4 * len(tasks), 3.2),
                             squeeze=False)
    for ax, nt in zip(axes[0], tasks):
        width = 0.36
        for k, (planner, color, label) in enumerate((
                (baseline, "#9aa5b1", "Static Dijkstra (baseline)"),
                (compare, "#2f6fb3", "Space-time A* (reservation)"))):
            xs, ys, errs = [], [], []
            for a in agvs:
                vals = [r["makespan_s"] for r in rows
                        if r["n_tasks"] == nt and r["n_agvs"] == a
                        and r["planner"] == planner]
                if not vals:
                    continue
                m = sum(vals) / len(vals)
                var = sum((v - m) ** 2 for v in vals) / max(1, len(vals) - 1)
                xs.append(a + (k - 0.5) * width)
                ys.append(m)
                errs.append(var ** 0.5)
            ax.bar(xs, ys, width=width, color=color, label=label,
                   yerr=errs, capsize=2.5, error_kw={"lw": 0.8})
        ax.set_xticks(agvs)
        ax.set_xlabel("Number of AGVs")
        ax.set_ylabel("Completion time (s)")
        ax.set_title("%d tasks" % nt, fontsize=9)
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
    axes[0][0].legend(frameon=False, fontsize=7.5)
    fig.suptitle("Baseline comparison: completion time (error bars = 1 SD, 10 seeds)", fontsize=10)
    fig.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig
