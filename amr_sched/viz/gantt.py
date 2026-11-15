"""Gantt chart: per-AGV timeline, task occupancy, and conflict waiting.

This is the most direct evidence figure for path and timing output: one glance
shows whether a vehicle was idle, where it waited, and for how long.
"""

from __future__ import annotations

import os
from collections import defaultdict

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from amr_sched.entities import AgvState
from amr_sched.simulation import SimulationResult
from amr_sched.viz import style


_STATE_COLORS = {
    AgvState.IDLE: "#d9dde1",
    AgvState.TO_PICKUP: "#4c9be8",
    AgvState.LOADING: "#f0c419",
    AgvState.TO_DROPOFF: "#2e8b57",
    AgvState.UNLOADING: "#e08b2e",
    AgvState.BLOCKED: "#d64545",
}
_STATE_LABELS = {
    AgvState.IDLE: "Idle",
    AgvState.TO_PICKUP: "Empty to pickup",
    AgvState.LOADING: "Loading",
    AgvState.TO_DROPOFF: "Loaded delivery",
    AgvState.UNLOADING: "Unloading",
    AgvState.BLOCKED: "Conflict wait",
}


def plot_gantt(result: SimulationResult,
               out_path: str | None = None,
               title: str = "AGV task Gantt chart"):
    """Gantt chart by task: one row per vehicle, one bar per task execution.

    Data comes from task timestamps (assigned / pickup / complete), not from the
    trajectory, so it works with ``record_trajectory=False``.
    """
    style.setup()
    agvs = result.agvs
    if not agvs:
        raise ValueError("no AGVs")

    tasks_by_agv: dict[str, list] = defaultdict(list)
    for t in result.tasks:
        if t.assigned_agv:
            tasks_by_agv[t.assigned_agv].append(t)

    fig, ax = plt.subplots(figsize=(10.5, 0.55 * len(agvs) + 2.0))
    rows = {a.agv_id: i for i, a in enumerate(agvs)}

    for agv in agvs:
        y = rows[agv.agv_id]
        for t in tasks_by_agv.get(agv.agv_id, []):
            start = t.assigned_step if t.assigned_step is not None else t.release_step
            end = t.complete_step if t.complete_step is not None else result.finished_step
            if end <= start:
                continue
            color = "#d64545" if t.is_urgent else "#4c9be8"
            if t.is_urgent:
                color = "#c0392b"
            ax.barh(y, end - start, left=start, height=0.62,
                    color=color, alpha=0.85, edgecolor="white", linewidth=0.5)
            ax.text((start + end) / 2.0, y, t.task_id,
                    ha="center", va="center", fontsize=6.5,
                    color="white", clip_on=True)
            # Due-time marker
            if t.due_step is not None:
                late = (t.complete_step or result.finished_step) > t.due_step
                ax.plot([t.due_step], [y], marker="|", markersize=9,
                        color=("#8b0000" if late else "#2e8b57"),
                        markeredgewidth=1.6, zorder=5)

    ax.set_yticks(range(len(agvs)))
    ax.set_yticklabels([a.agv_id for a in agvs])
    ax.set_xlabel("Simulation time (steps)")
    ax.set_xlim(0, result.finished_step * 1.02)
    ax.invert_yaxis()
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    handles = [
        Patch(facecolor="#4c9be8", label="Normal task"),
        Patch(facecolor="#c0392b", label="Urgent task"),
    ]
    ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=8)

    ax.set_title("%s\n%d steps total | on-time %.3f | utilization %.3f | conflict wait %d steps"
                 % (title,
                    result.finished_step,
                    result.on_time_rate(),
                    result.agv_utilization(),
                    result.conflict_wait_steps()), fontsize=10)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)
    return fig
