"""Single-instance scheduling report: all figures plus a Markdown summary in one call.

Usage:
    python -m amr_sched.viz.report --tasks 20 --agvs 2 --seed 0
"""

from __future__ import annotations

import argparse
import os

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.viz.animation import (agv_state_timeline, render_animation,
                                     task_progress_curve)
from amr_sched.viz.corridor import corridor_load, plot_corridor_heatmap
from amr_sched.viz.gantt import plot_gantt
from amr_sched.viz.layout import plot_layout
from amr_sched.viz.trajectory import plot_trajectories


def build_report(tasks, agvs, seed, planner_name, out_dir, policy):
    sc = ScenarioConfig(n_tasks=tasks, n_agvs=agvs, seed=seed)
    net, task_list, agv_list = generate_scenario(sc)
    planner = (SpaceTimeAStarPlanner if planner_name == "spacetime_astar"
               else StaticDijkstraPlanner)(max_speed=1.0, dt=1.0)
    res = Simulator(net, task_list, agv_list, planner, GreedyAllocator(),
                    SimulationConfig(max_steps=500_000,
                                     record_trajectory=True,
                                     reschedule_policy=policy)).run()

    os.makedirs(out_dir, exist_ok=True)
    plot_layout(net, agv_list, out_path=os.path.join(out_dir, "layout.png"))
    plot_gantt(res, out_path=os.path.join(out_dir, "gantt.png"))
    plot_trajectories(net, res, out_path=os.path.join(out_dir, "trajectory.png"))
    plot_corridor_heatmap(net, res, out_path=os.path.join(out_dir, "corridor.png"))
    # State monitoring and process visualization: static plots cannot show the
    # process, so these two cover it
    agv_state_timeline(res, out_path=os.path.join(out_dir, "agv_states.png"))
    task_progress_curve(res, out_path=os.path.join(out_dir, "task_progress.png"))
    render_animation(net, res, os.path.join(out_dir, "animation.gif"),
                     frame_every=20)

    s = res.summary()
    load = corridor_load(net, res)
    top = sorted(load.items(), key=lambda kv: -kv[1]["load"])[:5]

    lines = [
        "# Scheduling process report",
        "",
        "Instance: %d tasks / %d AGVs / seed=%d / planner=%s / reschedule=%s"
        % (tasks, agvs, seed, planner_name, policy.value),
        "",
        "## The four acceptance metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        "| Completion time | %.1f s |" % s["makespan_seconds"],
        "| On-time rate | %.3f |" % s["on_time_rate"],
        "| AGV utilization | %.3f |" % s["agv_utilization"],
        "| Conflict wait | %d steps |" % s["conflict_wait_steps"],
        "",
        "## Urgent tasks and execution quality",
        "",
        "| Item | Value |",
        "|---|---|",
        "| Tasks completed | %d / %d |" % (s["tasks_completed"], s["tasks_total"]),
        "| Urgent on-time rate | %.3f |" % s["urgent_on_time_rate"],
        "| Mean urgent response | %.1f steps |" % s["mean_urgent_response_steps"],
        "| Preemptions | %d |" % s["preemption_count"],
        "| Planning retries | %d |" % s["planning_retries"],
        "| Hard failures | %d |" % s["hard_failures"],
        "| Collision events | %d |" % s["collision_events"],
        "",
        "## Busiest aisles (top 5)",
        "",
        "| Aisle | Load | Capacity | Occupied steps |",
        "|---|---|---|---|",
    ]
    for (u, v), info in top:
        lines.append("| %s - %s | %.1f%% | %d | %d |"
                     % (u, v, info["load"] * 100, info["capacity"], info["occupied"]))
    lines += [
        "",
        "## Figures",
        "",
        "![Layout](layout.png)",
        "![Gantt](gantt.png)",
        "![Trajectory](trajectory.png)",
        "![Aisle load](corridor.png)",
        "",
        "## Process monitoring",
        "",
        "Static figures only show the outcome. The next two show the **process**: what",
        "state each vehicle is in at every moment, and how tasks migrate between states.",
        "",
        "![AGV state timeline](agv_states.png)",
        "![Task progress](task_progress.png)",
        "",
        "Animation: [animation.gif](animation.gif)",
    ]
    with open(os.path.join(out_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return res, os.path.join(out_dir, "report.md")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=20)
    p.add_argument("--agvs", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--planner", default="spacetime_astar",
                   choices=["spacetime_astar", "static_dijkstra"])
    p.add_argument("--reschedule", default="none",
                   choices=[x.value for x in ReschedulePolicy])
    p.add_argument("--out", default="results/figures")
    args = p.parse_args(argv)

    res, path = build_report(args.tasks, args.agvs, args.seed, args.planner,
                             args.out, ReschedulePolicy(args.reschedule))
    print("Completed %d tasks in %.0f s" % (res.summary()["tasks_completed"],
                                           res.makespan_seconds()))
    print("Report: %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
