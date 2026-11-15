"""加急任务动态重调度对比实验（任务书 R2.5）。

比较两种策略：
  none             —— 加急任务只在分配排序上占优，不打断任何执行中任务
  preempt_unpicked —— 允许抢占"空驶取货、尚未装货"的普通任务

输出：results/reschedule.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

FIELDS = [
    "n_tasks", "n_agvs", "planner", "policy", "seed",
    "tasks_completed", "makespan_s", "on_time_rate", "urgent_on_time_rate",
    "mean_urgent_response_steps", "agv_utilization", "conflict_wait_steps",
    "preemption_count", "planning_retries", "hard_failures", "collision_events",
]


def run_one(nt, na, planner_name, policy, seed):
    sc = ScenarioConfig(n_tasks=nt, n_agvs=na, seed=seed)
    net, tasks, agvs = generate_scenario(sc)
    planner = (SpaceTimeAStarPlanner if planner_name == "spacetime_astar"
               else StaticDijkstraPlanner)(max_speed=1.0, dt=1.0)
    r = Simulator(net, tasks, agvs, planner, GreedyAllocator(),
                  SimulationConfig(max_steps=500_000, seed=seed,
                                   reschedule_policy=policy)).run()
    s = r.summary()
    return {
        "n_tasks": nt, "n_agvs": na, "planner": planner_name,
        "policy": policy.value, "seed": seed,
        "tasks_completed": int(s["tasks_completed"]),
        "makespan_s": round(float(s["makespan_seconds"]), 2),
        "on_time_rate": round(float(s["on_time_rate"]), 4),
        "urgent_on_time_rate": round(float(s["urgent_on_time_rate"]), 4),
        "mean_urgent_response_steps": round(float(s["mean_urgent_response_steps"]), 1),
        "agv_utilization": round(float(s["agv_utilization"]), 4),
        "conflict_wait_steps": int(s["conflict_wait_steps"]),
        "preemption_count": int(s["preemption_count"]),
        "planning_retries": int(s["planning_retries"]),
        "hard_failures": int(s["hard_failures"]),
        "collision_events": int(s["collision_events"]),
    }


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--agvs", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--planner", default="spacetime_astar")
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", default="results/reschedule.csv")
    args = p.parse_args(argv)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = []
    total = len(args.tasks) * len(args.agvs) * 2 * len(args.seeds)
    i = 0
    for nt in args.tasks:
        for na in args.agvs:
            for pol in (ReschedulePolicy.NONE, ReschedulePolicy.PREEMPT_UNPICKED):
                for seed in args.seeds:
                    i += 1
                    row = run_one(nt, na, args.planner, pol, seed)
                    rows.append(row)
                    print("[%d/%d] %3d/%d %-16s seed=%d -> makespan=%8.1f "
                          "u_ontime=%.3f u_resp=%6.1f pmp=%d"
                          % (i, total, nt, na, pol.value, seed, row["makespan_s"],
                             row["urgent_on_time_rate"],
                             row["mean_urgent_response_steps"],
                             row["preemption_count"]), flush=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("")
    print("写入 %s（%d 行）" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
