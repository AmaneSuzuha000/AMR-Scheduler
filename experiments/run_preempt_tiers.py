"""R2.5 加急动态重调度：三档抢占策略的代价-收益对比。

三档策略
--------
none             —— 加急只在分配排序上占优，不打断任何执行中任务
preempt_unpicked —— 允许抢占"空驶取货、尚未装货"的车（零物料代价）
preempt_any      —— 允许抢占"已装货"的车（有物料代价，需送回取货点）

研究问题
--------
抢占粒度放得越宽，加急准时率越高，但物料浪费也越大。
本实验量化这条代价-收益曲线，为"该抢到哪一档"提供依据。

用法::

    python experiments/run_preempt_tiers.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

FIELDS = [
    "policy", "n_tasks", "n_agvs", "seed", "tasks_completed", "makespan_s",
    "on_time_rate", "urgent_on_time_rate", "mean_urgent_response_steps",
    "preemption_count", "material_waste_steps", "distance_m",
    "conflict_wait_steps", "conflict_wait_events", "collision_events",
    "hard_failures", "deadlock_breaks",
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--agvs", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", default="results/preempt_tiers.csv")
    args = p.parse_args(argv)

    rows = []
    pols = list(ReschedulePolicy)
    total = len(pols) * len(args.tasks) * len(args.agvs) * len(args.seeds)
    i = 0
    for pol in pols:
        for n in args.tasks:
            for a in args.agvs:
                for sd in args.seeds:
                    i += 1
                    sc = ScenarioConfig(n_tasks=n, n_agvs=a, seed=sd)
                    net, tasks, agvs = generate_scenario(sc)
                    sim = Simulator(
                        net, tasks, agvs,
                        SpaceTimeAStarPlanner(max_speed=sc.max_speed, dt=sc.dt),
                        GreedyAllocator(),
                        SimulationConfig(dt=sc.dt, max_steps=500_000,
                                         reschedule_policy=pol))
                    r = sim.run()
                    r.total_distance = sim._odometer
                    s = r.summary()
                    rows.append({
                        "policy": pol.value,
                        "n_tasks": n, "n_agvs": a, "seed": sd,
                        "tasks_completed": int(s["tasks_completed"]),
                        "makespan_s": round(float(s["makespan_seconds"]), 2),
                        "on_time_rate": round(float(s["on_time_rate"]), 4),
                        "urgent_on_time_rate": round(
                            float(s["urgent_on_time_rate"]), 4),
                        "mean_urgent_response_steps": round(
                            float(s["mean_urgent_response_steps"]), 2),
                        "preemption_count": int(s["preemption_count"]),
                        "material_waste_steps": round(
                            float(s["material_waste_steps"]), 2),
                        "distance_m": round(float(sim._odometer), 1),
                        "conflict_wait_steps": int(s["conflict_wait_steps"]),
                        "collision_events": int(s["collision_events"]),
                        "hard_failures": int(s["hard_failures"]),
                    })
                    if i % 30 == 0:
                        print("[%3d/%d] %s %d任务%d车 -> 加急准时%.3f 抢占%d 浪费%.0fs"
                              % (i, total, pol.value, n, a,
                                 s["urgent_on_time_rate"],
                                 s["preemption_count"],
                                 s["material_waste_steps"]), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print()
    print("写入 %s（%d 行）" % (args.out, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())