"""R5.6 基准策略对比：朴素基线 vs 本课题方法。

基线代表"上了 AGV 但没上调度系统"的做法：
  - nearest_shortest：车挑最近任务，静态最短路，不做冲突消解优化
  - fifo_shortest   ：先来后到配对，静态最短路

本课题方法：
  - spacetime_astar + priority+greedy（主方法）
  - static_dijkstra + priority+greedy（消融：证明时空预约的贡献）

用法::

    python experiments/run_baseline.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.baseline import (FifoShortestAllocator,
                                           NearestShortestAllocator)
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

FIELDS = [
    "strategy", "planner", "n_tasks", "n_agvs", "seed",
    "tasks_completed", "makespan_s", "on_time_rate", "urgent_on_time_rate",
    "agv_utilization", "conflict_wait_steps", "collision_events",
    "hard_failures", "deadlock_breaks",
]

STRATEGIES = [
    # (名称, 规划器类, 分配器工厂)
    ("nearest_shortest", StaticDijkstraPlanner, NearestShortestAllocator),
    ("fifo_shortest", StaticDijkstraPlanner, FifoShortestAllocator),
    ("dijkstra+greedy", StaticDijkstraPlanner, GreedyAllocator),
    ("spacetime+greedy", SpaceTimeAStarPlanner, GreedyAllocator),
]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--agvs", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(5)))
    p.add_argument("--out", default="results/baseline.csv")
    args = p.parse_args(argv)

    rows = []
    total = len(STRATEGIES) * len(args.tasks) * len(args.agvs) * len(args.seeds)
    i = 0
    for name, planner_cls, alloc_cls in STRATEGIES:
        for n_tasks in args.tasks:
            for n_agvs in args.agvs:
                for seed in args.seeds:
                    i += 1
                    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs,
                                        seed=seed)
                    net, tasks, agvs = generate_scenario(sc)
                    # 每次重建，绝不复用任务对象（曾因此得到假数据）
                    r = Simulator(net, tasks, agvs,
                                  planner_cls(max_speed=sc.max_speed, dt=sc.dt),
                                  alloc_cls(),
                                  SimulationConfig(dt=sc.dt,
                                                   max_steps=500_000)).run()
                    s = r.summary()
                    rows.append({
                        "strategy": name,
                        "planner": planner_cls.__name__,
                        "n_tasks": n_tasks,
                        "n_agvs": n_agvs,
                        "seed": seed,
                        "tasks_completed": int(s["tasks_completed"]),
                        "makespan_s": round(float(s["makespan_seconds"]), 2),
                        "on_time_rate": round(float(s["on_time_rate"]), 4),
                        "urgent_on_time_rate": round(
                            float(s["urgent_on_time_rate"]), 4),
                        "agv_utilization": round(
                            float(s["agv_utilization"]), 4),
                        "conflict_wait_steps": int(s["conflict_wait_steps"]),
                        "collision_events": int(s["collision_events"]),
                        "hard_failures": int(s["hard_failures"]),
                        "deadlock_breaks": int(s["deadlock_breaks"]),
                    })
                    print("[%3d/%d] %-16s %3d任务 %d车 seed=%d -> %5.0fs 完成%3d/%-3d"
                          % (i, total, name, n_tasks, n_agvs, seed,
                             s["makespan_seconds"], s["tasks_completed"],
                             n_tasks), flush=True)

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
