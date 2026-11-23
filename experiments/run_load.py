"""通道负荷算例。

"通道负荷"怎么定义
-----------------
本实验将通道负荷操作化为**两个可控维度**：

1. **需求密度**（到达窗口）：固定 50 个任务，用 release_window 控制它们多集中下达。
   * 低负荷：1800 步内陆续到达 —— 车大多闲着，通道几乎不拥堵；
   * 中负荷：600 步（默认）；
   * 高负荷：200 步 —— 几乎同时下达，所有车同时抢通道。

2. **通道容量**（结构）：主环是双车道还是单车道。
   * aisle_capacity = 2（默认，双车道）；
   * aisle_capacity = 1（单车道）—— 任何相向相遇都可能死锁。

同时输出**实测通道负荷率**：\\sum(占用时间) / (总步数 × 容量)，
这是"通道负荷"的直接度量，而不是只看输入参数。

用法::

    python experiments/run_load.py
    python experiments/run_load.py --out results/load.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.viz.corridor import capacity_ceiling, corridor_load

#: (档位名, 到达窗口)；任务数固定 50
LOAD_LEVELS = [("low", 1800), ("medium", 600), ("high", 200)]

PLANNERS = {
    "spacetime_astar": SpaceTimeAStarPlanner,
    "static_dijkstra": StaticDijkstraPlanner,
}

FIELDS = [
    "load_level", "release_window", "aisle_capacity", "n_tasks", "n_agvs",
    "planner", "seed", "tasks_completed", "all_done", "completion_rate", "makespan_s",
    "on_time_rate", "urgent_on_time_rate", "agv_utilization",
    "conflict_wait_steps", "mean_corridor_load", "peak_corridor_load",
    "load_ceiling", "load_ratio",
    "planning_retries", "hard_failures", "deadlock_breaks", "collision_events",
    "wall_clock_s",
]


def run_one(load_level: str, window: int, aisle_cap: int, n_tasks: int,
            n_agvs: int, planner_name: str, seed: int) -> dict:
    sc = ScenarioConfig(
        n_tasks=n_tasks, n_agvs=n_agvs, seed=seed,
        release_window=window, aisle_capacity=aisle_cap,
    )
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(
        net, tasks, agvs,
        PLANNERS[planner_name](max_speed=sc.max_speed, dt=sc.dt),
        GreedyAllocator(),
        SimulationConfig(dt=sc.dt, max_steps=100_000, seed=seed,
                         record_trajectory=True),
    )
    t0 = time.perf_counter()
    r = sim.run()
    elapsed = time.perf_counter() - t0
    s = r.summary()

    # 实测通道负荷（只在完成全部任务时有意义；未完成时也照样统计）
    loads = corridor_load(net, r)
    vals = [v["load"] for v in loads.values()]
    mean_load = sum(vals) / len(vals) if vals else 0.0
    peak_load = max(vals) if vals else 0.0
    # 归一化：原始负荷受车队规模稀释，必须除以物理可达上限才有可比性
    ceil = capacity_ceiling(net, len(agvs))
    load_ratio = (mean_load / ceil) if ceil > 0 else 0.0

    done = int(s["tasks_completed"])
    return {
        "load_level": load_level,
        "release_window": window,
        "aisle_capacity": aisle_cap,
        "n_tasks": n_tasks,
        "n_agvs": n_agvs,
        "planner": planner_name,
        "seed": seed,
        "tasks_completed": done,
        "all_done": 1 if done == n_tasks else 0,
        "completion_rate": round(done / n_tasks, 4),
        "makespan_s": round(float(s["makespan_seconds"]), 2),
        "on_time_rate": round(float(s["on_time_rate"]), 4),
        "urgent_on_time_rate": round(float(s["urgent_on_time_rate"]), 4),
        "agv_utilization": round(float(s["agv_utilization"]), 4),
        "conflict_wait_steps": int(s["conflict_wait_steps"]),
        "mean_corridor_load": round(mean_load, 4),
        "peak_corridor_load": round(peak_load, 4),
        "load_ceiling": round(ceil, 4),
        "load_ratio": round(load_ratio, 4),
        "planning_retries": int(s["planning_retries"]),
        "hard_failures": int(s["hard_failures"]),
        "deadlock_breaks": int(s["deadlock_breaks"]),
        "collision_events": int(s["collision_events"]),
        "wall_clock_s": round(elapsed, 3),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=50)
    p.add_argument("--agvs", type=int, nargs="+", default=[4])
    p.add_argument("--capacities", type=int, nargs="+", default=[2, 1])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", default="results/load.csv")
    args = p.parse_args(argv)

    combos = [(lv, w, c, a, pl, sd)
              for lv, w in LOAD_LEVELS
              for c in args.capacities
              for a in args.agvs
              for pl in PLANNERS
              for sd in args.seeds]
    total = len(combos)
    rows = []
    t_start = time.perf_counter()

    for i, (lv, w, c, a, pl, sd) in enumerate(combos, 1):
        row = run_one(lv, w, c, args.tasks, a, pl, sd)
        rows.append(row)
        print("[%3d/%d] %-6s cap=%d agv=%d %-16s seed=%d -> done=%2d/%d "
              "make=%8.1fs load=%.3f wait=%5d"
              % (i, total, lv, c, a, pl, sd, row["tasks_completed"],
                 args.tasks, row["makespan_s"], row["mean_corridor_load"],
                 row["conflict_wait_steps"]), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w_ = csv.DictWriter(fh, fieldnames=FIELDS)
        w_.writeheader()
        w_.writerows(rows)

    print()
    print("写入 %s（%d 行，用时 %.1fs）"
          % (args.out, len(rows), time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
