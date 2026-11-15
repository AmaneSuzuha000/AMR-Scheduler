"""主实验矩阵：3 规模 × 3 车队 × 2 规划器 × 10 种子 = 180 次。

数据版本 v0.4（2026-11-02）：
  修复死锁打破对已取货任务的错误处理（活锁 bug）后重新生成。
  与 v0.3 相比，18 个格中 16 格中位数完全不变，2 格（100任务/6车）
  分别变化 -0.77% 与 -0.12%，均为变快。结论不受影响。
"""

from __future__ import annotations

import csv
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig

OUT = "results/matrix.csv"

FIELDS = ["n_tasks", "n_agvs", "planner", "seed", "tasks_completed",
          "makespan_s", "on_time_rate", "agv_utilization",
          "conflict_wait_steps", "conflict_wait_events",
          "urgent_on_time_rate", "mean_urgent_response_steps",
          "preemption_count", "planning_retries", "hard_failures",
          "collision_events", "deadlock_breaks", "wall_clock_s"]


def main() -> int:
    _ap = argparse.ArgumentParser(description="重跑实验结果")
    _ap.add_argument("--out", default="results/matrix.csv",
                     help="输出 CSV 路径；默认写正式数据，重跑时请务必指定别的路径")
    out = _ap.parse_args().out

    rows = []
    total = 3 * 3 * 2 * 10
    done = 0
    for n in (20, 50, 100):
        for a in (2, 4, 6):
            for pname, pcls in [("static_dijkstra", StaticDijkstraPlanner),
                                ("spacetime_astar", SpaceTimeAStarPlanner)]:
                for seed in range(10):
                    sc = ScenarioConfig(
                        n_tasks=n, n_agvs=a, seed=seed, dt=1.0,
                        max_speed=1.0, urgent_ratio=0.15)
                    net, tasks, agvs = generate_scenario(sc)
                    t0 = time.time()
                    sim = Simulator(
                        net, tasks, agvs, pcls(max_speed=1.0, dt=1.0),
                        GreedyAllocator(),
                        SimulationConfig(dt=1.0, max_steps=60_000, seed=seed))
                    res = sim.run()
                    res.total_distance = sim._odometer
                    s = res.summary()
                    wall = time.time() - t0
                    rows.append({
                        "n_tasks": n,
                        "n_agvs": a,
                        "planner": pname,
                        "seed": seed,
                        "tasks_completed": s["tasks_completed"],
                        "makespan_s": round(s["makespan_seconds"], 3),
                        "on_time_rate": round(s["on_time_rate"], 6),
                        "agv_utilization": round(s["agv_utilization"], 6),
                        "conflict_wait_steps": s["conflict_wait_steps"],
                        "conflict_wait_events": s["conflict_wait_events"],
                        "urgent_on_time_rate": round(
                            s["urgent_on_time_rate"], 6),
                        "mean_urgent_response_steps": round(
                            s["mean_urgent_response_steps"], 3),
                        "preemption_count": s["preemption_count"],
                        "planning_retries": s["planning_retries"],
                        "hard_failures": s["hard_failures"],
                        "collision_events": s["collision_events"],
                        "deadlock_breaks": s["deadlock_breaks"],
                        "wall_clock_s": round(wall, 3),
                    })
                    done += 1
                    if done % 30 == 0:
                        print("  %d/%d" % (done, total), flush=True)

    os.makedirs("results", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("已写入 %s（%d 行）" % (OUT, len(rows)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())