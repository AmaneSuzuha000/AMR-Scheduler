"""补齐 conflict_wait_events 数据：在 baseline 的 4 策略 × 9 格上重跑。

原 baseline.csv 只有 conflict_wait_steps（步数），系统规范 要求的是次数。
本脚本复用同一接口，额外记录事件数，输出 baseline_events.csv。
"""

from __future__ import annotations

import csv
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.baseline import (
    FifoShortestAllocator, NearestShortestAllocator)
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig

OUT = "results/baseline_events.csv"

STRATEGIES = [
    ("baseline_nearest", NearestShortestAllocator, StaticDijkstraPlanner),
    ("baseline_fifo", FifoShortestAllocator, StaticDijkstraPlanner),
    ("dijkstra+greedy", GreedyAllocator, StaticDijkstraPlanner),
    ("spacetime+greedy", GreedyAllocator, SpaceTimeAStarPlanner),
]

FIELDS = ["strategy", "n_tasks", "n_agvs", "seed",
          "makespan_s", "on_time_rate", "agv_utilization",
          "conflict_wait_steps", "conflict_wait_events",
          "tasks_completed"]


def main() -> int:
    _ap = argparse.ArgumentParser(description="重跑实验结果")
    _ap.add_argument("--out", default="results/baseline_events.csv",
                     help="输出 CSV 路径；默认写正式数据，重跑时请务必指定别的路径")
    out = _ap.parse_args().out

    rows = []
    total = len(STRATEGIES) * 3 * 3 * 5
    done = 0
    for name, alloc_cls, planner_cls in STRATEGIES:
        for n in (20, 50, 100):
            for a in (2, 4, 6):
                for seed in range(5):
                    sc = ScenarioConfig(n_tasks=n, n_agvs=a, seed=seed,
                                        dt=1.0, max_speed=1.0, urgent_ratio=0.15)
                    net, tasks, agvs = generate_scenario(sc)
                    sim = Simulator(
                        net, tasks, agvs,
                        planner_cls(max_speed=1.0, dt=1.0),
                        alloc_cls(),
                        SimulationConfig(dt=1.0, max_steps=60000, seed=seed))
                    res = sim.run()
                    res.total_distance = sim._odometer
                    s = res.summary()
                    rows.append({
                        "strategy": name,
                        "n_tasks": n,
                        "n_agvs": a,
                        "seed": seed,
                        "makespan_s": round(s["makespan_seconds"], 3),
                        "on_time_rate": round(s["on_time_rate"], 6),
                        "agv_utilization": round(s["agv_utilization"], 6),
                        "conflict_wait_steps": s["conflict_wait_steps"],
                        "conflict_wait_events": s["conflict_wait_events"],
                        "tasks_completed": s["tasks_completed"],
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