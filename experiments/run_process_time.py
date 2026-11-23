"""工位加工时间敏感性实验（工位加工时间与服务驻留联合建模）。

旧模型「卸货即完成」让工位永远空闲，低估工位排队。
本实验在 5 档加工时间 × 9 格 × 5 种子上重复主配置。

输出 results/process_time.csv
"""

from __future__ import annotations

import csv
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig

OUT = "results/process_time.csv"

#: 加工时间档位（秒）。0 = 旧行为；600s = 典型的检修工序时长量级
PROCESS_TIMES = [0.0, 60.0, 120.0, 300.0, 600.0]

FIELDS = ["process_time", "planner", "n_tasks", "n_agvs", "seed",
          "makespan_s", "on_time_rate", "agv_utilization",
          "conflict_wait_steps", "conflict_wait_events",
          "tasks_completed"]


def main() -> int:
    _ap = argparse.ArgumentParser(description="重跑实验结果")
    _ap.add_argument("--out", default="results/process_time.csv",
                     help="输出 CSV 路径；默认写正式数据，重跑时请务必指定别的路径")
    out = _ap.parse_args().out

    rows = []
    total = len(PROCESS_TIMES) * 2 * 3 * 3 * 5
    done = 0
    for pt in PROCESS_TIMES:
        for pname, pcls in [("static", StaticDijkstraPlanner),
                            ("spacetime", SpaceTimeAStarPlanner)]:
            for n in (20, 50, 100):
                for a in (2, 4, 6):
                    for seed in range(5):
                        sc = ScenarioConfig(
                            n_tasks=n, n_agvs=a, seed=seed, dt=1.0,
                            max_speed=1.0, urgent_ratio=0.15)
                        net, tasks, agvs = generate_scenario(sc)
                        sim = Simulator(
                            net, tasks, agvs,
                            pcls(max_speed=1.0, dt=1.0),
                            GreedyAllocator(),
                            SimulationConfig(
                                dt=1.0, max_steps=200000, seed=seed,
                                process_time=pt))
                        res = sim.run()
                        res.total_distance = sim._odometer
                        s = res.summary()
                        rows.append({
                            "process_time": pt,
                            "planner": pname,
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
                        if done % 45 == 0:
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