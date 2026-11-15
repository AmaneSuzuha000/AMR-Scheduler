"""枢纽布局下的通道容量扫描（论文 6.9.4）。

为什么单独做一个脚本
--------------------
6.9.4 想说明的是「枢纽型布局的通过能力存在物理下限」。
这个结论必须由可复现的数据支撑，不能来自一次性手工探查。
因此把 hub 布局 × 4 档枢纽容量 × 2 种车队 × 3 种子 × 2 策略全部跑出来入库。
"""
from __future__ import annotations

import csv
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.baseline import NearestShortestAllocator
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

OUT = "results/hub_capacity.csv"
#: 种子数。默认 3（原始口径）；扩样本时用环境变量覆盖。
N_SEEDS = int(os.environ.get("HUB_SEEDS", "3"))

STRATEGIES = [
    ("nearest_shortest", NearestShortestAllocator, StaticDijkstraPlanner),
    ("dijkstra+greedy", GreedyAllocator, StaticDijkstraPlanner),
    ("spacetime+greedy", GreedyAllocator, SpaceTimeAStarPlanner),
]

FIELDS = ["capacity", "strategy", "n_agvs", "seed", "tasks_total",
          "tasks_completed", "completion_rate", "makespan_s",
          "deadlock_breaks", "conflict_wait_steps", "wall_clock_s"]


def main() -> int:
    _ap = argparse.ArgumentParser(description="重跑实验结果")
    _ap.add_argument("--out", default=os.environ.get("HUB_OUT", "results/hub_capacity.csv"),
                     help="输出 CSV 路径；默认写正式数据，重跑时请务必指定别的路径")
    out = _ap.parse_args().out

    out = os.environ.get("HUB_OUT", OUT)
    rows = []
    n_tasks = 100
    t_all = time.time()
    total = 4 * len(STRATEGIES) * 2 * N_SEEDS
    done = 0
    for cap in (2, 3, 4, 6):
        net = build_layout("hub", aisle_capacity=cap)
        for name, alloc_cls, planner_cls in STRATEGIES:
            for na in (4, 6):
                for seed in range(N_SEEDS):
                    t0 = time.time()
                    sc = ScenarioConfig(
                        n_tasks=n_tasks, n_agvs=na, seed=seed, dt=1.0,
                        max_speed=1.0, urgent_ratio=0.15)
                    _, tasks, agvs = generate_scenario(sc, network=net)
                    sim = Simulator(
                        net, tasks, agvs,
                        planner_cls(max_speed=1.0, dt=1.0),
                        alloc_cls(),
                        SimulationConfig(dt=1.0, max_steps=120_000, seed=seed))
                    res = sim.run()
                    s = res.summary()
                    rows.append({
                        "capacity": cap,
                        "strategy": name,
                        "n_agvs": na,
                        "seed": seed,
                        "tasks_total": n_tasks,
                        "tasks_completed": s["tasks_completed"],
                        "completion_rate": round(s["tasks_completed"] / n_tasks, 6),
                        "makespan_s": round(s["makespan_seconds"], 1)
                            if s["tasks_completed"] else "",
                        "deadlock_breaks": s["deadlock_breaks"],
                        "conflict_wait_steps": s["conflict_wait_steps"],
                        "wall_clock_s": round(time.time() - t0, 3),
                    })
                    done += 1
                    print("  %d/%d  cap=%d %-16s %d车 s%d  完成 %3d  死锁 %d"
                          % (done, total, cap, name, na, seed,
                             s["tasks_completed"], s["deadlock_breaks"]), flush=True)

    os.makedirs("results", exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print("已写入 %s（%d 行，耗时 %.0fs）" % (out, len(rows), time.time() - t_all))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())