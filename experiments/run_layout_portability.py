"""多布局验证：四条结论是否依赖单一的 17 节点环网布局。

论文 7.2 节把「单一布局」列为局限。本脚本在 ring / grid / hub 三种布局上
重复主对比实验，检验策略排序是否稳定。

输出 results/layout_portability.csv
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
from amr_sched.layouts import build_layout
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig

OUT = "results/layout_portability.csv"

STRATEGIES = [
    ("baseline_nearest", NearestShortestAllocator, StaticDijkstraPlanner),
    ("baseline_fifo", FifoShortestAllocator, StaticDijkstraPlanner),
    ("dijkstra+greedy", GreedyAllocator, StaticDijkstraPlanner),
    ("spacetime+greedy", GreedyAllocator, SpaceTimeAStarPlanner),
]

FIELDS = ["layout", "strategy", "n_tasks", "n_agvs", "seed",
          "makespan_s", "on_time_rate", "agv_utilization",
          "conflict_wait_steps", "conflict_wait_events",
          "tasks_completed", "n_nodes", "n_edges"]


# 第 64 轮修正：默认值原为 5，但 results/layout_portability.csv（1080 行）
# 是用 10 种子跑出来的。用默认值重跑只得 540 行，会被误判为"数据不一致"。
# 默认值改为 10，与冻结数据对齐。
N_SEEDS = int(os.environ.get("LP_SEEDS", "10"))
OUT = os.environ.get("LP_OUT", OUT)


def main() -> int:
    _ap = argparse.ArgumentParser(description="重跑实验结果")
    _ap.add_argument("--out", default=os.environ.get("LP_OUT", "results/layout_portability.csv"),
                     help="输出 CSV 路径；默认写正式数据，重跑时请务必指定别的路径")
    out = _ap.parse_args().out

    rows = []
    _only = os.environ.get("LP_LAYOUT", "")
    layouts = [_only] if _only else ["ring", "grid", "hub"]
    total = len(layouts) * len(STRATEGIES) * 3 * 3 * N_SEEDS
    done = 0
    for layout in layouts:
        net = build_layout(layout, aisle_capacity=2)
        for name, alloc_cls, planner_cls in STRATEGIES:
            for n in (20, 50, 100):
                for a in (2, 4, 6):
                    for seed in range(N_SEEDS):
                        sc = ScenarioConfig(
                            n_tasks=n, n_agvs=a, seed=seed, dt=1.0,
                            max_speed=1.0, urgent_ratio=0.15)
                        _, tasks, agvs = generate_scenario(sc, network=net)
                        sim = Simulator(
                            net, tasks, agvs,
                            planner_cls(max_speed=1.0, dt=1.0),
                            alloc_cls(),
                            SimulationConfig(
                                dt=1.0, max_steps=120000, seed=seed))
                        res = sim.run()
                        res.total_distance = sim._odometer
                        s = res.summary()
                        rows.append({
                            "layout": layout,
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
                            "n_nodes": len(net.nodes),
                            "n_edges": len(net.edges),
                        })
                        done += 1
                        if done % 60 == 0:
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