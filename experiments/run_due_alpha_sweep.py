"""截止期松弛系数 α 敏感性扫描（E1）。

两种做法分开做：
  1. 只改评价截止期：轨迹不变，对已完成轨迹重新判定准时与否；
  2. 改调度再重跑：α 进入 ScenarioConfig.deadline_slack，影响 EDD 决策。

为什么必须分开：混在一起无法判断「准时率变化」是来自指标定义
还是来自调度行为改变。

用法::

    python experiments/run_due_alpha_sweep.py
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator  # noqa: E402
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner  # noqa: E402
from amr_sched.scenarios import (  # noqa: E402
    DueModel, ScenarioConfig, generate_scenario,
)
from amr_sched.simulation import SimulationConfig, Simulator  # noqa: E402

ALPHAS = [1.5, 2.0, 3.0, 4.0]
SEEDS = list(range(10))


def run_one(n_tasks, n_agvs, seed, alpha):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed, dt=1.0,
                        urgent_ratio=0.15, max_speed=1.0,
                        deadline_slack=alpha,
                        due_model=DueModel.TRAVEL_ONLY)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=120_000, seed=seed))
    res = sim.run()
    s = res.summary()
    return {
        "n_tasks": n_tasks, "n_agvs": n_agvs, "seed": seed, "alpha": alpha,
        "tasks_completed": s["tasks_completed"],
        "makespan_s": round(s["makespan_seconds"], 1),
        "on_time_rate": round(s["on_time_rate"], 4),
        "urgent_on_time_rate": round(s["urgent_on_time_rate"], 4),
    }


def main() -> int:
    out = os.path.join("results", "due_alpha_sweep.csv")
    rows = []
    total = len(ALPHAS) * len(SEEDS) * 3
    k = 0
    for alpha in ALPHAS:
        for n_tasks in (20, 50, 100):
            for seed in SEEDS:
                rows.append(run_one(n_tasks, 4, seed, alpha))
                k += 1
                if k % 20 == 0:
                    print("  %d/%d" % (k, total), flush=True)
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("已写入 %s（%d 行）" % (out, len(rows)))

    # 汇总
    print("")
    print("α     任务规模   准时率均值   完成时间均值")
    for alpha in ALPHAS:
        for n in (20, 50, 100):
            sub = [r for r in rows if r["alpha"] == alpha and r["n_tasks"] == n]
            ot = sum(r["on_time_rate"] for r in sub) / len(sub)
            mk = sum(r["makespan_s"] for r in sub) / len(sub)
            print("%-4s  %-8d  %.4f       %.1f" % (alpha, n, ot, mk))
    return 0


if __name__ == "__main__":
    sys.exit(main())
