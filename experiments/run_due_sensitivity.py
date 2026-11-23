"""截止时间口径敏感性实验。

问题
----
当前截止期公式 `due = release + 静态最短行驶时间 × 3.0` **不含排队等待**。
高负荷下任务集中下达，车来不及一台台做完，于是准时率崩到 0.11；
但崩掉反映的是**截止期定得不合理**，不是调度做得差。

如果这项指标的数值主要由截止公式决定，那它就不能用来评价调度方法。

做法
----
**不重跑仿真**，对同一批完成的调度结果用三套口径重新判定：

| 口径 | 定义 |
|---|---|
| travel_only | due = release + 静态最短行驶时间 × 3.0（现行） |
| travel_queue | due = release + 行驶时间 + 每车分摊的排队作业量 |
| fixed | due = release + 600 步固定容差 |

这样能把"调度水平"和"截止口径"两个因素分开。

用法::

    python experiments/run_due_sensitivity.py
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import DueModel, ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

#: 负荷档（到达窗口）与 保持一致
LOAD_LEVELS = [("low", 1800), ("medium", 600), ("high", 200)]

FIELDS = [
    "load_level", "release_window", "n_tasks", "n_agvs", "seed",
    "tasks_completed", "makespan_s", "agv_utilization", "conflict_wait_steps",
    "on_time_travel_only", "on_time_travel_queue", "on_time_fixed",
    "urgent_travel_only", "urgent_travel_queue", "urgent_fixed",
]


def build_dues(result, net, model: DueModel, n_tasks: int, n_agvs: int,
               max_speed: float, dt: float, slack: float = 3.0,
               load_time: float = 5.0, unload_time: float = 5.0) -> dict:
    """按指定口径为每个任务重新计算截止步。"""
    dues = {}
    for t in result.tasks:
        tt = net.shortest_travel_times(t.pickup, max_speed)[t.dropoff]
        travel_steps = tt / dt
        if model is DueModel.TRAVEL_ONLY:
            lead = travel_steps * slack
        elif model is DueModel.TRAVEL_QUEUE:
            per_agv = max(1.0, n_tasks / max(1, n_agvs))
            opp = travel_steps + (load_time + unload_time) / dt
            lead = travel_steps + per_agv * opp
        else:
            lead = 600.0 * slack / 3.0
        dues[t.task_id] = t.release_step + max(1, int(round(lead)))
    return dues


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, default=50)
    p.add_argument("--agvs", type=int, default=4)
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", default="results/due_sensitivity.csv")
    args = p.parse_args(argv)

    rows = []
    total = len(LOAD_LEVELS) * len(args.seeds)
    i = 0
    for lv, win in LOAD_LEVELS:
        for sd in args.seeds:
            i += 1
            sc = ScenarioConfig(n_tasks=args.tasks, n_agvs=args.agvs,
                                seed=sd, release_window=win)
            net, tasks, agvs = generate_scenario(sc)
            r = Simulator(net, tasks, agvs,
                          SpaceTimeAStarPlanner(max_speed=sc.max_speed, dt=sc.dt),
                          GreedyAllocator(),
                          SimulationConfig(dt=sc.dt, max_steps=100_000,
                                           seed=sd)).run()
            s = r.summary()
            rec = {
                "load_level": lv,
                "release_window": win,
                "n_tasks": args.tasks,
                "n_agvs": args.agvs,
                "seed": sd,
                "tasks_completed": int(s["tasks_completed"]),
                "makespan_s": round(float(s["makespan_seconds"]), 2),
                "agv_utilization": round(float(s["agv_utilization"]), 4),
                "conflict_wait_steps": int(s["conflict_wait_steps"]),
            }
            for m in DueModel:
                dues = build_dues(r, net, m, args.tasks, args.agvs,
                                  sc.max_speed, sc.dt)
                got = r.rate_against_due(dues)
                rec["on_time_%s" % m.value] = round(got["on_time_rate"], 4)
                rec["urgent_%s" % m.value] = round(
                    got["urgent_on_time_rate"]
                    if got["urgent_on_time_rate"] == got["urgent_on_time_rate"] else 0.0, 4)
            rows.append(rec)
            print("[%3d/%d] %-6s seed=%d done=%2d/%d make=%8.1fs "
                  "准时率 %.3f / %.3f / %.3f"
                  % (i, total, lv, sd, rec["tasks_completed"], args.tasks,
                     rec["makespan_s"], rec["on_time_travel_only"],
                     rec["on_time_travel_queue"], rec["on_time_fixed"]),
                  flush=True)

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
