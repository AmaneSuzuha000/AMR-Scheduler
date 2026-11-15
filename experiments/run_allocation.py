"""分配策略与调度规则对比实验（任务书 R2.4a / R5.6）。

跑 任务规模 × AGV数量 × 策略 × 种子 的矩阵。

策略 = 调度规则 × 分配算法：
* 调度规则：fifo / edd / spt / cr / priority
* 分配算法：greedy / nearest / hungarian

用法::

    python experiments/run_allocation.py
    python experiments/run_allocation.py --tasks 20 --agvs 2
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.dispatch import DispatchRule
from amr_sched.allocation.rule_based import RuleBasedAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

#: (调度规则, 分配算法) 组合。priority+greedy 是当前默认，作为基线。
STRATEGIES: list[tuple[DispatchRule, str]] = [
    (DispatchRule.FIFO, "nearest"),
    (DispatchRule.EDD, "nearest"),
    (DispatchRule.SPT, "nearest"),
    (DispatchRule.CR, "nearest"),
    (DispatchRule.PRIORITY, "nearest"),
    (DispatchRule.PRIORITY, "greedy"),      # 基线
    (DispatchRule.FIFO, "hungarian"),
    (DispatchRule.EDD, "hungarian"),
    (DispatchRule.CR, "hungarian"),
    (DispatchRule.PRIORITY, "hungarian"),
]

FIELDS = [
    "n_tasks", "n_agvs", "dispatch", "assign", "strategy", "seed",
    "tasks_completed", "makespan_s", "on_time_rate", "urgent_on_time_rate",
    "mean_urgent_response_steps", "agv_utilization", "conflict_wait_steps",
    "planning_retries", "hard_failures", "collision_events", "wall_clock_s",
]


def run_one(n_tasks: int, n_agvs: int, dispatch: DispatchRule, assign: str,
            seed: int) -> dict:
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    net, tasks, agvs = generate_scenario(sc)
    alloc = RuleBasedAllocator(dispatch=dispatch, assign=assign)
    sim = Simulator(
        net, tasks, agvs,
        SpaceTimeAStarPlanner(max_speed=sc.max_speed, dt=sc.dt),
        alloc,
        SimulationConfig(dt=sc.dt, max_steps=500_000, seed=seed,
                         reschedule_policy=ReschedulePolicy.NONE),
    )
    t0 = time.perf_counter()
    r = sim.run()
    elapsed = time.perf_counter() - t0
    s = r.summary()
    return {
        "n_tasks": n_tasks,
        "n_agvs": n_agvs,
        "dispatch": dispatch.value,
        "assign": assign,
        "strategy": alloc.name,
        "seed": seed,
        "tasks_completed": int(s["tasks_completed"]),
        "makespan_s": round(float(s["makespan_seconds"]), 2),
        "on_time_rate": round(float(s["on_time_rate"]), 4),
        "urgent_on_time_rate": round(float(s["urgent_on_time_rate"]), 4),
        "mean_urgent_response_steps": round(float(s["mean_urgent_response_steps"]), 1),
        "agv_utilization": round(float(s["agv_utilization"]), 4),
        "conflict_wait_steps": int(s["conflict_wait_steps"]),
        "planning_retries": int(s["planning_retries"]),
        "hard_failures": int(s["hard_failures"]),
        "collision_events": int(s["collision_events"]),
        "wall_clock_s": round(elapsed, 3),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", type=int, nargs="+", default=[20, 50, 100])
    p.add_argument("--agvs", type=int, nargs="+", default=[2, 4, 6])
    p.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    p.add_argument("--out", default="results/allocation.csv")
    args = p.parse_args(argv)

    cells = [(t, a) for t in args.tasks for a in args.agvs]
    total = len(cells) * len(STRATEGIES) * len(args.seeds)
    done = 0
    rows: list[dict] = []

    t_start = time.perf_counter()
    for n_tasks, n_agvs in cells:
        for dispatch, assign in STRATEGIES:
            for seed in args.seeds:
                row = run_one(n_tasks, n_agvs, dispatch, assign, seed)
                rows.append(row)
                done += 1
                print("[%d/%d] %3d/%d %-22s seed=%d -> makespan=%8.1fs "
                      "ontime=%.3f util=%.3f wait=%6d"
                      % (done, total, n_tasks, n_agvs, row["strategy"], seed,
                         row["makespan_s"], row["on_time_rate"],
                         row["agv_utilization"], row["conflict_wait_steps"]),
                      flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)

    print()
    print("写入 %s（%d 行，用时 %.1fs）"
          % (args.out, len(rows), time.perf_counter() - t_start))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
