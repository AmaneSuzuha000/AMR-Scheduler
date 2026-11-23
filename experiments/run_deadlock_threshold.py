"""A13 阈值稳健性检验：阈值 100 会不会随场景漂移。

要回答的问题
--------------
上一轮在「50 任务 / 容量 2 / 3 布局」上发现：
死锁打破 >100 次 ⇒ 必然失败，≤100 次 ⇒ 必然全部完成。

但一个只在单一配置下成立的阈值，不是阈值，是巧合。
必须换任务数与通道容量重跑，看阈值是否漂移。

设计
----
两个正交维度：
  * 任务数：20 / 50 / 100（负载规模）
  * 通道容量：1 / 2（结构瓶颈强度）
布局固定为 hub（死锁最频繁的结构，最能暴露问题）。
车队固定 6 车（最容易饱和的规模）。

若阈值在 8 个格子里都稳定，A13 才配叫「跨场景」结论。
若漂移明显，就得改成「与场景相关的相对阈值」。
"""

from __future__ import annotations

import csv
import os
import statistics as st
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

# 默认输出路径；可用 --out 覆盖（避免重跑实验时覆盖正式数据）
DEFAULT_OUT = "results/deadlock_threshold.csv"
TASKS = [20, 50, 100]
CAPACITIES = [1, 2]
N_AGVS = 6
N_SEEDS = int(os.environ.get("DT_SEEDS", "5"))


def run_one(n_tasks: int, cap: int, seed: int) -> dict:
    net = build_layout("hub", aisle_capacity=cap)
    sc = ScenarioConfig(
        n_tasks=n_tasks, n_agvs=N_AGVS, seed=seed,
        dt=1.0, max_speed=1.0, urgent_ratio=0.0,
    )
    _, tasks, agvs = generate_scenario(sc, network=net)
    sim = Simulator(
        net, tasks, agvs,
        StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=150_000, seed=seed,
                         record_trajectory=False),
    )
    res = sim.run()
    s = res.summary()
    return {
        "n_tasks": n_tasks,
        "capacity": cap,
        "n_agvs": N_AGVS,
        "seed": seed,
        "tasks_completed": s["tasks_completed"],
        "completion_rate": round(s["tasks_completed"] / n_tasks, 4),
        "deadlock_breaks": s["deadlock_breaks"],
        "conflict_wait_steps": s["conflict_wait_steps"],
        "makespan_seconds": round(s["makespan_seconds"], 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="死锁阈值实验")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="输出 CSV 路径（默认写正式数据，重跑时请务必指定别的路径）")
    args = ap.parse_args(argv)
    out = args.out
    rows = []
    total = len(TASKS) * len(CAPACITIES) * N_SEEDS
    k = 0
    for n_tasks in TASKS:
        for cap in CAPACITIES:
            for seed in range(N_SEEDS):
                k += 1
                r = run_one(n_tasks, cap, seed)
                rows.append(r)
                print("%3d/%d  %3d任务 cap=%d s%d  完成 %3d/%d  死锁 %6d"
                      % (k, total, n_tasks, cap, seed,
                         r["tasks_completed"], n_tasks, r["deadlock_breaks"]),
                      flush=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("")
    print("已写入 %s（%d 行）" % (out, len(rows)))

    # ---- 分析 ----
    print("")
    print("=== 逐格：死锁 >100 是否仍然预测失败 ===")
    print("%-4s %-4s %-6s %-10s %-10s %-8s" %
          ("任务", "cap", "次数", ">100 失败数", "<=100 失败数", "阈值可分"))
    all_ok = True
    for n_tasks in TASKS:
        for cap in CAPACITIES:
            sub = [r for r in rows
                   if r["n_tasks"] == n_tasks and r["capacity"] == cap]
            hi = [r for r in sub if r["deadlock_breaks"] > 100]
            lo = [r for r in sub if r["deadlock_breaks"] <= 100]
            hi_fail = sum(1 for r in hi if r["completion_rate"] < 0.9999)
            lo_fail = sum(1 for r in lo if r["completion_rate"] < 0.9999)
            ok = (hi_fail == len(hi)) and (lo_fail == 0)
            all_ok = all_ok and ok
            print("%-4d %-4d %-6d %-10s %-10s %-8s" %
                  (n_tasks, cap, len(sub),
                   "%d/%d" % (hi_fail, len(hi)),
                   "%d/%d" % (lo_fail, len(lo)),
                   "是" if ok else "**否**"))

    print("")
    print("=== 全局二分类 ===")
    hi = [r for r in rows if r["deadlock_breaks"] > 100]
    lo = [r for r in rows if r["deadlock_breaks"] <= 100]
    hi_fail = sum(1 for r in hi if r["completion_rate"] < 0.9999)
    lo_fail = sum(1 for r in lo if r["completion_rate"] < 0.9999)
    print("  死锁 >100 : %2d 次，失败 %2d 次（%.0f%%）"
          % (len(hi), hi_fail, 100 * hi_fail / len(hi) if hi else 0))
    print("  死锁 <=100: %2d 次，失败 %2d 次（%.0f%%）"
          % (len(lo), lo_fail, 100 * lo_fail / len(lo) if lo else 0))
    print("")
    if all_ok:
        print("  阈值 100 在全部 %d 个格子中均成立。" % (len(TASKS) * len(CAPACITIES)))
    else:
        print("  阈值在部分格子失效 —— 需改为相对阈值。")

    # ---- 未失控运行的死锁量级随规模的变化 ----
    print("")
    print("=== 未失控运行（完成率=1）的死锁次数分布 ===")
    print("%-4s %-4s %-8s %-8s %-8s" % ("任务", "cap", "中位", "最大", "次数"))
    for n_tasks in TASKS:
        for cap in CAPACITIES:
            sub = [r["deadlock_breaks"] for r in rows
                   if r["n_tasks"] == n_tasks and r["capacity"] == cap
                   and r["completion_rate"] >= 0.9999]
            if not sub:
                print("%-4d %-4d %-8s %-8s %-8d" % (n_tasks, cap, "-", "-", 0))
                continue
            print("%-4d %-4d %-8.0f %-8d %-8d"
                  % (n_tasks, cap, st.median(sub), max(sub), len(sub)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
