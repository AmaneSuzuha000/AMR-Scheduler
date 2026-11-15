"""补齐 A13 的低死锁侧样本：找过渡带。

为什么需要这一轮
----------------
上一轮发现阈值 100 在 6 个格子里都成立，但要诚实标注一个局限：
50 任务 / 容量 1 与 100 任务 / 容量 1 两格里，5 次运行**全部失控**，
只有高死锁一侧的样本。那两格的「可分」是因为**没有反例可被证伪**。

看已有数据能定位问题：
    20任务 cap=1: [11, 16, 1447, 2887, 3200]   ← 有低侧
    50任务 cap=1: [3020, 3024, 3025, 3574, 3577] ← 全高侧
过渡带在 20 与 50 之间。补 30/40 任务，让低侧样本进入原来全高侧的格子。

若 30/40 任务在 cap=1 下能出现「低死锁 + 全完成」，
则「容量 1 必然失控」这个印象被修正：**失控取决于负载与容量的比值，不是容量本身**。
"""

from __future__ import annotations

import csv
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

# 默认输出路径；可用 --out 覆盖（避免重跑实验时覆盖正式数据）
DEFAULT_OUT = "results/deadlock_transition.csv"
TASKS = [25, 30, 35, 40]
N_AGVS = 6
N_SEEDS = int(os.environ.get("DX_SEEDS", "5"))


def run_one(n_tasks: int, seed: int) -> dict:
    """容量固定 1 —— 要补的正是这一档的低侧样本。"""
    net = build_layout("hub", aisle_capacity=1)
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
        "capacity": 1,
        "n_agvs": N_AGVS,
        "seed": seed,
        "tasks_completed": s["tasks_completed"],
        "completion_rate": round(s["tasks_completed"] / n_tasks, 4),
        "deadlock_breaks": s["deadlock_breaks"],
        "conflict_wait_steps": s["conflict_wait_steps"],
        "makespan_seconds": round(s["makespan_seconds"], 1),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="死锁过渡实验")
    ap.add_argument("--out", default=DEFAULT_OUT,
                    help="输出 CSV 路径（默认写正式数据，重跑时请务必指定别的路径）")
    args = ap.parse_args(argv)
    out = args.out
    rows = []
    total = len(TASKS) * N_SEEDS
    k = 0
    for n_tasks in TASKS:
        for seed in range(N_SEEDS):
            k += 1
            r = run_one(n_tasks, seed)
            rows.append(r)
            print("%2d/%d  %3d任务 cap=1 s%d  完成 %3d/%d  死锁 %6d"
                  % (k, total, n_tasks, seed,
                     r["tasks_completed"], n_tasks, r["deadlock_breaks"]),
                  flush=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("")
    print("已写入 %s（%d 行）" % (out, len(rows)))

    print("")
    print("=== 过渡带：cap=1 下死锁随任务数的变化 ===")
    print("%-6s %-24s %-8s %-8s" % ("任务", "死锁次数（逐种子）", "低侧数", "全完成数"))
    for n_tasks in TASKS:
        sub = sorted((r for r in rows if r["n_tasks"] == n_tasks),
                     key=lambda r: r["deadlock_breaks"])
        vals = [r["deadlock_breaks"] for r in sub]
        low = sum(1 for v in vals if v <= 100)
        full = sum(1 for r in sub if r["completion_rate"] >= 0.9999)
        print("%-6d %-24s %-8d %-8d"
              % (n_tasks, str(vals), low, full))

    print("")
    print("=== 合并三批（threshold + transition）看阈值 ===")
    base_rows = list(csv.DictReader(
        open(args.out.replace("deadlock_transition", "deadlock_threshold"), encoding="utf-8")))
    allrows = base_rows + rows
    lo = [r for r in allrows if int(r["deadlock_breaks"]) <= 100]
    hi = [r for r in allrows if int(r["deadlock_breaks"]) > 100]
    lf = sum(1 for r in lo if float(r["completion_rate"]) < 0.9999)
    hf = sum(1 for r in hi if float(r["completion_rate"]) < 0.9999)
    print("  <=100 : %2d 次，失败 %d 次" % (len(lo), lf))
    print("  >100  : %2d 次，失败 %2d 次" % (len(hi), hf))
    print("  合计 %d 次（threshold %d + transition %d）"
          % (len(allrows), len(base_rows), len(rows)))
    if lo and hi:
        print("  空档: (%.0f, %.0f)"
              % (max(int(r["deadlock_breaks"]) for r in lo),
                 min(int(r["deadlock_breaks"]) for r in hi)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
