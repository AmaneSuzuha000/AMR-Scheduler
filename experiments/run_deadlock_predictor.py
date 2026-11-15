"""A13 的普适性检验：死锁打破次数能否作为"配置饱和"的早期预警。

假设
----
在 hub 布局上发现：死锁打破次数与完成率的相关系数 r = −0.673。
若这只是 hub 的特例（枢纽节点被反复争抢），那它只配叫"hub 现象"。
若在 ring / grid 上也成立，它才配当**通用早期预警指标**。

设计
----
三种布局 × 2 档车队（4 / 6 车）× N 种子 × 固定 50 任务。
任务数与车队规模固定，只让布局与种子变化，
这样死锁次数与完成率的差异才归因于**结构**而不是负载。

统计
----
1. 每格的相关性；2. 合并全部样本的相关性；
3. 分布局算，看是否一致。
"""

from __future__ import annotations

import csv
import argparse
import os
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator

OUT = "results/deadlock_predictor.csv"
LAYOUTS = ["ring", "grid", "hub"]
AGVS = [4, 6]
N_SEEDS = int(os.environ.get("DP_SEEDS", "6"))
N_TASKS = 50


def run_one(layout: str, n_agvs: int, seed: int) -> dict:
    net = build_layout(layout, aisle_capacity=2)
    sc = ScenarioConfig(
        n_tasks=N_TASKS, n_agvs=n_agvs, seed=seed,
        dt=1.0, max_speed=1.0, urgent_ratio=0.0,
    )
    _, tasks, agvs = generate_scenario(sc, network=net)
    sim = Simulator(
        net, tasks, agvs,
        StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=120_000, seed=seed,
                         record_trajectory=False),
    )
    res = sim.run()
    s = res.summary()
    return {
        "layout": layout,
        "n_agvs": n_agvs,
        "seed": seed,
        "n_tasks": N_TASKS,
        "tasks_completed": s["tasks_completed"],
        "completion_rate": round(s["tasks_completed"] / N_TASKS, 4),
        "deadlock_breaks": s["deadlock_breaks"],
        "conflict_wait_steps": s["conflict_wait_steps"],
        "makespan_seconds": round(s["makespan_seconds"], 1),
        "n_nodes": len(net.nodes),
        "n_edges": len(net.edges),
    }


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) *
           sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


def spearman(xs, ys):
    """秩相关。对非线性单调关系更稳健，用作 Pearson 的旁证。"""
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    return pearson(rank(xs), rank(ys))


def main() -> int:
    _ap = argparse.ArgumentParser(description="重跑实验结果")
    _ap.add_argument("--out", default="results/deadlock_predictor.csv",
                     help="输出 CSV 路径；默认写正式数据，重跑时请务必指定别的路径")
    out = _ap.parse_args().out

    rows = []
    total = len(LAYOUTS) * len(AGVS) * N_SEEDS
    k = 0
    for layout in LAYOUTS:
        for n_agvs in AGVS:
            for seed in range(N_SEEDS):
                k += 1
                r = run_one(layout, n_agvs, seed)
                rows.append(r)
                print("%3d/%d  %-5s %d车 s%d  完成 %2d/%d  死锁 %5d"
                      % (k, total, layout, n_agvs, seed,
                         r["tasks_completed"], N_TASKS, r["deadlock_breaks"]),
                      flush=True)

    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("")
    print("已写入 %s（%d 行）" % (OUT, len(rows)))

    # 分析
    xs = [r["deadlock_breaks"] for r in rows]
    ys = [r["completion_rate"] for r in rows]
    print("")
    print("=== 相关性 ===")
    print("  全部样本 n=%d  Pearson r=%.3f  Spearman rho=%.3f"
          % (len(rows), pearson(xs, ys), spearman(xs, ys)))
    for layout in LAYOUTS:
        sub = [r for r in rows if r["layout"] == layout]
        x2 = [r["deadlock_breaks"] for r in sub]
        y2 = [r["completion_rate"] for r in sub]
        print("  %-5s n=%2d  Pearson r=%.3f  Spearman rho=%.3f"
              % (layout, len(sub), pearson(x2, y2), spearman(x2, y2)))
    for na in AGVS:
        sub = [r for r in rows if r["n_agvs"] == na]
        x2 = [r["deadlock_breaks"] for r in sub]
        y2 = [r["completion_rate"] for r in sub]
        print("  %d车   n=%2d  Pearson r=%.3f  Spearman rho=%.3f"
              % (na, len(sub), pearson(x2, y2), spearman(x2, y2)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
