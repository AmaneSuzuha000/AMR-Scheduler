#!/usr/bin/env python3
"""物料分类型最小案例（转向架 / 轮轴 / 制动阀）。

为什么做这个
------------
此前 4540 行实验里 Task.material_type 全为 None，所有物料同质。
区分三类检修物料的运输与优先级特征过——


本实验回答一个很窄的问题：
  当装卸时长随物料差异（120 / 60 / 30 步）时，
  调度指标会怎么变？调度算法本身不做任何改造，只换物料口径。

**这不是新方法，是口径敏感性检验。**
结论如果只有轻微变化，说明既有结论对物料异质性是稳健的；
以此评估不同物料配比下的策略适用范围。

设计
----
同一 seed、同一任务规模、同一算法，只切换 material_mix：
  A: material_mix=False  —— 物料同质（历史口径，load=5s）
  B: material_mix=True   —— 三类物料按 20/45/35 混合
配对比较，跑 10 个 seed。
"""
import sys, csv, statistics
sys.path.insert(0, ".")

from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig
from amr_sched.allocation import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner

SEEDS = 10
N_TASKS = 30
N_AGVS = 4


def run_once(seed, material_mix):
    sc = ScenarioConfig(n_tasks=N_TASKS, n_agvs=N_AGVS, seed=seed, dt=1.0,
                        max_speed=1.0, urgent_ratio=0.15,
                        material_mix=material_mix)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=120_000, seed=seed))
    res = sim.run()
    res.total_distance = sim._odometer
    s = res.summary()
    return s


def main():
    rows = []
    print("%4s | %-22s | %-22s | %s" % ("seed", "A 同质", "B 分类型", "差值"))
    print("-" * 84)
    for seed in range(SEEDS):
        a = run_once(seed, False)
        b = run_once(seed, True)
        rows.append((seed, a, b))
        print("%4d | 完成%3d 用时%6.0f | 完成%3d 用时%6.0f | 完成%+3d 用时%+6.0f"
              % (seed, a["tasks_completed"], a["makespan_seconds"],
                 b["tasks_completed"], b["makespan_seconds"],
                 b["tasks_completed"] - a["tasks_completed"],
                 b["makespan_seconds"] - a["makespan_seconds"]))

    def stats(key):
        d = [r[2][key] - r[1][key] for r in rows]
        m = statistics.mean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0.0
        return m, sd

    print("")
    print("=== 配对差值（B - A），n=%d ===" % len(rows))
    for key, label, unit in [("tasks_completed", "完成任务数", ""),
                             ("makespan_seconds", "完工时间", " s"),
                             ("on_time_rate", "准时率", ""),
                             ("conflict_wait_steps", "冲突等待步", ""),
                             ("deadlock_breaks", "死锁打破", "")]:
        m, sd = stats(key)
        print("  %-12s 均值 %+9.3f%s   标准差 %7.3f" % (label, m, unit, sd))

    # 存 CSV
    out = "results/material_mix_case.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["seed", "mode", "tasks_completed", "makespan_seconds",
                    "on_time_rate", "conflict_wait_steps", "deadlock_breaks"])
        for seed, a, b in rows:
            for mode, r in (("homogeneous", a), ("material_mix", b)):
                w.writerow([seed, mode, r["tasks_completed"], r["makespan_seconds"],
                            r["on_time_rate"], r["conflict_wait_steps"],
                            r["deadlock_breaks"]])
    print("")
    print("已写", out)


if __name__ == "__main__":
    main()
