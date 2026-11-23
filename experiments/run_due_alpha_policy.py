"""截止期 α 对**算法排序**的影响（E1 补充）。

上一轮只验证了「准时率绝对值随 α 剧烈变化」，
但没验证**方法之间的排序是否稳定**。

这是个要紧的缺口：如果排序会随 α 翻转，
那 6.4 节「抢占优于不抢占」的结论就依赖于 α=3.0 这个约定，
以此检验不同松弛系数下策略排序的稳健性。

设计：α ∈ {1.5, 2, 3, 4} × 三档抢占策略 × 10 种子。
判据：每个 α 下 preempt_unpicked 是否都优于 none。
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator  # noqa: E402
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner  # noqa: E402
from amr_sched.reschedule import ReschedulePolicy  # noqa: E402
from amr_sched.scenarios import DueModel, ScenarioConfig, generate_scenario  # noqa: E402
from amr_sched.simulation import SimulationConfig, Simulator  # noqa: E402

ALPHAS = [1.5, 2.0, 3.0, 4.0]
POLICIES = [ReschedulePolicy.NONE, ReschedulePolicy.PREEMPT_UNPICKED,
            ReschedulePolicy.PREEMPT_ANY]
SEEDS = list(range(10))


def run_one(alpha, policy, seed):
    sc = ScenarioConfig(n_tasks=50, n_agvs=4, seed=seed, dt=1.0,
                        urgent_ratio=0.15, max_speed=1.0,
                        deadline_slack=alpha, due_model=DueModel.TRAVEL_ONLY)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=120_000, seed=seed,
                                     reschedule_policy=policy))
    s = sim.run().summary()
    return {
        "alpha": alpha, "policy": policy.value, "seed": seed,
        "makespan_s": round(s["makespan_seconds"], 1),
        "on_time_rate": round(s["on_time_rate"], 4),
        "urgent_on_time_rate": round(s["urgent_on_time_rate"], 4),
        "mean_urgent_response_steps": round(s["mean_urgent_response_steps"], 1),
        "preemption_count": s["preemption_count"],
    }


def main() -> int:
    rows = []
    for alpha in ALPHAS:
        for pol in POLICIES:
            for seed in SEEDS:
                rows.append(run_one(alpha, pol, seed))
            print("  alpha=%.1f %s done" % (alpha, pol.value), flush=True)

    out = os.path.join("results", "due_alpha_policy.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("已写入 %s（%d 行）" % (out, len(rows)))

    print("")
    print("α      策略                加急准时率   完成时间     响应步数")
    for alpha in ALPHAS:
        for pol in POLICIES:
            sub = [r for r in rows
                   if r["alpha"] == alpha and r["policy"] == pol.value]
            u = sum(r["urgent_on_time_rate"] for r in sub) / len(sub)
            mk = sum(r["makespan_s"] for r in sub) / len(sub)
            rs = sum(r["mean_urgent_response_steps"] for r in sub) / len(sub)
            print("%-5.1f  %-18s  %.4f      %.1f      %.1f"
                  % (alpha, pol.value, u, mk, rs))

    print("")
    print("排序稳定性检验：每个 α 下 preempt_unpicked 是否优于 none")
    ok = 0
    for alpha in ALPHAS:
        a = [r for r in rows if r["alpha"] == alpha
             and r["policy"] == "preempt_unpicked"]
        b = [r for r in rows if r["alpha"] == alpha and r["policy"] == "none"]
        ua = sum(r["urgent_on_time_rate"] for r in a) / len(a)
        ub = sum(r["urgent_on_time_rate"] for r in b) / len(b)
        better = ua > ub
        ok += 1 if better else 0
        print("  α=%.1f  %.4f vs %.4f  %s"
              % (alpha, ua, ub, "优于" if better else "**不优于**"))
    print("")
    print("结论：4 个 α 中 %d 个保持同向" % ok)
    return 0


if __name__ == "__main__":
    sys.exit(main())
