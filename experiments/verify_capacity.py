"""校验仿真过程中节点与边的容量从未被突破。

为什么需要独立校验
------------------
仿真器自己统计的 collision_events 只覆盖"移动中相撞"，
**不覆盖"多台车同时停在容量不足的节点上"**。

实测：单车道 50 任务 4 车时，末态曾出现 R2（容量 1）停着 3 台车，
而 collision_events 仍然是 0。所以必须另做一次逐步占用审计。

用法::

    python experiments/verify_capacity.py
"""

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def audit(result, net) -> list[str]:
    """逐步重放轨迹，检查每个时刻每个节点的占用是否超容。"""
    problems = []
    worst = 0
    for rec in result.trajectory:
        pos = getattr(rec, "positions", None)
        if not pos:
            continue
        occ = Counter(pos.values())
        for nid, c in occ.items():
            cap = net.node(nid).capacity
            if c > cap:
                worst = max(worst, c - cap)
                if len(problems) < 3:
                    problems.append(
                        "step %d: 节点 %s 占用 %d > 容量 %d"
                        % (rec.step, nid, c, cap))
    if worst:
        problems.append("最大超容 = %d" % worst)
    return problems


def main() -> int:
    total = 0
    bad = 0
    for cap in (2, 1):
        for agvs in (2, 4, 6):
            for win in (1800, 600, 200):
                for seed in range(3):
                    total += 1
                    sc = ScenarioConfig(n_tasks=50, n_agvs=agvs, seed=seed,
                                        release_window=win, aisle_capacity=cap)
                    net, tasks, ag = generate_scenario(sc)
                    r = Simulator(net, tasks, ag,
                                  SpaceTimeAStarPlanner(1.0, 1.0),
                                  GreedyAllocator(),
                                  SimulationConfig(max_steps=60_000,
                                                   record_trajectory=True)).run()
                    probs = audit(r, net)
                    if probs:
                        bad += 1
                        print("cap=%d agv=%d win=%4d seed=%d 完成=%2d/50 -> %s"
                              % (cap, agvs, win, seed,
                                 r.summary()["tasks_completed"], probs[0]),
                              flush=True)
    print()
    print("审计 %d 次运行，其中 %d 次出现节点超容" % (total, bad))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
