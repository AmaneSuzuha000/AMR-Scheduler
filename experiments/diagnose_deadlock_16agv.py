"""16 台车时死锁打断发生在哪：把"猜测主通道交叉口"变成证据。

包装 Simulator._break_deadlock，记录每次打断时所有车的节点与状态，
统计出现最多的节点与构型。回合制仿真里这是唯一能拿到现场的办法。
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import WorkshopScene, legal_task_pairs, parking_slots


def main(seed: int = 0, n_agvs: int = 16, n_tasks: int = 100) -> int:
    scene = WorkshopScene.load()
    net = scene.network()
    cfg = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    _n, tasks, agvs = generate_scenario(
        cfg, network=net, pairs=legal_task_pairs(scene), start_pool=parking_slots(scene))
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(), SimulationConfig(dt=1.0, max_steps=1_500_000))

    snaps = []
    original = sim._break_deadlock

    def wrapped(*a, **kw):
        before = sim._deadlock_breaks
        out = original(*a, **kw)
        if sim._deadlock_breaks > before:
            # 只记真正打断的那一次：_break_deadlock 每步都被调用，绝大多数调用什么都没做。
            snaps.append({agv.agv_id: (str(agv.cur_node), str(getattr(agv, "state", "")))
                          for agv in sim.agvs})
        return out

    sim._break_deadlock = wrapped  # type: ignore[method-assign]
    res = sim.run()

    summ = res.summary()
    print("任务完成 %d/%d；summary 记 deadlock_breaks=%s，_break_deadlock 被调用 %d 次"
          % (len(res.completed_tasks()), n_tasks, summ["deadlock_breaks"], len(snaps)))
    if not snaps:
        return 0
    node_hits = Counter()
    for snap in snaps:
        for node, _state in snap.values():
            node_hits[node] += 1
    total = len(snaps)
    print("\n打断时被占住的节点（出现次数 / %d 次打断）：" % total)
    for node, cnt in node_hits.most_common(8):
        print("  %-10s %3d  (%.0f%%)" % (node, cnt, 100.0 * cnt / total))

    zone_hits = Counter()
    for node, cnt in node_hits.items():
        try:
            zone_hits[scene.zone_of(node).zone_id] += cnt
        except Exception:
            zone_hits["?"] += cnt
    print("\n按区域：" )
    for zone, cnt in zone_hits.most_common(6):
        print("  %-16s %3d" % (zone, cnt))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*[int(x) for x in sys.argv[1:]]))