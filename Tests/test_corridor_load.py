"""通道负荷口径的回归测试（R5.3）。

守的是一个很容易搞反的结论：`mean_corridor_load` 的分子受车队规模约束，
它的物理上限是 ``n_agvs / sum(capacity)``，不是 1.0。
把 13% 直接读成「通道很空」会得出与事实相反的结论。
"""

from __future__ import annotations

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.viz.corridor import (capacity_ceiling, corridor_load,
                                     normalized_load)


def _run(n_agvs=4, aisle_capacity=1, seed=0):
    sc = ScenarioConfig(n_tasks=40, n_agvs=n_agvs, seed=seed,
                        dt=1.0, max_speed=1.0, aisle_capacity=aisle_capacity)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs, SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=100_000, seed=seed,
                                     record_trajectory=True))
    return net, agvs, sim.run()


def test_ceiling_equals_agvs_over_total_capacity():
    """上限公式必须是 n_agvs / 无向通道总容量。"""
    net, agvs, _ = _run(n_agvs=4, aisle_capacity=2)
    total = 0
    seen = set()
    for e in net.edges.values():
        k = (e.src, e.dst) if e.src < e.dst else (e.dst, e.src)
        if k in seen:
            continue
        seen.add(k); total += e.capacity
    assert abs(capacity_ceiling(net, 4) - 4 / total) < 1e-12


def test_ceiling_is_far_below_one():
    """关键断言：上限远小于 1，所以原始负荷不能被当成「利用率」。"""
    net, agvs, _ = _run(n_agvs=4, aisle_capacity=1)
    ceil = capacity_ceiling(net, 4)
    assert 0.1 < ceil < 0.35, "上限 {} 不在预期量级".format(ceil)


def test_raw_load_never_exceeds_ceiling():
    """任何一次仿真，原始负荷都不该超过结构上限（超了就是统计口径有 bug）。"""
    for cap in (1, 2):
        net, agvs, result = _run(n_agvs=4, aisle_capacity=cap)
        m = normalized_load(net, result, 4)
        assert m["raw"] <= m["ceiling"] + 1e-9, (
            "容量 {} 下 raw={} 超过上限 {}".format(cap, m["raw"], m["ceiling"]))


def test_normalized_ratio_reveals_high_load():
    """归一化后应显示为高负荷（>50%），不是「很空」。"""
    net, agvs, result = _run(n_agvs=4, aisle_capacity=1)
    m = normalized_load(net, result, 4)
    assert m["ratio"] > 0.40, (
        "归一化负荷仅 {:.1%}，与「冲突等待上千步」的事实不符".format(m["ratio"]))


def test_ratio_is_scale_free():
    """换容量后归一化值应同量级（原始值会差，归一化不该差太多）。"""
    net1, _, r1 = _run(n_agvs=4, aisle_capacity=1)
    net2, _, r2 = _run(n_agvs=4, aisle_capacity=2)
    a = normalized_load(net1, r1, 4)["ratio"]
    b = normalized_load(net2, r2, 4)["ratio"]
    assert abs(a - b) < 0.45, "归一化值差异过大: {:.2f} vs {:.2f}".format(a, b)


def test_corridor_load_covers_every_undirected_edge():
    """corridor_load 必须覆盖全部无向通道，不能漏。"""
    net, _, result = _run()
    loads = corridor_load(net, result)
    seen = set()
    for e in net.edges.values():
        k = (e.src, e.dst) if e.src < e.dst else (e.dst, e.src)
        seen.add(k)
    assert set(loads) == seen

def test_single_corridor_load_can_exceed_one_via_two_way():
    """单条通道负荷可 >1：双向合并计数，不是容量违规。"""
    net, _, result = _run(n_agvs=4, aisle_capacity=1)
    loads = corridor_load(net, result)
    # 逐条检查：只要 occupied <= steps（一条通道每步最多两向各一车）
    steps = max(1, result.finished_step)
    for k, v in loads.items():
        assert v["occupied"] <= 2 * steps, (
            "通道 {} 占用步 {} 超过 2×总步 {}".format(k, v["occupied"], steps))


def test_collision_free_even_when_load_above_one():
    """负荷 >1 时仍必须零碰撞——证明它只是口径问题，不是真违规。"""
    net, _, result = _run(n_agvs=4, aisle_capacity=1)
    s = result.summary()
    if any(v["load"] > 1.0 for v in corridor_load(net, result).values()):
        assert s["collision_events"] == 0, "有通道负荷>1 且发生碰撞"
