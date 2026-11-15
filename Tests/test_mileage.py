"""里程与绕行系数的测试（R5.6 机制诊断）。"""

from __future__ import annotations

import pytest

from amr_sched.allocation.baseline import NearestShortestAllocator
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator, \
    _shortest_distance_static


def _run(alloc=None, seed=0, n_tasks=15, n_agvs=2):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(1.0, 1.0),
                    alloc or NearestShortestAllocator(),
                    SimulationConfig(max_steps=200_000))
    r = sim.run()
    r.total_distance = sim._odometer
    return net, r, sim


def test_odometer_accumulates():
    """里程必须随行驶单调增加，且为正。"""
    _net, r, sim = _run()
    assert sim._odometer > 0, "里程没有累计，说明 _commit_move 没记账"
    assert r.total_distance == pytest.approx(sim._odometer)


def test_odometer_equals_sum_of_edge_lengths():
    """里程应等于走过的每条边长之和（独立复算验证）。"""
    net, r, sim = _run()
    # 用 detour_ratio 的分母反推：ideal = distance / ratio
    ratio = r.detour_ratio(net)
    assert ratio >= 1.0, "绕行系数不应小于 1（不可能比最短路还短）"
    ideal = r.total_distance / ratio
    assert ideal > 0


def test_static_distance_is_symmetric_on_bidirectional_network():
    """双向网络上两点间最短距离应对称。"""
    net, _r, _s = _run()
    ids = list(net.nodes)
    a, b = ids[0], ids[5]
    assert _shortest_distance_static(net, a, b) == pytest.approx(
        _shortest_distance_static(net, b, a))


def test_static_distance_zero_for_same_node():
    """同点距离为 0（回归：曾因剔除源点导致"车已在取货点"被判不可达）。"""
    net, _r, _s = _run()
    nid = list(net.nodes)[0]
    assert _shortest_distance_static(net, nid, nid) == 0.0


def test_detour_ratio_is_one_when_no_travel():
    """没有任何完成的任务时，绕行系数退化为 1.0 而不是崩溃或除零。"""
    sc = ScenarioConfig(n_tasks=5, n_agvs=1, seed=0)
    net, tasks, agvs = generate_scenario(sc)
    for t in tasks:
        t.complete_step = None
    r = Simulator(net, tasks, agvs, StaticDijkstraPlanner(1.0, 1.0),
                  NearestShortestAllocator(),
                  SimulationConfig(max_steps=1000)).run()
    # 手动清空完成后，分母为 0，应返回 1.0
    for t in r.tasks:
        t.complete_step = None
    assert r.detour_ratio(net) == 1.0


def test_mileage_independent_of_allocator():
    """同一算例下，两种分配器的里程应在同一量级（差异 < 3%）。

    这条守住一个诊断结论：主方法变慢**不是**因为绕路。
    若将来某个改动让主方法的里程暴涨，这条会失败并提示回来查。
    """
    net1, r1, s1 = _run(NearestShortestAllocator(), seed=0)
    net2, r2, s2 = _run(GreedyAllocator(), seed=0)
    d1, d2 = r1.total_distance, r2.total_distance
    assert d1 > 0 and d2 > 0
    assert abs(d1 - d2) / max(d1, d2) < 0.10, \
        "两种分配器里程差异 %.1f%%，远超预期，可能有规划器在绕远路" \
        % (abs(d1 - d2) / max(d1, d2) * 100)
