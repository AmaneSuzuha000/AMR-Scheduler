"""路径规划器测试。"""

import pytest

from amr_sched.network import Edge, Network, Node, build_demo_network
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner, physical_edge_id
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.reservation import Interval, ReservationTable


@pytest.fixture
def line_net():
    """A--B--C--D 一条直线，每段 10 米。"""
    net = Network()
    for i, nid in enumerate("ABCD"):
        net.add_node(Node(nid, i * 10, 0))
    for a, b in [("A", "B"), ("B", "C"), ("C", "D")]:
        net.add_bidirectional(a, b, 10.0, speed_limit=1.0)
    return net


def test_static_planner_finds_shortest(line_net):
    p = StaticDijkstraPlanner(max_speed=1.0, dt=1.0)
    path = p.plan(line_net, "A", "D", 0, ReservationTable(), "agv1", 1000)
    assert path is not None
    assert path.nodes == ["B", "C", "D"]
    assert path.arrival_steps == [10, 20, 30]


def test_static_planner_same_node(line_net):
    p = StaticDijkstraPlanner(max_speed=1.0, dt=1.0)
    path = p.plan(line_net, "A", "A", 0, ReservationTable(), "agv1", 1000)
    assert path is not None and path.nodes == []


def test_static_planner_unreachable():
    net = Network()
    net.add_node(Node("A", 0, 0))
    net.add_node(Node("B", 1, 0))
    p = StaticDijkstraPlanner()
    assert p.plan(net, "A", "B", 0, ReservationTable(), "a", 100) is None


def test_spacetime_planner_basic(line_net):
    p = SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0)
    path = p.plan(line_net, "A", "D", 0, ReservationTable(), "agv1", 1000)
    assert path is not None
    assert path.nodes[-1] == "D"


def test_spacetime_planner_avoids_blocked_edge(line_net):
    """B->C 在 [0,50) 被占，规划结果要么绕行（无绕行则等待），到达时间必须 > 50。"""
    p = SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0)
    rt = ReservationTable()
    rt.reserve_edge("B", "C", Interval(0, 50), "other")
    rt.reserve_edge("C", "B", Interval(0, 50), "other")
    path = p.plan(line_net, "A", "D", 0, rt, "agv1", 2000)
    assert path is not None
    # 直线图无绕行，只能等
    assert path.arrival_steps[-1] >= 50


def test_spacetime_no_duplicate_consecutive_nodes():
    """回溯折叠必须消除 u->u 自环（等待会产生重复状态）。"""
    net = build_demo_network()
    p = SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0)
    rt = ReservationTable()
    path = p.plan(net, "B1", "W3", 0, rt, "agv1", 3000)
    assert path is not None
    for a, b in zip(path.nodes, path.nodes[1:]):
        assert a != b


def test_spacetime_respects_horizon(line_net):
    p = SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0)
    # 30 步才能到 D，horizon=20 必然失败
    assert p.plan(line_net, "A", "D", 0, ReservationTable(), "a", 20) is None


def test_physical_edge_id_symmetric():
    assert physical_edge_id("A", "B") == physical_edge_id("B", "A")


def test_steps_for_rounds_up():
    p = SpaceTimeAStarPlanner(max_speed=1.0, dt=0.5)
    assert p.steps_for(0.1) == 1        # 至少 1 步
    assert p.steps_for(1.0) == 2
