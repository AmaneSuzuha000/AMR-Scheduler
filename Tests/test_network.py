"""网络模型测试。"""

import pytest

from amr_sched.network import Edge, Network, Node, NodeKind, build_demo_network


def test_add_node_and_edge():
    net = Network()
    net.add_node(Node("A", 0, 0))
    net.add_node(Node("B", 10, 0))
    net.add_edge(Edge("A", "B", 10.0))
    assert len(net) == 2
    assert net.has_edge("A", "B")
    assert not net.has_edge("B", "A")


def test_duplicate_node_rejected():
    net = Network()
    net.add_node(Node("A", 0, 0))
    with pytest.raises(ValueError):
        net.add_node(Node("A", 1, 1))


def test_edge_rejects_missing_node():
    net = Network()
    net.add_node(Node("A", 0, 0))
    with pytest.raises(ValueError):
        net.add_edge(Edge("A", "Z", 1.0))


def test_edge_rejects_self_loop():
    net = Network()
    net.add_node(Node("A", 0, 0))
    with pytest.raises(ValueError):
        net.add_edge(Edge("A", "A", 1.0))


def test_zero_length_rejected():
    with pytest.raises(ValueError):
        Edge("A", "B", 0.0)


def test_shortest_travel_time_chain():
    net = Network()
    for i, nid in enumerate("ABC"):
        net.add_node(Node(nid, i, 0))
    net.add_bidirectional("A", "B", 10.0, speed_limit=1.0)
    net.add_bidirectional("B", "C", 20.0, speed_limit=1.0)
    d = net.shortest_travel_times("A", 1.0)
    assert d["B"] == pytest.approx(10.0)
    assert d["C"] == pytest.approx(30.0)


def test_shortest_prefers_short_way():
    net = Network()
    for nid, x, y in [("A", 0, 0), ("B", 10, 0), ("C", 10, 10), ("D", 0, 10)]:
        net.add_node(Node(nid, x, y))
    net.add_bidirectional("A", "B", 10.0)
    net.add_bidirectional("B", "C", 10.0)
    net.add_bidirectional("C", "D", 10.0)
    net.add_bidirectional("D", "A", 10.0)
    assert net.shortest_travel_times("A", 1.0)["C"] == pytest.approx(20.0)


def test_unreachable_pairs():
    net = Network()
    net.add_node(Node("A", 0, 0))
    net.add_node(Node("B", 1, 0))
    net.add_node(Node("C", 2, 0))
    net.add_edge(Edge("A", "B", 1.0))
    bad = net.unreachable_pairs(1.0)
    assert ("A", "C") in bad
    assert ("C", "A") in bad


def test_demo_network_is_connected_and_has_cycles():
    net = build_demo_network()
    assert len(net) >= 12
    assert len(net.edges) >= 20
    assert net.unreachable_pairs(1.0) == []


def test_node_capacity_validated():
    with pytest.raises(ValueError):
        Node("A", 0, 0, capacity=0)


def test_nodes_of_kind_filters():
    net = build_demo_network()
    stations = net.nodes_of_kind(NodeKind.WORKSTATION)
    assert len(stations) == 6
    assert all(n.kind is NodeKind.WORKSTATION for n in stations)
