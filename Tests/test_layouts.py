"""替代布局的单元测试（论文 7.2「单一布局」局限的验证基础）。"""

from __future__ import annotations

import pytest

from amr_sched.layouts import build_grid_network, build_hub_network, build_layout
from amr_sched.network import NodeKind, build_demo_network


@pytest.mark.parametrize("name", ["ring", "grid", "hub"])
def test_layout_has_required_zones(name):
    """每种布局都必须有拆解区、组装区、工位、缓存区，否则业务语义不成立。"""
    net = build_layout(name)
    kinds = {n.kind for n in net.nodes.values()}
    assert NodeKind.DISASSEMBLY in kinds, "%s 缺拆解区" % name
    assert NodeKind.ASSEMBLY in kinds, "%s 缺组装区" % name
    assert NodeKind.WORKSTATION in kinds, "%s 缺工位" % name
    assert NodeKind.BUFFER in kinds, "%s 缺缓存区" % name


@pytest.mark.parametrize("name", ["ring", "grid", "hub"])
def test_layout_fully_connected(name):
    """全连通是无例外前提：任一点不可达都会让任务永久挂住。"""
    net = build_layout(name)
    bad = net.unreachable_pairs(1.0)
    assert not bad, "%s 存在不可达对: %s" % (name, bad[:5])


@pytest.mark.parametrize("name", ["ring", "grid", "hub"])
def test_layout_has_multiple_workstations(name):
    """至少 6 个工位，才能体现多工位的调度问题。"""
    net = build_layout(name)
    ws = net.nodes_of_kind(NodeKind.WORKSTATION)
    assert len(ws) >= 6, "%s 只有 %d 个工位" % (name, len(ws))


def test_grid_offers_more_alternative_paths_than_hub():
    """网格的等长替代路径应比枢纽多——这是选它做对照的理由。"""
    grid = build_grid_network()
    hub = build_hub_network()
    assert len(grid.edges) > len(hub.edges)
    # 枢纽图中任一工位到另一工位的最短路都必经 H
    ws = [n.node_id for n in hub.nodes_of_kind(NodeKind.WORKSTATION)]
    for a in ws[:3]:
        times = hub.shortest_travel_times(a, max_speed=1.0)
        for b in ws[3:]:
            direct = times[b]
            via_h = times["H"] + hub.shortest_travel_times("H", 1.0)[b]
            assert direct == pytest.approx(via_h, rel=1e-6), (
                "枢纽布局中 %s->%s 未经过 H" % (a, b))


def test_unknown_layout_raises():
    """未知布局名必须报错，不能静默回退到默认布局。"""
    with pytest.raises(ValueError, match="未知布局"):
        build_layout("no_such_layout")


def test_layouts_are_deterministic():
    """同一布局构造两次，节点与边完全一致。"""
    for name in ("ring", "grid", "hub"):
        a = build_layout(name)
        b = build_layout(name)
        assert set(a.nodes.keys()) == set(b.nodes.keys())
        assert set(a.edges.keys()) == set(b.edges.keys())


def test_aisle_capacity_propagates():
    """通道容量参数必须传到环道节点上，否则负荷算例失效。"""
    net = build_layout("grid", aisle_capacity=1)
    aisles = net.nodes_of_kind(NodeKind.AISLE)
    assert aisles
    assert all(n.capacity == 1 for n in aisles)


def test_ring_layout_matches_demo_network():
    """ring 必须与原来的 build_demo_network 完全一致，保证历史数据可比。"""
    a = build_layout("ring")
    b = build_demo_network()
    assert set(a.nodes.keys()) == set(b.nodes.keys())
    assert len(a.edges) == len(b.edges)