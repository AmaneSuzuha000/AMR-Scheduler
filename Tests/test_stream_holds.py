"""The frame must publish what Python reserved, not just where vehicles stand.

The L2 requirement is to show junction reservations and congestion. A vehicle marker
cannot carry that: two vehicles queued on one node and two vehicles passing through look
the same, and a junction booked by a vehicle two edges away is invisible entirely. So the
reservation table has to go on the wire.
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from amr_sched.allocation.greedy import GreedyAllocator            # noqa: E402
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner  # noqa: E402
from amr_sched.simulation import SimulationConfig, Simulator       # noqa: E402
from amr_sched.stream import FrameBuilder                          # noqa: E402
from amr_sched.workshop import build_scenario                       # noqa: E402


def _sim(steps: int, n_agvs: int = 4, n_tasks: int = 12, seed: int = 0):
    net, tasks, agvs = build_scenario(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(1.0, 1.0), GreedyAllocator(),
                    SimulationConfig())
    sim.reset()
    for _ in range(steps):
        sim.step_once()
    return sim


def test_frame_publishes_node_holds():
    frame = FrameBuilder(_sim(40)).frame()
    assert "holds" in frame
    assert isinstance(frame["holds"], dict)
    assert frame["holds"], "at step 40 some vehicle holds a node"
    for node_id, owners in frame["holds"].items():
        assert isinstance(node_id, str) and node_id
        assert isinstance(owners, list) and owners
        assert all(isinstance(o, str) and o for o in owners)


def test_holds_come_from_the_reservation_table_not_from_positions():
    """Every published owner must be someone the reservation table says holds that node."""
    sim = _sim(40)
    step = sim._step_cursor
    frame = FrameBuilder(sim).frame()
    snapshot = sim._res.snapshot(step, step + 1)
    expected = {node: sorted({owner for (_s, _e, owner) in entries})
                for node, entries in snapshot.items()}
    assert frame["holds"] == expected


def test_congestion_reports_capacity_from_the_network():
    """A node is congested at *its own* capacity, not at a hard-coded one."""
    sim = _sim(40)
    frame = FrameBuilder(sim).frame()
    net = sim.network
    for node_id, info in frame["congestion"].items():
        assert info["capacity"] == int(getattr(net.nodes[node_id], "capacity", 1) or 1)
        assert info["claims"] == len(frame["holds"][node_id])
        assert info["claims"] >= info["capacity"]
        assert info["claims"] > 1, "a single claim is never congestion"


def test_every_congested_node_is_also_a_held_node():
    """Congestion is a view over holds; it must not invent nodes."""
    frame = FrameBuilder(_sim(40)).frame()
    assert set(frame["congestion"]) <= set(frame["holds"])


def test_a_capacity_two_node_is_not_called_congested_at_one_claim():
    """Pins the capacity comparison: capacity 2 with one claim is occupancy, not a queue."""
    sim = _sim(40)
    frame = FrameBuilder(sim).frame()
    net = sim.network
    for node_id, owners in frame["holds"].items():
        capacity = int(getattr(net.nodes[node_id], "capacity", 1) or 1)
        if len(owners) < capacity:
            assert node_id not in frame["congestion"]



def test_blocked_counts_only_claimants_that_are_actually_stuck():
    """A full node is not a queue: parked vehicles fill a bay without anyone waiting."""
    from amr_sched.stream import _congestion_from_holds
    holds = {"Q_PARK": ["AGV02", "AGV03"], "J_S_85": ["AGV00", "AGV01"]}
    network = _StubNetwork({"Q_PARK": 2, "J_S_85": 2})
    out = _congestion_from_holds(holds, network, {"AGV00"})
    assert out["Q_PARK"]["blocked"] == 0
    assert out["J_S_85"]["blocked"] == 1


class _StubNetwork:
    """Minimal network stand-in: only node capacity is read here."""

    def __init__(self, capacities):
        self.nodes = {k: type("N", (), {"capacity": v})() for k, v in capacities.items()}

def test_congestion_carries_how_long_it_has_lasted():
    """拥堵必须有**时长**：一次堵 1 步和堵 30 步不是一回事，圆圈一样大就把轻重抹平了。"""
    sim = _sim(0)
    builder = FrameBuilder(sim)
    durations = []
    for _ in range(120):
        sim.step_once()
        frame = builder.frame()
        for node_id, info in frame["congestion"].items():
            assert info["steps"] >= 1
            durations.append((frame["step"], node_id, info["steps"]))
    assert durations, "这个场景 120 步内应当出现拥堵"
    # 同一个节点连续出现时，步数必须递增
    by_node = {}
    for step, node_id, steps in durations:
        by_node.setdefault(node_id, []).append((step, steps))
    for node_id, seq in by_node.items():
        for (s0, d0), (s1, d1) in zip(seq, seq[1:]):
            if s1 == s0 + 1:
                assert d1 == d0 + 1, "%s 在连续两步间时长应该加一" % node_id


def test_a_node_that_clears_starts_over_at_one(monkeypatch):
    """堵过一次又通了、再堵，不能接着上一次的数字往上加。"""
    sim = _sim(0)
    builder = FrameBuilder(sim)
    # 直接用受控的 holds，而不是指望真实仿真在若干步内正好"堵了又通"：
    # 那种测试会因为调度策略或种子的变化而随机失效，而这里要钉的是计数逻辑。
    plans = [{"J": ["A", "B"]}, {"J": ["A", "B"]}, {}, {"J": ["A", "B"]}]
    seen = []
    for i, holds in enumerate(plans):
        monkeypatch.setattr(builder, "_holds", lambda step, h=holds: h)
        seen.append(builder._congestion(i).get("J", {}).get("steps"))
    assert seen == [1, 2, None, 1], seen


def test_rewinding_the_clock_clears_the_counters(monkeypatch):
    """同一台服务重跑一轮时步号归零，时长不能继承上一轮的历史。"""
    sim = _sim(0)
    builder = FrameBuilder(sim)
    monkeypatch.setattr(builder, "_holds", lambda step: {"J": ["A", "B"]})
    assert builder._congestion(500)["J"]["steps"] == 1
    assert builder._congestion(501)["J"]["steps"] == 2
    assert builder._congestion(3)["J"]["steps"] == 1, "步号回退说明换了一轮，必须重算"


@pytest.mark.parametrize("steps", [5, 40, 120])
def test_frame_stays_json_serialisable_throughout(steps):
    """The frame crosses a socket; anything not JSON-encodable fails at the boundary."""
    import json
    frame = FrameBuilder(_sim(steps)).frame()
    line = json.dumps(frame, ensure_ascii=False)
    back = json.loads(line)
    assert back["holds"] == frame["holds"]
    assert back["congestion"] == frame["congestion"]
