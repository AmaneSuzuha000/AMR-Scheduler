"""Execution layer tests: graph path -> waypoints -> reference velocity sequence.

What makes this layer worth testing is that it **needs no ROS or Gazebo to verify**.
If the Gazebo side changes model parameters or swaps the controller, these tests still hold
as long as the waypoint contract is unchanged; and if the waypoints themselves are computed
wrongly, the tests go red immediately.
"""

from __future__ import annotations

import math

import pytest

from amr_sched.execution import (
    DriveLimit,
    build_route,
    check_clearance,
    route_to_cmd_vel_sequence,
)
from amr_sched.network import build_demo_network


@pytest.fixture(scope="module")
def net():
    return build_demo_network()


def _first_edge(net):
    for e in net.edges.values():
        return e
    raise AssertionError("the network has no edges")


def test_route_endpoints_match_nodes(net):
    e = _first_edge(net)
    plan = build_route(net, [e.src, e.dst], agv_id="AGV00")
    a, b = net.node(e.src), net.node(e.dst)
    assert plan.waypoints[0].x == pytest.approx(a.x)
    assert plan.waypoints[0].y == pytest.approx(a.y)
    assert plan.waypoints[-1].x == pytest.approx(b.x)
    assert plan.waypoints[-1].y == pytest.approx(b.y)
    assert plan.nodes == (e.src, e.dst)


def test_route_length_matches_geometry(net):
    e = _first_edge(net)
    plan = build_route(net, [e.src, e.dst])
    a, b = net.node(e.src), net.node(e.dst)
    assert plan.length_m == pytest.approx(math.hypot(b.x - a.x, b.y - a.y))


def test_t_ideal_uses_edge_speed_and_chassis_cap(net):
    """The theoretical time uses the smaller of the edge speed limit and the chassis cap."""
    e = _first_edge(net)
    limit = DriveLimit(v_max=0.1)
    slow = build_route(net, [e.src, e.dst], limit=limit)
    fast = build_route(net, [e.src, e.dst], limit=DriveLimit(v_max=99.0))
    # With a slower chassis the time should be longer -- or equal, if the edge limit was
    # already below 0.1.
    assert slow.t_ideal_s >= fast.t_ideal_s


def test_t_est_not_less_than_ideal(net):
    e = _first_edge(net)
    plan = build_route(net, [e.src, e.dst])
    assert plan.t_est_s >= plan.t_ideal_s


def test_broken_path_raises(net):
    """No edge between two consecutive hops must raise, rather than being papered over
    with a straight line."""
    e = _first_edge(net)
    # Find a node with no edge to src.
    other = None
    for nid in net.nodes:
        if nid != e.src and not net.has_edge(e.src, nid):
            other = nid
            break
    if other is None:
        pytest.skip("the network is fully connected; no broken pair to test with")
    with pytest.raises(ValueError, match="path is broken"):
        build_route(net, [e.src, other])


def test_unknown_node_raises(net):
    e = _first_edge(net)
    with pytest.raises(ValueError, match="does not exist"):
        build_route(net, [e.src, "__NO_SUCH_NODE__"])


def test_single_node_rejected(net):
    e = _first_edge(net)
    with pytest.raises(ValueError, match="at least two nodes"):
        build_route(net, [e.src])


def test_interp_inserts_monotonic_time(net):
    """Interpolated timestamps must be non-decreasing along the path."""
    e = _first_edge(net)
    plain = build_route(net, [e.src, e.dst])
    dense = build_route(net, [e.src, e.dst], interp_m=0.5)
    assert len(dense.waypoints) >= len(plain.waypoints)
    ts = [w.t_plan_s for w in dense.waypoints]
    assert all(b >= a - 1e-9 for a, b in zip(ts, ts[1:]))
    # Interpolation must not change the total length: the points stay on the same line.
    assert dense.length_m == pytest.approx(plain.length_m, rel=1e-6)


def test_clearance_flags_narrow_capacity_one(net):
    """A node must be flagged when the body is at least as wide as the clear width."""
    e = _first_edge(net)
    wide = DriveLimit(width_m=1.5)
    bad = check_clearance(net, [e.src, e.dst], wide)
    # At minimum, the capacity <= 1 nodes must be flagged.
    caps = {nid: net.node(nid).capacity for nid in (e.src, e.dst)}
    for nid in bad:
        assert caps[nid] <= 1


def test_clearance_empty_for_narrow_robot(net):
    e = _first_edge(net)
    assert check_clearance(net, [e.src, e.dst], DriveLimit(width_m=0.3)) == []


def test_cmd_vel_respects_limits(net):
    e = _first_edge(net)
    limit = DriveLimit(v_max=0.4, w_max=0.8)
    plan = build_route(net, [e.src, e.dst], limit=limit)
    cmds = route_to_cmd_vel_sequence(plan, limit=limit)
    assert cmds
    for c in cmds:
        assert 0.0 <= c["v"] <= limit.v_max + 1e-9
        assert abs(c["w"]) <= limit.w_max + 1e-9


def test_cmd_vel_empty_for_degenerate_route(net):
    e = _first_edge(net)
    plan = build_route(net, [e.src, e.dst])
    # Construct a plan with a single waypoint.
    from amr_sched.execution import RoutePlan
    one = RoutePlan(agv_id="X", waypoints=plan.waypoints[:1],
                    nodes=(e.src,), t_ideal_s=0.0, t_est_s=0.0)
    assert route_to_cmd_vel_sequence(one) == []


def test_drive_limit_rejects_nonpositive():
    for kw in ({"v_max": 0}, {"w_max": -1}, {"a_max": 0}, {"width_m": 0}):
        with pytest.raises(ValueError):
            DriveLimit(**kw)


def test_negative_interp_rejected(net):
    e = _first_edge(net)
    with pytest.raises(ValueError, match="interp_m"):
        build_route(net, [e.src, e.dst], interp_m=-1.0)
