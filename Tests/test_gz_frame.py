"""Tests for the network <-> Gazebo coordinate transform.

The transform is the only place where "where the scheduler thinks the vehicle is"
and "where Gazebo puts the model" are related. If it were wrong the bridge would
drive to one point while the map drew another, and the symptom would look like a
controller fault rather than a mapping fault. So it is pinned here.
"""
from __future__ import annotations

import math

import pytest

from tools.live_bridge.gz_frame import DEMO_FRAME, GzFrame


def test_round_trip_without_rotation():
    """Every point must survive world -> network -> world unchanged."""
    for x, y in [(-25.0, 20.0), (0.0, 0.0), (40.0, 40.0), (105.0, 20.0)]:
        wx, wy = DEMO_FRAME.to_world(x, y)
        bx, by = DEMO_FRAME.to_network(wx, wy)
        assert bx == pytest.approx(x, abs=1e-9)
        assert by == pytest.approx(y, abs=1e-9)


def test_round_trip_with_rotation():
    """Rotation must be invertible as well, not just the scale part."""
    frame = GzFrame(network_origin=(10.0, 10.0), origin=(1.0, 2.0),
                    scale=0.5, rotation=0.7)
    for x, y in [(-30.0, 75.0), (0.0, 0.0), (110.0, -25.0)]:
        wx, wy = frame.to_world(x, y)
        bx, by = frame.to_network(wx, wy)
        assert bx == pytest.approx(x, abs=1e-9)
        assert by == pytest.approx(y, abs=1e-9)


def test_heading_inverts_with_the_same_rotation_as_position():
    """The position is not the only thing the transform has to undo.

    A heading left un-rotated would put the overlay at the right point facing the wrong
    way, which reads as a controller fault rather than a mapping fault.
    """
    frame = GzFrame(network_origin=(10.0, 10.0), origin=(1.0, 2.0),
                    scale=0.5, rotation=0.7)
    for yaw in (-3.0, -0.4, 0.0, 1.2, 3.0):
        # A heading that is correct in the world frame must come back unchanged.
        assert frame.to_network_yaw(yaw + frame.rotation) == pytest.approx(yaw, abs=1e-9)


def test_yaw_wraps_into_a_single_turn():
    """Bridges send unwrapped yaws; the UI must not draw a heading of 12 rad."""
    assert -math.pi < DEMO_FRAME.to_network_yaw(9.0) <= math.pi


def test_scale_shrinks_the_network_to_the_demo_area():
    """The demo frame must fit the 125 m x 80 m network into a few world metres."""
    # The two extremes of the workshop network.
    west = DEMO_FRAME.to_world(-25.0, 20.0)
    east = DEMO_FRAME.to_world(105.0, 20.0)
    span = abs(east[0] - west[0])
    assert span == pytest.approx(130.0 / 25.0, abs=1e-9)
    assert span < 6.0, "demo area should stay small enough to walk in seconds"


def test_origin_maps_to_origin():
    """R0 at (0,0) is the world origin in the demo frame; document that by test."""
    assert DEMO_FRAME.to_world(0.0, 0.0) == pytest.approx((0.0, 0.0))


def test_non_positive_scale_is_rejected():
    """A zero or negative scale would silently collapse the map; refuse it."""
    with pytest.raises(ValueError):
        GzFrame(scale=0.0)
    with pytest.raises(ValueError):
        GzFrame(scale=-1.0)


def test_path_to_world_follows_node_order():
    """Waypoint order must match the planned node order, not be sorted or deduped."""
    from amr_sched.scenarios import ScenarioConfig, generate_scenario

    net, _tasks, _agvs = generate_scenario(ScenarioConfig(n_tasks=5, n_agvs=1, seed=0))
    nodes = ["R0", "R1", "R4", "B1"]
    waypoints = DEMO_FRAME.path_to_world(nodes, net)

    assert len(waypoints) == len(nodes)
    for node_id, (wx, wy) in zip(nodes, waypoints):
        node = net.node(node_id)
        assert wx == pytest.approx(node.x / 25.0, abs=1e-9)
        assert wy == pytest.approx(node.y / 25.0, abs=1e-9)
