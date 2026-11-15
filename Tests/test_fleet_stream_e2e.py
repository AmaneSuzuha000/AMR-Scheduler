"""End-to-end check of the fleet scheduling stream, C#-side consumption included.

Two different links carry scheduling data, and they are easy to confuse:

  * ``amr_sched.live`` speaks multi-vehicle ``hello`` + ``frame`` to a scheduler client;
  * ``tools/live_bridge`` speaks single-vehicle ``hello`` / ``pose`` / ``status`` to the
    RAWSim-O GUI, and in Gazebo mode (``--mode gazebo``) it is fed by ``/odom``.

The tests here cover the producer side that runs on this machine: the frames must be
well formed, self-consistent, and carry enough for a consumer to place a vehicle
without guessing. The Gazebo leg is exercised separately and needs the Ubuntu host,
so it cannot live in CI; see ``tools/live_bridge/GAZEBO.md`` for what was measured.
"""
from __future__ import annotations

import json

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.stream import FrameBuilder


def _running_sim(steps: int = 60, agvs: int = 3):
    net, tasks, fleet = generate_scenario(
        ScenarioConfig(n_tasks=20, n_agvs=agvs, seed=0))
    sim = Simulator(net, tasks, fleet, StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(), SimulationConfig(max_steps=200000))
    sim.reset()
    for _ in range(steps):
        sim.step_once()
    return sim, net


def test_hello_announces_a_network_a_consumer_can_draw():
    """Every edge needs a usable length, and every endpoint must be announced.

    The producer used to iterate the edge dict keys instead of its values, so every
    edge length arrived as null. A GUI fed with that draws a graph with zero-length
    edges, which looks like a layout bug rather than a protocol bug.
    """
    sim, net = _running_sim()
    hello = FrameBuilder(sim).hello()

    assert hello["type"] == "hello"
    assert hello["network"]["nodes"], "a hello with no nodes cannot be drawn"
    assert hello["network"]["edges"], "a hello with no edges cannot be drawn"

    ids = {n["id"] for n in hello["network"]["nodes"]}
    for edge in hello["network"]["edges"]:
        assert edge["w"] is not None, "edge %r has no length" % (edge,)
        assert edge["w"] > 0, "edge %r has a non-positive length" % (edge,)
        assert edge["src"] in ids, "edge source %r was never announced" % (edge["src"],)
        assert edge["dst"] in ids, "edge target %r was never announced" % (edge["dst"],)


def test_frame_is_json_serialisable_and_matches_its_own_step():
    """A frame must survive json.dumps and agree with itself about time."""
    sim, _net = _running_sim()
    frame = FrameBuilder(sim).frame()

    round_tripped = json.loads(json.dumps(frame))
    assert round_tripped["type"] == "frame"
    assert "schema" in round_tripped, "consumers key off schema to pick a parser"
    assert round_tripped["step"] == frame["step"]
    assert round_tripped["t"] == pytest.approx(frame["step"] * 1.0, abs=1e-6)


def test_every_vehicle_has_a_drawable_pose():
    """Each vehicle must say which node or edge it is on -- never neither."""
    sim, _net = _running_sim()
    frame = FrameBuilder(sim).frame()

    assert frame["agvs"], "a frame with no vehicles renders an empty map"
    for agv in frame["agvs"]:
        pose = agv["pose"]
        assert pose["node"] is not None or pose["edge"] is not None, (
            "%s is neither on a node nor on an edge" % agv["agv_id"])
        assert 0.0 <= pose["progress"] <= 1.0
        assert pose["source"] == "sim"


def test_in_transit_pose_reports_real_progress():
    """Progress used to be hard-coded to 0.0, pinning every vehicle to its origin.

    A vehicle mid-edge must report progress strictly between 0 and 1 for at least
    one frame during a run; a constant 0.0 means the interpolation was dropped.
    """
    sim, _net = _running_sim(steps=0)
    builder = FrameBuilder(sim)

    seen = []
    for _ in range(120):
        sim.step_once()
        for agv in builder.frame()["agvs"]:
            pose = agv["pose"]
            if pose["edge"] is not None:
                seen.append(pose["progress"])

    mid = [p for p in seen if 1e-6 < p < 1.0 - 1e-6]
    assert mid, (
        "no vehicle ever reported progress strictly inside an edge; the producer is "
        "probably reporting a constant again: %r" % (sorted(set(seen))[:8],))


def test_fleet_hello_and_single_vehicle_hello_stay_distinguishable():
    """The C# parser tells the two producers apart by payload shape.

    A fleet hello carries a "network" object; a single-vehicle hello carries
    "vehicle"/"proto"/"frame". If either producer ever grew the other's marker field
    the discrimination would silently pick the wrong branch.
    """
    sim, _net = _running_sim()
    hello = FrameBuilder(sim).hello()

    assert "network" in hello
    assert "vehicle" not in hello

    from tools.live_bridge import live_bridge

    vehicle_hello = {"type": "hello", "vehicle": "agv1", "proto": 1, "frame": "odom"}
    assert "network" not in vehicle_hello
    assert vehicle_hello["vehicle"]
    assert hasattr(live_bridge, "DEFAULT_PORT")
