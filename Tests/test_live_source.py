"""Tests for the bridge -> console overlay.

Why this layer needs its own tests
----------------------------------
The overlay is the only path by which a pose this process did not compute reaches the
canvas. Two failure modes matter and neither is caught by the bridge tests:

* a pose that is reported as live when it has stopped updating (a dead bridge looks
  exactly like a parked vehicle), and
* a frame label ("odom") silently read as absolute localisation.

The coordinate conversion is pinned too: the mock bridge drives a circle of radius 0.9 m
in *world* metres, and the demo frame scales the world down by 25, so the same circle must
come back to the console as radius 22.5 network metres. A sign or scale error there would
put the ring in the wrong aisle while every unit test on the bridge still passed.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from mock_bridge import MockBridge  # noqa: E402

from tools.live_bridge.gz_frame import DEMO_FRAME  # noqa: E402
from web.live_source import LiveLink, LiveRegistry  # noqa: E402


@pytest.fixture()
def bridge():
    b = MockBridge(watchdog=0.5, rate=50.0)
    b.start()
    yield b
    b.stop()


def _wait_for_pose(link, timeout=3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        snap = link.snapshot()
        if snap is not None:
            return snap
        time.sleep(0.02)
    return None


def test_snapshot_is_none_before_the_first_pose_arrives():
    """No pose yet is a distinct state from "attached but the vehicle is at 0,0"."""
    link = LiveLink("127.0.0.1", 9, frame=DEMO_FRAME)
    assert link.snapshot() is None


def test_pose_is_converted_from_world_into_network_metres(bridge):
    link = LiveLink("127.0.0.1", bridge.port, frame=DEMO_FRAME)
    link.start()
    try:
        snap = _wait_for_pose(link)
        assert snap is not None, "no pose arrived from the mock bridge"
        # The mock drives a circle of radius 0.9 world m. DEMO_FRAME shrinks the network
        # by 25, so the same circle is 22.5 network m across.
        r = math.hypot(snap.x, snap.y)
        assert r == pytest.approx(0.9 * 25.0, rel=0.02)
    finally:
        link.stop()


def test_channel_label_is_passed_through_and_not_relabelled(bridge):
    """The UI prints this verbatim; renaming it would turn a channel into a claim."""
    link = LiveLink("127.0.0.1", bridge.port, frame=DEMO_FRAME)
    link.start()
    try:
        snap = _wait_for_pose(link)
        assert snap is not None
        assert snap.channel == "sim"          # MockBridge declares frame="sim"
        assert snap.as_dict()["provenance"] == "bridge:sim"
        assert snap.stale is False
    finally:
        link.stop()


def test_a_pose_that_stops_updating_is_marked_stale(bridge):
    """Age past the threshold must flip the flag -- this is the anti-"parked vehicle" guard."""
    link = LiveLink("127.0.0.1", bridge.port, frame=DEMO_FRAME, stale_after=0.3)
    link.start()
    try:
        assert _wait_for_pose(link) is not None
        # Pretend the last pose landed long ago.
        link._last_monotonic = time.monotonic() - 5.0
        snap = link.snapshot()
        assert snap is not None and snap.stale is True
        assert snap.age_s > 4.9
    finally:
        link.stop()


def test_a_watchdog_trip_marks_stale_even_while_poses_keep_arriving():
    """The newer bridge keeps emitting poses with an unchanged value when the watchdog
    trips. Age alone would look fresh, so the status has to be believed as well."""
    link = LiveLink("127.0.0.1", 9, frame=DEMO_FRAME, stale_after=100.0)
    from amr_sched.bridge_client import BridgeStatus, Pose2D
    link._last_pose = Pose2D(x=1.0, y=2.0, yaw=0.0, seq=1, t_wall=0.0, vehicle="agv1")
    link._last_monotonic = time.monotonic()
    assert link.snapshot().stale is False
    link.client.last_status = BridgeStatus(error="watchdog", t_wall=time.monotonic())
    assert link.snapshot().stale is True


def test_an_odom_stale_status_marks_stale_even_while_poses_keep_arriving():
    """Measured 2026-11-05 on the SCOUT Mini: the bridge's rclpy thread died and it went on
    publishing one frozen pose at 20 Hz with a fresh arrival time. Only the status line
     distinguishes that from a parked vehicle, so the overlay must believe it."""
    link = LiveLink("127.0.0.1", 9, frame=DEMO_FRAME, stale_after=100.0)
    from amr_sched.bridge_client import BridgeStatus, Pose2D
    link._last_pose = Pose2D(x=1.0, y=2.0, yaw=0.0, seq=1, t_wall=0.0, vehicle="agv1")
    link._last_monotonic = time.monotonic()
    assert link.snapshot().stale is False
    link.client.last_status = BridgeStatus(error="odom_stale", t_wall=time.monotonic())
    snap = link.snapshot()
    assert snap.stale is True
    # The position is still reported -- it is not invented -- but flagged as untrustworthy.
    # 1.0 world metre maps to 25 network metres under DEMO_FRAME, so this also re-checks
    # that the staleness flag did not short-circuit the coordinate transform.
    assert snap.x == pytest.approx(25.0)

def test_registry_reports_attached_and_waiting_separately(bridge):
    reg = LiveRegistry()
    reg.attach("127.0.0.1", bridge.port, frame=DEMO_FRAME)
    try:
        snap = reg.snapshot()
        assert snap is not None and snap["attached"] is True
        # Either the first pose has landed or it is explicitly waiting; never a silent gap.
        assert snap.get("waiting") is True or "x" in snap
        deadline = time.monotonic() + 3.0
        while time.monotonic() < deadline and reg.snapshot().get("waiting"):
            time.sleep(0.02)
        assert "x" in reg.snapshot()
    finally:
        reg.detach()
    assert reg.snapshot() is None


def test_attach_to_a_closed_port_raises_instead_of_reporting_attached():
    """A refused connection must surface. An attached-but-empty overlay is indistinguishable
    from a vehicle that never moved."""
    reg = LiveRegistry()
    with pytest.raises(OSError):
        reg.attach("127.0.0.1", 1, frame=DEMO_FRAME)
    assert reg.snapshot() is None


# ---------- HTTP surface ----------
# The overlay is only useful if the console can actually reach it over its own endpoints,
# so the wiring is pinned here rather than assumed.

fastapi_testclient = pytest.importorskip(
    "fastapi.testclient", reason="需要 fastapi 才能跑接口测试")
TestClient = fastapi_testclient.TestClient


@pytest.fixture()
def client():
    from web import server as srv
    srv.LIVE.detach()
    yield TestClient(srv.app)
    srv.LIVE.detach()


def test_map_endpoint_serves_geometry_without_a_run(client):
    """The canvas needs the workshop network before any job exists."""
    from web import server as srv
    m = client.get("/api/map?layout=ring").json()
    assert len(m["nodes"]) == 17 and len(m["edges"]) == 36
    assert srv.RUNS == {}, "fetching the map must not create a run"


def test_map_endpoint_rejects_an_unknown_layout(client):
    assert client.get("/api/map?layout=nope").status_code == 400


def test_live_is_off_by_default(client):
    d = client.get("/api/live").json()
    assert d["attached"] is False and d["live"] is None


def test_attach_to_a_dead_port_is_reported_as_502(client):
    """Refusing loudly is the point: a silent empty overlay looks like a parked vehicle."""
    r = client.post("/api/live/attach", json={"host": "127.0.0.1", "port": 1})
    assert r.status_code == 502
    assert "cannot reach" in r.json()["detail"]
    assert client.get("/api/live").json()["attached"] is False


def test_attach_then_detach_round_trip(client, bridge):
    r = client.post("/api/live/attach",
                    json={"host": "127.0.0.1", "port": bridge.port})
    assert r.status_code == 200, r.text
    assert r.json()["attached"] is True

    deadline = time.monotonic() + 3.0
    snap = client.get("/api/live").json()
    while time.monotonic() < deadline and snap["live"].get("waiting"):
        time.sleep(0.02)
        snap = client.get("/api/live").json()
    assert snap["attached"] is True
    assert "x" in snap["live"], snap
    assert snap["live"]["provenance"] == "bridge:sim"

    assert client.delete("/api/live").json()["attached"] is False
    assert client.get("/api/live").json()["attached"] is False


def test_run_frames_do_not_carry_the_live_pose(client, bridge):
    """A run record is persisted to frames.json.

    A pose that came from outside this process must never be written into a run's own
    record -- that would make the file claim the scheduler produced it.
    """
    from web import server as srv
    client.post("/api/live/attach", json={"host": "127.0.0.1", "port": bridge.port})
    body = {"layout": "ring", "n_tasks": 4, "n_agvs": 2, "seed": 1,
            "speed": 0, "max_steps": 60_000}
    rid = client.post("/api/run", json=body).json()["run_id"]
    frame = srv._frame(srv.RUNS[rid])
    assert "live" not in frame
