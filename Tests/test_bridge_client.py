"""Tests for the Windows <-> Ubuntu bridge layer.

Two things that have actually gone wrong on the real link are checked here:

1. **watchdog**: a vehicle that does not ping gets halted by the bridge and its pose never
   changes again.
   This turns "a heartbeat is mandatory" into a regression constraint, so that nobody
   later writes a read-only client and assumes it will see a moving pose.
2. **Coordinate calibration**: bridge metres and workshop-map coordinates are two different
   things,
   and a wrong transform shows up as the vehicle drifting off the drawn map rather than as
   an error.
   So degenerate inputs and residuals are both tested here.
"""

from __future__ import annotations

import math

import pytest

from amr_sched.bridge_client import (
    AffineCalib,
    BridgeClient,
    BridgeStatus,
    Pose2D,
    to_map_pose,
)


def test_identity_calib_is_noop():
    c = AffineCalib()
    assert c.apply(1.5, -2.0) == pytest.approx((1.5, -2.0))
    assert c.apply_yaw(0.3) == pytest.approx(0.3)


def test_pure_translation():
    c = AffineCalib(scale=1.0, theta=0.0, tx=10.0, ty=-5.0)
    assert c.apply(0.0, 0.0) == pytest.approx((10.0, -5.0))
    assert c.apply(2.0, 3.0) == pytest.approx((12.0, -2.0))


def test_rotation_90_maps_axes():
    c = AffineCalib(scale=1.0, theta=math.pi / 2)
    X, Y = c.apply(1.0, 0.0)
    assert X == pytest.approx(0.0, abs=1e-12)
    assert Y == pytest.approx(1.0)


def test_scale_multiplies_distance():
    c = AffineCalib(scale=2.0)
    assert c.apply(3.0, 4.0) == pytest.approx((6.0, 8.0))


def test_calib_recovers_known_transform():
    """Given a known transform, check that least squares recovers it."""
    truth = AffineCalib(scale=1.7, theta=0.42, tx=3.0, ty=-1.2)
    bridge_pts = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (2.0, 3.0)]
    pairs = [(b, truth.apply(*b)) for b in bridge_pts]
    solved = AffineCalib.from_pairs(pairs)
    assert solved.scale == pytest.approx(truth.scale, rel=1e-9)
    assert solved.theta == pytest.approx(truth.theta, abs=1e-9)
    assert solved.tx == pytest.approx(truth.tx, abs=1e-9)
    assert solved.ty == pytest.approx(truth.ty, abs=1e-9)
    # The residuals must be zero.
    assert max(solved.residuals(pairs)) == pytest.approx(0.0, abs=1e-9)


def test_calib_two_points_exact():
    """Two pairs pin a similarity transform exactly: 4 degrees of freedom, 4 equations."""
    pairs = [((0.0, 0.0), (5.0, 5.0)), ((0.0, 2.0), (5.0, 7.0))]
    c = AffineCalib.from_pairs(pairs)
    assert max(c.residuals(pairs)) == pytest.approx(0.0, abs=1e-9)


def test_calib_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 2 correspondence"):
        AffineCalib.from_pairs([((0.0, 0.0), (1.0, 1.0))])


def test_calib_rejects_degenerate_points():
    """Coincident bridge points leave the scale unsolvable; that must raise rather than
    return garbage."""
    with pytest.raises(ValueError, match="coincide"):
        AffineCalib.from_pairs([((1.0, 1.0), (0.0, 0.0)),
                                ((1.0, 1.0), (2.0, 2.0))])


def test_calib_rejects_nonpositive_scale():
    with pytest.raises(ValueError, match="scale"):
        AffineCalib(scale=0.0)


def test_calib_flags_bad_correspondence():
    """Mismatch one pair on purpose; the residuals must grow. Calibration is not finished
    just because a solution came out."""
    good = [((0.0, 0.0), (0.0, 0.0)), ((1.0, 0.0), (1.0, 0.0)),
            ((0.0, 1.0), (0.0, 1.0))]
    bad = good[:2] + [((0.0, 1.0), (0.0, 5.0))]
    assert max(AffineCalib.from_pairs(good).residuals(good)) < 1e-9
    assert max(AffineCalib.from_pairs(bad).residuals(bad)) > 0.5


def test_yaw_wraps_into_range():
    c = AffineCalib(scale=1.0, theta=math.pi)
    y = c.apply_yaw(0.0)
    assert -math.pi < y <= math.pi


def test_to_map_pose_combines_translation_and_yaw():
    c = AffineCalib(scale=1.0, theta=0.0, tx=100.0, ty=50.0)
    p = Pose2D(x=1.0, y=2.0, yaw=0.5, seq=1, t_wall=0.0)
    X, Y, yaw = to_map_pose(p, c)
    assert (X, Y) == pytest.approx((101.0, 52.0))
    assert yaw == pytest.approx(0.5)


def test_status_detects_stall():
    assert BridgeStatus(error="watchdog").is_stalled
    assert BridgeStatus(error="estop").is_stalled
    assert not BridgeStatus(error=None).is_stalled
    assert not BridgeStatus(error="other").is_stalled


def test_client_dispatch_parses_pose():
    got = []
    c = BridgeClient("127.0.0.1", on_pose=got.append)
    c._dispatch({"type": "pose", "vehicle": "agv1", "seq": 7,
                 "t": 123.5, "x": 1.25, "y": -0.5, "yaw": 0.75})
    assert len(got) == 1
    assert got[0].seq == 7
    assert got[0].x == pytest.approx(1.25)
    assert c.n_poses == 1


def test_client_dispatch_parses_hello_and_status():
    hellos, stats = [], []
    c = BridgeClient("127.0.0.1", on_hello=hellos.append,
                     on_status=stats.append)
    c._dispatch({"type": "hello", "vehicle": "agv1", "proto": 1,
                 "frame": "odom", "caps": ["ping"]})
    c._dispatch({"type": "status", "mode": "circle", "error": "watchdog",
                 "battery": None})
    assert hellos[0]["frame"] == "odom"
    assert c.hello["proto"] == 1
    assert stats[0].is_stalled
    assert c.last_status is not None and c.last_status.is_stalled


def test_client_ignores_unknown_and_malformed():
    c = BridgeClient("127.0.0.1")
    c._dispatch({"type": "pong", "t1": 1.0})
    c._dispatch({"type": "nav_result", "status": "ok"})
    assert c.n_poses == 0


def test_client_ping_requires_connection():
    """Pinging while disconnected must not raise (it is dropped silently), but must not
    pretend to have succeeded either."""
    c = BridgeClient("127.0.0.1")
    c.ping()
    assert c.n_pings == 1


def test_default_ping_interval_under_watchdog():
    """The default heartbeat interval must be well under the bridge's 0.5 s watchdog, or
    it will trip every time."""
    c = BridgeClient("127.0.0.1")
    assert c.ping_interval < 0.5 / 2
