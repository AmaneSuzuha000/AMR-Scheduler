"""Guard the route-to-Nav2 contract: dense samples, unchanged approved geometry."""
from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/live_bridge"))
from gz_frame import DEMO_FRAME
from nav2_path import sample_polyline,shift_tracking_reference
from amr_sched.workshop import WorkshopScene


def _first_rpp_carrot_x(points, robot_x, lookahead=0.5):
    """RPP's nearest-sample pruning and first-outside-circle rule on a straight line.

    This is the upstream failure mechanism, not the sampling implementation.
    A negative first carrot bypasses interpolation and requests a turn backwards.
    """
    nearest = min(range(len(points)), key=lambda i: abs(points[i][0] - robot_x))
    relative = [p[0] - robot_x for p in points[nearest:]]
    return next((x for x in relative if abs(x) >= lookahead), relative[-1])


def test_y30_regression_no_behind_robot_carrot_after_sampling():
    """Recorded failure x=0.884 m must have a forward carrot on the same route."""
    net = WorkshopScene.load().network()
    points = DEMO_FRAME.path_to_world(["J_S_8", "J_S_45", "J_S_85"], net)
    assert _first_rpp_carrot_x(points, 0.884) == pytest.approx(-0.564)
    sampled = sample_polyline(points)
    # Cover the entire first leg, including the observed failure interval.
    for robot_x in (0.32, 0.70, 0.821, 0.884, 1.02, 1.30, 1.79):
        assert _first_rpp_carrot_x(sampled, robot_x) > 0
    assert sampled[-1][:2] == points[-1]


@pytest.mark.parametrize("ids", [
    ["J_S_8", "J_S_45", "J_S_85"],
    ["J_N_25", "J_N_45", "J_N_65"],
])
def test_world_spacing_and_original_vertices_are_preserved(ids):
    points = DEMO_FRAME.path_to_world(ids, WorkshopScene.load().network())
    samples = sample_polyline(points, 0.05)
    xy = [(x, y) for x, y, _ in samples]
    assert xy[0] == points[0] and xy[-1] == points[-1]
    assert all(point in xy for point in points)
    assert max(math.dist(a, b) for a, b in zip(xy, xy[1:])) <= 0.05 + 1e-12
    assert sum(math.dist(a, b) for a, b in zip(xy, xy[1:])) == pytest.approx(
        sum(math.dist(a, b) for a, b in zip(points, points[1:])))


def test_sampling_never_cuts_a_corner_or_changes_segment_order():
    points = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
    samples = sample_polyline(points, 0.13)
    xy = [p[:2] for p in samples]
    i, j = xy.index(points[1]), xy.index(points[2])
    assert all(y == 0.0 for x, y in xy[:i + 1])
    assert all(x == 1.0 for x, y in xy[i:j + 1])
    assert all(x == pytest.approx(y) for x, y in xy[j:])
    assert samples[i][2] == pytest.approx(math.pi / 2)
    assert samples[j][2] == pytest.approx(math.pi / 4)
    assert samples[-1][2] == pytest.approx(math.pi / 4)


def test_only_consecutive_duplicate_vertices_are_removed():
    samples = sample_polyline([(0, 0), (0, 0), (1, 0), (1, 0), (0, 0)], 0.25)
    xy = [p[:2] for p in samples]
    assert xy.count((0.0, 0.0)) == 2  # The return visit is part of the route.
    assert xy.count((1.0, 0.0)) == 1
    assert all(a != b for a, b in zip(xy, xy[1:]))
    assert samples[-1][2] == pytest.approx(math.pi)


@pytest.mark.parametrize("spacing", [0, -0.1, math.nan, math.inf])
def test_invalid_spacing_is_rejected_before_creating_a_path(spacing):
    with pytest.raises(ValueError, match="finite and positive"):
        sample_polyline([(0, 0), (1, 0)], spacing)


@pytest.mark.parametrize("points", [[], [(0, 0)], [(1, 2), (1, 2)]])
def test_empty_or_stationary_path_is_rejected(points):
    with pytest.raises(ValueError, match="distinct"):
        sample_polyline(points)


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_coordinates_are_rejected(value):
    with pytest.raises(ValueError, match="finite"):
        sample_polyline([(0, 0), (value, 1)])


def test_unbounded_sampling_request_is_rejected():
    with pytest.raises(ValueError, match="100000"):
        sample_polyline([(0, 0), (1, 0)], 1e-12)


def test_reference_frame_conversion_preserves_center_route_and_world_collision_envelope():
    """Controller point changes must never shift the task or shrink the robot."""
    offset=.463951/2
    footprint=[(-.315,-.252),(.315,-.252),(.315,.252),(-.315,.252)]
    samples=sample_polyline([(0.,0.),(1.,0.),(1.,2.),(-1.,1.)])
    tracked=shift_tracking_reference(samples,offset)
    assert shift_tracking_reference(samples,0.)==samples
    for center,reference in zip(samples,tracked):
        x,y,yaw=center
        rx,ry,ryaw=reference
        assert ryaw==yaw
        c,s=math.cos(yaw),math.sin(yaw)
        assert (rx-offset*c,ry-offset*s)==pytest.approx((x,y))
        # Transform both footprints through their respective native base TF.
        for fx,fy in footprint:
            original=(x+c*fx-s*fy,y+s*fx+c*fy)
            represented=(rx+c*(fx-offset)-s*fy,ry+s*(fx-offset)+c*fy)
            assert represented==pytest.approx(original)
    for invalid in (-.1,.316,math.nan,math.inf):
        with pytest.raises(ValueError):
            shift_tracking_reference(samples,invalid)
