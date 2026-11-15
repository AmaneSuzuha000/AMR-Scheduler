"""Rules for the Gazebo route follower, each one tied to a failure that actually happened.

These are not style checks. Every assertion here corresponds to a defect that cost real
time, and the note in each test says which. When the architecture changed, the tests were
rewritten rather than deleted -- a rule that no longer applies is removed, a rule that
still applies keeps its evidence.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FOLLOW = os.path.join(ROOT, "tools", "live_bridge", "gz_follow_routes.py")
POSE_NODE = os.path.join(ROOT, "tools", "live_bridge", "gz_pose_node.py")
DOC = os.path.join(ROOT, "tools", "live_bridge", "GAZEBO.md")


def _src(path=FOLLOW):
    with open(path, encoding="utf-8") as f:
        return f.read()


def _code_only(src):
    """Source with comments and docstrings stripped, for rules about executable code."""
    import ast
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None)
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                body[0].value.value = ""
    return ast.unparse(tree)


def test_pose_comes_from_the_service_and_nowhere_else():
    """Wheel odometry measured 0.879 m against at most 0.45 m of possible travel.

    That is not a noisy estimate of position, it is a different quantity. It is not used as
    ground truth and not blended in as a hint, because a service read costs 0.2 ms and
    there is nothing to gain by avoiding it.
    """
    code = _code_only(_src())
    assert "/odom" not in code, (
        "odometry must not appear in executable code; it may only be discussed in prose")


def test_the_follower_declares_no_yaw_gain_as_measured():
    """Rotation is not repeatable here: 49, 49 and 25 degrees for one command.

    An earlier version carried a table of measured attenuation ratios. Every one of those
    numbers came from a single run, and tuning against them produced a controller that
    drove away from its waypoints. The constant must not be named as if it were measured.
    """
    # Checked against executable code only. The docstring deliberately quotes the old
    # guesses (0.16, 0.31, 0.62, 0.70) so a reader knows what was tried and why it failed;
    # searching the whole file would forbid recording that history.
    src = _src()
    code = _code_only(src)
    assert re.search(r"^YAW_RATIO = 0\.43$", src, re.M)
    for guess in ("YAW_NOMINAL", "TURN_EFFECTIVE"):
        assert guess not in code, (
            "a single-run yaw guess must not return as %s" % guess)


def test_the_documented_rotation_limit_is_stated_not_implied():
    """The spread must be in the document, because it bounds every accuracy claim."""
    doc = _src(DOC)
    # The envelope finding replaced the blanket "rotation is not repeatable" claim: part of
    # the command space does repeat, and which part is the useful result.
    assert re.search(r"14 (cells|of)", doc), (
        "the size of the repeatable envelope must stay recorded")
    assert "0.43" in doc, "the measured yaw ratio must stay recorded"


def test_waypoint_tolerance_is_a_named_constant():
    """A tolerance buried in a comparison is how 4/4 got reported at 1.14 m mean error.

    The runner counted a waypoint as reached when it was within 0.20 m *of the lookahead
    advance*, so waypoints the vehicle never approached were reported as reached. The
    summary line said PASS 4/4 while the mean error was over a metre.
    """
    src = _src()
    assert "REACH_TOL" in src
    assert "REACH_TOL = " in src


def test_reported_error_is_measured_at_the_waypoint_not_elsewhere():
    """The arrival error must be the distance to that waypoint, not to a later one."""
    src = _src()
    assert "math.hypot(wx - pos[0], wy - pos[1])" in src, (
        "the recorded error must be measured against the waypoint being claimed")


def test_routes_are_reset_to_a_known_pose_between_runs():
    """Without a reset each route starts wherever the previous one stopped.

    That makes per-route numbers incomparable, which is worse than not having them.
    """
    src = _src()
    assert "def teleport" in src
    assert "node.teleport()" in src


def test_the_pose_service_node_checks_success():
    """A missing model answers with (0,0,0) and success False.

    Treating that as a real position would silently put the vehicle at the origin and make
    every subsequent measurement wrong without any error being raised.
    """
    src = _src(POSE_NODE)
    assert "if not res.success" in src
    assert "status_message" in src


def test_the_pose_service_node_keeps_the_node_for_spinning():
    """In Humble spin_until_future_complete is rclpy.spin_until_future_complete(node, ...).

    Passing the client raises "Client object has no attribute subscriptions"; passing the
    node alone raises "Node object has no attribute spin_until_future_complete". Both were
    tried before the working form was found.
    """
    src = _src(POSE_NODE)
    assert "rclpy.spin_until_future_complete(self._node, future" in src
    assert "self._node.spin_until_future_complete" not in src
    assert "rclpy.spin_until_future_complete(self._get," not in src


def test_arrival_is_closest_approach_not_first_entry():
    """Declaring success on entering the tolerance reports the tolerance as a result.

    Every waypoint of every route came out at 0.42-0.45 m against a 0.45 m tolerance:
    precise-looking, and measuring nothing. The rule must wait until the distance grows
    again and record the minimum.
    """
    src = _src()
    assert "leaving" in src
    assert "best_for" in src
    assert "errors.append(best)" in src, (
        "the recorded error must be the closest approach, not the distance at entry")


def test_the_envelope_constants_come_from_the_grid():
    """The follower must stay inside the cells that repeated, not span the whole plane."""
    src = _src()
    assert re.search(r"^MIN_VX = 0\.15$", src, re.M), (
        "below 0.15 m/s the turning response was measured erratic")
    assert re.search(r"^MAX_VX = 0\.25$", src, re.M), (
        "0.30 straight was itself unstable at 0.044 m spread")


def test_the_envelope_size_is_recorded():
    """14 of 20 cells repeated; that number is the evidence for the constraints."""
    doc = _src(DOC)
    assert "14" in doc
    assert "20" in doc


def test_the_loop_rate_documented_matches_what_was_measured():
    """19.7 Hz was measured; a claim of 20 Hz must not drift away from it."""
    doc = _src(DOC)
    assert "19.7" in doc or "19.7 Hz" in doc


def test_the_command_clamp_is_bounded_by_the_measured_ceiling():
    """A yaw command above roughly 1.2 rad/s was measured to produce *less* rotation.

    An earlier version clamped at 2.5 in an attempt to turn faster and put every hard turn
    in the region where the plugin barely responds.
    """
    src = _src()
    m = re.search(r'--max-yaw",\s*type=float,\s*default=([0-9.]+|MAX_WZ)', src)
    assert m, "the yaw clamp must be an explicit argument"
    value = m.group(1)
    if value == "MAX_WZ":
        # Taken from the constant, which must itself be the measured ceiling.
        assert re.search(r"^MAX_WZ = 1\.20$", src, re.M), (
            "MAX_WZ must be the highest yaw command that repeats")
    else:
        assert float(value) <= 1.5, (
            "the clamp must stay at or below the measured usable ceiling")


def test_the_documented_limitations_match_the_code():
    """GAZEBO.md must agree with the code about /odom and about the pose services.

    Both halves of this have changed once already: the document used to say
    set_entity_state was absent, which was true until the gazebo_ros_state plugin was
    added to the world file, and then it was false while the code had already switched to
    the service.
    """
    doc = _src(DOC)
    assert "/odom" in doc
    assert "set_entity_state" in doc
    assert "1.95" in doc, "the measured odometry ratio must stay recorded"
    assert "not present" in doc.lower(), (
        "GAZEBO.md must record that the state plugin had to be added")


def test_routes_actually_run_report_their_failures():
    """A runner that hides a failed route is worse than one that has no routes."""
    src = _src()
    assert "failures += 1" in src
    assert "routes failed" in src


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
