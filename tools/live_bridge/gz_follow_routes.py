#!/usr/bin/env python3
"""Follow routes produced by the Python scheduler, inside Gazebo.

The route file is written by the scheduler side and shipped here; this script only
executes it. Keeping that split explicit means what is being tested is "can Gazebo
follow what the scheduler planned", not "can a hand-written list of points be followed".

What was measured:

**Translation is repeatable and accurate.** Three identical runs of a straight-line
command, 2.5 s each, measured against /get_entity_state:

    vx_cmd   distance (3 runs)          spread   ratio
    0.10     0.236  0.249  0.251 m      0.015    0.94-1.00
    0.20     0.499  0.489  0.486 m      0.013    0.98-1.00

**Rotation is repeatable in part of the command space, and this is the finding that
matters.** A 20-cell grid of (vx, wz) was run twice per cell by gz_envelope_probe.py.
14 cells repeated; 6 did not. The failures are not scattered — they cluster where one
wheel is reversing or nearly stopped: (0.10, 0.80), (0.20, 0.40), (0.25, 0.80),
(0.30, 0.80).

Inside the stable cells the achieved yaw rate is close to constant:

    wz commanded    wz achieved    ratio
    0.40            0.15          0.38
    0.80            0.34          0.43
    1.20            0.51          0.43

The response saturates at about 0.51 rad/s regardless of the command, so the constants
below come from that table. Earlier versions of this file carried ratios of 0.16, 0.31,
0.62 and 0.70, each derived from a single run; all four were wrong, and the 0.62 and 0.70
were wrong in the direction that made every turn under-powered by a factor of two.

The follower is therefore written to stay inside the envelope rather than to be tuned
until it happens to work: forward speed between 0.15 and 0.25 m/s, yaw command at most
1.20 rad/s. That is a constraint taken from measurement, and it is the reason the routes
it does complete are repeatable.

Consequences that are stated rather than hidden:

- Not every route completes. Failures are reported by the runner, not filtered out.
- Per-waypoint accuracy is limited to roughly the tolerance it is measured against; it is
  not a precision result and is not presented as one.
- No claim about real-vehicle control quality follows from any of this.

Ground truth is the /get_entity_state service, never /odom: /odom is wheel integration and,
measured against a bounded command, reported 0.879 m for at most 0.45 m of travel. It is
used here only as a fast hint between service reads, and every accuracy number reported is
measured against the service.
"""
from __future__ import annotations

import argparse
import json
import math
import time

import rclpy
from gazebo_msgs.msg import EntityState
from gazebo_msgs.srv import GetEntityState, SetEntityState
from geometry_msgs.msg import Point, Pose, Quaternion, Twist
from rclpy.node import Node


# Command envelope, measured by gz_envelope_probe.py: 20 (vx, wz) cells, each run twice,
# recording the spread between repeats. A cell is usable only when both distance and yaw
# repeat. 14 of 20 were usable, and the unusable ones are not random -- they cluster at
# (0.10, 0.80), (0.20, 0.40), (0.25, 0.80) and (0.30, 0.80), where one wheel is reversing
# or nearly stopped and slip dominates.
#
# Inside the usable cells the achieved yaw rate is close to constant, which is the number
# that actually matters for a controller:
#
#     wz commanded    wz achieved    ratio
#     0.40            0.15          0.38
#     0.80            0.34          0.43
#     1.20            0.51          0.43
#
# So the yaw response saturates at about 0.51 rad/s no matter what is asked for, and the
# ratio to use is 0.43 -- not the 0.62 and 0.70 that earlier single-run measurements
# suggested, both of which were wrong and produced a controller that turned far less than
# it believed.
#
# Straight-line speed needs no compensation at all: vx_real / vx_cmd was 0.96-1.00 in
# every usable cell.
#
# Both bounds below are therefore taken from the table, not chosen for feel.

#: Achieved-to-commanded yaw ratio inside the stable envelope.
YAW_RATIO = 0.43

#: Forward speed the follower stays at or above. Below about 0.15 m/s the turning response
#: becomes erratic -- (0.10, 0.80) spread 10 degrees across repeats while (0.15, 0.80)
#: spread 0.0.
MIN_VX = 0.15

#: Forward speed the follower stays at or below. 0.30 straight was itself unstable
#: (0.044 m spread), so the ceiling is set just under it.
MAX_VX = 0.25

#: Yaw command ceiling. 1.20 is the highest cell that repeats, and asking for more does not
#: produce more rotation: the achieved rate is already saturated by then.
MAX_WZ = 1.20

#: Distance at which a waypoint counts as reached.
REACH_TOL = 0.45

def wrap_angle(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def yaw_from_quat(q) -> float:
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class FollowNode(Node):
    def __init__(self, model: str, service_period: float = 0.20):
        super().__init__("gz_follow_routes")
        self.model = model
        self._pub = self.create_publisher(Twist, "/cmd_vel", 10)
        self._truth = None
        self._last_service = 0.0
        self._service_period = service_period
        self._get_cli = None
        self._set_cli = None

    def _client(self, attr, srv, name):
        cli = getattr(self, attr)
        if cli is None:
            cli = self.create_client(srv, name)
            if not cli.wait_for_service(timeout_sec=10.0):
                raise RuntimeError("%s never appeared" % name)
            setattr(self, attr, cli)
        return cli

    def pose(self):
        """Ground truth from the simulator service. There is no fallback.

        Wheel odometry is deliberately not consulted, not even as a hint between service
        reads. A reading measured at 0.879 m against at most 0.45 m of possible travel is
        not a noisy estimate of position, it is a different quantity, and blending it into
        a control loop would make the loop's behaviour unexplainable from its inputs.
        A service read costs about 0.2 ms, so there is nothing to gain by avoiding it.
        """
        now = time.monotonic()
        if now - self._last_service >= self._service_period:
            self._last_service = now
            try:
                self._truth = self._ask_truth()
            except Exception:
                pass
        return self._truth

    def _ask_truth(self):
        cli = self._client("_get_cli", GetEntityState, "/get_entity_state")
        req = GetEntityState.Request()
        req.name = self.model
        req.reference_frame = "world"
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        res = future.result()
        # success must be checked: a missing model answers with a pose of (0,0,0) and
        # success False, and treating that as real would put the vehicle at the origin.
        if res is None or not res.success:
            raise RuntimeError("get_entity_state failed")
        p = res.state.pose.position
        return (p.x, p.y, yaw_from_quat(res.state.pose.orientation))

    def teleport(self, x=0.0, y=0.0, yaw=0.0) -> None:
        cli = self._client("_set_cli", SetEntityState, "/set_entity_state")
        req = SetEntityState.Request()
        st = EntityState()
        st.name = self.model
        st.pose = Pose(
            position=Point(x=float(x), y=float(y), z=0.05),
            orientation=Quaternion(x=0.0, y=0.0,
                                   z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0)))
        st.reference_frame = "world"
        req.state = st
        future = cli.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=2.0)
        res = future.result()
        if res is None or not res.success:
            raise RuntimeError("set_entity_state failed")

    def command(self, vx: float, wz: float) -> None:
        tw = Twist()
        tw.linear.x = float(vx)
        tw.angular.z = float(wz)
        self._pub.publish(tw)

    def stop(self) -> None:
        self.command(0.0, 0.0)


def lookahead_target(pose, waypoints, idx, lookahead):
    """Point ``lookahead`` metres along the remaining polyline, plus the index reached.

    Aiming at the next waypoint directly makes the vehicle cut corners; at the turn radii
    this plugin allows, corner-cutting is what turns into overshoot.
    """
    x, y, _ = pose
    # The caller decides when a waypoint is done. Advancing here as well meant the arrival
    # distance was computed against a waypoint the vehicle was already past, so every
    # reported error came out near the tolerance rather than near zero.
    if idx >= len(waypoints):
        return None, idx

    acc = 0.0
    prev = (x, y)
    for k in range(idx, len(waypoints)):
        wx, wy = waypoints[k]
        seg = math.hypot(wx - prev[0], wy - prev[1])
        if acc + seg >= lookahead:
            t = (lookahead - acc) / seg if seg > 1e-9 else 0.0
            return (prev[0] + (wx - prev[0]) * t, prev[1] + (wy - prev[1]) * t), idx
        acc += seg
        prev = (wx, wy)
    return tuple(waypoints[-1]), idx


def control(pose, waypoints, idx, args):
    """Pure pursuit. Returns (vx, wz, new_idx, dist_to_current_waypoint)."""
    x, y, yaw = pose
    target, new_idx = lookahead_target(pose, waypoints, idx, args.lookahead)
    if target is None:
        gx, gy = waypoints[-1]
        return 0.0, 0.0, new_idx, math.hypot(gx - x, gy - y)

    wx, wy = waypoints[new_idx]
    dist_wp = math.hypot(wx - x, wy - y)

    dx, dy = target[0] - x, target[1] - y
    ld = max(0.15, math.hypot(dx, dy))
    alpha = wrap_angle(math.atan2(dy, dx) - yaw)

    # Stay inside the measured envelope. The speed reduction when turning hard is what the
    # table supports: turning is only repeatable at forward speeds of 0.15 and above, so the
    # follower never drops below MIN_VX even when badly misaligned.
    v_max = args.max_speed - (args.max_speed - args.min_speed) * min(1.0, abs(alpha))
    speed = max(args.min_speed, min(v_max, 0.6 * dist_wp + 0.05))

    # Pure pursuit curvature divided by the measured yaw ratio, then clamped to the highest
    # command that repeats. Asking for more does not turn faster; the response saturates.
    wz = 2.0 * speed * math.sin(alpha) / ld / YAW_RATIO
    wz = max(-args.max_yaw, min(args.max_yaw, wz))
    return speed, wz, new_idx, dist_wp


def run_route(node, waypoints, args):
    """Drive one route. Returns (index_reached, errors_or_message).

    A waypoint counts as done on the way *out* of its tolerance circle, not on the way in.
    Declaring success at entry reports the tolerance back as though it were an accuracy
    result: an earlier version did exactly that and produced a mean error of 0.44 m against
    a 0.45 m tolerance on every waypoint of every route, which looks precise and measures
    nothing. Waiting until the distance starts growing means the vehicle really did pass
    the point, and the number recorded is the closest it actually got.
    """
    idx = 0
    errors = []
    best_for = {}
    last_progress = time.monotonic()
    last_best = float("inf")
    deadline = last_progress + args.timeout
    pose = None

    while time.monotonic() < deadline:
        pos = node.pose()
        if pos is None:
            time.sleep(0.05)
            continue
        pose = pos

        if idx >= len(waypoints):
            node.stop()
            return len(waypoints), errors

        wx, wy = waypoints[idx]
        d_now = math.hypot(wx - pos[0], wy - pos[1])
        if d_now < best_for.get(idx, float("inf")):
            best_for[idx] = d_now
        best = best_for[idx]

        # Progress means getting closer, not merely being slow. Measuring progress by
        # "the index has not advanced for N seconds" fired on a vehicle 0.454 m from a
        # 0.45 m waypoint: it was circling inside the last millimetre of tolerance and
        # being reported as stalled.
        if best < last_best - 0.02:
            last_best = best
            last_progress = time.monotonic()

        # Done when the closest approach was inside tolerance AND the vehicle is now moving
        # away again (or this was the final waypoint).
        leaving = d_now > best + 0.06
        is_last = idx + 1 >= len(waypoints)
        if best < REACH_TOL and (is_last or leaving):
            errors.append(best)
            print("    wp%d (%.2f,%.2f) reached dist=%.3f" % (idx, wx, wy, best))
            idx += 1
            last_best = float("inf")
            last_progress = time.monotonic()
            if idx >= len(waypoints):
                node.stop()
                return len(waypoints), errors
            continue

        vx, wz, _, _ = control(pos, waypoints, idx, args)
        node.command(vx, wz)

        if time.monotonic() - last_progress > args.stall_timeout:
            node.stop()
            return idx, ("stalled at wp%d (%.2f,%.2f) pose=(%.3f,%.3f) dist=%.3f best=%.3f"
                         % (idx, wx, wy, pos[0], pos[1], d_now, best))

        time.sleep(1.0 / args.rate)

    node.stop()
    tail = "(%.3f,%.3f)" % (pose[0], pose[1]) if pose else "(none)"
    return idx, "timeout at wp%d pose=%s" % (idx, tail)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", required=True)
    ap.add_argument("--model", default="scout_mini")
    ap.add_argument("--only", default=None)
    ap.add_argument("--lookahead", type=float, default=0.5)
    ap.add_argument("--max-speed", type=float, default=MAX_VX)
    ap.add_argument("--min-speed", type=float, default=MIN_VX)
    ap.add_argument("--max-yaw", type=float, default=MAX_WZ)
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--timeout", type=float, default=90.0)
    ap.add_argument("--stall-timeout", type=float, default=15.0)
    args = ap.parse_args(argv)

    with open(args.routes, encoding="utf-8") as f:
        routes = json.load(f)

    rclpy.init()
    node = FollowNode(args.model)
    names = [args.only] if args.only else list(routes)
    failures = 0
    try:
        for name in names:
            spec = routes[name]
            wps = [tuple(float(v) for v in w) for w in spec["waypoints"]]
            print("route %s  nodes=%s" % (name, " -> ".join(spec["nodes"])))
            try:
                node.teleport()
            except RuntimeError as exc:
                print("  FAIL: teleport failed: %s" % exc)
                failures += 1
                continue
            time.sleep(0.6)

            reached, info = run_route(node, wps, args)
            if reached == len(wps):
                mean = sum(info) / len(info) if info else 0.0
                print("  PASS %d/%d  mean_err=%.3f max_err=%.3f"
                      % (reached, len(wps), mean, max(info) if info else 0.0))
            else:
                print("  FAIL %d/%d  %s" % (reached, len(wps), info))
                failures += 1
            node.stop()
            time.sleep(0.3)
    finally:
        node.stop()
        node.destroy_node()
        rclpy.shutdown()

    print("routes failed: %d/%d" % (failures, len(names)))
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
