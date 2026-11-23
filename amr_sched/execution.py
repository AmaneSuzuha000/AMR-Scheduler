"""Translate the scheduler's graph path into something an execution backend can run.

Why this layer exists
---------------------
The scheduler ``amr_sched`` speaks **graph semantics**: a node sequence such as
``["R1","N3","W5"]`` plus a traversal time for each leg. A SCOUT Mini in Gazebo
understands only ``/cmd_vel`` (linear and angular velocity), and a real chassis understands
nothing but velocity commands either. Three things sit between the two:

1. **Coordinates**: graph nodes are discrete (x, y) points; getting from one to the next
   means driving a straight line or a polyline.
2. **Heading**: a node has no orientation, but a robot has a yaw, and turning takes time.
3. **Speed**: the scheduler assumes each leg is covered in the shortest possible time, while
   a chassis accelerates, decelerates, and has a minimum turning radius.

This module handles item 1 and half of item 3 (path -> waypoints -> reference velocity
sequence), and leaves heading tracking to the execution backend (the Gazebo controller, or
the real-vehicle driver).

**What it does not do**: it produces no real trajectory, models no wheel-ground contact, and
makes no claim to have validated anything on a real vehicle. Those things can only come out
of Gazebo or out of the hardware itself.

Design principles
-----------------
* **Pure functions**: given a node sequence and a network, it returns a result, with no
  dependence on a clock or on IO. That is what lets correctness be pinned by unit tests on
  Windows, with no ROS installation anywhere in sight.
* **No information lost**: waypoints keep their source node and their expected cumulative
  time, so a planned arrival time can later be aligned, point by point, against the time the
  vehicle actually got there.
* **Impossible is rejected explicitly**: a broken path raises instead of quietly drawing a
  straight line -- which would make a demo look like it runs when it does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from .network import Network

#: Pose sources as the execution side sees them. Mirrors stream.PoseSource, but from the
#: execution backend's point of view.
EXEC_SOURCES = ("gazebo_sim", "scout_hardware")


@dataclass(frozen=True)
class DriveLimit:
    """Chassis kinematic limits. The defaults are conservative readings of the published
    SCOUT Mini specifications, not measurements."""

    v_max: float = 1.0      # max linear speed, m/s (nominal 1.5, with a safety margin)
    w_max: float = 1.5      # max angular speed, rad/s
    a_max: float = 0.5      # linear acceleration, m/s^2, used to estimate ramp time
    width_m: float = 0.65   # body width, m
    length_m: float = 0.80  # body length, m

    def __post_init__(self) -> None:
        for name in ("v_max", "w_max", "a_max", "width_m", "length_m"):
            v = getattr(self, name)
            if v <= 0:
                raise ValueError("%s must be positive, got %r" % (name, v))


@dataclass(frozen=True)
class Waypoint:
    """A single waypoint in metres."""

    x: float
    y: float
    node_id: str
    t_plan_s: float
    kind: str = "node"   # "node" is a graph node; "interp" is a smoothing insertion

    def to_dict(self) -> dict:
        return {
            "x": round(self.x, 4),
            "y": round(self.y, 4),
            "node_id": self.node_id,
            "t_plan_s": round(self.t_plan_s, 3),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class RoutePlan:
    """An executable route."""

    agv_id: str
    waypoints: tuple
    nodes: tuple
    t_ideal_s: float   # theoretical travel time: sum of length / speed limit per leg
    t_est_s: float     # estimate including one acceleration and one deceleration ramp

    @property
    def length_m(self) -> float:
        total = 0.0
        for a, b in zip(self.waypoints, self.waypoints[1:]):
            total += math.hypot(b.x - a.x, b.y - a.y)
        return total

    def to_dict(self) -> dict:
        return {
            "agv_id": self.agv_id,
            "nodes": list(self.nodes),
            "waypoints": [w.to_dict() for w in self.waypoints],
            "t_ideal_s": round(self.t_ideal_s, 3),
            "t_est_s": round(self.t_est_s, 3),
            "length_m": round(self.length_m, 3),
        }


def _seg_speed(net: Network, a: str, b: str, limit: DriveLimit) -> float:
    """Usable speed on the leg a->b: the smaller of the edge limit and the chassis cap."""
    if not net.has_edge(a, b):
        raise ValueError(
            "path is broken: no edge between %s -> %s. "
            "The node sequence the scheduler produced must be connected hop by hop; "
            "it cannot be patched up with a straight line." % (a, b))
    return min(float(net.edge(a, b).speed_limit), limit.v_max)

def _wrap(a: float) -> float:
    """Normalise an angle into (-pi, pi]."""
    while a > math.pi:
        a -= 2 * math.pi
    while a <= -math.pi:
        a += 2 * math.pi
    return a


def build_route(net: Network, nodes: Sequence, *,
                agv_id: str = "AGV", limit=None,
                interp_m: float = 0.0) -> RoutePlan:
    """Translate a graph node sequence into a waypoint route.

    Parameters
    ----------
    nodes:
        Graph path, at least two nodes. Repeated nodes are allowed, which is how a
        turn-around-in-place is expressed.
    interp_m:
        When greater than zero, interpolated points are inserted along each leg at this
        spacing, giving a controller such as pure pursuit a denser reference.

    Raises
    ------
    ValueError
        A node does not exist, or there is no edge between two consecutive hops.
    """
    if limit is None:
        limit = DriveLimit()
    if len(nodes) < 2:
        raise ValueError("a route needs at least two nodes, got %d" % len(nodes))
    if interp_m < 0:
        raise ValueError("interp_m cannot be negative")

    pts = []
    t_acc = 0.0
    speeds = []
    for i, nid in enumerate(nodes):
        if nid not in net:
            raise ValueError("path refers to a node that does not exist: %s" % nid)
        nd = net.node(nid)
        if i > 0:
            prev = nodes[i - 1]
            pnd = net.node(prev)
            spd = _seg_speed(net, prev, nid, limit)
            speeds.append(spd)
            seg_len = math.hypot(nd.x - pnd.x, nd.y - pnd.y)
            t_acc += seg_len / spd
            if interp_m > 0 and seg_len > interp_m:
                n_mid = int(seg_len // interp_m)
                for k in range(1, n_mid + 1):
                    f = k * interp_m / seg_len
                    if f >= 1.0:
                        break
                    pts.append(Waypoint(
                        x=pnd.x + (nd.x - pnd.x) * f,
                        y=pnd.y + (nd.y - pnd.y) * f,
                        node_id=prev,
                        t_plan_s=t_acc - (seg_len * (1 - f)) / spd,
                        kind="interp",
                    ))
        pts.append(Waypoint(x=nd.x, y=nd.y, node_id=nid,
                            t_plan_s=t_acc, kind="node"))

    # One acceleration at the start and one deceleration at the end, each taking half of
    # v_top / a_max, so together they add t_ramp.
    v_top = min(limit.v_max, max(speeds)) if speeds else limit.v_max
    t_est = t_acc + v_top / limit.a_max

    return RoutePlan(agv_id=agv_id, waypoints=tuple(pts),
                     nodes=tuple(nodes), t_ideal_s=t_acc, t_est_s=t_est)


def check_clearance(net: Network, nodes: Sequence, limit=None) -> list:
    """Check whether the chassis actually fits through the nodes the route passes.

    Why it is needed: the scheduler's capacity constraint counts vehicles, and knows nothing
    about how wide they are. A 1.0 m single lane passes a 0.65 m body; swap in a 1.2 m
    chassis and it does not. This check is what separates "the algorithm thinks it is
    feasible" from "it physically fits".

    Returns
    -------
    list: node ids that are too narrow; an empty list means everything passed.
    """
    if limit is None:
        limit = DriveLimit()
    bad = []
    for nid in dict.fromkeys(nodes):
        if nid not in net:
            bad.append(nid)
            continue
        nd = net.node(nid)
        # Convention: capacity <= 1 is treated as a narrow aisle or a single station, with a
        # clear width taken as 1.0 m; capacity >= 2 is treated as passable in parallel, at
        # 2.0 m. This is a conservative approximation. The real clear width has to come from
        # the workshop drawing, and nothing should be concluded on the strength of this.
        clear_w = 1.0 if nd.capacity <= 1 else 2.0
        if limit.width_m >= clear_w:
            bad.append(nid)
    return bad


def route_to_cmd_vel_sequence(plan: RoutePlan, *, limit=None) -> list:
    """Turn a waypoint sequence into a reference velocity sequence.

    These are not commands that can be sent to a vehicle. They are the expected linear and
    angular velocities, for a controller to be checked against. The angular term is a rough
    conversion of "heading change to the next point divided by the leg's travel time" and
    **includes no turning-radius constraint**, so it is a reference and nothing more. It must
    never be mistaken for a controller output.
    """
    if limit is None:
        limit = DriveLimit()
    wps = plan.waypoints
    if len(wps) < 2:
        return []
    out = []
    for i in range(len(wps) - 1):
        a, b = wps[i], wps[i + 1]
        dx, dy = b.x - a.x, b.y - a.y
        seg = math.hypot(dx, dy)
        heading = math.atan2(dy, dx)
        dt_seg = max(b.t_plan_s - a.t_plan_s, 1e-6)
        v = min(seg / dt_seg, limit.v_max)
        w = 0.0
        if i + 2 < len(wps):
            c = wps[i + 2]
            heading2 = math.atan2(c.y - b.y, c.x - b.x)
            w = max(-limit.w_max, min(limit.w_max,
                                     _wrap(heading2 - heading) / dt_seg))
        out.append({
            "x": round(a.x, 4), "y": round(a.y, 4),
            "heading": round(heading, 4),
            "v": round(v, 4), "w": round(w, 4),
        })
    return out
