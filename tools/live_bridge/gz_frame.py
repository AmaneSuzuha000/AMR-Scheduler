"""Workshop network  <->  Gazebo world coordinates.

Why a single explicit transform instead of ad-hoc numbers
--------------------------------------------------------
The scheduler's network spans roughly 125 m x 80 m (R0 at (0,0) to S at (105,20),
D at (-25,20)). The Gazebo world used for the motion demo is an unbounded ground
plane. Any mapping between the two is therefore a *choice*, and a choice that is
made in two places is a bug waiting to happen: the bridge would drive to one point
while the map drew another, and the discrepancy would look like a controller fault.

So the mapping lives here, is invertible, and is asserted round-trip in the tests.

What the transform is
---------------------
    world = origin + scale * rotate(network - network_origin)

scale defaults to 1/25, which turns the 125 m network into a 5 m Gazebo area.
That keeps the demo inside a region where a 0.5 m/s robot crosses the map in a few
seconds while still exercising real turning. It is a display/demo choice, not a
claim about the real workshop: nothing here says the workshop is 5 m wide.

Orientation is preserved (no rotation by default). If a rotation is ever needed it
must be added here with a matching test, not bolted on at a call site.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def _wrap_angle(a: float) -> float:
    """Normalise an angle into (-pi, pi]."""
    while a > math.pi:
        a -= 2.0 * math.pi
    while a <= -math.pi:
        a += 2.0 * math.pi
    return a


@dataclass(frozen=True)
class GzFrame:
    """An invertible affine map between network and Gazebo world coordinates."""

    #: Network coordinate that maps to "origin" in the world.
    network_origin: tuple[float, float] = (0.0, 0.0)
    #: World coordinate that "network_origin" maps to.
    origin: tuple[float, float] = (0.0, 0.0)
    #: Metres of world per metre of network. Must be > 0.
    scale: float = 1.0 / 25.0
    #: Rotation in radians, applied about "network_origin".
    rotation: float = 0.0

    def __post_init__(self) -> None:
        if not (self.scale > 0.0):
            raise ValueError("scale must be positive, got %r" % (self.scale,))

    def to_world(self, x: float, y: float) -> tuple[float, float]:
        """Map a network point into world coordinates."""
        dx = (x - self.network_origin[0]) * self.scale
        dy = (y - self.network_origin[1]) * self.scale
        if self.rotation:
            c, s = math.cos(self.rotation), math.sin(self.rotation)
            dx, dy = c * dx - s * dy, s * dx + c * dy
        return (self.origin[0] + dx, self.origin[1] + dy)

    def to_network(self, x: float, y: float) -> tuple[float, float]:
        """Inverse of :meth:`to_world`."""
        dx = x - self.origin[0]
        dy = y - self.origin[1]
        if self.rotation:
            c, s = math.cos(self.rotation), math.sin(self.rotation)
            dx, dy = c * dx + s * dy, -s * dx + c * dy
        return (self.network_origin[0] + dx / self.scale,
                self.network_origin[1] + dy / self.scale)

    def to_network_yaw(self, yaw: float) -> float:
        """Map a heading in world coordinates back into the network frame.

        The position is not the only thing the transform has to undo: a rotation part left
        un-applied here would put the vehicle at the right point facing the wrong way, and
        the arrow would disagree with the path it is following. Written as the exact inverse
        of :meth:`to_world`'s rotation so the two cannot drift apart.
        """
        return _wrap_angle(yaw - self.rotation)

    def path_to_world(self, nodes, network) -> list[tuple[float, float]]:
        """Turn a list of node ids into world waypoints, in order."""
        out = []
        for node_id in nodes:
            node = network.node(node_id)
            out.append(self.to_world(node.x, node.y))
        return out


#: Frame used by the motion demo: R0 (0,0) sits at the world origin, 25 network
#: metres become one world metre.
DEMO_FRAME = GzFrame(network_origin=(0.0, 0.0), origin=(0.0, 0.0), scale=1.0 / 25.0)
