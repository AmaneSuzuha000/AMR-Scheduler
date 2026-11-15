#!/usr/bin/env python3
"""Export scheduler routes for the Gazebo follower.

Why this file exists:

The Gazebo runner consumed a routes.json that was typed by hand. That left the most
important claim of the whole chain untested -- "the vehicle drives the route the scheduler
planned" -- because the route in the vehicle was never the scheduler's route, it was a
copy of one. Any divergence between the two would have been invisible.

So the route file is now produced from the same objects the scheduler uses: nodes and edges
from amr_sched.network, paths from the same planner, scaled into Gazebo world coordinates.

Coordinate scaling is a single explicit constant rather than an implicit convention.
amr_sched uses a 40 m aisle pitch; the Gazebo world uses 1.6 m, so the factor is 1/25.
Encoding it here, once, means a mismatch shows up as a wrong number in a test rather than
as a vehicle quietly driving to the wrong place.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from amr_sched.network import build_demo_network          # noqa: E402

#: amr_sched metres per Gazebo metre. The scheduler models 40 m between aisle nodes; the
#: Gazebo world places them 1.6 m apart so the whole network fits in the space the vehicle
#: can actually be driven around in during a test.
SCALE = 1.0 / 25.0

#: Node that sits at the Gazebo world origin. Every other node is offset from it, so the
#: network lands where the vehicle starts rather than wherever the raw coordinates happen
#: to fall.
ORIGIN_NODE = "R0"


def scaled_positions(network):
    """Node id -> (x, y) in Gazebo world coordinates."""
    origin = network.nodes[ORIGIN_NODE]
    return {nid: ((node.x - origin.x) * SCALE, (node.y - origin.y) * SCALE)
            for nid, node in network.nodes.items()}


def path_waypoints(path, positions):
    """Turn a node sequence into the waypoints the follower drives through."""
    missing = [n for n in path if n not in positions]
    if missing:
        raise KeyError("path refers to nodes with no position: %s" % missing)
    return [list(positions[n]) for n in path]


def build_routes(network=None, specs=None):
    """Build the route document.

    ``specs`` is a list of (name, path) pairs; the path is a node sequence as the planner
    would emit it. When omitted, the routes are the demo network's own node sequences, so
    the exporter can be exercised without a simulation run.
    """
    network = network or build_demo_network()
    positions = scaled_positions(network)
    if specs is None:
        specs = default_specs()
    out = {}
    for name, path in specs:
        out[name] = {
            "nodes": list(path),
            "waypoints": path_waypoints(path, positions),
        }
    return out


def default_specs():
    """The four routes exercised in GAZEBO.md.

    They are written as node sequences, not coordinates, precisely so that the geometry
    comes from the network rather than from this list.
    """
    return [
        ("D->W3", ["D", "R0", "R1", "R2", "W3"]),
        ("D->S", ["D", "R0", "R1", "R2", "S"]),
        ("W1->W5", ["W1", "R0", "R1", "R2", "R5", "W5"]),
        ("R0->B1", ["R0", "R1", "R4", "B1"]),
    ]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="routes.json")
    args = ap.parse_args(argv)

    routes = build_routes()
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(routes, f, indent=1)

    print("wrote %s: %d routes" % (args.out, len(routes)))
    for name, spec in routes.items():
        print("  %-8s %-26s %d waypoints"
              % (name, " -> ".join(spec["nodes"]), len(spec["waypoints"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
