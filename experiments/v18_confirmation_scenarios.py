"""Frozen inputs for the five-method wall-clock confirmation run."""

from dataclasses import replace

from amr_sched.network import Network, Node
from holdout_scenarios_v17 import build_holdout_v17_scenario


LAYOUTS = ("perimeter_bypass", "narrow_spine", "offset_dual_lane", "workshop_cells")
FIRST_FLOW_SEED = 14101
REGIMES = ("standard", "window_bottleneck", "execution_blockage")
BLOCKAGE = {
    "enable_travel_drift": False,
    "enable_dynamic_obstacles": True,
    "obstacle_count": 3,
    "obstacle_durations": (6, 38, 8),
    "obstacle_window_range": (12, 84),
    "est_clear_horizon_steps": 16,
    "enable_local_planner_fail": False,
    "enable_p4_belief": True,
    "enable_p4_voi_probe": True,
}


def build_confirmation_scenario(layout, seed, regime):
    source, tasks, agvs, ops, events = build_holdout_v17_scenario(
        "cross_wing_yard", seed, with_windows=regime == "window_bottleneck",
    )
    lane_changes = {
        "perimeter_bypass": (),
        "narrow_spine": (("BR_N1", "BR_N2"), ("BR_S1", "BR_S2")),
        "offset_dual_lane": (("MID_X", "EJ_N"), ("MID_X", "WJ_S")),
        "workshop_cells": (("BR_N1", "BR_N2"),),
    }
    omitted = {frozenset(lane) for lane in lane_changes[layout]}
    net = Network()
    for node in source.nodes.values():
        net.add_node(replace(node, capacity=1) if node.node_id == "MID_X" and layout == "narrow_spine" else node)
    for edge in source.edges.values():
        if frozenset((edge.src, edge.dst)) not in omitted:
            net.add_edge(edge)
    if layout == "perimeter_bypass":
        net.add_node(Node("OUTER_N", 0.0, 14.0, capacity=2))
        net.add_bidirectional("BAY_NW", "OUTER_N", 12.0, speed_limit=3.0)
        net.add_bidirectional("OUTER_N", "BAY_NE", 12.0, speed_limit=3.0)
    elif layout == "offset_dual_lane":
        net.add_bidirectional("BR_N1", "BR_S1", 11.0, speed_limit=2.0)
        net.add_bidirectional("BR_N2", "BR_S2", 11.0, speed_limit=2.0)
    elif layout == "workshop_cells":
        net.add_bidirectional("WJ_N", "WJ_S", 10.0, speed_limit=3.0, capacity=2)
        net.add_bidirectional("EJ_N", "EJ_S", 10.0, speed_limit=3.0, capacity=2)
        net.add_bidirectional("STG_W", "BAY_SW", 12.0, speed_limit=2.5)
        net.add_bidirectional("STG_E", "BAY_NE", 12.0, speed_limit=2.5)
    # The blocked regime uses seeded, exogenous obstacles. Avoid action-dependent native faults.
    return net, tasks, agvs, ops, dict(events, fault_step=999999, post_pickup_fault_step=999999)


def confirmation_specs(first_flow_seed=FIRST_FLOW_SEED):
    for layout_index, layout in enumerate(LAYOUTS):
        first_seed = first_flow_seed + 4 * layout_index
        for seed in range(first_seed, first_seed + 4):
            cluster = f"{layout}_{seed}"
            for regime in REGIMES:
                yield "CONFIRMATION", regime, [], {
                    "topo": "workshop", "confirmation_layout": layout, "seed": seed,
                    "episode_key": f"{cluster}_{regime}", "base_cluster": cluster,
                    "p2_dict": BLOCKAGE if regime == "execution_blockage" else None,
                }
