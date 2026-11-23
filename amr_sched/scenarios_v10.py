"""V10 Unseen Shop-Floor Topologies, 5-Axis Stress Suite & Causal Opening Features.

Provides:
  1. Two genuinely unseen layouts (never present in training checkpoints V4..V9):
     - `unseen_ladder`: 28-node asymmetric 2-rail ladder spine with single-lane cross-rungs
     - `unseen_choke_yard`: 32-node West/East dual-yard connected via narrow bridge + long bypass
  2. 5-Axis parameterized stress generator (`build_v10_stress_scenario`):
     - `standard`: canonical V9/V10 difficulty aligned to seen suites on new graphs
     - `converging_dag`: multi-branch converging assembly DAG under crew/tool contention
     - `multi_kitting`: 3-4 kit tasks per operation dispersed across opposite zones
     - `dual_fault`: two overlapping AGV faults cascading during active kitting windows
     - `flow_inversion`: mid-episode wave reversing spatial source->sink traffic flow
     - `fleet_density`: configurable 4-AGV (starved) or 8-AGV (congestion-gridlocked) fleet
  3. Strictly causal t=0 opening structural feature extractor (`extract_opening_structural_features`)
     for out-of-distribution layout-agnostic selector generalization.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.snapshot import NativeSnapshot, OperationSnapshot, op_predecessors
from amr_sched.entities import Agv, Task, TaskPriority
from amr_sched.network import Edge, Network, Node, NodeKind


def _stable_axis_offset(stress_axis: str) -> int:
    """Process-independent deterministic seed offset replacing Python randomized hash()."""
    dig = hashlib.sha256(str(stress_axis).encode("utf-8")).digest()
    return int.from_bytes(dig[:4], "little") % 997

UNSEEN_LAYOUT_NAMES = ("unseen_ladder", "unseen_choke_yard")
STRESS_AXES = (
    "standard",
    "converging_dag",
    "multi_kitting",
    "dual_fault",
    "flow_inversion",
    "fleet_density",
)

OPENING_FEATURE_NAMES = (
    "log_nodes",
    "edge_node_ratio",
    "mean_edge_cap",
    "frac_single_lane",
    "mean_out_degree",
    "tt_mean_norm",
    "tt_p90_norm",
    "tasks_per_agv",
    "ops_per_agv",
    "mean_kit_size",
    "initial_locked_ratio",
    "bay_flex_mean",
    "crew_flex_mean",
    "shared_tool_frac",
    "pred_dag_density",
    "mean_due_slack_norm",
)
OPENING_FEATURE_DIM = len(OPENING_FEATURE_NAMES)


def _add_bi(net: Network, a: str, b: str, length: float, speed: float = 3.0, cap: int = 1) -> None:
    """Add bidirectional edges cleanly if both endpoints exist."""
    net.add_bidirectional(a, b, length=float(length), speed_limit=float(speed), capacity=int(cap))


def build_unseen_ladder_network(aisle_capacity: int = 1, speed: float = 3.0) -> Network:
    """Build 28-node 2-rail asymmetric ladder topology with single-lane cross-rungs."""
    net = Network()
    # Two horizontal rails: LN_0..LN_5 (y=12) and LS_0..LS_5 (y=0)
    for i in range(6):
        x = 12.0 + i * 18.0
        net.add_node(Node(f"LN_{i}", x, 12.0, NodeKind.AISLE, capacity=2))
        net.add_node(Node(f"LS_{i}", x, 0.0, NodeKind.AISLE, capacity=2))

    # Disassembly docks on West, Assembly on East
    net.add_node(Node("DL_W1", 0.0, 15.0, NodeKind.DISASSEMBLY, capacity=2))
    net.add_node(Node("DL_W2", 0.0, -3.0, NodeKind.DISASSEMBLY, capacity=2))
    net.add_node(Node("AS_E1", 114.0, 15.0, NodeKind.ASSEMBLY, capacity=2))
    net.add_node(Node("AS_E2", 114.0, -3.0, NodeKind.ASSEMBLY, capacity=2))

    # Buffers at outer/middle bays
    for bid, x, y in (
        ("BUF_1", 12.0, 22.0),
        ("BUF_2", 12.0, -10.0),
        ("BUF_3", 66.0, 22.0),
        ("BUF_4", 66.0, -10.0),
    ):
        net.add_node(Node(bid, x, y, NodeKind.BUFFER, capacity=2))

    # 8 maintenance bays along upper & lower alcoves
    bay_specs = (
        ("W1", 30.0, 22.0, "LN_1"),
        ("W2", 30.0, -10.0, "LS_1"),
        ("W3", 48.0, 22.0, "LN_2"),
        ("W4", 48.0, -10.0, "LS_2"),
        ("W5", 84.0, 22.0, "LN_4"),
        ("W6", 84.0, -10.0, "LS_4"),
        ("W7", 102.0, 22.0, "LN_5"),
        ("W8", 102.0, -10.0, "LS_5"),
    )
    for wid, x, y, parent in bay_specs:
        net.add_node(Node(wid, x, y, NodeKind.WORKSTATION, capacity=1))
        _add_bi(net, parent, wid, 10.0, speed=speed, cap=max(1, aisle_capacity))

    # Connect horizontal rail backbone (upper slightly faster/longer than lower on central span)
    for i in range(5):
        _add_bi(net, f"LN_{i}", f"LN_{i + 1}", 18.0 + (i % 2) * 3.0, speed=speed, cap=2)
        _add_bi(net, f"LS_{i}", f"LS_{i + 1}", 18.0 + ((i + 1) % 2) * 3.0, speed=speed, cap=2)

    # Single-lane cross-rungs creating head-on congestion vs detour trade-offs
    for r_idx in range(6):
        cap_rung = 1 if r_idx in (1, 2, 3, 4) else max(1, aisle_capacity)
        _add_bi(net, f"LN_{r_idx}", f"LS_{r_idx}", 12.0 + (r_idx % 3) * 3.0, speed=speed, cap=cap_rung)

    _add_bi(net, "DL_W1", "LN_0", 12.0, speed=speed, cap=2)
    _add_bi(net, "DL_W2", "LS_0", 12.0, speed=speed, cap=2)
    _add_bi(net, "AS_E1", "LN_5", 12.0, speed=speed, cap=2)
    _add_bi(net, "AS_E2", "LS_5", 12.0, speed=speed, cap=2)
    _add_bi(net, "BUF_1", "LN_0", 10.0, speed=speed, cap=2)
    _add_bi(net, "BUF_2", "LS_0", 10.0, speed=speed, cap=2)
    _add_bi(net, "BUF_3", "LN_3", 10.0, speed=speed, cap=2)
    _add_bi(net, "BUF_4", "LS_3", 10.0, speed=speed, cap=2)
    return net


def build_unseen_choke_yard_network(aisle_capacity: int = 1, speed: float = 3.0) -> Network:
    """Build 32-node West/East dual-yard topology joined by a 1-lane choke bridge & long northern bypass."""
    net = Network()
    # West yard aisle 2x3 mesh
    for i in range(3):
        x = 12.0 + i * 16.0
        net.add_node(Node(f"WA_N{i}", x, 12.0, NodeKind.AISLE, capacity=2))
        net.add_node(Node(f"WA_S{i}", x, -12.0, NodeKind.AISLE, capacity=2))
    for i in range(3):
        _add_bi(net, f"WA_N{i}", f"WA_S{i}", 24.0, speed=speed, cap=max(1, aisle_capacity))
        if i < 2:
            _add_bi(net, f"WA_N{i}", f"WA_N{i + 1}", 16.0, speed=speed, cap=2)
            _add_bi(net, f"WA_S{i}", f"WA_S{i + 1}", 16.0, speed=speed, cap=2)

    # Central choke bridge + long northern bypass
    for nid, x, y in (
        ("CH_W", 56.0, 0.0),
        ("CH_M", 74.0, 0.0),
        ("CH_E", 92.0, 0.0),
        ("BYP_1", 65.0, 26.0),
        ("BYP_2", 83.0, 26.0),
    ):
        net.add_node(Node(nid, x, y, NodeKind.AISLE, capacity=1 if nid.startswith("CH_") else 2))

    # Short 1-lane central bridge (36m total) vs long 2-lane bypass (58m total)
    _add_bi(net, "WA_N2", "CH_W", 15.0, speed=speed, cap=2)
    _add_bi(net, "WA_S2", "CH_W", 15.0, speed=speed, cap=2)
    _add_bi(net, "CH_W", "CH_M", 18.0, speed=speed, cap=1)
    _add_bi(net, "CH_M", "CH_E", 18.0, speed=speed, cap=1)
    _add_bi(net, "WA_N2", "BYP_1", 21.0, speed=speed, cap=2)
    _add_bi(net, "BYP_1", "BYP_2", 18.0, speed=speed, cap=2)

    # East yard aisle 2x3 mesh
    for i in range(3):
        x = 104.0 + i * 16.0
        net.add_node(Node(f"EA_N{i}", x, 12.0, NodeKind.AISLE, capacity=2))
        net.add_node(Node(f"EA_S{i}", x, -12.0, NodeKind.AISLE, capacity=2))
    for i in range(3):
        _add_bi(net, f"EA_N{i}", f"EA_S{i}", 24.0, speed=speed, cap=max(1, aisle_capacity))
        if i < 2:
            _add_bi(net, f"EA_N{i}", f"EA_N{i + 1}", 16.0, speed=speed, cap=2)
            _add_bi(net, f"EA_S{i}", f"EA_S{i + 1}", 16.0, speed=speed, cap=2)
    _add_bi(net, "BYP_2", "EA_N0", 21.0, speed=speed, cap=2)
    _add_bi(net, "CH_E", "EA_N0", 15.0, speed=speed, cap=2)
    _add_bi(net, "CH_E", "EA_S0", 15.0, speed=speed, cap=2)

    # Docks, Buffers, Workstations split West vs East (forces cross-bridge material kitting!)
    net.add_node(Node("DL_W1", 0.0, 12.0, NodeKind.DISASSEMBLY, capacity=2))
    net.add_node(Node("DL_W2", 0.0, -12.0, NodeKind.DISASSEMBLY, capacity=2))
    _add_bi(net, "DL_W1", "WA_N0", 12.0, speed=speed, cap=2)
    _add_bi(net, "DL_W2", "WA_S0", 12.0, speed=speed, cap=2)

    net.add_node(Node("AS_E1", 148.0, 12.0, NodeKind.ASSEMBLY, capacity=2))
    net.add_node(Node("AS_E2", 148.0, -12.0, NodeKind.ASSEMBLY, capacity=2))
    _add_bi(net, "AS_E1", "EA_N2", 12.0, speed=speed, cap=2)
    _add_bi(net, "AS_E2", "EA_S2", 12.0, speed=speed, cap=2)

    for wid, x, y, parent in (
        ("W1", 12.0, 22.0, "WA_N0"),
        ("W2", 28.0, -22.0, "WA_S1"),
        ("W3", 44.0, -22.0, "WA_S2"),
        ("W4", 104.0, -22.0, "EA_S0"),
        ("W5", 120.0, 22.0, "EA_N1"),
        ("W6", 136.0, -22.0, "EA_S2"),
    ):
        net.add_node(Node(wid, x, y, NodeKind.WORKSTATION, capacity=1))
        _add_bi(net, parent, wid, 10.0, speed=speed, cap=max(1, aisle_capacity))

    for bid, x, y, parent in (
        ("WBUF1", 28.0, 22.0, "WA_N1"),
        ("EBUF1", 120.0, -22.0, "EA_S1"),
        ("EBUF2", 136.0, 22.0, "EA_N2"),
    ):
        net.add_node(Node(bid, x, y, NodeKind.BUFFER, capacity=2))
        _add_bi(net, parent, bid, 10.0, speed=speed, cap=2)

    return net


def get_unseen_layout(name: str, aisle_capacity: int = 1) -> Network:
    """Build one of the V10 genuinely unseen networks by name."""
    if name == "unseen_ladder":
        return build_unseen_ladder_network(aisle_capacity=aisle_capacity)
    if name == "unseen_choke_yard":
        return build_unseen_choke_yard_network(aisle_capacity=aisle_capacity)
    raise ValueError(f"Unknown unseen layout {name!r}; expected one of {UNSEEN_LAYOUT_NAMES}")


def build_v10_stress_scenario(
    topology: str,
    seed: int,
    stress_axis: str = "standard",
    n_agvs_override: Optional[int] = None,
    n_ops: int = 8,
) -> tuple[Network, list[Task], list[Agv], list[OperationSnapshot], dict[str, Any]]:
    """Build a deterministic unseen-layout and/or 5-axis parameterized stress scenario."""
    from amr_sched.rl.env import _tune_network, build_scenario_for_env

    rng = random.Random(int(seed) * 1009 + _stable_axis_offset(stress_axis))
    if topology in UNSEEN_LAYOUT_NAMES:
        net = get_unseen_layout(topology, aisle_capacity=1)
        src_nodes = sorted(
            n.node_id for n in net.nodes.values()
            if n.kind in (NodeKind.DISASSEMBLY, NodeKind.BUFFER)
        )
        bay_nodes = sorted(
            n.node_id for n in net.nodes.values()
            if n.kind in (NodeKind.WORKSTATION, NodeKind.ASSEMBLY)
        )
    elif stress_axis == "standard" and n_agvs_override is None:
        return build_scenario_for_env(topology, seed, scenario_version="v3")
    else:
        base_net, _, _, _, _ = build_scenario_for_env(topology, seed, scenario_version="v3")
        net = _tune_network(base_net, speed=3.0)
        src_nodes = sorted(
            n.node_id for n in net.nodes.values()
            if n.kind in (NodeKind.DISASSEMBLY, NodeKind.BUFFER)
        ) or sorted(net.nodes.keys())
        bay_nodes = sorted(
            n.node_id for n in net.nodes.values()
            if n.kind in (NodeKind.WORKSTATION, NodeKind.ASSEMBLY)
        ) or sorted(net.nodes.keys())

    all_nodes = sorted(net.nodes.keys())
    west_sources = [n for n in src_nodes if net.node(n).x <= 55.0] or src_nodes
    east_sources = [n for n in (src_nodes + bay_nodes) if net.node(n).x >= 80.0] or all_nodes
    west_bays = [n for n in bay_nodes if net.node(n).x <= 55.0] or bay_nodes
    east_bays = [n for n in bay_nodes if net.node(n).x >= 80.0] or bay_nodes

    kit_parts = 3 if stress_axis == "multi_kitting" else 2
    n_agvs = int(n_agvs_override) if n_agvs_override is not None else (4 if stress_axis == "fleet_density" and seed % 2 == 1 else (8 if stress_axis == "fleet_density" else 6))

    ops: list[OperationSnapshot] = []
    tasks: list[Task] = []
    for o_idx in range(n_ops):
        op_id = f"OP_{o_idx + 1}"
        # Precedence DAG structure (true multi-predecessor convergence on converging_dag)
        preds_tuple: tuple[str, ...] = ()
        if stress_axis == "converging_dag":
            # OP_1, OP_2 -> OP_5; OP_3, OP_4 -> OP_6; OP_5 -> OP_7; OP_6 -> OP_8
            if o_idx == 4:
                preds_tuple = ("OP_1", "OP_2")
            elif o_idx == 5:
                preds_tuple = ("OP_3", "OP_4")
            elif o_idx == 6:
                preds_tuple = ("OP_5",)
            elif o_idx == 7:
                preds_tuple = ("OP_6",)
        else:
            if o_idx % 4 != 0:
                preds_tuple = (f"OP_{o_idx}",)
        pred = preds_tuple[0] if preds_tuple else None

        if stress_axis == "flow_inversion":
            if o_idx < n_ops // 2:
                bay = rng.choice(east_bays)
                p_pool = west_sources
            else:
                bay = rng.choice(west_bays)
                p_pool = east_sources
        else:
            bay = bay_nodes[o_idx % len(bay_nodes)] if stress_axis == "standard" else rng.choice(bay_nodes)
            p_pool = [n for n in src_nodes if n != bay] or [n for n in all_nodes if n != bay]

        alt_bays = [b for b in bay_nodes if b != bay]
        alt_bay = rng.choice(alt_bays) if alt_bays else bay
        rel = (o_idx // 2) * 5 + rng.randint(0, 3)
        op_due = 135 + o_idx * 17 + rng.randint(0, 10)
        part_ids: list[str] = []
        n_parts = 4 if (stress_axis == "multi_kitting" and o_idx % 2 == 1) else kit_parts
        for k_i in range(n_parts):
            tid = f"T_{o_idx + 1}_{k_i + 1}"
            pick_cand = [n for n in p_pool if n != bay] or [n for n in all_nodes if n != bay]
            src = rng.choice(pick_cand)
            t_due = 112 + o_idx * 15 + k_i * 3 + rng.randint(0, 8)
            t_obj = Task(tid, pickup=src, dropoff=bay, release_step=rel, due_step=t_due)
            t_obj.kit_op_id = op_id
            tasks.append(t_obj)
            part_ids.append(tid)

        crew_id = 1 if (stress_axis == "converging_dag" and o_idx % 3 != 2) or o_idx < (n_ops // 2) else 2
        tool_id = 1 if (o_idx < n_ops // 2 or stress_axis == "converging_dag" and o_idx >= 4) else 2
        ops.append(
            OperationSnapshot(
                op_id=op_id,
                kit_task_ids=tuple(part_ids),
                bay_id=bay,
                crew_id=crew_id,
                tool_id=tool_id,
                proc_steps=16 + (o_idx % 2) * 3,
                due_step=op_due,
                predecessor_op_id=pred,
                predecessor_op_ids=preds_tuple,
                eligible_bays=tuple(sorted({bay, alt_bay})),
                eligible_crews=(1, 2),
            )
        )

    spawn_pool = [n for n in all_nodes if net.node(n).kind != NodeKind.WORKSTATION]
    if len(spawn_pool) < n_agvs:
        spawn_pool = all_nodes
    starts = [spawn_pool[(i * 3 + rng.randint(0, 2)) % len(spawn_pool)] for i in range(n_agvs)]
    agvs = [Agv(f"agv_{i}", max_speed=3.0, start_node=starts[i]) for i in range(n_agvs)]

    u_pick = rng.choice(east_sources if stress_axis == "flow_inversion" else src_nodes)
    u_drop_cands = [b for b in bay_nodes if b != u_pick] or bay_nodes
    u_drop = rng.choice(u_drop_cands)

    f1_step = rng.randint(10, 18) if stress_axis == "dual_fault" else rng.randint(10, 22)
    f2_step = f1_step + rng.randint(6, 10) if stress_axis == "dual_fault" else rng.randint(65, 85)
    f2_dur = 22 if stress_axis == "dual_fault" else 18
    disruptions = {
        "fault_agv": agvs[rng.randrange(len(agvs))].agv_id,
        "fault_step": f1_step,
        "fault_window": 3,
        "event2_regime": "OPEN_WINDOW_KITTING_BOTTLENECK",
        "kitting_bottleneck_window": 3,
        "kitting_bottleneck_delay": 45,
        "urgent_task_id": "T_URGENT_V10",
        "urgent_pickup": u_pick,
        "urgent_dropoff": u_drop,
        "urgent_release_step": rng.randint(30, 46),
        "urgent_due_step": rng.randint(38, 56),
        "post_pickup_fault_step": f2_step,
        "post_pickup_fault_duration": f2_dur,
        "stress_axis": stress_axis,
    }
    return net, tasks, agvs, ops, disruptions


def extract_opening_structural_features(
    network: Network,
    snapshot: NativeSnapshot,
    *,
    visibility: str = "planned",
) -> np.ndarray:
    """Extract 16 strictly causal t=0 structural features with zero layout identifier leakage.

    `visibility="planned"` uses the day's planned t=0 work order schedule (without exposing future
    disruptions or unreleased runtime events); `visibility="released_only"` restricts task-level
    workload/distance statistics strictly to tasks with `release_step <= snapshot.tick`.
    """
    n_nodes = max(1, len(network.nodes))
    n_edges = max(1, len(network.edges))
    edge_caps = [float(e.capacity) for e in network.edges.values()] or [1.0]
    mean_edge_cap = float(np.mean(edge_caps))
    frac_single_lane = float(sum(1 for c in edge_caps if c <= 1.0)) / float(len(edge_caps))
    out_degs = [len(network.out_edges(nid)) for nid in network.nodes] or [1]
    mean_out_degree = float(np.mean(out_degs))

    visible_tasks = (
        [t for t in snapshot.tasks if t.release_step <= snapshot.tick]
        if visibility == "released_only"
        else list(snapshot.tasks)
    )
    n_agvs = max(1, len(snapshot.robots))
    n_tasks = len(visible_tasks)
    n_ops = max(1, len(snapshot.operations))

    tt_cache: dict[str, dict[str, float]] = {}
    tts: list[float] = []
    for t in visible_tasks:
        dmap = tt_cache.get(t.pickup)
        if dmap is None:
            dmap = network.shortest_travel_times(t.pickup, 3.0)
            tt_cache[t.pickup] = dmap
        val = dmap.get(t.dropoff, 30.0)
        if math.isfinite(val):
            tts.append(float(val))
    tt_mean = float(np.mean(tts)) if tts else 15.0
    tt_p90 = float(np.percentile(tts, 90)) if tts else 25.0

    kit_sizes = [len(op.kit_task_ids) for op in snapshot.operations] or [2]
    locked_cnt = sum(1 for t in visible_tasks if t.assigned_agv is not None)
    bay_flex = [len(op.eligible_bays) if op.eligible_bays else 1 for op in snapshot.operations] or [1]
    crew_flex = [len(op.eligible_crews) if op.eligible_crews else 1 for op in snapshot.operations] or [1]

    tool_counts: dict[int, int] = {}
    for op in snapshot.operations:
        tool_counts[int(op.tool_id)] = tool_counts.get(int(op.tool_id), 0) + 1
    max_tool_share = max(tool_counts.values(), default=1) / float(n_ops)
    pred_edges = sum(len(op_predecessors(op)) for op in snapshot.operations)
    slacks = [float(op.due_step - op.proc_steps - snapshot.tick) for op in snapshot.operations] or [100.0]

    vec = np.asarray(
        [
            math.log1p(float(n_nodes)) / 5.0,
            float(n_edges) / float(n_nodes) / 4.0,
            mean_edge_cap / 2.0,
            frac_single_lane,
            mean_out_degree / 4.0,
            tt_mean / 40.0,
            tt_p90 / 60.0,
            float(n_tasks) / float(n_agvs) / 6.0,
            float(n_ops) / float(n_agvs) / 3.0,
            float(np.mean(kit_sizes)) / 4.0,
            float(locked_cnt) / max(1.0, float(n_tasks)),
            float(np.mean(bay_flex)) / 3.0,
            float(np.mean(crew_flex)) / 3.0,
            max_tool_share,
            float(pred_edges) / float(n_ops),
            float(np.mean(slacks)) / 200.0,
        ],
        dtype=np.float32,
    )
    return vec
