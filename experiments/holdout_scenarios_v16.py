"""Untouched Holdout Topologies (`HOLDOUT_NEW_V16`) & Deterministic 2-AMR Bridge Reversal.

Provides:
  1. `build_holdout_v16_scenario(variant, seed, with_windows=False)`: procedural unseen
     factory graphs (`hub_choke`, `double_ladder`) with zero overlap against `ring`, `grid`,
     `workshop`, `unseen_ladder`, or `unseen_choke_yard`, strictly reserved for holdout
     seeds `9001..9016`.
  2. `run_controlled_2amr_window_reversal()`: deterministic symmetric 2-AMR bottleneck bridge
     fixture demonstrating how reversing bottleneck passage order flips downstream production
     crew-window feasibility by >40 steps.
"""

from __future__ import annotations

import random
from typing import Any

from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.cvcr.snapshot import NativeSnapshot, RobotSnapshot, TaskSnapshot
from amr_sched.entities import Agv, Task, TaskPriority
from amr_sched.network import Edge, Network, Node, NodeKind


def _bi(net: Network, u: str, v: str, length: float, speed: float = 3.0, cap: int = 1) -> None:
    """Add symmetric directed edge pair between nodes `u` and `v`."""
    net.add_edge(Edge(u, v, length=length, speed_limit=speed, capacity=cap))
    net.add_edge(Edge(v, u, length=length, speed_limit=speed, capacity=cap))


def build_holdout_v16_scenario(
    variant: str,
    seed: int,
    *,
    with_windows: bool = False,
) -> tuple[Network, list[Task], list[Agv], list[OperationSnapshot], dict[str, Any]]:
    """Build procedural unseen topology (`hub_choke` or `double_ladder`) for final confirmation."""
    rng = random.Random(int(seed) * 233 + sum((i + 1) * ord(ch) for i, ch in enumerate(variant)))
    net = Network()
    if variant == "hub_choke":
        # Two production wings bridged by a narrow central hub (H_C1 <-> H_C2) and long outer ring
        nodes = {
            "D_W": (-15.0, 0.0, NodeKind.DOCK, 4),
            "D_E": (15.0, 0.0, NodeKind.DOCK, 4),
            "W_J1": (-9.0, 4.0, NodeKind.AISLE, 2),
            "W_J2": (-9.0, -4.0, NodeKind.AISLE, 2),
            "H_C1": (-3.0, 0.0, NodeKind.AISLE, 1),
            "H_C2": (3.0, 0.0, NodeKind.AISLE, 1),
            "E_J1": (9.0, 4.0, NodeKind.AISLE, 2),
            "E_J2": (9.0, -4.0, NodeKind.AISLE, 2),
            "RING_N": (0.0, 12.0, NodeKind.AISLE, 2),
            "RING_S": (0.0, -12.0, NodeKind.AISLE, 2),
            "WB1": (-12.0, 7.0, NodeKind.WORKSTATION, 1),
            "WB2": (-12.0, -7.0, NodeKind.WORKSTATION, 1),
            "WB3": (12.0, 7.0, NodeKind.WORKSTATION, 1),
            "WB4": (12.0, -7.0, NodeKind.WORKSTATION, 1),
            "WB5": (15.0, 6.0, NodeKind.WORKSTATION, 1),
            "WB6": (-15.0, -6.0, NodeKind.WORKSTATION, 1),
        }
        for nid, (x, y, kd, cp) in nodes.items():
            net.add_node(Node(nid, x, y, kind=kd, capacity=cp))
        _bi(net, "D_W", "W_J1", 9.0)
        _bi(net, "D_W", "W_J2", 9.0)
        _bi(net, "W_J1", "H_C1", 9.0)
        _bi(net, "W_J2", "H_C1", 9.0)
        _bi(net, "H_C1", "H_C2", 9.0, cap=1)
        _bi(net, "H_C2", "E_J1", 9.0)
        _bi(net, "H_C2", "E_J2", 9.0)
        _bi(net, "E_J1", "D_E", 9.0)
        _bi(net, "E_J2", "D_E", 9.0)
        _bi(net, "W_J1", "RING_N", 18.0)
        _bi(net, "RING_N", "E_J1", 18.0)
        _bi(net, "W_J2", "RING_S", 18.0)
        _bi(net, "RING_S", "E_J2", 18.0)
        _bi(net, "W_J1", "WB1", 6.0)
        _bi(net, "W_J2", "WB2", 6.0)
        _bi(net, "E_J1", "WB3", 6.0)
        _bi(net, "E_J2", "WB4", 6.0)
        _bi(net, "D_E", "WB5", 6.0)
        _bi(net, "D_W", "WB6", 6.0)
        west_src = ["D_W", "WB6"]
        east_src = ["D_E", "WB5"]
        bays = ["WB1", "WB2", "WB3", "WB4", "WB5", "WB6"]
        agvs = [
            Agv("AGV_1", max_speed=3.0, start_node="D_W", load_time=2.0, unload_time=2.0),
            Agv("AGV_2", max_speed=3.0, start_node="D_W", load_time=2.0, unload_time=2.0),
            Agv("AGV_3", max_speed=3.0, start_node="H_C1", load_time=2.0, unload_time=2.0),
            Agv("AGV_4", max_speed=3.0, start_node="D_E", load_time=2.0, unload_time=2.0),
            Agv("AGV_5", max_speed=3.0, start_node="D_E", load_time=2.0, unload_time=2.0),
            Agv("AGV_6", max_speed=3.0, start_node="E_J1", load_time=2.0, unload_time=2.0),
        ]
    else:  # double_ladder
        nodes = {
            "D_L": (-12.0, 0.0, NodeKind.DOCK, 4),
            "D_R": (12.0, 0.0, NodeKind.DOCK, 4),
            "L1": (-6.0, 5.0, NodeKind.AISLE, 1),
            "L2": (0.0, 5.0, NodeKind.AISLE, 1),
            "L3": (6.0, 5.0, NodeKind.AISLE, 1),
            "M1": (-6.0, 0.0, NodeKind.AISLE, 1),
            "M2": (0.0, 0.0, NodeKind.AISLE, 1),
            "M3": (6.0, 0.0, NodeKind.AISLE, 1),
            "R1": (-6.0, -5.0, NodeKind.AISLE, 1),
            "R2": (0.0, -5.0, NodeKind.AISLE, 1),
            "R3": (6.0, -5.0, NodeKind.AISLE, 1),
            "WA": (-6.0, 10.0, NodeKind.WORKSTATION, 1),
            "WB": (6.0, 10.0, NodeKind.WORKSTATION, 1),
            "WC": (-6.0, -10.0, NodeKind.WORKSTATION, 1),
            "WD": (6.0, -10.0, NodeKind.WORKSTATION, 1),
        }
        for nid, (x, y, kd, cp) in nodes.items():
            net.add_node(Node(nid, x, y, kind=kd, capacity=cp))
        for row, ln in ((("L1", "L2", "L3"), 9.0), (("M1", "M2", "M3"), 6.0), (("R1", "R2", "R3"), 9.0)):
            _bi(net, row[0], row[1], ln)
            _bi(net, row[1], row[2], ln)
        for c in ("1", "2", "3"):
            _bi(net, f"L{c}", f"M{c}", 6.0)
            _bi(net, f"M{c}", f"R{c}", 6.0)
        _bi(net, "D_L", "L1", 6.0)
        _bi(net, "D_L", "M1", 6.0)
        _bi(net, "D_L", "R1", 6.0)
        _bi(net, "D_R", "L3", 6.0)
        _bi(net, "D_R", "M3", 6.0)
        _bi(net, "D_R", "R3", 6.0)
        _bi(net, "L1", "WA", 6.0)
        _bi(net, "L3", "WB", 6.0)
        _bi(net, "R1", "WC", 6.0)
        _bi(net, "R3", "WD", 6.0)
        west_src = ["D_L", "WA", "WC"]
        east_src = ["D_R", "WB", "WD"]
        bays = ["WA", "WB", "WC", "WD"]
        agvs = [
            Agv("AGV_1", max_speed=3.0, start_node="D_L", load_time=2.0, unload_time=2.0),
            Agv("AGV_2", max_speed=3.0, start_node="D_L", load_time=2.0, unload_time=2.0),
            Agv("AGV_3", max_speed=3.0, start_node="M1", load_time=2.0, unload_time=2.0),
            Agv("AGV_4", max_speed=3.0, start_node="D_R", load_time=2.0, unload_time=2.0),
            Agv("AGV_5", max_speed=3.0, start_node="D_R", load_time=2.0, unload_time=2.0),
            Agv("AGV_6", max_speed=3.0, start_node="M3", load_time=2.0, unload_time=2.0),
        ]

    tasks: list[Task] = []
    ops: list[OperationSnapshot] = []
    for o_idx in range(8):
        op_id = f"OP_{o_idx + 1}"
        cross_east = (o_idx % 2 == 0)
        bay = rng.choice(bays[len(bays) // 2 :] if cross_east else bays[: len(bays) // 2])
        alt_bays = tuple(b for b in bays if b != bay)[:2]
        src_pool = west_src if cross_east else east_src
        n_kit = 3 if o_idx in (2, 5) else 2
        preds: tuple[str, ...] = ()
        if o_idx == 4:
            preds = ("OP_1",)
        elif o_idx == 6:
            preds = ("OP_3",)
        elif o_idx == 7:
            preds = ("OP_5",)

        due = 55 + o_idx * 18 + rng.randint(0, 14)
        rel_base = 0 if o_idx < 3 else (o_idx - 2) * 8 + rng.randint(0, 5)
        wins: tuple[tuple[int, int], ...] = ()
        if with_windows and o_idx in (1, 3, 5):
            wins = ((10, 42 + (o_idx % 3) * 8), (82, 160))

        kit_ids: list[str] = []
        for k in range(n_kit):
            tid = f"T_{op_id}_{k + 1}"
            kit_ids.append(tid)
            pu = rng.choice([s for s in src_pool if s != bay] or src_pool)
            rel_k = rel_base + k * 4
            t_obj = Task(
                task_id=tid,
                pickup=pu,
                dropoff=bay,
                release_step=rel_k,
                due_step=due - 6,
                priority=TaskPriority.URGENT if o_idx == 3 else TaskPriority.NORMAL,
            )
            t_obj.kit_op_id = op_id  # type: ignore[attr-defined]
            tasks.append(t_obj)
        ops.append(
            OperationSnapshot(
                op_id=op_id,
                kit_task_ids=tuple(kit_ids),
                bay_id=bay,
                crew_id=(o_idx % 3) + 1,
                tool_id=(o_idx % 3) + 1,
                proc_steps=8 + rng.randint(0, 6),
                due_step=due,
                predecessor_op_id=preds[0] if preds else None,
                predecessor_op_ids=preds,
                eligible_bays=tuple([bay] + list(alt_bays)),
                eligible_crews=(1, 2, 3),
                resource_windows=wins,
            )
        )

    disruptions = {
        "fault_agv_id": "AGV_2",
        "fault_duration": 26,
        "fault_window": 2,
        "post_pickup_fault": True,
        "urgent_pickup": west_src[0],
        "urgent_dropoff": bays[0],
    }
    return net, tasks, agvs, ops, disruptions


def run_controlled_2amr_window_reversal() -> dict[str, Any]:
    """Verify reversible passage order over shared bridge flips production shift-window completion."""
    net = Network()
    coords = {
        "DOCK": (-4.0, 0.0, NodeKind.DOCK, 2),
        "J_L": (-1.5, 0.0, NodeKind.AISLE, 1),
        "J_R": (1.5, 0.0, NodeKind.AISLE, 1),
        "BAY_A": (4.0, 1.5, NodeKind.WORKSTATION, 1),
        "BAY_B": (4.0, -1.5, NodeKind.WORKSTATION, 1),
    }
    for nid, (x, y, kd, cp) in coords.items():
        net.add_node(Node(nid, x, y, kind=kd, capacity=cp))
    for u, v, ln in (
        ("DOCK", "J_L", 3.0),
        ("J_L", "J_R", 12.0),  # 4 ticks single-lane bridge @ 3.0 m/step
        ("J_R", "BAY_A", 3.0),
        ("J_R", "BAY_B", 3.0),
    ):
        _bi(net, u, v, ln, speed=3.0, cap=1)

    # First AGV unloads by ~t=11; second must yield on single-lane bridge and finishes at ~t=16.
    # OP_A opens at t=20 (arriving early wastes no production time);
    # OP_B closes at t=14 then doesn't reopen until t=55!
    ops = [
        OperationSnapshot(
            op_id="OP_A",
            kit_task_ids=("T_A",),
            bay_id="BAY_A",
            crew_id=1,
            tool_id=1,
            proc_steps=5,
            due_step=30,
            resource_windows=((20, 45),),
        ),
        OperationSnapshot(
            op_id="OP_B",
            kit_task_ids=("T_B",),
            bay_id="BAY_B",
            crew_id=2,
            tool_id=2,
            proc_steps=5,
            due_step=24,
            resource_windows=((0, 12), (55, 85)),
        ),
    ]
    snap = NativeSnapshot(
        schema="v1",
        tick=0,
        version={"diag": 1},
        world_fingerprint="2amr-symmetric-bridge",
        robots=[RobotSnapshot("AGV_1", "DOCK", "idle"), RobotSnapshot("AGV_2", "DOCK", "idle")],
        tasks=[
            TaskSnapshot("T_A", "DOCK", "BAY_A", 0, 22, "pending", None, kit_op_id="OP_A"),
            TaskSnapshot("T_B", "DOCK", "BAY_B", 0, 22, "pending", None, kit_op_id="OP_B"),
        ],
        operations=ops,
        reservations=[],
        now_monotonic=0.0,
    )
    ev = CVCREvaluator(net, snap, max_speed=3.0)
    # AGV_1 takes corridor priority first inside evaluator ordering:
    a_first = ev.evaluate({"T_A": "AGV_1", "T_B": "AGV_2"})
    b_first = ev.evaluate({"T_A": "AGV_2", "T_B": "AGV_1"})
    return {
        "T_A_first": {
            "valid": bool(a_first.valid),
            "obj": round(float(a_first.objective), 2),
            "op_start": dict(a_first.op_start),
            "op_end": dict(a_first.op_end),
        },
        "T_B_first": {
            "valid": bool(b_first.valid),
            "obj": round(float(b_first.objective), 2),
            "op_start": dict(b_first.op_start),
            "op_end": dict(b_first.op_end),
        },
        "op_B_start_jump": int(a_first.op_start.get("OP_B", 0)) - int(b_first.op_start.get("OP_B", 0)),
        "reversible_obj_gain": round(float(a_first.objective) - float(b_first.objective), 2),
    }
