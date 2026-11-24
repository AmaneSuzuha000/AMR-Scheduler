"""Brand-New Untouched Procedural Holdout Suite (`HOLDOUT_V17_FINAL`: seeds `9201..9212`).

Strictly quarantined from ALL Stage 0-3 training/tuning scripts (`5101..5108`, `8101..8102`,
dev eval `7001..7006`, `8201..8202`, `9001..9002`). Adds a third unseen procedural layout
`cross_wing_yard` alongside fresh seeds on `hub_choke` and `double_ladder`.
"""

from __future__ import annotations

import random
from typing import Any

from experiments.holdout_scenarios_v16 import _bi, build_holdout_v16_scenario
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.entities import Agv, Task, TaskPriority
from amr_sched.network import Network, Node, NodeKind


def build_holdout_v17_scenario(
    variant: str,
    seed: int,
    *,
    with_windows: bool = True,
) -> tuple[Network, list[Task], list[Agv], list[OperationSnapshot], dict[str, Any]]:
    """Build untouched Stage-4 procedural holdout scenario (`cross_wing_yard` | `hub_choke` | `double_ladder`)."""
    if variant in ("hub_choke", "double_ladder"):
        return build_holdout_v16_scenario(variant, seed, with_windows=with_windows)

    rng = random.Random(int(seed) * 317 + sum((i + 1) * ord(ch) for i, ch in enumerate(variant)))
    net = Network()
    # Dual reversible capacity-1 bridge corridors (`BR_N1<->BR_N2` and `BR_S1<->BR_S2`) + outer bypass loop
    coords = {
        "STG_W": (-14.0, 0.0, NodeKind.DOCK, 4),
        "STG_E": (14.0, 0.0, NodeKind.DOCK, 4),
        "WJ_N": (-8.0, 5.0, NodeKind.AISLE, 2),
        "WJ_S": (-8.0, -5.0, NodeKind.AISLE, 2),
        "BR_N1": (-2.5, 5.0, NodeKind.AISLE, 1),
        "BR_N2": (2.5, 5.0, NodeKind.AISLE, 1),
        "BR_S1": (-2.5, -5.0, NodeKind.AISLE, 1),
        "BR_S2": (2.5, -5.0, NodeKind.AISLE, 1),
        "EJ_N": (8.0, 5.0, NodeKind.AISLE, 2),
        "EJ_S": (8.0, -5.0, NodeKind.AISLE, 2),
        "MID_X": (0.0, 0.0, NodeKind.AISLE, 2),
        "BAY_NW": (-10.0, 10.0, NodeKind.WORKSTATION, 1),
        "BAY_NE": (10.0, 10.0, NodeKind.WORKSTATION, 1),
        "BAY_SW": (-10.0, -10.0, NodeKind.WORKSTATION, 1),
        "BAY_SE": (10.0, -10.0, NodeKind.WORKSTATION, 1),
    }
    for nid, (x, y, kd, cp) in coords.items():
        net.add_node(Node(nid, x, y, kind=kd, capacity=cp))

    for u, v, ln in (
        ("STG_W", "WJ_N", 6.0),
        ("STG_W", "WJ_S", 6.0),
        ("STG_E", "EJ_N", 6.0),
        ("STG_E", "EJ_S", 6.0),
        ("WJ_N", "BR_N1", 6.0),
        ("BR_N1", "BR_N2", 9.0),
        ("BR_N2", "EJ_N", 6.0),
        ("WJ_S", "BR_S1", 6.0),
        ("BR_S1", "BR_S2", 9.0),
        ("BR_S2", "EJ_S", 6.0),
        ("WJ_N", "MID_X", 9.0),
        ("WJ_S", "MID_X", 9.0),
        ("MID_X", "EJ_N", 9.0),
        ("MID_X", "EJ_S", 9.0),
        ("WJ_N", "BAY_NW", 6.0),
        ("EJ_N", "BAY_NE", 6.0),
        ("WJ_S", "BAY_SW", 6.0),
        ("EJ_S", "BAY_SE", 6.0),
    ):
        _bi(net, u, v, ln, speed=3.0, cap=1)

    west_src = ["STG_W", "BAY_NW", "BAY_SW"]
    east_src = ["STG_E", "BAY_NE", "BAY_SE"]
    west_bays = ["BAY_NW", "BAY_SW"]
    east_bays = ["BAY_NE", "BAY_SE"]
    all_bays = west_bays + east_bays

    agvs = [
        Agv("AGV_1", max_speed=3.0, start_node="STG_W", load_time=2.0, unload_time=2.0),
        Agv("AGV_2", max_speed=3.0, start_node="STG_W", load_time=2.0, unload_time=2.0),
        Agv("AGV_3", max_speed=3.0, start_node="WJ_N", load_time=2.0, unload_time=2.0),
        Agv("AGV_4", max_speed=3.0, start_node="STG_E", load_time=2.0, unload_time=2.0),
        Agv("AGV_5", max_speed=3.0, start_node="STG_E", load_time=2.0, unload_time=2.0),
        Agv("AGV_6", max_speed=3.0, start_node="EJ_S", load_time=2.0, unload_time=2.0),
    ]

    tasks: list[Task] = []
    ops: list[OperationSnapshot] = []
    for o_idx in range(9):
        op_id = f"OP_{o_idx + 1}"
        go_east = (o_idx % 2 == 0)
        bay = rng.choice(east_bays if go_east else west_bays)
        alt_bays = tuple(b for b in all_bays if b != bay)[:2]
        src_pool = west_src if go_east else east_src
        n_kit = 3 if o_idx in (1, 4, 7) else 2
        preds: tuple[str, ...] = ()
        if o_idx == 5:
            preds = ("OP_1",)
        elif o_idx == 7:
            preds = ("OP_3",)
        elif o_idx == 8:
            preds = ("OP_5", "OP_6")

        due = 52 + o_idx * 16 + rng.randint(0, 12)
        rel_base = 0 if o_idx < 3 else (o_idx - 2) * 7 + rng.randint(0, 4)
        wins: tuple[tuple[int, int], ...] = ()
        if with_windows and o_idx in (1, 2, 4, 6):
            wins = ((8, 38 + (o_idx % 3) * 8), (80, 165))

        kit_ids: list[str] = []
        for k in range(n_kit):
            tid = f"T_{op_id}_{k + 1}"
            kit_ids.append(tid)
            pu = rng.choice([s for s in src_pool if s != bay] or src_pool)
            t_obj = Task(
                task_id=tid,
                pickup=pu,
                dropoff=bay,
                release_step=rel_base + k * 3,
                due_step=due - 6,
                priority=TaskPriority.URGENT if o_idx == 2 else TaskPriority.NORMAL,
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
                proc_steps=7 + rng.randint(0, 5),
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
        "urgent_dropoff": all_bays[0],
    }
    return net, tasks, agvs, ops, disruptions
