"""Paired Production-Kitting Window & Choke Dev Scenarios (V13 Research Benchmark).

Implements the exact 2x2x2x2 = 16 paired factor design for testing whether RL learns
discrete production resource window protection and multi-decision joint neighborhoods:
  1. `shift_timing`: `"early"` (crew window opens late -> speeding arrival saves 0)
                     vs `"late"` (window closes tightly -> catching cutoff saves ~40 steps)
  2. `missing_count`: `"single"` (1 bottleneck missing piece) vs `"multi"` (3 missing pieces)
  3. `corridor_mode`: `"choke"` (contested 1-lane central bridge)
                      vs `"bypass"` (long northern bypass option + bay diversion)
  4. `window_count`: `"one"` (continuous availability)
                     vs `"two"` (two consecutive shift windows with 40-step gap)
"""

from __future__ import annotations

import itertools
import random
from typing import Any

from amr_sched.cvcr.snapshot import OperationSnapshot
from amr_sched.entities import Agv, Task, TaskPriority
from amr_sched.network import Edge, Network, Node
from amr_sched.scenarios_v10 import build_unseen_choke_yard_network

V13_SHIFT_FACTORS = ("early", "late")
V13_MISSING_FACTORS = ("single", "multi")
V13_CORRIDOR_FACTORS = ("choke", "bypass")
V13_WINDOW_FACTORS = ("one", "two")

V13_FACTOR_COMBOS: list[tuple[str, str, str, str]] = list(
    itertools.product(
        V13_SHIFT_FACTORS,
        V13_MISSING_FACTORS,
        V13_CORRIDOR_FACTORS,
        V13_WINDOW_FACTORS,
    )
)


def build_v13_paired_dev_scenario(
    seed: int,
    *,
    shift_timing: str = "late",
    missing_count: str = "multi",
    corridor_mode: str = "choke",
    window_count: str = "two",
    n_agvs: int = 5,
    speed: float = 3.0,
) -> tuple[Any, list[Task], list[Agv], list[OperationSnapshot], dict[str, Any]]:
    """Build deterministic paired scenario on West/East choke-yard isolation testbed."""
    factor_tag = f"{shift_timing}:{missing_count}:{corridor_mode}:{window_count}"
    rng = random.Random(int(seed) * 131 + sum((i + 1) * ord(ch) for i, ch in enumerate(factor_tag)))

    base_net = build_unseen_choke_yard_network(aisle_capacity=1, speed=speed)
    net = Network()
    for nid, n_obj in base_net.nodes.items():
        cap = 3 if (corridor_mode == "bypass" and nid in ("BYP_1", "BYP_2")) else n_obj.capacity
        net.add_node(Node(nid, n_obj.x, n_obj.y, kind=n_obj.kind, capacity=cap))
    byp_len = 16.0 if corridor_mode == "bypass" else 36.0
    byp_pairs = {("WA_N2", "BYP_1"), ("BYP_1", "WA_N2"), ("BYP_2", "EA_N0"), ("EA_N0", "BYP_2")}
    for e_obj in base_net.edges.values():
        elen = byp_len if (e_obj.src, e_obj.dst) in byp_pairs else e_obj.length
        net.add_edge(Edge(e_obj.src, e_obj.dst, length=elen, speed_limit=e_obj.speed_limit, capacity=e_obj.capacity))

    east_bays = ["W4", "W5", "W6"]
    west_sources = ["DL_W1", "DL_W2", "WBUF1"]
    east_sources = ["EBUF1", "EBUF2"]

    tasks: list[Task] = []
    ops: list[OperationSnapshot] = []

    # OP_1..OP_6: Create 2 tight cross-bridge kitted operations (OP_3 & OP_4) plus flanking orders
    # so choosing WHICH kitting window to protect determines whether 40-step crew wait happens.
    base_kit_n = 3 if missing_count == "multi" else 1
    for o_idx in range(6):
        op_id = f"OP_{o_idx + 1}"
        # OP_5 depends on OP_3; OP_6 depends on OP_4 -> protecting OP_3/OP_4 ripples downstream!
        preds: tuple[str, ...] = ()
        if o_idx == 4:
            preds = ("OP_3",)
        elif o_idx == 5:
            preds = ("OP_4",)

        # Alternate West/East so cross-choke contention is guaranteed
        if o_idx in (0, 2, 3):
            bay = east_bays[o_idx % len(east_bays)]
            alt_bay = rng.choice([b for b in east_bays if b != bay])
            src_pool = west_sources
        else:
            bay = rng.choice(["W1", "W2", "W3"])
            alt_bay = rng.choice([b for b in ("W1", "W2", "W3") if b != bay])
            src_pool = east_sources

        n_parts = base_kit_n if o_idx in (2, 3) else 2
        rel = 0 if o_idx < 4 else 12 + rng.randint(0, 4)

        # Window calibration:
        # - OP_3 is high downstream impact (unlocks OP_5) with tight closing shift window around t ~ 46..54
        # - OP_1 has an early wait window (crew arrives late at t=52) in "early" mode so rushing it saves 0
        # - OP_4 competes on crew 1 with OP_3
        r_wins: tuple[tuple[int, int], ...] = ()
        if window_count == "two":
            if shift_timing == "early":
                if o_idx == 2:
                    # Shift opens late (58..74) then jumps +40 to (114..999): rushing before 58 saves 0 unless missing 74
                    r_wins = ((58, 74), (114, 999))
                elif o_idx == 3:
                    # Tight window (36..49) -> protecting OP_4 catches 49 instead of waiting until 89!
                    r_wins = ((36, 49), (89, 999))
            else:  # "late": tight closing crew windows right around arrival frontier!
                if o_idx == 2:
                    r_wins = ((0, 46), (86, 999))
                elif o_idx == 3:
                    r_wins = ((0, 56), (96, 999))
                elif o_idx == 1:
                    r_wins = ((45, 72), (112, 999))

        kit_ids: list[str] = []
        op_due = 88 + o_idx * 16 + rng.randint(0, 8)
        if o_idx == 2 and shift_timing == "late":
            op_due = 72
        elif o_idx == 3 and shift_timing == "early":
            op_due = 72

        for k_i in range(n_parts):
            tid = f"T_{o_idx + 1}_{k_i + 1}"
            # Last piece of critical op starts farther if missing_count=="single"
            src = west_sources[0] if (n_parts == 1 and o_idx in (2, 3)) else src_pool[(k_i + o_idx + seed) % len(src_pool)]
            t_due = op_due - 14 + k_i * 2
            t_obj = Task(
                tid,
                pickup=src,
                dropoff=bay,
                release_step=rel,
                due_step=t_due,
                priority=TaskPriority.URGENT if o_idx == 0 else TaskPriority.NORMAL,
            )
            t_obj.kit_op_id = op_id
            tasks.append(t_obj)
            kit_ids.append(tid)

        ops.append(
            OperationSnapshot(
                op_id=op_id,
                kit_task_ids=tuple(kit_ids),
                bay_id=bay,
                crew_id=1 if o_idx in (1, 2, 3) else 2,
                tool_id=1 if o_idx < 4 else 2,
                proc_steps=14 + (o_idx % 2) * 3,
                due_step=op_due,
                predecessor_op_id=preds[0] if preds else None,
                predecessor_op_ids=preds,
                is_urgent=(o_idx == 0),
                eligible_bays=tuple(sorted({bay, alt_bay})),
                eligible_crews=(1, 2),
                resource_windows=r_wins,
            )
        )

    spawn_candidates = [
        "WA_N0", "WA_S1", "WA_N1", "EA_N0", "EA_S1", "EA_N1",
        "WA_S0", "WA_N2", "EA_S0", "EA_N2", "WBUF1", "EBUF1", "EBUF2",
    ]
    rng.shuffle(spawn_candidates)
    spawn_counts: dict[str, int] = {}
    agvs: list[Agv] = []
    for i in range(n_agvs):
        chosen_node = spawn_candidates[i % len(spawn_candidates)]
        for cand_n in spawn_candidates:
            cap_n = int(net.node(cand_n).capacity)
            if spawn_counts.get(cand_n, 0) < cap_n:
                chosen_node = cand_n
                break
        spawn_counts[chosen_node] = spawn_counts.get(chosen_node, 0) + 1
        agvs.append(Agv(f"agv_{i}", max_speed=speed, start_node=chosen_node))

    disruptions = {
        "fault_agv": agvs[seed % len(agvs)].agv_id,
        "fault_step": 10 + (seed % 7),
        "fault_window": 3,
        "event2_regime": "OPEN_WINDOW_KITTING_BOTTLENECK",
        "kitting_bottleneck_window": 3,
        "kitting_bottleneck_delay": 25,
        "urgent_task_id": "T_URG_V13",
        "urgent_pickup": rng.choice(west_sources),
        "urgent_dropoff": rng.choice(["W5", "W6"]),
        "urgent_release_step": 22 + (seed % 10),
        "urgent_due_step": 44 + (seed % 10),
        "post_pickup_fault_step": 62 + (seed % 10),
        "post_pickup_fault_duration": 14,
        "v13_factors": factor_tag,
    }
    return net, tasks, agvs, ops, disruptions
