"""Stage 2 (E1) unit and integration tests for pairwise window payoff representation and candidate dedup."""

from __future__ import annotations

import numpy as np

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.policy import LGCVCRPolicy, _LazyKittingSurrogate
from amr_sched.cvcr.production import NativeProductionTracker
from amr_sched.cvcr.snapshot import NativeSnapshotCapture, OperationSnapshot
from amr_sched.cvcr.v15_ea_joint import V15ExecutionAwarePolicy, compute_action_diff_digest
from amr_sched.cvcr.v17_window_protection import (
    V17_PAIRWISE_DIM,
    compute_op_readiness_profile,
    compute_pairwise_payoff,
    generate_v17_dual_op_candidates,
    select_coupled_op_pair,
)
from amr_sched.entities import Agv, Task, TaskStatus
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import Hop, SimulationConfig, Simulator, _Job


def test_stage2_pairwise_payoff_and_action_diff_dedup() -> None:
    """Verify pairwise `(K, P, R, S)` payoff vector accurately captures window crossing & dedup works."""
    net = Network()
    for nid, (x, y, kind, cap) in {
        "LA": (0.0, 2.0, NodeKind.BUFFER, 2),
        "LB": (0.0, -2.0, NodeKind.BUFFER, 2),
        "C0": (2.0, 0.0, NodeKind.AISLE, 1),
        "C1": (6.0, 0.0, NodeKind.AISLE, 1),
        "WA": (8.0, 2.0, NodeKind.WORKSTATION, 2),
        "WB": (8.0, -2.0, NodeKind.WORKSTATION, 2),
    }.items():
        net.add_node(Node(nid, x, y, kind=kind, capacity=cap))
    for u, v, d in (("LA", "C0", 2.0), ("LB", "C0", 2.0), ("C0", "C1", 4.0), ("C1", "WA", 2.0), ("C1", "WB", 2.0)):
        net.add_edge(Edge(u, v, length=d, speed_limit=1.0, capacity=1))
        net.add_edge(Edge(v, u, length=d, speed_limit=1.0, capacity=1))

    ta = Task("T_A", "LA", "WA", release_step=0, due_step=20)
    ta.kit_op_id = "OP_A"  # type: ignore[attr-defined]
    tb = Task("T_B", "LB", "WB", release_step=0, due_step=16)
    tb.kit_op_id = "OP_B"  # type: ignore[attr-defined]
    ops = [
        OperationSnapshot("OP_A", ("T_A",), "WA", crew_id=1, tool_id=1, proc_steps=4, due_step=40, resource_windows=((12, 120),)),
        OperationSnapshot("OP_B", ("T_B",), "WB", crew_id=2, tool_id=2, proc_steps=4, due_step=20, resource_windows=((0, 11),)),
    ]
    sim = Simulator(
        net,
        [ta, tb],
        [Agv("agv_0", max_speed=1.0, start_node="LA"), Agv("agv_1", max_speed=1.0, start_node="LB")],
        SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=150, load_time=2.0, unload_time=2.0, idle_retreat=False),
    )
    sim.reset()
    ta.status = TaskStatus.PICKED
    ta.assigned_agv = "agv_0"
    tb.status = TaskStatus.PICKED
    tb.assigned_agv = "agv_1"
    sim.agvs[0].cur_node = "LA"
    sim.agvs[0].cur_task = "T_A"
    sim.agvs[1].cur_node = "LB"
    sim.agvs[1].cur_task = "T_B"
    sim._jobs["agv_0"] = _Job("T_A", "to_dropoff", [Hop("LA", "C0", 0, 2)], idx=0, goal="WA")
    sim._jobs["agv_1"] = _Job("T_B", "to_dropoff", [Hop("LB", "C0", 0, 2)], idx=0, goal="WB")
    NativeProductionTracker(ops, sim.tasks)

    snap = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=True)
    clf = classify_commitments(snap)
    ev = CVCREvaluator(net, snap, use_rust=False, mode="B1", max_speed=1.0)
    base_assign = {"T_A": "agv_0", "T_B": "agv_1"}
    base_bays = {"OP_A": "WA", "OP_B": "WB"}
    base_crews = {"OP_A": 1, "OP_B": 2}
    # Deliberately give OP_A lower number so incumbent picks OP_A first (causing OP_B to miss [0,11]!)
    base_prio = {"OP_A": 5, "OP_B": 15}
    base_eval = ev.evaluate(base_assign, op_priority=base_prio)
    assert base_eval.valid

    pair = select_coupled_op_pair(net, snap, base_eval, base_assign, base_prio)
    assert pair is not None
    op_a, op_b, pair_meta = pair
    assert {op_a, op_b} == {"OP_A", "OP_B"}
    assert "C0" in pair_meta["shared_bottleneck"] and "C1" in pair_meta["shared_bottleneck"]

    lg = LGCVCRPolicy("V17_EA_JOINT", net, use_rust=False)
    surr = _LazyKittingSurrogate(net, snap, base_prio, 3.0, 15.0)
    pol = V15ExecutionAwarePolicy(weights=None)
    import time as _t
    cands, _gf, meta = generate_v17_dual_op_candidates(
        lg, net, snap, clf, ev, surr, base_assign, base_bays, base_crews, base_eval, base_prio,
        ["agv_0", "agv_1"], {}, {}, pol, _t.perf_counter() + 0.5, 50.0, _t.perf_counter, sim=sim, include_dual_op=True
    )
    digests = [c.action_diff_digest for c in cands]
    assert len(digests) == len(set(digests)), "All emitted candidates must be strictly action-diff deduplicated"
    assert meta["dual_added_distinct"] >= 1

    # Verify pairwise feature signals rescue of OP_B back into [0, 11] window
    best_c = max((c for c in cands if c.eval_res.valid), key=lambda c: c.det_score)
    assert best_c.eval_res.op_start["OP_B"] <= 11
    assert best_c.eval_res.objective < base_eval.objective - 50.0
    assert best_c.pairwise_features is not None and best_c.pairwise_features.shape == (V17_PAIRWISE_DIM,)

    # Index 0..5: delta_S_A, delta_S_B, delta_tard_A, delta_tard_B, cross_A, cross_B
    if op_a == "OP_B":
        assert best_c.pairwise_features[0] < -0.5  # OP_B starts >= 20 steps earlier
        assert best_c.pairwise_features[4] == -1.0  # rescued OP_B across window back into win 0
    else:
        assert best_c.pairwise_features[1] < -0.5
        assert best_c.pairwise_features[5] == -1.0
