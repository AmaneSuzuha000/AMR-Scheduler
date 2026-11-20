"""Stage 1 (E0) unit and reconciliation tests for legal bridge-order control."""

from __future__ import annotations

from pathlib import Path

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.production import NativeProductionTracker
from amr_sched.cvcr.snapshot import NativeSnapshotCapture, OperationSnapshot
from amr_sched.cvcr.v17_e0_audit import run_e0_suite
from amr_sched.cvcr.v17_window_protection import plan_bridge_order_control
from amr_sched.entities import Agv, AgvState, Task, TaskStatus
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import Hop, SimulationConfig, Simulator, _Job


def test_stage1_e0_12_configs_plus_negative_controls(tmp_path: Path) -> None:
    """Verify Stage 1 E0 Go/No-Go gate across 12 controlled bridge configs and 2 negative controls."""
    res = run_e0_suite(output_dir=tmp_path)
    assert (tmp_path / "v17_e0_reconciliation.json").is_file()
    assert (tmp_path / "v17_e0_reconciliation.csv").is_file()
    rows = res["rows"]
    assert len(rows) == 14

    by_id = {r["case_id"]: r for r in rows}
    # 1. Pre-pickup & loaded_at_staging: perfect symmetric flip independent of agv_0/agv_1 IDs, 0 audit fails
    for st in ("pre_pickup", "loaded_at_staging"):
        for perm in ("normal_ids", "swapped_ids"):
            ra = by_id[f"{st}|A_tight_B_loose|{perm}"]
            rb = by_id[f"{st}|B_tight_A_loose|{perm}"]
            assert ra["winner"] == "protect_A"
            assert rb["winner"] == "protect_B"
            assert ra["delta_net_B_minus_A"] > 50.0
            assert rb["delta_net_B_minus_A"] < -50.0
            assert abs(ra["delta_net_B_minus_A"] + rb["delta_net_B_minus_A"]) < 1e-6
            assert ra["protect_A"]["aud_pass"] and ra["protect_B"]["aud_pass"]
            assert rb["protect_A"]["aud_pass"] and rb["protect_B"]["aud_pass"]
            assert ra["protect_A"]["committed"] and ra["protect_B"]["committed"]

    # 2. In-edge cases: cleanly masked & rejected by NativeTransactionOwner, 0 audit fails
    for wo in ("A_tight_B_loose", "B_tight_A_loose"):
        for perm in ("normal_ids", "swapped_ids"):
            ri = by_id[f"in_edge|{wo}|{perm}"]
            assert ri["winner"] == "tie"
            assert ri["delta_net_B_minus_A"] == 0.0
            assert not ri["protect_A"]["eligible"]
            assert not ri["protect_B"]["eligible"]
            assert ri["protect_A"]["fallback_reason"] == "in_edge_irrevocable"
            assert any("VIOLATION_LIVE_REPAIR_IN_TRANSIT" in rsn for rsn in ri["protect_A"]["receipt_reasons"])
            assert ri["protect_A"]["aud_pass"] and ri["protect_B"]["aud_pass"]

    # 3. Negative controls: crossing order changes bridge entrance, but net_obj diff is zero
    r_loose = by_id["neg_loose_windows"]
    assert r_loose["delta_net_B_minus_A"] == 0.0
    assert r_loose["protect_A"]["bridge_enter_A"] < r_loose["protect_A"]["bridge_enter_B"]
    assert r_loose["protect_B"]["bridge_enter_B"] < r_loose["protect_B"]["bridge_enter_A"]

    r_pred = by_id["neg_shared_long_predecessor"]
    assert r_pred["delta_net_B_minus_A"] == 0.0
    assert r_pred["protect_A"]["op_start_A"] == r_pred["protect_A"]["op_start_B"] == 45
    assert r_pred["protect_B"]["op_start_A"] == r_pred["protect_B"]["op_start_B"] == 45


def test_stage1_e0_no_safe_yield_haven_masked() -> None:
    """Verify holding a yielder on a capacity-1 choke directly blocking leader with no siding is masked."""
    # Single linear track: L_START (cap=2) -- C_BLK (cap=1) -- W_END (cap=2), no side spurs
    net = Network()
    net.add_node(Node("L_START", 0.0, 0.0, kind=NodeKind.BUFFER, capacity=2))
    net.add_node(Node("C_BLK", 2.0, 0.0, kind=NodeKind.AISLE, capacity=1))
    net.add_node(Node("W_END", 4.0, 0.0, kind=NodeKind.WORKSTATION, capacity=1))
    net.add_edge(Edge("L_START", "C_BLK", length=2.0, speed_limit=1.0, capacity=1))
    net.add_edge(Edge("C_BLK", "W_END", length=2.0, speed_limit=1.0, capacity=1))

    t1 = Task("T1", "L_START", "W_END", release_step=0, due_step=20)
    t1.kit_op_id = "OP1"  # type: ignore[attr-defined]
    t2 = Task("T2", "C_BLK", "W_END", release_step=0, due_step=40)
    t2.kit_op_id = "OP2"  # type: ignore[attr-defined]
    ops = [
        OperationSnapshot("OP1", ("T1",), "W_END", crew_id=1, tool_id=1, proc_steps=2, due_step=25),
        OperationSnapshot("OP2", ("T2",), "W_END", crew_id=2, tool_id=2, proc_steps=2, due_step=45),
    ]
    sim = Simulator(
        net,
        [t1, t2],
        [Agv("agv_0", max_speed=1.0, start_node="L_START"), Agv("agv_1", max_speed=1.0, start_node="C_BLK")],
        SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=60, idle_retreat=False),
    )
    sim.reset()
    t1.status = TaskStatus.PICKED
    t1.assigned_agv = "agv_0"
    t2.status = TaskStatus.PICKED
    t2.assigned_agv = "agv_1"
    sim.agvs[0].cur_node = "L_START"
    sim.agvs[0].cur_task = "T1"
    sim.agvs[0].state = AgvState.TO_DROPOFF
    sim.agvs[1].cur_node = "C_BLK"
    sim.agvs[1].cur_task = "T2"
    sim.agvs[1].state = AgvState.TO_DROPOFF
    sim._jobs["agv_0"] = _Job("T1", "to_dropoff", [Hop("L_START", "C_BLK", 0, 2), Hop("C_BLK", "W_END", 2, 4)], idx=0, goal="W_END")
    sim._jobs["agv_1"] = _Job("T2", "to_dropoff", [Hop("C_BLK", "W_END", 0, 2)], idx=0, goal="W_END")

    snap = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=True)
    ev = CVCREvaluator(net, snap, use_rust=False, mode="B1", max_speed=1.0)
    plan = plan_bridge_order_control(
        net,
        snap,
        ev,
        {"T1": "agv_0", "T2": "agv_1"},
        {"OP1": 10, "OP2": 20},
        leader_op_id="OP1",
        yielder_op_id="OP2",
        sim=sim,
    )
    assert not plan.eligible
    assert plan.reason == "no_safe_yield_haven"
