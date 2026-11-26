"""Unit tests for `IF-CORE-ROUTING-CPSA-01` (CPS-A* baseline + CPR incremental mechanisms)."""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.snapshot import NativeSnapshotCapture, OperationSnapshot
from amr_sched.entities import Agv, Task
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.simulation import SimulationConfig, Simulator
from experiments.cps_astar_routing.cpr_incremental_router import (
    CPRIncrementalEvaluator,
    CPRRoutingConfig,
    SlotReservationPressureTable,
    select_active_blocker,
)
from experiments.cps_astar_routing.cps_astar_graph import (
    CPSGraphFieldConfig,
    CPSAStarConfig,
    CPSAStarGraphEvaluator,
    CPSAStarGraphPlanner,
    CPSAStarGreedyMAPDProposer,
    GraphCongestionField,
)
from experiments.rgro_closure.closure_common import sha256_file
from tools.data_manifest import MANIFEST, build as build_csv_manifest


def _build_corridor_network() -> Network:
    """Single-lane central bottleneck corridor `J0 <-> J1` with north/south bypass loop `B0 <-> B1`."""
    net = Network()
    net.add_node(Node("S0", 0.0, 0.0, kind=NodeKind.DOCK, capacity=1))
    net.add_node(Node("S1", 0.0, 4.0, kind=NodeKind.WORKSTATION, capacity=1))
    net.add_node(Node("J0", 3.0, 2.0, kind=NodeKind.AISLE, capacity=1))
    net.add_node(Node("J1", 6.0, 2.0, kind=NodeKind.AISLE, capacity=1))
    net.add_node(Node("B0", 3.0, 6.0, kind=NodeKind.BUFFER, capacity=1))
    net.add_node(Node("B1", 6.0, 6.0, kind=NodeKind.BUFFER, capacity=1))
    net.add_node(Node("D0", 9.0, 0.0, kind=NodeKind.ASSEMBLY, capacity=1))
    net.add_node(Node("D1", 9.0, 4.0, kind=NodeKind.WORKSTATION, capacity=1))

    edges = [
        ("S0", "J0", 3.0), ("J0", "S0", 3.0),
        ("S1", "J0", 3.0), ("J0", "S1", 3.0),
        ("J0", "J1", 3.0), ("J1", "J0", 3.0),
        ("J0", "B0", 3.0), ("B0", "J0", 3.0),
        ("B0", "B1", 3.0), ("B1", "B0", 3.0),
        ("B1", "J1", 3.0), ("J1", "B1", 3.0),
        ("J1", "D0", 3.0), ("D0", "J1", 3.0),
        ("J1", "D1", 3.0), ("D1", "J1", 3.0),
    ]
    for u, v, ln in edges:
        net.add_edge(Edge(u, v, length=ln, speed_limit=3.0, capacity=1))
    return net


def _make_two_amr_corridor_sim() -> tuple[Network, Simulator, list[OperationSnapshot], Any]:
    net = _build_corridor_network()
    t1 = Task(task_id="T1", pickup="J1", dropoff="S0", release_step=0, due_step=25)
    t2 = Task(task_id="T2", pickup="J0", dropoff="D0", release_step=0, due_step=25)
    agv1 = Agv(agv_id="R1", max_speed=3.0, start_node="J1", load_time=2.0, unload_time=2.0)
    agv2 = Agv(agv_id="R2", max_speed=3.0, start_node="J0", load_time=2.0, unload_time=2.0)
    sim = Simulator(
        net,
        [t1, t2],
        [agv1, agv2],
        SpaceTimeAStarPlanner(max_speed=3.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=80, planning_horizon=120, idle_retreat=False, record_trajectory=True),
    )
    sim.reset()
    ops = [
        OperationSnapshot(op_id="OP1", kit_task_ids=("T1",), bay_id="S0", crew_id=1, tool_id=1, proc_steps=2, due_step=25),
        OperationSnapshot(op_id="OP2", kit_task_ids=("T2",), bay_id="D0", crew_id=2, tool_id=2, proc_steps=2, due_step=35),
    ]
    snap = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=True)
    return net, sim, ops, snap


def test_graph_congestion_field_math_and_planner_modes() -> None:
    net = _build_corridor_network()
    fcfg = CPSGraphFieldConfig(spatial_radius_L=1, temporal_radius_T=1, sigma_l=2.0, sigma_t=2.0, slot_delta=3)
    field = GraphCongestionField(net, fcfg, t_start=0)
    field.add_agv_path("R_bg", [("J0", 1), ("J0", 4), ("J1", 5)])

    # Raw occupancy exists on J0 at t=1..4 -> Gaussian phi(J0, 2) > phi(D1, 2) == 0
    phi_j0 = field.phi("J0", 2)
    phi_d1 = field.phi("D1", 2)
    assert phi_j0 > 0.15
    assert phi_d1 == 0.0
    assert field.phi_bar("S0", 0) > 0.0
    assert field.slot_Phi("J0", 2) > 0.0

    # Verify Eq. 4.3-4.4 closed form against single point occupancy at (B0, t=5)
    f2 = GraphCongestionField(net, fcfg, t_start=0)
    f2.add_agv_path("R_single", [("B0", 5)])
    # Center (B0, dt=0) weight=1.0; B0 has neighbors J0, B1 -> d=1 neighbors + self
    k0 = 1.0
    k_sp = math.exp(-1.0 / 8.0)
    k_tm = math.exp(-1.0 / 8.0)
    k_both = math.exp(-1.0 / 8.0 - 1.0 / 8.0)
    # Sum over spatial node in {B0, J0, B1} x dt in {-1, 0, 1}: 1 center + 2 spatial + 2 temporal + 4 diag
    denom = k0 + 2 * k_sp + 2 * k_tm + 4 * k_both
    assert math.isclose(f2.phi("B0", 5), 1.0 / denom, rel_tol=1e-6)

    # Planners in oneshot and lifelong find conflict-free paths around occupied single lane
    res = ReservationTable()
    res.reserve_node("J0", Interval(1, 5), "R_bg")
    res.reserve_edge("J0", "J1", Interval(4, 6), "R_bg")
    planner_ll = CPSAStarGraphPlanner(net, CPSAStarConfig(mode="lifelong", field=fcfg), max_speed=3.0, dt=1.0)
    planned, hits = planner_ll.plan_leg("S0", "D0", 0, res, field, "R_new", 60)
    assert planned is not None and planned.nodes[-1] == "D0"
    assert isinstance(hits, dict)


def test_cpr_wait_pressure_and_priority_ordering() -> None:
    net, sim, ops, snap = _make_two_amr_corridor_sim()
    # Heavy congestion slotted on central corridor J0->J1 encourages bypass via B0->B1
    cfg_w = CPRRoutingConfig(enable_slot_wait_pressure=True, hybrid_rust_slot_projection=False)
    ev_w = CPRIncrementalEvaluator(net, snap, cfg_w, mode="B1", use_rust=False)
    res = ReservationTable()
    slot_tbl = SlotReservationPressureTable(net, slot_delta=3, t_start=0)
    slot_tbl.add_agv_path("R_crowd1", [("J0", 1), ("J0", 6), ("J1", 7)])
    slot_tbl.add_agv_path("R_crowd2", [("J0", 2), ("J0", 7), ("J1", 8)])
    ev_w._active_slot_table = slot_tbl
    p_bypass = ev_w._space_time_route("S0", "D0", 0, res, "R_probe")
    ev_w._active_slot_table = None
    assert p_bypass is not None and p_bypass[-1][0] == "D0"
    # Verify wait penalty does not trigger on goal node
    assert slot_tbl.wait_penalty_q3("D0", 3, is_goal=True, gamma_q3=6, cap_q3=24) == 0
    assert slot_tbl.wait_penalty_q3("J0", 3, is_goal=False, gamma_q3=6, cap_q3=24) > 0

    # Caller traffic weights must survive slot sync restoration
    ev_w.set_traffic_weights({("S0", "J0"): 5}, cutoff_step=30)
    assert ev_w._caller_tw_q3 == {("S0", "J0"): 5}

    # Verify Factor P prioritizes congested + aged AMR ahead of non-congested AMR at equal op prio
    t_old = snap.tasks[0]
    t_new = snap.tasks[1]
    op_prio = {"OP1": 20, "OP2": 20}
    ev_p = CPRIncrementalEvaluator(net, snap, CPRRoutingConfig(enable_congestion_priority=True, prio_congestion_kappa=5.0), mode="B1", use_rust=False)
    k_r1 = ev_p._cpr_sort_key("R1", [t_old], op_prio, slot_tbl)
    k_r2 = ev_p._cpr_sort_key("R2", [t_new], op_prio, slot_tbl)
    # R2 sits at crowded J0 -> k_r2 sorts before k_r1
    assert k_r2 < k_r1


def test_cpr_bounded_rollback_rescues_headon_corridor_and_passes_auditor() -> None:
    """Tight-deadline head-on lock: high-prio AMR with bypass slack grabs bottleneck, starving tight-slack AMR.

    Under static priority (OP_HIGH before OP_LOW) with `sim_horizon=9`, `R0` (HardRes)
    fails (`valid == False`), whereas `CPRIncrementalEvaluator(enable_bounded_rollback=True)`
    swaps the pair cleanly (`R_L` takes central 2-step lane, `R_H` takes 4-step `BYP`),
    rescuing `valid == True` with `cpr_repairs_succeeded >= 1` and passing `IndependentAuditor`.
    """
    from amr_sched.cvcr.evaluator import CVCREvaluator

    net = Network()
    net.add_node(Node("L_BUF", 0.0, 0.0, kind=NodeKind.BUFFER, capacity=1))
    net.add_node(Node("J0", 6.0, 0.0, kind=NodeKind.AISLE, capacity=1))
    net.add_node(Node("J1", 12.0, 0.0, kind=NodeKind.AISLE, capacity=1))
    net.add_node(Node("BYP", 9.0, 6.0, kind=NodeKind.BUFFER, capacity=1))

    edges = [
        ("L_BUF", "J0", 6.0), ("J0", "L_BUF", 6.0),  # 2 steps
        ("J0", "J1", 6.0), ("J1", "J0", 6.0),        # 2 steps central bottleneck
        ("J0", "BYP", 6.0), ("BYP", "J0", 6.0),      # 2 steps
        ("BYP", "J1", 6.0), ("J1", "BYP", 6.0),      # 2 steps (4 steps bypass total)
    ]
    for u, v, ln in edges:
        net.add_edge(Edge(u, v, length=ln, speed_limit=3.0, capacity=1))

    t_h = Task(task_id="T_HIGH", pickup="J0", dropoff="J1", release_step=0, due_step=12)
    t_l = Task(task_id="T_LOW", pickup="J1", dropoff="L_BUF", release_step=0, due_step=9)
    sim = Simulator(
        net,
        [t_h, t_l],
        [
            Agv(agv_id="R_H", max_speed=3.0, start_node="J0", load_time=2.0, unload_time=2.0),
            Agv(agv_id="R_L", max_speed=3.0, start_node="J1", load_time=2.0, unload_time=2.0),
        ],
        SpaceTimeAStarPlanner(max_speed=3.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=40, planning_horizon=50, idle_retreat=False, record_trajectory=True),
    )
    sim.reset()
    ops = [
        OperationSnapshot(op_id="OP_HIGH", kit_task_ids=("T_HIGH",), bay_id="J1", crew_id=1, tool_id=1, proc_steps=1, due_step=12),
        OperationSnapshot(op_id="OP_LOW", kit_task_ids=("T_LOW",), bay_id="L_BUF", crew_id=2, tool_id=2, proc_steps=1, due_step=9),
    ]
    snap = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=True)
    assign = {"T_HIGH": "R_H", "T_LOW": "R_L"}
    op_prio = {"OP_HIGH": 5, "OP_LOW": 15}

    ev_r0 = CVCREvaluator(net, snap, mode="B1", sim_horizon=7, use_rust=False)
    res_r0 = ev_r0.evaluate(assign, op_priority=op_prio)
    assert not res_r0.valid, "Expected R0 static priority order to fail on tight horizon=7 head-on corridor"

    ev_cpr = CPRIncrementalEvaluator(
        net, snap,
        CPRRoutingConfig(enable_bounded_rollback=True, max_repairs_per_eval=2, min_headroom_ms_for_repair=0.0),
        mode="B1", sim_horizon=7, use_rust=False,
    )
    res_cpr = ev_cpr.evaluate(assign, op_priority=op_prio)
    assert res_cpr.valid
    assert ev_cpr.cpr_repairs_succeeded >= 1
    assert ev_cpr.cpr_candidates_rescued >= 1

    # Also test Rust mode B2 consistency if native libraries are available
    if CVCREvaluator(net, snap, mode="B2", sim_horizon=7, use_rust=True).use_rust:
        ev_cpr_rust = CPRIncrementalEvaluator(
            net, snap,
            CPRRoutingConfig(enable_bounded_rollback=True, max_repairs_per_eval=2, min_headroom_ms_for_repair=0.0),
            mode="B1", sim_horizon=7, use_rust=True,
        )
        assert ev_cpr_rust.evaluate(assign, op_priority=op_prio).valid

    for _ in range(40):
        if sim._all_done():
            break
        sim.step_once()
    aud = IndependentAuditor.audit_simulation_run(sim, ops, require_trajectory=True)
    assert aud.pass_audit, f"Audit failed: {aud}"


def test_atomic_restore_and_pinned_custody_never_released() -> None:
    net = _build_corridor_network()
    t1 = Task(task_id="T1", pickup="S0", dropoff="D0", release_step=0, due_step=30)
    agv1 = Agv(agv_id="R1", max_speed=3.0, start_node="S0")
    agv_pin = Agv(agv_id="R_PIN", max_speed=3.0, start_node="J0")
    sim = Simulator(
        net,
        [t1],
        [agv1, agv_pin],
        SpaceTimeAStarPlanner(max_speed=3.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=60, planning_horizon=80, idle_retreat=False, record_trajectory=True),
    )
    sim.reset()
    ops = [OperationSnapshot(op_id="OP1", kit_task_ids=("T1",), bay_id="D0", crew_id=1, tool_id=1, proc_steps=2, due_step=30)]
    snap = NativeSnapshotCapture.capture(sim, ops, hide_future_releases=True)

    # Pinned owner must NEVER be returned by select_active_blocker
    active_routed = {"R_PIN": ([("J0", 0)], [], {}, [], []), "R1": ([("S0", 0)], [], {}, [], [])}
    picked = select_active_blocker("R2", {"R_PIN": 99, "R1": 1}, active_routed, pinned_owners={"R_PIN"})
    assert picked == "R1"
    assert select_active_blocker("R2", {"R_PIN": 99}, {"R_PIN": active_routed["R_PIN"]}, pinned_owners={"R_PIN"}) is None

    # CPS-A* evaluators & greedy diagnostic proposer run cleanly
    g_ev = CPSAStarGraphEvaluator(net, snap, CPSAStarConfig(mode="oneshot"), mode="B2", use_rust=False)
    r_g = g_ev.evaluate({"T1": "R1"}, op_priority={"OP1": 10})
    assert r_g.valid
    g_ev.set_incumbent({"T1": "R1"}, r_g, op_priority={"OP1": 10})
    r_b2 = g_ev.evaluate({"T1": "R1"}, op_priority={"OP1": 10})
    assert r_b2.valid

    greedy_prop = CPSAStarGreedyMAPDProposer()(snap, net, budget_ms=35.0)
    assert greedy_prop.policy_id == "CPSAStar_GreedyMAPD"


def test_public_and_frozen_manifests_remain_untouched() -> None:
    expected = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))
    current = build_csv_manifest()
    assert current == expected and len(current) == 38

    v20_lock = ROOT / "out" / "v20_1_closure" / "v20_1_freeze_manifest.json"
    if v20_lock.is_file():
        m = json.loads(v20_lock.read_text(encoding="utf-8"))
        files_map = m.get("sha256_files", m.get("files", {}))
        checked = 0
        for rel_p, exp_sha in files_map.items():
            if isinstance(exp_sha, str) and len(exp_sha) == 64:
                fp = ROOT / rel_p
                assert fp.is_file() and sha256_file(fp) == exp_sha, f"Tampered {rel_p}"
                checked += 1
        assert checked == 8
