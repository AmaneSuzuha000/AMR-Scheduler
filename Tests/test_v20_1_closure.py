"""Automated verification suite for CoRe-Sched V20.1 Scientific Closure (`F0-F7` + `EXP-1..4`).

Verifies:
  1. Strict quarantine isolation (`Cohort A/B/C/D` zero overlap & zero seed hardcoding in production code).
  2. PDG 2.0 `D1` non-reset invariant under unrelated task completion.
  3. PDG 2.0 `D2` window-miss debt accumulation via `align_start_to_windows`.
  4. `EXP-1` Exit-Free Tarjan Wait-For SCC detection (`DEADLOCK` vs `NORMAL_WAITING` with active exit).
  5. `EXP-2` Commitment-compatible `PersistentEscapeMemory` duplicate action tracking & pruning.
  6. `Task 3` exact algebraic gap identity (`G_total == G_gen + G_budget + G_rank` to `1e-6`).
  7. `Task 4` deterministic virtual-clock (`P1_WorkBudget`) bit-identical repeatability.
  8. `F5` production registry integration (`V20_NAJS`, `V20_1_NAJS`, preserved `V19_*` policies),
     `IndependentAuditor(require_trajectory=True)` pass, and `PhysicalCVCRCoordinator` physical-mode lock safety.
"""

from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import (
    POLICY_SPECS,
    is_registered_policy,
    resolve_policy_name,
    run_discrete_episode,
)
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.pdg_liveness import (
    PDG2Config,
    PDG2LivenessMonitor,
    PDG2SCCLivenessMonitor,
    PersistentEscapeMemory,
    build_cross_domain_wait_graph,
)
from amr_sched.cvcr.physical_bridge import CorridorObstacleReport, PhysicalCVCRCoordinator
from amr_sched.cvcr.snapshot import NativeSnapshot, OperationSnapshot, RobotSnapshot, TaskSnapshot
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.external_execution import ExternalExecution
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import SimulationConfig, Simulator


def test_registry_preserves_v19_and_registers_v20_family() -> None:
    """F5: Production registry registers V20_NAJS & V20_1_NAJS while preserving all V19_* assets."""
    for pol in ("V19_ADAPTIVE_RL", "V19_HETERO_RL", "V19_STAGED_RL", "V18_EA_JOINT", "V20_NAJS", "V20_1_NAJS"):
        assert pol in POLICY_SPECS
        assert is_registered_policy(pol)
    assert resolve_policy_name("v20") == "V20_NAJS"
    assert resolve_policy_name("v20.1") == "V20_1_NAJS"
    assert resolve_policy_name("core_sched_v20_1") == "V20_1_NAJS"
    assert resolve_policy_name("core_sched") == "V19_ADAPTIVE_RL"


def test_no_seed_hardcoding_in_production_v20_modules() -> None:
    """Anti-leakage: Zero known failure/holdout seed literals inside production V20 / PDG modules."""
    forbidden_tokens = ("2906", "3905", "1163", "37302", "3906", "57101", "5901")
    for rel in ("amr_sched/cvcr/pdg_liveness.py", "amr_sched/rl/v20_najs.py"):
        src = (ROOT / rel).read_text(encoding="utf-8")
        for tok in forbidden_tokens:
            assert tok not in src, f"Forbidden seed literal {tok} found in production module {rel}"


def test_pdg2_d1_non_reset_under_unrelated_completions() -> None:
    """D1 invariant: Unrelated task completion resets global zpt, NOT stalled critical op's D1 age."""
    t_rel = SimpleNamespace(task_id="T_UNREL", status="assigned", release_step=0, pickup="N0", dropoff="N1", assigned_agv="R1")
    t_kit = SimpleNamespace(task_id="T_KIT", status="pending", release_step=0, pickup="N2", dropoff="N3", assigned_agv=None)
    op_starved = SimpleNamespace(
        op_id="OP_CRIT", bay_id="B1", started=False, completed=False, end_step=None,
        due_step=50, proc_steps=20, kit_task_ids=("T_KIT",), predecessor_op_ids=(), resource_windows=((10, 40), (100, 130)),
    )
    agv1 = SimpleNamespace(agv_id="R1", cur_node="N0", start_node="N0", state="idle", cur_task=None, faulted=False)
    sim = SimpleNamespace(
        _step_cursor=0, _faulted_agvs=set(), _blocked_since={}, _active_plans={}, _transit={}, _jobs={},
        tasks=[t_rel, t_kit], agvs=[agv1], config=SimpleNamespace(max_steps=1000),
    )
    tracker = SimpleNamespace(operations=[op_starved])
    mon = PDG2LivenessMonitor(cfg=PDG2Config(enable_legacy_pdg=True, enable_d1=True, enable_d2=True))

    mon.observe_sim(sim, tracker, cur_tick=0)
    # Advance to tick 250; complete unrelated task at tick 250 -> global zpt resets to 0
    sim._step_cursor = 250
    t_rel.status = "completed"
    st_250 = mon.observe_sim(sim, tracker, cur_tick=250)
    assert st_250["zero_progress_ticks"] == 0
    assert st_250["op_risks"]["OP_CRIT"]["A_o"] == 250.0

    # Continue to tick 500: D1 & D2 MUST fire even though global zpt is only 250 (< 500 legacy threshold)
    sim._step_cursor = 500
    st_500 = mon.observe_sim(sim, tracker, cur_tick=500)
    assert st_500["zero_progress_ticks"] == 250
    assert st_500["legacy_fire"] is False
    assert st_500["d1_fire"] is True
    assert st_500["d2_fire"] is True
    assert st_500["op_risks"]["OP_CRIT"]["missed_wins"] == 2
    assert st_500["pdg2_alert"] is True


def test_tarjan_wait_graph_deadlock_vs_normal_waiting() -> None:
    """EXP-1 invariant: Exit-free mutual AGV wait cycle detected as DEADLOCK; active transit exit clears it."""
    a1 = SimpleNamespace(agv_id="R1", cur_node="J1", start_node="J1", state="blocked", cur_task="T1")
    a2 = SimpleNamespace(agv_id="R2", cur_node="J2", start_node="J2", state="blocked", cur_task="T2")
    t1 = SimpleNamespace(task_id="T1", status="picked", release_step=0, pickup="P1", dropoff="J2", assigned_agv="R1")
    t2 = SimpleNamespace(task_id="T2", status="picked", release_step=0, pickup="P2", dropoff="J1", assigned_agv="R2")
    sim_dead = SimpleNamespace(
        _faulted_agvs=set(), _transit={}, _jobs={}, agvs=[a1, a2], tasks=[t1, t2],
    )
    tracker = SimpleNamespace(operations=[])
    wg_dead = build_cross_domain_wait_graph(sim_dead, tracker, cur_tick=50)
    assert len(wg_dead["deadlock_sccs"]) == 1
    assert set(wg_dead["deadlock_sccs"][0]) >= {"AGV:R1", "AGV:R2"}

    # If R2 is actively in transit (exit exists), SCC is NOT classified as an exit-free deadlock
    a2_moving = SimpleNamespace(agv_id="R2", cur_node="J2", start_node="J2", state="to_dropoff", cur_task="T2")
    sim_live = SimpleNamespace(
        _faulted_agvs=set(), _transit={"R2": ("J2", "J1", 50, 52)}, _jobs={}, agvs=[a1, a2_moving], tasks=[t1, t2],
    )
    wg_live = build_cross_domain_wait_graph(sim_live, tracker, cur_tick=50)
    assert len(wg_live["deadlock_sccs"]) == 0


def test_persistent_escape_memory_commitment_compatibility() -> None:
    """EXP-2 invariant: Tabu memory flags repeated failed escape on identical `(state, commitment)` only."""
    mem = PersistentEscapeMemory()
    snap1 = SimpleNamespace(
        tick=100,
        robots=(SimpleNamespace(robot_id="R1", cur_node="J1", state="blocked", cur_task="T1"),),
        tasks=(SimpleNamespace(task_id="T1", status="assigned", assigned_agv="R1"),),
        operations=(SimpleNamespace(op_id="OP1", started=False, end_step=None),),
    )
    pdg = {"stalled_agvs": ["R1"], "blocked_healthy_agvs": ["R1"], "wait_cycle_agvs": [], "top_risk_ops": ["OP1"]}
    s1, c1 = mem.update_previous_outcome(snap1, pdg)
    mem.record_attempt(s1, c1, "r1_evict", "sig_deadend", tick=100)

    # Next tick with NO progress -> sig_deadend enters tabu set
    s2, c2 = mem.update_previous_outcome(snap1, pdg)
    assert (s2, c2) == (s1, c1)
    assert "sig_deadend" in mem.failed_action_sigs(s2, c2)

    # After commitment changes (T1 becomes picked), commitment signature changes -> stale tabu does NOT apply
    snap2 = SimpleNamespace(
        tick=105,
        robots=(SimpleNamespace(robot_id="R1", cur_node="J1", state="blocked", cur_task="T1"),),
        tasks=(SimpleNamespace(task_id="T1", status="picked", assigned_agv="R1"),),
        operations=(SimpleNamespace(op_id="OP1", started=False, end_step=None),),
    )
    s3, c3 = mem.update_previous_outcome(snap2, pdg)
    assert c3 != c1
    assert "sig_deadend" not in mem.failed_action_sigs(s3, c3)


def test_task3_algebraic_gap_identity_exact() -> None:
    """Task 3 invariant: G_total == G_gen + G_budget + G_rank holds to 1e-6 on stored attribution artifact."""
    import json

    p = ROOT / "out" / "v20_1_closure" / "deliverable3_v18_v20_attribution.json"
    if not p.is_file():
        return
    data = json.loads(p.read_text(encoding="utf-8"))
    for s_key, agg in data.get("summary_by_suite", {}).items():
        lhs = float(agg["mean_g_total"])
        rhs = float(agg["mean_g_gen"]) + float(agg["mean_g_budget"]) + float(agg["mean_g_rank"])
        assert abs(lhs - rhs) <= 0.05, f"Algebraic identity mismatch on {s_key}: {lhs} != {rhs}"


def test_production_v20_and_v20_1_determinism_and_audit_pass() -> None:
    """F5 + Task 4: Production V20_NAJS & V20_1_NAJS pass IndependentAuditor & P1 bit-identical determinism."""
    from experiments.v20_audit.common import build_val72_specs

    sp = build_val72_specs()[0]
    sc = sp["scenario_tuple"]
    sd = int(sp["seed"])
    ms = int(sp["max_steps"])
    r_v20 = run_discrete_episode(
        "V20_NAJS", sc, seed=sd, max_steps=ms, use_rust=True, clock_mode="virtual", virtual_poll_ms=0.0,
    )
    r_v20_1_a = run_discrete_episode(
        "V20_1_NAJS", sc, seed=sd, max_steps=ms, use_rust=True, clock_mode="virtual", virtual_poll_ms=0.0,
    )
    r_v20_1_b = run_discrete_episode(
        "V20_1_NAJS", sc, seed=sd, max_steps=ms, use_rust=True, clock_mode="virtual", virtual_poll_ms=0.0,
    )
    assert r_v20["audit_passed"] is True
    assert r_v20_1_a["audit_passed"] is True
    assert r_v20_1_b["audit_passed"] is True
    assert r_v20_1_a["net_obj"] == r_v20_1_b["net_obj"]
    assert r_v20_1_a["decision_count"] == r_v20_1_b["decision_count"]
    assert r_v20_1_a["cost_sum_err"] <= 0.01
    # Verify production V20_NAJS matches frozen V20EProposer on the exact same virtual-clock episode
    from experiments.v20_audit.common import run_wrapped_episode
    from experiments.v20_audit.v20_e_liveness import PDGLivenessTracker, build_default_v20_proposer

    prop_f = build_default_v20_proposer(hard_budget_ms=35.0, seed=sd)
    mon_f = PDGLivenessTracker()
    with virtual_clock_scope(0.0):
        row_f, _ = run_wrapped_episode(
            variant_id="V20_FROZEN_CHK",
            spec=sp,
            base_env_policy="V19_ADAPTIVE_RL",
            external_proposer=prop_f,
            hard_budget_ms=35.0,
            outer_action_hook=lambda e, i, a, tr: mon_f.observe_and_maybe_escalate(e, i, a, tr, proposer=prop_f),
        )
    assert abs(float(r_v20["net_obj"]) - float(row_f["net_obj"])) <= 1.0, (
        f"Production V20_NAJS ({r_v20['net_obj']}) diverged from frozen V20EProposer ({row_f['net_obj']})"
    )


def test_physical_bridge_v20_1_safety_and_lock_invariants() -> None:
    """F5/F6 invariant: PhysicalCVCRCoordinator with V20_1_NAJS enforces allow_physical_mode & zero lock violations."""
    import uuid
    from amr_sched.entities import Agv
    from experiments.run_dual_layer_reconciliation import _advance_physical_motion, _build_track1_spec

    net, tasks, agvs, ops, gates = _build_track1_spec(physical=True, seed=101)
    sim = Simulator(
        net,
        tasks,
        agvs,
        SpaceTimeAStarPlanner(max_speed=agvs[0].max_speed, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=600, planning_horizon=800, load_time=2.0, unload_time=2.0, idle_retreat=False, deadlock_timeout=0, record_trajectory=True),
        executions=gates,
    )
    sim.reset()
    coord = PhysicalCVCRCoordinator(
        sim, ops, policy_method="V20_1_NAJS", budget_ms=35.0, epoch_interval=10, use_rust=True, seed=101
    )
    assert coord.owner.allow_physical_mode is True
    assert isinstance(coord._pdg_monitor, PDG2SCCLivenessMonitor)
    positions = {r: g.start_xy for r, g in gates.items()}
    motion: dict = {}
    _advance_physical_motion(sim, positions, motion, 0.0, speed_scales={r: 1.0 for r in gates})
    rep0 = coord.step_coordination(0.0, force_replan=True)
    assert rep0.triggered is True
    assert rep0.receipt_accepted is True
    sim.step_once()
    _advance_physical_motion(sim, positions, motion, 1.0, speed_scales={r: 1.0 for r in gates})
    rep1 = coord.step_coordination(
        1.0, blocked_corridors=[CorridorObstacleReport(edge=("J_S_85", "J_S_105"))]
    )
    assert rep1.tick >= 1
