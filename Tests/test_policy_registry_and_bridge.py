"""Tests for unified policy registry and PhysicalCVCRCoordinator pluggability."""

from __future__ import annotations

from pathlib import Path
import uuid

import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.baselines.policy_registry import (
    DEFAULT_12_POLICIES,
    SUPPORTED_POLICIES,
    build_external_proposer,
    is_external_policy,
    resolve_policy_name,
    run_discrete_episode,
)
from amr_sched.cvcr.physical_bridge import PhysicalCVCRCoordinator
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.entities import Agv, MaterialType, Task, TaskPriority
from amr_sched.external_execution import ExternalExecution
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import WorkshopScene
from tools.live_bridge.fleet_config import point_execution_network
from tools.wheelset_scene.workshop_v2_layout import build_improved_scene, build_v2_nav_graph

ROOT = Path(__file__).resolve().parents[1]


def _make_track1_physical_setup():
    scene = WorkshopScene.load(ROOT / "scenes" / "rail_freight_workshop_v1.json")
    net = point_execution_network(scene.network())
    starts = {
        "R1": "J_S_65",
        "R2": "J_N_45",
        "R3": "J_N_85",
        "R4": "W_DOOR",
        "R5": "W_COUPLER",
        "R6": "W_PARK",
    }
    tasks = [
        Task("K_BG_1", "Q_BOGIE", "D_BOGIE", release_step=0, due_step=1200, material_type=MaterialType.BOGIE),
        Task("K_WS_1", "J_S_65", "D_WHEELSET", release_step=0, due_step=2400, material_type=MaterialType.WHEELSET),
        Task("K_BR_1", "J_N_85", "D_BRAKE", release_step=0, due_step=1500, material_type=MaterialType.BRAKE_VALVE),
        Task("K_CP_1", "R_COUPLER", "D_COUPLER", release_step=0, due_step=1800, material_type=MaterialType.BRAKE_VALVE),
        Task("K_DR_1", "R_DOOR", "D_DOOR", release_step=0, due_step=2000, material_type=MaterialType.BRAKE_VALVE),
        Task("K_AS_1", "J_S_125", "D_ASSEMBLY", release_step=0, due_step=2600, priority=TaskPriority.URGENT, material_type=MaterialType.WHEELSET),
    ]
    ops = [
        OperationSnapshot("OP_BOGIE", bay_id="D_BOGIE", crew_id=1, tool_id=1, proc_steps=15, due_step=1500, kit_task_ids=("K_BG_1",)),
        OperationSnapshot("OP_WHEELSET", bay_id="D_WHEELSET", crew_id=2, tool_id=2, proc_steps=15, due_step=2600, predecessor_op_id="OP_BOGIE", kit_task_ids=("K_WS_1",)),
        OperationSnapshot("OP_BRAKE", bay_id="D_BRAKE", crew_id=3, tool_id=3, proc_steps=15, due_step=1800, kit_task_ids=("K_BR_1",)),
        OperationSnapshot("OP_COUPLER", bay_id="D_COUPLER", crew_id=4, tool_id=4, proc_steps=15, due_step=2200, eligible_bays=("D_COUPLER", "D_DOOR"), kit_task_ids=("K_CP_1",)),
        OperationSnapshot("OP_DOOR", bay_id="D_DOOR", crew_id=5, tool_id=5, proc_steps=15, due_step=2400, kit_task_ids=("K_DR_1",)),
        OperationSnapshot("OP_ASSEMBLY", bay_id="D_ASSEMBLY", crew_id=6, tool_id=6, proc_steps=20, due_step=2800, predecessor_op_id="OP_WHEELSET", is_urgent=True, kit_task_ids=("K_AS_1",)),
    ]
    run_id = uuid.uuid4().hex
    gates = {
        rid: ExternalExecution(
            "rail_freight_workshop_v1",
            rid,
            (net.nodes[s].x, net.nodes[s].y),
            source="localized",
            run_id=run_id,
            namespace_routes=True,
            arrival_tolerance=0.15,
            timeout=2.0,
        )
        for rid, s in starts.items()
    }
    agvs = [Agv(rid, max_speed=0.2, start_node=s, load_time=0.2, unload_time=0.2) for rid, s in starts.items()]
    sim = Simulator(
        net,
        tasks,
        agvs,
        SpaceTimeAStarPlanner(max_speed=0.2, dt=0.1),
        GreedyAllocator(),
        SimulationConfig(
            dt=0.1,
            max_steps=2000,
            planning_horizon=1200,
            load_time=0.2,
            unload_time=0.2,
            idle_retreat=False,
            deadlock_timeout=0,
            reschedule_policy=ReschedulePolicy.NONE,
            record_trajectory=True,
        ),
        executions=gates,
    )
    sim.reset()
    return sim, ops


def _make_track2_physical_setup():
    scene = WorkshopScene(build_v2_nav_graph(build_improved_scene()))
    net = point_execution_network(scene.network())
    starts = {"AGV_1": "J_S_65", "AGV_2": "J_S_90", "AGV_3": "J_S_142"}
    tasks = [
        Task("T_01_TD_INSP", pickup="D_TEARDOWN_0", dropoff="D_INSPECTION_0", release_step=0, due_step=260),
        Task("T_02_INSP_PREP", pickup="D_INSPECTION_1", dropoff="D_PREP_IN", release_step=0, due_step=280),
        Task("T_03_PREP_PRESS", pickup="D_PREP_OUT", dropoff="D_PRESS_IN", release_step=0, due_step=260),
        Task("T_04_PRESS_CLOSE", pickup="D_PRESS_OUT", dropoff="D_CLOSING_IN", release_step=0, due_step=220),
        Task("T_05_FIN_STORE_A", pickup="D_FINISHING_0", dropoff="D_STORAGE", release_step=0, due_step=280),
        Task("T_06_FIN_STORE_B", pickup="D_FINISHING_1", dropoff="D_UNKNOWN", release_step=0, due_step=340),
    ]
    ops = [
        OperationSnapshot("OP_V2_INSP", ("T_01_TD_INSP",), "D_INSPECTION_0", crew_id=1, tool_id=1, proc_steps=5, due_step=280),
        OperationSnapshot("OP_V2_PREP", ("T_02_INSP_PREP",), "D_PREP_IN", crew_id=2, tool_id=2, proc_steps=5, due_step=320, predecessor_op_id="OP_V2_INSP"),
        OperationSnapshot("OP_V2_PRESS", ("T_03_PREP_PRESS",), "D_PRESS_IN", crew_id=3, tool_id=3, proc_steps=5, due_step=360, predecessor_op_id="OP_V2_PREP"),
        OperationSnapshot("OP_V2_CLOSE", ("T_04_PRESS_CLOSE",), "D_CLOSING_IN", crew_id=2, tool_id=2, proc_steps=5, due_step=400, predecessor_op_id="OP_V2_PRESS"),
        OperationSnapshot("OP_V2_STORE", ("T_05_FIN_STORE_A",), "D_STORAGE", crew_id=1, tool_id=4, proc_steps=5, due_step=480, predecessor_op_id="OP_V2_CLOSE"),
        OperationSnapshot("OP_V2_VERIFY", ("T_06_FIN_STORE_B",), "D_UNKNOWN", crew_id=3, tool_id=5, proc_steps=5, due_step=520, predecessor_op_id="OP_V2_STORE"),
    ]
    run_id = uuid.uuid4().hex
    gates = {
        rid: ExternalExecution(
            "wheelset_workshop_v2",
            rid,
            (net.nodes[s].x, net.nodes[s].y),
            source="localized",
            run_id=run_id,
            namespace_routes=True,
            arrival_tolerance=0.25,
            timeout=2.0,
        )
        for rid, s in starts.items()
    }
    agvs = [Agv(rid, max_speed=0.65, start_node=s, load_time=2.0, unload_time=2.0) for rid, s in starts.items()]
    sim = Simulator(
        net,
        tasks,
        agvs,
        SpaceTimeAStarPlanner(max_speed=0.65, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(
            dt=1.0,
            max_steps=800,
            planning_horizon=1000,
            load_time=2.0,
            unload_time=2.0,
            idle_retreat=False,
            deadlock_timeout=0,
            record_trajectory=True,
        ),
        executions=gates,
    )
    sim.reset()
    return sim, ops


def test_policy_registry_aliases_and_coverage():
    assert len(DEFAULT_12_POLICIES) == 12
    assert len(SUPPORTED_POLICIES) == 19
    assert resolve_policy_name("v19_bc") == "V19_HETERO_BC"
    assert resolve_policy_name("v19_rl") == "V19_HETERO_RL"
    assert resolve_policy_name("v19_staged") == "V19_STAGED_RL"
    assert resolve_policy_name("v19_no_prod_comm") == "V19_NO_PROD_COMM"
    assert resolve_policy_name("v19_adaptive_rl") == "V19_ADAPTIVE_RL"
    assert resolve_policy_name("ahg_rl") == "V19_ADAPTIVE_RL"
    assert resolve_policy_name("ahg-rl") == "V19_ADAPTIVE_RL"
    assert resolve_policy_name("PGMPO_AMR_ALIGNED") == "IJCAI26_PGMPO"
    assert resolve_policy_name("v18") == "V18_EA_JOINT"
    assert resolve_policy_name("V10_RULE") == "rule_v10"
    assert resolve_policy_name("SPGA_COR2026") == "SPGA_AMR"
    assert resolve_policy_name("IJCAI25_BALANS_PLAIN") == "BALANS_IJCAI25"
    assert not is_external_policy("V18_EA_JOINT")
    assert is_external_policy("IJCAI26_PGMPO")
    assert build_external_proposer("V18_EA_JOINT") is None
    assert callable(build_external_proposer("rule_v10"))
    with pytest.raises(ValueError):
        resolve_policy_name("UNKNOWN_ALGO_XYZ")


@pytest.mark.parametrize("policy_name", SUPPORTED_POLICIES)
def test_all_registered_policies_step_in_physical_coordinator_both_tracks(policy_name: str):
    if policy_name in ("TII26_E2G_MoE", "ICRA26_Kitting_HGNN"):
        pytest.importorskip("torch")
    for setup_fn in (_make_track1_physical_setup, _make_track2_physical_setup):
        sim, ops = setup_fn()
        coord = PhysicalCVCRCoordinator(
            sim,
            ops,
            policy_method=policy_name,
            budget_ms=35.0,
            epoch_interval=16,
            use_rust=False,
            seed=42,
        )
        rep = coord.step_coordination(0.0, force_replan=True)
        assert rep.triggered is True
        assert rep.receipt_accepted is True
        assert rep.eval_objective is not None and rep.eval_objective >= 0.0
        assert len(coord.history) == 1


@pytest.mark.parametrize("policy_name", ["V18_EA_JOINT", "IJCAI26_PGMPO", "RAL26_DetourDelta", "V19_HETERO_RL"])
def test_run_discrete_episode_on_wheelset_workshop_v2(policy_name: str):
    pytest.importorskip("gymnasium")
    scene = WorkshopScene(build_v2_nav_graph(build_improved_scene()))
    net = scene.network()
    starts = {"AGV_1": "J_S_65", "AGV_2": "J_S_90", "AGV_3": "J_S_142"}
    agvs = [Agv(rid, max_speed=0.65, start_node=s, load_time=2.0, unload_time=2.0) for rid, s in starts.items()]
    tasks = [
        Task("T_01_TD_INSP", pickup="D_TEARDOWN_0", dropoff="D_INSPECTION_0", release_step=0, due_step=260),
        Task("T_02_INSP_PREP", pickup="D_INSPECTION_1", dropoff="D_PREP_IN", release_step=0, due_step=280),
        Task("T_03_PREP_PRESS", pickup="D_PREP_OUT", dropoff="D_PRESS_IN", release_step=0, due_step=260),
        Task("T_04_PRESS_CLOSE", pickup="D_PRESS_OUT", dropoff="D_CLOSING_IN", release_step=0, due_step=220),
        Task("T_05_FIN_STORE_A", pickup="D_FINISHING_0", dropoff="D_STORAGE", release_step=0, due_step=280),
        Task("T_06_FIN_STORE_B", pickup="D_FINISHING_1", dropoff="D_UNKNOWN", release_step=0, due_step=340),
    ]
    ops = [
        OperationSnapshot("OP_V2_INSP", ("T_01_TD_INSP",), "D_INSPECTION_0", crew_id=1, tool_id=1, proc_steps=5, due_step=280),
        OperationSnapshot("OP_V2_PREP", ("T_02_INSP_PREP",), "D_PREP_IN", crew_id=2, tool_id=2, proc_steps=5, due_step=320, predecessor_op_id="OP_V2_INSP"),
        OperationSnapshot("OP_V2_PRESS", ("T_03_PREP_PRESS",), "D_PRESS_IN", crew_id=3, tool_id=3, proc_steps=5, due_step=360, predecessor_op_id="OP_V2_PREP"),
        OperationSnapshot("OP_V2_CLOSE", ("T_04_PRESS_CLOSE",), "D_CLOSING_IN", crew_id=2, tool_id=2, proc_steps=5, due_step=400, predecessor_op_id="OP_V2_PRESS"),
        OperationSnapshot("OP_V2_STORE", ("T_05_FIN_STORE_A",), "D_STORAGE", crew_id=1, tool_id=4, proc_steps=5, due_step=480, predecessor_op_id="OP_V2_CLOSE"),
        OperationSnapshot("OP_V2_VERIFY", ("T_06_FIN_STORE_B",), "D_UNKNOWN", crew_id=3, tool_id=5, proc_steps=5, due_step=520, predecessor_op_id="OP_V2_STORE"),
    ]
    rec = run_discrete_episode(
        policy_name,
        (net, tasks, agvs, ops, {}),
        seed=101,
        max_steps=1600,
        use_rust=False,
        clock_mode="virtual",
        episode_key=f"wheelset_v2_test_{policy_name}",
    )
    assert rec["audit_passed"] is True
    assert rec["task_comp"] == pytest.approx(1.0)
    assert rec["prod_comp"] == pytest.approx(1.0)
    assert rec["cost_sum_err"] <= 0.0055


def test_summarize_physical_run_with_completed_tasks():
    from amr_sched.baselines.policy_registry import summarize_physical_run
    from amr_sched.entities import TaskStatus

    sim, ops = _make_track1_physical_setup()
    coord = PhysicalCVCRCoordinator(sim, ops, policy_method="V18_EA_JOINT", budget_ms=25.0, use_rust=False, seed=7)
    coord.step_coordination(0.0, force_replan=True)
    for idx, t in enumerate(sim.tasks):
        t.status = TaskStatus.COMPLETED
        t.complete_step = (t.due_step or 1000) + idx * 10
    for idx, op in enumerate(ops):
        op.end_step = (op.due_step or 1000) + idx * 5
    summary = summarize_physical_run(coord, ops)
    assert summary["policy"] == "V18_EA_JOINT"
    assert summary["triggered_count"] >= 1
    assert summary["accepted_count"] >= 1
    assert summary["task_tardiness"] == pytest.approx(sum(idx * 10.0 for idx in range(len(sim.tasks))))
    assert summary["op_tardiness"] == pytest.approx(sum(idx * 5.0 for idx in range(len(ops))))
    assert summary["makespan_steps"] == max(op.end_step for op in ops)


def test_v17_checkpoint_name_and_strict_non_downgrade(tmp_path, monkeypatch) -> None:
    """V17 must resolve to v17_window_protection.npz and fail instead of downgrading to V16."""
    import pytest
    from amr_sched.baselines import policy_registry as pr

    assert pr._V17_CKPT.name == "v17_window_protection.npz"
    assert pr._V17_CKPT.is_file()
    assert pr._V16_CKPT.is_file()

    missing_v17 = tmp_path / "missing_v17_window_protection.npz"
    monkeypatch.setattr(pr, "_V17_CKPT", missing_v17)
    with pytest.raises(FileNotFoundError):
        pr.load_internal_cvcr_policy("V17_EA_JOINT")
    with pytest.raises(FileNotFoundError):
        pr.load_internal_cvcr_policy("V18_EA_JOINT")


def test_scenario_digest_sensitive_to_disruptions_and_p2_cfg() -> None:
    """_scenario_digest must hash dynamic disruptions and P2 execution perturbations alongside base tasks/ops."""
    pytest.importorskip("gymnasium")
    from amr_sched.baselines.policy_registry import _scenario_digest, create_discrete_policy_env
    from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
    from tools.benchmarks.compare_workshop_algorithms import build_track2_scenario

    sc, p2_cfg, _ = build_track2_scenario(101, "execution_blockage")
    _, _, env_a = create_discrete_policy_env("rule_v10", seed=101, max_steps=40, use_rust=True, p2_cfg=p2_cfg)
    env_a.reset(seed=101, options={"scenario_tuple": sc})
    sha_a = _scenario_digest(env_a, p2_cfg=p2_cfg)

    p2_mod = P2PerturbationConfig(
        seed=p2_cfg.seed,
        enable_travel_drift=p2_cfg.enable_travel_drift,
        drift_prob=float(p2_cfg.drift_prob) + 0.05,
        early_max_steps=p2_cfg.early_max_steps,
        late_max_steps=p2_cfg.late_max_steps,
        enable_dynamic_obstacles=p2_cfg.enable_dynamic_obstacles,
        obstacle_count=p2_cfg.obstacle_count,
        obstacle_duration_steps=p2_cfg.obstacle_duration_steps,
        obstacle_window_range=p2_cfg.obstacle_window_range,
        est_clear_horizon_steps=p2_cfg.est_clear_horizon_steps,
        enable_local_planner_fail=p2_cfg.enable_local_planner_fail,
        local_fail_count=p2_cfg.local_fail_count,
        local_fail_duration_steps=p2_cfg.local_fail_duration_steps,
        local_fail_window_range=p2_cfg.local_fail_window_range,
        enable_p4_belief=p2_cfg.enable_p4_belief,
        enable_p4_voi_probe=p2_cfg.enable_p4_voi_probe,
        obstacle_durations=p2_cfg.obstacle_durations,
    )
    _, _, env_b = create_discrete_policy_env("rule_v10", seed=101, max_steps=40, use_rust=True, p2_cfg=p2_mod)
    env_b.reset(seed=101, options={"scenario_tuple": sc})
    sha_b = _scenario_digest(env_b, p2_cfg=p2_mod)
    assert sha_a != sha_b

    env_a.disruptions["injected_test_fault"] = {"tick": 12, "kind": "agv_fault", "agv_id": "AGV_1"}
    sha_c = _scenario_digest(env_a, p2_cfg=p2_cfg)
    assert sha_c != sha_a


def test_summarize_physical_run_extended_latency_metrics() -> None:
    """summarize_physical_run returns propose_ms_p99, deadline_overrun_rate, and e2e_feedback_to_commit_ms."""
    from amr_sched.allocation.greedy import GreedyAllocator
    from amr_sched.baselines.policy_registry import build_external_proposer, summarize_physical_run
    from amr_sched.cvcr.physical_bridge import PhysicalCVCRCoordinator
    from amr_sched.entities import Agv, Task
    from amr_sched.external_execution import ExternalExecution
    from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
    from amr_sched.reschedule import ReschedulePolicy
    from amr_sched.simulation import SimulationConfig, Simulator
    from Tests.test_ros2_nav2_cvcr_loop import _build_physical_bypass_network

    net, coords = _build_physical_bypass_network()
    gates = {
        "r0": ExternalExecution("sc", "r0", coords["S0"], run_id="pr_ext", source="localized", frame_id="map", namespace_routes=True),
        "r1": ExternalExecution("sc", "r1", coords["S1"], run_id="pr_ext", source="localized", frame_id="map", namespace_routes=True),
    }

    sim = Simulator(
        net,
        [Task("T0", "J0", "W0"), Task("T1", "S1", "W1")],
        [Agv("r0", max_speed=0.5, start_node="S0"), Agv("r1", max_speed=0.5, start_node="S1")],
        SpaceTimeAStarPlanner(max_speed=0.5, dt=0.5),
        GreedyAllocator(),
        SimulationConfig(dt=0.5, max_steps=200, load_time=0.5, unload_time=0.5, idle_retreat=False, deadlock_timeout=0, reschedule_policy=ReschedulePolicy.NONE),
        executions=gates,
    )
    sim.reset()
    for rid, g in gates.items():
        sim.execution_feedback({**g.envelope(), "event": "POSE", "pose": [*g.start_xy, 0.0], "seq": 1, "timestamp": 0.0}, 0.0)
    ext = build_external_proposer("IJCAI26_PGMPO", seed=7, use_rust=False, backend="numpy")
    coord = PhysicalCVCRCoordinator(sim, [], policy_method="IJCAI26_PGMPO", budget_ms=25.0, epoch_interval=6, use_rust=False, external_proposer=ext, seed=7)
    for k in range(3):
        coord.step_coordination(float(k + 1) * 0.5, force_replan=(k == 0))
        sim.step_once()
    summary = summarize_physical_run(coord, [], budget_ms=25.0)
    assert "propose_ms_p99" in summary
    assert "deadline_overrun_rate" in summary
    assert 0.0 <= float(summary["deadline_overrun_rate"]) <= 1.0
    e2e = summary["e2e_feedback_to_commit_ms"]
    assert e2e["count"] >= 1
    assert e2e["p99"] >= e2e["mean"] >= 0.0


def test_policy_registry_torch_vs_numpy_parity() -> None:
    """Registry build_external_proposer and discrete episode outcomes match across PyTorch and NumPy backends."""
    import pytest
    pytest.importorskip("torch")
    from amr_sched.baselines.policy_registry import build_external_proposer, run_discrete_episode
    from tools.benchmarks.compare_workshop_algorithms import build_track2_scenario

    sc, p2_cfg, _ = build_track2_scenario(101, "kitting_windows")
    for pol in ("IJCAI26_PGMPO", "EJOR26_Wu_GRU_Attn", "V19_HETERO_BC", "V19_HETERO_RL"):
        ep_t = run_discrete_episode(pol, sc, seed=101, max_steps=100, use_rust=True, p2_cfg=p2_cfg, clock_mode="virtual", backend="torch")
        ep_n = run_discrete_episode(pol, sc, seed=101, max_steps=100, use_rust=True, p2_cfg=p2_cfg, clock_mode="virtual", backend="numpy")
        assert ep_t["inference_backend"] == "torch"
        assert ep_n["inference_backend"] == "numpy"
        assert ep_t["scenario_sha"] == ep_n["scenario_sha"]
        assert ep_t["cmax"] == ep_n["cmax"]
        assert ep_t["task_tardiness_cost"] == ep_n["task_tardiness_cost"]
        assert ep_t["op_tardiness_cost"] == ep_n["op_tardiness_cost"]
        assert ep_t["churn"] == ep_n["churn"]
        assert ep_t["task_comp"] == ep_n["task_comp"]
        assert ep_t["prod_comp"] == ep_n["prod_comp"]
        assert build_external_proposer(pol, seed=101, backend="numpy") is not None
