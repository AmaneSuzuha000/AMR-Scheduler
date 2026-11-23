"""Dual-Layer Discrete (Layer A) vs Physical ROS2/Nav2/Gazebo (Layer B) Reconciliation.

Reconciles Layer A discrete scheduling (`run_discrete_episode`, `--clock wall`) against
Layer B continuous physical execution (`PhysicalCVCRCoordinator` + `ExternalExecution`
kinematic & headless WSL2 Gazebo + ROS 2 + Nav2 full-chain verification) across:
  - Track 1 (`rail_freight_workshop_v1`, 62 nodes, 6 AGVs)
  - Track 2 (`wheelset_workshop_v2`, 50 nodes, 3 AGVs)
for the 6 evaluated policies:
  `V19_HETERO_RL`, `V18_EA_JOINT`, `IJCAI26_PGMPO`, `rule_v10`,
  `EJOR26_Wu_GRU_Attn`, `RAL26_DetourDelta`.

Writes all outputs exclusively to `out/dual_layer/`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any
import uuid

for _k, _v in (
    ("OMP_NUM_THREADS", "1"),
    ("MKL_NUM_THREADS", "1"),
    ("OPENBLAS_NUM_THREADS", "1"),
    ("NUMEXPR_NUM_THREADS", "1"),
):
    os.environ.setdefault(_k, _v)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.baselines.policy_registry import (
    build_external_proposer,
    run_discrete_episode,
    summarize_physical_run,
)
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.physical_bridge import (
    CorridorObstacleReport,
    Nav2RobotTelemetry,
    PhysicalCVCRCoordinator,
)
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.entities import Agv, MaterialType, Task, TaskPriority, TaskStatus
from amr_sched.external_execution import ExternalExecution
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import WorkshopScene
from tools.live_bridge.fleet_config import point_execution_network
from tools.wheelset_scene.workshop_v2_layout import build_improved_scene, build_v2_nav_graph

EVAL_POLICIES: tuple[str, ...] = (
    "V19_HETERO_RL",
    "V18_EA_JOINT",
    "IJCAI26_PGMPO",
    "rule_v10",
    "EJOR26_Wu_GRU_Attn",
    "RAL26_DetourDelta",
)


def _build_track1_spec(
    physical: bool,
    seed: int = 101,
) -> tuple[Any, list[Task], list[Agv], list[OperationSnapshot], dict[str, ExternalExecution]]:
    """Build matched Track 1 (62-node, 6-AGV) discrete or physical setup."""
    scene = WorkshopScene.load(ROOT / "scenes" / "rail_freight_workshop_v1.json")
    net = point_execution_network(scene.network()) if physical else scene.network()
    starts = {
        "R1": "J_S_65",
        "R2": "J_N_45",
        "R3": "J_N_85",
        "R4": "W_DOOR",
        "R5": "W_COUPLER",
        "R6": "W_PARK",
    }
    s_off = max(0, (int(seed) // 100) - 1) * 3
    tasks = [
        Task("K_BG_1", "Q_BOGIE", "D_BOGIE", release_step=0, due_step=80 + s_off, material_type=MaterialType.BOGIE),
        Task("K_WS_1", "J_S_65", "D_WHEELSET", release_step=s_off // 2, due_step=110 + s_off, material_type=MaterialType.WHEELSET),
        Task("K_BR_1", "J_N_85", "D_BRAKE", release_step=0, due_step=90 + s_off, material_type=MaterialType.BRAKE_VALVE),
        Task("K_CP_1", "R_COUPLER", "D_COUPLER", release_step=s_off // 3, due_step=105 + s_off, material_type=MaterialType.BRAKE_VALVE),
        Task("K_DR_1", "R_DOOR", "D_DOOR", release_step=0, due_step=115 + s_off, material_type=MaterialType.BRAKE_VALVE),
        Task(
            "K_AS_1",
            "J_S_125",
            "D_ASSEMBLY",
            release_step=s_off // 2,
            due_step=130 + s_off,
            priority=TaskPriority.URGENT,
            material_type=MaterialType.WHEELSET,
        ),
    ]
    ops = [
        OperationSnapshot("OP_BOGIE", bay_id="D_BOGIE", crew_id=1, tool_id=1, proc_steps=15, due_step=95 + s_off, kit_task_ids=("K_BG_1",)),
        OperationSnapshot(
            "OP_WHEELSET",
            bay_id="D_WHEELSET",
            crew_id=2,
            tool_id=2,
            proc_steps=15,
            due_step=130 + s_off,
            predecessor_op_id="OP_BOGIE",
            kit_task_ids=("K_WS_1",),
        ),
        OperationSnapshot("OP_BRAKE", bay_id="D_BRAKE", crew_id=3, tool_id=3, proc_steps=15, due_step=110 + s_off, kit_task_ids=("K_BR_1",)),
        OperationSnapshot(
            "OP_COUPLER",
            bay_id="D_COUPLER",
            crew_id=4,
            tool_id=4,
            proc_steps=15,
            due_step=125 + s_off,
            eligible_bays=("D_COUPLER", "D_DOOR"),
            kit_task_ids=("K_CP_1",),
        ),
        OperationSnapshot("OP_DOOR", bay_id="D_DOOR", crew_id=5, tool_id=5, proc_steps=15, due_step=135 + s_off, kit_task_ids=("K_DR_1",)),
        OperationSnapshot(
            "OP_ASSEMBLY",
            bay_id="D_ASSEMBLY",
            crew_id=6,
            tool_id=6,
            proc_steps=20,
            due_step=160 + s_off,
            predecessor_op_id="OP_WHEELSET",
            is_urgent=True,
            kit_task_ids=("K_AS_1",),
        ),
    ]
    gates: dict[str, ExternalExecution] = {}
    if physical:
        run_id = uuid.uuid4().hex
        gates = {
            rid: ExternalExecution(
                "rail_freight_workshop_v1",
                rid,
                (net.nodes[s].x, net.nodes[s].y),
                source="localized",
                run_id=run_id,
                namespace_routes=True,
                arrival_tolerance=0.20,
                timeout=4.0,
            )
            for rid, s in starts.items()
        }
    agvs = [Agv(rid, max_speed=0.8, start_node=s, load_time=1.0, unload_time=1.0) for rid, s in starts.items()]
    return net, tasks, agvs, ops, gates


def _build_track2_spec(
    physical: bool,
    seed: int = 101,
) -> tuple[Any, list[Task], list[Agv], list[OperationSnapshot], dict[str, ExternalExecution]]:
    """Build matched Track 2 (50-node, 3-AGV) discrete or physical setup."""
    scene = WorkshopScene(build_v2_nav_graph(build_improved_scene()))
    net = point_execution_network(scene.network()) if physical else scene.network()
    starts = {"AGV_1": "J_S_65", "AGV_2": "J_S_90", "AGV_3": "J_S_142"}
    s_off = max(0, (int(seed) // 100) - 1) * 4
    tasks = [
        Task("T_01_TD_INSP", pickup="D_TEARDOWN_0", dropoff="D_INSPECTION_0", release_step=0, due_step=140 + s_off),
        Task("T_02_INSP_PREP", pickup="D_INSPECTION_1", dropoff="D_PREP_IN", release_step=s_off // 2, due_step=170 + s_off),
        Task("T_03_PREP_PRESS", pickup="D_PREP_OUT", dropoff="D_PRESS_IN", release_step=0, due_step=190 + s_off),
        Task("T_04_PRESS_CLOSE", pickup="D_PRESS_OUT", dropoff="D_CLOSING_IN", release_step=s_off // 2, due_step=180 + s_off),
        Task("T_05_FIN_STORE_A", pickup="D_FINISHING_0", dropoff="D_STORAGE", release_step=0, due_step=210 + s_off),
        Task("T_06_FIN_STORE_B", pickup="D_FINISHING_1", dropoff="D_UNKNOWN", release_step=s_off // 2, due_step=240 + s_off),
    ]
    ops = [
        OperationSnapshot("OP_V2_INSP", ("T_01_TD_INSP",), "D_INSPECTION_0", crew_id=1, tool_id=1, proc_steps=5, due_step=160 + s_off),
        OperationSnapshot(
            "OP_V2_PREP",
            ("T_02_INSP_PREP",),
            "D_PREP_IN",
            crew_id=2,
            tool_id=2,
            proc_steps=5,
            due_step=200 + s_off,
            predecessor_op_id="OP_V2_INSP",
        ),
        OperationSnapshot(
            "OP_V2_PRESS",
            ("T_03_PREP_PRESS",),
            "D_PRESS_IN",
            crew_id=3,
            tool_id=3,
            proc_steps=5,
            due_step=230 + s_off,
            predecessor_op_id="OP_V2_PREP",
        ),
        OperationSnapshot(
            "OP_V2_CLOSE",
            ("T_04_PRESS_CLOSE",),
            "D_CLOSING_IN",
            crew_id=2,
            tool_id=2,
            proc_steps=5,
            due_step=260 + s_off,
            predecessor_op_id="OP_V2_PRESS",
        ),
        OperationSnapshot(
            "OP_V2_STORE",
            ("T_05_FIN_STORE_A",),
            "D_STORAGE",
            crew_id=1,
            tool_id=4,
            proc_steps=5,
            due_step=300 + s_off,
            predecessor_op_id="OP_V2_CLOSE",
        ),
        OperationSnapshot(
            "OP_V2_VERIFY",
            ("T_06_FIN_STORE_B",),
            "D_UNKNOWN",
            crew_id=3,
            tool_id=5,
            proc_steps=5,
            due_step=330 + s_off,
            predecessor_op_id="OP_V2_STORE",
        ),
    ]
    gates: dict[str, ExternalExecution] = {}
    if physical:
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
                timeout=4.0,
            )
            for rid, s in starts.items()
        }
    agvs = [Agv(rid, max_speed=0.65, start_node=s, load_time=2.0, unload_time=2.0) for rid, s in starts.items()]
    return net, tasks, agvs, ops, gates


def _advance_physical_motion(
    sim: Simulator,
    positions: dict[str, tuple[float, float]],
    motion: dict[str, dict[str, Any]],
    now: float,
    *,
    speed_scales: dict[str, float],
) -> None:
    """Advance realistic Nav2/AMCL POSE, ARRIVED, and STOPPED feedback for bound physical gates."""
    for robot, gate in sim.executions.items():
        gate.drain_commands()
        xy = positions[robot]
        if gate.fault and gate.bound:
            evt = "STOPPED" if not gate.stopped else "POSE"
            sim.execution_feedback(
                {
                    **gate.envelope(),
                    "event": evt,
                    "pose": [xy[0], xy[1], 0.0],
                    "seq": gate.last_seq + 1,
                    "timestamp": now,
                },
                now,
            )
            continue

        command = gate.command
        event = "POSE"
        if command is not None:
            token = command["route_id"]
            if token not in motion:
                dist = math.dist(xy, command["target_xy"])
                scale = max(0.35, float(speed_scales.get(robot, 1.0)))
                eff_speed = max(0.08, float(command["speed"]) * scale)
                # Include continuous Nav2 acceleration/deceleration & heading settle envelope
                dur = max(0.5, (dist / eff_speed) * 1.12 + 0.15)
                motion[token] = {
                    "start": now,
                    "source": xy,
                    "target": tuple(command["target_xy"]),
                    "duration": dur,
                }
            act = motion[token]
            frac = min(1.0, (now - float(act["start"])) / max(1e-6, float(act["duration"])))
            xy = (
                float(act["source"][0]) + (float(act["target"][0]) - float(act["source"][0])) * frac,
                float(act["source"][1]) + (float(act["target"][1]) - float(act["source"][1])) * frac,
            )
            positions[robot] = xy
            if frac >= 1.0:
                event = "ARRIVED"
        res = sim.execution_feedback(
            {
                **gate.envelope(),
                "event": event,
                "pose": [xy[0], xy[1], 0.0],
                "seq": gate.last_seq + 1,
                "timestamp": now,
            },
            now,
        )
        if not res.get("accepted", False):
            raise RuntimeError(f"Physical feedback rejected for {robot} at t={now}: {res}")


def run_layer_b_physical_episode(track: str, policy_name: str, seed: int = 101) -> dict[str, Any]:
    """Run closed-loop Layer B continuous physical execution via PhysicalCVCRCoordinator + ExternalExecution."""
    s_idx = max(0, (int(seed) // 100) - 1)
    if track == "track1":
        net, tasks, agvs, ops, gates = _build_track1_spec(physical=True, seed=seed)
        dt = 1.0
        max_steps = 1600
        blocked_edge = ("J_S_85", "J_S_105")
        block_window = (2 + s_idx * 3, 24 + s_idx * 4)
        slow_robot = ("R2", "R3", "R1", "R2", "R3")[s_idx % 5]
    else:
        net, tasks, agvs, ops, gates = _build_track2_spec(physical=True, seed=seed)
        dt = 1.0
        max_steps = 1600
        blocked_edge = ("J_S_108", "J_S_125")
        block_window = (2 + s_idx * 3, 22 + s_idx * 4)
        slow_robot = ("AGV_2", "AGV_1", "AGV_2", "AGV_3", "AGV_2")[s_idx % 5]

    max_speed = agvs[0].max_speed
    load_t = agvs[0].load_time
    sim = Simulator(
        net,
        tasks,
        agvs,
        SpaceTimeAStarPlanner(max_speed=max_speed, dt=dt),
        GreedyAllocator(),
        SimulationConfig(
            dt=dt,
            max_steps=1800,
            planning_horizon=2200,
            load_time=load_t,
            unload_time=load_t,
            idle_retreat=False,
            deadlock_timeout=0,
            record_trajectory=True,
        ),
        executions=gates,
    )
    sim.reset()
    ext = build_external_proposer(policy_name, seed=seed, use_rust=True, backend="numpy")
    coord = PhysicalCVCRCoordinator(
        sim,
        ops,
        policy_method=policy_name,
        budget_ms=35.0,
        epoch_interval=12,
        use_rust=True,
        external_proposer=ext,
        seed=seed,
    )
    positions = {r: g.start_xy for r, g in gates.items()}
    motion: dict[str, dict[str, Any]] = {}

    for step in range(max_steps):
        now = float(step) * dt
        in_slow = (block_window[0] - 5) <= step < (block_window[1] + 10)
        speed_scales = {r: (0.55 if (r == slow_robot and in_slow) else 1.0) for r in gates}
        _advance_physical_motion(sim, positions, motion, now, speed_scales=speed_scales)

        blk: list[CorridorObstacleReport] = []
        if block_window[0] <= step < block_window[1] and blocked_edge in net.edges:
            blk = [
                CorridorObstacleReport(
                    blocked_edge,
                    estimated_clear_horizon_steps=max(4, block_window[1] - step),
                )
            ]
        telem = {
            r: Nav2RobotTelemetry(
                r,
                pose_xy_yaw=(positions[r][0], positions[r][1], 0.0),
                speed_scale=speed_scales[r],
                eta_drift_steps=(6 if speed_scales[r] < 0.8 else 0),
                local_planner_state=("slowed" if speed_scales[r] < 0.8 else "nominal"),
                receipt_age_s=0.04,
            )
            for r in gates
        }
        coord.step_coordination(
            now,
            nav2_telemetry=telem,
            blocked_corridors=blk,
            force_replan=(step == 0),
        )
        sim.step_once()
        if sim._all_done():
            if coord.tracker is not None:
                while not coord.tracker.all_done(sim._step_cursor) and sim._step_cursor < sim.config.max_steps:
                    sim._step_cursor += 1
                    coord.tracker.step(sim._step_cursor)
            break

    aud = IndependentAuditor.audit_simulation_run(sim, ops, require_trajectory=True)
    summary = summarize_physical_run(coord, ops, budget_ms=35.0)
    tasks_done = sum(1 for t in sim.tasks if t.status == TaskStatus.COMPLETED)
    ops_done = sum(1 for o in ops if o.end_step is not None)
    unfinished_tasks = max(0, len(sim.tasks) - tasks_done)
    unfinished_ops = max(0, len(ops) - ops_done)
    task_comp = float(tasks_done) / max(1, len(sim.tasks))
    prod_comp = float(ops_done) / max(1, len(ops))
    cmax = int(summary["makespan_steps"]) if (unfinished_tasks == 0 and unfinished_ops == 0) else max(
        int(summary["makespan_steps"]), int(sim._step_cursor)
    )
    tardiness = float(summary["task_tardiness"]) + float(summary["op_tardiness"])
    reassigned_cnt = sum(len(h.reassigned_tasks) for h in coord.history)
    total_compute_ms = float(summary["propose_ms_mean"]) * float(summary["triggered_count"])
    # Unified objective matching Layer A: cmax + 2*task_tard + 2*op_tard + 5000*unfin_tasks + 3000*unfin_ops + 15*churn + 0.05*compute_ms
    unified_obj = (
        float(cmax)
        + 2.0 * float(summary["task_tardiness"])
        + 2.0 * float(summary["op_tardiness"])
        + 5000.0 * float(unfinished_tasks)
        + 3000.0 * float(unfinished_ops)
        + 15.0 * float(reassigned_cnt)
        + 0.05 * float(total_compute_ms)
    )
    legacy_phys_obj = (
        float(cmax)
        + 3.0 * float(summary["task_tardiness"])
        + 4.0 * float(summary["op_tardiness"])
        + 15.0 * float(reassigned_cnt)
        + 2500.0 * float(unfinished_tasks)
        + 3000.0 * float(unfinished_ops)
    )
    return {
        "track": track,
        "policy": policy_name,
        "seed": seed,
        "audit_passed": bool(aud.pass_audit),
        "audit_issues": list(aud.issues),
        "task_comp": round(task_comp, 4),
        "prod_comp": round(prod_comp, 4),
        "makespan_steps": cmax,
        "task_tardiness": round(float(summary["task_tardiness"]), 3),
        "op_tardiness": round(float(summary["op_tardiness"]), 3),
        "total_tardiness": round(tardiness, 3),
        "unified_net_obj": round(unified_obj, 3),
        "physical_net_obj": round(unified_obj, 3),
        "legacy_physical_net_obj": round(legacy_phys_obj, 3),
        "compute_ms_total": round(total_compute_ms, 3),
        "triggered_count": int(summary["triggered_count"]),
        "accepted_count": int(summary["accepted_count"]),
        "detour_replans": int(summary["detour_events"]),
        "reassigned_tasks": int(reassigned_cnt),
        "propose_ms_mean": round(float(summary["propose_ms_mean"]), 3),
        "propose_ms_p95": round(float(summary["propose_ms_p95"]), 3),
        "propose_ms_p99": round(float(summary["propose_ms_p99"]), 3),
        "deadline_overrun_rate": round(float(summary["deadline_overrun_rate"]), 4),
        "e2e_feedback_to_commit_ms": summary["e2e_feedback_to_commit_ms"],
    }


def run_layer_a_discrete_episode(track: str, policy_name: str, seed: int = 101) -> dict[str, Any]:
    """Run matched Layer A discrete episode with wall-clock timing and torch-free NumPy backend."""
    from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig

    if track == "track1":
        net, tasks, agvs, ops, _ = _build_track1_spec(physical=False, seed=seed)
        max_steps = 1600
    else:
        net, tasks, agvs, ops, _ = _build_track2_spec(physical=False, seed=seed)
        max_steps = 1600
    p2_cfg = P2PerturbationConfig(
        seed=seed,
        enable_travel_drift=True,
        drift_prob=0.18,
        early_max_steps=1,
        late_max_steps=3,
        enable_dynamic_obstacles=True,
        obstacle_count=1,
        obstacle_duration_steps=20,
        obstacle_window_range=(2, 30),
        est_clear_horizon_steps=15,
        enable_local_planner_fail=False,
        local_fail_count=0,
        local_fail_duration_steps=0,
        local_fail_window_range=(10, 40),
        enable_p4_belief=True,
        enable_p4_voi_probe=False,
    )
    rec = run_discrete_episode(
        policy_name,
        (net, tasks, agvs, ops, {}),
        seed=seed,
        max_steps=max_steps,
        use_rust=True,
        p2_cfg=p2_cfg,
        clock_mode="wall",
        backend="numpy",
        episode_key=f"g4_layer_a_{track}_{policy_name}_{seed}",
    )
    return {
        "track": track,
        "policy": policy_name,
        "seed": seed,
        "audit_passed": bool(rec["audit_passed"]),
        "task_comp": round(float(rec["task_comp"]), 4),
        "prod_comp": round(float(rec["prod_comp"]), 4),
        "makespan_steps": int(rec["cmax"]),
        "total_tardiness": round(float(rec["tardiness"]), 3),
        "discrete_net_obj": round(float(rec["net_obj"]), 3),
        "churn": int(rec["churn"]),
        "decision_ms_mean": round(float(rec["mean_decision_ms"]), 3),
        "decision_ms_p95": round(float(rec["decision_ms_p95"]), 3),
        "decision_ms_p99": round(float(rec["decision_ms_p99"]), 3),
        "inference_backend": str(rec.get("inference_backend", "numpy")),
    }


def _rank_vector(vals: list[float]) -> list[float]:
    """Compute 1-based ascending ranks with average tie-breaking."""
    n = len(vals)
    indexed = sorted(enumerate(vals), key=lambda x: x[1])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(indexed[j + 1][1] - indexed[i][1]) <= 1e-9:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def _spearman_rho(xs: list[float], ys: list[float]) -> float:
    """Compute Spearman rank correlation coefficient."""
    rx = np.asarray(_rank_vector(xs), dtype=np.float64)
    ry = np.asarray(_rank_vector(ys), dtype=np.float64)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.linalg.norm(rx) * np.linalg.norm(ry))
    if denom <= 1e-12:
        return 1.0
    return float(np.dot(rx, ry) / denom)


def _kendall_tau(xs: list[float], ys: list[float]) -> float:
    """Compute Kendall rank correlation coefficient tau-a."""
    n = len(xs)
    if n < 2:
        return 1.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            prod = dx * dy
            if prod > 1e-12:
                concordant += 1
            elif prod < -1e-12:
                discordant += 1
    total_pairs = n * (n - 1) // 2
    return float(concordant - discordant) / max(1, total_pairs)


def _collect_existing_and_optional_gazebo_reports(out_dir: Path, run_gazebo_track2: bool) -> list[dict[str, Any]]:
    """Collect verified full-chain Gazebo ROS 2 + Nav2 reports and optionally run headless Track 2 V19_HETERO_RL."""
    reports: list[dict[str, Any]] = []
    known_paths = [
        ("track1_gazebo_v18", ROOT / "out" / "map01_full_chain_01" / "full_chain_report.json"),
        ("track2_gazebo_v18", ROOT / "out" / "wheelset_v2_full_chain_01" / "full_chain_report.json"),
        ("track2_gazebo_rule_v10", ROOT / "out" / "wheelset_v2_cmp_rule_v10" / "full_chain_report.json"),
    ]
    if run_gazebo_track2:
        gz_out = out_dir / "gazebo_wheelset_v2_v19_hetero_rl"
        gz_report_path = gz_out / "full_chain_report.json"
        if not gz_report_path.is_file():
            clean_cmd = (
                "pkill -9 -f 'gzserver|gzclient|ros2|nav2|ekf_node|amcl|lifecycle_manager|"
                "robot_state_publisher|rviz2|run_map01_full_chain|run_wheelset_v2_full_chain' 2>/dev/null || true"
            )
            subprocess.run(["wsl", "-e", "bash", "-lc", clean_cmd], check=False)
            wsl_cmd = (
                "source /opt/ros/humble/setup.bash && "
                "if [ -f /root/map01_guard_install/setup.bash ]; then "
                "source /root/map01_guard_install/setup.bash; fi && "
                "cd ${AMR_ROOT:-$(pwd)} && "
                "python3 -B tools/wheelset_scene/run_wheelset_v2_full_chain.py "
                "--policy V19_HETERO_RL "
                "--out out/dual_layer/gazebo_wheelset_v2_v19_hetero_rl "
                "--no-rviz --no-camera-capture --wait-wpf-s 15"
            )
            try:
                subprocess.run(["wsl", "-e", "bash", "-lc", wsl_cmd], check=True, timeout=600)
            finally:
                subprocess.run(["wsl", "-e", "bash", "-lc", clean_cmd], check=False)
        known_paths.append(("track2_gazebo_v19_hetero_rl", gz_report_path))

    for tag, path in known_paths:
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                pm = data.get("policy_metrics", {})
                reports.append(
                    {
                        "tag": tag,
                        "scene_id": data.get("scene_id"),
                        "policy": data.get("policy") or data.get("requested_policy") or "V18_EA_JOINT",
                        "passed": bool(data.get("passed", False)),
                        "sim_time_s": data.get("sim_time_s"),
                        "tasks_completed": data.get("tasks_completed"),
                        "tasks_total": data.get("tasks_total"),
                        "ops_completed": data.get("ops_completed"),
                        "ops_total": data.get("ops_total"),
                        "makespan_steps": pm.get("makespan_steps", data.get("audit", {}).get("makespan")),
                        "task_tardiness": pm.get("task_tardiness"),
                        "op_tardiness": pm.get("op_tardiness"),
                        "propose_ms_mean": pm.get("propose_ms_mean"),
                        "propose_ms_p95": pm.get("propose_ms_p95"),
                        "propose_ms_p99": pm.get("propose_ms_p99"),
                        "min_clearance_m": data.get("clearance_audit", {}).get("min_clearance_m"),
                    }
                )
            except Exception:
                pass
    return reports


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Dual-Layer Discrete-vs-Physical Reconciliation.")
    parser.add_argument(
        "--output",
        type=str,
        default="out/dual_layer/dual_layer_report.json",
        help="Output JSON path under out/dual_layer/",
    )
    parser.add_argument(
        "--with-gazebo-v19",
        action="store_true",
        help="Also run headless WSL2 Gazebo + ROS2 + Nav2 full-chain on Track 2 for V19_HETERO_RL.",
    )
    parser.add_argument(
        "--multi-seed",
        action="store_true",
        help="Run 5 matched seeds (101, 202, 303, 404, 505) per policy and track under the unified objective.",
    )
    args = parser.parse_args()

    out_path = ROOT / args.output
    out_dir = out_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_seeds = (101, 202, 303, 404, 505) if args.multi_seed else (101,)

    t0 = time.perf_counter()
    reconciliation_by_track: dict[str, Any] = {}

    for track in ("track1", "track2"):
        layer_a_rows: list[dict[str, Any]] = []
        layer_b_rows: list[dict[str, Any]] = []
        paired_rows: list[dict[str, Any]] = []

        for pol in EVAL_POLICIES:
            pol_ra: list[dict[str, Any]] = []
            pol_rb: list[dict[str, Any]] = []
            for s in eval_seeds:
                print(f"[Dual-Layer] Running {track} | {pol} | seed={s} (Layer A + Layer B)...", flush=True)
                ra = run_layer_a_discrete_episode(track, pol, seed=s)
                rb = run_layer_b_physical_episode(track, pol, seed=s)
                layer_a_rows.append(ra)
                layer_b_rows.append(rb)
                pol_ra.append(ra)
                pol_rb.append(rb)

            mean_a_obj = round(float(np.mean([r["discrete_net_obj"] for r in pol_ra])), 3)
            mean_b_obj = round(float(np.mean([r["unified_net_obj"] for r in pol_rb])), 3)
            mean_b_leg = round(float(np.mean([r["legacy_physical_net_obj"] for r in pol_rb])), 3)
            mean_a_cmax = round(float(np.mean([r["makespan_steps"] for r in pol_ra])), 2)
            mean_b_cmax = round(float(np.mean([r["makespan_steps"] for r in pol_rb])), 2)
            paired_rows.append(
                {
                    "track": track,
                    "policy": pol,
                    "num_seeds": len(eval_seeds),
                    "layer_a_net_obj": mean_a_obj,
                    "layer_b_net_obj": mean_b_obj,
                    "layer_b_legacy_obj": mean_b_leg,
                    "layer_a_makespan": mean_a_cmax,
                    "layer_b_makespan": mean_b_cmax,
                    "layer_a_tardiness": round(float(np.mean([r["total_tardiness"] for r in pol_ra])), 3),
                    "layer_b_tardiness": round(float(np.mean([r["total_tardiness"] for r in pol_rb])), 3),
                    "layer_a_task_comp": round(float(np.mean([r["task_comp"] for r in pol_ra])), 4),
                    "layer_b_task_comp": round(float(np.mean([r["task_comp"] for r in pol_rb])), 4),
                    "layer_a_decision_ms_p95": round(float(np.mean([r["decision_ms_p95"] for r in pol_ra])), 3),
                    "layer_b_propose_ms_p95": round(float(np.mean([r["propose_ms_p95"] for r in pol_rb])), 3),
                    "layer_b_propose_ms_p99": round(float(np.mean([r["propose_ms_p99"] for r in pol_rb])), 3),
                    "layer_b_e2e_p99_ms": round(float(np.mean([r["e2e_feedback_to_commit_ms"]["p99"] for r in pol_rb])), 3),
                    "layer_a_audit_passed": all(r["audit_passed"] for r in pol_ra),
                    "layer_b_audit_passed": all(r["audit_passed"] for r in pol_rb),
                    "makespan_ratio_b_over_a": round(
                        float(mean_b_cmax) / max(1.0, float(mean_a_cmax)), 4
                    ),
                }
            )

        a_objs = [r["layer_a_net_obj"] for r in paired_rows]
        b_objs = [r["layer_b_net_obj"] for r in paired_rows]
        b_leg_objs = [r["layer_b_legacy_obj"] for r in paired_rows]
        a_cmax = [float(r["layer_a_makespan"]) for r in paired_rows]
        b_cmax = [float(r["layer_b_makespan"]) for r in paired_rows]

        ranks_a = _rank_vector(a_objs)
        ranks_b = _rank_vector(b_objs)
        for idx, row in enumerate(paired_rows):
            row["rank_layer_a"] = ranks_a[idx]
            row["rank_layer_b"] = ranks_b[idx]
            row["rank_shift"] = round(abs(ranks_a[idx] - ranks_b[idx]), 2)

        top1_a = min(paired_rows, key=lambda r: r["layer_a_net_obj"])["policy"]
        top1_b = min(paired_rows, key=lambda r: r["layer_b_net_obj"])["policy"]

        reconciliation_by_track[track] = {
            "paired_rows": paired_rows,
            "layer_a_detail": layer_a_rows,
            "layer_b_detail": layer_b_rows,
            "rank_correlation": {
                "num_seeds": len(eval_seeds),
                "objective_formula": "J_unified = cmax + 2*T_task + 2*T_op + 5000*U_task + 3000*U_op + 15*C + 0.05*M",
                "net_obj_spearman_rho": round(_spearman_rho(a_objs, b_objs), 4),
                "net_obj_kendall_tau": round(_kendall_tau(a_objs, b_objs), 4),
                "legacy_net_obj_spearman_rho": round(_spearman_rho(a_objs, b_leg_objs), 4),
                "makespan_spearman_rho": round(_spearman_rho(a_cmax, b_cmax), 4),
                "top1_layer_a": top1_a,
                "top1_layer_b": top1_b,
                "all_audits_passed": all(
                    r["layer_a_audit_passed"] and r["layer_b_audit_passed"] for r in paired_rows
                ),
            },
        }

    gazebo_reports = _collect_existing_and_optional_gazebo_reports(out_dir, bool(args.with_gazebo_v19))

    report = {
        "gate": "Dual_Layer_Reconciliation",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "policies": list(EVAL_POLICIES),
        "tracks": reconciliation_by_track,
        "gazebo_full_chain_evidence": gazebo_reports,
    }
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# Dual-Layer Discrete (Layer A) vs Physical ROS 2 / Nav2 / Gazebo (Layer B) Reconciliation",
        "",
        "- **Clock & Backend**: Single-process wall-clock (`--clock wall`), torch-free NumPy inference (`backend='numpy'`), `budget_ms=35.0`.",
        "- **Evaluated Policies (6)**: `V19_HETERO_RL`, `V18_EA_JOINT`, `IJCAI26_PGMPO`, `rule_v10`, `EJOR26_Wu_GRU_Attn`, `RAL26_DetourDelta`.",
        "",
    ]

    for track, label in (
        ("track1", "Track 1 (`rail_freight_workshop_v1`, 62 nodes, 6 AGVs)"),
        ("track2", "Track 2 (`wheelset_workshop_v2`, 50 nodes, 3 AGVs)"),
    ):
        tdata = reconciliation_by_track[track]
        rc = tdata["rank_correlation"]
        md_lines.extend(
            [
                f"## {label}",
                "",
                f"- **Objective Spearman Rank Correlation ($\\rho$)**: `{rc['net_obj_spearman_rho']:.4f}`",
                f"- **Objective Kendall Rank Correlation ($\\tau$)**: `{rc['net_obj_kendall_tau']:.4f}`",
                f"- **Makespan Spearman Rank Correlation ($\\rho$)**: `{rc['makespan_spearman_rho']:.4f}`",
                f"- **Layer A Top-1**: `{rc['top1_layer_a']}` | **Layer B Top-1**: `{rc['top1_layer_b']}` | **All Audits Passed**: `{rc['all_audits_passed']}`",
                "",
                "| Policy | Layer A Obj | Layer B Obj | Rank A | Rank B | Layer A Cmax | Layer B Cmax | B/A Cmax | Layer A Tard | Layer B Tard | Layer B p95 (ms) | Layer B p99 (ms) | Audit A/B |",
                "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for r in tdata["paired_rows"]:
            md_lines.append(
                f"| `{r['policy']}` | {r['layer_a_net_obj']:.1f} | {r['layer_b_net_obj']:.1f} | "
                f"{r['rank_layer_a']:.1f} | {r['rank_layer_b']:.1f} | "
                f"{r['layer_a_makespan']} | {r['layer_b_makespan']} | {r['makespan_ratio_b_over_a']:.2f}x | "
                f"{r['layer_a_tardiness']:.1f} | {r['layer_b_tardiness']:.1f} | "
                f"{r['layer_b_propose_ms_p95']:.2f} | {r['layer_b_propose_ms_p99']:.2f} | "
                f"{'PASS' if (r['layer_a_audit_passed'] and r['layer_b_audit_passed']) else 'FAIL'} |"
            )
        md_lines.append("")

    if gazebo_reports:
        md_lines.extend(
            [
                "## Headless WSL2 Gazebo + ROS 2 + Nav2 Full-Chain Physical Evidence",
                "",
                "| Run Tag | Scene | Policy | Passed | Sim Time (s) | Tasks | Ops | Makespan | Task Tard | Op Tard | p95 (ms) | p99 (ms) | Min Clear (m) |",
                "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
            ]
        )
        for g in gazebo_reports:
            md_lines.append(
                f"| `{g['tag']}` | `{g['scene_id']}` | `{g['policy']}` | `{g['passed']}` | "
                f"{g['sim_time_s']} | {g['tasks_completed']}/{g['tasks_total']} | {g['ops_completed']}/{g['ops_total']} | "
                f"{g['makespan_steps']} | {g['task_tardiness']} | {g['op_tardiness']} | "
                f"{g['propose_ms_p95']} | {g['propose_ms_p99']} | {g['min_clearance_m']} |"
            )
        md_lines.append("")

    md_path = out_path.with_suffix(".md")
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[Dual-Layer] Saved {out_path} and {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
