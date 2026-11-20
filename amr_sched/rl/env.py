"""Gymnasium Environment for Online Multi-AMR Rescheduling (AMRReschedulingEnv).

Wraps AMR-Scheduler's physical simulation, CVCR incremental evaluator,
and single-writer transaction owner into a standard Gymnasium reinforcement learning interface.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
import random
import time
from typing import Any, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.clock import (
    is_virtual_clock_active,
    now_s as _cvcr_now_s,
    record_work_ms as _cvcr_record_work_ms,
)
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.production import NativeProductionTracker
from amr_sched.cvcr.snapshot import NativeSnapshotCapture, OperationSnapshot
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, AgvState, Task, TaskPriority
from amr_sched.layouts import build_layout
from amr_sched.network import Network, build_demo_network
from amr_sched.planners.rust_spacetime_astar import (
    RustSpaceTimeAStarPlanner,
    is_rust_planner_available,
)
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reservation import Interval
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import build_scenario
from amr_sched.rl.spaces import (
    ACTION_KEEP,
    ACTION_NAMES,
    EXPERT_ACTION_NAMES,
    EXPERT_V6_ACTION_NAMES,
    EXPERT_V7_ACTION_NAMES,
    EXPERT_V8_ACTION_NAMES,
    EXPERT_V9_ACTION_NAMES,
    NUM_ACTIONS,
    NUM_EXPERT_ACTIONS,
    NUM_EXPERT_V6_ACTIONS,
    NUM_EXPERT_V7_ACTIONS,
    NUM_EXPERT_V8_ACTIONS,
    NUM_EXPERT_V9_ACTIONS,
    NUM_EXPERT_V10_ACTIONS,
    NUM_EXPERT_V12_ACTIONS,
    STATE_DIM,
    ActionSpec,
    compute_action_mask,
    compute_anytime_state,
    compute_expert_action_mask,
    compute_expert_state,
    compute_expert_v6_action_mask,
    compute_expert_v6_state,
    compute_expert_v7_action_mask,
    compute_expert_v7_state,
    compute_expert_v8_action_mask,
    compute_expert_v8_dual_graph,
    compute_expert_v8_state,
    compute_expert_v9_action_mask,
    compute_expert_v9_dual_graph,
    compute_expert_v9_state,
    compute_expert_v10_action_mask,
    compute_expert_v12_action_mask,
    compute_expert_v12_dual_graph,
    compute_expert_v12_state,
    compute_macro_state,
    decode_action,
    decode_expert_action,
    decode_expert_v6_action,
    decode_expert_v7_action,
    decode_expert_v8_action,
    decode_expert_v9_action,
    decode_expert_v10_action,
    decode_expert_v12_action,
)


def _tune_network(net0: Network, speed: float = 3.0) -> Network:
    """Rebuild Network with uniform traversal speed and dual-lane minimum capacity."""
    net = Network()
    for n in net0.nodes.values():
        net.add_node(replace(n, capacity=max(2, n.capacity)))
    for e in net0.edges.values():
        net.add_edge(replace(e, speed_limit=speed, capacity=max(2, e.capacity)))
    return net


def build_scenario_for_env(
    topology: str,
    seed: int,
    scenario_version: str = "v3",
) -> tuple[Network, list[Task], list[Agv], list[OperationSnapshot], dict[str, Any]]:
    """Build reproducible, de-duplicated, completable multi-disruption scenarios across Ring, Grid, and Workshop."""
    rng = random.Random(f"{topology}_{seed}_{scenario_version}")
    n_tasks = 16
    is_v3 = scenario_version != "v2"
    fault_window = (1 if (seed % 2 == 1) else 6) if is_v3 else 8
    event2_regime = (
        ("OPEN_WINDOW_KITTING_BOTTLENECK" if (seed % 3 != 0) else "HIGH_COMMITMENT_FROZEN")
        if is_v3
        else "URGENT_STANDARD"
    )

    if topology == "ring":
        net = _tune_network(build_demo_network(aisle_capacity=2), speed=3.0)
        bays = ["W1", "W2", "W3", "W4", "W5", "W6"]
        ops = []
        tasks = []
        for o_idx in range(n_tasks // 2):
            op_id = f"OP_{o_idx + 1}"
            # Two parallel 4-stage assembly chains (OP_1..4, OP_5..8)
            pred = f"OP_{o_idx}" if o_idx % 4 != 0 else None
            bay = rng.choice(bays)
            alt_bay = rng.choice([b for b in bays if b != bay])
            alt_bays = tuple(sorted({bay, alt_bay}))
            t1_id = f"T{o_idx * 2 + 1}"
            t2_id = f"T{o_idx * 2 + 2}"
            rel = (o_idx // 2) * 5 + rng.randint(0, 3)
            op_due = (125 if is_v3 else 160) + o_idx * (16 if is_v3 else 20) + rng.randint(0, 10)
            t1_due = (105 if is_v3 else 150) + o_idx * (14 if is_v3 else 18) + rng.randint(0, 10)
            t2_due = (110 if is_v3 else 155) + o_idx * (14 if is_v3 else 18) + rng.randint(0, 10)
            ops.append(
                OperationSnapshot(
                    op_id=op_id,
                    kit_task_ids=(t1_id, t2_id),
                    bay_id=bay,
                    crew_id=1 if o_idx < 4 else 2,
                    tool_id=1 if o_idx < 4 else 2,
                    proc_steps=15 + (o_idx % 2) * 3,
                    due_step=op_due,
                    predecessor_op_id=pred,
                    eligible_bays=alt_bays,
                    eligible_crews=(1, 2),
                )
            )
            src_pool = [w for w in bays if w != bay]
            src1 = rng.choice(src_pool)
            src2 = rng.choice(src_pool)
            t1 = Task(t1_id, pickup=src1, dropoff=bay, release_step=rel, due_step=t1_due)
            t1.kit_op_id = op_id
            t2 = Task(t2_id, pickup=src2, dropoff=bay, release_step=rel, due_step=t2_due)
            t2.kit_op_id = op_id
            tasks.extend([t1, t2])
        start_nodes = rng.sample(bays, 6)
        agvs = [Agv(f"agv_{i}", max_speed=3.0, start_node=start_nodes[i]) for i in range(6)]
        u_pick, u_drop = rng.sample(bays, 2)
        disruptions = {
            "fault_agv": agvs[rng.randrange(len(agvs))].agv_id,
            "fault_step": rng.randint(10, 22),
            "fault_window": fault_window,
            "event2_regime": event2_regime,
            "kitting_bottleneck_window": 3,
            "kitting_bottleneck_delay": 45,
            "urgent_task_id": "T_URGENT_1",
            "urgent_pickup": u_pick,
            "urgent_dropoff": u_drop,
            "urgent_release_step": rng.randint(32, 48),
            "urgent_due_step": rng.randint(35, 55) if is_v3 else rng.randint(70, 95),
            "post_pickup_fault_step": rng.randint(65, 85),
            "post_pickup_fault_duration": 18,
        }

    elif topology == "grid":
        net = _tune_network(build_layout("grid"), speed=3.0)
        nodes = sorted(net.nodes.keys())
        ops = []
        tasks = []
        for o_idx in range(n_tasks // 2):
            op_id = f"OP_{o_idx + 1}"
            pred = f"OP_{o_idx}" if o_idx % 4 != 0 else None
            bay = rng.choice(nodes)
            alt_bay = rng.choice([n for n in nodes if n != bay])
            t1_id = f"T{o_idx * 2 + 1}"
            t2_id = f"T{o_idx * 2 + 2}"
            rel = (o_idx // 2) * 5 + rng.randint(0, 3)
            op_due = (128 if is_v3 else 165) + o_idx * (16 if is_v3 else 20) + rng.randint(0, 10)
            t1_due = (105 if is_v3 else 150) + o_idx * (14 if is_v3 else 18) + rng.randint(0, 10)
            t2_due = (110 if is_v3 else 155) + o_idx * (14 if is_v3 else 18) + rng.randint(0, 10)
            ops.append(
                OperationSnapshot(
                    op_id=op_id,
                    kit_task_ids=(t1_id, t2_id),
                    bay_id=bay,
                    crew_id=1 if o_idx < 4 else 2,
                    tool_id=1 if o_idx < 4 else 2,
                    proc_steps=15 + (o_idx % 2) * 3,
                    due_step=op_due,
                    predecessor_op_id=pred,
                    eligible_bays=tuple(sorted({bay, alt_bay})),
                    eligible_crews=(1, 2),
                )
            )
            src_pool = [n for n in nodes if n != bay]
            src1 = rng.choice(src_pool)
            src2 = rng.choice(src_pool)
            t1 = Task(t1_id, pickup=src1, dropoff=bay, release_step=rel, due_step=t1_due)
            t1.kit_op_id = op_id
            t2 = Task(t2_id, pickup=src2, dropoff=bay, release_step=rel, due_step=t2_due)
            t2.kit_op_id = op_id
            tasks.extend([t1, t2])
        start_nodes = rng.sample(nodes, 6)
        agvs = [Agv(f"agv_{i}", max_speed=3.0, start_node=start_nodes[i]) for i in range(6)]
        u_pick, u_drop = rng.sample(nodes, 2)
        disruptions = {
            "fault_agv": agvs[rng.randrange(len(agvs))].agv_id,
            "fault_step": rng.randint(10, 22),
            "fault_window": fault_window,
            "event2_regime": event2_regime,
            "kitting_bottleneck_window": 3,
            "kitting_bottleneck_delay": 45,
            "urgent_task_id": "T_URGENT_1",
            "urgent_pickup": u_pick,
            "urgent_dropoff": u_drop,
            "urgent_release_step": rng.randint(32, 48),
            "urgent_due_step": rng.randint(35, 55) if is_v3 else rng.randint(70, 95),
            "post_pickup_fault_step": rng.randint(65, 85),
            "post_pickup_fault_duration": 18,
        }

    else:
        # 62-node Workshop layout
        sub_seed = rng.randint(1, 1_000_000_000)
        net0, tasks_raw, agvs_raw = build_scenario(
            n_tasks=n_tasks, n_agvs=6, release_window=45, seed=sub_seed
        )
        net = _tune_network(net0, speed=3.0)
        valid_bays = sorted({t.dropoff for t in tasks_raw})
        valid_pickups = sorted({t.pickup for t in tasks_raw})
        ops = []
        tasks = []
        for o_idx in range(n_tasks // 2):
            op_id = f"OP_{o_idx + 1}"
            pred = f"OP_{o_idx}" if o_idx % 4 != 0 else None
            t_raw1 = tasks_raw[o_idx * 2]
            t_raw2 = tasks_raw[o_idx * 2 + 1]
            bay = t_raw1.dropoff
            alt_bays_cand = [b for b in valid_bays if b != bay]
            alt_bay = rng.choice(alt_bays_cand) if alt_bays_cand else bay
            rel = (o_idx // 2) * 5 + rng.randint(0, 3)
            op_due = (130 if is_v3 else 170) + o_idx * (16 if is_v3 else 20) + rng.randint(0, 10)
            t1_due = (108 if is_v3 else 150) + o_idx * (14 if is_v3 else 18) + rng.randint(0, 10)
            t2_due = (112 if is_v3 else 155) + o_idx * (14 if is_v3 else 18) + rng.randint(0, 10)

            t1 = Task(
                task_id=t_raw1.task_id,
                pickup=t_raw1.pickup,
                dropoff=bay,
                release_step=rel,
                due_step=t1_due,
            )
            t1.kit_op_id = op_id
            t2 = Task(
                task_id=t_raw2.task_id,
                pickup=t_raw2.pickup,
                dropoff=bay,
                release_step=rel,
                due_step=t2_due,
            )
            t2.kit_op_id = op_id
            tasks.extend([t1, t2])
            ops.append(
                OperationSnapshot(
                    op_id=op_id,
                    kit_task_ids=(t1.task_id, t2.task_id),
                    bay_id=bay,
                    crew_id=1 if o_idx < 4 else 2,
                    tool_id=1 if o_idx < 4 else 2,
                    proc_steps=15 + (o_idx % 2) * 3,
                    due_step=op_due,
                    predecessor_op_id=pred,
                    eligible_bays=tuple(sorted({bay, alt_bay})),
                    eligible_crews=(1, 2),
                )
            )
        agvs = [Agv(a.agv_id, max_speed=3.0, start_node=a.start_node) for a in agvs_raw]
        u_pick = rng.choice(valid_pickups)
        u_drop = rng.choice([b for b in valid_bays if b != u_pick] or valid_bays)
        disruptions = {
            "fault_agv": agvs[rng.randrange(len(agvs))].agv_id,
            "fault_step": rng.randint(10, 22),
            "fault_window": fault_window,
            "event2_regime": event2_regime,
            "kitting_bottleneck_window": 3,
            "kitting_bottleneck_delay": 45,
            "urgent_task_id": "T_URGENT_W",
            "urgent_pickup": u_pick,
            "urgent_dropoff": u_drop,
            "urgent_release_step": rng.randint(32, 48),
            "urgent_due_step": rng.randint(35, 55) if is_v3 else rng.randint(70, 95),
            "post_pickup_fault_step": rng.randint(65, 85),
            "post_pickup_fault_duration": 18,
        }

    return net, tasks, agvs, ops, disruptions


class AMRReschedulingEnv(gym.Env):
    """Gymnasium environment wrapping AMR-Scheduler simulation and CVCR replanning."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        topology: str = "workshop",
        seed: int = 42,
        max_steps: int = 1000,
        lambda_churn: float = 2.0,
        lambda_budget: float = 1.0,
        lambda_wait: float = 0.5,
        use_rust: bool = True,
        enable_commitment_lock: bool = True,
        enable_kitting_coupling: bool = True,
        enable_compute_latency: bool = True,
        scenario_version: str = "v3",
        allow_kitting_shortcut: bool = False,
        pure_physical_reward: bool = True,
        real_compute_window_coupling: bool = True,
        compute_step_ms: float = 20.0,
        solver_override: Optional[str] = None,
        jitter_windows: bool = False,
        expert_portfolio: bool = False,
        expert_v6_portfolio: bool = False,
        expert_v7_portfolio: bool = False,
        expert_v8_portfolio: bool = False,
        expert_v9_portfolio: bool = False,
        expert_v10_portfolio: bool = False,
        expert_v12_portfolio: bool = False,
        lrho_predictor: Any = None,
        destroy_selector: Any = None,
        record_trajectory: bool = False,
        external_proposer: Any = None,
        loaded_repair_mode: str = "none",
        loaded_repair_threshold: int = 8,
        kitting_ranker: Any = None,
        v13_kitting_mode: Optional[str] = None,
        v14_policy: Any = None,
        v15_policy: Any = None,
        execution_perturbation: Any = None,
        execution_aware: bool = True,
        p4_choke_robust: bool = False,
        enable_p4_voi_probe: Optional[bool] = None,
    ) -> None:
        super().__init__()
        if loaded_repair_mode not in ("none", "solo_reroute", "coordinated_neighbor_repair", "haven_v12"):
            raise ValueError(f"Unsupported loaded_repair_mode: {loaded_repair_mode}")
        self.topology = topology
        self.default_seed = seed
        self.max_steps = max_steps
        self.lambda_churn = lambda_churn
        self.lambda_budget = lambda_budget
        self.lambda_wait = lambda_wait
        self.use_rust = use_rust and is_rust_planner_available()
        self.enable_commitment_lock = bool(enable_commitment_lock)
        self.enable_kitting_coupling = bool(enable_kitting_coupling)
        self.enable_compute_latency = bool(enable_compute_latency)
        self.scenario_version = scenario_version
        self.allow_kitting_shortcut = bool(allow_kitting_shortcut)
        self.pure_physical_reward = bool(pure_physical_reward)
        self.real_compute_window_coupling = bool(real_compute_window_coupling)
        self.compute_step_ms = float(compute_step_ms)
        self.solver_override = solver_override
        self.jitter_windows = bool(jitter_windows)
        self.expert_portfolio = bool(expert_portfolio)
        self.expert_v6_portfolio = bool(expert_v6_portfolio)
        self.expert_v7_portfolio = bool(expert_v7_portfolio)
        self.expert_v8_portfolio = bool(expert_v8_portfolio)
        self.expert_v9_portfolio = bool(expert_v9_portfolio)
        self.expert_v10_portfolio = bool(expert_v10_portfolio)
        self.expert_v12_portfolio = bool(expert_v12_portfolio)
        self.lrho_predictor = lrho_predictor
        self.destroy_selector = destroy_selector
        self.record_trajectory = bool(record_trajectory)
        self.external_proposer = external_proposer
        self.loaded_repair_mode = str(loaded_repair_mode)
        self.loaded_repair_threshold = int(loaded_repair_threshold)
        self.kitting_ranker = kitting_ranker
        self.v13_kitting_mode = v13_kitting_mode
        self.v14_policy = v14_policy
        self.v15_policy = v15_policy
        self.execution_perturbation_spec = execution_perturbation
        self.execution_aware = bool(execution_aware)
        self.p4_choke_robust = bool(p4_choke_robust)
        self.enable_p4_voi_probe: Optional[bool] = (
            bool(enable_p4_voi_probe) if enable_p4_voi_probe is not None else None
        )
        self.execution_perturbation: Any = None
        self._active_execution_aware: bool = bool(execution_aware)
        self._active_p4_choke_robust: bool = bool(p4_choke_robust)
        self.proposal_task_history: dict[str, dict[str, Any]] = {}
        self.loaded_repair_events: dict[tuple[str, str], dict[str, Any]] = {}
        self.total_loaded_repairs: int = 0
        self.cp_stats_totals: dict[str, int] = {
            "cp_invoked": 0,
            "cp_feasible": 0,
            "cp_entered_eval": 0,
            "cp_accepted_improved": 0,
            "cp_queue_only": 0,
        }

        if self.expert_v12_portfolio:
            n_act = NUM_EXPERT_V12_ACTIONS
        elif self.expert_v10_portfolio:
            n_act = NUM_EXPERT_V10_ACTIONS
        elif self.expert_v9_portfolio:
            n_act = NUM_EXPERT_V9_ACTIONS
        elif self.expert_v8_portfolio:
            n_act = NUM_EXPERT_V8_ACTIONS
        elif self.expert_v7_portfolio:
            n_act = NUM_EXPERT_V7_ACTIONS
        elif self.expert_v6_portfolio:
            n_act = NUM_EXPERT_V6_ACTIONS
        elif self.expert_portfolio:
            n_act = NUM_EXPERT_ACTIONS
        else:
            n_act = NUM_ACTIONS
        self.action_space = spaces.Discrete(n_act)
        self.observation_space = spaces.Box(
            low=-2.0, high=5.0, shape=(STATE_DIM,), dtype=np.float32
        )

        # Environment runtime state
        self.sim: Optional[Simulator] = None
        self.owner: Optional[NativeTransactionOwner] = None
        self.tracker: Optional[NativeProductionTracker] = None
        self.disruptions: dict[str, Any] = {}
        self.fault_injected: bool = False
        self.fault_injected_step: int = -1
        self.urgent_injected: bool = False
        self.urgent_injected_step: int = -1
        self.post_pickup_fault_injected: bool = False
        self.post_pickup_fault_agv: Optional[str] = None
        self.post_pickup_recover_step: int = 0
        self.active_pre_pickup_fault_agv: Optional[str] = None
        self.active_fault_task_id: Optional[str] = None
        self.active_kitting_task_id: Optional[str] = None
        self.active_kitting_op_id: Optional[str] = None
        self.kitting_delay_applied: int = 0

        self.current_step: int = 0
        self.last_replan_tick: int = 0
        self.recent_conflict_ticks: int = 0
        self.conflict_window: list[int] = []

        # Episode telemetry
        self.total_reward: float = 0.0
        self.total_churn: int = 0
        self.total_replans: int = 0
        self.total_compute_ms: float = 0.0
        self.total_window_expirations: int = 0
        self.total_kitting_preemptions: int = 0
        self.decision_logs: list[dict[str, Any]] = []

    def get_anytime_obs(self, base_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute enriched 22-D anytime observation including search & production telemetry."""
        assert self.sim is not None and self.tracker is not None
        return compute_anytime_state(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
            decision_logs=self.decision_logs,
        )

    def get_expert_obs(self, base_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute enriched 24-D Multi-Expert observation including causal layout signals."""
        assert self.sim is not None and self.tracker is not None
        return compute_expert_state(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
            decision_logs=self.decision_logs,
        )

    def get_expert_v6_obs(self, base_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute enriched 26-D V6 Multi-Expert observation adding split-kit & bay contention."""
        assert self.sim is not None and self.tracker is not None
        return compute_expert_v6_state(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
            decision_logs=self.decision_logs,
        )

    def get_expert_v7_obs(self, base_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute enriched 30-D V7 Multi-Expert observation adding topology-agnostic bottleneck signals."""
        assert self.sim is not None and self.tracker is not None
        return compute_expert_v7_state(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
            decision_logs=self.decision_logs,
        )

    def get_expert_v8_obs(self, base_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute strictly causal 36-D V8 state combining 30-D V7 + 6 RCDR/LTM drift signals."""
        assert self.sim is not None and self.tracker is not None
        return compute_expert_v8_state(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
            decision_logs=self.decision_logs,
        )

    def get_expert_v8_dual_graph(self, base_obs: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        """Compute strictly G3-causal E2G-MoE dual-graph observation tensors and bandit summary."""
        assert self.sim is not None and self.tracker is not None
        return compute_expert_v8_dual_graph(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
            decision_logs=self.decision_logs,
        )

    def get_expert_v9_obs(self, base_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute strictly G3-causal 38-D V9 state adding window tightness & CP-SAT coupling."""
        assert self.sim is not None and self.tracker is not None
        return compute_expert_v9_state(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
            decision_logs=self.decision_logs,
        )

    def get_expert_v9_dual_graph(self, base_obs: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        """Compute strictly G3-causal V9 dual-graph observation tensors and 56-D bandit summary."""
        assert self.sim is not None and self.tracker is not None
        return compute_expert_v9_dual_graph(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
            decision_logs=self.decision_logs,
        )

    def get_expert_v12_obs(self, base_obs: Optional[np.ndarray] = None) -> np.ndarray:
        """Compute strictly causal 44-D V12 state adding Detour-Delta, Haven, & CP signals."""
        assert self.sim is not None and self.tracker is not None
        return compute_expert_v12_state(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
            decision_logs=self.decision_logs,
        )

    def get_expert_v12_dual_graph(self, base_obs: Optional[np.ndarray] = None) -> dict[str, np.ndarray]:
        """Compute strictly causal V12 dual-graph observation tensors and 44-D global vector."""
        assert self.sim is not None and self.tracker is not None
        return compute_expert_v12_dual_graph(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
            decision_logs=self.decision_logs,
        )

    def _current_mask(self, base_obs: np.ndarray) -> np.ndarray:
        """Return causal action mask for the configured portfolio mode."""
        if self.expert_v12_portfolio:
            return compute_expert_v12_action_mask(self.sim, last_replan_tick=self.last_replan_tick)
        if self.expert_v10_portfolio:
            return compute_expert_v10_action_mask(self.sim, last_replan_tick=self.last_replan_tick)
        if self.expert_v9_portfolio:
            return compute_expert_v9_action_mask(self.sim, last_replan_tick=self.last_replan_tick)
        if self.expert_v8_portfolio:
            return compute_expert_v8_action_mask(self.sim, last_replan_tick=self.last_replan_tick)
        if self.expert_v7_portfolio:
            return compute_expert_v7_action_mask(self.sim, last_replan_tick=self.last_replan_tick)
        if self.expert_v6_portfolio:
            return compute_expert_v6_action_mask(self.sim, last_replan_tick=self.last_replan_tick)
        if self.expert_portfolio:
            return compute_expert_action_mask(self.sim, last_replan_tick=self.last_replan_tick)
        return compute_action_mask(
            self.sim,
            self.tracker,
            base_obs,
            last_replan_tick=self.last_replan_tick,
        )

    def _decode_selected_action(self, action: int) -> ActionSpec:
        """Decode integer action using the configured portfolio specification."""
        if self.expert_v12_portfolio:
            return decode_expert_v12_action(action)
        if self.expert_v10_portfolio:
            return decode_expert_v10_action(action)
        if self.expert_v9_portfolio:
            return decode_expert_v9_action(action)
        if self.expert_v8_portfolio:
            return decode_expert_v8_action(action)
        if self.expert_v7_portfolio:
            return decode_expert_v7_action(action)
        if self.expert_v6_portfolio:
            return decode_expert_v6_action(action)
        if self.expert_portfolio:
            return decode_expert_action(action)
        return decode_action(action)

    def compute_episode_metrics(self) -> dict[str, Any]:
        """Compute canonical operational and net-benefit episode metrics."""
        assert self.sim is not None and self.tracker is not None
        res = self.sim._build_result()
        cmax = int(self.sim._step_cursor)

        completed_tasks = len(res.completed_tasks())
        total_tasks = len(self.sim.tasks)
        task_rate = float(completed_tasks) / max(1, total_tasks)

        completed_ops = sum(
            1 for op in self.tracker.operations
            if op.started and op.end_step is not None and cmax >= op.end_step
        )
        total_ops = len(self.tracker.operations)
        prod_rate = float(completed_ops) / max(1, total_ops)

        task_tardiness = sum(
            max(0, (int(t.complete_step) if t.complete_step is not None else cmax) - int(t.due_step))
            for t in self.sim.tasks
            if t.due_step is not None
        )
        op_tardiness = sum(
            max(0, (int(op.end_step) if (op.started and op.end_step is not None) else cmax) - int(op.due_step))
            for op in self.tracker.operations
        )
        total_tardiness = float(task_tardiness + op_tardiness)

        uncompleted_tasks = total_tasks - completed_tasks
        uncompleted_ops = total_ops - completed_ops
        task_tardiness_cost = 2.0 * float(task_tardiness)
        op_tardiness_cost = 2.0 * float(op_tardiness)
        unfinished_task_penalty = float(uncompleted_tasks) * 5000.0
        unfinished_op_penalty = float(uncompleted_ops) * 3000.0
        churn_cost = 15.0 * float(self.total_churn)
        compute_cost = 0.05 * float(self.total_compute_ms)
        total_obj = float(
            cmax
            + task_tardiness_cost
            + op_tardiness_cost
            + unfinished_task_penalty
            + unfinished_op_penalty
        )
        net_obj = float(
            total_obj + churn_cost + compute_cost
        )
        total_q_head = sum(int(d.get("queue_head_changes", 0)) for d in self.decision_logs)
        total_q_tail = sum(int(d.get("queue_tail_changes", 0)) for d in self.decision_logs)
        total_q_all = sum(int(d.get("queue_reorders", 0)) for d in self.decision_logs)
        total_reassign_only = sum(int(d.get("reassigned_existing_tasks", 0)) for d in self.decision_logs)
        total_first_urgent = sum(int(d.get("newly_assigned_urgent", 0)) for d in self.decision_logs)
        proposed_total = len(self.proposal_task_history)
        superseded_total = sum(int(v.get("superseded_count", 0)) for v in self.proposal_task_history.values())
        executed_prefix_total = sum(1 for v in self.proposal_task_history.values() if v.get("executed_prefix"))
        stall_durations = [
            int(ev["delivery_step"]) - int(ev["first_stall_step"])
            for ev in self.loaded_repair_events.values()
            if ev.get("delivery_step") is not None and ev.get("first_stall_step") is not None
        ]
        stall_recovery_moves = [
            int(ev["recovery_move_step"]) - int(ev["repaired_step"])
            for ev in self.loaded_repair_events.values()
            if ev.get("recovery_move_step") is not None and ev.get("repaired_step") is not None
        ]
        return {
            "cmax": cmax,
            "completed_tasks": f"{completed_tasks}/{total_tasks}",
            "task_completion_rate": round(task_rate, 4),
            "completed_ops": f"{completed_ops}/{total_ops}",
            "production_completion_rate": round(prod_rate, 4),
            "uncompleted_tasks": int(uncompleted_tasks),
            "uncompleted_ops": int(uncompleted_ops),
            "total_tardiness": round(total_tardiness, 2),
            "task_tardiness_cost": round(task_tardiness_cost, 4),
            "op_tardiness_cost": round(op_tardiness_cost, 4),
            "unfinished_task_penalty": round(unfinished_task_penalty, 4),
            "unfinished_op_penalty": round(unfinished_op_penalty, 4),
            "churn_cost": round(churn_cost, 4),
            "compute_cost": round(compute_cost, 4),
            "total_obj": round(total_obj, 2),
            "net_obj": round(net_obj, 2),
            "total_churn": int(self.total_churn),
            "total_reassigned_tasks": int(total_reassign_only),
            "total_first_urgent_dispatches": int(total_first_urgent),
            "total_queue_reorders": int(total_q_all),
            "total_queue_head_changes": int(total_q_head),
            "total_queue_tail_changes": int(total_q_tail),
            "total_replans": int(self.total_replans),
            "total_compute_ms": round(float(self.total_compute_ms), 2),
            "total_window_expirations": int(self.total_window_expirations),
            "total_kitting_preemptions": int(self.total_kitting_preemptions),
            "total_loaded_repairs": int(self.total_loaded_repairs),
            "stalled_picked_events": int(len(self.loaded_repair_events)),
            "stalled_picked_recovered_count": int(len(stall_durations)),
            "mean_stall_to_delivery_steps": (
                round(sum(stall_durations) / len(stall_durations), 2) if stall_durations else 0.0
            ),
            "mean_repair_to_move_steps": (
                round(sum(stall_recovery_moves) / len(stall_recovery_moves), 2) if stall_recovery_moves else 0.0
            ),
            "proposed_tasks_total": int(proposed_total),
            "superseded_proposals_total": int(superseded_total),
            "executed_prefix_count": int(executed_prefix_total),
            "cp_stats_totals": dict(self.cp_stats_totals),
            "execution_aware": bool(self._active_execution_aware),
            "p2_perturbation": (
                self.execution_perturbation.summary()
                if self.execution_perturbation is not None and hasattr(self.execution_perturbation, "summary")
                else None
            ),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        effective_seed = seed if seed is not None else self.default_seed
        if options and "topology" in options:
            self.topology = options["topology"]
        stress_axis = str(options.get("stress_axis", "standard")) if options else "standard"
        n_agvs_override = options.get("n_agvs_override") if options else None

        if options and "scenario_tuple" in options and options["scenario_tuple"] is not None:
            net, tasks, agvs, ops, disruptions = options["scenario_tuple"]
        elif options and "v13_kitting_spec" in options and options["v13_kitting_spec"] is not None:
            from amr_sched.scenarios_v13_kitting import build_v13_paired_dev_scenario
            spec_v13 = dict(options["v13_kitting_spec"])
            net, tasks, agvs, ops, disruptions = build_v13_paired_dev_scenario(
                seed=effective_seed,
                **spec_v13,
            )
        elif self.topology in ("unseen_ladder", "unseen_choke_yard") or stress_axis != "standard" or n_agvs_override is not None:
            from amr_sched.scenarios_v10 import build_v10_stress_scenario
            net, tasks, agvs, ops, disruptions = build_v10_stress_scenario(
                self.topology,
                effective_seed,
                stress_axis=stress_axis,
                n_agvs_override=n_agvs_override,
            )
        else:
            net, tasks, agvs, ops, disruptions = build_scenario_for_env(
                self.topology, effective_seed, scenario_version=self.scenario_version
            )
        if not self.enable_kitting_coupling:
            for op in ops:
                op.predecessor_op_id = None
                op.predecessor_op_ids = ()
        if self.jitter_windows:
            jrng = random.Random(int(effective_seed) * 997 + len(tasks) + len(ops))
            disruptions["fault_window"] = jrng.choice((1, 2, 3, 4, 6))
            disruptions["kitting_bottleneck_window"] = jrng.choice((1, 2, 3, 4, 6))
        self.disruptions = disruptions
        self.fault_injected = False
        self.fault_injected_step = -1
        self.urgent_injected = False
        self.urgent_injected_step = -1
        self.post_pickup_fault_injected = False
        self.post_pickup_fault_agv = None
        self.post_pickup_recover_step = 0
        self.active_pre_pickup_fault_agv = None
        self.active_fault_task_id = None
        self.active_kitting_task_id = None
        self.active_kitting_op_id = None
        self.kitting_delay_applied = 0

        planner_cls = RustSpaceTimeAStarPlanner if self.use_rust else SpaceTimeAStarPlanner
        planner = planner_cls(max_speed=3.0, dt=1.0)
        allocator = GreedyAllocator()
        ep_spec = options.get("execution_perturbation", self.execution_perturbation_spec) if options else self.execution_perturbation_spec
        self._active_execution_aware = bool(
            options.get("execution_aware", self.execution_aware) if options else self.execution_aware
        )
        self._active_p4_choke_robust = bool(
            options.get("p4_choke_robust", self.p4_choke_robust) if options else self.p4_choke_robust
        )
        if ep_spec is None:
            self.execution_perturbation = None
        else:
            from dataclasses import replace as _dc_replace
            from amr_sched.cvcr.p2_perturbation import ExecutionPerturbationModel, P2PerturbationConfig
            if isinstance(ep_spec, ExecutionPerturbationModel):
                self.execution_perturbation = ep_spec
            elif isinstance(ep_spec, P2PerturbationConfig):
                cfg_ep = ep_spec if int(ep_spec.seed) != 0 else _dc_replace(ep_spec, seed=int(effective_seed))
                self.execution_perturbation = ExecutionPerturbationModel(cfg_ep)
            elif isinstance(ep_spec, dict):
                d_ep = dict(ep_spec)
                d_ep.setdefault("seed", int(effective_seed))
                self.execution_perturbation = ExecutionPerturbationModel(P2PerturbationConfig(**d_ep))
            else:
                self.execution_perturbation = ep_spec
        self.sim = Simulator(
            network=net,
            tasks=tasks,
            agvs=agvs,
            planner=planner,
            allocator=allocator,
            config=SimulationConfig(
                dt=1.0,
                max_steps=self.max_steps,
                node_hold_steps=6,
                deadlock_timeout=20,
                record_trajectory=bool(
                    options.get("record_trajectory", self.record_trajectory)
                    if options is not None
                    else self.record_trajectory
                ),
            ),
            perturbation_model=self.execution_perturbation,
        )
        self.sim.reset()
        self.sim._p2_execution_aware = bool(self._active_execution_aware)
        self.sim._p4_choke_robust = bool(self._active_p4_choke_robust)
        if self._active_p4_choke_robust or str(self.v13_kitting_mode or "").startswith(("V15_", "V16_", "V17_")):
            from amr_sched.cvcr.p4_belief_choke import record_node_history_tick
            record_node_history_tick(self.sim, 0)

        self.tracker = NativeProductionTracker(ops, self.sim.tasks)
        self.sim._prod_tracker = self.tracker
        self.owner = NativeTransactionOwner(self.sim, ops, production_tracker=self.tracker)

        self.current_step = 0
        self.last_replan_tick = 0
        self.recent_conflict_ticks = 0
        self.conflict_window = [0] * 10
        bm = getattr(self.execution_perturbation, "_belief_model", None) if self.execution_perturbation is not None else None
        if bm is not None and self.execution_perturbation is not None:
            if self.enable_p4_voi_probe is not None:
                bm.enable_voi_probe = bool(self.enable_p4_voi_probe)
            elif str(self.v13_kitting_mode or "").startswith(("V13_", "V16_", "V17_")) and self.execution_perturbation.config.enable_p4_belief:
                bm.enable_voi_probe = True

        self.total_reward = 0.0
        self.total_churn = 0
        self.total_replans = 0
        self.total_compute_ms = 0.0
        self.total_window_expirations = 0
        self.total_kitting_preemptions = 0
        self.decision_logs = []
        self.sim._env_last_replan_tick = 0
        self.sim._env_recent_conflicts = 0
        self.sim._env_decision_logs = self.decision_logs
        self.sim._execution_feedback = None
        self.sim._execution_feedback_step = -1
        self.sim._causal_drift_observed = False
        if self.execution_perturbation is not None and self._active_execution_aware:
            fb0 = self.execution_perturbation.capture_feedback(self.sim, 0)
            self.sim._execution_feedback = fb0
            self.sim._execution_feedback_step = 0
            if fb0 is not None and not fb0.is_empty():
                self.sim._causal_drift_observed = True
        self.proposal_task_history = {}
        self.loaded_repair_events = {}
        self.total_loaded_repairs = 0
        self.cp_stats_totals = {
            "cp_invoked": 0,
            "cp_feasible": 0,
            "cp_entered_eval": 0,
            "cp_accepted_improved": 0,
            "cp_queue_only": 0,
        }

        obs = compute_macro_state(
            self.sim,
            self.tracker,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
        )
        mask = self._current_mask(obs)

        info = {
            "action_mask": mask,
            "step": self.current_step,
            "topology": self.topology,
            "seed": effective_seed,
        }
        return obs, info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.sim is not None and self.owner is not None and self.tracker is not None

        # 1. Action Mask Check
        cur_obs = compute_macro_state(
            self.sim,
            self.tracker,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
        )
        mask = self._current_mask(cur_obs)

        invalid_action_penalty = 0.0
        if mask[action] == 0:
            # Action is physically/logically masked out
            invalid_action_penalty = -5.0
            # Fall back to KEEP if masked action is requested
            action = ACTION_KEEP if mask[ACTION_KEEP] == 1 else int(np.argmax(mask))

        def _task_stat(t: Any) -> str:
            s = getattr(t, "status", "")
            return getattr(s, "value", str(s)).lower()

        def _agv_stat(a: Any) -> str:
            s = getattr(a, "state", "")
            return getattr(s, "value", str(s)).lower()

        spec = self._decode_selected_action(action)
        step_churn = 0
        changed_tasks = 0
        reassigned_existing_tasks = 0
        changed_ops = 0
        newly_assigned_urgent = 0
        queue_reorders = 0
        queue_head_changes = 0
        queue_tail_changes = 0
        step_compute_ms = 0.0
        receipt_accepted = False
        rescued_tasks = 0
        primary_rescued_tasks = 0
        preempted_kitting = 0
        urgent_scheduled = 0
        window_expired = False
        improved_delta = 0.0
        step_now = int(self.sim._step_cursor)
        faulted_before_step = set(getattr(self.sim, "_faulted_agvs", set()))

        # Factor C (disabled): relax post-pickup task freeze on faulted robots for counterfactual studies
        if not self.enable_commitment_lock:
            for t in self.sim.tasks:
                if _task_stat(t) == "picked" and t.assigned_agv in faulted_before_step:
                    from amr_sched.entities import TaskStatus
                    t.status = TaskStatus.ASSIGNED
                    t.adjustable_window_steps = 8

        fault_target_tasks = [
            t for t in self.sim.tasks
            if _task_stat(t) == "assigned" and t.assigned_agv in faulted_before_step
        ]
        open_fault_tasks = [
            t for t in fault_target_tasks
            if step_now >= getattr(t, "secondary_window_open_step", 0)
        ]
        in_secondary_hold = bool(fault_target_tasks and not open_fault_tasks)
        has_unrescued_fault_task = bool(open_fault_tasks)

        t_kit = (
            self.sim._task_by_id.get(self.active_kitting_task_id)
            if self.active_kitting_task_id
            else None
        )
        has_open_kitting_bottleneck = bool(
            t_kit is not None
            and getattr(t_kit, "kitting_bottleneck_active", False)
            and int(getattr(t_kit, "adjustable_window_steps", 0)) > 0
        )

        # 2. Rescheduling Decision Execution with Measured Compute-Window Coupling
        propose_ms = 0.0
        validate_commit_ms = 0.0
        window_cost_steps = 0
        live_repair_count = 0
        live_repair_stats: dict[str, Any] = {}
        scope_to_use = str(self.solver_override or spec.scope or "")
        if (
            self.v13_kitting_mode
            and spec.is_replan
            and action != 1
            and (str(self.v13_kitting_mode).startswith(("V15_", "V16_", "V17_")) or not (open_fault_tasks and action == 2))
        ):
            scope_to_use = str(self.v13_kitting_mode)
        if spec.is_replan and (spec.scope is not None or self.external_proposer is not None):
            faulted_now = faulted_before_step
            prev_assign = {t.task_id: t.assigned_agv for t in self.sim.tasks if t.assigned_agv is not None}
            prev_bays = {op.op_id: op.bay_id for op in self.tracker.operations}
            prev_crews = {op.op_id: op.crew_id for op in self.tracker.operations}
            prev_queues = {
                aid: list(q)
                for aid, q in getattr(self.sim, "_pinned_task_queues", {}).items()
            }
            prev_faulted_tasks = {t.task_id for t in open_fault_tasks}
            prev_urgent_unassigned = {
                t.task_id for t in self.sim.tasks
                if _task_stat(t) in ("pending", "unassigned")
                and t.release_step <= step_now
                and not getattr(t, "hidden_until_release", False)
                and (getattr(t, "is_urgent", False) or str(getattr(t, "priority", "")).lower() in ("urgent", "1", "taskpriority.urgent"))
            }
            prev_jobs_agvs = {
                a.agv_id: a.cur_task for a in self.sim.agvs
                if _agv_stat(a) == "to_pickup" and a.cur_task and a.agv_id not in faulted_now
            }

            proposal = None
            receipt_reasons: list[str] = []
            receipt_cand_digest = ""
            if self.enable_compute_latency and in_secondary_hold:
                window_expired = True
            else:
                if self.execution_perturbation is not None and self._active_execution_aware:
                    if getattr(self.sim, "_execution_feedback_step", -1) == step_now:
                        p2_fb = getattr(self.sim, "_execution_feedback", None)
                    else:
                        p2_fb = self.execution_perturbation.capture_feedback(self.sim, step_now)
                        self.sim._execution_feedback = p2_fb
                        self.sim._execution_feedback_step = step_now
                        if p2_fb is not None and not p2_fb.is_empty():
                            self.sim._causal_drift_observed = True
                else:
                    p2_fb = None
                snap = NativeSnapshotCapture.capture(
                    self.sim,
                    self.tracker.operations,
                    hide_future_releases=True,
                    execution_feedback=p2_fb,
                )
                t_p0 = _cvcr_now_s()
                if self.external_proposer is not None:
                    proposal = self.external_proposer(snap, self.sim.network, budget_ms=spec.budget_ms)
                else:
                    dd_sig = None
                    if scope_to_use.startswith(("V12_", "V13_", "V14_", "V15_", "V16_", "V17_")):
                        from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
                        dd_sig = compute_detour_delta_signals(
                            self.sim.network,
                            snap,
                            simulator=self.sim,
                            stall_threshold=self.loaded_repair_threshold,
                        )
                    v15_pol_arg = self.v15_policy
                    if scope_to_use.startswith("V17_") and v15_pol_arg is None:
                        from amr_sched.cvcr.v17_window_protection import load_v17_window_policy
                        v15_pol_arg = load_v17_window_policy()
                    elif scope_to_use.startswith("V16_") and v15_pol_arg is None:
                        from amr_sched.cvcr.v16_frontier import load_v16_frontier_policy
                        v15_pol_arg = load_v16_frontier_policy()
                    policy = LGCVCRPolicy(
                        scope_to_use,
                        self.sim.network,
                        use_rust=self.use_rust,
                        lrho_predictor=(self.lrho_predictor if scope_to_use.startswith("V10_") else None),
                        destroy_selector=(self.destroy_selector if scope_to_use.startswith(("V12_", "V13_", "V14_", "V15_", "V16_", "V17_")) else None),
                        bay_detour_delta=(dd_sig["bay_detour_delta"] if dd_sig is not None else None),
                        detour_weights_q3=(dd_sig["weights_q3"] if dd_sig is not None else None),
                        kitting_ranker=(self.kitting_ranker if scope_to_use.startswith(("V13_", "V14_")) else None),
                        v14_policy=(self.v14_policy if scope_to_use.startswith("V14_") else None),
                        v15_policy=(v15_pol_arg if scope_to_use.startswith(("V15_", "V16_", "V17_")) else None),
                    )
                    policy._prev_ltm_weights = dict(getattr(self.sim, "_traffic_weights_q3", {}))
                    proposal = policy.propose(snap, budget_ms=spec.budget_ms)
                allow_live_repair = (
                    (
                        not str(self.v13_kitting_mode or "").startswith("V15_")
                        and not (
                            str(self.v13_kitting_mode or "").startswith(("V16_", "V17_"))
                            and getattr(self.v15_policy, "active_subscope", None) is None
                        )
                    )
                    or action == 1
                )
                if (
                    self.loaded_repair_mode != "none"
                    and proposal is not None
                    and allow_live_repair
                ):
                    if self.loaded_repair_mode == "haven_v12":
                        from amr_sched.cvcr.v12_proactive_guard import plan_haven_safe_corridor_repair
                        repairs, live_repair_stats = plan_haven_safe_corridor_repair(
                            self.sim,
                            snap,
                            stall_threshold=self.loaded_repair_threshold,
                            enable_choke_oscillation_guard=bool(
                                self._active_p4_choke_robust or str(self.v13_kitting_mode or "").startswith(("V15_", "V16_", "V17_"))
                            ),
                        )
                    else:
                        from amr_sched.cvcr.loaded_corridor_repair import plan_loaded_corridor_repair
                        repairs, live_repair_stats = plan_loaded_corridor_repair(
                            self.sim,
                            snap,
                            mode=self.loaded_repair_mode,
                            stall_threshold=self.loaded_repair_threshold,
                        )
                    if repairs:
                        if is_virtual_clock_active():
                            _cvcr_record_work_ms(0.25 + 0.12 * len(repairs))
                        merged_rep = dict(getattr(proposal, "proposed_live_repairs", None) or {})
                        merged_rep.update(repairs)
                        proposal.proposed_live_repairs = merged_rep
                        proposal.telemetry["loaded_repair_stats"] = dict(live_repair_stats)
                        digest_base = json.dumps(
                            {
                                "base": proposal.candidate_digest,
                                "live_repairs": sorted(
                                    (aid, r.get("kind"), r.get("task_id"), r.get("goal"), len(r.get("hops", [])))
                                    for aid, r in repairs.items()
                                ),
                            },
                            sort_keys=True,
                        ).encode()
                        proposal.candidate_digest = hashlib.sha256(digest_base).hexdigest()
                propose_ms = (_cvcr_now_s() - t_p0) * 1000.0
                if proposal is not None:
                    cp_st = proposal.telemetry.get("cp_stats", {})
                    for k_cp in self.cp_stats_totals:
                        self.cp_stats_totals[k_cp] += int(cp_st.get(k_cp, 0))

                if self.real_compute_window_coupling:
                    window_cost_steps = int(math.floor(propose_ms / max(1.0, self.compute_step_ms)))
                else:
                    window_cost_steps = {20.0: 0, 50.0: 2, 100.0: 4, 150.0: 6}.get(
                        round(float(spec.budget_ms), 1), 6
                    )

                if self.enable_compute_latency:
                    if open_fault_tasks:
                        w_avail = max(int(getattr(t, "adjustable_window_steps", 6)) for t in open_fault_tasks)
                        if window_cost_steps > w_avail:
                            window_expired = True
                            for t in open_fault_tasks:
                                if not getattr(t, "window_missed", False):
                                    t.window_missed = True
                                    t.adjustable_window_steps = 0
                                    t.secondary_window_open_step = step_now + 40
                    elif has_open_kitting_bottleneck and t_kit is not None:
                        w_avail = int(getattr(t_kit, "adjustable_window_steps", 3))
                        if window_cost_steps > w_avail:
                            window_expired = True
                            t_kit.adjustable_window_steps = 0
                            t_kit.kitting_bottleneck_active = False
                            self.active_kitting_task_id = None
                            self.active_kitting_op_id = None

            if window_expired:
                step_compute_ms = propose_ms
                receipt_accepted = False
                self.total_window_expirations += 1
                if hasattr(self.sim, "_res"):
                    seen_exp_nodes: set[str] = set()
                    for agv_obj in self.sim.agvs:
                        if agv_obj.agv_id not in faulted_now and agv_obj.agv_id not in self.sim._transit:
                            nh = agv_obj.cur_node or agv_obj.start_node
                            if nh not in seen_exp_nodes:
                                seen_exp_nodes.add(nh)
                                self.sim._res.reserve_node(nh, Interval(step_now, step_now + 1), "SYNC_LOCK")
                self.last_replan_tick = step_now
            else:
                t_c0 = _cvcr_now_s()
                if self.execution_perturbation is not None and self._active_execution_aware:
                    if getattr(self.sim, "_execution_feedback_step", -1) == step_now:
                        p2_fb_commit = getattr(self.sim, "_execution_feedback", None)
                    else:
                        p2_fb_commit = self.execution_perturbation.capture_feedback(self.sim, step_now)
                else:
                    p2_fb_commit = None
                receipt = self.owner.validate_and_commit(proposal, execution_feedback=p2_fb_commit)
                receipt_reasons = list(getattr(receipt, "reasons", ()))
                receipt_cand_digest = str(getattr(proposal, "candidate_digest", "")) if proposal is not None else ""
                if is_virtual_clock_active():
                    _cvcr_record_work_ms(0.05)
                validate_commit_ms = (_cvcr_now_s() - t_c0) * 1000.0
                step_compute_ms = propose_ms + validate_commit_ms
                receipt_accepted = receipt.accepted
                if receipt.accepted:
                    live_rep_map = getattr(proposal, "proposed_live_repairs", None) or {}
                    live_repair_count = len(live_rep_map)
                    self.total_loaded_repairs += live_repair_count
                    for aid_lr, rep_lr in live_rep_map.items():
                        if str(rep_lr.get("kind", "")) == "picked_reroute" and rep_lr.get("task_id"):
                            ev_key = (aid_lr, str(rep_lr["task_id"]))
                            ev = self.loaded_repair_events.setdefault(
                                ev_key,
                                {
                                    "agv_id": aid_lr,
                                    "task_id": str(rep_lr["task_id"]),
                                    "first_stall_step": int(
                                        getattr(self.sim, "_blocked_since", {}).get(aid_lr, step_now)
                                    ),
                                    "repaired_step": None,
                                    "recovery_move_step": None,
                                    "delivery_step": None,
                                },
                            )
                            if ev.get("repaired_step") is None:
                                ev["repaired_step"] = step_now
                    for tid_p, tgt_agv in proposal.proposed_assignment.items():
                        t_p_obj = self.sim._task_by_id.get(tid_p)
                        if t_p_obj is not None and _task_stat(t_p_obj) in ("pending", "assigned"):
                            hist_entry = self.proposal_task_history.get(tid_p)
                            if hist_entry is None:
                                self.proposal_task_history[tid_p] = {
                                    "first_proposed_step": step_now,
                                    "last_proposed_agv": tgt_agv,
                                    "superseded_count": 0,
                                    "executed_prefix": False,
                                }
                            elif hist_entry["last_proposed_agv"] != tgt_agv:
                                hist_entry["superseded_count"] += 1
                                hist_entry["last_proposed_agv"] = tgt_agv
                    improved_delta = float(proposal.telemetry.get("improved_delta", 0.0))
                    changed_tasks = sum(
                        1 for tid, old_a in prev_assign.items()
                        if self.sim._task_by_id[tid].assigned_agv != old_a
                    )
                    reassigned_existing_tasks = int(changed_tasks)
                    newly_assigned_urgent = sum(
                        1 for tid in prev_urgent_unassigned
                        if self.sim._task_by_id[tid].assigned_agv is not None
                    )
                    changed_ops = sum(
                        1 for op in self.tracker.operations
                        if op.bay_id != prev_bays.get(op.op_id) or op.crew_id != prev_crews.get(op.op_id)
                    )
                    cur_queues = getattr(self.sim, "_pinned_task_queues", {})
                    queue_reorders = sum(
                        1 for aid in (set(prev_queues) | set(cur_queues))
                        if cur_queues.get(aid, []) != prev_queues.get(aid, [])
                    )
                    queue_head_changes = sum(
                        1 for aid in (set(prev_queues) | set(cur_queues))
                        if cur_queues.get(aid, []) != prev_queues.get(aid, [])
                        and cur_queues.get(aid, [])[:1] != prev_queues.get(aid, [])[:1]
                    )
                    queue_tail_changes = max(0, queue_reorders - queue_head_changes)
                    step_churn = changed_tasks + newly_assigned_urgent + changed_ops
                    rescued_tasks = sum(
                        1 for tid in prev_faulted_tasks
                        if self.sim._task_by_id[tid].assigned_agv not in faulted_now
                        and self.sim._task_by_id[tid].assigned_agv is not None
                    )
                    primary_rescued_tasks = sum(
                        1 for tid in prev_faulted_tasks
                        if self.sim._task_by_id[tid].assigned_agv not in faulted_now
                        and self.sim._task_by_id[tid].assigned_agv is not None
                        and not getattr(self.sim._task_by_id[tid], "window_missed", False)
                    )
                    urgent_scheduled = newly_assigned_urgent

                    if self.allow_kitting_shortcut:
                        if (
                            has_open_kitting_bottleneck
                            and t_kit is not None
                            and spec.scope in ("S2_CROSS_DOMAIN_CLOSURE", "S3_VALUE_EXPANDED_CVCR")
                            and spec.budget_ms >= 45.0
                            and self.active_kitting_op_id
                        ):
                            target_op = self.tracker.op_by_id.get(self.active_kitting_op_id)
                            if target_op is not None and not target_op.started:
                                target_op.proc_steps = max(1, target_op.proc_steps - self.kitting_delay_applied)
                                self.kitting_delay_applied = 0
                                t_kit.kitting_bottleneck_active = False
                                t_kit.adjustable_window_steps = 0
                                self.active_kitting_op_id = None
                                self.active_kitting_task_id = None
                                preempted_kitting = 1
                                self.total_kitting_preemptions += 1
                                step_churn = max(1, step_churn)
                    else:
                        # G0 strict causality: kitting bottlenecks resolve only via real bay/crew or urgent dispatch changes.
                        if (
                            has_open_kitting_bottleneck
                            and t_kit is not None
                            and (changed_ops > 0 or newly_assigned_urgent > 0)
                        ):
                            t_kit.kitting_bottleneck_active = False
                            t_kit.adjustable_window_steps = 0
                            self.active_kitting_op_id = None
                            self.active_kitting_task_id = None
                            preempted_kitting = 1
                            self.total_kitting_preemptions += 1

                    # Real physical handover pause (3 ticks) + decision-latency drag for heavy budgets
                    if preempted_kitting == 0:
                        lat_hold = (window_cost_steps // 2) if self.enable_compute_latency else 0
                        for aid, old_tid in prev_jobs_agvs.items():
                            t_obj = self.sim._task_by_id.get(old_tid)
                            if t_obj is not None and t_obj.assigned_agv != aid:
                                agv_obj = next((x for x in self.sim.agvs if x.agv_id == aid), None)
                                if agv_obj is not None and hasattr(self.sim, "_res"):
                                    node_h = agv_obj.cur_node or agv_obj.start_node
                                    self.sim._res.reserve_node(
                                        node_h, Interval(step_now, step_now + 3 + lat_hold), "HANDOVER_LOCK"
                                    )
                    # Shop-floor broadcast sync pause (1 tick) only when a non-empty broadcast change commits without rescue/preemption
                    _full_sync_scopes = (
                        "S2_CROSS_DOMAIN_CLOSURE",
                        "S3_VALUE_EXPANDED_CVCR",
                        "BEST_CVCR_FULL",
                        "OUTCOME_MLP_FULLSTACK",
                        "KEEP_RELEASE_MLP_FULLSTACK",
                        "ALNS_FULLSTACK",
                        "EXP_E2_PROD_FOCUSED",
                        "EXP_E3_JOINT_CORE",
                        "EXP_E4_CONGESTION_PACER",
                        "EXP_E5_AGGRESSIVE_GRID",
                        "V6_E2_CRIT_PROD_CI",
                        "V6_E3_JOINT_CI_HYS",
                        "V6_E4_PACER_CI",
                        "V6_E5_AGGRESSIVE_CI",
                        "V7_E2_CRIT_PROD_CI",
                        "V7_E3_JOINT_CI_HYS",
                        "V7_E4_PACER_CI",
                        "V7_E5_AGGRESSIVE_CI",
                        "V7_E6_BALANCE_KM",
                        "V8_E2_CRIT_PROD_CI",
                        "V8_E3_JOINT_PC_HYS",
                        "V8_E4_LTM_PACER_CI",
                        "V8_E5_AGGRESSIVE_LTM",
                        "V8_E6_BALANCE_KM_PC",
                        "V9_E2_CRIT_PROD_CI",
                        "V9_E3_JOINT_PC_HYS",
                        "V9_E4_LTM_GUIDE_CI",
                        "V9_E5_AGGRESSIVE_GUIDE",
                        "V9_E6_BALANCE_KM_GUIDE",
                        "V9_E7_CPSAT_JOINT_GUIDE",
                        "V10_E2_CRIT_PROD_CI",
                        "V10_E3_JOINT_PC_HYS",
                        "V10_E4_LTM_GUIDE_CI",
                        "V10_E5_AGGRESSIVE_GUIDE",
                        "V10_E6_BALANCE_KM_LRHO",
                        "V10_E7_COND_CPSAT",
                        "V12_CRIT_PROD_DESTROY_S",
                        "V12_PACER_CONSERVATIVE",
                        "V12_BALANCED_DESTROY_M",
                        "V12_BALANCED_DESTROY_L",
                        "V12_AGGRESSIVE_GUIDE_TPG",
                        "V12_SAFE_CPSAT_JOINT",
                        "V13_KITTING_DET",
                        "V13_KITTING_SUP",
                        "V13_KITTING_RL",
                        "V14_JOINT_RL",
                    )
                    if (
                        step_churn > 0
                        and rescued_tasks == 0
                        and preempted_kitting == 0
                        and (spec.scope in _full_sync_scopes or scope_to_use in _full_sync_scopes)
                        and hasattr(self.sim, "_res")
                    ):
                        seen_sync_nodes: set[str] = set()
                        for agv_obj in self.sim.agvs:
                            if agv_obj.agv_id not in faulted_now and agv_obj.agv_id not in self.sim._transit:
                                nh = agv_obj.cur_node or agv_obj.start_node
                                if nh not in seen_sync_nodes:
                                    seen_sync_nodes.add(nh)
                                    self.sim._res.reserve_node(nh, Interval(step_now, step_now + 1), "SYNC_LOCK")
                    self.last_replan_tick = step_now

            tel = proposal.telemetry if proposal is not None else {}
            kd = tel.get("kitting_decision") or {}
            self.decision_logs.append(
                {
                    "step": step_now,
                    "action_name": spec.name,
                    "scope": scope_to_use,
                    "t_requested_ms": round(float(spec.budget_ms), 3),
                    "t_propose_ms": round(propose_ms, 3),
                    "t_validate_commit_ms": round(validate_commit_ms, 3),
                    "t_total_ms": round(step_compute_ms, 3),
                    "eval_count": int(tel.get("eval_count", 0)),
                    "deadline_met": (
                        bool(tel.get("deadline_met", propose_ms <= float(spec.budget_ms) + 2.0))
                        if proposal is not None
                        else False
                    ),
                    "timed_out_gracefully": bool(tel.get("timed_out_gracefully", False)),
                    "window_cost_steps": int(window_cost_steps),
                    "window_expired": bool(window_expired),
                    "receipt_accepted": bool(receipt_accepted),
                    "requested_cand": kd.get("requested_cand_id"),
                    "selected_cand": kd.get("selected_cand_id"),
                    "fallback_reason": kd.get("fallback_reason"),
                    "committed_receipt": {
                        "accepted": bool(receipt_accepted),
                        "reasons": receipt_reasons,
                        "candidate_digest": receipt_cand_digest,
                    },
                    "first_executed_prefix": sorted(
                        tid_p for tid_p, he in self.proposal_task_history.items() if he.get("executed_prefix")
                    ),
                    "kitting_decision": tel.get("kitting_decision"),
                    "step_churn": int(step_churn),
                    "changed_tasks": int(changed_tasks),
                    "reassigned_existing_tasks": int(reassigned_existing_tasks),
                    "changed_ops": int(changed_ops),
                    "newly_assigned_urgent": int(newly_assigned_urgent),
                    "queue_reorders": int(queue_reorders),
                    "queue_head_changes": int(queue_head_changes),
                    "queue_tail_changes": int(queue_tail_changes),
                    "rescued_tasks": int(rescued_tasks),
                    "preempted_kitting": int(preempted_kitting),
                    "live_repair_count": int(live_repair_count),
                    "cp_stats": dict(tel.get("cp_stats", {})),
                    "improved_delta": round(float(improved_delta), 4),
                    "initial_obj": round(float(tel.get("initial_obj", 0.0)), 4),
                    "predicted_obj": round(float(proposal.predicted_obj), 4) if proposal is not None else 0.0,
                    "predicted_makespan": int(proposal.predicted_makespan) if proposal is not None else 0,
                }
            )
            self.total_replans += 1
            self.total_churn += step_churn
            self.total_compute_ms += step_compute_ms

        # 3. Dynamic Exogenous Disruption Injections
        step = int(self.sim._step_cursor)
        f_step = self.disruptions.get("fault_step", 999999)
        if not self.fault_injected and step >= f_step:
            pref_agv = self.disruptions["fault_agv"]
            assigned_candidates = [
                a.agv_id for a in self.sim.agvs
                if a.cur_task and _task_stat(self.sim._task_by_id[a.cur_task]) == "assigned"
            ]
            if assigned_candidates:
                self.fault_injected = True
                self.fault_injected_step = step
                f_agv = pref_agv if pref_agv in assigned_candidates else assigned_candidates[0]
                self.active_pre_pickup_fault_agv = f_agv
                self.sim.set_agv_faulted(f_agv, True)
                w_init = int(self.disruptions.get("fault_window", 6)) if self.enable_commitment_lock else 10
                for a in self.sim.agvs:
                    if a.agv_id == f_agv:
                        a.state = AgvState.BLOCKED
                        a.remaining_job_steps = 9999
                        a.faulted_until_step = step + 999
                        if a.cur_task and a.cur_task in self.sim._task_by_id:
                            ft = self.sim._task_by_id[a.cur_task]
                            ft.adjustable_window_steps = w_init
                            self.active_fault_task_id = ft.task_id

        if not self.urgent_injected and step >= self.disruptions.get("urgent_release_step", 999999):
            self.urgent_injected = True
            self.urgent_injected_step = step
            u_task = Task(
                self.disruptions["urgent_task_id"],
                pickup=self.disruptions["urgent_pickup"],
                dropoff=self.disruptions["urgent_dropoff"],
                release_step=step,
                due_step=step + self.disruptions["urgent_due_step"],
                priority=TaskPriority.URGENT,
            )
            if (
                self.enable_kitting_coupling
                and self.disruptions.get("event2_regime") == "OPEN_WINDOW_KITTING_BOTTLENECK"
                and self.tracker is not None
            ):
                pref_order = ("OP_3", "OP_7", "OP_2", "OP_6", "OP_4", "OP_8", "OP_1", "OP_5")
                target_op = None
                for oid in pref_order:
                    cand_op = self.tracker.op_by_id.get(oid)
                    if cand_op is not None and not cand_op.started:
                        target_op = cand_op
                        break
                if target_op is None:
                    target_op = next((op for op in self.tracker.operations if not op.started), None)
                if target_op is not None:
                    u_task.kit_op_id = target_op.op_id
                    u_task.dropoff = target_op.bay_id
                    delay = int(self.disruptions.get("kitting_bottleneck_delay", 45))
                    target_op.proc_steps += delay
                    self.kitting_delay_applied = delay
                    self.active_kitting_op_id = target_op.op_id
                    self.active_kitting_task_id = u_task.task_id
                    u_task.kitting_bottleneck_active = True
                    u_task.adjustable_window_steps = (
                        int(self.disruptions.get("kitting_bottleneck_window", 3))
                        if self.enable_commitment_lock
                        else 10
                    )
            self.sim.tasks.append(u_task)
            self.sim._task_by_id[u_task.task_id] = u_task

        # Event 3: Temporary post-pickup fault on a healthy loaded AGV (Last Adjustable Window CLOSED)
        pp_step = self.disruptions.get("post_pickup_fault_step", 999999)
        if not self.post_pickup_fault_injected and pp_step <= step <= pp_step + 20:
            loaded_candidates = [
                a for a in self.sim.agvs
                if a.agv_id != self.active_pre_pickup_fault_agv
                and a.agv_id not in getattr(self.sim, "_faulted_agvs", set())
                and a.cur_task
                and _task_stat(self.sim._task_by_id[a.cur_task]) == "picked"
            ]
            if loaded_candidates:
                f_agv2 = loaded_candidates[0]
                dur = int(self.disruptions.get("post_pickup_fault_duration", 18))
                self.post_pickup_fault_injected = True
                self.post_pickup_fault_agv = f_agv2.agv_id
                self.post_pickup_recover_step = step + dur
                f_agv2.faulted_until_step = self.post_pickup_recover_step
                f_agv2.state = AgvState.BLOCKED
                self.sim.set_agv_faulted(f_agv2.agv_id, True)
                if not self.enable_commitment_lock and f_agv2.cur_task in self.sim._task_by_id:
                    from amr_sched.entities import TaskStatus
                    ft2 = self.sim._task_by_id[f_agv2.cur_task]
                    ft2.status = TaskStatus.ASSIGNED
                    ft2.adjustable_window_steps = 8

        if self.post_pickup_fault_agv is not None and step >= self.post_pickup_recover_step:
            f_agv2_id = self.post_pickup_fault_agv
            agv_rec = next((a for a in self.sim.agvs if a.agv_id == f_agv2_id), None)
            can_merge = True
            if agv_rec is not None:
                cur_n = agv_rec.cur_node or agv_rec.start_node
                cap_n = int(self.sim.network.node(cur_n).capacity) if cur_n in self.sim.network.nodes else 1
                faulted_set_now = getattr(self.sim, "_faulted_agvs", set())
                transit_now = getattr(self.sim, "_transit", {})
                occ_now = sum(
                    1 for b in self.sim.agvs
                    if b.agv_id != f_agv2_id
                    and b.agv_id not in faulted_set_now
                    and (
                        (b.agv_id not in transit_now and b.cur_node == cur_n)
                        or (
                            b.agv_id in transit_now
                            and (transit_now[b.agv_id][1] == cur_n or transit_now[b.agv_id][0] == cur_n)
                        )
                    )
                )
                can_merge = occ_now < cap_n
            if can_merge:
                self.post_pickup_fault_agv = None
                if agv_rec is not None:
                    agv_rec.faulted_until_step = 0
                self.sim.set_agv_faulted(f_agv2_id, False)

        # 4. Advance Simulation Physics by 1 Tick
        more_steps = self.sim.step_once()
        self.current_step = int(self.sim._step_cursor)
        self.sim._env_last_replan_tick = self.last_replan_tick
        self.sim._env_recent_conflicts = self.recent_conflict_ticks
        if self.execution_perturbation is not None and self._active_execution_aware:
            fb_step = self.execution_perturbation.capture_feedback(self.sim, self.current_step)
            self.sim._execution_feedback = fb_step
            self.sim._execution_feedback_step = self.current_step
            if fb_step is not None and not fb_step.is_empty():
                self.sim._causal_drift_observed = True
        else:
            self.sim._execution_feedback = None
            self.sim._execution_feedback_step = self.current_step
        if self._active_p4_choke_robust or str(self.v13_kitting_mode or "").startswith(("V15_", "V16_", "V17_")):
            from amr_sched.cvcr.p4_belief_choke import record_node_history_tick
            record_node_history_tick(self.sim, self.current_step)
        self.tracker.set_op_priorities(getattr(self.sim, "_op_priorities", None))
        self.tracker.step(self.current_step)

        # Update task-proposal execution compliance & loaded-stall recovery lifecycle clocks
        for tid_p, hist_entry in self.proposal_task_history.items():
            if not hist_entry.get("executed_prefix"):
                t_p_obj = self.sim._task_by_id.get(tid_p)
                if t_p_obj is not None and _task_stat(t_p_obj) in ("picked", "completed"):
                    if t_p_obj.assigned_agv == hist_entry.get("last_proposed_agv"):
                        hist_entry["executed_prefix"] = True

        blocked_map_now = getattr(self.sim, "_blocked_since", {})
        transit_map_now = getattr(self.sim, "_transit", {})
        jobs_map_now = getattr(self.sim, "_jobs", {})
        faulted_map_now = getattr(self.sim, "_faulted_agvs", set())
        for aid_b, bs_step in blocked_map_now.items():
            if aid_b in faulted_map_now or aid_b in transit_map_now:
                continue
            j_b = jobs_map_now.get(aid_b)
            if j_b is not None and j_b.task_id:
                t_b = self.sim._task_by_id.get(j_b.task_id)
                if t_b is not None and _task_stat(t_b) == "picked" and (self.current_step - int(bs_step)) >= self.loaded_repair_threshold:
                    ev_key = (aid_b, j_b.task_id)
                    self.loaded_repair_events.setdefault(
                        ev_key,
                        {
                            "agv_id": aid_b,
                            "task_id": j_b.task_id,
                            "first_stall_step": int(bs_step),
                            "repaired_step": None,
                            "recovery_move_step": None,
                            "delivery_step": None,
                        },
                    )
        for (ev_aid, ev_tid), ev_rec in self.loaded_repair_events.items():
            if ev_rec.get("repaired_step") is not None and ev_rec.get("recovery_move_step") is None:
                if ev_aid in transit_map_now:
                    ev_rec["recovery_move_step"] = self.current_step
            if ev_rec.get("delivery_step") is None:
                t_ev = self.sim._task_by_id.get(ev_tid)
                if t_ev is not None and _task_stat(t_ev) == "completed":
                    ev_rec["delivery_step"] = int(t_ev.complete_step or self.current_step)

        # Update dynamic adjustable window expiration clocks
        faulted_set_after = getattr(self.sim, "_faulted_agvs", set())
        if self.active_fault_task_id and self.active_fault_task_id in self.sim._task_by_id:
            ft_now = self.sim._task_by_id[self.active_fault_task_id]
            if _task_stat(ft_now) == "assigned" and ft_now.assigned_agv in faulted_set_after:
                if (
                    getattr(ft_now, "window_missed", False)
                    and self.current_step >= getattr(ft_now, "secondary_window_open_step", 999999)
                ):
                    ft_now.adjustable_window_steps = 1
                elif (
                    not getattr(ft_now, "window_missed", False)
                    and self.fault_injected_step >= 0
                    and self.current_step > self.fault_injected_step + 3
                ):
                    ft_now.adjustable_window_steps = max(0, int(getattr(ft_now, "adjustable_window_steps", 0)) - 1)
                    if ft_now.adjustable_window_steps == 0:
                        ft_now.window_missed = True
                        ft_now.secondary_window_open_step = self.current_step + 40

        if self.active_kitting_task_id and self.active_kitting_task_id in self.sim._task_by_id:
            tk_now = self.sim._task_by_id[self.active_kitting_task_id]
            if (
                getattr(tk_now, "kitting_bottleneck_active", False)
                and self.urgent_injected_step >= 0
                and self.current_step > self.urgent_injected_step + 3
            ):
                tk_now.adjustable_window_steps = max(0, int(getattr(tk_now, "adjustable_window_steps", 0)) - 1)
                if tk_now.adjustable_window_steps == 0:
                    tk_now.kitting_bottleneck_active = False
                    self.active_kitting_task_id = None
                    self.active_kitting_op_id = None

        # 5. Measure Physical Delays and Conflicts
        current_tardiness_incurred = sum(
            1 for t in self.sim.tasks
            if _task_stat(t) not in ("completed", "failed") and t.due_step is not None and self.current_step > t.due_step
        )
        faulted_set = faulted_set_after
        blocked_since = getattr(self.sim, "_blocked_since", {})
        current_blocked_agvs = sum(
            1 for a in self.sim.agvs
            if a.agv_id not in faulted_set
            and not getattr(a, "faulted", False)
            and (a.agv_id in blocked_since or _agv_stat(a) in ("blocked", "parking"))
        )

        self.conflict_window.pop(0)
        self.conflict_window.append(current_blocked_agvs)
        self.recent_conflict_ticks = sum(self.conflict_window)

        # 6. Physical Step Reward (label-free when pure_physical_reward=True)
        r_tardiness = float(current_tardiness_incurred) * 0.1
        r_churn = self.lambda_churn * float(step_churn)
        r_budget = self.lambda_budget * (step_compute_ms / 150.0)
        r_wait = self.lambda_wait * float(current_blocked_agvs) * 0.1

        if self.pure_physical_reward:
            rescue_bonus = 0.0
            wasted_replan_penalty = (
                12.0
                if (spec.is_replan and step_churn == 0 and rescued_tasks == 0 and urgent_scheduled == 0)
                else 0.0
            ) + (8.0 if window_expired else 0.0)
            inaction_fault_penalty = 0.0
        else:
            effective_rescues = rescued_tasks + preempted_kitting
            optimal_fault_scope_budget = (
                spec.scope == "S0_DIRECT_AFFECTED" and spec.budget_ms <= 25.0
            )
            optimal_kit_scope_budget = (
                spec.scope == "S2_CROSS_DOMAIN_CLOSURE" and 45.0 <= spec.budget_ms <= 55.0
            )
            fault_credit = (
                60.0 * float(primary_rescued_tasks)
                if (primary_rescued_tasks > 0 and optimal_fault_scope_budget)
                else (
                    -25.0
                    if (rescued_tasks > 0 and not optimal_fault_scope_budget)
                    else 0.0
                )
            )
            kit_credit = (
                60.0 * float(preempted_kitting)
                if (preempted_kitting > 0 and optimal_kit_scope_budget)
                else (
                    -25.0
                    if (preempted_kitting > 0 and not optimal_kit_scope_budget)
                    else 0.0
                )
            )
            rescue_bonus = fault_credit + kit_credit
            wasted_replan_penalty = (
                12.0 if (spec.is_replan and effective_rescues == 0) else 0.0
            ) + (8.0 if window_expired else 0.0)
            inaction_fault_penalty = (
                4.0 if (not spec.is_replan and (has_unrescued_fault_task or has_open_kitting_bottleneck)) else 0.0
            )

        step_reward = rescue_bonus - (
            r_tardiness + r_churn + r_budget + r_wait + wasted_replan_penalty + inaction_fault_penalty
        ) + invalid_action_penalty

        # 7. Termination & Truncation Check
        all_transport_done = (not more_steps) or all(
            _task_stat(t) in ("completed", "failed") for t in self.sim.tasks
        )
        all_production_done = self.tracker.all_done(self.current_step) if self.tracker else True

        # If transport is completed but production operations are still in progress,
        # advance simulation clock until production finishes or max_steps reached
        if all_transport_done and not all_production_done and self.current_step < self.max_steps:
            self.sim._step_cursor += 1
            self.current_step = int(self.sim._step_cursor)
            self.tracker.set_op_priorities(getattr(self.sim, "_op_priorities", None))
            self.tracker.step(self.current_step)
            all_production_done = self.tracker.all_done(self.current_step)

        terminated = all_transport_done and all_production_done
        truncated = (not terminated) and (self.current_step >= self.max_steps)

        if terminated or truncated:
            res = self.sim._build_result()
            completed_tasks = len(res.completed_tasks())
            uncompleted_tasks = len(self.sim.tasks) - completed_tasks

            completed_ops = sum(
                1 for op in self.tracker.operations
                if op.started and op.end_step is not None and self.current_step >= op.end_step
            )
            uncompleted_ops = len(self.tracker.operations) - completed_ops

            # Terminal penalty for missed orders/production
            terminal_penalty = -(uncompleted_tasks * 50.0 + uncompleted_ops * 50.0)
            step_reward += terminal_penalty

        self.total_reward += step_reward

        # 8. Compute Next Observation and Mask
        next_obs = compute_macro_state(
            self.sim,
            self.tracker,
            last_replan_tick=self.last_replan_tick,
            recent_conflict_ticks=self.recent_conflict_ticks,
        )
        next_mask = self._current_mask(next_obs)

        info = {
            "action_mask": next_mask,
            "action_name": spec.name,
            "receipt_accepted": receipt_accepted,
            "step": self.current_step,
            "churn": step_churn,
            "rescued_tasks": rescued_tasks,
            "urgent_scheduled": urgent_scheduled,
            "preempted_kitting": preempted_kitting,
            "window_expired": window_expired,
            "improved_delta": improved_delta,
            "propose_ms": propose_ms,
            "validate_commit_ms": validate_commit_ms,
            "compute_ms": step_compute_ms,
            "total_churn": self.total_churn,
            "total_replans": self.total_replans,
            "total_compute_ms": self.total_compute_ms,
            "total_reward": self.total_reward,
        }

        return next_obs, step_reward, terminated, truncated, info
