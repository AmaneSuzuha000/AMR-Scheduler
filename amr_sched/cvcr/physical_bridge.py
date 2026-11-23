"""P3 Physical ROS 2 / Nav2 / Gazebo <-> Execution-Aware CVCR Coordination Bridge.

Connects L3 production-and-fleet CVCR (`LGCVCRPolicy`, `NativeTransactionOwner`) cleanly
to L1/L2 physical robot adapters (`ExternalExecution`, `ExternalHandling`, `FleetRuntime`,
Nav2 `FollowPath`, AMCL / local-costmap obstacle streams) while strictly preserving
physical occupancy pins, mid-edge non-preemption, and loaded cargo custody.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import time
from typing import Any, Optional

from amr_sched.cvcr.execution_feedback import (
    CorridorFeedback,
    ExecutionFeedbackPacket,
)
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy
from amr_sched.cvcr.production import NativeProductionTracker, OperationSnapshot
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.transaction import NativeTransactionOwner, TransactionReceipt
from amr_sched.entities import TaskStatus
from amr_sched.monitor import DeviationEvent


@dataclass(frozen=True)
class Nav2RobotTelemetry:
    """Structured L1/L2 Nav2, localization, and safety-guard telemetry for one AMR."""

    robot_id: str
    pose_xy_yaw: Optional[tuple[float, float, float]] = None
    localization_source: str = "amcl"
    speed_scale: float = 1.0
    eta_drift_steps: int = 0
    local_planner_state: str = "nominal"
    receipt_age_s: float = 0.0
    handling_delay_steps: int = 0
    fault_reason: Optional[str] = None

    def to_telemetry_dict(self) -> dict[str, Any]:
        """Convert to `ExecutionFeedbackPacket.from_simulator_and_events` telemetry format."""
        out: dict[str, Any] = {
            "localization_source": self.localization_source,
            "speed_scale": float(self.speed_scale),
            "eta_drift_steps": int(self.eta_drift_steps),
            "local_planner_state": self.local_planner_state,
            "receipt_age_s": float(self.receipt_age_s),
            "handling_delay_steps": int(self.handling_delay_steps),
        }
        if self.pose_xy_yaw is not None:
            out["pose_xy_yaw"] = self.pose_xy_yaw
        if self.fault_reason:
            out["fault_reason"] = str(self.fault_reason)
        return out


@dataclass(frozen=True)
class CorridorObstacleReport:
    """Structured Nav2 costmap / LiDAR dynamic obstacle observation on one corridor."""

    edge: tuple[str, str]
    status: str = "blocked"
    estimated_clear_horizon_steps: int = 15
    source: str = "nav2_costmap"
    reason: str = "dynamic_obstacle"


@dataclass
class PhysicalCoordinationStepReport:
    """Auditable per-turn summary of physical CVCR coordination and reconciliations."""

    tick: int
    triggered: bool
    trigger_reason: str
    receipt_accepted: bool = False
    receipt_reasons: tuple[str, ...] = ()
    closed_corridors: tuple[tuple[str, str], ...] = ()
    reopened_corridors: tuple[tuple[str, str], ...] = ()
    replanned_detours: tuple[str, ...] = ()
    reassigned_tasks: tuple[str, ...] = ()
    feedback_signature: Optional[str] = None
    propose_ms: float = 0.0
    e2e_ms: float = 0.0
    eval_objective: Optional[float] = None
    actions_log: list[dict[str, Any]] = field(default_factory=list)


class PhysicalCVCRCoordinator:
    """Execution-Aware L3 CVCR coordinator over physical ROS 2 / Nav2 `ExternalExecution` fleets."""

    def __init__(
        self,
        simulator: Any,
        operations: Optional[list[OperationSnapshot]] = None,
        *,
        production_tracker: Optional[NativeProductionTracker] = None,
        policy_method: str = "V13_KITTING_DET",
        budget_ms: float = 35.0,
        epoch_interval: int = 12,
        use_rust: bool = False,
        default_obstacle_horizon_steps: int = 15,
        kitting_ranker: Any = None,
        destroy_selector: Any = None,
        recovery_backends: Optional[dict[str, Any]] = None,
        enable_belief_estimation: bool = True,
        external_proposer: Optional[Any] = None,
        seed: int = 0,
    ) -> None:
        if not getattr(simulator, "executions", None):
            raise ValueError("PhysicalCVCRCoordinator requires Simulator with physical executions")
        self.simulator = simulator
        self.operations: list[OperationSnapshot] = list(operations or [])
        self.tracker = (
            production_tracker
            if production_tracker is not None
            else (NativeProductionTracker(self.operations, simulator.tasks) if self.operations else None)
        )
        self.owner = NativeTransactionOwner(
            simulator,
            self.operations,
            production_tracker=self.tracker,
            allow_physical_mode=True,
        )
        from amr_sched.baselines.policy_registry import (
            build_external_proposer,
            is_external_policy,
            is_registered_policy,
            load_internal_cvcr_policy,
            resolve_policy_name,
        )

        raw_method = str(policy_method)
        self.requested_policy_method = raw_method
        self.policy_method = (
            resolve_policy_name(raw_method) if is_registered_policy(raw_method) else raw_method
        )
        self.budget_ms = float(budget_ms)
        self.epoch_interval = max(1, int(epoch_interval))
        self.use_rust = bool(use_rust)
        self.seed = int(seed)
        self.default_obstacle_horizon_steps = max(2, int(default_obstacle_horizon_steps))
        self.kitting_ranker = kitting_ranker
        self.destroy_selector = destroy_selector
        self._recovery_backends: dict[str, Any] = dict(recovery_backends or {})
        self._external_proposer: Optional[Any] = external_proposer
        if self._external_proposer is None and is_registered_policy(self.policy_method):
            if is_external_policy(self.policy_method):
                self._external_proposer = build_external_proposer(
                    self.policy_method,
                    seed=self.seed,
                    use_rust=self.use_rust,
                )
        self._cached_v_policy: Optional[Any] = None
        if self._external_proposer is None and is_registered_policy(self.policy_method):
            self._cached_v_policy = load_internal_cvcr_policy(self.policy_method)

        self.last_replan_tick: int = -999
        self._last_fb_sig: Optional[str] = None
        self._last_struct_sig: Optional[tuple] = None
        self._managed_closures: set[tuple[str, str]] = set()
        self._active_blocked_corridors: list[CorridorObstacleReport | tuple[str, str]] = []
        self.history: list[PhysicalCoordinationStepReport] = []
        self._belief_model: Optional[Any] = None
        if enable_belief_estimation:
            from amr_sched.cvcr.p4_belief_choke import CorridorDurationBeliefModel
            self._belief_model = CorridorDurationBeliefModel(
                prior_scale=max(6.0, float(self.default_obstacle_horizon_steps)),
                enable_voi_probe=self.policy_method.startswith("V16_"),
            )

    def set_recovery_backends(self, backends: Optional[dict[str, Any]]) -> None:
        """Bind native ROS 2 / Nav2 recovery adapters (`resume_execution`/`replan_execution`) per robot."""
        self._recovery_backends = dict(backends or {})

    def set_blocked_corridors(
        self,
        blocked_corridors: Optional[list[CorridorObstacleReport | tuple[str, str]]],
    ) -> None:
        """Update active Nav2/costmap corridor obstacles held by the coordinator."""
        self._active_blocked_corridors = list(blocked_corridors or ())

    @staticmethod
    def _undir_edge(edge: tuple[str, str]) -> tuple[str, str]:
        return (str(edge[0]), str(edge[1])) if str(edge[0]) <= str(edge[1]) else (str(edge[1]), str(edge[0]))

    def _physically_locked_robots(self) -> set[str]:
        sim = self.simulator
        tick = int(sim._step_cursor)
        locked: set[str] = set()
        handlings = getattr(sim, "handlings", None) or {}
        for aid, gate in sim.executions.items():
            h = handlings.get(aid)
            h_act = bool(h is not None and (getattr(h, "command", None) is not None or getattr(h, "in_progress", False)))
            if (
                aid in getattr(sim, "_transit", {})
                or getattr(gate, "command", None) is not None
                or aid in getattr(sim, "_physical_detours", {})
                or aid in getattr(sim, "_physical_preemptions", {})
                or int(getattr(sim, "_load_end", {}).get(aid, 0)) > tick
                or int(getattr(sim, "_drop_end", {}).get(aid, 0)) > tick
                or h_act
            ):
                locked.add(aid)
        return locked

    def sync_corridor_obstacles(
        self,
        now_wall: float,
        blocked_corridors: Optional[list[CorridorObstacleReport | tuple[str, str]]] = None,
    ) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        """Synchronize live Nav2/costmap obstacles cleanly with `close/reopen_physical_corridor`."""
        sim = self.simulator
        eff_blocks = self._active_blocked_corridors if blocked_corridors is None else blocked_corridors
        desired: set[tuple[str, str]] = set()
        for item in eff_blocks:
            if isinstance(item, CorridorObstacleReport):
                if item.status == "blocked":
                    desired.add(self._undir_edge(item.edge))
            else:
                desired.add(self._undir_edge(item))

        closed_now: list[tuple[str, str]] = []
        reopened_now: list[tuple[str, str]] = []

        for edge in sorted(desired):
            if edge not in sim._closed_corridors:
                ans = sim.close_physical_corridor(edge, float(now_wall))
                if ans.get("accepted"):
                    self._managed_closures.add(edge)
                    closed_now.append(edge)

        for edge in sorted(list(self._managed_closures)):
            if edge not in desired and edge in sim._closed_corridors:
                rev = sim._corridor_revisions.get(edge)
                if rev is not None:
                    ans = sim.reopen_physical_corridor(edge, int(rev))
                    if ans.get("accepted"):
                        self._managed_closures.discard(edge)
                        reopened_now.append(edge)
            elif edge not in sim._closed_corridors:
                self._managed_closures.discard(edge)

        return closed_now, reopened_now

    def reconcile_detours(self, now_wall: float) -> list[str]:
        """Advance safe mid-edge detours to their endpoint node, then run node-arrival replanning."""
        sim = self.simulator
        now_f = float(now_wall)
        replanned: list[str] = []
        for aid in sorted(list(getattr(sim, "_physical_detours", {}).keys())):
            gate = sim.execution_for(aid)
            if gate is None or not gate.fault or not gate.stopped or gate._outbox:
                continue
            backend = self._recovery_backends.get(aid)
            token = gate.recovery_token()
            if token is None:
                continue
            tr = sim._transit.get(aid)
            if tr is not None and gate.command is not None:
                # Safe mid-edge hop whose active corridor is still open: resume to node before replanning
                if tuple(sorted(tr[:2])) not in getattr(sim, "_closed_corridors", set()):
                    if backend is not None and hasattr(backend, "resume_execution"):
                        backend.resume_execution(token)
                    else:
                        sim.resume_execution(token, now_f, robot_id=aid)
                continue
            if tr is None and gate.command is None:
                if backend is not None and hasattr(backend, "replan_execution"):
                    ans = backend.replan_execution(token)
                else:
                    ans = sim.replan_execution(token, now_f, robot_id=aid)
                if isinstance(ans, dict) and ans.get("accepted"):
                    replanned.append(aid)
        return replanned

    def build_execution_feedback(
        self,
        now_wall: float,
        *,
        nav2_telemetry: Optional[dict[str, Nav2RobotTelemetry | dict[str, Any]]] = None,
        blocked_corridors: Optional[list[CorridorObstacleReport | tuple[str, str]]] = None,
        deviations: Optional[list[DeviationEvent]] = None,
    ) -> Optional[ExecutionFeedbackPacket]:
        """Aggregate Nav2/AMCL/guard/monitor signals into a strictly causal `ExecutionFeedbackPacket`."""
        sim = self.simulator
        tick = int(sim._step_cursor)
        telem_dict: dict[str, dict[str, Any]] = {}
        for rid, item in (nav2_telemetry or {}).items():
            telem_dict[str(rid)] = (
                item.to_telemetry_dict() if isinstance(item, Nav2RobotTelemetry) else dict(item)
            )
        pkt = ExecutionFeedbackPacket.from_simulator_and_events(
            sim,
            deviations=deviations,
            robot_telemetry=telem_dict,
            now_wall=now_wall,
        )
        robot_states = dict(pkt.robot_states) if pkt is not None else {}
        corridor_updates = dict(pkt.corridor_updates) if pkt is not None else {}
        events = list(pkt.source_events) if pkt is not None else []

        # Merge physically closed corridors and reported Nav2 costmap obstacles causally
        blk_specs: list[CorridorObstacleReport] = []
        eff_blocks = self._active_blocked_corridors if blocked_corridors is None else blocked_corridors
        for item in eff_blocks:
            if isinstance(item, CorridorObstacleReport):
                if item.status == "blocked":
                    blk_specs.append(item)
            else:
                blk_specs.append(
                    CorridorObstacleReport(
                        edge=(str(item[0]), str(item[1])),
                        estimated_clear_horizon_steps=self.default_obstacle_horizon_steps,
                    )
                )
        for edge in sorted(sim._closed_corridors):
            if not any(self._undir_edge(b.edge) == edge for b in blk_specs):
                blk_specs.append(
                    CorridorObstacleReport(
                        edge=edge,
                        estimated_clear_horizon_steps=self.default_obstacle_horizon_steps,
                        source="physical_closure",
                        reason="closed_corridor",
                    )
                )

        if self._belief_model is not None:
            blk_undir_now = {self._undir_edge(spec.edge) for spec in blk_specs}
            self._belief_model.observe_step(tick, blk_undir_now)

        for spec in blk_specs:
            u, v = str(spec.edge[0]), str(spec.edge[1])
            hz = max(2, int(spec.estimated_clear_horizon_steps))
            for a, b in ((u, v), (v, u)):
                if sim.network.has_edge(a, b):
                    if self._belief_model is not None:
                        corridor_updates[(a, b)] = self._belief_model.build_corridor_feedback(
                            sim.network, a, b, tick, source=spec.source, reason=spec.reason
                        )
                    else:
                        corridor_updates[(a, b)] = CorridorFeedback(
                            src=a,
                            dst=b,
                            status="blocked",
                            extra_delay_steps=hz,
                            blocked_until_step=tick + hz,
                            source=spec.source,
                            reason=spec.reason,
                        )
            events.append(f"CORRIDOR_BLOCKED:{u}<->{v}@{tick}")

        if not robot_states and not corridor_updates:
            return None
        return ExecutionFeedbackPacket(
            tick=tick,
            robot_states=robot_states,
            corridor_updates=corridor_updates,
            source_events=tuple(sorted(set(str(e) for e in events))),
        )

    def _sanitize_proposal_for_physical(
        self,
        proposal: CandidateProposal,
        locked_robots: set[str],
    ) -> CandidateProposal:
        """Preserve in-flight/edge-pinned robot commitments while keeping safe traffic/bay/idle updates."""
        sim = self.simulator
        tmap = {t.task_id: t for t in sim.tasks}
        pins_chk = sim._res.pins() if hasattr(sim, "_res") else {"edges": {}}
        edge_pin_owners = {o for owners in pins_chk.get("edges", {}).values() for o in owners}
        immobile_owners = locked_robots | edge_pin_owners

        clean_assign: dict[str, str] = {}
        for tid, new_aid in proposal.proposed_assignment.items():
            t = tmap.get(tid)
            if t is None:
                continue
            if t.status in (TaskStatus.PICKED, TaskStatus.COMPLETED) and t.assigned_agv is not None:
                clean_assign[tid] = t.assigned_agv
            elif t.assigned_agv is not None and t.assigned_agv in immobile_owners and t.assigned_agv != new_aid:
                clean_assign[tid] = t.assigned_agv
            else:
                clean_assign[tid] = new_aid

        clean_bays = dict(proposal.proposed_bays)
        for op in self.operations:
            if op.op_id in clean_bays and clean_bays[op.op_id] != op.bay_id:
                for ktid in op.kit_task_ids:
                    kt = tmap.get(ktid)
                    if kt is not None and (
                        kt.status in (TaskStatus.PICKED, TaskStatus.COMPLETED)
                        or kt.assigned_agv in locked_robots
                    ):
                        clean_bays[op.op_id] = op.bay_id
                        break

        return CandidateProposal(
            schema=proposal.schema,
            policy_id=proposal.policy_id,
            snapshot_digest=proposal.snapshot_digest,
            affected_robots=proposal.affected_robots,
            affected_tasks=proposal.affected_tasks,
            affected_operations=proposal.affected_operations,
            proposed_assignment=clean_assign,
            proposed_routes=proposal.proposed_routes,
            proposed_bays=clean_bays,
            proposed_crews=proposal.proposed_crews,
            predicted_obj=proposal.predicted_obj,
            predicted_makespan=proposal.predicted_makespan,
            candidate_digest=proposal.candidate_digest,
            telemetry=proposal.telemetry,
            proposed_admitted_agvs=proposal.proposed_admitted_agvs,
            proposed_task_sequences=proposal.proposed_task_sequences,
            proposed_traffic_weights=proposal.proposed_traffic_weights,
            proposed_traffic_cutoff=proposal.proposed_traffic_cutoff,
            proposed_guide_routes=proposal.proposed_guide_routes,
            proposed_op_priorities=proposal.proposed_op_priorities,
            proposed_live_repairs=None,
        )

    def _should_trigger(
        self,
        tick: int,
        fb: Optional[ExecutionFeedbackPacket],
        closed_now: list[tuple[str, str]],
        reopened_now: list[tuple[str, str]],
        force_replan: bool,
    ) -> tuple[bool, str]:
        sim = self.simulator
        active_tasks = [
            t for t in sim.tasks
            if t.release_step <= tick and t.status not in (TaskStatus.COMPLETED, TaskStatus.FAILED)
        ]
        if not active_tasks:
            return False, "no_active_tasks"
        if force_replan:
            return True, "forced"
        if closed_now or reopened_now:
            return True, "corridor_topology_change"
        struct_sig = (
            (
                tuple(sorted(fb.corridor_updates.keys())),
                tuple(sorted((rid, st.local_planner_state, bool(st.is_degraded)) for rid, st in fb.robot_states.items())),
            )
            if fb is not None
            else None
        )
        if struct_sig is not None and struct_sig != self._last_struct_sig:
            # Immediate wakeup when feedback shows new corridor block or newly degraded robot
            degraded_unpicked = any(
                t.status == TaskStatus.ASSIGNED
                and t.assigned_agv is not None
                and fb is not None
                and t.assigned_agv in fb.robot_states
                and fb.robot_states[t.assigned_agv].is_degraded
                for t in active_tasks
            )
            if degraded_unpicked or (fb is not None and fb.corridor_updates):
                return True, "execution_feedback_transition"
        if tick == 0 or self.last_replan_tick < 0:
            return True, "initial_dispatch"
        if (tick - self.last_replan_tick) >= self.epoch_interval:
            pending_unassigned = any(t.status == TaskStatus.PENDING for t in active_tasks)
            detour_waiting = bool(getattr(sim, "_physical_detours", {}))
            if pending_unassigned or detour_waiting or fb is not None:
                return True, "periodic_epoch"
        return False, "cooldown"

    def step_coordination(
        self,
        now_wall: float,
        *,
        nav2_telemetry: Optional[dict[str, Nav2RobotTelemetry | dict[str, Any]]] = None,
        blocked_corridors: Optional[list[CorridorObstacleReport | tuple[str, str]]] = None,
        deviations: Optional[list[DeviationEvent]] = None,
        force_replan: bool = False,
    ) -> PhysicalCoordinationStepReport:
        """Run one closed-loop L3 CVCR <-> L1/L2 Nav2 coordination step prior to `sim.step_once()`."""
        t_step_start = time.perf_counter()
        sim = self.simulator
        tick = int(sim._step_cursor)
        if self.tracker is not None:
            self.tracker.set_op_priorities(getattr(sim, "_op_priorities", None))
            self.tracker.step(tick)

        closed_now, reopened_now = self.sync_corridor_obstacles(now_wall, blocked_corridors)
        fb = self.build_execution_feedback(
            now_wall,
            nav2_telemetry=nav2_telemetry,
            blocked_corridors=blocked_corridors,
            deviations=deviations,
        )
        fb_sig = fb.signature() if fb is not None else None
        struct_sig = (
            (
                tuple(sorted(fb.corridor_updates.keys())),
                tuple(sorted((rid, st.local_planner_state, bool(st.is_degraded)) for rid, st in fb.robot_states.items())),
            )
            if fb is not None
            else None
        )
        trig, trig_reason = self._should_trigger(tick, fb, closed_now, reopened_now, force_replan)

        rcpt_ok = False
        rcpt_reasons: list[str] = []
        reassigned: list[str] = []
        propose_ms = 0.0
        e2e_ms = 0.0
        eval_obj: Optional[float] = None

        if trig and not getattr(sim, "_operator_paused", False):
            # Capture snapshot strictly causally with current execution feedback
            snap = NativeSnapshotCapture.capture(
                sim,
                self.operations,
                hide_future_releases=True,
                execution_feedback=fb,
            )
            prev_assign = {t.task_id: t.assigned_agv for t in sim.tasks if t.assigned_agv is not None}
            t0 = time.perf_counter()
            if self._external_proposer is not None:
                raw_prop = self._external_proposer(snap, sim.network, budget_ms=self.budget_ms)
            else:
                v15_pol_phys = self._cached_v_policy
                dd_bay = None
                dd_w = None
                if self.policy_method.startswith(("V15_", "V16_", "V17_", "V18_")):
                    from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
                    dd_sig = compute_detour_delta_signals(sim.network, snap, simulator=sim, stall_threshold=20)
                    dd_bay = dd_sig["bay_detour_delta"]
                    dd_w = dd_sig["weights_q3"]
                    if v15_pol_phys is None:
                        if self.policy_method.startswith("V18_"):
                            from amr_sched.cvcr.v18_frontier_synthesis import load_v18_synthesis_policy
                            v15_pol_phys = load_v18_synthesis_policy()
                        elif self.policy_method.startswith("V17_"):
                            from amr_sched.cvcr.v17_window_protection import load_v17_window_policy
                            v15_pol_phys = load_v17_window_policy()
                        elif self.policy_method.startswith("V16_"):
                            from amr_sched.cvcr.v16_frontier import load_v16_frontier_policy
                            v15_pol_phys = load_v16_frontier_policy()
                policy = LGCVCRPolicy(
                    self.policy_method,
                    sim.network,
                    use_rust=self.use_rust,
                    destroy_selector=self.destroy_selector,
                    bay_detour_delta=dd_bay,
                    detour_weights_q3=dd_w,
                    kitting_ranker=self.kitting_ranker,
                    v15_policy=v15_pol_phys,
                )
                raw_prop = policy.propose(snap, budget_ms=self.budget_ms)
            propose_ms = (time.perf_counter() - t0) * 1000.0
            eval_obj = float(raw_prop.predicted_obj)

            locked = self._physically_locked_robots()
            clean_prop = self._sanitize_proposal_for_physical(raw_prop, locked)
            rcpt: TransactionReceipt = self.owner.validate_and_commit(clean_prop, execution_feedback=fb)
            e2e_ms = (time.perf_counter() - t_step_start) * 1000.0
            rcpt_ok = bool(rcpt.accepted)
            rcpt_reasons = list(rcpt.reasons)
            if rcpt.accepted:
                self.last_replan_tick = tick
                self._last_fb_sig = fb_sig
                self._last_struct_sig = struct_sig
                for t in sim.tasks:
                    if t.task_id in prev_assign and t.assigned_agv != prev_assign[t.task_id]:
                        reassigned.append(t.task_id)
                if self.tracker is not None:
                    self.tracker.set_op_priorities(getattr(sim, "_op_priorities", None))

        replanned_detours = self.reconcile_detours(now_wall)
        if fb_sig is not None and not trig:
            self._last_fb_sig = fb_sig
        if struct_sig is not None and not trig:
            self._last_struct_sig = struct_sig
        elif struct_sig is None:
            self._last_struct_sig = None

        rep = PhysicalCoordinationStepReport(
            tick=tick,
            triggered=trig,
            trigger_reason=trig_reason,
            receipt_accepted=rcpt_ok,
            receipt_reasons=tuple(rcpt_reasons),
            closed_corridors=tuple(closed_now),
            reopened_corridors=tuple(reopened_now),
            replanned_detours=tuple(replanned_detours),
            reassigned_tasks=tuple(sorted(reassigned)),
            feedback_signature=fb_sig,
            propose_ms=round(propose_ms, 3),
            e2e_ms=round(e2e_ms, 3),
            eval_objective=(round(eval_obj, 3) if eval_obj is not None else None),
        )
        self.history.append(rep)
        return rep
