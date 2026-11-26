"""PDG 2.0 Detection Module (D1 + D2): Local Cross-Domain Liveness & Progressive Starvation Monitor.

Implements two independently toggleable detection mechanisms on top of optional legacy global PDG:
  - D1: Independent per-operation critical-chain progress tracking (immune to resets from unrelated tasks,
        while discounting legitimate active processing, active upstream machining, and unreleased waves).
  - D2: Kitting starvation lag S_o(t), cumulative resource-window miss debt W_o(t), due-date slippage D_o(t),
        and execution oscillation / idle-blocker detection O_exec(t).

Composite per-op risk:
    R_o(t) = alpha * A_o(t) + beta * W_o(t) + gamma * D_o(t) + eta * S_o(t)

Note: Framed strictly as a risk detection and escalation trigger — NOT a formal universal liveness guarantee.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from amr_sched.cvcr.snapshot import align_start_to_windows, op_predecessors
from experiments.v20_audit.task_c_liveness_forensics import compute_pdg_state


def _stat_lower(obj: Any) -> str:
    s = getattr(obj, "status", getattr(obj, "state", ""))
    return str(getattr(s, "value", s)).lower()


@dataclass
class PDG2Config:
    """Configuration for PDG 2.0 independent mechanism ablations."""

    enable_legacy_pdg: bool = True
    enable_d1: bool = True
    enable_d2: bool = True
    cooldown_ticks: int = 12
    # Risk equation weights: R_o(t) = alpha*A_o + beta*W_o + gamma*D_o + eta*S_o
    alpha_age: float = 1.0
    beta_window: float = 0.85
    gamma_due: float = 0.35
    eta_kit: float = 0.90
    # Horizon-normalized thresholds (calibrated strictly on DEV VAL-72 / VAL-Static-35)
    d1_age_threshold: float = 475.0
    d2_risk_threshold: float = 760.0
    dwell_stall_threshold: int = 45
    osc_zpt_threshold: int = 160
    max_r4_retries: int = 6


@dataclass
class PDG2LivenessMonitor:
    """Episode-level PDG 2.0 monitor with independent per-op dependency clocks and structured risk output."""

    cfg: PDG2Config = field(default_factory=PDG2Config)
    # Global baseline counters (for legacy PDG parity)
    last_done_t: int = 0
    last_done_o: int = 0
    last_progress_tick: int = 0
    last_escalation_tick: int = -100
    recent_reject_streak: int = 0
    last_logged_decisions: int = 0
    op_first_kit_done_tick: dict[str, int] = field(default_factory=dict)

    # D1: Per-operation independent chain progress state
    op_last_chain_adv_tick: dict[str, int] = field(default_factory=dict)
    op_prev_kit_sig: dict[str, tuple[str, ...]] = field(default_factory=dict)
    op_prev_exec_sig: dict[str, tuple[bool, bool]] = field(default_factory=dict)
    op_descendant_count: dict[str, int] = field(default_factory=dict)
    _dag_initialized: bool = False

    # D2: Execution oscillation & non-recovery escalation tracking
    agv_prev_node: dict[str, str] = field(default_factory=dict)
    agv_dwell_ticks: dict[str, int] = field(default_factory=dict)
    consecutive_no_recovery: int = 0
    last_risk_peak: float = 0.0
    prev_escalation_level: int = 0

    # Telemetry
    alert_ticks: list[int] = field(default_factory=list)
    escalation_ticks: list[int] = field(default_factory=list)
    escalation_levels_used: list[int] = field(default_factory=list)
    alert_sources: dict[str, int] = field(default_factory=dict)
    latest_state: Optional[dict[str, Any]] = None

    def _init_dag(self, operations: Any) -> None:
        ops = list(operations or ())
        children: dict[str, list[str]] = {op.op_id: [] for op in ops}
        for op in ops:
            for p in op_predecessors(op):
                if p in children:
                    children[p].append(op.op_id)

        def _count_desc(oid: str, visited: set[str]) -> int:
            for ch in children.get(oid, ()):
                if ch not in visited:
                    visited.add(ch)
                    _count_desc(ch, visited)
            return len(visited)

        for op in ops:
            self.op_descendant_count[op.op_id] = _count_desc(op.op_id, set())
            self.op_last_chain_adv_tick.setdefault(op.op_id, 0)
        self._dag_initialized = True

    def observe(self, env: Any) -> dict[str, Any]:
        """Compute full PDG 2.0 state without mutating simulation physics."""
        cur_tick = int(env.current_step)
        ops = list(getattr(env.tracker, "operations", ()) or ())
        if not self._dag_initialized:
            self._init_dag(ops)

        n_logs = len(getattr(env, "decision_logs", []))
        if n_logs > self.last_logged_decisions:
            self.last_logged_decisions = n_logs
            last_dl = env.decision_logs[-1]
            if not bool(last_dl.get("receipt_accepted", False)):
                self.recent_reject_streak += 1
            else:
                self.recent_reject_streak = 0

        c_t, c_o = env._completion_counts(int(env.sim._step_cursor))
        made_global_progress = bool(c_t > self.last_done_t or c_o > self.last_done_o)
        if made_global_progress:
            self.last_done_t, self.last_done_o = c_t, c_o
            self.last_progress_tick = cur_tick

        legacy = compute_pdg_state(
            env.sim,
            env.tracker,
            last_progress_tick=self.last_progress_tick,
            op_first_kit_done_tick=self.op_first_kit_done_tick,
            recent_reject_streak=self.recent_reject_streak,
        )

        t_by_id = {t.task_id: t for t in env.sim.tasks}
        op_by_id = {op.op_id: op for op in ops}
        faulted = set(getattr(env.sim, "_faulted_agvs", set()))
        max_steps = int(getattr(getattr(env.sim, "config", None), "max_steps", getattr(env, "max_steps", 1000)))
        # Scale smoothly by horizon length (observable config parameter, never scenario/seed name)
        horizon_scale = max(1.0, min(1.85, float(max_steps) / 1000.0))

        schedulable_rem = sum(
            1
            for t in env.sim.tasks
            if _stat_lower(t) in ("unassigned", "pending", "assigned")
            and int(getattr(t, "release_step", 0)) <= cur_tick
        )

        # Track per-AGV dwell (execution oscillation / blocked congestion) and idle blockers
        stalled_agvs: list[str] = []
        blocked_healthy_agvs: list[str] = []
        active_needed_nodes: set[str] = set()
        for t in env.sim.tasks:
            st_t = _stat_lower(t)
            if st_t in ("pending", "unassigned", "assigned", "picked") and int(getattr(t, "release_step", 0)) <= cur_tick:
                if st_t != "picked":
                    active_needed_nodes.add(str(t.pickup))
                active_needed_nodes.add(str(t.dropoff))

        agv_node_map: dict[str, str] = {}
        for a in env.sim.agvs:
            aid = str(a.agv_id)
            nd = str(a.cur_node or a.start_node)
            agv_node_map[aid] = nd
            st_a = _stat_lower(a)
            if aid in faulted:
                self.agv_dwell_ticks[aid] = 0
                self.agv_prev_node[aid] = nd
                continue
            if st_a == "blocked":
                blocked_healthy_agvs.append(aid)
            if st_a in ("blocked", "to_pickup", "to_dropoff") and self.agv_prev_node.get(aid) == nd:
                self.agv_dwell_ticks[aid] = self.agv_dwell_ticks.get(aid, 0) + 1
            else:
                self.agv_dwell_ticks[aid] = 0
            self.agv_prev_node[aid] = nd
            if self.agv_dwell_ticks[aid] >= self.cfg.dwell_stall_threshold:
                stalled_agvs.append(aid)

        # Idle healthy AGVs occupying a pickup/dropoff or same node as a blocked/stalled robot
        stalled_nodes = {agv_node_map[aid] for aid in set(stalled_agvs) | set(blocked_healthy_agvs) if aid in agv_node_map}
        idle_blocker_agvs: list[str] = []
        for a in env.sim.agvs:
            aid = str(a.agv_id)
            if aid in faulted or a.cur_task is not None:
                continue
            if _stat_lower(a) == "idle":
                nd = agv_node_map[aid]
                if nd in active_needed_nodes or nd in stalled_nodes:
                    idle_blocker_agvs.append(aid)

        # D1 & D2 per-operation critical-chain risk computation
        op_risk_map: dict[str, dict[str, Any]] = {}
        d1_trigger_ops: list[str] = []
        d2_trigger_ops: list[str] = []
        max_op_age = 0.0
        max_op_risk = 0.0

        # Precompute which ops are actively machining
        op_machining: dict[str, bool] = {}
        op_done_map: dict[str, bool] = {}
        for op in ops:
            done = bool(op.end_step is not None and int(op.end_step) <= cur_tick)
            started = bool(getattr(op, "started", False))
            op_done_map[op.op_id] = done
            op_machining[op.op_id] = bool(started and not done)

        # Identify active due-date frontier among observable unstarted root ops
        root_unstarted_dues: list[int] = []
        for op in ops:
            if op_done_map[op.op_id] or op_machining[op.op_id]:
                continue
            preds = op_predecessors(op)
            if any(not op_done_map.get(p, True) for p in preds):
                continue
            kits_f = [t_by_id[kt] for kt in getattr(op, "kit_task_ids", ()) if kt in t_by_id]
            if kits_f and all(int(getattr(kt, "release_step", 0)) > cur_tick for kt in kits_f):
                continue
            root_unstarted_dues.append(int(getattr(op, "due_step", max_steps)))
        min_frontier_due = min(root_unstarted_dues) if root_unstarted_dues else max_steps

        for op in ops:
            oid = op.op_id
            done = op_done_map[oid]
            started = bool(getattr(op, "started", False))
            kits = [t_by_id[kt] for kt in getattr(op, "kit_task_ids", ()) if kt in t_by_id]
            kit_sig = tuple(_stat_lower(kt) for kt in kits)
            exec_sig = (started, done)
            preds = op_predecessors(op)
            pred_done_cnt = sum(1 for p in preds if op_done_map.get(p, False))
            full_sig = kit_sig + (f"p{pred_done_cnt}",)

            if self.op_prev_kit_sig.get(oid) != full_sig or self.op_prev_exec_sig.get(oid) != exec_sig:
                self.op_last_chain_adv_tick[oid] = cur_tick
                self.op_prev_kit_sig[oid] = full_sig
                self.op_prev_exec_sig[oid] = exec_sig

            if done:
                continue

            # Ignore unlaunched future waves whose kit tasks have not even been released yet
            if kits and all(int(getattr(kt, "release_step", 0)) > cur_tick for kt in kits):
                continue
            # Discount legitimate active machining or active immediate predecessor machining
            if op_machining[oid] or any(op_machining.get(p, False) for p in preds):
                continue
            # Also ignore if an upstream predecessor is still unstarted (attribute stall to the root upstream blocker!)
            if any(not op_done_map.get(p, True) for p in preds):
                continue

            # Check whether this root unstarted op has actionable missing kits or stalled carriers
            k_done = sum(1 for kt in kits if _stat_lower(kt) == "completed")
            k_tot = len(kits)
            k_missing = [kt for kt in kits if _stat_lower(kt) != "completed"]
            k_schedulable = sum(1 for kt in k_missing if _stat_lower(kt) in ("unassigned", "pending", "assigned"))
            k_stalled_carrier = any(
                (kt.assigned_agv in stalled_agvs or kt.assigned_agv in blocked_healthy_agvs)
                for kt in k_missing
                if kt.assigned_agv
            )
            # If all remaining kits for this op are actively in-transit (picked) and unstalled, do not false-alert
            if k_missing and k_schedulable == 0 and not k_stalled_carrier:
                continue

            due = int(getattr(op, "due_step", max_steps))
            # Only attribute starvation to ops on the active EDD frontier or already holding partial kits
            on_frontier = (due <= min_frontier_due + 100) or (0 < k_done < k_tot)
            if not on_frontier:
                continue

            # D1: Independent stall age A_o(t) weighted by downstream chain criticality
            raw_age = float(max(0, cur_tick - self.op_last_chain_adv_tick.get(oid, 0)))
            crit_mult = 1.0 + 0.12 * float(min(5, self.op_descendant_count.get(oid, 0)))
            a_o = raw_age * crit_mult

            # D2: Kitting starvation S_o(t), Resource Window Miss Debt W_o(t), Due-Date Slippage D_o(t)
            s_o = 0.0
            if 0 < k_done < k_tot:
                lag = float(max(0, cur_tick - self.op_first_kit_done_tick.get(oid, cur_tick)))
                s_o = lag * (float(k_tot - k_done) / float(max(1, k_tot)))

            wins = tuple(getattr(op, "resource_windows", ()) or ())
            w_o = 0.0
            missed_wins = 0
            if wins:
                missed_wins = sum(1 for _, w_end in wins if cur_tick > int(w_end))
                align_gap = float(max(0, align_start_to_windows(cur_tick, wins) - cur_tick))
                w_o = 110.0 * float(missed_wins) + 0.45 * align_gap

            proc = int(getattr(op, "proc_steps", 0))
            d_o = float(max(0, (cur_tick + proc) - due))

            r_o = (
                self.cfg.alpha_age * a_o
                + self.cfg.beta_window * w_o
                + self.cfg.gamma_due * d_o
                + self.cfg.eta_kit * s_o
            )
            norm_age = a_o / horizon_scale
            norm_risk = r_o / horizon_scale

            max_op_age = max(max_op_age, norm_age)
            max_op_risk = max(max_op_risk, norm_risk)
            op_risk_map[oid] = {
                "op_id": oid,
                "A_o": round(a_o, 1),
                "W_o": round(w_o, 1),
                "D_o": round(d_o, 1),
                "S_o": round(s_o, 1),
                "R_o": round(r_o, 1),
                "norm_age": round(norm_age, 1),
                "norm_risk": round(norm_risk, 1),
                "missed_wins": missed_wins,
                "k_done": k_done,
                "k_tot": k_tot,
                "descendants": self.op_descendant_count.get(oid, 0),
            }
            if self.cfg.enable_d1 and norm_age >= self.cfg.d1_age_threshold and cur_tick > due:
                d1_trigger_ops.append(oid)
            if (
                self.cfg.enable_d2
                and norm_risk >= self.cfg.d2_risk_threshold
                and cur_tick > due
                and (missed_wins >= 1 or s_o >= 260.0 or raw_age >= 360.0)
            ):
                d2_trigger_ops.append(oid)

        # Actionable legacy PDG alert (V20.0 parity baseline)
        actionable_legacy = bool(
            legacy["pdg_alert"]
            and (
                int(legacy["wait_cycle_count"]) >= 1
                or int(legacy["blocked_healthy"]) >= 1
                or int(legacy["max_loaded_stall_ticks"]) >= 5
                or schedulable_rem >= 1
                or int(legacy["zero_progress_ticks"]) >= 540
            )
        )
        legacy_fire = bool(self.cfg.enable_legacy_pdg and actionable_legacy)
        d1_fire = bool(self.cfg.enable_d1 and d1_trigger_ops)
        # D2 also triggers on sustained corridor dwell + blocked congestion when global progress stalled
        d2_osc_fire = bool(
            self.cfg.enable_d2
            and len(stalled_agvs) >= 1
            and len(blocked_healthy_agvs) >= 1
            and int(legacy["zero_progress_ticks"]) >= self.cfg.osc_zpt_threshold
        )
        d2_fire = bool(self.cfg.enable_d2 and (d2_trigger_ops or d2_osc_fire))

        is_alert = bool(legacy_fire or d1_fire or d2_fire)

        # Classify observable blocking structure (drives state-dependent escalation R1..R4)
        if int(legacy["wait_cycle_count"]) >= 1 or (len(stalled_agvs) >= 1 and (blocked_healthy_agvs or idle_blocker_agvs)):
            blocking_mode = "corridor_stall"
        elif d2_trigger_ops and any(op_risk_map[o]["missed_wins"] >= 1 or op_risk_map[o]["S_o"] >= 180.0 for o in d2_trigger_ops):
            blocking_mode = "kitting_window_starvation"
        elif d1_trigger_ops:
            blocking_mode = "critical_chain_stall"
        elif legacy_fire:
            blocking_mode = "global_deadlock"
        else:
            blocking_mode = "calm"

        # Update consecutive non-recovery streak cleanly
        if is_alert:
            if not made_global_progress and max_op_risk >= self.last_risk_peak - 25.0:
                self.consecutive_no_recovery += 1
            elif made_global_progress:
                self.consecutive_no_recovery = max(0, self.consecutive_no_recovery - 2)
            self.last_risk_peak = max(self.last_risk_peak * 0.98, max_op_risk)
            self.alert_ticks.append(cur_tick)
            for tag, fired in (("legacy", legacy_fire), ("D1", d1_fire), ("D2_op", bool(d2_trigger_ops)), ("D2_osc", d2_osc_fire)):
                if fired:
                    self.alert_sources[tag] = self.alert_sources.get(tag, 0) + 1
        else:
            self.consecutive_no_recovery = 0
            self.last_risk_peak = 0.0

        top_risk_ops = [
            k for k, _ in sorted(op_risk_map.items(), key=lambda kv: kv[1]["norm_risk"], reverse=True)[:4]
        ]

        state = dict(legacy)
        state.update(
            {
                "pdg_alert": is_alert,
                "pdg2_alert": is_alert,
                "legacy_fire": legacy_fire,
                "d1_fire": d1_fire,
                "d2_fire": d2_fire,
                "blocking_mode": blocking_mode,
                "schedulable_rem_tasks": int(schedulable_rem),
                "max_op_norm_age": round(max_op_age, 1),
                "max_op_norm_risk": round(max_op_risk, 1),
                "top_risk_ops": top_risk_ops,
                "op_risks": op_risk_map,
                "stalled_agvs": sorted(set(stalled_agvs)),
                "blocked_healthy_agvs": sorted(set(blocked_healthy_agvs)),
                "idle_blocker_agvs": sorted(set(idle_blocker_agvs)),
                "consecutive_no_recovery": int(self.consecutive_no_recovery),
            }
        )
        self.latest_state = state
        return state

    def observe_and_maybe_escalate(
        self,
        env: Any,
        info: dict[str, Any],
        base_action: int,
        _traces: Any = None,
        *,
        proposer: Optional[Any] = None,
    ) -> int:
        """Observe PDG 2.0 state, push structured diagnostics to proposer, and trigger outer action escalation."""
        state = self.observe(env)
        cur_tick = int(env.current_step)
        is_alert = bool(state["pdg2_alert"])

        if proposer is not None:
            if hasattr(proposer, "set_pdg2_context"):
                proposer.set_pdg2_context(state if is_alert else None, sim_ref=env.sim if is_alert else None)
            elif hasattr(proposer, "set_pdg_state"):
                proposer.set_pdg_state(state if is_alert else None)

        if (
            is_alert
            and int(base_action) == 0
            and (cur_tick - self.last_escalation_tick) >= self.cfg.cooldown_ticks
        ):
            self.last_escalation_tick = cur_tick
            self.escalation_ticks.append(cur_tick)
            mask = info.get("action_mask")
            # For corridor stalls, prefer action=1 (surgical TPG repair, guarantees allow_live_repair=True in env.py)
            if state["blocking_mode"] == "corridor_stall" and mask is not None and len(mask) > 1 and int(mask[1]) == 1:
                return 1
            if mask is not None and len(mask) > 2:
                legal_replans = [i for i in range(1, len(mask)) if int(mask[i]) == 1]
                if legal_replans:
                    return int(legal_replans[-1])
            return 1 if state["blocking_mode"] == "corridor_stall" else 2
        return int(base_action)
