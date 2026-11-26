"""Production Cross-Domain Progress Dependency Graph (PDG 1.0 / PDG 2.0) Liveness & Recovery.

Self-contained production module for CoRe-Sched V20.0 (`V20_NAJS`) and V20.1 (`V20_1_NAJS`):
  - `compute_pdg_state` & `PDGLivenessTracker`: exact frozen V20.0 global PDG liveness monitor.
  - `PDG2Config` & `PDG2LivenessMonitor`: V20.1 per-operation independent critical-chain stall (`D1`)
    with active EDD frontier filtering + window-miss debt & oscillation detector (`D2`).
  - `build_cross_domain_wait_graph` & `PDG2SCCLivenessMonitor`: Petri-Net inspired cross-domain
    Wait-For Graph + Exit-Free Tarjan Deadlock-SCC confirmation & 4-class structural telemetry.
  - `PersistentEscapeMemory`: commitment-compatible `(state_sig, commitment_sig, action_sig)`
    non-destructive tabu memory suppressing repeated dead-end recovery attempts.
  - `plan_idle_blocker_evacuations` & `build_critical_bottleneck_locks`: commitment-preserving
    corridor evacuation and bottleneck passage locks (`E1`).

Note: Framed strictly as risk detection and structured recovery under online commitment
constraints — NOT a formal universal liveness guarantee.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Optional

from amr_sched.cvcr.snapshot import NativeSnapshot, align_start_to_windows, op_predecessors
from amr_sched.cvcr.v12_proactive_guard import analyze_network_havens
from amr_sched.network import Network


def _stat_lower(obj: Any) -> str:
    s = getattr(obj, "status", getattr(obj, "state", ""))
    return str(getattr(s, "value", s)).lower()


def _completion_counts_from_sim(sim: Any, tracker: Any, tick: int) -> tuple[int, int]:
    c_t = sum(1 for t in getattr(sim, "tasks", ()) if _stat_lower(t) == "completed")
    c_o = sum(
        1
        for op in getattr(tracker, "operations", ())
        if getattr(op, "end_step", None) is not None and int(op.end_step) <= int(tick)
    )
    return c_t, c_o


def compute_pdg_state(
    sim: Any,
    tracker: Any,
    *,
    last_progress_tick: int,
    op_first_kit_done_tick: dict[str, int],
    recent_reject_streak: int,
) -> dict[str, Any]:
    """Compute topology-agnostic global Progress Dependency Graph (PDG 1.0) metrics."""
    tick = int(getattr(sim, "_step_cursor", 0))
    faulted = set(getattr(sim, "_faulted_agvs", set()))
    blocked_since = dict(getattr(sim, "_blocked_since", {}))
    tasks_by_id = getattr(sim, "_task_by_id", {t.task_id: t for t in sim.tasks})

    healthy_agvs = [a for a in sim.agvs if a.agv_id not in faulted and not getattr(a, "faulted", False)]
    node_occupant: dict[str, str] = {}
    for a in sim.agvs:
        cur_n = getattr(a, "cur_node", None) or getattr(a, "start_node", None)
        if cur_n:
            node_occupant[str(cur_n)] = a.agv_id

    wait_adj: dict[str, set[str]] = {a.agv_id: set() for a in healthy_agvs}
    max_loaded_stall = 0
    blocked_healthy = 0
    for a in healthy_agvs:
        aid = a.agv_id
        st_name = _stat_lower(a)
        is_blocked = (aid in blocked_since) or st_name in ("blocked", "parking")
        if is_blocked:
            blocked_healthy += 1
            stall_len = max(1, tick - int(blocked_since.get(aid, tick)))
            t_cur = tasks_by_id.get(getattr(a, "cur_task", None) or "")
            t_stat = _stat_lower(t_cur) if t_cur else ""
            if t_stat == "picked":
                max_loaded_stall = max(max_loaded_stall, stall_len)
            nxt_nodes: list[str] = []
            plan = getattr(sim, "_active_plans", {}).get(aid)
            if plan and getattr(plan, "nodes", None):
                cur_n = getattr(a, "cur_node", None)
                for n_step in plan.nodes:
                    if str(n_step) != str(cur_n):
                        nxt_nodes.append(str(n_step))
                        break
            elif t_cur is not None:
                tgt = str(t_cur.dropoff if t_stat == "picked" else t_cur.pickup)
                if tgt:
                    nxt_nodes.append(tgt)
            for nxt in nxt_nodes:
                occ = node_occupant.get(nxt)
                if occ and occ != aid and occ in wait_adj:
                    wait_adj[aid].add(occ)

    cycle_nodes: set[str] = set()
    cycle_count = 0
    color: dict[str, int] = {}
    stack: list[str] = []

    def _dfs(u: str) -> None:
        nonlocal cycle_count
        color[u] = 1
        stack.append(u)
        for v in sorted(wait_adj.get(u, ())):
            c = color.get(v, 0)
            if c == 1 and v in stack:
                cycle_count += 1
                cycle_nodes.update(stack[stack.index(v) :])
            elif c == 0:
                _dfs(v)
        stack.pop()
        color[u] = 2

    for u in sorted(wait_adj):
        if color.get(u, 0) == 0:
            _dfs(u)

    kit_split_lags: list[int] = []
    for op in getattr(tracker, "operations", ()) or ():
        if getattr(op, "started", False) or getattr(op, "completed", False):
            continue
        k_done = sum(
            1
            for kt in getattr(op, "kit_task_ids", ())
            if kt in tasks_by_id and _stat_lower(tasks_by_id[kt]) == "completed"
        )
        if 0 < k_done < len(getattr(op, "kit_task_ids", ())):
            first_t = op_first_kit_done_tick.setdefault(op.op_id, tick)
            kit_split_lags.append(tick - first_t)
        elif op.op_id in op_first_kit_done_tick and k_done == len(getattr(op, "kit_task_ids", ())):
            op_first_kit_done_tick.pop(op.op_id, None)

    max_kit_split_lag = max(kit_split_lags) if kit_split_lags else 0
    rem_tasks = sum(1 for t in sim.tasks if _stat_lower(t) not in ("completed", "failed"))
    zero_progress_ticks = (tick - last_progress_tick) if rem_tasks > 0 else 0
    blocked_ratio = blocked_healthy / max(1, len(healthy_agvs))

    alert = bool(
        rem_tasks > 0
        and (
            (cycle_count >= 1 and zero_progress_ticks >= 40)
            or (blocked_ratio >= 0.75 and max_loaded_stall >= 25 and zero_progress_ticks >= 40)
            or (zero_progress_ticks >= 420 and max_kit_split_lag >= 200)
            or zero_progress_ticks >= 500
        )
    )
    return {
        "tick": tick,
        "rem_tasks": rem_tasks,
        "blocked_healthy": blocked_healthy,
        "blocked_ratio": round(blocked_ratio, 3),
        "max_loaded_stall_ticks": max_loaded_stall,
        "wait_cycle_count": cycle_count,
        "wait_cycle_agvs": sorted(cycle_nodes),
        "max_kit_split_lag": max_kit_split_lag,
        "zero_progress_ticks": zero_progress_ticks,
        "recent_reject_streak": recent_reject_streak,
        "pdg_alert": alert,
    }


@dataclass
class PDGLivenessTracker:
    """Frozen CoRe-Sched V20.0 episode-level PDG 1.0 liveness monitor."""

    cooldown_ticks: int = 12
    last_done_t: int = 0
    last_done_o: int = 0
    last_progress_tick: int = 0
    last_escalation_tick: int = -100
    recent_reject_streak: int = 0
    last_logged_decisions: int = 0
    op_first_kit_done_tick: dict[str, int] = field(default_factory=dict)
    alert_ticks: list[int] = field(default_factory=list)
    escalation_ticks: list[int] = field(default_factory=list)
    latest_pdg: Optional[dict[str, Any]] = None

    def observe_sim(
        self,
        sim: Any,
        tracker: Any,
        *,
        cur_tick: Optional[int] = None,
        decision_logs: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        tick = int(getattr(sim, "_step_cursor", 0) if cur_tick is None else cur_tick)
        if decision_logs is not None:
            n_logs = len(decision_logs)
            if n_logs > self.last_logged_decisions:
                self.last_logged_decisions = n_logs
                if not bool(decision_logs[-1].get("receipt_accepted", False)):
                    self.recent_reject_streak += 1
                else:
                    self.recent_reject_streak = 0

        c_t, c_o = _completion_counts_from_sim(sim, tracker, int(getattr(sim, "_step_cursor", tick)))
        if c_t > self.last_done_t or c_o > self.last_done_o:
            self.last_done_t, self.last_done_o = c_t, c_o
            self.last_progress_tick = tick

        pdg = compute_pdg_state(
            sim,
            tracker,
            last_progress_tick=self.last_progress_tick,
            op_first_kit_done_tick=self.op_first_kit_done_tick,
            recent_reject_streak=self.recent_reject_streak,
        )
        schedulable_rem = sum(
            1
            for t in sim.tasks
            if _stat_lower(t) in ("unassigned", "pending", "assigned")
            and int(getattr(t, "release_step", 0)) <= tick
        )
        actionable_distress = bool(
            int(pdg["wait_cycle_count"]) >= 1
            or int(pdg["blocked_healthy"]) >= 1
            or int(pdg["max_loaded_stall_ticks"]) >= 5
            or schedulable_rem >= 1
            or int(pdg["zero_progress_ticks"]) >= 540
        )
        is_alert = bool(pdg["pdg_alert"]) and actionable_distress
        pdg["schedulable_rem_tasks"] = int(schedulable_rem)
        pdg["pdg_alert"] = is_alert
        pdg["liveness_class"] = (
            "DEADLOCK" if int(pdg["wait_cycle_count"]) >= 1 else ("STARVATION" if is_alert else "NORMAL_WAITING")
        )
        self.latest_pdg = pdg
        if is_alert:
            self.alert_ticks.append(tick)
        return pdg

    def observe_and_maybe_escalate(
        self,
        env: Any,
        info: dict[str, Any],
        base_action: int,
        _traces: Any = None,
        *,
        proposer: Optional[Any] = None,
    ) -> int:
        cur_tick = int(env.current_step)
        pdg = self.observe_sim(
            env.sim,
            env.tracker,
            cur_tick=cur_tick,
            decision_logs=getattr(env, "decision_logs", None),
        )
        is_alert = bool(pdg["pdg_alert"])
        if proposer is not None and hasattr(proposer, "set_pdg_state"):
            proposer.set_pdg_state(pdg if is_alert else None)

        if (
            is_alert
            and int(base_action) == 0
            and (cur_tick - self.last_escalation_tick) >= self.cooldown_ticks
        ):
            self.last_escalation_tick = cur_tick
            self.escalation_ticks.append(cur_tick)
            mask = info.get("action_mask")
            if mask is not None and len(mask) > 2:
                legal_replans = [i for i in range(1, len(mask)) if int(mask[i]) == 1]
                if legal_replans:
                    return int(legal_replans[-1])
            return 2
        return int(base_action)


def build_cross_domain_wait_graph(sim: Any, tracker: Any, cur_tick: int) -> dict[str, Any]:
    """Build explicit directed Wait-For Graph `u -> v` and compute exact exit-free Tarjan SCCs."""
    adj: dict[str, set[str]] = {}

    def _edge(u: str, v: str) -> None:
        if u != v:
            adj.setdefault(u, set()).add(v)
            adj.setdefault(v, set())

    ops = list(getattr(tracker, "operations", ()) or ())
    tasks = list(getattr(sim, "tasks", ()) or ())
    agvs = list(getattr(sim, "agvs", ()) or ())
    t_by_id = {t.task_id: t for t in tasks}
    op_done = {op.op_id: bool(op.end_step is not None and int(op.end_step) <= cur_tick) for op in ops}
    op_machining = {op.op_id: bool(getattr(op, "started", False) and not op_done[op.op_id]) for op in ops}

    bay_holder: dict[str, str] = {}
    for op in ops:
        if op_machining[op.op_id]:
            bay_holder[str(op.bay_id)] = op.op_id

    node_holders: dict[str, list[str]] = {}
    agv_node: dict[str, str] = {}
    faulted = set(getattr(sim, "_faulted_agvs", set()))
    active_exits: set[str] = set()

    for a in agvs:
        aid = str(a.agv_id)
        nd = str(a.cur_node or a.start_node)
        agv_node[aid] = nd
        node_holders.setdefault(nd, []).append(aid)
        st = _stat_lower(a)
        if aid not in faulted and st in ("to_pickup", "to_dropoff", "loading", "unloading") and aid in getattr(sim, "_transit", {}):
            active_exits.add(f"AGV:{aid}")

    for op in ops:
        if op_done[op.op_id]:
            continue
        if op_machining[op.op_id]:
            active_exits.add(f"OP:{op.op_id}")
            continue
        kits = [t_by_id[kt] for kt in getattr(op, "kit_task_ids", ()) if kt in t_by_id]
        if kits and all(int(getattr(kt, "release_step", 0)) > cur_tick for kt in kits):
            continue
        u_op = f"OP:{op.op_id}"
        adj.setdefault(u_op, set())
        for p in op_predecessors(op):
            if not op_done.get(p, True):
                _edge(u_op, f"OP:{p}")
        for kt in kits:
            if _stat_lower(kt) != "completed":
                _edge(u_op, f"TASK:{kt.task_id}")
        b_hold = bay_holder.get(str(op.bay_id))
        if b_hold and b_hold != op.op_id:
            _edge(u_op, f"OP:{b_hold}")

    for t in tasks:
        st_t = _stat_lower(t)
        if st_t in ("completed", "failed") or int(getattr(t, "release_step", 0)) > cur_tick:
            continue
        u_t = f"TASK:{t.task_id}"
        adj.setdefault(u_t, set())
        if t.assigned_agv:
            _edge(u_t, f"AGV:{t.assigned_agv}")
        target_nd = str(t.dropoff) if st_t == "picked" else str(t.pickup)
        for occ_a in node_holders.get(target_nd, ()):
            if occ_a != t.assigned_agv:
                _edge(u_t, f"AGV:{occ_a}")

    jobs = getattr(sim, "_jobs", {}) or {}
    for a in agvs:
        aid = str(a.agv_id)
        if aid in faulted:
            continue
        st_a = _stat_lower(a)
        if st_a not in ("blocked", "to_pickup", "to_dropoff"):
            continue
        u_a = f"AGV:{aid}"
        adj.setdefault(u_a, set())
        next_nodes: list[str] = []
        jb = jobs.get(aid)
        if jb is not None:
            for h in (getattr(jb, "hops", None) or ())[:2]:
                dst = str(getattr(h, "dst", ""))
                if dst and dst != agv_node[aid]:
                    next_nodes.append(dst)
        if not next_nodes and a.cur_task and a.cur_task in t_by_id:
            tk = t_by_id[a.cur_task]
            tgt = str(tk.dropoff) if _stat_lower(tk) == "picked" else str(tk.pickup)
            if tgt != agv_node[aid]:
                next_nodes.append(tgt)
        for nd in next_nodes:
            for occ_a in node_holders.get(nd, ()):
                if occ_a != aid:
                    _edge(u_a, f"AGV:{occ_a}")

    # Exact Tarjan SCC
    index_counter = 0
    indices: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    stack: list[str] = []
    on_stack: set[str] = set()
    sccs: list[list[str]] = []

    def _strongconnect(v: str) -> None:
        nonlocal index_counter
        indices[v] = index_counter
        lowlink[v] = index_counter
        index_counter += 1
        stack.append(v)
        on_stack.add(v)
        for w in sorted(adj.get(v, ())):
            if w not in indices:
                _strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])
        if lowlink[v] == indices[v]:
            comp: list[str] = []
            while True:
                w = stack.pop()
                on_stack.remove(w)
                comp.append(w)
                if w == v:
                    break
            if len(comp) >= 2:
                sccs.append(sorted(comp))

    for node in sorted(adj):
        if node not in indices:
            _strongconnect(node)

    deadlock_sccs = [comp for comp in sccs if not any(n in active_exits for n in comp)]
    return {
        "n_nodes": len(adj),
        "n_edges": sum(len(v) for v in adj.values()),
        "sccs": sccs,
        "deadlock_sccs": deadlock_sccs,
        "active_exits": sorted(active_exits),
    }


@dataclass
class PDG2Config:
    """DEV-calibrated configuration for production PDG 2.0 (D1 + D2 + Exit-Free Tarjan SCC)."""

    enable_legacy_pdg: bool = True
    enable_d1: bool = True
    enable_d2: bool = True
    cooldown_ticks: int = 12
    alpha_age: float = 1.0
    beta_window: float = 0.85
    gamma_due: float = 0.35
    eta_kit: float = 0.90
    d1_age_threshold: float = 475.0
    d2_risk_threshold: float = 760.0
    dwell_stall_threshold: int = 45
    osc_zpt_threshold: int = 160
    scc_persist_ticks: int = 45
    scc_zpt_min: int = 90
    max_r4_retries: int = 6


@dataclass
class PDG2LivenessMonitor:
    """Production PDG 2.0 monitor (`D1` per-op critical chain + `D2` window/oscillation)."""

    cfg: PDG2Config = field(default_factory=PDG2Config)
    last_done_t: int = 0
    last_done_o: int = 0
    last_progress_tick: int = 0
    last_escalation_tick: int = -100
    recent_reject_streak: int = 0
    last_logged_decisions: int = 0
    op_first_kit_done_tick: dict[str, int] = field(default_factory=dict)

    op_last_chain_adv_tick: dict[str, int] = field(default_factory=dict)
    op_prev_kit_sig: dict[str, tuple[str, ...]] = field(default_factory=dict)
    op_prev_exec_sig: dict[str, tuple[bool, bool]] = field(default_factory=dict)
    op_descendant_count: dict[str, int] = field(default_factory=dict)
    _dag_initialized: bool = False

    agv_prev_node: dict[str, str] = field(default_factory=dict)
    agv_dwell_ticks: dict[str, int] = field(default_factory=dict)
    consecutive_no_recovery: int = 0
    last_risk_peak: float = 0.0

    alert_ticks: list[int] = field(default_factory=list)
    escalation_ticks: list[int] = field(default_factory=list)
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

    def observe_sim(
        self,
        sim: Any,
        tracker: Any,
        *,
        cur_tick: Optional[int] = None,
        decision_logs: Optional[list[dict[str, Any]]] = None,
        max_steps_hint: int = 1000,
    ) -> dict[str, Any]:
        tick = int(getattr(sim, "_step_cursor", 0) if cur_tick is None else cur_tick)
        ops = list(getattr(tracker, "operations", ()) or ())
        if not self._dag_initialized:
            self._init_dag(ops)

        if decision_logs is not None:
            n_logs = len(decision_logs)
            if n_logs > self.last_logged_decisions:
                self.last_logged_decisions = n_logs
                if not bool(decision_logs[-1].get("receipt_accepted", False)):
                    self.recent_reject_streak += 1
                else:
                    self.recent_reject_streak = 0

        c_t, c_o = _completion_counts_from_sim(sim, tracker, int(getattr(sim, "_step_cursor", tick)))
        made_global_progress = bool(c_t > self.last_done_t or c_o > self.last_done_o)
        if made_global_progress:
            self.last_done_t, self.last_done_o = c_t, c_o
            self.last_progress_tick = tick

        legacy = compute_pdg_state(
            sim,
            tracker,
            last_progress_tick=self.last_progress_tick,
            op_first_kit_done_tick=self.op_first_kit_done_tick,
            recent_reject_streak=self.recent_reject_streak,
        )

        t_by_id = {t.task_id: t for t in sim.tasks}
        faulted = set(getattr(sim, "_faulted_agvs", set()))
        max_steps = int(getattr(getattr(sim, "config", None), "max_steps", max_steps_hint))
        horizon_scale = max(1.0, min(1.85, float(max_steps) / 1000.0))

        schedulable_rem = sum(
            1
            for t in sim.tasks
            if _stat_lower(t) in ("unassigned", "pending", "assigned")
            and int(getattr(t, "release_step", 0)) <= tick
        )

        stalled_agvs: list[str] = []
        blocked_healthy_agvs: list[str] = []
        active_needed_nodes: set[str] = set()
        for t in sim.tasks:
            st_t = _stat_lower(t)
            if st_t in ("pending", "unassigned", "assigned", "picked") and int(getattr(t, "release_step", 0)) <= tick:
                if st_t != "picked":
                    active_needed_nodes.add(str(t.pickup))
                active_needed_nodes.add(str(t.dropoff))

        agv_node_map: dict[str, str] = {}
        for a in sim.agvs:
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

        stalled_nodes = {agv_node_map[aid] for aid in set(stalled_agvs) | set(blocked_healthy_agvs) if aid in agv_node_map}
        idle_blocker_agvs: list[str] = []
        for a in sim.agvs:
            aid = str(a.agv_id)
            if aid in faulted or a.cur_task is not None:
                continue
            if _stat_lower(a) == "idle":
                nd = agv_node_map[aid]
                if nd in active_needed_nodes or nd in stalled_nodes:
                    idle_blocker_agvs.append(aid)

        op_risk_map: dict[str, dict[str, Any]] = {}
        d1_trigger_ops: list[str] = []
        d2_trigger_ops: list[str] = []
        max_op_age = 0.0
        max_op_risk = 0.0

        op_machining: dict[str, bool] = {}
        op_done_map: dict[str, bool] = {}
        for op in ops:
            done = bool(op.end_step is not None and int(op.end_step) <= tick)
            started = bool(getattr(op, "started", False))
            op_done_map[op.op_id] = done
            op_machining[op.op_id] = bool(started and not done)

        root_unstarted_dues: list[int] = []
        for op in ops:
            if op_done_map[op.op_id] or op_machining[op.op_id]:
                continue
            if any(not op_done_map.get(p, True) for p in op_predecessors(op)):
                continue
            kits_f = [t_by_id[kt] for kt in getattr(op, "kit_task_ids", ()) if kt in t_by_id]
            if kits_f and all(int(getattr(kt, "release_step", 0)) > tick for kt in kits_f):
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
                self.op_last_chain_adv_tick[oid] = tick
                self.op_prev_kit_sig[oid] = full_sig
                self.op_prev_exec_sig[oid] = exec_sig

            if done:
                continue
            if kits and all(int(getattr(kt, "release_step", 0)) > tick for kt in kits):
                continue
            if op_machining[oid] or any(op_machining.get(p, False) for p in preds):
                continue
            if any(not op_done_map.get(p, True) for p in preds):
                continue

            k_done = sum(1 for kt in kits if _stat_lower(kt) == "completed")
            k_tot = len(kits)
            k_missing = [kt for kt in kits if _stat_lower(kt) != "completed"]
            k_schedulable = sum(1 for kt in k_missing if _stat_lower(kt) in ("unassigned", "pending", "assigned"))
            k_stalled_carrier = any(
                (kt.assigned_agv in stalled_agvs or kt.assigned_agv in blocked_healthy_agvs)
                for kt in k_missing
                if kt.assigned_agv
            )
            if k_missing and k_schedulable == 0 and not k_stalled_carrier:
                continue

            due = int(getattr(op, "due_step", max_steps))
            if not ((due <= min_frontier_due + 100) or (0 < k_done < k_tot)):
                continue

            raw_age = float(max(0, tick - self.op_last_chain_adv_tick.get(oid, 0)))
            crit_mult = 1.0 + 0.12 * float(min(5, self.op_descendant_count.get(oid, 0)))
            a_o = raw_age * crit_mult

            s_o = 0.0
            if 0 < k_done < k_tot:
                lag = float(max(0, tick - self.op_first_kit_done_tick.get(oid, tick)))
                s_o = lag * (float(k_tot - k_done) / float(max(1, k_tot)))

            wins = tuple(getattr(op, "resource_windows", ()) or ())
            w_o = 0.0
            missed_wins = 0
            if wins:
                missed_wins = sum(1 for _, w_end in wins if tick > int(w_end))
                align_gap = float(max(0, align_start_to_windows(tick, wins) - tick))
                w_o = 110.0 * float(missed_wins) + 0.45 * align_gap

            proc = int(getattr(op, "proc_steps", 0))
            d_o = float(max(0, (tick + proc) - due))
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
            if self.cfg.enable_d1 and norm_age >= self.cfg.d1_age_threshold and tick > due:
                d1_trigger_ops.append(oid)
            if (
                self.cfg.enable_d2
                and norm_risk >= self.cfg.d2_risk_threshold
                and tick > due
                and (missed_wins >= 1 or s_o >= 260.0 or raw_age >= 360.0)
            ):
                d2_trigger_ops.append(oid)

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
        d2_osc_fire = bool(
            self.cfg.enable_d2
            and len(stalled_agvs) >= 1
            and len(blocked_healthy_agvs) >= 1
            and int(legacy["zero_progress_ticks"]) >= self.cfg.osc_zpt_threshold
        )
        d2_fire = bool(self.cfg.enable_d2 and (d2_trigger_ops or d2_osc_fire))
        is_alert = bool(legacy_fire or d1_fire or d2_fire)

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

        if is_alert:
            if not made_global_progress and max_op_risk >= self.last_risk_peak - 25.0:
                self.consecutive_no_recovery += 1
            elif made_global_progress:
                self.consecutive_no_recovery = max(0, self.consecutive_no_recovery - 2)
            self.last_risk_peak = max(self.last_risk_peak * 0.98, max_op_risk)
            self.alert_ticks.append(tick)
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

    def observe(self, env: Any) -> dict[str, Any]:
        return self.observe_sim(
            env.sim,
            env.tracker,
            cur_tick=int(env.current_step),
            decision_logs=getattr(env, "decision_logs", None),
            max_steps_hint=int(getattr(env, "max_steps", 1000)),
        )


@dataclass
class PDG2SCCLivenessMonitor:
    """Minimal approved V20.1 monitor: PDG 2.0 (D1+D2) + Exit-Free Tarjan Deadlock-SCC confirmation."""

    cfg: PDG2Config = field(default_factory=PDG2Config)
    base_monitor: PDG2LivenessMonitor = field(init=False)
    scc_streak: int = 0
    alert_ticks: list[int] = field(default_factory=list)
    escalation_ticks: list[int] = field(default_factory=list)
    last_escalation_tick: int = -100
    state_class_counts: dict[str, int] = field(default_factory=dict)
    latest_state: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.base_monitor = PDG2LivenessMonitor(cfg=self.cfg)

    def observe_sim(
        self,
        sim: Any,
        tracker: Any,
        *,
        cur_tick: Optional[int] = None,
        decision_logs: Optional[list[dict[str, Any]]] = None,
        max_steps_hint: int = 1000,
    ) -> dict[str, Any]:
        tick = int(getattr(sim, "_step_cursor", 0) if cur_tick is None else cur_tick)
        st = self.base_monitor.observe_sim(
            sim,
            tracker,
            cur_tick=tick,
            decision_logs=decision_logs,
            max_steps_hint=max_steps_hint,
        )

        deadlock_sccs: list[list[str]] = []
        if int(st["zero_progress_ticks"]) >= 30 and (len(st["stalled_agvs"]) > 0 or len(st["blocked_healthy_agvs"]) > 0):
            wg = build_cross_domain_wait_graph(sim, tracker, tick)
            deadlock_sccs = wg["deadlock_sccs"]
        if deadlock_sccs:
            self.scc_streak += 1
        else:
            self.scc_streak = 0

        scc_fire = bool(
            self.scc_streak >= self.cfg.scc_persist_ticks
            and int(st["zero_progress_ticks"]) >= self.cfg.scc_zpt_min
        )
        if scc_fire or int(st["wait_cycle_count"]) >= 1:
            liveness_class = "DEADLOCK"
        elif st["d2_fire"] and st["blocking_mode"] == "kitting_window_starvation":
            liveness_class = "WINDOW_FAILURE"
        elif st["d1_fire"] or st["legacy_fire"]:
            liveness_class = "STARVATION"
        else:
            liveness_class = "NORMAL_WAITING"
        self.state_class_counts[liveness_class] = self.state_class_counts.get(liveness_class, 0) + 1

        is_alert = bool(st["pdg2_alert"] or scc_fire)
        st["pdg_alert"] = is_alert
        st["pdg2_alert"] = is_alert
        st["liveness_class"] = liveness_class
        st["deadlock_sccs"] = deadlock_sccs
        st["scc_streak"] = self.scc_streak
        if deadlock_sccs and is_alert:
            scc_agvs = {n.split(":", 1)[1] for comp in deadlock_sccs for n in comp if n.startswith("AGV:")}
            st["wait_cycle_agvs"] = sorted(set(st.get("wait_cycle_agvs") or ()) | scc_agvs)
            if scc_fire:
                st["blocking_mode"] = "corridor_stall"

        self.latest_state = st
        if is_alert:
            self.alert_ticks.append(tick)
        return st

    def observe_and_maybe_escalate(
        self,
        env: Any,
        info: dict[str, Any],
        base_action: int,
        _traces: Any = None,
        *,
        proposer: Optional[Any] = None,
    ) -> int:
        cur_tick = int(env.current_step)
        st = self.observe_sim(
            env.sim,
            env.tracker,
            cur_tick=cur_tick,
            decision_logs=getattr(env, "decision_logs", None),
            max_steps_hint=int(getattr(env, "max_steps", 1000)),
        )
        is_alert = bool(st["pdg2_alert"])
        if proposer is not None:
            if hasattr(proposer, "set_pdg2_context"):
                proposer.set_pdg2_context(st if is_alert else None, sim_ref=env.sim if is_alert else None)
            elif hasattr(proposer, "set_pdg_state"):
                proposer.set_pdg_state(st if is_alert else None)

        if (
            is_alert
            and int(base_action) == 0
            and (cur_tick - self.last_escalation_tick) >= self.cfg.cooldown_ticks
        ):
            self.last_escalation_tick = cur_tick
            self.escalation_ticks.append(cur_tick)
            mask = info.get("action_mask")
            if st["liveness_class"] == "DEADLOCK" and mask is not None and len(mask) > 1 and int(mask[1]) == 1:
                return 1
            if mask is not None and len(mask) > 2:
                legal = [i for i in range(1, len(mask)) if int(mask[i]) == 1]
                if legal:
                    return int(legal[-1])
            return 1 if st["liveness_class"] == "DEADLOCK" else 2
        return int(base_action)


@dataclass
class RecoveryMemoryEntry:
    state_sig: str
    commitment_sig: str
    neighborhood_key: str
    action_sig: str
    tick: int
    outcome: str = "pending"


@dataclass
class PersistentEscapeMemory:
    """Commitment-compatible non-destructive escape memory preventing repeated dead-end loops."""

    max_entries: int = 128
    entries: list[RecoveryMemoryEntry] = field(default_factory=list)
    duplicate_candidates_pruned: int = 0
    _last_entry: Optional[RecoveryMemoryEntry] = None
    _last_completed_counts: tuple[int, int] = (0, 0)

    @staticmethod
    def compute_signatures(snapshot: NativeSnapshot, pdg: dict[str, Any]) -> tuple[str, str]:
        dist = set(pdg.get("stalled_agvs") or ()) | set(pdg.get("blocked_healthy_agvs") or ()) | set(pdg.get("wait_cycle_agvs") or ())
        r_sig = sorted(
            (r.robot_id, r.cur_node, r.state, r.cur_task)
            for r in snapshot.robots
            if r.robot_id in dist or r.state.lower() in ("blocked", "faulted")
        )
        t_done = sum(1 for t in snapshot.tasks if t.status.lower() == "completed")
        o_done = sum(1 for op in snapshot.operations if op.end_step is not None)
        state_raw = json.dumps({"r": r_sig, "td": t_done, "od": o_done, "ops": pdg.get("top_risk_ops", [])[:3]}, separators=(",", ":")).encode()
        comm_raw = json.dumps(
            {
                "picked": sorted((t.task_id, t.assigned_agv) for t in snapshot.tasks if t.status.lower() == "picked"),
                "started_ops": sorted(op.op_id for op in snapshot.operations if op.started),
            },
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(state_raw).hexdigest()[:14], hashlib.sha256(comm_raw).hexdigest()[:14]

    def update_previous_outcome(self, snapshot: NativeSnapshot, pdg: dict[str, Any]) -> tuple[str, str]:
        s_sig, c_sig = self.compute_signatures(snapshot, pdg)
        c_counts = (
            sum(1 for t in snapshot.tasks if t.status.lower() == "completed"),
            sum(1 for op in snapshot.operations if op.end_step is not None),
        )
        if self._last_entry is not None and self._last_entry.outcome == "pending":
            if c_counts > self._last_completed_counts or s_sig != self._last_entry.state_sig:
                self._last_entry.outcome = "progressed"
            else:
                self._last_entry.outcome = "no_progress"
        self._last_completed_counts = c_counts
        return s_sig, c_sig

    def failed_action_sigs(self, state_sig: str, commitment_sig: str) -> set[str]:
        return {
            e.action_sig
            for e in self.entries
            if e.state_sig == state_sig and e.commitment_sig == commitment_sig and e.outcome == "no_progress"
        }

    def record_attempt(self, state_sig: str, commitment_sig: str, neighborhood_key: str, action_sig: str, tick: int) -> None:
        entry = RecoveryMemoryEntry(
            state_sig=state_sig,
            commitment_sig=commitment_sig,
            neighborhood_key=neighborhood_key,
            action_sig=action_sig,
            tick=tick,
            outcome="pending",
        )
        self.entries.append(entry)
        if len(self.entries) > self.max_entries:
            self.entries.pop(0)
        self._last_entry = entry


_HAVEN_CACHE: dict[tuple[int, int, int], Any] = {}


def _cached_network_havens(network: Network) -> Any:
    key = (id(network), len(network.nodes), len(network.edges))
    val = _HAVEN_CACHE.get(key)
    if val is None:
        if len(_HAVEN_CACHE) > 32:
            _HAVEN_CACHE.clear()
        val = analyze_network_havens(network)
        _HAVEN_CACHE[key] = val
    return val


def plan_idle_blocker_evacuations(
    snapshot: NativeSnapshot,
    network: Network,
    sim_ref: Any,
    idle_blocker_agvs: list[str],
    faulted_robots: set[str],
) -> dict[str, dict[str, Any]]:
    """Build transaction-legal `idle_park` repairs moving idle AGVs off active/blocked nodes."""
    if not idle_blocker_agvs or sim_ref is None:
        return {}
    tick = int(snapshot.tick)
    max_steps = int(getattr(getattr(sim_ref, "config", None), "max_steps", 2000))
    transit = getattr(sim_ref, "_transit", {})
    agv_by_id = {str(a.agv_id): a for a in sim_ref.agvs}

    occupied_nodes = {str(r.cur_node) for r in snapshot.robots if r.cur_node}
    faulted_nodes = {str(r.cur_node) for r in snapshot.robots if r.robot_id in faulted_robots and r.cur_node}
    needed_nodes: set[str] = set()
    for t in snapshot.tasks:
        if t.status.lower() not in ("completed", "failed") and int(t.release_step) <= tick:
            needed_nodes.add(str(t.pickup))
            needed_nodes.add(str(t.dropoff))

    havens = _cached_network_havens(network)
    adj: dict[str, list[tuple[str, int]]] = {nid: [] for nid in network.nodes}
    for e in network.edges.values():
        step_t = max(1, int(round(float(getattr(e, "length", 1.0)) / max(0.2, float(getattr(e, "speed_limit", 1.0))))))
        adj.setdefault(str(e.src), []).append((str(e.dst), min(6, step_t)))

    repairs: dict[str, dict[str, Any]] = {}
    claimed_targets: set[str] = set()

    for aid in idle_blocker_agvs[:2]:
        if aid in faulted_robots or aid in transit:
            continue
        agv = agv_by_id.get(aid)
        if agv is None or agv.cur_task is not None:
            continue
        start_nd = str(agv.cur_node or agv.start_node)
        if start_nd not in adj:
            continue
        q: deque[tuple[str, list[tuple[str, str, int]]]] = deque([(start_nd, [])])
        visited = {start_nd}
        found_path: Optional[tuple[str, list[tuple[str, str, int]]]] = None
        while q:
            u, path = q.popleft()
            if path and u not in occupied_nodes and u not in needed_nodes and u not in claimed_targets:
                found_path = (u, path)
                break
            if len(path) >= 4:
                continue
            nbrs = sorted(adj.get(u, ()), key=lambda x: (0 if x[0] in havens.safe_havens else 1, x[1], x[0]))
            for v, dt in nbrs:
                if v not in visited and v not in faulted_nodes:
                    visited.add(v)
                    q.append((v, path + [(u, v, dt)]))
        if found_path is None:
            for v, dt in adj.get(start_nd, ()):
                if v not in occupied_nodes and v not in claimed_targets and v not in faulted_nodes:
                    found_path = (v, [(start_nd, v, dt)])
                    break
        if found_path is not None:
            goal_nd, edges_p = found_path
            cur_t = tick
            hops: list[tuple[str, str, int, int]] = []
            valid_h = True
            for u, v, dt in edges_p:
                if not network.has_edge(u, v) or cur_t + dt >= max_steps - 2:
                    valid_h = False
                    break
                hops.append((u, v, cur_t, cur_t + dt))
                cur_t += dt
            if valid_h and hops:
                claimed_targets.add(goal_nd)
                occupied_nodes.add(goal_nd)
                repairs[aid] = {
                    "kind": "idle_park",
                    "goal": goal_nd,
                    "hops": hops,
                    "min_reserve_hops": len(hops),
                }
    return repairs


def build_critical_bottleneck_locks(
    snapshot: NativeSnapshot,
    network: Network,
    stalled_agvs: list[str],
    faulted_robots: set[str],
) -> dict[str, list[tuple[str, int, int]]]:
    """Emit short, transaction-legal corridor passage locks on choke junctions for stalled healthy AGVs."""
    if not stalled_agvs:
        return {}
    tick = int(snapshot.tick)
    havens = _cached_network_havens(network)
    r_by_id = {r.robot_id: r for r in snapshot.robots}
    locks: dict[str, list[tuple[str, int, int]]] = {}
    for idx, aid in enumerate(stalled_agvs[:2]):
        if aid in faulted_robots:
            continue
        r = r_by_id.get(aid)
        if r is None or not r.route:
            continue
        for nd in r.route[1:4]:
            if nd in havens.choke_nodes and nd in network.nodes:
                lo = tick + 1 + idx * 3
                hi = lo + 4
                locks[aid] = [(str(nd), lo, hi)]
                break
    return locks


__all__ = [
    "PDG2Config",
    "PDG2LivenessMonitor",
    "PDG2SCCLivenessMonitor",
    "PDGLivenessTracker",
    "PersistentEscapeMemory",
    "RecoveryMemoryEntry",
    "build_critical_bottleneck_locks",
    "build_cross_domain_wait_graph",
    "compute_pdg_state",
    "plan_idle_blocker_evacuations",
]
