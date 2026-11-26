"""EXP-1 (P0): Structural Progress Dependency Graph + Tarjan SCC Analysis (Petri Net Inspired).

Inspired by Lee & Lin (1995) & Mejía & Montoya (2009) Petri Net blocking analysis:
  1. Builds an explicit cross-domain directed Wait-For Graph over Ops, Kit Tasks, AGVs, Bays, and Nodes.
  2. Runs Tarjan SCC + active physical exit/releasability analysis to separate 5 concrete liveness states:
     - DEADLOCK       (non-trivial SCC with no active machining/transit release exit)
     - LIVELOCK       (repeated physical state signature cycle without forward progress)
     - STARVATION     (frontier critical op stalled with zero SCC)
     - WINDOW_FAILURE (unstarted observable op accumulating closed-window misses)
     - NORMAL_WAITING (explicit active machining/transit exit exists -> hold plan)
  3. Supports 4-way controlled falsification (`S0_Legacy_PDG`, `S1_Local_D1_Only`,
     `S2_WaitGraph_SCC_Only`, `S3_Joint_SCC_Plus_Local`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
import hashlib
import json
from typing import Any, Optional

from amr_sched.cvcr.snapshot import op_predecessors
from experiments.v20_1_closure.pdg2_monitor import PDG2Config, PDG2LivenessMonitor, _stat_lower


def build_cross_domain_wait_graph(sim: Any, tracker: Any, cur_tick: int) -> dict[str, Any]:
    """Build explicit directed Wait-For Graph `u -> v` (u waits on v to release/satisfy precondition)."""
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

    # Bay occupancy by active machining ops
    bay_holder: dict[str, str] = {}
    for op in ops:
        if op_machining[op.op_id]:
            bay_holder[str(op.bay_id)] = op.op_id

    # Node occupancy by AGVs
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

    # Task -> AGV & AGV -> Blocking AGV/Node edges
    for t in tasks:
        st_t = _stat_lower(t)
        if st_t in ("completed", "failed") or int(getattr(t, "release_step", 0)) > cur_tick:
            continue
        u_t = f"TASK:{t.task_id}"
        adj.setdefault(u_t, set())
        if t.assigned_agv:
            _edge(u_t, f"AGV:{t.assigned_agv}")
        # Pickup / dropoff physical occupancy dependency
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
        # Next hop node from active job route or assigned task endpoint
        next_nodes: list[str] = []
        jb = jobs.get(aid)
        if jb is not None:
            hops = getattr(jb, "hops", None) or ()
            for h in hops[:2]:
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

    # Compute exact Tarjan Strongly Connected Components
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

    # Check which SCCs have NO active physical release exit (true structural deadlock vs transient wait)
    deadlock_sccs: list[list[str]] = []
    for comp in sccs:
        comp_set = set(comp)
        has_active_exit = any(n in active_exits for n in comp_set)
        if not has_active_exit:
            deadlock_sccs.append(comp)

    # Compute in-degree / blocking centrality for Cross-Domain ADDRESS (EXP-3)
    in_deg: dict[str, int] = {n: 0 for n in adj}
    for u, nbrs in adj.items():
        for v in nbrs:
            in_deg[v] = in_deg.get(v, 0) + 1

    return {
        "n_nodes": len(adj),
        "n_edges": sum(len(v) for v in adj.values()),
        "sccs": sccs,
        "deadlock_sccs": deadlock_sccs,
        "active_exits": sorted(active_exits),
        "in_degree": in_deg,
        "adj": {k: sorted(v) for k, v in adj.items()},
    }


@dataclass
class StructuralPDGMonitor:
    """Unified EXP-1 Structural Wait-For Graph + Tarjan SCC + Local Progress Classifier."""

    mode: str = "S3_Joint_SCC_Plus_Local"  # S0_Legacy_PDG | S1_Local_D1_Only | S2_WaitGraph_SCC_Only | S3_Joint_SCC_Plus_Local
    scc_persist_ticks: int = 22
    livelock_repeat_threshold: int = 4
    base_monitor: PDG2LivenessMonitor = field(default_factory=PDG2LivenessMonitor)
    scc_streak: int = 0
    recent_state_hashes: deque[str] = field(default_factory=lambda: deque(maxlen=80))
    state_class_counts: dict[str, int] = field(default_factory=dict)
    alert_ticks: list[int] = field(default_factory=list)
    escalation_ticks: list[int] = field(default_factory=list)
    last_escalation_tick: int = -100
    latest_state: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        if self.mode == "S0_Legacy_PDG":
            self.base_monitor = PDG2LivenessMonitor(cfg=PDG2Config(enable_legacy_pdg=True, enable_d1=False, enable_d2=False))
        elif self.mode == "S1_Local_D1_Only":
            self.base_monitor = PDG2LivenessMonitor(cfg=PDG2Config(enable_legacy_pdg=True, enable_d1=True, enable_d2=False))
        elif self.mode == "S2_WaitGraph_SCC_Only":
            self.base_monitor = PDG2LivenessMonitor(cfg=PDG2Config(enable_legacy_pdg=True, enable_d1=False, enable_d2=False))
        else:
            self.base_monitor = PDG2LivenessMonitor(cfg=PDG2Config(enable_legacy_pdg=True, enable_d1=True, enable_d2=True))

    def _physical_state_hash(self, env: Any) -> str:
        agv_sig = sorted((str(a.agv_id), str(a.cur_node), _stat_lower(a), str(a.cur_task)) for a in env.sim.agvs)
        t_done = sum(1 for t in env.sim.tasks if _stat_lower(t) == "completed")
        o_done = sum(1 for o in env.tracker.operations if o.end_step is not None)
        raw = json.dumps({"a": agv_sig, "td": t_done, "od": o_done}, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()[:14]

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
        st = self.base_monitor.observe(env)
        wg = build_cross_domain_wait_graph(env.sim, env.tracker, cur_tick)

        has_deadlock_scc = bool(wg["deadlock_sccs"])
        if has_deadlock_scc:
            self.scc_streak += 1
        else:
            self.scc_streak = 0

        # Livelock detection: identical (AGV poses/states + completion counts) oscillating >= threshold times while stalled
        sh = self._physical_state_hash(env)
        if cur_tick % 5 == 0:
            self.recent_state_hashes.append(sh)
        repeat_cnt = self.recent_state_hashes.count(sh)
        livelock_detected = bool(
            repeat_cnt >= self.livelock_repeat_threshold
            and int(st["zero_progress_ticks"]) >= 120
            and (len(st["stalled_agvs"]) >= 1 or len(st["blocked_healthy_agvs"]) >= 1)
        )

        scc_fire = bool(
            self.mode in ("S2_WaitGraph_SCC_Only", "S3_Joint_SCC_Plus_Local")
            and self.scc_streak >= self.scc_persist_ticks
            and int(st["zero_progress_ticks"]) >= 35
        )
        ll_fire = bool(self.mode in ("S2_WaitGraph_SCC_Only", "S3_Joint_SCC_Plus_Local") and livelock_detected)

        # 5-class Petri-Net / SCC structural liveness classification
        if scc_fire or int(st["wait_cycle_count"]) >= 1:
            liveness_class = "DEADLOCK"
        elif ll_fire:
            liveness_class = "LIVELOCK"
        elif st["d2_fire"] and st["blocking_mode"] == "kitting_window_starvation":
            liveness_class = "WINDOW_FAILURE"
        elif st["d1_fire"] or st["legacy_fire"]:
            liveness_class = "STARVATION"
        else:
            liveness_class = "NORMAL_WAITING"

        self.state_class_counts[liveness_class] = self.state_class_counts.get(liveness_class, 0) + 1

        if self.mode == "S2_WaitGraph_SCC_Only":
            is_alert = bool(st["legacy_fire"] or scc_fire or ll_fire)
        elif self.mode == "S3_Joint_SCC_Plus_Local":
            is_alert = bool(st["pdg2_alert"] or scc_fire or ll_fire)
        else:
            is_alert = bool(st["pdg2_alert"])

        st["pdg_alert"] = is_alert
        st["pdg2_alert"] = is_alert
        st["liveness_class"] = liveness_class
        st["deadlock_sccs"] = wg["deadlock_sccs"]
        st["scc_streak"] = self.scc_streak
        st["wait_graph_in_degree"] = wg["in_degree"]
        if scc_fire:
            st["blocking_mode"] = "corridor_stall"
            # Add AGVs inside deadlock SCCs to wait_cycle_agvs so escape prioritizes breaking the SCC
            scc_agvs = {n.split(":", 1)[1] for comp in wg["deadlock_sccs"] for n in comp if n.startswith("AGV:")}
            st["wait_cycle_agvs"] = sorted(set(st.get("wait_cycle_agvs") or ()) | scc_agvs)

        self.latest_state = st
        if is_alert:
            self.alert_ticks.append(cur_tick)

        if proposer is not None:
            if hasattr(proposer, "set_pdg2_context"):
                proposer.set_pdg2_context(st if is_alert else None, sim_ref=env.sim if is_alert else None)
            elif hasattr(proposer, "set_pdg_state"):
                proposer.set_pdg_state(st if is_alert else None)

        if is_alert and int(base_action) == 0 and (cur_tick - self.last_escalation_tick) >= self.base_monitor.cfg.cooldown_ticks:
            self.last_escalation_tick = cur_tick
            self.escalation_ticks.append(cur_tick)
            mask = info.get("action_mask")
            if liveness_class in ("DEADLOCK", "LIVELOCK") and mask is not None and len(mask) > 1 and int(mask[1]) == 1:
                return 1
            if mask is not None and len(mask) > 2:
                legal = [i for i in range(1, len(mask)) if int(mask[i]) == 1]
                if legal:
                    return int(legal[-1])
            return 1 if liveness_class in ("DEADLOCK", "LIVELOCK") else 2
        return int(base_action)
