"""P4 Online Obstacle Duration Belief Estimation & Choke-Robust Multi-AMR Passage Coordination.

Addresses two open scientific gaps observed in the 720-episode P2 Execution-Aware study:
  1. Online Bayesian / Hazard-Rate Obstacle Duration Estimation (`CorridorDurationBeliefModel`)
     with Detour-Cost vs Causal-Wait Regret Gating (`belief_corridor_feedback`), eliminating
     blind fixed rolling horizons (`step + 12`) that cause full-cycle detours near clearance
     while maintaining proactive avoidance when cheap parallel bypasses exist.
  2. Causal Choke-Oscillation Detection & Multi-Hop Bridge/Haven Passage Tokens
     (`detect_choke_oscillation_agvs`, `plan_choke_passage_repairs`), resolving coupled single-lane
     `loaded_yield` ping-pong oscillations (such as West/East choke bridges under Travel Drift)
     even when global triggers are calm-muted or permanently faulted AGVs exist elsewhere.

Strictly Causal Zero-Leakage Guarantee:
  Neither component reads future obstacle clearance instants, future travel realizations, or
  unreleased task schedules. Beliefs update exclusively from past reopened corridors and live elapsed
  dwell times.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import heapq
import math
from typing import Any, Optional

from amr_sched.cvcr.execution_feedback import CorridorFeedback, ExecutionFeedbackPacket
from amr_sched.cvcr.loaded_corridor_repair import (
    _clone_res_from,
    _plan_on_sim_with_weights,
    _release_future_on_clone,
)
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.entities import TaskStatus
from amr_sched.network import Network, NodeKind
from amr_sched.reservation import Interval


def _undir_edge(u: str, v: str) -> tuple[str, str]:
    """Canonicalize undirected corridor key."""
    su, sv = str(u), str(v)
    return (su, sv) if su <= sv else (sv, su)


@dataclass(frozen=True)
class BeliefDurationEstimate:
    """Strictly causal remaining duration posterior summary and barrier decision."""

    edge: tuple[str, str]
    elapsed_steps: int
    expected_remaining: float
    lower_q25: float
    upper_cvar75: float
    p_clear_within_6: float
    detour_overhead_steps: float
    barrier_horizon_steps: int
    recommended_q3_weight: int
    prefer_wait_over_detour: bool


class CorridorDurationBeliefModel:
    """Online Kaplan-Meier / increasing-hazard Weibull Bayesian survival estimator.

    Learns empirical obstacle clearance durations online strictly from corridors that have
    already reopened (`blocked -> open`), blended smoothly with an increasing-hazard unimodal
    workshop blockage prior plus a small chronic heavy-tail mixture component.
    """

    def __init__(
        self,
        *,
        prior_scale: float = 13.5,
        prior_shape: float = 2.6,
        tail_scale: float = 36.0,
        persistent_mix: float = 0.03,
        max_horizon: int = 80,
        enable_voi_probe: bool = False,
    ) -> None:
        """Initialize Bayesian corridor blockage duration belief model."""
        self.prior_scale = max(2.0, float(prior_scale))
        self.prior_shape = max(1.0, float(prior_shape))
        self.tail_scale = max(self.prior_scale, float(tail_scale))
        self.persistent_mix = max(0.0, min(0.4, float(persistent_mix)))
        self.max_horizon = max(20, int(max_horizon))
        self.enable_voi_probe = bool(enable_voi_probe)

        self._active_starts: dict[tuple[str, str], int] = {}
        self._completed_durations: list[int] = []
        self._last_observed_step: int = -1

    def reset(self) -> None:
        """Clear active observation windows and learned historical durations."""
        self._active_starts.clear()
        self._completed_durations.clear()
        self._last_observed_step = -1

    def observe_step(self, step: int, blocked_undir: set[tuple[str, str]]) -> None:
        """Record newly started blockages and commit durations of newly cleared corridors."""
        s = int(step)
        if s < self._last_observed_step:
            return
        cur_set = {_undir_edge(u, v) for (u, v) in blocked_undir}
        for ek in list(self._active_starts.keys()):
            if ek not in cur_set:
                start_s = self._active_starts.pop(ek)
                dur = max(1, s - int(start_s))
                self._completed_durations.append(dur)
        for ek in cur_set:
            if ek not in self._active_starts:
                self._active_starts[ek] = s
        self._last_observed_step = s

    def observe_packet(self, pkt: Optional[ExecutionFeedbackPacket]) -> None:
        """Update survival state from an `ExecutionFeedbackPacket` cleanly."""
        if pkt is None:
            return
        cur_set = {
            _undir_edge(u, v)
            for (u, v), cf in pkt.corridor_updates.items()
            if cf.status == "blocked"
        }
        self.observe_step(int(pkt.tick), cur_set)

    def _posterior_scale(self) -> float:
        """Empirical-Bayes exponentially-weighted scale update blending prior with completed clears."""
        n_done = len(self._completed_durations)
        if n_done <= 0:
            return self.prior_scale
        nu0 = 0.45
        decay = 0.80
        w_sum = 0.0
        d_sum = 0.0
        for age, d in enumerate(reversed(self._completed_durations)):
            w = decay ** float(age)
            w_sum += w
            d_sum += w * float(max(2, int(d)))
        return max(3.5, (nu0 * self.prior_scale + d_sum) / (nu0 + w_sum))

    def _prior_survival(self, t: float, scale_override: Optional[float] = None) -> float:
        """Increasing-hazard Weibull + small exponential chronic tail mixture."""
        if t <= 0.0:
            return 1.0
        scale = self.prior_scale if scale_override is None else max(3.5, float(scale_override))
        s_weib = math.exp(-((float(t) / scale) ** self.prior_shape))
        s_tail = math.exp(-(float(t) / max(scale, self.tail_scale)))
        return (1.0 - self.persistent_mix) * s_weib + self.persistent_mix * s_tail

    def survival(self, t: float) -> float:
        """Return Bayesian survival probability `P(Duration > t)` blending prior and completed clears."""
        if t <= 0.0:
            return 1.0
        lam_post = self._posterior_scale()
        p_scale = self._prior_survival(t, scale_override=lam_post)
        n_done = len(self._completed_durations)
        if n_done <= 0:
            return p_scale
        # Smoothly combine scale-adapted Weibull prior with Kaplan-Meier empirical survival
        n_above = sum(1 for d in self._completed_durations if float(d) > t)
        s_emp = (float(n_above) + p_scale * 2.0) / (float(n_done) + 2.0)
        w_emp = min(0.55, float(n_done) / (float(n_done) + 2.0))
        return (1.0 - w_emp) * p_scale + w_emp * s_emp

    def estimate_remaining(
        self,
        edge: tuple[str, str],
        step: int,
        *,
        detour_overhead_steps: float = 0.0,
    ) -> BeliefDurationEstimate:
        """Compute posterior remaining duration stats and wait-vs-detour barrier for `edge` at `step`."""
        ek = _undir_edge(edge[0], edge[1])
        s = int(step)
        start_s = self._active_starts.get(ek, s)
        elapsed = max(0, s - int(start_s))

        s_e = max(1e-6, self.survival(float(elapsed)))
        pmf: list[float] = []
        s_prev = s_e
        for r in range(1, self.max_horizon):
            s_cur = min(s_prev, self.survival(float(elapsed + r)))
            prob = max(0.0, (s_prev - s_cur) / s_e)
            pmf.append(prob)
            s_prev = s_cur
        # Put any residual tail mass at max_horizon
        tail_mass = max(0.0, 1.0 - sum(pmf))
        pmf.append(tail_mass)

        mu_rem = sum(float(r + 1) * p for r, p in enumerate(pmf))
        cdf = 0.0
        q25 = float(self.max_horizon)
        found_q25 = False
        q75 = float(self.max_horizon)
        found_q75 = False
        p_clear_6 = 0.0
        top_mass = 0.0
        top_sum = 0.0
        for r, p in enumerate(pmf):
            val = float(r + 1)
            cdf += p
            if r < 6:
                p_clear_6 += p
            if not found_q25 and cdf >= 0.25:
                q25 = val
                found_q25 = True
            if not found_q75 and cdf >= 0.75:
                q75 = val
                found_q75 = True
            if cdf > 0.75:
                over = p if (cdf - p) >= 0.75 else (cdf - 0.75)
                top_mass += over
                top_sum += val * over
        cvar75 = (top_sum / top_mass) if top_mass > 1e-6 else q75

        detour_gap = max(0.0, float(detour_overhead_steps))
        lam_post = self._posterior_scale()
        # Continuous CVaR75 risk-adjusted wait-vs-detour pricing:
        #   R_wait = E[rem] + beta * (CVaR_0.75[rem] - E[rem])
        # Compare R_wait directly against physical alternate-path detour overhead (no brittle threshold).
        r_wait = float(mu_rem) + 0.35 * max(0.0, float(cvar75) - float(mu_rem))
        is_chronic_regime = len(self._completed_durations) >= 1 and lam_post > (self.prior_scale * 1.15)
        tau_probe = 0
        if (
            self.enable_voi_probe
            and not is_chronic_regime
            and not math.isinf(detour_gap)
            and detour_gap >= 5.0
        ):
            tau_probe = min(8, max(3, int(round(0.75 * detour_gap))))
            if len(self._completed_durations) >= 2 and lam_post <= 10.0:
                tau_probe = max(tau_probe, min(9, int(round(0.85 * lam_post))))
        allow_continuous_wait = (
            (len(self._completed_durations) >= 1)
            or (elapsed >= (tau_probe if tau_probe > 0 else 10))
        )
        if math.isinf(detour_gap):
            prefer_wait = True
            h_bar = max(4, min(28, int(round(mu_rem))))
            q3_w = 0
        elif tau_probe > 0 and elapsed < tau_probe:
            # Causal VoI early wait-and-observe hold: avoid irreversible full-loop detours before probe expires
            prefer_wait = True
            h_bar = max(2, tau_probe - elapsed)
            q3_w = 0
        elif allow_continuous_wait and detour_gap >= 6.0 and r_wait <= 0.85 * detour_gap:
            prefer_wait = True
            h_bar = max(3, min(20, int(round(mu_rem))))
            q3_w = max(1, int(round(min(14.0, 12.0 * r_wait / max(1.0, detour_gap)))))
        else:
            prefer_wait = False
            use_cvar = is_chronic_regime and elapsed <= 4
            raw_h = float(cvar75) if use_cvar else float(mu_rem)
            h_bar = max(4, min(30, int(round(raw_h))))
            q3_w = 32

        return BeliefDurationEstimate(
            edge=ek,
            elapsed_steps=elapsed,
            expected_remaining=round(mu_rem, 3),
            lower_q25=round(q25, 3),
            upper_cvar75=round(cvar75, 3),
            p_clear_within_6=round(p_clear_6, 4),
            detour_overhead_steps=(9999.0 if math.isinf(detour_gap) else round(detour_gap, 3)),
            barrier_horizon_steps=int(h_bar),
            recommended_q3_weight=int(q3_w),
            prefer_wait_over_detour=prefer_wait,
        )

    def build_corridor_feedback(
        self,
        network: Network,
        u: str,
        v: str,
        step: int,
        *,
        source: str = "p4_belief_lidar",
        reason: str = "dynamic_obstacle",
    ) -> CorridorFeedback:
        """Create a regret-calibrated `CorridorFeedback` entry for directed edge `(u, v)`."""
        oh = corridor_detour_overhead(network, u, v)
        est = self.estimate_remaining((u, v), step, detour_overhead_steps=oh)
        return CorridorFeedback(
            src=str(u),
            dst=str(v),
            status="blocked",
            extra_delay_steps=int(est.barrier_horizon_steps),
            blocked_until_step=int(step) + int(est.barrier_horizon_steps),
            source=source,
            reason=reason,
            q3_penalty_override=int(est.recommended_q3_weight),
        )


_DETOUR_OVERHEAD_CACHE: dict[int, tuple[Network, dict[tuple[str, str], float]]] = {}


def corridor_detour_overhead(network: Network, u: str, v: str) -> float:
    """Compute shortest alternate detour step overhead `(L_bypass - L_direct)` when `{u, v}` closes."""
    if not network.has_edge(u, v) and not network.has_edge(v, u):
        return math.inf
    net_id = id(network)
    entry = _DETOUR_OVERHEAD_CACHE.get(net_id)
    if entry is None or entry[0] is not network:
        if len(_DETOUR_OVERHEAD_CACHE) >= 32:
            _DETOUR_OVERHEAD_CACHE.clear()
        entry = (network, {})
        _DETOUR_OVERHEAD_CACHE[net_id] = entry
    cache = entry[1]
    ek = _undir_edge(u, v)
    if ek in cache:
        return cache[ek]

    def _edge_steps(src: str, dst: str) -> float:
        e = network.edge(src, dst)
        spd = max(0.1, float(e.speed_limit))
        return max(1.0, float(e.length) / spd)

    direct = (
        _edge_steps(u, v)
        if network.has_edge(u, v)
        else _edge_steps(v, u)
    )
    # Dijkstra u -> v without undirected edge ek
    dist: dict[str, float] = {str(u): 0.0}
    pq: list[tuple[float, str]] = [(0.0, str(u))]
    best_alt = math.inf
    while pq:
        d, cur = heapq.heappop(pq)
        if d > dist.get(cur, math.inf) + 1e-9:
            continue
        if cur == v:
            best_alt = d
            break
        for e in network.out_edges(cur):
            if _undir_edge(e.src, e.dst) == ek:
                continue
            nd = d + _edge_steps(e.src, e.dst)
            if nd < dist.get(e.dst, math.inf) - 1e-9:
                dist[e.dst] = nd
                heapq.heappush(pq, (nd, e.dst))
    overhead = max(0.0, best_alt - direct) if math.isfinite(best_alt) else math.inf
    cache[ek] = overhead
    return overhead


def record_node_history_tick(simulator: Any, step: int) -> None:
    """Append current `(step, cur_node, task_id, leg)` observation per non-faulted AGV causally."""
    hist_map: Optional[dict[str, deque[tuple[int, str, Optional[str], Optional[str]]]]] = getattr(
        simulator, "_p4_node_hist", None
    )
    if hist_map is None:
        hist_map = {}
        simulator._p4_node_hist = hist_map
    faulted = set(getattr(simulator, "_faulted_agvs", set()))
    transit = getattr(simulator, "_transit", {})
    jobs = getattr(simulator, "_jobs", {})
    s = int(step)
    for agv in getattr(simulator, "agvs", []):
        aid = agv.agv_id
        if aid in faulted or aid in transit:
            continue
        nd = agv.cur_node or agv.start_node
        if not nd:
            continue
        job = jobs.get(aid)
        tid = job.task_id if job is not None else None
        leg = job.leg if job is not None else None
        dq = hist_map.get(aid)
        if dq is None:
            dq = deque(maxlen=48)
            hist_map[aid] = dq
        if not dq or dq[-1][0] != s:
            dq.append((s, str(nd), tid, leg))


def _choke_corridor_nodes(network: Network) -> set[str]:
    """Return all non-leaf `capacity <= 1` aisle/junction bottleneck nodes."""
    chokes: set[str] = set()
    undir: dict[str, set[str]] = {nid: set() for nid in network.nodes}
    for e in network.edges.values():
        if e.src in undir:
            undir[e.src].add(e.dst)
        if e.dst in undir:
            undir[e.dst].add(e.src)
    for nid, nobj in network.nodes.items():
        if int(nobj.capacity) <= 1 and nobj.kind == NodeKind.AISLE and len(undir.get(nid, ())) >= 2:
            chokes.add(nid)
    return chokes


def predict_and_arbitrate_choke_conflicts(
    simulator: Any,
    feedback: Optional[ExecutionFeedbackPacket],
    ops: Any = (),
    *,
    lookahead: int = 14,
) -> list[str]:
    """Proactively arbitrate opposite-direction single-lane choke arrivals under execution drift.

    Strictly inactive on nominal runs (`feedback is None` or `feedback.is_empty()`),
    returning `[]` so nominal schedules remain zero-diff. When drift or local obstacle slowdowns
    shift arrival windows, yields lower-slack contenders BEFORE they collide inside `cap=1` chokes.
    """
    if feedback is None or feedback.is_empty():
        return []
    network: Network = simulator.network
    chokes = _choke_corridor_nodes(network)
    if not chokes:
        return []

    step = int(getattr(simulator, "_step_cursor", 0))
    faulted = set(getattr(simulator, "_faulted_agvs", set()))
    jobs = getattr(simulator, "_jobs", {})
    tmap = getattr(simulator, "_task_by_id", {})
    last_rep = getattr(simulator, "_last_loaded_repair_at", {})
    fb_states = getattr(feedback, "robot_states", None) or {}
    op_by_id = {getattr(o, "op_id", ""): o for o in ops} if ops else {}

    def _op_slack(tid: str) -> float:
        t = tmap.get(tid)
        if t is None:
            return 1e6
        due = float(t.due_step if t.due_step is not None else 999)
        kit_op = getattr(t, "kit_op_id", None)
        if kit_op and kit_op in op_by_id:
            op_o = op_by_id[kit_op]
            due = min(due, float(op_o.due_step) - float(op_o.proc_steps))
        return due - float(step)

    # Estimate drift-shifted choke entry/exit windows per active non-faulted robot
    # aid -> (choke_nodes_set, directed_choke_edges, t_enter, t_exit, slack, has_drift)
    contenders: dict[str, tuple[set[str], set[tuple[str, str]], float, float, float, bool]] = {}
    for agv in getattr(simulator, "agvs", []):
        aid = agv.agv_id
        if aid in faulted:
            continue
        prev_rep = last_rep.get(aid)
        if prev_rep is not None and (step - int(prev_rep[0])) < 10:
            continue
        job = jobs.get(aid)
        if job is None or not job.task_id:
            continue
        t = tmap.get(job.task_id)
        if t is None or t.status not in (TaskStatus.PICKED, TaskStatus.ASSIGNED):
            continue
        st_fb = fb_states.get(aid)
        spd = max(0.35, float(getattr(st_fb, "speed_scale", 1.0))) if st_fb is not None else 1.0
        eta_d = max(0, int(getattr(st_fb, "eta_drift_steps", 0))) if st_fb is not None else 0
        lp_s = str(getattr(st_fb, "local_planner_state", "nominal")) if st_fb is not None else "nominal"
        is_drifted = (spd < 0.90) or (eta_d > 0) or (lp_s != "nominal")

        c_nodes: set[str] = set()
        c_edges: set[tuple[str, str]] = set()
        t_first: Optional[float] = None
        t_last: float = float(step)
        cur_cursor = float(step + eta_d)
        for h in job.hops[job.idx : job.idx + int(lookahead)]:
            nom_dt = max(1, int(getattr(h, "arrive", 1)) - int(getattr(h, "depart", 0)))
            eff_dt = float(nom_dt) / spd if (spd < 1.0 or eta_d > 0) else float(nom_dt)
            dep_e = cur_cursor
            arr_e = cur_cursor + eff_dt
            cur_cursor = arr_e
            if h.src in chokes or h.dst in chokes:
                if h.src in chokes:
                    c_nodes.add(h.src)
                if h.dst in chokes:
                    c_nodes.add(h.dst)
                c_edges.add((h.src, h.dst))
                if t_first is None:
                    t_first = dep_e
                t_last = max(t_last, arr_e + 1.0)
        if c_nodes and t_first is not None:
            contenders[aid] = (c_nodes, c_edges, t_first, t_last, _op_slack(t.task_id), is_drifted)

    yield_ids: set[str] = set()
    c_ids = sorted(contenders.keys())
    for idx_i, aid in enumerate(c_ids):
        na, ea, sa0, sa1, sla, da = contenders[aid]
        for oid in c_ids[idx_i + 1 :]:
            nb, eb, sb0, sb1, slb, db = contenders[oid]
            if not (da or db):
                continue
            shared_nodes = na & nb
            opposing_edges = any((v, u) in eb for (u, v) in ea)
            if not (shared_nodes or opposing_edges):
                continue
            # Windows overlap or within 3-step arrival safety margin
            if min(sa1, sb1) + 3.0 >= max(sa0, sb0):
                # Yield lower-priority (larger slack) contender cleanly before entering choke
                if (sla, aid) <= (slb, oid):
                    yield_ids.add(oid)
                else:
                    yield_ids.add(aid)
    return sorted(yield_ids)


def detect_choke_oscillation_agvs(
    simulator: Any,
    *,
    window: int = 36,
    min_revisits: int = 2,
) -> list[str]:
    """Detect non-faulted active AGVs caught in 1-lane choke `yield` or back-and-forth ping-pong."""
    network: Network = simulator.network
    chokes = _choke_corridor_nodes(network)
    if not chokes:
        return []
    step = int(getattr(simulator, "_step_cursor", 0))
    faulted = set(getattr(simulator, "_faulted_agvs", set()))
    transit = getattr(simulator, "_transit", {})
    jobs = getattr(simulator, "_jobs", {})
    tmap = getattr(simulator, "_task_by_id", {})
    hist_map = getattr(simulator, "_p4_node_hist", {})
    blocked_since = getattr(simulator, "_blocked_since", {})
    last_rep = getattr(simulator, "_last_loaded_repair_at", {})

    occ_active: dict[str, str] = {}
    for agv in getattr(simulator, "agvs", []):
        aid = agv.agv_id
        if aid in faulted or aid in transit:
            continue
        nd = agv.cur_node or agv.start_node
        if nd:
            occ_active[aid] = str(nd)

    osc: set[str] = set()
    for agv in getattr(simulator, "agvs", []):
        aid = agv.agv_id
        if aid in faulted or aid in transit:
            continue
        here = occ_active.get(aid)
        if not here:
            continue
        prev_rep = last_rep.get(aid)
        if prev_rep is not None and (step - int(prev_rep[0])) < 10:
            continue
        job = jobs.get(aid)
        if job is None or not job.task_id:
            continue
        t = tmap.get(job.task_id)
        if t is None or t.status not in (TaskStatus.PICKED, TaskStatus.ASSIGNED):
            continue

        near_choke = (here in chokes) or any(h.dst in chokes or h.src in chokes for h in job.hops[job.idx : job.idx + 3])
        if not near_choke:
            continue

        # 1) Repeated yield / post-yield choke oscillation on the exact same choke node & task
        #    (single isolated or smooth 2-hop retreat onto a safe haven resolves naturally in nominal runs)
        dq = hist_map.get(aid)
        recent = [row for row in dq if (step - row[0]) <= window and row[2] == job.task_id] if dq else []
        yield_choke_counts: dict[str, int] = {}
        yield_episodes = 0
        prev_is_yield = False
        prev_nd = ""
        for _, nd_r, _, leg_r in recent:
            is_y = leg_r in ("loaded_yield", "empty_yield")
            if is_y and not prev_is_yield:
                yield_episodes += 1
                if nd_r in chokes:
                    yield_choke_counts[nd_r] = yield_choke_counts.get(nd_r, 0) + 1
            elif is_y and nd_r in chokes and nd_r != prev_nd:
                # Track entering an interior choke node while still stuck yielding
                yield_choke_counts[nd_r] = yield_choke_counts.get(nd_r, 0) + 1
            prev_is_yield = is_y
            prev_nd = nd_r
        cur_is_yield = job.leg in ("loaded_yield", "empty_yield")
        if cur_is_yield and not prev_is_yield:
            yield_episodes += 1
            if here in chokes:
                yield_choke_counts[here] = yield_choke_counts.get(here, 0) + 1

        transitions: list[str] = []
        for _, nd_r, _, _ in recent:
            if not transitions or transitions[-1] != nd_r:
                transitions.append(nd_r)
        if not transitions or transitions[-1] != here:
            transitions.append(here)
        choke_counts: dict[str, int] = {}
        for nd_r in transitions:
            if nd_r in chokes:
                choke_counts[nd_r] = choke_counts.get(nd_r, 0) + 1
        max_choke_visits = max(choke_counts.values(), default=0)

        if any(cnt >= 2 for cnt in yield_choke_counts.values()) or yield_episodes >= 3:
            osc.add(aid)
            continue

        # 2) Sustained back-and-forth choke node revisit oscillation (>= 4 visits in window)
        if max_choke_visits >= max(4, int(min_revisits) + 2):
            osc.add(aid)
            continue

        # 3) Sustained head-on lock inside a choke (>= 8 ticks stationary block)
        if here in chokes and aid in blocked_since and (step - int(blocked_since[aid])) >= 8 and job.idx < len(job.hops):
            nxt = job.hops[job.idx].dst
            if nxt in chokes:
                for oid, ond in occ_active.items():
                    if oid != aid and ond == nxt:
                        osc.add(aid)
                        osc.add(oid)

    # 4) Causal execution-drift predictive choke convergence (never fires cleanly on Nominal 1.0x runs)
    fb = getattr(simulator, "_execution_feedback", None)
    fb_states = getattr(fb, "robot_states", None) or {}
    drifted_agvs: set[str] = {
        aid
        for aid, bs in blocked_since.items()
        if (step - int(bs)) >= 7 and occ_active.get(aid) in chokes
    }
    for rid, st_fb in fb_states.items():
        if int(getattr(st_fb, "eta_drift_steps", 0)) > 0 or float(getattr(st_fb, "speed_scale", 1.0)) < 0.90:
            drifted_agvs.add(rid)

    if drifted_agvs:
        active_list = [agv.agv_id for agv in getattr(simulator, "agvs", []) if agv.agv_id in occ_active]
        choke_edges: dict[str, set[tuple[str, str]]] = {}
        for aid in active_list:
            job = jobs.get(aid)
            if job is None or not job.task_id:
                continue
            t = tmap.get(job.task_id)
            if t is None or t.status not in (TaskStatus.PICKED, TaskStatus.ASSIGNED):
                continue
            prev_rep = last_rep.get(aid)
            if prev_rep is not None and (step - int(prev_rep[0])) < 10:
                continue
            ce = {
                (h.src, h.dst)
                for h in job.hops[job.idx : job.idx + 3]
                if (h.src in chokes or h.dst in chokes)
            }
            if ce:
                choke_edges[aid] = ce
        cids = sorted(choke_edges.keys())
        for idx_i, aid in enumerate(cids):
            for oid in cids[idx_i + 1 :]:
                if aid not in drifted_agvs and oid not in drifted_agvs:
                    continue
                ce_a = choke_edges[aid]
                ce_b = choke_edges[oid]
                if any((v, u) in ce_b for (u, v) in ce_a):
                    osc.add(aid)
                    osc.add(oid)

    # 5) Proactive causal drift-window choke passage arbitration (active ONLY under execution feedback)
    if fb is not None and not fb.is_empty():
        proactive_yielders = predict_and_arbitrate_choke_conflicts(simulator, fb, lookahead=14)
        for py_id in proactive_yielders:
            if py_id in occ_active:
                osc.add(py_id)

    return sorted(osc)


def plan_choke_passage_repairs(
    simulator: Any,
    snapshot: NativeSnapshot,
    osc_agvs: list[str],
    coord_res: Any,
    staged_edges: list[tuple[str, str, int, int]],
    base_weights: dict[tuple[str, str], int],
    *,
    max_horizon: int = 45,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    """Arbitrate multi-hop directional Choke Tokens & Safe Haven holds/bypasses over `cap=1` bridges.

    Ensures an oscillating or head-on pair in a narrow choke corridor is separated cleanly:
    either yielding peers take a parallel multi-lane bypass completely off the choke cluster, or
    retreat to `cap >= 2` Safe Havens with full-bridge Leader reservations + `extra_node_locks`
    blocking `Simulator` early-departure until the Leader exits onto `cap >= 2` space.
    """
    repairs: dict[str, dict[str, Any]] = {}
    stats = {"choke_oscillation_repairs": 0, "choke_bypasses": 0, "choke_token_locks": 0}
    if not osc_agvs:
        return repairs, stats

    network: Network = simulator.network
    chokes = _choke_corridor_nodes(network)
    if not chokes:
        return repairs, stats

    step = int(getattr(simulator, "_step_cursor", 0))
    faulted = set(getattr(simulator, "_faulted_agvs", set()))
    transit = getattr(simulator, "_transit", {})
    agv_by_id = {a.agv_id: a for a in getattr(simulator, "agvs", [])}
    jobs = getattr(simulator, "_jobs", {})
    tmap = getattr(simulator, "_task_by_id", {})
    succ_cnt: dict[str, int] = {}
    for op_s in getattr(snapshot, "operations", ()):
        for pid in op_predecessors(op_s):
            succ_cnt[pid] = succ_cnt.get(pid, 0) + 1

    # Expand connected choke component seeded from oscillating AGVs
    cluster_nodes: set[str] = set()
    q: deque[str] = deque()
    for aid in osc_agvs:
        agv = agv_by_id.get(aid)
        if agv is None:
            continue
        here = str(agv.cur_node or agv.start_node or "")
        if here in chokes and here not in cluster_nodes:
            cluster_nodes.add(here)
            q.append(here)
        job = jobs.get(aid)
        if job is not None:
            for h in job.hops[job.idx : job.idx + 4]:
                for nd_h in (h.src, h.dst):
                    if nd_h in chokes and nd_h not in cluster_nodes:
                        cluster_nodes.add(nd_h)
                        q.append(nd_h)
    while q:
        cur = q.popleft()
        for e in network.out_edges(cur):
            if e.dst in chokes and e.dst not in cluster_nodes:
                cluster_nodes.add(e.dst)
                q.append(e.dst)
    if not cluster_nodes:
        return repairs, stats

    # Collect all non-faulted/non-transit active AGVs inside or approaching this choke cluster
    contenders: list[str] = []
    for agv in getattr(simulator, "agvs", []):
        aid = agv.agv_id
        if aid in faulted or aid in transit:
            continue
        job = jobs.get(aid)
        if job is None or not job.task_id:
            continue
        t = tmap.get(job.task_id)
        if t is None or t.status not in (TaskStatus.PICKED, TaskStatus.ASSIGNED):
            continue
        here = str(agv.cur_node or agv.start_node or "")
        in_cluster = here in cluster_nodes
        approaching = any(h.dst in cluster_nodes for h in job.hops[job.idx : job.idx + 3])
        if aid in osc_agvs or in_cluster or approaching:
            contenders.append(aid)
    if len(contenders) < 2 and not any(
        jobs.get(a) is not None and jobs[a].leg in ("loaded_yield", "empty_yield") for a in contenders
    ):
        return repairs, stats

    # Find safe cap>=2 perimeter nodes directly adjacent to cluster_nodes
    haven_exits: set[str] = set()
    for cn in cluster_nodes:
        for e in network.out_edges(cn):
            if e.dst not in cluster_nodes and int(network.node(e.dst).capacity) >= 2:
                haven_exits.add(e.dst)

    def _goal_for(aid: str) -> str:
        job = jobs[aid]
        t = tmap[job.task_id]
        return t.dropoff if t.status is TaskStatus.PICKED else t.pickup

    # Arbitrate single directional Leader (deepest inside cluster -> closest exit / highest priority)
    def _leader_key(aid: str) -> tuple[Any, ...]:
        agv = agv_by_id[aid]
        job = jobs[aid]
        t = tmap[job.task_id]
        here = str(agv.cur_node or agv.start_node or "")
        in_choke = 0 if here in cluster_nodes else 1
        # Nodes closer to bridge exit have an adjacent safe haven or shorter remaining path
        dist_exit = min((simulator._travel_time(here, hv) for hv in haven_exits), default=99.0)
        is_picked = 0 if t.status is TaskStatus.PICKED else 1
        urg = 0 if t.is_urgent else 1
        due = int(t.due_step or 999)
        succ = -int(succ_cnt.get(getattr(t, "kit_op_id", None) or "", 0))
        return (is_picked, in_choke, dist_exit, urg, succ, due, aid)

    ordered = sorted(contenders, key=_leader_key)
    leader_id = ordered[0]
    yielders = [a for a in ordered[1:]]

    for aid in [leader_id] + yielders:
        _release_future_on_clone(simulator, coord_res, aid)

    # First retreat any Yielder physically standing inside `cluster_nodes` onto nearest `cap>=2` Haven
    # (or directly take a choke-free parallel bypass all the way to goal if one exists!)
    w_avoid_choke = dict(base_weights)
    for cn in cluster_nodes:
        for e in network.out_edges(cn):
            w_avoid_choke[(e.src, e.dst)] = 200
            if network.has_edge(e.dst, e.src):
                w_avoid_choke[(e.dst, e.src)] = 200

    yielder_haven_ready: dict[str, tuple[str, int, list[tuple[str, str, int, int]]]] = {}
    for yid in yielders:
        agv_y = agv_by_id[yid]
        here_y = str(agv_y.cur_node or agv_y.start_node or "")
        goal_y = _goal_for(yid)
        t_y = tmap[jobs[yid].task_id]
        # Option 1: full choke-free bypass directly from here_y -> goal_y
        hops_byp = _plan_on_sim_with_weights(
            simulator, agv_y, here_y, goal_y, step, w_avoid_choke,
            res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges,
        )
        if hops_byp and not any(dst in cluster_nodes for _, dst, _, _ in hops_byp):
            kind_y = "picked_reroute" if t_y.status is TaskStatus.PICKED else "assigned_reroute"
            repairs[yid] = {
                "kind": kind_y,
                "task_id": t_y.task_id,
                "goal": goal_y,
                "hops": hops_byp,
                "traffic_weights": w_avoid_choke,
                "traffic_cutoff": step + 60,
                "min_reserve_hops": len(hops_byp),
            }
            for u, v, dep, arr in hops_byp:
                staged_edges.append((u, v, dep, arr))
                coord_res.reserve_edge(u, v, Interval(dep, arr), "__CHOKE_BYPASS__")
            stats["choke_oscillation_repairs"] += 1
            stats["choke_bypasses"] += 1
            continue

        # Option 2: Step out of cluster_nodes into nearest reachable cap>=2 Haven first
        if here_y in cluster_nodes and haven_exits:
            best_ret: Optional[list[tuple[str, str, int, int]]] = None
            best_hv = None
            for hv in sorted(haven_exits):
                h_ret = _plan_on_sim_with_weights(
                    simulator, agv_y, here_y, hv, step, base_weights,
                    res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges,
                )
                if h_ret and (best_ret is None or h_ret[-1][3] < best_ret[-1][3]):
                    best_ret = h_ret
                    best_hv = hv
            if best_ret and best_hv is not None:
                for u, v, dep, arr in best_ret:
                    staged_edges.append((u, v, dep, arr))
                    coord_res.reserve_edge(u, v, Interval(dep, arr), "__CHOKE_RETREAT__")
                    coord_res.reserve_node(v, Interval(arr, arr + 1), "__CHOKE_RETREAT__")
                yielder_haven_ready[yid] = (best_hv, int(best_ret[-1][3]), best_ret)
            else:
                yielder_haven_ready[yid] = (here_y, step, [])
        else:
            yielder_haven_ready[yid] = (here_y, step, [])

    # Now route Leader all the way across & cleanly out of cluster_nodes -> goal_L
    agv_l = agv_by_id[leader_id]
    here_l = str(agv_l.cur_node or agv_l.start_node or "")
    goal_l = _goal_for(leader_id)
    t_l = tmap[jobs[leader_id].task_id]
    hops_l = _plan_on_sim_with_weights(
        simulator, agv_l, here_l, goal_l, step, base_weights,
        res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges,
    )
    t_bridge_clear = step + 12
    bridge_lock_nodes: list[tuple[str, int, int]] = []
    if hops_l:
        last_choke_arr = step
        for u, v, dep, arr in hops_l:
            staged_edges.append((u, v, dep, arr))
            coord_res.reserve_edge(u, v, Interval(dep, arr), "__CHOKE_LEADER__")
            coord_res.reserve_node(v, Interval(arr, arr + 2), "__CHOKE_LEADER__")
            if u in cluster_nodes or v in cluster_nodes:
                last_choke_arr = max(last_choke_arr, int(arr) + 1)
                if v in cluster_nodes:
                    # Only lock node during Leader's actual approach/traversal window so Yielders can vacate beforehand
                    bridge_lock_nodes.append((v, max(step, int(dep)), int(arr) + 2))
        t_bridge_clear = max(t_bridge_clear, last_choke_arr)
        kind_l = "picked_reroute" if t_l.status is TaskStatus.PICKED else "assigned_reroute"
        repairs[leader_id] = {
            "kind": kind_l,
            "task_id": t_l.task_id,
            "goal": goal_l,
            "hops": hops_l,
            "traffic_weights": dict(base_weights),
            "traffic_cutoff": max(step + max_horizon, t_bridge_clear + 10),
            "min_reserve_hops": len(hops_l),
            "extra_node_locks": bridge_lock_nodes,
        }
        stats["choke_oscillation_repairs"] += 1
        stats["choke_token_locks"] += 1

    # Finally plan any Yielder sitting in Haven (bypass or wait until Leader clears bridge)
    for yid, (hv, t_hv, ret_hops) in yielder_haven_ready.items():
        if yid in repairs:
            continue
        agv_y = agv_by_id[yid]
        goal_y = _goal_for(yid)
        t_y = tmap[jobs[yid].task_id]
        # Try bypass first from Haven hv (often via northern bypass lanes!)
        hops_tail = _plan_on_sim_with_weights(
            simulator, agv_y, hv, goal_y, max(t_hv, step), w_avoid_choke,
            res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges,
        )
        used_bypass = bool(hops_tail and not any(dst in cluster_nodes for _, dst, _, _ in hops_tail))
        if not used_bypass:
            # Lock choke cluster nodes until t_bridge_clear so Simulator cannot early-depart Yielder
            lock_res = _clone_res_from(coord_res)
            lock_start = max(t_hv, step)
            lock_end = t_bridge_clear + 1
            if lock_end > lock_start:
                for cn in cluster_nodes:
                    lock_res.reserve_node(cn, Interval(lock_start, lock_end), "__CHOKE_WAIT__")
            hops_tail = _plan_on_sim_with_weights(
                simulator, agv_y, hv, goal_y, max(lock_start, lock_end), base_weights,
                res_table=lock_res, extra_blocked_edges=staged_edges,
            )
        if not hops_tail:
            hops_tail = _plan_on_sim_with_weights(
                simulator, agv_y, hv, goal_y, max(t_hv, step), base_weights,
                res_table=_clone_res_from(coord_res), extra_blocked_edges=staged_edges,
            )
        full_y_hops = list(ret_hops) + list(hops_tail or [])
        if full_y_hops:
            kind_y = "picked_reroute" if t_y.status is TaskStatus.PICKED else "assigned_reroute"
            repairs[yid] = {
                "kind": kind_y,
                "task_id": t_y.task_id,
                "goal": goal_y,
                "hops": full_y_hops,
                "traffic_weights": w_avoid_choke if used_bypass else dict(base_weights),
                "traffic_cutoff": max(step + 60, t_bridge_clear + 20),
                "min_reserve_hops": len(full_y_hops),
            }
            for u, v, dep, arr in full_y_hops:
                staged_edges.append((u, v, dep, arr))
                coord_res.reserve_edge(u, v, Interval(dep, arr), "__CHOKE_YIELDER__")
            stats["choke_oscillation_repairs"] += 1
            if used_bypass:
                stats["choke_bypasses"] += 1

    return repairs, stats
