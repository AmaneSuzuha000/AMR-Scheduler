"""Direction A Core: Execution-Aware Local Reservation Repair (`EALR`) for Lifelong Multi-AGV Coordination.

Links directly to CPS-A* Section 6 Future Work (service-time uncertainty, heterogeneous vehicle
dynamics, communication delays) without expanding into complex FMS production scheduling:
  - Mechanism 1 (`detect_reservation_mismatches`): identifies active AGVs whose observed physical
    state/progress drifts from planned spatiotemporal reservations beyond allowed comm lag.
  - Mechanism 2 (`propagate_reservation_dependencies`): expands forward reservation dependency
    graph across shared vertices & bidirectional corridors over `[t, t + prop_window]` to isolate
    the minimal affected subset `R_loc` (`|R_loc| << |I|`), leaving unrelated AGVs untouched.
  - Mechanism 3 (`local_reservation_repair`): preserves physical pins (`transfer_pins`) and
    immutable comm-lag prefix `[t, t + comm_lag_steps]`, incrementally updates the CPS-A*
    congestion field Phi, replans `R_loc` via CPS-A* (or safe fallback hold), and enforces zero
    physical collisions.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Any, Optional

from amr_sched.cvcr.evaluator import physical_edge_id
from amr_sched.network import Network
from amr_sched.reservation import Interval, ReservationTable
from experiments.cps_astar_routing.cps_astar_graph import (
    CPSAStarConfig,
    CPSAStarGraphPlanner,
    GraphCongestionField,
)
from experiments.track_r_cpsa.cpsa_pure_runner import (
    expand_planned_to_step_positions,
    reserve_mapf_path,
    shortest_dist_steps,
)


def _fast_reserve_path(
    res_table: Any,
    planner: Any,
    network: Network,
    agv_id: str,
    step_positions: list[tuple[str, int]],
    *,
    park_until: int | None = None,
) -> None:
    """Use native single-call Rust indexed reservation when running E2E zero-shadow tables."""
    reserve_fn = getattr(planner, "reserve_step_points_indexed", None)
    if reserve_fn is not None and not getattr(res_table, "_track_shadow", True):
        reserve_fn(res_table, agv_id, step_positions, park_until=park_until)
        return
    reserve_mapf_path(res_table, agv_id, step_positions, park_until=park_until)


@dataclass(frozen=True)
class EALRConfig:
    """Hyperparameters for Execution-Aware Local Reservation Repair (`EALR v1` & migrated `v2`)."""

    prop_window_steps: int = 24
    prop_max_depth: int = 2
    max_repair_agents: int = 12
    comm_lag_steps: int = 0
    local_rollback_limit: int = 3
    replan_horizon: int = 100
    prio_age_tau: float = 0.25
    prio_cong_kappa: float = 4.0
    # Prior-research direct migrations (Time-Shift Absorption + P4 Belief + Choke Arbitration)
    enable_v2_migrations: bool = True
    enable_belief_gating: bool = True
    enable_cone_pull: bool = False
    enable_siding_yield: bool = False
    enable_choke_arbitration: bool = True
    max_cone_pull: int = 2


class DelayDurationBeliefTracker:
    """Strictly causal Empirical-Bayes Weibull + Kaplan-Meier delay tracker (from `p4_belief_choke`)."""

    def __init__(self, *, prior_scale: float = 2.0, prior_shape: float = 2.1, max_horizon: int = 24) -> None:
        self.prior_scale = max(1.0, float(prior_scale))
        self.prior_shape = max(1.0, float(prior_shape))
        self.max_horizon = max(6, int(max_horizon))
        self._active_starts: dict[str, int] = {}
        self._completed_durations: list[int] = []
        self._last_tick: int = -1

    def observe_step(self, tick: int, delayed_agvs: set[str]) -> None:
        s = int(tick)
        if s < self._last_tick:
            return
        for aid in list(self._active_starts.keys()):
            if aid not in delayed_agvs:
                dur = max(1, s - int(self._active_starts.pop(aid)))
                self._completed_durations.append(dur)
        for aid in delayed_agvs:
            if aid not in self._active_starts:
                self._active_starts[aid] = s
        self._last_tick = s

    def _posterior_scale(self) -> float:
        if not self._completed_durations:
            return self.prior_scale
        w_sum, d_sum, decay = 0.0, 0.0, 0.80
        for age, d in enumerate(reversed(self._completed_durations[-32:])):
            w = decay ** float(age)
            w_sum += w
            d_sum += w * float(max(1, int(d)))
        return max(1.2, (0.5 * self.prior_scale + d_sum) / (0.5 + w_sum))

    def _survival(self, t: float) -> float:
        if t <= 0.0:
            return 1.0
        sc = self._posterior_scale()
        p_w = math.exp(-((float(t) / sc) ** self.prior_shape))
        n_done = len(self._completed_durations)
        if n_done <= 0:
            return p_w
        n_abv = sum(1 for d in self._completed_durations if float(d) > t)
        s_emp = (float(n_abv) + p_w * 2.0) / (float(n_done) + 2.0)
        w_emp = min(0.55, float(n_done) / (float(n_done) + 2.0))
        return (1.0 - w_emp) * p_w + w_emp * s_emp

    def expected_remaining(self, agv_id: str, tick: int) -> float:
        elapsed = max(0, int(tick) - int(self._active_starts.get(agv_id, tick)))
        s_e = max(1e-6, self._survival(float(elapsed)))
        s_prev = s_e
        mu_rem = 0.0
        for r in range(1, self.max_horizon):
            s_cur = min(s_prev, self._survival(float(elapsed + r)))
            prob = max(0.0, (s_prev - s_cur) / s_e)
            mu_rem += float(r) * prob
            s_prev = s_cur
        mu_rem += float(self.max_horizon) * max(0.0, s_prev / s_e)
        return mu_rem


@dataclass(frozen=True)
class MismatchSignal:
    """One detected mismatch between actual/delayed vehicle execution and planned reservations."""

    agv_id: str
    obs_node: str
    expected_node: str
    estimated_delay_steps: int
    reason: str  # "node_mismatch" | "active_stall" | "service_delay"


def detect_reservation_mismatches(
    agv_ids: list[str],
    obs_pos: dict[str, str],
    obs_delay_remaining: dict[str, int],
    obs_service_remaining: dict[str, int],
    plan_steps: dict[str, dict[int, str]],
    plan_end_t: dict[str, int],
    goals: dict[str, Optional[str]],
    *,
    tick: int,
) -> dict[str, MismatchSignal]:
    """Mechanism 1: Detect vehicles whose observed position or ongoing delay invalidates their plan."""
    mismatches: dict[str, MismatchSignal] = {}
    for aid in agv_ids:
        cur_p = obs_pos[aid]
        exp_p = plan_steps[aid].get(tick, cur_p)
        rem_d = int(obs_delay_remaining.get(aid, 0))
        rem_s = int(obs_service_remaining.get(aid, 0))
        gl = goals.get(aid)
        if cur_p != exp_p:
            # Expected to have moved elsewhere by `tick`, but delayed on `cur_p`
            delay_est = max(1, rem_d + rem_s)
            mismatches[aid] = MismatchSignal(aid, cur_p, exp_p, delay_est, "node_mismatch")
        elif gl is not None and (rem_d > 0 or rem_s > 0):
            # Still dwelling/slowed while plan intends to move within next delay steps
            next_scheduled_move = any(
                plan_steps[aid].get(tick + k, cur_p) != cur_p
                for k in range(1, max(2, rem_d + rem_s + 1))
            )
            if next_scheduled_move or tick < plan_end_t.get(aid, tick):
                mismatches[aid] = MismatchSignal(
                    aid, cur_p, exp_p, max(1, rem_d + rem_s),
                    "service_delay" if rem_s > 0 else "active_stall",
                )
    return mismatches


def propagate_reservation_dependencies(
    mismatches: dict[str, MismatchSignal],
    agv_ids: list[str],
    cur_pos: dict[str, str],
    plan_steps: dict[str, dict[int, str]],
    plan_end_t: dict[str, int],
    goals: dict[str, Optional[str]],
    res_table: ReservationTable,
    field: GraphCongestionField,
    network: Network,
    cfg: EALRConfig,
    *,
    tick: int,
    belief_tracker: Optional[DelayDurationBeliefTracker] = None,
) -> list[str]:
    """Mechanism 2: Propagate spatiotemporal reservation conflicts forward to form minimal `R_loc`."""
    if not mismatches:
        return []

    use_belief = bool(cfg.enable_v2_migrations and cfg.enable_belief_gating and belief_tracker is not None)
    max_rem_belief = max(
        (belief_tracker.expected_remaining(aid, tick) for aid in mismatches),
        default=float(cfg.prop_window_steps),
    ) if use_belief else float(cfg.prop_window_steps)
    max_sig_d = max((sig.estimated_delay_steps for sig in mismatches.values()), default=1)
    has_service_sig = any(sig.reason == "service_delay" for sig in mismatches.values())

    # Belief-gated windowing: tight healthy 2-hop window (w=18) for brief jitter, expand to full w=24+ on sustained stalls
    if use_belief and (not has_service_sig) and max_sig_d <= 1 and max_rem_belief <= 2.1:
        eff_window = min(int(cfg.prop_window_steps), 18)
        eff_depth = int(cfg.prop_max_depth)
    else:
        eff_window = max(4, int(cfg.prop_window_steps))
        eff_depth = int(cfg.prop_max_depth)

    w_end = int(tick) + max(4, eff_window)
    r_set: set[str] = set(mismatches.keys())
    eff_cap = int(cfg.max_repair_agents)
    frontier: deque[tuple[str, int]] = deque((aid, 0) for aid in sorted(mismatches))

    # Pre-index intended nodes & corridors over [tick, w_end] per AGV for O(W) conflict detection
    agv_node_times: dict[str, dict[int, str]] = {}
    for aid in agv_ids:
        tmap: dict[int, str] = {}
        last_n = cur_pos[aid]
        pe = max(tick, int(plan_end_t.get(aid, tick)))
        for s in range(tick, w_end + 1):
            last_n = plan_steps[aid].get(s, last_n if s > pe else cur_pos[aid])
            tmap[s] = last_n
        agv_node_times[aid] = tmap

    while frontier and len(r_set) < eff_cap:
        src_aid, depth = frontier.popleft()
        if depth >= eff_depth:
            continue

        sig = mismatches.get(src_aid)
        d_shift = sig.estimated_delay_steps if sig is not None else 1
        src_pos = cur_pos[src_aid]

        # Collect nodes & shifted windows occupied by `src_aid`
        probe_windows: list[tuple[str, int, int]] = [(src_pos, tick, min(w_end, tick + d_shift + 2))]
        # Also include its planned trajectory near-horizon [tick, w_end]
        for s in range(tick, min(w_end, plan_end_t.get(src_aid, tick) + 1)):
            nd_s = agv_node_times[src_aid][s]
            probe_windows.append((nd_s, s, min(w_end + 1, s + d_shift + 2)))

        candidates: dict[str, float] = {}
        for nd, lo, hi in probe_windows:
            if hi <= lo:
                continue
            sp = Interval(lo, hi)
            for other in res_table.node_conflicts(nd, sp, exclude_agv=src_aid):
                if other in r_set or other not in agv_node_times:
                    continue
                if goals.get(other) is None:
                    # Only include stationary idle blocker if directly on shifted path & movable, skip fixed
                    pass
                cong_boost = 1.0 + float(field.slot_Phi(nd, lo))
                candidates[other] = max(candidates.get(other, 0.0), cong_boost)

            # Check adjacent physical corridor conflicts for next steps
            for e in network.out_edges(nd):
                if e.dst not in network.nodes:
                    continue
                for other in res_table.edge_conflicts(e.src, e.dst, sp, exclude_agv=src_aid):
                    if other not in r_set and other in agv_node_times:
                        candidates[other] = max(candidates.get(other, 0.0), 1.5 + float(field.slot_Phi(nd, lo)))
                if network.has_edge(e.dst, e.src):
                    for other in res_table.edge_conflicts(e.dst, e.src, sp, exclude_agv=src_aid):
                        if other not in r_set and other in agv_node_times:
                            candidates[other] = max(candidates.get(other, 0.0), 1.5 + float(field.slot_Phi(nd, lo)))

        # Also catch upstream AGVs directly scheduled to visit `src_pos` within [tick, tick + d_shift + 1]
        for other in agv_ids:
            if other in r_set:
                continue
            for s in range(tick, min(w_end, tick + d_shift + 3)):
                if agv_node_times[other].get(s) == src_pos:
                    candidates[other] = max(candidates.get(other, 0.0), 2.0)

        # v3 Migration (`p4_belief_choke` + `loaded_corridor_repair`): pair opposing corridor traffic
        if cfg.enable_v2_migrations and cfg.enable_choke_arbitration:
            s_lim = min(w_end, tick + 6)
            src_hops: set[tuple[str, str]] = set()
            for s in range(tick, s_lim):
                u_s = agv_node_times[src_aid].get(s, src_pos)
                v_s = agv_node_times[src_aid].get(s + 1, u_s)
                if u_s != v_s and (len(network.out_edges(u_s)) <= 2 or len(network.out_edges(v_s)) <= 2):
                    src_hops.add((u_s, v_s))
            if src_hops:
                for other in agv_ids:
                    if other in r_set or goals.get(other) is None:
                        continue
                    for s in range(tick, s_lim):
                        ou = agv_node_times[other].get(s)
                        ov = agv_node_times[other].get(s + 1)
                        if ou is not None and ov is not None and (ov, ou) in src_hops:
                            candidates[other] = max(candidates.get(other, 0.0), 2.5)
                            break

        for nxt_aid, _ in sorted(candidates.items(), key=lambda kv: (-kv[1], kv[0])):
            if len(r_set) >= eff_cap:
                break
            r_set.add(nxt_aid)
            frontier.append((nxt_aid, depth + 1))

    return sorted(r_set)


def try_time_shift_absorption(
    mismatches: dict[str, MismatchSignal],
    cur_pos: dict[str, str],
    delay_remaining: dict[str, int],
    service_remaining: dict[str, int],
    goals: dict[str, Optional[str]],
    plan_steps: dict[str, dict[int, str]],
    plan_end_t: dict[str, int],
    res_table: ReservationTable,
    field: GraphCongestionField,
    network: Network,
    *,
    tick: int,
    horizon: int,
    comm_lag_steps: int = 0,
) -> tuple[dict[str, MismatchSignal], list[str]]:
    """Zero-expansion temporal shift along existing spatial path before invoking A* cone propagation.

    Only absorbs when `comm_lag_steps == 0` and shifted `(node, edge)` intervals are 100% conflict-free.
    Returns `(remaining_mismatches, absorbed_agv_ids)`.
    """
    if comm_lag_steps > 0 or not mismatches:
        return dict(mismatches), []

    remaining: dict[str, MismatchSignal] = {}
    absorbed: list[str] = []
    for aid in sorted(mismatches.keys()):
        sig = mismatches[aid]
        gl = goals.get(aid)
        if gl is None:
            remaining[aid] = sig
            continue
        hold_d = int(delay_remaining.get(aid, 0)) + int(service_remaining.get(aid, 0))
        hold_d = max(1, hold_d)
        old_end = int(plan_end_t.get(aid, tick))
        if old_end <= tick:
            remaining[aid] = sig
            continue
        # Extract untouched spatial suffix starting from `cur_pos[aid]`
        spatial_seq: list[str] = [cur_pos[aid]]
        started = False
        for s in range(tick, old_end + 1):
            nd = plan_steps[aid].get(s)
            if nd is None:
                continue
            if not started:
                if nd == cur_pos[aid]:
                    started = True
                elif network.has_edge(cur_pos[aid], nd):
                    started = True
                    spatial_seq.append(nd)
                continue
            if nd != spatial_seq[-1]:
                if network.has_edge(spatial_seq[-1], nd):
                    spatial_seq.append(nd)
                else:
                    spatial_seq = []
                    break
        if len(spatial_seq) <= 1 or spatial_seq[-1] != gl:
            remaining[aid] = sig
            continue
        new_end = tick + hold_d + (len(spatial_seq) - 1)
        if new_end >= horizon:
            remaining[aid] = sig
            continue

        # Build shifted trajectory: stay at cur_pos[aid] for [tick..tick+hold_d], then step along spatial_seq
        shifted: list[tuple[str, int]] = [(cur_pos[aid], tick + k) for k in range(hold_d + 1)]
        cur_t = tick + hold_d
        for nd in spatial_seq[1:]:
            cur_t += 1
            shifted.append((nd, cur_t))

        # Verify 100% conflict-free in res_table (excluding aid) before committing
        ok = True
        for i, (u_n, u_t) in enumerate(shifted):
            if not res_table.is_node_free(u_n, Interval(u_t, u_t + 1), network.node(u_n).capacity, exclude_agv=aid):
                ok = False
                break
            if i + 1 < len(shifted):
                v_n, v_t = shifted[i + 1]
                if u_n != v_n:
                    phys = physical_edge_id(u_n, v_n)
                    ecap = (
                        min(network.edge(u_n, v_n).capacity, network.edge(v_n, u_n).capacity)
                        if network.has_edge(v_n, u_n)
                        else network.edge(u_n, v_n).capacity
                    )
                    occ = set(res_table.edge_conflicts(phys[0], phys[1], Interval(u_t, v_t), exclude_agv=aid))
                    if network.has_edge(phys[1], phys[0]) and phys[1] != phys[0]:
                        occ |= set(res_table.edge_conflicts(phys[1], phys[0], Interval(u_t, v_t), exclude_agv=aid))
                    if len(occ) >= ecap:
                        ok = False
                        break
        if ok:
            ln, lt = shifted[-1]
            park_end = min(horizon, lt + 2)
            if park_end > lt + 1 and not res_table.is_node_free(ln, Interval(lt + 1, park_end), network.node(ln).capacity, exclude_agv=aid):
                ok = False
        if not ok:
            remaining[aid] = sig
            continue

        res_table.transfer_pins(aid, [cur_pos[aid]])
        res_table.release_all(aid)
        field.remove_agv(aid)
        res_table.transfer_pins(aid, [])
        reserve_mapf_path(res_table, aid, shifted, park_until=min(horizon, shifted[-1][1] + 2))
        plan_steps[aid] = {st: nd for nd, st in shifted}
        plan_end_t[aid] = shifted[-1][1]
        field.add_agv_path(aid, shifted, horizon=horizon)
        absorbed.append(aid)

    return remaining, absorbed


def _commit_prefix_path(
    aid: str,
    cur_node: str,
    plan_steps: dict[str, dict[int, str]],
    *,
    tick: int,
    prefix_steps: int,
    delay_hold_steps: int,
) -> list[tuple[str, int]]:
    """Build immutable physical prefix `[tick .. tick + prefix_steps]` under comms lag & delay."""
    chain: list[tuple[str, int]] = [(cur_node, tick)]
    n_hold = max(0, int(delay_hold_steps))
    cur = cur_node
    for dt in range(1, max(0, int(prefix_steps)) + 1):
        st = tick + dt
        if dt <= n_hold:
            chain.append((cur, st))
        else:
            nxt = plan_steps[aid].get(st, cur)
            cur = nxt
            chain.append((cur, st))
    return chain


def _build_agv_anchor(
    aid: str,
    cur_node: str,
    delay_remaining: dict[str, int],
    service_remaining: dict[str, int],
    plan_steps: dict[str, dict[int, str]],
    *,
    tick: int,
    comm_lag: int,
    horizon: int,
) -> list[tuple[str, int]]:
    hold_d = int(delay_remaining.get(aid, 0)) + int(service_remaining.get(aid, 0))
    prefix = _commit_prefix_path(
        aid, cur_node, plan_steps, tick=tick, prefix_steps=comm_lag, delay_hold_steps=hold_d
    )
    end_n, end_t = prefix[-1]
    extra_hold = max(0, hold_d - comm_lag)
    for k in range(1, extra_hold + 1):
        if end_t + k <= horizon:
            prefix.append((end_n, end_t + k))
    return prefix


def _find_siding_yield_targets(
    network: Network,
    start_node: str,
    start_step: int,
    res_table: ReservationTable,
    field: GraphCongestionField,
    agv_id: str,
    horizon: int,
    *,
    max_hops: int = 4,
    limit: int = 4,
) -> list[str]:
    """Rank nearby free junction/siding cells (from `loaded_corridor_repair`) to vacate blocked lanes."""
    dist: dict[str, int] = {start_node: 0}
    dq: deque[str] = deque([start_node])
    cands: list[tuple[int, float, int, str]] = []
    while dq:
        u = dq.popleft()
        d = dist[u]
        if d >= max_hops:
            continue
        for e in network.out_edges(u):
            v = e.dst
            if v in dist:
                continue
            nd = d + 1
            dist[v] = nd
            dq.append(v)
            arr_est = min(horizon, start_step + nd)
            park_end = min(horizon, arr_est + 4)
            cap_v = network.node(v).capacity
            if park_end > arr_est and res_table.is_node_free(v, Interval(arr_est, park_end), cap_v, exclude_agv=agv_id):
                deg = len(network.out_edges(v))
                if deg >= 3:
                    cands.append((0, float(field.slot_Phi(v, arr_est)), nd, v))
    cands.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    return [x[3] for x in cands[:limit]]


def local_reservation_repair(
    r_loc: list[str],
    mismatches: dict[str, MismatchSignal],
    agv_ids: list[str],
    cur_pos: dict[str, str],
    delay_remaining: dict[str, int],
    service_remaining: dict[str, int],
    goals: dict[str, Optional[str]],
    task_rel_step: dict[str, int],
    plan_steps: dict[str, dict[int, str]],
    plan_end_t: dict[str, int],
    res_table: ReservationTable,
    field: GraphCongestionField,
    planner: CPSAStarGraphPlanner,
    network: Network,
    cfg: EALRConfig,
    dist_cache: dict[tuple[str, str], int],
    *,
    tick: int,
    horizon: int,
    failed_prev_ids: Optional[set[str]] = None,
) -> dict[str, Any]:
    """Mechanism 3 + Migrated Cone Pull-In & Siding Yield: Pin-protected local replanning over `R_loc`."""
    if not r_loc:
        return {
            "repaired": 0, "expansions": 0, "rollbacks": 0,
            "fallback_holds": 0, "cone_pulls": 0, "siding_yields": 0,
            "repaired_ids": [], "failed_ids": [],
        }

    comm_lag = max(0, int(cfg.comm_lag_steps))
    prev_failed = set(failed_prev_ids or ())
    r_loc_set: set[str] = set(r_loc)
    anchors: dict[str, list[tuple[str, int]]] = {}
    for aid in r_loc:
        prefix = _build_agv_anchor(
            aid, cur_pos[aid], delay_remaining, service_remaining, plan_steps,
            tick=tick, comm_lag=comm_lag, horizon=horizon,
        )
        anchors[aid] = prefix
        res_table.transfer_pins(aid, [cur_pos[aid]])
        res_table.release_all(aid)
        field.remove_agv(aid)
        res_table.transfer_pins(aid, [])
        _fast_reserve_path(res_table, planner, network, aid, prefix, park_until=None)

    def _repair_sort_key(aid: str) -> tuple[int, float, str]:
        gl = goals.get(aid)
        if gl is None:
            return (1, 0.0, aid)
        a_n, a_t = anchors[aid][-1]
        age = float(tick - task_rel_step.get(aid, tick))
        cong = field.slot_Phi(a_n, a_t)
        dist = float(shortest_dist_steps(network, a_n, gl, dist_cache))
        stall_bonus = 6.0 if aid in mismatches else 0.0
        # Arbitrate narrow chokes (`loaded_corridor_repair` rule): cars inside narrow deg<=2 bridges clear first
        choke_prio = 4.0 if (cfg.enable_v2_migrations and cfg.enable_choke_arbitration and len(network.out_edges(a_n)) <= 2) else 0.0
        score = dist - float(cfg.prio_age_tau) * age - float(cfg.prio_cong_kappa) * cong - stall_bonus - choke_prio
        return (0, score, aid)

    pq: deque[str] = deque(sorted(r_loc, key=_repair_sort_key))
    uncommitted: dict[str, list[tuple[str, int]]] = {}
    retries: dict[str, int] = {aid: 0 for aid in r_loc}
    expansions_sum = 0
    rollbacks = 0
    fallback_holds = 0
    cone_pulls = 0
    siding_yields = 0
    repaired_cnt = 0
    failed_ids: list[str] = []
    pulled_budget = max(0, int(cfg.max_cone_pull)) if (cfg.enable_v2_migrations and cfg.enable_cone_pull) else 0
    stall_pull_budget = 1 if (cfg.enable_v2_migrations and pulled_budget == 0) else pulled_budget
    leg_h = min(horizon, tick + int(cfg.replan_horizon))
    repair_max_exp = max(1600, 4 * len(network.nodes))
    streak_max_exp = min(repair_max_exp, 700)
    probe_max_exp = min(repair_max_exp, 1100)
    max_tick_rollbacks = 4

    while pq:
        aid = pq.popleft()
        anchor_pts = anchors[aid]
        a_node, a_step = anchor_pts[-1]
        gl = goals.get(aid)
        aid_max_exp = streak_max_exp if aid in prev_failed else repair_max_exp

        if gl is None or a_step >= leg_h:
            _fast_reserve_path(res_table, planner, network, aid, anchor_pts, park_until=min(horizon, a_step + 15))
            for nd, st in anchor_pts:
                plan_steps[aid][st] = nd
            plan_end_t[aid] = a_step
            uncommitted[aid] = anchor_pts
            repaired_cnt += 1
            continue

        planned, hits = planner.plan_leg(
            a_node, gl, a_step, res_table, field, aid, leg_h,
            goal_park_steps=1, max_expansions_override=aid_max_exp,
        )
        expansions_sum += int(getattr(planner, "last_expansions", 0))

        if planned is not None:
            tail_pts = expand_planned_to_step_positions(a_node, a_step, planned)
            full_pts = anchor_pts[:-1] + tail_pts
            _fast_reserve_path(res_table, planner, network, aid, full_pts, park_until=min(horizon, full_pts[-1][1] + 2))
            plan_steps[aid] = {st: nd for nd, st in full_pts}
            plan_end_t[aid] = full_pts[-1][1]
            uncommitted[aid] = full_pts
            field.add_agv_path(aid, full_pts, horizon=horizon)
            repaired_cnt += 1
            continue

        if retries[aid] < int(cfg.local_rollback_limit) and uncommitted and rollbacks < max_tick_rollbacks:
            # Require actual blocker hit overlap (`hits[j] > 0`) to prevent blind non-blocking thrashing
            active_j = [j for j in uncommitted if j != aid and int(hits.get(j, 0)) > 0]
            if active_j:
                rollbacks += 1
                blocker_j = max(active_j, key=lambda j: (int(hits.get(j, 0)), j))
                saved_j = uncommitted.pop(blocker_j)
                res_table.release_all(blocker_j)
                field.remove_agv(blocker_j)
                _fast_reserve_path(res_table, planner, network, blocker_j, anchors[blocker_j], park_until=None)

                retries[aid] += 1
                p_retry, _ = planner.plan_leg(
                    a_node, gl, a_step, res_table, field, aid, leg_h,
                    goal_park_steps=1, max_expansions_override=probe_max_exp,
                )
                expansions_sum += int(getattr(planner, "last_expansions", 0))
                if p_retry is not None:
                    tail_pts = expand_planned_to_step_positions(a_node, a_step, p_retry)
                    full_pts = anchor_pts[:-1] + tail_pts
                    _fast_reserve_path(res_table, planner, network, aid, full_pts, park_until=min(horizon, full_pts[-1][1] + 2))
                    plan_steps[aid] = {st: nd for nd, st in full_pts}
                    plan_end_t[aid] = full_pts[-1][1]
                    uncommitted[aid] = full_pts
                    field.add_agv_path(aid, full_pts, horizon=horizon)
                    repaired_cnt += 1
                    pq.append(blocker_j)
                    continue
                _fast_reserve_path(res_table, planner, network, blocker_j, saved_j, park_until=min(horizon, saved_j[-1][1] + 2))
                field.add_agv_path(blocker_j, saved_j, horizon=horizon)
                uncommitted[blocker_j] = saved_j

        # Targeted outer blocker pull (`causal_repair_cone`) when enabled OR persistent goal/corridor trap
        cone_rescued = False
        allow_outer_pull = (pulled_budget > 0) or (stall_pull_budget > 0 and aid in prev_failed)
        if allow_outer_pull and len(r_loc_set) < int(cfg.max_repair_agents) + 2:
            outer_cands = [
                k for k in hits
                if k in plan_steps and k not in r_loc_set and goals.get(k) is not None and int(hits.get(k, 0)) > 0
            ]
            outer_cands.sort(key=lambda k: (-int(hits.get(k, 0)), k))
            for cand_k in outer_cands[:1]:
                end_k = max(tick, int(plan_end_t.get(cand_k, tick)))
                saved_pts_k: list[tuple[str, int]] = []
                last_ck = cur_pos[cand_k]
                for s_i in range(tick, end_k + 1):
                    last_ck = plan_steps[cand_k].get(s_i, last_ck)
                    saved_pts_k.append((last_ck, s_i))
                anchor_k = _build_agv_anchor(
                    cand_k, cur_pos[cand_k], delay_remaining, service_remaining, plan_steps,
                    tick=tick, comm_lag=comm_lag, horizon=horizon,
                )
                res_table.transfer_pins(cand_k, [cur_pos[cand_k]])
                res_table.release_all(cand_k)
                field.remove_agv(cand_k)
                res_table.transfer_pins(cand_k, [])
                _fast_reserve_path(res_table, planner, network, cand_k, anchor_k, park_until=None)

                p_cone, _ = planner.plan_leg(
                    a_node, gl, a_step, res_table, field, aid, leg_h,
                    goal_park_steps=1, max_expansions_override=probe_max_exp,
                )
                expansions_sum += int(getattr(planner, "last_expansions", 0))
                if p_cone is not None:
                    tail_pts = expand_planned_to_step_positions(a_node, a_step, p_cone)
                    full_pts = anchor_pts[:-1] + tail_pts
                    _fast_reserve_path(res_table, planner, network, aid, full_pts, park_until=min(horizon, full_pts[-1][1] + 2))
                    plan_steps[aid] = {st: nd for nd, st in full_pts}
                    plan_end_t[aid] = full_pts[-1][1]
                    uncommitted[aid] = full_pts
                    field.add_agv_path(aid, full_pts, horizon=horizon)
                    repaired_cnt += 1
                    r_loc_set.add(cand_k)
                    anchors[cand_k] = anchor_k
                    retries[cand_k] = 0
                    pq.append(cand_k)
                    if pulled_budget > 0:
                        pulled_budget -= 1
                    stall_pull_budget = max(0, stall_pull_budget - 1)
                    cone_pulls += 1
                    cone_rescued = True
                    break
                res_table.release_all(cand_k)
                ln_k, lt_k = saved_pts_k[-1]
                park_k = min(horizon, lt_k + 2)
                if park_k > lt_k + 1 and not res_table.is_node_free(
                    ln_k, Interval(lt_k + 1, park_k), network.node(ln_k).capacity, exclude_agv=cand_k
                ):
                    park_k = lt_k + 1
                _fast_reserve_path(res_table, planner, network, cand_k, saved_pts_k, park_until=park_k)
                field.add_agv_path(cand_k, saved_pts_k, horizon=horizon)

        if cone_rescued:
            continue

        # Migration Pillar 3 (`loaded_corridor_repair`): yield off narrow choke onto nearest free siding
        siding_rescued = False
        if cfg.enable_v2_migrations and cfg.enable_siding_yield and a_step + 4 <= leg_h:
            for siding in _find_siding_yield_targets(network, a_node, a_step, res_table, field, aid, horizon):
                p_sid, _ = planner.plan_leg(
                    a_node, siding, a_step, res_table, field, aid, min(leg_h, a_step + 12),
                    goal_park_steps=2, max_expansions_override=probe_max_exp,
                )
                expansions_sum += int(getattr(planner, "last_expansions", 0))
                if p_sid is not None and p_sid.nodes:
                    tail_pts = expand_planned_to_step_positions(a_node, a_step, p_sid)
                    full_pts = anchor_pts[:-1] + tail_pts
                    _fast_reserve_path(res_table, planner, network, aid, full_pts, park_until=min(horizon, full_pts[-1][1] + 3))
                    plan_steps[aid] = {st: nd for nd, st in full_pts}
                    plan_end_t[aid] = full_pts[-1][1]
                    uncommitted[aid] = full_pts
                    field.add_agv_path(aid, full_pts, horizon=horizon)
                    siding_yields += 1
                    repaired_cnt += 1
                    siding_rescued = True
                    break

        if siding_rescued:
            continue

        fallback_holds += 1
        failed_ids.append(aid)
        cap_a = network.node(a_node).capacity
        park_end = None
        for probe_k in (4, 2, 1):
            cand = min(horizon, a_step + probe_k)
            if cand > a_step and res_table.is_node_free(a_node, Interval(a_step + 1, cand + 1), cap_a, exclude_agv=aid):
                park_end = cand + 1
                break
        _fast_reserve_path(res_table, planner, network, aid, anchor_pts, park_until=park_end)
        plan_steps[aid] = {st: nd for nd, st in anchor_pts}
        plan_end_t[aid] = a_step
        uncommitted[aid] = anchor_pts

    return {
        "repaired": repaired_cnt,
        "expansions": expansions_sum,
        "rollbacks": rollbacks,
        "fallback_holds": fallback_holds,
        "cone_pulls": cone_pulls,
        "siding_yields": siding_yields,
        "repaired_ids": sorted(r_loc_set),
        "failed_ids": failed_ids,
    }
