"""Task 3: Execution-Aware Upper Planners (`U0`, `U1a`, `U1b`, `U2`, `U3`, `U4`) sharing fixed `R1_Robust`."""

from __future__ import annotations

from collections import defaultdict
import heapq
import math
from typing import Any, Callable, Optional

from amr_sched.network import Edge, Network
from amr_sched.planners.base import PlannedPath
from amr_sched.reservation import Interval
from amr_sched.rust_reservation import RustReservationTable
from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.geometry_and_shield import wrap_pi
from experiments.res_guided_avoidance.metric_world import MetricMap
from experiments.stp_research_reset.audit_env import ExecutionAuditSim, _reserve_exact_span

# Shortest Euclidean distance tables cached per `(id(network), goal_node)`
_DIST_HEUR_CACHE: dict[tuple[int, str], dict[str, float]] = {}
N_DIR_BINS = 16


def _rev_dijkstra_dist(net: Network, goal_node: str) -> dict[str, float]:
    key = (id(net), goal_node)
    cached = _DIST_HEUR_CACHE.get(key)
    if cached is not None:
        return cached
    rev: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for e in net.edges.values():
        rev[e.dst].append((e.src, float(e.length)))
    dist: dict[str, float] = {goal_node: 0.0}
    pq: list[tuple[float, str]] = [(0.0, goal_node)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, 1e18) + 1e-9:
            continue
        for p, ln in rev.get(u, ()):
            nd = d + ln
            if nd < dist.get(p, 1e18) - 1e-9:
                dist[p] = nd
                heapq.heappush(pq, (nd, p))
    if len(_DIST_HEUR_CACHE) > 2048:
        _DIST_HEUR_CACHE.clear()
    _DIST_HEUR_CACHE[key] = dist
    return dist


def _dir_bin(ang_rad: float) -> int:
    return int(round(wrap_pi(ang_rad) / (2.0 * math.pi / N_DIR_BINS))) % N_DIR_BINS


def _edge_kinodynamic_ticks(
    mmap: MetricMap,
    edge: Edge,
    prev_ang: Optional[float],
    init_theta: float,
    dt_plan_s: float,
    *,
    is_first_hop: bool,
) -> tuple[int, float, bool]:
    """Return `(traversal_subticks, edge_ang_rad, is_sharp_turn)` calibrated to `R1_Robust` kinematics."""
    net = mmap.network
    nu, nv = net.node(edge.src), net.node(edge.dst)
    edge_ang = math.atan2(nv.y - nu.y, nv.x - nu.x)
    hw = mmap.local_half_width(edge.src, edge.dst)
    # Physical unicycle under m2 uses FOLLOW=0.95 / ACCELERATE=1.25; narrow bays restrict to ~0.85
    v_cruise = 1.05 if hw >= 1.25 else 0.85

    sharp = False
    rot_s = 0.0
    if is_first_hop and prev_ang is None:
        d_ang = abs(wrap_pi(edge_ang - init_theta))
        if d_ang >= 0.75:
            sharp = True
            rot_s = max(0.0, (d_ang - 0.40) / 1.45)
    elif prev_ang is not None:
        d_ang = abs(wrap_pi(edge_ang - prev_ang))
        if d_ang >= 0.70:
            sharp = True
            rot_s = 0.65 * d_ang / 1.40

    dur_s = (float(edge.length) / v_cruise) + rot_s
    tr_ticks = max(1, int(math.ceil(dur_s / max(0.1, dt_plan_s) - 0.15)))
    return tr_ticks, edge_ang, sharp


def _build_occupancy_counts(
    sim: ExecutionAuditSim,
    exclude_rid: str,
) -> dict[str, list[tuple[int, int]]]:
    """Build lightweight node occupancy intervals from active plans for slot-congestion scoring."""
    occ: dict[str, list[tuple[int, int]]] = defaultdict(list)
    div = sim._sub_tick_div
    for oid, (sn, st, pl, bt) in sim._active_leg_plans.items():
        if oid == exclude_rid:
            continue
        cur_t = int(st)
        occ[sn].append((cur_t, cur_t + max(1, div // 2)))
        for nd, arr_t in zip(pl.nodes, pl.arrival_steps):
            arr_i = int(arr_t)
            occ[nd].append((max(cur_t, arr_i - 1), arr_i + max(1, div // 2) + int(bt)))
            cur_t = arr_i
    return occ


def _slot_pressure(occ: dict[str, list[tuple[int, int]]], node_id: str, t0: int, t1: int) -> float:
    hits = 0.0
    for a, b in occ.get(node_id, ()):
        if a <= t1 and t0 <= b:
            hits += 1.0
    return hits


def plan_kinodynamic_spacetime_astar(
    sim: ExecutionAuditSim,
    rid: str,
    goal_node: str,
    t_plan_start: int,
    *,
    res_table: RustReservationTable | None = None,
    gamma: float = 4.0,
    turn_cost_s: float = 0.5,
    buffer_ticks: int = 0,
    max_expansions: int = 12_000,
) -> PlannedPath | None:
    """Sub-macro kinodynamic space-time A* accounting for heading align, turn delay & full edge span."""
    mmap = sim.mmap
    net = mmap.network
    rb = sim.robots[rid]
    start_node = rb.cur_node
    dt_plan_s = sim.plan_dt_s
    div = sim._sub_tick_div
    rt = res_table if res_table is not None else sim.active_res_table

    if start_node == goal_node:
        return PlannedPath(nodes=[goal_node], arrival_steps=[t_plan_start + 1], wait_steps=1)

    h_dist = _rev_dijkstra_dist(net, goal_node)
    if start_node not in h_dist:
        return None

    occ = _build_occupancy_counts(sim, exclude_rid=rid) if gamma > 1e-6 else {}
    buf = max(0, int(buffer_ticks))
    hold_ticks = max(1, (div + 1) // 2)
    h_span_ticks = max(int(round(220.0 / dt_plan_s)), int(math.ceil(h_dist[start_node] / 0.35 / dt_plan_s)) + 60)
    max_t = t_plan_start + h_span_ticks
    v_heur = 1.15  # admissible speed upper bound wrt calibrated 1.05 m/s nominal cruise

    # State key: (node_id, dir_bin_or_neg1, t_sub)
    start_key = (start_node, -1, int(t_plan_start))
    best_g: dict[tuple[str, int, int], float] = {start_key: 0.0}
    parent: dict[tuple[str, int, int], tuple[tuple[str, int, int] | None, float | None]] = {start_key: (None, None)}

    h0 = (h_dist[start_node] / v_heur) / dt_plan_s
    # heap entry: (f_ticks, g_ticks, node_id, dir_bin, t_sub, last_ang_or_none)
    pq: list[tuple[float, float, str, int, int, Optional[float]]] = [
        (h0, 0.0, start_node, -1, int(t_plan_start), None)
    ]

    goal_state: Optional[tuple[str, int, int]] = None
    expansions = 0
    while pq and expansions < max_expansions:
        f_cur, g_cur, u, d_bin, t_cur, prev_ang = heapq.heappop(pq)
        st_key = (u, d_bin, t_cur)
        if g_cur > best_g.get(st_key, 1e18) + 1e-9:
            continue
        if u == goal_node:
            # Verify destination park interval `[t_cur, t_cur + hold_ticks + buf)` free
            g_cap = net.node(goal_node).capacity
            if rt.is_node_free(goal_node, Interval(t_cur, t_cur + hold_ticks + buf), g_cap, exclude_agv=rid):
                goal_state = st_key
                break

        expansions += 1
        # Wait 1 sub-tick at current node u
        t_wait = t_cur + 1
        if t_wait <= max_t:
            u_cap = net.node(u).capacity
            if rt.is_node_free(u, Interval(t_cur, t_wait + 1), u_cap, exclude_agv=rid):
                phi = _slot_pressure(occ, u, t_cur, t_wait) if occ else 0.0
                g_w = g_cur + 1.05 + (gamma * 0.25 * phi)
                wk = (u, d_bin, t_wait)
                if g_w < best_g.get(wk, 1e18) - 1e-9:
                    best_g[wk] = g_w
                    parent[wk] = (st_key, prev_ang)
                    hw_est = (h_dist.get(u, 0.0) / v_heur) / dt_plan_s
                    heapq.heappush(pq, (g_w + hw_est, g_w, u, d_bin, t_wait, prev_ang))

        # Traverse each outgoing edge u -> v
        is_first = prev_ang is None
        for edge in net.out_edges(u):
            v = edge.dst
            if v not in h_dist:
                continue
            tr_ticks, e_ang, is_sharp = _edge_kinodynamic_ticks(
                mmap, edge, prev_ang, rb.theta, dt_plan_s, is_first_hop=is_first
            )
            arr_t = t_cur + tr_ticks
            if arr_t + hold_ticks + buf > max_t:
                continue
            e_span = Interval(t_cur, max(t_cur + 1, arr_t + buf))
            if not rt.is_edge_free(u, v, e_span, edge.capacity, exclude_agv=rid):
                continue
            if not rt.is_opposing_edge_free(u, v, e_span, exclude_agv=rid):
                continue
            v_cap = net.node(v).capacity
            n_span = Interval(arr_t, arr_t + hold_ticks + buf)
            if not rt.is_node_free(v, n_span, v_cap, exclude_agv=rid):
                continue

            phi = _slot_pressure(occ, v, t_cur, arr_t + buf) if occ else 0.0
            turn_pen = (turn_cost_s / dt_plan_s) if is_sharp else 0.0
            g_nxt = g_cur + float(tr_ticks) + turn_pen + (gamma * 0.25 * phi)
            nb = _dir_bin(e_ang)
            nk = (v, nb, arr_t)
            if g_nxt < best_g.get(nk, 1e18) - 1e-9:
                best_g[nk] = g_nxt
                parent[nk] = (st_key, e_ang)
                hn = (h_dist[v] / v_heur) / dt_plan_s
                heapq.heappush(pq, (g_nxt + hn, g_nxt, v, nb, arr_t, e_ang))

    if goal_state is None:
        return None

    # Reconstruct PlannedPath (nodes after start_node, matching BasePlannedPath format)
    chain: list[tuple[str, int]] = []
    cur: Optional[tuple[str, int, int]] = goal_state
    while cur is not None:
        chain.append((cur[0], cur[2]))
        cur = parent[cur][0]
    chain.reverse()
    nodes: list[str] = []
    arrivals: list[int] = []
    waits = 0
    for k in range(1, len(chain)):
        nd, tk = chain[k]
        if nd == chain[k - 1][0]:
            waits += 1
        nodes.append(nd)
        arrivals.append(tk)
    if not nodes:
        nodes = [goal_node]
        arrivals = [t_plan_start + 1]
    return PlannedPath(nodes=nodes, arrival_steps=arrivals, wait_steps=waits)


def u1a_plan_fn(sim: ExecutionAuditSim, rid: str, goal_node: str, t_plan_tick: int) -> tuple[PlannedPath | None, int]:
    """`U1a_EdgeSpan_Fix`: original Rust CPS-A* planner, fixing only multi-tick edge span reservation."""
    pl, _ = sim.planner.plan_leg(
        sim.robots[rid].cur_node,
        goal_node,
        t_plan_tick,
        sim.active_res_table,
        sim.field,
        rid,
        min(sim.plan_horizon_ticks, t_plan_tick + 120),
        goal_park_steps=1,
    )
    if pl is None or not pl.nodes:
        pl, _ = sim.planner.plan_leg(
            sim.robots[rid].cur_node,
            goal_node,
            t_plan_tick + 2,
            sim.active_res_table,
            sim.field,
            rid,
            min(sim.plan_horizon_ticks, t_plan_tick + 122),
            goal_park_steps=1,
        )
    return pl, 0


def u0_buf1_plan_fn(sim: ExecutionAuditSim, rid: str, goal_node: str, t_plan_tick: int) -> tuple[PlannedPath | None, int]:
    """Factorial isolation: original Rust CPS-A* + exact edge span + 1 macro-tick post-arrival buffer."""
    pl, _ = sim.planner.plan_leg(
        sim.robots[rid].cur_node,
        goal_node,
        t_plan_tick,
        sim.active_res_table,
        sim.field,
        rid,
        min(sim.plan_horizon_ticks, t_plan_tick + 120),
        goal_park_steps=1,
    )
    return pl, 1


def make_u1b_plan_fn(*, gamma: float = 4.0, turn_cost_s: float = 0.5) -> Callable[[ExecutionAuditSim, str, str, int], tuple[PlannedPath | None, int]]:
    def _fn(sim: ExecutionAuditSim, rid: str, goal_node: str, t_plan_tick: int) -> tuple[PlannedPath | None, int]:
        pl = plan_kinodynamic_spacetime_astar(
            sim, rid, goal_node, t_plan_tick, gamma=gamma, turn_cost_s=turn_cost_s, buffer_ticks=0
        )
        if pl is None:
            pl = plan_kinodynamic_spacetime_astar(
                sim,
                rid,
                goal_node,
                t_plan_tick,
                res_table=RustReservationTable.fast(),
                gamma=gamma,
                turn_cost_s=turn_cost_s,
                buffer_ticks=0,
            )
        return pl, 0

    return _fn


def make_u2_fixed_buffer_fn(buffer_s: float) -> Callable[[ExecutionAuditSim, str, str, int], tuple[PlannedPath | None, int]]:
    def _fn(sim: ExecutionAuditSim, rid: str, goal_node: str, t_plan_tick: int) -> tuple[PlannedPath | None, int]:
        buf_ticks = max(1, int(round(buffer_s / sim.plan_dt_s)))
        pl = plan_kinodynamic_spacetime_astar(
            sim, rid, goal_node, t_plan_tick, gamma=4.0, turn_cost_s=0.5, buffer_ticks=buf_ticks
        )
        if pl is None:
            # Relax buffer before dropping table reservation
            pl = plan_kinodynamic_spacetime_astar(
                sim, rid, goal_node, t_plan_tick, gamma=4.0, turn_cost_s=0.5, buffer_ticks=0
            )
            return pl, 0
        return pl, buf_ticks

    return _fn


def u3_state_dep_buffer_fn(sim: ExecutionAuditSim, rid: str, goal_node: str, t_plan_tick: int) -> tuple[PlannedPath | None, int]:
    """`U3_State_Dep_Res_Buffer`: deterministic route-turn / narrow-corridor / congestion / noise buffer."""
    dt_plan_s = sim.plan_dt_s
    rb = sim.robots[rid]
    base_pl = plan_kinodynamic_spacetime_astar(
        sim, rid, goal_node, t_plan_tick, gamma=4.0, turn_cost_s=0.5, buffer_ticks=0
    )
    if base_pl is None or not base_pl.nodes:
        base_pl = plan_kinodynamic_spacetime_astar(
            sim, rid, goal_node, t_plan_tick, res_table=RustReservationTable.fast(), gamma=4.0, turn_cost_s=0.5, buffer_ticks=0
        )
        return base_pl, 0

    # Extract unique waypoint chain for geometric/congestion features
    seq = [rb.cur_node]
    for nd in base_pl.nodes:
        if seq[-1] != nd:
            seq.append(nd)
    net = sim.mmap.network
    turns = 0
    narrow_cnt = 0
    edges_cnt = max(1, len(seq) - 1)
    for u, v in zip(seq[:-1], seq[1:]):
        if sim.mmap.local_half_width(u, v) <= 1.30:
            narrow_cnt += 1
    init_err = 0.0
    if len(seq) >= 2:
        n0, n1 = net.node(seq[0]), net.node(seq[1])
        init_err = abs(wrap_pi(math.atan2(n1.y - n0.y, n1.x - n0.x) - rb.theta))
    for a, b, c in zip(seq[:-2], seq[1:-1], seq[2:]):
        na, nb, nc = net.node(a), net.node(b), net.node(c)
        dx1, dy1 = nb.x - na.x, nb.y - na.y
        dx2, dy2 = nc.x - nb.x, nc.y - nb.y
        l1, l2 = math.hypot(dx1, dy1), math.hypot(dx2, dy2)
        if l1 > 1e-6 and l2 > 1e-6 and ((dx1 * dx2 + dy1 * dy2) / (l1 * l2)) < 0.707:
            turns += 1

    occ = _build_occupancy_counts(sim, exclude_rid=rid)
    occ_hits = 0.0
    for nd, arr_t in zip(base_pl.nodes[:6], base_pl.arrival_steps[:6]):
        occ_hits += _slot_pressure(occ, nd, int(arr_t) - 2, int(arr_t) + 2)

    buf_s = 0.0
    buf_s += 0.30 * min(3, turns)
    if init_err >= 0.785:
        buf_s += 0.35
    if (narrow_cnt / edges_cnt) > 0.5:
        buf_s += 0.35
    buf_s += min(0.70, 0.25 * occ_hits)
    if sim.obs_noise_pos_m > 0.05:
        buf_s += 0.65
    if sim.reservation_lag_s > 0.50:
        buf_s += 0.45
    buf_s = min(2.60, buf_s)
    buf_ticks = max(0, int(round(buf_s / dt_plan_s)))
    if buf_ticks == 0:
        return base_pl, 0

    buf_pl = plan_kinodynamic_spacetime_astar(
        sim, rid, goal_node, t_plan_tick, gamma=4.0, turn_cost_s=0.5, buffer_ticks=buf_ticks
    )
    if buf_pl is not None and buf_pl.nodes:
        return buf_pl, buf_ticks
    # If full buffer infeasible under dense reservations, retry half buffer
    half_b = max(1, buf_ticks // 2)
    half_pl = plan_kinodynamic_spacetime_astar(
        sim, rid, goal_node, t_plan_tick, gamma=4.0, turn_cost_s=0.5, buffer_ticks=half_b
    )
    if half_pl is not None and half_pl.nodes:
        return half_pl, half_b
    return base_pl, 0


class U4CounterfactualOracle:
    """Finite-Pool Counterfactual Oracle over upper `(path_variant, res_buffer)` candidates."""

    def __init__(self, rollout_steps: int = 80) -> None:
        self.rollout_steps = int(rollout_steps)
        self.decision_count: int = 0
        self.differed_from_u3_count: int = 0
        self.differed_from_u1b_count: int = 0
        self.chosen_tags: dict[str, int] = defaultdict(int)
        self.oracle_vs_u1b_score_gaps: list[float] = []
        self._u1b_fallback = make_u1b_plan_fn(gamma=4.0, turn_cost_s=0.5)

    @staticmethod
    def _lift_u0_candidate(sim: ExecutionAuditSim, rid: str, goal_node: str, t_plan_tick: int) -> Optional[PlannedPath]:
        div = sim._sub_tick_div
        t_macro = int(t_plan_tick // div)
        empty_rt = RustReservationTable.fast()
        raw_pl, _ = sim.planner.plan_leg(
            sim.robots[rid].cur_node, goal_node, t_macro, empty_rt, sim.planner.create_field(), rid, t_macro + 120, goal_park_steps=1
        )
        if raw_pl is None or not raw_pl.nodes:
            return None
        return PlannedPath(
            nodes=list(raw_pl.nodes),
            arrival_steps=[t_plan_tick + max(1, (int(s) - t_macro) * div) for s in raw_pl.arrival_steps],
            wait_steps=int(raw_pl.wait_steps) * div,
        )

    def _commit_candidate_on_clone(
        self,
        cl: ExecutionAuditSim,
        rid: str,
        start_node: str,
        t_plan_tick: int,
        pl: PlannedPath,
        buf_ticks: int,
    ) -> None:
        div = cl._sub_tick_div
        plan_rt = cl.active_res_table
        plan_rt.release_all(rid)
        cl.field.remove_agv(rid)
        lock_len = 2 * div
        plan_rt.reserve_node(start_node, Interval(t_plan_tick, t_plan_tick + lock_len), rid)
        if div > 1:
            cl._sub_phys_locks[rid] = (start_node, t_plan_tick, t_plan_tick + lock_len)
        pts = _reserve_exact_span(
            plan_rt, cl.mmap, rid, start_node, t_plan_tick, pl, buffer_ticks=buf_ticks, sub_tick_div=div, park_until=None
        )
        cl.field.add_agv_path(rid, pts)
        cl._active_pts[rid] = list(pts)
        cl._active_leg_plans[rid] = (start_node, t_plan_tick, pl, int(buf_ticks))
        cl._active_buf_s[rid] = float(buf_ticks) * cl.plan_dt_s

        rb = cl.robots[rid]
        w_nodes = [start_node]
        w_times_sub = [t_plan_tick]
        for nd_id, st in zip(pl.nodes, pl.arrival_steps):
            if w_nodes[-1] != nd_id:
                w_nodes.append(nd_id)
                w_times_sub.append(int(st))
        plan_s = [float(tk) * cl.plan_dt_s for tk in w_times_sub]
        rb.route_nodes = w_nodes
        rb.route_times = [int(math.ceil(s / cl.mmap.macro_step_s - 1e-9)) for s in plan_s]
        rb.route_idx = 0
        rb.prev_rem_dist_m = cl.remaining_route_dist_m(rb)
        cl._route_times_sec[rid] = plan_s

    def _score_rollout(
        self,
        base_sim: ExecutionAuditSim,
        rid: str,
        start_node: str,
        t_plan_tick: int,
        pl: PlannedPath,
        buf_ticks: int,
    ) -> tuple[float, int]:
        cl = base_sim.clone_snapshot()
        # Inside counterfactual rollouts, future replans use fast deterministic U1b (no recursion)
        cl.upper_mode = "U1b_KinoTime_CPSA"
        cl.custom_plan_fn = self._u1b_fallback
        self._commit_candidate_on_clone(cl, rid, start_node, t_plan_tick, pl, buf_ticks)

        base_comp = base_sim.completed_tasks
        base_col = base_sim.wall_collisions + base_sim.peer_collisions + base_sim.intruder_collisions
        base_waits = base_sim._wait_steps
        base_rem = {r: base_sim.remaining_route_dist_m(base_sim.robots[r]) for r in base_sim.robot_ids}

        for _ in range(self.rollout_steps):
            if cl.step(m2_res_margin_selector, controller_name="U4_CF_Rollout"):
                break

        d_col = (cl.wall_collisions + cl.peer_collisions + cl.intruder_collisions) - base_col
        d_comp = cl.completed_tasks - base_comp
        d_waits = cl._wait_steps - base_waits
        ego_rb = cl.robots[rid]
        ego_rem = cl.remaining_route_dist_m(ego_rb)
        peer_prog = sum(max(0.0, base_rem[r] - cl.remaining_route_dist_m(cl.robots[r])) for r in cl.robot_ids if r != rid)
        ego_prog = max(0.0, cl.remaining_route_dist_m(base_sim.robots[rid]) - ego_rem)
        score = (100.0 * float(d_comp)) + 1.0 * ego_prog + 0.35 * peer_prog - 0.10 * float(d_waits)
        if d_col > 0:
            score -= 10000.0 * float(d_col)
        return float(score), int(d_col)

    def __call__(self, sim: ExecutionAuditSim, rid: str, goal_node: str, t_plan_tick: int) -> tuple[PlannedPath | None, int]:
        rb = sim.robots[rid]
        start_node = rb.cur_node
        b1 = max(1, int(round(1.0 / sim.plan_dt_s)))
        b2 = max(1, int(round(2.0 / sim.plan_dt_s)))

        raw_cands: list[tuple[str, Optional[PlannedPath], int]] = [
            ("U1b_nom", plan_kinodynamic_spacetime_astar(sim, rid, goal_node, t_plan_tick, gamma=4.0, turn_cost_s=0.5, buffer_ticks=0), 0),
            ("U1b_short", plan_kinodynamic_spacetime_astar(sim, rid, goal_node, t_plan_tick, gamma=0.0, turn_cost_s=0.25, buffer_ticks=0), 0),
            ("U1b_lowturn", plan_kinodynamic_spacetime_astar(sim, rid, goal_node, t_plan_tick, gamma=4.0, turn_cost_s=2.5, buffer_ticks=0), 0),
            ("U1b_detour", plan_kinodynamic_spacetime_astar(sim, rid, goal_node, t_plan_tick, gamma=10.0, turn_cost_s=0.5, buffer_ticks=0), 0),
            ("U1b_B1s", plan_kinodynamic_spacetime_astar(sim, rid, goal_node, t_plan_tick, gamma=4.0, turn_cost_s=0.5, buffer_ticks=b1), b1),
            ("U1b_B2s", plan_kinodynamic_spacetime_astar(sim, rid, goal_node, t_plan_tick, gamma=4.0, turn_cost_s=0.5, buffer_ticks=b2), b2),
            ("U0_lift", self._lift_u0_candidate(sim, rid, goal_node, t_plan_tick), 0),
        ]
        u3_pl, u3_b = u3_state_dep_buffer_fn(sim, rid, goal_node, t_plan_tick)
        raw_cands.append(("U3_sdep", u3_pl, u3_b))

        # Deduplicate identical `(nodes, arrival_steps, buffer_ticks)` while preserving order
        seen_sig: set[tuple[tuple[str, ...], tuple[int, ...], int]] = set()
        unique_cands: list[tuple[str, PlannedPath, int]] = []
        for tag, pl, bt in raw_cands:
            if pl is None or not pl.nodes:
                continue
            sig = (tuple(pl.nodes), tuple(int(s) for s in pl.arrival_steps), int(bt))
            if sig not in seen_sig:
                seen_sig.add(sig)
                unique_cands.append((tag, pl, int(bt)))

        if not unique_cands:
            fb, _ = self._u1b_fallback(sim, rid, goal_node, t_plan_tick)
            return fb, 0
        if len(unique_cands) == 1:
            self.decision_count += 1
            self.chosen_tags[unique_cands[0][0]] += 1
            self.oracle_vs_u1b_score_gaps.append(0.0)
            return unique_cands[0][1], unique_cands[0][2]

        scored: list[tuple[float, str, PlannedPath, int]] = []
        u1b_score: Optional[float] = None
        for tag, pl, bt in unique_cands:
            sc, _ = self._score_rollout(sim, rid, start_node, t_plan_tick, pl, bt)
            scored.append((sc, tag, pl, bt))
            if tag == "U1b_nom":
                u1b_score = sc

        scored.sort(key=lambda item: item[0], reverse=True)
        best_sc, best_tag, best_pl, best_bt = scored[0]
        self.decision_count += 1
        self.chosen_tags[best_tag] += 1
        if best_tag != "U1b_nom":
            self.differed_from_u1b_count += 1
        if u3_pl is not None and (tuple(best_pl.nodes), int(best_bt)) != (tuple(u3_pl.nodes), int(u3_b)):
            self.differed_from_u3_count += 1
        if u1b_score is not None:
            self.oracle_vs_u1b_score_gaps.append(max(0.0, best_sc - u1b_score))
        return best_pl, best_bt
