"""Controlled Execution-Delay & Uncertainty Lifelong MAPD Simulator (`R2-R3-R4`).

Evaluates four comparators under identical seeded delay trajectories & physical safety pins:
  1. `naive_no_repair`: plans per CPS-A* Lifelong on new segments, no execution-delay repair.
  2. `adg_order_hold`: ADG/TPG-equivalent order-preserving execution guard along fixed spatial paths.
  3. `global_rolling_cpsa`: periodic/triggered full-fleet global CPS-A* replanning beyond comm-lag.
  4. `ealr_local_repair`: Direction A (`Mechanism 1 + 2 + 3` execution-aware local repair + pins).

Zero production-scheduling/DAG/RL assumptions; pure CPS-A* MAPD physics + Section 6 Future Work.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any, Optional

from amr_sched.cvcr.evaluator import physical_edge_id
from amr_sched.network import Network
from amr_sched.planners.rust_spacetime_astar import _HAS_CPS_FFI, RustCPSAStarGraphPlanner
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.rust_reservation import RustReservationTable, is_rust_core_available
from experiments.cps_astar_routing.cps_astar_graph import (
    CPSGraphFieldConfig,
    CPSAStarConfig,
    CPSAStarGraphPlanner,
    GraphCongestionField,
)
from experiments.track_r_cpsa.cpsa_pure_runner import (
    expand_planned_to_step_positions,
    reserve_mapf_path,
    shortest_dist_steps,
    verify_mapf_trace_safety,
)
from experiments.track_r_cpsa.local_reservation_repair import (
    DelayDurationBeliefTracker,
    EALRConfig,
    _fast_reserve_path,
    detect_reservation_mismatches,
    local_reservation_repair,
    propagate_reservation_dependencies,
    try_time_shift_absorption,
)
from experiments.track_r_cpsa.maps_and_workloads import GridMapSpec


@dataclass(frozen=True)
class DelayScenarioConfig:
    """Section 6 Future Work uncertainty parameters."""

    motion_delay_prob: float = 0.0        # per-step probability of a 1-step transit stall
    service_delay_prob: float = 0.0       # probability of extra +1..max_service_delay steps at pickup/dropoff
    max_service_delay: int = 3
    heterogeneous_slow_frac: float = 0.0  # fraction of heavy AGVs taking periodic extra transit stalls
    comm_lag_steps: int = 0               # telemetry age / immutable command prefix steps
    horizon: int = 500


@dataclass(frozen=True)
class DelayedLifelongResult:
    solver: str
    map_name: str
    seed: int
    n_agvs: int
    scenario_tag: str
    completed_tasks: int
    throughput: float
    mean_waiting_time: float
    mean_queue_len: float
    total_expansions: int
    repair_events: int
    avg_repair_fraction: float
    near_miss_blocks: int
    zero_conflicts: bool


def _sample_delay_schedule(
    agv_ids: list[str],
    horizon: int,
    dcfg: DelayScenarioConfig,
    seed: int,
) -> tuple[set[str], dict[tuple[str, int], bool], dict[tuple[str, int], int]]:
    rng = random.Random(int(seed) * 1009 + 17)
    n_slow = int(round(float(dcfg.heterogeneous_slow_frac) * len(agv_ids)))
    slow_agvs = set(rng.sample(agv_ids, min(len(agv_ids), max(0, n_slow))))

    motion_stalls: dict[tuple[str, int], bool] = {}
    service_delays: dict[tuple[str, int], int] = {}
    for aid in agv_ids:
        is_slow = aid in slow_agvs
        for t in range(horizon):
            p = float(dcfg.motion_delay_prob)
            if is_slow and (t % 3 == 1):
                p = max(p, 0.55)
            motion_stalls[(aid, t)] = (rng.random() < p)
            if rng.random() < float(dcfg.service_delay_prob) and int(dcfg.max_service_delay) > 0:
                service_delays[(aid, t)] = rng.randint(1, int(dcfg.max_service_delay))
            else:
                service_delays[(aid, t)] = 0
    return slow_agvs, motion_stalls, service_delays


def _resolve_physical_step(
    agv_ids: list[str],
    cur_pos: dict[str, str],
    desired_next: dict[str, str],
    network: Network,
) -> tuple[dict[str, str], int]:
    """Fixed-point physical safety gate enforcing transfer-pin exclusivity & swap safety."""
    nxt = dict(desired_next)
    blocked_cnt = 0
    for _ in range(len(agv_ids) + 2):
        changed = False
        staying_nodes = {cur_pos[a] for a in agv_ids if nxt[a] == cur_pos[a]}
        # Also detect multiple AGVs claiming same target node or swapping head-on
        claimers: dict[str, list[str]] = {}
        for a in agv_ids:
            if nxt[a] != cur_pos[a]:
                claimers.setdefault(nxt[a], []).append(a)

        for a in agv_ids:
            u = cur_pos[a]
            v = nxt[a]
            if u == v:
                continue
            if not network.has_edge(u, v):
                nxt[a] = u
                blocked_cnt += 1
                changed = True
                continue
            if v in staying_nodes:
                nxt[a] = u
                blocked_cnt += 1
                changed = True
                continue
            # Head-on swap check
            swap_hit = any(nxt[b] == u and cur_pos[b] == v for b in agv_ids if b != a)
            if swap_hit:
                nxt[a] = u
                blocked_cnt += 1
                changed = True
                continue
            # Vertex competition: keep lexicographically smallest AGV if two target same node
            if len(claimers.get(v, [])) > 1 and a != min(claimers[v]):
                nxt[a] = u
                blocked_cnt += 1
                changed = True
        if not changed:
            break
    return nxt, blocked_cnt


def simulate_lifelong_with_delays(
    grid: GridMapSpec,
    init_positions: dict[str, str],
    request_stream: list[tuple[str, str, str]],
    dcfg: DelayScenarioConfig,
    *,
    solver: str = "ealr_local_repair",
    ealr_cfg: EALRConfig = EALRConfig(),
    seed: int = 0,
    scenario_tag: str = "delay_default",
    use_rust_e2e: bool = True,
) -> DelayedLifelongResult:
    """Run delayed Lifelong MAPD under one of the 4 exact comparators."""
    horizon = int(dcfg.horizon)
    plan_h = horizon + 120
    agv_ids = sorted(init_positions.keys())
    _, motion_stalls, service_delays = _sample_delay_schedule(agv_ids, horizon, dcfg, seed)

    cps_cfg = CPSAStarConfig(mode="lifelong", beta=0.5, gamma=5.0, batch_update_K=20, retry_limit_R=5)
    rust_active = bool(use_rust_e2e and is_rust_core_available() and _HAS_CPS_FFI)
    if rust_active:
        planner: Any = RustCPSAStarGraphPlanner(grid.network, cps_cfg, max_speed=1.0, dt=1.0)
        res_table: Any = RustReservationTable.fast()
        field: Any = planner.create_field(max_track_horizon=plan_h + 20, t_start=0)
    else:
        planner = CPSAStarGraphPlanner(grid.network, cps_cfg, max_speed=1.0, dt=1.0)
        res_table = ReservationTable()
        field = GraphCongestionField(grid.network, CPSGraphFieldConfig(max_track_horizon=plan_h + 20), t_start=0)
    dist_cache: dict[tuple[str, str], int] = {}

    pos: dict[str, str] = dict(init_positions)
    phase: dict[str, str] = {a: "idle" for a in agv_ids}
    cur_req: dict[str, Optional[tuple[str, str, str, int]]] = {a: None for a in agv_ids}
    pickup_serviced: dict[str, bool] = {a: False for a in agv_ids}
    task_rel: dict[str, int] = {}
    service_rem: dict[str, int] = {a: 0 for a in agv_ids}
    delay_rem: dict[str, int] = {a: 0 for a in agv_ids}
    fail_streak: dict[str, int] = {a: 0 for a in agv_ids}
    fail_cooldown: dict[str, int] = {a: 0 for a in agv_ids}

    # `spatial_q[a]` holds remaining planned cell sequence for ADG_OrderHold & cursor tracking
    spatial_q: dict[str, deque[str]] = {a: deque() for a in agv_ids}
    plan_steps: dict[str, dict[int, str]] = {a: {0: pos[a]} for a in agv_ids}
    plan_end_t: dict[str, int] = {a: 0 for a in agv_ids}
    traces: dict[str, dict[int, str]] = {a: {0: pos[a]} for a in agv_ids}
    pos_history: dict[str, dict[int, str]] = {a: {0: pos[a]} for a in agv_ids}

    # ADG shared-node passing queue: for each node, ordered deque of AGVs committed to traverse it
    adg_node_order: dict[str, deque[str]] = {}

    for aid in agv_ids:
        res_table.reserve_node(pos[aid], Interval(0, plan_h), aid)

    task_q: deque[tuple[str, str, str]] = deque(request_stream)
    completed = 0
    wait_sum = 0
    q_len_sum = 0
    total_exp = 0
    repair_events = 0
    repair_frac_sum = 0.0
    near_miss_blocks = 0

    eff_cfg = EALRConfig(
        prop_window_steps=ealr_cfg.prop_window_steps,
        prop_max_depth=ealr_cfg.prop_max_depth,
        max_repair_agents=ealr_cfg.max_repair_agents,
        comm_lag_steps=int(dcfg.comm_lag_steps),
        local_rollback_limit=ealr_cfg.local_rollback_limit,
        replan_horizon=ealr_cfg.replan_horizon,
        prio_age_tau=ealr_cfg.prio_age_tau,
        prio_cong_kappa=ealr_cfg.prio_cong_kappa,
        enable_v2_migrations=ealr_cfg.enable_v2_migrations,
        enable_belief_gating=ealr_cfg.enable_belief_gating,
        enable_cone_pull=ealr_cfg.enable_cone_pull,
        enable_siding_yield=ealr_cfg.enable_siding_yield,
        enable_choke_arbitration=ealr_cfg.enable_choke_arbitration,
        max_cone_pull=ealr_cfg.max_cone_pull,
    )
    belief_tracker = DelayDurationBeliefTracker()

    def _cur_goal(aid: str) -> Optional[str]:
        req = cur_req[aid]
        if req is None:
            return None
        if phase[aid] == "to_pickup":
            return req[1]
        if phase[aid] == "to_dropoff":
            return req[2]
        return None

    for t in range(horizon):
        if t > 0 and (t % 25 == 0):
            res_table.purge_before(t)

        # Check completions & handle service uncertainty at pickup / dropoff
        for aid in agv_ids:
            req = cur_req[aid]
            if req is not None:
                _, pu, do, rel_t = req
                if phase[aid] == "to_pickup" and pos[aid] == pu and not spatial_q[aid] and service_rem[aid] == 0:
                    if not pickup_serviced[aid]:
                        pickup_serviced[aid] = True
                        service_rem[aid] = int(service_delays.get((aid, t), 0))
                    if service_rem[aid] == 0:
                        phase[aid] = "need_dropoff"
                elif phase[aid] == "to_dropoff" and pos[aid] == do and not spatial_q[aid] and service_rem[aid] == 0:
                    completed += 1
                    wait_sum += max(0, t - rel_t)
                    phase[aid] = "idle"
                    cur_req[aid] = None
                    pickup_serviced[aid] = False

        # Assign new requests / next legs & re-queue stranded empty-queue active legs
        need_plan: list[tuple[str, str]] = []
        failed_this_tick: set[str] = set()
        for aid in agv_ids:
            if phase[aid] == "need_dropoff" and cur_req[aid] is not None:
                phase[aid] = "to_dropoff"
                fail_cooldown[aid] = 0
                need_plan.append((aid, cur_req[aid][2]))
            elif (
                phase[aid] == "to_pickup"
                and cur_req[aid] is not None
                and not spatial_q[aid]
                and service_rem[aid] == 0
                and t >= plan_end_t[aid]
                and pos[aid] != cur_req[aid][1]
            ):
                if fail_cooldown[aid] > 0:
                    fail_cooldown[aid] -= 1
                else:
                    need_plan.append((aid, cur_req[aid][1]))
            elif (
                phase[aid] == "to_dropoff"
                and cur_req[aid] is not None
                and not spatial_q[aid]
                and service_rem[aid] == 0
                and t >= plan_end_t[aid]
                and pos[aid] != cur_req[aid][2]
            ):
                if fail_cooldown[aid] > 0:
                    fail_cooldown[aid] -= 1
                else:
                    need_plan.append((aid, cur_req[aid][2]))
            elif phase[aid] == "idle" and task_q:
                req_id, pu, do = task_q.popleft()
                cur_req[aid] = (req_id, pu, do, t)
                task_rel[aid] = t
                pickup_serviced[aid] = False
                fail_streak[aid] = 0
                fail_cooldown[aid] = 0
                if pos[aid] == pu:
                    pickup_serviced[aid] = True
                    service_rem[aid] = int(service_delays.get((aid, t), 0))
                    phase[aid] = "to_dropoff"
                    need_plan.append((aid, do))
                else:
                    phase[aid] = "to_pickup"
                    need_plan.append((aid, pu))

        q_len_sum += len(task_q)

        # Plan newly assigned legs (all solvers start with CPS-A* planning for new legs)
        if need_plan:
            for aid, _ in need_plan:
                res_table.release_all(aid)
                field.remove_agv(aid)
                res_table.reserve_node(pos[aid], Interval(t, t + 1), aid)
                cap_p = grid.network.node(pos[aid]).capacity
                if t + 2 <= plan_h and res_table.is_node_free(pos[aid], Interval(t + 1, t + 2), cap_p, exclude_agv=aid):
                    res_table.reserve_node(pos[aid], Interval(t + 1, t + 2), aid)

            goal_map = dict(need_plan)

            def _np_key(aid: str) -> tuple[float, str]:
                age = float(t - task_rel.get(aid, t))
                d0 = float(shortest_dist_steps(grid.network, pos[aid], goal_map[aid], dist_cache))
                c0 = field.slot_Phi(pos[aid], t)
                return (d0 - 0.25 * age - 4.0 * c0, aid)

            leg_h = min(plan_h, t + int(eff_cfg.replan_horizon))
            for aid in sorted(goal_map.keys(), key=_np_key):
                res_table.release_all(aid)
                res_table.reserve_node(pos[aid], Interval(t, t + 1), aid)
                start_step_eff = min(plan_h, t + service_rem[aid])
                if service_rem[aid] > 0:
                    res_table.reserve_node(pos[aid], Interval(t, start_step_eff + 1), aid)
                np_max_exp = 900 if (solver == "ealr_local_repair" and fail_streak[aid] > 0) else max(1800, 4 * len(grid.network.nodes))
                planned, _ = planner.plan_leg(
                    pos[aid], goal_map[aid], start_step_eff, res_table, field, aid, leg_h,
                    goal_park_steps=1, max_expansions_override=np_max_exp,
                )
                total_exp += int(getattr(planner, "last_expansions", 0))
                if planned is not None:
                    fail_streak[aid] = 0
                    fail_cooldown[aid] = 0
                    pts = expand_planned_to_step_positions(pos[aid], start_step_eff, planned)
                    pre = [(pos[aid], tick_i) for tick_i in range(t, start_step_eff)]
                    full_pts = pre + pts
                    _fast_reserve_path(res_table, planner, grid.network, aid, full_pts, park_until=min(plan_h, full_pts[-1][1] + 2))
                    plan_steps[aid] = {st: nd for nd, st in full_pts}
                    plan_end_t[aid] = full_pts[-1][1]
                    spatial_q[aid] = deque([nd for nd, _ in pts[1:]])
                    field.add_agv_path(aid, full_pts, horizon=plan_h)
                    if solver == "adg_order_hold":
                        for nd, _ in pts[1:]:
                            q_adg = adg_node_order.setdefault(nd, deque())
                            if not q_adg or q_adg[-1] != aid:
                                q_adg.append(aid)
                else:
                    fail_streak[aid] += 1
                    fail_cooldown[aid] = min(2, fail_streak[aid]) if solver == "ealr_local_repair" else 0
                    failed_this_tick.add(aid)
                    res_table.reserve_node(pos[aid], Interval(t, min(plan_h, t + 2)), aid)
                    plan_steps[aid] = {t: pos[aid]}
                    plan_end_t[aid] = t
                    spatial_q[aid] = deque()

        # Check online execution-delay mismatch handling BEFORE stepping `t -> t+1`
        comm_lag = max(0, int(dcfg.comm_lag_steps))
        t_obs = max(0, t - comm_lag)
        obs_pos_map = {a: pos_history[a].get(t_obs, pos[a]) for a in agv_ids}
        goals_now = {a: _cur_goal(a) for a in agv_ids}

        # Check delay state at tick t (motion stall + ongoing service delay)
        stall_now: dict[str, bool] = {}
        for aid in agv_ids:
            if service_rem[aid] > 0:
                stall_now[aid] = True
                delay_rem[aid] = service_rem[aid]
            elif goals_now[aid] is not None and motion_stalls.get((aid, t), False):
                stall_now[aid] = True
                delay_rem[aid] = 1
            else:
                stall_now[aid] = False
                delay_rem[aid] = 0

        belief_tracker.observe_step(t, {aid for aid, st in stall_now.items() if st})

        if solver == "ealr_local_repair":
            mismatches = detect_reservation_mismatches(
                agv_ids,
                obs_pos_map if comm_lag > 0 else pos,
                delay_rem,
                service_rem,
                plan_steps,
                plan_end_t,
                goals_now,
                tick=t,
            )
            if mismatches:
                absorbed_ids: list[str] = []
                rem_mismatches = mismatches
                if eff_cfg.enable_v2_migrations and eff_cfg.enable_choke_arbitration:
                    rem_mismatches, absorbed_ids = try_time_shift_absorption(
                        mismatches, pos, delay_rem, service_rem, goals_now,
                        plan_steps, plan_end_t, res_table, field, grid.network,
                        tick=t, horizon=plan_h, comm_lag_steps=comm_lag,
                    )
                    for aid in absorbed_ids:
                        seq_cells_a: list[str] = []
                        prev_ca = pos[aid]
                        for s_i in range(t + 1, plan_end_t[aid] + 1):
                            cn = plan_steps[aid].get(s_i, prev_ca)
                            if cn != prev_ca:
                                seq_cells_a.append(cn)
                                prev_ca = cn
                        spatial_q[aid] = deque(seq_cells_a)
                r_loc = propagate_reservation_dependencies(
                    rem_mismatches, agv_ids, pos, plan_steps, plan_end_t, goals_now,
                    res_table, field, grid.network, eff_cfg, tick=t, belief_tracker=belief_tracker,
                )
                if r_loc:
                    if rem_mismatches and eff_cfg.max_repair_agents > 0:
                        skip_cd = {a for a in agv_ids if fail_cooldown[a] > 0 and a not in rem_mismatches}
                    else:
                        skip_cd = set()
                    r_eff = [a for a in r_loc if a not in skip_cd] or r_loc
                    prev_failed_set = {a for a in agv_ids if fail_streak[a] >= 1} | failed_this_tick
                    rep = local_reservation_repair(
                        r_eff, rem_mismatches, agv_ids, pos, delay_rem, service_rem, goals_now, task_rel,
                        plan_steps, plan_end_t, res_table, field, planner, grid.network, eff_cfg, dist_cache,
                        tick=t, horizon=plan_h, failed_prev_ids=prev_failed_set,
                    )
                    total_exp += int(rep["expansions"])
                    repaired_ids = rep.get("repaired_ids") or r_loc
                    failed_now = set(rep.get("failed_ids") or ())
                    for aid in repaired_ids:
                        if aid in failed_now:
                            fail_streak[aid] += 1
                            fail_cooldown[aid] = min(2, fail_streak[aid])
                        else:
                            fail_streak[aid] = 0
                            fail_cooldown[aid] = 0
                    repair_events += 1
                    repair_frac_sum += float(len(repaired_ids)) / max(1.0, float(len(agv_ids)))
                    for aid in repaired_ids:
                        # Sync spatial_q to newly repaired plan suffix after comm_lag + delay
                        seq_cells: list[str] = []
                        prev_c = pos[aid]
                        for s_i in range(t + 1, plan_end_t[aid] + 1):
                            cn = plan_steps[aid].get(s_i, prev_c)
                            if cn != prev_c:
                                seq_cells.append(cn)
                                prev_c = cn
                        spatial_q[aid] = deque(seq_cells)
                elif absorbed_ids:
                    repair_events += 1
                    repair_frac_sum += float(len(absorbed_ids)) / max(1.0, float(len(agv_ids)))

        elif solver == "global_rolling_cpsa":
            mismatches = detect_reservation_mismatches(
                agv_ids, obs_pos_map if comm_lag > 0 else pos, delay_rem, service_rem, plan_steps, plan_end_t, goals_now, tick=t
            )
            if mismatches:
                active_all = [a for a in agv_ids if goals_now[a] is not None or a in mismatches]
                rep_cfg = EALRConfig(
                    max_repair_agents=len(agv_ids),
                    comm_lag_steps=comm_lag,
                    replan_horizon=eff_cfg.replan_horizon,
                    enable_v2_migrations=False,
                )
                rep = local_reservation_repair(
                    active_all, mismatches, agv_ids, pos, delay_rem, service_rem, goals_now, task_rel,
                    plan_steps, plan_end_t, res_table, field, planner, grid.network, rep_cfg, dist_cache,
                    tick=t, horizon=plan_h,
                )
                total_exp += int(rep["expansions"])
                repair_events += 1
                repair_frac_sum += float(len(active_all)) / max(1.0, float(len(agv_ids)))
                for aid in active_all:
                    seq_cells = []
                    prev_c = pos[aid]
                    for s_i in range(t + 1, plan_end_t[aid] + 1):
                        cn = plan_steps[aid].get(s_i, prev_c)
                        if cn != prev_c:
                            seq_cells.append(cn)
                            prev_c = cn
                    spatial_q[aid] = deque(seq_cells)

        # Compute intended next step per solver semantics
        desired: dict[str, str] = {}
        for aid in agv_ids:
            if stall_now[aid]:
                desired[aid] = pos[aid]
            elif solver == "adg_order_hold":
                # Follow spatial_q[aid] strictly if head of ADG order at target cell
                if spatial_q[aid]:
                    tgt = spatial_q[aid][0]
                    q_adg = adg_node_order.get(tgt)
                    if q_adg and q_adg[0] != aid:
                        desired[aid] = pos[aid]
                    else:
                        desired[aid] = tgt
                else:
                    desired[aid] = pos[aid]
            elif solver == "naive_no_repair":
                # Follows planned path step at `t+1` blindly when matching current node, else walks next spatial hop
                if spatial_q[aid]:
                    desired[aid] = spatial_q[aid][0]
                else:
                    desired[aid] = plan_steps[aid].get(t + 1, pos[aid])
            else:
                desired[aid] = plan_steps[aid].get(t + 1, pos[aid])

        # Physical execution safety gate & physical transfer pins
        next_pos, blk = _resolve_physical_step(agv_ids, pos, desired, grid.network)
        near_miss_blocks += blk

        for aid in agv_ids:
            old_p = pos[aid]
            if next_pos[aid] != old_p:
                if spatial_q[aid] and spatial_q[aid][0] == next_pos[aid]:
                    spatial_q[aid].popleft()
                if solver == "adg_order_hold":
                    q_old = adg_node_order.get(old_p)
                    if q_old and q_old[0] == aid:
                        q_old.popleft()
            if service_rem[aid] > 0:
                service_rem[aid] -= 1
            pos[aid] = next_pos[aid]
            traces[aid][t + 1] = pos[aid]
            pos_history[aid][t + 1] = pos[aid]
            if phase[aid] == "idle":
                res_table.release_all(aid)
                res_table.reserve_node(pos[aid], Interval(t + 1, plan_h), aid)

    safe_ok, _ = verify_mapf_trace_safety(grid.network, traces)
    return DelayedLifelongResult(
        solver=solver,
        map_name=grid.name,
        seed=int(seed),
        n_agvs=len(agv_ids),
        scenario_tag=scenario_tag,
        completed_tasks=completed,
        throughput=round(float(completed) / max(1, horizon), 4),
        mean_waiting_time=round(float(wait_sum) / max(1, completed), 2),
        mean_queue_len=round(float(q_len_sum) / max(1, horizon), 2),
        total_expansions=total_exp,
        repair_events=repair_events,
        avg_repair_fraction=round(repair_frac_sum / max(1, repair_events), 4),
        near_miss_blocks=near_miss_blocks,
        zero_conflicts=safe_ok,
    )
