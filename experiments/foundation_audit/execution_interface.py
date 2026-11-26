"""F1 Read-Only Unified Execution Interface (`Discrete` <-> `Continuous`).

Provides a clean, non-invasive abstraction mapping shared transport/layout
specifications (`UnifiedTaskSpec`, `UnifiedRobotSpec`) onto BOTH:
  - `DiscreteBackendAdapter`: `amr_sched.simulation.Simulator` (`dt=1.0`,
    `RustSpaceTimeAStarPlanner` / `SpaceTimeAStarPlanner`, `RuleBasedAllocator`
    or fixed queue/assignment replay), with read-only per-leg telemetry capture.
  - `ContinuousBackendAdapter`: Track A `ExecutionAuditSim` running locked
    `U1a_EdgeSpan_Fix` + `R1_Robust` (`_reserve_exact_span` +
    `m2_res_margin_selector`), supporting deterministic dispatch rules
    (`fifo+greedy`, `edd+hungarian`, `priority+nearest`, `spt+greedy`,
    `cr+greedy`, `nearest_idle`) and fixed dispatch-schedule replay.

Strict architectural invariant: ZERO edits to frozen `amr_sched/`, `crates/`,
`results/`, `experiments/res_guided_avoidance/`, `experiments/track_r_cpsa/`,
or `experiments/stp_research_reset/`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
import math
import pathlib
import sys
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.allocation.base import AllocationStrategy
from amr_sched.allocation.dispatch import DispatchRule
from amr_sched.allocation.rule_based import RuleBasedAllocator
from amr_sched.entities import Agv, AgvState, Task, TaskPriority, TaskStatus
from amr_sched.network import Edge, Network, Node
from amr_sched.planners.rust_spacetime_astar import (
    RustSpaceTimeAStarPlanner,
    is_rust_planner_available,
)
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import SimulationConfig, Simulator
from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.metric_world import (
    ContTask,
    CorridorCapsule,
    MetricMap,
    RobotContinuousState,
)
from experiments.stp_research_reset.audit_env import ExecutionAuditSim
from experiments.stp_research_reset.upper_planners import u1a_plan_fn
from tools.live_bridge.fleet_config import point_execution_network


@dataclass(frozen=True)
class UnifiedTaskSpec:
    """Backend-agnostic transport task specification in SI seconds."""

    task_id: str
    pickup: str
    dropoff: str
    release_s: float = 0.0
    due_s: float | None = None
    priority: str = "normal"  # "normal" | "urgent"
    service_s: float = 2.0
    assigned_agv: str | None = None
    dispatch_s: float | None = None
    queue_order: int = 0


@dataclass(frozen=True)
class UnifiedRobotSpec:
    """Backend-agnostic mobile robot specification in SI units."""

    robot_id: str
    start_node: str
    max_speed: float = 1.0
    load_s: float = 2.0
    unload_s: float = 0.0


@dataclass(frozen=True)
class UnifiedLegRecord:
    """Per-leg travel and ETA prediction telemetry."""

    task_id: str
    robot_id: str
    phase: str  # "to_pickup" | "to_dropoff"
    start_node: str
    goal_node: str
    planned_dur_s: float
    actual_dur_s: float
    deta_s: float
    euclidean_m: float
    graph_shortest_m: float
    completed: bool = True


@dataclass
class UnifiedBackendResult:
    """Normalized execution metrics produced by either fidelity backend."""

    backend: str
    scenario_id: str
    policy: str
    seed: int
    completed_tasks: int
    total_tasks: int
    makespan_s: float
    zero_collisions: bool
    collision_count: int
    wait_ratio: float
    total_replans: int
    mean_deta_s: float
    mae_eta_s: float
    on_time_rate: float
    total_tardiness_s: float
    task_completions_s: dict[str, float] = field(default_factory=dict)
    task_assignments: dict[str, str] = field(default_factory=dict)
    leg_records: list[UnifiedLegRecord] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


def _shortest_path_length_m(
    network: Network, src: str, dst: str, cache: dict[tuple[str, str], float]
) -> float:
    if src == dst:
        return 0.0
    key = (src, dst)
    if key not in cache:
        dists = network.shortest_travel_times(src, max_speed=1.0)
        # Note shortest_travel_times multiplies by traversal_time = length / speed_limit;
        # compute pure topological edge lengths with Dijkstra when needed.
        import heapq

        pq: list[tuple[float, str]] = [(0.0, src)]
        len_map: dict[str, float] = {src: 0.0}
        while pq:
            d, u = heapq.heappop(pq)
            if d > len_map.get(u, float("inf")):
                continue
            for e in network.out_edges(u):
                nd = d + float(e.length)
                if nd < len_map.get(e.dst, float("inf")):
                    len_map[e.dst] = nd
                    heapq.heappush(pq, (nd, e.dst))
        for k, v in len_map.items():
            cache[(src, k)] = float(v)
        cache.setdefault(key, float("inf"))
    return float(cache[key])


def network_to_metric_map(
    name: str,
    network: Network,
    *,
    half_width_m: float = 1.45,
    macro_step_s: float = 2.0,
    start_pool: tuple[str, ...] = (),
    pickup_nodes: tuple[str, ...] = (),
    dropoff_nodes: tuple[str, ...] = (),
) -> MetricMap:
    """Lift any topological `Network` with 2D node `(x, y)` coordinates into `MetricMap`."""
    capsules: list[CorridorCapsule] = []
    ehw: dict[tuple[str, str], float] = {}
    seen_pairs: set[tuple[str, str]] = set()
    xs = [float(n.x) for n in network.nodes.values()] or [0.0]
    ys = [float(n.y) for n in network.nodes.values()] or [0.0]

    for (u, v), e in network.edges.items():
        pair = (u, v) if u <= v else (v, u)
        ehw[(u, v)] = float(half_width_m)
        if network.has_edge(v, u):
            ehw[(v, u)] = float(half_width_m)
        if pair in seen_pairs:
            continue
        seen_pairs.add(pair)
        nu, nv = network.node(u), network.node(v)
        capsules.append(CorridorCapsule(float(nu.x), float(nu.y), float(nv.x), float(nv.y), float(half_width_m)))

    pad_r = max(float(half_width_m), 1.35)
    for nid, nd in network.nodes.items():
        node_pad = max(pad_r, 2.2) if int(nd.capacity) > 1 else pad_r
        capsules.append(CorridorCapsule(float(nd.x), float(nd.y), float(nd.x), float(nd.y), node_pad))

    pad_b = max(4.0, float(half_width_m) * 3.0)
    bounds = (min(xs) - pad_b, min(ys) - pad_b, max(xs) + pad_b, max(ys) + pad_b)
    all_nodes = tuple(sorted(network.nodes.keys()))
    return MetricMap(
        name=name,
        network=network,
        world_bounds=bounds,
        corridor_capsules=tuple(capsules),
        edge_half_widths=ehw,
        pickup_nodes=pickup_nodes or all_nodes,
        dropoff_nodes=dropoff_nodes or all_nodes,
        start_pool=start_pool or all_nodes,
        macro_step_s=float(macro_step_s),
    )


def scale_network_for_calibration(
    network: Network,
    *,
    edge_duration_factors: dict[tuple[str, str], float] | None = None,
    global_speed_scale: float = 1.0,
) -> Network:
    """Create a read-only clone of `network` with calibrated edge lengths / speed limits."""
    cloned = Network()
    for nd in network.nodes.values():
        cloned.add_node(
            Node(
                node_id=nd.node_id,
                x=float(nd.x),
                y=float(nd.y),
                kind=nd.kind,
                capacity=int(nd.capacity),
            )
        )
    for (u, v), e in network.edges.items():
        mult = 1.0
        if edge_duration_factors and (u, v) in edge_duration_factors:
            mult = float(edge_duration_factors[(u, v)])
        eff_speed = max(0.25, float(e.speed_limit) * float(global_speed_scale))
        kw: dict[str, Any] = {"speed_limit": eff_speed}
        if abs(mult - 1.0) > 1e-9:
            kw["length"] = max(0.5, float(e.length) * mult)
            kw["centerline"] = ()
        cloned.add_edge(replace(e, **kw))
    return cloned


def _parse_policy_rule(policy: str) -> tuple[DispatchRule, str]:
    key = str(policy).strip().lower()
    if key in ("nearest", "nearest_idle"):
        return DispatchRule.FIFO, "nearest"
    if "+" in key:
        disp_s, assign_s = key.split("+", 1)
        return DispatchRule(disp_s), assign_s
    return DispatchRule.FIFO, "greedy"


class _FixedDispatchAllocator(AllocationStrategy):
    """Deterministic replay allocator honoring pre-exported `(assigned_agv, dispatch_step, queue_order)`."""

    name = "fixed_replay"

    def __init__(
        self,
        assignments: dict[str, str],
        dispatch_steps: dict[str, int],
        queue_orders: dict[str, int],
    ) -> None:
        self.assignments = assignments
        self.dispatch_steps = dispatch_steps
        self.queue_orders = queue_orders

    def allocate(
        self,
        network: Network,
        idle_agvs: list[Agv],
        pending_tasks: list[Task],
        current_step: int,
        max_speed: float,
        travel_time_cache: dict[tuple[str, str], float],
    ) -> list[tuple[Agv, Task]]:
        idle_map = {a.agv_id: a for a in idle_agvs}
        by_agv: dict[str, list[Task]] = {}
        for t in pending_tasks:
            owner = self.assignments.get(t.task_id)
            if owner is None or owner not in idle_map:
                continue
            d_step = self.dispatch_steps.get(t.task_id, t.release_step)
            if current_step < d_step:
                continue
            by_agv.setdefault(owner, []).append(t)

        pairs: list[tuple[Agv, Task]] = []
        for owner, cands in by_agv.items():
            cands.sort(key=lambda t: (self.queue_orders.get(t.task_id, 0), self.dispatch_steps.get(t.task_id, 0), t.task_id))
            pairs.append((idle_map[owner], cands[0]))
        return pairs


class DiscreteBackendAdapter:
    """Read-only adapter executing `UnifiedTaskSpec`/`UnifiedRobotSpec` on `amr_sched.simulation.Simulator`."""

    def run(
        self,
        scenario_id: str,
        network: Network,
        tasks: list[UnifiedTaskSpec],
        robots: list[UnifiedRobotSpec],
        *,
        policy: str = "fifo+greedy",
        dt: float = 1.0,
        horizon_s: float = 500.0,
        seed: int = 0,
        reservation_window: int = 1,
        point_execution: bool = True,
        fixed_dispatches: bool = False,
        enforce_dispatch_step: bool = True,
        use_rust: bool = True,
        extra_service_s: float = 0.0,
    ) -> UnifiedBackendResult:
        eff_net = point_execution_network(network) if point_execution else network
        max_spd = max((float(r.max_speed) for r in robots), default=1.0)
        mean_load = float(np.mean([r.load_s for r in robots])) if robots else 2.0
        mean_unload = float(np.mean([r.unload_s for r in robots])) if robots else 0.0
        max_steps = max(10, int(math.ceil(horizon_s / dt)))

        agv_objs = [
            Agv(
                agv_id=r.robot_id,
                max_speed=float(r.max_speed),
                start_node=r.start_node,
                load_time=float(r.load_s + extra_service_s),
                unload_time=float(r.unload_s),
            )
            for r in robots
        ]
        task_objs: list[Task] = []
        assign_map: dict[str, str] = {}
        dispatch_step_map: dict[str, int] = {}
        queue_order_map: dict[str, int] = {}
        per_agv_queues: dict[str, list[str]] = {}
        due_map_s: dict[str, float] = {}

        ordered_tasks = sorted(tasks, key=lambda t: (t.queue_order, t.release_s, t.task_id))
        for t in ordered_tasks:
            rel_step = max(0, int(round(t.release_s / dt)))
            due_step = max(rel_step, int(round(t.due_s / dt))) if t.due_s is not None else None
            if t.due_s is not None:
                due_map_s[t.task_id] = float(t.due_s)
            prio = TaskPriority.URGENT if str(t.priority).lower() == "urgent" else TaskPriority.NORMAL
            task_objs.append(
                Task(
                    task_id=t.task_id,
                    pickup=t.pickup,
                    dropoff=t.dropoff,
                    release_step=rel_step,
                    due_step=due_step,
                    priority=prio,
                )
            )
            if t.assigned_agv is not None:
                assign_map[t.task_id] = str(t.assigned_agv)
                d_step = max(rel_step, int(round(t.dispatch_s / dt))) if (enforce_dispatch_step and t.dispatch_s is not None) else rel_step
                dispatch_step_map[t.task_id] = d_step
                queue_order_map[t.task_id] = int(t.queue_order)
                per_agv_queues.setdefault(str(t.assigned_agv), []).append(t.task_id)

        if fixed_dispatches:
            allocator: AllocationStrategy = _FixedDispatchAllocator(assign_map, dispatch_step_map, queue_order_map)
        else:
            disp_rule, assign_alg = _parse_policy_rule(policy)
            allocator = RuleBasedAllocator(disp_rule, assign_alg)

        if use_rust and is_rust_planner_available():
            planner = RustSpaceTimeAStarPlanner(max_speed=max_spd, dt=dt, max_expansions=6_000)
        else:
            planner = SpaceTimeAStarPlanner(max_speed=max_spd, dt=dt, max_expansions=6_000)

        cfg = SimulationConfig(
            dt=float(dt),
            max_steps=max_steps,
            seed=int(seed),
            load_time=mean_load + extra_service_s,
            unload_time=mean_unload,
            reservation_window=int(reservation_window),
            record_trajectory=False,
        )
        sim = Simulator(eff_net, task_objs, agv_objs, planner, allocator, config=cfg)
        sim.reset()
        if fixed_dispatches and not enforce_dispatch_step and per_agv_queues:
            sim.set_task_assignment_pins(assign_map)
            sim.set_task_queues(per_agv_queues)

        # Read-only per-leg telemetry capture across step_once()
        active_leg: dict[str, dict[str, Any]] = {}
        arrived_job_key: dict[str, tuple[str, str]] = {}
        legs: list[UnifiedLegRecord] = []
        len_cache: dict[tuple[str, str], float] = {}

        def _close_leg(rid: str, end_step: int, completed_flag: bool) -> None:
            st = active_leg.pop(rid, None)
            if st is None:
                return
            actual_s = max(0.0, float(end_step - st["start_step"]) * dt)
            planned_s = max(0.0, float(st["planned_end_step"] - st["start_step"]) * dt)
            sn, gn = st["start_node"], st["goal_node"]
            nu, nv = eff_net.node(sn), eff_net.node(gn)
            euc = float(math.hypot(nv.x - nu.x, nv.y - nu.y))
            gdist = _shortest_path_length_m(eff_net, sn, gn, len_cache)
            legs.append(
                UnifiedLegRecord(
                    task_id=str(st["task_id"]),
                    robot_id=rid,
                    phase=str(st["phase"]),
                    start_node=sn,
                    goal_node=gn,
                    planned_dur_s=round(planned_s, 3),
                    actual_dur_s=round(actual_s, 3),
                    deta_s=round(actual_s - planned_s, 3),
                    euclidean_m=round(euc, 3),
                    graph_shortest_m=round(gdist, 3),
                    completed=completed_flag,
                )
            )

        while sim.step_once():
            step_now = sim._step_cursor - 1
            for agv in sim.agvs:
                rid = agv.agv_id
                job = sim._jobs.get(rid)
                cur_state = active_leg.get(rid)
                if job is None or job.leg not in ("to_pickup", "to_dropoff"):
                    if cur_state is not None:
                        _close_leg(rid, step_now, completed_flag=False)
                    arrived_job_key.pop(rid, None)
                    continue
                jkey = (str(job.task_id), str(job.leg))
                if cur_state is None or (cur_state["task_id"], cur_state["phase"]) != jkey:
                    if cur_state is not None:
                        _close_leg(rid, step_now, completed_flag=False)
                    if arrived_job_key.get(rid) == jkey:
                        continue
                    if job.hops and job.idx < len(job.hops):
                        sn = job.hops[0].src
                        init_end = job.hops[-1].arrive
                        active_leg[rid] = {
                            "task_id": str(job.task_id),
                            "phase": str(job.leg),
                            "start_step": step_now,
                            "start_node": sn,
                            "goal_node": str(job.goal),
                            "planned_end_step": int(init_end),
                            "arrived": False,
                        }
                    elif agv.cur_node == job.goal:
                        arrived_job_key[rid] = jkey
                cur_state = active_leg.get(rid)
                if (
                    cur_state is not None
                    and not cur_state["arrived"]
                    and job.idx >= len(job.hops)
                    and rid not in sim._transit
                    and agv.cur_node == cur_state["goal_node"]
                ):
                    cur_state["arrived"] = True
                    arrived_job_key[rid] = jkey
                    _close_leg(rid, step_now + 1, completed_flag=True)

        res = sim._build_result(sim._step_cursor)
        for rid in list(active_leg.keys()):
            _close_leg(rid, res.finished_step, completed_flag=False)

        completions_s: dict[str, float] = {}
        assignments_out: dict[str, str] = {}
        due_cnt = 0
        on_time_cnt = 0
        tardiness_s = 0.0
        for tk in sim.tasks:
            if tk.assigned_agv is not None:
                assignments_out[tk.task_id] = str(tk.assigned_agv)
            if tk.complete_step is not None:
                comp_s = float(tk.complete_step) * dt
                completions_s[tk.task_id] = round(comp_s, 3)
            if tk.task_id in due_map_s:
                due_cnt += 1
                if tk.complete_step is not None:
                    comp_s = float(tk.complete_step) * dt
                    tard = max(0.0, comp_s - due_map_s[tk.task_id])
                    tardiness_s += tard
                    if tard <= 1e-6:
                        on_time_cnt += 1
                else:
                    tardiness_s += max(0.0, float(horizon_s) - due_map_s[tk.task_id])

        comp_legs = [lg for lg in legs if lg.completed]
        mean_deta = float(np.mean([lg.deta_s for lg in comp_legs])) if comp_legs else 0.0
        mae_eta = float(np.mean([abs(lg.deta_s) for lg in comp_legs])) if comp_legs else 0.0
        denom_steps = max(1, len(sim.agvs) * max(1, res.finished_step))
        wait_ratio = float(res.conflict_wait_steps()) / float(denom_steps)

        raw_ms_s = float(res.makespan_seconds())
        mk_s = raw_ms_s if math.isfinite(raw_ms_s) else float(res.finished_step) * dt

        return UnifiedBackendResult(
            backend="discrete",
            scenario_id=scenario_id,
            policy=policy,
            seed=int(seed),
            completed_tasks=len(res.completed_tasks()),
            total_tasks=len(tasks),
            makespan_s=round(mk_s, 3),
            zero_collisions=(int(res.collision_events) == 0),
            collision_count=int(res.collision_events),
            wait_ratio=round(wait_ratio, 4),
            total_replans=int(res.replan_count),
            mean_deta_s=round(mean_deta, 3),
            mae_eta_s=round(mae_eta, 3),
            on_time_rate=round(on_time_cnt / max(1, due_cnt), 4),
            total_tardiness_s=round(tardiness_s, 3),
            task_completions_s=completions_s,
            task_assignments=assignments_out,
            leg_records=legs,
            meta={
                "dt": dt,
                "finished_step": int(res.finished_step),
                "point_execution": bool(point_execution),
                "reservation_window": int(reservation_window),
            },
        )


class _UnifiedAuditBridgeSim(ExecutionAuditSim):
    """Exact-task & exact-robot ExecutionAuditSim subclass for unified F1-F5 evaluation."""

    def __init__(
        self,
        mmap: MetricMap,
        tasks_spec: list[UnifiedTaskSpec],
        robots_spec: list[UnifiedRobotSpec],
        *,
        policy: str = "fifo+greedy",
        horizon_s: float = 300.0,
        seed: int = 0,
        fixed_dispatches: bool = False,
        enforce_dispatch_time: bool = False,
        robust_safety: bool = True,
        obs_noise_pos_m: float = 0.0,
        obs_noise_vel_mps: float = 0.0,
        reservation_lag_s: float = 0.0,
        reservation_dropout: float = 0.0,
        disturb_slow_prob: float = 0.0,
        n_intruders: int = 0,
    ) -> None:
        self._tasks_spec = list(tasks_spec)
        self._robots_spec = list(robots_spec)
        self._policy_key = policy
        self._fixed_dispatches = bool(fixed_dispatches)
        self._enforce_dispatch_time = bool(enforce_dispatch_time)
        self._spec_by_int_id: dict[int, UnifiedTaskSpec] = {}
        self._leg_task_id: dict[int, str] = {}
        self._fixed_queues: dict[str, deque[int]] = {}
        self._task_completions_s: dict[str, float] = {}
        self._task_assignments_out: dict[str, str] = {}
        self._travel_cache: dict[tuple[str, str], float] = {}
        self._rule_allocator: RuleBasedAllocator | None = None
        if not self._fixed_dispatches and policy not in ("nearest", "nearest_idle"):
            disp_r, assign_a = _parse_policy_rule(policy)
            self._rule_allocator = RuleBasedAllocator(disp_r, assign_a)

        super().__init__(
            mmap,
            upper_mode="U1a_EdgeSpan_Fix",
            sub_tick_div=1,
            custom_plan_fn=u1a_plan_fn,
            record_telemetry=True,
            n_robots=len(robots_spec),
            n_tasks=len(tasks_spec),
            horizon_s=horizon_s,
            seed=seed,
            n_intruders=n_intruders,
            obs_noise_pos_m=obs_noise_pos_m,
            obs_noise_vel_mps=obs_noise_vel_mps,
            reservation_lag_s=reservation_lag_s,
            reservation_dropout=reservation_dropout,
            disturb_slow_prob=disturb_slow_prob,
            robust_safety=robust_safety,
        )

    def reset(self) -> None:
        super().reset()
        # Replace synthetic random placement and random tasks with exact unified specifications.
        self.robots.clear()
        self.robot_ids.clear()
        self.pending.clear()
        self.ready.clear()
        self.res_table.clear()
        self.res_table = self.res_table.__class__.fast()
        self.field = self.planner.create_field()
        self._phys_locks.clear()
        self._active_pts.clear()
        self._active_leg_plans.clear()
        self._route_times_sec.clear()
        self._active_buf_s.clear()
        self.active_legs.clear()
        self.closed_legs.clear()
        self._task_completions_s.clear()
        self._task_assignments_out.clear()
        self._spec_by_int_id.clear()
        self._leg_task_id.clear()
        self._fixed_queues.clear()

        placed_xy: list[tuple[float, float]] = []
        for idx, rspec in enumerate(self._robots_spec):
            nd = self.mmap.network.node(rspec.start_node)
            rx, ry = float(nd.x), float(nd.y)
            # Guard against capacity>1 dock nodes with co-located starting AGVs (e.g. STG_W)
            coloc_idx = sum(1 for px, py in placed_xy if math.hypot(rx - px, ry - py) < 1.25)
            if coloc_idx > 0:
                ang = (2.0 * math.pi * coloc_idx) / max(4, coloc_idx + 2)
                rx += 1.35 * math.cos(ang)
                ry += 1.35 * math.sin(ang)
            placed_xy.append((rx, ry))
            outs = self.mmap.network.out_edges(rspec.start_node)
            if outs:
                nb = self.mmap.network.node(outs[0].dst)
                yaw = math.atan2(nb.y - ry, nb.x - rx)
            else:
                yaw = 0.0
            self.robots[rspec.robot_id] = RobotContinuousState(
                robot_id=rspec.robot_id,
                x=rx,
                y=ry,
                theta=yaw,
                cur_node=rspec.start_node,
                route_nodes=[rspec.start_node],
                route_times=[0],
                route_idx=0,
            )
            self.robot_ids.append(rspec.robot_id)
            from amr_sched.rust_reservation import Interval

            self.res_table.reserve_node(rspec.start_node, Interval(0, 2), rspec.robot_id)
            self._phys_locks[rspec.robot_id] = (rspec.start_node, 0, 2)
            self._prev_v[rspec.robot_id] = 0.0
            self._prev_a[rspec.robot_id] = 0.0

        ordered = sorted(self._tasks_spec, key=lambda t: (t.release_s, t.queue_order, t.task_id))
        pending_tmp: list[ContTask] = []
        for idx, ts in enumerate(ordered):
            due_val = float(ts.due_s) if ts.due_s is not None else float(self.horizon_s + 200.0)
            ct = ContTask(
                task_id=idx,
                pickup=ts.pickup,
                dropoff=ts.dropoff,
                release_s=float(ts.release_s),
                due_s=due_val,
                service_s=float(ts.service_s),
            )
            self._spec_by_int_id[idx] = ts
            pending_tmp.append(ct)
        self.pending = deque(sorted(pending_tmp, key=lambda c: c.release_s))

        if self._fixed_dispatches:
            for ts in sorted(self._tasks_spec, key=lambda x: (x.queue_order, x.dispatch_s or 0.0, x.task_id)):
                if ts.assigned_agv:
                    int_ids = [iid for iid, sp in self._spec_by_int_id.items() if sp.task_id == ts.task_id]
                    if int_ids:
                        self._fixed_queues.setdefault(str(ts.assigned_agv), deque()).append(int_ids[0])

    def _open_leg_record(
        self,
        rid: str,
        goal_node: str,
        t_tick: int,
        *,
        is_replan: bool,
        is_drift: bool,
        waited: int,
    ) -> None:
        super()._open_leg_record(
            rid,
            goal_node,
            t_tick,
            is_replan=is_replan,
            is_drift=is_drift,
            waited=waited,
        )
        rb = self.robots.get(rid)
        if rid in self.active_legs and rb is not None and rb.cur_task is not None:
            sp = self._spec_by_int_id.get(rb.cur_task.task_id)
            if sp is not None:
                self._leg_task_id[self.active_legs[rid].leg_id] = sp.task_id

    def _macro_coordinate(self) -> None:
        t_tick = self.macro_tick
        while self.pending and self.pending[0].release_s <= self.t_s + 1e-6:
            self.ready.append(self.pending.popleft())

        # Record task dropoff completions cleanly before delegating
        for rid in self.robot_ids:
            rb = self.robots[rid]
            if rb.cur_task is not None and rb.phase == "to_dropoff" and rb.service_rem_s <= 0.0:
                tgt = self.mmap.network.node(rb.cur_task.dropoff)
                if math.hypot(rb.x - tgt.x, rb.y - tgt.y) <= 0.85:
                    sp = self._spec_by_int_id[rb.cur_task.task_id]
                    self._task_completions_s[sp.task_id] = round(float(self.t_s), 3)

        # Dispatch idle robots strictly according to configured allocator / fixed queues
        idle_rids = [
            rid
            for rid in self.robot_ids
            if self.robots[rid].phase == "idle" and self.robots[rid].service_rem_s <= 0.0
        ]
        if idle_rids and self.ready:
            if self._fixed_dispatches:
                ready_by_id = {ct.task_id: ct for ct in self.ready}
                used: set[int] = set()
                for rid in idle_rids:
                    q = self._fixed_queues.get(rid)
                    if not q:
                        continue
                    nxt_id = q[0]
                    ct = ready_by_id.get(nxt_id)
                    if ct is None:
                        continue
                    sp = self._spec_by_int_id[nxt_id]
                    if self._enforce_dispatch_time and sp.dispatch_s is not None and self.t_s + 1e-6 < sp.dispatch_s:
                        continue
                    q.popleft()
                    used.add(nxt_id)
                    rb = self.robots[rid]
                    rb.cur_task = ct
                    rb.phase = "to_pickup"
                    self._task_assignments_out[sp.task_id] = rid
                    self._plan_robot(rid, ct.pickup, t_tick)
                if used:
                    self.ready = [ct for ct in self.ready if ct.task_id not in used]
            elif self._rule_allocator is not None:
                # Reuse discrete RuleBasedAllocator cleanly without code duplication
                agv_proxies = [
                    Agv(agv_id=rid, max_speed=1.0, start_node=self.robots[rid].cur_node)
                    for rid in idle_rids
                ]
                ct_by_tid: dict[str, ContTask] = {}
                t_proxies: list[Task] = []
                for ct in self.ready:
                    sp = self._spec_by_int_id[ct.task_id]
                    ct_by_tid[sp.task_id] = ct
                    t_proxies.append(
                        Task(
                            task_id=sp.task_id,
                            pickup=sp.pickup,
                            dropoff=sp.dropoff,
                            release_step=int(round(sp.release_s)),
                            due_step=int(round(sp.due_s)) if sp.due_s is not None else None,
                            priority=TaskPriority.URGENT if sp.priority == "urgent" else TaskPriority.NORMAL,
                        )
                    )
                pairs = self._rule_allocator.allocate(
                    self.mmap.network,
                    agv_proxies,
                    t_proxies,
                    int(round(self.t_s)),
                    1.0,
                    self._travel_cache,
                )
                used_ids: set[int] = set()
                for agv_p, t_p in pairs:
                    ct = ct_by_tid[t_p.task_id]
                    used_ids.add(ct.task_id)
                    rb = self.robots[agv_p.agv_id]
                    rb.cur_task = ct
                    rb.phase = "to_pickup"
                    self._task_assignments_out[t_p.task_id] = agv_p.agv_id
                    self._plan_robot(agv_p.agv_id, ct.pickup, t_tick)
                if used_ids:
                    self.ready = [ct for ct in self.ready if ct.task_id not in used_ids]

        # Mask unassigned tasks when not in default nearest_idle so base MetricWorldSim
        # never steals queued/policy-allocated tasks.
        if self._fixed_dispatches or self._rule_allocator is not None:
            saved_ready = self.ready
            self.ready = []
            try:
                super()._macro_coordinate()
            finally:
                self.ready = saved_ready
        else:
            # Record nearest_idle task assignment before delegating
            super()._macro_coordinate()
            for rid in self.robot_ids:
                rb = self.robots[rid]
                if rb.cur_task is not None:
                    sp = self._spec_by_int_id[rb.cur_task.task_id]
                    self._task_assignments_out.setdefault(sp.task_id, rid)


class ContinuousBackendAdapter:
    """Read-only adapter executing `UnifiedTaskSpec`/`UnifiedRobotSpec` on Track A `U1a + R1_Robust`."""

    def run(
        self,
        scenario_id: str,
        mmap: MetricMap,
        tasks: list[UnifiedTaskSpec],
        robots: list[UnifiedRobotSpec],
        *,
        policy: str = "fifo+greedy",
        horizon_s: float = 400.0,
        seed: int = 0,
        fixed_dispatches: bool = False,
        enforce_dispatch_time: bool = False,
        robust_safety: bool = True,
        obs_noise_pos_m: float = 0.0,
        obs_noise_vel_mps: float = 0.0,
        reservation_lag_s: float = 0.0,
        reservation_dropout: float = 0.0,
        disturb_slow_prob: float = 0.0,
        n_intruders: int = 0,
    ) -> UnifiedBackendResult:
        sim = _UnifiedAuditBridgeSim(
            mmap,
            tasks,
            robots,
            policy=policy,
            horizon_s=horizon_s,
            seed=seed,
            fixed_dispatches=fixed_dispatches,
            enforce_dispatch_time=enforce_dispatch_time,
            robust_safety=robust_safety,
            obs_noise_pos_m=obs_noise_pos_m,
            obs_noise_vel_mps=obs_noise_vel_mps,
            reservation_lag_s=reservation_lag_s,
            reservation_dropout=reservation_dropout,
            disturb_slow_prob=disturb_slow_prob,
            n_intruders=n_intruders,
        )
        res = sim.run_episode(m2_res_margin_selector, controller_name=f"Cont_U1a_R1_{policy}")

        # Convert LegTelemetry + active unfinished legs into UnifiedLegRecord list
        len_cache: dict[tuple[str, str], float] = {}
        legs: list[UnifiedLegRecord] = []
        for lg in sim.closed_legs:
            if lg.end_t_s is None:
                continue
            actual_s = max(0.0, float(lg.end_t_s - lg.plan_t_s))
            planned_s = float(lg.planned_dur_s)
            nu = mmap.network.node(lg.start_node)
            nv = mmap.network.node(lg.goal_node)
            euc = float(math.hypot(nv.x - nu.x, nv.y - nu.y))
            gdist = _shortest_path_length_m(mmap.network, lg.start_node, lg.goal_node, len_cache)
            tid_str = sim._leg_task_id.get(lg.leg_id, f"leg_{lg.leg_id}")
            legs.append(
                UnifiedLegRecord(
                    task_id=tid_str,
                    robot_id=str(lg.robot_id),
                    phase=str(lg.phase),
                    start_node=str(lg.start_node),
                    goal_node=str(lg.goal_node),
                    planned_dur_s=round(planned_s, 3),
                    actual_dur_s=round(actual_s, 3),
                    deta_s=round(actual_s - planned_s, 3),
                    euclidean_m=round(euc, 3),
                    graph_shortest_m=round(gdist, 3),
                    completed=bool(lg.completed),
                )
            )

        due_cnt = 0
        on_time_cnt = 0
        tardiness_s = 0.0
        for ts in tasks:
            if ts.due_s is not None:
                due_cnt += 1
                comp_s = sim._task_completions_s.get(ts.task_id)
                if comp_s is not None:
                    tard = max(0.0, comp_s - float(ts.due_s))
                    tardiness_s += tard
                    if tard <= 1e-6:
                        on_time_cnt += 1
                else:
                    tardiness_s += max(0.0, float(horizon_s) - float(ts.due_s))

        comp_legs = [lg for lg in legs if lg.completed]
        mean_deta = float(np.mean([lg.deta_s for lg in comp_legs])) if comp_legs else 0.0
        mae_eta = float(np.mean([abs(lg.deta_s) for lg in comp_legs])) if comp_legs else 0.0
        col_tot = int(res.wall_collisions + res.peer_collisions + res.intruder_collisions)

        return UnifiedBackendResult(
            backend="continuous",
            scenario_id=scenario_id,
            policy=policy,
            seed=int(seed),
            completed_tasks=int(res.completed_tasks),
            total_tasks=len(tasks),
            makespan_s=float(res.makespan_s),
            zero_collisions=(col_tot == 0),
            collision_count=col_tot,
            wait_ratio=float(res.wait_ratio),
            total_replans=int(res.replans),
            mean_deta_s=round(mean_deta, 3),
            mae_eta_s=round(mae_eta, 3),
            on_time_rate=round(on_time_cnt / max(1, due_cnt), 4),
            total_tardiness_s=round(tardiness_s, 3),
            task_completions_s=dict(sim._task_completions_s),
            task_assignments=dict(sim._task_assignments_out),
            leg_records=legs,
            meta={
                "eta_drift_replans": int(sim.eta_drift_replans),
                "row_gate_events": int(sim.row_gate_events),
                "shield_interventions": int(res.shield_interventions),
                "independent_collision_events": int(sim.independent_collision_events),
                "macro_step_s": float(mmap.macro_step_s),
            },
        )
