"""P2 L1/L2 Execution Uncertainty & Causal Disturbance Model for AMR Closed-Loop Simulation.

Introduces three orthogonal physical execution uncertainties decoupled cleanly from
deterministic nominal baselines:
  1. **Stochastic Travel-Time Uncertainty**: per-hop early/late realization relative
     to nominal shortest-path ETA, bounded by destination capacity & horizon safety.
  2. **Dynamic Corridor Obstacles (`ROUTE_DEVIATION`)**: temporary aisle blockages
     placed strictly on graph-verified bypassable high-centrality corridors so all
     tasks remain physically completable via alternate routing.
  3. **Transient L1 Local-Planner / Watchdog Failures (`local_failed`)**: bounded
     Nav2-style local recovery pauses on active AMRs with elevated receipt age.

Strictly Zero Future Leakage:
  `capture_feedback(sim, step)` and `active_disturbance_sig(sim, step)` only observe
  already-started disturbances `(t_start <= step)`, instantaneous transit progress /
  speed scaling, and a fixed causal recovery horizon (`est_clear_horizon_steps`),
  never exposing true future clearance instants `(t_end)`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import hashlib
import random
from typing import Any, Optional

from amr_sched.cvcr.execution_feedback import (
    AmrExecutionStatus,
    CorridorFeedback,
    ExecutionFeedbackPacket,
)
from amr_sched.network import Network


@dataclass(frozen=True)
class P2PerturbationConfig:
    """Configuration for P2 L1/L2 stochastic execution disturbances."""

    seed: int = 0
    enable_travel_drift: bool = True
    drift_prob: float = 0.28
    early_max_steps: int = 1
    late_max_steps: int = 2
    enable_dynamic_obstacles: bool = True
    obstacle_count: int = 2
    obstacle_duration_steps: int = 14
    obstacle_window_range: tuple[int, int] = (14, 72)
    est_clear_horizon_steps: int = 12
    enable_local_planner_fail: bool = True
    local_fail_count: int = 2
    local_fail_duration_steps: int = 10
    local_fail_window_range: tuple[int, int] = (18, 76)
    enable_p4_belief: bool = False
    enable_p4_voi_probe: bool = False
    obstacle_durations: Optional[tuple[int, ...]] = None


@dataclass(frozen=True)
class ScheduledObstacle:
    """Scheduled exogenous dynamic obstacle on an undirected corridor `(u, v)`."""

    u: str
    v: str
    start_step: int
    end_step: int


@dataclass(frozen=True)
class ScheduledLocalFail:
    """Scheduled transient L1 local-planner / localization recovery window."""

    agv_id: str
    start_step: int
    end_step: int


class ExecutionPerturbationModel:
    """Seeded, replay-deterministic L1/L2 execution disturbance generator and telemetry bridge."""

    def __init__(self, config: P2PerturbationConfig | None = None) -> None:
        self.config = config or P2PerturbationConfig()
        self.obstacles: list[ScheduledObstacle] = []
        self.local_fails: list[ScheduledLocalFail] = []
        self._obs_map: dict[tuple[str, str], list[tuple[int, int]]] = {}
        self._fail_map: dict[str, list[tuple[int, int]]] = {}
        # agv_id -> (src, dst, depart_step, nominal_arrive, actual_arrive, speed_scale)
        self._active_transits: dict[str, tuple[str, str, int, int, int, float]] = {}
        self._recorded_obs_hits: set[tuple[str, str, str, int]] = set()
        self._recorded_fail_starts: set[tuple[str, int]] = set()
        self.deviation_log: list[dict[str, Any]] = []
        self._belief_model: Optional[Any] = None
        if self.config.enable_p4_belief:
            from amr_sched.cvcr.p4_belief_choke import CorridorDurationBeliefModel
            self._belief_model = CorridorDurationBeliefModel(
                enable_voi_probe=bool(self.config.enable_p4_voi_probe)
            )
        self._stats: dict[str, int] = {
            "drift_events": 0,
            "early_hops": 0,
            "late_hops": 0,
            "total_drift_steps": 0,
            "obstacle_blocks_hit": 0,
            "local_fail_ticks": 0,
            "local_fail_events": 0,
        }

    @staticmethod
    def _undir_key(u: str, v: str) -> tuple[str, str]:
        return (u, v) if u <= v else (v, u)

    @staticmethod
    def _shortest_path_nodes(
        adj: dict[str, list[str]],
        start: str,
        goal: str,
        blocked: set[tuple[str, str]],
    ) -> Optional[list[str]]:
        if start == goal:
            return [start]
        if start not in adj or goal not in adj:
            return None
        q: deque[str] = deque([start])
        parent: dict[str, Optional[str]] = {start: None}
        while q:
            cur = q.popleft()
            for nxt in adj.get(cur, ()):
                uk = ExecutionPerturbationModel._undir_key(cur, nxt)
                if uk in blocked or nxt in parent:
                    continue
                parent[nxt] = cur
                if nxt == goal:
                    out = [goal]
                    p = cur
                    while p is not None:
                        out.append(p)
                        p = parent[p]
                    out.reverse()
                    return out
                q.append(nxt)
        return None

    def _build_undir_adj(self, network: Network) -> dict[str, list[str]]:
        adj: dict[str, set[str]] = {nid: set() for nid in network.nodes}
        for e in network.edges.values():
            if e.src in adj and e.dst in adj and e.src != e.dst:
                adj[e.src].add(e.dst)
                adj[e.dst].add(e.src)
        return {k: sorted(v) for k, v in adj.items()}

    def _keeps_tasks_connected(
        self,
        adj: dict[str, list[str]],
        pairs: list[tuple[str, str]],
        blocked: set[tuple[str, str]],
    ) -> bool:
        for src, dst in pairs:
            if self._shortest_path_nodes(adj, src, dst, blocked) is None:
                return False
        return True

    def reset(
        self,
        network: Network,
        tasks: list[Any],
        agvs: list[Any],
        *,
        excluded_agvs: tuple[str, ...] | list[str] = (),
    ) -> None:
        """Deterministically sample corridor obstacles & L1 fails verified to preserve task feasibility."""
        self.obstacles.clear()
        self.local_fails.clear()
        self._obs_map.clear()
        self._fail_map.clear()
        self._active_transits.clear()
        self._recorded_obs_hits.clear()
        self._recorded_fail_starts.clear()
        self.deviation_log.clear()
        if self._belief_model is not None:
            self._belief_model.reset()
        for k in self._stats:
            self._stats[k] = 0

        cfg = self.config
        rng = random.Random(int(cfg.seed) * 10007 + len(tasks) * 31 + len(agvs) * 7)
        adj = self._build_undir_adj(network)

        required_pairs: set[tuple[str, str]] = set()
        for t in tasks:
            required_pairs.add((t.pickup, t.dropoff))
        start_nodes = sorted({a.start_node for a in agvs if getattr(a, "start_node", None)})
        for t in tasks[: min(len(tasks), 8)]:
            for sn in start_nodes:
                required_pairs.add((sn, t.pickup))
        pair_list = sorted(required_pairs)

        # 1. Dynamic corridor obstacles on bypassable high-traffic corridors
        if cfg.enable_dynamic_obstacles and cfg.obstacle_count > 0 and pair_list:
            edge_hits: dict[tuple[str, str], int] = {}
            empty_blk: set[tuple[str, str]] = set()
            for s_nd, g_nd in pair_list:
                sp = self._shortest_path_nodes(adj, s_nd, g_nd, empty_blk)
                if sp and len(sp) >= 2:
                    for i in range(len(sp) - 1):
                        ek = self._undir_key(sp[i], sp[i + 1])
                        edge_hits[ek] = edge_hits.get(ek, 0) + 1

            # Fall back to any non-bridge edge if no shortest path hit was collected
            if not edge_hits:
                for u, nbrs in adj.items():
                    for v in nbrs:
                        ek = self._undir_key(u, v)
                        edge_hits.setdefault(ek, 1)

            ranked_edges = sorted(edge_hits.items(), key=lambda kv: (-kv[1], kv[0]))
            bypassable: list[tuple[str, str]] = []
            for ek, _cnt in ranked_edges:
                if self._keeps_tasks_connected(adj, pair_list, {ek}):
                    bypassable.append(ek)

            if bypassable:
                lo, hi = int(cfg.obstacle_window_range[0]), int(cfg.obstacle_window_range[1])
                base_dur = max(2, int(cfg.obstacle_duration_steps))
                n_obs = int(cfg.obstacle_count)
                span = max(base_dur + 4, (hi - lo) // max(1, n_obs))
                chosen_edges: list[tuple[str, str]] = []
                # Pick distinct top-ranked bypassable edges across top-K pool with stagger
                pool = bypassable[: max(len(bypassable), min(len(bypassable), n_obs * 3))]
                for k in range(n_obs):
                    dur = (
                        max(2, int(cfg.obstacle_durations[k % len(cfg.obstacle_durations)]))
                        if cfg.obstacle_durations
                        else base_dur
                    )
                    w_lo = lo + k * span
                    w_hi = max(w_lo, min(hi, w_lo + max(4, span - dur)))
                    t_start = rng.randint(w_lo, w_hi)
                    t_end = t_start + dur
                    # Prefer distinct corridor across staggered windows
                    cand_choices = [e for e in pool[: max(1, min(len(pool), 6))] if e not in chosen_edges] or pool
                    ek = cand_choices[rng.randrange(len(cand_choices))]
                    chosen_edges.append(ek)
                    # Check if overlapping with any active obstacle preserves connectivity
                    simultaneous = {ek} | {
                        self._undir_key(o.u, o.v)
                        for o in self.obstacles
                        if not (t_end <= o.start_step or t_start >= o.end_step)
                    }
                    if len(simultaneous) > 1 and not self._keeps_tasks_connected(adj, pair_list, simultaneous):
                        continue
                    ob = ScheduledObstacle(u=ek[0], v=ek[1], start_step=t_start, end_step=t_end)
                    self.obstacles.append(ob)
                    self._obs_map.setdefault(ek, []).append((t_start, t_end))

        # 2. Transient L1 local-planner / localization watchdog failures
        if cfg.enable_local_planner_fail and cfg.local_fail_count > 0 and agvs:
            excl = set(excluded_agvs)
            cand_agvs = [a.agv_id for a in agvs if a.agv_id not in excl] or [a.agv_id for a in agvs]
            cand_agvs = sorted(cand_agvs)
            lo_f, hi_f = int(cfg.local_fail_window_range[0]), int(cfg.local_fail_window_range[1])
            dur_f = max(2, int(cfg.local_fail_duration_steps))
            n_f = int(cfg.local_fail_count)
            span_f = max(dur_f + 4, (hi_f - lo_f) // max(1, n_f))
            rng.shuffle(cand_agvs)
            for k in range(n_f):
                w_lo = lo_f + k * span_f
                w_hi = max(w_lo, min(hi_f, w_lo + max(4, span_f - dur_f)))
                t_start = rng.randint(w_lo, w_hi)
                aid = cand_agvs[k % len(cand_agvs)]
                lf = ScheduledLocalFail(agv_id=aid, start_step=t_start, end_step=t_start + dur_f)
                self.local_fails.append(lf)
                self._fail_map.setdefault(aid, []).append((lf.start_step, lf.end_step))

    def _hop_rng(self, agv_id: str, src: str, dst: str, depart_step: int) -> random.Random:
        payload = f"{int(self.config.seed)}|{agv_id}|{src}|{dst}|{int(depart_step)}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return random.Random(int.from_bytes(digest, "little"))

    def sample_hop_steps(
        self,
        agv_id: str,
        src: str,
        dst: str,
        depart_step: int,
        nominal_steps: int,
    ) -> int:
        """Sample actual realized hop traversal duration for one edge transition."""
        n_nom = max(1, int(nominal_steps))
        cfg = self.config
        if not cfg.enable_travel_drift or cfg.drift_prob <= 0.0:
            return n_nom
        hrng = self._hop_rng(agv_id, src, dst, depart_step)
        if hrng.random() >= float(cfg.drift_prob):
            return n_nom
        # Bias slightly toward cautionary slowdowns (~68% late, ~32% early when n_nom >= 2)
        if n_nom >= 2 and cfg.early_max_steps >= 1 and hrng.random() < 0.32:
            delta = -hrng.randint(1, min(int(cfg.early_max_steps), n_nom - 1))
            self._stats["early_hops"] += 1
        else:
            delta = hrng.randint(1, max(1, int(cfg.late_max_steps)))
            self._stats["late_hops"] += 1
        self._stats["drift_events"] += 1
        self._stats["total_drift_steps"] += abs(delta)
        return max(1, n_nom + delta)

    def record_transit(
        self,
        agv_id: str,
        src: str,
        dst: str,
        depart_step: int,
        nominal_steps: int,
        actual_steps: int,
    ) -> None:
        """Record live hop dispatch for causal transit velocity observation."""
        n_nom = max(1, int(nominal_steps))
        n_act = max(1, int(actual_steps))
        scale = float(n_nom) / float(n_act)
        self._active_transits[agv_id] = (
            src,
            dst,
            int(depart_step),
            int(depart_step) + n_nom,
            int(depart_step) + n_act,
            scale,
        )

    def bump_transit_arrival(self, agv_id: str, new_arrive: int) -> None:
        """Update tracked transit arrival instant when stalled mid-edge."""
        rec = self._active_transits.get(agv_id)
        if rec is None:
            return
        src, dst, dep, nom_arr, _act_arr, scale = rec
        self._active_transits[agv_id] = (src, dst, dep, nom_arr, int(new_arrive), min(scale, 0.25))

    def clear_transit(self, agv_id: str) -> None:
        """Remove completed or faulted transit entry."""
        self._active_transits.pop(agv_id, None)

    def is_corridor_blocked(self, src: str, dst: str, step: int) -> bool:
        """Whether undirected corridor `(src, dst)` is physically blocked at `step`."""
        spans = self._obs_map.get(self._undir_key(src, dst))
        if not spans:
            return False
        s = int(step)
        return any(st <= s < en for st, en in spans)

    def is_local_stalled(self, agv_id: str, step: int) -> bool:
        """Whether `agv_id` is experiencing an active L1 local-planner recovery hold at `step`."""
        spans = self._fail_map.get(agv_id)
        if not spans:
            return False
        s = int(step)
        return any(st <= s < en for st, en in spans)

    def on_blocked_entry(self, agv_id: str, src: str, dst: str, step: int) -> None:
        """Record a blocked-corridor delay event when an AGV tries to enter a disturbed edge."""
        self._stats["obstacle_blocks_hit"] += 1
        ek = self._undir_key(src, dst)
        for st, _en in self._obs_map.get(ek, ()):
            if st <= step < _en:
                sig = (agv_id, ek[0], ek[1], st)
                if sig not in self._recorded_obs_hits:
                    self._recorded_obs_hits.add(sig)
                    self.deviation_log.append(
                        {
                            "category": "ROUTE_DEVIATION",
                            "agv_id": agv_id,
                            "edge": ek,
                            "step": int(step),
                            "obstacle_start": int(st),
                        }
                    )
                break

    def on_local_fail_tick(self, agv_id: str, step: int) -> None:
        """Record one L1 local-planner recovery hold tick."""
        self._stats["local_fail_ticks"] += 1
        for st, en in self._fail_map.get(agv_id, ()):
            if st <= step < en:
                sig = (agv_id, st)
                if sig not in self._recorded_fail_starts:
                    self._recorded_fail_starts.add(sig)
                    self._stats["local_fail_events"] += 1
                    self.deviation_log.append(
                        {
                            "category": "LOCAL_PLANNER_FAIL",
                            "agv_id": agv_id,
                            "step": int(step),
                            "fail_start": int(st),
                        }
                    )
                break

    def capture_feedback(self, sim: Any, step: int) -> Optional[ExecutionFeedbackPacket]:
        """Build a strictly causal `ExecutionFeedbackPacket` from active observations at `step`.

        No future oracle leakage:
          - Blocked corridors advertise a fixed rolling sensor horizon `(step + est_clear_horizon_steps)`,
            never reading true `end_step`.
          - L1 `local_failed` reports elapsed timeout age & causal remaining prior estimate.
          - In-flight travel drift uses only elapsed steps and instantaneous observed `speed_scale`.
        """
        s = int(step)
        hz = max(2, int(self.config.est_clear_horizon_steps))
        robot_states: dict[str, AmrExecutionStatus] = {}
        corridor_updates: dict[tuple[str, str], CorridorFeedback] = {}
        events: list[str] = []

        # Active dynamic obstacles visible at step s (causal: end_step > s never exposed)
        active_blk = {
            self._undir_key(ob.u, ob.v)
            for ob in self.obstacles
            if ob.start_step <= s < ob.end_step
        }
        if self._belief_model is not None:
            self._belief_model.observe_step(s, active_blk)

        for ob in self.obstacles:
            if ob.start_step <= s < ob.end_step:
                if self._belief_model is not None and getattr(sim, "network", None) is not None:
                    corridor_updates[(ob.u, ob.v)] = self._belief_model.build_corridor_feedback(
                        sim.network, ob.u, ob.v, s, source="p4_belief_lidar", reason="dynamic_obstacle"
                    )
                    corridor_updates[(ob.v, ob.u)] = self._belief_model.build_corridor_feedback(
                        sim.network, ob.v, ob.u, s, source="p4_belief_lidar", reason="dynamic_obstacle"
                    )
                else:
                    cf = CorridorFeedback(
                        src=ob.u,
                        dst=ob.v,
                        status="blocked",
                        extra_delay_steps=hz,
                        blocked_until_step=s + hz,
                        source="p2_obstacle_lidar",
                        reason="dynamic_obstacle",
                    )
                    corridor_updates[(ob.u, ob.v)] = cf
                    corridor_updates[(ob.v, ob.u)] = CorridorFeedback(
                        src=ob.v,
                        dst=ob.u,
                        status="blocked",
                        extra_delay_steps=hz,
                        blocked_until_step=s + hz,
                        source="p2_obstacle_lidar",
                        reason="dynamic_obstacle",
                    )
                events.append(f"ROUTE_DEVIATION:{ob.u}<->{ob.v}@{s}")

        # Active L1 local-planner fails + causal in-flight speed drift
        for agv in getattr(sim, "agvs", ()):
            aid = agv.agv_id
            fail_start: Optional[int] = None
            for st, en in self._fail_map.get(aid, ()):
                if st <= s < en:
                    fail_start = st
                    break
            tr_info = self._active_transits.get(aid)
            if fail_start is not None:
                elapsed = max(0, s - fail_start)
                est_rem = max(2, hz - min(hz - 2, elapsed))
                tr_leg = (tr_info[0], tr_info[1]) if tr_info is not None else None
                robot_states[aid] = AmrExecutionStatus(
                    robot_id=aid,
                    localization_source="amcl",
                    active_leg=tr_leg,
                    eta_drift_steps=int(est_rem),
                    speed_scale=0.05,
                    local_planner_state="local_failed",
                    receipt_age_s=2.0 + 0.5 * float(elapsed),
                    fault_reason="l1_local_planner_recovery",
                )
                events.append(f"LOCAL_FAIL:{aid}@{s}")
            elif tr_info is not None:
                src, dst, dep, nom_arr, _act_arr, scale = tr_info
                # Strictly causal velocity observation only AFTER departure (s > dep)
                if s > dep and scale < 0.99:
                    est_total = max(1, int(round(float(nom_arr - dep) / max(0.25, scale))))
                    causal_drift = max(1, min(6, (dep + est_total) - nom_arr))
                    prog = min(0.95, float(s - dep) / float(max(1, est_total)))
                    robot_states[aid] = AmrExecutionStatus(
                        robot_id=aid,
                        localization_source="amcl",
                        active_leg=(src, dst),
                        leg_progress=prog,
                        eta_drift_steps=int(causal_drift),
                        speed_scale=float(round(scale, 3)),
                        local_planner_state="slowed",
                        receipt_age_s=0.1,
                    )

        if not robot_states and not corridor_updates:
            return None
        return ExecutionFeedbackPacket(
            tick=s,
            robot_states=robot_states,
            corridor_updates=corridor_updates,
            source_events=tuple(sorted(set(events))),
        )

    def active_disturbance_sig(
        self,
        sim: Any,
        step: int,
    ) -> dict[str, Any]:
        """Compute strictly causal active P2 disturbance flags for execution-aware wakeups."""
        s = int(step)
        blocked_undir = {
            self._undir_key(ob.u, ob.v)
            for ob in self.obstacles
            if ob.start_step <= s < ob.end_step
        }
        trig_blocked_undir = set(blocked_undir)
        if self._belief_model is not None and getattr(sim, "network", None) is not None:
            from amr_sched.cvcr.p4_belief_choke import corridor_detour_overhead
            self._belief_model.observe_step(s, blocked_undir)
            trig_blocked_undir = set()
            for ek in blocked_undir:
                est = self._belief_model.estimate_remaining(
                    ek, s, detour_overhead_steps=corridor_detour_overhead(sim.network, ek[0], ek[1])
                )
                if (not est.prefer_wait_over_detour) or est.expected_remaining >= 14.0:
                    trig_blocked_undir.add(ek)
        failed_agvs = {
            lf.agv_id
            for lf in self.local_fails
            if lf.start_step <= s < lf.end_step
        }
        jobs = getattr(sim, "_jobs", {})
        tmap = getattr(sim, "_task_by_id", {})
        transit = getattr(sim, "_transit", {})
        faulted = getattr(sim, "_faulted_agvs", set())

        blocked_picked_agvs: list[str] = []
        blocked_assigned_agvs: list[str] = []
        fail_assigned_agvs: list[str] = []
        fail_picked_agvs: list[str] = []

        for agv in getattr(sim, "agvs", ()):
            aid = agv.agv_id
            if aid in faulted:
                continue
            job = jobs.get(aid)
            task = tmap.get(job.task_id) if (job is not None and job.task_id) else None
            t_stat = (
                getattr(task.status, "value", str(task.status)).lower()
                if task is not None
                else ""
            )
            if aid in failed_agvs and task is not None:
                if t_stat == "assigned":
                    fail_assigned_agvs.append(aid)
                elif t_stat == "picked":
                    fail_picked_agvs.append(aid)
            if (
                trig_blocked_undir
                and aid not in transit
                and job is not None
                and task is not None
                and job.idx < len(job.hops)
            ):
                upcoming = job.hops[job.idx : job.idx + 3]
                if any(self._undir_key(h.src, h.dst) in trig_blocked_undir for h in upcoming):
                    if t_stat == "picked":
                        blocked_picked_agvs.append(aid)
                    elif t_stat == "assigned":
                        blocked_assigned_agvs.append(aid)

        return {
            "blocked_edges": tuple(sorted(blocked_undir)),
            "failed_agvs": tuple(sorted(failed_agvs)),
            "blocked_picked_agvs": tuple(sorted(blocked_picked_agvs)),
            "blocked_assigned_agvs": tuple(sorted(blocked_assigned_agvs)),
            "fail_assigned_agvs": tuple(sorted(fail_assigned_agvs)),
            "fail_picked_agvs": tuple(sorted(fail_picked_agvs)),
        }

    def summary(self) -> dict[str, Any]:
        """Return compact episode disturbance counters for telemetry & ablation reports."""
        return {
            "obstacle_count": len(self.obstacles),
            "local_fail_count": len(self.local_fails),
            **{k: int(v) for k, v in self._stats.items()},
        }
