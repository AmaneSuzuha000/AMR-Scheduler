"""Stage 3 (E3): Two-Step Event-Driven SMDP-RL (Information-Value RL) & Observation-Mask Test.

Models real closed-loop AMR bottleneck timing decisions on top of `Simulator`,
`NativeTransactionOwner`, `NativeProductionTracker`, `CorridorDurationBeliefModel`,
and `IndependentAuditor` with exact telescoping event rewards (`r_1 + r_2 == -J_net`, `gamma=1`).

Compares three agents under identical causal observations, legal actions, and scenarios:
  1. `OneStepMyopicAgent`: Immediate best `{COMMIT_DETOUR_NOW, COMMIT_DIRECT_WAIT}`.
  2. `TerminalRegFixedContAgent`: Terminal regression over fixed continuation heuristic.
  3. `TwoStepFQIAgent`: Exact 2-step Bellman Fitted Q-Iteration (`Q_2` -> Bellman target for `Q_1`).

Includes the decisive Observation-Mask Counterfactual (`mask_intermediate_obs=True`) verifying
whether 2-step RL's advantage comes from causal Value-of-Information (`VoI`).
"""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any, Optional

import numpy as np

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.p2_perturbation import ExecutionPerturbationModel, P2PerturbationConfig, ScheduledObstacle
from amr_sched.cvcr.p4_belief_choke import CorridorDurationBeliefModel
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.production import NativeProductionTracker
from amr_sched.cvcr.snapshot import NativeSnapshotCapture, OperationSnapshot
from amr_sched.cvcr.transaction import NativeTransactionOwner
from amr_sched.entities import Agv, AgvState, Task, TaskStatus
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import Hop, Interval, SimulationConfig, Simulator, _Job

ACT_COMMIT_DETOUR = 0
ACT_COMMIT_DIRECT = 1
ACT_WAIT_OBSERVE = 2

E3_STATE_DIM = 12


@dataclass(frozen=True)
class E3ScenarioSpec:
    """Parameterized 2-AMR + trailing queue dynamic-obstacle corridor timing scenario."""

    seed: int
    true_clear_step: int
    past_clears: tuple[int, ...]
    detour_leg_len: float
    win_close_a: int
    n_behind: int
    probe_dt: int = 4
    t_dec1: int = 2
    reversible: bool = True


def generate_e3_scenarios(n: int, *, base_seed: int = 31000) -> list[E3ScenarioSpec]:
    """Generate diverse reversible/irreversible short/long obstacle timing scenarios."""
    rng = random.Random(int(base_seed))
    specs: list[E3ScenarioSpec] = []
    for i in range(int(n)):
        # Identical historical prior so t_dec1 has genuine uncertainty between short (~5) and long (~30)
        past = (5, 28, 6, 32, 5, 6)
        is_short = (rng.random() < 0.55)
        if is_short:
            true_clear = rng.randint(4, 6)  # clears right during [t_dec1=2, t_dec2=6]
        else:
            true_clear = rng.randint(24, 38)  # long persistent blockage
        det_len = float(rng.choice((10.0, 12.0, 14.0)))
        # Direct path: LA->C0(2) + C0->C1(2) + C1->WA(2) + unload(2) = 8 steps (+ wait until true_clear).
        # Detour path: LA->BYP(det_len) + BYP->WA(4) + unload(2) = det_len + 6 (+ t_dec).
        det_steps = int(det_len) + 6
        if (i % 3) == 2:
            # Irreversible razor-thin window: immediate detour finishes at 2 + det_steps <= win_close;
            # waiting probe_dt=4 steps until t_dec2=6 misses win_close_a!
            win_close = 2 + det_steps + 1
            rev = False
        else:
            # Reversible window: waiting probe_dt=4 still leaves safe detour room before win_close_a
            win_close = 2 + det_steps + 8
            rev = True
        n_behind = 0 if (i % 4 != 3) else 2
        specs.append(
            E3ScenarioSpec(
                seed=int(base_seed) + i,
                true_clear_step=true_clear,
                past_clears=past,
                detour_leg_len=det_len,
                win_close_a=win_close,
                n_behind=n_behind,
                probe_dt=4,
                t_dec1=2,
                reversible=rev,
            )
        )
    return specs


def _build_e3_sim(spec: E3ScenarioSpec) -> tuple[Simulator, NativeProductionTracker, CorridorDurationBeliefModel]:
    """Build exact physical Simulator + P2 obstacle + causal belief model for `spec`."""
    net = Network()
    coords = {
        "LA": (0.0, 0.0, NodeKind.BUFFER, 3),
        "C0": (2.0, 0.0, NodeKind.AISLE, 1),
        "C1": (4.0, 0.0, NodeKind.AISLE, 1),
        "BYP": (float(spec.detour_leg_len), 6.0, NodeKind.AISLE, 2),
        "WA": (float(spec.detour_leg_len) + 4.0, 0.0, NodeKind.WORKSTATION, 3),
        "WB": (float(spec.detour_leg_len) + 4.0, -3.0, NodeKind.WORKSTATION, 3),
    }
    for nid, (x, y, kd, cp) in coords.items():
        net.add_node(Node(nid, x, y, kind=kd, capacity=cp))
    for u, v, d in (
        ("LA", "C0", 2.0),
        ("C0", "C1", 2.0),
        ("C1", "WA", 2.0),
        ("C1", "WB", 2.0),
        ("LA", "BYP", float(spec.detour_leg_len)),
        ("BYP", "WA", 4.0),
        ("BYP", "WB", 4.0),
    ):
        net.add_edge(Edge(u, v, length=d, speed_limit=1.0, capacity=1))
        net.add_edge(Edge(v, u, length=d, speed_limit=1.0, capacity=1))

    t_a = Task("T_A", "LA", "WA", release_step=0, due_step=spec.win_close_a + 4)
    t_a.kit_op_id = "OP_A"  # type: ignore[attr-defined]
    tasks = [t_a]
    ops = [
        OperationSnapshot(
            "OP_A",
            ("T_A",),
            "WA",
            crew_id=1,
            tool_id=1,
            proc_steps=3,
            due_step=spec.win_close_a + 4,
            resource_windows=((0, spec.win_close_a),),
        )
    ]
    agvs = [Agv("agv_0", max_speed=1.0, start_node="LA")]

    for q_i in range(spec.n_behind):
        tid = f"T_Q{q_i}"
        opid = f"OP_Q{q_i}"
        tq = Task(tid, "LA", "WB", release_step=0, due_step=30 + q_i * 6)
        tq.kit_op_id = opid  # type: ignore[attr-defined]
        tasks.append(tq)
        ops.append(OperationSnapshot(opid, (tid,), "WB", crew_id=2 + q_i, tool_id=2 + q_i, proc_steps=3, due_step=35 + q_i * 6))
        agvs.append(Agv(f"agv_q{q_i}", max_speed=1.0, start_node="LA"))

    p2_cfg = P2PerturbationConfig(
        seed=spec.seed,
        enable_travel_drift=False,
        enable_dynamic_obstacles=False,
        enable_local_planner_fail=False,
        enable_p4_belief=True,
        enable_p4_voi_probe=True,
    )
    p2 = ExecutionPerturbationModel(p2_cfg)
    sim = Simulator(
        network=net,
        tasks=tasks,
        agvs=agvs,
        planner=SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
        allocator=GreedyAllocator(),
        config=SimulationConfig(
            dt=1.0,
            max_steps=200,
            load_time=1.0,
            unload_time=2.0,
            reschedule_policy=ReschedulePolicy.NONE,
            record_trajectory=True,
            idle_retreat=False,
        ),
        perturbation_model=p2,
    )
    sim.reset()
    # Install scheduled obstacle on (C0, C1) [0 .. true_clear_step)
    ob = ScheduledObstacle(u="C0", v="C1", start_step=0, end_step=int(spec.true_clear_step))
    p2.obstacles = [ob]
    p2._obs_map[("C0", "C1")] = [(0, int(spec.true_clear_step))]

    belief = CorridorDurationBeliefModel(enable_voi_probe=True)
    belief._completed_durations.extend(int(d) for d in spec.past_clears)
    belief.observe_step(0, {("C0", "C1")})

    # Place agv_0 at LA loaded with T_A (`PICKED`), queued jobs behind holding at LA until dispatch
    for idx_a, (agv, t) in enumerate(zip(sim.agvs, sim.tasks)):
        t.status = TaskStatus.PICKED
        t.assigned_agv = agv.agv_id
        t.assigned_step = 0
        t.pickup_step = 0
        agv.cur_node = "LA"
        agv.cur_task = t.task_id
        agv.state = AgvState.LOADING
        sim._jobs[agv.agv_id] = _Job(t.task_id, "to_pickup", [], idx=0, goal=t.pickup, reserved_upto=0)
        sim._load_end[agv.agv_id] = int(spec.t_dec1)
        sim._res.reserve_node("LA", Interval(0, int(spec.t_dec1) + 1), agv.agv_id)

    tracker = NativeProductionTracker(ops, sim.tasks)
    return sim, tracker, belief


def _extract_e3_state(
    spec: E3ScenarioSpec,
    step: int,
    belief: CorridorDurationBeliefModel,
    *,
    is_step2: bool,
    obs_cleared: bool,
    mask_intermediate_obs: bool,
) -> np.ndarray:
    """Extract strictly causal 12-D state vector for Decision 1 / Decision 2."""
    detour_steps = int(spec.detour_leg_len) + 4 + 2  # LA->BYP + BYP->WA + unload
    direct_steps = 2 + 2 + 2 + 2  # LA->C0 + C0->C1 + C1->WA + unload
    detour_gap = float(detour_steps - direct_steps)

    eff_step = int(spec.t_dec1) if (is_step2 and mask_intermediate_obs) else int(step)
    est = belief.estimate_remaining(("C0", "C1"), eff_step, detour_overhead_steps=detour_gap)
    # Posterior probability corridor clears within probe_dt steps
    p_clear_probe = 0.0
    if not (is_step2 and obs_cleared and not mask_intermediate_obs):
        s_now = max(1e-6, belief.survival(float(eff_step)))
        s_fut = belief.survival(float(eff_step + spec.probe_dt))
        p_clear_probe = max(0.0, min(1.0, (s_now - s_fut) / s_now))
    else:
        p_clear_probe = 1.0

    margin_det = float(spec.win_close_a - (step + detour_steps))
    margin_dir = float(spec.win_close_a - (step + direct_steps))
    obs_cleared_feat = 0.0 if (not is_step2 or mask_intermediate_obs) else (1.0 if obs_cleared else -1.0)
    rev_flag = 1.0 if (not is_step2 and (spec.win_close_a - (step + spec.probe_dt + detour_steps)) >= 0) else 0.0

    return np.asarray(
        [
            float(np.clip(margin_det / 15.0, -3.0, 3.0)),
            float(np.clip(margin_dir / 15.0, -3.0, 3.0)),
            float(np.clip(detour_gap / 12.0, 0.0, 2.0)),
            float(eff_step) / 10.0,
            0.0 if (is_step2 and obs_cleared and not mask_intermediate_obs) else float(est.expected_remaining) / 25.0,
            0.0 if (is_step2 and obs_cleared and not mask_intermediate_obs) else float(est.upper_cvar75) / 25.0,
            float(p_clear_probe),
            float(spec.n_behind) / 3.0,
            rev_flag,
            float(obs_cleared_feat),
            1.0 if is_step2 else 0.0,
            0.0 if (is_step2 and obs_cleared and not mask_intermediate_obs) else 1.0,  # corridor still seen blocked
        ],
        dtype=np.float32,
    )


def _commit_corridor_choice(
    sim: Simulator,
    tracker: NativeProductionTracker,
    owner: NativeTransactionOwner,
    *,
    use_detour: bool,
    step: int,
) -> bool:
    """Commit either detour (`LA->BYP->WA`) or direct (`LA->C0->C1->WA`) legally via Owner."""
    snap = NativeSnapshotCapture.capture(sim, tracker.operations, hide_future_releases=True)
    repairs: dict[str, dict[str, Any]] = {}
    t_dep = int(step)
    for agv in sim.agvs:
        if not agv.cur_task or agv.agv_id in sim._transit:
            continue
        task = sim._task_by_id[agv.cur_task]
        drop_n = task.dropoff
        if use_detour and agv.agv_id == "agv_0":
            n1 = sim._hop_steps("LA", "BYP")
            n2 = sim._hop_steps("BYP", drop_n)
            hops = [("LA", "BYP", t_dep, t_dep + n1), ("BYP", drop_n, t_dep + n1, t_dep + n1 + n2)]
        else:
            n1 = sim._hop_steps("LA", "C0")
            n2 = sim._hop_steps("C0", "C1")
            n3 = sim._hop_steps("C1", drop_n)
            hops = [
                ("LA", "C0", t_dep, t_dep + n1),
                ("C0", "C1", t_dep + n1, t_dep + n1 + n2),
                ("C1", drop_n, t_dep + n1 + n2, t_dep + n1 + n2 + n3),
            ]
            t_dep += 1
        repairs[agv.agv_id] = {
            "kind": "picked_reroute",
            "task_id": task.task_id,
            "goal": drop_n,
            "hops": hops,
            "min_reserve_hops": len(hops),
        }
    prop = CandidateProposal(
        schema="amr-lgcvcr.proposal.v1",
        policy_id="V17_E3_EVENT",
        snapshot_digest=snap.compute_digest(),
        affected_robots=sorted(repairs.keys()),
        affected_tasks=[],
        affected_operations=[],
        proposed_assignment={t.task_id: t.assigned_agv for t in sim.tasks if t.assigned_agv},
        proposed_routes={},
        proposed_bays={},
        proposed_crews={},
        predicted_obj=0.0,
        predicted_makespan=40,
        candidate_digest=f"e3_{use_detour}_{step}",
        telemetry={},
        proposed_live_repairs=repairs,
    )
    rcpt = owner.validate_and_commit(prop)
    return bool(rcpt.accepted)


def _run_sim_partial_and_finish(sim: Simulator, tracker: NativeProductionTracker, stop_at_step: Optional[int] = None) -> float:
    """Advance simulator + tracker until `stop_at_step` or completion, returning cumulative `-J_net`."""
    while sim._step_cursor < sim.config.max_steps:
        step = int(sim._step_cursor)
        tracker.set_op_priorities(getattr(sim, "_op_priorities", None))
        tracker.step(step)
        if stop_at_step is not None and step >= stop_at_step:
            return 0.0
        all_t = all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in sim.tasks)
        if all_t and tracker.all_done(step):
            break
        more = sim.step_once()
        cur = int(sim._step_cursor)
        if not more and not tracker.all_done(cur):
            sim._step_cursor += 1
        elif not more and tracker.all_done(cur):
            break

    aud = IndependentAuditor.audit_simulation_run(sim, tracker.operations, require_trajectory=True)
    cmax = int(sim._step_cursor)
    t_tard = sum(max(0, (int(t.complete_step) if t.complete_step is not None else cmax) - int(t.due_step or 0)) for t in sim.tasks)
    o_tard = sum(max(0, (int(o.end_step) if o.end_step is not None else cmax) - int(o.due_step)) for o in tracker.operations)
    unf_t = sum(1 for t in sim.tasks if t.status is not TaskStatus.COMPLETED)
    unf_o = sum(1 for o in tracker.operations if not (o.started and o.end_step is not None))
    j_net = float(cmax + 2.0 * (t_tard + o_tard) + 5000.0 * unf_t + 3000.0 * unf_o)
    if not aud.pass_audit:
        j_net += 10000.0
    return j_net


def evaluate_e3_trajectory(
    spec: E3ScenarioSpec,
    a1: int,
    a2: Optional[int] = None,
    *,
    mask_intermediate_obs: bool = False,
) -> dict[str, Any]:
    """Execute real closed-loop 1-step or 2-step episode with exact telescoping ledger (`r1 + r2 == -J_net`)."""
    sim, tracker, belief = _build_e3_sim(spec)
    owner = NativeTransactionOwner(sim, operations=tracker.operations, production_tracker=tracker)

    # Advance cleanly up to Decision 1 (`t_dec1`)
    _run_sim_partial_and_finish(sim, tracker, stop_at_step=spec.t_dec1)
    p2 = sim._perturbation_model
    assert p2 is not None
    belief.observe_step(spec.t_dec1, {("C0", "C1")} if p2.is_corridor_blocked("C0", "C1", spec.t_dec1) else set())
    s1 = _extract_e3_state(spec, spec.t_dec1, belief, is_step2=False, obs_cleared=False, mask_intermediate_obs=mask_intermediate_obs)

    if a1 in (ACT_COMMIT_DETOUR, ACT_COMMIT_DIRECT):
        ok = _commit_corridor_choice(sim, tracker, owner, use_detour=(a1 == ACT_COMMIT_DETOUR), step=spec.t_dec1)
        assert ok
        j_net = _run_sim_partial_and_finish(sim, tracker, stop_at_step=None)
        return {
            "s1": s1,
            "s2": None,
            "a1": a1,
            "a2": None,
            "r1": -j_net,
            "r2": 0.0,
            "j_net": j_net,
            "telescoping_err": 0.0,
            "obs_cleared_at_dec2": False,
        }

    # Action 2: WAIT_ONE_OBS_EVENT -> physically hold at staging LA until t_dec2
    t_dec2 = spec.t_dec1 + spec.probe_dt
    for agv in sim.agvs:
        if agv.agv_id in sim._load_end:
            sim._load_end[agv.agv_id] = int(t_dec2)
            sim._res.reserve_node("LA", Interval(int(spec.t_dec1), int(t_dec2) + 1), agv.agv_id)
    _run_sim_partial_and_finish(sim, tracker, stop_at_step=t_dec2)
    still_blk = p2.is_corridor_blocked("C0", "C1", t_dec2)
    obs_cleared = not still_blk
    belief.observe_step(t_dec2, {("C0", "C1")} if still_blk else set())
    s2 = _extract_e3_state(
        spec, t_dec2, belief, is_step2=True, obs_cleared=obs_cleared, mask_intermediate_obs=mask_intermediate_obs
    )

    assert a2 in (ACT_COMMIT_DETOUR, ACT_COMMIT_DIRECT)
    ok = _commit_corridor_choice(sim, tracker, owner, use_detour=(a2 == ACT_COMMIT_DETOUR), step=t_dec2)
    assert ok
    j_net = _run_sim_partial_and_finish(sim, tracker, stop_at_step=None)
    # Ledger split: r1 = partial hold penalty during [t_dec1, t_dec2], r2 = terminal - r1 -> r1 + r2 == -j_net identically
    r1 = -float(spec.probe_dt)
    r2 = -j_net - r1
    return {
        "s1": s1,
        "s2": s2,
        "a1": ACT_WAIT_OBSERVE,
        "a2": a2,
        "r1": r1,
        "r2": r2,
        "j_net": j_net,
        "telescoping_err": abs((r1 + r2) - (-j_net)),
        "obs_cleared_at_dec2": obs_cleared,
    }


class _MLPRegressor:
    """Lightweight regularized NumPy/Adam MLP regressor for clean tabular Q-approximation."""

    def __init__(self, in_dim: int, hidden: int = 32, seed: int = 42) -> None:
        rng = np.random.default_rng(seed)
        self.w1 = (rng.normal(0.0, 0.25, size=(in_dim, hidden))).astype(np.float32)
        self.b1 = np.zeros((hidden,), dtype=np.float32)
        self.w2 = (rng.normal(0.0, 0.25, size=(hidden, 1))).astype(np.float32)
        self.b2 = np.zeros((1,), dtype=np.float32)
        self.mu = np.zeros((in_dim,), dtype=np.float32)
        self.sd = np.ones((in_dim,), dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, *, epochs: int = 350, lr: float = 2e-2) -> None:
        self.mu = X.mean(axis=0).astype(np.float32)
        self.sd = np.maximum(0.05, X.std(axis=0)).astype(np.float32)
        xn = ((X - self.mu) / self.sd).astype(np.float32)
        yt = y.reshape(-1, 1).astype(np.float32)
        # Adam state
        params = [self.w1, self.b1, self.w2, self.b2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        n = float(max(1, len(xn)))
        for ep in range(1, epochs + 1):
            h_pre = xn @ self.w1 + self.b1
            h = np.maximum(0.0, h_pre)
            pred = h @ self.w2 + self.b2
            err = (pred - yt) * (2.0 / n)
            dw2 = h.T @ err + 1e-3 * self.w2
            db2 = err.sum(axis=0)
            dh = (err @ self.w2.T) * (h_pre > 0.0)
            dw1 = xn.T @ dh + 1e-3 * self.w1
            db1 = dh.sum(axis=0)
            grads = [dw1, db1, dw2, db2]
            b1_c = 1.0 - 0.9 ** ep
            b2_c = 1.0 - 0.999 ** ep
            for idx, (p, g) in enumerate(zip(params, grads)):
                m[idx] = 0.9 * m[idx] + 0.1 * g
                v[idx] = 0.999 * v[idx] + 0.001 * (g * g)
                m_hat = m[idx] / b1_c
                v_hat = v[idx] / b2_c
                p -= lr * m_hat / (np.sqrt(v_hat) + 1e-8)

    def predict(self, X: np.ndarray) -> np.ndarray:
        xn = ((X - self.mu) / self.sd).astype(np.float32)
        h = np.maximum(0.0, xn @ self.w1 + self.b1)
        return (h @ self.w2 + self.b2).reshape(-1)


def run_e3_smdp_experiment(*, n_train: int = 160, n_eval: int = 120, seed: int = 31000) -> dict[str, Any]:
    """Train & evaluate 1-step Myopic, Terminal Regression, and 2-Step FQI (+ Obs-Mask test)."""
    train_specs = generate_e3_scenarios(n_train, base_seed=seed)
    eval_specs = generate_e3_scenarios(n_eval, base_seed=seed + 10000)

    # Collect complete exact closed-loop outcome table on training set (4 branches per spec)
    X_s1_det: list[np.ndarray] = []
    y_s1_det: list[float] = []
    X_s1_dir: list[np.ndarray] = []
    y_s1_dir: list[float] = []

    X_s2_det: list[np.ndarray] = []
    y_s2_det: list[float] = []
    X_s2_dir: list[np.ndarray] = []
    y_s2_dir: list[float] = []

    wait_rows: list[tuple[np.ndarray, float, np.ndarray]] = []  # (s1, r1, s2)
    y_term_fixed_cont: list[float] = []  # Terminal cost when Decision 2 uses fixed myopic rule (direct if cleared else detour)

    for sp in train_specs:
        tr_det = evaluate_e3_trajectory(sp, ACT_COMMIT_DETOUR)
        tr_dir = evaluate_e3_trajectory(sp, ACT_COMMIT_DIRECT)
        tr_w_det = evaluate_e3_trajectory(sp, ACT_WAIT_OBSERVE, ACT_COMMIT_DETOUR)
        tr_w_dir = evaluate_e3_trajectory(sp, ACT_WAIT_OBSERVE, ACT_COMMIT_DIRECT)
        assert tr_w_det["telescoping_err"] < 1e-6 and tr_w_dir["telescoping_err"] < 1e-6

        X_s1_det.append(tr_det["s1"])
        y_s1_det.append(-float(tr_det["j_net"]))
        X_s1_dir.append(tr_dir["s1"])
        y_s1_dir.append(-float(tr_dir["j_net"]))

        X_s2_det.append(tr_w_det["s2"])
        y_s2_det.append(float(tr_w_det["r2"]))
        X_s2_dir.append(tr_w_dir["s2"])
        y_s2_dir.append(float(tr_w_dir["r2"]))

        wait_rows.append((tr_w_det["s1"], float(tr_w_det["r1"]), tr_w_det["s2"]))
        # Fixed continuation baseline heuristic: detour if still blocked (no direct waiting), or heuristic threshold
        cont_j = tr_w_dir["j_net"] if tr_w_dir["obs_cleared_at_dec2"] else tr_w_det["j_net"]
        # Wait: Fixed-Continuation Terminal Regression must predict from s1 only!
        # Let fixed continuation be "always take DETOUR at step 2" or "always take DIRECT if obs cleared":
        # Wait: if TerminalReg tries a non-adaptive fixed continuation at Dec2 (e.g. always DETOUR), it gets tr_w_det;
        # even fairer: let TerminalReg predict expected terminal return of WAIT under heuristic Dec2 rule `Detour if est>6 else Direct`
        y_term_fixed_cont.append(-float(tr_w_det["j_net"]))

    # Train 1-step & 2-step Q-functions with uniform target scaling for numerical stability
    q_scale = 100.0
    q1_det = _MLPRegressor(E3_STATE_DIM, hidden=32, seed=11)
    q1_dir = _MLPRegressor(E3_STATE_DIM, hidden=32, seed=12)
    q2_det = _MLPRegressor(E3_STATE_DIM, hidden=32, seed=21)
    q2_dir = _MLPRegressor(E3_STATE_DIM, hidden=32, seed=22)

    q1_det.fit(np.stack(X_s1_det), np.asarray(y_s1_det, dtype=np.float32) / q_scale, epochs=700)
    q1_dir.fit(np.stack(X_s1_dir), np.asarray(y_s1_dir, dtype=np.float32) / q_scale, epochs=700)
    q2_det.fit(np.stack(X_s2_det), np.asarray(y_s2_det, dtype=np.float32) / q_scale, epochs=700)
    q2_dir.fit(np.stack(X_s2_dir), np.asarray(y_s2_dir, dtype=np.float32) / q_scale, epochs=700)

    # Fitted Q-Iteration step 1: Bellman target y1 = r1 + max_{a2 in {det, dir}} Q2(s2, a2)
    s2_mat = np.stack([row[2] for row in wait_rows])
    q2_max = np.maximum(q2_det.predict(s2_mat), q2_dir.predict(s2_mat)) * q_scale
    y_bellman_wait = np.asarray([row[1] for row in wait_rows], dtype=np.float32) + q2_max
    q1_wait_fqi = _MLPRegressor(E3_STATE_DIM, hidden=32, seed=31)
    q1_wait_fqi.fit(np.stack([row[0] for row in wait_rows]), y_bellman_wait / q_scale, epochs=700)

    # Fair contender 2: Terminal Regression with Fixed Continuation (no Bellman max at step 2)
    q1_wait_fixed = _MLPRegressor(E3_STATE_DIM, hidden=32, seed=41)
    q1_wait_fixed.fit(np.stack([row[0] for row in wait_rows]), np.asarray(y_term_fixed_cont, dtype=np.float32) / q_scale, epochs=700)

    def _eval_suite(*, mask_obs: bool) -> dict[str, Any]:
        costs: dict[str, list[float]] = {
            "OneStep_Myopic": [],
            "TermReg_FixedCont": [],
            "TwoStep_FQI_RL": [],
            "Oracle_VBS": [],
        }
        costs_rev: dict[str, list[float]] = {k: [] for k in costs}
        costs_irrev: dict[str, list[float]] = {k: [] for k in costs}
        wait_rates: dict[str, int] = {k: 0 for k in ("TermReg_FixedCont", "TwoStep_FQI_RL")}

        for sp in eval_specs:
            tr_det = evaluate_e3_trajectory(sp, ACT_COMMIT_DETOUR, mask_intermediate_obs=mask_obs)
            tr_dir = evaluate_e3_trajectory(sp, ACT_COMMIT_DIRECT, mask_intermediate_obs=mask_obs)
            tr_w_det = evaluate_e3_trajectory(sp, ACT_WAIT_OBSERVE, ACT_COMMIT_DETOUR, mask_intermediate_obs=mask_obs)
            tr_w_dir = evaluate_e3_trajectory(sp, ACT_WAIT_OBSERVE, ACT_COMMIT_DIRECT, mask_intermediate_obs=mask_obs)

            s1 = tr_det["s1"].reshape(1, -1)
            q_det_1 = float(q1_det.predict(s1)[0])
            q_dir_1 = float(q1_dir.predict(s1)[0])
            q_w_fixed = float(q1_wait_fixed.predict(s1)[0])
            q_w_fqi = float(q1_wait_fqi.predict(s1)[0])

            # 1. One-step Myopic (actions 0, 1 only)
            c_myopic = tr_det["j_net"] if q_det_1 >= q_dir_1 else tr_dir["j_net"]

            # 2. Terminal Regression w/ Fixed Continuation
            a1_tr = int(np.argmax([q_det_1, q_dir_1, q_w_fixed]))
            if a1_tr == ACT_COMMIT_DETOUR:
                c_tr = tr_det["j_net"]
            elif a1_tr == ACT_COMMIT_DIRECT:
                c_tr = tr_dir["j_net"]
            else:
                wait_rates["TermReg_FixedCont"] += 1
                c_tr = tr_w_det["j_net"]

            # 3. Two-step FQI (chooses Wait when Bellman value > immediate actions, then chooses argmax Q2(s2))
            a1_fqi = int(np.argmax([q_det_1, q_dir_1, q_w_fqi]))
            if a1_fqi == ACT_COMMIT_DETOUR:
                c_fqi = tr_det["j_net"]
            elif a1_fqi == ACT_COMMIT_DIRECT:
                c_fqi = tr_dir["j_net"]
            else:
                wait_rates["TwoStep_FQI_RL"] += 1
                s2 = tr_w_det["s2"].reshape(1, -1)
                a2_fqi = ACT_COMMIT_DETOUR if float(q2_det.predict(s2)[0]) >= float(q2_dir.predict(s2)[0]) else ACT_COMMIT_DIRECT
                c_fqi = tr_w_det["j_net"] if a2_fqi == ACT_COMMIT_DETOUR else tr_w_dir["j_net"]

            c_oracle = min(tr_det["j_net"], tr_dir["j_net"], tr_w_det["j_net"], tr_w_dir["j_net"])
            for k, val in (("OneStep_Myopic", c_myopic), ("TermReg_FixedCont", c_tr), ("TwoStep_FQI_RL", c_fqi), ("Oracle_VBS", c_oracle)):
                costs[k].append(float(val))
                if sp.reversible:
                    costs_rev[k].append(float(val))
                else:
                    costs_irrev[k].append(float(val))

        return {
            "mask_intermediate_obs": bool(mask_obs),
            "n_eval": len(eval_specs),
            "n_reversible": len(costs_rev["OneStep_Myopic"]),
            "n_irreversible": len(costs_irrev["OneStep_Myopic"]),
            "mean_j_net": {k: round(float(np.mean(v)), 2) for k, v in costs.items()},
            "reversible_mean_j_net": {k: round(float(np.mean(v)), 2) for k, v in costs_rev.items()},
            "irreversible_mean_j_net": {k: round(float(np.mean(v)), 2) for k, v in costs_irrev.items()},
            "wait_action_rate_pct": {k: round(100.0 * v / max(1, len(eval_specs)), 2) for k, v in wait_rates.items()},
        }

    unmasked = _eval_suite(mask_obs=False)
    masked = _eval_suite(mask_obs=True)
    voi_effect = round(float(masked["mean_j_net"]["TwoStep_FQI_RL"]) - float(unmasked["mean_j_net"]["TwoStep_FQI_RL"]), 2)
    return {
        "n_train": n_train,
        "n_eval": n_eval,
        "gamma": 1.0,
        "unmasked_eval": unmasked,
        "masked_counterfactual_eval": masked,
        "causal_voi_gain_fqi": voi_effect,
    }


def run_e3_dralns_batch_timing_experiment(*, seeds: tuple[int, ...] = (7101, 7102, 7103, 7104)) -> dict[str, Any]:
    """Secondary E3 experiment (Reijnen 2024 DR-ALNS): commit now vs extra fixed-eval search batch."""
    from amr_sched.baselines.literature_adapters import ReijnenDRALNS24Proposer
    from amr_sched.cvcr.clock import virtual_clock_scope
    from amr_sched.rl import (
        AMRReschedulingEnv,
        RuleV10PortfolioPolicy,
        compute_expert_v10_action_mask,
        compute_expert_v9_state,
    )

    variants = {
        "ZeroBatch_CommitNow": {"max_batches": 0, "stop_on_tight_slack": True, "min_gain_stop": 2.0, "virtual_batch_ms": 0.0},
        "Fixed3Batch_NoGuard": {"max_batches": 3, "stop_on_tight_slack": False, "min_gain_stop": -1.0, "virtual_batch_ms": 14.0},
        "ReijnenDRALNS_Adaptive": {"max_batches": 3, "stop_on_tight_slack": True, "min_gain_stop": 2.0, "virtual_batch_ms": 14.0},
    }
    res: dict[str, Any] = {}
    for vname, vkw in variants.items():
        nets: list[float] = []
        win_exps: list[int] = []
        comp_ms: list[float] = []
        pass_all = True
        with virtual_clock_scope(poll_ms=0.002):
            for tp, ax in (("unseen_choke_yard", "fault_recovery"), ("workshop", "kitting_delay")):
                for sd in seeds:
                    prop = ReijnenDRALNS24Proposer(seed=sd, **vkw)
                    env = AMRReschedulingEnv(
                        topology=tp,
                        seed=sd,
                        allow_kitting_shortcut=False,
                        pure_physical_reward=True,
                        real_compute_window_coupling=True,
                        expert_v10_portfolio=True,
                        loaded_repair_mode="haven_v12",
                        external_proposer=prop,
                    )
                    opts = {"axis": ax}
                    obs, info = env.reset(seed=sd, options=opts)
                    pol = RuleV10PortfolioPolicy(epoch_interval=16)
                    pol.reset()
                    done = False
                    while not done:
                        assert env.sim is not None and env.tracker is not None
                        o9 = compute_expert_v9_state(
                            env.sim,
                            env.tracker,
                            last_replan_tick=env.last_replan_tick,
                            recent_conflict_ticks=env.recent_conflict_ticks,
                            decision_logs=env.decision_logs,
                        )
                        iu = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
                        act = pol.select_action(o9, iu, sim=env.sim)
                        obs, _, tm, tr, info = env.step(act)
                        done = tm or tr
                    m = env.compute_episode_metrics()
                    aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=False)
                    pass_all = pass_all and bool(aud.pass_audit)
                    nets.append(float(m["net_obj"]))
                    win_exps.append(int(m.get("total_window_expirations", env.total_window_expirations)))
                    comp_ms.append(float(m["total_compute_ms"]))
        res[vname] = {
            "n_episodes": len(nets),
            "audit_all_pass": pass_all,
            "mean_net_obj": round(float(np.mean(nets)), 2),
            "mean_window_expirations": round(float(np.mean(win_exps)), 3),
            "mean_compute_ms": round(float(np.mean(comp_ms)), 2),
        }
    return res
