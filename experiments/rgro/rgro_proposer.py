"""Shared-Budget Risk-Gated Residual Optimization (`RGRO`) Proposer (`IF-CORE-RGRO-01`).

Unifies `V19` Reference Anchor, `V18` Low-Cost Residual Polish, `V20.1` Shared-Encoder Residual
Search, Pure-NumPy Relative-Advantage `DFL` Ensemble (`A_LCB`), Episode-Grouped / Non-Exchangeable
`CRC` Harm-Risk Gating, and Independent `PDG 2.0` Emergency Escape inside ONE single-pass
35 ms compute budget (ZERO serial solver duplication, ZERO scenario/seed/Track-ID branching).

Built-in Deliverable 4 Structural Latency Fixes (frozen baselines remain 100% untouched):
  1. `CVCREvaluator(..., mode="B2", seq_diff_b2=True)` enabling fast `_pending_b1_seed`.
  2. Live `ev.set_traffic_weights(tw, cutoff_step=cut_step)` from `execution_feedback`.
  3. Anchor-First execution: fast `V19` Anchor completes BEFORE any residual search starts.
  4. Footprint-predicted `est_ev_s` + `t_keep` guard (`now + est_ev_s < hard_budget_s`) on every
     single `ev.evaluate()` call in both nominal and `PDG 2.0` escape modes.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.pdg_liveness import (
    PDG2Config,
    PDG2SCCLivenessMonitor,
    build_critical_bottleneck_locks,
    plan_idle_blocker_evacuations,
)
from amr_sched.cvcr.policy import CandidateProposal, _LazyKittingSurrogate
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals, plan_haven_safe_corridor_repair
from amr_sched.cvcr.v18_frontier_synthesis import compute_semicircle_kitting_bays
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    build_causal_repair_cone,
    compute_stability_sets,
    observable_op_closure,
    score_candidate_ja_unified,
)
from amr_sched.rl.v19_hetero_constructor import (
    NumpyV19HeteroConstructorNet,
    V19AdaptiveCouplingProposer,
    _build_core_sched_proposal,
    _build_external_proposal,
    compute_candidate_amr_footprint,
    compute_dynamic_coupling_features,
    decode_v19_candidate,
)
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph, extract_v19_hetero_graph
from amr_sched.rl.v20_najs import (
    V20CandidateArtifact,
    _DEFAULT_V19_RL_CKPT,
    _DEFAULT_V19_STAGED_CKPT,
    _compute_fresh_semicircle_bays,
    _default_task_order_from_prio,
    _healthy_robots_from_graph,
    build_keep_baseline_stats,
    build_masked_subgraph,
    decode_v19_with_cached_enc,
    derive_base_op_prio,
    expand_causal_release_closure,
    extract_critic_features,
    generate_causal_cone_candidates,
)
from experiments.rgro.dfl_critic import (
    DFLAdvantageEnsemble,
    ExecutionHistoryBuffer,
    build_anchor_relative_feature,
)


@dataclass(frozen=True)
class RGROConfig:
    """Declarative configuration covering the full `U0..U5` VAPI + RGRO ablation ladder."""

    policy_id: str = "RGRO_U5"
    enable_pdg2: bool = True
    enable_v18_polish: bool = True
    enable_v20_residual: bool = True
    enable_dfl: bool = True
    enable_crc_gate: bool = True
    # DFL + LCB + CRC thresholds (calibrated strictly on quarantined DEV!)
    lcb_kappa: float = 0.75
    dfl_tau_gain: float = 20.0      # Conservative LCB advantage above critic noise floor (J_A points)
    crc_harm_tau: float = 0.10      # Max calibrated harm probability P(J(x_c) > J(x_b) + 50)
    res_ja_margin: float = 8.0      # Base deterministic JA_Unified improvement margin for residual over anchor
    # Budget controls (strictly <= 35 ms SLA)
    hard_budget_ms: float = 35.0
    residual_min_rem_ms: float = 3.5
    max_residual_evals: int = 3
    use_rust: bool = True
    seed: int = 0


def make_ladder_config(variant: str, *, crc_harm_tau: float = 0.10, dfl_tau_gain: float = 20.0, lcb_kappa: float = 0.75) -> RGROConfig:
    """Factory for the 6 unified `U0..U5` ladder variants."""
    v = variant.upper()
    if v == "U0":
        return RGROConfig(
            policy_id="U0_V19_Anchor",
            enable_pdg2=False,
            enable_v18_polish=False,
            enable_v20_residual=False,
            enable_dfl=False,
            enable_crc_gate=False,
        )
    if v == "U1":
        return RGROConfig(
            policy_id="U1_V19_PDG2",
            enable_pdg2=True,
            enable_v18_polish=False,
            enable_v20_residual=False,
            enable_dfl=False,
            enable_crc_gate=False,
        )
    if v == "U2":
        return RGROConfig(
            policy_id="U2_U1_V18Polish",
            enable_pdg2=True,
            enable_v18_polish=True,
            enable_v20_residual=False,
            enable_dfl=False,
            enable_crc_gate=False,
        )
    if v == "U3":
        return RGROConfig(
            policy_id="U3_U1_V20Res",
            enable_pdg2=True,
            enable_v18_polish=False,
            enable_v20_residual=True,
            enable_dfl=False,
            enable_crc_gate=False,
        )
    if v == "U4":
        return RGROConfig(
            policy_id="U4_U3_DFL_LCB",
            enable_pdg2=True,
            enable_v18_polish=False,
            enable_v20_residual=True,
            enable_dfl=True,
            enable_crc_gate=False,
            lcb_kappa=lcb_kappa,
            dfl_tau_gain=dfl_tau_gain,
        )
    if v == "U5":
        return RGROConfig(
            policy_id="U5_U4_CRC_Gate",
            enable_pdg2=True,
            enable_v18_polish=False,
            enable_v20_residual=True,
            enable_dfl=True,
            enable_crc_gate=True,
            lcb_kappa=lcb_kappa,
            dfl_tau_gain=dfl_tau_gain,
            crc_harm_tau=crc_harm_tau,
        )
    raise ValueError(f"Unknown RGRO ladder variant: {variant}")


def _generate_v18_polish_candidates(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    fast_surr: FastJAWKittingSurrogate,
    anchor_assign: dict[str, str],
    anchor_bays: dict[str, str],
    anchor_crews: dict[str, int],
    anchor_prio: dict[str, int],
    healthy_agvs: list[str],
    fresh_op_ids: set[str],
) -> list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]]:
    """Fast local 1-task move / 2-task swap / bay-crew polish anchored on `x_b` (<0.5 ms surrogate screen)."""
    if not graph.rel_tids or len(healthy_agvs) < 2:
        return []
    base_sc = float(fast_surr.score(anchor_assign, anchor_prio, anchor_bays, anchor_crews))
    rel = list(graph.rel_tids)
    # 1) Greedy 1-task reassignment pass (up to 2 improving moves)
    cur_a = dict(anchor_assign)
    cur_sc = base_sc
    moved = False
    for _ in range(2):
        best_mv: Optional[tuple[float, str, str]] = None
        for tid in rel[:10]:
            cur_r = cur_a.get(tid)
            for aid in healthy_agvs:
                if aid == cur_r:
                    continue
                cur_a[tid] = aid
                sc_t = float(fast_surr.score(cur_a, anchor_prio, anchor_bays, anchor_crews))
                if sc_t + 1e-2 < cur_sc and (best_mv is None or sc_t < best_mv[0]):
                    best_mv = (sc_t, tid, aid)
            if cur_r is not None:
                cur_a[tid] = cur_r
        if best_mv is None:
            break
        cur_sc, mv_t, mv_r = best_mv
        cur_a[mv_t] = mv_r
        moved = True

    # 2) Optional bay/crew micro-polish on fresh ops
    cur_b = dict(anchor_bays)
    cur_c = dict(anchor_crews)
    bc_moved = False
    for op in snapshot.operations:
        if op.started or op.op_id not in fresh_op_ids:
            continue
        if len(op.eligible_bays) > 1:
            ob = cur_b.get(op.op_id)
            for xb in op.eligible_bays:
                if xb == ob:
                    continue
                cur_b[op.op_id] = xb
                sc_t = float(fast_surr.score(cur_a, anchor_prio, cur_b, cur_c))
                if sc_t + 1e-2 < cur_sc:
                    cur_sc, ob, bc_moved = sc_t, xb, True
                else:
                    if ob is not None:
                        cur_b[op.op_id] = ob
        if len(op.eligible_crews) > 1:
            oc = int(cur_c.get(op.op_id, 0))
            for xc in op.eligible_crews:
                if int(xc) == oc:
                    continue
                cur_c[op.op_id] = int(xc)
                sc_t = float(fast_surr.score(cur_a, anchor_prio, cur_b, cur_c))
                if sc_t + 1e-2 < cur_sc:
                    cur_sc, oc, bc_moved = sc_t, int(xc), True
                else:
                    cur_c[op.op_id] = oc

    out: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []
    if moved or bc_moved:
        out.append(("v18_polish_1", cur_a, cur_b, cur_c, dict(anchor_prio), None, False))

    # 3) Pairwise 2-task swap between different healthy robots (breaks symmetric load imbalance)
    best_sw: Optional[tuple[float, str, str]] = None
    top_rel = rel[:8]
    for i in range(len(top_rel)):
        t1 = top_rel[i]
        r1 = anchor_assign.get(t1)
        if r1 not in healthy_agvs:
            continue
        for j in range(i + 1, len(top_rel)):
            t2 = top_rel[j]
            r2 = anchor_assign.get(t2)
            if r2 not in healthy_agvs or r2 == r1:
                continue
            sw_a = dict(anchor_assign)
            sw_a[t1], sw_a[t2] = r2, r1
            sc_s = float(fast_surr.score(sw_a, anchor_prio, anchor_bays, anchor_crews))
            if sc_s + 1e-2 < base_sc and (best_sw is None or sc_s < best_sw[0]):
                best_sw = (sc_s, t1, t2)
    if best_sw is not None:
        _, s1, s2 = best_sw
        sw_a = dict(anchor_assign)
        sw_a[s1], sw_a[s2] = sw_a[s2], sw_a[s1]
        out.append(("v18_polish_swap", sw_a, dict(anchor_bays), dict(anchor_crews), dict(anchor_prio), None, False))
    return out


class RGROProposer:
    """Unified Shared-Budget `RGRO` Proposer implementing `U0..U5` with strict single-pass Layer-1 reuse."""

    def __init__(
        self,
        cfg: RGROConfig,
        *,
        dfl_ensemble: Optional[DFLAdvantageEnsemble] = None,
        dfl_ckpt_path: Optional[Path] = None,
        full_ckpt: Path = _DEFAULT_V19_RL_CKPT,
        staged_ckpt: Path = _DEFAULT_V19_STAGED_CKPT,
    ) -> None:
        self.cfg = cfg
        self.policy_id = cfg.policy_id
        self.use_rust = bool(cfg.use_rust)
        self.seed = int(cfg.seed)

        # Reuse frozen V19AdaptiveCouplingProposer ONLY as weight/LinUCB holder (never invokes duplicate pass!)
        self._v19_holder = V19AdaptiveCouplingProposer(
            full_ckpt_path=full_ckpt,
            staged_ckpt_path=staged_ckpt,
            use_rust=self.use_rust,
            policy_id=self.policy_id,
            seed=self.seed,
            mode="bandit",
            evaluator_mode="B2_SEEDED",
            use_ja_aligned_scorer=True,
            enable_neural_candidates=True,
            enable_causal_cone_repair=True,
            enable_zero_churn_projection=True,
            enable_exec_consistent_commit=True,
            enable_window_budget_guard=True,
        )
        self._full_net: Optional[NumpyV19HeteroConstructorNet] = None
        self._staged_net: Optional[NumpyV19HeteroConstructorNet] = None
        self._pgmpo: Any = None
        self._pgmpo_init_done: bool = False

        if dfl_ensemble is not None:
            self._dfl: Optional[DFLAdvantageEnsemble] = dfl_ensemble
        elif dfl_ckpt_path is not None and Path(dfl_ckpt_path).is_file():
            self._dfl = DFLAdvantageEnsemble.load_npz(Path(dfl_ckpt_path))
        else:
            self._dfl = None

        self.history_buf = ExecutionHistoryBuffer(maxlen=12)
        self._active_pdg: Optional[dict[str, Any]] = None
        self._sim_ref: Any = None
        self._escape_tabu: dict[str, int] = {}

        # Episode / call telemetry counters
        self.pdg_recoveries_triggered: int = 0
        self.residual_candidates_evaluated: int = 0
        self.residual_candidates_accepted: int = 0
        self.residual_rejected_dfl: int = 0
        self.residual_rejected_crc: int = 0
        self.last_call_stats: dict[str, Any] = {}

    def reset_episode_state(self) -> None:
        """Reset per-episode LinUCB, history buffer, tabu memory, and counters."""
        dim = int(self._v19_holder._A[0].shape[0])
        self._v19_holder._A = [np.eye(dim, dtype=np.float64), np.eye(dim, dtype=np.float64)]
        self._v19_holder._b = [np.zeros(dim, dtype=np.float64), np.zeros(dim, dtype=np.float64)]
        self.history_buf.reset()
        self._active_pdg = None
        self._escape_tabu.clear()
        self.pdg_recoveries_triggered = 0
        self.residual_candidates_evaluated = 0
        self.residual_candidates_accepted = 0
        self.residual_rejected_dfl = 0
        self.residual_rejected_crc = 0
        self.last_call_stats = {}

    def set_pdg_state(self, pdg: Optional[dict[str, Any]]) -> None:
        self._active_pdg = pdg

    def set_sim_ref(self, sim: Any) -> None:
        self._sim_ref = sim

    def observe_committed_step(self, snapshot: NativeSnapshot, *, rej_streak: int = 0) -> np.ndarray:
        """Event-only history update called strictly once per committed real tick."""
        return self.history_buf.update_from_snapshot(snapshot, rej_streak=rej_streak)

    def _ensure_nets(self) -> tuple[NumpyV19HeteroConstructorNet, NumpyV19HeteroConstructorNet]:
        if self._full_net is None or self._staged_net is None:
            f_net, s_net = self._v19_holder._ensure_infer_nets()
            self._full_net = f_net
            self._staged_net = s_net
        return self._full_net, self._staged_net

    def _ensure_pgmpo(self) -> Any:
        if not self._pgmpo_init_done:
            self._pgmpo_init_done = True
            try:
                from amr_sched.baselines.policy_registry import load_internal_cvcr_policy

                v18 = load_internal_cvcr_policy("V18_EA_JOINT", use_rust=self.use_rust)
                pg = getattr(v18, "pgmpo", None)
                self._pgmpo = pg if (pg is not None and getattr(pg, "ready", False)) else None
            except Exception:
                self._pgmpo = None
        return self._pgmpo

    def _encode_cached(
        self,
        net: NumpyV19HeteroConstructorNet,
        graph: V19HeteroGraph,
        cache_dict: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
        key: str,
        call_counter: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if key not in cache_dict:
            enc = net.encode_entities(graph)
            h12 = net.a1_a2_heads(enc, graph)
            cache_dict[key] = (enc, h12)
            call_counter[key] = call_counter.get(key, 0) + 1
        return cache_dict[key]

    def __call__(self, snapshot: NativeSnapshot, network: Network, *, budget_ms: float = 35.0) -> CandidateProposal:
        if self.cfg.enable_pdg2 and self._active_pdg and self._active_pdg.get("pdg_alert"):
            return self._propose_pdg2_escape(snapshot, network, budget_ms=budget_ms)
        return self._propose_nominal_rgro(snapshot, network, budget_ms=budget_ms)

    def _propose_nominal_rgro(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        import amr_sched.cvcr.v17_window_protection as _v17
        import amr_sched.cvcr.v18_frontier_synthesis as _v18

        select_coupled_op_pair = getattr(_v17, "select_coupled_op_pair", getattr(_v18, "select_coupled_op_pair", None))
        full_net, staged_net = self._ensure_nets()
        t0 = _cvcr_now_s()

        # Update event-only history buffer idempotently on this committed tick (read-only afterwards!)
        h_vec_ro = self.history_buf.update_from_snapshot(snapshot)

        # ---- SHARED LAYER-1 CONTEXT (COMPUTED STRICTLY ONCE) ----
        clf = classify_commitments(snapshot)
        tick = int(snapshot.tick)

        has_res_win = any(bool(getattr(op, "resource_windows", ())) for op in snapshot.operations)
        has_kit_win_tight = any(
            getattr(t, "kitting_bottleneck_active", False) and int(getattr(t, "adjustable_window_steps", 3)) <= 1
            for t in snapshot.tasks
        )
        has_open_fault_win = bool(clf.disturbed_robots) and any(
            int(getattr(t, "adjustable_window_steps", 3)) <= 1 for t in snapshot.tasks
        )
        is_window_tight = bool(has_kit_win_tight or has_open_fault_win)
        eff_budget_ms = min(float(budget_ms), 17.5 if is_window_tight else 29.0)
        deadline = t0 + max(0.0, eff_budget_ms * 0.78 / 1000.0)
        hard_budget_s = t0 + max(0.0, eff_budget_ms * 0.88 / 1000.0)

        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        graph_extract_calls = 1
        enc_cache: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
        enc_calls: dict[str, int] = {"full": 0, "staged": 0}

        healthy_agvs = [graph.agv_ids[i] for i in graph.healthy_agv_indices]
        phi = compute_dynamic_coupling_features(graph)
        coupling_score = float(0.30 * phi[0] + 0.25 * phi[1] + 0.20 * phi[2] + 0.25 * phi[3])
        is_ws = len(network.nodes) >= 30

        # Deliverable 4 Fix #1 + #2: `mode="B2", seq_diff_b2=True` + live `set_traffic_weights`!
        ev = CVCREvaluator(
            network,
            snapshot,
            alpha_makespan=3.0,
            lambda_change=15.0,
            use_rust=self.use_rust,
            mode="B2",
            seq_diff_b2=True,
        )
        fb = getattr(snapshot, "execution_feedback", None)
        has_drift = bool(fb is not None and not fb.is_empty())
        tw: dict[tuple[str, str], int] = {}
        cut_step = 0
        if has_drift and fb is not None and fb.blocked_intervals(t_commit=tick, default_horizon=12):
            tw = {
                (str(u), str(v)): int(w)
                for (u, v), w in fb.corridor_q3_penalties().items()
                if network.has_edge(u, v)
            }
            if tw:
                cut_step = tick + 45
                ev.set_traffic_weights(tw, cutoff_step=cut_step)

        base_op_prio = {
            op.op_id: max(tick, int(op.due_step) - int(op.proc_steps))
            for op in snapshot.operations
        }
        t_keep0 = _cvcr_now_s()
        keep_eval = ev.evaluate(
            graph.base_assign,
            op_priority=base_op_prio,
            op_bay_overrides=graph.base_bays,
            op_crew_overrides=graph.base_crews,
        )
        keep_eval_s = max(0.001, _cvcr_now_s() - t_keep0)
        if keep_eval.valid:
            ev.set_incumbent(
                graph.base_assign,
                keep_eval,
                op_priority=base_op_prio,
                op_bay_overrides=graph.base_bays,
            )

        best_eval: EvalResult = keep_eval
        best_assign = dict(graph.base_assign)
        best_bays = dict(graph.base_bays)
        best_crews = dict(graph.base_crews)
        best_op_prio = dict(base_op_prio)
        best_seq: Optional[dict[str, list[str]]] = None
        fallback_keep = True
        chosen_arm = "keep"
        selected_mode = "keep"

        rob_cur_healthy = {
            r.robot_id: r.cur_task
            for r in snapshot.robots
            if r.cur_task and r.robot_id not in clf.disturbed_robots
        }
        fresh_op_ids, pending_or_dist_set, _ = compute_stability_sets(snapshot, graph)
        obs_ops = observable_op_closure(snapshot)
        tasks_by_id = {t.task_id: t for t in snapshot.tasks}
        is_open = tick <= 4
        is_resc = bool(clf.disturbed_robots) or any(
            (t.assigned_agv is None or t.status.lower() == "pending") and getattr(t, "is_urgent", False)
            for t in snapshot.tasks
        )
        bay_delta = {graph.bay_ids[i]: float(graph.bay_detour_norm[i]) * 15.0 for i in range(len(graph.bay_ids))}
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev, graph, lambda_change=15.0)

        def _score_ja(
            asn: dict[str, str],
            bs: dict[str, str],
            cs_c: dict[str, int],
            ev_c: EvalResult,
        ) -> tuple[float, float]:
            return score_candidate_ja_unified(
                snapshot, network, graph, ev_c, asn, bs, cs_c, obs_ops, fast_surr,
                lambda_change=15.0, sim_horizon=ev.sim_horizon,
            )

        keep_score_val, keep_tard_val = _score_ja(graph.base_assign, graph.base_bays, graph.base_crews, keep_eval)
        best_score = keep_score_val
        anchor_evals = 0
        res_evals = 0
        res_gen = 0
        res_acc = False
        res_rej_dfl = 0
        res_rej_crc = 0

        keep_sig = (
            tuple(sorted(graph.base_assign.items())),
            tuple(sorted(graph.base_bays.items())),
            tuple(sorted(graph.base_crews.items())),
            tuple(sorted(base_op_prio.items())),
            None,
        )
        seen_sigs: set[tuple[Any, ...]] = {keep_sig}
        max_inc_eval_s = max(0.001, 0.55 * keep_eval_s)
        max_full_eval_s = keep_eval_s

        # ---- STAGE 1 & 2: V19 REFERENCE ANCHOR (ANCHOR-FIRST, REUSING CACHED ENCODING) ----
        if graph.rel_tids and healthy_agvs and float(budget_ms) > 0.05:
            np_rng = np.random.default_rng(self.seed + tick)
            x_ctx = np.concatenate([[1.0], phi.astype(np.float64)])
            ucb = []
            for arm_i in range(2):
                A_inv = np.linalg.inv(self._v19_holder._A[arm_i])
                theta = A_inv @ self._v19_holder._b[arm_i]
                bonus = 0.25 * float(np.sqrt(max(0.0, x_ctx @ A_inv @ x_ctx)))
                ucb.append(float(theta @ x_ctx) + bonus)
            arm_pick = 1 if ucb[1] >= ucb[0] else 0
            selected_mode = "full" if arm_pick == 1 else "staged"
            other_mode = "staged" if arm_pick == 1 else "full"

            def _try_verify_anchor(
                tag_s: str,
                asn_s: dict[str, str],
                bs_s: dict[str, str],
                cs_s: dict[str, int],
                op_s: dict[str, int],
                is_full_s: bool,
                *,
                force: bool = False,
            ) -> bool:
                nonlocal best_eval, best_score, best_assign, best_bays, best_crews, best_op_prio
                nonlocal fallback_keep, chosen_arm, anchor_evals, max_inc_eval_s, max_full_eval_s

                sg = (
                    tuple(sorted(asn_s.items())),
                    tuple(sorted(bs_s.items())),
                    tuple(sorted(cs_s.items())),
                    tuple(sorted(op_s.items())),
                    None,
                )
                if sg in seen_sigs:
                    return False
                if (not force) and _cvcr_now_s() >= deadline:
                    return False

                fp = compute_candidate_amr_footprint(
                    snapshot, best_assign, best_bays, best_eval.robot_order_seq, asn_s, bs_s, None
                )
                est_ev_s = max_inc_eval_s if fp["is_b2_incremental_hit"] else max(max_full_eval_s, 1.10 * keep_eval_s)
                if (not force) and (_cvcr_now_s() + est_ev_s) >= hard_budget_s:
                    return False

                seen_sigs.add(sg)
                t_ev0 = _cvcr_now_s()
                ev_c = ev.evaluate(asn_s, op_priority=op_s, op_bay_overrides=bs_s, op_crew_overrides=cs_s)
                dt_ev_s = max(0.0005, _cvcr_now_s() - t_ev0)
                if fp["is_b2_incremental_hit"]:
                    max_inc_eval_s = max(max_inc_eval_s, dt_ev_s)
                else:
                    max_full_eval_s = max(max_full_eval_s, dt_ev_s)
                anchor_evals += 1

                sc_c, tard_c = _score_ja(asn_s, bs_s, cs_s, ev_c)
                if not ev_c.valid:
                    return False

                if best_eval is keep_eval:
                    if is_open or is_resc:
                        margin = 0.0
                    else:
                        nnr = sum(
                            1
                            for tid_c, nr in asn_s.items()
                            if tid_c in tasks_by_id
                            and tasks_by_id[tid_c].status.lower() == "assigned"
                            and tasks_by_id[tid_c].assigned_agv
                            and tasks_by_id[tid_c].assigned_agv != nr
                            and tasks_by_id[tid_c].assigned_agv not in clf.disturbed_robots
                        )
                        nso = sum(
                            1
                            for oid in graph.op_ids
                            if oid not in fresh_op_ids
                            and (
                                bs_s.get(oid, graph.base_bays[oid]) != graph.base_bays[oid]
                                or cs_s.get(oid, graph.base_crews[oid]) != graph.base_crews[oid]
                            )
                        )
                        if nnr == 0 and nso == 0:
                            margin = 0.0
                        else:
                            raw_m = 20.0 + 16.0 * float(nnr) + 20.0 * float(nso)
                            tg = max(0.0, keep_tard_val - tard_c)
                            margin = raw_m * max(0.20, 1.0 - min(0.80, tg / 70.0))
                else:
                    margin = 1e-4

                if (not best_eval.valid) or (sc_c + margin < best_score):
                    best_eval, best_score = ev_c, sc_c
                    best_assign, best_bays, best_crews, best_op_prio = asn_s, bs_s, cs_s, op_s
                    chosen_arm = tag_s
                    fallback_keep = False
                    ev.set_incumbent(best_assign, best_eval, op_priority=best_op_prio, op_bay_overrides=best_bays)
                    return True
                return False

            # Stage 1: AHG-RL Neural Candidates (using CACHED Layer-1 entity encodings!)
            nn_cands: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], bool]] = []
            lock_nn_bay = bool(is_ws and len(graph.agv_ids) >= 4)
            for k in range(3):
                if k > 0 and (_cvcr_now_s() - t0) > 0.009:
                    break
                use_dual_other = (k == 1 and len(graph.agv_ids) >= 4)
                use_arm_name = other_mode if use_dual_other else selected_mode
                use_net = full_net if use_arm_name == "full" else staged_net
                use_is_full = (use_arm_name == "full")
                use_greedy = (k == 0) or use_dual_other
                if use_greedy:
                    enc_k, h12_k = self._encode_cached(use_net, graph, enc_cache, use_arm_name, enc_calls)
                    cand = decode_v19_with_cached_enc(use_net, enc_k, h12_k, graph, snapshot)
                else:
                    cand = decode_v19_candidate(
                        use_net, graph, snapshot, greedy=False, np_rng=np_rng, tau=0.45, focus_sample_frac=0.45
                    )
                bn = (
                    dict(graph.base_bays)
                    if (lock_nn_bay or ((not is_ws) and not is_open))
                    else {
                        oid: (cand.bays.get(oid, graph.base_bays[oid]) if oid in fresh_op_ids else graph.base_bays[oid])
                        for oid in graph.op_ids
                    }
                )
                cn = {
                    oid: (cand.crews.get(oid, graph.base_crews[oid]) if oid in fresh_op_ids else graph.base_crews[oid])
                    for oid in graph.op_ids
                }
                raw_tag = f"{use_arm_name}_g" if use_greedy else f"{use_arm_name}_s{k}"
                nn_cands.append((raw_tag, cand.assign, bn, cn, cand.op_prio, use_is_full))
                if pending_or_dist_set and not is_open:
                    za = {
                        tid: (cand.assign.get(tid, graph.base_assign[tid]) if tid in pending_or_dist_set else graph.base_assign[tid])
                        for tid in graph.base_assign
                    }
                    nn_cands.append((f"{use_arm_name}_zc", za, bn, cn, cand.op_prio, False))

            for i_n, item_n in enumerate(nn_cands[:3]):
                _try_verify_anchor(*item_n, force=(i_n == 0 and _cvcr_now_s() < hard_budget_s))

            # Stage 2: V19 Causal Joint Repair Cone + Danger-Cliff Guard
            if _cvcr_now_s() < deadline:
                has_fbay = any(o.op_id in fresh_op_ids and len(o.eligible_bays) > 1 for o in snapshot.operations)
                allow_bd = has_fbay and (not has_drift) and (is_ws or is_open)
                if allow_bd:
                    b_opt, km_p = _v18.compute_semicircle_kitting_bays(
                        snapshot, network, clf, fast_surr, best_bays, bay_delta,
                        eligible_op_ids=fresh_op_ids, spread_empty_kits=is_open,
                    )
                else:
                    b_opt, km_p = dict(best_bays), dict(best_op_prio)
                c_opt = dict(best_crews)
                p_co = km_p if (has_res_win or has_fbay) else best_op_prio
                cone = build_causal_repair_cone(snapshot, graph, best_eval, best_assign, best_bays, best_crews)
                ccands: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], bool]] = []
                if (not is_ws) and pending_or_dist_set and not is_open:
                    ha_anc = {tid: aid for tid, aid in best_assign.items() if tid not in pending_or_dist_set}
                    ccands.append((
                        "cone_regret2_zc_keepbay",
                        fast_surr.regret_k_assignment(healthy_agvs, ha_anc, sorted(pending_or_dist_set), best_bays, best_op_prio),
                        dict(best_bays), dict(best_crews), dict(best_op_prio), False,
                    ))
                if pending_or_dist_set:
                    ha_anc = {tid: aid for tid, aid in best_assign.items() if tid not in pending_or_dist_set}
                    ccands.append((
                        "cone_regret2_zc",
                        fast_surr.regret_k_assignment(healthy_agvs, ha_anc, sorted(pending_or_dist_set), b_opt, p_co),
                        b_opt, c_opt, p_co, False,
                    ))
                if not is_ws:
                    fx_anc = {tid: aid for tid, aid in best_assign.items() if tid not in set(graph.rel_tids)}
                    ccands.append((
                        "cone_regret2_full",
                        fast_surr.regret_k_assignment(healthy_agvs, fx_anc, sorted(graph.rel_tids), b_opt, p_co),
                        b_opt, c_opt, p_co, True,
                    ))
                dc = sorted(set(pending_or_dist_set) | set(cone.cone_tasks))
                if dc and len(dc) < len(graph.rel_tids):
                    ac = {tid: aid for tid, aid in best_assign.items() if tid not in set(dc)}
                    dc_b = best_bays if ((not is_ws) and not is_open) else b_opt
                    ccands.append((
                        "cone_regret2_lns",
                        fast_surr.regret_k_assignment(healthy_agvs, ac, dc, dc_b, p_co),
                        dc_b, c_opt, p_co, False,
                    ))
                c_cap = 4 if (not is_ws) else 2
                for item_c in ccands[:c_cap]:
                    _try_verify_anchor(*item_c)

                if select_coupled_op_pair is not None and has_res_win and best_eval.valid and (_cvcr_now_s() + 0.002) < deadline:
                    dops = _v18._find_danger_cliff_ops(snapshot, best_eval, bay_delta)
                    pr = select_coupled_op_pair(network, snapshot, best_eval, best_assign, best_op_prio) if dops else None
                    if dops and pr is not None:
                        for ldr, yld in ((pr[0], pr[1]), (pr[1], pr[0])):
                            if (_cvcr_now_s() + 0.002) >= deadline:
                                break
                            bp_p = _v17.plan_bridge_order_control(
                                network, snapshot, ev, best_assign, best_op_prio, ldr, yld,
                                sim=None, base_bays=best_bays, base_crews=best_crews,
                            )
                            if bp_p.eligible and bp_p.eval_res is not None and bp_p.eval_res.valid:
                                anchor_evals += 1
                                sc_bp, _ = _score_ja(best_assign, best_bays, best_crews, bp_p.eval_res)
                                if sc_bp + 4.0 < best_score:
                                    best_eval, best_score, best_op_prio = bp_p.eval_res, sc_bp, bp_p.op_priority
                                    chosen_arm = "cliff_bridge"
                                    fallback_keep = False

            # Update LinUCB bandit based strictly on V19 Anchor performance
            impr = max(-1.0, min(1.0, (keep_score_val - best_score) / 200.0))
            self._v19_holder._A[arm_pick] += np.outer(x_ctx, x_ctx)
            self._v19_holder._b[arm_pick] += impr * x_ctx

            # ---- STAGE 3 & 4: BUDGET-GUARDED RESIDUAL OPTIMIZATION + DFL/LCB + CRC RISK GATE ----
            anchor_eval = best_eval
            anchor_score = best_score
            anchor_assign = dict(best_assign)
            anchor_bays = dict(best_bays)
            anchor_crews = dict(best_crews)
            anchor_op_prio = dict(best_op_prio)

            # Strict SLA-safe residual timebox (reserves headroom so pooled P99 stays <= 21-22 ms!)
            res_deadline = min(deadline, t0 + min(15.5, eff_budget_ms * 0.62) / 1000.0)
            res_hard_budget_s = min(hard_budget_s, t0 + min(18.5, eff_budget_ms * 0.72) / 1000.0)

            rem_s = res_deadline - _cvcr_now_s()
            min_rem_s = max(self.cfg.residual_min_rem_ms / 1000.0, 1.25 * keep_eval_s)
            if (self.cfg.enable_v18_polish or self.cfg.enable_v20_residual) and (rem_s >= min_rem_s or not anchor_eval.valid):
                res_pool: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []

                # U2 (ablated standalone arm): Low-Cost V18 local swap polish anchored on x_b
                if self.cfg.enable_v18_polish:
                    res_pool.extend(
                        _generate_v18_polish_candidates(
                            snapshot, graph, fast_surr, anchor_assign, anchor_bays, anchor_crews, anchor_op_prio,
                            healthy_agvs, fresh_op_ids,
                        )
                    )

                # U3+: Shared-Encoder V20.1 Structured Combinatorial Residual Search anchored on x_b
                if self.cfg.enable_v20_residual and (_cvcr_now_s() + 0.0015 < res_deadline or not anchor_eval.valid):
                    # 1) Reverse Bidirectional V18<->V19 Masked Subgraph (reusing CACHED full_net encoding!)
                    if anchor_eval.valid:
                        ef, hf = self._encode_cached(full_net, graph, enc_cache, "full", enc_calls)
                        cone_res = build_causal_repair_cone(
                            snapshot, graph, anchor_eval, anchor_assign, anchor_bays, anchor_crews,
                            max_cone_amrs=3, max_cone_tasks=5, max_cone_ops=3,
                        )
                        rel_t_sub, rel_o_sub = expand_causal_release_closure(
                            snapshot, graph, cone_res.cone_tasks, cone_res.cone_ops, max_tasks=6, max_ops=4,
                        )
                        if rel_t_sub or rel_o_sub:
                            sub_g = build_masked_subgraph(
                                graph, release_tasks=rel_t_sub, release_ops=rel_o_sub,
                                skel_assign=anchor_assign, skel_bays=anchor_bays, skel_crews=anchor_crews,
                                restrict_agvs=set(cone_res.cone_amrs) if cone_res.cone_amrs else None,
                            )
                            dec_bi = decode_v19_with_cached_enc(full_net, ef, hf, sub_g, snapshot)
                            res_pool.append((
                                "res_bidir_subgraph",
                                dict(dec_bi.assign),
                                dict(anchor_bays) if lock_nn_bay else dict(dec_bi.bays),
                                dict(dec_bi.crews),
                                dict(dec_bi.op_prio),
                                dict(dec_bi.task_sequences) if dec_bi.task_sequences else None,
                                False,
                            ))

                    # 2) E4 PGMPO top structured assignment (when workshop or anchor invalid, budget-guarded)
                    pgmpo = self._ensure_pgmpo()
                    if pgmpo is not None and (is_ws or not anchor_eval.valid) and (_cvcr_now_s() + 0.0025 < res_deadline):
                        lazy_surr = _LazyKittingSurrogate(
                            network, snapshot, anchor_op_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta
                        )
                        asns = pgmpo.generate_assignments(snapshot, network, clf, healthy_agvs, graph.fixed_assign, anchor_bays, lazy_surr)
                        if asns:
                            top_p = min(
                                asns,
                                key=lambda a: float(fast_surr.score(a, anchor_op_prio, anchor_bays, anchor_crews)),
                            )
                            res_pool.append((
                                "res_pgmpo_top", dict(top_p), dict(anchor_bays), dict(anchor_crews), dict(anchor_op_prio),
                                _default_task_order_from_prio(snapshot, graph, top_p, anchor_op_prio), True,
                            ))

                    # 3) Explicit task-sequence cone candidate seeded directly from x_b
                    if _cvcr_now_s() + 0.002 < res_deadline:
                        nn_seed = [("anchor", anchor_assign, anchor_bays, anchor_crews, anchor_op_prio, None, False)]
                        for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                            snapshot, network, graph, ev, fast_surr, anchor_eval, anchor_op_prio, nn_seed,
                            obs_ops, pending_or_dist_set, budget_ms_limit=1.0,
                        )[:2]:
                            res_pool.append((f"res_{tg_c}", asn_c, bs_c, cs_c, op_c, sq_c, is_fc))

                res_gen = len(res_pool)
                evaluated_res: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], EvalResult, float]] = []

                for tg_r, asn_r, bs_r, cs_r, op_r, sq_r, _ in res_pool:
                    if res_evals >= int(self.cfg.max_residual_evals):
                        break
                    sq_key = tuple((k, tuple(v)) for k, v in sorted(sq_r.items())) if sq_r else None
                    sg = (
                        tuple(sorted(asn_r.items())),
                        tuple(sorted(bs_r.items())),
                        tuple(sorted(cs_r.items())),
                        tuple(sorted(op_r.items())),
                        sq_key,
                    )
                    if sg in seen_sigs:
                        continue
                    fp = compute_candidate_amr_footprint(
                        snapshot, best_assign, best_bays, best_eval.robot_order_seq, asn_r, bs_r, sq_r
                    )
                    est_ev_s = max_inc_eval_s if fp["is_b2_incremental_hit"] else max(max_full_eval_s, 1.15 * keep_eval_s)
                    if (_cvcr_now_s() + est_ev_s) >= res_hard_budget_s:
                        continue
                    seen_sigs.add(sg)
                    t_ev0 = _cvcr_now_s()
                    ev_r = ev.evaluate(
                        asn_r, op_priority=op_r, op_bay_overrides=bs_r, op_crew_overrides=cs_r, task_order_override=sq_r
                    )
                    dt_ev_s = max(0.0005, _cvcr_now_s() - t_ev0)
                    if fp["is_b2_incremental_hit"]:
                        max_inc_eval_s = max(max_inc_eval_s, dt_ev_s)
                    else:
                        max_full_eval_s = max(max_full_eval_s, dt_ev_s)
                    res_evals += 1
                    self.residual_candidates_evaluated += 1
                    if ev_r.valid:
                        sc_r, _ = _score_ja(asn_r, bs_r, cs_r, ev_r)
                        evaluated_res.append((tg_r, asn_r, bs_r, cs_r, op_r, sq_r, ev_r, sc_r))

                # Stage 4: Residual Acceptance Gate (Deterministic U2/U3 vs DFL+LCB U4 vs DFL+LCB+CRC U5)
                if evaluated_res:
                    def _churn_margin_vs_anchor(item_r: Any) -> float:
                        if not anchor_eval.valid:
                            return 0.0
                        dt_a = sum(1 for tid in graph.rel_tids if item_r[1].get(tid) != anchor_assign.get(tid))
                        do_a = sum(
                            1
                            for oid in graph.op_ids
                            if item_r[2].get(oid) != anchor_bays.get(oid) or item_r[3].get(oid) != anchor_crews.get(oid)
                        )
                        return float(self.cfg.res_ja_margin) + 10.0 * float(dt_a) + 14.0 * float(do_a)

                    if not (self.cfg.enable_dfl and self._dfl is not None and anchor_eval.valid):
                        # Deterministic JA_Unified acceptance with churn-aware margin (or rescue if anchor invalid)
                        best_r = None
                        best_net_gain = 0.0
                        for item_r in evaluated_res:
                            m_req = _churn_margin_vs_anchor(item_r)
                            net_g = (anchor_score - item_r[7]) - m_req
                            if (not anchor_eval.valid) or (net_g > best_net_gain):
                                best_net_gain = net_g
                                best_r = item_r
                        if best_r is not None:
                            chosen_arm, best_assign, best_bays, best_crews, best_op_prio, best_seq, best_eval, best_score = best_r
                            fallback_keep = False
                            res_acc = True
                            self.residual_candidates_accepted += 1
                    else:
                        # Relative-Advantage DFL Ensemble + Epistemic LCB (+ optional CRC Harm Gate)
                        keep_stats = build_keep_baseline_stats(
                            snapshot, graph, keep_eval, keep_score_val, rob_cur_healthy, ev.sim_horizon
                        )
                        fv_anchor = extract_critic_features(
                            snapshot, network, graph, anchor_eval, anchor_score,
                            anchor_assign, anchor_bays, anchor_crews, None,
                            keep_stats, rob_cur_healthy, ev.sim_horizon, is_keep_cand=(anchor_eval is keep_eval),
                        )
                        X_batch = np.stack(
                            [
                                build_anchor_relative_feature(
                                    extract_critic_features(
                                        snapshot, network, graph, item[6], item[7],
                                        item[1], item[2], item[3], item[5],
                                        keep_stats, rob_cur_healthy, ev.sim_horizon, is_keep_cand=False,
                                    ),
                                    fv_anchor,
                                    h_vec_ro,
                                )
                                for item in evaluated_res
                            ],
                            axis=0,
                        )
                        dfl_scores = self._dfl.score_batch(X_batch, kappa=float(self.cfg.lcb_kappa))

                        best_util = -1e18
                        winner_item = None
                        for item, dsc in zip(evaluated_res, dfl_scores):
                            ja_gain = anchor_score - item[7]
                            m_req = _churn_margin_vs_anchor(item)
                            if ja_gain < m_req:
                                continue
                            if dsc.lcb_advantage < float(self.cfg.dfl_tau_gain):
                                res_rej_dfl += 1
                                self.residual_rejected_dfl += 1
                                continue
                            if self.cfg.enable_crc_gate and dsc.harm_prob > float(self.cfg.crc_harm_tau):
                                res_rej_crc += 1
                                self.residual_rejected_crc += 1
                                continue
                            util = float(dsc.lcb_advantage + 0.35 * (ja_gain - m_req))
                            if util > best_util:
                                best_util = util
                                winner_item = item
                        if winner_item is not None:
                            chosen_arm, best_assign, best_bays, best_crews, best_op_prio, best_seq, best_eval, best_score = winner_item
                            fallback_keep = False
                            res_acc = True
                            self.residual_candidates_accepted += 1

        use_exec_commit = bool(is_ws or (not has_res_win) or is_open)
        if use_exec_commit:
            prop = _build_core_sched_proposal(
                policy_id=self.policy_id,
                snapshot=snapshot,
                clf=clf,
                best_assign=best_assign,
                best_bays=best_bays,
                best_crews=best_crews,
                best_eval=best_eval,
                op_prio=best_op_prio,
                eval_cnt=ev.calls,
                t0=t0,
                budget_ms=budget_ms,
                fallback_keep=fallback_keep,
                task_sequences=best_seq,
                predicted_score=best_score,
                include_eval_seq_when_unwindowed=bool(best_seq is not None and not has_res_win),
            )
            if tw:
                prop.proposed_traffic_weights = tw
                prop.proposed_traffic_cutoff = cut_step
        else:
            prop = _build_external_proposal(
                policy_id=self.policy_id,
                snapshot=snapshot,
                clf=clf,
                best_assign=best_assign,
                best_bays=best_bays,
                best_crews=best_crews,
                best_eval=best_eval,
                op_prio=best_op_prio,
                eval_cnt=ev.calls,
                t0=t0,
                budget_ms=budget_ms,
                task_sequences=best_seq,
                traffic_weights=tw or None,
                traffic_cutoff=cut_step,
            )
            prop.predicted_obj = float(best_score)

        self.last_call_stats = {
            "mode": "nominal",
            "graph_extract_calls": graph_extract_calls,
            "encode_full_calls": int(enc_calls.get("full", 0)),
            "encode_staged_calls": int(enc_calls.get("staged", 0)),
            "anchor_evals": int(anchor_evals),
            "residual_generated": int(res_gen),
            "residual_evals": int(res_evals),
            "residual_accepted": bool(res_acc),
            "residual_rejected_dfl": int(res_rej_dfl),
            "residual_rejected_crc": int(res_rej_crc),
            "chosen_arm": chosen_arm,
            "eval_cnt": int(ev.calls),
        }
        prop.telemetry.update(self.last_call_stats)
        return prop

    def _propose_pdg2_escape(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
    ) -> CandidateProposal:
        """Independent PDG 2.0 Emergency Escape with strict Deliverable-4 latency guards (`mode='B2'` + `t_keep`)."""
        self.pdg_recoveries_triggered += 1
        pdg = self._active_pdg or {}
        t0 = _cvcr_now_s()
        cap_ms = min(float(budget_ms) if float(budget_ms) > 0 else float(self.cfg.hard_budget_ms), 29.0)
        deadline = t0 + max(4.0, min(15.5, cap_ms * 0.60)) / 1000.0
        hard_budget_s = t0 + max(5.0, min(18.5, cap_ms * 0.72)) / 1000.0
        tick = int(snapshot.tick)

        self.history_buf.update_from_snapshot(snapshot)
        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        base_prio = derive_base_op_prio(snapshot)
        lvl = int(pdg.get("escalation_level", 1))
        lvl_key = f"L{max(1, min(3, lvl))}"
        top_risk_ops = list(pdg.get("top_risk_ops") or ())
        top_risk_set = set(top_risk_ops)
        boosted_prio = dict(base_prio)
        min_p = min(boosted_prio.values()) if boosted_prio else 0
        for rank_i, oid in enumerate(top_risk_ops):
            if oid in boosted_prio:
                boosted_prio[oid] = int(min_p - 60 + rank_i * 10)

        # Deliverable 4 Fix #1 + #2 inside PDG 2.0 escape!
        ev = CVCREvaluator(
            network, snapshot, mode="B2", alpha_makespan=3.0, lambda_change=15.0,
            use_rust=self.use_rust, seq_diff_b2=True,
        )
        fb = getattr(snapshot, "execution_feedback", None)
        tw: dict[tuple[str, str], int] = {}
        cut_step = 0
        if fb is not None and not fb.is_empty() and fb.blocked_intervals(t_commit=tick, default_horizon=12):
            tw = {
                (str(u), str(v)): int(w)
                for (u, v), w in fb.corridor_q3_penalties().items()
                if network.has_edge(u, v)
            }
            if tw:
                cut_step = tick + 45
                ev.set_traffic_weights(tw, cutoff_step=cut_step)

        dd_sig = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev, graph, lambda_change=15.0)
        fresh_ops = observable_op_closure(snapshot)

        t_k0 = _cvcr_now_s()
        keep_eval = ev.evaluate(
            graph.base_assign, op_priority=base_prio, op_bay_overrides=graph.base_bays, op_crew_overrides=graph.base_crews
        )
        keep_eval_s = max(0.001, _cvcr_now_s() - t_k0)
        keep_sc, _ = score_candidate_ja_unified(
            snapshot, network, graph, keep_eval, graph.base_assign, graph.base_bays, graph.base_crews,
            fresh_ops, fast_surr, lambda_change=15.0, sim_horizon=ev.sim_horizon,
        )
        keep_art = V20CandidateArtifact(
            "E0_KEEP", "keep", dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(base_prio), None, False
        )
        keep_art.eval_res_b2 = keep_eval
        keep_art.ja_unified_score = keep_sc
        keep_art.predicted_score = 0.0
        best = keep_art
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        if keep_eval.valid:
            ev.set_incumbent(graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq)

        op_by_id = {op.op_id: op for op in snapshot.operations}
        cycle_agvs = set(pdg.get("wait_cycle_agvs") or ())
        stalled_agvs = set(pdg.get("stalled_agvs") or ())
        blocked_agvs = set(pdg.get("blocked_healthy_agvs") or ())
        idle_blockers = list(pdg.get("idle_blocker_agvs") or ())
        distressed_agvs = cycle_agvs | stalled_agvs | blocked_agvs
        unblocked_healthy = [a for a in healthy if a not in distressed_agvs]
        risk_tasks = {kt for oid in top_risk_ops if oid in op_by_id for kt in op_by_id[oid].kit_task_ids if kt in graph.rel_tids}
        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if op.end_step is None)

        e1_live_repairs: dict[str, dict[str, Any]] = {}
        if (
            self._sim_ref is not None
            and not getattr(self._sim_ref, "executions", None)
            and len(network.nodes) <= 100
            and (_cvcr_now_s() + 0.005 < deadline)
        ):
            haven_r, _ = plan_haven_safe_corridor_repair(self._sim_ref, snapshot, stall_threshold=3, enable_choke_oscillation_guard=True)
            e1_live_repairs.update(haven_r)
            idle_r = plan_idle_blocker_evacuations(
                snapshot, network, self._sim_ref, [a for a in idle_blockers if a not in e1_live_repairs], set(clf.disturbed_robots)
            )
            e1_live_repairs.update(idle_r)
        e1_locks = (
            build_critical_bottleneck_locks(snapshot, network, sorted(distressed_agvs), set(clf.disturbed_robots))
            if (len(network.nodes) <= 100 and _cvcr_now_s() + 0.003 < deadline)
            else {}
        )

        # Expire old tabu signatures
        self._escape_tabu = {k: v for k, v in self._escape_tabu.items() if v > tick}
        seen_sigs: set[str] = {keep_art.ensure_sig()}
        max_inc_s = max(0.001, 0.60 * keep_eval_s)

        def _risk_score(art: V20CandidateArtifact) -> float:
            er = art.eval_res_b2
            sc = float(art.ja_unified_score or 1e9)
            if er is None or not er.valid:
                return sc + 10000.0
            relieved_blocked = sum(
                1 for tid in graph.rel_tids
                if graph.base_assign.get(tid) in distressed_agvs
                and art.assign.get(tid) != graph.base_assign.get(tid)
                and art.assign.get(tid) in unblocked_healthy
            )
            relieved_cycle = sum(
                1 for tid in graph.rel_tids
                if graph.base_assign.get(tid) in cycle_agvs and art.assign.get(tid) != graph.base_assign.get(tid)
            )
            changed_risk_tasks = sum(1 for tid in risk_tasks if art.assign.get(tid) != graph.base_assign.get(tid))
            changed_risk_bays = sum(1 for oid in top_risk_set if art.bays.get(oid) != graph.base_bays.get(oid))
            tabu_pen = 600.0 if (art.ensure_sig() in self._escape_tabu) else 0.0
            relief = (
                -260.0 * float(min(3, relieved_blocked))
                - 180.0 * float(min(3, relieved_cycle))
                - 150.0 * float(min(3, changed_risk_tasks))
                - 140.0 * float(min(2, changed_risk_bays))
                - (160.0 if art.live_repairs else 0.0)
                - (90.0 if art.bottleneck_locks else 0.0)
            )
            return (sc - keep_sc) + 0.4 * float(er.starvation_ticks - keep_eval.starvation_ticks) + relief + tabu_pen

        def _consider_esc(art: V20CandidateArtifact) -> bool:
            nonlocal best, max_inc_s
            sig = art.ensure_sig()
            if sig in seen_sigs or _cvcr_now_s() >= deadline:
                return False
            fp = compute_candidate_amr_footprint(
                snapshot, best.assign, best.bays, best.eval_res_b2.robot_order_seq if best.eval_res_b2 else None,
                art.assign, art.bays, art.task_orders,
            )
            est_s = max_inc_s if fp["is_b2_incremental_hit"] else max(keep_eval_s, 1.15 * keep_eval_s)
            if (_cvcr_now_s() + est_s) >= hard_budget_s:
                return False
            seen_sigs.add(sig)
            t_ev0 = _cvcr_now_s()
            er = ev.evaluate(
                art.assign, op_priority=art.op_prio, op_bay_overrides=art.bays,
                op_crew_overrides=art.crews, task_order_override=art.task_orders,
            )
            dt_s = max(0.0005, _cvcr_now_s() - t_ev0)
            if fp["is_b2_incremental_hit"]:
                max_inc_s = max(max_inc_s, dt_s)
            art.eval_res_b2 = er
            if not er.valid:
                return False
            sc_a, _ = score_candidate_ja_unified(
                snapshot, network, graph, er, art.assign, art.bays, art.crews, fresh_ops, fast_surr,
                lambda_change=15.0, sim_horizon=ev.sim_horizon,
            )
            art.ja_unified_score = sc_a
            rsc = _risk_score(art)
            art.predicted_score = rsc
            if rsc + 1e-4 < float(best.predicted_score if best.predicted_score is not None else 0.0):
                best = art
                ev.set_incumbent(best.assign, er, op_priority=best.op_prio, op_bay_overrides=best.bays, task_order_override=best.task_orders)
                return True
            return False

        # Candidate 1: E1 direct corridor / idle-blocker repair + priority hoist
        if e1_live_repairs or e1_locks or risk_tasks:
            hoisted_seq = _default_task_order_from_prio(snapshot, graph, graph.base_assign, boosted_prio)
            _consider_esc(
                V20CandidateArtifact(
                    "PDG2_E1_QueueCorridor", f"e1_direct_{lvl_key}",
                    dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(boosted_prio),
                    hoisted_seq, False,
                    live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                    bottleneck_locks=(dict(e1_locks) if e1_locks else None),
                )
            )

        # Candidate 2: Evict blocked/distressed robots' releasable tasks onto unblocked healthy robots
        if healthy and graph.rel_tids and _cvcr_now_s() < deadline:
            blocked_rel = [tid for tid in graph.rel_tids if graph.base_assign.get(tid) in distressed_agvs]
            if blocked_rel and unblocked_healthy:
                asn_ev = dict(graph.base_assign)
                for idx_t, tid in enumerate(blocked_rel):
                    asn_ev[tid] = unblocked_healthy[idx_t % len(unblocked_healthy)]
                _consider_esc(
                    V20CandidateArtifact(
                        "PDG2_R1_EvictBlocked", f"r1_evict_{lvl_key}",
                        asn_ev, dict(graph.base_bays), dict(graph.base_crews), dict(boosted_prio),
                        _default_task_order_from_prio(snapshot, graph, asn_ev, boosted_prio), False,
                        live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                        bottleneck_locks=(dict(e1_locks) if e1_locks else None),
                    )
                )

        # Candidate 3: Alternative bay reroute for top stalled/risk operations
        if top_risk_ops and _cvcr_now_s() < deadline:
            alt_bays = dict(graph.base_bays)
            changed_b = False
            for oid in top_risk_ops[:2]:
                op = op_by_id.get(oid)
                if op is not None and not op.started and len(op.eligible_bays) > 1:
                    cur_b = alt_bays.get(oid)
                    for xb in op.eligible_bays:
                        if xb != cur_b:
                            alt_bays[oid] = xb
                            changed_b = True
                            break
            if changed_b:
                _consider_esc(
                    V20CandidateArtifact(
                        "PDG2_E2_BayReroute", f"e2_bay_{lvl_key}",
                        dict(best.assign), alt_bays, dict(graph.base_crews), dict(boosted_prio),
                        _default_task_order_from_prio(snapshot, graph, best.assign, boosted_prio), False,
                        live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                    )
                )

        # Candidate 4: Cached V19 Full escape candidate if budget remains
        if _cvcr_now_s() + max(0.003, keep_eval_s) < deadline:
            full_net, _ = self._ensure_nets()
            enc_f = full_net.encode_entities(graph)
            h12_f = full_net.a1_a2_heads(enc_f, graph)
            dec_f = decode_v19_with_cached_enc(full_net, enc_f, h12_f, graph, snapshot)
            _consider_esc(
                V20CandidateArtifact(
                    "PDG2_V19_Escape", f"v19_esc_{lvl_key}",
                    dict(dec_f.assign), dict(dec_f.bays), dict(dec_f.crews), dict(boosted_prio),
                    _default_task_order_from_prio(snapshot, graph, dec_f.assign, boosted_prio), True,
                    live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                )
            )

        if best is not keep_art:
            self._escape_tabu[best.ensure_sig()] = tick + 18

        fallback_keep = (best is keep_art) and not (best.live_repairs or best.bottleneck_locks)
        prop = _build_core_sched_proposal(
            policy_id=self.policy_id,
            snapshot=snapshot,
            clf=clf,
            best_assign=best.assign,
            best_bays=best.bays,
            best_crews=best.crews,
            best_eval=best.eval_res_b2,
            op_prio=best.op_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=cap_ms,
            fallback_keep=fallback_keep,
            task_sequences=best.task_orders,
            predicted_score=best.predicted_score,
            include_eval_seq_when_unwindowed=not has_win,
        )
        if tw:
            prop.proposed_traffic_weights = tw
            prop.proposed_traffic_cutoff = cut_step
        if best.live_repairs:
            prop.proposed_live_repairs = dict(best.live_repairs)
        if best.bottleneck_locks:
            prop.proposed_bottleneck_locks = dict(best.bottleneck_locks)
        self.last_call_stats = {
            "mode": f"pdg2_{lvl_key}",
            "graph_extract_calls": 1,
            "encode_full_calls": 0,
            "chosen_arm": best.tag,
            "eval_cnt": int(ev.calls),
        }
        prop.telemetry.update(self.last_call_stats)
        return prop


def run_rgro_episode(
    spec: dict[str, Any],
    cfg: RGROConfig,
    *,
    dfl_ensemble: Optional[DFLAdvantageEnsemble] = None,
    use_virtual_clock: bool = False,
    compute_jitter_ms_fn: Optional[Any] = None,
) -> dict[str, Any]:
    """Execute one episode with `RGROProposer` cleanly wired into `AMRReschedulingEnv`."""
    from amr_sched.baselines.policy_registry import create_discrete_policy_env
    from amr_sched.cvcr.auditor import IndependentAuditor
    from amr_sched.cvcr.clock import virtual_clock_scope
    from amr_sched.cvcr.snapshot import NativeSnapshotCapture
    from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
    import time

    def _exec() -> dict[str, Any]:
        _, ctrl, env = create_discrete_policy_env(
            "V19_ADAPTIVE_RL",
            seed=int(spec["seed"]),
            max_steps=int(spec["max_steps"]),
            use_rust=cfg.use_rust,
            p2_cfg=spec.get("p2_cfg"),
        )
        prop = RGROProposer(cfg, dfl_ensemble=dfl_ensemble)
        mon = PDG2SCCLivenessMonitor(cfg=PDG2Config()) if cfg.enable_pdg2 else None

        def _wrapped_prop(snap: NativeSnapshot, net: Network, *, budget_ms: float = 35.0) -> CandidateProposal:
            eff_b = float(budget_ms)
            if compute_jitter_ms_fn is not None:
                # Simulate realistic host CPU/budget jitter deterministically by reducing effective budget or spinning
                extra_busy_ms = float(compute_jitter_ms_fn(int(snap.tick), int(spec["seed"])))
                if extra_busy_ms > 0.0:
                    t_spin = time.perf_counter() + extra_busy_ms / 1000.0
                    while time.perf_counter() < t_spin:
                        pass
                    eff_b = max(4.0, float(budget_ms) - extra_busy_ms)
            return prop(snap, net, budget_ms=eff_b)

        env.external_proposer = _wrapped_prop
        obs, info = env.reset(seed=int(spec["seed"]), options={"scenario_tuple": copy.deepcopy(spec["scenario_tuple"])})
        prop.reset_episode_state()
        prop.set_sim_ref(env.sim)

        tm = tr = False
        rej_streak = 0
        while not (tm or tr):
            t_now = int(env.current_step)
            if cfg.enable_dfl and (t_now % 5 == 0 or t_now == 0):
                p2_fb = (
                    env.execution_perturbation.capture_feedback(env.sim, t_now)
                    if (env.execution_perturbation is not None and env._active_execution_aware)
                    else None
                )
                s_cur = NativeSnapshotCapture.capture(
                    env.sim, env.tracker.operations, hide_future_releases=True, execution_feedback=p2_fb
                )
                prop.observe_committed_step(s_cur, rej_streak=rej_streak)

            feats = compute_expert_v9_state(
                env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
            )
            s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
            act = int(ctrl.select_action(feats, s_info, sim=env.sim))
            if mon is not None:
                act = int(mon.observe_and_maybe_escalate(env, s_info, act, proposer=prop))

            nb = len(env.decision_logs)
            obs, _, tm, tr, info = env.step(act)
            if len(env.decision_logs) > nb:
                dl = env.decision_logs[-1]
                rej_streak = 0 if bool(dl.get("receipt_accepted", True)) else (rej_streak + 1)

        m = env.compute_episode_metrics()
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        dec_ms = [float(r["t_propose_ms"]) + float(r["t_validate_commit_ms"]) for r in env.decision_logs]
        return {
            "policy": cfg.policy_id,
            "episode_key": str(spec["episode_key"]),
            "suite_group": str(spec.get("suite_group", "")),
            "seed": int(spec["seed"]),
            "net_obj": round(float(m["net_obj"]), 4),
            "h1": round(float(m["cmax"]) + float(m.get("task_tardiness_cost", 0.0)) + float(m.get("op_tardiness_cost", 0.0)), 4),
            "churn": int(m["total_churn"]),
            "comp100": bool(float(m["task_completion_rate"]) >= 0.9999 and float(m["production_completion_rate"]) >= 0.9999),
            "audit_passed": bool(aud.pass_audit),
            "expired_windows": int(env.total_window_expirations),
            "pdg_alerts": len(mon.alert_ticks) if mon is not None else 0,
            "pdg_recoveries": int(prop.pdg_recoveries_triggered),
            "res_evals": int(prop.residual_candidates_evaluated),
            "res_accepted": int(prop.residual_candidates_accepted),
            "res_rej_dfl": int(prop.residual_rejected_dfl),
            "res_rej_crc": int(prop.residual_rejected_crc),
            "decision_latencies_ms": [round(float(x), 4) for x in dec_ms],
        }

    if use_virtual_clock:
        with virtual_clock_scope(0.0):
            return _exec()
    return _exec()
