"""Unified Closure Solver `RGROClosureProposer` (`IF-CORE-RGRO-CLOSURE-01` Tasks 1-5).

Combines cleanly without any scenario/seed/Track-ID branching:
  1. `UnifiedDeadlineManager` + Zero-Expiry Transaction Hard Veto (Task 1),
  2. Shared Layer-1 context (`mode="B2", seq_diff_b2=True` + live `set_traffic_weights`),
  3. Toggleable structured `V18` Quality Recovery (`O1` Opening, `O2` Queue-Only, `O3` Low-Churn Diverse) (Task 3),
  4. Toggleable `DFLAdvantageEnsemble` (single-horizon `48t`) or `MultiHorizonDFLEnsemble` (`[16,48,128,full]`)
     with LCB uncertainty & optional CRC harm-risk gate (Tasks 2, 4, 5),
  5. Independent `PDG 2.0` Emergency Escape under the same unified absolute deadline + veto check.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
import gc
from pathlib import Path
import time
from typing import Any, Callable, Optional

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
from experiments.rgro.dfl_critic import DFLAdvantageEnsemble, ExecutionHistoryBuffer, build_anchor_relative_feature
from experiments.rgro_closure.deadline_manager import UnifiedDeadlineContext, UnifiedDeadlineManager
from experiments.rgro_closure.multi_horizon_dfl import MultiHorizonDFLEnsemble
from experiments.rgro_closure.v18_recovery_operators import (
    generate_o1_opening_candidates,
    generate_o2_queue_refinement,
    generate_o3_diverse_low_churn,
    is_opening_zero_commitment,
)


@dataclass(frozen=True)
class ClosureConfig:
    """Declarative configuration for all closure ablations & Cohort F comparators."""

    policy_id: str = "RGRO_U4_Closure"
    enable_pdg2: bool = True
    enable_v20_residual: bool = True
    # V18 quality recovery toggles (Task 3)
    enable_o1_opening: bool = False
    enable_o2_queue: bool = False
    enable_o3_diverse: bool = False
    # Critic / Risk gate toggles (Tasks 2 & 4)
    critic_mode: str = "dfl_lcb"  # "none" (Det), "dfl_nolcb", "dfl_lcb", "dfl_lcb_crc", "mh_dfl_lcb", "mh_dfl_lcb_crc"
    lcb_kappa: float = 0.75
    dfl_tau_gain: float = 20.0
    crc_harm_tau: float = 0.15
    res_ja_margin: float = 8.0
    # Unified deadline manager (Task 1)
    sla_target_ms: float = 20.5
    window_tight_sla_ms: float = 17.5
    max_residual_evals: int = 3
    use_rust: bool = True
    seed: int = 0


class RGROClosureProposer:
    """Shared-Budget `RGRO` Proposer with Unified Deadline Closure & Zero-Expiry Hard Veto."""

    def __init__(
        self,
        cfg: ClosureConfig,
        *,
        dfl_ensemble: Optional[DFLAdvantageEnsemble] = None,
        mh_ensemble: Optional[MultiHorizonDFLEnsemble] = None,
        full_ckpt: Path = _DEFAULT_V19_RL_CKPT,
        staged_ckpt: Path = _DEFAULT_V19_STAGED_CKPT,
    ) -> None:
        self.cfg = cfg
        self.policy_id = cfg.policy_id
        self.use_rust = bool(cfg.use_rust)
        self.seed = int(cfg.seed)

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

        self._dfl = dfl_ensemble
        self._mh_dfl = mh_ensemble
        self.history_buf = ExecutionHistoryBuffer(maxlen=12)
        self.dl_mgr = UnifiedDeadlineManager(
            sla_target_ms=cfg.sla_target_ms,
            window_tight_sla_ms=cfg.window_tight_sla_ms,
        )

        self._active_pdg: Optional[dict[str, Any]] = None
        self._sim_ref: Any = None
        self._escape_tabu: dict[str, int] = {}

        # Per-episode counters
        self.pdg_recoveries_triggered: int = 0
        self.residual_candidates_evaluated: int = 0
        self.residual_candidates_accepted: int = 0
        self.veto_count: int = 0
        self.fallback_count: int = 0
        self.internal_overrun_count: int = 0
        self.last_call_stats: dict[str, Any] = {}
        self.call_latency_records: list[dict[str, Any]] = []
        if self.cfg.enable_o1_opening or self.cfg.enable_v20_residual:
            self._ensure_pgmpo()

    def reset_episode_state(self) -> None:
        dim = int(self._v19_holder._A[0].shape[0])
        self._v19_holder._A = [np.eye(dim, dtype=np.float64), np.eye(dim, dtype=np.float64)]
        self._v19_holder._b = [np.zeros(dim, dtype=np.float64), np.zeros(dim, dtype=np.float64)]
        self.history_buf.reset()
        self.dl_mgr.reset()
        self._active_pdg = None
        self._escape_tabu.clear()
        self.pdg_recoveries_triggered = 0
        self.residual_candidates_evaluated = 0
        self.residual_candidates_accepted = 0
        self.veto_count = 0
        self.fallback_count = 0
        self.internal_overrun_count = 0
        self.last_call_stats = {}
        self.call_latency_records = []

    def set_pdg_state(self, pdg: Optional[dict[str, Any]]) -> None:
        self._active_pdg = pdg

    def set_pdg2_context(self, pdg: Optional[dict[str, Any]], *, sim_ref: Any = None) -> None:
        self._active_pdg = pdg
        if sim_ref is not None:
            self._sim_ref = sim_ref

    def set_sim_ref(self, sim: Any) -> None:
        self._sim_ref = sim

    def observe_committed_step(self, snapshot: NativeSnapshot, *, rej_streak: int = 0) -> np.ndarray:
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

                v18 = load_internal_cvcr_policy("V18_EA_JOINT")
                pg = getattr(v18, "pgmpo", None)
                self._pgmpo = pg if (pg is not None and getattr(pg, "ready", False)) else None
            except Exception as exc:
                raise RuntimeError(f"Failed to load V18 PGMPO generator: {exc}") from exc
        return self._pgmpo

    def _encode_cached(
        self,
        net: NumpyV19HeteroConstructorNet,
        graph: V19HeteroGraph,
        cache_dict: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]],
        key: str,
        enc_calls: dict[str, int],
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if key not in cache_dict:
            enc = net.encode_entities(graph)
            h12 = net.a1_a2_heads(enc, graph)
            cache_dict[key] = (enc, h12)
            enc_calls[key] = enc_calls.get(key, 0) + 1
        return cache_dict[key]

    def __call__(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float = 35.0,
        injected_jitter_ms: float = 0.0,
    ) -> CandidateProposal:
        t0 = _cvcr_now_s()
        gc_pause_ms = 0.0
        gc_t0 = [0.0]

        def _gc_cb(phase: str, _info: dict[str, Any]) -> None:
            nonlocal gc_pause_ms
            if phase == "start":
                gc_t0[0] = _cvcr_now_s()
            elif phase == "stop" and gc_t0[0] > 0.0:
                gc_pause_ms += max(0.0, (_cvcr_now_s() - gc_t0[0]) * 1000.0)
                gc_t0[0] = 0.0

        gc.callbacks.append(_gc_cb)
        try:
            if self.cfg.enable_pdg2 and self._active_pdg and self._active_pdg.get("pdg_alert"):
                prop = self._propose_pdg2_escape(
                    snapshot, network, budget_ms=budget_ms, t0_s=t0, injected_jitter_ms=injected_jitter_ms
                )
            else:
                prop = self._propose_nominal(
                    snapshot, network, budget_ms=budget_ms, t0_s=t0, injected_jitter_ms=injected_jitter_ms
                )
        finally:
            if _gc_cb in gc.callbacks:
                gc.callbacks.remove(_gc_cb)

        # Backfill exact GC pause and decompose internal vs OS jitter
        tel = prop.telemetry or {}
        tot_prop_ms = (_cvcr_now_s() - t0) * 1000.0
        os_j = float(tel.get("os_jitter_ms", 0.0)) + gc_pause_ms
        int_ms = max(0.0, tot_prop_ms - os_j - float(injected_jitter_ms))
        prop.telemetry["gc_pause_ms"] = round(gc_pause_ms, 3)
        prop.telemetry["os_jitter_ms"] = round(os_j + float(injected_jitter_ms), 3)
        prop.telemetry["internal_algo_ms"] = round(int_ms, 3)
        prop.telemetry["total_propose_wall_ms"] = round(tot_prop_ms, 3)
        self.call_latency_records.append(dict(prop.telemetry))
        return prop

    def _emit_proposal(
        self,
        *,
        snapshot: NativeSnapshot,
        clf: Any,
        graph: V19HeteroGraph,
        best_assign: dict[str, str],
        best_bays: dict[str, str],
        best_crews: dict[str, int],
        best_op_prio: dict[str, int],
        best_seq: Optional[dict[str, list[str]]],
        best_eval: EvalResult,
        best_score: float,
        fallback_keep: bool,
        eval_cnt: int,
        tw: dict[tuple[str, str], int],
        cut_step: int,
        dl: UnifiedDeadlineContext,
        budget_ms: float,
        is_ws: bool,
        has_res_win: bool,
        is_open: bool,
        include_eval_seq_unwin: bool = True,
    ) -> CandidateProposal:
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
                eval_cnt=eval_cnt,
                t0=dl.t0_s,
                budget_ms=budget_ms,
                fallback_keep=fallback_keep,
                task_sequences=best_seq,
                predicted_score=best_score,
                include_eval_seq_when_unwindowed=bool(include_eval_seq_unwin and best_seq is not None and not has_res_win),
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
                eval_cnt=eval_cnt,
                t0=dl.t0_s,
                budget_ms=budget_ms,
                task_sequences=best_seq,
                traffic_weights=tw or None,
                traffic_cutoff=cut_step,
            )
            prop.predicted_obj = float(best_score)
        return prop

    def _propose_nominal(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float,
        t0_s: float,
        injected_jitter_ms: float,
    ) -> CandidateProposal:
        import amr_sched.cvcr.v17_window_protection as _v17
        import amr_sched.cvcr.v18_frontier_synthesis as _v18

        select_coupled_op_pair = getattr(_v17, "select_coupled_op_pair", getattr(_v18, "select_coupled_op_pair", None))
        full_net, staged_net = self._ensure_nets()

        h_vec_ro = self.history_buf.update_from_snapshot(snapshot)
        clf = classify_commitments(snapshot)
        dl = self.dl_mgr.begin_decision(
            snapshot, clf, budget_ms=budget_ms, t0_s=t0_s, injected_jitter_ms=injected_jitter_ms
        )
        tick = int(snapshot.tick)

        has_res_win = any(bool(getattr(op, "resource_windows", ())) for op in snapshot.operations)
        t_g0 = _cvcr_now_s()
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        g_ms = (_cvcr_now_s() - t_g0) * 1000.0
        dl.record_phase("preamble_graph", g_ms)
        base_pre = self.dl_mgr.observe_graph_ms(g_ms)
        dl.detect_os_jitter(base_pre)

        enc_cache: dict[str, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}
        enc_calls: dict[str, int] = {"full": 0, "staged": 0}
        healthy_agvs = [graph.agv_ids[i] for i in graph.healthy_agv_indices]
        phi = compute_dynamic_coupling_features(graph)
        is_ws = len(network.nodes) >= 30
        is_open = (tick <= 4)
        is_open_zero = is_opening_zero_commitment(snapshot)

        ev = CVCREvaluator(
            network, snapshot, alpha_makespan=3.0, lambda_change=15.0,
            use_rust=self.use_rust, mode="B2", seq_diff_b2=True,
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

        base_op_prio = {op.op_id: max(tick, int(op.due_step) - int(op.proc_steps)) for op in snapshot.operations}
        t_keep0 = _cvcr_now_s()
        keep_eval = ev.evaluate(
            graph.base_assign, op_priority=base_op_prio, op_bay_overrides=graph.base_bays, op_crew_overrides=graph.base_crews
        )
        keep_eval_s = max(0.001, _cvcr_now_s() - t_keep0)
        dl.record_phase("keep_eval", keep_eval_s * 1000.0)
        if keep_eval.valid:
            ev.set_incumbent(graph.base_assign, keep_eval, op_priority=base_op_prio, op_bay_overrides=graph.base_bays)

        best_eval: EvalResult = keep_eval
        best_assign = dict(graph.base_assign)
        best_bays = dict(graph.base_bays)
        best_crews = dict(graph.base_crews)
        best_op_prio = dict(base_op_prio)
        best_seq: Optional[dict[str, list[str]]] = None
        fallback_keep = True
        chosen_arm = "keep"

        rob_cur_healthy = {
            r.robot_id: r.cur_task for r in snapshot.robots if r.cur_task and r.robot_id not in clf.disturbed_robots
        }
        fresh_op_ids, pending_or_dist_set, _ = compute_stability_sets(snapshot, graph)
        obs_ops = observable_op_closure(snapshot)
        tasks_by_id = {t.task_id: t for t in snapshot.tasks}
        is_resc = bool(clf.disturbed_robots) or any(
            (t.assigned_agv is None or t.status.lower() == "pending") and getattr(t, "is_urgent", False)
            for t in snapshot.tasks
        )
        bay_delta = {graph.bay_ids[i]: float(graph.bay_detour_norm[i]) * 15.0 for i in range(len(graph.bay_ids))}
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev, graph, lambda_change=15.0)

        def _score_ja(asn: dict[str, str], bs: dict[str, str], cs_c: dict[str, int], ev_c: EvalResult) -> tuple[float, float]:
            return score_candidate_ja_unified(
                snapshot, network, graph, ev_c, asn, bs, cs_c, obs_ops, fast_surr,
                lambda_change=15.0, sim_horizon=ev.sim_horizon,
            )

        keep_score_val, keep_tard_val = _score_ja(graph.base_assign, graph.base_bays, graph.base_crews, keep_eval)
        best_score = keep_score_val
        anchor_evals = 0
        res_evals = 0
        res_acc = False

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

        # Immediate short-circuit ONLY if preamble + keep_eval already nearly exhausted propose cap
        if graph.rel_tids and healthy_agvs and dl.can_run_before(dl.abs_propose_deadline_s, 0.0012):
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
                tag_s: str, asn_s: dict[str, str], bs_s: dict[str, str], cs_s: dict[str, int],
                op_s: dict[str, int], seq_s: Optional[dict[str, list[str]]] = None,
                *, allow_until_abs: bool = False,
            ) -> bool:
                nonlocal best_eval, best_score, best_assign, best_bays, best_crews, best_op_prio, best_seq
                nonlocal fallback_keep, chosen_arm, anchor_evals, max_inc_eval_s, max_full_eval_s

                sq_k = tuple((k, tuple(v)) for k, v in sorted(seq_s.items())) if seq_s else None
                sg = (
                    tuple(sorted(asn_s.items())),
                    tuple(sorted(bs_s.items())),
                    tuple(sorted(cs_s.items())),
                    tuple(sorted(op_s.items())),
                    sq_k,
                )
                use_abs = bool(allow_until_abs and (best_eval is keep_eval))
                cutoff_s = dl.abs_propose_deadline_s if use_abs else dl.anchor_hard_s
                if sg in seen_sigs or not dl.can_run_before(cutoff_s, 0.0008):
                    return False

                fp = compute_candidate_amr_footprint(
                    snapshot, best_assign, best_bays, best_eval.robot_order_seq, asn_s, bs_s, seq_s
                )
                est_ev_s = max_inc_eval_s if fp["is_b2_incremental_hit"] else max(max_full_eval_s, 1.05 * keep_eval_s)
                if not dl.can_run_before(cutoff_s, est_ev_s + 0.0012):
                    return False

                seen_sigs.add(sg)
                t_ev0 = _cvcr_now_s()
                ev_c = ev.evaluate(
                    asn_s, op_priority=op_s, op_bay_overrides=bs_s, op_crew_overrides=cs_s, task_order_override=seq_s
                )
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
                            1 for tid_c, nr in asn_s.items()
                            if tid_c in tasks_by_id and tasks_by_id[tid_c].status.lower() == "assigned"
                            and tasks_by_id[tid_c].assigned_agv and tasks_by_id[tid_c].assigned_agv != nr
                            and tasks_by_id[tid_c].assigned_agv not in clf.disturbed_robots
                        )
                        nso = sum(
                            1 for oid in graph.op_ids
                            if oid not in fresh_op_ids
                            and (bs_s.get(oid, graph.base_bays[oid]) != graph.base_bays[oid] or cs_s.get(oid, graph.base_crews[oid]) != graph.base_crews[oid])
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
                    best_assign, best_bays, best_crews, best_op_prio, best_seq = asn_s, bs_s, cs_s, op_s, seq_s
                    chosen_arm = tag_s
                    fallback_keep = False
                    ev.set_incumbent(best_assign, best_eval, op_priority=best_op_prio, op_bay_overrides=best_bays, task_order_override=best_seq)
                    return True
                return False

            # Stage 1: Fast V19 Anchor Neural Candidates (reusing Layer-1 encodings ONCE)
            nn_cands = []
            lock_nn_bay = bool(is_ws and len(graph.agv_ids) >= 4 and (not is_open))
            nn_loop_cap = 1 if (self.cfg.enable_o1_opening and is_open_zero) else 3
            for k in range(nn_loop_cap):
                if k > 0 and ((not keep_eval.valid) or dl.elapsed_ms() > 7.5 or not dl.can_run_before(dl.anchor_soft_s, 0.0025)):
                    break
                use_dual_other = (k == 1 and len(graph.agv_ids) >= 4)
                use_arm_name = other_mode if use_dual_other else selected_mode
                use_net = full_net if use_arm_name == "full" else staged_net
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
                sq_nn = dict(cand.task_sequences) if (is_open_zero and getattr(cand, "task_sequences", None)) else None
                nn_cands.append((raw_tag, cand.assign, bn, cn, cand.op_prio, sq_nn))
                if pending_or_dist_set and not is_open:
                    za = {
                        tid: (cand.assign.get(tid, graph.base_assign[tid]) if tid in pending_or_dist_set else graph.base_assign[tid])
                        for tid in graph.base_assign
                    }
                    nn_cands.append((f"{use_arm_name}_zc", za, bn, cn, cand.op_prio, None))

            for i_n, item_n in enumerate(nn_cands[:3]):
                _try_verify_anchor(*item_n, allow_until_abs=(i_n == 0))

            # Task 3 O1: V18 Opening Combinatorial Constructor (ONLY when 0 execution commitments)
            if self.cfg.enable_o1_opening and is_open_zero and dl.can_run_before(dl.abs_propose_deadline_s, 0.0018):
                o1_cands = generate_o1_opening_candidates(
                    snapshot, network, clf, graph, fast_surr, base_op_prio, bay_delta, healthy_agvs, fresh_op_ids,
                    self._ensure_pgmpo(), max_candidates=2,
                )
                for tg_o, asn_o, bs_o, cs_o, op_o, sq_o, _ in o1_cands:
                    _try_verify_anchor(tg_o, asn_o, bs_o, cs_o, op_o, sq_o, allow_until_abs=True)

            # Stage 2: V19 Causal Repair Cone (guarded by anchor_soft_s, or abs_propose_deadline_s if still invalid!)
            if dl.can_run_before(dl.anchor_soft_s, 0.0018) or ((not best_eval.valid) and dl.can_run_before(dl.abs_propose_deadline_s, 0.0022)):
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
                ccands = []
                if (not is_ws) and pending_or_dist_set and not is_open:
                    ha_anc = {tid: aid for tid, aid in best_assign.items() if tid not in pending_or_dist_set}
                    ccands.append((
                        "cone_regret2_zc_keepbay",
                        fast_surr.regret_k_assignment(healthy_agvs, ha_anc, sorted(pending_or_dist_set), best_bays, best_op_prio),
                        dict(best_bays), dict(best_crews), dict(best_op_prio), None,
                    ))
                if pending_or_dist_set:
                    ha_anc = {tid: aid for tid, aid in best_assign.items() if tid not in pending_or_dist_set}
                    ccands.append((
                        "cone_regret2_zc",
                        fast_surr.regret_k_assignment(healthy_agvs, ha_anc, sorted(pending_or_dist_set), b_opt, p_co),
                        b_opt, c_opt, p_co, None,
                    ))
                if not is_ws:
                    fx_anc = {tid: aid for tid, aid in best_assign.items() if tid not in set(graph.rel_tids)}
                    ccands.append((
                        "cone_regret2_full",
                        fast_surr.regret_k_assignment(healthy_agvs, fx_anc, sorted(graph.rel_tids), b_opt, p_co),
                        b_opt, c_opt, p_co, None,
                    ))
                dc = sorted(set(pending_or_dist_set) | set(cone.cone_tasks))
                if dc and len(dc) < len(graph.rel_tids):
                    ac = {tid: aid for tid, aid in best_assign.items() if tid not in set(dc)}
                    dc_b = best_bays if ((not is_ws) and not is_open) else b_opt
                    ccands.append((
                        "cone_regret2_lns",
                        fast_surr.regret_k_assignment(healthy_agvs, ac, dc, dc_b, p_co),
                        dc_b, c_opt, p_co, None,
                    ))
                c_cap = 3 if (not is_ws) else 2
                for item_c in ccands[:c_cap]:
                    _try_verify_anchor(*item_c)

                if select_coupled_op_pair is not None and has_res_win and best_eval.valid and dl.can_run_before(dl.anchor_hard_s, 0.002):
                    dops = _v18._find_danger_cliff_ops(snapshot, best_eval, bay_delta)
                    pr = select_coupled_op_pair(network, snapshot, best_eval, best_assign, best_op_prio) if dops else None
                    if dops and pr is not None:
                        for ldr, yld in ((pr[0], pr[1]), (pr[1], pr[0])):
                            if not dl.can_run_before(dl.anchor_hard_s, max(0.0015, max_inc_eval_s)):
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

            impr = max(-1.0, min(1.0, (keep_score_val - best_score) / 200.0))
            self._v19_holder._A[arm_pick] += np.outer(x_ctx, x_ctx)
            self._v19_holder._b[arm_pick] += impr * x_ctx

            anchor_eval = best_eval
            anchor_score = best_score
            anchor_assign = dict(best_assign)
            anchor_bays = dict(best_bays)
            anchor_crews = dict(best_crews)
            anchor_op_prio = dict(best_op_prio)
            anchor_seq = best_seq
            anchor_was_keep = bool(fallback_keep)
            anchor_arm = chosen_arm

            # ---- STAGE 3 & 4: RESIDUAL SEARCH + O2/O3 RECOVERY + DFL/MH-DFL + CRC ----
            if (self.cfg.enable_v20_residual or self.cfg.enable_o2_queue or self.cfg.enable_o3_diverse) and (
                dl.can_run_before(dl.res_soft_s, max(0.0018, 1.05 * max_inc_eval_s + 0.0012)) or not anchor_eval.valid
            ):
                res_pool = []

                if self.cfg.enable_o2_queue and dl.can_run_before(dl.res_soft_s, 0.0012):
                    res_pool.extend(
                        generate_o2_queue_refinement(
                            snapshot, graph, fast_surr, anchor_assign, anchor_bays, anchor_crews, anchor_op_prio, anchor_eval
                        )
                    )

                if self.cfg.enable_o3_diverse and dl.can_run_before(dl.res_soft_s, 0.0015):
                    res_pool.extend(
                        generate_o3_diverse_low_churn(
                            snapshot, graph, fast_surr, anchor_assign, anchor_bays, anchor_crews, anchor_op_prio,
                            anchor_eval, healthy_agvs, fresh_op_ids,
                        )
                    )

                if self.cfg.enable_v20_residual and dl.can_run_before(dl.res_soft_s, 0.0015):
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

                    pgmpo = self._ensure_pgmpo()
                    pgmpo_cutoff = dl.abs_propose_deadline_s if (not anchor_eval.valid) else dl.res_soft_s
                    if pgmpo is not None and (is_ws or not anchor_eval.valid) and dl.can_run_before(pgmpo_cutoff, 0.0020):
                        b_pg = dict(anchor_bays)
                        if not anchor_eval.valid:
                            b_pg, _ = _v18.compute_semicircle_kitting_bays(
                                snapshot, network, clf, fast_surr, anchor_bays, bay_delta, eligible_op_ids=fresh_op_ids
                            )
                        lazy_surr = _LazyKittingSurrogate(
                            network, snapshot, anchor_op_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta
                        )
                        asns = pgmpo.generate_assignments(snapshot, network, clf, healthy_agvs, graph.fixed_assign, b_pg, lazy_surr)
                        if asns:
                            ranked_p = sorted(
                                asns, key=lambda a: float(fast_surr.score(a, anchor_op_prio, b_pg, anchor_crews))
                            )
                            for i_p, top_p in enumerate(ranked_p[: (2 if not anchor_eval.valid else 1)]):
                                res_pool.append((
                                    f"res_pgmpo_{i_p}", dict(top_p), dict(b_pg), dict(anchor_crews), dict(anchor_op_prio),
                                    _default_task_order_from_prio(snapshot, graph, top_p, anchor_op_prio), True,
                                ))

                    if dl.can_run_before(dl.res_soft_s if anchor_eval.valid else dl.abs_propose_deadline_s, 0.0018):
                        nn_seed = [("anchor", anchor_assign, anchor_bays, anchor_crews, anchor_op_prio, anchor_seq, False)]
                        for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                            snapshot, network, graph, ev, fast_surr, anchor_eval, anchor_op_prio, nn_seed,
                            obs_ops, pending_or_dist_set, budget_ms_limit=1.0,
                        )[:2]:
                            res_pool.append((f"res_{tg_c}", asn_c, bs_c, cs_c, op_c, sq_c, is_fc))

                evaluated_res = []
                for tg_r, asn_r, bs_r, cs_r, op_r, sq_r, _ in res_pool:
                    if res_evals >= int(self.cfg.max_residual_evals):
                        break
                    sq_k = tuple((k, tuple(v)) for k, v in sorted(sq_r.items())) if sq_r else None
                    sg = (tuple(sorted(asn_r.items())), tuple(sorted(bs_r.items())), tuple(sorted(cs_r.items())), tuple(sorted(op_r.items())), sq_k)
                    if sg in seen_sigs:
                        continue
                    fp = compute_candidate_amr_footprint(
                        snapshot, best_assign, best_bays, best_eval.robot_order_seq, asn_r, bs_r, sq_r
                    )
                    est_ev_s = max_inc_eval_s if fp["is_b2_incremental_hit"] else max(max_full_eval_s, 1.15 * keep_eval_s)
                    res_eval_cutoff = dl.abs_propose_deadline_s if (not best_eval.valid) else dl.res_hard_s
                    if not dl.can_run_before(res_eval_cutoff, est_ev_s + 0.0014):
                        continue
                    seen_sigs.add(sg)
                    t_ev0 = _cvcr_now_s()
                    ev_r = ev.evaluate(asn_r, op_priority=op_r, op_bay_overrides=bs_r, op_crew_overrides=cs_r, task_order_override=sq_r)
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

                if evaluated_res:
                    def _churn_margin(item_r: Any) -> float:
                        if not anchor_eval.valid:
                            return 0.0
                        dt_a = sum(1 for tid in graph.rel_tids if item_r[1].get(tid) != anchor_assign.get(tid))
                        do_a = sum(
                            1 for oid in graph.op_ids
                            if item_r[2].get(oid) != anchor_bays.get(oid) or item_r[3].get(oid) != anchor_crews.get(oid)
                        )
                        if dt_a == 0 and do_a == 0:
                            # O2 queue-only refinement incurs ZERO assignment/bay churn -> lower bar!
                            return 3.0
                        return float(self.cfg.res_ja_margin) + 10.0 * float(dt_a) + 14.0 * float(do_a)

                    cm = self.cfg.critic_mode.lower()
                    use_mh = cm.startswith("mh_") and (self._mh_dfl is not None)
                    use_1h = (not use_mh) and cm.startswith("dfl") and (self._dfl is not None)
                    if (not anchor_eval.valid) or cm == "none" or ((not use_mh) and (not use_1h)):
                        best_r, best_net_g = None, 0.0
                        for ir in evaluated_res:
                            if not anchor_eval.valid:
                                if best_r is None or float(ir[7]) < float(best_r[7]):
                                    best_r = ir
                            else:
                                ng = (anchor_score - ir[7]) - _churn_margin(ir)
                                if ng > best_net_g:
                                    best_net_g, best_r = ng, ir
                        if best_r is not None:
                            chosen_arm, best_assign, best_bays, best_crews, best_op_prio, best_seq, best_eval, best_score = best_r
                            fallback_keep = False
                            res_acc = True
                    else:
                        keep_stats = build_keep_baseline_stats(snapshot, graph, keep_eval, keep_score_val, rob_cur_healthy, ev.sim_horizon)
                        fv_anc = extract_critic_features(
                            snapshot, network, graph, anchor_eval, anchor_score,
                            anchor_assign, anchor_bays, anchor_crews, anchor_seq,
                            keep_stats, rob_cur_healthy, ev.sim_horizon, is_keep_cand=anchor_was_keep,
                        )
                        X_b = np.stack(
                            [
                                build_anchor_relative_feature(
                                    extract_critic_features(
                                        snapshot, network, graph, ir[6], ir[7], ir[1], ir[2], ir[3], ir[5],
                                        keep_stats, rob_cur_healthy, ev.sim_horizon, is_keep_cand=False,
                                    ),
                                    fv_anc,
                                    h_vec_ro,
                                )
                                for ir in evaluated_res
                            ],
                            axis=0,
                        )
                        use_lcb = ("nolcb" not in cm)
                        use_crc = cm.endswith("_crc")
                        kappa_eff = float(self.cfg.lcb_kappa) if use_lcb else 0.0

                        if use_mh and self._mh_dfl is not None:
                            scores = self._mh_dfl.score_batch(X_b, kappa=kappa_eff, use_full_risk_term=use_lcb)
                        else:
                            assert self._dfl is not None
                            scores = self._dfl.score_batch(X_b, kappa=kappa_eff)

                        best_u = -1e18
                        winner = None
                        for ir, sc_obj in zip(evaluated_res, scores):
                            ja_g = anchor_score - ir[7]
                            m_req = _churn_margin(ir)
                            if ja_g < m_req:
                                continue
                            is_zero_churn_q = (ir[0] == "o2_queue_opt")
                            adv_chk = sc_obj.lcb_advantage if use_lcb else getattr(sc_obj, "pred_advantage", getattr(sc_obj, "pred_composite_advantage", 0.0))
                            req_adv = -15.0 if is_zero_churn_q else float(self.cfg.dfl_tau_gain)
                            if adv_chk < req_adv:
                                continue
                            if use_crc and sc_obj.harm_prob > float(self.cfg.crc_harm_tau):
                                continue
                            u = float(max(0.0, adv_chk) + 0.35 * (ja_g - m_req))
                            if u > best_u:
                                best_u, winner = u, ir
                        if winner is not None:
                            chosen_arm, best_assign, best_bays, best_crews, best_op_prio, best_seq, best_eval, best_score = winner
                            fallback_keep = False
                            res_acc = True

                    if res_acc:
                        self.residual_candidates_accepted += 1
        else:
            anchor_eval = keep_eval
            anchor_score = keep_score_val
            anchor_assign = dict(graph.base_assign)
            anchor_bays = dict(graph.base_bays)
            anchor_crews = dict(graph.base_crews)
            anchor_op_prio = dict(base_op_prio)
            anchor_seq = None
            anchor_was_keep = True
            anchor_arm = "keep"
            dl.used_fallback = True

        # ---- ZERO-EXPIRY TRANSACTION HARD VETO ----
        act_dir = dl.finalize_veto_check(
            is_keep_cand=fallback_keep,
            has_valid_anchor=bool(anchor_eval.valid),
            anchor_is_keep=anchor_was_keep,
            has_residual_selected=res_acc,
        )
        if act_dir == "revert_anchor":
            chosen_arm = f"veto_to_{anchor_arm}"
            best_assign, best_bays, best_crews, best_op_prio, best_seq, best_eval, best_score = (
                anchor_assign, anchor_bays, anchor_crews, anchor_op_prio, anchor_seq, anchor_eval, anchor_score
            )
            fallback_keep = anchor_was_keep
            res_acc = False
        elif act_dir == "revert_keep":
            chosen_arm = "veto_to_keep"
            best_assign = dict(graph.base_assign)
            best_bays = dict(graph.base_bays)
            best_crews = dict(graph.base_crews)
            best_op_prio = dict(base_op_prio)
            best_seq = None
            best_eval = keep_eval
            best_score = keep_score_val
            fallback_keep = True
            res_acc = False

        if dl.vetoed:
            self.veto_count += 1
        if dl.used_fallback:
            self.fallback_count += 1
        if dl.internal_overrun:
            self.internal_overrun_count += 1

        prop = self._emit_proposal(
            snapshot=snapshot, clf=clf, graph=graph,
            best_assign=best_assign, best_bays=best_bays, best_crews=best_crews,
            best_op_prio=best_op_prio, best_seq=best_seq, best_eval=best_eval,
            best_score=best_score, fallback_keep=fallback_keep, eval_cnt=ev.calls,
            tw=tw, cut_step=cut_step, dl=dl, budget_ms=budget_ms,
            is_ws=is_ws, has_res_win=has_res_win, is_open=is_open,
        )
        self.last_call_stats = {
            "mode": "nominal",
            "graph_extract_calls": 1,
            "encode_full_calls": int(enc_calls.get("full", 0)),
            "encode_staged_calls": int(enc_calls.get("staged", 0)),
            "anchor_evals": int(anchor_evals),
            "residual_evals": int(res_evals),
            "residual_accepted": bool(res_acc),
            "chosen_arm": chosen_arm,
            "eval_cnt": int(ev.calls),
            "vetoed": bool(dl.vetoed),
            "veto_reason": dl.veto_reason,
            "used_fallback": bool(dl.used_fallback),
            "internal_overrun": bool(dl.internal_overrun),
            "abs_propose_cap_ms": dl.abs_propose_cap_ms,
            "window_slack_steps": dl.window_slack_steps,
            "os_jitter_ms": dl.os_jitter_est_ms,
        }
        prop.telemetry.update(self.last_call_stats)
        return prop

    def _propose_pdg2_escape(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        *,
        budget_ms: float,
        t0_s: float,
        injected_jitter_ms: float,
    ) -> CandidateProposal:
        self.pdg_recoveries_triggered += 1
        pdg = self._active_pdg or {}
        tick = int(snapshot.tick)
        self.history_buf.update_from_snapshot(snapshot)
        clf = classify_commitments(snapshot)
        dl = self.dl_mgr.begin_decision(
            snapshot, clf, budget_ms=budget_ms, t0_s=t0_s, injected_jitter_ms=injected_jitter_ms
        )

        t_g0 = _cvcr_now_s()
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        g_ms = (_cvcr_now_s() - t_g0) * 1000.0
        dl.record_phase("preamble_graph", g_ms)
        dl.detect_os_jitter(self.dl_mgr.observe_graph_ms(g_ms))

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

        ev = CVCREvaluator(
            network, snapshot, mode="B2", alpha_makespan=3.0, lambda_change=15.0,
            use_rust=self.use_rust, seq_diff_b2=True,
        )
        fb = getattr(snapshot, "execution_feedback", None)
        tw: dict[tuple[str, str], int] = {}
        cut_step = 0
        if fb is not None and not fb.is_empty() and fb.blocked_intervals(t_commit=tick, default_horizon=12):
            tw = {(str(u), str(v)): int(w) for (u, v), w in fb.corridor_q3_penalties().items() if network.has_edge(u, v)}
            if tw:
                cut_step = tick + 45
                ev.set_traffic_weights(tw, cutoff_step=cut_step)

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
        if self._sim_ref is not None and not getattr(self._sim_ref, "executions", None) and len(network.nodes) <= 100 and dl.can_run_before(dl.anchor_soft_s, 0.003):
            haven_r, _ = plan_haven_safe_corridor_repair(self._sim_ref, snapshot, stall_threshold=3, enable_choke_oscillation_guard=True)
            e1_live_repairs.update(haven_r)
            idle_r = plan_idle_blocker_evacuations(
                snapshot, network, self._sim_ref, [a for a in idle_blockers if a not in e1_live_repairs], set(clf.disturbed_robots)
            )
            e1_live_repairs.update(idle_r)
        e1_locks = (
            build_critical_bottleneck_locks(snapshot, network, sorted(distressed_agvs), set(clf.disturbed_robots))
            if (len(network.nodes) <= 100 and dl.can_run_before(dl.anchor_soft_s, 0.002))
            else {}
        )

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
            relieved_cycle = sum(1 for tid in graph.rel_tids if graph.base_assign.get(tid) in cycle_agvs and art.assign.get(tid) != graph.base_assign.get(tid))
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
            if sig in seen_sigs or not dl.can_run_before(dl.anchor_soft_s, 0.0012):
                return False
            fp = compute_candidate_amr_footprint(
                snapshot, best.assign, best.bays, best.eval_res_b2.robot_order_seq if best.eval_res_b2 else None,
                art.assign, art.bays, art.task_orders,
            )
            est_s = max_inc_s if fp["is_b2_incremental_hit"] else max(keep_eval_s, 1.15 * keep_eval_s)
            if not dl.can_run_before(dl.anchor_hard_s, est_s + 0.0012):
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

        if (e1_live_repairs or e1_locks or risk_tasks) and dl.can_run_before(dl.anchor_hard_s, keep_eval_s):
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

        if healthy and graph.rel_tids and dl.can_run_before(dl.anchor_soft_s, keep_eval_s):
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

        if best is not keep_art:
            self._escape_tabu[best.ensure_sig()] = tick + 18

        fallback_keep = (best is keep_art) and not (best.live_repairs or best.bottleneck_locks)
        act_dir = dl.finalize_veto_check(
            is_keep_cand=fallback_keep,
            has_valid_anchor=bool(best is not keep_art and best.eval_res_b2 and best.eval_res_b2.valid),
            anchor_is_keep=(best is keep_art),
            has_residual_selected=False,
        )
        if act_dir == "revert_keep":
            best = keep_art
            fallback_keep = True

        if dl.vetoed:
            self.veto_count += 1
        if dl.used_fallback:
            self.fallback_count += 1
        if dl.internal_overrun:
            self.internal_overrun_count += 1

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
            t0=dl.t0_s,
            budget_ms=budget_ms,
            fallback_keep=fallback_keep,
            task_sequences=best.task_orders,
            predicted_score=best.predicted_score,
            include_eval_seq_when_unwindowed=not has_win,
        )
        if tw:
            prop.proposed_traffic_weights = tw
            prop.proposed_traffic_cutoff = cut_step
        if best.live_repairs and not dl.vetoed:
            prop.proposed_live_repairs = dict(best.live_repairs)
        if best.bottleneck_locks and not dl.vetoed:
            prop.proposed_bottleneck_locks = dict(best.bottleneck_locks)

        self.last_call_stats = {
            "mode": f"pdg2_{lvl_key}",
            "graph_extract_calls": 1,
            "encode_full_calls": 0,
            "chosen_arm": "veto_to_keep" if dl.vetoed else best.tag,
            "eval_cnt": int(ev.calls),
            "vetoed": bool(dl.vetoed),
            "veto_reason": dl.veto_reason,
            "used_fallback": bool(dl.used_fallback),
            "internal_overrun": bool(dl.internal_overrun),
            "abs_propose_cap_ms": dl.abs_propose_cap_ms,
            "window_slack_steps": dl.window_slack_steps,
            "os_jitter_ms": dl.os_jitter_est_ms,
        }
        prop.telemetry.update(self.last_call_stats)
        return prop


def run_closure_episode(
    spec: dict[str, Any],
    *,
    proposer_factory: Callable[[], Any],
    base_env_policy: str = "V19_ADAPTIVE_RL",
    use_pdg2: bool = True,
    use_virtual_clock: bool = False,
    jitter_fn: Optional[Callable[[int, int], float]] = None,
) -> dict[str, Any]:
    """Unified clean runner supporting RGROClosureProposer, V201PatchedProposer, and frozen comparators."""
    from amr_sched.baselines.policy_registry import create_discrete_policy_env
    from amr_sched.cvcr.auditor import IndependentAuditor
    from amr_sched.cvcr.clock import virtual_clock_scope
    from amr_sched.cvcr.snapshot import NativeSnapshotCapture
    from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state

    def _exec() -> dict[str, Any]:
        _, ctrl, env = create_discrete_policy_env(
            base_env_policy,
            seed=int(spec["seed"]),
            max_steps=int(spec["max_steps"]),
            use_rust=True,
            p2_cfg=spec.get("p2_cfg"),
        )
        prop = proposer_factory()
        mon = PDG2SCCLivenessMonitor(cfg=PDG2Config()) if use_pdg2 else None

        def _wrapped(snap: NativeSnapshot, net: Network, *, budget_ms: float = 35.0) -> CandidateProposal:
            extra_ms = float(jitter_fn(int(snap.tick), int(spec["seed"]))) if jitter_fn is not None else 0.0
            if extra_ms > 0.0:
                t_end = time.perf_counter() + extra_ms / 1000.0
                while time.perf_counter() < t_end:
                    pass
            try:
                return prop(snap, net, budget_ms=budget_ms, injected_jitter_ms=extra_ms)
            except TypeError:
                eff_b = max(4.0, float(budget_ms) - extra_ms) if extra_ms > 0.0 else float(budget_ms)
                return prop(snap, net, budget_ms=eff_b)

        env.external_proposer = _wrapped
        obs, info = env.reset(seed=int(spec["seed"]), options={"scenario_tuple": copy.deepcopy(spec["scenario_tuple"])})
        if hasattr(prop, "reset_episode_state"):
            prop.reset_episode_state()
        if hasattr(prop, "set_sim_ref"):
            prop.set_sim_ref(env.sim)
        else:
            prop._sim_ref = env.sim

        tm = tr = False
        rej_streak = 0
        while not (tm or tr):
            t_now = int(env.current_step)
            if hasattr(prop, "observe_committed_step") and (t_now % 5 == 0 or t_now == 0):
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
                dl_row = env.decision_logs[-1]
                rej_streak = 0 if bool(dl_row.get("receipt_accepted", True)) else (rej_streak + 1)
                if hasattr(prop, "dl_mgr"):
                    prop.dl_mgr.observe_decision_log(dl_row)

        m = env.compute_episode_metrics()
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        dec_ms = [float(r["t_propose_ms"]) + float(r["t_validate_commit_ms"]) for r in env.decision_logs]
        val_ms = [float(r["t_validate_commit_ms"]) for r in env.decision_logs]
        recs = list(getattr(prop, "call_latency_records", []))
        int_ms = [float(x.get("internal_algo_ms", 0.0)) for x in recs]
        jit_ms = [float(x.get("os_jitter_ms", 0.0)) for x in recs]
        fb_ms = [
            float(d["t_propose_ms"]) + float(d["t_validate_commit_ms"])
            for d, r in zip(env.decision_logs, recs)
            if r.get("used_fallback")
        ]
        return {
            "policy": getattr(prop, "policy_id", base_env_policy),
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
            "veto_count": int(getattr(prop, "veto_count", 0)),
            "fallback_count": int(getattr(prop, "fallback_count", 0)),
            "internal_overrun_count": int(getattr(prop, "internal_overrun_count", 0)),
            "decision_latencies_ms": [round(float(x), 4) for x in dec_ms],
            "validate_commit_ms": [round(float(x), 4) for x in val_ms],
            "internal_algo_ms": [round(float(x), 4) for x in int_ms],
            "os_jitter_ms": [round(float(x), 4) for x in jit_ms],
            "fallback_response_ms": [round(float(x), 4) for x in fb_ms],
        }

    if use_virtual_clock:
        with virtual_clock_scope(0.0):
            return _exec()
    return _exec()
