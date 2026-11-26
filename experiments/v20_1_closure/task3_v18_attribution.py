"""Task 3 (F3): V18–V20 Headroom Attribution & State-Risk Critic Applicability Study.

Decomposes the gap between frozen `V20_NAJS` (`V20BProposer`) and `V18_EA_JOINT` on identical
snapshots under identical evaluation (`CVCREvaluator B2_SEEDED`, `JA_Unified`, and `+48t` closed-loop
rollout via `rollout_48_and_optional_full`) into four exact additive/causal components:
  1. Candidate Generation Gap:       G_gen    = J*(C_V20_untrunc) - J*(C_union)
  2. Budget Truncation Regret:       G_budget = J*(C_V20)         - J*(C_V20_untrunc)
  3. Ranking / Selection Regret:     G_rank   = J(c_selected)     - J*(C_V20)
     (exact algebraic identity: J(c_selected) - J*(C_union) == G_gen + G_budget + G_rank)
  4. Long-Horizon Evaluation Error:  G_eval   = J_48t(c_selected) - min_{c in key_frontier} J_48t(c)

Also tests continuous observable state-risk Critic applicability gating vs hardcoded `ws_only`
strictly on quarantined `VAL-72`.
Writes Deliverable 3 (`out/v20_1_closure/deliverable3_v18_v20_attribution.{json,md}`).
"""

from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.snapshot import NativeSnapshot, NativeSnapshotCapture
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.causal_repair_cone import FastJAWKittingSurrogate, observable_op_closure
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    CandidateArtifact,
    RecordedStep,
    assert_stage1_quarantine,
    build_val72_specs,
    make_v18_external_proposer,
    rollout_48_and_optional_full,
    run_wrapped_episode,
    summarize_episode_rows,
)
from experiments.v20_audit.proposer_variants import (
    ExpertSuiteEvaluator,
    WindowAwareKittingSurrogate,
    _LazyKittingSurrogate,
    derive_base_op_prio,
)
from experiments.v20_audit.v20_b_bidirectional import V20BProposer
from experiments.v20_audit.v20_e_liveness import build_default_v20_proposer

OUT_CLOSURE_DIR = ROOT / "out" / "v20_1_closure"


def _make_v20_backbone(*, soft_ms: float = 15.0, hard_ms: float = 35.0, critic_beta: float = 0.25, critic_mode: str = "ws_only") -> V20BProposer:
    return V20BProposer(
        hard_budget_ms=hard_ms,
        soft_deadline_ms=soft_ms,
        seed=0,
        enable_e4=True,
        e4_selector="fast_surr",
        e4_include_zc=False,
        e4_mode="ws_or_invalid",
        staged_mode="ws_or_invalid",
        enable_bidir_v18_to_v19=True,
        enable_e6_static_or_headroom=False,
        enable_e7=True,
        enable_e9=False,
        enable_early_stop=False,
        compact_calm_ms=None,
        enable_r2_global_ws=False,
        r2_zc_ws_mode="never",
        include_staged_g="on_invalid",
        unwin_seq_horizon=2,
        seq_diff_b2=True,
        critic_weights_path=OUT_AUDIT_DIR / "v20_a_critic_weights.npz",
        critic_beta=critic_beta,
        critic_mode=critic_mode,
    )


def _capture_v20_pool(prop: V20BProposer, snap: NativeSnapshot, net: Any, *, budget_ms: float) -> tuple[list[CandidateArtifact], Any]:
    """Run `prop` with a non-invasive spy on `_finish_artifact` to capture the exact evaluated candidate pool."""
    captured: list[CandidateArtifact] = []
    orig_fin = prop._suite._finish_artifact

    def _spy(art: CandidateArtifact, *args: Any, **kwargs: Any):
        res = orig_fin(art, *args, **kwargs)
        captured.append(copy.deepcopy(art))
        return res

    prop._suite._finish_artifact = _spy
    try:
        out_prop = prop(snap, net, budget_ms=budget_ms)
    finally:
        prop._suite._finish_artifact = orig_fin
    return captured, out_prop


def _evaluate_candidate_uniform(
    art: CandidateArtifact,
    *,
    suite: ExpertSuiteEvaluator,
    snap: NativeSnapshot,
    graph: Any,
    ev_b2: CVCREvaluator,
    w_surr: Any,
    rob_cur_h: dict[str, str],
    fresh_ops: set[str],
    fast_surr: Any,
) -> float:
    a = copy.deepcopy(art)
    suite._finish_artifact(
        a, snapshot=snap, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
        rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
    )
    if a.eval_res_b2 is None or not a.eval_res_b2.valid or a.ja_unified_score is None:
        return 1e9
    return float(a.ja_unified_score)


def _select_stratified_val_episodes() -> list[dict[str, Any]]:
    """Select 24 stratified quarantined VAL-72 episodes covering all 4-Topo layouts, Track 1, and Track 2."""
    s72 = build_val72_specs()
    picked = []
    # 12 4-Topo (all 4 layouts x 3 regimes on seeds 16701/16702)
    for sp in s72:
        if sp["suite_group"] == "4topo" and int(sp["seed"]) == 16702:
            picked.append(sp)
    # 6 Track 1 (all 4 regimes on 1161 + 2 on 1162)
    t1 = [s for s in s72 if s["suite_group"] == "track1"]
    picked.extend(t1[:6])
    # 6 Track 2 (all 4 regimes on 1161 + 2 on 1162)
    t2 = [s for s in s72 if s["suite_group"] == "track2"]
    picked.extend(t2[:6])
    assert_stage1_quarantine([s["seed"] for s in picked], label="Task3_Stratified_VAL")
    return picked


def _collect_episode_snapshots(sp: dict[str, Any], max_snaps: int = 2) -> list[dict[str, Any]]:
    """Run frozen V20_NAJS recording prefix steps & capture up to `max_snaps` non-trivial replan snapshots."""
    sc_copy = copy.deepcopy(sp["scenario_tuple"])
    p2_copy = copy.deepcopy(sp["p2_cfg"])
    seed = int(sp["seed"])
    max_steps = int(sp["max_steps"])
    _, ctrl, env = create_discrete_policy_env(
        "V19_ADAPTIVE_RL", topology=str(sc_copy[0]), seed=seed, max_steps=max_steps, use_rust=True, p2_cfg=p2_copy
    )
    prop = build_default_v20_proposer(hard_budget_ms=35.0, seed=0, enable_pdg_recovery=False)
    recorded: list[RecordedStep] = []
    snaps_meta: list[dict[str, Any]] = []

    def _cap(snap: NativeSnapshot, net: Any, *, budget_ms: float = 35.0):
        p = prop(snap, net, budget_ms=budget_ms)
        recorded.append(RecordedStep(step_before=int(snap.tick), action=-1, triggered_replan=True, proposal=copy.deepcopy(p)))
        if len(snaps_meta) < max_snaps and int(snap.tick) >= 8:
            clf = classify_commitments(snap)
            g = extract_v19_hetero_graph(snap, net, clf=clf)
            if len(g.rel_tids) >= 3:
                snaps_meta.append(
                    {
                        "tick": int(snap.tick),
                        "snap": copy.deepcopy(snap),
                        "recorded_prefix": copy.deepcopy(recorded),
                    }
                )
        return p

    env.external_proposer = _cap
    obs, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
    tm = tr = False
    while not (tm or tr):
        sb = int(env.current_step)
        feats = compute_expert_v9_state(
            env.sim, env.tracker,
            last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks,
            decision_logs=env.decision_logs,
        )
        mask = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
        act = int(ctrl.select_action(feats, dict(info, action_mask=mask), sim=env.sim))
        bn = len(recorded)
        bl = len(env.decision_logs)
        obs, _, tm, tr, info = env.step(act)
        if len(recorded) > bn:
            recorded[-1].action = act
            if len(env.decision_logs) > bl:
                recorded[-1].propose_ms = float(env.decision_logs[-1].get("t_propose_ms", 0.0))
            for sm in snaps_meta:
                if sm["tick"] == sb and "action" not in sm:
                    sm["action"] = act
                    sm["recorded_prefix"][-1].action = act
                    sm["recorded_prefix"][-1].propose_ms = recorded[-1].propose_ms
        else:
            recorded.append(RecordedStep(step_before=sb, action=act, triggered_replan=False, proposal=None))
        if len(snaps_meta) >= max_snaps and all("action" in sm for sm in snaps_meta):
            break
    return [sm for sm in snaps_meta if "action" in sm]


class StateRiskCriticProposer(V20BProposer):
    """Replaces hardcoded `ws_only` network-size check with continuous observable state-risk gating."""

    def __init__(self, *, risk_gate_mode: str = "continuous_coupling", **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.risk_gate_mode = str(risk_gate_mode)
        self.critic_active_calls: int = 0
        self.critic_total_calls: int = 0

    def __call__(self, snapshot: NativeSnapshot, network: Any, *, budget_ms: float = 35.0):
        # Compute continuous observable structural coupling risk without reading topology/track names:
        #   - kit_coupling: fraction of active ops requiring >= 2 kitting deliveries
        #   - window_pressure: fraction of uncompleted ops carrying explicit resource windows
        #   - dag_depth_proxy: fraction of uncompleted ops having precedence links
        unfin_ops = [op for op in snapshot.operations if op.end_step is None]
        n_u = max(1, len(unfin_ops))
        n_multi_kit = sum(1 for op in unfin_ops if len(op.kit_task_ids) >= 2)
        n_win = sum(1 for op in unfin_ops if getattr(op, "resource_windows", ()))
        n_prec = sum(1 for op in unfin_ops if getattr(op, "predecessor_op_id", None) or getattr(op, "predecessor_op_ids", ()))
        coupling_score = (n_multi_kit / n_u) + 0.6 * (n_win / n_u) + 0.5 * (n_prec / n_u)
        self.critic_total_calls += 1
        orig_beta = self.critic_beta
        orig_mode = self.critic_mode
        try:
            if self.risk_gate_mode == "continuous_coupling":
                # Smoothly gate critic proportional to structural coupling risk (active on high-coupling states only)
                if coupling_score >= 0.75 and len(network.nodes) >= 35:
                    self.critic_mode = "always"
                    self.critic_beta = orig_beta * min(1.0, coupling_score / 1.1)
                    self.critic_active_calls += 1
                else:
                    self.critic_beta = 0.0
            elif self.risk_gate_mode == "always_scaled":
                self.critic_mode = "always"
                self.critic_beta = orig_beta * min(1.0, max(0.0, (coupling_score - 0.35) / 0.8))
                if self.critic_beta > 1e-4:
                    self.critic_active_calls += 1
            return super().__call__(snapshot, network, budget_ms=budget_ms)
        finally:
            self.critic_beta = orig_beta
            self.critic_mode = orig_mode


def run_task3_attribution() -> dict[str, Any]:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    ep_specs = _select_stratified_val_episodes()
    v20_bud = _make_v20_backbone(soft_ms=15.0, hard_ms=35.0)
    v20_untr = _make_v20_backbone(soft_ms=400.0, hard_ms=500.0)
    v18_pol = make_v18_external_proposer(hard_budget_ms=35.0, seq_diff_b2=True)
    from amr_sched.baselines.policy_registry import build_external_proposer as _build_reg_ext
    v19_full_pol = _build_reg_ext("V19_HETERO_RL")
    suite_eval = ExpertSuiteEvaluator(seed=0)

    snap_rows: list[dict[str, Any]] = []
    missing_winner_families: Counter[str] = Counter()

    print(f"=== [Task 3.1] Collecting & attributing common snapshots across {len(ep_specs)} stratified VAL episodes ===", flush=True)
    for sp in ep_specs:
        snaps = _collect_episode_snapshots(sp, max_snaps=2)
        net = copy.deepcopy(sp["scenario_tuple"][0])
        # Recover network object cleanly from env
        _, _, tmp_env = create_discrete_policy_env(
            "V19_ADAPTIVE_RL", topology=str(net), seed=int(sp["seed"]), max_steps=int(sp["max_steps"]), use_rust=True
        )
        tmp_env.reset(seed=int(sp["seed"]), options={"scenario_tuple": copy.deepcopy(sp["scenario_tuple"])})
        net_obj = tmp_env.sim.network

        for sm in snaps:
            snap: NativeSnapshot = sm["snap"]
            clf = classify_commitments(snap)
            graph = extract_v19_hetero_graph(snap, net_obj, clf=clf)
            base_prio = derive_base_op_prio(snap)
            ev_b2 = CVCREvaluator(net_obj, snap, mode="B2_SEEDED", alpha_makespan=3.0, lambda_change=15.0, use_rust=True, seq_diff_b2=True)
            dd_sig = compute_detour_delta_signals(net_obj, snap)
            bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
            lazy_surr = _LazyKittingSurrogate(net_obj, snap, base_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta)
            w_surr = WindowAwareKittingSurrogate(lazy_surr, ev_b2, snap)
            fast_surr = FastJAWKittingSurrogate(net_obj, snap, ev_b2, graph, lambda_change=15.0)
            fresh_ops = observable_op_closure(snap)
            rob_cur_h = {r.robot_id: r.cur_task for r in snap.robots if r.cur_task and r.robot_id not in clf.disturbed_robots}

            # 1. V20 budgeted pool & actual selection
            c_v20, prop_sel = _capture_v20_pool(v20_bud, snap, net_obj, budget_ms=35.0)
            # 2. V20 untruncated pool
            c_v20_u, _ = _capture_v20_pool(v20_untr, snap, net_obj, budget_ms=500.0)
            # 3. V18 candidate families (E1, E4, E5, E6 + live V18_EA_JOINT winner)
            all_exp = suite_eval.evaluate_all_experts(snap, net_obj)
            c_v18: list[CandidateArtifact] = [
                all_exp[eid] for eid in ("E1_V18_Regret2", "E4_V18_PGMPO", "E5_V18_MIT_DyMAB_LNS", "E6_V18_Polish") if eid in all_exp
            ]
            p_v18_live = v18_pol(snap, net_obj, budget_ms=35.0)
            c_v18.append(
                CandidateArtifact(
                    "V18_EA_LiveWinner", "v18_live",
                    {**graph.base_assign, **dict(p_v18_live.proposed_assignment)},
                    {**graph.base_bays, **dict(p_v18_live.proposed_bays)},
                    {**graph.base_crews, **dict(p_v18_live.proposed_crews)},
                    {**base_prio, **dict(p_v18_live.proposed_op_priorities or {})},
                    dict(p_v18_live.proposed_task_sequences) if p_v18_live.proposed_task_sequences else None,
                    False,
                )
            )
            # 4. V19_HETERO_RL full multi-sample neural joint constructor anchor (Track 2 global coupling check)
            p_v19_full = v19_full_pol(snap, net_obj, budget_ms=35.0)
            c_v19_full = [
                CandidateArtifact(
                    "V19_HETERO_RL_Full", "v19_hetero_live",
                    {**graph.base_assign, **dict(p_v19_full.proposed_assignment)},
                    {**graph.base_bays, **dict(p_v19_full.proposed_bays)},
                    {**graph.base_crews, **dict(p_v19_full.proposed_crews)},
                    {**base_prio, **dict(p_v19_full.proposed_op_priorities or {})},
                    dict(p_v19_full.proposed_task_sequences) if p_v19_full.proposed_task_sequences else None,
                    True,
                )
            ]
            art_sel = CandidateArtifact(
                "V20_Selected", "v20_selected",
                {**graph.base_assign, **dict(prop_sel.proposed_assignment)},
                {**graph.base_bays, **dict(prop_sel.proposed_bays)},
                {**graph.base_crews, **dict(prop_sel.proposed_crews)},
                {**base_prio, **dict(prop_sel.proposed_op_priorities or {})},
                dict(prop_sel.proposed_task_sequences) if prop_sel.proposed_task_sequences else None,
                False,
            )

            # Evaluate every candidate under the EXACT SAME evaluator & scorer
            eval_cache: dict[str, tuple[float, CandidateArtifact]] = {}

            def _score_pool(pool: list[CandidateArtifact]) -> tuple[float, CandidateArtifact | None]:
                best_sc = 1e9
                best_a = None
                for a in pool:
                    sig = a.ensure_sig()
                    if sig not in eval_cache:
                        sc = _evaluate_candidate_uniform(
                            a, suite=suite_eval, snap=snap, graph=graph, ev_b2=ev_b2,
                            w_surr=w_surr, rob_cur_h=rob_cur_h, fresh_ops=fresh_ops, fast_surr=fast_surr,
                        )
                        eval_cache[sig] = (sc, a)
                    sc_v, a_v = eval_cache[sig]
                    if sc_v < best_sc:
                        best_sc, best_a = sc_v, a_v
                return best_sc, best_a

            j_sel, _ = _score_pool([art_sel])
            j_v20, a_v20_best = _score_pool(c_v20 + [art_sel])
            j_v20_u, a_v20u_best = _score_pool(c_v20 + [art_sel] + c_v20_u)
            j_v18, a_v18_best = _score_pool(c_v18)
            j_v19f, a_v19f_best = _score_pool(c_v19_full)
            j_union_v18, _ = _score_pool(c_v20 + [art_sel] + c_v20_u + c_v18)
            j_union_v19, _ = _score_pool(c_v20 + [art_sel] + c_v20_u + c_v19_full)
            j_union, a_union_best = _score_pool(c_v20 + [art_sel] + c_v20_u + c_v18 + c_v19_full)

            if j_sel >= 1e8 or j_union >= 1e8:
                continue

            g_gen = float(j_v20_u - j_union)
            g_gen_v18 = float(j_v20_u - j_union_v18)
            g_gen_v19 = float(j_v20_u - j_union_v19)
            g_budget = float(j_v20 - j_v20_u)
            g_rank = float(j_sel - j_v20)
            total_gap = float(j_sel - j_union)
            assert abs((g_gen + g_budget + g_rank) - total_gap) < 1e-5, "Accounting identity mismatch!"

            if g_gen > 1.0 and a_union_best is not None:
                missing_winner_families[a_union_best.expert_id] += 1

            # Verify closed-loop +48t rollout error G_eval on key candidates (c_sel vs best_v20 vs best_v18 vs best_v19_full)
            key_cands: list[tuple[str, CandidateArtifact]] = [("sel", art_sel)]
            if a_v20_best is not None and a_v20_best.ensure_sig() != art_sel.ensure_sig():
                key_cands.append(("v20_best", a_v20_best))
            if a_v18_best is not None and a_v18_best.ensure_sig() not in {x[1].ensure_sig() for x in key_cands}:
                key_cands.append(("v18_best", a_v18_best))
            if a_v19f_best is not None and a_v19f_best.ensure_sig() not in {x[1].ensure_sig() for x in key_cands}:
                key_cands.append(("v19_full_best", a_v19f_best))

            j48_map: dict[str, float] = {}
            for lbl_k, art_k in key_cands:
                # Re-build valid proposal via uniform eval
                a_tmp = copy.deepcopy(art_k)
                suite_eval._finish_artifact(
                    a_tmp, snapshot=snap, graph=graph, ev_b2=ev_b2, w_surr=w_surr,
                    rob_cur_h=rob_cur_h, obs_ops=fresh_ops, fast_surr=fast_surr, compute_details=False,
                )
                pr_k = a_tmp.to_proposal(
                    policy_id=f"T3_{lbl_k}", snapshot=snap, clf=clf, eval_res=a_tmp.eval_res_b2,
                    eval_cnt=1, t0=0.0, budget_ms=35.0, fallback_keep=False,
                )
                ro = rollout_48_and_optional_full(
                    sp, sm["recorded_prefix"], sm["tick"], pr_k,
                    action_at_target=int(sm["action"]), rollout_horizon=48, full_continuation=False,
                )
                j48_map[lbl_k] = float(ro["j48_with_residual"])

            g_eval_48t = float(j48_map["sel"] - min(j48_map.values()))
            v18_beats_sel_48t = bool("v18_best" in j48_map and j48_map["v18_best"] + 1e-4 < j48_map["sel"])
            v19_beats_sel_48t = bool("v19_full_best" in j48_map and j48_map["v19_full_best"] + 1e-4 < j48_map["sel"])

            snap_rows.append(
                {
                    "episode_key": sp["episode_key"],
                    "suite_group": sp["suite_group"],
                    "tick": sm["tick"],
                    "n_c_v20": len({a.ensure_sig() for a in c_v20}),
                    "n_c_v20_untrunc": len({a.ensure_sig() for a in c_v20_u}),
                    "j_sel": round(j_sel, 3),
                    "j_v20_best": round(j_v20, 3),
                    "j_v20_untrunc_best": round(j_v20_u, 3),
                    "j_v18_best": round(j_v18, 3),
                    "j_v19_full_best": round(j_v19f, 3),
                    "j_union_best": round(j_union, 3),
                    "G_gen": round(g_gen, 3),
                    "G_gen_V18": round(g_gen_v18, 3),
                    "G_gen_V19": round(g_gen_v19, 3),
                    "G_budget": round(g_budget, 3),
                    "G_rank": round(g_rank, 3),
                    "G_total": round(total_gap, 3),
                    "G_eval_48t": round(g_eval_48t, 3),
                    "v18_beats_sel_48t": v18_beats_sel_48t,
                    "v19_beats_sel_48t": v19_beats_sel_48t,
                    "union_winner_expert": a_union_best.expert_id if a_union_best else "none",
                }
            )

    # Aggregate attribution by suite
    by_suite: dict[str, Any] = {}
    for grp in ("all", "4topo", "track1", "track2"):
        sub = snap_rows if grp == "all" else [r for r in snap_rows if r["suite_group"] == grp]
        if not sub:
            continue
        by_suite[grp] = {
            "n_snaps": len(sub),
            "mean_G_gen": round(float(np.mean([r["G_gen"] for r in sub])), 2),
            "mean_G_gen_V18": round(float(np.mean([r["G_gen_V18"] for r in sub])), 2),
            "mean_G_gen_V19": round(float(np.mean([r["G_gen_V19"] for r in sub])), 2),
            "mean_G_budget": round(float(np.mean([r["G_budget"] for r in sub])), 2),
            "mean_G_rank": round(float(np.mean([r["G_rank"] for r in sub])), 2),
            "mean_G_total": round(float(np.mean([r["G_total"] for r in sub])), 2),
            "median_G_total": round(float(np.median([r["G_total"] for r in sub])), 2),
            "mean_G_eval_48t": round(float(np.mean([r["G_eval_48t"] for r in sub])), 2),
            "pct_v18_beats_sel_proxy": round(100.0 * float(np.mean([1.0 if r["G_gen_V18"] > 1.0 else 0.0 for r in sub])), 1),
            "pct_v19_beats_sel_proxy": round(100.0 * float(np.mean([1.0 if r["G_gen_V19"] > 1.0 else 0.0 for r in sub])), 1),
            "pct_v18_beats_sel_48t": round(100.0 * float(np.mean([1.0 if r["v18_beats_sel_48t"] else 0.0 for r in sub])), 1),
            "pct_v19_beats_sel_48t": round(100.0 * float(np.mean([1.0 if r["v19_beats_sel_48t"] else 0.0 for r in sub])), 1),
        }
        print(
            f"  Attribution [{grp:6s}] (n={len(sub):2d}): Total={by_suite[grp]['mean_G_total']:6.2f} = "
            f"Gen({by_suite[grp]['mean_G_gen']:6.2f} [V18={by_suite[grp]['mean_G_gen_V18']:.1f}, V19={by_suite[grp]['mean_G_gen_V19']:.1f}]) + "
            f"Bud({by_suite[grp]['mean_G_budget']:5.2f}) + Rank({by_suite[grp]['mean_G_rank']:6.2f}) | Eval48t={by_suite[grp]['mean_G_eval_48t']:6.2f}",
            flush=True,
        )

    # 2. State-Risk Critic Applicability Study on full quarantined VAL-72
    print("=== [Task 3.2] State-Risk Critic Applicability Gating on VAL-72 ===", flush=True)
    val72 = build_val72_specs()
    critic_modes_out: dict[str, Any] = {}
    for label, prop_fn in (
        ("Critic_OFF", lambda: _make_v20_backbone(critic_beta=0.0, critic_mode="ws_only")),
        ("Critic_WS_Only_025", lambda: _make_v20_backbone(critic_beta=0.25, critic_mode="ws_only")),
        (
            "Critic_StateRisk_Gate",
            lambda: StateRiskCriticProposer(
                risk_gate_mode="continuous_coupling",
                hard_budget_ms=35.0, soft_deadline_ms=15.0, seed=0,
                enable_e4=True, e4_selector="fast_surr", e4_include_zc=False,
                e4_mode="ws_or_invalid", staged_mode="ws_or_invalid",
                enable_bidir_v18_to_v19=True, enable_e6_static_or_headroom=False,
                enable_e7=True, enable_e9=False, enable_early_stop=False, compact_calm_ms=None,
                enable_r2_global_ws=False, r2_zc_ws_mode="never", include_staged_g="on_invalid",
                unwin_seq_horizon=2, seq_diff_b2=True,
                critic_weights_path=OUT_AUDIT_DIR / "v20_a_critic_weights.npz",
                critic_beta=0.25, critic_mode="always",
            ),
        ),
    ):
        rows = []
        act_c = tot_c = 0
        for sp in val72:
            p_inst = prop_fn()
            r, _ = run_wrapped_episode(variant_id=label, spec=sp, base_env_policy="V19_ADAPTIVE_RL", external_proposer=p_inst, hard_budget_ms=35.0)
            rows.append(r)
            if isinstance(p_inst, StateRiskCriticProposer):
                act_c += p_inst.critic_active_calls
                tot_c += p_inst.critic_total_calls
        ov = summarize_episode_rows(rows)
        by_s = {g: summarize_episode_rows([x for x in rows if x["suite_group"] == g])["mean_net_obj"] for g in ("4topo", "track1", "track2")}
        critic_modes_out[label] = {
            **ov,
            "by_suite_net": by_s,
            "critic_active_frac": round(act_c / max(1, tot_c), 4) if tot_c else None,
        }
        print(f"  {label:22s}: JA={ov['mean_net_obj']:7.2f} Med={ov['median_net_obj']:6.2f} Comp={ov['comp100_count']}/72 | 4T={by_s['4topo']} T1={by_s['track1']} T2={by_s['track2']}", flush=True)

    report = {
        "schema": "v20_1_closure.deliverable3_dual_anchor_attribution.v1",
        "attribution_by_suite": by_suite,
        "missing_winner_families": dict(missing_winner_families),
        "critic_applicability_val72": critic_modes_out,
        "snapshot_details": snap_rows,
    }
    (OUT_CLOSURE_DIR / "deliverable3_v18_v20_attribution.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    md = [
        "# Deliverable 3: Dual-Anchor Candidate Attribution (`V18-EA` + `V19_HETERO_RL` vs `V20`) & Critic Applicability",
        "",
        "## 1. Exact Same-Snapshot Gap Decomposition (`G_total = G_gen + G_budget + G_rank`)",
        "",
        "| Suite | Snaps | Mean `G_total` | `G_gen` (Union) | `G_gen_V18` | `G_gen_V19` | `G_budget` | `G_rank` | `G_eval` (+48t) | V18>Sel (+48t%) | V19>Sel (+48t%) |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for grp, s in by_suite.items():
        md.append(
            f"| {grp} | {s['n_snaps']} | {s['mean_G_total']:.2f} | {s['mean_G_gen']:.2f} | {s['mean_G_gen_V18']:.2f} | "
            f"{s['mean_G_gen_V19']:.2f} | {s['mean_G_budget']:.2f} | {s['mean_G_rank']:.2f} | {s['mean_G_eval_48t']:.2f} | "
            f"{s['pct_v18_beats_sel_48t']:.1f}% | {s['pct_v19_beats_sel_48t']:.1f}% |"
        )
    md.extend([
        "",
        f"- **Anchor Families Generating Missing Union Optima (`G_gen > 1.0`)**: `{dict(missing_winner_families)}`",
        "",
        "## 2. Critic Applicability Gating (`VAL-72`, 35 ms)",
        "",
        "| Variant | Comp100 | Mean JA ↓ | Median JA ↓ | 4-Topo | Track 1 | Track 2 | Pool P95 |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for lbl, s in critic_modes_out.items():
        bs = s["by_suite_net"]
        md.append(
            f"| {lbl} | {s['comp100_count']}/72 | {s['mean_net_obj']:.2f} | {s['median_net_obj']:.2f} | "
            f"{bs['4topo']:.2f} | {bs['track1']:.2f} | {bs['track2']:.2f} | {s['pooled_p95_ms']:.2f} ms |"
        )

    (OUT_CLOSURE_DIR / "deliverable3_v18_v20_attribution.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[TASK 3 DONE] Wrote Deliverable 3 -> {OUT_CLOSURE_DIR / 'deliverable3_v18_v20_attribution.md'}", flush=True)
    return report


if __name__ == "__main__":
    run_task3_attribution()
