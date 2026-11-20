"""Stage 0 regression and contract tests for V17 causal routing, parity, probe logging, and 7-column costs."""

from __future__ import annotations

import hashlib
import numpy as np
import pytest

from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.cvcr.v15_ea_joint import (
    V15Candidate,
    V15ExecutionAwarePolicy,
)
from amr_sched.cvcr.v16_frontier import (
    FEATURE_SCHEMA_VERSION,
    V16AdaptiveController,
    V16FrontierPolicy,
    extract_v16_context,
    load_v16_frontier_policy,
)
from amr_sched.entities import Task
from amr_sched.rl.spaces import (
    EXPERT_V12_FEATURE_NAMES,
    EXPERT_V12_STATE_DIM,
    compute_expert_v7_state,
)


def test_stage0_feature_index_mapping_and_future_task_filter() -> None:
    """Verify `extract_v16_context` reads `conflict_wait_rate[9]`, `detour_delta_p90[41]`, and filters future tasks."""
    assert FEATURE_SCHEMA_VERSION == "v17_causal_v1"
    idx_cwr = EXPERT_V12_FEATURE_NAMES.index("conflict_wait_rate")
    idx_dd90 = EXPERT_V12_FEATURE_NAMES.index("detour_delta_p90")
    assert idx_cwr == 9
    assert idx_dd90 == 41

    obs12 = np.zeros(EXPERT_V12_STATE_DIM, dtype=np.float32)
    obs12[4] = 1.77   # min_slack_ratio (old bug index for wait_sig)
    obs12[40] = 1.88  # detour_delta_mean (old bug index for bay_dd_p90)
    obs12[idx_cwr] = 0.63
    obs12[idx_dd90] = 0.42
    obs7 = np.zeros(30, dtype=np.float32)
    obs7[19] = 0.25

    class _DummySim:
        network = None
        agvs = ()
        _perturbation_model = object()  # Must NEVER be peeked in causal mode
        _execution_feedback = None
        _causal_drift_observed = False
        _t_hid = Task("T_FUT_HID", "N0", "N1", release_step=5, due_step=80)
        _t_hid.hidden_until_release = True  # type: ignore[attr-defined]
        tasks = [
            Task("T_NOW", "N0", "N1", release_step=0, due_step=50),
            Task("T_FUT_REL", "N0", "N1", release_step=20, due_step=80),
            _t_hid,
        ]

    ctx = extract_v16_context(_DummySim(), None, obs12, obs7, tick=10, has_exec_drift=False, causal=True)
    assert abs(float(ctx[4]) - 0.0) < 1e-6, "Oracle _perturbation_model must not set has_drift before feedback"
    assert abs(float(ctx[6]) - 0.0) < 1e-6, "Oracle _perturbation_model must not set p2_active in causal mode"
    assert abs(float(ctx[7]) - 0.25) < 1e-6, "Only released non-hidden T_NOW counts toward uncomp_tasks"
    assert abs(float(ctx[8]) - 0.63) < 1e-6, "Must read obs_v12[9] conflict_wait_rate, not [4]"
    assert abs(float(ctx[9]) - 0.42) < 1e-6, "Must read obs_v12[41] detour_delta_p90, not [40]"
    assert abs(float(ctx[10]) - 0.25) < 1e-6


def test_stage0_strict_causal_prefix_invariance_before_first_disturbance() -> None:
    """Two runs with identical prefix up to tick=18 and differing future events must match 100% before tick=18."""
    pytest.importorskip("gymnasium")
    from amr_sched.rl.env import AMRReschedulingEnv

    def _run_prefix(p2_cfg: P2PerturbationConfig | None, extra_future_task: bool) -> list[tuple[int, str, int]]:
        trace: list[tuple[int, str, int]] = []
        with virtual_clock_scope(poll_ms=0.002):
            env = AMRReschedulingEnv(
                topology="ring",
                seed=7001,
                allow_kitting_shortcut=False,
                pure_physical_reward=True,
                real_compute_window_coupling=True,
                expert_v12_portfolio=True,
                loaded_repair_mode="haven_v12",
                loaded_repair_threshold=20,
                record_trajectory=False,
                v13_kitting_mode="V16_EA_JOINT",
                execution_perturbation=p2_cfg,
                execution_aware=True,
                p4_choke_robust=True,
            )
            obs, info = env.reset(seed=7001, options={"topology": "ring", "stress_axis": "standard"})
            assert env.sim is not None and env.tracker is not None
            if extra_future_task:
                fut = Task("T_FUTURE_99", "W0", "W1", release_step=30, due_step=200)
                fut.hidden_until_release = True  # type: ignore[attr-defined]
                env.sim.tasks.append(fut)
                env.sim._task_by_id["T_FUTURE_99"] = fut
            ctrl = V16AdaptiveController(epoch_interval=16, profile="quality")
            ctrl.reset()

            for step_i in range(18):
                o12 = env.get_expert_v12_obs(obs)
                o7 = compute_expert_v7_state(
                    env.sim,
                    env.tracker,
                    o12[:22],
                    last_replan_tick=int(getattr(env.sim, "_env_last_replan_tick", 0)),
                    recent_conflict_ticks=int(getattr(env.sim, "_env_recent_conflicts", 0)),
                    decision_logs=getattr(env.sim, "_env_decision_logs", None),
                )
                ctx = extract_v16_context(env.sim, env.tracker, o12, o7, tick=step_i, has_exec_drift=False, causal=True)
                h_ctx = hashlib.sha256(np.round(ctx, 5).tobytes()).hexdigest()
                act = ctrl.select_action(o12, info, sim=env.sim)
                trace.append((step_i, h_ctx, int(act)))
                obs, _, tm, tr, info = env.step(act)
                if tm or tr:
                    break
        return trace

    # Future obstacles & local fails occur strictly at tick >= 22, drift disabled for exact t < 22 prefix match
    p2_future = P2PerturbationConfig(
        seed=999,
        enable_travel_drift=False,
        enable_dynamic_obstacles=True,
        obstacle_count=2,
        obstacle_duration_steps=20,
        obstacle_window_range=(22, 60),
        enable_local_planner_fail=True,
        local_fail_count=1,
        local_fail_duration_steps=8,
        local_fail_window_range=(24, 55),
        enable_p4_belief=True,
        enable_p4_voi_probe=True,
    )
    t_clean = _run_prefix(None, False)
    t_future = _run_prefix(p2_future, True)
    assert len(t_clean) == len(t_future) == 18
    assert t_clean == t_future


def test_stage0_pytorch_numpy_ranker_parity_with_keep_zero() -> None:
    """Verify Set-Attention PyTorch forward and `V15ExecutionAwarePolicy.score_candidates` match < 1e-6."""
    torch = pytest.importorskip("torch")
    from amr_sched.cvcr.v15_ea_joint import V15_CAND_FEAT_DIM, V15_GLOBAL_FEAT_DIM

    class _RefSetAttnCandRanker(torch.nn.Module):
        def __init__(self, weights_npz: dict[str, np.ndarray]) -> None:
            super().__init__()
            self.c_lin1 = torch.nn.Linear(V15_CAND_FEAT_DIM, 48)
            self.att_q = torch.nn.Linear(48, 24, bias=False)
            self.att_k = torch.nn.Linear(48, 24, bias=False)
            self.att_v = torch.nn.Linear(48, 48, bias=False)
            self.c_lin2 = torch.nn.Linear(96, 48)
            self.c_lin3 = torch.nn.Linear(48, 1)
            self.g_lin1 = torch.nn.Linear(V15_GLOBAL_FEAT_DIM, 24)
            self.g_lin2 = torch.nn.Linear(24, 1)
            with torch.no_grad():
                self.c_lin1.weight.copy_(torch.tensor(weights_npz["cand_w1"].T))
                self.c_lin1.bias.copy_(torch.tensor(weights_npz["cand_b1"]))
                self.att_q.weight.copy_(torch.tensor(weights_npz["att_q"].T))
                self.att_k.weight.copy_(torch.tensor(weights_npz["att_k"].T))
                self.att_v.weight.copy_(torch.tensor(weights_npz["att_v"].T))
                self.c_lin2.weight.copy_(torch.tensor(weights_npz["cand_w2"].T))
                self.c_lin2.bias.copy_(torch.tensor(weights_npz["cand_b2"]))
                self.c_lin3.weight.copy_(torch.tensor(weights_npz["cand_w3"].T))
                self.c_lin3.bias.copy_(torch.tensor(weights_npz["cand_b3"]))
                self.g_lin1.weight.copy_(torch.tensor(weights_npz["g_w1"].T))
                self.g_lin1.bias.copy_(torch.tensor(weights_npz["g_b1"]))
                self.g_lin2.weight.copy_(torch.tensor(weights_npz["g_w2"].T))
                self.g_lin2.bias.copy_(torch.tensor(weights_npz["g_b2"]))

        def forward(
            self,
            cand_norm: torch.Tensor,
            g_norm: torch.Tensor,
            det_vec: torch.Tensor,
            valid_mask: torch.Tensor,
            *,
            is_keep_mask: torch.Tensor,
            gate_scale: float = 0.25,
        ) -> torch.Tensor:
            h = torch.relu(self.c_lin1(cand_norm))
            q, k, v = self.att_q(h), self.att_k(h), self.att_v(h)
            att = torch.softmax((q @ k.T) / (24.0 ** 0.5), dim=-1)
            h_att = torch.relu(self.c_lin2(torch.cat([h, att @ v], dim=-1)))
            delta = self.c_lin3(h_att).squeeze(-1)
            hg = torch.relu(self.g_lin1(g_norm))
            trust = torch.sigmoid(self.g_lin2(hg)).squeeze() * 2.0 * gate_scale
            adj = torch.tanh(delta / 2.0) * trust * (1.0 - is_keep_mask)
            return (det_vec + adj) - (1.0 - valid_mask) * 1e6

    pol = load_v16_frontier_policy()
    assert pol.w is not None and "cand_w1" in pol.w
    w = pol.w
    net = _RefSetAttnCandRanker(w).eval()

    rng = np.random.default_rng(2027)
    n_cands = 5
    cm = rng.normal(0.0, 1.0, size=(n_cands, 36)).astype(np.float32)
    cm[:, 16:21] = 0.0
    cm[0, 16] = 1.0  # C0 is keep_incumbent
    cm[1, 17] = 1.0
    cm[2, 18] = 1.0
    cm[3, 19] = 1.0
    cm[4, 20] = 1.0
    gf = rng.normal(0.0, 1.0, size=(14,)).astype(np.float32)
    det_scores = np.array([0.0, 0.35, -0.2, 0.8, 0.1], dtype=np.float32)
    valid = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    fams = [
        "keep_incumbent",
        "fast_detour_regret",
        "balans_adaptive_destroy",
        "v10_crit_ci_balance",
        "selective_cliff_rescue",
    ]

    class _DummyEval:
        valid = True

    cands = [
        V15Candidate(
            cand_id=f"C{i}",
            source_kind=fams[i],
            assignment={},
            bay_overrides={},
            crew_overrides={},
            op_priority={},
            eval_res=_DummyEval(),  # type: ignore[arg-type]
            traffic_weights={},
            traffic_cutoff=0,
            guide_routes={},
            commit_seq_horizon=0,
            destroy_tids=(),
            cliff_saved=False,
            det_score=float(det_scores[i]),
            features=cm[i],
        )
        for i in range(n_cands)
    ]

    np_scores = pol.score_candidates(gf, cands)
    with torch.no_grad():
        cn = torch.tensor(np.clip((cm - w["c_mu"]) / np.maximum(0.05, w["c_sd"]), -6.0, 6.0), dtype=torch.float32)
        gn = torch.tensor(np.clip((gf - w["g_mu"]) / np.maximum(0.05, w["g_sd"]), -6.0, 6.0), dtype=torch.float32)
        dt = torch.tensor(det_scores, dtype=torch.float32)
        vt = torch.tensor(valid, dtype=torch.float32)
        km = torch.tensor((cm[:, 16] > 0.5).astype(np.float32), dtype=torch.float32)
        pt_scores = net(cn, gn, dt, vt, is_keep_mask=km).numpy()

    max_diff = float(np.max(np.abs(np_scores - pt_scores)))
    assert max_diff < 1e-5, f"PyTorch vs NumPy mismatch max_diff={max_diff}"
    assert abs(float(pt_scores[0]) - float(det_scores[0])) < 1e-6, "keep_incumbent residual must be 0"


def test_stage0_probe_override_and_seven_cost_columns() -> None:
    """Verify `allow_probe_override` bypasses silent fallback and env reports exact 7 orthogonal costs."""
    pytest.importorskip("gymnasium")
    from amr_sched.rl.env import AMRReschedulingEnv

    # 1. Verify 7 orthogonal cost columns sum cleanly to total_obj & net_obj
    with virtual_clock_scope(poll_ms=0.002):
        env = AMRReschedulingEnv(
            topology="ring",
            seed=7001,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            expert_v12_portfolio=True,
            loaded_repair_mode="haven_v12",
            loaded_repair_threshold=20,
            record_trajectory=False,
            v13_kitting_mode="V16_EA_JOINT",
            execution_aware=True,
            p4_choke_robust=True,
        )
        obs, info = env.reset(seed=7001, options={"topology": "ring", "stress_axis": "standard"})
        ctrl = V16AdaptiveController(epoch_interval=16, profile="quality")
        ctrl.reset()
        done = False
        while not done:
            act = ctrl.select_action(env.get_expert_v12_obs(obs), info, sim=env.sim)
            obs, _, tm, tr, info = env.step(act)
            done = tm or tr
        m = env.compute_episode_metrics()

    for k in (
        "cmax",
        "task_tardiness_cost",
        "op_tardiness_cost",
        "unfinished_task_penalty",
        "unfinished_op_penalty",
        "churn_cost",
        "compute_cost",
    ):
        assert k in m, f"Missing cost breakdown column: {k}"
    assert m["unfinished_task_penalty"] == 0.0
    assert m["unfinished_op_penalty"] == 0.0
    tot_reconstructed = (
        float(m["cmax"])
        + float(m["task_tardiness_cost"])
        + float(m["op_tardiness_cost"])
        + float(m["unfinished_task_penalty"])
        + float(m["unfinished_op_penalty"])
    )
    net_reconstructed = tot_reconstructed + float(m["churn_cost"]) + float(m["compute_cost"])
    assert abs(tot_reconstructed - float(m["total_obj"])) <= 0.05
    assert abs(net_reconstructed - float(m["net_obj"])) <= 0.05

    # 2. Verify probe override observability inside policy._search_v15_execution_aware via mock candidate scoring
    from amr_sched.cvcr.commitment import classify_commitments
    from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult

    with virtual_clock_scope(poll_ms=0.002):
        env2 = AMRReschedulingEnv(topology="ring", seed=7002, v13_kitting_mode="V16_EA_JOINT")
        env2.reset(seed=7002)
        assert env2.sim is not None and env2.tracker is not None
        snap = NativeSnapshotCapture.capture(env2.sim, env2.tracker.operations, hide_future_releases=True)
        clf = classify_commitments(snap)
        ev = CVCREvaluator(env2.sim.network, snap, use_rust=False, mode="B2")
        op_prio = {op.op_id: int(op.due_step) for op in snap.operations}
        pol_obj = LGCVCRPolicy("V16_EA_JOINT", env2.sim.network, use_rust=False)
        healthy = pol_obj._healthy_robots(snap, clf.disturbed_robots)
        base_assign = pol_obj._base_assignment(snap, healthy, op_prio)
        base_ev = ev.evaluate(base_assign, op_priority=op_prio)

        class _ForcedWorstProbe(V16FrontierPolicy):
            def score_candidates(self, g_feats: np.ndarray, cands: list[V15Candidate]) -> np.ndarray:
                out = np.full((len(cands),), -1e5, dtype=np.float32)
                if len(cands) >= 2:
                    # Force C1 even if we artificially inflate C1's objective gap > det_best + 12.0
                    cands[1].eval_res = EvalResult(
                        robot_paths=dict(cands[1].eval_res.robot_paths),
                        robot_order_seq=dict(cands[1].eval_res.robot_order_seq),
                        order_unload_ticks=dict(cands[1].eval_res.order_unload_ticks),
                        op_start=dict(cands[1].eval_res.op_start),
                        op_end=dict(cands[1].eval_res.op_end),
                        op_res_ready=dict(cands[1].eval_res.op_res_ready),
                        objective=float(cands[0].eval_res.objective) + 50.0,
                        valid=True,
                        starvation_ticks=int(cands[1].eval_res.starvation_ticks),
                        eval_latency_ms=0.0,
                        mode="B2",
                    )
                    cands[1].det_score = float(cands[0].det_score) - 5.0
                    cands[1].cliff_saved = False
                    out[1] = 1e5
                return out

        for override_flag, expect_fallback in ((False, "det_guard_obj_gt_12"), (True, "none")):
            pr = _ForcedWorstProbe(None, allow_probe_override=override_flag)
            p_inst = LGCVCRPolicy("V16_EA_JOINT", env2.sim.network, use_rust=False, v15_policy=pr)
            import time as _t
            p_inst._search_v15_execution_aware(
                snap, clf, ev, base_assign, base_ev, healthy, op_prio, _t.perf_counter() + 0.2, budget_ms=50.0
            )
            kd = p_inst._last_kitting_decision
            assert kd["fallback_reason"] == expect_fallback
            assert kd["requested_cand_id"] != kd["selected_cand_id"] if not override_flag else kd["requested_cand_id"] == kd["selected_cand_id"]
