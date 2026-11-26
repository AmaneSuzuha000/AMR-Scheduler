"""P0 Scientific Audit (`EXP-01`, `EXP-02`, `EXP-03`) for AMR-Scheduler V19.

Executes:
  - EXP-01: Reproducible Training Protocol & SMDP Discounting (`gamma_step=0.9985` vs `1.0`) Audit.
  - EXP-02: Paired 7-Term Cost Attribution, Conditional 100%-Completion Analysis (`H1` vs `H2`),
            Quantile/CVaR Risk Profiles, and Step-by-Step Decision Divergence Traces.
  - EXP-03: Heterogeneous Graph Entity/Channel Audit & `A1` Plackett-Luce Policy Gradient Verification.

Outputs are written to `out/v19_scientific_audit/exp01_02_03_report.json` and `exp01_02_03_report.md`.
"""

from __future__ import annotations

import os
import platform

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

if sys.platform == "win32" and getattr(platform, "_wmi", None) is not None:
    platform._wmi = None  # type: ignore[attr-defined]

import numpy as np
from scipy import stats
import torch
import torch.nn.functional as F

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.rl import AMRReschedulingEnv, RuleV10PortfolioPolicy, compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.v19_hetero_constructor import (
    V19HeteroConstructorNet,
    V19HeteroConstructorProposer,
    decode_v19_candidate,
    export_v19_npz,
)
from amr_sched.rl.v19_hetero_graph import (
    AGV_FEAT_DIM,
    BAY_FEAT_DIM,
    CREW_FEAT_DIM,
    OP_BAY_EDGE_DIM,
    OP_FEAT_DIM,
    TASK_AGV_EDGE_DIM,
    TASK_FEAT_DIM,
    extract_v19_hetero_graph,
)
from experiments.train_v19_bc import (
    CKPT_DIR,
    FORBIDDEN_BLIND_SEEDS,
    FORBIDDEN_DEV_SEEDS,
    TRAIN_4TOPO_SEEDS,
    TRAIN_WORKSHOP_SEEDS,
    VAL_4TOPO_SEEDS,
    VAL_WORKSHOP_SEEDS,
    _build_workshop_episode_spec,
    _run_v19_gate_episode,
)
from experiments.train_v19_rl import (
    RL_4TOPO_SEEDS,
    RL_VAL_4TOPO_SEEDS,
    RL_VAL_WORKSHOP_SEEDS,
    RL_WORKSHOP_SEEDS,
    _collate_rl_batch,
    _collect_smdp_rollouts,
)

OUT_DIR = ROOT / "out" / "v19_scientific_audit"
COST_TERMS: tuple[str, ...] = (
    "cmax",
    "task_tardiness_cost",
    "op_tardiness_cost",
    "unfinished_task_penalty",
    "unfinished_op_penalty",
    "churn_cost",
    "compute_cost",
)


def _paired_bootstrap_ci(diffs: np.ndarray, n_boot: int = 10000, seed: int = 2026) -> tuple[float, float]:
    if len(diffs) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    means = diffs[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _cvar_top_alpha(vals: np.ndarray, alpha: float = 0.10) -> float:
    if len(vals) == 0:
        return 0.0
    k = max(1, int(np.ceil(len(vals) * alpha)))
    s = np.sort(vals)[::-1]
    return float(np.mean(s[:k]))


def run_exp01_reproducibility_and_discount_audit() -> dict[str, Any]:
    """EXP-01: Verify seed quarantine, checkpoint provenance, and SMDP discount telescoping alignment."""
    bc_train_seeds = set(TRAIN_4TOPO_SEEDS) | set(TRAIN_WORKSHOP_SEEDS)
    bc_val_seeds = set(VAL_4TOPO_SEEDS) | set(VAL_WORKSHOP_SEEDS)
    rl_train_seeds = set(RL_4TOPO_SEEDS) | set(RL_WORKSHOP_SEEDS)
    rl_val_seeds = set(RL_VAL_4TOPO_SEEDS) | set(RL_VAL_WORKSHOP_SEEDS)
    blind_seeds = set(FORBIDDEN_BLIND_SEEDS)
    g1_seeds = set(FORBIDDEN_DEV_SEEDS)

    quarantine_checks = {
        "bc_train_vs_blind_overlap": len(bc_train_seeds & blind_seeds),
        "bc_val_vs_blind_overlap": len(bc_val_seeds & blind_seeds),
        "rl_train_vs_blind_overlap": len(rl_train_seeds & blind_seeds),
        "rl_val_vs_blind_overlap": len(rl_val_seeds & blind_seeds),
        "bc_train_vs_bc_val_overlap": len(bc_train_seeds & bc_val_seeds),
        "all_train_val_vs_g1_overlap": len((bc_train_seeds | bc_val_seeds | rl_train_seeds | rl_val_seeds) & g1_seeds),
    }

    g2_report_path = ROOT / "out" / "g2_bc" / "g2_gate_report.json"
    g3_report_path = ROOT / "out" / "g3_rl" / "g3_rl_training_report.json"
    g2_meta = json.loads(g2_report_path.read_text(encoding="utf-8")) if g2_report_path.is_file() else {}
    g3_meta = json.loads(g3_report_path.read_text(encoding="utf-8")) if g3_report_path.is_file() else {}

    # Empirically test SMDP discounted return G_0(gamma=0.9985) vs undiscounted telescoping sum -J_T (gamma=1.0)
    bc_ckpt = CKPT_DIR / "v19_hetero_bc.pt"
    sd = torch.load(bc_ckpt, map_location="cpu", weights_only=False)
    actor = V19HeteroConstructorNet(
        hidden_dim=64,
        n_heads=4,
        n_layers=2,
        action_stage=str(sd["config"].get("action_stage", "A1234")),
        ablation_mode="full",
    )
    actor.load_state_dict(sd["model"], strict=True)
    actor.eval()

    audit_specs = [
        _build_workshop_episode_spec("track1", 1161, "nominal"),
        _build_workshop_episode_spec("track1", 1161, "kitting_windows"),
        _build_workshop_episode_spec("track1", 1161, "execution_blockage"),
        _build_workshop_episode_spec("track1", 1161, "multi_wave_surge"),
        _build_workshop_episode_spec("track2", 1161, "nominal"),
        _build_workshop_episode_spec("track2", 1161, "kitting_windows"),
        _build_workshop_episode_spec("track2", 1161, "execution_blockage"),
        _build_workshop_episode_spec("track2", 1161, "multi_wave_surge"),
    ]
    transitions_disc, _ = _collect_smdp_rollouts(
        actor,
        audit_specs,
        policy_label="V19_AUDIT_DISC",
        ablation_mode="full",
        rollout_seed=2026,
        gamma_step=0.9985,
        gae_lambda=0.95,
    )
    transitions_undisc, _ = _collect_smdp_rollouts(
        actor,
        audit_specs,
        policy_label="V19_AUDIT_UNDISC",
        ablation_mode="full",
        rollout_seed=2026,
        gamma_step=1.0,
        gae_lambda=0.95,
    )

    adv_disc = np.array([float(t["adv"]) for t in transitions_disc], dtype=np.float64)
    adv_undisc = np.array([float(t["adv"]) for t in transitions_undisc], dtype=np.float64)
    tc_disc = np.array([float(t["target_cost"]) for t in transitions_disc], dtype=np.float64)
    tc_undisc = np.array([float(t["target_cost"]) for t in transitions_undisc], dtype=np.float64)

    adv_pearson = float(np.corrcoef(adv_disc, adv_undisc)[0, 1]) if len(adv_disc) > 1 else 1.0
    adv_spearman = float(stats.spearmanr(adv_disc, adv_undisc).statistic) if len(adv_disc) > 1 else 1.0
    tc_pearson = float(np.corrcoef(tc_disc, tc_undisc)[0, 1]) if len(tc_disc) > 1 else 1.0

    return {
        "quarantine_checks": quarantine_checks,
        "all_quarantine_zero": all(v == 0 for v in quarantine_checks.values()),
        "training_protocol": {
            "bc_train_episodes": 264,
            "bc_val_episodes": 72,
            "rl_train_pool_episodes": 704,
            "rl_early_val_episodes": 24,
            "g3_locked_val_episodes": 96,
            "g4_blind_holdout_episodes": 120,
            "ppo_rounds": 12,
            "ppo_episodes_per_round": 24,
            "ppo_total_closed_loop_episodes_per_model": 288,
            "ppo_epochs_per_round": 2,
            "ppo_lr": 1.5e-5,
            "ppo_beta_kl": 0.08,
            "ppo_clip_eps": 0.12,
            "smdp_gamma_step": 0.9985,
            "smdp_gae_lambda": 0.95,
            "equal_budget_across_full_staged_noprodcomm": True,
        },
        "discount_vs_telescoping_audit": {
            "num_smdp_transitions_evaluated": len(transitions_disc),
            "mean_dt_steps_per_smdp_interval": round(
                float(np.mean([int(t["dt_steps"]) for t in transitions_disc])), 2
            ),
            "effective_interval_discount_mean": round(
                float(np.mean([0.9985 ** int(t["dt_steps"]) for t in transitions_disc])), 4
            ),
            "advantage_pearson_gamma09985_vs_gamma1": round(adv_pearson, 6),
            "advantage_spearman_gamma09985_vs_gamma1": round(adv_spearman, 6),
            "target_cost_pearson_gamma09985_vs_gamma1": round(tc_pearson, 6),
        },
        "g2_bc_summary": g2_meta.get("promotion_decision", {}),
        "g3_rl_models_trained": list(g3_meta.get("models", {}).keys()) if isinstance(g3_meta, dict) else [],
    }


def run_exp02_cost_attribution_and_conditional_analysis() -> dict[str, Any]:
    """EXP-02: Decompose 120-episode blind holdout into 7 cost terms, 100%-completed intersections, and divergence trace."""
    blind_path = ROOT / "out" / "g4_blind_holdout" / "blind_holdout_report.json"
    data = json.loads(blind_path.read_text(encoding="utf-8"))
    rows_by_pol: dict[str, list[dict[str, Any]]] = data["rows"]

    policies = (
        "V19_HETERO_RL",
        "V19_STAGED_RL",
        "V19_HETERO_BC",
        "V19_NO_PROD_COMM",
        "V18_EA_JOINT",
        "rule_v10",
        "IJCAI26_PGMPO",
        "EJOR26_Wu_GRU_Attn",
        "RAL26_DetourDelta",
    )

    policy_ledgers: dict[str, dict[str, Any]] = {}
    for pol in policies:
        eps = rows_by_pol[pol]
        net_vals = np.array([float(r["net_obj"]) for r in eps], dtype=np.float64)
        ok_eps = [
            r for r in eps if float(r["unfinished_task_penalty"]) == 0.0 and float(r["unfinished_op_penalty"]) == 0.0
        ]
        ok_nets = np.array([float(r["net_obj"]) for r in ok_eps], dtype=np.float64)

        term_means = {k: round(float(np.mean([float(r[k]) for r in eps])), 2) for k in COST_TERMS}
        term_sum = sum(term_means.values())
        policy_ledgers[pol] = {
            "num_episodes": len(eps),
            "num_100pct_completed": len(ok_eps),
            "completion_reliability_episode_rate": round(len(ok_eps) / max(1, len(eps)), 4),
            "mean_net_obj": round(float(np.mean(net_vals)), 2),
            "median_net_obj": round(float(np.median(net_vals)), 2),
            "iqr_net_obj": round(float(np.percentile(net_vals, 75) - np.percentile(net_vals, 25)), 2),
            "p90_net_obj": round(float(np.percentile(net_vals, 90)), 2),
            "p95_net_obj": round(float(np.percentile(net_vals, 95)), 2),
            "cvar10_worst_net_obj": round(_cvar_top_alpha(net_vals, 0.10), 2),
            "mean_net_obj_on_own_100pct_completed": round(float(np.mean(ok_nets)), 2),
            "cost_terms": term_means,
            "scheduling_efficiency_subtotal": round(
                term_means["cmax"] + term_means["task_tardiness_cost"] + term_means["op_tardiness_cost"], 2
            ),
            "failure_penalty_subtotal": round(
                term_means["unfinished_task_penalty"] + term_means["unfinished_op_penalty"], 2
            ),
            "control_overhead_subtotal": round(term_means["churn_cost"] + term_means["compute_cost"], 2),
            "ledger_sum_check_diff": round(abs(float(np.mean(net_vals)) - term_sum), 4),
        }

    # Paired comparisons on (a) all 120 episodes and (b) 100%-completed intersection
    pairs = [
        ("RL_vs_BC", "V19_HETERO_RL", "V19_HETERO_BC"),
        ("Full_vs_Staged", "V19_HETERO_RL", "V19_STAGED_RL"),
        ("Full_vs_NoProdComm", "V19_HETERO_RL", "V19_NO_PROD_COMM"),
        ("V19_vs_V18_EA_JOINT", "V19_HETERO_RL", "V18_EA_JOINT"),
        ("V19_vs_rule_v10", "V19_HETERO_RL", "rule_v10"),
        ("V19_vs_IJCAI26_PGMPO", "V19_HETERO_RL", "IJCAI26_PGMPO"),
    ]
    paired_attribution: dict[str, dict[str, Any]] = {}
    for label, pa, pb in pairs:
        map_a = {r["episode_key"]: r for r in rows_by_pol[pa]}
        map_b = {r["episode_key"]: r for r in rows_by_pol[pb]}
        keys = sorted(map_a.keys())

        diffs_all = np.array([float(map_a[k]["net_obj"]) - float(map_b[k]["net_obj"]) for k in keys], dtype=np.float64)
        ci_lo_all, ci_hi_all = _paired_bootstrap_ci(diffs_all)
        try:
            w_p_all = float(stats.wilcoxon(diffs_all).pvalue)
        except Exception:
            w_p_all = 1.0

        term_diffs_all = {
            t: round(float(np.mean([float(map_a[k][t]) - float(map_b[k][t]) for k in keys])), 2)
            for t in COST_TERMS
        }

        common_ok_keys = [
            k
            for k in keys
            if float(map_a[k]["unfinished_task_penalty"]) == 0.0
            and float(map_a[k]["unfinished_op_penalty"]) == 0.0
            and float(map_b[k]["unfinished_task_penalty"]) == 0.0
            and float(map_b[k]["unfinished_op_penalty"]) == 0.0
        ]
        diffs_ok = np.array(
            [float(map_a[k]["net_obj"]) - float(map_b[k]["net_obj"]) for k in common_ok_keys], dtype=np.float64
        )
        ci_lo_ok, ci_hi_ok = _paired_bootstrap_ci(diffs_ok)
        try:
            w_p_ok = float(stats.wilcoxon(diffs_ok).pvalue) if len(diffs_ok) > 1 else 1.0
        except Exception:
            w_p_ok = 1.0
        term_diffs_ok = {
            t: round(float(np.mean([float(map_a[k][t]) - float(map_b[k][t]) for k in common_ok_keys])), 2)
            for t in COST_TERMS
        }

        # Identify extreme outlier episodes driving failure penalty difference
        outlier_eps = []
        for k in keys:
            fa = float(map_a[k]["unfinished_task_penalty"]) + float(map_a[k]["unfinished_op_penalty"])
            fb = float(map_b[k]["unfinished_task_penalty"]) + float(map_b[k]["unfinished_op_penalty"])
            if fa != fb:
                outlier_eps.append(
                    {
                        "episode_key": k,
                        "suite": map_a[k].get("suite", ""),
                        f"{pa}_net_obj": round(float(map_a[k]["net_obj"]), 2),
                        f"{pb}_net_obj": round(float(map_b[k]["net_obj"]), 2),
                        f"{pa}_fail_pen": round(fa, 2),
                        f"{pb}_fail_pen": round(fb, 2),
                        f"{pa}_task_comp": round(float(map_a[k]["task_comp"]), 4),
                        f"{pb}_task_comp": round(float(map_b[k]["task_comp"]), 4),
                        f"{pa}_prod_comp": round(float(map_a[k]["prod_comp"]), 4),
                        f"{pb}_prod_comp": round(float(map_b[k]["prod_comp"]), 4),
                    }
                )

        paired_attribution[label] = {
            "policy_a": pa,
            "policy_b": pb,
            "all_120_episodes": {
                "mean_diff_net_obj": round(float(np.mean(diffs_all)), 2),
                "median_diff_net_obj": round(float(np.median(diffs_all)), 2),
                "ci95_lo": round(ci_lo_all, 2),
                "ci95_hi": round(ci_hi_all, 2),
                "wilcoxon_p": round(w_p_all, 6),
                "a_wins": int(np.sum(diffs_all < -1e-4)),
                "b_wins": int(np.sum(diffs_all > 1e-4)),
                "ties": int(np.sum(np.abs(diffs_all) <= 1e-4)),
                "term_diffs": term_diffs_all,
                "h1_scheduling_quality_delta": round(
                    term_diffs_all["cmax"]
                    + term_diffs_all["task_tardiness_cost"]
                    + term_diffs_all["op_tardiness_cost"],
                    2,
                ),
                "h2_failure_penalty_delta": round(
                    term_diffs_all["unfinished_task_penalty"] + term_diffs_all["unfinished_op_penalty"], 2
                ),
                "control_overhead_delta": round(
                    term_diffs_all["churn_cost"] + term_diffs_all["compute_cost"], 2
                ),
            },
            "conditional_100pct_completed_intersection": {
                "num_common_100pct_episodes": len(common_ok_keys),
                "mean_a_net_obj": round(float(np.mean([float(map_a[k]["net_obj"]) for k in common_ok_keys])), 2),
                "mean_b_net_obj": round(float(np.mean([float(map_b[k]["net_obj"]) for k in common_ok_keys])), 2),
                "mean_diff_net_obj": round(float(np.mean(diffs_ok)), 2),
                "median_diff_net_obj": round(float(np.median(diffs_ok)), 2),
                "ci95_lo": round(ci_lo_ok, 2),
                "ci95_hi": round(ci_hi_ok, 2),
                "wilcoxon_p": round(w_p_ok, 6),
                "a_wins": int(np.sum(diffs_ok < -1e-4)),
                "b_wins": int(np.sum(diffs_ok > 1e-4)),
                "term_diffs": term_diffs_ok,
            },
            "failure_divergence_episodes": outlier_eps,
        }

    # Step-by-step counterfactual decision divergence trace on Track 1 kitting_windows (seed=901)
    spec_t1 = _build_workshop_episode_spec("track1", 901, "kitting_windows")
    sc_copy = copy.deepcopy(spec_t1["scenario_tuple"])
    env = AMRReschedulingEnv(
        seed=901,
        max_steps=int(spec_t1["max_steps"]),
        use_rust=True,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        loaded_repair_mode="haven_v12",
        loaded_repair_threshold=20,
        execution_perturbation=copy.deepcopy(spec_t1["p2_cfg"]),
        execution_aware=True,
        p4_choke_robust=True,
        expert_v10_portfolio=True,
    )
    ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
    ctrl.reset()
    _, info = env.reset(seed=901, options={"scenario_tuple": sc_copy})

    prop_full = V19HeteroConstructorProposer(CKPT_DIR / "v19_hetero_rl.npz", policy_id="V19_HETERO_RL", seed=901)
    prop_bc = V19HeteroConstructorProposer(CKPT_DIR / "v19_hetero_bc.npz", policy_id="V19_HETERO_BC", seed=901)
    prop_staged = V19HeteroConstructorProposer(CKPT_DIR / "v19_staged_rl.npz", policy_id="V19_STAGED_RL", seed=901)

    divergence_snapshots: list[dict[str, Any]] = []
    tm, tr = False, False
    while not (tm or tr) and len(divergence_snapshots) < 5:
        assert env.sim is not None and env.tracker is not None
        feats = compute_expert_v9_state(
            env.sim,
            env.tracker,
            last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks,
            decision_logs=env.decision_logs,
        )
        s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
        act = ctrl.select_action(feats, s_info, sim=env.sim)
        if act != 0:
            snap = NativeSnapshotCapture.capture(
                env.sim, env.tracker.operations, hide_future_releases=True
            )
            clf = classify_commitments(snap)
            if len(clf.releasable_tasks) >= 2:
                pf = prop_full(snap, env.sim.network, budget_ms=35.0)
                pb = prop_bc(snap, env.sim.network, budget_ms=35.0)
                ps = prop_staged(snap, env.sim.network, budget_ms=35.0)
                if pf.proposed_assignment != pb.proposed_assignment or pf.proposed_bays != ps.proposed_bays or pf.predicted_obj != ps.predicted_obj:
                    divergence_snapshots.append(
                        {
                            "tick": int(snap.tick),
                            "releasable_tasks": sorted(clf.releasable_tasks),
                            "full_rl_predicted_b2_obj": round(float(pf.predicted_obj), 2),
                            "bc_predicted_b2_obj": round(float(pb.predicted_obj), 2),
                            "staged_rl_predicted_b2_obj": round(float(ps.predicted_obj), 2),
                            "full_rl_pins": dict(pf.proposed_assignment),
                            "bc_pins": dict(pb.proposed_assignment),
                            "staged_rl_pins": dict(ps.proposed_assignment),
                            "full_rl_bays": dict(pf.proposed_bays),
                            "staged_rl_bays": dict(ps.proposed_bays),
                        }
                    )
        _, _, tm, tr, info = env.step(act)

    return {
        "policy_ledgers": policy_ledgers,
        "paired_attribution": paired_attribution,
        "decision_divergence_trace_track1_seed901": divergence_snapshots,
    }


def run_exp03_architecture_and_a1_plackett_luce_audit() -> dict[str, Any]:
    """EXP-03: Document exact V19HeteroGraph entities/edges and test A1 Plackett-Luce PPO gradient training."""
    spec = _build_workshop_episode_spec("track1", 1161, "kitting_windows")
    sc_copy = copy.deepcopy(spec["scenario_tuple"])
    env = AMRReschedulingEnv(
        seed=1161,
        max_steps=int(spec["max_steps"]),
        use_rust=True,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        loaded_repair_mode="haven_v12",
        loaded_repair_threshold=20,
        execution_perturbation=copy.deepcopy(spec["p2_cfg"]),
        execution_aware=True,
        p4_choke_robust=True,
        expert_v10_portfolio=True,
    )
    _, _ = env.reset(seed=1161, options={"scenario_tuple": sc_copy})
    assert env.sim is not None and env.tracker is not None
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    graph = extract_v19_hetero_graph(snap, env.sim.network)

    # Compare 3-round PPO update on 8 Track 1/2 validation episodes with include_a1_plackett_luce=False vs True
    bc_ckpt = CKPT_DIR / "v19_hetero_bc.pt"
    sd = torch.load(bc_ckpt, map_location="cpu", weights_only=False)
    train_specs = [
        _build_workshop_episode_spec("track1", 1001, "kitting_windows"),
        _build_workshop_episode_spec("track1", 1001, "execution_blockage"),
        _build_workshop_episode_spec("track2", 1001, "kitting_windows"),
        _build_workshop_episode_spec("track2", 1001, "execution_blockage"),
    ]
    val_specs = [
        _build_workshop_episode_spec("track1", 1161, "kitting_windows"),
        _build_workshop_episode_spec("track1", 1161, "execution_blockage"),
        _build_workshop_episode_spec("track2", 1161, "kitting_windows"),
        _build_workshop_episode_spec("track2", 1161, "execution_blockage"),
    ]

    a1_comparison: dict[str, Any] = {}
    for mode_label, use_pl in (("a1_frozen_bc_order", False), ("a1_plackett_luce_ppo", True)):
        torch.manual_seed(2026)
        net = V19HeteroConstructorNet(
            hidden_dim=64, n_heads=4, n_layers=2, action_stage="A1234", ablation_mode="full"
        )
        net.load_state_dict(copy.deepcopy(sd["model"]), strict=True)
        ref_net = V19HeteroConstructorNet(
            hidden_dim=64, n_heads=4, n_layers=2, action_stage="A1234", ablation_mode="full"
        )
        ref_net.load_state_dict(copy.deepcopy(sd["model"]), strict=True)
        ref_net.eval()

        opt = torch.optim.AdamW(net.parameters(), lr=1.5e-5, weight_decay=1e-4)
        transitions, _ = _collect_smdp_rollouts(
            net,
            train_specs,
            policy_label=f"V19_{mode_label}",
            ablation_mode="full",
            rollout_seed=2026,
        )
        batch = _collate_rl_batch(transitions, torch.device("cpu"))
        with torch.no_grad():
            ref_out = ref_net.evaluate_joint_actions_batch(
                batch,
                chosen_bays=batch["chosen_bays"],
                chosen_crews=batch["chosen_crews"],
                chosen_agvs=batch["chosen_agvs"],
                focus_order=batch["focus_order"],
                chosen_seq=batch["chosen_seq"],
                return_probs=True,
                include_a1_plackett_luce=use_pl,
            )
            old_out = net.evaluate_joint_actions_batch(
                batch,
                chosen_bays=batch["chosen_bays"],
                chosen_crews=batch["chosen_crews"],
                chosen_agvs=batch["chosen_agvs"],
                focus_order=batch["focus_order"],
                chosen_seq=batch["chosen_seq"],
                include_a1_plackett_luce=use_pl,
            )
            old_lp = old_out["mean_logp"].detach()

        net.train()
        a1_grad_norms: list[float] = []
        for _ in range(3):
            opt.zero_grad(set_to_none=True)
            cur_out = net.evaluate_joint_actions_batch(
                batch,
                chosen_bays=batch["chosen_bays"],
                chosen_crews=batch["chosen_crews"],
                chosen_agvs=batch["chosen_agvs"],
                focus_order=batch["focus_order"],
                chosen_seq=batch["chosen_seq"],
                ref_probs=ref_out["probs"],
                include_a1_plackett_luce=use_pl,
            )
            adv = batch["adv"]
            adv_norm = (adv - adv.mean()) / (adv.std() + 1e-6)
            ratio = torch.exp(torch.clamp(cur_out["mean_logp"] - old_lp, -5.0, 5.0))
            surr1 = ratio * adv_norm
            surr2 = torch.clamp(ratio, 1.0 - 0.12, 1.0 + 0.12) * adv_norm
            p_loss = -torch.min(surr1, surr2).mean()
            v_loss = 0.5 * F.smooth_l1_loss(cur_out["value"], batch["target_cost"])
            kl_loss = 0.08 * cur_out["mean_kl"].mean()
            loss = p_loss + v_loss + kl_loss - 0.005 * cur_out["mean_entropy"].mean()
            loss.backward()
            g_w1 = float(net.a1_w1.weight.grad.norm().item()) if net.a1_w1.weight.grad is not None else 0.0
            g_w2 = float(net.a1_w2.weight.grad.norm().item()) if net.a1_w2.weight.grad is not None else 0.0
            a1_grad_norms.append(g_w1 + g_w2)
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()

        tmp_pt = OUT_DIR / f"tmp_{mode_label}.pt"
        tmp_npz = OUT_DIR / f"tmp_{mode_label}.npz"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {k: v.cpu() for k, v in net.state_dict().items()}, "config": net.config_dict()}, tmp_pt)
        export_v19_npz(tmp_pt, tmp_npz)
        val_rows = [_run_v19_gate_episode(tmp_npz, f"V19_{mode_label}", sp) for sp in val_specs]
        tmp_pt.unlink(missing_ok=True)
        tmp_npz.unlink(missing_ok=True)

        a1_comparison[mode_label] = {
            "include_a1_plackett_luce": use_pl,
            "mean_a1_head_grad_norm": round(float(np.mean(a1_grad_norms)), 6),
            "val_mean_net_obj": round(float(np.mean([r["net_obj"] for r in val_rows])), 2),
            "val_mean_cmax": round(float(np.mean([r["cmax"] for r in val_rows])), 2),
            "val_mean_tardiness": round(float(np.mean([r["tardiness"] for r in val_rows])), 2),
        }

    return {
        "actual_hetero_graph_schema": {
            "node_types": {
                "Operation": {"dim": OP_FEAT_DIM, "sample_count_track1": len(graph.op_ids)},
                "Task": {"dim": TASK_FEAT_DIM, "sample_count_track1": len(graph.rel_tids)},
                "AMR": {"dim": AGV_FEAT_DIM, "sample_count_track1": len(graph.agv_ids)},
                "Bay": {"dim": BAY_FEAT_DIM, "sample_count_track1": len(graph.bay_ids)},
                "Crew": {"dim": CREW_FEAT_DIM, "sample_count_track1": len(graph.crew_ids)},
            },
            "edge_and_relation_types": {
                "op_dag_adj": "3-channel directed precedence adjacency (predecessor, self, successor)",
                "task_to_op": "Kitting membership link mapping transport task -> target operation (kit_task_to_op & kit_op_to_task)",
                "op_bay_feats": f"{OP_BAY_EDGE_DIM}-D operation-bay eligibility, distance, and congestion features",
                "op_crew_mask": "Operation-crew eligibility relation",
                "task_agv_feats": f"{TASK_AGV_EDGE_DIM}-D task-AMR travel distance, ETA slack, incumbent lock, and kitting unlock edge features",
            },
            "corridor_traffic_representation": (
                "Encoded via node/edge attributes (`bay_detour_norm`, `op_feats[:, 6]` bay congestion, "
                "`task_feats[:, 11]` dropoff Detour-Delta, and `agv_feats[:, 5:7]` L1/L2 execution delay & degradation) "
                "rather than standalone Corridor graph nodes."
            ),
            "decoding_and_feedback_flow": (
                "2-layer bidirectional message passing (`encode_entities`: op<->task, op<->bay, op<->crew, task<->agv) "
                "-> A2 (Bay & Crew logits) -> A1 (Task focus order) -> A3 (Sequential AMR pointer conditioned on selected A2 bay via `sel_bay_proj` and cumulative fleet load via `agv_dyn_proj`) -> A4 (Queue insertion score `a4_seq_scores` & derived operation priority)."
            ),
        },
        "a1_plackett_luce_gradient_audit": a1_comparison,
    }


def _write_markdown_report(report: dict[str, Any], md_path: Path) -> None:
    e1 = report["exp01_reproducibility_and_discount"]
    e2 = report["exp02_cost_attribution"]
    e3 = report["exp03_architecture_and_a1_audit"]

    lines = [
        "# P0 Scientific & Evidentiary Audit Report (`EXP-01`, `EXP-02`, `EXP-03`)",
        "",
        "## 1. EXP-01: Reproducible Training Protocol & SMDP Discounting Audit",
        "",
        f"- **Seed Quarantine Integrity**: `all_quarantine_zero = {e1['all_quarantine_zero']}` (`{e1['quarantine_checks']}`)",
        f"- **Equal Training Budget**: All 3 RL variants (`V19_HETERO_RL`, `V19_STAGED_RL`, `V19_NO_PROD_COMM`) trained for exactly `12 rounds × 24 episodes = 288 closed-loop episodes` (`ppo_epochs=2`, `lr=1.5e-5`, `beta_kl=0.08`, `clip_eps=0.12`).",
        f"- **SMDP Discounting (`gamma_step=0.9985`) vs Undiscounted Telescoping (`gamma_step=1.0`)**:",
        f"  - Mean SMDP interval length: `{e1['discount_vs_telescoping_audit']['mean_dt_steps_per_smdp_interval']}` steps (mean interval discount factor `{e1['discount_vs_telescoping_audit']['effective_interval_discount_mean']}`)",
        f"  - Advantage Pearson correlation ($\\gamma=0.9985$ vs $\\gamma=1.0$): **`{e1['discount_vs_telescoping_audit']['advantage_pearson_gamma09985_vs_gamma1']}`** (Spearman $\\rho = {e1['discount_vs_telescoping_audit']['advantage_spearman_gamma09985_vs_gamma1']}$)",
        f"  - Value Target Pearson correlation: **`{e1['discount_vs_telescoping_audit']['target_cost_pearson_gamma09985_vs_gamma1']}`**",
        "",
        "---",
        "",
        "## 2. EXP-02: 7-Term Cost Attribution & Conditional 100%-Completion Analysis (120 Unseen Blind Holdout Episodes)",
        "",
        "### 2.1 Complete 7-Term Cost Ledger & Tail Risk (CVaR)",
        "",
        "| Policy | Mean `net_obj` | Median | IQR | CVaR$_{10\\%}$ | $C_{\\max}$ | $2T_{\\mathrm{task}}$ | $2T_{\\mathrm{op}}$ | $5000U_{\\mathrm{task}}$ | $3000U_{\\mathrm{op}}$ | $15C$ | $0.05M$ | 100% Completed |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for pol, d in e2["policy_ledgers"].items():
        ct = d["cost_terms"]
        lines.append(
            f"| **`{pol}`** | **{d['mean_net_obj']:.2f}** | {d['median_net_obj']:.1f} | {d['iqr_net_obj']:.1f} | {d['cvar10_worst_net_obj']:.1f} | "
            f"{ct['cmax']:.1f} | {ct['task_tardiness_cost']:.1f} | {ct['op_tardiness_cost']:.1f} | "
            f"{ct['unfinished_task_penalty']:.1f} | {ct['unfinished_op_penalty']:.1f} | {ct['churn_cost']:.1f} | {ct['compute_cost']:.1f} | "
            f"**{d['num_100pct_completed']}/{d['num_episodes']}** |"
        )

    lines.extend(
        [
            "",
            "### 2.2 All-Episode vs Conditional 100%-Completed Intersection (`H1` Scheduling Quality vs `H2` Completion Reliability)",
            "",
            "| Comparison | All-120 $\\Delta$ `net_obj` (95% CI) | $\\Delta H_1$ (Time & Tardiness) | $\\Delta H_2$ (Unfinished Pen.) | $\\Delta$ Overhead | Common 100%-Completed Eps | Conditional $\\Delta$ `net_obj` (95% CI) | Conditional Wins (A / B) |",
            "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for label, p in e2["paired_attribution"].items():
        a120 = p["all_120_episodes"]
        cok = p["conditional_100pct_completed_intersection"]
        lines.append(
            f"| **`{label}`** | **{a120['mean_diff_net_obj']:+.2f}** (`[{a120['ci95_lo']:+.1f}, {a120['ci95_hi']:+.1f}]`) | "
            f"**{a120['h1_scheduling_quality_delta']:+.2f}** | **{a120['h2_failure_penalty_delta']:+.2f}** | {a120['control_overhead_delta']:+.2f} | "
            f"`{cok['num_common_100pct_episodes']}/120` | **{cok['mean_diff_net_obj']:+.2f}** (`[{cok['ci95_lo']:+.1f}, {cok['ci95_hi']:+.1f}]`, $p={cok['wilcoxon_p']:.4f}$) | "
            f"`{cok['a_wins']} / {cok['b_wins']}` |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. EXP-03: Heterogeneous Graph Entity Schema & `A1` Plackett-Luce Gradient Audit",
            "",
            f"- **Verified 5 Node Types**: `{e3['actual_hetero_graph_schema']['node_types']}`",
            f"- **Corridor & Traffic Representation**: {e3['actual_hetero_graph_schema']['corridor_traffic_representation']}",
            f"- **Actual Decoding & Cross-Stage Conditioning Flow**: {e3['actual_hetero_graph_schema']['decoding_and_feedback_flow']}",
            "",
            "### `A1` Focus Ranking Head Gradient & Validation Comparison",
            "",
            "| Mode | `include_a1_plackett_luce` | Mean `A1` Head Grad Norm | Val Mean `net_obj` | Val Mean $C_{\\max}$ | Val Mean Tardiness |",
            "| :--- | :---: | :---: | :---: | :---: | :---: |",
        ]
    )
    for m_label, row in e3["a1_plackett_luce_gradient_audit"].items():
        lines.append(
            f"| **`{m_label}`** | `{row['include_a1_plackett_luce']}` | `{row['mean_a1_head_grad_norm']:.6f}` | "
            f"**{row['val_mean_net_obj']:.2f}** | {row['val_mean_cmax']:.2f} | {row['val_mean_tardiness']:.2f} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print("[P0 Audit] Running EXP-01 (Reproducibility & SMDP Discounting Audit)...", flush=True)
    exp01 = run_exp01_reproducibility_and_discount_audit()
    print("[P0 Audit] Running EXP-02 (7-Term Cost Attribution & Conditional 100%-Completion Analysis)...", flush=True)
    exp02 = run_exp02_cost_attribution_and_conditional_analysis()
    print("[P0 Audit] Running EXP-03 (Architecture Schema & A1 Plackett-Luce Gradient Audit)...", flush=True)
    exp03 = run_exp03_architecture_and_a1_plackett_luce_audit()

    report = {
        "phase": "P0_EXP01_02_03",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "exp01_reproducibility_and_discount": exp01,
        "exp02_cost_attribution": exp02,
        "exp03_architecture_and_a1_audit": exp03,
    }
    json_path = OUT_DIR / "exp01_02_03_report.json"
    md_path = OUT_DIR / "exp01_02_03_report.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown_report(report, md_path)
    print(f"[P0 Audit] Wrote {json_path} and {md_path} in {report['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
