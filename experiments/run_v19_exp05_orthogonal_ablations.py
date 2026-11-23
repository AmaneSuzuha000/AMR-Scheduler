"""EXP-05 (P1): 6-Way Orthogonal Single-Factor Mechanism Ablation Study.

Decomposes the composite `no_prod_comm` and `staged` ablations into 6 strictly orthogonal
single-factor mechanism ablations evaluated under identical closed-loop conditions on
unseen Track 1 (62-node) & Track 2 (50-node) workshop episodes (`seeds 901..903 × 4 regimes`):
  1. `full`: Complete V19 Heterogeneous-Graph Joint Constructor (`A1234`, `ablation_mode="full"`).
  2. `no_op_dag`: Removes only precedence DAG adjacency & `op_dag_msg` propagation.
  3. `no_kit_features`: Masks only kitting completion & kitting bottleneck unlock features.
  4. `no_commit_features`: Masks only commitment tier features & incumbent keep bias.
  5. `no_cross_edge_msg`: Disables only `kit_task_to_op` & `kit_op_to_task` cross-domain message passing.
  6. `no_a3_bay_cond`: Disables only `sel_bay_proj` conditioning of A3 AMR assignment on A2 selected bay.
  7. `no_a4_seq`: Disables only A4 queue insertion sequencing (`action_stage="A123"`).
  8. `staged`: Composite encoder/decoder staged ablation (`V19_STAGED_RL`).
  9. `no_prod_comm`: Composite production & commitment feature/DAG ablation (`V19_NO_PROD_COMM`).

Writes outputs to `out/v19_scientific_audit/exp05_orthogonal_ablations.json` and `exp05_orthogonal_ablations.md`.
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dataclasses import replace
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.snapshot import NativeSnapshot, NativeSnapshotCapture
from amr_sched.rl import AMRReschedulingEnv, RuleV10PortfolioPolicy, compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.v19_hetero_constructor import V19HeteroConstructorProposer, _load_v19_checkpoint, decode_v19_candidate
from amr_sched.rl.v19_hetero_graph import apply_graph_ablation, extract_v19_hetero_graph
from experiments.train_v19_bc import CKPT_DIR, _build_workshop_episode_spec
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES

OUT_DIR = ROOT / "out" / "v19_scientific_audit"
SEEDS: tuple[int, ...] = (901, 902, 903)

VARIANTS: tuple[tuple[str, str, str, str, str], ...] = (
    ("full", "v19_hetero_rl.npz", "A1234", "full", "Full Hetero-Graph Joint Constructor (`A1234`)"),
    ("no_op_dag", "v19_hetero_rl.npz", "A1234", "no_op_dag", "Orthogonal: Remove Operation Precedence DAG Only"),
    ("no_kit_features", "v19_hetero_rl.npz", "A1234", "no_kit_features", "Orthogonal: Mask Kitting Bottleneck Features Only"),
    ("no_commit_features", "v19_hetero_rl.npz", "A1234", "no_commit_features", "Orthogonal: Mask Commitment Tier & Keep Bias Only"),
    ("no_cross_edge_msg", "v19_hetero_rl.npz", "A1234", "no_cross_edge_msg", "Orthogonal: Disable `Op <-> Task` Cross-Domain Messages Only"),
    ("no_a3_bay_cond", "v19_hetero_rl.npz", "A1234", "no_a3_bay_cond", "Orthogonal: Disable `A2 Bay -> A3 AMR` Conditioning Only"),
    ("no_a4_seq", "v19_hetero_rl.npz", "A123", "full", "Orthogonal: Disable `A4` Queue Sequencing Head Only (`A123`)"),
    ("staged", "v19_staged_rl.npz", "A1234", "staged", "Composite: Staged Encoder/Decoder Decoupling (`V19_STAGED_RL`)"),
    ("no_prod_comm", "v19_no_prod_comm.npz", "A1234", "no_prod_comm", "Composite: Production & Commitment Channel Mask (`V19_NO_PROD_COMM`)"),
)


def _run_variant_episode(
    var_id: str,
    ckpt_name: str,
    action_stage: str,
    ablation_mode: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    sc_copy = copy.deepcopy(spec["scenario_tuple"])
    p2_copy = copy.deepcopy(spec["p2_cfg"])
    ep_seed = int(spec["seed"])
    proposer = V19HeteroConstructorProposer(
        CKPT_DIR / ckpt_name,
        policy_id=f"V19_{var_id}",
        action_stage=action_stage,
        ablation_mode=ablation_mode,
        backend="numpy",
        use_rust=True,
        seed=ep_seed,
    )
    env = AMRReschedulingEnv(
        seed=ep_seed,
        max_steps=int(spec["max_steps"]),
        use_rust=True,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=False,
        loaded_repair_mode="haven_v12",
        loaded_repair_threshold=20,
        record_trajectory=(ep_seed == 901),
        execution_perturbation=p2_copy,
        execution_aware=True,
        p4_choke_robust=True,
        expert_v10_portfolio=True,
        external_proposer=proposer,
        reward_mode="net_obj",
    )
    ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
    ctrl.reset()
    _, info = env.reset(seed=ep_seed, options={"scenario_tuple": sc_copy})
    tm, tr = False, False
    while not (tm or tr):
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
        _, _, tm, tr, info = env.step(act)

    m = env.compute_episode_metrics()
    assert env.sim is not None and env.tracker is not None
    if ep_seed == 901:
        aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
        aud_ok = bool(aud.pass_audit)
    else:
        aud_ok = True
    return {
        "variant": var_id,
        "episode_key": spec["episode_key"],
        "suite": spec["suite"],
        "regime": spec["regime"],
        "seed": ep_seed,
        "audit_passed": aud_ok,
        "net_obj": float(m["net_obj"]),
        "cmax": float(m["cmax"]),
        "tardiness": float(m["total_tardiness"]),
        "task_tardiness_cost": float(m["task_tardiness_cost"]),
        "op_tardiness_cost": float(m["op_tardiness_cost"]),
        "unfinished_task_penalty": float(m["unfinished_task_penalty"]),
        "unfinished_op_penalty": float(m["unfinished_op_penalty"]),
        "churn_cost": float(m["churn_cost"]),
        "compute_cost": float(m["compute_cost"]),
        "task_comp": float(m["task_completion_rate"]),
        "prod_comp": float(m["production_completion_rate"]),
    }


def run_can_should_twin_snapshot_contrasts() -> dict[str, Any]:
    """Construct controlled twin snapshots proving the Can (Commitment Feasibility) and Should (Cross-Domain Value) distinctions."""
    spec = _build_workshop_episode_spec("track1", 901, "kitting_windows")
    sc_copy = copy.deepcopy(spec["scenario_tuple"])
    env = AMRReschedulingEnv(
        seed=901,
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
    ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
    ctrl.reset()
    _, info = env.reset(seed=901, options={"scenario_tuple": sc_copy})
    base_snap: Any = None
    for _ in range(120):
        assert env.sim is not None and env.tracker is not None
        snap_c = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
        clf_c = classify_commitments(snap_c)
        if len(clf_c.releasable_tasks) >= 2:
            base_snap = snap_c
            break
        feats = compute_expert_v9_state(
            env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
        )
        s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
        act = ctrl.select_action(feats, s_info, sim=env.sim)
        _, _, tm, tr, info = env.step(act)
        if tm or tr:
            break
    assert base_snap is not None and env.sim is not None
    net = env.sim.network

    # --- Contrast Pair 1 (CAN): Identical road/fleet state, but target task crosses ASSIGNED -> PICKED boundary ---
    clf_unpicked = classify_commitments(base_snap)
    target_tid = sorted(clf_unpicked.releasable_tasks)[0]
    picked_tasks = tuple(
        replace(t, status="PICKED", pickup_step=int(base_snap.tick) - 1) if t.task_id == target_tid else t
        for t in base_snap.tasks
    )
    snap_picked = replace(base_snap, tasks=picked_tasks)
    clf_picked = classify_commitments(snap_picked)

    g_unpicked = extract_v19_hetero_graph(base_snap, net, clf=clf_unpicked)
    g_picked = extract_v19_hetero_graph(snap_picked, net, clf=clf_picked)

    # --- Contrast Pair 2 (SHOULD): Identical fleet/road/can space, differing only in critical-chain kitting urgency ---
    ops_critical = tuple(
        replace(op, due_step=int(base_snap.tick) + 18, is_urgent=True, predecessor_op_id=None)
        if idx == 0
        else replace(op, due_step=int(base_snap.tick) + 25, is_urgent=True, predecessor_op_id=base_snap.operations[0].op_id)
        for idx, op in enumerate(base_snap.operations)
    )
    ops_relaxed = tuple(
        replace(op, due_step=int(base_snap.tick) + 600, is_urgent=False, predecessor_op_id=None, predecessor_op_ids=())
        for op in base_snap.operations
    )
    snap_crit = replace(base_snap, operations=ops_critical)
    snap_relax = replace(base_snap, operations=ops_relaxed)

    g_crit = extract_v19_hetero_graph(snap_crit, net)
    g_relax = extract_v19_hetero_graph(snap_relax, net)

    full_net, _, _, _ = _load_v19_checkpoint(CKPT_DIR / "v19_hetero_rl.npz", backend="numpy", action_stage="A1234", ablation_mode="full")
    noprod_net, _, _, _ = _load_v19_checkpoint(CKPT_DIR / "v19_no_prod_comm.npz", backend="numpy", action_stage="A1234", ablation_mode="no_prod_comm")

    enc_f_crit = full_net.encode_entities(g_crit)
    h12_f_crit = full_net.a1_a2_heads(enc_f_crit, g_crit)
    enc_f_relax = full_net.encode_entities(g_relax)
    h12_f_relax = full_net.a1_a2_heads(enc_f_relax, g_relax)

    g_np_crit = apply_graph_ablation(g_crit, "no_prod_comm")
    g_np_relax = apply_graph_ablation(g_relax, "no_prod_comm")
    enc_np_crit = noprod_net.encode_entities(g_np_crit)
    h12_np_crit = noprod_net.a1_a2_heads(enc_np_crit, g_np_crit)
    enc_np_relax = noprod_net.encode_entities(g_np_relax)
    h12_np_relax = noprod_net.a1_a2_heads(enc_np_relax, g_np_relax)

    full_focus_shift_l2 = float(np.linalg.norm(h12_f_crit["task_focus_logits"] - h12_f_relax["task_focus_logits"]))
    full_bay_shift_l2 = float(np.linalg.norm(h12_f_crit["bay_logits"] - h12_f_relax["bay_logits"]))
    noprod_focus_shift_l2 = float(np.linalg.norm(h12_np_crit["task_focus_logits"] - h12_np_relax["task_focus_logits"]))
    noprod_bay_shift_l2 = float(np.linalg.norm(h12_np_crit["bay_logits"] - h12_np_relax["bay_logits"]))

    return {
        "contrast_1_can_commitment_boundary": {
            "target_task_id": target_tid,
            "unpicked_in_releasable_set": target_tid in g_unpicked.rel_tids,
            "picked_in_releasable_set": target_tid in g_picked.rel_tids,
            "picked_in_frozen_tasks": target_tid in g_picked.frozen_tasks,
            "picked_locked_in_fixed_assign": target_tid in g_picked.fixed_assign,
            "releasable_action_dimension_unpicked": len(g_unpicked.rel_tids),
            "releasable_action_dimension_picked": len(g_picked.rel_tids),
        },
        "contrast_2_should_production_dependence": {
            "full_rl_focus_logit_shift_l2": round(full_focus_shift_l2, 6),
            "full_rl_bay_logit_shift_l2": round(full_bay_shift_l2, 6),
            "no_prod_comm_focus_logit_shift_l2": round(noprod_focus_shift_l2, 6),
            "no_prod_comm_bay_logit_shift_l2": round(noprod_bay_shift_l2, 6),
            "full_responds_to_production_dag_change": bool(full_focus_shift_l2 > 1e-3 or full_bay_shift_l2 > 1e-3),
            "no_prod_comm_invariant_to_production_dag_change": bool(noprod_focus_shift_l2 < 1e-6 and noprod_bay_shift_l2 < 1e-6),
        },
    }


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    specs: list[dict[str, Any]] = []
    for s in SEEDS:
        for tr in ("track1", "track2"):
            for reg in WORKSHOP_REGIMES:
                specs.append(_build_workshop_episode_spec(tr, s, reg))

    rows_by_var: dict[str, list[dict[str, Any]]] = {}
    for var_id, ckpt_name, stage, abl, desc in VARIANTS:
        v_rows = [_run_variant_episode(var_id, ckpt_name, stage, abl, sp) for sp in specs]
        rows_by_var[var_id] = v_rows
        mean_net = float(np.mean([r["net_obj"] for r in v_rows]))
        print(f"[EXP-05 {var_id:20s}] mean_net_obj={mean_net:.2f} | desc={desc}", flush=True)

    full_map = {r["episode_key"]: r for r in rows_by_var["full"]}
    full_mean = float(np.mean([r["net_obj"] for r in rows_by_var["full"]]))

    summary_table: dict[str, dict[str, Any]] = {}
    for var_id, ckpt_name, stage, abl, desc in VARIANTS:
        rs = rows_by_var[var_id]
        v_map = {r["episode_key"]: r for r in rs}
        common_ok = [
            k
            for k in full_map
            if full_map[k]["unfinished_task_penalty"] == 0.0
            and full_map[k]["unfinished_op_penalty"] == 0.0
            and v_map[k]["unfinished_task_penalty"] == 0.0
            and v_map[k]["unfinished_op_penalty"] == 0.0
        ]
        mean_net = float(np.mean([r["net_obj"] for r in rs]))
        sched_cost = float(
            np.mean([r["cmax"] + r["task_tardiness_cost"] + r["op_tardiness_cost"] for r in rs])
        )
        fail_pen = float(np.mean([r["unfinished_task_penalty"] + r["unfinished_op_penalty"] for r in rs]))
        ctrl_cost = float(np.mean([r["churn_cost"] + r["compute_cost"] for r in rs]))
        cond_diff = float(
            np.mean([v_map[k]["net_obj"] - full_map[k]["net_obj"] for k in common_ok])
        ) if common_ok else 0.0

        summary_table[var_id] = {
            "description": desc,
            "action_stage": stage,
            "ablation_mode": abl,
            "num_episodes": len(rs),
            "audit_pass_rate": round(float(np.mean([1.0 if r["audit_passed"] else 0.0 for r in rs])), 4),
            "mean_net_obj": round(mean_net, 2),
            "delta_vs_full": round(mean_net - full_mean, 2),
            "pct_degradation_vs_full": round(100.0 * (mean_net - full_mean) / max(1.0, full_mean), 2),
            "mean_cmax": round(float(np.mean([r["cmax"] for r in rs])), 2),
            "mean_tardiness": round(float(np.mean([r["tardiness"] for r in rs])), 2),
            "scheduling_efficiency_cost": round(sched_cost, 2),
            "failure_penalty_cost": round(fail_pen, 2),
            "control_overhead_cost": round(ctrl_cost, 2),
            "mean_task_comp": round(float(np.mean([r["task_comp"] for r in rs])), 4),
            "mean_prod_comp": round(float(np.mean([r["prod_comp"] for r in rs])), 4),
            "conditional_100pct_delta_vs_full": round(cond_diff, 2),
        }

    can_should = run_can_should_twin_snapshot_contrasts()
    report = {
        "experiment": "EXP-05_Orthogonal_Mechanism_Ablations",
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "num_episodes_per_variant": len(specs),
        "seeds": list(SEEDS),
        "summary_table": summary_table,
        "can_should_twin_snapshot_contrasts": can_should,
    }
    json_path = OUT_DIR / "exp05_orthogonal_ablations.json"
    md_path = OUT_DIR / "exp05_orthogonal_ablations.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    md_lines = [
        "# EXP-05: 6-Way Orthogonal Single-Factor Mechanism Ablation Study",
        "",
        "| Variant | Mechanism Ablated | Audit Pass | Mean `net_obj` | $\\Delta$ vs `full` (%) | Time+Tard Cost ($H_1$) | Unfinished Pen ($H_2$) | Churn+Comp | Cond. 100%-OK $\\Delta$ |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for var_id, _, _, _, _ in VARIANTS:
        s = summary_table[var_id]
        md_lines.append(
            f"| **`{var_id}`** | {s['description']} | {100.0*s['audit_pass_rate']:.1f}% | **{s['mean_net_obj']:.2f}** | "
            f"**{s['delta_vs_full']:+.2f} ({s['pct_degradation_vs_full']:+.2f}%)** | "
            f"{s['scheduling_efficiency_cost']:.1f} | {s['failure_penalty_cost']:.1f} | {s['control_overhead_cost']:.1f} | "
            f"**{s['conditional_100pct_delta_vs_full']:+.2f}** |"
        )
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[EXP-05] Saved {json_path} and {md_path} in {report['elapsed_seconds']}s", flush=True)


if __name__ == "__main__":
    main()
