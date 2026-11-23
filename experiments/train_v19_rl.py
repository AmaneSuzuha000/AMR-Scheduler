"""Warm-started KL-anchored SMDP PPO under exact `reward_mode="net_obj"`."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "_scratch"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.top_venue_baselines import _build_external_proposal
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.rl import AMRReschedulingEnv, RuleV10PortfolioPolicy, compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.v19_hetero_constructor import (
    DecodedJointCandidate,
    V19HeteroConstructorNet,
    V19HeteroConstructorProposer,
    export_v19_npz,
    graph_to_torch_tensors,
)
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph, apply_graph_ablation, extract_v19_hetero_graph
from experiments.train_v19_bc import (
    CKPT_DIR,
    DEFAULT_DATASET_PATH,
    FORBIDDEN_BLIND_SEEDS,
    FORBIDDEN_DEV_SEEDS,
    _aggregate_gate_rows,
    _build_4topo_episode_spec,
    _build_workshop_episode_spec,
    _cand_true_obj,
    _collate_v19_batch,
    _compute_bc_batch_loss,
    _derive_op_prio,
    _run_v19_gate_episode,
    _verify_pt_npz_parity,
    assert_seed_quarantine,
)
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES

OUT_G3_DIR = ROOT / "out" / "g3_rl"
BC_WARM_CKPT = CKPT_DIR / "v19_hetero_bc.pt"

RL_4TOPO_SEEDS: tuple[int, ...] = tuple(
    blk * 100 + idx for blk in range(161, 167) for idx in range(1, 17)
)
RL_WORKSHOP_SEEDS: tuple[int, ...] = tuple(range(1001, 1161))
RL_VAL_4TOPO_SEEDS: tuple[int, ...] = (16701, 16702)
RL_VAL_WORKSHOP_SEEDS: tuple[int, ...] = (1161,)

MODEL_SPECS: dict[str, dict[str, str]] = {
    "V19_HETERO_RL": {"ablation_mode": "full", "stem": "v19_hetero_rl"},
    "V19_STAGED_RL": {"ablation_mode": "staged", "stem": "v19_staged_rl"},
    "V19_NO_PROD_COMM": {"ablation_mode": "no_prod_comm", "stem": "v19_no_prod_comm"},
}


def build_rl_train_pool() -> list[dict[str, Any]]:
    """Build full quarantined RL training pool (`16101..16616` & `Track 1/2 1001..1160`)."""
    assert_seed_quarantine(RL_4TOPO_SEEDS, label="RL_4TOPO_SEEDS")
    assert_seed_quarantine(RL_WORKSHOP_SEEDS, label="RL_WORKSHOP_SEEDS")
    pool: list[dict[str, Any]] = []
    for s in RL_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                pool.append(_build_4topo_episode_spec(lay, s, reg))
    for s in RL_WORKSHOP_SEEDS:
        for tr in ("track1", "track2"):
            for reg in WORKSHOP_REGIMES:
                pool.append(_build_workshop_episode_spec(tr, s, reg))
    return pool


def build_rl_early_val_specs() -> list[dict[str, Any]]:
    """Build fast 24-episode early-stop val suite disjoint from `16801..16901` & `1162..1163`."""
    assert_seed_quarantine(RL_VAL_4TOPO_SEEDS, label="RL_VAL_4TOPO_SEEDS")
    assert_seed_quarantine(RL_VAL_WORKSHOP_SEEDS, label="RL_VAL_WORKSHOP_SEEDS")
    specs: list[dict[str, Any]] = []
    for s in RL_VAL_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in ("window_bottleneck", "execution_blockage"):
                specs.append(_build_4topo_episode_spec(lay, s, reg))
    for s in RL_VAL_WORKSHOP_SEEDS:
        for tr in ("track1", "track2"):
            for reg in WORKSHOP_REGIMES:
                specs.append(_build_workshop_episode_spec(tr, s, reg))
    return specs


def _ablate_bc_sample(sample: dict[str, Any], ablation_mode: str) -> dict[str, Any]:
    if ablation_mode in ("full", "staged"):
        return sample
    out = dict(sample)
    ten = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in sample["tensors"].items()}
    if ten["op_x"].shape[1] > 0:
        ten["op_x"][:, :, [0, 1, 2, 3, 4, 5, 9, 10, 11]] = 0.0
    if ten["task_x"].shape[1] > 0:
        ten["task_x"][:, :, [7, 8, 9, 10]] = 0.0
    if ten["task_agv_x"].shape[1] > 0:
        ten["task_agv_x"][:, :, :, [3, 4, 7]] = 0.0
    dag = torch.zeros_like(ten["op_dag_adj"])
    for i in range(int(dag.shape[1])):
        dag[:, i, i, 1] = 1.0
    ten["op_dag_adj"] = dag
    out["tensors"] = ten
    return out


def _calibrate_ablation_bc_prior(
    warm_state: dict[str, torch.Tensor],
    action_stage: str,
    ablation_mode: str,
    *,
    dataset_path: Path,
    device: torch.device,
    epochs: int = 8,
    batch_size: int = 64,
    seed: int = 2026,
) -> dict[str, torch.Tensor]:
    """Calibrate ablated model from BC weights so KL reference matches its own input channels."""
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    raw_samples: list[dict[str, Any]] = payload["samples"]
    assert_seed_quarantine([int(s["seed"]) for s in raw_samples], label="bc_calibration")
    samples = [_ablate_bc_sample(s, ablation_mode) for s in raw_samples]

    net = V19HeteroConstructorNet(
        hidden_dim=64, n_heads=4, n_layers=2, action_stage=action_stage, ablation_mode=ablation_mode
    ).to(device)
    net.load_state_dict(warm_state, strict=True)
    opt = torch.optim.AdamW(net.parameters(), lr=6e-4, weight_decay=1e-4)
    rng = np.random.default_rng(seed + 77)

    net.train()
    for ep in range(1, epochs + 1):
        order_idx = rng.permutation(len(samples))
        losses: list[float] = []
        for start_i in range(0, len(order_idx), batch_size):
            chunk = [samples[int(j)] for j in order_idx[start_i : start_i + batch_size]]
            batch = _collate_v19_batch(chunk, device)
            opt.zero_grad(set_to_none=True)
            loss = _compute_bc_batch_loss(net, batch, action_stage=action_stage)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.5)
            opt.step()
            losses.append(float(loss.detach().item()))
        if ep == 1 or ep == epochs:
            print(f"[RL calibrate {ablation_mode}] ep {ep:02d}/{epochs} bc_loss={float(np.mean(losses)):.4f}", flush=True)

    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def _sample_rollout_action(
    actor_cpu: V19HeteroConstructorNet,
    graph: V19HeteroGraph,
    snapshot: Any,
    *,
    np_rng: np.random.Generator,
    greedy: bool,
    tau: float = 0.70,
    seq_sigma: float = 0.35,
) -> tuple[DecodedJointCandidate, dict[str, Any]]:
    """Sample joint `(A1 -> A2 -> A3 [-> A4])` action plus exact PPO step tensors."""
    N_op = len(graph.op_ids)
    N_t = len(graph.rel_tids)
    N_a = len(graph.agv_ids)
    eff_stage = actor_cpu.action_stage
    ten = graph_to_torch_tensors(graph, device="cpu")

    with torch.inference_mode():
        enc = actor_cpu.encode_entities(
            ten["op_x"], ten["task_x"], ten["agv_x"], ten["bay_x"], ten["crew_x"], ten["task_to_op"], ten["op_dag_adj"]
        )
        h12 = actor_cpu.a1_a2_heads(enc, ten["op_bay_x"], ten["op_bay_mask"], ten["op_crew_mask"])
    focus_np = h12["task_focus_logits"][0].numpy()
    bay_logits_np = h12["bay_logits"][0].numpy()
    crew_logits_np = h12["crew_logits"][0].numpy()
    val_scalar = float(h12["value"][0].item())

    def _pick(lg_1d: np.ndarray) -> int:
        if greedy:
            return int(np.argmax(lg_1d))
        sc = np.asarray(lg_1d, dtype=np.float64) / max(0.05, float(tau))
        sc = sc - np.max(sc)
        ex = np.exp(sc)
        probs = ex / max(1e-12, float(ex.sum()))
        return int(np_rng.choice(len(lg_1d), p=probs))

    bay_indices = np.zeros((N_op,), dtype=np.int64)
    crew_indices = np.zeros((N_op,), dtype=np.int64)
    bays_dict: dict[str, str] = {}
    crews_dict: dict[str, int] = {}
    for oi, oid in enumerate(graph.op_ids):
        if oid in graph.frozen_ops:
            bi = int(graph.op_incumbent_bay_idx[oi])
        else:
            bi = _pick(bay_logits_np[oi])
        bay_indices[oi] = bi
        bays_dict[oid] = graph.bay_ids[bi]

        if graph.op_crew_mask[oi].sum() <= 1:
            ci = int(graph.op_incumbent_crew_idx[oi])
        else:
            ci = _pick(crew_logits_np[oi])
        crew_indices[oi] = ci
        crews_dict[oid] = int(graph.crew_ids[ci])

    assign_dict = dict(graph.fixed_assign)
    agv_indices = np.zeros((N_t,), dtype=np.int64)
    focus_order = sorted(range(N_t), key=lambda i: (-float(focus_np[i]), i))
    if N_t > 0:
        added_load = torch.zeros((1, N_a), dtype=torch.float32)
        added_dur = torch.zeros((1, N_a), dtype=torch.float32)
        for ti in focus_order:
            oi = int(graph.task_to_op[ti])
            sel_bi = int(bay_indices[oi]) if oi >= 0 else int(graph.task_incumbent_bay_idx[ti])
            b_tr = float(graph.bay_xy_travel_from_pickup[ti, sel_bi])
            with torch.inference_mode():
                lg_t = actor_cpu.a3_step_logits(
                    enc,
                    ti,
                    torch.tensor([sel_bi], dtype=torch.long),
                    ten["task_agv_x"][:, ti, :, :],
                    ten["task_agv_mask"][:, ti, :],
                    torch.tensor([b_tr], dtype=torch.float32),
                    torch.tensor([float(graph.bay_detour_norm[sel_bi])], dtype=torch.float32),
                    added_load,
                    added_dur,
                )
            ai = _pick(lg_t[0].numpy())
            agv_indices[ti] = ai
            assign_dict[graph.rel_tids[ti]] = graph.agv_ids[ai]
            added_load[0, ai] += 1.0
            added_dur[0, ai] += b_tr + float(graph.task_agv_feats[ti, ai, 0])

    if N_t > 0 and eff_stage == "A1234":
        with torch.inference_mode():
            seq_mu = actor_cpu.a4_seq_scores(enc, torch.from_numpy(agv_indices).unsqueeze(0))[0].numpy()
        seq_scores_np = seq_mu if greedy else (seq_mu + np_rng.normal(0.0, seq_sigma, size=seq_mu.shape).astype(np.float32))
        by_agv_rel: dict[str, list[tuple[float, str]]] = {}
        for ti, tid in enumerate(graph.rel_tids):
            by_agv_rel.setdefault(assign_dict[tid], []).append((-float(seq_scores_np[ti]), tid))
        task_sequences: Optional[dict[str, list[str]]] = {}
        snap_by_id = {t.task_id: t for t in snapshot.tasks}
        rob_by_id = {r.robot_id: r for r in snapshot.robots}
        for aid, pairs in by_agv_rel.items():
            pairs.sort()
            ordered = [tid for _, tid in pairs]
            rob = rob_by_id.get(aid)
            if rob and rob.cur_task:
                cur_t = snap_by_id.get(rob.cur_task)
                if cur_t is not None and cur_t.assigned_agv == aid:
                    is_frozen_head = rob.cur_task in graph.frozen_tasks
                    is_active_head = (
                        rob.cur_task in ordered
                        and rob.cur_task not in graph.clf.disturbed_tasks
                        and not (ordered and snap_by_id.get(ordered[0]) is not None and snap_by_id[ordered[0]].is_urgent and not cur_t.is_urgent)
                    )
                    if is_frozen_head and cur_t.task_id not in ordered:
                        ordered = [cur_t.task_id] + ordered
                    elif is_active_head and ordered[0] != rob.cur_task:
                        ordered = [rob.cur_task] + [tid for tid in ordered if tid != rob.cur_task]
            if ordered:
                task_sequences[aid] = ordered
        op_best_rank: dict[str, float] = {}
        for ti, tid in enumerate(graph.rel_tids):
            oi = int(graph.task_to_op[ti])
            if oi >= 0:
                oid = graph.op_ids[oi]
                op_best_rank[oid] = min(op_best_rank.get(oid, 1e9), -float(seq_scores_np[ti]))
        op_prio = {
            op.op_id: int(round(max(snapshot.tick, op.due_step - op.proc_steps) + 8.0 * op_best_rank.get(op.op_id, 0.0)))
            for op in snapshot.operations
        }
    else:
        seq_scores_np = np.zeros((N_t,), dtype=np.float32)
        task_sequences = None
        op_prio = {
            op.op_id: max(int(snapshot.tick), int(op.due_step) - int(op.proc_steps))
            for op in snapshot.operations
        }

    cand = DecodedJointCandidate(
        assign=assign_dict,
        bays=bays_dict,
        crews=crews_dict,
        op_prio=op_prio,
        task_sequences=task_sequences,
        bay_indices=bay_indices,
        crew_indices=crew_indices,
        agv_indices=agv_indices,
        seq_scores=seq_scores_np,
    )
    meta = {
        "tensors": {k: v for k, v in ten.items()},
        "op_bay_mask": torch.from_numpy(graph.op_bay_mask),
        "op_crew_mask": torch.from_numpy(graph.op_crew_mask),
        "task_agv_mask": torch.from_numpy(graph.task_agv_mask),
        "chosen_bays": torch.from_numpy(bay_indices.copy()),
        "chosen_crews": torch.from_numpy(crew_indices.copy()),
        "chosen_agvs": torch.from_numpy(agv_indices.copy()),
        "focus_order": torch.tensor(focus_order, dtype=torch.long),
        "chosen_seq": torch.from_numpy(seq_scores_np.copy()),
        "old_value": float(val_scalar),
    }
    return cand, meta


def _collect_smdp_rollouts(
    actor_cpu: V19HeteroConstructorNet,
    specs: list[dict[str, Any]],
    *,
    policy_label: str,
    ablation_mode: str,
    rollout_seed: int,
    gamma_step: float = 0.9985,
    gae_lambda: float = 0.95,
    seq_sigma: float = 0.35,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    """Run closed-loop `reward_mode='net_obj'` episodes and return GAE SMDP transitions."""
    transitions: list[dict[str, Any]] = []
    ep_net_objs: list[float] = []
    ep_task_comps: list[float] = []
    ep_prod_comps: list[float] = []
    ep_churns: list[float] = []
    actor_cpu.eval()

    for idx_ep, spec in enumerate(specs):
        sc_copy = copy.deepcopy(spec["scenario_tuple"])
        p2_copy = copy.deepcopy(spec["p2_cfg"])
        ep_seed = int(spec["seed"])
        np_rng = np.random.default_rng(rollout_seed * 131 + ep_seed + idx_ep)
        eff_stage = actor_cpu.action_stage

        # Holder tracks active SMDP interval opened on the latest decision
        ep_recs: list[dict[str, Any]] = []
        active_interval = {"rew": 0.0, "steps": 0, "has_rec": False}

        def _rollout_proposer(snapshot: Any, network: Any, budget_ms: float = 50.0) -> Any:
            t0 = time.perf_counter()
            clf = classify_commitments(snapshot)
            raw_g = extract_v19_hetero_graph(snapshot, network, clf=clf)
            g = apply_graph_ablation(raw_g, ablation_mode)
            ev = CVCREvaluator(network, snapshot, alpha_makespan=3.0, lambda_change=15.0, use_rust=True, mode="B2")
            base_prio = {
                op.op_id: max(int(snapshot.tick), int(op.due_step) - int(op.proc_steps))
                for op in snapshot.operations
            }
            keep_eval = ev.evaluate(
                g.base_assign, op_priority=base_prio, op_bay_overrides=g.base_bays, op_crew_overrides=g.base_crews
            )
            keep_true = _cand_true_obj(keep_eval, g.base_bays, g.base_crews, g, snapshot, None)

            best_assign = dict(g.base_assign)
            best_bays = dict(g.base_bays)
            best_crews = dict(g.base_crews)
            best_prio = dict(base_prio)
            best_seq = None
            best_eval = keep_eval

            if len(g.rel_tids) > 0 and len(g.agv_ids) >= 1:
                use_greedy = bool(idx_ep % 4 == 0)
                cand, rec = _sample_rollout_action(
                    actor_cpu, g, snapshot, np_rng=np_rng, greedy=use_greedy, tau=0.45, seq_sigma=seq_sigma
                )
                ev_c = ev.evaluate(
                    cand.assign,
                    op_priority=cand.op_prio,
                    op_bay_overrides=cand.bays,
                    op_crew_overrides=cand.crews,
                    task_order_override=cand.task_sequences,
                )
                c_true = _cand_true_obj(ev_c, cand.bays, cand.crews, g, snapshot, cand.task_sequences)
                if ev_c.valid and (not keep_eval.valid or c_true <= keep_true + 10.0):
                    best_eval = ev_c
                    best_assign = cand.assign
                    best_bays = cand.bays
                    best_crews = cand.crews
                    best_prio = cand.op_prio
                    best_seq = cand.task_sequences
                else:
                    # Keep incumbent: still bind chosen indices cleanly onto incumbent choices
                    agv_map = {aid: i for i, aid in enumerate(g.agv_ids)}
                    inc_agvs = np.asarray([agv_map[g.base_assign[tid]] for tid in g.rel_tids], dtype=np.int64)
                    rec["chosen_bays"] = torch.from_numpy(g.op_incumbent_bay_idx.copy())
                    rec["chosen_crews"] = torch.from_numpy(g.op_incumbent_crew_idx.copy())
                    rec["chosen_agvs"] = torch.from_numpy(inc_agvs)

                # Close previous interval when a new decision opens
                if active_interval["has_rec"] and ep_recs:
                    ep_recs[-1]["smdp_reward"] = active_interval["rew"]
                    ep_recs[-1]["dt_steps"] = max(1, active_interval["steps"])
                active_interval["rew"] = 0.0
                active_interval["steps"] = 0
                active_interval["has_rec"] = True
                rec["tick"] = int(snapshot.tick)
                rec["suite"] = str(spec["suite"])
                ep_recs.append(rec)

            return _build_external_proposal(
                policy_id=policy_label,
                snapshot=snapshot,
                clf=clf,
                best_assign=best_assign,
                best_bays=best_bays,
                best_crews=best_crews,
                best_eval=best_eval,
                op_prio=best_prio,
                eval_cnt=ev.calls,
                t0=t0,
                budget_ms=budget_ms,
                task_sequences=best_seq if eff_stage == "A1234" else None,
            )

        with virtual_clock_scope(poll_ms=0.002):
            env = AMRReschedulingEnv(
                seed=ep_seed,
                max_steps=int(spec["max_steps"]),
                use_rust=True,
                allow_kitting_shortcut=False,
                pure_physical_reward=True,
                real_compute_window_coupling=False,
                loaded_repair_mode="haven_v12",
                loaded_repair_threshold=20,
                record_trajectory=False,
                execution_perturbation=p2_copy,
                execution_aware=True,
                p4_choke_robust=True,
                expert_v10_portfolio=True,
                external_proposer=_rollout_proposer,
                reward_mode="net_obj",
            )
            ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
            ctrl.reset()
            _, info = env.reset(seed=ep_seed, options={"scenario_tuple": sc_copy})
            tm = False
            tr = False
            while not (tm or tr):
                assert env.sim is not None and env.tracker is not None
                feats = compute_expert_v9_state(
                    env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
                    recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
                )
                s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
                act = ctrl.select_action(feats, s_info, sim=env.sim)
                _, step_r, tm, tr, info = env.step(act)
                if active_interval["has_rec"]:
                    active_interval["rew"] += float(step_r) / 1000.0
                    active_interval["steps"] += 1

        if active_interval["has_rec"] and ep_recs:
            ep_recs[-1]["smdp_reward"] = active_interval["rew"]
            ep_recs[-1]["dt_steps"] = max(1, active_interval["steps"])

        m = env.compute_episode_metrics()
        ep_net_objs.append(float(m["net_obj"]))
        ep_task_comps.append(float(m["task_completion_rate"]))
        ep_prod_comps.append(float(m["production_completion_rate"]))
        ep_churns.append(float(m["total_churn"]))

        # Compute SMDP time-scaled GAE (value outputs remaining +cost / 1000.0 -> ret_val = -cost)
        adv_next = 0.0
        for k_rev in range(len(ep_recs) - 1, -1, -1):
            cur = ep_recs[k_rev]
            v_cur = -float(cur["old_value"])
            if k_rev == len(ep_recs) - 1:
                v_next = 0.0
            else:
                v_next = -float(ep_recs[k_rev + 1]["old_value"])
            g_dt = float(gamma_step) ** max(1, int(cur["dt_steps"]))
            delta = float(cur["smdp_reward"]) + g_dt * v_next - v_cur
            adv = delta + g_dt * float(gae_lambda) * adv_next
            adv_next = adv
            ret = adv + v_cur
            cur["adv"] = float(adv)
            cur["target_cost"] = float(max(0.0, min(40.0, -ret)))
        transitions.extend(ep_recs)

    rollout_stats = {
        "episodes": float(len(specs)),
        "transitions": float(len(transitions)),
        "mean_net_obj": float(np.mean(ep_net_objs)) if ep_net_objs else 0.0,
        "mean_task_comp": float(np.mean(ep_task_comps)) if ep_task_comps else 0.0,
        "mean_prod_comp": float(np.mean(ep_prod_comps)) if ep_prod_comps else 0.0,
        "mean_churn": float(np.mean(ep_churns)) if ep_churns else 0.0,
    }
    return transitions, rollout_stats


def _collate_rl_batch(chunk: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    """Collate SMDP RL rollout transitions cleanly onto CUDA/CPU."""
    B = len(chunk)
    Op_m = max(int(s["tensors"]["op_x"].shape[1]) for s in chunk)
    T_m = max(int(s["tensors"]["task_x"].shape[1]) for s in chunk)
    A_m = max(int(s["tensors"]["agv_x"].shape[1]) for s in chunk)
    B_m = max(int(s["tensors"]["bay_x"].shape[1]) for s in chunk)
    C_m = max(int(s["tensors"]["crew_x"].shape[1]) for s in chunk)
    t0_ref = chunk[0]["tensors"]

    op_x = torch.zeros((B, Op_m, int(t0_ref["op_x"].shape[-1])), dtype=torch.float32)
    task_x = torch.zeros((B, T_m, int(t0_ref["task_x"].shape[-1])), dtype=torch.float32)
    agv_x = torch.zeros((B, A_m, int(t0_ref["agv_x"].shape[-1])), dtype=torch.float32)
    bay_x = torch.zeros((B, B_m, int(t0_ref["bay_x"].shape[-1])), dtype=torch.float32)
    crew_x = torch.zeros((B, C_m, int(t0_ref["crew_x"].shape[-1])), dtype=torch.float32)
    op_valid = torch.zeros((B, Op_m), dtype=torch.bool)
    task_valid = torch.zeros((B, T_m), dtype=torch.bool)
    agv_valid = torch.zeros((B, A_m), dtype=torch.bool)
    bay_valid = torch.zeros((B, B_m), dtype=torch.bool)
    crew_valid = torch.zeros((B, C_m), dtype=torch.bool)
    task_to_op = torch.full((B, T_m), -1, dtype=torch.long)
    op_dag_adj = torch.zeros((B, Op_m, Op_m, int(t0_ref["op_dag_adj"].shape[-1])), dtype=torch.float32)
    op_bay_x = torch.zeros((B, Op_m, B_m, int(t0_ref["op_bay_x"].shape[-1])), dtype=torch.float32)
    op_bay_mask = torch.zeros((B, Op_m, B_m), dtype=torch.bool)
    op_crew_mask = torch.zeros((B, Op_m, C_m), dtype=torch.bool)
    task_agv_x = torch.zeros((B, T_m, A_m, int(t0_ref["task_agv_x"].shape[-1])), dtype=torch.float32)
    task_agv_mask = torch.zeros((B, T_m, A_m), dtype=torch.bool)
    bay_travel = torch.zeros((B, T_m, B_m), dtype=torch.float32)
    bay_dd_norm = torch.zeros((B, B_m), dtype=torch.float32)
    task_inc_bay = torch.zeros((B, T_m), dtype=torch.long)

    chosen_bays = torch.zeros((B, Op_m), dtype=torch.long)
    chosen_crews = torch.zeros((B, Op_m), dtype=torch.long)
    chosen_agvs = torch.zeros((B, T_m), dtype=torch.long)
    focus_order = torch.zeros((B, T_m), dtype=torch.long)
    chosen_seq = torch.zeros((B, T_m), dtype=torch.float32)
    n_tasks = torch.zeros((B,), dtype=torch.long)
    adv = torch.zeros((B,), dtype=torch.float32)
    target_cost = torch.zeros((B,), dtype=torch.float32)

    for b_i, s in enumerate(chunk):
        ten = s["tensors"]
        no, nt, na, nb, nc = (
            int(ten["op_x"].shape[1]),
            int(ten["task_x"].shape[1]),
            int(ten["agv_x"].shape[1]),
            int(ten["bay_x"].shape[1]),
            int(ten["crew_x"].shape[1]),
        )
        op_x[b_i, :no] = ten["op_x"][0]
        task_x[b_i, :nt] = ten["task_x"][0]
        agv_x[b_i, :na] = ten["agv_x"][0]
        bay_x[b_i, :nb] = ten["bay_x"][0]
        crew_x[b_i, :nc] = ten["crew_x"][0]
        op_valid[b_i, :no] = True
        task_valid[b_i, :nt] = True
        agv_valid[b_i, :na] = True
        bay_valid[b_i, :nb] = True
        crew_valid[b_i, :nc] = True
        task_to_op[b_i, :nt] = ten["task_to_op"][0]
        op_dag_adj[b_i, :no, :no] = ten["op_dag_adj"][0]
        op_bay_x[b_i, :no, :nb] = ten["op_bay_x"][0]
        op_bay_mask[b_i, :no, :nb] = s["op_bay_mask"]
        op_crew_mask[b_i, :no, :nc] = s["op_crew_mask"]
        task_agv_x[b_i, :nt, :na] = ten["task_agv_x"][0]
        task_agv_mask[b_i, :nt, :na] = s["task_agv_mask"]
        bay_travel[b_i, :nt, :nb] = ten["bay_travel"][0]
        bay_dd_norm[b_i, :nb] = ten["bay_dd_norm"][0]
        task_inc_bay[b_i, :nt] = ten["task_inc_bay"][0]

        chosen_bays[b_i, :no] = s["chosen_bays"]
        chosen_crews[b_i, :no] = s["chosen_crews"]
        chosen_agvs[b_i, :nt] = s["chosen_agvs"]
        focus_order[b_i, :nt] = s["focus_order"]
        chosen_seq[b_i, :nt] = s["chosen_seq"]
        n_tasks[b_i] = nt
        adv[b_i] = float(s["adv"])
        target_cost[b_i] = float(s["target_cost"])

    return {
        "op_x": op_x.to(device),
        "task_x": task_x.to(device),
        "agv_x": agv_x.to(device),
        "bay_x": bay_x.to(device),
        "crew_x": crew_x.to(device),
        "op_valid": op_valid.to(device),
        "task_valid": task_valid.to(device),
        "agv_valid": agv_valid.to(device),
        "bay_valid": bay_valid.to(device),
        "crew_valid": crew_valid.to(device),
        "task_to_op": task_to_op.to(device),
        "op_dag_adj": op_dag_adj.to(device),
        "op_bay_x": op_bay_x.to(device),
        "op_bay_mask": op_bay_mask.to(device),
        "op_crew_mask": op_crew_mask.to(device),
        "task_agv_x": task_agv_x.to(device),
        "task_agv_mask": task_agv_mask.to(device),
        "bay_travel": bay_travel.to(device),
        "bay_dd_norm": bay_dd_norm.to(device),
        "task_inc_bay": task_inc_bay.to(device),
        "chosen_bays": chosen_bays.to(device),
        "chosen_crews": chosen_crews.to(device),
        "chosen_agvs": chosen_agvs.to(device),
        "focus_order": focus_order.to(device),
        "chosen_seq": chosen_seq.to(device),
        "n_tasks": n_tasks.to(device),
        "adv": adv.to(device),
        "target_cost": target_cost.to(device),
    }


def _evaluate_val_subset(
    actor_state: dict[str, torch.Tensor],
    action_stage: str,
    ablation_mode: str,
    val_specs: list[dict[str, Any]],
    tmp_ckpt_path: Path,
) -> dict[str, Any]:
    """Export temporary `.npz` and evaluate deterministic online policy on unseen val specs."""
    tmp_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_net = V19HeteroConstructorNet(
        hidden_dim=64, n_heads=4, n_layers=2, action_stage=action_stage, ablation_mode=ablation_mode
    )
    cpu_net.load_state_dict(actor_state, strict=True)
    torch.save({"model": actor_state, "config": cpu_net.config_dict()}, tmp_ckpt_path)
    tmp_npz = tmp_ckpt_path.with_suffix(".npz")
    export_v19_npz(tmp_ckpt_path, tmp_npz)
    rows = [_run_v19_gate_episode(tmp_npz, f"V19_VAL_{ablation_mode}", sp) for sp in val_specs]
    return _aggregate_gate_rows(rows)


def train_single_g3_model(
    model_name: str,
    *,
    rounds: int = 12,
    eps_per_round: int = 24,
    ppo_epochs: int = 2,
    batch_size: int = 64,
    lr: float = 1.5e-5,
    beta_kl: float = 0.08,
    clip_eps: float = 0.12,
    seq_sigma: float = 0.20,
    seed: int = 2026,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> dict[str, Any]:
    """Train one RL policy via closed-loop SMDP PPO anchored to its calibrated BC prior."""
    if not BC_WARM_CKPT.is_file():
        raise FileNotFoundError(f"BC promoted checkpoint not found: {BC_WARM_CKPT}")
    bc_ck = torch.load(BC_WARM_CKPT, map_location="cpu", weights_only=False)
    action_stage = str(bc_ck["config"]["action_stage"])
    ablation_mode = MODEL_SPECS[model_name]["ablation_mode"]
    stem = MODEL_SPECS[model_name]["stem"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"[RL RL {model_name}] stage={action_stage} ablation={ablation_mode} device={device} rounds={rounds} eps/rnd={eps_per_round}",
        flush=True,
    )

    warm_state: dict[str, torch.Tensor] = {k: v.cpu().clone() for k, v in bc_ck["model"].items()}
    if ablation_mode != "full":
        warm_state = _calibrate_ablation_bc_prior(
            warm_state,
            action_stage,
            ablation_mode,
            dataset_path=dataset_path,
            device=device,
            epochs=8,
            batch_size=64,
            seed=seed,
        )

    bc_ref = V19HeteroConstructorNet(
        hidden_dim=64, n_heads=4, n_layers=2, action_stage=action_stage, ablation_mode=ablation_mode
    ).to(device)
    bc_ref.load_state_dict(warm_state, strict=True)
    bc_ref.eval()
    for p in bc_ref.parameters():
        p.requires_grad_(False)

    rl_net = V19HeteroConstructorNet(
        hidden_dim=64, n_heads=4, n_layers=2, action_stage=action_stage, ablation_mode=ablation_mode
    ).to(device)
    rl_net.load_state_dict(warm_state, strict=True)

    actor_cpu = V19HeteroConstructorNet(
        hidden_dim=64, n_heads=4, n_layers=2, action_stage=action_stage, ablation_mode=ablation_mode
    )
    opt = torch.optim.AdamW(rl_net.parameters(), lr=lr, weight_decay=1e-4)

    train_pool = build_rl_train_pool()
    topo_pool = [sp for sp in train_pool if sp["suite"] == "topo4"]
    work_pool = [sp for sp in train_pool if sp["suite"] != "topo4"]
    val_specs = build_rl_early_val_specs()
    tmp_val_pt = OUT_G3_DIR / f"_tmp_{stem}.pt"

    # Initial round-0 val baseline
    best_state = {k: v.cpu().clone() for k, v in warm_state.items()}
    val0 = _evaluate_val_subset(best_state, action_stage, ablation_mode, val_specs, tmp_val_pt)
    best_val_key = (-val0["audit_pass_rate"], -val0["mean_task_comp"], -val0["mean_prod_comp"], val0["mean_net_obj"])
    best_round = 0
    best_val_summary = val0
    print(f"[RL RL {model_name}] round 00 val={val0}", flush=True)

    rng = np.random.default_rng(seed + (101 if ablation_mode == "full" else (202 if ablation_mode == "staged" else 303)))

    def _sample_balanced_specs(n_total: int) -> list[dict[str, Any]]:
        n_w = n_total // 2
        n_t = n_total - n_w
        idx_t = rng.choice(len(topo_pool), size=min(n_t, len(topo_pool)), replace=False)
        idx_w = rng.choice(len(work_pool), size=min(n_w, len(work_pool)), replace=False)
        out = [topo_pool[int(i)] for i in idx_t] + [work_pool[int(i)] for i in idx_w]
        rng.shuffle(out)
        return out

    # Calibrate critic head (val_w1, val_w2) onto exact SMDP remaining cost-to-go while freezing actor
    actor_cpu.load_state_dict({k: v.detach().cpu() for k, v in rl_net.state_dict().items()}, strict=True)
    warm_trans, _ = _collect_smdp_rollouts(
        actor_cpu,
        _sample_balanced_specs(24),
        policy_label=model_name,
        ablation_mode=ablation_mode,
        rollout_seed=seed,
        seq_sigma=seq_sigma,
    )
    if warm_trans:
        val_opt = torch.optim.AdamW(list(rl_net.val_w1.parameters()) + list(rl_net.val_w2.parameters()), lr=8e-4, weight_decay=1e-4)
        rl_net.train()
        for _ in range(8):
            o_w = rng.permutation(len(warm_trans))
            for st_i in range(0, len(o_w), batch_size):
                b_ten = _collate_rl_batch([warm_trans[int(j)] for j in o_w[st_i : st_i + batch_size]], device)
                out_w = rl_net.evaluate_joint_actions_batch(
                    b_ten,
                    chosen_bays=b_ten["chosen_bays"],
                    chosen_crews=b_ten["chosen_crews"],
                    chosen_agvs=b_ten["chosen_agvs"],
                    focus_order=b_ten["focus_order"],
                    chosen_seq=b_ten["chosen_seq"],
                    seq_sigma=seq_sigma,
                )
                v_loss = F.smooth_l1_loss(out_w["value"], b_ten["target_cost"])
                val_opt.zero_grad(set_to_none=True)
                v_loss.backward()
                torch.nn.utils.clip_grad_norm_(rl_net.parameters(), 1.0)
                val_opt.step()

    history: list[dict[str, Any]] = [{"round": 0, "val": val0}]
    t0 = time.perf_counter()

    for rnd in range(1, rounds + 1):
        round_specs = _sample_balanced_specs(eps_per_round)
        actor_cpu.load_state_dict({k: v.detach().cpu() for k, v in rl_net.state_dict().items()}, strict=True)

        transitions, roll_stats = _collect_smdp_rollouts(
            actor_cpu,
            round_specs,
            policy_label=model_name,
            ablation_mode=ablation_mode,
            rollout_seed=seed + rnd,
            seq_sigma=seq_sigma,
        )
        if not transitions:
            continue

        # Normalize advantages separately per suite (topo4 vs track1 vs track2) to avoid cross-scale dominance
        by_suite: dict[str, list[float]] = {}
        for t_rec in transitions:
            by_suite.setdefault(str(t_rec.get("suite", "topo4")), []).append(float(t_rec["adv"]))
        suite_stats = {
            k: (float(np.mean(v)), float(np.std(v)) + 1e-6)
            for k, v in by_suite.items()
        }
        for t_rec in transitions:
            mu_s, sd_s = suite_stats[str(t_rec.get("suite", "topo4"))]
            t_rec["adv"] = float(np.clip((float(t_rec["adv"]) - mu_s) / sd_s, -3.5, 3.5))

        # Pre-collate minibatches on CUDA + compute old_logp & bc_ref_probs ONCE per batch
        batches: list[dict[str, Any]] = []
        rl_net.eval()
        order_idx = rng.permutation(len(transitions))
        for start_i in range(0, len(order_idx), batch_size):
            chunk = [transitions[int(j)] for j in order_idx[start_i : start_i + batch_size]]
            b_ten = _collate_rl_batch(chunk, device)
            with torch.inference_mode():
                ref_out = bc_ref.evaluate_joint_actions_batch(
                    b_ten,
                    chosen_bays=b_ten["chosen_bays"],
                    chosen_crews=b_ten["chosen_crews"],
                    chosen_agvs=b_ten["chosen_agvs"],
                    focus_order=b_ten["focus_order"],
                    chosen_seq=b_ten["chosen_seq"],
                    seq_sigma=seq_sigma,
                    return_probs=True,
                )
                old_out = rl_net.evaluate_joint_actions_batch(
                    b_ten,
                    chosen_bays=b_ten["chosen_bays"],
                    chosen_crews=b_ten["chosen_crews"],
                    chosen_agvs=b_ten["chosen_agvs"],
                    focus_order=b_ten["focus_order"],
                    chosen_seq=b_ten["chosen_seq"],
                    ref_probs=ref_out["probs"],
                    seq_sigma=seq_sigma,
                    return_probs=False,
                )
            batches.append({
                "batch": b_ten,
                "ref_probs": ref_out["probs"],
                "old_logp": old_out["mean_logp"].detach(),
            })

        # KL-anchored Clipped SMDP PPO updates on RTX 5060 CUDA
        rl_net.train()
        ppo_losses: list[float] = []
        kls: list[float] = []
        for _ in range(ppo_epochs):
            ep_kls: list[float] = []
            for mb_idx in rng.permutation(len(batches)):
                mb = batches[int(mb_idx)]
                b_ten = mb["batch"]
                out = rl_net.evaluate_joint_actions_batch(
                    b_ten,
                    chosen_bays=b_ten["chosen_bays"],
                    chosen_crews=b_ten["chosen_crews"],
                    chosen_agvs=b_ten["chosen_agvs"],
                    focus_order=b_ten["focus_order"],
                    chosen_seq=b_ten["chosen_seq"],
                    ref_probs=mb["ref_probs"],
                    seq_sigma=seq_sigma,
                    return_probs=False,
                )
                ratio = torch.exp(torch.clamp(out["mean_logp"] - mb["old_logp"], -2.0, 2.0))
                adv_t = b_ten["adv"]
                surr1 = ratio * adv_t
                surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv_t
                mask_d = out["has_decisions"].float()
                pg_loss = -(torch.min(surr1, surr2) * mask_d).sum() / torch.clamp(mask_d.sum(), min=1.0)
                val_loss = F.smooth_l1_loss(out["value"], b_ten["target_cost"])
                kl_loss = (out["mean_kl"] * mask_d).sum() / torch.clamp(mask_d.sum(), min=1.0)
                ent_bonus = (out["mean_entropy"] * mask_d).sum() / torch.clamp(mask_d.sum(), min=1.0)

                total_loss = pg_loss + 0.5 * val_loss + beta_kl * kl_loss - 0.002 * ent_bonus
                opt.zero_grad(set_to_none=True)
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(rl_net.parameters(), 1.0)
                opt.step()
                ppo_losses.append(float(total_loss.detach().item()))
                kl_v = float(kl_loss.detach().item())
                kls.append(kl_v)
                ep_kls.append(kl_v)
            if ep_kls and float(np.mean(ep_kls)) > 0.045:
                break

        log_rec: dict[str, Any] = {
            "round": rnd,
            "roll": {k: round(v, 4) for k, v in roll_stats.items()},
            "ppo_loss": round(float(np.mean(ppo_losses)) if ppo_losses else 0.0, 4),
            "mean_kl": round(float(np.mean(kls)) if kls else 0.0, 5),
        }
        msg = (
            f"[RL RL {model_name}] rnd {rnd:02d}/{rounds} ({time.perf_counter() - t0:.1f}s) "
            f"roll_obj={roll_stats['mean_net_obj']:.1f} churn={roll_stats['mean_churn']:.1f} "
            f"kl={log_rec['mean_kl']:.4f}"
        )

        if rnd % 3 == 0 or rnd == rounds:
            cur_state = {k: v.detach().cpu().clone() for k, v in rl_net.state_dict().items()}
            val_m = _evaluate_val_subset(cur_state, action_stage, ablation_mode, val_specs, tmp_val_pt)
            log_rec["val"] = val_m
            v_key = (-val_m["audit_pass_rate"], -val_m["mean_task_comp"], -val_m["mean_prod_comp"], val_m["mean_net_obj"])
            if v_key < best_val_key:
                best_val_key = v_key
                best_state = cur_state
                best_round = rnd
                best_val_summary = val_m
            msg += f" | val_obj={val_m['mean_net_obj']:.1f} churn={val_m['mean_churn']:.1f} tc={val_m['mean_task_comp']:.3f} best@{best_round}={best_val_summary['mean_net_obj']:.1f}"
        history.append(log_rec)
        print(msg, flush=True)

    for p_rm in (tmp_val_pt, tmp_val_pt.with_suffix(".npz")):
        if p_rm.exists():
            p_rm.unlink()

    cpu_final = V19HeteroConstructorNet(
        hidden_dim=64, n_heads=4, n_layers=2, action_stage=action_stage, ablation_mode=ablation_mode
    )
    cpu_final.load_state_dict(best_state, strict=True)
    cpu_final.eval()

    pt_path = CKPT_DIR / f"{stem}.pt"
    npz_path = CKPT_DIR / f"{stem}.npz"
    torch.save(
        {
            "model": best_state,
            "config": cpu_final.config_dict(),
            "best_round": best_round,
            "best_val": best_val_summary,
            "train_seed": seed,
            "history": history,
        },
        pt_path,
    )
    export_v19_npz(pt_path, npz_path)
    max_parity = _verify_pt_npz_parity(pt_path, npz_path)
    if max_parity >= 1e-5:
        raise RuntimeError(f"Parity check failed for {model_name}: {max_parity:.3e}")

    print(
        f"[RL RL {model_name}] Saved {pt_path.name} & {npz_path.name} (best@{best_round}, val_obj={best_val_summary['mean_net_obj']:.2f}, parity={max_parity:.2e})",
        flush=True,
    )
    return {
        "model_name": model_name,
        "pt_path": str(pt_path),
        "npz_path": str(npz_path),
        "best_round": best_round,
        "best_val": best_val_summary,
        "max_pt_npz_diff": max_parity,
        "history": history,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train G3 KL-anchored SMDP PPO models & ablations.")
    parser.add_argument("--models", nargs="+", default=list(MODEL_SPECS.keys()))
    parser.add_argument("--rounds", type=int, default=12)
    parser.add_argument("--eps-per-round", type=int, default=24)
    parser.add_argument("--ppo-epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.5e-5)
    parser.add_argument("--beta-kl", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    OUT_G3_DIR.mkdir(parents=True, exist_ok=True)
    sum_path = OUT_G3_DIR / "g3_train_summary.json"
    summary: dict[str, Any] = json.loads(sum_path.read_text(encoding="utf-8")) if sum_path.is_file() else {}
    for m_name in args.models:
        if m_name not in MODEL_SPECS:
            raise ValueError(f"Unknown G3 model {m_name!r}; expected one of {list(MODEL_SPECS)}")
        summary[m_name] = train_single_g3_model(
            m_name,
            rounds=args.rounds,
            eps_per_round=args.eps_per_round,
            ppo_epochs=args.ppo_epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            beta_kl=args.beta_kl,
            seed=args.seed,
        )
        sum_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
