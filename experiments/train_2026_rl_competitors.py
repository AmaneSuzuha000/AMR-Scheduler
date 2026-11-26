"""Unified Quarantined SMDP-PPO + IL Trainer for Gate G2 2026 Dynamic FJSP-AGV Neural Competitors.

Trains BOTH:
  1. `ASC26_MA_TRANS` (Shen, Zhang & Jin, Applied Soft Computing 193, 2026, 114899):
     2-layer Transformer job/op encoder + 1-layer Resource GAT + Dual-Agent PPO.
  2. `FGCS26_REN_GNN` (Ren, Liu, Huang & Cao, Future Generation Computer Systems 182, 2026):
     2-hop 3-Plane 3D Disjunctive Graph GNN + Disruption-Aware Residual Gate + Composite PPO.

Under 100% IDENTICAL quarantined data splits and training budget as CoRe-Sched (`AHG-RL`) and
`IJPR25_DFJSPT_HMARL`:
  - Stage 1 (IL Warmup): 20 epochs over the exact 264 quarantined expert trajectories
    (`v19_g2_dataset.pt`, 2,473 transitions, zero overlap with ANY blind holdout seeds).
  - Stage 2 (Closed-Loop SMDP-PPO + IL Regularization): 12 rounds x 24 closed-loop episodes
    = 288 episodes drawn from the exact quarantined RL pool (`RL_4TOPO_SEEDS`, `RL_WORKSHOP_SEEDS`),
    vectorized `[B=64]` GPU PPO updates, keeping the best checkpoint on the 24-episode early-stop
    validation split.
  - Exports BOTH PyTorch `.pt` and exact pure-NumPy `.npz` (`< 1e-5` max abs diff asserted on
    real workshop + 4-Topo snapshots!) to:
      * `amr_sched/checkpoints/baselines_2026/asc26_ma_trans.{pt,npz}`
      * `amr_sched/checkpoints/baselines_2026/fgcs26_ren_gnn.{pt,npz}`
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.baselines.fgcs26_ren_gnn import (
    FGCS26_REN_GNN_CKPT,
    FGCS26RenGNNNet,
    FGCS26RenGNNProposer,
    decode_fgcs26_candidate,
    export_fgcs26_ren_npz,
    load_fgcs26_ren_model,
)
from amr_sched.baselines.ma_trans_asc26 import (
    ASC26_MA_TRANS_CKPT,
    MATransASC26Proposer,
    MATransNet,
    decode_ma_trans_candidate,
    export_ma_trans_npz,
    load_ma_trans_model,
)
from amr_sched.baselines.top_venue_baselines import _build_external_proposal
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.rl import (
    AMRReschedulingEnv,
    RuleV10PortfolioPolicy,
    compute_expert_v10_action_mask,
    compute_expert_v9_state,
)
from amr_sched.rl.v19_hetero_constructor import graph_to_torch_tensors
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from amr_sched.simulation import Simulator
from experiments.train_v19_bc import (
    DEFAULT_DATASET_PATH,
    _collate_v19_batch,
    assert_seed_quarantine,
)
from experiments.train_v19_rl import (
    build_rl_early_val_specs,
    build_rl_train_pool,
)

OUT_DIR = ROOT / "out" / "v19_scientific_audit"


def _compute_il_loss(net: torch.nn.Module, batch: dict[str, Any]) -> tuple[torch.Tensor, dict[str, float]]:
    tb = batch["t_bays"]
    tc = batch["t_crews"]
    ta = batch["t_agvs"]
    tf = batch["t_focus"]
    tv = batch["t_val"]
    sw = batch["weights"]

    out = net.forward_teacher_forced(batch, tb, ta, tf)
    job_lg = out["job_logits"]
    Nt = job_lg.shape[1]
    if Nt >= 2:
        t_valid = batch["task_valid"].bool()
        diff_true = tf.unsqueeze(2) - tf.unsqueeze(1)
        diff_pred = job_lg.unsqueeze(2) - job_lg.unsqueeze(1)
        pair_mask = (diff_true > 1e-4) & t_valid.unsqueeze(2) & t_valid.unsqueeze(1)
        loss_job = F.softplus(-diff_pred[pair_mask]).mean() if pair_mask.any() else job_lg.sum() * 0.0
    else:
        loss_job = job_lg.sum() * 0.0

    bay_lg = out["bay_logits"]
    bay_mask = batch["op_valid"].bool() & (tb >= 0) & (batch["op_bay_mask"].sum(dim=-1) > 1.5)
    loss_bay = F.cross_entropy(bay_lg[bay_mask], tb[bay_mask]) if bay_mask.any() else bay_lg.sum() * 0.0

    crew_lg = out["crew_logits"]
    crew_mask = batch["op_valid"].bool() & (tc >= 0) & (batch["op_crew_mask"].sum(dim=-1) > 1.5)
    loss_crew = F.cross_entropy(crew_lg[crew_mask], tc[crew_mask]) if crew_mask.any() else crew_lg.sum() * 0.0

    agv_lg = out["agv_logits"]
    agv_mask = batch["task_valid"].bool() & (ta >= 0) & (batch["task_agv_mask"].sum(dim=-1) > 1.5)
    loss_agv = F.cross_entropy(agv_lg[agv_mask], ta[agv_mask]) if agv_mask.any() else agv_lg.sum() * 0.0

    loss_val = (F.smooth_l1_loss(out["value"], tv.squeeze(-1), reduction="none") * sw).mean()
    total = 0.35 * loss_job + 1.0 * loss_bay + 0.5 * loss_crew + 1.2 * loss_agv + 0.1 * loss_val
    return total, {
        "loss_job": float(loss_job.item()),
        "loss_bay": float(loss_bay.item()),
        "loss_crew": float(loss_crew.item()),
        "loss_agv": float(loss_agv.item()),
        "loss_val": float(loss_val.item()),
    }


def _evaluate_policy_batch(net: torch.nn.Module, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tb, tc, ta, tf = batch["t_bays"], batch["t_crews"], batch["t_agvs"], batch["t_focus"]
    out = net.forward_teacher_forced(batch, tb, ta, tf)
    B, Nt = tf.shape
    device = tf.device
    logp = torch.zeros(B, dtype=torch.float32, device=device)
    ent = torch.zeros(B, dtype=torch.float32, device=device)

    t_valid = batch["task_valid"].bool()
    if Nt >= 2:
        job_lg = out["job_logits"]
        sort_key = tf.masked_fill(~t_valid, -1e9)
        order_idx = torch.argsort(sort_key, dim=1, descending=True)
        ord_lg = torch.gather(job_lg, 1, order_idx)
        ord_valid = torch.gather(t_valid, 1, order_idx)
        ord_lg_m = ord_lg.masked_fill(~ord_valid, -1e9)
        lse = torch.flip(torch.logcumsumexp(torch.flip(ord_lg_m, dims=[1]), dim=1), dims=[1])
        rem_cnt = torch.flip(torch.cumsum(torch.flip(ord_valid.long(), dims=[1]), dim=1), dims=[1])
        pl_mask = ord_valid & (rem_cnt >= 2)
        logp = logp + (ord_lg - lse).masked_fill(~pl_mask, 0.0).sum(dim=1)
        j_lp0 = F.log_softmax(job_lg.masked_fill(~t_valid, -1e9), dim=-1)
        j_p0 = torch.where(t_valid, j_lp0.exp(), torch.zeros_like(j_lp0))
        j_ent0 = -(j_p0 * torch.where(t_valid, j_lp0, torch.zeros_like(j_lp0))).sum(dim=-1)
        ent = ent + torch.where(t_valid.sum(dim=1) >= 2, j_ent0, torch.zeros_like(j_ent0))

    op_valid = batch["op_valid"].bool()
    if op_valid.shape[1] > 0:
        bay_lp_all = F.log_softmax(out["bay_logits"], dim=-1)
        bay_mask = op_valid & (tb >= 0) & (batch["op_bay_mask"].sum(dim=-1) > 1.5)
        bay_sel = torch.gather(bay_lp_all, 2, tb.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        logp = logp + bay_sel.masked_fill(~bay_mask, 0.0).sum(dim=1)
        b_ok = batch["op_bay_mask"].bool()
        b_p = torch.where(b_ok, bay_lp_all.exp(), torch.zeros_like(bay_lp_all))
        b_e = -(b_p * torch.where(b_ok, bay_lp_all, torch.zeros_like(bay_lp_all))).sum(dim=-1)
        ent = ent + b_e.masked_fill(~bay_mask, 0.0).sum(dim=1)

        crew_lp_all = F.log_softmax(out["crew_logits"], dim=-1)
        crew_mask = op_valid & (tc >= 0) & (batch["op_crew_mask"].sum(dim=-1) > 1.5)
        crew_sel = torch.gather(crew_lp_all, 2, tc.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        logp = logp + crew_sel.masked_fill(~crew_mask, 0.0).sum(dim=1)
        c_ok = batch["op_crew_mask"].bool()
        c_p = torch.where(c_ok, crew_lp_all.exp(), torch.zeros_like(crew_lp_all))
        c_e = -(c_p * torch.where(c_ok, crew_lp_all, torch.zeros_like(crew_lp_all))).sum(dim=-1)
        ent = ent + c_e.masked_fill(~crew_mask, 0.0).sum(dim=1)

    if Nt > 0:
        agv_lp_all = F.log_softmax(out["agv_logits"], dim=-1)
        agv_mask = t_valid & (ta >= 0) & (batch["task_agv_mask"].sum(dim=-1) > 1.5)
        agv_sel = torch.gather(agv_lp_all, 2, ta.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        logp = logp + agv_sel.masked_fill(~agv_mask, 0.0).sum(dim=1)
        a_ok = batch["task_agv_mask"].bool()
        a_p = torch.where(a_ok, agv_lp_all.exp(), torch.zeros_like(agv_lp_all))
        a_e = -(a_p * torch.where(a_ok, agv_lp_all, torch.zeros_like(agv_lp_all))).sum(dim=-1)
        ent = ent + a_e.masked_fill(~agv_mask, 0.0).sum(dim=1)

    return logp, ent, out["value"]


def _rollout_episode(
    cpu_net: torch.nn.Module,
    decode_fn: Callable[..., Any],
    policy_tag: str,
    spec: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], float]:
    cpu_net.eval()
    sc_copy = copy.deepcopy(spec["scenario_tuple"])
    p2_copy = copy.deepcopy(spec["p2_cfg"])
    rng = np.random.default_rng(seed)
    transitions: list[dict[str, Any]] = []

    def _collect_proposer(snapshot: Any, network: Any, budget_ms: float = 35.0) -> Any:
        t0 = time.perf_counter()
        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        ev = CVCREvaluator(network, snapshot, alpha_makespan=3.0, lambda_change=15.0, use_rust=True, mode="B2", seq_diff_b2=True)
        base_prio = {op.op_id: max(int(snapshot.tick), int(op.due_step) - int(op.proc_steps)) for op in snapshot.operations}
        keep_ev = ev.evaluate(graph.base_assign, op_priority=base_prio, op_bay_overrides=graph.base_bays, op_crew_overrides=graph.base_crews)
        if keep_ev.valid:
            ev.set_incumbent(graph.base_assign, keep_ev, op_priority=base_prio, op_bay_overrides=graph.base_bays)

        best_ev = keep_ev
        best_assign, best_bays, best_crews = dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews)
        if len(graph.rel_tids) > 0 and len(graph.agv_ids) >= 1:
            cand = decode_fn(cpu_net, graph, snapshot, greedy=False, tau=0.60, rng=rng)
            ev_c = ev.evaluate(cand.assign, op_priority=cand.op_prio, op_bay_overrides=cand.bays, op_crew_overrides=cand.crews)
            ten = graph_to_torch_tensors(graph)
            Nt = len(graph.rel_tids)
            focus_rank = np.zeros(Nt, dtype=np.float32)
            for pos, idx in enumerate(cand.task_order):
                if 0 <= idx < Nt:
                    focus_rank[idx] = float(Nt - pos)
            transitions.append(
                {
                    "tick": int(snapshot.tick),
                    "tensors": ten,
                    "op_bay_mask": ten["op_bay_mask"][0],
                    "op_crew_mask": ten["op_crew_mask"][0],
                    "task_agv_mask": ten["task_agv_mask"][0],
                    "target_bays": torch.tensor(cand.chosen_bay_idx, dtype=torch.long),
                    "target_crews": torch.tensor(cand.chosen_crew_idx, dtype=torch.long),
                    "target_agvs": torch.tensor(cand.chosen_agv_idx, dtype=torch.long),
                    "target_focus": torch.from_numpy(focus_rank),
                    "target_seq_rank": torch.from_numpy(focus_rank),
                    "target_value": torch.zeros(1, dtype=torch.float32),
                    "sample_weight": 1.0,
                    "eval_obj": float(ev_c.objective) if ev_c.valid else float(keep_ev.objective) + 500.0,
                    "keep_obj": float(keep_ev.objective),
                }
            )
            if ev_c.valid and ev_c.objective + 10.0 < keep_ev.objective:
                best_ev = ev_c
                best_assign, best_bays, best_crews = cand.assign, cand.bays, cand.crews

        return _build_external_proposal(
            policy_id=policy_tag,
            snapshot=snapshot,
            clf=clf,
            best_assign=best_assign,
            best_bays=best_bays,
            best_crews=best_crews,
            best_eval=best_ev,
            op_prio=base_prio,
            eval_cnt=ev.calls,
            t0=t0,
            budget_ms=budget_ms,
        )

    env = AMRReschedulingEnv(
        seed=int(spec["seed"]),
        max_steps=int(spec["max_steps"]),
        use_rust=True,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        loaded_repair_mode="haven_v12",
        record_trajectory=False,
        execution_perturbation=p2_copy,
        execution_aware=True,
        p4_choke_robust=True,
        expert_v10_portfolio=True,
        external_proposer=_collect_proposer,
    )
    ctrl = RuleV10PortfolioPolicy(epoch_interval=20)
    ctrl.reset()
    with virtual_clock_scope(True):
        obs, info = env.reset(seed=int(spec["seed"]), options={"scenario_tuple": sc_copy})
        tm = False
        tr = False
        while not (tm or tr):
            assert env.sim is not None and env.tracker is not None
            feats = compute_expert_v9_state(
                env.sim, env.tracker,
                last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks,
                decision_logs=env.decision_logs,
            )
            step_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
            act = ctrl.select_action(feats, step_info, sim=env.sim)
            obs, _, tm, tr, info = env.step(act)
    m = env.compute_episode_metrics()
    final_obj = float(m["net_obj"])
    if transitions:
        gamma_step = 0.9985
        final_tick = int(getattr(env.sim, "_step_cursor", transitions[-1]["tick"])) if env.sim is not None else transitions[-1]["tick"]
        term_ret = -final_obj / 1000.0
        for tr_item in transitions:
            dt_rem = max(1, final_tick - int(tr_item["tick"]))
            imm = float(np.clip((tr_item["keep_obj"] - tr_item["eval_obj"]) / 1000.0, -2.0, 2.0))
            tr_item["ret"] = float(term_ret * (gamma_step ** dt_rem) + 0.15 * imm)
    return transitions, final_obj


def _eval_validation(
    net: torch.nn.Module,
    export_npz_fn: Callable[[Any, Path], Path],
    proposer_cls: type,
    val_specs: list[dict[str, Any]],
    tmp_pt: Path,
) -> float:
    torch.save({"state_dict": net.state_dict(), "config": net.config_dict()}, tmp_pt)
    export_npz_fn(net, tmp_pt.with_suffix(".npz"))
    objs: list[float] = []
    for sp in val_specs:
        prop = proposer_cls(tmp_pt, backend="numpy", use_rust=True, seed=int(sp["seed"]))
        sc_copy = copy.deepcopy(sp["scenario_tuple"])
        p2_copy = copy.deepcopy(sp["p2_cfg"])
        env = AMRReschedulingEnv(
            seed=int(sp["seed"]),
            max_steps=int(sp["max_steps"]),
            use_rust=True,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=True,
            loaded_repair_mode="haven_v12",
            record_trajectory=False,
            execution_perturbation=p2_copy,
            execution_aware=True,
            p4_choke_robust=True,
            expert_v10_portfolio=True,
            external_proposer=prop,
        )
        ctrl = RuleV10PortfolioPolicy(epoch_interval=20)
        ctrl.reset()
        with virtual_clock_scope(True):
            obs, info = env.reset(seed=int(sp["seed"]), options={"scenario_tuple": sc_copy})
            tm = False
            tr = False
            while not (tm or tr):
                assert env.sim is not None and env.tracker is not None
                feats = compute_expert_v9_state(
                    env.sim, env.tracker,
                    last_replan_tick=env.last_replan_tick,
                    recent_conflict_ticks=env.recent_conflict_ticks,
                    decision_logs=env.decision_logs,
                )
                step_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
                act = ctrl.select_action(feats, step_info, sim=env.sim)
                obs, _, tm, tr, info = env.step(act)
        objs.append(float(env.compute_episode_metrics()["net_obj"]))
    return float(np.mean(objs))


def _verify_pt_npz_parity(
    policy_name: str,
    pt_path: Path,
    load_fn: Callable[..., tuple[Any, str, Any]],
    val_specs: list[dict[str, Any]],
) -> float:
    pt_m, _, _ = load_fn(pt_path, backend="torch")
    np_m, _, _ = load_fn(pt_path, backend="numpy")
    max_diff = 0.0
    for sp in (val_specs[0], val_specs[-1]):
        net0, tasks0, agvs0, ops0, _ = copy.deepcopy(sp["scenario_tuple"])
        sim0 = Simulator(net0, tasks0, agvs0, StaticDijkstraPlanner(), GreedyAllocator())
        snap0 = NativeSnapshotCapture.capture(sim0, ops0)
        g0 = extract_v19_hetero_graph(snap0, net0)
        t0_ten = graph_to_torch_tensors(g0)
        with torch.no_grad():
            enc_pt = pt_m.encode(t0_ten)
            if policy_name == "ASC26_MA_TRANS":
                jl_pt, bl_pt, cl_pt = pt_m.forward_machine_heads(enc_pt, t0_ten)
            else:
                jl_pt, bl_pt, cl_pt = pt_m.forward_static_heads(enc_pt, t0_ten)
        enc_np = np_m.encode(g0)
        if policy_name == "ASC26_MA_TRANS":
            jl_np, bl_np, cl_np = np_m.predict_machine_heads(enc_np, g0)
        else:
            jl_np, bl_np, cl_np = np_m.predict_static_heads(enc_np, g0)

        if len(jl_np):
            max_diff = max(max_diff, float(np.max(np.abs(jl_pt[0].numpy() - jl_np))))
        if bl_np.size:
            b_ok = g0.op_bay_mask
            if b_ok.any():
                max_diff = max(max_diff, float(np.max(np.abs(bl_pt[0].numpy()[b_ok] - bl_np[b_ok]))))
        if cl_np.size:
            c_ok = g0.op_crew_mask
            if c_ok.any():
                max_diff = max(max_diff, float(np.max(np.abs(cl_pt[0].numpy()[c_ok] - cl_np[c_ok]))))
        if len(g0.rel_tids) > 0 and len(g0.agv_ids) > 0:
            dyn = np.zeros((len(g0.agv_ids), 2), dtype=np.float32)
            D = enc_np["h_bay"].shape[-1] if len(g0.bay_ids) else enc_np["h_agv"].shape[-1]
            sb_np = enc_np["h_bay"][0] if len(g0.bay_ids) else np.zeros((D,), dtype=np.float32)
            al_np = np_m.predict_agv_step(enc_np, g0, 0, sb_np, dyn)
            with torch.no_grad():
                g_ctx_t = enc_pt["g_res"] if policy_name == "ASC26_MA_TRANS" else enc_pt["g_pool"]
                al_pt = pt_m.forward_agv_step(
                    enc_pt["h_task"][:, 0],
                    enc_pt["h_agv"],
                    t0_ten["task_agv_x"][:, 0],
                    torch.from_numpy(sb_np[None, :]),
                    torch.from_numpy(dyn[None, :, :]),
                    g_ctx_t,
                    t0_ten["task_agv_mask"][:, 0],
                )[0].numpy()
            a_ok = g0.task_agv_mask[0]
            if a_ok.any():
                max_diff = max(max_diff, float(np.max(np.abs(al_pt[a_ok] - al_np[a_ok]))))

    assert max_diff < 1e-5, f"{policy_name} PyTorch-NumPy parity error {max_diff:.2e} >= 1e-5!"
    return max_diff


def train_one_competitor(
    policy_name: str,
    net_ctor: Callable[[], torch.nn.Module],
    decode_fn: Callable[..., Any],
    export_npz_fn: Callable[[Any, Path], Path],
    load_fn: Callable[..., tuple[Any, str, Any]],
    proposer_cls: type,
    pt_out: Path,
    seed: int,
    il_samples: list[dict[str, Any]],
    val_specs: list[dict[str, Any]],
    train_pool: list[dict[str, Any]],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[{policy_name}] Training on device={device} (seed={seed}, quarantined 264 IL + 288 SMDP-PPO)", flush=True)

    torch.manual_seed(seed)
    np.random.seed(seed)
    net = net_ctor().to(device)
    n_params = sum(p.numel() for p in net.parameters())
    print(f"[{policy_name}] Parameters: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    pt_out.parent.mkdir(parents=True, exist_ok=True)
    npz_out = pt_out.with_suffix(".npz")
    tmp_pt = pt_out.parent / f"_{policy_name.lower()}_val_tmp.pt"

    # Stage 1: 20-epoch IL warmup
    il_epochs = 20
    batch_size = 64
    for ep in range(il_epochs):
        net.train()
        perm = rng.permutation(len(il_samples))
        losses: list[float] = []
        for s_i in range(0, len(il_samples), batch_size):
            batch = _collate_v19_batch([il_samples[int(j)] for j in perm[s_i : s_i + batch_size]], device)
            opt.zero_grad(set_to_none=True)
            loss, _ = _compute_il_loss(net, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            losses.append(float(loss.item()))
        if (ep + 1) in (1, 10, il_epochs):
            print(f"[{policy_name} IL] Epoch {ep+1:02d}/{il_epochs} | loss={np.mean(losses):.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)

    best_val = _eval_validation(net, export_npz_fn, proposer_cls, val_specs, tmp_pt)
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    torch.save({"state_dict": best_state, "config": net.config_dict()}, pt_out)
    export_npz_fn(net, npz_out)
    print(f"[{policy_name} IL] Post-IL 24-ep early-val mean_net_obj = {best_val:.2f}", flush=True)

    # Stage 2: 12 rounds x 24 closed-loop SMDP-PPO + IL episodes (288 episodes)
    ppo_opt = torch.optim.AdamW(net.parameters(), lr=2.5e-4, weight_decay=1e-4)
    cpu_net = net_ctor().to(torch.device("cpu"))
    n_rounds = 12
    eps_per_round = 24
    history: list[dict[str, Any]] = []
    for rnd in range(n_rounds):
        cpu_net.load_state_dict({k: v.detach().cpu() for k, v in net.state_dict().items()})
        cpu_net.eval()
        picks = rng.choice(len(train_pool), size=eps_per_round, replace=False)
        round_trans: list[dict[str, Any]] = []
        round_objs: list[float] = []
        for p_idx in picks:
            tr_list, f_obj = _rollout_episode(cpu_net, decode_fn, f"{policy_name}_TRAIN", train_pool[int(p_idx)], seed=seed + rnd * 100 + int(p_idx))
            round_trans.extend(tr_list)
            round_objs.append(f_obj)

        if round_trans:
            net.eval()
            with torch.no_grad():
                for start in range(0, len(round_trans), 64):
                    chunk = round_trans[start : start + 64]
                    b_ten = _collate_v19_batch(chunk, device)
                    old_lp, _, old_v = _evaluate_policy_batch(net, b_ten)
                    old_lp_np = old_lp.cpu().numpy()
                    old_v_np = old_v.cpu().numpy()
                    for k_i, tr_item in enumerate(chunk):
                        tr_item["old_logp"] = float(old_lp_np[k_i])
                        tr_item["old_val"] = float(old_v_np[k_i])
                        tr_item["adv"] = float(tr_item["ret"] - tr_item["old_val"])

            advs = np.array([t["adv"] for t in round_trans], dtype=np.float32)
            adv_mean, adv_std = float(advs.mean()), max(1e-4, float(advs.std()))
            for t in round_trans:
                t["adv_norm"] = float((t["adv"] - adv_mean) / adv_std)

            net.train()
            il_weight = max(0.05, 0.35 * (1.0 - rnd / float(n_rounds)))
            for _ppo_ep in range(3):
                tr_perm = rng.permutation(len(round_trans))
                for start in range(0, len(round_trans), 64):
                    chunk = [round_trans[int(idx)] for idx in tr_perm[start : start + 64]]
                    b_ten = _collate_v19_batch(chunk, device)
                    old_lp_t = torch.tensor([c["old_logp"] for c in chunk], dtype=torch.float32, device=device)
                    adv_t = torch.tensor([c["adv_norm"] for c in chunk], dtype=torch.float32, device=device)
                    ret_t = torch.tensor([c["ret"] for c in chunk], dtype=torch.float32, device=device)

                    ppo_opt.zero_grad(set_to_none=True)
                    lp, ent, val = _evaluate_policy_batch(net, b_ten)
                    ratio = torch.exp(torch.clamp(lp - old_lp_t, -10.0, 10.0))
                    ppo_loss = -torch.min(ratio * adv_t, torch.clamp(ratio, 0.8, 1.2) * adv_t).mean()
                    val_loss = F.smooth_l1_loss(val, ret_t)
                    ent_bonus = ent.mean()

                    il_idxs = rng.choice(len(il_samples), size=min(64, len(il_samples)), replace=False)
                    il_batch = _collate_v19_batch([il_samples[int(j)] for j in il_idxs], device)
                    l_il, _ = _compute_il_loss(net, il_batch)
                    loss_step = ppo_loss + 0.25 * val_loss - 0.005 * ent_bonus + il_weight * l_il
                    loss_step.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    ppo_opt.step()

        val_obj = _eval_validation(net, export_npz_fn, proposer_cls, val_specs, tmp_pt)
        improved = val_obj < best_val
        if improved:
            best_val = val_obj
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save({"state_dict": best_state, "config": net.config_dict()}, pt_out)
            export_npz_fn(net, npz_out)
        history.append(
            {
                "round": rnd + 1,
                "train_mean_obj": round(float(np.mean(round_objs)), 2),
                "transitions": len(round_trans),
                "val_mean_obj": round(val_obj, 2),
                "best_val": round(best_val, 2),
                "improved": improved,
            }
        )
        print(
            f"[{policy_name} PPO+IL] Round {rnd+1:02d}/{n_rounds} | train={np.mean(round_objs):7.1f} "
            f"| trans={len(round_trans):4d} | val={val_obj:7.1f} (best={best_val:7.1f}) ({time.perf_counter()-t0:.1f}s)",
            flush=True,
        )

    # Restore best state, write final .pt & .npz, clean temp, verify < 1e-5 parity!
    net.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    torch.save({"state_dict": best_state, "config": net.config_dict()}, pt_out)
    export_npz_fn(net, npz_out)
    for p_rm in (tmp_pt, tmp_pt.with_suffix(".npz")):
        if p_rm.is_file():
            p_rm.unlink()

    max_diff = _verify_pt_npz_parity(policy_name, pt_out, load_fn, val_specs)
    print(f"[{policy_name}] DONE! best_val={best_val:.2f}, pt-npz max_diff={max_diff:.2e}, saved to {pt_out}", flush=True)
    return {
        "policy": policy_name,
        "params": n_params,
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "best_val_mean_net_obj": round(best_val, 2),
        "pt_npz_max_abs_diff": max_diff,
        "pt_path": str(pt_out),
        "npz_path": str(npz_out),
        "rounds": history,
    }


def main(only: str = "all") -> dict[str, Any]:
    payload = torch.load(DEFAULT_DATASET_PATH, map_location="cpu", weights_only=False)
    il_samples: list[dict[str, Any]] = payload["samples"]
    assert_seed_quarantine([int(s["seed"]) for s in il_samples], label="gate_g2_il_dataset")
    val_specs = build_rl_early_val_specs()
    train_pool = build_rl_train_pool()

    results: dict[str, Any] = {}
    if only in ("all", "ma_trans", "asc26"):
        results["ASC26_MA_TRANS"] = train_one_competitor(
            policy_name="ASC26_MA_TRANS",
            net_ctor=lambda: MATransNet(d_model=64, n_heads=4, d_ff=128),
            decode_fn=decode_ma_trans_candidate,
            export_npz_fn=export_ma_trans_npz,
            load_fn=load_ma_trans_model,
            proposer_cls=MATransASC26Proposer,
            pt_out=ASC26_MA_TRANS_CKPT,
            seed=20261,
            il_samples=il_samples,
            val_specs=val_specs,
            train_pool=train_pool,
        )
    if only in ("all", "ren_gnn", "fgcs26"):
        results["FGCS26_REN_GNN"] = train_one_competitor(
            policy_name="FGCS26_REN_GNN",
            net_ctor=lambda: FGCS26RenGNNNet(hidden_dim=64),
            decode_fn=decode_fgcs26_candidate,
            export_npz_fn=export_fgcs26_ren_npz,
            load_fn=load_fgcs26_ren_model,
            proposer_cls=FGCS26RenGNNProposer,
            pt_out=FGCS26_REN_GNN_CKPT,
            seed=20262,
            il_samples=il_samples,
            val_specs=val_specs,
            train_pool=train_pool,
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_p = OUT_DIR / "train_2026_rl_competitors.json"
    if out_p.exists() and only != "all":
        prev = json.loads(out_p.read_text(encoding="utf-8"))
        prev.update(results)
        results = prev
    out_p.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\n[Gate G2 Trainer] Saved summary to {out_p}", flush=True)
    return results


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="all", choices=["all", "ma_trans", "asc26", "ren_gnn", "fgcs26"])
    args = ap.parse_args()
    main(only=args.only)
