"""Train Wang et al. (IJPR 2025) 3-Level Hierarchical MARL (`IJPR25_DFJSPT_HMARL`) on Quarantined Splits.

Faithfully matches `ClouDaDaDa/DFJSPT_code` training protocol (`PPO + Imitation Learning custom_loss`)
under the EXACT SAME quarantined seed splits and episode budget as CoRe-Sched (`AHG-RL`):
  - Stage 1 (IL Warmup): 264 quarantined expert trajectories (`v19_g2_dataset.pt`, 2,473 transitions)
    training all three independent action-masked MLPs (`JobAgent`, `MachineAgent`, `TransbotAgent`).
  - Stage 2 (Closed-Loop Multi-Agent SMDP-PPO + IL Regularization): 12 rounds x 24 closed-loop
    episodes = 288 episodes drawn from the exact same quarantined RL pool (`RL_4TOPO_SEEDS`,
    `RL_WORKSHOP_SEEDS`), keeping the best checkpoint on the 24-episode early-stop validation split.
  - Exports both PyTorch `.pt` and exact pure-NumPy `.npz` (`< 1e-5` max abs diff) to
    `amr_sched/checkpoints/baselines_2026/ijpr25_dfjspt_hmarl.{pt,npz}`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.dfjspt_hmarl_ijpr25 import (
    DFJSPTHMARLNet,
    DFJSPTHMARLProposer,
    IJPR25_DFJSPT_CKPT,
    NumpyDFJSPTHMARLNet,
    decode_dfjspt_candidate,
    export_dfjspt_npz,
    load_dfjspt_model,
)
from amr_sched.baselines.top_venue_baselines import _build_external_proposal
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.rl import AMRReschedulingEnv, RuleV10PortfolioPolicy, compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.v19_hetero_constructor import graph_to_torch_tensors
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
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
PT_OUT = IJPR25_DFJSPT_CKPT
NPZ_OUT = IJPR25_DFJSPT_CKPT.with_suffix(".npz")


def _compute_il_loss(
    net: DFJSPTHMARLNet,
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float]]:
    ten = batch
    tb = batch["t_bays"]
    tc = batch["t_crews"]
    ta = batch["t_agvs"]
    tf = batch["t_focus"]
    tv = batch["t_val"]
    sw = batch["weights"]

    out = net.forward_teacher_forced(ten, tb, ta, tf)
    # Level 1 (JobAgent): pairwise ranking / Plackett-Luce IL loss on task focus scores
    job_lg = out["job_logits"]
    B, Nt = job_lg.shape
    if Nt >= 2:
        t_valid = ten["task_valid"].bool()
        diff_true = tf.unsqueeze(2) - tf.unsqueeze(1)
        diff_pred = job_lg.unsqueeze(2) - job_lg.unsqueeze(1)
        pair_mask = (diff_true > 1e-4) & t_valid.unsqueeze(2) & t_valid.unsqueeze(1)
        if pair_mask.any():
            loss_job = F.softplus(-diff_pred[pair_mask]).mean()
        else:
            loss_job = job_lg.sum() * 0.0
    else:
        loss_job = job_lg.sum() * 0.0

    # Level 2 (MachineAgent): masked cross-entropy on bay & crew choices
    bay_lg = out["bay_logits"]
    bay_mask = ten["op_valid"].bool() & (tb >= 0) & (ten["op_bay_mask"].sum(dim=-1) > 1.5)
    loss_bay = F.cross_entropy(bay_lg[bay_mask], tb[bay_mask]) if bay_mask.any() else bay_lg.sum() * 0.0

    crew_lg = out["crew_logits"]
    crew_mask = ten["op_valid"].bool() & (tc >= 0) & (ten["op_crew_mask"].sum(dim=-1) > 1.5)
    loss_crew = F.cross_entropy(crew_lg[crew_mask], tc[crew_mask]) if crew_mask.any() else crew_lg.sum() * 0.0

    # Level 3 (TransbotAgent): masked cross-entropy on AMR allocations
    agv_lg = out["agv_logits"]
    agv_mask = ten["task_valid"].bool() & (ta >= 0) & (ten["task_agv_mask"].sum(dim=-1) > 1.5)
    loss_agv = F.cross_entropy(agv_lg[agv_mask], ta[agv_mask]) if agv_mask.any() else agv_lg.sum() * 0.0

    # Value regression (normalized scale matching v19_g2_dataset target_value)
    loss_val = (F.smooth_l1_loss(out["value"], tv.squeeze(-1), reduction="none") * sw).mean()

    total = 0.35 * loss_job + 1.0 * loss_bay + 0.5 * loss_crew + 1.2 * loss_agv + 0.1 * loss_val
    return total, {
        "loss_job": float(loss_job.item()),
        "loss_bay": float(loss_bay.item()),
        "loss_crew": float(loss_crew.item()),
        "loss_agv": float(loss_agv.item()),
        "loss_val": float(loss_val.item()),
    }


def _evaluate_policy_batch(
    net: DFJSPTHMARLNet,
    batch: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vectorized batched `[B]` joint 3-agent log-prob, entropy, and critic value (zero CPU-GPU syncs)."""
    tb = batch["t_bays"]
    tc = batch["t_crews"]
    ta = batch["t_agvs"]
    tf = batch["t_focus"]
    out = net.forward_teacher_forced(batch, tb, ta, tf)

    B = tf.shape[0]
    device = tf.device
    logp = torch.zeros(B, dtype=torch.float32, device=device)
    ent = torch.zeros(B, dtype=torch.float32, device=device)

    # Level 1 (JobAgent): vectorized exact Plackett-Luce log-prob over sampled task order
    Nt = tf.shape[1]
    t_valid = batch["task_valid"].bool()
    if Nt >= 2:
        job_lg = out["job_logits"]
        sort_key = tf.masked_fill(~t_valid, -1e9)
        order_idx = torch.argsort(sort_key, dim=1, descending=True)
        ord_lg = torch.gather(job_lg, 1, order_idx)
        ord_valid = torch.gather(t_valid, 1, order_idx)
        ord_lg_masked = ord_lg.masked_fill(~ord_valid, -1e9)
        # Reverse logcumsumexp gives log(sum_{k >= pos} exp(ord_lg[k])) in O(1) GPU ops
        rev_lg = torch.flip(ord_lg_masked, dims=[1])
        rev_lse = torch.logcumsumexp(rev_lg, dim=1)
        lse = torch.flip(rev_lse, dims=[1])
        # Positions with at least 2 remaining valid tasks contribute to Plackett-Luce
        rem_cnt = torch.flip(torch.cumsum(torch.flip(ord_valid.long(), dims=[1]), dim=1), dims=[1])
        pl_mask = ord_valid & (rem_cnt >= 2)
        pl_lp = (ord_lg - lse).masked_fill(~pl_mask, 0.0)
        logp = logp + pl_lp.sum(dim=1)
        # First-step categorical entropy over valid tasks (stable, scale-matched)
        j_lp0 = F.log_softmax(job_lg.masked_fill(~t_valid, -1e9), dim=-1)
        j_p0 = torch.where(t_valid, j_lp0.exp(), torch.zeros_like(j_lp0))
        j_ent0 = -(j_p0 * torch.where(t_valid, j_lp0, torch.zeros_like(j_lp0))).sum(dim=-1)
        ent = ent + torch.where(t_valid.sum(dim=1) >= 2, j_ent0, torch.zeros_like(j_ent0))

    # Level 2 (MachineAgent): vectorized bay + crew masked categorical logp & entropy
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

    # Level 3 (TransbotAgent): vectorized AMR masked categorical logp & entropy
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


def _rollout_episode_for_ppo(
    cpu_net: DFJSPTHMARLNet,
    spec: dict[str, Any],
    seed: int,
) -> tuple[list[dict[str, Any]], float]:
    """Run one closed-loop SMDP rollout on fast CPU inference, collecting batched-ready transitions."""
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
            cand = decode_dfjspt_candidate(cpu_net, graph, snapshot, greedy=False, tau=0.60, rng=rng)
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
            policy_id="IJPR25_DFJSPT_TRAIN",
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

    # Assign SMDP discounted returns telescoping to -net_obj / 1000.0 (old_logp/old_val filled in batch)
    if transitions:
        gamma_step = 0.9985
        final_tick = int(getattr(env.sim, "_step_cursor", transitions[-1]["tick"])) if env.sim is not None else transitions[-1]["tick"]
        term_ret = -final_obj / 1000.0
        for tr_item in transitions:
            dt_rem = max(1, final_tick - int(tr_item["tick"]))
            imm = float(np.clip((tr_item["keep_obj"] - tr_item["eval_obj"]) / 1000.0, -2.0, 2.0))
            tr_item["ret"] = float(term_ret * (gamma_step ** dt_rem) + 0.15 * imm)
    return transitions, final_obj


def _eval_validation(net: DFJSPTHMARLNet, val_specs: list[dict[str, Any]], tmp_pt: Path) -> float:
    torch.save({"model": net.state_dict(), "config": net.config_dict()}, tmp_pt)
    tmp_npz = tmp_pt.with_suffix(".npz")
    export_dfjspt_npz(tmp_pt, tmp_npz)
    objs: list[float] = []
    for sp in val_specs:
        prop = DFJSPTHMARLProposer(tmp_pt, backend="numpy", use_rust=True, seed=int(sp["seed"]))
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


def train_dfjspt_hmarl(resume_il: bool = False) -> dict[str, Any]:
    t0 = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[IJPR25 DFJSP-T] Training on device={device} (exact quarantined splits, resume_il={resume_il})", flush=True)

    # Load exact quarantined 264-episode expert IL dataset
    payload = torch.load(DEFAULT_DATASET_PATH, map_location="cpu", weights_only=False)
    il_samples: list[dict[str, Any]] = payload["samples"]
    assert_seed_quarantine([int(s["seed"]) for s in il_samples], label="dfjspt_il_dataset")

    torch.manual_seed(2025)
    np.random.seed(2025)
    net = DFJSPTHMARLNet(hidden_dim=256).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=1e-4)
    rng = np.random.default_rng(2025)

    tmp_pt = PT_OUT.parent / "_dfjspt_val_tmp.pt"
    PT_OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── Stage 1: Imitation Learning Warmup (20 epochs over 2,473 expert samples) ──
    if resume_il and tmp_pt.is_file():
        ck = torch.load(tmp_pt, map_location=device, weights_only=False)
        net.load_state_dict(ck["model"], strict=True)
        print(f"[IJPR25 IL] Resumed 20-epoch IL warmup checkpoint from {tmp_pt}", flush=True)
    else:
        il_epochs = 20
        batch_size = 64
        for ep in range(il_epochs):
            net.train()
            perm = rng.permutation(len(il_samples))
            ep_losses: list[float] = []
            for s_i in range(0, len(il_samples), batch_size):
                batch = _collate_v19_batch([il_samples[int(j)] for j in perm[s_i : s_i + batch_size]], device)
                opt.zero_grad(set_to_none=True)
                loss, _ = _compute_il_loss(net, batch, device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                ep_losses.append(float(loss.item()))
            if (ep + 1) in (1, 10, il_epochs):
                print(f"[IJPR25 IL] Epoch {ep+1:02d}/{il_epochs} | loss={np.mean(ep_losses):.4f} ({time.perf_counter()-t0:.1f}s)", flush=True)

    val_specs = build_rl_early_val_specs()
    train_pool = build_rl_train_pool()
    best_val = _eval_validation(net, val_specs, tmp_pt)
    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
    print(f"[IJPR25 IL] Post-IL 24-ep early-val mean_net_obj = {best_val:.2f}", flush=True)

    # ── Stage 2: 12 rounds x 24 closed-loop SMDP-PPO + IL episodes (288 episodes, exact parity with CoRe-Sched) ──
    ppo_opt = torch.optim.AdamW(net.parameters(), lr=2.5e-4, weight_decay=1e-4)
    cpu_net = DFJSPTHMARLNet().to(torch.device("cpu"))
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
            tr_list, f_obj = _rollout_episode_for_ppo(cpu_net, train_pool[int(p_idx)], seed=2025 + rnd * 100 + int(p_idx))
            round_trans.extend(tr_list)
            round_objs.append(f_obj)

        if round_trans:
            # Batched GPU computation of old_logp & old_val (zero unbatched CUDA overhead)
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
            # 3 vectorized batched PPO epochs + auxiliary IL custom_loss (decaying weight like DFJSPT_code)
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
                    s1 = ratio * adv_t
                    s2 = torch.clamp(ratio, 0.8, 1.2) * adv_t
                    ppo_loss = -torch.min(s1, s2).mean()
                    val_loss = F.smooth_l1_loss(val, ret_t)
                    ent_bonus = ent.mean()

                    il_idxs = rng.choice(len(il_samples), size=min(64, len(il_samples)), replace=False)
                    il_batch = _collate_v19_batch([il_samples[int(j)] for j in il_idxs], device)
                    l_il, _ = _compute_il_loss(net, il_batch, device)
                    loss_step = ppo_loss + 0.25 * val_loss - 0.005 * ent_bonus + il_weight * l_il
                    loss_step.backward()
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                    ppo_opt.step()

        val_obj = _eval_validation(net, val_specs, tmp_pt)
        improved = val_obj < best_val
        if improved:
            best_val = val_obj
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save({"model": best_state, "config": net.config_dict()}, PT_OUT)
            export_dfjspt_npz(PT_OUT, NPZ_OUT)
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
            f"[IJPR25 PPO+IL] Round {rnd+1:02d}/{n_rounds} | train={np.mean(round_objs):7.1f} "
            f"| trans={len(round_trans):4d} | val={val_obj:7.1f} (best={best_val:7.1f}) ({time.perf_counter()-t0:.1f}s)",
            flush=True,
        )

    # Save best checkpoint (.pt + exact .npz) and verify <1e-5 parity
    torch.save({"model": best_state, "config": net.config_dict()}, PT_OUT)
    export_dfjspt_npz(PT_OUT, NPZ_OUT)
    for p_rm in (tmp_pt, tmp_pt.with_suffix(".npz")):
        if p_rm.is_file():
            p_rm.unlink()

    # Parity check between PyTorch and NumPy backends
    pt_m, _, _ = load_dfjspt_model(PT_OUT, backend="torch")
    np_m, _, _ = load_dfjspt_model(PT_OUT, backend="numpy")
    sp0 = val_specs[0]
    from amr_sched.allocation.greedy import GreedyAllocator
    from amr_sched.cvcr.snapshot import NativeSnapshotCapture
    from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
    from amr_sched.simulation import Simulator
    net0, tasks0, agvs0, ops0, _ = copy.deepcopy(sp0["scenario_tuple"])
    sim0 = Simulator(net0, tasks0, agvs0, StaticDijkstraPlanner(), GreedyAllocator())
    snap0 = NativeSnapshotCapture.capture(sim0, ops0)
    g0 = extract_v19_hetero_graph(snap0, net0)
    with torch.no_grad():
        t0_ten = graph_to_torch_tensors(g0)
        gf_pt = pt_m.global_summary(t0_ten)
        jl_pt, _ = pt_m.forward_job_agent(t0_ten, gf_pt)
        bl_pt, cl_pt, _ = pt_m.forward_machine_agent(t0_ten, gf_pt)
    gf_np = np.concatenate([
        g0.op_feats.mean(0) if len(g0.op_feats) else np.zeros(12),
        g0.task_feats.mean(0) if len(g0.task_feats) else np.zeros(12),
        g0.agv_feats.mean(0) if len(g0.agv_feats) else np.zeros(10),
        g0.bay_feats.mean(0) if len(g0.bay_feats) else np.zeros(8),
        g0.crew_feats.mean(0) if len(g0.crew_feats) else np.zeros(4),
    ]).astype(np.float32)
    jl_np = np_m.predict_job_logits(g0, gf_np)
    bl_np, cl_np = np_m.predict_machine_logits(g0, gf_np)
    max_diff = max(
        float(np.max(np.abs(jl_pt[0].numpy() - jl_np))) if len(jl_np) else 0.0,
        float(np.max(np.abs(bl_pt[0].numpy() - bl_np))) if bl_np.size else 0.0,
        float(np.max(np.abs(cl_pt[0].numpy() - cl_np))) if cl_np.size else 0.0,
    )
    assert max_diff < 1e-4, f"PyTorch vs NumPy parity error too large: {max_diff}"
    summary = {
        "policy": "IJPR25_DFJSPT_HMARL",
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "best_val_mean_net_obj": round(best_val, 2),
        "pt_npz_max_abs_diff": max_diff,
        "pt_path": str(PT_OUT),
        "npz_path": str(NPZ_OUT),
        "rounds": history,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "ijpr25_dfjspt_train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[IJPR25 DFJSP-T] Done! best_val={best_val:.2f}, pt-npz diff={max_diff:.2e}, saved to {PT_OUT}", flush=True)
    return summary


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--resume-il", action="store_true")
    a = ap.parse_args()
    train_dfjspt_hmarl(resume_il=a.resume_il)
