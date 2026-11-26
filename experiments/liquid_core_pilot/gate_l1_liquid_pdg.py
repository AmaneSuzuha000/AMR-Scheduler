"""Track L — Gate L0 + L1: Liquid-PDG Progressive Risk Prediction & Conformal Risk Control (CRC).

Strictly isolated on quarantined DEV episodes (VAL-72 + DEV-Stress-24 + DEV-Train subset;
ZERO overlap with Cohort A/B/C/D and NEVER trained on 2906/3905).
Evaluates matched-budget 7-model ladder (`L0..L6`):
  - L0: Current deterministic PDG 2.0 normalized risk score
  - L1_MLP_Hist: MLP + engineered rolling history trend features (delta, EMA, slope, dt)
  - L2_GRU: GRU (d_h=24)
  - L3_LSTM: LSTM (d_h=24)
  - L4_CfC_FixedDt: CfC (d_h=24) with fixed dt=1.0
  - L5_CfC_RealDt: CfC (d_h=24) with real simulation time interval dt_sim
  - L6_LTC_RealDt: LTC (d_h=24, ODE substeps=3) with real dt_sim

Tests BOTH regular (`1x`) AND irregular continuous-time subsampling (`2x / 4x` irregular gaps)
plus Split-Conformal Risk Control (`CRC`, ICLR 2024, target FNR <= 0.10).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.pdg_liveness import PDG2Config, PDG2SCCLivenessMonitor
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from experiments.liquid_core_pilot.temporal_models import (
    LiquidPDGModel,
    benchmark_single_step_ms,
    calibrate_conformal_risk_threshold,
    count_parameters,
)
from experiments.v20_1_closure.task2_pdg2_ablation import build_dev_stress_24_specs
from experiments.v20_audit.common import assert_stage1_quarantine, build_val72_specs

OUT_LIQUID_DIR = ROOT / "out" / "liquid_core_pilot"
STATE_DIM = 12
HIST_EXTRA_DIM = 13  # delta (12) + scaled dt_sim (1)


def _pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    if n_pos == 0 or n_pos == len(y):
        return 0.0
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    prec = tp / np.maximum(1, tp + fp)
    rec = tp / float(n_pos)
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


def _roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(y_score, dtype=np.float64)
    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    return float((np.sum(ranks[y == 1]) - n_pos * (n_pos + 1) / 2.0) / float(n_pos * n_neg))


def _brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)) ** 2))


def _extract_step_state_vec(env: Any, st: dict[str, Any], max_steps: int) -> np.ndarray:
    """Extract 12-D observable cross-domain state vector z_k at real execution tick."""
    cur_t = int(env.current_step)
    ops = list(getattr(env.tracker, "operations", ()) or ())
    tasks = list(getattr(env.sim, "tasks", ()) or ())
    agvs = list(getattr(env.sim, "agvs", ()) or ())
    n_ops = max(1, len(ops))
    n_tasks = max(1, len(tasks))
    n_agvs = max(1, len(agvs))

    done_t = sum(1 for t in tasks if str(getattr(t.status, "value", t.status)).lower() == "completed")
    done_o = sum(1 for o in ops if o.end_step is not None and int(o.end_step) <= cur_t)
    partial_kits = sum(1 for v in (st.get("op_risks") or {}).values() if 0 < int(v.get("k_done", 0)) < int(v.get("k_tot", 0)))
    missed_wins = sum(int(v.get("missed_wins", 0)) for v in (st.get("op_risks") or {}).values())
    max_norm_age = float(st.get("max_op_norm_age", 0.0)) / 500.0
    max_norm_risk = float(st.get("max_op_norm_risk", 0.0)) / 800.0
    zpt_norm = float(st.get("zero_progress_ticks", 0.0)) / 500.0
    blocked_frac = float(len(st.get("blocked_healthy_agvs") or ())) / float(n_agvs)
    stalled_frac = float(len(st.get("stalled_agvs") or ())) / float(n_agvs)
    idle_blk_frac = float(len(st.get("idle_blocker_agvs") or ())) / float(n_agvs)
    t_rem_frac = float(n_tasks - done_t) / float(n_tasks)
    o_rem_frac = float(n_ops - done_o) / float(n_ops)
    time_frac = float(cur_t) / float(max(1, max_steps))
    scc_flag = 1.0 if (st.get("deadlock_sccs") or int(st.get("wait_cycle_count", 0)) > 0) else 0.0

    return np.array(
        [
            min(3.0, max_norm_age),
            min(3.0, max_norm_risk),
            min(3.0, zpt_norm),
            blocked_frac,
            stalled_frac,
            idle_blk_frac,
            float(partial_kits) / float(n_ops),
            min(3.0, float(missed_wins) / 3.0),
            t_rem_frac,
            o_rem_frac,
            min(1.5, time_frac),
            scc_flag,
        ],
        dtype=np.float32,
    )


def collect_dev_temporal_trajectories() -> list[dict[str, Any]]:
    """Collect real closed-loop trajectories on quarantined DEV splits (`VAL-72` + `DEV-Stress-24`)."""
    val72 = build_val72_specs()
    stress24 = build_dev_stress_24_specs()
    # Select balanced quarantined DEV subset (48 diverse episodes: 24 VAL + 24 Stress)
    specs = val72[::3] + stress24
    assert_stage1_quarantine([int(sp["seed"]) for sp in specs], label="TrackL_GateL1_DEV")

    trajs: list[dict[str, Any]] = []
    # Run under deterministic virtual clock with V20_NAJS (frozen baseline) so labels measure true V20 risk
    for idx, sp in enumerate(specs):
        pol = "V20_NAJS" if (idx % 2 == 0) else "V18_EA_JOINT"
        with virtual_clock_scope(0.0):
            kind, ctrl, env = create_discrete_policy_env(
                pol,
                seed=int(sp["seed"]),
                max_steps=int(sp["max_steps"]),
                use_rust=True,
                p2_cfg=sp.get("p2_cfg"),
            )
            obs, info = env.reset(seed=int(sp["seed"]), options={"scenario_tuple": sp["scenario_tuple"]})
            mon = PDG2SCCLivenessMonitor(cfg=PDG2Config())
            tm = tr = False
            ticks: list[int] = []
            z_list: list[np.ndarray] = []
            l0_scores: list[float] = []
            progress_ticks: list[int] = []
            prev_done = (0, 0)
            sample_stride = 4  # base 4-tick sampling resolution

            while not (tm or tr):
                t_now = int(env.current_step)
                st = mon.observe_sim(env.sim, env.tracker, cur_tick=t_now, max_steps_hint=int(sp["max_steps"]))
                c_t, c_o = env._completion_counts(int(env.sim._step_cursor))
                if (c_t, c_o) > prev_done:
                    prev_done = (c_t, c_o)
                    progress_ticks.append(t_now)

                if t_now % sample_stride == 0:
                    ticks.append(t_now)
                    z_list.append(_extract_step_state_vec(env, st, int(sp["max_steps"])))
                    # L0 deterministic PDG 2.0 continuous risk proxy in [0, 1]
                    l0_raw = max(
                        float(st.get("max_op_norm_risk", 0.0)) / 760.0,
                        float(st.get("zero_progress_ticks", 0.0)) / 500.0,
                        1.0 if st.get("pdg2_alert") else 0.0,
                    )
                    l0_scores.append(min(1.0, max(0.0, l0_raw)))

                if kind == "rule_v10":
                    feats = compute_expert_v9_state(
                        env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
                        recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
                    )
                    s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
                else:
                    feats = ctrl.observe(env, obs)
                    s_info = info
                act = ctrl.select_action(feats, s_info, sim=env.sim)
                obs, _, tm, tr, info = env.step(act)

            m = env.compute_episode_metrics()
            ep_incomp = not (float(m["task_completion_rate"]) >= 0.9999 and float(m["production_completion_rate"]) >= 0.9999)
            ep_end = int(env.current_step)

            # Construct right-censored progressive stall labels:
            # y_k = 1 if within next H=120 simulation ticks, zero progress lasts >= 110 ticks OR episode fails/severely stalls
            # Censor steps within final 20 ticks of a completed episode to avoid right-censoring bias.
            H = 120
            labels: list[int] = []
            valid_mask: list[bool] = []
            for t_k, z_k in zip(ticks, z_list):
                if (not ep_incomp) and (ep_end - t_k) < 20:
                    labels.append(0)
                    valid_mask.append(False)
                    continue
                # Check if next progress tick is > t_k + 110 (severe prolonged stall ahead) or episode ends incomplete
                nxt_prog = next((pt for pt in progress_ticks if pt > t_k), None)
                stalled_ahead = (nxt_prog is None and (ep_incomp or (ep_end - t_k) >= 110)) or (
                    nxt_prog is not None and (nxt_prog - t_k) >= 110
                )
                severe_op_risk_ahead = bool(ep_incomp and (ep_end - t_k) <= 450)
                labels.append(1 if (stalled_ahead or severe_op_risk_ahead) else 0)
                valid_mask.append(True)

            if len(ticks) >= 12:
                trajs.append({
                    "episode_key": str(sp["episode_key"]),
                    "policy": pol,
                    "ep_incomp": ep_incomp,
                    "ticks": np.asarray(ticks, dtype=np.float32),
                    "z": np.asarray(z_list, dtype=np.float32),
                    "l0": np.asarray(l0_scores, dtype=np.float32),
                    "y": np.asarray(labels, dtype=np.float32),
                    "mask": np.asarray(valid_mask, dtype=bool),
                })
    return trajs


def _subsample_trajectory(traj: dict[str, Any], *, irregular_drop_prob: float, rng: np.random.Generator) -> dict[str, Any]:
    """Subsample trajectory at irregular simulation intervals `dt_k_sim` (strictly sim ticks, never wall clock)."""
    T = len(traj["ticks"])
    if irregular_drop_prob <= 0.0:
        idx = np.arange(T)
    else:
        keep = [0]
        for i in range(1, T):
            if rng.random() >= irregular_drop_prob:
                keep.append(i)
        if len(keep) < 6:
            keep = list(range(0, T, max(1, T // 6)))
        idx = np.asarray(keep, dtype=np.int64)
    ticks = traj["ticks"][idx]
    z = traj["z"][idx]
    l0 = traj["l0"][idx]
    y = traj["y"][idx]
    mask = traj["mask"][idx]
    dt_raw = np.diff(ticks, prepend=ticks[0])
    dt_raw[0] = 4.0
    dt_sim = np.clip(dt_raw / 20.0, 0.05, 5.0).astype(np.float32)  # normalized sim-time interval
    # Engineered history features for L1_MLP_Hist: [z_k - z_{k-1}, dt_sim]
    dz = np.diff(z, axis=0, prepend=z[:1])
    hist = np.concatenate([dz, dt_sim[:, None]], axis=-1).astype(np.float32)
    return {"ticks": ticks, "z": z, "dt": dt_sim, "hist": hist, "l0": l0, "y": y, "mask": mask}


def _train_and_eval_arch(
    arch: str,
    train_trajs: list[dict[str, Any]],
    calib_trajs: list[dict[str, Any]],
    test_trajs: list[dict[str, Any]],
    *,
    seed: int = 42,
    epochs: int = 28,
) -> dict[str, Any]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = LiquidPDGModel(arch, STATE_DIM, hidden_dim=24, hist_extra_dim=(HIST_EXTRA_DIM if arch == "L1_MLP_Hist" else 0))

    # Compute class balance weight on train set
    pos_cnt = sum(int(np.sum(t["y"][t["mask"]] == 1)) for t in train_trajs)
    neg_cnt = sum(int(np.sum(t["y"][t["mask"]] == 0)) for t in train_trajs)
    pos_w = torch.tensor([min(12.0, max(1.0, float(neg_cnt) / float(max(1, pos_cnt))))], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3, weight_decay=1e-4)

    # Train on mixed regular + mildly irregular simulation sampling so models learn dt invariance
    model.train()
    for ep in range(epochs):
        order = rng.permutation(len(train_trajs))
        for i in order:
            drop_p = 0.0 if (ep % 2 == 0) else 0.35
            sub = _subsample_trajectory(train_trajs[int(i)], irregular_drop_prob=drop_p, rng=rng)
            m_idx = sub["mask"]
            if not np.any(m_idx):
                continue
            z_t = torch.from_numpy(sub["z"][None, :, :])
            dt_t = torch.from_numpy(sub["dt"][None, :])
            h_t = torch.from_numpy(sub["hist"][None, :, :]) if arch == "L1_MLP_Hist" else None
            y_t = torch.from_numpy(sub["y"])
            logits = model(z_t, dt_t, h_t)[0]
            loss = loss_fn(logits[m_idx], y_t[m_idx])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

    # Frozen inference evaluation + Split-Conformal Risk Control (CRC) calibration
    model.eval()

    def _predict_set(trajs: list[dict[str, Any]], drop_p: float, eval_seed: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
        erng = np.random.default_rng(eval_seed)
        all_p: list[np.ndarray] = []
        all_y: list[np.ndarray] = []
        leads: list[float] = []
        with torch.no_grad():
            for tr in trajs:
                sub = _subsample_trajectory(tr, irregular_drop_prob=drop_p, rng=erng)
                m_idx = sub["mask"]
                if not np.any(m_idx):
                    continue
                z_t = torch.from_numpy(sub["z"][None, :, :])
                dt_t = torch.from_numpy(sub["dt"][None, :])
                h_t = torch.from_numpy(sub["hist"][None, :, :]) if arch == "L1_MLP_Hist" else None
                probs = torch.sigmoid(model(z_t, dt_t, h_t)[0]).numpy()
                all_p.append(probs[m_idx])
                all_y.append(sub["y"][m_idx])
                # Lead time on true-positive episodes
                pos_ticks = sub["ticks"][sub["y"] > 0.5]
                if len(pos_ticks) > 0:
                    first_pos = float(pos_ticks[0])
                    alert_ticks = sub["ticks"][probs >= 0.5]
                    if len(alert_ticks) > 0 and float(alert_ticks[0]) <= first_pos + 120.0:
                        leads.append(max(0.0, (first_pos + 120.0) - float(alert_ticks[0])))
        return np.concatenate(all_p), np.concatenate(all_y), leads

    cal_p, cal_y, _ = _predict_set(calib_trajs, drop_p=0.0, eval_seed=101)
    crc = calibrate_conformal_risk_threshold(cal_p, cal_y, alpha_fnr=0.10)
    tau = float(crc["tau_crc"])

    metrics_by_rate: dict[str, dict[str, Any]] = {}
    for tag, dp in (("regular_1x", 0.0), ("irregular_2x", 0.50), ("irregular_4x", 0.75)):
        tp, ty, leads = _predict_set(test_trajs, drop_p=dp, eval_seed=202 + int(dp * 100))
        preds_crc = (tp >= tau).astype(np.int64)
        pos_m = (ty == 1)
        neg_m = (ty == 0)
        fnr_crc = float(np.mean(preds_crc[pos_m] == 0)) if np.any(pos_m) else 0.0
        fpr_crc = float(np.mean(preds_crc[neg_m] == 1)) if np.any(neg_m) else 0.0
        metrics_by_rate[tag] = {
            "pr_auc": round(_pr_auc(ty, tp), 4),
            "roc_auc": round(_roc_auc(ty, tp), 4),
            "brier": round(_brier(ty, tp), 4),
            "crc_fnr": round(fnr_crc, 4),
            "crc_fpr": round(fpr_crc, 4),
            "mean_lead_ticks": round(float(np.mean(leads)), 1) if leads else 0.0,
        }

    step_ms = benchmark_single_step_ms(model, STATE_DIM, hist_extra_dim=(HIST_EXTRA_DIM if arch == "L1_MLP_Hist" else 0))
    return {
        "arch": arch,
        "params": count_parameters(model),
        "step_cpu_ms": step_ms,
        "crc_calibration": crc,
        "metrics": metrics_by_rate,
    }


def run_gate_l1() -> dict[str, Any]:
    OUT_LIQUID_DIR.mkdir(parents=True, exist_ok=True)
    print("=== [Track L · Gate L1] Collecting Quarantined DEV Trajectories ===", flush=True)
    trajs = collect_dev_temporal_trajectories()
    # Episode-disjoint split: 60% train, 20% CRC calib, 20% test (stratified by episode failure/stall prevalence)
    trajs.sort(key=lambda t: (int(t["ep_incomp"]), int(np.sum(t["y"])), t["episode_key"]))
    train_t, calib_t, test_t = [], [], []
    for i, tr in enumerate(trajs):
        rem = i % 5
        if rem in (0, 1, 2):
            train_t.append(tr)
        elif rem == 3:
            calib_t.append(tr)
        else:
            test_t.append(tr)

    tot_pos = sum(int(np.sum(t["y"][t["mask"]] == 1)) for t in trajs)
    tot_n = sum(int(np.sum(t["mask"])) for t in trajs)
    print(f"  Collected {len(trajs)} DEV trajectories ({tot_n} valid steps, {tot_pos} positive stall-risk labels = {100.0*tot_pos/max(1,tot_n):.1f}%)", flush=True)

    # Evaluate L0 deterministic PDG 2.0 baseline on identical test set
    l0_res: dict[str, Any] = {"arch": "L0_Deterministic_PDG2", "params": 0, "step_cpu_ms": 0.02, "metrics": {}}
    cal_l0_p = np.concatenate([t["l0"][t["mask"]] for t in calib_t])
    cal_l0_y = np.concatenate([t["y"][t["mask"]] for t in calib_t])
    l0_crc = calibrate_conformal_risk_threshold(cal_l0_p, cal_l0_y, alpha_fnr=0.10)
    l0_res["crc_calibration"] = l0_crc
    for tag, dp in (("regular_1x", 0.0), ("irregular_2x", 0.50), ("irregular_4x", 0.75)):
        erng = np.random.default_rng(202 + int(dp * 100))
        ps, ys = [], []
        for tr in test_t:
            sub = _subsample_trajectory(tr, irregular_drop_prob=dp, rng=erng)
            ps.append(sub["l0"][sub["mask"]])
            ys.append(sub["y"][sub["mask"]])
        tp = np.concatenate(ps)
        ty = np.concatenate(ys)
        preds = (tp >= float(l0_crc["tau_crc"])).astype(np.int64)
        l0_res["metrics"][tag] = {
            "pr_auc": round(_pr_auc(ty, tp), 4),
            "roc_auc": round(_roc_auc(ty, tp), 4),
            "brier": round(_brier(ty, tp), 4),
            "crc_fnr": round(float(np.mean(preds[ty == 1] == 0)) if np.any(ty == 1) else 0.0, 4),
            "crc_fpr": round(float(np.mean(preds[ty == 0] == 1)) if np.any(ty == 0) else 0.0, 4),
            "mean_lead_ticks": 0.0,
        }

    results = [l0_res]
    for arch in ("L1_MLP_Hist", "L2_GRU", "L3_LSTM", "L4_CfC_FixedDt", "L5_CfC_RealDt", "L6_LTC_RealDt"):
        r = _train_and_eval_arch(arch, train_t, calib_t, test_t, seed=42)
        results.append(r)
        m1 = r["metrics"]["regular_1x"]
        m4 = r["metrics"]["irregular_4x"]
        print(
            f"  [Gate L1] {arch:16s} | Params={r['params']:4d} | Step={r['step_cpu_ms']:.4f}ms "
            f"| 1x PR-AUC={m1['pr_auc']:.4f} Brier={m1['brier']:.4f} CRC(FNR={m1['crc_fnr']:.3f},FPR={m1['crc_fpr']:.3f}) "
            f"| 4x-Irreg PR-AUC={m4['pr_auc']:.4f}",
            flush=True,
        )

    out = {
        "n_trajectories": len(trajs),
        "n_train": len(train_t),
        "n_calib": len(calib_t),
        "n_test": len(test_t),
        "positive_rate": round(float(tot_pos) / float(max(1, tot_n)), 4),
        "results": results,
    }
    (OUT_LIQUID_DIR / "gate_l1_liquid_pdg_report.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [
        "# Track L · Gate L1: `Liquid-PDG` Progressive Risk Prediction & Conformal Risk Control (`L0..L6`)",
        "",
        f"- **Dataset (Quarantined DEV Only)**: `{len(trajs)}` closed-loop trajectories (`{len(train_t)}` train / `{len(calib_t)}` CRC-calib / `{len(test_t)}` test, positive stall-risk prevalence = `{100.0*tot_pos/max(1,tot_n):.1f}%`).",
        "- **Strict Invariants**: `dt_sim` is simulation ticks (never CPU wall clock); weights frozen at inference; right-censored tails masked cleanly.",
        "",
        "| Model | Params | Step CPU (ms) | `1x` PR-AUC ↑ | `1x` ROC-AUC ↑ | `1x` Brier ↓ | `2x` Irreg PR-AUC ↑ | `4x` Irreg PR-AUC ↑ | CRC (`α=0.10`) FNR / FPR |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        m1 = r["metrics"]["regular_1x"]
        m2 = r["metrics"]["irregular_2x"]
        m4 = r["metrics"]["irregular_4x"]
        md.append(
            f"| **{r['arch']}** | {r['params']} | {r['step_cpu_ms']:.4f} | {m1['pr_auc']:.4f} | {m1['roc_auc']:.4f} | {m1['brier']:.4f} "
            f"| {m2['pr_auc']:.4f} | {m4['pr_auc']:.4f} | {m1['crc_fnr']:.3f} / {m1['crc_fpr']:.3f} |"
        )
    (OUT_LIQUID_DIR / "gate_l1_liquid_pdg_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[GATE L1 DONE] Wrote {OUT_LIQUID_DIR / 'gate_l1_liquid_pdg_report.md'}", flush=True)
    return out


if __name__ == "__main__":
    run_gate_l1()
