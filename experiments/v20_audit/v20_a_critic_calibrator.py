"""Phase V20-A Task A2: Train & validate Calibrated Long-Horizon Value Critic V_phi (`TRAIN` vs disjoint `VAL`)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from experiments.train_v19_bc import (
    TRAIN_4TOPO_SEEDS,
    TRAIN_WORKSHOP_SEEDS,
    VAL_4TOPO_SEEDS,
    VAL_WORKSHOP_SEEDS,
    _build_4topo_episode_spec,
    _build_workshop_episode_spec,
)
from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    RecordedStep,
    assert_stage1_quarantine,
    make_v18_external_proposer,
    rollout_48_and_optional_full,
)
from experiments.v20_audit.critic_model import (
    CRITIC_FEAT_DIM,
    CRITIC_FEAT_NAMES,
    NumpyValueCritic,
    build_keep_baseline_stats,
    extract_critic_features,
)
from experiments.v20_audit.proposer_variants import ExpertSuiteEvaluator
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES

CRITIC_EXPERTS: tuple[str, ...] = (
    "E0_KEEP",
    "E1_V18_Regret2",
    "E2_AHG_Full",
    "E3_AHG_Staged",
    "E4_V18_PGMPO",
    "E6_V18_Polish",
    "E7_CausalCone_VOI",
    "E9_Local_CPSAT",
)

CRITIC_NPZ_PATH = OUT_AUDIT_DIR / "v20_a_critic_weights.npz"
CRITIC_DATASET_NPZ = OUT_AUDIT_DIR / "v20_a_critic_dataset.npz"

_WORKER_SUITE: ExpertSuiteEvaluator | None = None


def _init_rollout_worker() -> None:
    """Initialize one read-only ExpertSuiteEvaluator once per spawned rollout worker."""
    global _WORKER_SUITE
    _WORKER_SUITE = ExpertSuiteEvaluator(seed=0)


def _process_episode_full_worker(
    task: tuple[int, dict[str, Any], str, str],
) -> tuple[int, str, list[list[tuple[np.ndarray, float, float, float, float]]]]:
    """Collect up to 6 stratified replan snapshots in one episode and run full-continuation rollouts."""
    ep_order, sp, src_pol, grp = task
    global _WORKER_SUITE
    if _WORKER_SUITE is None:
        _WORKER_SUITE = ExpertSuiteEvaluator(seed=0)

    sc_copy = copy.deepcopy(sp["scenario_tuple"])
    p2_copy = copy.deepcopy(sp["p2_cfg"])
    seed = int(sp["seed"])
    kind, ctrl, env = create_discrete_policy_env(
        src_pol,
        topology=str(sc_copy[0]),
        seed=seed,
        max_steps=int(sp["max_steps"]),
        use_rust=True,
        p2_cfg=p2_copy,
    )
    raw_ext = make_v18_external_proposer(env_ref=env) if src_pol == "V18_EA_JOINT" else env.external_proposer
    assert raw_ext is not None
    prefix_steps: list[RecordedStep] = []
    snaps_ep: list[dict[str, Any]] = []

    def _hook(snap: NativeSnapshot, net: Any, *, budget_ms: float = 35.0) -> CandidateProposal:
        prop = raw_ext(snap, net, budget_ms=budget_ms)
        prefix_steps.append(RecordedStep(step_before=int(snap.tick), action=-1, triggered_replan=True, proposal=copy.deepcopy(prop)))
        snaps_ep.append(
            {
                "network": copy.deepcopy(env.sim.network),
                "snapshot": copy.deepcopy(snap),
                "prefix_steps": [copy.copy(x) for x in prefix_steps[:-1]],
                "action_at_tick": -1,
            }
        )
        return prop

    env.external_proposer = _hook
    with virtual_clock_scope(poll_ms=0.002):
        obs, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
        tm = tr = False
        while not (tm or tr):
            if kind == "rule_v10":
                f = compute_expert_v9_state(
                    env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
                    recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
                )
                si = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
            else:
                f = ctrl.observe(env, obs)
                si = info
            act = int(ctrl.select_action(f, si, sim=env.sim))
            sb = int(env.current_step)
            bn = len(prefix_steps)
            obs, _, tm, tr, info = env.step(act)
            if len(prefix_steps) > bn:
                prefix_steps[-1].action = act
                if env.decision_logs:
                    prefix_steps[-1].propose_ms = float(env.decision_logs[-1].get("t_propose_ms", 0.0))
                snaps_ep[-1]["action_at_tick"] = act
            else:
                prefix_steps.append(RecordedStep(step_before=sb, action=act, triggered_replan=False, proposal=None))

    if len(snaps_ep) > 6:
        picks = sorted({int(round(q * (len(snaps_ep) - 1))) for q in np.linspace(0.0, 1.0, 6)})
        snaps_ep = [snaps_ep[k] for k in picks]

    snap_blocks: list[list[tuple[np.ndarray, float, float, float, float]]] = []
    for item in snaps_ep:
        snap: NativeSnapshot = item["snapshot"]
        net = item["network"]
        clf = classify_commitments(snap)
        graph = extract_v19_hetero_graph(snap, net, clf=clf)
        if not graph.rel_tids:
            continue
        arts = _WORKER_SUITE.evaluate_all_experts(snap, net, expert_ids=CRITIC_EXPERTS)
        keep_art = arts["E0_KEEP"]
        if not (keep_art.eval_res_b2 and keep_art.eval_res_b2.valid and keep_art.ja_unified_score is not None):
            continue
        rob_cur_h = {r.robot_id: r.cur_task for r in snap.robots if r.cur_task and r.robot_id not in clf.disturbed_robots}
        sim_h = int(keep_art.ja_proxy_details.get("sim_horizon", max(600, int(snap.tick) + 300)))
        keep_stats = build_keep_baseline_stats(snap, graph, keep_art.eval_res_b2, float(keep_art.ja_unified_score), rob_cur_h, sim_h)

        dedup_ro: dict[str, dict[str, Any]] = {}
        cands: list[tuple[np.ndarray, float, float, float, float]] = []
        for eid in CRITIC_EXPERTS:
            art = arts[eid]
            if not (art.eval_res_b2 and art.eval_res_b2.valid and art.ja_unified_score is not None):
                continue
            sig = art.ensure_sig()
            if sig not in dedup_ro:
                prop_opt = None if eid == "E0_KEEP" else art.to_proposal(
                    policy_id=f"CR_{eid}", snapshot=snap, clf=clf, eval_res=art.eval_res_b2, eval_cnt=1, t0=0.0,
                    budget_ms=35.0, fallback_keep=False, include_eval_seq_when_unwindowed=False,
                )
                dedup_ro[sig] = rollout_48_and_optional_full(
                    sp, item["prefix_steps"], int(snap.tick), prop_opt,
                    action_at_target=int(item["action_at_tick"]), base_env_policy=src_pol,
                    rollout_horizon=48, full_continuation=True,
                )
            ro_res = dedup_ro[sig]
            fv = extract_critic_features(
                snap, net, graph, art.eval_res_b2, float(art.ja_unified_score),
                art.assign, art.bays, art.crews, art.task_orders, keep_stats, rob_cur_h, sim_h,
                is_keep_cand=(eid == "E0_KEEP"),
            )
            cands.append((
                fv,
                float(ro_res["full_net_obj"]),
                float(ro_res["j48_with_residual"]),
                float(art.ja_unified_score),
                float(keep_stats.ja_unified),
            ))
        if len(cands) >= 3:
            snap_blocks.append(cands)

    return ep_order, grp, snap_blocks


def _collect_parallel_split_dataset(
    specs: list[dict[str, Any]],
    *,
    split_name: str,
    max_total: int,
    seed_rng: int,
    max_workers: int,
) -> dict[str, np.ndarray]:
    """Build balanced snapshot rollout dataset with true full-continuation returns using parallel workers."""
    rng = np.random.default_rng(seed_rng)
    by_grp: dict[str, list[dict[str, Any]]] = {"4topo": [], "track1": [], "track2": []}
    for sp in specs:
        by_grp[str(sp["suite_group"])].append(sp)
    for grp, lst in by_grp.items():
        perm = rng.permutation(len(lst))
        by_grp[grp] = [lst[i] for i in perm]

    targets = {
        "4topo": int(max_total * 0.40),
        "track1": int(max_total * 0.30),
        "track2": max_total - int(max_total * 0.40) - int(max_total * 0.30),
    }

    tasks: list[tuple[int, dict[str, Any], str, str]] = []
    ep_counter = 0
    for grp, grp_specs in by_grp.items():
        # ~5.5 snapshots expected per episode -> take tight buffer cleanly
        need_eps = min(len(grp_specs), max(targets[grp] // 5 + 4, int(math.ceil(targets[grp] / 4.5))))
        for sp in grp_specs[:need_eps]:
            src_pol = "V19_ADAPTIVE_RL" if (ep_counter % 2 == 0) else "V18_EA_JOINT"
            tasks.append((ep_counter, sp, src_pol, grp))
            ep_counter += 1

    print(f"  [{split_name}] Dispatching {len(tasks)} episodes across {max_workers} workers (full_continuation=True)...", flush=True)
    ep_results: list[tuple[int, str, list[list[tuple[np.ndarray, float, float, float, float]]]]] = []
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init_rollout_worker) as pool:
        for done_i, res in enumerate(pool.map(_process_episode_full_worker, tasks), start=1):
            ep_results.append(res)
            if done_i % 15 == 0 or done_i == len(tasks):
                print(f"  [{split_name}] Completed {done_i}/{len(tasks)} episodes...", flush=True)

    ep_results.sort(key=lambda x: x[0])
    grp_snaps_taken: dict[str, int] = {"4topo": 0, "track1": 0, "track2": 0}
    suite_code_map = {"4topo": 0, "track1": 1, "track2": 2}

    feats_all: list[np.ndarray] = []
    y_full_all: list[float] = []
    y48_all: list[float] = []
    ja_u_all: list[float] = []
    keep_ja_all: list[float] = []
    snap_id_all: list[int] = []
    ep_id_all: list[int] = []
    suite_all: list[int] = []

    snap_cursor = 0
    for ep_order, grp, snap_blocks in ep_results:
        for cands in snap_blocks:
            if grp_snaps_taken[grp] >= targets[grp]:
                break
            keep_full = cands[0][1]
            keep_j48 = cands[0][2]
            for fv, f_v, j48_v, ja_v, k_ja in cands:
                feats_all.append(fv)
                y_full_all.append(float(f_v - keep_full))
                y48_all.append(float(j48_v - keep_j48))
                ja_u_all.append(ja_v)
                keep_ja_all.append(k_ja)
                snap_id_all.append(snap_cursor)
                ep_id_all.append(ep_order)
                suite_all.append(suite_code_map[grp])
            grp_snaps_taken[grp] += 1
            snap_cursor += 1

    print(f"  [{split_name}] Collected {snap_cursor} snaps ({grp_snaps_taken}), {len(feats_all)} candidate rows.", flush=True)
    return {
        "X": np.stack(feats_all, axis=0).astype(np.float32),
        "y_delta_full": np.asarray(y_full_all, dtype=np.float32),
        "y_delta48": np.asarray(y48_all, dtype=np.float32),
        "ja_unified": np.asarray(ja_u_all, dtype=np.float32),
        "keep_ja": np.asarray(keep_ja_all, dtype=np.float32),
        "snap_id": np.asarray(snap_id_all, dtype=np.int32),
        "ep_id": np.asarray(ep_id_all, dtype=np.int32),
        "suite": np.asarray(suite_all, dtype=np.int32),
    }


def _evaluate_ranking_metrics(
    X: np.ndarray,
    y_target: np.ndarray,
    scores: np.ndarray,
    snap_id: np.ndarray,
    suite: np.ndarray,
) -> dict[str, Any]:
    rhos: list[float] = []
    concs: list[float] = []
    regrets: list[float] = []
    regret_by_suite: dict[int, list[float]] = {0: [], 1: [], 2: []}
    rho_by_suite: dict[int, list[float]] = {0: [], 1: [], 2: []}

    for sid in np.unique(snap_id):
        mask = snap_id == sid
        y = y_target[mask]
        s = scores[mask]
        su = int(suite[mask][0])
        if len(y) < 3 or float(np.std(y)) < 1e-6 or float(np.std(s)) < 1e-6:
            continue
        rv = spearmanr(s, y).statistic
        if rv is not None and not np.isnan(rv):
            rhos.append(float(rv))
            rho_by_suite[su].append(float(rv))
        reg = float(y[int(np.argmin(s))] - np.min(y))
        regrets.append(reg)
        regret_by_suite[su].append(reg)
        con = tot = 0
        for i in range(len(y)):
            for j in range(i + 1, len(y)):
                if abs(y[i] - y[j]) > 1e-6 and abs(s[i] - s[j]) > 1e-6:
                    tot += 1
                    if (y[i] < y[j]) == (s[i] < s[j]):
                        con += 1
        if tot > 0:
            concs.append(con / float(tot))

    suite_names = {0: "4topo", 1: "track1", 2: "track2"}
    return {
        "n_scored_snapshots": len(regrets),
        "mean_spearman_rho_48": round(float(np.mean(rhos)), 4) if rhos else 0.0,
        "pairwise_concordance_48": round(float(np.mean(concs)), 4) if concs else 0.0,
        "mean_top1_regret_48": round(float(np.mean(regrets)), 3) if regrets else 0.0,
        "by_suite": {
            suite_names[k]: {
                "rho_48": round(float(np.mean(rho_by_suite[k])), 4) if rho_by_suite[k] else 0.0,
                "regret_48": round(float(np.mean(regret_by_suite[k])), 3) if regret_by_suite[k] else 0.0,
            }
            for k in (0, 1, 2)
        },
    }


def _fit_expected_regret_linear(
    Xs: np.ndarray,
    ys: np.ndarray,
    ss: np.ndarray,
    *,
    y_scale: float,
    clip_y: float,
    l2: float,
    rw: float,
    pwk: float,
    hw: float,
    temp: float,
    jmix: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Fit PyTorch Expected-Regret + Pairwise Logistic + Huber linear critic on vectorized snapshot batches."""
    mu = np.mean(Xs, axis=0).astype(np.float32)
    sd = np.maximum(1e-4, np.std(Xs, axis=0)).astype(np.float32)
    Xtrn = torch.tensor((Xs - mu) / sd, dtype=torch.float32, device=device)

    groups = [g for g in (np.where(ss == sid)[0] for sid in np.unique(ss)) if len(g) >= 2 and float(ys[g].std()) > 1e-6]
    max_m = max(len(g) for g in groups)
    idx_pad = np.zeros((len(groups), max_m), dtype=np.int64)
    mask_pad = np.zeros((len(groups), max_m), dtype=bool)
    reg_mat = np.zeros((len(groups), max_m), dtype=np.float32)
    pi: list[int] = []
    pj: list[int] = []
    pw: list[float] = []
    for k, g in enumerate(groups):
        idx_pad[k, : len(g)] = g
        mask_pad[k, : len(g)] = True
        reg_mat[k, : len(g)] = np.clip((ys[g] - ys[g].min()) / y_scale, 0.0, 6.0)
        for a in range(len(g)):
            for b in range(a + 1, len(g)):
                df = float(ys[g[a]] - ys[g[b]])
                if abs(df) < 5.0:
                    continue
                ia, ib = int(g[a]), int(g[b])
                if df < 0:
                    pi.append(ia)
                    pj.append(ib)
                else:
                    pi.append(ib)
                    pj.append(ia)
                pw.append(min(3.0, math.log1p(abs(df) / (0.16 * y_scale))))

    idx_t = torch.tensor(idx_pad, dtype=torch.long, device=device)
    m_t = torch.tensor(mask_pad, dtype=torch.bool, device=device)
    reg_t = torch.tensor(reg_mat, dtype=torch.float32, device=device)
    pi_t = torch.tensor(pi, dtype=torch.long, device=device)
    pj_t = torch.tensor(pj, dtype=torch.long, device=device)
    pw_t = torch.tensor(pw, dtype=torch.float32, device=device)
    y_c = torch.tensor(np.clip(ys / y_scale, -clip_y, clip_y), dtype=torch.float32, device=device)

    lin = nn.Linear(Xs.shape[1], 1).to(device)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    opt = torch.optim.Adam(lin.parameters(), lr=0.04)
    for _ in range(320):
        opt.zero_grad()
        sc = lin(Xtrn)[:, 0]
        lh = F.huber_loss(sc, y_c, delta=1.5)
        g_sc = sc[idx_t].masked_fill(~m_t, 1e9)
        probs = F.softmax(-g_sc / temp, dim=1) * m_t.float()
        l_reg = torch.mean(torch.sum(probs * reg_t, dim=1))
        l_pair = torch.mean(pw_t * F.softplus((sc[pi_t] - sc[pj_t]) * 1.5)) if pi else torch.tensor(0.0, device=device)
        loss = rw * l_reg + pwk * l_pair + hw * lh + l2 * torch.sum(lin.weight ** 2)
        loss.backward()
        opt.step()

    w_out = lin.weight.detach().cpu().numpy()[0].astype(np.float32)
    w_out[12] = 0.0
    w_out[23] = 0.0
    b_out = float(lin.bias.detach().cpu().item())
    return mu, sd, w_out, b_out, float(jmix)


def train_and_validate_critic(*, refresh_dataset: bool = False) -> dict[str, Any]:
    """Build strictly disjoint TRAIN/VAL full-continuation datasets, select hyperparams by blocked-episode TRAIN CV, and validate."""
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    need_build = refresh_dataset or (not CRITIC_DATASET_NPZ.is_file())
    if not need_build:
        d = np.load(str(CRITIC_DATASET_NPZ))
        if "tr_y_delta_full" not in d.files or "va_ep_id" not in d.files:
            need_build = True
        else:
            print(f"[Task A2] Loading cached full-continuation dataset from {CRITIC_DATASET_NPZ}...", flush=True)
            tr = {k[3:]: d[k] for k in d.files if k.startswith("tr_")}
            va = {k[3:]: d[k] for k in d.files if k.startswith("va_")}

    if need_build:
        tr_4t = TRAIN_4TOPO_SEEDS[:5]
        tr_ws = TRAIN_WORKSHOP_SEEDS[:8]
        val_4t = VAL_4TOPO_SEEDS
        val_ws = VAL_WORKSHOP_SEEDS
        assert_stage1_quarantine(list(tr_4t) + list(tr_ws), label="Critic_TRAIN")
        assert_stage1_quarantine(list(val_4t) + list(val_ws), label="Critic_VAL")
        assert not (set(tr_4t) & set(val_4t)) and not (set(tr_ws) & set(val_ws)), "TRAIN/VAL seed leakage!"

        tr_specs: list[dict[str, Any]] = []
        for s in tr_4t:
            for lay in TOPO4_LAYOUTS:
                for reg in TOPO4_REGIMES:
                    sp = _build_4topo_episode_spec(lay, s, reg)
                    sp["suite_group"] = "4topo"
                    tr_specs.append(sp)
        for tr_name in ("track1", "track2"):
            for s in tr_ws:
                for reg in WORKSHOP_REGIMES:
                    sp = _build_workshop_episode_spec(tr_name, s, reg)
                    sp["suite_group"] = tr_name
                    tr_specs.append(sp)

        va_specs: list[dict[str, Any]] = []
        for s in val_4t:
            for lay in TOPO4_LAYOUTS:
                for reg in TOPO4_REGIMES:
                    sp = _build_4topo_episode_spec(lay, s, reg)
                    sp["suite_group"] = "4topo"
                    va_specs.append(sp)
        for tr_name in ("track1", "track2"):
            for s in val_ws:
                for reg in WORKSHOP_REGIMES:
                    sp = _build_workshop_episode_spec(tr_name, s, reg)
                    sp["suite_group"] = tr_name
                    va_specs.append(sp)

        workers = min(10, max(2, (os.cpu_count() or 4) // 3))
        print("[Task A2] Building TRAIN & disjoint VAL datasets with true full-continuation rollouts...", flush=True)
        tr = _collect_parallel_split_dataset(tr_specs, split_name="TRAIN", max_total=360, seed_rng=101, max_workers=workers)
        va = _collect_parallel_split_dataset(va_specs, split_name="VAL", max_total=180, seed_rng=202, max_workers=workers)
        save_kw = {f"tr_{k}": v for k, v in tr.items()}
        save_kw.update({f"va_{k}": v for k, v in va.items()})
        np.savez_compressed(str(CRITIC_DATASET_NPZ), **save_kw)

    X_tr, y_tr_full, y_tr_48, s_tr, ep_tr, su_tr = (
        tr["X"].copy(), tr["y_delta_full"], tr["y_delta48"], tr["snap_id"], tr["ep_id"], tr["suite"]
    )
    X_va, y_va_full, y_va_48, s_va, su_va = (
        va["X"].copy(), va["y_delta_full"], va["y_delta48"], va["snap_id"], va["suite"]
    )
    X_tr[:, 12] = 0.0
    X_tr[:, 23] = 0.0
    X_va[:, 12] = 0.0
    X_va[:, 23] = 0.0

    # Robust hybrid target combining within-snapshot rank and asinh-compressed terminal cost
    # to prevent rare +-34,000 downstream timeout spikes from flipping physical weights.
    y_rank_tr = np.zeros_like(y_tr_full)
    for sid in np.unique(s_tr):
        m = s_tr == sid
        if m.sum() > 1 and float(y_tr_full[m].std()) > 1e-6:
            r = np.argsort(np.argsort(y_tr_full[m])).astype(np.float32)
            y_rank_tr[m] = (r - r.mean()) / max(1.0, r.std())
    y_asinh_tr = np.clip(np.arcsinh(y_tr_full / 200.0).astype(np.float32) / 2.5, -2.0, 2.0)
    y_hyb_tr = 0.5 * y_rank_tr + 0.5 * y_asinh_tr

    y_scale_4t = 120.0
    y_scale_ws = 400.0
    torch.manual_seed(20261124)

    # Select regularization strictly via Episode-Blocked GroupKFold on TRAIN
    gkf = GroupKFold(n_splits=5)
    m4_idx = np.where(su_tr == 0)[0]
    mw_idx = np.where(su_tr != 0)[0]

    best_cfg = (400.0, 0.8, 0.5)
    best_cv_score = -1e18
    for r_mix in (0.5, 0.7, 1.0):
        tgt = r_mix * y_rank_tr + (1.0 - r_mix) * y_asinh_tr
        for alpha in (200.0, 400.0, 800.0):
            for jmix in (0.4, 0.8, 1.2):
                oof = np.zeros_like(y_tr_full)
                for tr_i, va_i in gkf.split(X_tr, y_tr_full, groups=ep_tr):
                    mu, sd = X_tr[tr_i].mean(0), np.maximum(1e-4, X_tr[tr_i].std(0))
                    reg = Ridge(alpha=alpha).fit((X_tr[tr_i] - mu) / sd, tgt[tr_i])
                    oof[va_i] = reg.predict((X_tr[va_i] - mu) / sd) + jmix * X_tr[va_i, 0]
                m_f = _evaluate_ranking_metrics(X_tr, y_tr_full, oof, s_tr, su_tr)
                m_48 = _evaluate_ranking_metrics(X_tr, y_tr_48, oof, s_tr, su_tr)
                # Balanced OOF ranking + regret reduction across both full and +48t targets
                cv_sc = 0.6 * m_f["mean_spearman_rho_48"] + 0.4 * m_48["mean_spearman_rho_48"] - 0.0002 * m_f["mean_top1_regret_48"] - 0.005 * m_48["mean_top1_regret_48"]
                if cv_sc > best_cv_score:
                    best_cv_score = cv_sc
                    best_cfg = (alpha, jmix, r_mix)

    alpha_sel, jmix_sel, rmix_sel = best_cfg
    tgt_final = rmix_sel * y_rank_tr + (1.0 - rmix_sel) * y_asinh_tr
    mu_4t = np.mean(X_tr, axis=0).astype(np.float32)
    sd_4t = np.maximum(1e-4, np.std(X_tr, axis=0)).astype(np.float32)
    ridge_fin = Ridge(alpha=alpha_sel).fit((X_tr - mu_4t) / sd_4t, tgt_final)
    w_4t = ridge_fin.coef_.astype(np.float32)
    w_4t[12] = 0.0
    w_4t[23] = 0.0
    b_4t = float(ridge_fin.intercept_)
    jmix_4t = float(jmix_sel)

    # Also store regime-matched Workshop head (or shared robust parameters when shared generalizes best)
    mu_ws, sd_ws, w_ws, b_ws, jmix_ws = mu_4t, sd_4t, w_4t.copy(), b_4t, jmix_4t

    # Raw linear baseline on un-compressed y_full for comparison
    ridge_base = Ridge(alpha=150.0).fit(
        (X_tr - mu_4t) / sd_4t,
        np.clip(np.where(su_tr == 0, y_tr_full / 250.0, y_tr_full / 1000.0), -4.0, 4.0),
    )
    pred_ridge_va = ridge_base.predict((X_va - mu_4t) / sd_4t) * np.where(su_va == 0, 250.0, 1000.0)

    np.savez(
        str(CRITIC_NPZ_PATH),
        scale=np.array([y_scale_4t], dtype=np.float32),
        scale_ws=np.array([y_scale_ws], dtype=np.float32),
        mean_4t=mu_4t,
        std_4t=sd_4t,
        w_4t=w_4t,
        b_4t=np.array([b_4t], dtype=np.float32),
        jmix_4t=np.array([jmix_4t], dtype=np.float32),
        mean_ws=mu_ws,
        std_ws=sd_ws,
        w_ws=w_ws,
        b_ws=np.array([b_ws], dtype=np.float32),
        jmix_ws=np.array([jmix_ws], dtype=np.float32),
    )

    pred_ref_va = np.zeros(len(X_va), dtype=np.float32)
    is_ws = X_va[:, 22] >= 0.5
    pred_ref_va[~is_ws] = (((X_va[~is_ws] - mu_4t) / sd_4t) @ w_4t + b_4t + jmix_4t * X_va[~is_ws, 0]) * y_scale_4t
    pred_ref_va[is_ws] = (((X_va[is_ws] - mu_ws) / sd_ws) @ w_ws + b_ws + jmix_ws * X_va[is_ws, 0]) * y_scale_ws

    np_critic = NumpyValueCritic(CRITIC_NPZ_PATH)
    pred_np_va = np.array([np_critic.predict_delta_keep(va["X"][i]) for i in range(len(X_va))], dtype=np.float32)
    max_parity_err = float(np.max(np.abs(pred_np_va - pred_ref_va)))
    assert max_parity_err < 1e-2, f"NumPy critic parity mismatch: {max_parity_err}"

    m_ja_full = _evaluate_ranking_metrics(X_va, y_va_full, va["ja_unified"], s_va, su_va)
    m_ridge_full = _evaluate_ranking_metrics(X_va, y_va_full, pred_ridge_va, s_va, su_va)
    m_critic_full = _evaluate_ranking_metrics(X_va, y_va_full, pred_np_va, s_va, su_va)

    m_ja_48 = _evaluate_ranking_metrics(X_va, y_va_48, va["ja_unified"], s_va, su_va)
    m_critic_48 = _evaluate_ranking_metrics(X_va, y_va_48, pred_np_va, s_va, su_va)

    top_feats = sorted([(CRITIC_FEAT_NAMES[i], round(float(w_4t[i]), 4)) for i in range(CRITIC_FEAT_DIM)], key=lambda x: abs(x[1]), reverse=True)[:10]

    report = {
        "schema": "v20_a.critic_calibration.v4_robust_rank_asinh",
        "train_samples": int(len(X_tr)),
        "val_samples": int(len(X_va)),
        "train_episodes": int(len(np.unique(ep_tr))),
        "cv_selected_cfg": {"alpha": alpha_sel, "jmix": jmix_sel, "rank_mix": rmix_sel},
        "numpy_parity_max_abs_err": round(max_parity_err, 6),
        "val_metrics_vs_full_net_obj": {
            "JA_Unified_Baseline": m_ja_full,
            "Ridge_Linear_Baseline": m_ridge_full,
            "RobustRankAsinh_Critic": m_critic_full,
        },
        "val_metrics_vs_j48": {
            "JA_Unified_Baseline": m_ja_48,
            "RobustRankAsinh_Critic": m_critic_48,
        },
        "top_weights": top_feats,
        "gate_a_critic_j48_pass": bool(
            m_critic_48["mean_spearman_rho_48"] >= 0.62
            and m_critic_48["mean_top1_regret_48"] <= 0.75 * max(1e-6, m_ja_48["mean_top1_regret_48"])
        ),
        "gate_a_critic_full_improves_both": bool(
            m_critic_full["mean_spearman_rho_48"] > m_ja_full["mean_spearman_rho_48"]
            and m_critic_full["mean_top1_regret_48"] <= m_ja_full["mean_top1_regret_48"]
        ),
    }
    (OUT_AUDIT_DIR / "v20_a_critic_calibration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("[Task A2 Blocked-Episode CV Selection]", flush=True)
    print(f"  -> Best cfg: alpha={alpha_sel}, jmix={jmix_sel}, rank_mix={rmix_sel}", flush=True)
    print("[Task A2 Validation Metrics vs Disjoint VAL `full_net_obj`]", flush=True)
    for k, v in report["val_metrics_vs_full_net_obj"].items():
        print(f"  -> {k}: rho_full={v['mean_spearman_rho_48']}, conc={v['pairwise_concordance_48']}, top1_regret={v['mean_top1_regret_48']} ({v['by_suite']})", flush=True)
    print("[Task A2 Validation Metrics vs Disjoint VAL `+48t` (`j48_with_residual`)]", flush=True)
    for k, v in report["val_metrics_vs_j48"].items():
        print(f"  -> {k}: rho_48={v['mean_spearman_rho_48']}, conc={v['pairwise_concordance_48']}, top1_regret={v['mean_top1_regret_48']} ({v['by_suite']})", flush=True)
    print(f"  -> GateA J48 Pass={report['gate_a_critic_j48_pass']}, Full Improves={report['gate_a_critic_full_improves_both']}, parity_err={max_parity_err:.6f}", flush=True)
    return report


if __name__ == "__main__":
    train_and_validate_critic(refresh_dataset=False)
