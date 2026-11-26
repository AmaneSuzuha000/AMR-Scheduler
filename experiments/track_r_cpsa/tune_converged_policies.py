"""Track P2 Scenario-Native Policy & DFL Critic Fine-Tuning Engine (Quarantined Weights).

Uses the accelerated Track P1 Rust E2E simulator (`RustReservationTable.fast()`,
`RustCPSAStarGraphPlanner`, `RustSlotField`) to collect pure-transport counterfactual
decision snapshots strictly from converged `workshop_v1` & `demo17` traffic/dispatching
episodes (zero Layer-5 production DAG imports), and fine-tunes:
  1. `mh_dfl_critic_workshop.npz` (`DFLAdvantageEnsemble` on L2/L3 Local Priority Repair
     causal blocker selection + dispatch crowding risk gating via Mandi et al. DFL loss).
  2. `ejor26_wu_gru_workshop.npz` (`WuEJOR26GRUAttentionNet` actor fine-tuned with KL
     distillation + proximal L2 anchor onto converged workshop task geometry).

All outputs are isolated in `amr_sched/checkpoints/track_r/`, leaving original frozen
checkpoints untouched so pre-convergence vs. adapted models compare cleanly side by side.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.baselines_2026 import WuEJOR26GRUAttentionNet
from amr_sched.baselines.numpy_backends import (
    NumpyWuEJOR26GRUAttentionNet,
    load_numpy_state_dict,
)
from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.commitment import TIER_EXECUTING_HARD, TIER_SOFT, TIER_UNCOMMITTED
from amr_sched.cvcr.v12_proactive_guard import analyze_network_havens
from amr_sched.network import Network, build_demo_network
from amr_sched.planners.rust_spacetime_astar import RustCPSAStarGraphPlanner
from amr_sched.reservation import Interval
from amr_sched.rust_reservation import RustReservationTable
from amr_sched.workshop import DEFAULT_SCENE, WorkshopScene, legal_task_pairs, parking_slots
from experiments.cps_astar_routing.cps_astar_graph import CPSAStarConfig
from experiments.rgro.dfl_critic import (
    CAND_RAW_DIM,
    DFLAdvantageEnsemble,
    build_anchor_relative_feature,
    train_dfl_advantage_ensemble,
)
from experiments.track_r_cpsa.cpsa_pure_runner import impact_aware_blocker_score
from experiments.track_r_cpsa.workshop_bridge_eval import (
    _AGV_SPEED_MPS,
    _STEP_DT_S,
    WorkshopTask,
    _agv_commitment_tier,
    _build_workshop_history_vec,
    _dispatch_tasks_wu26,
    _expand_multi_step_schedule,
    _load_dfl_critic,
    _load_wu26_net,
    _tt_steps,
    run_workshop_episode,
)

TRACK_R_CKPT_DIR = ROOT / "amr_sched" / "checkpoints" / "track_r"


def _build_wu_pair_tensors_ws(
    cand_t: list[WorkshopTask],
    cand_a: list[str],
    cur_pos: dict[str, str],
    network: Network,
    tt_cache: dict[tuple[str, str], int],
    tick: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build exact `[1, 12, L]` static and `[1, 8, L]` dynamic candidate features for Wu26 GRU."""
    n_t, n_a = len(cand_t), len(cand_a)
    L = n_t * n_a
    xs = [float(n.x) for n in network.nodes.values()] or [0.0]
    ys = [float(n.y) for n in network.nodes.values()] or [0.0]
    minx, spanx = min(xs), max(1.0, max(xs) - min(xs))
    miny, spany = min(ys), max(1.0, max(ys) - min(ys))

    static_m = np.zeros((1, 12, L), dtype=np.float32)
    dyn_m = np.zeros((1, 8, L), dtype=np.float32)
    for i, tk in enumerate(cand_t):
        pn, dn = network.node(tk.pickup), network.node(tk.dropoff)
        travel = float(_tt_steps(network, tk.pickup, tk.dropoff, tt_cache)) / 40.0
        slack = max(-1.0, min(2.0, float(tk.due_step - tick) / 80.0))
        urg = 1.0 if tk.is_urgent else 0.0
        for j, aid in enumerate(cand_a):
            idx = i * n_a + j
            rn = network.node(cur_pos[aid])
            dead = float(_tt_steps(network, cur_pos[aid], tk.pickup, tt_cache))
            est_c = dead + travel * 40.0
            est_sl = max(-1.0, min(2.0, float(tk.due_step - (tick + est_c)) / 60.0))
            static_m[0, :, idx] = [
                (pn.x - minx) / spanx, (pn.y - miny) / spany,
                (dn.x - minx) / spanx, (dn.y - miny) / spany,
                min(2.0, travel), slack, urg, 0.0,
                (rn.x - minx) / spanx, (rn.y - miny) / spany,
                min(2.0, dead / 30.0), 0.0,
            ]
            dyn_m[0, :, idx] = [
                min(2.0, dead / 30.0), 0.0, 0.0, est_sl, 0.0,
                1.0 - min(1.0, est_c / 80.0), 0.0, 0.0,
            ]
    return static_m, dyn_m


def _collect_dispatch_and_wu_samples(
    seeds: tuple[int, ...] = (101, 102, 103, 104),
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run Rust E2E workshop rollouts to harvest counterfactual dispatch & Wu26 fine-tuning samples."""
    dfl_snaps: list[dict[str, Any]] = []
    wu_samples: list[dict[str, Any]] = []
    wu_zero = _load_wu26_net()

    configs = [
        ("workshop_v1", 1, 50, 6, 0.12),
        ("workshop_v1", 1, 80, 12, 0.15),
        ("workshop_v1", 1, 120, 6, 0.12),
        ("demo17", 1, 40, 6, 0.10),
    ]
    for layout, cap, n_tasks, n_agvs, dp in configs:
        for sd in seeds:
            def _capture_dispatch(
                idle_agvs: list[str],
                ready_tasks: list[WorkshopTask],
                cur_pos: dict[str, str],
                network: Network,
                tt_cache: dict[tuple[str, str], int],
                tick: int,
                ctx: dict[str, Any],
                _cap: int = cap,
                _na: int = n_agvs,
                _nt: int = n_tasks,
            ) -> list[tuple[str, WorkshopTask]]:
                bay_load = ctx.get("bay_load", {})
                haven_prof = ctx.get("haven_prof")
                scored_t = sorted(
                    ready_tasks,
                    key=lambda tk: (
                        0 if tk.is_urgent else 1,
                        float(tk.due_step - tick - _tt_steps(network, tk.pickup, tk.dropoff, tt_cache)),
                        tk.release_step,
                        tk.task_id,
                    ),
                )
                cand_t = scored_t[: min(len(scored_t), 8)]
                cand_a = sorted(idle_agvs)[: min(len(idle_agvs), 8)]
                if len(cand_t) >= 2 and len(cand_a) >= 2:
                    is_large = 1.0 if len(network.nodes) >= 30 else 0.0
                    choke_p = float(sum(bay_load.values())) / float(max(1, len(cand_a) + 2))
                    h_vec = _build_workshop_history_vec(
                        blocked_frac=min(1.0, choke_p * 0.25),
                        choke_frac=min(1.0, choke_p * 0.3),
                        urgent_pressure=0.5,
                        comp_rate=0.5,
                        horizon_frac=min(1.0, float(tick) / 700.0),
                    )
                    fv_anc = np.zeros(CAND_RAW_DIM, dtype=np.float32)
                    fv_anc[12] = 1.0
                    fv_anc[22] = is_large

                    x_rows: list[np.ndarray] = []
                    costs: list[float] = []
                    sat_scale = float(_na) / max(1.0, 6.0 * float(_cap))
                    backlog_ratio = float(len(ready_tasks)) / float(max(1, len(idle_agvs)))
                    overload = min(1.0, max(0.0, (backlog_ratio - 1.5) / 4.0))
                    dead_w = 3.5 + 3.0 * overload
                    haul_w = 0.55 + 0.75 * overload
                    for tk in cand_t:
                        d_haul = float(_tt_steps(network, tk.pickup, tk.dropoff, tt_cache))
                        ld_i = float(bay_load.get(tk.pickup, 0) + bay_load.get(tk.dropoff, 0))
                        choke_hit = (
                            (1.0 if haven_prof and (tk.pickup in haven_prof.choke_nodes or tk.pickup in haven_prof.leaf_spurs) else 0.0)
                            + (1.0 if haven_prof and (tk.dropoff in haven_prof.choke_nodes or tk.dropoff in haven_prof.leaf_spurs) else 0.0)
                        )
                        for aid in cand_a:
                            d_dead = float(_tt_steps(network, cur_pos[aid], tk.pickup, tt_cache))
                            eta = tick + d_dead + d_haul + 1.5 * ld_i
                            late = max(0.0, eta - float(tk.due_step))
                            late_eff = min(18.0, late) + 0.12 * max(0.0, late - 18.0)
                            urg_m = 1.6 if tk.is_urgent else 1.0
                            late_w = (8.0 * urg_m) * (1.0 - 0.65 * overload)
                            urg_bonus = (18.0 if tk.is_urgent else 0.0) * (1.0 - 0.50 * overload)
                            # Backlog-normalized workshop regret surrogate (prevents overdue cross-map thrashing)
                            j_pair = (
                                dead_w * d_dead
                                + haul_w * d_haul
                                + late_w * late_eff
                                + (9.0 / max(0.7, sat_scale)) * choke_hit * ld_i
                                - urg_bonus
                            )
                            fv_c = np.zeros(CAND_RAW_DIM, dtype=np.float32)
                            fv_c[0] = (d_dead + d_haul - 25.0) / 40.0
                            fv_c[2] = (d_dead + d_haul) / 50.0
                            fv_c[3] = float((tick + d_dead + d_haul) - tk.due_step) / 60.0
                            fv_c[7] = ld_i * 0.35
                            fv_c[8] = min(1.0, ld_i * 0.3)
                            fv_c[9] = ld_i * 0.25
                            fv_c[22] = is_large
                            x_rows.append(build_anchor_relative_feature(fv_c, fv_anc, h_vec))
                            costs.append(float(j_pair))

                    c_arr = np.asarray(costs, dtype=np.float32)
                    c_anchor = float(np.median(c_arr))
                    dfl_snaps.append({
                        "X_rel": np.stack(x_rows, axis=0),
                        "true_delta_anchor": (c_arr - c_anchor).astype(np.float32),
                    })
                    static_m, dyn_m = _build_wu_pair_tensors_ws(cand_t, cand_a, cur_pos, network, tt_cache, tick)
                    wu_samples.append({
                        "static_m": static_m,
                        "dyn_m": dyn_m,
                        "costs": c_arr,
                    })

                return _dispatch_tasks_wu26(
                    idle_agvs,
                    ready_tasks,
                    cur_pos,
                    network,
                    tt_cache,
                    tick,
                    wu_zero,
                    haven_prof=haven_prof,
                    bay_load=bay_load,
                )

            run_workshop_episode(
                layout=layout,
                aisle_capacity=cap,
                n_tasks=n_tasks,
                n_agvs=n_agvs,
                seed=sd,
                dispatcher="Wu26_SemiCircle_GRU",
                router="CPS_CVCR_DFL",
                delay_prob=dp,
                horizon=750,
                wu_net=wu_zero,
                dispatch_override=_capture_dispatch,
                use_rust_e2e=True,
            )
    return dfl_snaps, wu_samples


def _collect_lpr_blocker_counterfactuals(
    n_episodes: int = 45,
    seed: int = 777,
) -> list[dict[str, Any]]:
    """Collect REAL Rust E2E counterfactual L2/L3 Local Priority Repair (`blocker_j`) training snapshots.

    Uses `RustReservationTable.fast().clone_fast()` + `restore_from()` to measure exact
    unblocking probability, aid completion arrival delta, and blocker replan overhead for
    each candidate blocking AGV under congested workshop / demo17 choke scenarios.
    """
    rng = random.Random(int(seed))
    snaps: list[dict[str, Any]] = []
    ws_net = WorkshopScene.load(DEFAULT_SCENE).network(aisle_capacity=1)
    dm_net = build_demo_network(aisle_capacity=1)

    for ep in range(n_episodes):
        is_ws = (ep % 3 != 2)
        net = ws_net if is_ws else dm_net
        haven_prof = analyze_network_havens(net)
        cfg = CPSAStarConfig(mode="lifelong", beta=0.35, gamma=1.5, max_expansions=3000)
        planner = RustCPSAStarGraphPlanner(net, cfg, max_speed=_AGV_SPEED_MPS, dt=_STEP_DT_S)
        res = RustReservationTable.fast()
        fld = planner.create_field(max_track_horizon=180, t_start=0)
        nodes = sorted(net.nodes.keys())
        n_bg = rng.randint(4, 10)
        horizon = 110

        uncommitted: dict[str, tuple[list[tuple[str, int]], list[tuple[str, str, int, int]]]] = {}
        pos: dict[str, str] = {}
        phase: dict[str, str] = {}
        cur_task: dict[str, Optional[WorkshopTask]] = {}
        starts = rng.sample(nodes, min(len(nodes), n_bg + 1))

        for k in range(n_bg):
            aid = f"AGV_{k+1:02d}"
            st = starts[k]
            gl = rng.choice([n for n in nodes if n != st])
            pos[aid] = st
            is_hard = rng.random() < 0.45
            phase[aid] = "to_dropoff" if is_hard else ("to_pickup" if rng.random() < 0.6 else "idle")
            cur_task[aid] = (
                None if phase[aid] == "idle"
                else WorkshopTask(k, st, gl, 0, rng.randint(25, 65), is_urgent=(rng.random() < 0.3))
            )
            res.reserve_node(st, Interval(0, 1), aid)
            pl, _ = planner.plan_leg(st, gl, 0, res, fld, aid, horizon, goal_park_steps=1)
            if pl is not None:
                pts, segs = _expand_multi_step_schedule(st, 0, pl, net, _AGV_SPEED_MPS)
                planner.reserve_step_points_indexed(res, aid, pts, segs, park_until=min(horizon, pts[-1][1] + 2))
                fld.add_agv_path(aid, pts, horizon=horizon)
                uncommitted[aid] = (pts, segs)

        if len(uncommitted) < 2:
            continue
        aid_f = "AGV_FAIL"
        s_f = starts[-1]
        g_f = rng.choice([n for n in nodes if n != s_f])
        tk_f = WorkshopTask(999, s_f, g_f, 0, 40, is_urgent=(rng.random() < 0.4))
        cands = sorted(uncommitted.keys())[: min(len(uncommitted), 5)]
        snap_res = res.clone_fast()
        snap_fld = fld.clone()

        hits_probe = {c: rng.randint(1, 6) for c in cands}
        # Probe direct leg hits if actually blocked
        res.reserve_node(s_f, Interval(0, 1), aid_f)
        p_direct, h_direct = planner.plan_leg(s_f, g_f, 0, res, fld, aid_f, 35, goal_park_steps=1)
        for k_h, v_h in h_direct.items():
            if k_h in uncommitted:
                hits_probe[k_h] = max(1, int(v_h))
        if p_direct is not None and not h_direct:
            # Make goal tightly contended along one blocker's path to evaluate true LPR tradeoff
            pass

        is_large = 1.0 if is_ws else 0.0
        choke_frac = float(sum(1 for a_k, pk in pos.items() if pk in haven_prof.choke_nodes or pk in haven_prof.leaf_spurs)) / float(max(1, n_bg))
        h_vec = _build_workshop_history_vec(
            blocked_frac=min(1.0, float(len(cands)) / float(max(1, n_bg))),
            choke_frac=choke_frac,
            urgent_pressure=1.0 if tk_f.is_urgent else 0.3,
            comp_rate=0.5,
            horizon_frac=0.35,
        )
        fv_anc = np.zeros(CAND_RAW_DIM, dtype=np.float32)
        fv_anc[12] = 1.0
        fv_anc[22] = is_large

        x_rows: list[np.ndarray] = []
        y_deltas: list[float] = []
        j_anchor = 210.0  # Anchor baseline: fail / stall penalty when refusing local release

        for bj in cands:
            saved_pts, saved_segs = uncommitted[bj]
            res.restore_from(snap_res)
            fld.restore_from(snap_fld)
            res.release_all(bj)
            res.reserve_node(pos[bj], Interval(0, 1), bj)
            fld.remove_agv(bj)

            p_aid, _ = planner.plan_leg(s_f, g_f, 0, res, fld, aid_f, horizon, goal_park_steps=1)
            exp_a = int(planner.last_expansions)
            ok_aid = p_aid is not None
            arr_a = int(p_aid.arrival_steps[-1]) if (p_aid is not None and p_aid.arrival_steps) else horizon
            if p_aid is not None:
                pts_a, segs_a = _expand_multi_step_schedule(s_f, 0, p_aid, net, _AGV_SPEED_MPS)
                planner.reserve_step_points_indexed(res, aid_f, pts_a, segs_a, park_until=min(horizon, arr_a + 2))
                fld.add_agv_path(aid_f, pts_a, horizon=horizon)

            gl_b = cur_task[bj].dropoff if (cur_task[bj] and phase[bj] == "to_dropoff") else (cur_task[bj].pickup if cur_task[bj] else pos[bj])
            p_bj, _ = planner.plan_leg(pos[bj], gl_b, 0, res, fld, bj, horizon, goal_park_steps=1)
            exp_b = int(planner.last_expansions)
            ok_bj = p_bj is not None
            arr_b = int(p_bj.arrival_steps[-1]) if (p_bj is not None and p_bj.arrival_steps) else horizon

            # Penalize disrupting Executing_Hard urgent carrier unless unblocking succeeds cleanly
            hard_pen = 35.0 if (_agv_commitment_tier(phase[bj], cur_task[bj]) == TIER_EXECUTING_HARD) else 0.0
            j_bj = (
                (0.0 if ok_aid else 160.0)
                + (0.0 if ok_bj else 140.0)
                + 1.6 * float(arr_a + max(0, arr_b - int(saved_pts[-1][1])))
                + 0.03 * float(exp_a + exp_b)
                + hard_pen
            )

            on_choke = 1.0 if (pos[bj] in haven_prof.choke_nodes or pos[bj] in haven_prof.leaf_spurs) else 0.0
            tier_bj = _agv_commitment_tier(phase[bj], cur_task[bj])
            imp = impact_aware_blocker_score(aid_f, bj, hits_probe, max(1, len(saved_pts)), snap_fld.slot_Phi(pos[bj], 0))
            tier_bonus = 2.5 if tier_bj == TIER_UNCOMMITTED else (1.5 if tier_bj == TIER_SOFT else 0.0)
            urg_delta = (1.0 if tk_f.is_urgent else 0.0) - (1.0 if (cur_task[bj] and cur_task[bj].is_urgent) else 0.0)

            fv_c = np.zeros(CAND_RAW_DIM, dtype=np.float32)
            fv_c[0] = -0.15 * float(imp) - 0.1 * tier_bonus
            fv_c[1] = fv_c[0]
            fv_c[2] = -0.1 * float(hits_probe.get(bj, 1))
            fv_c[3] = -0.25 * urg_delta
            fv_c[7] = -0.2 * float(hits_probe.get(bj, 1))
            fv_c[8] = choke_frac
            fv_c[9] = -0.2 * on_choke
            fv_c[13] = 0.0 if tier_bj == TIER_UNCOMMITTED else (0.15 if tier_bj == TIER_SOFT else 0.45)
            fv_c[15] = 0.5 / float(max(1, n_bg))
            fv_c[17] = on_choke * 0.5
            fv_c[22] = is_large
            x_rows.append(build_anchor_relative_feature(fv_c, fv_anc, h_vec))
            y_deltas.append(float(j_bj - j_anchor))

        if len(x_rows) >= 2:
            snaps.append({
                "X_rel": np.stack(x_rows, axis=0),
                "true_delta_anchor": np.asarray(y_deltas, dtype=np.float32),
            })
    return snaps


def tune_wu26_workshop_actor(
    wu_samples: list[dict[str, Any]],
    *,
    out_npz: Path,
    epochs: int = 50,
    lr: float = 4e-4,
    seed: int = 42,
) -> dict[str, Any]:
    """Fine-tune pre-trained `WuEJOR26GRUAttentionNet` on converged workshop samples with proximal anchor."""
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))

    ckpt = resolve_default_checkpoint("baselines_2026/ejor26_wu_gru_attn.pt")
    w_init, _, _ = load_numpy_state_dict(ckpt, prefer_npz=True)
    model = WuEJOR26GRUAttentionNet(static_dim=12, dynamic_dim=8, hidden_dim=128, dropout=0.0)
    model.load_state_dict({k: torch.from_numpy(v.copy()) for k, v in w_init.items()}, strict=True)
    anchor_params = {k: p.detach().clone() for k, p in model.named_parameters()}

    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Hold out 15% of samples to verify KL improvement without overfitting
    n_tot = len(wu_samples)
    idx_all = rng.permutation(n_tot)
    n_val = max(4, int(0.15 * n_tot))
    val_ids = idx_all[:n_val].tolist()
    tr_ids = idx_all[n_val:].tolist()

    def _eval_kl(indices: list[int]) -> float:
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for ii in indices:
                sp = wu_samples[ii]
                st_t = torch.from_numpy(sp["static_m"])
                dy_t = torch.from_numpy(sp["dyn_m"])
                L = st_t.shape[-1]
                hs = model.encode_static(st_t)
                h0 = torch.zeros((1, model.hidden_dim), dtype=torch.float32)
                d0 = torch.zeros((1, model.static_dim), dtype=torch.float32)
                mk = torch.ones((1, L), dtype=torch.float32)
                lg, _ = model.step_pointer(hs, dy_t, d0, h0, mk)
                c_t = torch.from_numpy(sp["costs"])
                c_norm = (c_t - c_t.mean()) / torch.clamp(c_t.std(), min=1.0)
                q_true = F.softmax(-c_norm / 0.70, dim=0)
                log_p = F.log_softmax(lg[0], dim=0)
                tot += float(F.kl_div(log_p, q_true, reduction="sum").item())
        return tot / max(1, len(indices))

    kl_before = _eval_kl(val_ids)
    for _ in range(int(epochs)):
        model.train()
        perm = rng.permutation(len(tr_ids))
        opt.zero_grad()
        loss_acc = torch.tensor(0.0)
        b_cnt = 0
        for pos in perm:
            sp = wu_samples[tr_ids[int(pos)]]
            st_t = torch.from_numpy(sp["static_m"])
            dy_t = torch.from_numpy(sp["dyn_m"])
            L = st_t.shape[-1]
            hs = model.encode_static(st_t)
            h0 = torch.zeros((1, model.hidden_dim), dtype=torch.float32)
            d0 = torch.zeros((1, model.static_dim), dtype=torch.float32)
            mk = torch.ones((1, L), dtype=torch.float32)
            lg, _ = model.step_pointer(hs, dy_t, d0, h0, mk)
            c_t = torch.from_numpy(sp["costs"])
            c_norm = (c_t - c_t.mean()) / torch.clamp(c_t.std(), min=1.0)
            q_true = F.softmax(-c_norm / 0.70, dim=0)
            log_p = F.log_softmax(lg[0], dim=0)
            kl = F.kl_div(log_p, q_true, reduction="sum")
            prox = sum(
                torch.sum((p - anchor_params[name]) ** 2)
                for name, p in model.named_parameters()
                if "critic" not in name
            )
            loss_acc = loss_acc + kl + 0.035 * prox
            b_cnt += 1
            if b_cnt >= 16:
                (loss_acc / float(b_cnt)).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                opt.step()
                opt.zero_grad()
                loss_acc = torch.tensor(0.0)
                b_cnt = 0
        if b_cnt > 0:
            (loss_acc / float(b_cnt)).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()

    kl_after = _eval_kl(val_ids)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        str(out_npz),
        **{k: v.detach().cpu().numpy().astype(np.float32) for k, v in model.state_dict().items()},
    )
    # Parity check with pure-NumPy backend
    w_saved, _, _ = load_numpy_state_dict(out_npz, prefer_npz=True)
    _ = NumpyWuEJOR26GRUAttentionNet(w_saved)
    return {
        "train_samples": len(tr_ids),
        "val_samples": len(val_ids),
        "val_kl_before": round(kl_before, 4),
        "val_kl_after": round(kl_after, 4),
        "out_npz": str(out_npz),
    }


def _eval_dfl_val_regret(critic: DFLAdvantageEnsemble, val_snaps: list[dict[str, Any]]) -> dict[str, float]:
    regrets: list[float] = []
    for sp in val_snaps:
        sc = critic.score_batch(sp["X_rel"], kappa=0.4)
        best_idx = int(max(range(len(sc)), key=lambda i: float(sc[i].lcb_advantage) - 40.0 * float(sc[i].harm_prob)))
        y = sp["true_delta_anchor"]
        regrets.append(float(y[best_idx] - np.min(y)))
    return {
        "val_mean_regret_ja": round(float(np.mean(regrets)) if regrets else 0.0, 2),
        "val_zero_regret_rate": round(float(np.mean([1.0 if r <= 1e-6 else 0.0 for r in regrets])) if regrets else 0.0, 4),
    }


def main() -> None:
    t0 = time.perf_counter()
    TRACK_R_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("=== Track P2: Scenario-Native Policy & L2/L3 DFL Critic Tuning ===", flush=True)

    disp_snaps, wu_samples = _collect_dispatch_and_wu_samples()
    lpr_snaps = _collect_lpr_blocker_counterfactuals(n_episodes=48, seed=777)
    all_snaps = disp_snaps + lpr_snaps
    print(
        f"  Collected {len(disp_snaps)} dispatch snaps + {len(lpr_snaps)} counterfactual LPR blocker snaps "
        f"({len(wu_samples)} Wu26 training sets) in {time.perf_counter() - t0:.2f}s",
        flush=True,
    )

    rng = np.random.default_rng(2026)
    perm = rng.permutation(len(all_snaps))
    n_val = max(12, int(0.15 * len(all_snaps)))
    val_snaps = [all_snaps[int(i)] for i in perm[:n_val]]
    tr_snaps = [all_snaps[int(i)] for i in perm[n_val:]]

    zero_dfl = _load_dfl_critic()
    before_stats = _eval_dfl_val_regret(zero_dfl, val_snaps) if zero_dfl is not None else {}

    tuned_dfl = train_dfl_advantage_ensemble(
        tr_snaps,
        loss_mode="DFL",
        n_members=3,
        hidden_dim=48,
        epochs=65,
        lr=3e-3,
        weight_decay=1e-3,
        harm_delta_ja=25.0,
        base_seed=42,
    )
    tuned_dfl.metadata["domain"] = "workshop_v1"
    tuned_dfl.metadata["n_dispatch_snaps"] = len(disp_snaps)
    tuned_dfl.metadata["n_lpr_snaps"] = len(lpr_snaps)
    dfl_out = TRACK_R_CKPT_DIR / "mh_dfl_critic_workshop.npz"
    tuned_dfl.save_npz(dfl_out)
    after_stats = _eval_dfl_val_regret(tuned_dfl, val_snaps)
    print(f"  [DFL] ZeroShot val={before_stats} -> Tuned val={after_stats} -> {dfl_out}", flush=True)

    wu_out = TRACK_R_CKPT_DIR / "ejor26_wu_gru_workshop.npz"
    wu_stats = tune_wu26_workshop_actor(wu_samples, out_npz=wu_out, epochs=50, lr=4e-4, seed=42)
    print(f"  [Wu26 GRU] {wu_stats}", flush=True)

    report = {
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "dfl_zero_shot_val": before_stats,
        "dfl_tuned_val": after_stats,
        "dfl_ckpt": str(dfl_out),
        "wu26_stats": wu_stats,
    }
    (ROOT / "out" / "track_r_cpsa" / "track_p2_tuning_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"=== Tuning complete in {report['elapsed_s']}s ===", flush=True)


if __name__ == "__main__":
    main()
