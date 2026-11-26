"""Railway Freight Workshop Bridge (`Wu EJOR'26` + `Guo & Huang CPS-A*` + Taskbook R2-R5).

Cleanly converges the two advisor-anchored papers onto the graduation-thesis railway freight
maintenance workshop without importing any frozen FMS/DFL/CRC/RGRO modules:
  1. Upper Dispatching Layer (`Wu et al., EJOR 2026`, DOI 10.1016/j.ejor.2026.08.040):
     Pure-NumPy GRU Pointer-Attention (`NumpyWuEJOR26GRUAttentionNet`, trained checkpoint) +
     semi-circle proximity priority (`p_m` + load-balanced staging) vs `FIFO_RoundRobin` &
     `Nearest_Idle`.
  2. Lower Routing & Conflict-Resolution Layer (`Guo & Huang, TRC R1 2026 CPS-A*`):
     `Static_Hold` vs `CA*` vs `CPS-A*` vs `CPS-A* (IALPR + EALR)` with exact weighted edge
     interval reservations (`max_speed=5.0 m/step`) and physical collision-free step gating.
  3. Taskbook Metrics (R5):
     Outputs makespan, mean completion time, on-time rate (%), AGV utilization (%), conflict
     wait count, and fleet-size/channel-load rationality across `demo17` (17-node ring) and
     `rail_freight_workshop_v1` (62-node / 144-edge scene).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.numpy_backends import (
    NumpyWuEJOR26GRUAttentionNet,
    load_numpy_state_dict,
)
from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.commitment import TIER_EXECUTING_HARD, TIER_SOFT, TIER_UNCOMMITTED
from amr_sched.cvcr.evaluator import physical_edge_id
from amr_sched.cvcr.v12_proactive_guard import NetworkHavenProfile, analyze_network_havens
from amr_sched.network import Network, NodeKind, build_demo_network
from amr_sched.planners.rust_spacetime_astar import _HAS_CPS_FFI, RustCPSAStarGraphPlanner
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.rust_reservation import RustReservationTable, is_rust_core_available
from amr_sched.workshop import DEFAULT_SCENE, WorkshopScene, legal_task_pairs, parking_slots
from experiments.cps_astar_routing.cps_astar_graph import (
    CPSAStarConfig,
    CPSAStarGraphPlanner,
    CPSGraphFieldConfig,
    GraphCongestionField,
    PlannedPath,
)
from experiments.rgro.dfl_critic import (
    CAND_RAW_DIM,
    DFLAdvantageEnsemble,
    HIST_DIM,
    build_anchor_relative_feature,
)
from experiments.track_r_cpsa.cpsa_pure_runner import impact_aware_blocker_score

OUT_DIR = ROOT / "out" / "track_r_cpsa"
_AGV_SPEED_MPS = 1.0
_STEP_DT_S = 5.0


@dataclass(frozen=True)
class WorkshopTask:
    task_id: int
    pickup: str
    dropoff: str
    release_step: int
    due_step: int
    is_urgent: bool = False
    service_steps: int = 1


@dataclass(frozen=True)
class WorkshopRunResult:
    layout: str
    aisle_capacity: int
    n_tasks: int
    n_agvs: int
    seed: int
    dispatcher: str
    router: str
    delay_prob: float
    completed_tasks: int
    makespan: int
    mean_completion_time: float
    on_time_rate: float
    agv_utilization: float
    conflict_wait_count: int
    total_expansions: int
    wall_ms: float
    zero_conflicts: bool
    lpr_attempts: int = 0
    lpr_wasted: int = 0


def _edge_steps(network: Network, u: str, v: str, speed: float = _AGV_SPEED_MPS) -> int:
    """Exact discrete ticks matching `CPSAStarGraphPlanner` (`ceil(traversal_time / dt)`)."""
    e = network.edge(u, v)
    return max(1, int(math.ceil(float(e.traversal_time(speed)) / _STEP_DT_S)))


def _tt_steps(
    network: Network,
    src: str,
    dst: str,
    cache: dict[tuple[str, str], int],
    speed: float = _AGV_SPEED_MPS,
) -> int:
    key = (src, dst)
    if key in cache:
        return cache[key]
    dists = network.shortest_travel_times(src, speed)
    for k, val in dists.items():
        cache[(src, k)] = max(0, int(math.ceil(float(val) / _STEP_DT_S))) if k != src else 0
    if key not in cache:
        cache[key] = 999999
    return cache[key]


def generate_workshop_tasks(
    network: Network,
    *,
    layout: str,
    n_tasks: int,
    seed: int,
    urgent_frac: float = 0.20,
    arrival_rate: float = 0.16,
) -> list[WorkshopTask]:
    """Generate deterministic railway-workshop pickup/dropoff streams including urgent insertions."""
    rng = random.Random(int(seed) * 1009 + int(n_tasks) * 17)
    tt_cache: dict[tuple[str, str], int] = {}

    if layout == "workshop_v1":
        sc = WorkshopScene.load(DEFAULT_SCENE)
        pairs = legal_task_pairs(sc)
    else:
        stations = [n.node_id for n in network.nodes_of_kind(NodeKind.WORKSTATION)]
        bufs = [n.node_id for n in network.nodes_of_kind(NodeKind.BUFFER)]
        dis = [n.node_id for n in network.nodes_of_kind(NodeKind.DISASSEMBLY)] or ["D"]
        asm = [n.node_id for n in network.nodes_of_kind(NodeKind.ASSEMBLY)] or ["S"]
        pairs = []
        for w in stations:
            pairs.append((dis[0], w))
            pairs.append((w, asm[0]))
        for b in bufs:
            for w in stations[:3]:
                pairs.append((b, w))

    tasks: list[WorkshopTask] = []
    cur_rel = 0.0
    for i in range(n_tasks):
        pu, do = rng.choice(pairs)
        if i >= max(2, int(0.15 * n_tasks)):
            cur_rel += rng.expovariate(max(0.05, arrival_rate))
        rel = int(round(cur_rel))
        urg = rng.random() < urgent_frac
        travel = _tt_steps(network, pu, do, tt_cache)
        # Urgent insertions receive tighter due windows inside achievable turnaround budgets
        slack = rng.randint(25, 45) if urg else rng.randint(55, 95)
        svc = rng.randint(1, 2)
        tasks.append(WorkshopTask(i, pu, do, rel, rel + travel + slack, is_urgent=urg, service_steps=svc))
    return sorted(tasks, key=lambda t: (t.release_step, 0 if t.is_urgent else 1, t.task_id))


def _load_wu26_net(ckpt_path: Optional[Path] = None) -> Optional[NumpyWuEJOR26GRUAttentionNet]:
    ckpt = ckpt_path if ckpt_path is not None else resolve_default_checkpoint("baselines_2026/ejor26_wu_gru_attn.pt")
    try:
        weights, _, _ = load_numpy_state_dict(ckpt, prefer_npz=True)
        return NumpyWuEJOR26GRUAttentionNet(weights)
    except Exception:
        return None


def _load_dfl_critic(ckpt_path: Optional[Path] = None) -> Optional[DFLAdvantageEnsemble]:
    """Load trained pure-NumPy DFL 3-member critic ensemble (`0.04 ms` inference, zero PyTorch)."""
    if ckpt_path is not None and Path(ckpt_path).is_file():
        try:
            return DFLAdvantageEnsemble.load_npz(ckpt_path)
        except Exception:
            pass
    for rel in (
        ROOT / "out" / "rgro_closure" / "mh_dfl_critic.npz",
        ROOT / "out" / "rgro" / "dfl_critic_v19.npz",
        ROOT / "amr_sched" / "checkpoints" / "track_r" / "mh_dfl_critic_workshop.npz",
    ):
        if rel.is_file():
            try:
                return DFLAdvantageEnsemble.load_npz(rel)
            except Exception:
                continue
    return None


def _agv_commitment_tier(phase_str: str, task_obj: Optional[WorkshopTask]) -> str:
    """Map workshop execution phase to CVCR commitment tier."""
    if phase_str in ("need_dropoff", "to_dropoff") and task_obj is not None:
        return TIER_EXECUTING_HARD
    if phase_str == "to_pickup" and task_obj is not None:
        return TIER_SOFT
    return TIER_UNCOMMITTED


def _build_workshop_history_vec(
    *,
    blocked_frac: float,
    choke_frac: float,
    urgent_pressure: float,
    comp_rate: float,
    horizon_frac: float,
) -> np.ndarray:
    """Construct compact 16-D causal congestion history vector matching `ExecutionHistoryBuffer`."""
    return np.clip(
        np.array(
            [
                blocked_frac,
                choke_frac,
                urgent_pressure,
                0.0,
                urgent_pressure * 1.5,
                comp_rate,
                0.0,
                0.0,
                0.0,
                comp_rate,
                horizon_frac,
                blocked_frac,
                urgent_pressure * 1.5,
                choke_frac,
                0.0,
                1.0,
            ],
            dtype=np.float32,
        ),
        -6.0,
        6.0,
    )


def _select_blocker_cvcr_dfl(
    aid: str,
    cands: list[str],
    hits: dict[str, int],
    pos: dict[str, str],
    phase: dict[str, str],
    cur_task: dict[str, Optional[WorkshopTask]],
    uncommitted: dict[str, tuple[list[tuple[str, int]], list[tuple[str, str, int, int]]]],
    haven_prof: NetworkHavenProfile,
    net: Network,
    field: GraphCongestionField,
    tick: int,
    n_agvs: int,
    horizon: int,
    completed_cnt: int,
    n_tasks: int,
    dfl_critic: Optional[DFLAdvantageEnsemble],
) -> Optional[str]:
    """CVCR Commitment + DFL Conservative Advantage blocker selector (RGRO/CRC risk-gated)."""
    tier_aid = _agv_commitment_tier(phase[aid], cur_task[aid])
    tk_aid = cur_task[aid]
    # Protect loaded material carriers against disruption by idle evacuees
    filtered = [
        j for j in cands
        if not (tier_aid == TIER_UNCOMMITTED and _agv_commitment_tier(phase[j], cur_task[j]) == TIER_EXECUTING_HARD)
    ]
    if not filtered:
        return None

    is_large = 1.0 if len(net.nodes) >= 30 else 0.0
    blocked_frac = float(len(hits)) / float(max(1, n_agvs))
    choke_frac = float(
        sum(1 for a_k in pos if pos[a_k] in haven_prof.choke_nodes or pos[a_k] in haven_prof.leaf_spurs)
    ) / float(max(1, n_agvs))
    comp_rate = float(completed_cnt) / float(max(1, n_tasks))
    urg_p = 1.0 if (tk_aid is not None and tk_aid.is_urgent) else 0.3
    h_vec = _build_workshop_history_vec(
        blocked_frac=blocked_frac,
        choke_frac=choke_frac,
        urgent_pressure=urg_p,
        comp_rate=comp_rate,
        horizon_frac=min(1.0, float(tick) / float(max(1, horizon))),
    )
    fv_anchor = np.zeros(CAND_RAW_DIM, dtype=np.float32)
    fv_anchor[12] = 1.0  # is_keep_cand = 1.0
    fv_anchor[22] = is_large

    x_rows: list[np.ndarray] = []
    meta_scores: list[tuple[float, float, float, str]] = []
    for bj in filtered:
        tier_bj = _agv_commitment_tier(phase[bj], cur_task[bj])
        tk_bj = cur_task[bj]
        on_choke = 1.0 if (pos[bj] in haven_prof.choke_nodes or pos[bj] in haven_prof.leaf_spurs) else 0.0
        rem_len = max(1, len(uncommitted.get(bj, ([], []))[0]))
        imp = impact_aware_blocker_score(aid, bj, hits, rem_len, field.slot_Phi(pos[bj], tick))
        tier_bonus = 0.0
        if tier_bj == TIER_UNCOMMITTED:
            tier_bonus = 2.5
        elif tier_bj == TIER_SOFT and tier_aid == TIER_EXECUTING_HARD:
            tier_bonus = 1.5
        elif tier_bj == TIER_EXECUTING_HARD and tier_aid != TIER_EXECUTING_HARD:
            tier_bonus = -3.0

        urg_delta = (1.0 if (tk_aid and tk_aid.is_urgent) else 0.0) - (1.0 if (tk_bj and tk_bj.is_urgent) else 0.0)
        fv_c = np.zeros(CAND_RAW_DIM, dtype=np.float32)
        fv_c[0] = -0.15 * float(imp) - 0.1 * tier_bonus
        fv_c[1] = fv_c[0]
        fv_c[2] = -0.1 * float(hits.get(bj, 1))
        fv_c[3] = -0.25 * urg_delta
        fv_c[7] = -0.2 * float(hits.get(bj, 1))
        fv_c[8] = choke_frac
        fv_c[9] = -0.2 * on_choke
        fv_c[13] = 0.0 if tier_bj == TIER_UNCOMMITTED else (0.15 if tier_bj == TIER_SOFT else 0.45)
        fv_c[15] = 0.5 / float(max(1, n_agvs))
        fv_c[17] = on_choke * 0.5
        fv_c[22] = is_large
        x_rows.append(build_anchor_relative_feature(fv_c, fv_anchor, h_vec))
        meta_scores.append((imp, tier_bonus, on_choke, tier_bj))

    dfl_out = dfl_critic.score_batch(np.stack(x_rows, axis=0), kappa=0.5) if dfl_critic is not None else None
    meta_trained_ws = str(getattr(dfl_critic, "metadata", {}).get("domain", "")) == "workshop_v1" if dfl_critic is not None else False
    ranked: list[tuple[float, str]] = []
    for idx, bj in enumerate(filtered):
        imp, tier_bonus, on_choke, tier_bj = meta_scores[idx]
        lcb_adv = 0.0
        harm_p = 0.25
        ep_conf = 1.0
        if dfl_out is not None:
            lcb_adv = max(-3.0, min(3.0, float(dfl_out[idx].lcb_advantage) / 100.0))
            harm_p = float(dfl_out[idx].harm_prob)
            if not meta_trained_ws:
                ep_conf = 1.0 / (1.0 + float(dfl_out[idx].epistemic_std) / 80.0)
        # CRC/RGRO harm veto: avoid breaking committed urgent carriers under high predicted harm unless inside choke
        if tier_bj == TIER_EXECUTING_HARD and tier_aid != TIER_EXECUTING_HARD and harm_p > 0.72 and on_choke < 0.5 and (meta_trained_ws or ep_conf > 0.45):
            continue
        l_w = 0.75 if meta_trained_ws else (0.50 * ep_conf)
        h_w = 2.20 if meta_trained_ws else (1.40 * ep_conf)
        score = imp + tier_bonus + 1.2 * on_choke + l_w * lcb_adv - h_w * harm_p
        ranked.append((score, bj))
    if not ranked:
        return None
    ranked.sort(key=lambda kv: (-kv[0], kv[1]))
    return ranked[0][1]


def _dispatch_tasks_wu26(
    idle_agvs: list[str],
    ready_tasks: list[WorkshopTask],
    cur_pos: dict[str, str],
    network: Network,
    tt_cache: dict[tuple[str, str], int],
    tick: int,
    wu_net: Optional[NumpyWuEJOR26GRUAttentionNet],
    *,
    dfl_critic: Optional[DFLAdvantageEnsemble] = None,
    haven_prof: Optional[NetworkHavenProfile] = None,
    bay_load: Optional[dict[str, int]] = None,
) -> list[tuple[str, WorkshopTask]]:
    """Wu et al. (EJOR 2026) semi-circle composite priority p_m + GRU Pointer-Attention + optional DFL detour gate."""
    if not idle_agvs or not ready_tasks:
        return []
    # Keep per-tick pointer inference compact & fast (<= 8 highest-priority tasks x <= 8 idle AGVs)
    scored_tasks = sorted(
        ready_tasks,
        key=lambda tk: (
            0 if tk.is_urgent else 1,
            float(tk.due_step - tick - _tt_steps(network, tk.pickup, tk.dropoff, tt_cache)),
            tk.release_step,
            tk.task_id,
        ),
    )
    cand_t = scored_tasks[: min(len(scored_tasks), 8)]
    cand_a = sorted(idle_agvs)[: min(len(idle_agvs), 8)]
    n_t, n_a = len(cand_t), len(cand_a)

    # Semi-circle proximity composite score matrix (+ optional trained GRU pointer residual)
    xs = [float(n.x) for n in network.nodes.values()] or [0.0]
    ys = [float(n.y) for n in network.nodes.values()] or [0.0]
    minx, spanx = min(xs), max(1.0, max(xs) - min(xs))
    miny, spany = min(ys), max(1.0, max(ys) - min(ys))

    scores = np.zeros((n_t, n_a), dtype=np.float64)
    for i, tk in enumerate(cand_t):
        d_haul = float(_tt_steps(network, tk.pickup, tk.dropoff, tt_cache))
        # Semi-circle proximity p_m: urgent/tight-slack gets elevated weight along shortest approach
        pm = (1.8 if tk.is_urgent else 1.0) + max(0.0, 20.0 / max(4.0, float(tk.due_step - tick)))
        spur_pen = 0.0
        if bay_load is not None:
            p_ld = int(bay_load.get(tk.pickup, 0))
            d_ld = int(bay_load.get(tk.dropoff, 0))
            if haven_prof is not None and (tk.pickup in haven_prof.leaf_spurs or tk.pickup in haven_prof.choke_nodes):
                spur_pen += 2.0 * float(p_ld)
            if haven_prof is not None and (tk.dropoff in haven_prof.leaf_spurs or tk.dropoff in haven_prof.choke_nodes):
                spur_pen += 2.0 * float(d_ld)
        for j, aid in enumerate(cand_a):
            d_dead = float(_tt_steps(network, cur_pos[aid], tk.pickup, tt_cache))
            # Prefer shortest deadhead, weighted by semi-circle priority p_m
            scores[i, j] = pm * 12.0 - d_dead - 0.15 * d_haul - spur_pen

    if wu_net is not None and n_t * n_a >= 2:
        L = n_t * n_a
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
        hs = wu_net.encode_static(static_m)
        h_t = np.zeros((1, wu_net.hidden_dim), dtype=np.float32)
        dec_in = np.zeros((1, wu_net.static_dim), dtype=np.float32)
        mask = np.ones((1, L), dtype=np.float32)
        logits, _ = wu_net.step_pointer(hs, dyn_m, dec_in, h_t, mask)
        lg = logits[0].reshape(n_t, n_a).astype(np.float64)
        if np.all(np.isfinite(lg)):
            # Blend normalized pointer preference smoothly with exact semi-circle proximity cost
            lg_std = float(np.std(lg)) or 1.0
            scores += 3.5 * ((lg - np.mean(lg)) / lg_std)

    if dfl_critic is not None and n_t * n_a >= 2:
        is_large = 1.0 if len(network.nodes) >= 30 else 0.0
        choke_p = float(sum((bay_load or {}).values())) / float(max(1, len(cand_a) + 2))
        h_vec = _build_workshop_history_vec(
            blocked_frac=min(1.0, choke_p * 0.25),
            choke_frac=min(1.0, choke_p * 0.3),
            urgent_pressure=0.5,
            comp_rate=0.5,
            horizon_frac=0.3,
        )
        fv_anc = np.zeros(CAND_RAW_DIM, dtype=np.float32)
        fv_anc[12] = 1.0
        fv_anc[22] = is_large
        x_dfl: list[np.ndarray] = []
        pairs_ij: list[tuple[int, int]] = []
        for i, tk in enumerate(cand_t):
            d_haul = float(_tt_steps(network, tk.pickup, tk.dropoff, tt_cache))
            ld_i = float((bay_load or {}).get(tk.pickup, 0) + (bay_load or {}).get(tk.dropoff, 0))
            for j, aid in enumerate(cand_a):
                d_dead = float(_tt_steps(network, cur_pos[aid], tk.pickup, tt_cache))
                fv_c = np.zeros(CAND_RAW_DIM, dtype=np.float32)
                fv_c[0] = (d_dead + d_haul - 25.0) / 40.0
                fv_c[2] = (d_dead + d_haul) / 50.0
                fv_c[3] = float((tick + d_dead + d_haul) - tk.due_step) / 60.0
                fv_c[7] = ld_i * 0.35
                fv_c[8] = min(1.0, ld_i * 0.3)
                fv_c[9] = ld_i * 0.25
                fv_c[22] = is_large
                x_dfl.append(build_anchor_relative_feature(fv_c, fv_anc, h_vec))
                pairs_ij.append((i, j))
        d_res = dfl_critic.score_batch(np.stack(x_dfl, axis=0), kappa=0.4)
        # Saturation-, backlog-, and uncertainty-aware gating prevents DFL from starving 1-lane fleets
        meta_trained_ws = str(getattr(dfl_critic, "metadata", {}).get("domain", "")) == "workshop_v1"
        backlog_p = min(1.0, max(0.0, (float(len(ready_tasks)) / float(max(1, len(idle_agvs))) - 2.0) / 4.0))
        mean_ep_std = float(np.mean([float(r.epistemic_std) for r in d_res])) if d_res else 0.0
        ep_conf = (1.0 / (1.0 + mean_ep_std / 120.0)) if meta_trained_ws else (1.0 / (1.0 + mean_ep_std / 80.0))
        cong_damp = (
            max(0.35, 1.0 - 0.14 * max(0.0, choke_p - 0.8) - 0.35 * backlog_p)
            if meta_trained_ws
            else max(0.20, 1.0 - 0.18 * choke_p)
        )
        gate_w = ep_conf * cong_damp
        harm_w = (0.35 if meta_trained_ws else 0.20) * gate_w
        lcb_w = (0.45 if meta_trained_ws else 0.25) * gate_w
        for idx, (i, j) in enumerate(pairs_ij):
            lcb = max(-3.0, min(3.0, float(d_res[idx].lcb_advantage) / 100.0))
            scores[i, j] += lcb_w * lcb - harm_w * float(d_res[idx].harm_prob)

    chosen: list[tuple[str, WorkshopTask]] = []
    used_t: set[int] = set()
    used_a: set[int] = set()
    flat_pairs = sorted(
        ((float(scores[i, j]), i, j) for i in range(n_t) for j in range(n_a)),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    for _, i, j in flat_pairs:
        if i in used_t or j in used_a:
            continue
        used_t.add(i)
        used_a.add(j)
        chosen.append((cand_a[j], cand_t[i]))
    return chosen


def _dispatch_tasks(
    dispatcher: str,
    idle_agvs: list[str],
    ready_tasks: list[WorkshopTask],
    cur_pos: dict[str, str],
    network: Network,
    tt_cache: dict[tuple[str, str], int],
    tick: int,
    wu_net: Optional[NumpyWuEJOR26GRUAttentionNet],
    *,
    dfl_critic: Optional[DFLAdvantageEnsemble] = None,
    haven_prof: Optional[NetworkHavenProfile] = None,
    bay_load: Optional[dict[str, int]] = None,
) -> list[tuple[str, WorkshopTask]]:
    if not idle_agvs or not ready_tasks:
        return []
    if dispatcher == "FIFO_RoundRobin":
        ordered_t = sorted(ready_tasks, key=lambda tk: (tk.release_step, tk.task_id))
        ordered_a = sorted(idle_agvs)
        k = min(len(ordered_t), len(ordered_a))
        return [(ordered_a[i], ordered_t[i]) for i in range(k)]
    if dispatcher == "Nearest_Idle":
        pairs = sorted(
            (
                (float(_tt_steps(network, cur_pos[a], tk.pickup, tt_cache)), tk.release_step, a, tk.task_id, tk)
                for a in idle_agvs
                for tk in ready_tasks
            ),
            key=lambda x: (x[0], x[1], x[2], x[3]),
        )
        used_t, used_a = set(), set()
        out: list[tuple[str, WorkshopTask]] = []
        for _, _, aid, tid, tk in pairs:
            if aid in used_a or tid in used_t:
                continue
            used_a.add(aid)
            used_t.add(tid)
            out.append((aid, tk))
        return out
    if dispatcher == "Wu26_SemiCircle":
        return _dispatch_tasks_wu26(idle_agvs, ready_tasks, cur_pos, network, tt_cache, tick, None)
    if dispatcher == "Wu26_SemiCircle_DFL":
        return _dispatch_tasks_wu26(
            idle_agvs,
            ready_tasks,
            cur_pos,
            network,
            tt_cache,
            tick,
            wu_net,
            dfl_critic=dfl_critic,
            haven_prof=haven_prof,
            bay_load=bay_load,
        )
    return _dispatch_tasks_wu26(idle_agvs, ready_tasks, cur_pos, network, tt_cache, tick, wu_net)


def _expand_multi_step_schedule(
    start_node: str,
    start_step: int,
    planned: PlannedPath,
    network: Network,
    speed: float,
) -> tuple[list[tuple[str, int]], list[tuple[str, str, int, int]]]:
    """Expand `PlannedPath` over weighted-time edges into per-tick nodes AND exact `(u, v, dep_t, arr_t)` edges.

    While traversing `(u, v)` during `[dep_t, arr_t)`, `step_states` keeps `node=u` for `[dep_t, arr_t)`
    until arriving at `v` at `arr_t` (for physical verification & EALR pause compatibility), while
    `edge_segments` records exact `[dep_t, arr_t)` channel occupation.
    """
    node_pts: list[tuple[str, int]] = [(start_node, int(start_step))]
    edge_segs: list[tuple[str, str, int, int]] = []
    cur_n = start_node
    cur_t = int(start_step)
    for nxt_n, arr_t in zip(planned.nodes, planned.arrival_steps):
        arr_i = int(arr_t)
        n_st = _edge_steps(network, cur_n, nxt_n, speed)
        dep_t = max(cur_t, arr_i - n_st)
        while cur_t < dep_t:
            cur_t += 1
            node_pts.append((cur_n, cur_t))
        if cur_t < arr_i:
            edge_segs.append((cur_n, nxt_n, cur_t, arr_i))
            while cur_t + 1 < arr_i:
                cur_t += 1
                node_pts.append((cur_n, cur_t))
            cur_t = arr_i
            node_pts.append((nxt_n, cur_t))
        cur_n = nxt_n
    return node_pts, edge_segs


def _reserve_workshop_path(
    res_table: Any,
    agv_id: str,
    node_pts: list[tuple[str, int]],
    edge_segs: list[tuple[str, str, int, int]],
    *,
    park_until: Optional[int] = None,
    planner: Any = None,
) -> None:
    if not node_pts:
        return
    reserve_fn = getattr(planner, "reserve_step_points_indexed", None) if planner is not None else None
    if reserve_fn is not None and not getattr(res_table, "_track_shadow", True):
        reserve_fn(res_table, agv_id, node_pts, edge_segs, park_until=park_until)
        return
    for nd, tk in node_pts:
        res_table.reserve_node(nd, Interval(tk, tk + 1), agv_id)
    for u, v, d_t, a_t in edge_segs:
        if a_t > d_t:
            res_table.reserve_edge(u, v, Interval(d_t, a_t), agv_id)
    last_n, last_t = node_pts[-1]
    if park_until is not None and park_until > last_t + 1:
        res_table.reserve_node(last_n, Interval(last_t + 1, int(park_until)), agv_id)


def run_workshop_episode(
    *,
    layout: str = "demo17",
    aisle_capacity: int = 1,
    n_tasks: int = 30,
    n_agvs: int = 4,
    seed: int = 42,
    dispatcher: str = "Wu26_SemiCircle_GRU",
    router: str = "CPS_AStar",
    delay_prob: float = 0.0,
    horizon: int = 900,
    wu_net: Optional[NumpyWuEJOR26GRUAttentionNet] = None,
    dfl_critic: Optional[DFLAdvantageEnsemble] = None,
    dispatch_override: Optional[Any] = None,
    use_rust_e2e: bool = True,
    route_b_telemetry_hook: Optional[Any] = None,
    lpr_selector_mode: str = "auto",
) -> WorkshopRunResult:
    """Execute one complete closed-loop railway-workshop schedule under Taskbook R2-R5 metrics."""
    net = (
        WorkshopScene.load(DEFAULT_SCENE).network(aisle_capacity=aisle_capacity)
        if layout == "workshop_v1"
        else build_demo_network(aisle_capacity=aisle_capacity)
    )
    haven_prof = analyze_network_havens(net)
    tasks = generate_workshop_tasks(net, layout=layout, n_tasks=n_tasks, seed=seed)
    agv_ids = [f"AGV_{i+1:02d}" for i in range(n_agvs)]
    if layout == "workshop_v1":
        pool = parking_slots(WorkshopScene.load(DEFAULT_SCENE))
        raw_pool = list(dict.fromkeys(pool + list(haven_prof.safe_havens)))
        if n_agvs <= len(pool):
            init_pos = {aid: pool[i % len(pool)] for i, aid in enumerate(agv_ids)}
        else:
            slot_pool = [nd for nd in raw_pool if nd in net.nodes for _ in range(max(1, int(net.node(nd).capacity)))]
            init_pos = {aid: slot_pool[i % len(slot_pool)] for i, aid in enumerate(agv_ids)}
    else:
        pool = ["B1", "B2", "B3", "D", "S", "R0", "R2", "R4"]
        if n_agvs <= len(pool):
            init_pos = {aid: pool[i % len(pool)] for i, aid in enumerate(agv_ids)}
        else:
            slot_pool = [nd for nd in pool if nd in net.nodes for _ in range(max(1, int(net.node(nd).capacity)))]
            init_pos = {aid: slot_pool[i % len(slot_pool)] for i, aid in enumerate(agv_ids)}

    is_static = (router == "Static_Hold")
    is_ca = (router == "CA_Star")
    is_cvcr = (router == "CPS_CVCR_DFL")
    is_ialpr = (router in ("CPS_IALPR", "CPS_IALPR_EALR", "CPS_CVCR_DFL"))
    is_ealr = (router in ("CPS_IALPR_EALR", "CPS_CVCR_DFL"))
    if (is_cvcr or dispatcher == "Wu26_SemiCircle_DFL") and dfl_critic is None:
        dfl_critic = _load_dfl_critic()

    max_exp_budget = max(4000, 140 * len(net.nodes)) if not is_cvcr else max(5000, 160 * len(net.nodes))
    cps_cfg = CPSAStarConfig(
        mode="lifelong",
        beta=0.0 if (is_static or is_ca) else (0.35 if is_cvcr else 0.5),
        gamma=0.0 if (is_static or is_ca) else (1.5 if is_cvcr else 5.0),
        batch_update_K=10,
        retry_limit_R=0 if (is_static or is_ca) else (5 if is_cvcr else 4),
        max_expansions=max_exp_budget,
    )
    rust_active = bool(use_rust_e2e and is_rust_core_available() and _HAS_CPS_FFI)
    if rust_active:
        planner: Any = RustCPSAStarGraphPlanner(net, cps_cfg, max_speed=_AGV_SPEED_MPS, dt=_STEP_DT_S)
        res_table: Any = RustReservationTable.fast()
        empty_res: Any = RustReservationTable.fast()
        plan_h = horizon + 120
        field: Any = planner.create_field(max_track_horizon=plan_h + 20, t_start=0)
    else:
        planner = CPSAStarGraphPlanner(net, cps_cfg, max_speed=_AGV_SPEED_MPS, dt=_STEP_DT_S)
        plan_h = horizon + 120
        res_table = ReservationTable()
        empty_res = ReservationTable()
        field = GraphCongestionField(net, CPSGraphFieldConfig(max_track_horizon=plan_h + 20), t_start=0)
    tt_cache: dict[tuple[str, str], int] = {}

    rng = random.Random(int(seed) * 313 + n_agvs * 7)
    motion_stalls: dict[tuple[str, int], bool] = {
        (aid, tk): (rng.random() < delay_prob)
        for aid in agv_ids for tk in range(horizon)
    } if delay_prob > 0.0 else {}

    pos: dict[str, str] = dict(init_pos)
    phase: dict[str, str] = {a: "idle" for a in agv_ids}  # idle | to_pickup | to_dropoff
    cur_task: dict[str, Optional[WorkshopTask]] = {a: None for a in agv_ids}
    service_rem: dict[str, int] = {a: 0 for a in agv_ids}
    in_edge: dict[str, Optional[tuple[str, str, int]]] = {a: None for a in agv_ids}  # (u, v, rem_steps)
    plan_steps: dict[str, dict[int, str]] = {a: {0: pos[a]} for a in agv_ids}
    plan_edges: dict[str, dict[int, tuple[str, str, int, int]]] = {a: {} for a in agv_ids}
    plan_end_t: dict[str, int] = {a: 0 for a in agv_ids}
    cooldown: dict[str, int] = {a: 0 for a in agv_ids}
    fail_plan_cnt: dict[str, int] = {a: 0 for a in agv_ids}
    stall_ticks: dict[str, int] = {a: 0 for a in agv_ids}

    for aid in agv_ids:
        res_table.reserve_node(pos[aid], Interval(0, plan_h), aid)

    pending: deque[WorkshopTask] = deque(tasks)
    ready: list[WorkshopTask] = []
    done_times: list[int] = []
    on_time_cnt = 0
    busy_steps = 0
    conflict_wait_cnt = 0
    total_exp = 0
    lpr_attempts_cnt = 0
    lpr_wasted_cnt = 0
    violated = False
    makespan = horizon
    t0_wall = time.perf_counter()

    def _plan_for_agv(
        aid_x: str,
        goal_x: str,
        t_start_x: int,
        use_res: ReservationTable,
    ) -> tuple[list[tuple[str, int]] | None, list[tuple[str, str, int, int]], dict[str, int]]:
        nonlocal total_exp
        # Find path from current node at t_start_x to goal_x
        pl, hits = planner.plan_leg(
            pos[aid_x], goal_x, t_start_x, use_res, field, aid_x, min(plan_h, t_start_x + 180),
            goal_park_steps=1, max_expansions_override=max_exp_budget,
        )
        total_exp += int(getattr(planner, "last_expansions", 0))
        if pl is None:
            return None, [], hits
        n_pts, e_segs = _expand_multi_step_schedule(pos[aid_x], t_start_x, pl, net, _AGV_SPEED_MPS)
        return n_pts, e_segs, hits

    for t in range(horizon):
        if t > 0 and (t % 30 == 0):
            res_table.purge_before(t)

        # Track P3 (Route B Hook): ingest Nav2 / local obstacle telemetry & ETA drift cleanly into reservations/service
        if route_b_telemetry_hook is not None:
            pkt = route_b_telemetry_hook(t, net, pos, in_edge, phase)
            if isinstance(pkt, dict):
                for obs in pkt.get("obstacles", ()):
                    edge_uv = getattr(obs, "edge", None) or (obs.get("edge") if isinstance(obs, dict) else None)
                    hz_obs = int(getattr(obs, "estimated_clear_horizon_steps", 8) if not isinstance(obs, dict) else obs.get("estimated_clear_horizon_steps", 8))
                    if edge_uv and len(edge_uv) == 2 and net.has_edge(edge_uv[0], edge_uv[1]):
                        obs_tag = f"__L1_OBSTACLE_{edge_uv[0]}_{edge_uv[1]}__"
                        res_table.release_all(obs_tag)
                        res_table.reserve_edge(edge_uv[0], edge_uv[1], Interval(t, min(plan_h, t + max(1, hz_obs))), obs_tag)
                for aid_tm, tm in pkt.get("telemetry", {}).items():
                    drift = int(getattr(tm, "eta_drift_steps", 0) if not isinstance(tm, dict) else tm.get("eta_drift_steps", 0))
                    if aid_tm in service_rem and drift > service_rem[aid_tm]:
                        service_rem[aid_tm] = drift

        while pending and pending[0].release_step <= t:
            ready.append(pending.popleft())

        # Check pickup/dropoff completions for AGVs settled cleanly on target nodes
        for aid in agv_ids:
            tk = cur_task[aid]
            if tk is None or in_edge[aid] is not None or service_rem[aid] > 0:
                continue
            if phase[aid] == "to_pickup" and pos[aid] == tk.pickup:
                service_rem[aid] = tk.service_steps
                phase[aid] = "need_dropoff"
            elif phase[aid] == "to_dropoff" and pos[aid] == tk.dropoff:
                done_times.append(t - tk.release_step)
                if t <= tk.due_step:
                    on_time_cnt += 1
                phase[aid] = "idle"
                cur_task[aid] = None
                fail_plan_cnt[aid] = 0
                plan_steps[aid] = {t: pos[aid]}
                plan_edges[aid] = {}
                plan_end_t[aid] = t

        if len(done_times) == n_tasks:
            makespan = t
            break

        # Dispatch ready tasks to idle AGVs (guard capacity-1 leaf workstations against simultaneous double-entry)
        idle_now = [a for a in agv_ids if phase[a] == "idle" and in_edge[a] is None and service_rem[a] == 0]
        if idle_now and ready:
            busy_leaf_stations: set[str] = set()
            bay_load_cur: dict[str, int] = {}
            for a2 in agv_ids:
                bay_load_cur[pos[a2]] = bay_load_cur.get(pos[a2], 0) + 1
                if net.node(pos[a2]).kind == NodeKind.WORKSTATION and net.node(pos[a2]).capacity == 1:
                    busy_leaf_stations.add(pos[a2])
                tk2 = cur_task[a2]
                if tk2 is not None:
                    bay_load_cur[tk2.pickup] = bay_load_cur.get(tk2.pickup, 0) + 1
                    bay_load_cur[tk2.dropoff] = bay_load_cur.get(tk2.dropoff, 0) + 1
                    for st_nd in (tk2.pickup, tk2.dropoff):
                        if net.node(st_nd).kind == NodeKind.WORKSTATION and net.node(st_nd).capacity == 1:
                            busy_leaf_stations.add(st_nd)
            eligible_ready = [
                tk for tk in ready
                if tk.pickup not in busy_leaf_stations and tk.dropoff not in busy_leaf_stations
            ]
            if not eligible_ready:
                assignments = []
            elif dispatch_override is not None:
                assignments = dispatch_override(
                    idle_now,
                    eligible_ready,
                    pos,
                    net,
                    tt_cache,
                    t,
                    {
                        "cur_task": cur_task,
                        "phase": phase,
                        "in_edge": in_edge,
                        "plan_steps": plan_steps,
                        "bay_load": bay_load_cur,
                        "haven_prof": haven_prof,
                        "agv_ids": agv_ids,
                        "wu_net": wu_net,
                        "dfl_critic": dfl_critic,
                    },
                )
            else:
                assignments = _dispatch_tasks(
                    dispatcher,
                    idle_now,
                    eligible_ready,
                    pos,
                    net,
                    tt_cache,
                    t,
                    wu_net,
                    dfl_critic=dfl_critic,
                    haven_prof=haven_prof,
                    bay_load=bay_load_cur,
                )
            assigned_ids = {tk.task_id for _, tk in assignments}
            ready = [tk for tk in ready if tk.task_id not in assigned_ids]
            for aid, tk in assignments:
                cur_task[aid] = tk
                cooldown[aid] = 0
                fail_plan_cnt[aid] = 0
                if pos[aid] == tk.pickup:
                    service_rem[aid] = tk.service_steps
                    phase[aid] = "need_dropoff"
                else:
                    phase[aid] = "to_pickup"

        park_goals: dict[str, str] = {}
        park_candidates = (
            parking_slots(WorkshopScene.load(DEFAULT_SCENE))
            if layout == "workshop_v1"
            else ["B1", "B2", "B3", "D", "S"]
        )
        haven_targets = list(haven_prof.safe_havens) if haven_prof.safe_havens else list(park_candidates)
        occ_cur: dict[str, int] = {}
        for a2 in agv_ids:
            occ_cur[pos[a2]] = occ_cur.get(pos[a2], 0) + 1
        reserved_havens: dict[str, int] = {}
        for aid in agv_ids:
            if (
                phase[aid] == "idle"
                and cur_task[aid] is None
                and in_edge[aid] is None
                and t >= plan_end_t[aid]
            ):
                need_evac = (
                    net.node(pos[aid]).kind == NodeKind.WORKSTATION
                    or (is_cvcr and (pos[aid] in haven_prof.choke_nodes or pos[aid] in haven_prof.leaf_spurs))
                )
                if not need_evac:
                    continue
                if cooldown[aid] > 0:
                    cooldown[aid] -= 1
                else:
                    if is_cvcr:
                        best_p = min(
                            haven_targets,
                            key=lambda p_nd: (
                                0 if (occ_cur.get(p_nd, 0) + reserved_havens.get(p_nd, 0)) < net.node(p_nd).capacity else 1,
                                _tt_steps(net, pos[aid], p_nd, tt_cache),
                                p_nd,
                            ),
                        )
                        reserved_havens[best_p] = reserved_havens.get(best_p, 0) + 1
                    else:
                        best_p = min(
                            park_candidates,
                            key=lambda p_nd: (_tt_steps(net, pos[aid], p_nd, tt_cache), p_nd),
                        )
                    if best_p != pos[aid]:
                        park_goals[aid] = best_p

        # Identify AGVs needing initial planning or delayed EALR local repair (only if NOT mid-edge!)
        need_plan: list[tuple[str, str]] = []
        delayed_mismatches: list[str] = []
        for aid in agv_ids:
            if in_edge[aid] is not None:
                continue
            tk = cur_task[aid]
            if tk is None:
                if aid in park_goals:
                    need_plan.append((aid, park_goals[aid]))
                continue
            if phase[aid] == "need_dropoff":
                phase[aid] = "to_dropoff"
                need_plan.append((aid, tk.dropoff))
            elif phase[aid] in ("to_pickup", "to_dropoff"):
                gl = tk.pickup if phase[aid] == "to_pickup" else tk.dropoff
                exp_here = plan_steps[aid].get(t, pos[aid])
                if service_rem[aid] > 0:
                    if is_ealr and t < plan_end_t[aid]:
                        delayed_mismatches.append(aid)
                    continue
                if pos[aid] != exp_here:
                    if is_ealr:
                        delayed_mismatches.append(aid)
                    else:
                        need_plan.append((aid, gl))
                elif t >= plan_end_t[aid] and pos[aid] != gl:
                    if cooldown[aid] > 0:
                        cooldown[aid] -= 1
                    else:
                        need_plan.append((aid, gl))

        # EALR: expand minimal affected subset around mismatched AGVs (keeping mid-edge AGVs frozen!)
        if delayed_mismatches:
            r_set: set[str] = set(delayed_mismatches)
            for src_a in list(delayed_mismatches):
                for other in agv_ids:
                    if other in r_set or in_edge[other] is not None or cur_task[other] is None:
                        continue
                    for s_i in range(t, min(plan_h, t + 10)):
                        if plan_steps[other].get(s_i) == pos[src_a]:
                            r_set.add(other)
                            break
            for aid in sorted(r_set):
                tk = cur_task[aid]
                if tk is not None:
                    gl = tk.pickup if phase[aid] == "to_pickup" else tk.dropoff
                    if (aid, gl) not in need_plan and service_rem[aid] == 0:
                        need_plan.append((aid, gl))

        # Protect idle AGVs safely (only reserve indefinitely on non-workstation / safe haven slots)
        need_set = {a for a, _ in need_plan}
        for aid in agv_ids:
            if aid not in need_set and phase[aid] == "idle" and in_edge[aid] is None and t >= plan_end_t[aid]:
                res_table.release_all(aid)
                is_choke_or_ws = (
                    net.node(pos[aid]).kind == NodeKind.WORKSTATION
                    or (is_cvcr and (pos[aid] in haven_prof.choke_nodes or pos[aid] in haven_prof.leaf_spurs))
                )
                if is_choke_or_ws:
                    res_table.reserve_node(pos[aid], Interval(t, min(plan_h, t + 2)), aid)
                else:
                    res_table.reserve_node(pos[aid], Interval(t, plan_h), aid)

        if need_plan:
            for aid, _ in need_plan:
                res_table.release_all(aid)
                res_table.reserve_node(pos[aid], Interval(t, min(plan_h, t + 1)), aid)
                field.remove_agv(aid)

            goal_map = dict(need_plan)

            def _prio_k(aid_x: str) -> tuple[Any, ...]:
                tk = cur_task[aid_x]
                tier = _agv_commitment_tier(phase[aid_x], tk)
                tier_rank = 0 if (is_cvcr and tier == TIER_EXECUTING_HARD) else (1 if tier == TIER_SOFT else 2)
                urg_b = 12.0 if (tk and tk.is_urgent) else 0.0
                slk = float((tk.due_step - t) if tk else 50)
                cong = 0.0 if (is_static or is_ca) else float(field.slot_Phi(pos[aid_x], t))
                choke_b = -8.0 if (is_cvcr and pos[aid_x] in haven_prof.choke_nodes) else 0.0
                return (tier_rank, slk - urg_b - 3.0 * cong + choke_b, aid_x)

            pq_aids: deque[str] = deque(sorted(goal_map.keys(), key=_prio_k))
            uncommitted: dict[str, tuple[list[tuple[str, int]], list[tuple[str, str, int, int]]]] = {}
            retries: dict[str, int] = {a: 0 for a in goal_map}
            r_lim = 0 if (is_static or is_ca) else (5 if is_cvcr else 3)

            while pq_aids:
                aid = pq_aids.popleft()
                gl = goal_map[aid]
                t_eff = min(plan_h - 1, t + service_rem[aid])
                res_table.release_all(aid)
                res_table.reserve_node(pos[aid], Interval(t, t_eff + 1), aid)
                use_r = empty_res if is_static else res_table
                n_pts, e_segs, hits = _plan_for_agv(aid, gl, t_eff, use_r)
                tk_aid = cur_task[aid]
                svc_hold = (tk_aid.service_steps + 6) if (tk_aid is not None and phase[aid] == "to_pickup") else 2
                if n_pts is not None:
                    pre = [(pos[aid], tk_i) for tk_i in range(t, t_eff)]
                    full_pts = pre + n_pts
                    if not is_static:
                        _reserve_workshop_path(res_table, aid, full_pts, e_segs, park_until=min(plan_h, full_pts[-1][1] + svc_hold), planner=planner)
                    plan_steps[aid] = {st: nd for nd, st in full_pts}
                    plan_edges[aid] = {seg[2]: seg for seg in e_segs}
                    plan_end_t[aid] = full_pts[-1][1]
                    uncommitted[aid] = (full_pts, e_segs)
                    cooldown[aid] = 0
                    fail_plan_cnt[aid] = 0
                    if not (is_static or is_ca):
                        field.add_agv_path(aid, full_pts, horizon=plan_h)
                    continue

                if retries[aid] < r_lim:
                    cands = [j for j in uncommitted if j != aid and int(hits.get(j, 0)) > 0]
                    # Also allow local head-on recovery if an outer stationary/unstarted blocker blocks a leaf exit
                    if not cands and not is_static and not is_ca:
                        cands = [
                            j for j, hv in hits.items()
                            if j in in_edge and j != aid and int(hv) > 0 and in_edge[j] is None and service_rem[j] == 0
                            and pos[j] != plan_steps[j].get(t + 1, pos[j])
                        ]
                    if cands:
                        eff_lpr = lpr_selector_mode if lpr_selector_mode != "auto" else (
                            "dfl" if is_cvcr else ("impact" if is_ialpr else "max_hits")
                        )
                        if eff_lpr == "dfl":
                            bj = _select_blocker_cvcr_dfl(
                                aid, cands, hits, pos, phase, cur_task, uncommitted,
                                haven_prof, net, field, t, n_agvs, horizon,
                                len(done_times), n_tasks, dfl_critic,
                            )
                        elif eff_lpr == "impact":
                            bj = max(
                                cands,
                                key=lambda j: (impact_aware_blocker_score(aid, j, hits, max(1, len(uncommitted.get(j, ([], []))[0])), field.slot_Phi(pos[j], t)), j),
                            )
                        else:
                            bj = max(cands, key=lambda j: (int(hits.get(j, 0)), max(1, len(uncommitted.get(j, ([], []))[0])), j))
                    else:
                        bj = None
                    if bj is not None:
                        lpr_attempts_cnt += 1
                        was_uncomm = bj in uncommitted
                        saved_j = uncommitted.pop(bj) if was_uncomm else ([], [])
                        saved_outer_bj = (dict(plan_steps[bj]), dict(plan_edges[bj]), int(plan_end_t[bj]))
                        res_table.release_all(bj)
                        res_table.reserve_node(pos[bj], Interval(t, min(plan_h, t + 1)), bj)
                        field.remove_agv(bj)
                        retries[aid] += 1
                        n2, e2, _ = _plan_for_agv(aid, gl, t_eff, res_table)
                        if n2 is not None:
                            pre = [(pos[aid], tk_i) for tk_i in range(t, t_eff)]
                            full_pts = pre + n2
                            _reserve_workshop_path(res_table, aid, full_pts, e2, park_until=min(plan_h, full_pts[-1][1] + svc_hold), planner=planner)
                            plan_steps[aid] = {st: nd for nd, st in full_pts}
                            plan_edges[aid] = {seg[2]: seg for seg in e2}
                            plan_end_t[aid] = full_pts[-1][1]
                            uncommitted[aid] = (full_pts, e2)
                            fail_plan_cnt[aid] = 0
                            field.add_agv_path(aid, full_pts, horizon=plan_h)
                            if was_uncomm or bj in goal_map:
                                pq_aids.append(bj)
                            else:
                                tk_b = cur_task[bj]
                                if tk_b is not None:
                                    goal_map[bj] = tk_b.pickup if phase[bj] == "to_pickup" else tk_b.dropoff
                                    retries.setdefault(bj, 0)
                                    pq_aids.append(bj)
                            continue
                        lpr_wasted_cnt += 1
                        if was_uncomm:
                            _reserve_workshop_path(res_table, bj, saved_j[0], saved_j[1], park_until=min(plan_h, saved_j[0][-1][1] + 2), planner=planner)
                            field.add_agv_path(bj, saved_j[0], horizon=plan_h)
                            uncommitted[bj] = saved_j
                        else:
                            # Restore outer committed blocker cleanly on failed rollback probe
                            plan_steps[bj], plan_edges[bj], plan_end_t[bj] = saved_outer_bj
                            pts_bj = [(saved_outer_bj[0][st], st) for st in sorted(saved_outer_bj[0]) if st >= t]
                            if not pts_bj:
                                pts_bj = [(pos[bj], t)]
                            segs_bj = [saved_outer_bj[1][st] for st in sorted(saved_outer_bj[1]) if st >= t]
                            _reserve_workshop_path(res_table, bj, pts_bj, segs_bj, park_until=min(plan_h, pts_bj[-1][1] + 2), planner=planner)
                            field.add_agv_path(bj, pts_bj, horizon=plan_h)

                # CVCR Safe-Haven Yield Fallback when stalled inside a 1-lane choke / spur
                fail_plan_cnt[aid] += 1
                if is_cvcr and fail_plan_cnt[aid] >= 2 and (pos[aid] in haven_prof.choke_nodes or pos[aid] in haven_prof.leaf_spurs):
                    staged_ok = False
                    for hv in sorted(haven_targets, key=lambda h_n: (_tt_steps(net, pos[aid], h_n, tt_cache), h_n)):
                        if hv == pos[aid]:
                            continue
                        n_hv, e_hv, _ = _plan_for_agv(aid, hv, t_eff, res_table)
                        if n_hv is not None:
                            pre = [(pos[aid], tk_i) for tk_i in range(t, t_eff)]
                            full_pts = pre + n_hv
                            _reserve_workshop_path(res_table, aid, full_pts, e_hv, park_until=min(plan_h, full_pts[-1][1] + 2), planner=planner)
                            plan_steps[aid] = {st: nd for nd, st in full_pts}
                            plan_edges[aid] = {seg[2]: seg for seg in e_hv}
                            plan_end_t[aid] = full_pts[-1][1]
                            uncommitted[aid] = (full_pts, e_hv)
                            field.add_agv_path(aid, full_pts, horizon=plan_h)
                            cooldown[aid] = 1
                            fail_plan_cnt[aid] = 0
                            staged_ok = True
                            break
                    if staged_ok:
                        continue

                # Failed locally -> safe short hold & cooldown
                cooldown[aid] = 1
                res_table.reserve_node(pos[aid], Interval(t, min(plan_h, t + 2)), aid)
                plan_steps[aid] = {t: pos[aid]}
                plan_edges[aid] = {}
                plan_end_t[aid] = t

        # Physical step execution over `t -> t + 1`: Nodes-First Fixed-Point Mutex Guard
        def _shift_schedule_one_tick(aid_x: str) -> None:
            plan_steps[aid_x] = {
                (st + 1 if st >= t else st): nd
                for st, nd in plan_steps[aid_x].items()
            }
            plan_steps[aid_x][t] = pos[aid_x]
            plan_edges[aid_x] = {
                (st + 1 if st >= t else st): (seg[0], seg[1], seg[2] + (1 if seg[2] >= t else 0), seg[3] + (1 if seg[3] > t else 0))
                for st, seg in plan_edges[aid_x].items()
            }
            plan_end_t[aid_x] = int(plan_end_t[aid_x]) + 1

        next_pos = dict(pos)
        next_in_edge = dict(in_edge)
        staying_on_node: set[str] = set()
        want_land_edge: list[tuple[str, str, str]] = []  # (aid, u_e, v_e) for rem_e == 1
        hold_mid_edge: set[str] = set()
        want_depart: dict[str, tuple[str, int]] = {}     # aid -> (v_e, n_st)

        for aid in agv_ids:
            if cur_task[aid] is not None:
                busy_steps += 1
            ie = in_edge[aid]
            if ie is not None:
                u_e, v_e, rem_e = ie
                if motion_stalls.get((aid, t), False):
                    hold_mid_edge.add(aid)
                    conflict_wait_cnt += 1
                elif rem_e > 1:
                    next_in_edge[aid] = (u_e, v_e, rem_e - 1)
                    hold_mid_edge.add(aid)
                else:
                    want_land_edge.append((aid, u_e, v_e))
            else:
                if service_rem[aid] > 0:
                    service_rem[aid] -= 1
                    staying_on_node.add(aid)
                elif motion_stalls.get((aid, t), False) and cur_task[aid] is not None:
                    staying_on_node.add(aid)
                    conflict_wait_cnt += 1
                else:
                    seg = plan_edges[aid].get(t)
                    if seg is not None and seg[0] == pos[aid]:
                        want_depart[aid] = (seg[1], seg[3] - seg[2])
                    else:
                        staying_on_node.add(aid)
                        if cur_task[aid] is not None:
                            tk_c = cur_task[aid]
                            gl_c = tk_c.pickup if phase[aid] == "to_pickup" else tk_c.dropoff
                            if pos[aid] != gl_c:
                                conflict_wait_cnt += 1

        # Fixed-point convergence resolving edge arrivals + departures around staying nodes
        land_ok = {aid for aid, _, _ in want_land_edge}
        depart_ok = set(want_depart.keys())

        for _ in range(len(agv_ids) + 4):
            changed = False
            node_occ: dict[str, int] = {}
            edge_occ: dict[tuple[str, str], int] = {}
            dir_occ: dict[tuple[str, str], int] = {}

            for aid in staying_on_node:
                node_occ[pos[aid]] = node_occ.get(pos[aid], 0) + 1
            for aid in hold_mid_edge:
                u_e, v_e, _ = in_edge[aid]  # type: ignore[misc]
                phys = physical_edge_id(u_e, v_e)
                edge_occ[phys] = edge_occ.get(phys, 0) + 1
                dir_occ[(u_e, v_e)] = dir_occ.get((u_e, v_e), 0) + 1
            for aid, u_e, v_e in want_land_edge:
                if aid in land_ok:
                    node_occ[v_e] = node_occ.get(v_e, 0) + 1
                else:
                    phys = physical_edge_id(u_e, v_e)
                    edge_occ[phys] = edge_occ.get(phys, 0) + 1
                    dir_occ[(u_e, v_e)] = dir_occ.get((u_e, v_e), 0) + 1
            for aid, (v_e, n_st) in want_depart.items():
                if aid not in depart_ok:
                    node_occ[pos[aid]] = node_occ.get(pos[aid], 0) + 1
                else:
                    u_e = pos[aid]
                    phys = physical_edge_id(u_e, v_e)
                    edge_occ[phys] = edge_occ.get(phys, 0) + 1
                    dir_occ[(u_e, v_e)] = dir_occ.get((u_e, v_e), 0) + 1
                    if n_st <= 1:
                        node_occ[v_e] = node_occ.get(v_e, 0) + 1

            # Check deadlocked arrivals first (`rem_e == 1`): never overfill `v_e`
            for aid, u_e, v_e in sorted(want_land_edge, key=lambda x: x[0], reverse=True):
                if aid in land_ok and node_occ.get(v_e, 0) > net.node(v_e).capacity:
                    land_ok.remove(aid)
                    node_occ[v_e] -= 1
                    changed = True

            # Check departures: reject external / lower-commitment contenders first so dead-end & choke occupants can exit cleanly
            dep_reject_order = sorted(
                list(depart_ok),
                key=lambda a: (
                    1 if len(net.out_edges(pos[a])) <= 1 else 0,
                    1 if pos[a] in haven_prof.choke_nodes else 0,
                    1 if _agv_commitment_tier(phase[a], cur_task[a]) == TIER_EXECUTING_HARD else 0,
                    a,
                ),
            )
            for aid in dep_reject_order:
                v_e, n_st = want_depart[aid]
                u_e = pos[aid]
                phys = physical_edge_id(u_e, v_e)
                e = net.edge(u_e, v_e)
                ecap = min(e.capacity, net.edge(v_e, u_e).capacity) if net.has_edge(v_e, u_e) else e.capacity
                opp_cnt = dir_occ.get((v_e, u_e), 0)
                bad_edge = (edge_occ.get(phys, 0) > ecap) or (ecap <= 1 and opp_cnt > 0)
                bad_node = (n_st <= 1 and node_occ.get(v_e, 0) > net.node(v_e).capacity)
                # Dead-end leaf guard: never enter degree-1 terminal edge if occupant is staying there
                bad_leaf = (
                    n_st > 1
                    and len(net.out_edges(v_e)) <= 1
                    and node_occ.get(v_e, 0) >= net.node(v_e).capacity
                )
                if bad_edge or bad_node or bad_leaf:
                    depart_ok.remove(aid)
                    edge_occ[phys] -= 1
                    dir_occ[(u_e, v_e)] -= 1
                    if n_st <= 1:
                        node_occ[v_e] -= 1
                    changed = True
            if not changed:
                break

        # Commit exact verified state transitions and shift delayed schedules cleanly
        for aid in hold_mid_edge:
            if motion_stalls.get((aid, t), False):
                _shift_schedule_one_tick(aid)
                u_e, v_e, rem_e = in_edge[aid]  # type: ignore[misc]
                if t + rem_e + 1 <= plan_h:
                    res_table.reserve_edge(u_e, v_e, Interval(t, t + rem_e + 1), aid)
            else:
                stall_ticks[aid] = 0
        for aid, u_e, v_e in want_land_edge:
            if aid in land_ok:
                next_pos[aid] = v_e
                next_in_edge[aid] = None
                stall_ticks[aid] = 0
            else:
                conflict_wait_cnt += 1
                stall_ticks[aid] += 1
                _shift_schedule_one_tick(aid)
                if t + 2 <= plan_h:
                    res_table.reserve_edge(u_e, v_e, Interval(t, t + 2), aid)
        for aid in staying_on_node:
            if cur_task[aid] is not None and motion_stalls.get((aid, t), False):
                _shift_schedule_one_tick(aid)
                if t + 2 <= plan_h and res_table.is_node_free(pos[aid], Interval(t + 1, t + 2), net.node(pos[aid]).capacity, exclude_agv=aid):
                    res_table.reserve_node(pos[aid], Interval(t + 1, t + 2), aid)
            elif cur_task[aid] is not None and service_rem[aid] == 0:
                stall_ticks[aid] += 1
            else:
                stall_ticks[aid] = 0
        for aid, (v_e, n_st) in want_depart.items():
            if aid in depart_ok:
                stall_ticks[aid] = 0
                if n_st <= 1:
                    next_pos[aid] = v_e
                else:
                    next_in_edge[aid] = (pos[aid], v_e, n_st - 1)
            else:
                conflict_wait_cnt += 1
                stall_ticks[aid] += 1
                if is_cvcr and stall_ticks[aid] >= 6:
                    # Break infinite schedule shift loops in CVCR mode -> force immediate local replan next tick
                    plan_steps[aid] = {t + 1: pos[aid]}
                    plan_edges[aid] = {}
                    plan_end_t[aid] = t + 1
                    stall_ticks[aid] = 0
                else:
                    _shift_schedule_one_tick(aid)

        # Final independent safety verification across nodes & edges at t+1
        occ_node_next: dict[str, int] = {}
        occ_phys_edges: dict[tuple[str, str], int] = {}
        for aid in agv_ids:
            if next_in_edge[aid] is not None:
                phys = physical_edge_id(next_in_edge[aid][0], next_in_edge[aid][1])
                occ_phys_edges[phys] = occ_phys_edges.get(phys, 0) + 1
            else:
                occ_node_next[next_pos[aid]] = occ_node_next.get(next_pos[aid], 0) + 1
        for nd_k, cnt in occ_node_next.items():
            if cnt > net.node(nd_k).capacity:
                violated = True
        for phys_k, cnt in occ_phys_edges.items():
            e = net.edge(phys_k[0], phys_k[1]) if net.has_edge(phys_k[0], phys_k[1]) else net.edge(phys_k[1], phys_k[0])
            ecap = min(e.capacity, net.edge(phys_k[1], phys_k[0]).capacity) if (net.has_edge(phys_k[0], phys_k[1]) and net.has_edge(phys_k[1], phys_k[0])) else e.capacity
            if cnt > ecap:
                violated = True

        pos = next_pos
        in_edge = next_in_edge

    wall_ms = (time.perf_counter() - t0_wall) * 1000.0
    comp = len(done_times)
    mean_ct = float(sum(done_times)) / max(1, comp) if comp else float(makespan)
    on_time = float(on_time_cnt) / max(1, n_tasks)
    util = float(busy_steps) / max(1, max(1, makespan) * n_agvs)
    return WorkshopRunResult(
        layout=layout,
        aisle_capacity=aisle_capacity,
        n_tasks=n_tasks,
        n_agvs=n_agvs,
        seed=int(seed),
        dispatcher=dispatcher,
        router=router,
        delay_prob=round(float(delay_prob), 3),
        completed_tasks=comp,
        makespan=makespan,
        mean_completion_time=round(mean_ct, 2),
        on_time_rate=round(on_time, 4),
        agv_utilization=round(util, 4),
        conflict_wait_count=conflict_wait_cnt,
        total_expansions=total_exp,
        wall_ms=round(wall_ms, 2),
        zero_conflicts=not violated,
        lpr_attempts=lpr_attempts_cnt,
        lpr_wasted=lpr_wasted_cnt,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    wu_net = _load_wu26_net()
    dfl_critic = _load_dfl_critic()
    print(
        f"=== Workshop Bridge Evaluation (Wu EJOR'26={wu_net is not None}, DFL Critic={dfl_critic is not None}) ===",
        flush=True,
    )

    # Exp W1: Decoupled Upper-Layer (FIFO / Nearest / Wu26 / Wu26+DFL) x Lower-Layer (Static / CA* / CPS-A* / CPS+CVCR+DFL)
    combos_w1 = [
        ("FIFO_RoundRobin", "Static_Hold"),
        ("FIFO_RoundRobin", "CA_Star"),
        ("FIFO_RoundRobin", "CPS_AStar"),
        ("Nearest_Idle", "CPS_AStar"),
        ("Wu26_SemiCircle", "CA_Star"),
        ("Wu26_SemiCircle", "CPS_AStar"),
        ("Wu26_SemiCircle", "CPS_IALPR_EALR"),
        ("Wu26_SemiCircle_GRU", "CPS_AStar"),
        ("Wu26_SemiCircle", "CPS_CVCR_DFL"),
        ("Wu26_SemiCircle_DFL", "CPS_CVCR_DFL"),
    ]
    w1_summary: list[dict[str, Any]] = []
    for dp, rt in combos_w1:
        rows = [
            run_workshop_episode(
                layout="demo17", aisle_capacity=1, n_tasks=30, n_agvs=4,
                seed=s, dispatcher=dp, router=rt, delay_prob=0.10,
                wu_net=wu_net, dfl_critic=dfl_critic,
            )
            for s in (42, 43, 44)
        ]
        rec = {
            "dispatcher": dp,
            "router": rt,
            "completed": round(sum(r.completed_tasks for r in rows) / 3.0, 1),
            "makespan": round(sum(r.makespan for r in rows) / 3.0, 1),
            "mean_ct": round(sum(r.mean_completion_time for r in rows) / 3.0, 2),
            "on_time": round(sum(r.on_time_rate for r in rows) / 3.0, 4),
            "util": round(sum(r.agv_utilization for r in rows) / 3.0, 4),
            "waits": round(sum(r.conflict_wait_count for r in rows) / 3.0, 1),
            "safe": all(r.zero_conflicts for r in rows),
        }
        w1_summary.append(rec)
        print(
            f"  [W1 demo17 | {dp:19s} + {rt:14s}] done={rec['completed']:4.1f}/30 MS={rec['makespan']:5.1f} "
            f"OnTime={rec['on_time']*100:5.1f}% Util={rec['util']*100:4.1f}% Waits={rec['waits']:5.1f} Safe={rec['safe']}",
            flush=True,
        )

    # Exp W2: Taskbook R5 Scale x Fleet Size x Lane Load (`CPS_AStar` vs converged `CPS_CVCR_DFL`)
    w2_rows: list[dict[str, Any]] = []
    for cap in (1, 2):
        for nt in (20, 50, 100):
            for na in (2, 4, 6, 8):
                r_base = run_workshop_episode(
                    layout="demo17", aisle_capacity=cap, n_tasks=nt, n_agvs=na,
                    seed=42, dispatcher="Wu26_SemiCircle", router="CPS_AStar",
                    delay_prob=0.0, horizon=max(900, (nt * 40) // max(1, na)), wu_net=None,
                )
                r_cvcr = run_workshop_episode(
                    layout="demo17", aisle_capacity=cap, n_tasks=nt, n_agvs=na,
                    seed=42, dispatcher="Wu26_SemiCircle_DFL", router="CPS_CVCR_DFL",
                    delay_prob=0.0, horizon=max(900, (nt * 40) // max(1, na)),
                    wu_net=wu_net, dfl_critic=dfl_critic,
                )
                w2_rows.append({
                    "cap": cap, "n_tasks": nt, "n_agvs": na,
                    "completed": r_base.completed_tasks, "makespan": r_base.makespan,
                    "mean_ct": r_base.mean_completion_time, "on_time": r_base.on_time_rate,
                    "util": r_base.agv_utilization, "waits": r_base.conflict_wait_count, "safe": r_base.zero_conflicts,
                    "cvcr_completed": r_cvcr.completed_tasks, "cvcr_makespan": r_cvcr.makespan,
                    "cvcr_on_time": r_cvcr.on_time_rate, "cvcr_waits": r_cvcr.conflict_wait_count,
                    "cvcr_safe": r_cvcr.zero_conflicts,
                })
    print("  [W2 Taskbook R5 Scale x Fleet Matrix (`demo17`, `CPS_AStar` vs `CPS_CVCR_DFL`)] completed.", flush=True)

    # Exp W3: Detailed `rail_freight_workshop_v1` (62 nodes / 144 edges) Comparison under Urgent + 12% Delay
    w3_summary: list[dict[str, Any]] = []
    for dp, rt in (
        ("FIFO_RoundRobin", "CA_Star"),
        ("Wu26_SemiCircle", "CA_Star"),
        ("Wu26_SemiCircle", "CPS_AStar"),
        ("Wu26_SemiCircle", "CPS_IALPR_EALR"),
        ("Wu26_SemiCircle_GRU", "CPS_AStar"),
        ("Wu26_SemiCircle", "CPS_CVCR_DFL"),
        ("Wu26_SemiCircle_GRU", "CPS_CVCR_DFL"),
        ("Wu26_SemiCircle_DFL", "CPS_CVCR_DFL"),
    ):
        rows = [
            run_workshop_episode(
                layout="workshop_v1", aisle_capacity=1, n_tasks=30, n_agvs=4,
                seed=s, dispatcher=dp, router=rt, delay_prob=0.12,
                wu_net=wu_net, dfl_critic=dfl_critic,
            )
            for s in (42, 43)
        ]
        rec = {
            "dispatcher": dp,
            "router": rt,
            "completed": round(sum(r.completed_tasks for r in rows) / 2.0, 1),
            "makespan": round(sum(r.makespan for r in rows) / 2.0, 1),
            "mean_ct": round(sum(r.mean_completion_time for r in rows) / 2.0, 2),
            "on_time": round(sum(r.on_time_rate for r in rows) / 2.0, 4),
            "util": round(sum(r.agv_utilization for r in rows) / 2.0, 4),
            "waits": round(sum(r.conflict_wait_count for r in rows) / 2.0, 1),
            "safe": all(r.zero_conflicts for r in rows),
        }
        w3_summary.append(rec)
        print(
            f"  [W3 workshop_v1 (62n/144e) | {dp:19s} + {rt:14s}] done={rec['completed']:4.1f}/30 MS={rec['makespan']:5.1f} "
            f"OnTime={rec['on_time']*100:5.1f}% Waits={rec['waits']:5.1f} Safe={rec['safe']}",
            flush=True,
        )

    elapsed_s = round(time.perf_counter() - t_start, 2)
    payload = {
        "protocol": "TRACK-R-CPSA-01 Workshop Bridge (Wu EJOR'26 + CPS-A* + CVCR Havens/Commit + DFL Critic + Taskbook R2-R5)",
        "elapsed_s": elapsed_s,
        "wu26_weights_loaded": wu_net is not None,
        "dfl_critic_loaded": dfl_critic is not None,
        "w1_decoupled_demo17": w1_summary,
        "w2_taskbook_scale_fleet_matrix": w2_rows,
        "w3_workshop_v1_delayed": w3_summary,
    }
    (OUT_DIR / "workshop_bridge_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# 铁路货车检修车间桥接实测报告（`Wu EJOR'26` + `CPS-A*` + `CVCR/DFL` + Converged Workshop Constraints）",
        "",
        f"> 运行耗时：`{elapsed_s}s` · 纯 NumPy `WuEJOR26GRUAttentionNet`=`{wu_net is not None}` · `DFLAdvantageEnsemble`=`{dfl_critic is not None}` · 双层收敛护栏（零大厂生产 DAG 耦合）。",
        "",
        "## 1. Exp W1：`17` 节点检修环网（`aisle_capacity=1`, 30 项加急/常规任务, 4 台 AGV, 10% 运行延迟, 3 种子）",
        "",
        "| 上层分配 (`Task Assignment`) | 底层路由 (`Path & Conflict`) | 完工数 | 总完成时间 (`Makespan`) | 任务准时率 | AGV 利用率 | 冲突等待次数 | 零碰撞 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in w1_summary:
        md.append(
            f"| `{r['dispatcher']}` | `{r['router']}` | {r['completed']:.1f}/30 | {r['makespan']:.1f} | "
            f"{r['on_time']*100:.1f}% | {r['util']*100:.1f}% | {r['waits']:.1f} | {r['safe']} |"
        )

    md.extend([
        "",
        "## 2. Exp W2：车间任务规模 × AGV 编队配置敏感性矩阵（`demo17`, `CPS_AStar` vs `CPS_CVCR_DFL`）",
        "",
        "| 容量 | 任务数 | AGV 数 | `CPS-A*` 完工 | `CPS-A*` MS | `CPS-A*` 准时率 | `CPS-A*` 等待 | `CVCR+DFL` 完工 | `CVCR+DFL` MS | `CVCR+DFL` 准时率 | `CVCR+DFL` 等待 | 零碰撞 |",
        "| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in w2_rows:
        both_safe = bool(r["safe"] and r["cvcr_safe"])
        md.append(
            f"| {r['cap']} | {r['n_tasks']} | {r['n_agvs']} | {r['completed']}/{r['n_tasks']} | {r['makespan']} | "
            f"{r['on_time']*100:.1f}% | {r['waits']} | {r['cvcr_completed']}/{r['n_tasks']} | {r['cvcr_makespan']} | "
            f"{r['cvcr_on_time']*100:.1f}% | {r['cvcr_waits']} | {both_safe} |"
        )

    md.extend([
        "",
        "## 3. Exp W3：`rail_freight_workshop_v1`（62 节点 / 144 有向边, 12% 延误, 30 任务 × 4 AGV）",
        "",
        "| 上层分配 | 底层路由 | 完工数 | Makespan | 准时率 | AGV 利用率 | 冲突等待次数 | 零碰撞 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in w3_summary:
        md.append(
            f"| `{r['dispatcher']}` | `{r['router']}` | {r['completed']:.1f}/30 | {r['makespan']:.1f} | "
            f"{r['on_time']*100:.1f}% | {r['util']*100:.1f}% | {r['waits']:.1f} | {r['safe']} |"
        )
    (OUT_DIR / "workshop_bridge_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"=== Wrote {OUT_DIR / 'workshop_bridge_report.md'} ({elapsed_s}s) ===", flush=True)


if __name__ == "__main__":
    main()
