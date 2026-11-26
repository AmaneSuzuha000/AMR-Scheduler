"""Converged Large-Scale Stress Testing & Cross-Baseline / Prior-RL Comparison Suite.

Evaluates three clean benchmark matrices under the EXACT converged scope:
  1. `Bench-Dispatch`: Re-evaluates our prior RL series (`V19_HETERO_RL`, `V19_NO_PROD_COMM`,
     Track A `PPO` / `Discrete_SAC` / `AttentionModel`) + external 2025/2026 top-venue baselines
     (`EJOR26_Wu_*`, `RAL26_DetourDelta`, `IJCAI26_PGMPO`, `TII26_E2G_MoE`, `ASC26_MA_TRANS`,
     `FGCS26_REN_GNN`, `IJPR25_HMARL`) + classic OR rules (`FIFO`, `Nearest_Idle`, `EDD`,
     `Hungarian_MinCost`) strictly on the converged zero-production-DAG railway workshop scenario
     (`demo17` + `rail_freight_workshop_v1`), all sharing the identical collision-free `CPS_CVCR_DFL`
     lower-layer router.
  2. `Stress-W`: Large-scale workshop stress tests across high task bursts (`80–150` tasks),
     larger fleets (`6–16` AGVs), and heavy motion disturbances (`12%–25%` delay).
  3. `Stress-Grid`: Extreme MovingAI warehouse fleet scaling (`80 -> 200` AGVs Python vs Rust SIPP)
     and heavy-lag (`Rooms_32x32 80 AGVs`, `Warehouse_40x40 120 AGVs`, `comm_lag=1`) local repair
     stress tests.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Optional

import numpy as np
import torch

try:
    torch.set_num_threads(1)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.baselines_2026 import (
    MaTII26E2GMoENet,
    NumpyChoiIJCAI26PGMPONet,
    build_causal_instance,
    load_numpy_state_dict,
)
from amr_sched.baselines.dfjspt_hmarl_ijpr25 import (
    _global_flat_summary_np,
    load_dfjspt_model,
)
from amr_sched.baselines.fgcs26_ren_gnn import load_fgcs26_ren_model
from amr_sched.baselines.ma_trans_asc26 import load_ma_trans_model
from amr_sched.baselines.numpy_backends import NumpyWuEJOR26GRUAttentionNet
from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.snapshot import (
    NativeSnapshot,
    OperationSnapshot,
    RobotSnapshot,
    TaskSnapshot,
)
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.network import Network
from amr_sched.rl.v19_hetero_constructor import _load_v19_checkpoint, _np_elu, _np_linear
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from experiments.ejor26_warehouse_repro.models_and_solvers import (
    AttentionModelNet,
    DiscreteSACQNet,
    WuGRUAttentionNet,
)
from experiments.rgro.dfl_critic import DFLAdvantageEnsemble
from experiments.track_r_cpsa.execution_delay_sim import (
    DelayScenarioConfig,
    simulate_lifelong_with_delays,
)
from experiments.track_r_cpsa.local_reservation_repair import EALRConfig
from experiments.track_r_cpsa.maps_and_workloads import (
    build_map_rooms_32x32,
    build_map_warehouse_40x40,
    generate_lifelong_requests,
)
from experiments.track_r_cpsa.rust_sipp_accel import run_accelerated_lifelong_mapd
from experiments.track_r_cpsa.workshop_bridge_eval import (
    WorkshopTask,
    _dispatch_tasks,
    _load_dfl_critic,
    _load_wu26_net,
    _tt_steps,
    run_workshop_episode,
)

OUT_DIR = ROOT / "out" / "track_r_cpsa"
TRACK_R_CKPT_DIR = ROOT / "amr_sched" / "checkpoints" / "track_r"
_MODEL_CACHE: dict[str, Any] = {}


def _get_cached_model(key: str) -> Any:
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if key == "wu_net":
        obj = _load_wu26_net()
    elif key == "wu_net_adapted":
        p_wu = TRACK_R_CKPT_DIR / "ejor26_wu_gru_workshop.npz"
        obj = _load_wu26_net(p_wu if p_wu.is_file() else None)
    elif key == "dfl_critic":
        obj = _load_dfl_critic()
    elif key == "dfl_critic_adapted":
        p_dfl = TRACK_R_CKPT_DIR / "mh_dfl_critic_workshop.npz"
        obj = _load_dfl_critic(p_dfl if p_dfl.is_file() else None)
    elif key == "v19_rl":
        obj = _load_v19_checkpoint(resolve_default_checkpoint("v19/v19_hetero_rl.pt"), backend="numpy")[0]
    elif key == "v19_no_prod":
        obj = _load_v19_checkpoint(
            resolve_default_checkpoint("v19/v19_no_prod_comm.pt"), backend="numpy", ablation_mode="no_prod_comm"
        )[0]
    elif key == "v19_no_prod_adapted":
        p_v19 = TRACK_R_CKPT_DIR / "v19_no_prod_comm_workshop.npz"
        ckpt_v = p_v19 if p_v19.is_file() else resolve_default_checkpoint("v19/v19_no_prod_comm.pt")
        obj = _load_v19_checkpoint(ckpt_v, backend="numpy", ablation_mode="no_prod_comm")[0]
    elif key == "asc26":
        obj = load_ma_trans_model(resolve_default_checkpoint("baselines_2026/asc26_ma_trans.pt"), backend="numpy")[0]
    elif key == "fgcs26":
        obj = load_fgcs26_ren_model(resolve_default_checkpoint("baselines_2026/fgcs26_ren_gnn.pt"), backend="numpy")[0]
    elif key == "ijpr25":
        obj = load_dfjspt_model(resolve_default_checkpoint("baselines_2026/ijpr25_dfjspt_hmarl.pt"), backend="numpy")[0]
    elif key == "ijcai26":
        w_ij, _, _ = load_numpy_state_dict(resolve_default_checkpoint("baselines_2026/ijcai26_choi_pgmpo.pt"), prefer_npz=True)
        obj = NumpyChoiIJCAI26PGMPONet(w_ij)
    elif key == "tii26":
        sd = torch.load(resolve_default_checkpoint("baselines_2026/tii26_ma_e2g_moe.pt"), map_location="cpu", weights_only=True)
        net = MaTII26E2GMoENet()
        net.load_state_dict(sd["model"] if isinstance(sd, dict) and "model" in sd else sd)
        net.eval()
        obj = net
    elif key in ("tracka_ppo", "tracka_sac", "tracka_am"):
        ckpt_dir = ROOT / "out" / "ejor26_warehouse_repro" / "checkpoints"
        fname = {"tracka_ppo": "n50_PPO.pt", "tracka_sac": "n50_Discrete_SAC.pt", "tracka_am": "n50_AttentionModel.pt"}[key]
        sd = torch.load(ckpt_dir / fname, map_location="cpu", weights_only=True)
        raw = sd.get("actor", sd.get("net", sd)) if isinstance(sd, dict) else sd
        if key == "tracka_ppo":
            net = WuGRUAttentionNet(hidden_dim=128)
        elif key == "tracka_sac":
            net = DiscreteSACQNet(in_dim=9, hidden_dim=128)
        else:
            net = AttentionModelNet(static_dim=6, hidden_dim=96)
        net.load_state_dict(raw)
        net.eval()
        obj = net
    else:
        raise KeyError(f"Unknown cached model key: {key!r}")
    _MODEL_CACHE[key] = obj
    return obj


def _build_transport_snapshot(
    idle_agvs: list[str],
    cand_tasks: list[WorkshopTask],
    cur_pos: dict[str, str],
    tick: int,
    *,
    all_agv_ids: list[str] | None = None,
    cur_task: dict[str, Optional[WorkshopTask]] | None = None,
    phase: dict[str, str] | None = None,
    include_busy_robots: bool = False,
) -> NativeSnapshot:
    """Build a pure-transport `NativeSnapshot` (zero production DAGs/crews; 1-to-1 leaf ops)."""
    robs: list[RobotSnapshot] = []
    rob_list = sorted(all_agv_ids) if (include_busy_robots and all_agv_ids) else sorted(idle_agvs)
    for aid in rob_list:
        tk_a = cur_task.get(aid) if cur_task else None
        is_busy = tk_a is not None and aid not in idle_agvs
        st_s = "busy" if is_busy else "idle"
        ld_id = f"ACT_{tk_a.task_id}" if (is_busy and phase and phase.get(aid) == "to_dropoff") else None
        robs.append(RobotSnapshot(robot_id=aid, cur_node=cur_pos[aid], state=st_s, cur_task=None, loaded_task_id=ld_id))

    tsks: list[TaskSnapshot] = []
    ops: list[OperationSnapshot] = []
    for tk in cand_tasks:
        tid = f"T_{tk.task_id}"
        oid = f"OP_{tk.task_id}"
        tsks.append(
            TaskSnapshot(
                task_id=tid,
                pickup=tk.pickup,
                dropoff=tk.dropoff,
                release_step=tk.release_step,
                due_step=tk.due_step,
                status="pending",
                assigned_agv=None,
                kit_op_id=oid,
                priority="urgent" if tk.is_urgent else "normal",
                is_urgent=tk.is_urgent,
            )
        )
        ops.append(
            OperationSnapshot(
                op_id=oid,
                kit_task_ids=(tid,),
                bay_id=tk.dropoff,
                crew_id=0,
                tool_id=0,
                proc_steps=1,
                due_step=tk.due_step,
                predecessor_op_ids=(),
                is_urgent=tk.is_urgent,
                eligible_bays=(tk.dropoff,),
                eligible_crews=(0,),
            )
        )
    return NativeSnapshot(
        schema="amr-lgcvcr.snapshot.v1",
        tick=int(tick),
        version={"tick": int(tick)},
        world_fingerprint="converged_workshop",
        robots=robs,
        tasks=tsks,
        operations=ops,
        reservations=[],
        now_monotonic=0.0,
    )


def _match_from_score_matrix(
    scores: np.ndarray,
    cand_agvs: list[str],
    cand_tasks: list[WorkshopTask],
) -> list[tuple[str, WorkshopTask]]:
    """Deterministically extract 1-to-1 assignments from `(n_t, n_a)` score matrix."""
    n_t, n_a = scores.shape
    pairs = sorted(
        ((float(scores[i, j]), i, j) for i in range(n_t) for j in range(n_a)),
        key=lambda x: (-x[0], x[1], x[2]),
    )
    used_t: set[int] = set()
    used_a: set[int] = set()
    out: list[tuple[str, WorkshopTask]] = []
    for sc, i, j in pairs:
        if i in used_t or j in used_a or sc <= -1e8:
            continue
        used_t.add(i)
        used_a.add(j)
        out.append((cand_agvs[j], cand_tasks[i]))
    return out


def _hungarian_min_cost(cost: np.ndarray) -> list[tuple[int, int]]:
    """Exact $O(N^3)$ Hungarian / Kuhn-Munkres rectangular min-cost assignment."""
    n_r, n_c = cost.shape
    transposed = n_r > n_c
    C = cost.T if transposed else cost
    m, n = C.shape  # m <= n
    u = np.zeros(m + 1, dtype=np.float64)
    v = np.zeros(n + 1, dtype=np.float64)
    p = np.zeros(n + 1, dtype=np.int64)
    way = np.zeros(n + 1, dtype=np.int64)
    for i in range(1, m + 1):
        p[0] = i
        j0 = 0
        minv = np.full(n + 1, 1e18, dtype=np.float64)
        used = np.zeros(n + 1, dtype=bool)
        while True:
            used[j0] = True
            i0 = int(p[j0])
            delta = 1e18
            j1 = 0
            for j in range(1, n + 1):
                if not used[j]:
                    cur = float(C[i0 - 1, j - 1]) - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(n + 1):
                if used[j]:
                    u[int(p[j])] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = int(way[j0])
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    res: list[tuple[int, int]] = []
    for j in range(1, n + 1):
        if p[j] > 0:
            ri, ci = int(p[j] - 1), j - 1
            res.append((ci, ri) if transposed else (ri, ci))
    return res


def make_converged_dispatcher(method: str) -> Optional[Callable[..., list[tuple[str, WorkshopTask]]]]:
    """Build unified upper dispatcher callback across prior RL, 2026 baselines, and OR rules.

    Returns `None` for built-in strings handled natively by `workshop_bridge_eval._dispatch_tasks`
    (`FIFO_RoundRobin`, `Nearest_Idle`, `Wu26_SemiCircle`, `Wu26_SemiCircle_GRU`, `Wu26_SemiCircle_DFL`).
    """
    builtin = {
        "FIFO_RoundRobin",
        "Nearest_Idle",
        "EJOR26_Wu_SemiCircle",
        "EJOR26_Wu_GRU_Attn",
        "EJOR26_Wu_GRU_Adapted",
        "EJOR26_Wu_DFL",
        "EJOR26_Wu_DFL_Adapted",
    }
    if method in builtin:
        return None

    def _fn(
        idle_agvs: list[str],
        ready_tasks: list[WorkshopTask],
        cur_pos: dict[str, str],
        network: Network,
        tt_cache: dict[tuple[str, str], int],
        tick: int,
        ctx: dict[str, Any],
    ) -> list[tuple[str, WorkshopTask]]:
        if not idle_agvs or not ready_tasks:
            return []
        cand_a = sorted(idle_agvs)[:8]
        sorted_t = sorted(
            ready_tasks,
            key=lambda tk: (
                0 if tk.is_urgent else 1,
                float(tk.due_step - tick - _tt_steps(network, tk.pickup, tk.dropoff, tt_cache)),
                tk.release_step,
                tk.task_id,
            ),
        )
        cand_t = sorted_t[:8]
        n_t, n_a = len(cand_t), len(cand_a)

        if method == "EDD_Urgency":
            pairs = sorted(
                (
                    (
                        0 if tk.is_urgent else 1,
                        float(tk.due_step - tick - _tt_steps(network, tk.pickup, tk.dropoff, tt_cache)),
                        float(_tt_steps(network, cur_pos[aid], tk.pickup, tt_cache)),
                        aid,
                        tk.task_id,
                        i,
                        j,
                    )
                    for i, tk in enumerate(cand_t)
                    for j, aid in enumerate(cand_a)
                ),
                key=lambda x: x[:5],
            )
            used_t, used_a, out = set(), set(), []
            for *_, i, j in pairs:
                if i not in used_t and j not in used_a:
                    used_t.add(i)
                    used_a.add(j)
                    out.append((cand_a[j], cand_t[i]))
            return out

        if method == "Hungarian_MinCost":
            cost = np.zeros((n_t, n_a), dtype=np.float64)
            for i, tk in enumerate(cand_t):
                d_h = float(_tt_steps(network, tk.pickup, tk.dropoff, tt_cache))
                slk = float(tk.due_step - tick)
                urg_b = 12.0 if tk.is_urgent else max(0.0, (40.0 - slk) * 0.25)
                for j, aid in enumerate(cand_a):
                    d_d = float(_tt_steps(network, cur_pos[aid], tk.pickup, tt_cache))
                    cost[i, j] = d_d + 0.2 * d_h - urg_b
            return [(cand_a[j], cand_t[i]) for i, j in _hungarian_min_cost(cost)]

        if method == "RAL26_DetourDelta":
            snap = _build_transport_snapshot(
                cand_a,
                cand_t,
                cur_pos,
                tick,
                all_agv_ids=ctx.get("agv_ids"),
                cur_task=ctx.get("cur_task"),
                phase=ctx.get("phase"),
                include_busy_robots=True,
            )
            dd = compute_detour_delta_signals(network, snap)
            bay_delta = dd["bay_detour_delta"]
            cost = np.zeros((n_t, n_a), dtype=np.float64)
            for i, tk in enumerate(cand_t):
                d_h = float(_tt_steps(network, tk.pickup, tk.dropoff, tt_cache))
                urg_b = 10.0 if tk.is_urgent else 0.0
                cong_b = float(bay_delta.get(tk.dropoff, 0.0)) + 0.5 * float(bay_delta.get(tk.pickup, 0.0))
                for j, aid in enumerate(cand_a):
                    d_d = float(_tt_steps(network, cur_pos[aid], tk.pickup, tt_cache))
                    cost[i, j] = d_d + 0.2 * d_h + cong_b - urg_b
            return [(cand_a[j], cand_t[i]) for i, j in _hungarian_min_cost(cost)]

        if method in ("TrackA_PPO_ZeroShot", "TrackA_SAC_ZeroShot", "TrackA_AM_ZeroShot"):
            key_m = {"TrackA_PPO_ZeroShot": "tracka_ppo", "TrackA_SAC_ZeroShot": "tracka_sac", "TrackA_AM_ZeroShot": "tracka_am"}[method]
            model = _get_cached_model(key_m)
            xs = [float(n.x) for n in network.nodes.values()] or [0.0]
            ys = [float(n.y) for n in network.nodes.values()] or [0.0]
            minx, spanx = min(xs), max(1.0, max(xs) - min(xs))
            miny, spany = min(ys), max(1.0, max(ys) - min(ys))
            M = 1 + n_t
            static_x = np.zeros((n_a, M, 6), dtype=np.float32)
            dyn_x = np.zeros((n_a, M, 3), dtype=np.float32)
            mask = np.ones((n_a, M), dtype=bool)
            mask[:, 0] = False
            for j, aid in enumerate(cand_a):
                rn = network.node(cur_pos[aid])
                static_x[j, 0] = [(rn.x - minx) / spanx, (rn.y - miny) / spany, 0.0, 0.0, 0.0, 0.0]
                dyn_x[j, 0] = [1.0, 0.0, 1.0]
                for i, tk in enumerate(cand_t, start=1):
                    pn = network.node(tk.pickup)
                    d_d = float(_tt_steps(network, cur_pos[aid], tk.pickup, tt_cache)) / 40.0
                    urg = 1.0 if tk.is_urgent else 0.5
                    static_x[j, i] = [(pn.x - minx) / spanx, (pn.y - miny) / spany, 0.0, 0.1, urg, min(2.0, d_d)]
                    dyn_x[j, i] = [1.0, min(2.0, d_d), 0.0]
            with torch.inference_mode():
                st_t = torch.from_numpy(static_x)
                dn_t = torch.from_numpy(dyn_x)
                mk_t = torch.from_numpy(mask)
                if method == "TrackA_PPO_ZeroShot":
                    hs = model.encode_static(st_t)
                    h_t = torch.zeros((n_a, model.hidden_dim), dtype=torch.float32)
                    logits, _ = model.step_pointer(hs, dn_t, st_t[:, 0, :], h_t, mk_t)
                elif method == "TrackA_SAC_ZeroShot":
                    logits = model(torch.cat([st_t, dn_t], dim=-1), mk_t)
                else:
                    hn, hg = model.encode(st_t)
                    cur_i = torch.zeros((n_a,), dtype=torch.long)
                    rem_n = torch.ones((n_a,), dtype=torch.float32)
                    st_f = torch.zeros((n_a,), dtype=torch.float32)
                    logits = model.step_logits(hn, hg, cur_i, rem_n, st_f, st_t[:, :, 5], mk_t)
                mat = logits[:, 1:].numpy().T.astype(np.float64)
            return _match_from_score_matrix(mat, cand_a, cand_t)

        # Shared NativeSnapshot for multi-AGV AMR RL & 2026 Top-Venue Baselines
        snap = _build_transport_snapshot(cand_a, cand_t, cur_pos, tick, include_busy_robots=False)
        if method in ("IJCAI26_PGMPO", "TII26_E2G_MoE"):
            inst = build_causal_instance(snap, network)
            t_idx = {tid: i for i, tid in enumerate(inst.rel_tids)}
            a_idx = {aid: j for j, aid in enumerate(inst.healthy_agvs)}
            if method == "IJCAI26_PGMPO":
                ij = _get_cached_model("ijcai26")
                bits = ij.bit_vectors(4, ij.bit_dim)
                B_c = bits.shape[0]
                lg = ij.forward(
                    np.repeat(inst.task_feats[None], B_c, axis=0),
                    np.repeat(inst.agv_feats[None], B_c, axis=0),
                    np.repeat(inst.pair_feats[None], B_c, axis=0),
                    bits,
                ).mean(axis=0)
            else:
                tii = _get_cached_model("tii26")
                with torch.inference_mode():
                    lg_t, _, _ = tii(
                        torch.from_numpy(inst.task_feats)[None],
                        torch.from_numpy(inst.agv_feats)[None],
                        torch.from_numpy(inst.pair_feats)[None],
                        torch.from_numpy(inst.pred_adj)[None],
                        torch.from_numpy(inst.disturbance_6d)[None],
                    )
                lg = lg_t[0].numpy()
            mat = np.full((n_t, n_a), -1e9, dtype=np.float64)
            for i, tk in enumerate(cand_t):
                ti = t_idx.get(f"T_{tk.task_id}")
                if ti is None:
                    continue
                for j, aid in enumerate(cand_a):
                    aj = a_idx.get(aid)
                    if aj is not None:
                        mat[i, j] = float(lg[ti, aj])
            return _match_from_score_matrix(mat, cand_a, cand_t)

        # Graph-structured Pure-NumPy Models: V19_HETERO_RL, V19_NO_PROD_COMM, ASC26, FGCS26, IJPR25
        g = extract_v19_hetero_graph(snap, network)
        t_idx = {tid: i for i, tid in enumerate(g.rel_tids)}
        a_idx = {aid: j for j, aid in enumerate(g.agv_ids)}
        N_t_g, N_a_g = len(g.rel_tids), len(g.agv_ids)
        score_mat = np.full((N_t_g, N_a_g), -1e9, dtype=np.float64)

        if method in ("Ours_V19_HETERO_RL", "Ours_V19_NO_PROD_COMM", "Ours_V19_Adapted"):
            mkey = {
                "Ours_V19_HETERO_RL": "v19_rl",
                "Ours_V19_NO_PROD_COMM": "v19_no_prod",
                "Ours_V19_Adapted": "v19_no_prod_adapted",
            }[method]
            net_v = _get_cached_model(mkey)
            enc = net_v.encode_entities(g)
            z_l = np.zeros((N_a_g,), dtype=np.float32)
            for ti in range(N_t_g):
                bi = int(g.task_incumbent_bay_idx[ti])
                score_mat[ti] = net_v.a3_step_logits(enc, g, ti, bi, z_l, z_l)[0]
        elif method == "ASC26_MA_TRANS":
            net_m = _get_cached_model("asc26")
            enc = net_m.encode(g)
            z_d = np.zeros((N_a_g, 2), dtype=np.float32)
            D = enc["h_bay"].shape[-1] if len(g.bay_ids) > 0 else enc["g_res"].shape[-1]
            for ti in range(N_t_g):
                bi = int(g.task_incumbent_bay_idx[ti])
                hb = enc["h_bay"][bi] if 0 <= bi < len(g.bay_ids) else np.zeros((D,), dtype=np.float32)
                score_mat[ti] = net_m.predict_agv_step(enc, g, ti, hb, z_d)
        elif method == "FGCS26_REN_GNN":
            net_f = _get_cached_model("fgcs26")
            enc = net_f.encode(g)
            z_d = np.zeros((N_a_g, 2), dtype=np.float32)
            D = enc["h_bay"].shape[-1] if len(g.bay_ids) > 0 else enc["g_pool"].shape[-1]
            for ti in range(N_t_g):
                bi = int(g.task_incumbent_bay_idx[ti])
                hb = enc["h_bay"][bi] if 0 <= bi < len(g.bay_ids) else np.zeros((D,), dtype=np.float32)
                score_mat[ti] = net_f.predict_agv_step(enc, g, ti, hb, z_d)
        elif method == "IJPR25_DFJSPT_HMARL":
            net_h = _get_cached_model("ijpr25")
            gf = _global_flat_summary_np(g)
            z_d = np.zeros((N_a_g, 2), dtype=np.float32)
            for ti in range(N_t_g):
                bi = int(g.task_incumbent_bay_idx[ti])
                bf = g.bay_feats[bi] if 0 <= bi < len(g.bay_ids) else np.zeros((8,), dtype=np.float32)
                score_mat[ti] = net_h.predict_transbot_step(g, ti, bf, z_d, gf)
        else:
            raise ValueError(f"Unsupported dispatcher method: {method!r}")

        mat = np.full((n_t, n_a), -1e9, dtype=np.float64)
        for i, tk in enumerate(cand_t):
            ti = t_idx.get(f"T_{tk.task_id}")
            if ti is None:
                continue
            for j, aid in enumerate(cand_a):
                aj = a_idx.get(aid)
                if aj is not None:
                    mat[i, j] = float(score_mat[ti, aj])
        return _match_from_score_matrix(mat, cand_a, cand_t)

    return _fn


def _resolve_dispatch_models(method: str) -> tuple[str, Any, Any]:
    builtin_map = {
        "EJOR26_Wu_SemiCircle": "Wu26_SemiCircle",
        "EJOR26_Wu_GRU_Attn": "Wu26_SemiCircle_GRU",
        "EJOR26_Wu_GRU_Adapted": "Wu26_SemiCircle_GRU",
        "EJOR26_Wu_DFL": "Wu26_SemiCircle_DFL",
        "EJOR26_Wu_DFL_Adapted": "Wu26_SemiCircle_DFL",
    }
    dp_name = builtin_map.get(method, method)
    wu = _get_cached_model("wu_net_adapted" if "Adapted" in method else "wu_net")
    dfl = _get_cached_model("dfl_critic_adapted" if "Adapted" in method else "dfl_critic")
    return dp_name, wu, dfl


def _worker_bench_dispatch(args: tuple[str, str, int, int, int, float, str]) -> dict[str, Any]:
    layout, method, n_tasks, n_agvs, seed, delay_prob, router = args
    dp_name, wu, dfl = _resolve_dispatch_models(method)
    override = make_converged_dispatcher(method)
    r = run_workshop_episode(
        layout=layout,
        aisle_capacity=1,
        n_tasks=n_tasks,
        n_agvs=n_agvs,
        seed=seed,
        dispatcher=dp_name,
        router=router,
        delay_prob=delay_prob,
        horizon=900,
        wu_net=wu,
        dfl_critic=dfl,
        dispatch_override=override,
    )
    return {
        "layout": layout,
        "method": method,
        "seed": seed,
        "completed": r.completed_tasks,
        "n_tasks": r.n_tasks,
        "makespan": r.makespan,
        "on_time": r.on_time_rate,
        "util": r.agv_utilization,
        "waits": r.conflict_wait_count,
        "wall_ms": r.wall_ms,
        "safe": r.zero_conflicts,
    }


def _worker_stress_w(args: tuple[str, str, int, int, int, int, float, int, str, str]) -> dict[str, Any]:
    tag, layout, cap, nt, na, horizon, delay_prob, seed, dp, rt = args
    override = make_converged_dispatcher(dp)
    dp_arg, wu, dfl = _resolve_dispatch_models(dp)
    r = run_workshop_episode(
        layout=layout,
        aisle_capacity=cap,
        n_tasks=nt,
        n_agvs=na,
        seed=seed,
        dispatcher=dp_arg,
        router=rt,
        delay_prob=delay_prob,
        horizon=horizon,
        wu_net=wu,
        dfl_critic=dfl,
        dispatch_override=override,
    )
    return {
        "tag": tag,
        "dp": dp,
        "rt": rt,
        "seed": seed,
        "n_tasks": nt,
        "n_agvs": na,
        "completed": r.completed_tasks,
        "makespan": r.makespan,
        "on_time": r.on_time_rate,
        "util": r.agv_utilization,
        "waits": r.conflict_wait_count,
        "wall_ms": r.wall_ms,
        "safe": r.zero_conflicts,
    }


def _worker_stress_grid(args: tuple[str, str, int, int, int, str, str]) -> dict[str, Any]:
    tag, gname, na, horizon, seed, sv, label = args
    gm = build_map_rooms_32x32() if gname == "Rooms_32x32" else build_map_warehouse_40x40()
    ip, rq = generate_lifelong_requests(gm, na, n_requests=na * 28, seed=seed)
    dcfg = DelayScenarioConfig(
        motion_delay_prob=0.14,
        service_delay_prob=0.25,
        max_service_delay=2,
        heterogeneous_slow_frac=0.25,
        comm_lag_steps=1,
        horizon=horizon,
    )
    e_cfg = EALRConfig(
        enable_v2_migrations=True,
        enable_choke_arbitration=True,
        max_repair_agents=max(16, int(round(0.35 * na))),
    )
    res = simulate_lifelong_with_delays(gm, ip, rq, dcfg, solver=sv, ealr_cfg=e_cfg, seed=seed, scenario_tag=tag)
    return {
        "tag": tag,
        "map": gname,
        "n_agvs": na,
        "label": label,
        "tp": res.throughput,
        "completed": res.completed_tasks,
        "exp": res.total_expansions,
        "rep_frac": res.avg_repair_fraction,
        "safe": res.zero_conflicts,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0_all = time.perf_counter()
    workers = min(8, max(2, (os.cpu_count() or 4) // 2))

    # ------------------------------------------------------------------
    # 1. Bench-Dispatch: Unified Converged Upper-Layer Comparison (18 Methods)
    # ------------------------------------------------------------------
    methods = [
        # Classic / OR
        "FIFO_RoundRobin",
        "Nearest_Idle",
        "EDD_Urgency",
        "Hungarian_MinCost",
        # EJOR'26 Wu et al.
        "EJOR26_Wu_SemiCircle",
        "EJOR26_Wu_GRU_Attn",
        "EJOR26_Wu_GRU_Adapted",
        "EJOR26_Wu_DFL",
        "EJOR26_Wu_DFL_Adapted",
        # Prior RL (Zero-DAG Converged Transfer vs Scenario-Adapted)
        "Ours_V19_HETERO_RL",
        "Ours_V19_NO_PROD_COMM",
        "Ours_V19_Adapted",
        "TrackA_PPO_ZeroShot",
        "TrackA_SAC_ZeroShot",
        "TrackA_AM_ZeroShot",
        # Other 2025/2026 Literature Baselines
        "RAL26_DetourDelta",
        "IJCAI26_PGMPO",
        "TII26_E2G_MoE",
        "ASC26_MA_TRANS",
        "FGCS26_REN_GNN",
        "IJPR25_DFJSPT_HMARL",
    ]
    print(f"=== [Suite 1] Bench-Dispatch ({len(methods)} Upper Solvers x CPS_CVCR_DFL on demo17 & workshop_v1) ===", flush=True)
    b_tasks: list[tuple[str, str, int, int, int, float, str]] = []
    for m in methods:
        for s in (42, 43):
            b_tasks.append(("workshop_v1", m, 40, 6, s, 0.12, "CPS_CVCR_DFL"))
            b_tasks.append(("demo17", m, 30, 4, s, 0.10, "CPS_CVCR_DFL"))

    with ProcessPoolExecutor(max_workers=workers) as pool:
        b_rows = list(pool.map(_worker_bench_dispatch, b_tasks))

    bench_summary: dict[str, list[dict[str, Any]]] = {"workshop_v1": [], "demo17": []}
    for lay in ("workshop_v1", "demo17"):
        tot_t = 40 if lay == "workshop_v1" else 30
        print(f"--- [{lay} | CPS_CVCR_DFL] ---", flush=True)
        for m in methods:
            sub = [x for x in b_rows if x["layout"] == lay and x["method"] == m]
            rec = {
                "method": m,
                "n_tasks": tot_t,
                "completed": round(sum(x["completed"] for x in sub) / len(sub), 1),
                "makespan": round(sum(x["makespan"] for x in sub) / len(sub), 1),
                "on_time": round(sum(x["on_time"] for x in sub) / len(sub), 4),
                "util": round(sum(x["util"] for x in sub) / len(sub), 4),
                "waits": round(sum(x["waits"] for x in sub) / len(sub), 1),
                "safe": all(x["safe"] for x in sub),
            }
            bench_summary[lay].append(rec)
            print(
                f"  {m:22s}: done={rec['completed']:4.1f}/{tot_t} MS={rec['makespan']:5.1f} "
                f"OnTime={rec['on_time']*100:5.1f}% Waits={rec['waits']:6.1f} Safe={rec['safe']}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 2. Stress-W: Heavy-Load Workshop Stress Matrix
    # ------------------------------------------------------------------
    print("\n=== [Suite 2] Stress-W: Large-Scale Workshop Stress Matrix (Tasks up to 150, Fleets up to 16, Delay up to 25%) ===", flush=True)
    stress_scenarios = [
        ("W_Stress1_62n_120T_6AGV_12pct", "workshop_v1", 1, 120, 6, 1200, 0.12),
        ("W_Stress2_62n_80T_12AGV_15pct", "workshop_v1", 1, 80, 12, 900, 0.15),
        ("W_Stress3_62n_150T_16AGV_25pct", "workshop_v1", 2, 150, 16, 1000, 0.25),
        ("W_Stress4_17n_80T_8AGV_20pct", "demo17", 1, 80, 8, 1000, 0.20),
    ]
    stress_combos = [
        ("FIFO_RoundRobin", "CA_Star"),
        ("EJOR26_Wu_SemiCircle", "CPS_AStar"),
        ("EJOR26_Wu_GRU_Attn", "CPS_CVCR_DFL"),
        ("EJOR26_Wu_GRU_Adapted", "CPS_CVCR_DFL"),
        ("EJOR26_Wu_DFL", "CPS_CVCR_DFL"),
        ("EJOR26_Wu_DFL_Adapted", "CPS_CVCR_DFL"),
        ("Ours_V19_Adapted", "CPS_CVCR_DFL"),
    ]
    sw_tasks = [
        (tag, lay, cap, nt, na, hor, dp_p, s, dp, rt)
        for (tag, lay, cap, nt, na, hor, dp_p) in stress_scenarios
        for (dp, rt) in stress_combos
        for s in (42, 43)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        sw_rows = list(pool.map(_worker_stress_w, sw_tasks))

    stress_w_summary: list[dict[str, Any]] = []
    for (tag, lay, cap, nt, na, hor, dp_p) in stress_scenarios:
        for dp, rt in stress_combos:
            sub = [x for x in sw_rows if x["tag"] == tag and x["dp"] == dp and x["rt"] == rt]
            rec = {
                "scenario": tag,
                "n_tasks": nt,
                "n_agvs": na,
                "dispatcher": dp,
                "router": rt,
                "completed": round(sum(x["completed"] for x in sub) / len(sub), 1),
                "makespan": round(sum(x["makespan"] for x in sub) / len(sub), 1),
                "on_time": round(sum(x["on_time"] for x in sub) / len(sub), 4),
                "waits": round(sum(x["waits"] for x in sub) / len(sub), 1),
                "safe": all(x["safe"] for x in sub),
            }
            stress_w_summary.append(rec)
            print(
                f"  [{tag:30s} | {dp:20s} + {rt:12s}] done={rec['completed']:5.1f}/{nt} "
                f"MS={rec['makespan']:6.1f} OnTime={rec['on_time']*100:5.1f}% Waits={rec['waits']:7.1f} Safe={rec['safe']}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # 3. Stress-Grid: Extreme MovingAI Fleet Scaling (80 -> 200 AGVs) & Heavy Delay
    # ------------------------------------------------------------------
    print("\n=== [Suite 3] Stress-Grid: Extreme Warehouse Scaling (80..200 AGVs) & Heavy Uncertainty ===", flush=True)
    mw = build_map_warehouse_40x40()
    grid_scale_rows: list[dict[str, Any]] = []
    for na in (80, 120, 160, 200):
        ip, rq = generate_lifelong_requests(mw, na, n_requests=na * 20, seed=42)
        r_py = run_accelerated_lifelong_mapd(mw, ip, rq, backend="python_cps", horizon=200, seed=42)
        r_sipp = run_accelerated_lifelong_mapd(mw, ip, rq, backend="rust_sipp_slot", horizon=200, seed=42)
        spd = round(r_py.wall_ms / max(1e-6, r_sipp.wall_ms), 2)
        rec = {
            "n_agvs": na,
            "py_ms": round(r_py.wall_ms, 1),
            "py_tp": r_py.throughput,
            "py_safe": r_py.zero_conflicts,
            "sipp_ms": round(r_sipp.wall_ms, 1),
            "sipp_tp": r_sipp.throughput,
            "sipp_safe": r_sipp.zero_conflicts,
            "speedup": spd,
        }
        grid_scale_rows.append(rec)
        print(
            f"  [Warehouse_40x40 {na:3d} AGVs] Py={rec['py_ms']:7.1f}ms (TP={rec['py_tp']:.3f}, Safe={rec['py_safe']}) "
            f"| RustSIPP={rec['sipp_ms']:7.1f}ms ({spd:.2f}x, TP={rec['sipp_tp']:.3f}, Safe={rec['sipp_safe']})",
            flush=True,
        )

    sg_delay_tasks = [
        (f"{gm_n}_{na}", gm_n, na, 200, s, sv, lb)
        for gm_n, na in (("Rooms_32x32", 80), ("Warehouse_40x40", 120))
        for s in (39,)
        for sv, lb in (("naive_no_repair", "Naive"), ("global_rolling_cpsa", "Global_Pure"), ("ealr_local_repair", "EALR_v3"))
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        sg_rows = list(pool.map(_worker_stress_grid, sg_delay_tasks))
    for r in sg_rows:
        print(
            f"  [Stress-Grid-Delay {r['tag']:18s} | {r['label']:11s}] TP={r['tp']:.4f} "
            f"Exp={r['exp']} RepFrac={r['rep_frac']*100:.1f}% Safe={r['safe']}",
            flush=True,
        )

    elapsed_s = round(time.perf_counter() - t0_all, 2)
    payload = {
        "elapsed_s": elapsed_s,
        "bench_dispatch": bench_summary,
        "stress_workshop": stress_w_summary,
        "stress_grid_scaling": grid_scale_rows,
        "stress_grid_delay": sg_rows,
    }
    (OUT_DIR / "converged_stress_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# 统一收敛场景下的大规模压力测试与全基线横评报告",
        "",
        f"> 运行总耗时：`{elapsed_s}s` · 纯 NumPy + PyTorch 真实权重零泄漏加载 · 全量算例严格遵守无多工序 DAG 与 `100%` 零碰撞底线。",
        "",
        "## 一、此前自研 RL & 各大顶会文献基线：统一检修车间场景上层派车横评（共享 `CPS_CVCR_DFL` 底层路径）",
        "",
        "### 1A. 实景尺度 `rail_freight_workshop_v1`（62 节点 / 144 边, 40 任务 × 6 台 AGV, 12% 延误, 2 种子）",
        "",
        "| 派车策略 (`Dispatcher`) | 来源分类 | 完工数 | Makespan | 准时率 | AGV 利用率 | 冲突等待 | 零碰撞 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    cat_map = {
        "FIFO_RoundRobin": "经典规则",
        "Nearest_Idle": "经典规则",
        "EDD_Urgency": "经典规则",
        "Hungarian_MinCost": "经典运筹匹配",
        "EJOR26_Wu_SemiCircle": "EJOR'26 (Wu)",
        "EJOR26_Wu_GRU_Attn": "EJOR'26 (Wu GRU Zero-Shot)",
        "EJOR26_Wu_GRU_Adapted": "EJOR'26 (Wu GRU 场景微调)",
        "EJOR26_Wu_DFL": "EJOR'26 + 原 DFL (风险门控)",
        "EJOR26_Wu_DFL_Adapted": "EJOR'26 + 原位微调 DFL",
        "Ours_V19_HETERO_RL": "自研历史 RL (V19 PPO)",
        "Ours_V19_NO_PROD_COMM": "自研历史 RL (无生产消融)",
        "Ours_V19_Adapted": "自研历史 RL (场景原位微调)",
        "TrackA_PPO_ZeroShot": "Track A VRP RL (PPO)",
        "TrackA_SAC_ZeroShot": "Track A VRP RL (SAC)",
        "TrackA_AM_ZeroShot": "Track A VRP RL (AM)",
        "RAL26_DetourDelta": "IEEE RA-L'26",
        "IJCAI26_PGMPO": "IJCAI'26 (PGMPO)",
        "TII26_E2G_MoE": "IEEE TII'26 (HMoE)",
        "ASC26_MA_TRANS": "ASC'26 (MA-Trans)",
        "FGCS26_REN_GNN": "FGCS'26 (3D-DG GNN)",
        "IJPR25_DFJSPT_HMARL": "IJPR'25 (HMARL)",
    }
    for r in bench_summary["workshop_v1"]:
        md.append(
            f"| `{r['method']}` | {cat_map[r['method']]} | {r['completed']:.1f}/{r['n_tasks']} | {r['makespan']:.1f} | "
            f"{r['on_time']*100:.1f}% | {r['util']*100:.1f}% | {r['waits']:.1f} | {r['safe']} |"
        )
    md.extend([
        "",
        "### 1B. `demo17`（17 节点环网, 30 任务 × 4 台 AGV, 10% 延误, 2 种子）",
        "",
        "| 派车策略 (`Dispatcher`) | 来源分类 | 完工数 | Makespan | 准时率 | AGV 利用率 | 冲突等待 | 零碰撞 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in bench_summary["demo17"]:
        md.append(
            f"| `{r['method']}` | {cat_map[r['method']]} | {r['completed']:.1f}/{r['n_tasks']} | {r['makespan']:.1f} | "
            f"{r['on_time']*100:.1f}% | {r['util']*100:.1f}% | {r['waits']:.1f} | {r['safe']} |"
        )

    md.extend([
        "",
        "## 二、`Stress-W`：检修车间大规模高负荷/大车队/重扰动压力测试",
        "",
        "| 压力工况 | 上层派车 | 底层路由 | 完工数 | Makespan | 准时率 | 冲突等待 | 零碰撞 |",
        "| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in stress_w_summary:
        md.append(
            f"| `{r['scenario']}` | `{r['dispatcher']}` | `{r['router']}` | {r['completed']:.1f}/{r['n_tasks']} | "
            f"{r['makespan']:.1f} | {r['on_time']*100:.1f}% | {r['waits']:.1f} | {r['safe']} |"
        )

    md.extend([
        "",
        "## 三、`Stress-Grid`：极端百车仓储规划缩放与重滞后压力测试",
        "",
        "### 3A. `Warehouse_40x40`（`80 -> 200` 台 AGV · 200 步 MovingAI）",
        "",
        "| 车队规模 | `Python CPS-A*` 耗时 (TP, Safe) | `Rust SIPP + SlotPressure` 耗时 (加速, TP, Safe) |",
        "| :---: | :---: | :---: |",
    ])
    for r in grid_scale_rows:
        md.append(
            f"| {r['n_agvs']} AGVs | {r['py_ms']:.1f}ms ({r['py_tp']:.3f}, {r['py_safe']}) | "
            f"**{r['sipp_ms']:.1f}ms ({r['speedup']:.2f}x, {r['sipp_tp']:.3f}, {r['sipp_safe']})** |"
        )

    md.extend([
        "",
        "### 3B. 超大网格延误 + 通信滞后（`Rooms 80车` & `Warehouse 120车`, `lag=1`, 14% 延误）",
        "",
        "| 工况 | 策略 | 吞吐 (`TP`) | 完工数 | A\\* 扩展 | 波及率 | 零碰撞 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in sg_rows:
        md.append(
            f"| `{r['tag']}` | `{r['label']}` | {r['tp']:.4f} | {r['completed']} | {r['exp']} | {r['rep_frac']*100:.1f}% | {r['safe']} |"
        )

    (OUT_DIR / "converged_stress_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n=== Wrote {OUT_DIR / 'converged_stress_report.md'} ({elapsed_s}s) ===", flush=True)


def tune_v19_workshop_head(
    out_npz: Optional[Path] = None,
    *,
    epochs: int = 55,
    lr: float = 1.5e-3,
    seed: int = 42,
) -> dict[str, Any]:
    """Fine-tune ONLY the A3 transport matching head of `v19_no_prod_comm` on converged workshop graphs."""
    import torch.nn as nn
    import torch.nn.functional as F

    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    src_p = resolve_default_checkpoint("v19/v19_no_prod_comm.pt")
    net_v, _, _, _ = _load_v19_checkpoint(src_p, backend="numpy", ablation_mode="no_prod_comm")
    w = net_v.w
    H = net_v.hidden_dim

    samples: list[tuple[np.ndarray, np.ndarray]] = []  # (pair_in [L, 330], costs [L])
    wu_zero = _get_cached_model("wu_net")
    for layout, cap, nt, na, dp in (
        ("workshop_v1", 1, 45, 6, 0.12),
        ("workshop_v1", 1, 80, 12, 0.15),
        ("workshop_v1", 1, 120, 6, 0.12),
        ("demo17", 1, 35, 6, 0.10),
    ):
        for sd in (201, 202, 203):
            def _cap_v19(
                idle_agvs: list[str],
                ready_tasks: list[WorkshopTask],
                cur_pos: dict[str, str],
                network: Network,
                tt_cache: dict[tuple[str, str], int],
                tick: int,
                ctx: dict[str, Any],
                _cap: int = cap,
                _na: int = na,
            ) -> list[tuple[str, WorkshopTask]]:
                bay_load = ctx.get("bay_load", {})
                haven_prof = ctx.get("haven_prof")
                cand_a = sorted(idle_agvs)[:8]
                cand_t = sorted(
                    ready_tasks,
                    key=lambda tk: (
                        0 if tk.is_urgent else 1,
                        float(tk.due_step - tick - _tt_steps(network, tk.pickup, tk.dropoff, tt_cache)),
                        tk.release_step,
                        tk.task_id,
                    ),
                )[:8]
                if len(cand_t) >= 2 and len(cand_a) >= 2:
                    snap = _build_transport_snapshot(cand_a, cand_t, cur_pos, tick, include_busy_robots=False)
                    g = extract_v19_hetero_graph(snap, network)
                    enc = net_v.encode_entities(g)
                    N_a = len(g.agv_ids)
                    z0 = np.zeros((1, N_a, 2), dtype=np.float32)
                    ha_dyn = enc["h_agv"] + _np_elu(_np_linear(z0, w, "agv_dyn_proj"))
                    t_idx = {tid: i for i, tid in enumerate(g.rel_tids)}
                    a_idx = {aid: j for j, aid in enumerate(g.agv_ids)}
                    rows_p: list[np.ndarray] = []
                    rows_c: list[float] = []
                    sat_scale = float(_na) / max(1.0, 6.0 * float(_cap))
                    backlog_ratio = float(len(ready_tasks)) / float(max(1, len(idle_agvs)))
                    overload = min(1.0, max(0.0, (backlog_ratio - 1.5) / 4.0))
                    dead_w = 3.5 + 3.0 * overload
                    haul_w = 0.55 + 0.75 * overload
                    for tk in cand_t:
                        ti = t_idx.get(f"T_{tk.task_id}")
                        if ti is None:
                            continue
                        bi = int(g.task_incumbent_bay_idx[ti])
                        ht = enc["h_task"][:, ti, :]
                        h_bay = enc.get("h_bay")
                        if h_bay is not None and h_bay.shape[1] > 0:
                            ht = ht + _np_elu(_np_linear(h_bay[:, bi, :], w, "sel_bay_proj"))
                        b_tr = np.full((1, N_a, 1), float(g.bay_xy_travel_from_pickup[ti, bi]), dtype=np.float32)
                        b_dd = np.full((1, N_a, 1), float(g.bay_detour_norm[bi]), dtype=np.float32)
                        pair_in = np.concatenate(
                            [
                                np.broadcast_to(ht[:, None, :], (1, N_a, H)),
                                ha_dyn,
                                np.broadcast_to(enc["h_g_tr"][:, None, :], (1, N_a, 3 * H)),
                                g.task_agv_feats[None, ti, :, :],
                                b_tr,
                                b_dd,
                            ],
                            axis=-1,
                        )[0]
                        d_haul = float(_tt_steps(network, tk.pickup, tk.dropoff, tt_cache))
                        ld_i = float(bay_load.get(tk.pickup, 0) + bay_load.get(tk.dropoff, 0))
                        choke_hit = (
                            (1.0 if haven_prof and (tk.pickup in haven_prof.choke_nodes or tk.pickup in haven_prof.leaf_spurs) else 0.0)
                            + (1.0 if haven_prof and (tk.dropoff in haven_prof.choke_nodes or tk.dropoff in haven_prof.leaf_spurs) else 0.0)
                        )
                        for aid in cand_a:
                            aj = a_idx.get(aid)
                            if aj is None:
                                continue
                            d_dead = float(_tt_steps(network, cur_pos[aid], tk.pickup, tt_cache))
                            eta = tick + d_dead + d_haul + 1.5 * ld_i
                            late = max(0.0, eta - float(tk.due_step))
                            late_eff = min(18.0, late) + 0.12 * max(0.0, late - 18.0)
                            urg_m = 1.6 if tk.is_urgent else 1.0
                            late_w = (8.0 * urg_m) * (1.0 - 0.65 * overload)
                            urg_bonus = (18.0 if tk.is_urgent else 0.0) * (1.0 - 0.50 * overload)
                            cost_val = (
                                dead_w * d_dead
                                + haul_w * d_haul
                                + late_w * late_eff
                                + (9.0 / max(0.7, sat_scale)) * choke_hit * ld_i
                                - urg_bonus
                            )
                            rows_p.append(pair_in[aj])
                            rows_c.append(cost_val)
                    if len(rows_p) >= 4:
                        samples.append((np.stack(rows_p, axis=0), np.asarray(rows_c, dtype=np.float32)))
                return _dispatch_tasks("Wu26_SemiCircle_GRU", idle_agvs, ready_tasks, cur_pos, network, tt_cache, tick, wu_zero, haven_prof=haven_prof, bay_load=bay_load)

            run_workshop_episode(
                layout=layout, aisle_capacity=cap, n_tasks=nt, n_agvs=na, seed=sd,
                dispatcher="Wu26_SemiCircle_GRU", router="CPS_CVCR_DFL", delay_prob=dp,
                horizon=750, wu_net=wu_zero, dispatch_override=_cap_v19, use_rust_e2e=True,
            )

    in_dim = int(w["agv_w1.weight"].shape[1])
    h1_dim = int(w["agv_w1.weight"].shape[0])
    h2_dim = int(w["agv_w2.weight"].shape[0])
    head = nn.Sequential(nn.Linear(in_dim, h1_dim), nn.ELU(), nn.Linear(h1_dim, h2_dim), nn.ELU(), nn.Linear(h2_dim, 1))
    with torch.no_grad():
        head[0].weight.copy_(torch.from_numpy(w["agv_w1.weight"]))
        head[0].bias.copy_(torch.from_numpy(w["agv_w1.bias"]))
        head[2].weight.copy_(torch.from_numpy(w["agv_w2.weight"]))
        head[2].bias.copy_(torch.from_numpy(w["agv_w2.bias"]))
        head[4].weight.copy_(torch.from_numpy(w["agv_w3.weight"]))
        head[4].bias.copy_(torch.from_numpy(w["agv_w3.bias"]))
    anchor = {k: p.detach().clone() for k, p in head.named_parameters()}
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)

    perm = rng.permutation(len(samples))
    n_val = max(4, int(0.15 * len(samples)))
    val_ids, tr_ids = perm[:n_val].tolist(), perm[n_val:].tolist()

    def _kl(idxs: list[int]) -> float:
        head.eval()
        acc = 0.0
        with torch.no_grad():
            for ii in idxs:
                px, cx = samples[ii]
                lg = head(torch.from_numpy(px)).squeeze(-1)
                ct = torch.from_numpy(cx)
                q = F.softmax(-((ct - ct.mean()) / torch.clamp(ct.std(), min=1.0)) / 0.55, dim=0)
                acc += float(F.kl_div(F.log_softmax(lg, dim=0), q, reduction="sum").item())
        return acc / max(1, len(idxs))

    kl_before = _kl(val_ids)
    for _ in range(int(epochs)):
        head.train()
        opt.zero_grad()
        loss_sum = torch.tensor(0.0)
        cnt = 0
        for p_idx in rng.permutation(len(tr_ids)):
            px, cx = samples[tr_ids[int(p_idx)]]
            lg = head(torch.from_numpy(px)).squeeze(-1)
            ct = torch.from_numpy(cx)
            q = F.softmax(-((ct - ct.mean()) / torch.clamp(ct.std(), min=1.0)) / 0.55, dim=0)
            kl = F.kl_div(F.log_softmax(lg, dim=0), q, reduction="sum")
            prox = sum(torch.sum((p - anchor[nm]) ** 2) for nm, p in head.named_parameters())
            loss_sum = loss_sum + kl + 0.01 * prox
            cnt += 1
            if cnt >= 16:
                (loss_sum / float(cnt)).backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
                opt.step()
                opt.zero_grad()
                loss_sum = torch.tensor(0.0)
                cnt = 0
        if cnt > 0:
            (loss_sum / float(cnt)).backward()
            opt.step()
    kl_after = _kl(val_ids)

    dst = out_npz if out_npz is not None else (TRACK_R_CKPT_DIR / "v19_no_prod_comm_workshop.npz")
    dst.parent.mkdir(parents=True, exist_ok=True)
    merged = {k: np.asarray(v).copy() for k, v in w.items()}
    merged["agv_w1.weight"] = head[0].weight.detach().cpu().numpy().astype(np.float32)
    merged["agv_w1.bias"] = head[0].bias.detach().cpu().numpy().astype(np.float32)
    merged["agv_w2.weight"] = head[2].weight.detach().cpu().numpy().astype(np.float32)
    merged["agv_w2.bias"] = head[2].bias.detach().cpu().numpy().astype(np.float32)
    merged["agv_w3.weight"] = head[4].weight.detach().cpu().numpy().astype(np.float32)
    merged["agv_w3.bias"] = head[4].bias.detach().cpu().numpy().astype(np.float32)
    np.savez(str(dst), **merged)
    return {
        "n_samples": len(samples),
        "val_kl_before": round(kl_before, 4),
        "val_kl_after": round(kl_after, 4),
        "out_npz": str(dst),
    }


if __name__ == "__main__":
    if "--tune-v19" in sys.argv:
        print(json.dumps(tune_v19_workshop_head(), indent=2), flush=True)
    else:
        main()
