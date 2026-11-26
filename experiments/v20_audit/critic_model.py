"""Pure-NumPy 24-D Long-Horizon Progress & Commitment-Churn Value Critic (`V_phi`) for V20-A/B."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.evaluator import EvalResult
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import compute_candidate_ja_components
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph

CRITIC_FEAT_DIM = 24
CRITIC_FEAT_NAMES: tuple[str, ...] = (
    "d_ja_norm",
    "d_b2_norm",
    "d_cmax_norm",
    "d_ttard_norm",
    "d_otard_norm",
    "d_deliv_norm",
    "d_starv_norm",
    "d_wait_norm",
    "cand_wait_ratio",
    "d_overlap_norm",
    "cand_spread_norm",
    "d_spread_norm",
    "is_keep",
    "changed_assign_frac",
    "changed_bay_crew_frac",
    "q_head_change_frac",
    "q_tail_change_frac",
    "rescued_dist_frac",
    "split_kit_ops_frac",
    "d_kit_spread_norm",
    "d_win_cliff_frac",
    "d_late_ops_frac",
    "ws_flag",
    "abs_ja_log",
)


def _clip8(val: float) -> float:
    if not math.isfinite(val):
        return 0.0
    return float(max(-8.0, min(8.0, val)))


@dataclass
class KeepBaselineStats:
    """Cached per-snapshot KEEP reference statistics used to extract fast relative candidate features."""

    ja_unified: float
    b2_obj: float
    cmax_p: float
    task_tard_p: float
    op_tard_p: float
    deliv_p: float
    starv: float
    wait_steps: float
    node_overlap: float
    agv_spread: float
    max_kit_spread: float
    win_cliff_cnt: float
    late_ops_cnt: float
    keep_orders: dict[str, list[str]]


def _extract_path_metrics(res: EvalResult, tick_now: int) -> tuple[float, float, float]:
    """Compute `(wait_steps, node_time_overlap, agv_finish_spread)` from `EvalResult.robot_paths`."""
    waits = 0.0
    finish_times: list[float] = []
    occ: dict[tuple[str, int], int] = {}
    for rid, path in (res.robot_paths or {}).items():
        if not path:
            continue
        prev_n: Optional[str] = None
        prev_t: Optional[int] = None
        for nd, t_step in path:
            st = int(t_step)
            if prev_n is not None and nd == prev_n and prev_t is not None:
                waits += float(max(0, st - prev_t))
            key = (str(nd), st)
            occ[key] = occ.get(key, 0) + 1
            prev_n = str(nd)
            prev_t = st
        finish_times.append(float(path[-1][1]))
    overlap = float(sum(max(0, cnt - 1) for cnt in occ.values()))
    spread = float(max(finish_times) - min(finish_times)) if len(finish_times) >= 2 else 0.0
    if not finish_times:
        spread = 0.0
    return waits, overlap, spread


def _extract_op_window_metrics(snapshot: NativeSnapshot, res: EvalResult) -> tuple[float, float, float, float]:
    """Compute `(split_kit_ops_cnt, max_kit_spread, win_cliff_cnt, late_ops_cnt)`."""
    unloads = res.order_unload_ticks or {}
    op_starts = res.op_start or {}
    op_ends = res.op_end or {}
    split_cnt = 0.0
    max_spread = 0.0
    win_cliff = 0.0
    late_ops = 0.0
    tick = int(snapshot.tick)

    for op in snapshot.operations:
        if getattr(op, "completed", False):
            continue
        kt_times = [float(unloads.get(kt, tick + 40)) for kt in (op.kit_task_ids or ())]
        if len(kt_times) >= 2:
            sp = max(kt_times) - min(kt_times)
            max_spread = max(max_spread, sp)
            if sp > 20.0:
                split_cnt += 1.0
        st_op = int(op_starts.get(op.op_id, tick + 40))
        en_op = int(op_ends.get(op.op_id, st_op + int(op.proc_steps)))
        if en_op > int(op.due_step):
            late_ops += 1.0
        wins = getattr(op, "resource_windows", ()) or ()
        if wins:
            # Count op as cliff-risk if start misses or lands within 6 steps of window end
            in_good = False
            near_edge = False
            for w0, w1 in wins:
                if int(w0) <= st_op and (st_op + int(op.proc_steps)) <= int(w1):
                    in_good = True
                    if (int(w1) - (st_op + int(op.proc_steps))) <= 6:
                        near_edge = True
                    break
            if (not in_good) or near_edge:
                win_cliff += 1.0

    return split_cnt, max_spread, win_cliff, late_ops


def build_keep_baseline_stats(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    keep_eval: EvalResult,
    keep_ja_u: float,
    rob_cur_h: dict[str, str],
    sim_horizon: int,
) -> KeepBaselineStats:
    """Compute KEEP anchor stats once per snapshot."""
    comps = compute_candidate_ja_components(
        snapshot,
        graph,
        keep_eval,
        graph.base_assign,
        graph.base_bays,
        graph.base_crews,
        keep_eval.robot_order_seq or None,
        rob_cur_h,
        lambda_change=15.0,
        sim_horizon=sim_horizon,
    )
    waits, overlap, spread = _extract_path_metrics(keep_eval, int(snapshot.tick))
    _, max_ks, w_cliff, late_ops = _extract_op_window_metrics(snapshot, keep_eval)
    keep_orders = {k: list(v) for k, v in (keep_eval.robot_order_seq or {}).items() if v}
    cmax_fallback = float(max((keep_eval.op_end or {}).values(), default=int(snapshot.tick)))
    return KeepBaselineStats(
        ja_unified=float(keep_ja_u),
        b2_obj=float(keep_eval.objective),
        cmax_p=float(comps.get("cmax", cmax_fallback)),
        task_tard_p=float(comps.get("t_task_cost", 0.0)),
        op_tard_p=float(comps.get("t_op_cost", 0.0)),
        deliv_p=float(comps.get("deliv_cost", 0.0)),
        starv=float(keep_eval.starvation_ticks),
        wait_steps=waits,
        node_overlap=overlap,
        agv_spread=spread,
        max_kit_spread=max_ks,
        win_cliff_cnt=w_cliff,
        late_ops_cnt=late_ops,
        keep_orders=keep_orders,
    )


def extract_critic_features(
    snapshot: NativeSnapshot,
    network: Network,
    graph: V19HeteroGraph,
    ev_res: EvalResult,
    ja_unified_score: float,
    assign: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    orders: Optional[dict[str, list[str]]],
    keep_stats: KeepBaselineStats,
    rob_cur_h: dict[str, str],
    sim_horizon: int,
    *,
    is_keep_cand: bool = False,
) -> np.ndarray:
    """Build exact 24-D float32 feature vector comparing candidate vs snapshot KEEP anchor."""
    comps = compute_candidate_ja_components(
        snapshot,
        graph,
        ev_res,
        assign,
        bays,
        crews,
        orders,
        rob_cur_h,
        lambda_change=15.0,
        sim_horizon=sim_horizon,
    )
    cmax_fallback = float(max((ev_res.op_end or {}).values(), default=int(snapshot.tick)))
    cmax_p = float(comps.get("cmax", cmax_fallback))
    ttard_p = float(comps.get("t_task_cost", 0.0))
    otard_p = float(comps.get("t_op_cost", 0.0))
    deliv_p = float(comps.get("deliv_cost", 0.0))

    waits, overlap, spread = _extract_path_metrics(ev_res, int(snapshot.tick))
    split_ops, max_ks, w_cliff, late_ops = _extract_op_window_metrics(snapshot, ev_res)

    j_scale = max(100.0, abs(keep_stats.ja_unified))
    b2_scale = max(100.0, abs(keep_stats.b2_obj))
    n_agvs = max(1, len(graph.agv_ids))
    n_healthy = max(1, len(graph.healthy_agv_indices))
    n_rel = max(1, len(graph.rel_tids))
    n_ops = max(1, len(graph.op_ids))
    unstarted_ops = max(1, sum(1 for op in snapshot.operations if not getattr(op, "completed", False)))
    win_ops = max(1, sum(1 for op in snapshot.operations if getattr(op, "resource_windows", ())))

    changed_assign = sum(
        1 for tid in graph.rel_tids if graph.base_assign.get(tid) and assign.get(tid) != graph.base_assign.get(tid)
    )
    changed_bays = sum(1 for oid in graph.op_ids if bays.get(oid, graph.base_bays.get(oid)) != graph.base_bays.get(oid))
    changed_crews = sum(
        1 for oid in graph.op_ids if int(crews.get(oid, graph.base_crews.get(oid, 0))) != int(graph.base_crews.get(oid, 0))
    )

    eff_orders = orders if orders is not None else (ev_res.robot_order_seq or {})
    q_head_chg = 0
    q_tail_chg = 0
    for aid in keep_stats.keep_orders.keys() | eff_orders.keys():
        old_q = keep_stats.keep_orders.get(aid, [])
        new_q = list(eff_orders.get(aid, []))
        if old_q != new_q:
            if old_q[:1] != new_q[:1]:
                q_head_chg += 1
            else:
                q_tail_chg += 1

    dist_tids = [
        t.task_id
        for t in snapshot.tasks
        if t.status.lower() in ("pending", "assigned")
        and t.assigned_agv is not None
        and t.assigned_agv in graph.clf.disturbed_robots
    ]
    rescued_dist = sum(1 for tid in dist_tids if assign.get(tid) and assign.get(tid) not in graph.clf.disturbed_robots)

    feats = np.array(
        [
            _clip8((float(ja_unified_score) - keep_stats.ja_unified) / j_scale),
            _clip8((float(ev_res.objective) - keep_stats.b2_obj) / b2_scale),
            _clip8((cmax_p - keep_stats.cmax_p) / max(20.0, keep_stats.cmax_p)),
            _clip8((ttard_p - keep_stats.task_tard_p) / j_scale),
            _clip8((otard_p - keep_stats.op_tard_p) / j_scale),
            _clip8((deliv_p - keep_stats.deliv_p) / max(50.0, keep_stats.deliv_p)),
            _clip8((float(ev_res.starvation_ticks) - keep_stats.starv) / max(50.0, keep_stats.starv + 50.0)),
            _clip8((waits - keep_stats.wait_steps) / max(5.0, float(n_agvs) * 4.0)),
            _clip8(waits / max(10.0, max(1.0, cmax_p) * float(n_agvs))),
            _clip8((overlap - keep_stats.node_overlap) / max(2.0, float(n_agvs))),
            _clip8(spread / max(20.0, cmax_p)),
            _clip8((spread - keep_stats.agv_spread) / max(20.0, cmax_p)),
            1.0 if is_keep_cand else 0.0,
            _clip8(float(changed_assign) / float(n_rel)),
            _clip8(float(changed_bays + changed_crews) / float(n_ops)),
            _clip8(float(q_head_chg) / float(n_healthy)),
            _clip8(float(q_tail_chg) / float(n_healthy)),
            _clip8(float(rescued_dist) / float(max(1, len(dist_tids)))),
            _clip8(float(split_ops) / float(unstarted_ops)),
            _clip8((max_ks - keep_stats.max_kit_spread) / 30.0),
            _clip8((w_cliff - keep_stats.win_cliff_cnt) / float(win_ops)),
            _clip8((late_ops - keep_stats.late_ops_cnt) / float(unstarted_ops)),
            1.0 if len(network.nodes) >= 30 else 0.0,
            _clip8(math.log1p(max(0.0, float(ja_unified_score))) / 10.0),
        ],
        dtype=np.float32,
    )
    return feats


class NumpyValueCritic:
    """Fast pure-NumPy regime-conditioned Expected-Regret Value Critic loaded from `.npz` weights."""

    def __init__(self, weights_path: Path) -> None:
        data = np.load(str(weights_path), allow_pickle=False)
        self.scale = float(data["scale"][0])
        self.scale_ws = float(data["scale_ws"][0]) if "scale_ws" in data.files else self.scale
        self.mean_4t = np.asarray(data["mean_4t"], dtype=np.float32)
        self.std_4t = np.asarray(data["std_4t"], dtype=np.float32)
        self.w_4t = np.asarray(data["w_4t"], dtype=np.float32)
        self.b_4t = float(data["b_4t"][0])
        self.jmix_4t = float(data["jmix_4t"][0])

        self.mean_ws = np.asarray(data["mean_ws"], dtype=np.float32)
        self.std_ws = np.asarray(data["std_ws"], dtype=np.float32)
        self.w_ws = np.asarray(data["w_ws"], dtype=np.float32)
        self.b_ws = float(data["b_ws"][0])
        self.jmix_ws = float(data["jmix_ws"][0])

    def predict_delta_keep(self, feat_vec: np.ndarray) -> float:
        """Predict relative full-episode delta-cost vs snapshot KEEP (lower is strictly better)."""
        fv = np.asarray(feat_vec, dtype=np.float32).copy()
        fv[12] = 0.0
        fv[23] = 0.0
        is_ws = float(fv[22]) >= 0.5
        if is_ws:
            x = (fv - self.mean_ws) / self.std_ws
            raw = float(x @ self.w_ws + self.b_ws) + self.jmix_ws * float(fv[0])
            return float(raw * self.scale_ws)
        x = (fv - self.mean_4t) / self.std_4t
        raw = float(x @ self.w_4t + self.b_4t) + self.jmix_4t * float(fv[0])
        return float(raw * self.scale)
