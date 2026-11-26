"""Production CoRe-Sched V20.0 (`V20_NAJS`) and V20.1 (`V20_1_NAJS`) Solvers.

Self-contained pure-NumPy + Rust-evaluator production implementation (zero runtime dependency
on `experiments/` or PyTorch during inference):
  - `RobustRankAsinhCritic`: 24-D long-horizon expected-regret value critic (`checkpoints/v20/v20_critic_weights.npz`).
  - `V20BidirectionalBackbone`: Deterministic V18 <-> V19 bidirectional joint search with
    shared Layer-1 entity encoding, commitment-aware causal cones, and `seq_diff_b2`.
  - `V20NAJSProposer`: Frozen CoRe-Sched V20.0 (`V20BidirectionalBackbone` + PDG 1.0 escape).
  - `V201NAJSProposer`: Minimal effective CoRe-Sched V20.1 (`V20BidirectionalBackbone` +
    PDG 2.0 `E1+E2` commitment-preserving escape + non-destructive `PersistentEscapeMemory`).
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.pdg_liveness import (
    PersistentEscapeMemory,
    build_critical_bottleneck_locks,
    plan_idle_blocker_evacuations,
)
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy, _LazyKittingSurrogate
from amr_sched.cvcr.snapshot import NativeSnapshot
from amr_sched.cvcr.v12_proactive_guard import (
    compute_detour_delta_signals,
    plan_haven_safe_corridor_repair,
)
from amr_sched.cvcr.v18_frontier_synthesis import (
    WindowAwareKittingSurrogate,
    _surrogate_polish_assign,
    compute_semicircle_kitting_bays,
    load_v18_synthesis_policy,
)
from amr_sched.network import Network
from amr_sched.rl.causal_repair_cone import (
    FastJAWKittingSurrogate,
    _compute_fresh_semicircle_bays,
    _regret2_insert_tasks,
    build_causal_repair_cone,
    compute_candidate_ja_components,
    generate_causal_cone_candidates,
    observable_op_closure,
    project_zero_churn_candidate,
    score_candidate_ja_unified,
)
from amr_sched.rl.v19_hetero_constructor import (
    DecodedJointCandidate,
    NumpyV19HeteroConstructorNet,
    V19AdaptiveCouplingProposer,
    _build_core_sched_proposal,
)
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph, extract_v19_hetero_graph

_DEFAULT_V19_RL_CKPT = resolve_default_checkpoint("v19/v19_hetero_rl.pt")
_DEFAULT_V19_STAGED_CKPT = resolve_default_checkpoint("v19/v19_staged_rl.pt")
_DEFAULT_V17_CKPT = resolve_default_checkpoint("v17_window_protection.npz")
_DEFAULT_V20_CRITIC_CKPT = resolve_default_checkpoint("v20/v20_critic_weights.npz")


def _clip8(val: float) -> float:
    if not math.isfinite(val):
        return 0.0
    return float(max(-8.0, min(8.0, val)))


@dataclass
class KeepBaselineStats:
    """Cached per-snapshot KEEP anchor statistics for 24-D critic feature extraction."""

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


def _extract_path_metrics(res: EvalResult) -> tuple[float, float, float]:
    waits = 0.0
    finish_times: list[float] = []
    occ: dict[tuple[str, int], int] = {}
    for _, path in (res.robot_paths or {}).items():
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
    return waits, overlap, spread


def _extract_op_window_metrics(snapshot: NativeSnapshot, res: EvalResult) -> tuple[float, float, float, float]:
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
    comps = compute_candidate_ja_components(
        snapshot, graph, keep_eval, graph.base_assign, graph.base_bays, graph.base_crews,
        keep_eval.robot_order_seq or None, rob_cur_h, lambda_change=15.0, sim_horizon=sim_horizon,
    )
    waits, overlap, spread = _extract_path_metrics(keep_eval)
    _, max_ks, w_cliff, late_ops = _extract_op_window_metrics(snapshot, keep_eval)
    keep_orders = {k: list(v) for k, v in (keep_eval.robot_order_seq or {}).items() if v}
    cmax_fb = float(max((keep_eval.op_end or {}).values(), default=int(snapshot.tick)))
    return KeepBaselineStats(
        ja_unified=float(keep_ja_u),
        b2_obj=float(keep_eval.objective),
        cmax_p=float(comps.get("cmax", cmax_fb)),
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
    comps = compute_candidate_ja_components(
        snapshot, graph, ev_res, assign, bays, crews, orders, rob_cur_h, lambda_change=15.0, sim_horizon=sim_horizon,
    )
    cmax_fb = float(max((ev_res.op_end or {}).values(), default=int(snapshot.tick)))
    cmax_p = float(comps.get("cmax", cmax_fb))
    ttard_p = float(comps.get("t_task_cost", 0.0))
    otard_p = float(comps.get("t_op_cost", 0.0))
    deliv_p = float(comps.get("deliv_cost", 0.0))

    waits, overlap, spread = _extract_path_metrics(ev_res)
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

    return np.array(
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


class RobustRankAsinhCritic:
    """Fast pure-NumPy regime-conditioned Expected-Regret Value Critic loaded from `.npz` weights."""

    def __init__(self, weights_path: Path) -> None:
        self.weights_path = Path(weights_path)
        data = np.load(str(self.weights_path), allow_pickle=False)
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
        fv = np.asarray(feat_vec, dtype=np.float32).copy()
        fv[12] = 0.0
        fv[23] = 0.0
        if float(fv[22]) >= 0.5:
            x = (fv - self.mean_ws) / self.std_ws
            raw = float(x @ self.w_ws + self.b_ws) + self.jmix_ws * float(fv[0])
            return float(raw * self.scale_ws)
        x = (fv - self.mean_4t) / self.std_4t
        raw = float(x @ self.w_4t + self.b_4t) + self.jmix_4t * float(fv[0])
        return float(raw * self.scale)


def candidate_action_signature(
    assign: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    op_prio: dict[str, int],
    task_orders: Optional[dict[str, list[str]]] = None,
    live_repairs: Optional[dict[str, Any]] = None,
    bottleneck_locks: Optional[dict[str, Any]] = None,
) -> str:
    payload = (
        sorted((str(k), str(v)) for k, v in assign.items()),
        sorted((str(k), str(v)) for k, v in bays.items()),
        sorted((str(k), int(v)) for k, v in crews.items()),
        sorted((str(k), int(v)) for k, v in op_prio.items()),
        sorted((str(k), [str(x) for x in v]) for k, v in (task_orders or {}).items()),
        sorted((str(k), str(v.get("kind", "")), str(v.get("goal", ""))) for k, v in (live_repairs or {}).items()),
        sorted((str(k), [(str(a), int(b), int(c)) for a, b, c in v]) for k, v in (bottleneck_locks or {}).items()),
    )
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]


@dataclass
class V20CandidateArtifact:
    expert_id: str
    tag: str
    assign: dict[str, str]
    bays: dict[str, str]
    crews: dict[str, int]
    op_prio: dict[str, int]
    task_orders: Optional[dict[str, list[str]]] = None
    is_full: bool = False
    predicted_score: Optional[float] = None
    live_repairs: Optional[dict[str, dict[str, Any]]] = None
    bottleneck_locks: Optional[dict[str, list[tuple[str, int, int]]]] = None
    eval_res_b2: Optional[EvalResult] = None
    b2_ms: float = 0.0
    ja_unified_score: Optional[float] = None
    ja_proxy_details: dict[str, Any] = field(default_factory=dict)
    action_sig: str = ""

    def ensure_sig(self) -> str:
        if not self.action_sig:
            self.action_sig = candidate_action_signature(
                self.assign, self.bays, self.crews, self.op_prio,
                self.task_orders, self.live_repairs, self.bottleneck_locks,
            )
        return self.action_sig

    def to_proposal(
        self,
        *,
        policy_id: str,
        snapshot: NativeSnapshot,
        clf: Any,
        eval_res: Optional[EvalResult],
        eval_cnt: int,
        t0: float,
        budget_ms: float,
        fallback_keep: bool,
        include_eval_seq_when_unwindowed: bool = True,
    ) -> CandidateProposal:
        return _build_core_sched_proposal(
            policy_id=policy_id,
            snapshot=snapshot,
            clf=clf,
            best_assign=self.assign,
            best_bays=self.bays,
            best_crews=self.crews,
            best_eval=eval_res,
            op_prio=self.op_prio,
            eval_cnt=eval_cnt,
            t0=t0,
            budget_ms=budget_ms,
            fallback_keep=fallback_keep,
            task_sequences=self.task_orders,
            predicted_score=self.predicted_score,
            include_eval_seq_when_unwindowed=include_eval_seq_when_unwindowed,
        )


def derive_base_op_prio(snapshot: NativeSnapshot) -> dict[str, int]:
    return {op.op_id: int(op.due_step) - int(op.proc_steps) for op in snapshot.operations}


def _healthy_robots_from_graph(graph: V19HeteroGraph) -> list[str]:
    return [graph.agv_ids[i] for i in graph.healthy_agv_indices]


def _default_task_order_from_prio(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    assign: dict[str, str],
    op_prio: dict[str, int],
) -> dict[str, list[str]]:
    t_by_id = {t.task_id: t for t in snapshot.tasks}
    out: dict[str, list[str]] = {}
    for aid in _healthy_robots_from_graph(graph):
        tids = [tid for tid in graph.rel_tids if assign.get(tid) == aid]
        if tids:
            tids.sort(
                key=lambda tid: (
                    0 if t_by_id[tid].status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                    int(op_prio.get(t_by_id[tid].kit_op_id or "", 99999)),
                    int(t_by_id[tid].due_step),
                    int(t_by_id[tid].release_step),
                    tid,
                )
            )
            out[aid] = tids
    return out


def decode_v19_with_cached_enc(
    np_net: NumpyV19HeteroConstructorNet,
    enc: dict[str, np.ndarray],
    h12_raw: dict[str, np.ndarray],
    graph: V19HeteroGraph,
    snapshot: NativeSnapshot,
) -> DecodedJointCandidate:
    N_op = len(graph.op_ids)
    N_t = len(graph.rel_tids)
    N_a = len(graph.agv_ids)
    eff_stage = getattr(np_net, "action_stage", "A123")

    task_focus_np = h12_raw["task_focus_logits"][0]
    bay_logits_np = np.where(graph.op_bay_mask, h12_raw["bay_logits"][0], -1e9)
    crew_logits_np = np.where(graph.op_crew_mask, h12_raw["crew_logits"][0], -1e9)

    bay_indices = np.zeros((N_op,), dtype=np.int64)
    crew_indices = np.zeros((N_op,), dtype=np.int64)
    bays_dict: dict[str, str] = {}
    crews_dict: dict[str, int] = {}
    for oi, oid in enumerate(graph.op_ids):
        if oid in graph.frozen_ops:
            bi = int(graph.op_incumbent_bay_idx[oi])
        else:
            bi = int(np.argmax(bay_logits_np[oi])) if bay_logits_np.shape[1] > 0 else 0
        bay_indices[oi] = bi
        bays_dict[oid] = graph.bay_ids[bi]

        if graph.op_crew_mask[oi].sum() <= 1:
            ci = int(graph.op_incumbent_crew_idx[oi])
        else:
            ci = int(np.argmax(crew_logits_np[oi])) if crew_logits_np.shape[1] > 0 else 0
        crew_indices[oi] = ci
        crews_dict[oid] = int(graph.crew_ids[ci])

    assign_dict = dict(graph.fixed_assign)
    agv_indices = np.zeros((N_t,), dtype=np.int64)
    if N_t > 0:
        order = sorted(range(N_t), key=lambda i: (-float(task_focus_np[i]), i))
        added_load = np.zeros((N_a,), dtype=np.float32)
        added_dur = np.zeros((N_a,), dtype=np.float32)
        for ti in order:
            oi = int(graph.task_to_op[ti])
            sel_bi = int(bay_indices[oi]) if oi >= 0 else int(graph.task_incumbent_bay_idx[ti])
            b_tr = float(graph.bay_xy_travel_from_pickup[ti, sel_bi])
            lg = np_net.a3_step_logits(enc, graph, ti, sel_bi, added_load, added_dur)[0]
            ai = int(np.argmax(lg))
            agv_indices[ti] = ai
            assign_dict[graph.rel_tids[ti]] = graph.agv_ids[ai]
            added_load[ai] += 1.0
            added_dur[ai] += b_tr + float(graph.task_agv_feats[ti, ai, 0])

    if N_t > 0 and eff_stage == "A1234":
        seq_scores_np = np_net.a4_seq_scores(enc, agv_indices)[0]
        by_agv_rel: dict[str, list[tuple[float, str]]] = {}
        for ti, tid in enumerate(graph.rel_tids):
            aid = assign_dict[tid]
            by_agv_rel.setdefault(aid, []).append((-float(seq_scores_np[ti]), tid))
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
                        and not (
                            ordered
                            and snap_by_id.get(ordered[0]) is not None
                            and snap_by_id[ordered[0]].is_urgent
                            and not cur_t.is_urgent
                        )
                    )
                    if is_frozen_head and cur_t.task_id not in ordered:
                        ordered = [cur_t.task_id] + ordered
                    elif is_active_head and ordered[0] != rob.cur_task:
                        ordered = [rob.cur_task] + [tid for tid in ordered if tid != rob.cur_task]
            if ordered:
                task_sequences[aid] = ordered
        op_best_rank: dict[str, float] = {}
        for ti, _ in enumerate(graph.rel_tids):
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

    return DecodedJointCandidate(
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


def build_masked_subgraph(
    graph: V19HeteroGraph,
    *,
    release_tasks: set[str],
    release_ops: set[str],
    skel_assign: dict[str, str],
    skel_bays: dict[str, str],
    skel_crews: dict[str, int],
    restrict_agvs: Optional[set[str]] = None,
) -> V19HeteroGraph:
    sub = copy.copy(graph)
    sub.frozen_ops = set(graph.frozen_ops)
    sub.op_crew_mask = graph.op_crew_mask.copy()
    sub.task_agv_mask = graph.task_agv_mask.copy()
    sub.op_incumbent_bay_idx = graph.op_incumbent_bay_idx.copy()
    sub.op_incumbent_crew_idx = graph.op_incumbent_crew_idx.copy()
    sub.task_incumbent_agv_idx = graph.task_incumbent_agv_idx.copy()

    agv_idx = {aid: i for i, aid in enumerate(graph.agv_ids)}
    bay_idx = {bid: i for i, bid in enumerate(graph.bay_ids)}
    crew_idx = {int(cid): i for i, cid in enumerate(graph.crew_ids)}

    for oi, oid in enumerate(graph.op_ids):
        if oid in release_ops or oid in graph.frozen_ops:
            continue
        sub.frozen_ops.add(oid)
        sb = skel_bays.get(oid)
        if sb in bay_idx and graph.op_bay_mask[oi, bay_idx[sb]]:
            sub.op_incumbent_bay_idx[oi] = bay_idx[sb]
        sc = skel_crews.get(oid)
        if sc is not None and int(sc) in crew_idx and graph.op_crew_mask[oi, crew_idx[int(sc)]]:
            ci = crew_idx[int(sc)]
            sub.op_incumbent_crew_idx[oi] = ci
            sub.op_crew_mask[oi, :] = False
            sub.op_crew_mask[oi, ci] = True

    restrict_indices = {agv_idx[aid] for aid in restrict_agvs if aid in agv_idx} if restrict_agvs else None
    for ti, tid in enumerate(graph.rel_tids):
        sa = skel_assign.get(tid)
        if sa in agv_idx and graph.task_agv_mask[ti, agv_idx[sa]]:
            ai = agv_idx[sa]
            sub.task_incumbent_agv_idx[ti] = ai
            if tid not in release_tasks:
                sub.task_agv_mask[ti, :] = False
                sub.task_agv_mask[ti, ai] = True
            elif restrict_indices:
                allowed = [idx for idx in restrict_indices if graph.task_agv_mask[ti, idx]]
                if ai not in allowed and graph.task_agv_mask[ti, ai]:
                    allowed.append(ai)
                if allowed:
                    sub.task_agv_mask[ti, :] = False
                    for idx in allowed:
                        sub.task_agv_mask[ti, idx] = True
    return sub


def expand_causal_release_closure(
    snapshot: NativeSnapshot,
    graph: V19HeteroGraph,
    seed_tasks: list[str],
    seed_ops: list[str],
    *,
    max_tasks: int,
    max_ops: int,
) -> tuple[set[str], set[str]]:
    rel_set = set(graph.rel_tids)
    modifiable_ops = {oid for oid in graph.op_ids if oid not in graph.frozen_ops}
    op_by_id = {op.op_id: op for op in snapshot.operations}
    t_by_id = {t.task_id: t for t in snapshot.tasks}

    out_ops: list[str] = []
    out_tasks: list[str] = []

    def _add_op(oid: str) -> None:
        if oid in modifiable_ops and oid not in out_ops and len(out_ops) < max_ops:
            out_ops.append(oid)

    def _add_task(tid: str) -> None:
        if tid in rel_set and tid not in out_tasks and len(out_tasks) < max_tasks:
            out_tasks.append(tid)

    for oid in seed_ops:
        _add_op(oid)
    for tid in seed_tasks:
        _add_task(tid)
        t_obj = t_by_id.get(tid)
        if t_obj and t_obj.kit_op_id:
            _add_op(t_obj.kit_op_id)
            op_obj = op_by_id.get(t_obj.kit_op_id)
            if op_obj:
                for kt in op_obj.kit_task_ids:
                    _add_task(kt)
    for oid in list(out_ops):
        op_obj = op_by_id.get(oid)
        if op_obj:
            for kt in op_obj.kit_task_ids:
                _add_task(kt)
    return set(out_tasks), set(out_ops)


class V20BidirectionalBackbone:
    """Production CoRe-Sched V20 Bidirectional Search Backbone (V18 <-> V19 + Shared Layer-1 + Critic)."""

    def __init__(
        self,
        *,
        hard_budget_ms: float = 35.0,
        soft_deadline_ms: Optional[float] = 15.0,
        seed: int = 0,
        use_rust: bool = True,
        critic_weights_path: Optional[Path] = None,
        critic_beta: float = 0.25,
        critic_mode: str = "ws_only",
        unwin_seq_horizon: Optional[int] = 2,
        seq_diff_b2: bool = True,
        policy_id: str = "V20_NAJS",
    ) -> None:
        self.hard_budget_ms = float(hard_budget_ms)
        self.soft_deadline_ms = None if soft_deadline_ms is None else float(soft_deadline_ms)
        self.seed = int(seed)
        self.use_rust = bool(use_rust)
        self.critic_beta = float(critic_beta)
        self.critic_mode = str(critic_mode)
        self.unwin_seq_horizon = None if unwin_seq_horizon is None else int(unwin_seq_horizon)
        self.seq_diff_b2 = bool(seq_diff_b2)
        self.policy_id = str(policy_id)

        c_path = Path(critic_weights_path) if critic_weights_path is not None else _DEFAULT_V20_CRITIC_CKPT
        self._critic: Optional[RobustRankAsinhCritic] = None
        self._checkpoint_sha256: Optional[str] = None
        if c_path.is_file() and self.critic_beta > 0.0:
            self._critic = RobustRankAsinhCritic(c_path)
            self._checkpoint_sha256 = hashlib.sha256(c_path.read_bytes()).hexdigest()

        self._v19_anchor = V19AdaptiveCouplingProposer(
            _DEFAULT_V19_RL_CKPT,
            _DEFAULT_V19_STAGED_CKPT,
            backend="numpy",
            use_rust=self.use_rust,
            seed=self.seed,
        )
        self._v19_anchor._ensure_nets()
        self._v19_anchor._ensure_infer_nets()
        self._v18_pol = load_v18_synthesis_policy(path=_DEFAULT_V17_CKPT, allow_missing_pgmpo=True)

    def _effective_backend(self) -> str:
        return "numpy"

    def _finish_artifact(
        self,
        art: V20CandidateArtifact,
        *,
        snapshot: NativeSnapshot,
        graph: V19HeteroGraph,
        ev_b2: CVCREvaluator,
        obs_ops: Optional[set[str]] = None,
        fast_surr: Optional[FastJAWKittingSurrogate] = None,
    ) -> V20CandidateArtifact:
        t_eval0 = _cvcr_now_s()
        t_ord = art.task_orders if art.task_orders else None
        res = ev_b2.evaluate(
            art.assign,
            op_priority=art.op_prio,
            op_bay_overrides=art.bays,
            op_crew_overrides=art.crews,
            task_order_override=t_ord,
        )
        art.b2_ms = (_cvcr_now_s() - t_eval0) * 1000.0
        art.eval_res_b2 = res
        eff_seq = {k: list(v) for k, v in res.robot_order_seq.items() if v} if res.robot_order_seq else t_ord
        if art.task_orders is None and eff_seq:
            art.task_orders = eff_seq
        eff_ops = obs_ops if obs_ops is not None else observable_op_closure(snapshot)
        sc_u, _ = score_candidate_ja_unified(
            snapshot, ev_b2.network, graph, res, art.assign, art.bays, art.crews, eff_ops, fast_surr,
            lambda_change=15.0, sim_horizon=ev_b2.sim_horizon,
        )
        art.ja_unified_score = float(sc_u)
        art.ja_proxy_details["sim_horizon"] = int(ev_b2.sim_horizon)
        art.ensure_sig()
        return art

    def _compute_pending_or_disturbed(self, snapshot: NativeSnapshot, graph: V19HeteroGraph, clf: Any) -> set[str]:
        t_by_id = {t.task_id: t for t in snapshot.tasks}
        return {
            tid
            for tid in graph.rel_tids
            if t_by_id[tid].status.lower() == "pending"
            or (t_by_id[tid].assigned_agv is not None and t_by_id[tid].assigned_agv in clf.disturbed_robots)
            or (
                clf.disturbed_robots
                and any(
                    op.kit_task_ids and tid in op.kit_task_ids
                    for op in snapshot.operations
                    if any(t_by_id.get(kt) and t_by_id[kt].assigned_agv in clf.disturbed_robots for kt in op.kit_task_ids)
                )
            )
        }

    def propose_nominal(self, snapshot: NativeSnapshot, network: Network, *, budget_ms: float = 35.0) -> CandidateProposal:
        t0 = _cvcr_now_s()
        cap_ms = float(budget_ms) if float(budget_ms) > 0 else float(self.hard_budget_ms)
        soft_ms = (
            min(float(self.soft_deadline_ms), max(3.5, cap_ms * 0.72))
            if self.soft_deadline_ms is not None
            else max(3.5, cap_ms * 0.68)
        )
        deadline = t0 + max(3.5, soft_ms) / 1000.0

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        base_prio = derive_base_op_prio(snapshot)
        ev_b2 = CVCREvaluator(
            network, snapshot, mode="B2_SEEDED", alpha_makespan=3.0, lambda_change=15.0,
            use_rust=self.use_rust, seq_diff_b2=bool(self.seq_diff_b2),
        )
        dd_sig = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
        lazy_surr = _LazyKittingSurrogate(
            network, snapshot, base_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta
        )
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev_b2, graph, lambda_change=15.0)
        fresh_ops = observable_op_closure(snapshot)
        rob_cur_h = {r.robot_id: r.cur_task for r in snapshot.robots if r.cur_task and r.robot_id not in clf.disturbed_robots}

        keep_art = V20CandidateArtifact(
            "E0_KEEP", "keep", dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(base_prio), None, False
        )
        self._finish_artifact(keep_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, obs_ops=fresh_ops, fast_surr=fast_surr)
        best = keep_art
        keep_sc = float(keep_art.ja_unified_score or 1e9)
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev_b2.set_incumbent(graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq)

        is_ws_net = len(network.nodes) >= 30
        use_critic = bool(
            self._critic is not None and self.critic_beta > 0.0 and keep_eval.valid and (self.critic_mode != "ws_only" or is_ws_net)
        )
        keep_stats = None
        keep_critic_raw = 0.0
        sim_h_critic = int(keep_art.ja_proxy_details.get("sim_horizon", max(600, int(snapshot.tick) + 300)))
        if use_critic:
            keep_stats = build_keep_baseline_stats(snapshot, graph, keep_eval, keep_sc, rob_cur_h, sim_h_critic)
            fv_keep = extract_critic_features(
                snapshot, network, graph, keep_eval, keep_sc,
                keep_art.assign, keep_art.bays, keep_art.crews, keep_art.task_orders,
                keep_stats, rob_cur_h, sim_h_critic, is_keep_cand=True,
            )
            keep_critic_raw = self._critic.predict_delta_keep(fv_keep)
        keep_art.predicted_score = 0.0

        def _score_cand_delta(art: V20CandidateArtifact) -> float:
            ja_delta = float(art.ja_unified_score or 1e9) - keep_sc
            if not use_critic or keep_stats is None or art.eval_res_b2 is None or not art.eval_res_b2.valid:
                return ja_delta
            fv_c = extract_critic_features(
                snapshot, network, graph, art.eval_res_b2, float(art.ja_unified_score or keep_sc),
                art.assign, art.bays, art.crews, art.task_orders,
                keep_stats, rob_cur_h, sim_h_critic, is_keep_cand=False,
            )
            c_delta = float(self._critic.predict_delta_keep(fv_c) - keep_critic_raw)
            return float(ja_delta + self.critic_beta * c_delta)

        bays_semi, crews_semi = compute_semicircle_kitting_bays(snapshot, network, clf, lazy_surr, graph.base_bays, bay_delta)
        _, _, km_prio = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
        eff_km_prio = km_prio or base_prio
        pending_or_dist = self._compute_pending_or_disturbed(snapshot, graph, clf)
        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if not getattr(op, "completed", False))

        eval_cnt = 1
        seen_sigs: set[str] = {keep_art.ensure_sig()}
        nn_eval_pool: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []
        v18_skeletons: list[V20CandidateArtifact] = []

        def _consider(art: V20CandidateArtifact, *, is_skeleton: bool = False) -> bool:
            nonlocal best, eval_cnt
            sig = art.ensure_sig()
            if sig in seen_sigs or _cvcr_now_s() >= deadline:
                return False
            seen_sigs.add(sig)
            self._finish_artifact(art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, obs_ops=fresh_ops, fast_surr=fast_surr)
            eval_cnt += 1
            if art.eval_res_b2 and art.eval_res_b2.valid:
                nn_eval_pool.append((art.tag, art.assign, art.bays, art.crews, art.op_prio, art.task_orders, art.is_full))
                if is_skeleton:
                    v18_skeletons.append(art)
                sc_new = _score_cand_delta(art)
                art.predicted_score = sc_new
                sc_best = float(best.predicted_score if best.predicted_score is not None else 0.0)
                if sc_new + 1e-4 < sc_best:
                    best = art
                    ev_b2.set_incumbent(
                        best.assign, best.eval_res_b2, op_priority=best.op_prio, op_bay_overrides=best.bays, task_order_override=best.task_orders
                    )
                return True
            return False

        np_full_net = self._v19_anchor._np_full_net
        np_staged_net = self._v19_anchor._np_staged_net
        enc_full = None
        h12_full = None

        def _ensure_full_enc() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
            nonlocal enc_full, h12_full
            if enc_full is None:
                enc_full = np_full_net.encode_entities(graph)
                h12_full = np_full_net.a1_a2_heads(enc_full, graph)
            assert enc_full is not None and h12_full is not None
            return enc_full, h12_full

        if healthy and graph.rel_tids:
            if not is_ws_net:
                a_g, sq_g = _regret2_insert_tasks(snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_km_prio, bays_semi)
                _consider(V20CandidateArtifact("E1_V18_Regret2", "r2_global", a_g, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_g, True), is_skeleton=True)
                if pending_or_dist and _cvcr_now_s() < deadline:
                    anc = {tid: aid for tid, aid in graph.base_assign.items() if tid not in pending_or_dist}
                    a_z, sq_z = _regret2_insert_tasks(snapshot, graph, fast_surr, anc, sorted(pending_or_dist), dict(eff_km_prio), dict(bays_semi))
                    _consider(V20CandidateArtifact("E1_V18_Regret2", "r2_zc", a_z, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_z, False), is_skeleton=True)

            if _cvcr_now_s() + 0.002 < deadline:
                ef, hf = _ensure_full_enc()
                dec_f = decode_v19_with_cached_enc(np_full_net, ef, hf, graph, snapshot)
                _consider(
                    V20CandidateArtifact(
                        "E2_AHG_Full", "ahg_full_g",
                        dict(dec_f.assign), dict(dec_f.bays), dict(dec_f.crews), dict(dec_f.op_prio),
                        dict(dec_f.task_sequences) if dec_f.task_sequences else None, True,
                    )
                )

            if (is_ws_net or not keep_eval.valid) and _cvcr_now_s() + 0.002 < deadline:
                enc_s = np_staged_net.encode_entities(graph)
                h12_s = np_staged_net.a1_a2_heads(enc_s, graph)
                dec_s = decode_v19_with_cached_enc(np_staged_net, enc_s, h12_s, graph, snapshot)
                if not keep_eval.valid:
                    _consider(
                        V20CandidateArtifact(
                            "E3_AHG_Staged", "ahg_staged_g",
                            dict(dec_s.assign), dict(dec_s.bays), dict(dec_s.crews), dict(dec_s.op_prio),
                            dict(dec_s.task_sequences) if dec_s.task_sequences else None, False,
                        )
                    )
                zc_a, zc_b, zc_c, zc_p, zc_sq = project_zero_churn_candidate(
                    snapshot, graph, dec_s.assign, dec_s.bays, dec_s.crews, dec_s.op_prio, dec_s.task_sequences,
                    fresh_ops, pending_or_dist, rob_cur_h,
                )
                _consider(V20CandidateArtifact("E3_AHG_Staged", "ahg_staged_zc", zc_a, zc_b, zc_c, zc_p, zc_sq, False))

            if (is_ws_net or not keep_eval.valid) and (_cvcr_now_s() + 0.0035 < deadline) and getattr(self._v18_pol, "pgmpo", None) is not None and self._v18_pol.pgmpo.ready:
                asns = self._v18_pol.pgmpo.generate_assignments(snapshot, network, clf, healthy, graph.fixed_assign, bays_semi, lazy_surr)
                if asns:
                    top_p = min(
                        asns,
                        key=lambda a: float(
                            fast_surr.score(a, eff_km_prio, bays_semi, crews_semi, _default_task_order_from_prio(snapshot, graph, a, eff_km_prio))
                        ),
                    )
                    p_ord = _default_task_order_from_prio(snapshot, graph, top_p, eff_km_prio)
                    if _cvcr_now_s() + 0.002 < deadline:
                        _consider(
                            V20CandidateArtifact("E4_V18_PGMPO", "pgmpo_top", dict(top_p), dict(bays_semi), dict(crews_semi), dict(eff_km_prio), p_ord, True),
                            is_skeleton=True,
                        )

            if (_cvcr_now_s() + 0.0025 < deadline) and v18_skeletons:
                best_skel = min(v18_skeletons, key=lambda c: float(c.predicted_score if c.predicted_score is not None else 0.0))
                if best_skel.eval_res_b2 and best_skel.eval_res_b2.valid:
                    ef, hf = _ensure_full_enc()
                    cone = build_causal_repair_cone(
                        snapshot, graph, best_skel.eval_res_b2, best_skel.assign, best_skel.bays, best_skel.crews,
                        max_cone_amrs=3, max_cone_tasks=5, max_cone_ops=3,
                    )
                    rel_t_sub, rel_o_sub = expand_causal_release_closure(
                        snapshot, graph, cone.cone_tasks, cone.cone_ops, max_tasks=6, max_ops=4,
                    )
                    if rel_t_sub or rel_o_sub:
                        sub_g = build_masked_subgraph(
                            graph, release_tasks=rel_t_sub, release_ops=rel_o_sub,
                            skel_assign=best_skel.assign, skel_bays=best_skel.bays, skel_crews=best_skel.crews,
                            restrict_agvs=set(cone.cone_amrs) if cone.cone_amrs else None,
                        )
                        dec_bi = decode_v19_with_cached_enc(np_full_net, ef, hf, sub_g, snapshot)
                        _consider(
                            V20CandidateArtifact(
                                "V20B_Bidir_V18ToV19", "bidir_v18_to_v19",
                                dict(dec_bi.assign), dict(dec_bi.bays), dict(dec_bi.crews), dict(dec_bi.op_prio),
                                dict(dec_bi.task_sequences) if dec_bi.task_sequences else None, False,
                            )
                        )

            if _cvcr_now_s() + 0.0025 < deadline:
                nn_t = nn_eval_pool[:4] if nn_eval_pool else [(best.tag, best.assign, best.bays, best.crews, best.op_prio, best.task_orders, best.is_full)]
                for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                    snapshot, network, graph, ev_b2, fast_surr, keep_eval, base_prio, nn_t, fresh_ops, pending_or_dist,
                    budget_ms_limit=1.5,
                )[:4]:
                    _consider(V20CandidateArtifact("E7_CausalCone_VOI", tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc))

        changed_ops_cnt = sum(
            1
            for oid in graph.op_ids
            if best.bays.get(oid, graph.base_bays.get(oid)) != graph.base_bays.get(oid)
            or int(best.crews.get(oid, graph.base_crews.get(oid, 0))) != int(graph.base_crews.get(oid, 0))
        )
        changed_tasks_cnt = sum(
            1 for tid in graph.rel_tids if graph.base_assign.get(tid) and best.assign.get(tid) != graph.base_assign.get(tid)
        )
        eff_hys = 8.0 if (changed_ops_cnt > 0 or changed_tasks_cnt > 0) else 0.5
        best_rel_delta = float(best.predicted_score if best.predicted_score is not None else 0.0)
        fallback_keep = not (best.eval_res_b2 is not None and best.eval_res_b2.valid and best_rel_delta + eff_hys < 0.0)
        if fallback_keep:
            best = keep_art

        seq_horizon = int(self.unwin_seq_horizon) if self.unwin_seq_horizon is not None else 0
        inc_unwin = False
        orig_task_orders = best.task_orders
        if seq_horizon > 0 and not has_win:
            inc_unwin = True
            if isinstance(orig_task_orders, dict):
                best.task_orders = {k: list(v)[:seq_horizon] for k, v in orig_task_orders.items() if v}
        try:
            prop = best.to_proposal(
                policy_id=self.policy_id,
                snapshot=snapshot,
                clf=clf,
                eval_res=best.eval_res_b2,
                eval_cnt=eval_cnt,
                t0=t0,
                budget_ms=cap_ms,
                fallback_keep=fallback_keep,
                include_eval_seq_when_unwindowed=inc_unwin,
            )
        finally:
            best.task_orders = orig_task_orders

        prop.telemetry.update(
            {
                "best_expert": best.expert_id,
                "best_tag": best.tag,
                "fallback_keep": fallback_keep,
                "eval_cnt": int(eval_cnt),
                "keep_obj": round(keep_sc, 4),
                "best_obj": round(float(best.ja_unified_score or keep_sc), 4),
                "checkpoint_sha256": self._checkpoint_sha256,
                "inference_backend": "numpy",
            }
        )
        return prop


class V20NAJSProposer(V20BidirectionalBackbone):
    """Production Frozen CoRe-Sched V20.0 (`V20_NAJS`): Backbone + PDG 1.0 Liveness Escape."""

    def __init__(self, **kwargs: Any) -> None:
        kwargs.setdefault("policy_id", "V20_NAJS")
        super().__init__(**kwargs)
        self._active_pdg: Optional[dict[str, Any]] = None
        self.pdg_recoveries_triggered: int = 0

    def set_pdg_state(self, pdg: Optional[dict[str, Any]]) -> None:
        self._active_pdg = pdg

    def __call__(self, snapshot: NativeSnapshot, network: Network, *, budget_ms: float = 35.0) -> CandidateProposal:
        if not (self._active_pdg and self._active_pdg.get("pdg_alert")):
            return self.propose_nominal(snapshot, network, budget_ms=budget_ms)

        self.pdg_recoveries_triggered += 1
        pdg = self._active_pdg
        t0 = _cvcr_now_s()
        cap_ms = float(budget_ms) if float(budget_ms) > 0 else float(self.hard_budget_ms)
        deadline = t0 + max(5.0, min(cap_ms * 0.72, 23.0)) / 1000.0

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        base_prio = derive_base_op_prio(snapshot)
        ev_b2 = CVCREvaluator(network, snapshot, mode="B2_SEEDED", alpha_makespan=3.0, lambda_change=15.0, use_rust=self.use_rust, seq_diff_b2=bool(self.seq_diff_b2))
        dd_sig = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
        lazy_surr = _LazyKittingSurrogate(network, snapshot, base_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta)
        w_surr = WindowAwareKittingSurrogate(lazy_surr, ev_b2, snapshot)
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev_b2, graph, lambda_change=15.0)
        fresh_ops = observable_op_closure(snapshot)

        keep_art = V20CandidateArtifact("E0_KEEP", "keep", dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(base_prio), None, False)
        self._finish_artifact(keep_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, obs_ops=fresh_ops, fast_surr=fast_surr)
        best = keep_art
        keep_sc = float(keep_art.ja_unified_score or 1e9)
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev_b2.set_incumbent(graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq)

        t_by_id = {t.task_id: t for t in snapshot.tasks}
        protected_transit = {tid for tid in graph.rel_tids if t_by_id[tid].status.lower() in ("picked", "to_dropoff", "loading", "unloading")}
        cycle_agvs = set(pdg.get("wait_cycle_agvs") or ())
        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if not getattr(op, "completed", False))

        def _risk_score(art: V20CandidateArtifact) -> float:
            er = art.eval_res_b2
            sc = float(art.ja_unified_score or 1e9)
            if er is None or not er.valid:
                return sc + 10000.0
            changed_cyc = sum(1 for tid in graph.rel_tids if graph.base_assign.get(tid) in cycle_agvs and art.assign.get(tid) != graph.base_assign.get(tid))
            cyc_bonus = -120.0 * float(min(3, changed_cyc)) if cycle_agvs else 0.0
            starv_diff = float(er.starvation_ticks - keep_eval.starvation_ticks)
            esc_bonus = -40.0 if (art.assign != graph.base_assign or art.bays != graph.base_bays) else 0.0
            return sc + 0.5 * starv_diff + cyc_bonus + esc_bonus - keep_sc

        keep_art.predicted_score = 0.0
        bays_semi, crews_semi = compute_semicircle_kitting_bays(snapshot, network, clf, lazy_surr, graph.base_bays, bay_delta)
        _, _, km_prio = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
        eff_km_prio = km_prio or base_prio
        pending_or_dist = self._compute_pending_or_disturbed(snapshot, graph, clf)
        eval_cnt = 1
        seen_sigs: set[str] = {keep_art.ensure_sig()}
        nn_eval_pool: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []

        def _consider(art: V20CandidateArtifact) -> bool:
            nonlocal best, eval_cnt
            sig = art.ensure_sig()
            if sig in seen_sigs or _cvcr_now_s() >= deadline:
                return False
            seen_sigs.add(sig)
            self._finish_artifact(art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, obs_ops=fresh_ops, fast_surr=fast_surr)
            eval_cnt += 1
            if art.eval_res_b2 and art.eval_res_b2.valid:
                nn_eval_pool.append((art.tag, art.assign, art.bays, art.crews, art.op_prio, art.task_orders, art.is_full))
                r_sc = _risk_score(art)
                art.predicted_score = r_sc
                if r_sc + 1e-4 < float(best.predicted_score if best.predicted_score is not None else 0.0):
                    best = art
                    ev_b2.set_incumbent(best.assign, best.eval_res_b2, op_priority=best.op_prio, op_bay_overrides=best.bays, task_order_override=best.task_orders)
                return True
            return False

        if healthy and graph.rel_tids:
            a_g, sq_g = _regret2_insert_tasks(snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_km_prio, bays_semi)
            _consider(V20CandidateArtifact("V20E_PDG_Escape", "pdg_r2_global", a_g, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_g, True))
            if pending_or_dist or protected_transit:
                rel_e = (set(graph.rel_tids) - protected_transit) | pending_or_dist
                anc_e = {tid: aid for tid, aid in graph.base_assign.items() if tid not in rel_e}
                if rel_e:
                    a_e, sq_e = _regret2_insert_tasks(snapshot, graph, fast_surr, anc_e, sorted(rel_e), eff_km_prio, bays_semi)
                    _consider(V20CandidateArtifact("V20E_PDG_Escape", "pdg_r2_antithrash", a_e, dict(bays_semi), dict(crews_semi), dict(eff_km_prio), sq_e, False))
            np_full = self._v19_anchor._np_full_net
            np_staged = self._v19_anchor._np_staged_net
            if _cvcr_now_s() + 0.002 < deadline:
                ef = np_full.encode_entities(graph)
                hf = np_full.a1_a2_heads(ef, graph)
                df = decode_v19_with_cached_enc(np_full, ef, hf, graph, snapshot)
                _consider(V20CandidateArtifact("E2_AHG_Full", "pdg_ahg_full", dict(df.assign), dict(df.bays), dict(df.crews), dict(df.op_prio), df.task_sequences, True))
            if _cvcr_now_s() + 0.002 < deadline:
                es = np_staged.encode_entities(graph)
                hs = np_staged.a1_a2_heads(es, graph)
                ds = decode_v19_with_cached_enc(np_staged, es, hs, graph, snapshot)
                _consider(V20CandidateArtifact("E3_AHG_Staged", "pdg_ahg_staged_g", dict(ds.assign), dict(ds.bays), dict(ds.crews), dict(ds.op_prio), ds.task_sequences, False))
            if _cvcr_now_s() + 0.003 < deadline and getattr(self._v18_pol, "pgmpo", None) is not None and self._v18_pol.pgmpo.ready:
                for idx_p, p_asn in enumerate(self._v18_pol.pgmpo.generate_assignments(snapshot, network, clf, healthy, graph.fixed_assign, bays_semi, lazy_surr)[:2]):
                    if _cvcr_now_s() >= deadline:
                        break
                    _consider(V20CandidateArtifact("E4_V18_PGMPO", f"pdg_pgmpo_{idx_p}", dict(p_asn), dict(bays_semi), dict(crews_semi), dict(eff_km_prio), _default_task_order_from_prio(snapshot, graph, p_asn, eff_km_prio), True))
            if _cvcr_now_s() + 0.0025 < deadline:
                nnt = nn_eval_pool[:4] if nn_eval_pool else [(best.tag, best.assign, best.bays, best.crews, best.op_prio, best.task_orders, best.is_full)]
                for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                    snapshot, network, graph, ev_b2, fast_surr, keep_eval, base_prio, nnt, fresh_ops, pending_or_dist, budget_ms_limit=2.5
                )[:6]:
                    _consider(V20CandidateArtifact("E7_CausalCone_VOI", f"pdg_{tg_c}", asn_c, bs_c, cs_c, op_c, sq_c, is_fc))
            if _cvcr_now_s() + 0.003 < deadline and len(healthy) >= 2:
                pol_a, _ = _surrogate_polish_assign(w_surr, graph.rel_tids, healthy, dict(best.assign), dict(best.bays), dict(best.crews), dict(best.op_prio), max_passes=2, max_focus_tasks=10, do_swaps=True)
                _consider(V20CandidateArtifact("E6_V18_Polish", "pdg_polish", pol_a, dict(best.bays), dict(best.crews), dict(best.op_prio), _default_task_order_from_prio(snapshot, graph, pol_a, best.op_prio), False))

        best_rel = float(best.predicted_score if best.predicted_score is not None else 0.0)
        fallback_keep = not (best.eval_res_b2 is not None and best.eval_res_b2.valid and (best is not keep_art) and (best_rel < 0.0 or not keep_eval.valid))
        if fallback_keep:
            best = keep_art

        seq_horizon = int(self.unwin_seq_horizon) if self.unwin_seq_horizon is not None else 0
        inc_unwin = False
        orig_task_orders = best.task_orders
        if seq_horizon > 0 and not has_win:
            inc_unwin = True
            if isinstance(orig_task_orders, dict):
                best.task_orders = {k: list(v)[:seq_horizon] for k, v in orig_task_orders.items() if v}
        try:
            prop = best.to_proposal(
                policy_id=self.policy_id, snapshot=snapshot, clf=clf, eval_res=best.eval_res_b2, eval_cnt=eval_cnt,
                t0=t0, budget_ms=cap_ms, fallback_keep=fallback_keep, include_eval_seq_when_unwindowed=inc_unwin,
            )
        finally:
            best.task_orders = orig_task_orders
        prop.telemetry.update({"best_expert": best.expert_id, "best_tag": best.tag, "pdg_risk_mode": True, "fallback_keep": fallback_keep, "eval_cnt": int(eval_cnt), "checkpoint_sha256": self._checkpoint_sha256, "inference_backend": "numpy"})
        return prop


class V201NAJSProposer(V20BidirectionalBackbone):
    """Production Minimal Effective CoRe-Sched V20.1 (`V20_1_NAJS`).

    Combines frozen V20.0 bidirectional backbone (`0 ms` added nominal overhead) with:
      - PDG 2.0 `E1` (commitment-preserving corridor/idle-blocker evacuation + queue hoisting + bottleneck locks),
      - PDG 2.0 `E2` (state-driven `R1..R4` escalation ladder & structural relief scoring),
      - Non-destructive commitment-compatible `PersistentEscapeMemory` tabu tracking.
    """

    def __init__(self, *, max_r4_retries: int = 6, **kwargs: Any) -> None:
        kwargs.setdefault("policy_id", "V20_1_NAJS")
        super().__init__(**kwargs)
        self.max_r4_retries = int(max_r4_retries)
        self._active_pdg: Optional[dict[str, Any]] = None
        self._sim_ref: Any = None
        self.memory = PersistentEscapeMemory()
        self.pdg_recoveries_triggered: int = 0
        self.escalation_level_counts: dict[str, int] = {}
        self.recovery_durations_ms: list[float] = []
        self.irrecoverable_reports: int = 0

    def set_pdg_state(self, pdg: Optional[dict[str, Any]]) -> None:
        self._active_pdg = pdg

    def set_pdg2_context(self, pdg: Optional[dict[str, Any]], *, sim_ref: Any = None) -> None:
        self._active_pdg = pdg
        self._sim_ref = sim_ref

    @staticmethod
    def _select_escalation_level(pdg: dict[str, Any], n_rel: int) -> int:
        mode = str(pdg.get("blocking_mode", "global_deadlock"))
        no_rec = int(pdg.get("consecutive_no_recovery", 0))
        if n_rel == 0:
            return 2 if mode == "corridor_stall" else 4
        if no_rec >= 36:
            return 4
        if mode == "corridor_stall":
            return 2 if no_rec < 18 else 3
        if mode == "kitting_window_starvation":
            return 3 if no_rec >= 12 else 1
        if mode == "critical_chain_stall":
            return 1 if no_rec < 14 else (2 if no_rec < 26 else 3)
        return 3 if no_rec >= 12 else 2

    @staticmethod
    def _proposal_sig(prop: CandidateProposal) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "a": sorted((prop.proposed_assignment or {}).items()),
                    "b": sorted((prop.proposed_bays or {}).items()),
                    "lr": sorted((prop.proposed_live_repairs or {}).keys()),
                },
                separators=(",", ":"),
            ).encode()
        ).hexdigest()[:16]

    def __call__(self, snapshot: NativeSnapshot, network: Network, *, budget_ms: float = 35.0) -> CandidateProposal:
        if not (self._active_pdg and self._active_pdg.get("pdg_alert")):
            return self.propose_nominal(snapshot, network, budget_ms=budget_ms)

        self.pdg_recoveries_triggered += 1
        pdg = self._active_pdg
        s_sig, c_sig = self.memory.update_previous_outcome(snapshot, pdg)
        tabu_actions = self.memory.failed_action_sigs(s_sig, c_sig)

        t0 = _cvcr_now_s()
        cap_ms = float(budget_ms) if float(budget_ms) > 0 else float(self.hard_budget_ms)
        deadline = t0 + max(5.0, min(cap_ms * 0.72, 23.0)) / 1000.0

        clf = classify_commitments(snapshot)
        graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
        healthy = _healthy_robots_from_graph(graph)
        level = self._select_escalation_level(pdg, len(graph.rel_tids))
        lvl_key = f"R{level}"
        self.escalation_level_counts[lvl_key] = self.escalation_level_counts.get(lvl_key, 0) + 1
        if level == 4 and int(pdg.get("consecutive_no_recovery", 0)) >= 36 + self.max_r4_retries * 12:
            self.irrecoverable_reports += 1

        base_prio = derive_base_op_prio(snapshot)
        top_risk_ops = list(pdg.get("top_risk_ops") or ())
        top_risk_set = set(top_risk_ops)
        boosted_prio = dict(base_prio)
        min_p = min(boosted_prio.values()) if boosted_prio else 0
        for rank_i, oid in enumerate(top_risk_ops):
            if oid in boosted_prio:
                boosted_prio[oid] = int(min_p - 60 + rank_i * 10)

        ev_b2 = CVCREvaluator(network, snapshot, mode="B2_SEEDED", alpha_makespan=3.0, lambda_change=15.0, use_rust=self.use_rust, seq_diff_b2=bool(self.seq_diff_b2))
        dd_sig = compute_detour_delta_signals(network, snapshot)
        bay_delta = dd_sig["bay_detour_delta"] if dd_sig else {}
        lazy_surr = _LazyKittingSurrogate(network, snapshot, boosted_prio, alpha_makespan=3.0, lambda_change=15.0, bay_detour_delta=bay_delta)
        w_surr = WindowAwareKittingSurrogate(lazy_surr, ev_b2, snapshot)
        fast_surr = FastJAWKittingSurrogate(network, snapshot, ev_b2, graph, lambda_change=15.0)
        fresh_ops = observable_op_closure(snapshot)

        keep_art = V20CandidateArtifact("E0_KEEP", "keep", dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(base_prio), None, False)
        self._finish_artifact(keep_art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, obs_ops=fresh_ops, fast_surr=fast_surr)
        best = keep_art
        keep_sc = float(keep_art.ja_unified_score or 1e9)
        keep_eval = keep_art.eval_res_b2
        assert keep_eval is not None
        keep_seq = {k: list(v) for k, v in keep_eval.robot_order_seq.items() if v} if keep_eval.robot_order_seq else None
        ev_b2.set_incumbent(graph.base_assign, keep_eval, op_priority=base_prio, op_bay_overrides=graph.base_bays, task_order_override=keep_seq)

        op_by_id = {op.op_id: op for op in snapshot.operations}
        cycle_agvs = set(pdg.get("wait_cycle_agvs") or ())
        stalled_agvs = set(pdg.get("stalled_agvs") or ())
        blocked_agvs = set(pdg.get("blocked_healthy_agvs") or ())
        idle_blockers = list(pdg.get("idle_blocker_agvs") or ())
        distressed_agvs = cycle_agvs | stalled_agvs | blocked_agvs
        unblocked_healthy = [a for a in healthy if a not in distressed_agvs]
        risk_tasks = {kt for oid in top_risk_ops if oid in op_by_id for kt in op_by_id[oid].kit_task_ids if kt in graph.rel_tids}
        has_win = any(getattr(op, "resource_windows", ()) for op in snapshot.operations if op.end_step is None)

        e1_live_repairs: dict[str, dict[str, Any]] = {}
        if (
            self._sim_ref is not None
            and not getattr(self._sim_ref, "executions", None)
            and len(network.nodes) <= 100
            and (_cvcr_now_s() + 0.006 < deadline)
        ):
            haven_r, _ = plan_haven_safe_corridor_repair(self._sim_ref, snapshot, stall_threshold=3, enable_choke_oscillation_guard=True)
            e1_live_repairs.update(haven_r)
            idle_r = plan_idle_blocker_evacuations(
                snapshot, network, self._sim_ref, [a for a in idle_blockers if a not in e1_live_repairs], set(clf.disturbed_robots)
            )
            e1_live_repairs.update(idle_r)
        e1_locks = (
            build_critical_bottleneck_locks(snapshot, network, sorted(distressed_agvs), set(clf.disturbed_robots))
            if (len(network.nodes) <= 100 and _cvcr_now_s() + 0.004 < deadline)
            else {}
        )

        def _risk_score(art: V20CandidateArtifact) -> float:
            er = art.eval_res_b2
            sc = float(art.ja_unified_score or 1e9)
            if er is None or not er.valid:
                return sc + 10000.0
            relieved_blocked = sum(
                1 for tid in graph.rel_tids
                if graph.base_assign.get(tid) in distressed_agvs
                and art.assign.get(tid) != graph.base_assign.get(tid)
                and art.assign.get(tid) in unblocked_healthy
            )
            relieved_cycle = sum(
                1 for tid in graph.rel_tids
                if graph.base_assign.get(tid) in cycle_agvs and art.assign.get(tid) != graph.base_assign.get(tid)
            )
            changed_risk_tasks = sum(1 for tid in risk_tasks if art.assign.get(tid) != graph.base_assign.get(tid))
            changed_risk_bays = sum(1 for oid in top_risk_set if art.bays.get(oid) != graph.base_bays.get(oid))
            queue_hoisted = bool(
                art.task_orders
                and any(
                    seq and seq[0] in risk_tasks and (keep_seq is None or keep_seq.get(aid, [None])[0] != seq[0])
                    for aid, seq in art.task_orders.items()
                )
            )
            relief_bonus = (
                -260.0 * float(min(3, relieved_blocked))
                - 180.0 * float(min(3, relieved_cycle))
                - 150.0 * float(min(3, changed_risk_tasks))
                - 140.0 * float(min(2, changed_risk_bays))
                - (160.0 if art.live_repairs else 0.0)
                - (90.0 if art.bottleneck_locks else 0.0)
                - (110.0 if queue_hoisted else 0.0)
            )
            return (sc - keep_sc) + 0.4 * float(er.starvation_ticks - keep_eval.starvation_ticks) + relief_bonus

        keep_art.predicted_score = 0.0
        bays_semi, crews_semi = compute_semicircle_kitting_bays(snapshot, network, clf, lazy_surr, graph.base_bays, bay_delta)
        _, _, km_prio = _compute_fresh_semicircle_bays(snapshot, network, graph, fast_surr, fresh_ops)
        eff_prio = dict(boosted_prio)
        if km_prio:
            for k_o, v_o in km_prio.items():
                if k_o not in top_risk_set:
                    eff_prio[k_o] = v_o
        pending_or_dist = self._compute_pending_or_disturbed(snapshot, graph, clf)
        eval_cnt = 1
        seen_sigs: set[str] = {keep_art.ensure_sig()}
        nn_eval_pool: list[tuple[str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], Optional[dict[str, list[str]]], bool]] = []

        def _consider(art: V20CandidateArtifact) -> bool:
            nonlocal best, eval_cnt
            sig = art.ensure_sig()
            if sig in seen_sigs or _cvcr_now_s() >= deadline:
                return False
            seen_sigs.add(sig)
            self._finish_artifact(art, snapshot=snapshot, graph=graph, ev_b2=ev_b2, obs_ops=fresh_ops, fast_surr=fast_surr)
            eval_cnt += 1
            if art.eval_res_b2 and art.eval_res_b2.valid:
                nn_eval_pool.append((art.tag, art.assign, art.bays, art.crews, art.op_prio, art.task_orders, art.is_full))
                r_sc = _risk_score(art)
                art.predicted_score = r_sc
                if r_sc + 1e-4 < float(best.predicted_score if best.predicted_score is not None else 0.0):
                    best = art
                    ev_b2.set_incumbent(best.assign, best.eval_res_b2, op_priority=best.op_prio, op_bay_overrides=best.bays, task_order_override=best.task_orders)
                return True
            return False

        if e1_live_repairs or e1_locks or risk_tasks:
            hoisted_seq = _default_task_order_from_prio(snapshot, graph, graph.base_assign, eff_prio)
            _consider(
                V20CandidateArtifact(
                    "PDG2_E1_QueueCorridor", f"e1_direct_{lvl_key}",
                    dict(graph.base_assign), dict(graph.base_bays), dict(graph.base_crews), dict(eff_prio), hoisted_seq, False,
                    live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                    bottleneck_locks=(dict(e1_locks) if e1_locks else None),
                )
            )

        if healthy and graph.rel_tids:
            blocked_rel = [tid for tid in graph.rel_tids if graph.base_assign.get(tid) in distressed_agvs]
            if blocked_rel and unblocked_healthy:
                asn_ev = dict(graph.base_assign)
                for idx_t, tid in enumerate(blocked_rel):
                    asn_ev[tid] = unblocked_healthy[idx_t % len(unblocked_healthy)]
                _consider(
                    V20CandidateArtifact(
                        "PDG2_R1_EvictBlocked", f"r1_evict_{lvl_key}",
                        asn_ev, dict(graph.base_bays), dict(graph.base_crews), dict(eff_prio),
                        _default_task_order_from_prio(snapshot, graph, asn_ev, eff_prio), False,
                        live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                        bottleneck_locks=(dict(e1_locks) if e1_locks else None),
                    )
                )
            focus_t = sorted(set(risk_tasks) | set(blocked_rel) | set(pending_or_dist))
            if focus_t:
                anc_f = {tid: aid for tid, aid in graph.base_assign.items() if tid not in focus_t}
                a_f, sq_f = _regret2_insert_tasks(snapshot, graph, fast_surr, anc_f, focus_t, eff_prio, bays_semi)
                _consider(
                    V20CandidateArtifact(
                        "PDG2_R1_CritRegret2", f"r1_crit_r2_{lvl_key}", a_f, dict(bays_semi), dict(crews_semi), dict(eff_prio), sq_f, False,
                        live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                    )
                )
            if level >= 3 and _cvcr_now_s() + 0.0025 < deadline:
                rt, ro = expand_causal_release_closure(snapshot, graph, focus_t[:6] or list(graph.rel_tids)[:6], top_risk_ops[:4], max_tasks=8, max_ops=8)
                if rt or ro:
                    sub_g = build_masked_subgraph(
                        graph, release_tasks=rt, release_ops=ro, skel_assign=best.assign, skel_bays=best.bays, skel_crews=best.crews,
                        restrict_agvs=set(unblocked_healthy) if unblocked_healthy else None,
                    )
                    np_full = self._v19_anchor._np_full_net
                    ef = np_full.encode_entities(sub_g)
                    hf = np_full.a1_a2_heads(ef, sub_g)
                    df = decode_v19_with_cached_enc(np_full, ef, hf, sub_g, snapshot)
                    _consider(
                        V20CandidateArtifact(
                            "PDG2_R3_SubgraphAHG", f"r3_sub_ahg_{lvl_key}",
                            dict(df.assign), dict(df.bays), dict(df.crews), dict(eff_prio), df.task_sequences, False,
                            live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None),
                            bottleneck_locks=(dict(e1_locks) if e1_locks else None),
                        )
                    )
            if level >= 2 and _cvcr_now_s() + 0.0025 < deadline:
                nnt = nn_eval_pool[:4] if nn_eval_pool else [(best.tag, best.assign, best.bays, best.crews, best.op_prio, best.task_orders, best.is_full)]
                for tg_c, asn_c, bs_c, cs_c, op_c, sq_c, is_fc in generate_causal_cone_candidates(
                    snapshot, network, graph, ev_b2, fast_surr, keep_eval, eff_prio, nnt, fresh_ops, pending_or_dist, budget_ms_limit=2.5
                )[:5]:
                    _consider(V20CandidateArtifact("E7_CausalCone_VOI", f"pdg2_{tg_c}", asn_c, bs_c, cs_c, op_c, sq_c, is_fc, live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None)))
            if level >= 3 and _cvcr_now_s() + 0.003 < deadline:
                a_g, sq_g = _regret2_insert_tasks(snapshot, graph, fast_surr, dict(graph.fixed_assign), list(graph.rel_tids), eff_prio, bays_semi)
                _consider(V20CandidateArtifact("PDG2_R4_SoftReinit", f"r4_reinit_{lvl_key}", a_g, dict(bays_semi), dict(crews_semi), dict(eff_prio), sq_g, True, live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None)))
            if level >= 3 and _cvcr_now_s() + 0.004 < deadline and len(healthy) >= 2:
                pol_a, _ = _surrogate_polish_assign(w_surr, graph.rel_tids, unblocked_healthy if len(unblocked_healthy) >= 2 else healthy, dict(best.assign), dict(best.bays), dict(best.crews), dict(best.op_prio), max_passes=2, max_focus_tasks=10, do_swaps=True)
                _consider(V20CandidateArtifact("E6_V18_Polish", f"pdg2_polish_{lvl_key}", pol_a, dict(best.bays), dict(best.crews), dict(best.op_prio), _default_task_order_from_prio(snapshot, graph, pol_a, best.op_prio), False, live_repairs=(dict(e1_live_repairs) if e1_live_repairs else None)))

        best_rel = float(best.predicted_score if best.predicted_score is not None else 0.0)
        fallback_keep = not (best.eval_res_b2 is not None and best.eval_res_b2.valid and (best is not keep_art) and (best_rel < -1e-4 or not keep_eval.valid))
        if fallback_keep:
            best = keep_art

        seq_horizon = int(self.unwin_seq_horizon) if self.unwin_seq_horizon is not None else 0
        inc_unwin = False
        orig_task_orders = best.task_orders
        if seq_horizon > 0 and not has_win and best.expert_id != "PDG2_E1_QueueCorridor":
            inc_unwin = True
            if isinstance(orig_task_orders, dict):
                best.task_orders = {k: list(v)[:seq_horizon] for k, v in orig_task_orders.items() if v}
        elif best.expert_id == "PDG2_E1_QueueCorridor" and best.task_orders:
            inc_unwin = True
        try:
            prop = best.to_proposal(
                policy_id=self.policy_id, snapshot=snapshot, clf=clf, eval_res=best.eval_res_b2, eval_cnt=eval_cnt,
                t0=t0, budget_ms=cap_ms, fallback_keep=fallback_keep, include_eval_seq_when_unwindowed=inc_unwin,
            )
        finally:
            best.task_orders = orig_task_orders

        if not fallback_keep and best.live_repairs:
            prop.proposed_live_repairs = dict(best.live_repairs)
        if not fallback_keep and best.bottleneck_locks:
            prop.proposed_bottleneck_locks = dict(best.bottleneck_locks)

        a_sig = self._proposal_sig(prop)
        dup_detected = bool(tabu_actions and a_sig in tabu_actions and not fallback_keep)
        if dup_detected:
            self.memory.duplicate_candidates_pruned += 1
        self.memory.record_attempt(s_sig, c_sig, best.tag, a_sig, int(snapshot.tick))

        dur_ms = (_cvcr_now_s() - t0) * 1000.0
        self.recovery_durations_ms.append(round(dur_ms, 3))
        prop.telemetry.update(
            {
                "best_expert": best.expert_id,
                "best_tag": best.tag,
                "pdg2_mode": True,
                "escalation_level": lvl_key,
                "blocking_mode": pdg.get("blocking_mode"),
                "liveness_class": pdg.get("liveness_class", "STARVATION"),
                "fallback_keep": fallback_keep,
                "persistent_tabu_duplicate_detected": dup_detected,
                "persistent_memory_pruned": int(self.memory.duplicate_candidates_pruned),
                "eval_cnt": int(eval_cnt),
                "checkpoint_sha256": self._checkpoint_sha256,
                "inference_backend": "numpy",
            }
        )
        return prop


def build_v20_najs_proposer(*, hard_budget_ms: float = 35.0, seed: int = 0, use_rust: bool = True, critic_weights_path: Optional[Path] = None) -> V20NAJSProposer:
    return V20NAJSProposer(hard_budget_ms=hard_budget_ms, seed=seed, use_rust=use_rust, critic_weights_path=critic_weights_path)


def build_v20_1_najs_proposer(*, hard_budget_ms: float = 35.0, seed: int = 0, use_rust: bool = True, critic_weights_path: Optional[Path] = None) -> V201NAJSProposer:
    return V201NAJSProposer(hard_budget_ms=hard_budget_ms, seed=seed, use_rust=use_rust, critic_weights_path=critic_weights_path)


__all__ = [
    "KeepBaselineStats",
    "RobustRankAsinhCritic",
    "V201NAJSProposer",
    "V20BidirectionalBackbone",
    "V20CandidateArtifact",
    "V20NAJSProposer",
    "build_masked_subgraph",
    "build_v20_1_najs_proposer",
    "build_v20_najs_proposer",
    "candidate_action_signature",
    "decode_v19_with_cached_enc",
    "derive_base_op_prio",
    "expand_causal_release_closure",
]
