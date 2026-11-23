"""V18 AMR scheduling with surrogate screening and exact route verification.

Candidates combine congestion-aware workstation assignment, precedence-closure
regret insertion, and a sliding-window neighborhood selector. A NumPy generator
uses the bit-conditioned PGMPO checkpoint with cached attention and approximate
load updates. The shortlist is checked by the existing route evaluator before
ranking. Production window protection and transaction checks reuse V17 machinery.
"""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Any, Optional

import numpy as np

from amr_sched.baselines.numpy_backends import (
    elu as _elu,
    file_sha256,
    layer_norm as _layer_norm,
    load_numpy_state_dict,
    mha_batch as _mha_batch,
)
from amr_sched.cvcr.checkpoints import resolve_default_checkpoint
from amr_sched.cvcr.clock import is_virtual_clock_active, record_work_ms as _cvcr_record_work_ms
from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.snapshot import NativeSnapshot, op_predecessors
from amr_sched.cvcr.v12_proactive_guard import compute_kitting_marginal_start_value
from amr_sched.cvcr.v15_ea_joint import (
    V15Candidate,
    _build_candidate_features,
    compute_action_diff_digest,
    extract_v15_global_features,
    _find_danger_cliff_ops,
)
from amr_sched.cvcr.v16_frontier import V16FrontierPolicy
from amr_sched.cvcr.v17_window_protection import (
    DEFAULT_V17_CKPT,
    V17_PAIRWISE_DIM,
    V17AdaptiveController,
    V17WindowProtectionPolicy,
    compute_pairwise_payoff,
    load_v17_window_policy,
    plan_bridge_order_control,
    select_coupled_op_pair,
)
from amr_sched.network import Network

_PGMPO_NPZ_PATH = resolve_default_checkpoint("baselines_2026/ijcai26_choi_pgmpo.npz")
_PGMPO_PT_PATH = resolve_default_checkpoint("baselines_2026/ijcai26_choi_pgmpo.pt")


def _load_pgmpo_numpy_weights(
    ckpt_path: Optional[Path | str] = None,
    *,
    allow_missing: bool = False,
) -> tuple[Optional[dict[str, np.ndarray]], Optional[str]]:
    """Load Choi IJCAI-26 PGMPO weights into float32 NumPy arrays; raise on missing unless `allow_missing=True`."""
    p = Path(ckpt_path) if ckpt_path is not None else (_PGMPO_NPZ_PATH if _PGMPO_NPZ_PATH.is_file() else _PGMPO_PT_PATH)
    try:
        weights, _, sha = load_numpy_state_dict(p, prefer_npz=True)
        return weights, sha
    except FileNotFoundError:
        if allow_missing:
            return None, None
        raise
    except Exception:
        if allow_missing:
            return None, None
        raise


class NumpyPGMPOGenerator:
    """Zero-PyTorch-overhead pure-NumPy inference engine for Choi et al. (IJCAI-26) Bit-DAN."""

    def __init__(
        self,
        ckpt_path: Optional[Path | str] = None,
        c_policies: int = 8,
        *,
        allow_missing: bool = False,
    ) -> None:
        """Initialize cached NumPy weights and deterministic `C` bit-conditioning vectors."""
        self.w, self.checkpoint_sha256 = _load_pgmpo_numpy_weights(ckpt_path, allow_missing=allow_missing)
        self.c_policies = max(2, min(16, int(c_policies)))
        self.bit_dim = 4
        bits = [[(c >> b) & 1 for b in range(self.bit_dim)] for c in range(self.c_policies)]
        self.bits = np.asarray(bits, dtype=np.float32)

    @property
    def ready(self) -> bool:
        """Return True iff trained quarantined PGMPO weights are loaded in NumPy."""
        return self.w is not None and "W1.weight" in self.w

    def _encode_attention(
        self,
        op_x: np.ndarray,
        res_x: np.ndarray,
        bits: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run 2-layer Bit-Conditioned Dual Attention ONCE returning `(h_o, h_m, h_g)`."""
        w = self.w
        assert w is not None
        h_o = (op_x @ w["W1.weight"].T + w["W1.bias"]) + (bits @ w["W3.weight"].T)[:, None, :]
        h_m = (res_x @ w["W2.weight"].T + w["W2.bias"]) + (bits @ w["W4.weight"].T)[:, None, :]
        for l_idx in range(2):
            o_att = _mha_batch(
                h_o,
                h_o,
                w[f"omb_layers.{l_idx}.in_proj_weight"],
                w[f"omb_layers.{l_idx}.in_proj_bias"],
                w[f"omb_layers.{l_idx}.out_proj.weight"],
                w[f"omb_layers.{l_idx}.out_proj.bias"],
                n_heads=4,
            )
            h_o = _layer_norm(h_o + o_att, w[f"omb_norms.{l_idx}.weight"], w[f"omb_norms.{l_idx}.bias"])
            m_att = _mha_batch(
                h_m,
                h_o,
                w[f"mmb_layers.{l_idx}.in_proj_weight"],
                w[f"mmb_layers.{l_idx}.in_proj_bias"],
                w[f"mmb_layers.{l_idx}.out_proj.weight"],
                w[f"mmb_layers.{l_idx}.out_proj.bias"],
                n_heads=4,
            )
            h_m = _layer_norm(h_m + m_att, w[f"mmb_norms.{l_idx}.weight"], w[f"mmb_norms.{l_idx}.bias"])
        h_g = np.concatenate([h_o.mean(axis=1), h_m.mean(axis=1)], axis=-1)
        return h_o, h_m, h_g

    def _score_task_step(
        self,
        h_o_ti: np.ndarray,
        h_m: np.ndarray,
        h_g: np.ndarray,
        pair_ti: np.ndarray,
        dyn_res_delta: np.ndarray,
    ) -> np.ndarray:
        """Fast single-task O(C * N_a * H) pair_mlp decode reusing cached Dual-Attention embeddings."""
        w = self.w
        assert w is not None
        C, N_a, H = h_m.shape
        hm_dyn = h_m + dyn_res_delta
        ho_e = np.broadcast_to(h_o_ti[:, None, :], (C, N_a, H))
        hg_e = np.broadcast_to(h_g[:, None, :], (C, N_a, h_g.shape[-1]))
        feat = np.concatenate([ho_e, hm_dyn, hg_e, pair_ti], axis=-1)
        h1 = _elu(feat @ w["pair_mlp.0.weight"].T + w["pair_mlp.0.bias"])
        h2 = _elu(h1 @ w["pair_mlp.2.weight"].T + w["pair_mlp.2.bias"])
        return (h2 @ w["pair_mlp.4.weight"].T + w["pair_mlp.4.bias"])[..., 0]

    def generate_assignments(
        self,
        snapshot: NativeSnapshot,
        network: Network,
        clf: CommitmentClassification,
        healthy_robots: list[str],
        fixed_assign: dict[str, str],
        op_bays: dict[str, str],
        surr: Any,
    ) -> list[dict[str, str]]:
        """Run `C` sequential autoregressive load-updating bit-conditioned policies in pure NumPy."""
        if not self.ready:
            return []
        rel_tids = sorted(clf.releasable_tasks)
        n_t = len(rel_tids)
        n_a = len(healthy_robots)
        if n_t == 0 or n_a < 2:
            return []

        xs = [float(n.x) for n in network.nodes.values()] or [0.0]
        ys = [float(n.y) for n in network.nodes.values()] or [0.0]
        minx, spanx = min(xs), max(1.0, max(xs) - min(xs))
        miny, spany = min(ys), max(1.0, max(ys) - min(ys))
        tick = int(snapshot.tick)
        tmap = {t.task_id: t for t in snapshot.tasks}
        op_map = {op.op_id: op for op in snapshot.operations}
        rob_map = {r.robot_id: r for r in snapshot.robots}

        fixed_counts = {aid: 0 for aid in healthy_robots}
        for tid, aid in fixed_assign.items():
            if aid in fixed_counts:
                fixed_counts[aid] += 1
        res_counts = {aid: 0 for aid in healthy_robots}
        for rv in snapshot.reservations:
            if rv.agv_id in res_counts:
                res_counts[rv.agv_id] += 1

        agv_feats = np.zeros((n_a, 6), dtype=np.float32)
        for j, aid in enumerate(healthy_robots):
            r = rob_map.get(aid)
            nd = network.nodes.get(r.cur_node) if (r and r.cur_node in network.nodes) else None
            nx = ((float(nd.x) - minx) / spanx) if nd else 0.5
            ny = ((float(nd.y) - miny) / spany) if nd else 0.5
            busy = 1.0 if (r and r.cur_task is not None) else 0.0
            loaded = 1.0 if (r and r.loaded_task_id is not None) else 0.0
            q_norm = min(1.0, fixed_counts[aid] / 6.0)
            rem_steps = float(len(r.route)) / 20.0 if r else 0.0
            agv_feats[j] = [nx, ny, busy, loaded, q_norm, min(1.0, rem_steps)]

        task_feats = np.zeros((n_t, 8), dtype=np.float32)
        pair_feats = np.zeros((n_t, n_a, 6), dtype=np.float32)
        for i, tid in enumerate(rel_tids):
            t = tmap[tid]
            pn = network.nodes.get(t.pickup)
            dn_id = op_bays.get(t.kit_op_id or "", t.dropoff) if t.kit_op_id else t.dropoff
            dn = network.nodes.get(dn_id)
            px = ((float(pn.x) - minx) / spanx) if pn else 0.5
            py = ((float(pn.y) - miny) / spany) if pn else 0.5
            dx = ((float(dn.x) - minx) / spanx) if dn else 0.5
            dy = ((float(dn.y) - miny) / spany) if dn else 0.5
            travel = float(surr.tt(t.pickup, dn_id)) / 40.0
            due = float(t.due_step if t.due_step is not None else (tick + 120))
            slack = max(-1.0, min(2.0, (due - tick) / 80.0))
            urg = 1.0 if t.is_urgent else 0.0
            op = op_map.get(t.kit_op_id or "")
            win_tight = 1.0 if (op is not None and getattr(op, "resource_windows", ())) else 0.0
            task_feats[i] = [px, py, dx, dy, min(2.0, travel), slack, urg, win_tight]
            for j, aid in enumerate(healthy_robots):
                r = rob_map.get(aid)
                start_nd = r.cur_node if (r and r.cur_node in network.nodes) else t.pickup
                deadhead = float(surr.tt(start_nd, t.pickup))
                keep = 1.0 if t.assigned_agv == aid else 0.0
                same_bay_busy = min(1.0, fixed_counts[aid] / 5.0)
                est_comp = deadhead + travel * 40.0
                est_slack = max(-1.0, min(2.0, (due - (tick + est_comp)) / 60.0))
                res_load = min(1.0, res_counts[aid] / 25.0)
                pair_feats[i, j] = [
                    min(2.0, deadhead / 30.0),
                    keep,
                    same_bay_busy,
                    est_slack,
                    res_load,
                    1.0 - min(1.0, est_comp / 80.0),
                ]

        C = self.c_policies
        task_order = sorted(range(n_t), key=lambda i: (float(task_feats[i, 5]), -float(task_feats[i, 6]), i))
        op_b = np.broadcast_to(task_feats[None, :, :], (C, n_t, 8))
        res_0 = np.repeat(agv_feats[None, :, :], C, axis=0)
        res_cur = res_0.copy()
        pair_cur = np.repeat(pair_feats[None, :, :, :], C, axis=0)
        # Reuse the attention context and approximate the effect of incremental loads.
        h_o, h_m, h_g = self._encode_attention(op_b, res_0, self.bits)
        w2_t = self.w["W2.weight"].T  # type: ignore[index]
        choices = np.zeros((C, n_t), dtype=np.int64)
        c_ar = np.arange(C)

        for ti in task_order:
            dyn_delta = (res_cur - res_0) @ w2_t
            step_logits = self._score_task_step(h_o[:, ti, :], h_m, h_g, pair_cur[:, ti, :, :], dyn_delta)
            chosen_ai = np.argmax(step_logits, axis=-1)
            choices[:, ti] = chosen_ai
            res_cur[c_ar, chosen_ai, 2] = 1.0
            res_cur[c_ar, chosen_ai, 4] = np.minimum(1.0, res_cur[c_ar, chosen_ai, 4] + (1.0 / 6.0))
            res_cur[c_ar, chosen_ai, 5] = np.minimum(1.0, res_cur[c_ar, chosen_ai, 5] + 0.18)
            pair_cur[c_ar, :, chosen_ai, 2] = np.minimum(1.0, pair_cur[c_ar, :, chosen_ai, 2] + 0.20)
            pair_cur[c_ar, :, chosen_ai, 3] = np.maximum(-1.0, pair_cur[c_ar, :, chosen_ai, 3] - 0.22)
            pair_cur[c_ar, :, chosen_ai, 4] = np.minimum(1.0, pair_cur[c_ar, :, chosen_ai, 4] + 0.15)
            pair_cur[c_ar, :, chosen_ai, 5] = np.maximum(0.0, pair_cur[c_ar, :, chosen_ai, 5] - 0.18)

        out: list[dict[str, str]] = []
        seen: set[tuple[tuple[str, str], ...]] = set()
        for c_i in range(C):
            cand = dict(fixed_assign)
            for i, tid in enumerate(rel_tids):
                cand[tid] = healthy_robots[int(choices[c_i, i])]
            sig = tuple(sorted(cand.items()))
            if sig not in seen:
                seen.add(sig)
                out.append(cand)
        return out


class WindowAwareKittingSurrogate:
    """Estimate route times and reuse the production timing model for scoring."""

    def __init__(
        self,
        base_surr: Any,
        evaluator: CVCREvaluator,
        snapshot: NativeSnapshot,
    ) -> None:
        """Wrap `_LazyKittingSurrogate` and exact production propagator for zero-Rust screening."""
        self.surr = base_surr
        self.evaluator = evaluator
        self.snapshot = snapshot
        self.tasks = list(snapshot.tasks)
        self.robots_by_id = {r.robot_id: r for r in snapshot.robots}

    def predict_unloads(
        self,
        assign: dict[str, str],
        op_prio: dict[str, int],
        op_bays: dict[str, str],
    ) -> dict[str, int]:
        """Predict task unload ticks using O(N) shortest-path travel times & Detour-Delta bay delays."""
        unloads: dict[str, int] = {}
        by_r: dict[str, list[Any]] = {}
        tick = int(self.snapshot.tick)
        for t in self.tasks:
            st = t.status.lower()
            if st == "completed":
                unloads[t.task_id] = int(t.complete_step or tick)
            else:
                rid = assign.get(t.task_id)
                if rid:
                    by_r.setdefault(rid, []).append(t)

        surr = self.surr
        for rid, tlist in by_r.items():
            rob = self.robots_by_id.get(rid)
            if rob is None:
                continue
            cur_n = rob.cur_node
            cur_t = float(max(tick, rob.faulted_until))
            tlist.sort(
                key=lambda x: (
                    0 if x.status.lower() in ("picked", "loading", "to_dropoff", "unloading") else 1,
                    op_prio.get(x.kit_op_id or "", 0),
                    x.release_step,
                    x.task_id,
                )
            )
            cong = 1.0 + 0.045 * max(0, len(tlist) - 2)
            for t in tlist:
                st = t.status.lower()
                drop = op_bays.get(t.kit_op_id, t.dropoff) if (t.kit_op_id and st in ("pending", "assigned")) else t.dropoff
                if st in ("picked", "loading", "to_dropoff", "unloading"):
                    arr = cur_t + cong * surr.tt(cur_n, drop) + surr.bay_delay(drop) + 2.0
                else:
                    tp = max(cur_t + cong * surr.tt(cur_n, t.pickup), float(t.release_step)) + 2.0
                    arr = tp + cong * surr.tt(t.pickup, drop) + surr.bay_delay(drop) + 2.0
                unloads[t.task_id] = int(round(arr + 0.499))
                cur_t = arr
                cur_n = drop
        return unloads

    def score(
        self,
        assign: dict[str, str],
        op_prio: dict[str, int],
        op_bays: dict[str, str],
        op_crews: dict[str, int],
    ) -> float:
        """Score estimated unloads with production precedence, windows, and resource constraints."""
        unloads = self.predict_unloads(assign, op_prio, op_bays)
        _, _, _, obj, _ = self.evaluator._compute_objective(
            assign,
            unloads,
            op_prio,
            True,
            op_bays,
            op_crews,
        )
        return float(obj)


def compute_semicircle_kitting_bays(
    snapshot: NativeSnapshot,
    network: Network,
    clf: CommitmentClassification,
    surr: Any,
    base_bays: dict[str, str],
    bay_delta: dict[str, float],
    eligible_op_ids: Optional[set[str]] = None,
    spread_empty_kits: bool = False,
) -> tuple[dict[str, str], dict[str, int]]:
    """Wu EJOR'26 Semi-Circle Proximity + Zhang ICRA'26 Switch-Cost Bay Co-Locator (ACID & fresh-op locked)."""
    tmap = {t.task_id: t for t in snapshot.tasks}
    bays = dict(base_bays)
    km = compute_kitting_marginal_start_value(snapshot)
    bay_load: dict[str, int] = {}
    ops_sorted = sorted(
        snapshot.operations,
        key=lambda o: (
            0 if o.started else 1,
            0 if any(kt in clf.frozen_tasks for kt in o.kit_task_ids) else 1,
            0 if (eligible_op_ids is not None and o.op_id not in eligible_op_ids) else 1,
            0 if o.is_urgent else 1,
            int(o.due_step),
            -len(o.kit_task_ids),
            o.op_id,
        ),
    )
    km_prio: dict[str, int] = {}
    for op in ops_sorted:
        k_min = min(
            (km[kt][0] * 1000 + km[kt][4] for kt in op.kit_task_ids if kt in km),
            default=int(op.due_step),
        )
        km_prio[op.op_id] = int(k_min)
        kit_locked = any(kt in clf.frozen_tasks for kt in op.kit_task_ids)
        kits = [tmap[tid] for tid in op.kit_task_ids if tid in tmap and tmap[tid].status.lower() != "completed"]
        can_spread = bool(spread_empty_kits and not kits)
        if (
            op.started
            or kit_locked
            or (not kits and not can_spread)
            or not op.eligible_bays
            or len(op.eligible_bays) <= 1
            or (eligible_op_ids is not None and (not can_spread) and op.op_id not in eligible_op_ids)
        ):
            cur_b = bays.get(op.op_id, op.bay_id)
            bay_load[cur_b] = bay_load.get(cur_b, 0) + 1
            continue

        pm = (1.5 if op.is_urgent else 1.0) + 0.25 * len(kits)
        has_win = bool(getattr(op, "resource_windows", ()))
        best_b = op.bay_id
        best_cost = float("inf")
        for cand_b in op.eligible_bays:
            if cand_b not in network.nodes:
                continue
            d_sum = sum(float(surr.tt(t.pickup, cand_b)) for t in kits) if kits else 0.0
            ld = bay_load.get(cand_b, 0)
            # Wu'26 semi-circle proximity + ICRA'26 Eq.(4) switch/congestion pressure + RA-L'26 Detour-Delta
            sw_est = 0.8 * ld + 0.5 * max(0, len(kits) - 1) + (1.2 if has_win else 0.0)
            sw_w = 1.0 / (1.0 + math.exp(-0.3 * sw_est))
            cong_pen = 6.0 * ld + 2.5 * sw_w
            dd_pen = 1.0 * float(bay_delta.get(cand_b, 0.0))
            score = pm * d_sum + cong_pen + dd_pen
            if score < best_cost:
                best_cost = score
                best_b = cand_b
        bays[op.op_id] = best_b
        bay_load[best_b] = bay_load.get(best_b, 0) + 1
    return bays, km_prio


def _surrogate_polish_assign(
    w_surr: WindowAwareKittingSurrogate,
    rel_tids: list[str],
    healthy_robots: list[str],
    assign_init: dict[str, str],
    bays: dict[str, str],
    crews: dict[str, int],
    op_prio: dict[str, int],
    *,
    max_passes: int = 1,
    max_focus_tasks: int = 8,
    do_swaps: bool = False,
) -> tuple[dict[str, str], float]:
    """Search single-task reassignments for the most critical releasable tasks."""
    cur = dict(assign_init)
    best_sc = w_surr.score(cur, op_prio, bays, crews)
    if not rel_tids or len(healthy_robots) < 2:
        return cur, best_sc

    tmap = {t.task_id: t for t in w_surr.tasks}
    # Prioritize unassigned, urgent, and tightest-due tasks.
    focus = sorted(
        rel_tids,
        key=lambda tid: (
            0 if (tmap[tid].assigned_agv is None or tmap[tid].is_urgent) else 1,
            int(op_prio.get(tmap[tid].kit_op_id or "", 999)),
            int(tmap[tid].due_step if tmap[tid].due_step is not None else 999),
            tid,
        ),
    )[:max_focus_tasks]
    for _ in range(max_passes):
        improved = False
        for tid in focus:
            orig = cur.get(tid)
            best_r = orig
            for rid in healthy_robots:
                if rid == orig:
                    continue
                cur[tid] = rid
                sc = w_surr.score(cur, op_prio, bays, crews)
                if sc + 1.0 < best_sc:
                    best_sc = sc
                    best_r = rid
                    improved = True
            if best_r is not None:
                cur[tid] = best_r
        if not improved:
            break

    if do_swaps and 2 <= len(focus) <= 6:
        n_f = len(focus)
        for i in range(n_f):
            ti = focus[i]
            ai = cur.get(ti)
            for j in range(i + 1, n_f):
                tj = focus[j]
                aj = cur.get(tj)
                if not ai or not aj or ai == aj:
                    continue
                cur[ti], cur[tj] = aj, ai
                sc = w_surr.score(cur, op_prio, bays, crews)
                if sc + 2.0 < best_sc:
                    best_sc = sc
                    ai, aj = aj, ai
                else:
                    cur[ti], cur[tj] = ai, aj
    return cur, best_sc


def _mit26_precedence_closure_destroy(
    snapshot: NativeSnapshot,
    rel_tids: list[str],
    base_eval: EvalResult,
    assign: dict[str, str],
    mode_idx: int,
) -> list[str]:
    """Parimi & Williams (MIT CSAIL 2026) precedence-wait / low-slack / conflict destroy + transitive closure."""
    if not rel_tids:
        return []
    tmap = {t.task_id: t for t in snapshot.tasks}
    op_map = {op.op_id: op for op in snapshot.operations}
    rel_set = set(rel_tids)

    if mode_idx == 0:
        # LOW_SLACK: tightest due_step - op_end
        scored = []
        for tid in rel_tids:
            op = op_map.get(tmap[tid].kit_op_id or "")
            end_t = float(base_eval.op_end.get(op.op_id, snapshot.tick)) if (op and base_eval.valid) else float(snapshot.tick)
            due = float(op.due_step if op else (tmap[tid].due_step or 999))
            scored.append((due - end_t, tid))
        seeds = [tid for _, tid in sorted(scored)[:4]]
    elif mode_idx == 1:
        # AGENT_CONFLICT: heaviest-loaded AGVs
        loads: dict[str, int] = {}
        for _, aid in assign.items():
            loads[aid] = loads.get(aid, 0) + 1
        seeds = sorted(rel_tids, key=lambda tid: (-loads.get(assign.get(tid, ""), 0), tid))[:4]
    else:
        # PRECEDENCE_WAIT: largest wait between release & op start
        scored = []
        for tid in rel_tids:
            t = tmap[tid]
            op = op_map.get(t.kit_op_id or "")
            st_t = float(base_eval.op_start.get(op.op_id, snapshot.tick)) if (op and base_eval.valid) else float(snapshot.tick)
            scored.append((-(st_t - float(t.release_step)), tid))
        seeds = [tid for _, tid in sorted(scored)[:4]]

    # Transitive successor closure over production DAG
    succ_ops: dict[str, list[str]] = {}
    for op in snapshot.operations:
        for p in op_predecessors(op):
            succ_ops.setdefault(p, []).append(op.op_id)
    seed_ops: deque[str] = deque()
    for tid in seeds:
        oid = tmap[tid].kit_op_id
        if oid:
            seed_ops.append(oid)
    closed_ops: set[str] = set()
    while seed_ops:
        cur_o = seed_ops.popleft()
        if cur_o in closed_ops:
            continue
        closed_ops.add(cur_o)
        for nxt in succ_ops.get(cur_o, ()):
            if nxt not in closed_ops:
                seed_ops.append(nxt)
    out = set(seeds)
    for oid in closed_ops:
        op = op_map.get(oid)
        if op:
            for kt in op.kit_task_ids:
                if kt in rel_set:
                    out.add(kt)
    return sorted(out)


class V18FrontierSynthesisPolicy(V17WindowProtectionPolicy):
    """V18 unified policy embedding DyMAB non-stationary bandit, pure-NumPy PGMPO, and V17 ranker."""

    DYMAB_ARMS = ("H_rand", "H_sync", "H_entr")

    VALID_CANDIDATE_MODES = ("full", "no_neural", "neural_only")
    VALID_HYSTERESIS_MODES = ("default", "none", "adaptive")

    def __init__(
        self,
        weights: Optional[dict[str, np.ndarray]] = None,
        *,
        pgmpo_ckpt: Optional[Path | str] = None,
        conservative_gate: float = 1.5,
        zero_keep_residual: bool = True,
        allow_probe_override: bool = False,
        router_mode: str = "learned",
        ranker_mode: str = "pairwise",
        enable_restraint: bool = True,
        max_rust_verifications: int = 4,
        dymab_window: int = 12,
        split_search: bool = False,
        candidate_mode: str = "full",
        hysteresis_mode: str = "default",
        allow_missing_pgmpo: bool = False,
        v17_checkpoint_sha256: Optional[str] = None,
    ) -> None:
        """Initialize V18 synthesis policy with sliding-window DyMAB, NumPy PGMPO generator, and counterfactual attribution modes."""
        if candidate_mode not in self.VALID_CANDIDATE_MODES:
            raise ValueError(f"Unsupported candidate_mode {candidate_mode!r}; expected one of {self.VALID_CANDIDATE_MODES}")
        if hysteresis_mode not in self.VALID_HYSTERESIS_MODES:
            raise ValueError(f"Unsupported hysteresis_mode {hysteresis_mode!r}; expected one of {self.VALID_HYSTERESIS_MODES}")
        self.split_search = bool(split_search)
        self.candidate_mode = str(candidate_mode)
        self.hysteresis_mode = str(hysteresis_mode)
        self.allow_missing_pgmpo = bool(allow_missing_pgmpo)
        self.v17_checkpoint_sha256 = v17_checkpoint_sha256
        self.pgmpo = NumpyPGMPOGenerator(ckpt_path=pgmpo_ckpt, c_policies=8, allow_missing=self.allow_missing_pgmpo)
        self.max_rust_verifications = max(2, int(max_rust_verifications))
        self.dymab_window = max(4, int(dymab_window))
        super().__init__(
            weights=weights,
            conservative_gate=conservative_gate,
            zero_keep_residual=zero_keep_residual,
            allow_probe_override=allow_probe_override,
            router_mode=router_mode,
            ranker_mode=ranker_mode,
            enable_restraint=enable_restraint,
        )

    def reset(self) -> None:
        """Reset episodic bandit statistics including Sow JAIR'26 sliding-window DyMAB queues."""
        super().reset()
        self._dymab_q = {h: deque(maxlen=self.dymab_window) for h in self.DYMAB_ARMS}
        self._dymab_n = {h: 0 for h in self.DYMAB_ARMS}
        self._dymab_iter = 0
        self.last_funnel: Optional[dict[str, Any]] = None
        self.funnel_history: list[dict[str, Any]] = []

    def select_dymab_arm(self) -> str:
        """Select neighborhood arm via Sow et al. (JAIR'26) sliding-window alpha-UCB."""
        self._dymab_iter += 1
        alpha = 0.65 / max(1.0, math.log(self._dymab_iter + 2.0))
        best_h = self.DYMAB_ARMS[0]
        best_sc = -1e18
        for h in self.DYMAB_ARMS:
            q = self._dymab_q[h]
            mu = (sum(q) / len(q)) if q else 0.5
            bonus = math.sqrt(2.0 * alpha / (self._dymab_n[h] + 1.0))
            if mu + bonus > best_sc:
                best_sc = mu + bonus
                best_h = h
        return best_h

    def observe_dymab_reward(self, arm: str, reward: float) -> None:
        """Push latest non-stationary reward into sliding window `Q_i` of size `W`."""
        if arm in self._dymab_q:
            self._dymab_q[arm].append(float(max(0.0, min(2.0, reward))))
            self._dymab_n[arm] += 1


def load_v18_synthesis_policy(
    path: Optional[str | Path] = None,
    *,
    pgmpo_ckpt: Optional[str | Path] = None,
    max_rust_verifications: int = 4,
    router_mode: str = "learned",
    ranker_mode: str = "pairwise",
    enable_restraint: bool = True,
    split_search: bool = False,
    candidate_mode: str = "full",
    hysteresis_mode: str = "default",
    allow_missing_pgmpo: bool = False,
    allow_v17_fallback: bool = False,
) -> V18FrontierSynthesisPolicy:
    """Load `V18FrontierSynthesisPolicy` combining V17 checkpoint weights and pure-NumPy PGMPO."""
    v17_path = Path(path or DEFAULT_V17_CKPT)
    base_v17 = load_v17_window_policy(
        path=v17_path,
        router_mode=router_mode,
        ranker_mode=ranker_mode,
        enable_restraint=enable_restraint,
        allow_fallback=allow_v17_fallback,
    )
    v17_sha = file_sha256(v17_path) if v17_path.is_file() else None
    return V18FrontierSynthesisPolicy(
        weights=base_v17.w,
        pgmpo_ckpt=pgmpo_ckpt,
        router_mode=router_mode,
        ranker_mode=ranker_mode,
        enable_restraint=enable_restraint,
        max_rust_verifications=max_rust_verifications,
        split_search=split_search,
        candidate_mode=candidate_mode,
        hysteresis_mode=hysteresis_mode,
        allow_missing_pgmpo=allow_missing_pgmpo,
        v17_checkpoint_sha256=v17_sha,
    )


def _eval_op_tardiness(snapshot: NativeSnapshot, ev_res: EvalResult) -> float:
    """Compute sum of positive operation tardiness steps from an evaluated rollout."""
    if not ev_res.valid:
        return 0.0
    total = 0.0
    for op in snapshot.operations:
        if op.due_step is not None and op.op_id in ev_res.op_end:
            total += float(max(0, int(ev_res.op_end[op.op_id]) - int(op.due_step)))
    return total


def generate_v18_synthesis_candidates(
    policy_owner: Any,
    network: Network,
    snapshot: NativeSnapshot,
    clf: CommitmentClassification,
    evaluator: CVCREvaluator,
    surr: Any,
    base_assign: dict[str, str],
    base_bays: dict[str, str],
    base_crews: dict[str, int],
    base_eval: EvalResult,
    base_prio: dict[str, int],
    healthy_robots: list[str],
    bay_delta: dict[str, float],
    tw_seed: dict[tuple[str, str], int],
    v18_pol: V18FrontierSynthesisPolicy,
    deadline: float,
    budget_ms: float,
    now_fn: Any,
) -> tuple[list[V15Candidate], np.ndarray, dict[str, Any]]:
    """Rank joint scheduling candidates by surrogate cost and verify the shortlist."""
    g_feats = extract_v15_global_features(snapshot, base_eval, clf, bay_delta)
    cutoff_base = int(snapshot.tick) + 45 if tw_seed else 0
    tw_active = dict(tw_seed) if tw_seed else {}

    has_res_win = any(bool(getattr(op, "resource_windows", ())) for op in snapshot.operations)
    fb_snap = getattr(snapshot, "execution_feedback", None)
    has_exec_drift = bool(fb_snap is not None and not fb_snap.is_empty())
    # Commit full `proposed_task_sequences` (`h=2`) on ALL unwindowed regimes (including P4.5 drift, matching V7_CALM!),
    # while keeping flexible soft-pin (`h=0/1`) under discrete resource windows!
    use_seq_commit = not has_res_win
    default_h = 2 if use_seq_commit else 0

    pair = select_coupled_op_pair(network, snapshot, base_eval, base_assign, base_prio) if has_res_win else None
    op_a = pair[0] if pair is not None else ""
    op_b = pair[1] if pair is not None else ""
    pair_meta = pair[2] if pair is not None else {}
    sim = getattr(policy_owner, "_active_sim_ref", None)

    cands: list[V15Candidate] = []
    seen_digests: set[str] = set()

    def _add_verified_cand(
        cand_id: str,
        kind: str,
        assign: dict[str, str],
        bays: dict[str, str],
        crews: dict[str, int],
        prio: dict[str, int],
        ev: EvalResult,
        tw: dict[tuple[str, str], int],
        cut: int,
        horizon: int,
        rel_tids_t: tuple[str, ...],
        cliff: bool,
        radius_used: int = 999,
        live_repairs: Optional[dict[str, dict[str, Any]]] = None,
        bottleneck_locks: Optional[dict[str, list[tuple[str, int, int]]]] = None,
    ) -> Optional[V15Candidate]:
        rem_now = max(0.0, min(1.0, (deadline - now_fn()) / max(0.01, float(budget_ms) / 1000.0)))
        feat, ds = _build_candidate_features(
            kind,
            ev,
            base_eval,
            assign,
            base_assign,
            bays,
            crews,
            snapshot,
            horizon,
            cliff,
            v18_pol.family_ucb_bonus(kind),
            rem_now,
            g_feats,
        )
        c = V15Candidate(
            cand_id=cand_id,
            source_kind=kind,
            assignment=dict(assign),
            bay_overrides=dict(bays),
            crew_overrides=dict(crews),
            op_priority=dict(prio),
            eval_res=ev,
            traffic_weights=dict(tw),
            traffic_cutoff=int(cut),
            guide_routes={},
            commit_seq_horizon=int(horizon),
            destroy_tids=rel_tids_t,
            cliff_saved=bool(cliff),
            det_score=float(ds),
            features=feat,
            radius_used=int(radius_used),
            live_repairs=live_repairs,
            bottleneck_locks=bottleneck_locks,
        )
        dig = compute_action_diff_digest(c, base_assign, base_bays, base_crews, base_prio, live_repairs=live_repairs)
        c.action_diff_digest = dig
        if op_a and op_b:
            c.pairwise_features = compute_pairwise_payoff(
                network,
                snapshot,
                op_a,
                op_b,
                ev,
                base_eval,
                base_assign,
                c.assignment,
                base_prio,
                c.op_priority,
                c.bay_overrides,
                c.crew_overrides,
                pair_meta=pair_meta,
                reversible=True,
            )
        else:
            c.pairwise_features = np.zeros((V17_PAIRWISE_DIM,), dtype=np.float32)
        if dig in seen_digests:
            return None
        seen_digests.add(dig)
        cands.append(c)
        return c

    # C0: Keep incumbent anchor (already evaluated -> 0 extra Rust calls!)
    _add_verified_cand(
        "C0_keep",
        "keep_incumbent",
        base_assign,
        base_bays,
        base_crews,
        base_prio,
        base_eval,
        tw_active,
        cutoff_base,
        default_h,
        (),
        False,
    )

    rel_all = sorted(clf.releasable_tasks)
    rel_set = set(rel_all)
    if len(healthy_robots) < 2 or not rel_all:
        return cands, g_feats, {"mode": "v18_trivial", "rust_verifications": 0}

    tick_now = int(snapshot.tick)
    snap_by_id = {t.task_id: t for t in snapshot.tasks}
    has_unassigned_urgent = any(
        (t.assigned_agv is None or t.status.lower() == "pending") and t.is_urgent for t in snapshot.tasks
    )
    is_opening = tick_now <= 4
    is_rescue_tick = bool(clf.disturbed_robots) or has_unassigned_urgent

    w_surr = WindowAwareKittingSurrogate(surr, evaluator, snapshot)
    fixed_assign = {tid: aid for tid, aid in base_assign.items() if tid not in rel_set}

    # Causal fresh-operation set: newly released / unassigned / fault-affected ops get zero-friction bay co-location,
    # while already-dispatched healthy operations keep their bays locked (eliminates mid-episode bay thrashing!).
    fresh_op_ids: set[str] = set()
    for op in snapshot.operations:
        if op.started or any(kt in clf.frozen_tasks for kt in op.kit_task_ids):
            continue
        vis_kits = [snap_by_id[kt] for kt in op.kit_task_ids if kt in snap_by_id and snap_by_id[kt].status.lower() != "completed"]
        if not vis_kits:
            if is_opening:
                fresh_op_ids.add(op.op_id)
            continue
        if is_opening or any(
            (t.assigned_agv is None or t.status.lower() == "pending" or t.assigned_agv in clf.disturbed_robots)
            for t in vis_kits
        ):
            fresh_op_ids.add(op.op_id)

    # Zero-churn partition: separate newly pending / fault-disturbed tasks from healthy already-assigned tasks!
    pending_or_dist = [
        tid
        for tid in rel_all
        if (
            snap_by_id[tid].assigned_agv is None
            or snap_by_id[tid].status.lower() == "pending"
            or snap_by_id[tid].assigned_agv in clf.disturbed_robots
            or snap_by_id[tid].assigned_agv not in healthy_robots
        )
    ]
    pending_or_dist_set = set(pending_or_dist)
    healthy_anchor_assign = {tid: aid for tid, aid in base_assign.items() if tid not in pending_or_dist_set}

    cand_mode = getattr(v18_pol, "candidate_mode", "full")
    hys_mode = getattr(v18_pol, "hysteresis_mode", "default")

    def _hysteresis_req(
        assign_c: dict[str, str],
        bays_c: dict[str, str],
        cliff: bool = False,
        *,
        ev_c: Optional[EvalResult] = None,
        for_surrogate: bool = False,
    ) -> float:
        """Compute V7-inspired non-rescue stability margin preventing mid-episode churn & bay thrashing."""
        if hys_mode == "none" or is_opening or cliff:
            return 0.0
        n_non_rescue = 0
        for tid, nr in assign_c.items():
            t_orig = snap_by_id.get(tid)
            if (
                t_orig is not None
                and t_orig.status.lower() == "assigned"
                and t_orig.assigned_agv is not None
                and t_orig.assigned_agv != nr
                and t_orig.assigned_agv not in clf.disturbed_robots
            ):
                n_non_rescue += 1
        # Freshly released ops co-locate at zero hysteresis penalty; already-dispatched ops resist thrashing!
        n_stale_bay_flips = sum(
            1 for oid, b in bays_c.items() if base_bays.get(oid) != b and oid not in fresh_op_ids
        )
        if n_non_rescue == 0 and n_stale_bay_flips == 0:
            return 0.0
        base_m = 28.0 if has_exec_drift else 24.0
        raw_req = float(base_m + 18.0 * n_non_rescue + 22.0 * n_stale_bay_flips)
        if hys_mode == "adaptive":
            if for_surrogate:
                return 0.5 * raw_req
            if ev_c is not None and ev_c.valid and base_eval.valid:
                op_td_gain = max(0.0, _eval_op_tardiness(snapshot, base_eval) - _eval_op_tardiness(snapshot, ev_c))
                scale = max(0.15, 1.0 - min(0.85, (2.0 * op_td_gain) / 60.0))
                return raw_req * scale
        return raw_req

    # Paradigm 1: Wu EJOR'26 Semi-Circle + ICRA'26 Switch-Cost Bay Co-Location strictly on `fresh_op_ids`!
    if cand_mode != "neural_only":
        bays_semi, km_prio = compute_semicircle_kitting_bays(
            snapshot,
            network,
            clf,
            surr,
            base_bays,
            bay_delta,
            eligible_op_ids=fresh_op_ids,
            spread_empty_kits=is_opening,
        )
        has_flex_bay = any(op.op_id in fresh_op_ids and len(op.eligible_bays) > 1 for op in snapshot.operations)
        # Under P4.5 execution drift on unwindowed topologies, preserve incumbent bays after opening to avoid blocked alleys
        allow_bay_divert = has_flex_bay and (is_opening or has_res_win or not has_exec_drift)
        prio_co = km_prio if (has_res_win or has_flex_bay) else base_prio
        # Keep pure Wu'26 semi-circle bays intact (no bridge-crossing surrogate distortion); polish crews only!
        bays_opt = dict(bays_semi) if allow_bay_divert else dict(base_bays)
        crews_opt = dict(base_crews)
        if allow_bay_divert and any(len(op.eligible_crews) > 1 for op in snapshot.operations if op.op_id in fresh_op_ids):
            best_c_sc = w_surr.score(base_assign, prio_co, bays_opt, crews_opt)
            for op in snapshot.operations:
                if op.started or op.op_id not in fresh_op_ids or len(op.eligible_crews) <= 1:
                    continue
                cur_c = crews_opt.get(op.op_id, op.crew_id)
                for cand_c in op.eligible_crews:
                    if int(cand_c) == int(cur_c):
                        continue
                    crews_opt[op.op_id] = int(cand_c)
                    sc = w_surr.score(base_assign, prio_co, bays_opt, crews_opt)
                    if sc + 1e-3 < best_c_sc:
                        best_c_sc = sc
                        cur_c = int(cand_c)
                    else:
                        crews_opt[op.op_id] = cur_c
    else:
        bays_semi, km_prio = dict(base_bays), dict(base_prio)
        has_flex_bay = False
        allow_bay_divert = False
        prio_co = dict(base_prio)
        bays_opt = dict(base_bays)
        crews_opt = dict(base_crews)

    bays_inc_opt, crews_inc_opt = dict(base_bays), dict(base_crews)

    # Candidate hypothesis pool & funnel counters
    pool: list[tuple[float, str, str, str, dict[str, str], dict[str, str], dict[str, int], dict[str, int], tuple[str, ...], int]] = []
    seen_hyps: set[tuple[Any, ...]] = set()
    funnel_by_fam: dict[str, dict[str, int]] = {}

    def _bump_funnel(fam: str, metric: str, inc: int = 1) -> None:
        bucket = funnel_by_fam.setdefault(
            fam,
            {
                "offered_raw": 0,
                "offered": 0,
                "shortlisted": 0,
                "b2_evaluated": 0,
                "b2_valid": 0,
                "not_better_than_keep": 0,
                "hysteresis_rejected": 0,
                "hysteresis_blocked_winning": 0,
                "admitted_to_ranker": 0,
            },
        )
        bucket[metric] += inc

    def _offer(
        group_tag: str,
        cid: str,
        kind: str,
        assign: dict[str, str],
        bays: dict[str, str],
        crews: dict[str, int],
        prio: dict[str, int],
        rel_sub: list[str] | tuple[str, ...],
        rad: int = 999,
    ) -> float:
        sig = (
            tuple(sorted(assign.items())),
            tuple(sorted(bays.items())),
            tuple(sorted(crews.items())),
            tuple(sorted(prio.items())),
        )
        if getattr(v18_pol, "split_search", False):
            changes_assignment = assign != base_assign
            changes_production = bays != base_bays or crews != base_crews or prio != base_prio
            if changes_assignment and changes_production:
                # Ablation: assignment moves keep the incumbent production plan. Production-only moves are offered separately.
                bays, crews, prio = dict(base_bays), dict(base_crews), dict(base_prio)
                sig = (
                    tuple(sorted(assign.items())),
                    tuple(sorted(bays.items())),
                    tuple(sorted(crews.items())),
                    tuple(sorted(prio.items())),
                )
        sc = w_surr.score(assign, prio, bays, crews) + (
            0.0 if group_tag == "keep" else _hysteresis_req(assign, bays, for_surrogate=True)
        )
        if group_tag != "keep":
            _bump_funnel(group_tag, "offered_raw")
        if sig in seen_hyps:
            return sc
        seen_hyps.add(sig)
        if group_tag != "keep":
            _bump_funnel(group_tag, "offered")
        pool.append((sc, group_tag, cid, kind, dict(assign), dict(bays), dict(crews), dict(prio), tuple(rel_sub), rad))
        return sc

    # Incumbent baseline signature so we never re-verify identical incumbent
    base_surr_sc = _offer("keep", "C0_keep_ref", "keep_incumbent", base_assign, base_bays, base_crews, base_prio, ())

    if cand_mode != "neural_only":
        # Hypothesis Z0/Z1: ZERO-CHURN incremental insertion (inserts ONLY newly pending/disturbed tasks around healthy plan!)
        if pending_or_dist:
            zc_inc = surr.regret_k_assignment(healthy_robots, healthy_anchor_assign, pending_or_dist, op_bays=bays_inc_opt)
            _offer("zero_churn", "H_zc_inc", "balans_adaptive_destroy", zc_inc, bays_inc_opt, crews_inc_opt, base_prio, pending_or_dist, len(pending_or_dist))
            if allow_bay_divert and (bays_opt != base_bays or crews_opt != base_crews):
                zc_semi = surr.regret_k_assignment(healthy_robots, healthy_anchor_assign, pending_or_dist, op_bays=bays_opt)
                _offer("zero_churn_semi", "H_zc_semi_km", "fast_detour_regret", zc_semi, bays_opt, crews_opt, prio_co, pending_or_dist, len(pending_or_dist))
                if prio_co != base_prio:
                    _offer("zero_churn_semi", "H_zc_semi_base", "fast_detour_regret", zc_semi, bays_opt, crews_opt, base_prio, pending_or_dist, len(pending_or_dist))

        # Hypothesis B1/B2: Full-fleet Regret-2 (incumbent bays + Wu'26 semi-circle bays)
        reg_inc = surr.regret_k_assignment(healthy_robots, fixed_assign, rel_all, op_bays=bays_inc_opt)
        _offer("inc_bay", "H_reg_inc", "balans_adaptive_destroy", reg_inc, bays_inc_opt, crews_inc_opt, base_prio, rel_all)
        if allow_bay_divert and (bays_opt != base_bays or crews_opt != base_crews):
            _offer("co_bay_semi", "H_semi_keepa_km", "fast_detour_regret", base_assign, bays_opt, crews_opt, prio_co, ())
            reg_semi = surr.regret_k_assignment(healthy_robots, fixed_assign, rel_all, op_bays=bays_opt)
            _offer("co_bay_semi", "H_semi_direct_km", "fast_detour_regret", reg_semi, bays_opt, crews_opt, prio_co, rel_all)
        if km_prio != base_prio:
            _offer("inc_bay", "H_km_base", "v10_crit_ci_balance", base_assign, base_bays, base_crews, km_prio, ())

        # In a sparse opening, test one bay move without coupling it to unrelated moves.
        if is_opening and has_res_win and allow_bay_divert and len(rel_all) < len(healthy_robots):
            visible_gains = []
            for op in snapshot.operations:
                oid = op.op_id
                old_bay = base_bays[oid]
                new_bay = bays_opt[oid]
                if old_bay == new_bay:
                    continue
                gain = sum(
                    surr.tt(snap_by_id[tid].pickup, old_bay) - surr.tt(snap_by_id[tid].pickup, new_bay)
                    for tid in op.kit_task_ids if tid in rel_set
                )
                if gain > 0:
                    visible_gains.append((gain, oid, new_bay))
            if visible_gains:
                _, oid, new_bay = max(visible_gains)
                partial_bays = dict(base_bays)
                partial_bays[oid] = new_bay
                _offer(
                    "opening_partial", "H_opening_partial", "fast_detour_regret",
                    base_assign, partial_bays, base_crews, base_prio, (),
                )
                if not base_eval.valid:
                    repaired_assign = surr.regret_k_assignment(
                        healthy_robots, fixed_assign, rel_all, op_bays=partial_bays,
                    )
                    _offer(
                        "opening_repair", "H_opening_partial_regret", "fast_detour_regret",
                        repaired_assign, partial_bays, base_crews, base_prio, rel_all,
                    )

    # Paradigm 2: Choi IJCAI-26 Pure-NumPy Bit-Conditioned Multi-Policy (`C=8` fast cached-encoder rollouts)
    if cand_mode != "no_neural" and v18_pol.pgmpo.ready:
        active_bays_for_nn = bays_opt if (allow_bay_divert and cand_mode != "neural_only") else bays_inc_opt
        active_crews_for_nn = crews_opt if (allow_bay_divert and cand_mode != "neural_only") else crews_inc_opt
        active_prio_for_nn = prio_co if (allow_bay_divert and cand_mode != "neural_only") else base_prio
        nn_assigns = v18_pol.pgmpo.generate_assignments(
            snapshot, network, clf, healthy_robots, fixed_assign, active_bays_for_nn, surr
        )
        for idx_nn, nn_a in enumerate(nn_assigns):
            _offer(
                "pgmpo",
                f"H_pgmpo_c{idx_nn}",
                "v10_crit_ci_balance",
                nn_a,
                active_bays_for_nn,
                active_crews_for_nn,
                active_prio_for_nn,
                rel_all,
            )
            # Also project neural choices strictly onto `pending_or_dist` for zero-churn mid-episode dispatch!
            if (not is_opening) and pending_or_dist and len(pending_or_dist) < len(rel_all):
                zc_nn = dict(healthy_anchor_assign)
                for tid in pending_or_dist:
                    zc_nn[tid] = nn_a.get(tid, base_assign.get(tid, healthy_robots[0]))
                _offer(
                    "pgmpo_zc",
                    f"H_pgmpo_zc_c{idx_nn}",
                    "v10_crit_ci_balance",
                    zc_nn,
                    active_bays_for_nn,
                    active_crews_for_nn,
                    active_prio_for_nn,
                    pending_or_dist,
                    len(pending_or_dist),
                )

    dymab_arm = "disabled"
    if cand_mode != "neural_only":
        # Paradigm 3: Parimi MIT'26 Precedence-Closure Destroy + Regret-2 Reinsertion (LOW_SLACK)
        dest = _mit26_precedence_closure_destroy(snapshot, rel_all, base_eval, base_assign, 0)
        if dest:
            dest_set = set(dest)
            b_use = bays_semi if allow_bay_divert else bays_inc_opt
            c_use = crews_opt if allow_bay_divert else crews_inc_opt
            p_use = prio_co if allow_bay_divert else base_prio
            mit_a = surr.regret_k_assignment(
                healthy_robots,
                {t: a for t, a in base_assign.items() if t not in dest_set},
                dest,
                op_bays=b_use,
                rng=v18_pol._rng,
                noise=0.02,
            )
            _offer("mit_lns", "H_mit_low_slack", "balans_adaptive_destroy", mit_a, b_use, c_use, p_use, dest, len(dest))

        # Paradigm 4: Sow JAIR'26 Non-Stationary Sliding-Window DyMAB Adaptive Destroy
        dymab_arm = v18_pol.select_dymab_arm()
        rad_k = v18_pol.select_destroy_radius()
        if dymab_arm == "H_rand":
            sub = list(rel_all)
            v18_pol._rng.shuffle(sub)
            sub = sub[: min(len(sub), max(4, rad_k))]
        elif dymab_arm == "H_sync":
            op_map = {op.op_id: op for op in snapshot.operations}
            sub = sorted(
                rel_all,
                key=lambda tid: (
                    -max(0, int(base_eval.op_end.get(snap_by_id[tid].kit_op_id or "", 0)) - int(op_map[snap_by_id[tid].kit_op_id].due_step))
                    if (snap_by_id[tid].kit_op_id in op_map and base_eval.valid)
                    else 0,
                    tid,
                ),
            )[: min(len(rel_all), max(4, rad_k))]
        else:
            sub = sorted(
                rel_all,
                key=lambda tid: (-float(bay_delta.get(bays_opt.get(snap_by_id[tid].kit_op_id or "", snap_by_id[tid].dropoff), 0.0)), tid),
            )[: min(len(rel_all), max(4, rad_k))]
        if sub:
            sub_set = set(sub)
            b_dy = bays_semi if allow_bay_divert else bays_inc_opt
            c_dy = crews_opt if allow_bay_divert else crews_inc_opt
            p_dy = prio_co if allow_bay_divert else base_prio
            dy_a = surr.regret_k_assignment(
                healthy_robots,
                {t: a for t, a in base_assign.items() if t not in sub_set},
                sub,
                op_bays=b_dy,
                rng=v18_pol._rng,
                noise=0.04,
            )
            _offer("dymab", f"H_dymab_{dymab_arm}", "balans_adaptive_destroy", dy_a, b_dy, c_dy, p_dy, sub, len(sub))

        # Polish a bounded shortlist seed when the active task count is small.
        non_keep = [h for h in pool if h[1] != "keep"]
        non_keep.sort(key=lambda x: x[0])
        if non_keep and len(rel_all) <= 20 and non_keep[0][0] < base_surr_sc + 40.0:
            seed_h = non_keep[0]
            pol_a, _ = _surrogate_polish_assign(
                w_surr, rel_all, healthy_robots, seed_h[4], seed_h[5], seed_h[6], seed_h[7],
                max_passes=1, max_focus_tasks=8, do_swaps=False,
            )
            _offer("polish_0", "H_polish_0", "fast_detour_regret", pol_a, seed_h[5], seed_h[6], seed_h[7], rel_all, len(rel_all))

    # Synthetic benchmark charge for candidate generation.
    if is_virtual_clock_active():
        _cvcr_record_work_ms(0.20)

    # Verify a bounded shortlist before running the separate window protection stage.
    candidates_ranked = [h for h in pool if h[1] != "keep"]
    candidates_ranked.sort(key=lambda x: x[0])

    # Early exit when mid-episode incumbent ALREADY dominates every surrogate hypothesis cleanly
    if not is_opening and not is_rescue_tick and candidates_ranked and candidates_ranked[0][0] > base_surr_sc + 35.0:
        k_cap = 1
    else:
        k_cap = v18_pol.max_rust_verifications
        if is_opening or is_rescue_tick:
            k_cap = min(5, k_cap + 1)

    selected_hyps: list[Any] = []
    used_sigs: set[int] = set()
    used_groups: set[str] = set()
    for group in ("opening_partial", "opening_repair"):
        for idx_h, h in enumerate(candidates_ranked):
            if h[1] == group:
                selected_hyps.append(h)
                used_sigs.add(idx_h)
                used_groups.add(h[1])
                break
    for idx_h, h in enumerate(candidates_ranked):
        if len(selected_hyps) >= k_cap:
            break
        if len(selected_hyps) == 0 or h[1] not in used_groups:
            selected_hyps.append(h)
            used_sigs.add(idx_h)
            used_groups.add(h[1])
    for idx_h, h in enumerate(candidates_ranked):
        if len(selected_hyps) >= k_cap:
            break
        if idx_h not in used_sigs:
            selected_hyps.append(h)
            used_sigs.add(idx_h)

    for _, grp_s, *_ in selected_hyps:
        _bump_funnel(grp_s, "shortlisted")

    evaluator.set_traffic_weights(tw_active or None, cutoff_step=cutoff_base)
    evaluator.set_incumbent(base_assign, base_eval, op_priority=base_prio, op_bay_overrides=base_bays)
    calls_before_verification = evaluator.calls
    best_rust_cand: Optional[V15Candidate] = cands[0]
    base_obj_val = float(base_eval.objective) if base_eval.valid else 1e12
    best_valid_neural_b2_obj: Optional[float] = None
    best_valid_heuristic_b2_obj: Optional[float] = None

    for _, grp, cid, kind, cand_a, cand_b, cand_c, cand_p, rel_t, rad in selected_hyps:
        if not policy_owner._has_budget(deadline, evaluator, min_ms=2.0):
            break
        ev_c = evaluator.evaluate(
            cand_a,
            op_priority=cand_p,
            op_bay_overrides=cand_b,
            op_crew_overrides=cand_c,
        )
        _bump_funnel(grp, "b2_evaluated")
        if ev_c.valid:
            _bump_funnel(grp, "b2_valid")
            obj_val = float(ev_c.objective)
            if grp in ("pgmpo", "pgmpo_zc"):
                if best_valid_neural_b2_obj is None or obj_val < best_valid_neural_b2_obj:
                    best_valid_neural_b2_obj = obj_val
            else:
                if best_valid_heuristic_b2_obj is None or obj_val < best_valid_heuristic_b2_obj:
                    best_valid_heuristic_b2_obj = obj_val
        req_m = _hysteresis_req(cand_a, cand_b, cliff=False, ev_c=ev_c)
        beats_keep = (not base_eval.valid) or float(ev_c.objective) < base_obj_val
        passes_hys = ev_c.valid and ((not base_eval.valid) or float(ev_c.objective) + req_m < base_obj_val)
        if grp == "dymab" and base_eval.valid and ev_c.valid:
            gain_norm = max(0.0, (base_obj_val - float(ev_c.objective)) / 80.0)
            v18_pol.observe_dymab_reward(dymab_arm, gain_norm)
        if not passes_hys:
            if ev_c.valid:
                if beats_keep and req_m > 0.0:
                    _bump_funnel(grp, "hysteresis_rejected")
                    _bump_funnel(grp, "hysteresis_blocked_winning")
                else:
                    _bump_funnel(grp, "not_better_than_keep")
            continue
        added = _add_verified_cand(
            cid,
            kind,
            cand_a,
            cand_b,
            cand_c,
            cand_p,
            ev_c,
            tw_active,
            cutoff_base,
            default_h,
            rel_t,
            False,
            radius_used=rad,
        )
        if added is not None:
            _bump_funnel(grp, "admitted_to_ranker")
            if best_rust_cand is None or not best_rust_cand.eval_res.valid or ev_c.objective < best_rust_cand.eval_res.objective:
                best_rust_cand = added
                evaluator.set_incumbent(cand_a, ev_c, op_priority=cand_p, op_bay_overrides=cand_b)

    # Conditional V17 Dual-Op Bridge-Order & Resource-Window Cliff Protection (ONLY when real window cliff exists!)
    anchor = best_rust_cand if (best_rust_cand is not None and best_rust_cand.eval_res.valid) else cands[0]
    danger_ops = _find_danger_cliff_ops(snapshot, anchor.eval_res, bay_delta) if has_res_win else []
    dual_added = 0
    if danger_ops and pair is not None and policy_owner._has_budget(deadline, evaluator, min_ms=2.5):
        for cid_d, ldr, yld in (("D1_protect_A", op_a, op_b), ("D2_protect_B", op_b, op_a)):
            if not policy_owner._has_budget(deadline, evaluator, min_ms=2.0):
                break
            bp_plan = plan_bridge_order_control(
                network,
                snapshot,
                evaluator,
                anchor.assignment,
                anchor.op_priority,
                ldr,
                yld,
                sim=sim,
                base_bays=anchor.bay_overrides,
                base_crews=anchor.crew_overrides,
            )
            if bp_plan.eligible and bp_plan.eval_res is not None and bp_plan.eval_res.valid:
                if float(bp_plan.eval_res.objective) + 8.0 < float(anchor.eval_res.objective):
                    c_d = _add_verified_cand(
                        cid_d,
                        "selective_cliff_rescue",
                        anchor.assignment,
                        anchor.bay_overrides,
                        anchor.crew_overrides,
                        bp_plan.op_priority,
                        bp_plan.eval_res,
                        tw_active,
                        cutoff_base,
                        default_h,
                        anchor.destroy_tids,
                        True,
                        radius_used=anchor.radius_used,
                        live_repairs=bp_plan.live_repairs,
                        bottleneck_locks=bp_plan.bottleneck_locks,
                    )
                    if c_d is not None:
                        dual_added += 1
                        anchor = c_d

    # Zero-cost C_softpin_h1 under windowed/drift regimes (0 extra Rust calls!)
    if (not use_seq_commit) and anchor.cand_id != "C0_keep" and anchor.eval_res.valid and anchor.commit_seq_horizon == 0:
        _add_verified_cand(
            "C_v18_softpin_h1",
            "v10_crit_ci_balance",
            anchor.assignment,
            anchor.bay_overrides,
            anchor.crew_overrides,
            anchor.op_priority,
            anchor.eval_res,
            anchor.traffic_weights,
            anchor.traffic_cutoff,
            1,
            anchor.destroy_tids,
            anchor.cliff_saved,
            radius_used=anchor.radius_used,
            live_repairs=anchor.live_repairs,
            bottleneck_locks=anchor.bottleneck_locks,
        )

    totals = {
        k: sum(b[k] for b in funnel_by_fam.values())
        for k in (
            "offered_raw",
            "offered",
            "shortlisted",
            "b2_evaluated",
            "b2_valid",
            "not_better_than_keep",
            "hysteresis_rejected",
            "hysteresis_blocked_winning",
            "admitted_to_ranker",
        )
    }
    funnel = {
        "candidate_mode": cand_mode,
        "hysteresis_mode": hys_mode,
        "by_family": funnel_by_fam,
        "totals": totals,
        "base_valid": bool(base_eval.valid),
        "base_obj": round(float(base_eval.objective), 4) if base_eval.valid else None,
        "best_valid_neural_b2_obj": round(best_valid_neural_b2_obj, 4) if best_valid_neural_b2_obj is not None else None,
        "best_valid_heuristic_b2_obj": round(best_valid_heuristic_b2_obj, 4) if best_valid_heuristic_b2_obj is not None else None,
    }
    v18_pol.last_funnel = funnel
    if not hasattr(v18_pol, "funnel_history"):
        v18_pol.funnel_history = []
    v18_pol.funnel_history.append(funnel)

    meta = {
        "mode": "v18_frontier_synthesis",
        "candidate_mode": cand_mode,
        "hysteresis_mode": hys_mode,
        "surrogate_hyps": len(pool),
        "rust_verifications": evaluator.calls - calls_before_verification,
        "dual_added": dual_added,
        "dymab_arm": dymab_arm,
        "base_surr_score": round(float(base_surr_sc), 2),
        "has_flex_bay": has_flex_bay,
        "allow_bay_divert": allow_bay_divert,
        "pgmpo_active": v18_pol.pgmpo.ready,
        "funnel": funnel,
    }
    return cands, g_feats, meta


class V18AdaptiveController(V17AdaptiveController):
    """Retain V17 control triggers and route portfolio replans to V18."""

    def __init__(
        self,
        *,
        epoch_interval: int = 16,
        stall_threshold: int = 20,
        policy: Optional[V18FrontierSynthesisPolicy] = None,
        trigger: Optional[Any] = None,
        profile: str = "quality",
        enable_restraint: Optional[bool] = None,
    ) -> None:
        """Initialize the restraint controller and V18 scheduling policy."""
        pol = policy if policy is not None else load_v18_synthesis_policy()
        super().__init__(
            epoch_interval=epoch_interval,
            stall_threshold=stall_threshold,
            policy=pol,
            trigger=trigger,
            profile=profile,
            enable_restraint=enable_restraint,
        )

    def observe(self, env: Any, base_obs: np.ndarray) -> np.ndarray:
        """Compute the observation required by the active controller mode."""
        if any(op.resource_windows for op in env.tracker.operations):
            # Windowed budget selection uses the trigger features and the legal action mask.
            return env.get_expert_v9_obs(base_obs)
        if self.locked_mode in ("V7_CALM", "V10_GUIDED"):
            return env.get_anytime_obs(base_obs)
        return env.get_expert_v12_obs(base_obs)

    def select_action(self, obs_v12: np.ndarray, info: dict[str, Any], sim: Any = None) -> int:
        """Select action via V17 restraint + surgical safety, keeping `active_subscope=None` so V18 solver executes."""
        act = super().select_action(obs_v12, info, sim=sim)
        # Use the latched mode to trigger replans while keeping V18 as the solver.
        if isinstance(self.policy, V16FrontierPolicy):
            self.policy.active_subscope = None
        return act
