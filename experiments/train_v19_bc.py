"""Multi-expert B2 dataset collection, CUDA BC training, and validation gate for V19."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.baselines_2026 import IJCAI26PGMPOProposer
from amr_sched.baselines.policy_registry import (
    _COST_COLUMNS,
    _IJCAI26_CKPT,
    RuleV10PhysicalProposer,
    load_internal_cvcr_policy,
    run_discrete_episode,
)
from amr_sched.baselines.spga_cor2026 import SPGAConfig, run_spga_policy_proposal
from amr_sched.baselines.top_venue_baselines import DetourDeltaRAL26Proposer
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.snapshot import NativeSnapshot, NativeSnapshotCapture
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.rl import AMRReschedulingEnv, RuleV10PortfolioPolicy, compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.v19_hetero_constructor import (
    V19HeteroConstructorNet,
    V19HeteroConstructorProposer,
    export_v19_npz,
    graph_to_torch_tensors,
)
from amr_sched.rl.v19_hetero_graph import V19HeteroGraph, extract_v19_hetero_graph
from tools.benchmarks.compare_workshop_algorithms import (
    REGIMES as WORKSHOP_REGIMES,
    build_track1_scenario,
    build_track2_scenario,
)
from experiments.v18_confirmation_scenarios import BLOCKAGE as TOPO4_BLOCKAGE, LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES, build_confirmation_scenario

DEFAULT_DATASET_PATH = Path(
    str(ROOT / "out" / "v19_bc" / "v19_bc_dataset.pt")
)
CKPT_DIR = ROOT / "amr_sched" / "checkpoints" / "v19"
OUT_DIR = ROOT / "out" / "g2_bc"

FORBIDDEN_BLIND_SEEDS: frozenset[int] = frozenset(range(17101, 17317)) | frozenset(range(901, 907))
FORBIDDEN_DEV_SEEDS: frozenset[int] = (
    frozenset({101, 202, 303, 404, 505, 606})
    | frozenset(range(14101, 14617))
    | frozenset(range(15101, 15717))
    | frozenset(range(15801, 15817))
)

TRAIN_4TOPO_SEEDS: tuple[int, ...] = (
    16101, 16102, 16103, 16104,
    16201, 16202, 16203, 16204,
    16301, 16302, 16303, 16304,
)
TRAIN_WORKSHOP_SEEDS: tuple[int, ...] = tuple(range(1001, 1016))  # 15 seeds * 4 regimes * 2 tracks = 120 eps

VAL_4TOPO_SEEDS: tuple[int, ...] = (16701, 16702, 16703, 16704)  # 4 * 4 layouts * 3 regimes = 48 eps
VAL_WORKSHOP_SEEDS: tuple[int, ...] = (1161, 1162, 1163)        # 3 * 4 regimes * 2 tracks = 24 eps


def assert_seed_quarantine(seeds: list[int] | tuple[int, ...], *, label: str) -> None:
    """Abort immediately if any dev/blind-holdout seed touches training or BC validation."""
    s_set = set(int(s) for s in seeds)
    bad_blind = sorted(s_set & FORBIDDEN_BLIND_SEEDS)
    if bad_blind:
        raise RuntimeError(f"QUARANTINE VIOLATION in {label}: blind seeds {bad_blind[:8]}")
    bad_dev = sorted(s_set & FORBIDDEN_DEV_SEEDS)
    if bad_dev:
        raise RuntimeError(f"QUARANTINE VIOLATION in {label}: dev/holdout seeds {bad_dev[:8]}")


def _build_4topo_episode_spec(layout: str, seed: int, regime: str) -> dict[str, Any]:
    sc_tuple = build_confirmation_scenario(layout, seed, regime)
    p2_cfg = P2PerturbationConfig(seed=seed * 17 + 5, **TOPO4_BLOCKAGE) if regime == "execution_blockage" else None
    return {
        "episode_key": f"topo4_{layout}_{regime}_{seed}",
        "cluster_key": f"topo4_{layout}_{seed}",
        "suite": "topo4",
        "layout": layout,
        "regime": regime,
        "seed": int(seed),
        "max_steps": 1000,
        "scenario_tuple": sc_tuple,
        "p2_cfg": p2_cfg,
    }


def _build_workshop_episode_spec(track: str, seed: int, regime: str) -> dict[str, Any]:
    sc_tuple, p2_cfg, max_steps = (
        build_track1_scenario(seed, regime) if track == "track1" else build_track2_scenario(seed, regime)
    )
    return {
        "episode_key": f"{track}_{regime}_{seed}",
        "cluster_key": f"{track}_{seed}",
        "suite": track,
        "layout": track,
        "regime": regime,
        "seed": int(seed),
        "max_steps": int(max_steps),
        "scenario_tuple": sc_tuple,
        "p2_cfg": p2_cfg,
    }


def build_train_specs() -> list[dict[str, Any]]:
    """Build quarantined BC training episode list (144 4-topo + 120 Track 1/2 = 264)."""
    assert_seed_quarantine(TRAIN_4TOPO_SEEDS, label="TRAIN_4TOPO_SEEDS")
    assert_seed_quarantine(TRAIN_WORKSHOP_SEEDS, label="TRAIN_WORKSHOP_SEEDS")
    specs: list[dict[str, Any]] = []
    for s in TRAIN_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                specs.append(_build_4topo_episode_spec(lay, s, reg))
    for s in TRAIN_WORKSHOP_SEEDS:
        for tr in ("track1", "track2"):
            for reg in WORKSHOP_REGIMES:
                specs.append(_build_workshop_episode_spec(tr, s, reg))
    return specs


def build_val_specs() -> list[dict[str, Any]]:
    """Build quarantined BC validation gate episode list (48 4-topo + 24 Track 1/2 = 72)."""
    assert_seed_quarantine(VAL_4TOPO_SEEDS, label="VAL_4TOPO_SEEDS")
    assert_seed_quarantine(VAL_WORKSHOP_SEEDS, label="VAL_WORKSHOP_SEEDS")
    specs: list[dict[str, Any]] = []
    for s in VAL_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                specs.append(_build_4topo_episode_spec(lay, s, reg))
    for s in VAL_WORKSHOP_SEEDS:
        for tr in ("track1", "track2"):
            for reg in WORKSHOP_REGIMES:
                specs.append(_build_workshop_episode_spec(tr, s, reg))
    return specs


def _cand_true_obj(
    ev_res: EvalResult,
    bays: dict[str, str],
    crews: dict[str, int],
    graph: V19HeteroGraph,
    snapshot: NativeSnapshot,
    task_orders: Optional[dict[str, list[str]]],
    lambda_change: float = 15.0,
) -> float:
    """Compute switching-aware B2 objective matching real `AMRReschedulingEnv` net_obj terms."""
    if not ev_res.valid:
        return float(ev_res.objective) + 5000.0
    changed_ops = sum(
        1
        for oid in graph.op_ids
        if bays.get(oid, graph.base_bays.get(oid)) != graph.base_bays.get(oid)
        or int(crews.get(oid, graph.base_crews.get(oid, 0))) != int(graph.base_crews.get(oid, 0))
    )
    qh = 0
    if task_orders:
        for r in snapshot.robots:
            if r.cur_task and r.robot_id not in graph.clf.disturbed_robots:
                seq = task_orders.get(r.robot_id)
                if seq and seq[0] != r.cur_task and r.cur_task in graph.clf.releasable_tasks:
                    qh += 1
    return float(ev_res.objective) + float(lambda_change) * float(changed_ops + qh)


def _derive_op_prio(snapshot: NativeSnapshot, seq_orders: Optional[dict[str, list[str]]] = None) -> dict[str, int]:
    """Build critical-chain op priorities optionally refined by task dispatch rank."""
    base = {
        op.op_id: max(int(snapshot.tick), int(op.due_step) - int(op.proc_steps))
        for op in snapshot.operations
    }
    if not seq_orders:
        return base
    task_op = {t.task_id: t.kit_op_id for t in snapshot.tasks if t.kit_op_id}
    rank_shift: dict[str, int] = {}
    for _, seq in seq_orders.items():
        for pos, tid in enumerate(seq):
            oid = task_op.get(tid)
            if oid and oid in base:
                rank_shift[oid] = min(rank_shift.get(oid, pos), pos)
    for oid, r_idx in rank_shift.items():
        base[oid] = int(base[oid] + 4 * r_idx)
    return base


def _polish_joint_b2(
    ev: CVCREvaluator,
    graph: V19HeteroGraph,
    snapshot: NativeSnapshot,
    best_assign: dict[str, str],
    best_bays: dict[str, str],
    best_crews: dict[str, int],
    best_orders: Optional[dict[str, list[str]]],
    best_eval: EvalResult,
    best_true_obj: float,
    max_polish_evals: int = 20,
) -> tuple[dict[str, str], dict[str, str], dict[str, int], Optional[dict[str, list[str]]], EvalResult, float]:
    """Run bounded 1-pass critical-chain 1-opt B2 polish across op bays/crews and task AGVs."""
    used_evals = 0
    # 1. Polish unfinished non-frozen multi-bay ops sorted by tardiness
    late_ops = sorted(
        graph.op_ids,
        key=lambda oid: -max(0, int(best_eval.op_end.get(oid, snapshot.tick)) - next(o.due_step for o in snapshot.operations if o.op_id == oid)),
    )
    for oid in late_ops:
        if used_evals >= max_polish_evals or oid in graph.frozen_ops:
            break
        oi = graph.op_ids.index(oid)
        if int(graph.op_bay_mask[oi].sum()) > 1:
            cur_b = best_bays[oid]
            for bi, ok in enumerate(graph.op_bay_mask[oi]):
                cand_b = graph.bay_ids[bi]
                if not ok or cand_b == cur_b or used_evals >= max_polish_evals:
                    continue
                nbays = dict(best_bays)
                nbays[oid] = cand_b
                op_prio = _derive_op_prio(snapshot, best_orders)
                ev_c = ev.evaluate(
                    best_assign, op_priority=op_prio, op_bay_overrides=nbays,
                    op_crew_overrides=best_crews, task_order_override=best_orders,
                )
                used_evals += 1
                tobj = _cand_true_obj(ev_c, nbays, best_crews, graph, snapshot, best_orders)
                if ev_c.valid and tobj + 1e-4 < best_true_obj:
                    best_bays, best_eval, best_true_obj = nbays, ev_c, tobj
        if int(graph.op_crew_mask[oi].sum()) > 1 and used_evals < max_polish_evals:
            cur_c = int(best_crews[oid])
            for ci, ok in enumerate(graph.op_crew_mask[oi]):
                cand_c = int(graph.crew_ids[ci])
                if not ok or cand_c == cur_c or used_evals >= max_polish_evals:
                    continue
                ncrews = dict(best_crews)
                ncrews[oid] = cand_c
                op_prio = _derive_op_prio(snapshot, best_orders)
                ev_c = ev.evaluate(
                    best_assign, op_priority=op_prio, op_bay_overrides=best_bays,
                    op_crew_overrides=ncrews, task_order_override=best_orders,
                )
                used_evals += 1
                tobj = _cand_true_obj(ev_c, best_bays, ncrews, graph, snapshot, best_orders)
                if ev_c.valid and tobj + 1e-4 < best_true_obj:
                    best_crews, best_eval, best_true_obj = ncrews, ev_c, tobj

    # 2. Polish releasable tasks sorted by urgency & slack
    rel_sorted = sorted(
        range(len(graph.rel_tids)),
        key=lambda ti: (float(graph.task_due_steps[ti]), -float(graph.task_feats[ti, 6])),
    )
    for ti in rel_sorted:
        if used_evals >= max_polish_evals:
            break
        tid = graph.rel_tids[ti]
        cur_a = best_assign.get(tid)
        for ai in graph.healthy_agv_indices:
            cand_a = graph.agv_ids[ai]
            if not graph.task_agv_mask[ti, ai] or cand_a == cur_a or used_evals >= max_polish_evals:
                continue
            nass = dict(best_assign)
            nass[tid] = cand_a
            op_prio = _derive_op_prio(snapshot, best_orders)
            ev_c = ev.evaluate(
                nass, op_priority=op_prio, op_bay_overrides=best_bays,
                op_crew_overrides=best_crews, task_order_override=None,
            )
            used_evals += 1
            tobj = _cand_true_obj(ev_c, best_bays, best_crews, graph, snapshot, None)
            if ev_c.valid and tobj + 1e-4 < best_true_obj:
                best_assign, best_orders, best_eval, best_true_obj = nass, None, ev_c, tobj

    return best_assign, best_bays, best_crews, best_orders, best_eval, best_true_obj


def _oracle_label_snapshot(
    snapshot: NativeSnapshot,
    network: Any,
    experts: dict[str, Any],
    seed: int,
) -> Optional[dict[str, Any]]:
    """Run multi-expert B2 ensemble + 1-opt critical-chain polish on one causal snapshot."""
    clf = classify_commitments(snapshot)
    graph = extract_v19_hetero_graph(snapshot, network, clf=clf)
    if len(graph.rel_tids) == 0 or len(graph.agv_ids) == 0 or len(graph.op_ids) == 0:
        return None

    ev = CVCREvaluator(network, snapshot, alpha_makespan=3.0, lambda_change=15.0, use_rust=True, mode="B2")
    base_prio = _derive_op_prio(snapshot, None)
    keep_eval = ev.evaluate(
        graph.base_assign, op_priority=base_prio, op_bay_overrides=graph.base_bays, op_crew_overrides=graph.base_crews
    )
    keep_true_obj = _cand_true_obj(keep_eval, graph.base_bays, graph.base_crews, graph, snapshot, None)

    best_assign = dict(graph.base_assign)
    best_bays = dict(graph.base_bays)
    best_crews = dict(graph.base_crews)
    best_orders: Optional[dict[str, list[str]]] = dict(keep_eval.robot_order_seq)
    best_eval = keep_eval
    best_true_obj = keep_true_obj
    winner_name = "keep"

    def _consider(
        cand_name: str,
        raw_assign: dict[str, str],
        raw_bays: dict[str, str],
        raw_crews: dict[str, int],
        raw_orders: Optional[dict[str, list[str]]],
    ) -> None:
        nonlocal best_assign, best_bays, best_crews, best_orders, best_eval, best_true_obj, winner_name
        assign = dict(graph.fixed_assign)
        for tid in graph.rel_tids:
            agv = raw_assign.get(tid, graph.base_assign.get(tid))
            if agv in graph.agv_ids:
                ai = graph.agv_ids.index(agv)
                if not graph.task_agv_mask[graph.rel_tids.index(tid), ai]:
                    agv = graph.base_assign[tid]
            else:
                agv = graph.base_assign[tid]
            assign[tid] = agv
        bays = dict(graph.base_bays)
        crews = dict(graph.base_crews)
        for oi, oid in enumerate(graph.op_ids):
            if oid not in graph.frozen_ops and oid in raw_bays:
                b = raw_bays[oid]
                if b in graph.bay_ids and graph.op_bay_mask[oi, graph.bay_ids.index(b)]:
                    bays[oid] = b
            if graph.op_crew_mask[oi].sum() > 1 and oid in raw_crews:
                c = int(raw_crews[oid])
                if c in graph.crew_ids and graph.op_crew_mask[oi, graph.crew_ids.index(c)]:
                    crews[oid] = c
        op_prio = _derive_op_prio(snapshot, raw_orders)
        ev_c = ev.evaluate(
            assign, op_priority=op_prio, op_bay_overrides=bays,
            op_crew_overrides=crews, task_order_override=raw_orders,
        )
        t_obj = _cand_true_obj(ev_c, bays, crews, graph, snapshot, raw_orders)
        if ev_c.valid and (not best_eval.valid or t_obj + 1e-4 < best_true_obj):
            best_assign = assign
            best_bays = bays
            best_crews = crews
            best_orders = raw_orders
            best_eval = ev_c
            best_true_obj = t_obj
            winner_name = cand_name

    # Query teacher experts cleanly
    for name, fn in experts.items():
        try:
            prop = fn(snapshot, network)
        except Exception:
            continue
        if prop is None or not getattr(prop, "proposed_assignment", None):
            continue
        _consider(
            name,
            dict(prop.proposed_assignment),
            dict(getattr(prop, "proposed_bays", {}) or {}),
            dict(getattr(prop, "proposed_crews", {}) or {}),
            getattr(prop, "proposed_task_sequences", None),
        )

    best_assign, best_bays, best_crews, best_orders, best_eval, best_true_obj = _polish_joint_b2(
        ev, graph, snapshot, best_assign, best_bays, best_crews, best_orders, best_eval, best_true_obj
    )

    if not best_eval.valid:
        return None

    bay_idx_map = {bid: i for i, bid in enumerate(graph.bay_ids)}
    crew_idx_map = {int(cid): i for i, cid in enumerate(graph.crew_ids)}
    agv_idx_map = {aid: i for i, aid in enumerate(graph.agv_ids)}

    target_bays = np.asarray([bay_idx_map[best_bays[oid]] for oid in graph.op_ids], dtype=np.int64)
    target_crews = np.asarray([crew_idx_map[int(best_crews[oid])] for oid in graph.op_ids], dtype=np.int64)
    target_agvs = np.asarray([agv_idx_map[best_assign[tid]] for tid in graph.rel_tids], dtype=np.int64)

    # Focus order score (A1): higher when critical/urgent/earliest unload completion in oracle schedule
    order_seq = best_eval.robot_order_seq or {}
    unload_ticks = best_eval.order_unload_ticks or {}
    seq_pos: dict[str, int] = {}
    for _, seq_r in order_seq.items():
        for pos, tid in enumerate(seq_r):
            seq_pos[tid] = pos
    tasks_by_id = {t.task_id: t for t in snapshot.tasks}
    target_focus = np.zeros((len(graph.rel_tids),), dtype=np.float32)
    target_seq_rank = np.zeros((len(graph.rel_tids),), dtype=np.float32)
    for ti, tid in enumerate(graph.rel_tids):
        t_obj = tasks_by_id[tid]
        urg_bonus = 1.0 if getattr(t_obj, "is_urgent", False) else 0.0
        due_norm = max(-2.0, min(4.0, (float(t_obj.due_step or 200) - float(snapshot.tick)) / 80.0))
        target_focus[ti] = urg_bonus - 0.5 * due_norm + float(graph.task_feats[ti, 7])
        un_t = float(unload_ticks.get(tid, snapshot.tick + 200))
        pos = float(seq_pos.get(tid, 5))
        target_seq_rank[ti] = -(un_t - float(snapshot.tick)) / 40.0 - 0.25 * pos + 0.5 * urg_bonus

    b2_gain = max(0.0, keep_true_obj - best_true_obj)
    has_dist_or_urg = bool(clf.disturbed_robots or any(getattr(t, "is_urgent", False) for t in snapshot.tasks))
    sample_weight = min(4.0, 1.0 + b2_gain / 100.0 + (0.5 if has_dist_or_urg else 0.0))

    ten = graph_to_torch_tensors(graph, device="cpu")
    return {
        "tensors": {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in ten.items()},
        "op_bay_mask": torch.from_numpy(graph.op_bay_mask),
        "op_crew_mask": torch.from_numpy(graph.op_crew_mask),
        "task_agv_mask": torch.from_numpy(graph.task_agv_mask),
        "task_to_op": torch.from_numpy(graph.task_to_op),
        "target_bays": torch.from_numpy(target_bays),
        "target_crews": torch.from_numpy(target_crews),
        "target_agvs": torch.from_numpy(target_agvs),
        "target_focus": torch.from_numpy(target_focus),
        "target_seq_rank": torch.from_numpy(target_seq_rank),
        "target_value": torch.tensor([min(30.0, max(0.0, best_true_obj / 1000.0))], dtype=torch.float32),
        "sample_weight": float(sample_weight),
        "winner": winner_name,
        "keep_true_obj": float(keep_true_obj),
        "best_true_obj": float(best_true_obj),
    }


def collect_g2_dataset(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    max_episodes: Optional[int] = None,
    max_snaps_per_ep: int = 12,
    seed: int = 2026,
) -> dict[str, Any]:
    """Collect multi-expert B2 optimal joint labels over quarantined train episodes."""
    specs = build_train_specs()
    if max_episodes is not None:
        specs = specs[: int(max_episodes)]

    rule_expert = RuleV10PhysicalProposer(use_rust=True)
    ral_expert = DetourDeltaRAL26Proposer(use_rust=True)
    ijcai_expert = IJCAI26PGMPOProposer(
        ckpt_path=_IJCAI26_CKPT, use_rust=True, seed=seed, backend="numpy", c_policies=8
    )
    ijcai_expert._get_net()
    v18_pol = load_internal_cvcr_policy("V18_EA_JOINT")

    def _v18_expert(snap: NativeSnapshot, net: Any) -> Any:
        dd = compute_detour_delta_signals(net, snap, simulator=None)
        return LGCVCRPolicy(
            "V18_SAFE_CPSAT_JOINT",
            net,
            use_rust=True,
            bay_detour_delta=dd["bay_detour_delta"],
            detour_weights_q3=dd["weights_q3"],
            v15_policy=v18_pol,
        ).propose(snap, budget_ms=60.0)

    def _spga_expert(snap: NativeSnapshot, net: Any) -> Any:
        cfg = SPGAConfig(pop_size=20, max_iters=12, mode="g1_unified", budget_ms=35.0, seed=seed + int(snap.tick))
        return run_spga_policy_proposal(snap, net, cfg, use_rust=True)

    experts = {
        "V18_EA_JOINT": _v18_expert,
        "rule_v10": lambda s, n: rule_expert(s, n, budget_ms=50.0),
        "RAL26_DetourDelta": lambda s, n: ral_expert(s, n, budget_ms=30.0),
        "IJCAI26_PGMPO": lambda s, n: ijcai_expert(s, n, budget_ms=45.0),
        "SPGA_AMR": _spga_expert,
    }

    samples: list[dict[str, Any]] = []
    winner_counts: dict[str, int] = {}
    t0 = time.perf_counter()

    for idx_ep, spec in enumerate(specs):
        ep_seed = int(spec["seed"])
        env = AMRReschedulingEnv(
            seed=ep_seed,
            max_steps=int(spec["max_steps"]),
            use_rust=True,
            allow_kitting_shortcut=False,
            pure_physical_reward=True,
            real_compute_window_coupling=False,
            loaded_repair_mode="haven_v12",
            loaded_repair_threshold=20,
            execution_perturbation=spec["p2_cfg"],
            execution_aware=True,
            p4_choke_robust=True,
            expert_v10_portfolio=True,
        )
        ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
        ctrl.reset()
        _, info = env.reset(seed=ep_seed, options={"scenario_tuple": spec["scenario_tuple"]})
        tm = False
        tr = False
        ep_snaps = 0
        while not (tm or tr):
            assert env.sim is not None and env.tracker is not None
            feats = compute_expert_v9_state(
                env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
            )
            s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
            act = ctrl.select_action(feats, s_info, sim=env.sim)
            if act > 0 and ep_snaps < max_snaps_per_ep:
                snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
                rec = _oracle_label_snapshot(snap, env.sim.network, experts, seed=ep_seed + env.current_step)
                if rec is not None:
                    rec["episode_key"] = spec["episode_key"]
                    rec["cluster_key"] = spec["cluster_key"]
                    rec["suite"] = spec["suite"]
                    rec["seed"] = ep_seed
                    samples.append(rec)
                    w = str(rec["winner"])
                    winner_counts[w] = winner_counts.get(w, 0) + 1
                    ep_snaps += 1
            _, _, tm, tr, info = env.step(act)

        if (idx_ep + 1) % 24 == 0 or (idx_ep + 1) == len(specs):
            el = time.perf_counter() - t0
            print(
                f"[BC collect] {idx_ep + 1}/{len(specs)} eps | {len(samples)} samples | "
                f"{el:.1f}s | winners={winner_counts}",
                flush=True,
            )

    dataset_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "samples": samples,
        "meta": {
            "num_episodes": len(specs),
            "num_samples": len(samples),
            "train_4topo_seeds": list(TRAIN_4TOPO_SEEDS),
            "train_workshop_seeds": list(TRAIN_WORKSHOP_SEEDS),
            "winner_counts": winner_counts,
            "collect_seconds": round(time.perf_counter() - t0, 2),
        },
    }
    tmp_p = dataset_path.with_suffix(".pt.tmp")
    torch.save(payload, tmp_p)
    shutil.move(str(tmp_p), str(dataset_path))
    print(f"[BC collect] Saved dataset to {dataset_path} ({len(samples)} samples)", flush=True)
    return payload["meta"]


def _collate_v19_batch(chunk: list[dict[str, Any]], device: torch.device) -> dict[str, torch.Tensor]:
    """Collate variable-size V19 graphs into zero-padded `[B, ...]` batch tensors on `device`."""
    B = len(chunk)
    Op_m = max(int(s["tensors"]["op_x"].shape[1]) for s in chunk)
    T_m = max(int(s["tensors"]["task_x"].shape[1]) for s in chunk)
    A_m = max(int(s["tensors"]["agv_x"].shape[1]) for s in chunk)
    B_m = max(int(s["tensors"]["bay_x"].shape[1]) for s in chunk)
    C_m = max(int(s["tensors"]["crew_x"].shape[1]) for s in chunk)

    t0_ref = chunk[0]["tensors"]
    op_x = torch.zeros((B, Op_m, int(t0_ref["op_x"].shape[-1])), dtype=torch.float32)
    task_x = torch.zeros((B, T_m, int(t0_ref["task_x"].shape[-1])), dtype=torch.float32)
    agv_x = torch.zeros((B, A_m, int(t0_ref["agv_x"].shape[-1])), dtype=torch.float32)
    bay_x = torch.zeros((B, B_m, int(t0_ref["bay_x"].shape[-1])), dtype=torch.float32)
    crew_x = torch.zeros((B, C_m, int(t0_ref["crew_x"].shape[-1])), dtype=torch.float32)

    op_valid = torch.zeros((B, Op_m), dtype=torch.bool)
    task_valid = torch.zeros((B, T_m), dtype=torch.bool)
    agv_valid = torch.zeros((B, A_m), dtype=torch.bool)
    bay_valid = torch.zeros((B, B_m), dtype=torch.bool)
    crew_valid = torch.zeros((B, C_m), dtype=torch.bool)

    task_to_op = torch.full((B, T_m), -1, dtype=torch.long)
    op_dag_adj = torch.zeros((B, Op_m, Op_m, int(t0_ref["op_dag_adj"].shape[-1])), dtype=torch.float32)
    op_bay_x = torch.zeros((B, Op_m, B_m, int(t0_ref["op_bay_x"].shape[-1])), dtype=torch.float32)
    op_bay_mask = torch.zeros((B, Op_m, B_m), dtype=torch.bool)
    op_crew_mask = torch.zeros((B, Op_m, C_m), dtype=torch.bool)
    task_agv_x = torch.zeros((B, T_m, A_m, int(t0_ref["task_agv_x"].shape[-1])), dtype=torch.float32)
    task_agv_mask = torch.zeros((B, T_m, A_m), dtype=torch.bool)
    bay_travel = torch.zeros((B, T_m, B_m), dtype=torch.float32)
    bay_dd_norm = torch.zeros((B, B_m), dtype=torch.float32)
    task_inc_bay = torch.zeros((B, T_m), dtype=torch.long)

    t_bays = torch.zeros((B, Op_m), dtype=torch.long)
    t_crews = torch.zeros((B, Op_m), dtype=torch.long)
    t_agvs = torch.zeros((B, T_m), dtype=torch.long)
    t_focus = torch.zeros((B, T_m), dtype=torch.float32)
    t_seq = torch.zeros((B, T_m), dtype=torch.float32)
    t_val = torch.zeros((B, 1), dtype=torch.float32)
    weights = torch.zeros((B,), dtype=torch.float32)
    n_tasks = torch.zeros((B,), dtype=torch.long)

    for b_i, s in enumerate(chunk):
        ten = s["tensors"]
        no = int(ten["op_x"].shape[1])
        nt = int(ten["task_x"].shape[1])
        na = int(ten["agv_x"].shape[1])
        nb = int(ten["bay_x"].shape[1])
        nc = int(ten["crew_x"].shape[1])

        op_x[b_i, :no] = ten["op_x"][0]
        task_x[b_i, :nt] = ten["task_x"][0]
        agv_x[b_i, :na] = ten["agv_x"][0]
        bay_x[b_i, :nb] = ten["bay_x"][0]
        crew_x[b_i, :nc] = ten["crew_x"][0]

        op_valid[b_i, :no] = True
        task_valid[b_i, :nt] = True
        agv_valid[b_i, :na] = True
        bay_valid[b_i, :nb] = True
        crew_valid[b_i, :nc] = True

        task_to_op[b_i, :nt] = ten["task_to_op"][0]
        op_dag_adj[b_i, :no, :no] = ten["op_dag_adj"][0]
        op_bay_x[b_i, :no, :nb] = ten["op_bay_x"][0]
        op_bay_mask[b_i, :no, :nb] = s["op_bay_mask"]
        op_crew_mask[b_i, :no, :nc] = s["op_crew_mask"]
        task_agv_x[b_i, :nt, :na] = ten["task_agv_x"][0]
        task_agv_mask[b_i, :nt, :na] = s["task_agv_mask"]
        bay_travel[b_i, :nt, :nb] = ten["bay_travel"][0]
        bay_dd_norm[b_i, :nb] = ten["bay_dd_norm"][0]
        task_inc_bay[b_i, :nt] = ten["task_inc_bay"][0]

        t_bays[b_i, :no] = s["target_bays"]
        t_crews[b_i, :no] = s["target_crews"]
        t_agvs[b_i, :nt] = s["target_agvs"]
        t_focus[b_i, :nt] = s["target_focus"]
        t_seq[b_i, :nt] = s["target_seq_rank"]
        t_val[b_i] = s["target_value"]
        weights[b_i] = float(s["sample_weight"])
        n_tasks[b_i] = nt

    return {
        "op_x": op_x.to(device),
        "task_x": task_x.to(device),
        "agv_x": agv_x.to(device),
        "bay_x": bay_x.to(device),
        "crew_x": crew_x.to(device),
        "op_valid": op_valid.to(device),
        "task_valid": task_valid.to(device),
        "agv_valid": agv_valid.to(device),
        "bay_valid": bay_valid.to(device),
        "crew_valid": crew_valid.to(device),
        "task_to_op": task_to_op.to(device),
        "op_dag_adj": op_dag_adj.to(device),
        "op_bay_x": op_bay_x.to(device),
        "op_bay_mask": op_bay_mask.to(device),
        "op_crew_mask": op_crew_mask.to(device),
        "task_agv_x": task_agv_x.to(device),
        "task_agv_mask": task_agv_mask.to(device),
        "bay_travel": bay_travel.to(device),
        "bay_dd_norm": bay_dd_norm.to(device),
        "task_inc_bay": task_inc_bay.to(device),
        "t_bays": t_bays.to(device),
        "t_crews": t_crews.to(device),
        "t_agvs": t_agvs.to(device),
        "t_focus": t_focus.to(device),
        "t_seq": t_seq.to(device),
        "t_val": t_val.to(device),
        "weights": weights.to(device),
        "n_tasks": n_tasks.to(device),
    }


def _compute_bc_batch_loss(
    net: V19HeteroConstructorNet,
    batch: dict[str, torch.Tensor],
    *,
    action_stage: str,
) -> torch.Tensor:
    """Compute vectorized weighted multi-head BC loss across `[B, ...]` padded batch."""
    enc = net.encode_entities(
        batch["op_x"],
        batch["task_x"],
        batch["agv_x"],
        batch["bay_x"],
        batch["crew_x"],
        batch["task_to_op"],
        batch["op_dag_adj"],
        op_valid=batch["op_valid"],
        task_valid=batch["task_valid"],
        agv_valid=batch["agv_valid"],
        bay_valid=batch["bay_valid"],
        crew_valid=batch["crew_valid"],
    )
    h12 = net.a1_a2_heads(enc, batch["op_bay_x"], batch["op_bay_mask"], batch["op_crew_mask"])
    weights = batch["weights"]
    B, T_m = batch["t_agvs"].shape
    A_m = batch["agv_x"].shape[1]
    device = weights.device

    total_loss = torch.zeros((), dtype=torch.float32, device=device)

    # A2: Bays & Crews
    bay_multi = batch["op_valid"] & (batch["op_bay_mask"].sum(dim=-1) > 1)
    if bay_multi.any():
        w_b = weights.unsqueeze(1).expand_as(bay_multi)[bay_multi]
        ce_b = F.cross_entropy(h12["bay_logits"][bay_multi], batch["t_bays"][bay_multi], reduction="none")
        total_loss = total_loss + 0.65 * (ce_b * w_b).mean()

    crew_multi = batch["op_valid"] & (batch["op_crew_mask"].sum(dim=-1) > 1)
    if crew_multi.any():
        w_c = weights.unsqueeze(1).expand_as(crew_multi)[crew_multi]
        ce_c = F.cross_entropy(h12["crew_logits"][crew_multi], batch["t_crews"][crew_multi], reduction="none")
        total_loss = total_loss + 0.45 * (ce_c * w_c).mean()

    # A1 focus ranking
    tv = batch["task_valid"]
    if T_m > 1:
        tv2 = tv.unsqueeze(2) & tv.unsqueeze(1)
        f_diff = batch["t_focus"].unsqueeze(2) - batch["t_focus"].unsqueeze(1)
        p_diff = h12["task_focus_logits"].unsqueeze(2) - h12["task_focus_logits"].unsqueeze(1)
        pair_f = tv2 & (f_diff > 0.15)
        if pair_f.any():
            w_f = weights.view(B, 1, 1).expand_as(pair_f)[pair_f]
            total_loss = total_loss + 0.25 * (F.softplus(-p_diff[pair_f]) * w_f).mean()

    # A3 batched sequential pointer loss (zero CPU sync)
    if T_m > 0:
        with torch.no_grad():
            focus_masked = h12["task_focus_logits"].masked_fill(~tv, -1e9)
            order = torch.argsort(focus_masked, dim=-1, descending=True)  # [B, T_m]

        b_idx = torch.arange(B, device=device)
        added_load = torch.zeros((B, A_m), dtype=torch.float32, device=device)
        added_dur = torch.zeros((B, A_m), dtype=torch.float32, device=device)
        step_losses: list[torch.Tensor] = []

        for step_k in range(T_m):
            ti_b = order[:, step_k]
            active_b = step_k < batch["n_tasks"]
            op_idx_b = batch["task_to_op"][b_idx, ti_b]
            has_op = op_idx_b >= 0
            sel_bay_b = torch.where(has_op, batch["t_bays"][b_idx, op_idx_b.clamp(min=0)], batch["task_inc_bay"][b_idx, ti_b])
            b_tr_b = batch["bay_travel"][b_idx, ti_b, sel_bay_b]
            b_dd_b = batch["bay_dd_norm"][b_idx, sel_bay_b]
            tx_b = batch["task_agv_x"][b_idx, ti_b]
            tmask_b = batch["task_agv_mask"][b_idx, ti_b]

            lg = net.a3_step_logits(
                enc, ti_b, sel_bay_b, tx_b, tmask_b, b_tr_b, b_dd_b, added_load, added_dur
            )
            chosen_a = batch["t_agvs"][b_idx, ti_b]
            valid_ce = active_b & (tmask_b.sum(dim=-1) > 1)
            if valid_ce.any():
                ce_a = F.cross_entropy(lg[valid_ce], chosen_a[valid_ce], reduction="none") * weights[valid_ce]
                step_losses.append(ce_a)

            upd_l = torch.zeros_like(added_load)
            upd_l.scatter_(1, chosen_a.unsqueeze(1), active_b.float().unsqueeze(1))
            dur_val = (b_tr_b + tx_b[b_idx, chosen_a, 0]) * active_b.float()
            upd_d = torch.zeros_like(added_dur)
            upd_d.scatter_(1, chosen_a.unsqueeze(1), dur_val.unsqueeze(1))
            added_load = added_load + upd_l
            added_dur = added_dur + upd_d

        if step_losses:
            total_loss = total_loss + 1.0 * torch.cat(step_losses, dim=0).mean()

        if action_stage == "A1234" and T_m > 1:
            seq_sc = net.a4_seq_scores(enc, batch["t_agvs"])
            tv2 = tv.unsqueeze(2) & tv.unsqueeze(1)
            same_agv = batch["t_agvs"].unsqueeze(2) == batch["t_agvs"].unsqueeze(1)
            s_diff = batch["t_seq"].unsqueeze(2) - batch["t_seq"].unsqueeze(1)
            pair_s = tv2 & same_agv & (s_diff > 0.05)
            if pair_s.any():
                w_s = weights.view(B, 1, 1).expand_as(pair_s)[pair_s]
                total_loss = total_loss + 0.50 * (F.softplus(-(seq_sc.unsqueeze(2) - seq_sc.unsqueeze(1))[pair_s]) * w_s).mean()

    val_loss = F.smooth_l1_loss(h12["value"].view(-1), batch["t_val"].view(-1), reduction="none") * weights
    total_loss = total_loss + 0.10 * val_loss.mean()
    return total_loss


def train_g2_models(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    *,
    stages: tuple[str, ...] = ("A123", "A1234"),
    epochs: int = 28,
    lr: float = 1.2e-3,
    batch_size: int = 64,
    seed: int = 2026,
) -> dict[str, Any]:
    """Train `V19_HETERO_BC_A123` and `V19_HETERO_BC_A1234` on RTX 5060 CUDA and export `.npz`."""
    if not dataset_path.is_file():
        raise FileNotFoundError(f"G2 dataset not found: {dataset_path}")
    payload = torch.load(dataset_path, map_location="cpu", weights_only=False)
    samples: list[dict[str, Any]] = payload["samples"]
    assert_seed_quarantine([int(s["seed"]) for s in samples], label="loaded_dataset_samples")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[BC train] device={device} samples={len(samples)} epochs={epochs} batch_size={batch_size}", flush=True)

    clusters = sorted({str(s["cluster_key"]) for s in samples})
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(clusters))
    n_val_c = max(1, int(round(0.1 * len(clusters))))
    val_clusters = {clusters[int(i)] for i in perm[:n_val_c]}
    train_samples = [s for s in samples if str(s["cluster_key"]) not in val_clusters]
    val_samples = [s for s in samples if str(s["cluster_key"]) in val_clusters]
    if not val_samples:
        val_samples = train_samples[-max(1, len(train_samples) // 10) :]

    # Pre-collate fixed val batches onto CUDA once
    val_batches = [
        _collate_v19_batch(val_samples[i : i + batch_size], device)
        for i in range(0, len(val_samples), batch_size)
    ]

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"device": str(device), "train_samples": len(train_samples), "val_samples": len(val_samples), "stages": {}}

    for stage in stages:
        if stage not in ("A123", "A1234"):
            raise ValueError(f"Unsupported G2 action_stage: {stage!r}")
        torch.manual_seed(seed + (11 if stage == "A123" else 29))
        net = V19HeteroConstructorNet(hidden_dim=64, n_heads=4, n_layers=2, action_stage=stage, ablation_mode="full").to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, epochs))

        best_val_loss = float("inf")
        best_state: Optional[dict[str, torch.Tensor]] = None
        best_epoch = 0
        t_stage0 = time.perf_counter()

        for ep in range(1, epochs + 1):
            net.train()
            order_idx = rng.permutation(len(train_samples))
            ep_losses: list[float] = []
            for start_i in range(0, len(order_idx), batch_size):
                chunk = [train_samples[int(j)] for j in order_idx[start_i : start_i + batch_size]]
                batch = _collate_v19_batch(chunk, device)
                opt.zero_grad(set_to_none=True)
                loss = _compute_bc_batch_loss(net, batch, action_stage=stage)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.5)
                opt.step()
                ep_losses.append(float(loss.detach().item()))
            sched.step()

            net.eval()
            val_losses: list[float] = []
            with torch.inference_mode():
                for vb in val_batches:
                    vl = _compute_bc_batch_loss(net, vb, action_stage=stage)
                    val_losses.append(float(vl.item()))
            mean_tr = float(np.mean(ep_losses)) if ep_losses else 0.0
            mean_va = float(np.mean(val_losses)) if val_losses else 0.0
            if mean_va < best_val_loss:
                best_val_loss = mean_va
                best_epoch = ep
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            if ep == 1 or ep % 5 == 0 or ep == epochs:
                print(
                    f"[BC train {stage}] ep {ep:02d}/{epochs} ({time.perf_counter() - t_stage0:.1f}s) "
                    f"train={mean_tr:.4f} val={mean_va:.4f} best_val={best_val_loss:.4f}@{best_epoch}",
                    flush=True,
                )

        assert best_state is not None
        cpu_net = V19HeteroConstructorNet(hidden_dim=64, n_heads=4, n_layers=2, action_stage=stage, ablation_mode="full")
        cpu_net.load_state_dict(best_state, strict=True)
        cpu_net.eval()

        pt_path = CKPT_DIR / f"v19_hetero_bc_{stage.lower()}.pt"
        npz_path = CKPT_DIR / f"v19_hetero_bc_{stage.lower()}.npz"
        torch.save(
            {
                "model": best_state,
                "config": cpu_net.config_dict(),
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "train_seed": seed,
            },
            pt_path,
        )
        export_v19_npz(pt_path, npz_path)

        # Verify numerical parity between .pt and .npz on CPU right after export
        max_parity_err = _verify_pt_npz_parity(pt_path, npz_path)
        if max_parity_err >= 1e-5:
            raise RuntimeError(f"Parity error for {stage}: {max_parity_err:.3e} >= 1e-5")

        summary["stages"][stage] = {
            "pt_path": str(pt_path),
            "npz_path": str(npz_path),
            "best_epoch": best_epoch,
            "best_val_loss": round(best_val_loss, 6),
            "train_seconds": round(time.perf_counter() - t_stage0, 2),
            "max_pt_npz_diff": max_parity_err,
        }
        print(
            f"[BC train {stage}] Exported {pt_path.name} & {npz_path.name} (parity={max_parity_err:.2e})",
            flush=True,
        )
    return summary


def _verify_pt_npz_parity(pt_path: Path, npz_path: Path) -> float:
    from amr_sched.rl.v19_hetero_constructor import NumpyV19HeteroConstructorNet
    sc, p2, ms = build_track1_scenario(1001, "kitting_windows")
    env = AMRReschedulingEnv(seed=1001, max_steps=ms, use_rust=True, execution_perturbation=p2, execution_aware=True)
    env.reset(seed=1001, options={"scenario_tuple": sc})
    assert env.sim is not None and env.tracker is not None
    ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
    ctrl.reset()
    for _ in range(40):
        obs = compute_expert_v9_state(env.sim, env.tracker)
        msk = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
        _, _, tm, tr, _ = env.step(ctrl.select_action(obs, {"action_mask": msk}, sim=env.sim))
        if tm or tr:
            break
    snap = NativeSnapshotCapture.capture(env.sim, env.tracker.operations, hide_future_releases=True)
    g = extract_v19_hetero_graph(snap, env.sim.network)
    pt_ck = torch.load(pt_path, map_location="cpu", weights_only=False)
    pt_net = V19HeteroConstructorNet(**pt_ck["config"])
    pt_net.load_state_dict(pt_ck["model"], strict=True)
    pt_net.eval()
    np_net = NumpyV19HeteroConstructorNet(dict(np.load(str(npz_path), allow_pickle=False)))
    ten = graph_to_torch_tensors(g, device="cpu")
    with torch.inference_mode():
        pe = pt_net.encode_entities(ten["op_x"], ten["task_x"], ten["agv_x"], ten["bay_x"], ten["crew_x"], ten["task_to_op"], ten["op_dag_adj"])
        ph = pt_net.a1_a2_heads(pe, ten["op_bay_x"], ten["op_bay_mask"], ten["op_crew_mask"])
    ne = np_net.encode_entities(g)
    nh = np_net.a1_a2_heads(ne, g)
    def _rel_err(a: np.ndarray, b: np.ndarray) -> float:
        valid = np.abs(a) < 1e6
        if not np.any(valid):
            return 0.0
        scale = np.maximum(1.0, np.abs(a[valid]))
        return float(np.max(np.abs(a[valid] - b[valid]) / scale))

    return float(max(
        _rel_err(ph["bay_logits"][0].numpy(), nh["bay_logits"][0]),
        _rel_err(ph["crew_logits"][0].numpy(), nh["crew_logits"][0]),
        _rel_err(ph["value"][0].numpy(), nh["value"][0]),
    ))


def _run_v19_gate_episode(
    ckpt_npz: Path,
    policy_label: str,
    spec: dict[str, Any],
) -> dict[str, Any]:
    """Run one single-process real-wall-clock episode using exact `policy_registry` discrete contract."""
    import copy
    sc_copy = copy.deepcopy(spec["scenario_tuple"])
    p2_copy = copy.deepcopy(spec["p2_cfg"])
    ep_seed = int(spec["seed"])
    proposer = V19HeteroConstructorProposer(ckpt_npz, policy_id=policy_label, backend="numpy", use_rust=True, seed=ep_seed)
    proposer._get_net()
    env = AMRReschedulingEnv(
        seed=ep_seed,
        max_steps=int(spec["max_steps"]),
        use_rust=True,
        allow_kitting_shortcut=False,
        pure_physical_reward=True,
        real_compute_window_coupling=True,
        loaded_repair_mode="haven_v12",
        loaded_repair_threshold=20,
        record_trajectory=True,
        execution_perturbation=p2_copy,
        execution_aware=True,
        p4_choke_robust=True,
        expert_v10_portfolio=True,
        external_proposer=proposer,
    )
    ctrl = RuleV10PortfolioPolicy(epoch_interval=16)
    ctrl.reset()
    t0 = time.perf_counter()
    _, info = env.reset(seed=ep_seed, options={"scenario_tuple": sc_copy})
    tm = False
    tr = False
    while not (tm or tr):
        assert env.sim is not None and env.tracker is not None
        feats = compute_expert_v9_state(
            env.sim, env.tracker, last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks, decision_logs=env.decision_logs,
        )
        s_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
        act = ctrl.select_action(feats, s_info, sim=env.sim)
        _, _, tm, tr, info = env.step(act)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    m = env.compute_episode_metrics()
    aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
    dec_ms = [float(r["t_total_ms"]) for r in env.decision_logs]
    overruns = sum(1 for r in env.decision_logs if not bool(r.get("deadline_met", True)))
    return {
        "policy": policy_label,
        "episode_key": spec["episode_key"],
        "suite": spec["suite"],
        "seed": ep_seed,
        "audit_passed": bool(aud.pass_audit),
        "net_obj": round(float(m["net_obj"]), 4),
        "cmax": round(float(m["cmax"]), 4),
        "tardiness": round(float(m["total_tardiness"]), 4),
        "churn": int(m["total_churn"]),
        "task_comp": round(float(m["task_completion_rate"]), 6),
        "prod_comp": round(float(m["production_completion_rate"]), 6),
        "compute_ms": round(float(m["total_compute_ms"]), 3),
        "wall_ms": round(wall_ms, 2),
        "mean_decision_ms": round(float(np.mean(dec_ms)) if dec_ms else 0.0, 3),
        "deadline_overrun_rate": round(float(overruns / max(1, len(dec_ms))), 6),
    }


def _aggregate_gate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "audit_pass_rate": round(float(np.mean([1.0 if r["audit_passed"] else 0.0 for r in rows])), 6),
        "mean_net_obj": round(float(np.mean([r["net_obj"] for r in rows])), 4),
        "mean_cmax": round(float(np.mean([r["cmax"] for r in rows])), 4),
        "mean_tardiness": round(float(np.mean([r["tardiness"] for r in rows])), 4),
        "mean_churn": round(float(np.mean([r["churn"] for r in rows])), 4),
        "mean_task_comp": round(float(np.mean([r["task_comp"] for r in rows])), 6),
        "mean_prod_comp": round(float(np.mean([r["prod_comp"] for r in rows])), 6),
        "mean_decision_ms": round(float(np.mean([r["mean_decision_ms"] for r in rows])), 3),
        "deadline_overrun_rate": round(float(np.mean([r["deadline_overrun_rate"] for r in rows])), 6),
    }


def run_g2_val_gate() -> dict[str, Any]:
    """Evaluate `V19_HETERO_BC_A123` vs `V19_HETERO_BC_A1234` vs `V18_EA_JOINT` on 72 unseen val episodes."""
    specs = build_val_specs()
    ckpt_a123 = CKPT_DIR / "v19_hetero_bc_a123.npz"
    ckpt_a1234 = CKPT_DIR / "v19_hetero_bc_a1234.npz"
    for p in (ckpt_a123, ckpt_a1234):
        if not p.is_file():
            raise FileNotFoundError(f"Missing G2 checkpoint for validation gate: {p}")

    rows_by_pol: dict[str, list[dict[str, Any]]] = {
        "V19_HETERO_BC_A123": [],
        "V19_HETERO_BC_A1234": [],
        "V18_EA_JOINT": [],
    }
    t0 = time.perf_counter()
    for i, sp in enumerate(specs):
        rows_by_pol["V19_HETERO_BC_A123"].append(_run_v19_gate_episode(ckpt_a123, "V19_HETERO_BC_A123", sp))
        rows_by_pol["V19_HETERO_BC_A1234"].append(_run_v19_gate_episode(ckpt_a1234, "V19_HETERO_BC_A1234", sp))
        v18_rec = run_discrete_episode(
            "V18_EA_JOINT",
            scenario_tuple=sp["scenario_tuple"],
            seed=int(sp["seed"]),
            max_steps=int(sp["max_steps"]),
            p2_cfg=sp["p2_cfg"],
            clock_mode="wall",
            episode_key=sp["episode_key"],
            backend="numpy",
        )
        rows_by_pol["V18_EA_JOINT"].append(
            {
                "policy": "V18_EA_JOINT",
                "episode_key": sp["episode_key"],
                "suite": sp["suite"],
                "seed": int(sp["seed"]),
                "audit_passed": bool(v18_rec["audit_passed"]),
                "net_obj": float(v18_rec["net_obj"]),
                "cmax": float(v18_rec["cmax"]),
                "tardiness": float(v18_rec["tardiness"]),
                "churn": int(v18_rec["churn"]),
                "task_comp": float(v18_rec["task_comp"]),
                "prod_comp": float(v18_rec["prod_comp"]),
                "compute_ms": float(v18_rec["compute_ms"]),
                "wall_ms": float(v18_rec["wall_ms"]),
                "mean_decision_ms": float(v18_rec["mean_decision_ms"]),
                "deadline_overrun_rate": 0.0,
            }
        )
        if (i + 1) % 18 == 0 or (i + 1) == len(specs):
            print(f"[BC gate] {i + 1}/{len(specs)} val episodes complete ({time.perf_counter() - t0:.1f}s)", flush=True)

    summary = {pol: _aggregate_gate_rows(rows) for pol, rows in rows_by_pol.items()}
    summary_by_suite = {
        pol: {
            suite_k: _aggregate_gate_rows([r for r in rows if (r["suite"] == "topo4" if suite_k == "topo4" else r["suite"] in ("track1", "track2"))])
            for suite_k in ("topo4", "workshop")
        }
        for pol, rows in rows_by_pol.items()
    }

    s123 = summary["V19_HETERO_BC_A123"]
    s1234 = summary["V19_HETERO_BC_A1234"]
    key123 = (-s123["audit_pass_rate"], -s123["mean_task_comp"], -s123["mean_prod_comp"], s123["mean_net_obj"])
    key1234 = (-s1234["audit_pass_rate"], -s1234["mean_task_comp"], -s1234["mean_prod_comp"], s1234["mean_net_obj"])
    winner_pol = "V19_HETERO_BC_A123" if key123 <= key1234 else "V19_HETERO_BC_A1234"
    winning_stage = "A123" if winner_pol == "V19_HETERO_BC_A123" else "A1234"

    win_pt = CKPT_DIR / f"v19_hetero_bc_{winning_stage.lower()}.pt"
    win_npz = CKPT_DIR / f"v19_hetero_bc_{winning_stage.lower()}.npz"
    shutil.copy2(win_pt, CKPT_DIR / "v19_hetero_bc.pt")
    shutil.copy2(win_npz, CKPT_DIR / "v19_hetero_bc.npz")

    report = {
        "gate": "G2_BC_VAL",
        "clock_mode": "wall",
        "workers": 1,
        "val_episodes": len(specs),
        "val_4topo_seeds": list(VAL_4TOPO_SEEDS),
        "val_workshop_seeds": list(VAL_WORKSHOP_SEEDS),
        "winning_policy": winner_pol,
        "winning_stage": winning_stage,
        "summary": summary,
        "summary_by_suite": summary_by_suite,
        "rows": rows_by_pol,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = OUT_DIR / "g2_gate_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[BC gate] Winner={winner_pol} ({winning_stage}) | summary={summary} -> {report_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Train & gate V19 Gate G2 Behavior Cloning.")
    parser.add_argument("--stage", choices=("collect", "train", "gate", "all"), default="all")
    parser.add_argument("--stages", nargs="+", default=["A123", "A1234"])
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-snaps-per-ep", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.stage in ("collect", "all"):
        collect_g2_dataset(
            args.dataset, max_episodes=args.max_episodes, max_snaps_per_ep=args.max_snaps_per_ep, seed=args.seed
        )
    if args.stage in ("train", "all"):
        train_g2_models(
            args.dataset,
            stages=tuple(args.stages),
            epochs=args.epochs,
            lr=args.lr,
            batch_size=args.batch_size,
            seed=args.seed,
        )
    if args.stage in ("gate", "all"):
        run_g2_val_gate()


if __name__ == "__main__":
    main()
