"""Shared quarantined scenario suites, strict seed-disjointness guards, multi-horizon
counterfactual rollout engine, and statistical summary utilities for IF-CORE-RGRO-01.

Quarantine Policy (Hard Enforced):
  - DEV splits (`TRAIN`, `VAL-72`, `VAL-Static-35`, `DEV-Stress-24`) are STRICTLY disjoint
    from ALL historical cohorts (`A / B / C / D / D-Struct / G1 / Legacy Blind`) AND from
    ALL brand-new unseen confirmation suites (`Cohort E / E-Struct / E-Compute`).
  - `Cohort E / E-Struct / E-Compute` are ONLY instantiated inside Task 4 after SHA-256 lock.
"""

from __future__ import annotations

import copy
from dataclasses import replace
import hashlib
import math
from pathlib import Path
import sys
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.entities import Task, TaskPriority
from amr_sched.network import Edge, Network, Node
from amr_sched.rl import RuleV10PortfolioPolicy, compute_expert_v10_action_mask, compute_expert_v9_state
from experiments.train_v19_bc import (
    FORBIDDEN_BLIND_SEEDS,
    FORBIDDEN_DEV_SEEDS,
    TRAIN_4TOPO_SEEDS,
    TRAIN_WORKSHOP_SEEDS,
    VAL_4TOPO_SEEDS,
    VAL_WORKSHOP_SEEDS,
    _build_4topo_episode_spec,
    _build_workshop_episode_spec,
)
from experiments.v20_audit.common import (
    COHORT_A_SEEDS,
    COHORT_B_SEEDS,
    COHORT_C_RESERVED_SEEDS,
    RecordedStep,
    build_val72_specs,
    build_val_static35_specs,
    replay_fork_env_at_tick,
)
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES

OUT_RGRO_DIR = ROOT / "out" / "rgro"

# Historical Cohort D / D-Struct (archived diagnostic only; forbidden in DEV & Cohort E)
COHORT_D_4TOPO_SEEDS: tuple[int, ...] = (57101, 57102, 57201, 57202, 57301, 57302)
COHORT_D_WORKSHOP_SEEDS: tuple[int, ...] = (5901, 5902, 5903, 5904, 5905, 5906)
COHORT_D_STRUCT_SEEDS: tuple[int, ...] = tuple(range(6901, 6913))
COHORT_D_ALL_SEEDS: frozenset[int] = frozenset(
    set(COHORT_D_4TOPO_SEEDS) | set(COHORT_D_WORKSHOP_SEEDS) | set(COHORT_D_STRUCT_SEEDS)
)

# Quarantined DEV Stress seeds (strictly DEV-only for Tasks 1-3)
DEV_STRESS_SEEDS: tuple[int, ...] = (17901, 17902, 17903, 17904, 17905, 17906)

# Brand-new unseen confirmation cohorts (STRICTLY LOCKED for Task 4 single-shot only)
COHORT_E_4TOPO_SEEDS: tuple[int, ...] = (77101, 77102, 77201, 77202, 77301, 77302)
COHORT_E_WORKSHOP_SEEDS: tuple[int, ...] = (7901, 7902, 7903, 7904, 7905, 7906)
COHORT_E_STRUCT_SEEDS: tuple[int, ...] = tuple(range(8901, 8913))
COHORT_E_COMPUTE_4TOPO_SEEDS: tuple[int, ...] = (97101, 97201, 97301)
COHORT_E_COMPUTE_WS_SEEDS: tuple[int, ...] = (9901, 9902, 9903)

COHORT_E_ALL_SEEDS: frozenset[int] = frozenset(
    set(COHORT_E_4TOPO_SEEDS)
    | set(COHORT_E_WORKSHOP_SEEDS)
    | set(COHORT_E_STRUCT_SEEDS)
    | set(COHORT_E_COMPUTE_4TOPO_SEEDS)
    | set(COHORT_E_COMPUTE_WS_SEEDS)
)

ALL_HISTORICAL_AND_HOLDOUT_SEEDS: frozenset[int] = frozenset(
    set(FORBIDDEN_BLIND_SEEDS)
    | set(FORBIDDEN_DEV_SEEDS)
    | set(COHORT_A_SEEDS)
    | set(COHORT_B_SEEDS)
    | set(COHORT_C_RESERVED_SEEDS)
    | set(COHORT_D_ALL_SEEDS)
)


def assert_dev_quarantine(seeds: list[int] | tuple[int, ...] | set[int] | frozenset[int], *, label: str) -> None:
    """Assert seeds are strictly quarantined DEV (zero overlap with A/B/C/D/D-Struct/E/G1/Blind)."""
    s = {int(x) for x in seeds}
    forbidden = set(ALL_HISTORICAL_AND_HOLDOUT_SEEDS) | set(COHORT_E_ALL_SEEDS)
    bad = sorted(s & forbidden)
    if bad:
        raise RuntimeError(f"RGRO DEV QUARANTINE VIOLATION in {label}: {bad[:12]}")


def assert_cohort_e_disjoint() -> None:
    """Verify Cohort E / E-Struct / E-Compute have ZERO overlap with any DEV or historical cohort."""
    dev_all = (
        set(TRAIN_4TOPO_SEEDS)
        | set(TRAIN_WORKSHOP_SEEDS)
        | set(VAL_4TOPO_SEEDS)
        | set(VAL_WORKSHOP_SEEDS)
        | set(DEV_STRESS_SEEDS)
    )
    prior_all = set(ALL_HISTORICAL_AND_HOLDOUT_SEEDS) | dev_all
    overlap = sorted(set(COHORT_E_ALL_SEEDS) & prior_all)
    if overlap:
        raise RuntimeError(f"COHORT E DISJOINTNESS VIOLATION: {overlap[:12]}")
    # Also verify internal disjointness among E, E-Struct, and E-Compute
    e_std = set(COHORT_E_4TOPO_SEEDS) | set(COHORT_E_WORKSHOP_SEEDS)
    e_str = set(COHORT_E_STRUCT_SEEDS)
    e_cmp = set(COHORT_E_COMPUTE_4TOPO_SEEDS) | set(COHORT_E_COMPUTE_WS_SEEDS)
    if (e_std & e_str) or (e_std & e_cmp) or (e_str & e_cmp):
        raise RuntimeError("Internal overlap among Cohort E / E-Struct / E-Compute seeds!")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _rebuild_network_without_pairs(net: Network, drop_pairs: set[tuple[str, str]]) -> Network:
    """Build a fresh, adjacency-consistent `Network` excluding undirected edge pairs."""
    out = Network()
    for n in net.nodes.values():
        out.add_node(
            Node(
                n.node_id,
                float(n.x),
                float(n.y),
                kind=n.kind,
                capacity=int(getattr(n, "capacity", 1)),
            )
        )
    for e in net.edges.values():
        pair = (e.src, e.dst) if e.src < e.dst else (e.dst, e.src)
        if pair in drop_pairs:
            continue
        out.add_edge(Edge(e.src, e.dst, length=float(e.length), speed_limit=float(e.speed_limit)))
    return out


def _is_strongly_connected(net: Network) -> bool:
    if not net.nodes:
        return True
    adj: dict[str, list[str]] = {nid: [] for nid in net.nodes}
    for e in net.edges.values():
        adj.setdefault(e.src, []).append(e.dst)
    start = next(iter(net.nodes))
    seen = {start}
    q = [start]
    while q:
        u = q.pop()
        for dst in adj.get(u, ()):
            if dst not in seen:
                seen.add(dst)
                q.append(dst)
    return len(seen) == len(net.nodes)


def _build_structural_stress_specs(seeds: tuple[int, ...], *, prefix: str, group_prefix: str) -> list[dict[str, Any]]:
    """Build 4 structurally diverse stress families per seed (4 * len(seeds) episodes)."""
    specs: list[dict[str, Any]] = []
    for idx, sd in enumerate(seeds):
        # Family 1: Asymmetric Bottleneck Graph (cleanly remove 2 non-bridge interior corridor edges)
        sp1 = _build_workshop_episode_spec("track1", int(sd), "kitting_windows")
        net1, tasks1, agvs1, ops1, meta1 = copy.deepcopy(sp1["scenario_tuple"])
        cand_pairs = sorted(
            {
                (e.src, e.dst) if e.src < e.dst else (e.dst, e.src)
                for e in net1.edges.values()
                if str(e.src).startswith("J_") and str(e.dst).startswith("J_")
            }
        )
        dropped: set[tuple[str, str]] = set()
        cur_net = net1
        for pair in cand_pairs:
            if len(dropped) >= 2:
                break
            trial_drop = dropped | {pair}
            trial_net = _rebuild_network_without_pairs(net1, trial_drop)
            if _is_strongly_connected(trial_net):
                dropped = trial_drop
                cur_net = trial_net
        sp1["scenario_tuple"] = (cur_net, tasks1, agvs1, ops1, meta1)
        sp1["episode_key"] = f"{prefix}_bottleneck_{sd}"
        sp1["suite_group"] = f"{group_prefix}-Bottleneck"
        sp1["suite"] = f"{prefix}_struct"
        specs.append(sp1)

        # Family 2: 3-Kit Fan-In & Tight Staggered Resource Windows
        sp2 = _build_workshop_episode_spec("track2", int(sd), "kitting_windows")
        net2, tasks2, agvs2, ops2, meta2 = copy.deepcopy(sp2["scenario_tuple"])
        extra_added = False
        new_ops2 = []
        t2_map = {t.task_id: t for t in tasks2}
        for op in ops2:
            if not extra_added and len(op.kit_task_ids) >= 2 and op.kit_task_ids[0] in t2_map:
                ref_t = t2_map[op.kit_task_ids[0]]
                tid_3 = f"{ref_t.task_id}_KIT3"
                tasks2.append(
                    Task(
                        task_id=tid_3,
                        pickup=ref_t.pickup,
                        dropoff=ref_t.dropoff,
                        release_step=int(ref_t.release_step) + 12,
                        due_step=int(ref_t.due_step or 260) + 18,
                        priority=TaskPriority.URGENT,
                        material_type=ref_t.material_type,
                    )
                )
                wins = tuple(
                    (int(w0), max(int(w0) + int(op.proc_steps) + 4, int(w1) - 8))
                    for w0, w1 in (getattr(op, "resource_windows", ()) or ())
                )
                new_ops2.append(replace(op, kit_task_ids=tuple(op.kit_task_ids) + (tid_3,), resource_windows=wins))
                extra_added = True
            else:
                new_ops2.append(op)
        sp2["scenario_tuple"] = (net2, tasks2, agvs2, new_ops2, meta2)
        sp2["episode_key"] = f"{prefix}_3kit_windows_{sd}"
        sp2["suite_group"] = f"{group_prefix}-3KitDAG"
        sp2["suite"] = f"{prefix}_struct"
        sp2["max_steps"] = int(sp2["max_steps"]) + 120
        specs.append(sp2)

        # Family 3: Compound Multi-Wave Blockage + Heavy P2 Execution Drift & Obstacles
        sp3 = _build_workshop_episode_spec("track1", int(sd), "execution_blockage")
        sp3["p2_cfg"] = P2PerturbationConfig(
            seed=int(sd) * 31 + 7,
            enable_travel_drift=True,
            drift_prob=0.35,
            late_max_steps=3,
            enable_dynamic_obstacles=True,
            obstacle_count=3,
            obstacle_duration_steps=24,
            enable_local_planner_fail=True,
            local_fail_count=3,
            local_fail_duration_steps=12,
        )
        sp3["episode_key"] = f"{prefix}_compound_drift_{sd}"
        sp3["suite_group"] = f"{group_prefix}-Compound"
        sp3["suite"] = f"{prefix}_struct"
        specs.append(sp3)

        # Family 4: High-Density Multi-Wave Surge / Tight Windows + P2 Perturbations
        if idx % 2 == 0:
            sp4 = _build_4topo_episode_spec(TOPO4_LAYOUTS[idx % len(TOPO4_LAYOUTS)], int(sd), "window_bottleneck")
        else:
            sp4 = _build_workshop_episode_spec("track2", int(sd), "multi_wave_surge")
        sp4["p2_cfg"] = P2PerturbationConfig(
            seed=int(sd) * 37 + 11,
            enable_travel_drift=True,
            drift_prob=0.28,
            late_max_steps=3,
            enable_dynamic_obstacles=True,
            obstacle_count=2,
            obstacle_duration_steps=18,
            enable_local_planner_fail=True,
            local_fail_count=2,
            local_fail_duration_steps=10,
        )
        sp4["episode_key"] = f"{prefix}_surge_p2_{sd}"
        sp4["suite_group"] = f"{group_prefix}-SurgeP2"
        sp4["suite"] = f"{prefix}_struct"
        specs.append(sp4)
    return specs


def build_dev_stress_24_specs() -> list[dict[str, Any]]:
    """Build 24 quarantined DEV structural-stress episodes (zero overlap with Cohort D-Struct / E-Struct)."""
    assert_dev_quarantine(DEV_STRESS_SEEDS, label="DEV_STRESS_24")
    specs = _build_structural_stress_specs(DEV_STRESS_SEEDS, prefix="devstress", group_prefix="DEV-Stress")
    assert len(specs) == 24
    return specs


def build_dev_train_specs(max_4topo_seeds: int = 3, max_ws_seeds: int = 4) -> list[dict[str, Any]]:
    """Build stratified quarantined TRAIN specs strictly from TRAIN_4TOPO_SEEDS & TRAIN_WORKSHOP_SEEDS."""
    s4 = tuple(int(x) for x in TRAIN_4TOPO_SEEDS[:max_4topo_seeds])
    sw = tuple(int(x) for x in TRAIN_WORKSHOP_SEEDS[:max_ws_seeds])
    assert_dev_quarantine(set(s4) | set(sw), label="DEV_TRAIN")
    specs: list[dict[str, Any]] = []
    for s in s4:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["suite_group"] = "4topo"
                sp["split"] = "train"
                specs.append(sp)
    for tr in ("track1", "track2"):
        for s in sw:
            for reg in WORKSHOP_REGIMES:
                sp = _build_workshop_episode_spec(tr, s, reg)
                sp["suite_group"] = tr
                sp["split"] = "train"
                specs.append(sp)
    return specs


def build_cohort_e_120_specs() -> list[dict[str, Any]]:
    """Build 120 brand-new unseen standard Cohort E episodes (Task 4 single-shot ONLY)."""
    assert_cohort_e_disjoint()
    specs: list[dict[str, Any]] = []
    for s in COHORT_E_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                sp = _build_4topo_episode_spec(lay, int(s), reg)
                sp["suite_group"] = "4topo"
                specs.append(sp)
    for tr in ("track1", "track2"):
        for s in COHORT_E_WORKSHOP_SEEDS:
            for reg in WORKSHOP_REGIMES:
                sp = _build_workshop_episode_spec(tr, int(s), reg)
                sp["suite_group"] = tr
                specs.append(sp)
    assert len(specs) == 120
    return specs


def build_cohort_e_struct_48_specs() -> list[dict[str, Any]]:
    """Build 48 brand-new unseen OOD structural Cohort E-Struct episodes (Task 4 single-shot ONLY)."""
    assert_cohort_e_disjoint()
    specs = _build_structural_stress_specs(COHORT_E_STRUCT_SEEDS, prefix="estruct", group_prefix="E-Struct")
    assert len(specs) == 48
    return specs


def build_cohort_e_compute_36_specs() -> list[dict[str, Any]]:
    """Build 36 brand-new unseen compute-jitter episodes for Cohort E-Compute (Task 4 single-shot ONLY)."""
    assert_cohort_e_disjoint()
    specs: list[dict[str, Any]] = []
    for s in COHORT_E_COMPUTE_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in ("standard", "window_bottleneck"):
                sp = _build_4topo_episode_spec(lay, int(s), reg)
                sp["suite_group"] = "E-Compute-4Topo"
                sp["episode_key"] = f"ecomp_{sp['episode_key']}"
                specs.append(sp)
    for tr in ("track1", "track2"):
        for s in COHORT_E_COMPUTE_WS_SEEDS:
            for reg in ("multi_wave_surge", "execution_blockage"):
                sp = _build_workshop_episode_spec(tr, int(s), reg)
                sp["suite_group"] = f"E-Compute-{tr}"
                sp["episode_key"] = f"ecomp_{sp['episode_key']}"
                specs.append(sp)
    assert len(specs) == 36
    return specs


def rollout_multi_horizon(
    spec: dict[str, Any],
    recorded_steps: list[RecordedStep],
    target_tick: int,
    proposal_or_none: Optional[CandidateProposal],
    *,
    action_at_target: int,
    base_env_policy: str = "V19_ADAPTIVE_RL",
    horizons: tuple[int, ...] = (16, 48, 128),
    full_continuation: bool = False,
) -> dict[str, Any]:
    """Single fork rollout recording exact counterfactual cost at multiple horizons + optional full terminal J_A."""
    from amr_sched.rl.spaces import compute_macro_state

    env, _, _ = replay_fork_env_at_tick(
        spec,
        recorded_steps,
        target_tick,
        base_env_policy=base_env_policy,
    )

    if proposal_or_none is not None:
        env.external_proposer = lambda _s, _n, *, budget_ms=35.0: copy.deepcopy(proposal_or_none)
        cur_obs = compute_macro_state(
            env.sim,
            env.tracker,
            last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks,
        )
        mask_now = env._current_mask(cur_obs)
        eff_action = int(action_at_target)
        if eff_action <= 0 or eff_action >= len(mask_now) or mask_now[eff_action] == 0:
            preferred = [4, 2, 1, 3]
            picked = next((a for a in preferred if a < len(mask_now) and mask_now[a] == 1), None)
            if picked is None:
                picked = next((i for i in range(1, len(mask_now)) if mask_now[i] == 1), 0)
            eff_action = int(picked)
    else:
        env.external_proposer = None
        eff_action = 0

    def _snapshot_cost(tm_now: bool, tr_now: bool) -> float:
        c = env._cost_ledger(terminal=(tm_now or tr_now))
        ct, co = env._completion_counts(int(env.sim._step_cursor))
        rem_t = len(env.sim.tasks) - ct
        rem_o = len(env.tracker.operations) - co
        return float(
            c["cmax"]
            + c["task_tardiness_cost"]
            + c["op_tardiness_cost"]
            + c["churn_cost"]
            + c["compute_cost"]
            + 120.0 * rem_t
            + 80.0 * rem_o
        )

    obs, _, tm, tr, info_first = env.step(eff_action)
    receipt_accepted = bool(info_first.get("receipt_accepted", eff_action == 0))
    churn_step0 = int(info_first.get("churn", 0))

    max_h = max(int(h) for h in horizons)
    h_sorted = sorted(set(int(h) for h in horizons))
    h_costs: dict[int, float] = {}
    ticks_elapsed = 1

    for h in h_sorted:
        if ticks_elapsed >= h:
            h_costs[h] = _snapshot_cost(tm, tr)

    while not (tm or tr) and ticks_elapsed < max_h:
        obs, _, tm, tr, _ = env.step(0)
        ticks_elapsed += 1
        if ticks_elapsed in h_sorted and ticks_elapsed not in h_costs:
            h_costs[ticks_elapsed] = _snapshot_cost(tm, tr)

    # Fill any horizons beyond early termination
    final_partial = _snapshot_cost(tm, tr)
    for h in h_sorted:
        if h not in h_costs:
            h_costs[h] = final_partial

    out: dict[str, Any] = {
        "target_tick": int(target_tick),
        "receipt_accepted": receipt_accepted,
        "churn_step0": churn_step0,
        "starvation_steps_max_h": int(env.tracker.starvation_steps),
    }
    for h in h_sorted:
        out[f"j_{h}"] = round(float(h_costs[h]), 4)

    if full_continuation:
        if not (tm or tr):
            ctrl_cont = RuleV10PortfolioPolicy(epoch_interval=16)
            ctrl_cont.reset()
            env.external_proposer = None
            while not (tm or tr):
                f_c = compute_expert_v9_state(
                    env.sim,
                    env.tracker,
                    last_replan_tick=env.last_replan_tick,
                    recent_conflict_ticks=env.recent_conflict_ticks,
                    decision_logs=env.decision_logs,
                )
                m_c = compute_expert_v10_action_mask(env.sim, env.last_replan_tick)
                a_c = ctrl_cont.select_action(f_c, {"action_mask": m_c, "step": env.current_step}, sim=env.sim)
                obs, _, tm, tr, _ = env.step(a_c)
        m_end = env.compute_episode_metrics()
        aud_end = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=False)
        out["j_full"] = round(float(m_end["net_obj"]), 4)
        out["full_comp100"] = bool(
            float(m_end["task_completion_rate"]) >= 0.9999 and float(m_end["production_completion_rate"]) >= 0.9999
        )
        out["full_audit_passed"] = bool(aud_end.pass_audit)

    return out


def pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    pos = (float(q) / 100.0) * float(len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(len(s) - 1, lo + 1)
    frac = pos - float(lo)
    return float(s[lo] * (1.0 - frac) + s[hi] * frac)


def cvar95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    s = sorted(float(v) for v in vals)
    k = max(1, int(math.ceil(0.05 * len(s))))
    return float(np.mean(s[-k:]))


def bootstrap_ci_mean(diffs: list[float], n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    if not diffs:
        return 0.0, 0.0
    arr = np.asarray(diffs, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(n_boot, len(arr)))
    means = np.mean(arr[idx], axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def paired_stats(cand_rows: list[dict[str, Any]], ref_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired comparison of cand vs ref aligned on episode_key (negative delta = cand better)."""
    rmap = {r["episode_key"]: r for r in ref_rows}
    pairs = [(r, rmap[r["episode_key"]]) for r in cand_rows if r["episode_key"] in rmap]
    if not pairs:
        return {}
    diffs = [float(a["net_obj"]) - float(b["net_obj"]) for a, b in pairs]
    wins = sum(1 for d in diffs if d < -1e-6)
    ties = sum(1 for d in diffs if abs(d) <= 1e-6)
    losses = sum(1 for d in diffs if d > 1e-6)
    mean_cand = float(np.mean([a["net_obj"] for a, _ in pairs]))
    mean_ref = float(np.mean([b["net_obj"] for _, b in pairs]))
    rel_pct = 100.0 * (mean_cand - mean_ref) / max(1e-9, abs(mean_ref))
    ci_lo, ci_hi = bootstrap_ci_mean(diffs)
    nz = [d for d in diffs if abs(d) > 1e-9]
    if len(nz) >= 5:
        try:
            _, p_two = wilcoxon(nz, alternative="two-sided")
            _, p_less = wilcoxon(nz, alternative="less")
        except Exception:
            p_two, p_less = 1.0, 1.0
    else:
        p_two, p_less = 1.0, 1.0
    return {
        "n_pairs": len(pairs),
        "mean_cand": round(mean_cand, 2),
        "mean_ref": round(mean_ref, 2),
        "mean_delta": round(float(np.mean(diffs)), 2),
        "rel_delta_pct": round(rel_pct, 2),
        "ci95_delta": [round(ci_lo, 2), round(ci_hi, 2)],
        "wins": int(wins),
        "ties": int(ties),
        "losses": int(losses),
        "wilcoxon_p_2sided": round(float(p_two), 6),
        "wilcoxon_p_1sided_better": round(float(p_less), 6),
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {}
    objs = [float(r["net_obj"]) for r in rows]
    h1s = [float(r.get("h1", 0.0)) for r in rows]
    lats = [float(x) for r in rows for x in r.get("decision_latencies_ms", [])]
    comp_cnt = sum(1 for r in rows if bool(r.get("comp100", False)))
    aud_cnt = sum(1 for r in rows if bool(r.get("audit_passed", False)))
    over35 = sum(1 for x in lats if x > 35.0)
    return {
        "n": len(rows),
        "comp100_count": int(comp_cnt),
        "comp100_rate": round(comp_cnt / len(rows), 4),
        "audit_pass_rate": round(aud_cnt / len(rows), 4),
        "expired_windows_total": int(sum(int(r.get("expired_windows", 0)) for r in rows)),
        "mean_ja": round(float(np.mean(objs)), 2),
        "median_ja": round(float(np.median(objs)), 2),
        "cvar95_ja": round(cvar95(objs), 2),
        "mean_h1": round(float(np.mean(h1s)), 2),
        "mean_churn": round(float(np.mean([float(r.get("churn", 0.0)) for r in rows])), 2),
        "n_decisions": len(lats),
        "mean_dec_ms": round(float(np.mean(lats)), 3) if lats else 0.0,
        "p95_ms": round(pct(lats, 95.0), 3),
        "p99_ms": round(pct(lats, 99.0), 3),
        "max_ms": round(max(lats), 3) if lats else 0.0,
        "over35_rate_pct": round(100.0 * over35 / max(1, len(lats)), 3),
    }
