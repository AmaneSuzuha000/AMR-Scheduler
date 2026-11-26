"""Final Frozen Blind Holdout Runner for CoRe-Sched V20 (NAJS) across Cohort A (120), Cohort B (120), and Cohort C (120).

Strict scientific protocol:
1. All code and weights MUST be verified against `out/v20_scientific_audit/v20_freeze_manifest.json` BEFORE any blind seed runs.
2. ZERO hyperparameter tuning or code edits after freeze lock (`IF-V20-FREEZE`).
3. Cohort A & Cohort B compare frozen `CoRe-Sched V20` (`V20_NAJS`) and `BALANS_IJCAI25` (+ fresh same-machine `V19_ADAPTIVE_RL` / `V18_EA_JOINT`)
   against the immutable 12 core frozen baselines in `results/g3_fresh_blind_holdout.csv` and `results/g5_pristine_blind_holdout.csv`.
4. Brand-new unseen `Cohort C (120)` runs ALL 14 methods (`V20_NAJS` + 12 `G3_CORE_12_POLICIES` + `BALANS_IJCAI25`) in ONE continuous uncontended wall-clock session.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.baselines.policy_registry import create_discrete_policy_env, run_discrete_episode
from amr_sched.cvcr.policy import CandidateProposal
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from experiments.run_g3_fresh_frozen_holdout import (
    FRESH_4TOPO_SEEDS,
    FRESH_WORKSHOP_SEEDS,
    G3_CORE_12_POLICIES,
    build_fresh_120_specs,
)
from experiments.run_g5_pristine_blind_holdout import (
    PRISTINE_4TOPO_SEEDS,
    PRISTINE_WORKSHOP_SEEDS,
    build_pristine_120_specs,
)
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
from experiments.train_v19_rl import RL_4TOPO_SEEDS, RL_WORKSHOP_SEEDS
from experiments.v20_audit.common import (
    COHORT_C_RESERVED_SEEDS,
    OUT_AUDIT_DIR,
    assert_git_amr_sched_clean_relative_to_baseline,
    build_val72_specs,
)
from experiments.v20_audit.v20_e_liveness import PDGLivenessTracker, build_default_v20_proposer
import subprocess
from tools.benchmarks.compare_workshop_algorithms import REGIMES as WORKSHOP_REGIMES
from experiments.v18_confirmation_scenarios import LAYOUTS as TOPO4_LAYOUTS, REGIMES as TOPO4_REGIMES


def _verify_data_manifest() -> tuple[bool, str]:
    proc = subprocess.run(
        [sys.executable, str(ROOT / "tools" / "data_manifest.py"), "--check"],
        capture_output=True,
        text=True,
    )
    msg = (proc.stdout or proc.stderr or "").strip()
    return (proc.returncode == 0), msg

RESULTS_DIR = ROOT / "results"
FREEZE_MANIFEST_PATH = OUT_AUDIT_DIR / "v20_freeze_manifest.json"

COHORT_C_4TOPO_SEEDS: tuple[int, ...] = (47101, 47102, 47201, 47202, 47301, 47302)
COHORT_C_WORKSHOP_SEEDS: tuple[int, ...] = (4901, 4902, 4903, 4904, 4905, 4906)

CSV_FIELDNAMES: list[str] = [
    "policy",
    "label",
    "episode_key",
    "suite_group",
    "seed",
    "net_obj",
    "h1",
    "cmax",
    "task_tardiness_cost",
    "op_tardiness_cost",
    "unfinished_task_penalty",
    "unfinished_op_penalty",
    "churn_cost",
    "compute_cost",
    "churn",
    "task_comp",
    "prod_comp",
    "comp100",
    "audit_passed",
    "decision_count",
    "mean_decision_ms",
    "decision_ms_p95",
    "decision_ms_p99",
    "expired_windows",
    "transaction_rejections",
    "checkpoint_sha256",
    "pdg_alerts",
    "pdg_escalations",
    "pdg_inner_recoveries",
]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = max(0.0, min(1.0, float(q))) * float(len(sorted_vals) - 1)
    lo = int(pos)
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - float(lo)
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


def _h1(r: dict[str, Any]) -> float:
    return float(r["cmax"]) + float(r["task_tardiness_cost"]) + float(r["op_tardiness_cost"])


def _comp100(r: dict[str, Any]) -> bool:
    return bool(float(r.get("task_comp", 0.0)) >= 0.9999 and float(r.get("prod_comp", 0.0)) >= 0.9999)


def assert_cohort_c_quarantine() -> None:
    """Assert unseen Cohort C seeds are strictly disjoint from all prior splits and within COHORT_C_RESERVED_SEEDS."""
    prior = (
        set(FORBIDDEN_BLIND_SEEDS)
        | set(FORBIDDEN_DEV_SEEDS)
        | set(TRAIN_4TOPO_SEEDS)
        | set(TRAIN_WORKSHOP_SEEDS)
        | set(VAL_4TOPO_SEEDS)
        | set(VAL_WORKSHOP_SEEDS)
        | set(RL_4TOPO_SEEDS)
        | set(RL_WORKSHOP_SEEDS)
        | set(FRESH_4TOPO_SEEDS)
        | set(FRESH_WORKSHOP_SEEDS)
        | set(PRISTINE_4TOPO_SEEDS)
        | set(PRISTINE_WORKSHOP_SEEDS)
    )
    c_seeds = set(COHORT_C_4TOPO_SEEDS) | set(COHORT_C_WORKSHOP_SEEDS)
    if c_seeds & prior:
        raise RuntimeError(f"COHORT C QUARANTINE VIOLATION! Overlap with prior splits: {sorted(c_seeds & prior)}")
    if not c_seeds <= set(COHORT_C_RESERVED_SEEDS):
        raise RuntimeError(f"COHORT C seeds outside reserved set: {sorted(c_seeds - set(COHORT_C_RESERVED_SEEDS))}")


def build_cohort_c_120_specs() -> list[dict[str, Any]]:
    """Build brand-new 120-episode Cohort C holdout mirroring Cohort A & Cohort B structure."""
    assert_cohort_c_quarantine()
    specs: list[dict[str, Any]] = []
    for s in COHORT_C_4TOPO_SEEDS:
        for lay in TOPO4_LAYOUTS:
            for reg in TOPO4_REGIMES:
                sp = _build_4topo_episode_spec(lay, s, reg)
                sp["suite_group"] = "4topo"
                specs.append(sp)
    for track, grp in (("track1", "track1"), ("track2", "track2")):
        for s in COHORT_C_WORKSHOP_SEEDS:
            for reg in WORKSHOP_REGIMES:
                sp = _build_workshop_episode_spec(track, s, reg)
                sp["suite_group"] = grp
                specs.append(sp)
    assert len(specs) == 120, f"Expected 120 Cohort C episodes, got {len(specs)}"
    return specs


def _tracked_files_for_freeze() -> list[Path]:
    files = sorted((ROOT / "experiments" / "v20_audit").glob("*.py"))
    files.extend(
        [
            ROOT / "Tests" / "test_v20_audit_integrity.py",
            OUT_AUDIT_DIR / "v20_a_critic_weights.npz",
            OUT_AUDIT_DIR / "v20_c_neighborhood_head.npz",
            OUT_AUDIT_DIR / "v20_d_anytime_weights.npz",
            RESULTS_DIR / "g3_fresh_blind_holdout.csv",
            RESULTS_DIR / "g5_pristine_blind_holdout.csv",
        ]
    )
    return files


def write_freeze_manifest() -> dict[str, Any]:
    """Verify Gates A/B/E + clean git baseline + data manifest and lock SHA-256 freeze manifest."""
    assert_git_amr_sched_clean_relative_to_baseline()
    ok_data, msg_data = _verify_data_manifest()
    if not ok_data:
        raise RuntimeError(f"Data manifest verification failed before freeze: {msg_data}")

    gate_a = json.loads((OUT_AUDIT_DIR / "v20_a_gate_eval.json").read_text(encoding="utf-8"))
    gate_b = json.loads((OUT_AUDIT_DIR / "v20_b_gate_eval.json").read_text(encoding="utf-8"))
    gate_e = json.loads((OUT_AUDIT_DIR / "v20_e_gate_eval.json").read_text(encoding="utf-8"))
    if not bool(gate_a.get("gate_a_overall_pass")):
        raise RuntimeError("Cannot freeze V20: Gate A did not pass.")
    if not bool(gate_b.get("gate_b_pass")):
        raise RuntimeError("Cannot freeze V20: Gate B did not pass.")
    if not bool(gate_e.get("gate_e_pass")):
        raise RuntimeError("Cannot freeze V20: Gate E did not pass.")

    checksums: dict[str, str] = {}
    for p in _tracked_files_for_freeze():
        if not p.exists():
            raise FileNotFoundError(f"Required frozen artifact missing: {p}")
        rel = str(p.relative_to(ROOT)).replace("\\", "/")
        checksums[rel] = _sha256_file(p)

    manifest = {
        "schema": "v20.freeze_manifest.v1",
        "locked": True,
        "frozen_architecture": {
            "base_proposer": "V20EProposer (V20-B winning bidirectional + WS critic + PDG actionable liveness escape)",
            "pruned_modules": [
                "V20-C learned N_k selector (pruned per Gate C Wilcoxon p=0.4265 >= 0.05; retain deterministic causal cone)",
                "V20-D learned anytime controller (pruned per Gate D 35ms Wilcoxon p=0.0607 >= 0.05; retain deterministic V20-B loop)",
            ],
            "gates_verified": {
                "gate_a_pass": True,
                "gate_b_pass": True,
                "gate_e_pass": True,
            },
        },
        "data_manifest_status": msg_data,
        "sha256_files": checksums,
    }
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    FREEZE_MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_freeze_manifest() -> dict[str, Any]:
    """Verify frozen manifest exists and every tracked code/weight/baseline artifact matches bit-for-bit."""
    if not FREEZE_MANIFEST_PATH.exists():
        raise RuntimeError(f"Missing freeze manifest: {FREEZE_MANIFEST_PATH}. Run with --write-freeze first.")
    manifest = json.loads(FREEZE_MANIFEST_PATH.read_text(encoding="utf-8"))
    assert_git_amr_sched_clean_relative_to_baseline()
    ok_data, msg_data = _verify_data_manifest()
    if not ok_data:
        raise RuntimeError(f"Data manifest violation: {msg_data}")
    recorded = dict(manifest.get("sha256_files") or {})
    for rel, expected_sha in recorded.items():
        fpath = ROOT / rel
        if not fpath.exists():
            raise RuntimeError(f"Frozen artifact missing: {rel}")
        actual_sha = _sha256_file(fpath)
        if actual_sha != expected_sha:
            raise RuntimeError(
                f"FREEZE INTEGRITY VIOLATION! Modified after IF-V20-FREEZE lock: {rel} "
                f"(expected {expected_sha[:12]}, got {actual_sha[:12]})"
            )
    return manifest


def run_v20_blind_episode(
    sp: dict[str, Any],
    *,
    hard_budget_ms: float = 35.0,
    include_decision_latencies: bool = True,
) -> dict[str, Any]:
    """Run frozen CoRe-Sched V20 under exact `run_discrete_episode(..., clock_mode='wall')` protocol."""
    scenario_copy = copy.deepcopy(sp["scenario_tuple"])
    p2_copy = copy.deepcopy(sp["p2_cfg"])
    seed = int(sp["seed"])
    max_steps = int(sp["max_steps"])
    topo = str(scenario_copy[0])

    _, ctrl, env = create_discrete_policy_env(
        "V19_ADAPTIVE_RL",
        topology=topo,
        seed=seed,
        max_steps=max_steps,
        use_rust=True,
        p2_cfg=p2_copy,
        allow_missing_pgmpo=False,
    )
    prop = build_default_v20_proposer(hard_budget_ms=hard_budget_ms, seed=0, enable_pdg_recovery=True)
    tracker = PDGLivenessTracker(cooldown_ticks=12)

    proposer_stats: Counter[str] = Counter()
    observed_ckpt_sha = str(OUT_AUDIT_DIR / "v20_a_critic_weights.npz")
    if (OUT_AUDIT_DIR / "v20_a_critic_weights.npz").exists():
        observed_ckpt_sha = _sha256_file(OUT_AUDIT_DIR / "v20_a_critic_weights.npz")

    def _tracked_proposer(*args: Any, **kwargs: Any) -> CandidateProposal:
        cand = prop(*args, **kwargs)
        if cand is not None:
            telem = cand.telemetry or {}
            proposer_stats["keep_fallbacks"] += int(bool(telem.get("fallback_keep", False)))
            proposer_stats["deadline_overruns"] += int(not bool(telem.get("deadline_met", True)))
        return cand

    env.external_proposer = _tracked_proposer
    obs, info = env.reset(seed=seed, options={"scenario_tuple": scenario_copy})
    assert env.sim is not None and env.tracker is not None

    tm = False
    tr = False
    while not (tm or tr):
        feats = compute_expert_v9_state(
            env.sim,
            env.tracker,
            last_replan_tick=env.last_replan_tick,
            recent_conflict_ticks=env.recent_conflict_ticks,
            decision_logs=env.decision_logs,
        )
        step_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
        act = int(ctrl.select_action(feats, step_info, sim=env.sim))
        act = tracker.observe_and_maybe_escalate(env, step_info, act, proposer=prop)
        obs, _, tm, tr, info = env.step(act)

    m = env.compute_episode_metrics()
    aud = IndependentAuditor.audit_simulation_run(
        env.sim,
        env.tracker.operations,
        require_trajectory=True,
    )
    decision_ms = [
        float(row["t_propose_ms"]) + float(row["t_validate_commit_ms"])
        for row in env.decision_logs
    ]
    decision_sorted = sorted(decision_ms)
    h1_val = float(m["cmax"]) + float(m.get("task_tardiness_cost", 0.0)) + float(m.get("op_tardiness_cost", 0.0))
    task_comp = round(float(m["task_completion_rate"]), 6)
    prod_comp = round(float(m["production_completion_rate"]), 6)

    rec: dict[str, Any] = {
        "policy": "V20_NAJS",
        "label": "CoRe-Sched V20",
        "episode_key": sp["episode_key"],
        "suite_group": sp["suite_group"],
        "seed": seed,
        "net_obj": round(float(m["net_obj"]), 4),
        "h1": round(h1_val, 4),
        "cmax": round(float(m["cmax"]), 4),
        "task_tardiness_cost": round(float(m.get("task_tardiness_cost", 0.0)), 4),
        "op_tardiness_cost": round(float(m.get("op_tardiness_cost", 0.0)), 4),
        "unfinished_task_penalty": round(float(m.get("unfinished_task_penalty", 0.0)), 4),
        "unfinished_op_penalty": round(float(m.get("unfinished_op_penalty", 0.0)), 4),
        "churn_cost": round(float(m.get("churn_cost", 15.0 * float(m.get("total_churn", 0)))), 4),
        "compute_cost": round(float(m.get("compute_cost", 0.05 * float(m.get("total_compute_ms", 0.0)))), 4),
        "churn": int(m["total_churn"]),
        "task_comp": task_comp,
        "prod_comp": prod_comp,
        "comp100": bool(task_comp >= 0.9999 and prod_comp >= 0.9999),
        "audit_passed": bool(aud.pass_audit),
        "decision_count": len(decision_ms),
        "mean_decision_ms": round(sum(decision_ms) / len(decision_ms), 3) if decision_ms else 0.0,
        "decision_ms_p95": round(_percentile(decision_sorted, 0.95), 3),
        "decision_ms_p99": round(_percentile(decision_sorted, 0.99), 3),
        "expired_windows": int(env.total_window_expirations),
        "transaction_rejections": sum(not row["receipt_accepted"] for row in env.decision_logs),
        "checkpoint_sha256": observed_ckpt_sha,
        "pdg_alerts": len(tracker.alert_ticks),
        "pdg_escalations": len(tracker.escalation_ticks),
        "pdg_inner_recoveries": int(prop.pdg_recoveries_triggered),
    }
    if include_decision_latencies:
        rec["decision_latencies_ms"] = [round(float(v), 4) for v in decision_ms]
    return rec


def run_discrete_policy_episode_row(
    pol_key: str,
    label: str,
    sp: dict[str, Any],
    *,
    include_decision_latencies: bool = True,
) -> dict[str, Any]:
    res = run_discrete_episode(
        pol_key,
        copy.deepcopy(sp["scenario_tuple"]),
        seed=int(sp["seed"]),
        max_steps=int(sp["max_steps"]),
        use_rust=True,
        p2_cfg=copy.deepcopy(sp["p2_cfg"]),
        episode_key=sp["episode_key"],
        clock_mode="wall",
        include_decision_latencies=include_decision_latencies,
    )
    h1_val = float(res["cmax"]) + float(res["task_tardiness_cost"]) + float(res["op_tardiness_cost"])
    comp100 = bool(float(res.get("task_comp", 0.0)) >= 0.9999 and float(res.get("prod_comp", 0.0)) >= 0.9999)
    row: dict[str, Any] = {
        "policy": pol_key,
        "label": label,
        "episode_key": sp["episode_key"],
        "suite_group": sp["suite_group"],
        "seed": int(sp["seed"]),
        "net_obj": res["net_obj"],
        "h1": round(h1_val, 4),
        "cmax": res["cmax"],
        "task_tardiness_cost": res["task_tardiness_cost"],
        "op_tardiness_cost": res["op_tardiness_cost"],
        "unfinished_task_penalty": res["unfinished_task_penalty"],
        "unfinished_op_penalty": res["unfinished_op_penalty"],
        "churn_cost": res["churn_cost"],
        "compute_cost": res["compute_cost"],
        "churn": res["churn"],
        "task_comp": res["task_comp"],
        "prod_comp": res["prod_comp"],
        "comp100": comp100,
        "audit_passed": res["audit_passed"],
        "decision_count": res["decision_count"],
        "mean_decision_ms": res["mean_decision_ms"],
        "decision_ms_p95": res["decision_ms_p95"],
        "decision_ms_p99": res["decision_ms_p99"],
        "expired_windows": res["expired_windows"],
        "transaction_rejections": res["transaction_rejections"],
        "checkpoint_sha256": res.get("checkpoint_sha256") or "",
        "pdg_alerts": 0,
        "pdg_escalations": 0,
        "pdg_inner_recoveries": 0,
    }
    if include_decision_latencies:
        row["decision_latencies_ms"] = [float(x) for x in res.get("decision_latencies_ms", [])]
    return row


def _load_csv_rows_by_policy(csv_path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row: dict[str, Any] = dict(raw)
            for k in (
                "net_obj", "h1", "cmax", "task_tardiness_cost", "op_tardiness_cost",
                "unfinished_task_penalty", "unfinished_op_penalty", "churn_cost", "compute_cost",
                "task_comp", "prod_comp", "mean_decision_ms", "decision_ms_p95", "decision_ms_p99",
            ):
                row[k] = float(row[k])
            for k in (
                "seed", "churn", "decision_count", "expired_windows", "transaction_rejections",
                "pdg_alerts", "pdg_escalations", "pdg_inner_recoveries",
            ):
                if k in row and row[k] != "":
                    row[k] = int(float(row[k]))
                else:
                    row[k] = 0
            row["comp100"] = str(row["comp100"]).lower() == "true"
            row["audit_passed"] = str(row["audit_passed"]).lower() == "true"
            out.setdefault(str(row["policy"]), []).append(row)
    return out


def _write_csv(csv_path: Path, rows: list[dict[str, Any]]) -> None:
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    clean_rows = [{k: r.get(k, "") for k in CSV_FIELDNAMES} for r in rows]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        w.writeheader()
        w.writerows(clean_rows)


def _summarize(rows: list[dict[str, Any]], lats: list[float] | None = None) -> dict[str, Any]:
    comp_rows = [r for r in rows if _comp100(r)]
    s_lats = sorted(float(x) for x in lats) if lats else []
    by_suite: dict[str, Any] = {}
    for sk in ("4topo", "track1", "track2"):
        sub = [r for r in rows if r["suite_group"] == sk]
        sub_c = [r for r in sub if _comp100(r)]
        by_suite[sk] = {
            "n": len(sub),
            "comp100_count": len(sub_c),
            "mean_ja": round(float(np.mean([r["net_obj"] for r in sub])), 2) if sub else 0.0,
            "median_ja": round(float(np.median([r["net_obj"] for r in sub])), 2) if sub else 0.0,
            "mean_h1_all": round(float(np.mean([_h1(r) for r in sub])), 2) if sub else 0.0,
            "mean_h1_comp": round(float(np.mean([_h1(r) for r in sub_c])), 2) if sub_c else 0.0,
        }
    return {
        "n": len(rows),
        "audit_pass": sum(1 for r in rows if bool(r["audit_passed"])),
        "comp100_count": len(comp_rows),
        "comp100_pct": round(100.0 * len(comp_rows) / max(1, len(rows)), 2),
        "mean_ja": round(float(np.mean([r["net_obj"] for r in rows])), 2),
        "median_ja": round(float(np.median([r["net_obj"] for r in rows])), 2),
        "pure_h1_all_mean": round(float(np.mean([_h1(r) for r in rows])), 2),
        "pure_h1_comp_mean": round(float(np.mean([_h1(r) for r in comp_rows])), 2) if comp_rows else 0.0,
        "churn_mean": round(float(np.mean([r["churn"] for r in rows])), 2),
        "expired_windows": sum(int(r.get("expired_windows", 0)) for r in rows),
        "transaction_rejections": sum(int(r.get("transaction_rejections", 0)) for r in rows),
        "mean_decision_ms": round(float(np.mean(s_lats)), 3) if s_lats else round(float(np.mean([r["mean_decision_ms"] for r in rows])), 3),
        "pooled_p95_ms": round(_percentile(s_lats, 0.95), 3) if s_lats else None,
        "pooled_p99_ms": round(_percentile(s_lats, 0.99), 3) if s_lats else None,
        "mean_ep_p95_ms": round(float(np.mean([r["decision_ms_p95"] for r in rows])), 3),
        "mean_ep_p99_ms": round(float(np.mean([r["decision_ms_p99"] for r in rows])), 3),
        "pdg_alert_episodes": sum(1 for r in rows if int(r.get("pdg_alerts", 0)) > 0),
        "by_suite": by_suite,
    }


def _paired_stats(core_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cb = {r["episode_key"]: r for r in core_rows}
    bb = {r["episode_key"]: r for r in base_rows}
    keys = sorted(set(cb) & set(bb))
    c_ja = np.array([float(cb[k]["net_obj"]) for k in keys], dtype=np.float64)
    b_ja = np.array([float(bb[k]["net_obj"]) for k in keys], dtype=np.float64)
    diffs = b_ja - c_ja
    mean_c, mean_b = float(c_ja.mean()), float(b_ja.mean())
    pct = 100.0 * (mean_b - mean_c) / max(1e-9, mean_b)
    p_val = float(wilcoxon(diffs, zero_method="wilcox", alternative="greater").pvalue) if np.any(np.abs(diffs) > 1e-9) else 1.0
    rng = np.random.default_rng(20261125)
    boots = np.empty(2000, dtype=np.float64)
    for b in range(2000):
        idx = rng.integers(0, len(keys), size=len(keys))
        boots[b] = 100.0 * float(b_ja[idx].mean() - c_ja[idx].mean()) / max(1e-9, float(b_ja[idx].mean()))
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    wins = int(np.sum(diffs > 1e-6))
    losses = int(np.sum(diffs < -1e-6))
    ties = len(keys) - wins - losses
    return {
        "n": len(keys),
        "v20_mean_ja": round(mean_c, 2),
        "base_mean_ja": round(mean_b, 2),
        "delta_ja_pct": round(pct, 2),
        "ci95_lo": round(ci_lo, 2),
        "ci95_hi": round(ci_hi, 2),
        "wilcoxon_p_v20_better": float(f"{p_val:.4e}"),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }


def _run_cohort(
    cohort_name: str,
    specs: list[dict[str, Any]],
    policies: list[tuple[str, str]],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, list[float]]]:
    """Run all requested policies sequentially over `specs` with uncontended wall-clock timing & prior warmup."""
    print(f"\n=== [{cohort_name}] Warming up {len(policies)} policies on {specs[0]['episode_key']} ===", flush=True)
    for pol_key, _ in policies:
        if pol_key == "V20_NAJS":
            _ = run_v20_blind_episode(specs[0], hard_budget_ms=35.0, include_decision_latencies=False)
        else:
            _ = run_discrete_policy_episode_row(pol_key, pol_key, specs[0], include_decision_latencies=False)

    rows_by_pol: dict[str, list[dict[str, Any]]] = {}
    lats_by_pol: dict[str, list[float]] = {}
    for pol_idx, (pol_key, label) in enumerate(policies, 1):
        t0 = time.perf_counter()
        prows: list[dict[str, Any]] = []
        plats: list[float] = []
        for ep_i, sp in enumerate(specs, 1):
            if pol_key == "V20_NAJS":
                r = run_v20_blind_episode(sp, hard_budget_ms=35.0, include_decision_latencies=True)
            else:
                r = run_discrete_policy_episode_row(pol_key, label, sp, include_decision_latencies=True)
            plats.extend(float(x) for x in r.pop("decision_latencies_ms", []))
            prows.append(r)
            if ep_i % 40 == 0 or ep_i == len(specs):
                cur_ja = float(np.mean([x["net_obj"] for x in prows]))
                cur_c = sum(1 for x in prows if _comp100(x))
                cur_p95 = float(np.percentile(plats, 95)) if plats else 0.0
                print(
                    f"[{cohort_name}] ({pol_idx:02d}/{len(policies)}) {label:18s} | {ep_i:3d}/{len(specs)} | "
                    f"JA={cur_ja:8.1f} | Comp={cur_c:3d}/{ep_i} | PoolP95={cur_p95:5.2f}ms ({time.perf_counter()-t0:.1f}s)",
                    flush=True,
                )
        rows_by_pol[pol_key] = prows
        lats_by_pol[pol_key] = plats
    return rows_by_pol, lats_by_pol


def _build_report_and_save(
    *,
    cohort_a_fresh: tuple[dict[str, list[dict[str, Any]]], dict[str, list[float]]] | None,
    cohort_b_fresh: tuple[dict[str, list[dict[str, Any]]], dict[str, list[float]]] | None,
    cohort_c_fresh: tuple[dict[str, list[dict[str, Any]]], dict[str, list[float]]] | None,
) -> dict[str, Any]:
    g3_csv = _load_csv_rows_by_policy(RESULTS_DIR / "g3_fresh_blind_holdout.csv")
    g5_csv = _load_csv_rows_by_policy(RESULTS_DIR / "g5_pristine_blind_holdout.csv")

    report: dict[str, Any] = {
        "schema": "v20.final_blind_holdout.v1",
        "freeze_manifest_sha256": _sha256_file(FREEZE_MANIFEST_PATH),
        "cohorts": {},
    }

    def _assemble_cohort(
        name: str,
        fresh_tuple: tuple[dict[str, list[dict[str, Any]]], dict[str, list[float]]] | None,
        csv_ref: dict[str, list[dict[str, Any]]] | None,
    ) -> dict[str, Any]:
        sums: dict[str, Any] = {}
        merged_rows: dict[str, list[dict[str, Any]]] = {}
        if csv_ref is not None:
            for pol, rows in csv_ref.items():
                key = f"{pol}__HISTORICAL_CSV"
                merged_rows[key] = rows
                sums[key] = _summarize(rows, None)
        if fresh_tuple is not None:
            frows, flats = fresh_tuple
            for pol, rows in frows.items():
                merged_rows[pol] = rows
                sums[pol] = _summarize(rows, flats.get(pol))
        v20_rows = merged_rows.get("V20_NAJS", [])
        paired: dict[str, Any] = {}
        if v20_rows:
            for pol, rows in merged_rows.items():
                if pol != "V20_NAJS":
                    paired[pol] = _paired_stats(v20_rows, rows)
        # Task C outlier highlight if Cohort B
        highlights = {}
        if name == "cohort_b" and v20_rows:
            for k_out in ("topo4_offset_dual_lane_window_bottleneck_37302", "track2_kitting_windows_3906"):
                m_v20 = next((r for r in v20_rows if r["episode_key"] == k_out), None)
                m_v19 = next((r for r in merged_rows.get("V19_ADAPTIVE_RL", []) if r["episode_key"] == k_out), None)
                if m_v20:
                    highlights[k_out] = {"v20": m_v20, "v19_fresh": m_v19}
        return {"summary": sums, "paired_vs_v20": paired, "highlights": highlights}

    if cohort_a_fresh is not None:
        report["cohorts"]["cohort_a"] = _assemble_cohort("cohort_a", cohort_a_fresh, g3_csv)
    elif (OUT_AUDIT_DIR / "v20_cohort_a_rows.csv").exists():
        loaded_a = _load_csv_rows_by_policy(OUT_AUDIT_DIR / "v20_cohort_a_rows.csv")
        la_path = OUT_AUDIT_DIR / "v20_cohort_a_lats.json"
        la = json.loads(la_path.read_text(encoding="utf-8")) if la_path.exists() else {}
        report["cohorts"]["cohort_a"] = _assemble_cohort("cohort_a", (loaded_a, la), g3_csv)

    if cohort_b_fresh is not None:
        report["cohorts"]["cohort_b"] = _assemble_cohort("cohort_b", cohort_b_fresh, g5_csv)
    elif (OUT_AUDIT_DIR / "v20_cohort_b_rows.csv").exists():
        loaded_b = _load_csv_rows_by_policy(OUT_AUDIT_DIR / "v20_cohort_b_rows.csv")
        lb_path = OUT_AUDIT_DIR / "v20_cohort_b_lats.json"
        lb = json.loads(lb_path.read_text(encoding="utf-8")) if lb_path.exists() else {}
        report["cohorts"]["cohort_b"] = _assemble_cohort("cohort_b", (loaded_b, lb), g5_csv)

    if cohort_c_fresh is not None:
        report["cohorts"]["cohort_c"] = _assemble_cohort("cohort_c", cohort_c_fresh, None)
    elif (OUT_AUDIT_DIR / "v20_cohort_c_rows.csv").exists():
        loaded_c = _load_csv_rows_by_policy(OUT_AUDIT_DIR / "v20_cohort_c_rows.csv")
        lc_path = OUT_AUDIT_DIR / "v20_cohort_c_lats.json"
        lc = json.loads(lc_path.read_text(encoding="utf-8")) if lc_path.exists() else {}
        report["cohorts"]["cohort_c"] = _assemble_cohort("cohort_c", (loaded_c, lc), None)

    (OUT_AUDIT_DIR / "v20_final_blind_holdout.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Write clean markdown table
    lines = ["# CoRe-Sched V20 (NAJS) Frozen Blind Holdout Report", ""]
    for cname in ("cohort_a", "cohort_b", "cohort_c"):
        if cname not in report["cohorts"]:
            continue
        cdata = report["cohorts"][cname]
        lines.append(f"## {cname.upper()} (120 episodes)")
        lines.append("")
        lines.append("| Method | Comp100 | Mean JA | Median JA | Pure H1 | 4-Topo | Track 1 | Track 2 | Pool P95 (ms) | Pool P99 (ms) | Δ JA vs V20 | Wilcoxon p |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
        paired = cdata["paired_vs_v20"]
        for pol, s in sorted(cdata["summary"].items(), key=lambda kv: kv[1]["mean_ja"]):
            bs = s["by_suite"]
            pr = paired.get(pol)
            dlt = f"{pr['delta_ja_pct']:+.2f}%" if pr else "ref"
            pval = f"{pr['wilcoxon_p_v20_better']:.3e}" if pr else "-"
            p95s = f"{s['pooled_p95_ms']:.2f}" if s["pooled_p95_ms"] is not None else f"{s['mean_ep_p95_ms']:.2f}*"
            p99s = f"{s['pooled_p99_ms']:.2f}" if s["pooled_p99_ms"] is not None else f"{s['mean_ep_p99_ms']:.2f}*"
            lines.append(
                f"| {pol} | {s['comp100_count']}/120 ({s['comp100_pct']:.1f}%) | {s['mean_ja']:.1f} | {s['median_ja']:.1f} | "
                f"{s['pure_h1_all_mean']:.1f} | {bs['4topo']['mean_ja']:.1f} | {bs['track1']['mean_ja']:.1f} | {bs['track2']['mean_ja']:.1f} | "
                f"{p95s} | {p99s} | {dlt} | {pval} |"
            )
        lines.append("")
    (OUT_AUDIT_DIR / "v20_final_blind_holdout.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-freeze", action="store_true")
    ap.add_argument("--dry-run-quarantined", action="store_true")
    ap.add_argument("--stage", choices=("cohort_ab", "cohort_c", "all"), default="all")
    args = ap.parse_args()

    if args.dry_run_quarantined:
        sp0 = build_val72_specs()[0]
        r_v20 = run_v20_blind_episode(sp0, hard_budget_ms=35.0, include_decision_latencies=True)
        r_bal = run_discrete_policy_episode_row("BALANS_IJCAI25", "BALANS-IJCAI25", sp0, include_decision_latencies=True)
        print("DRY-RUN V20:", {k: r_v20[k] for k in ("policy", "episode_key", "net_obj", "h1", "comp100", "audit_passed", "mean_decision_ms", "decision_ms_p95", "pdg_alerts")})
        print("DRY-RUN BALANS:", {k: r_bal[k] for k in ("policy", "episode_key", "net_obj", "h1", "comp100", "audit_passed", "mean_decision_ms", "decision_ms_p95")})
        return

    if args.write_freeze:
        man = write_freeze_manifest()
        print(f"Wrote frozen manifest ({len(man['sha256_files'])} files locked) -> {FREEZE_MANIFEST_PATH}")
        return

    verify_freeze_manifest()
    print(f"[IF-V20-FREEZE VERIFIED] Manifest locked: {FREEZE_MANIFEST_PATH}", flush=True)

    ab_policies: list[tuple[str, str]] = [
        ("V20_NAJS", "CoRe-Sched V20"),
        ("V19_ADAPTIVE_RL", "CoRe-Sched"),
        ("V18_EA_JOINT", "V18-EA"),
        ("BALANS_IJCAI25", "BALANS-IJCAI25"),
    ]
    c_policies: list[tuple[str, str]] = [("V20_NAJS", "CoRe-Sched V20")] + list(G3_CORE_12_POLICIES) + [("BALANS_IJCAI25", "BALANS-IJCAI25")]

    ca_res = cb_res = cc_res = None
    if args.stage in ("cohort_ab", "all"):
        specs_a = build_fresh_120_specs()
        ca_rows, ca_lats = _run_cohort("COHORT_A", specs_a, ab_policies)
        _write_csv(OUT_AUDIT_DIR / "v20_cohort_a_rows.csv", [r for rs in ca_rows.values() for r in rs])
        (OUT_AUDIT_DIR / "v20_cohort_a_lats.json").write_text(json.dumps(ca_lats), encoding="utf-8")
        ca_res = (ca_rows, ca_lats)

        specs_b = build_pristine_120_specs()
        cb_rows, cb_lats = _run_cohort("COHORT_B", specs_b, ab_policies)
        _write_csv(OUT_AUDIT_DIR / "v20_cohort_b_rows.csv", [r for rs in cb_rows.values() for r in rs])
        (OUT_AUDIT_DIR / "v20_cohort_b_lats.json").write_text(json.dumps(cb_lats), encoding="utf-8")
        cb_res = (cb_rows, cb_lats)

    if args.stage in ("cohort_c", "all"):
        specs_c = build_cohort_c_120_specs()
        cc_rows, cc_lats = _run_cohort("COHORT_C", specs_c, c_policies)
        _write_csv(OUT_AUDIT_DIR / "v20_cohort_c_rows.csv", [r for rs in cc_rows.values() for r in rs])
        (OUT_AUDIT_DIR / "v20_cohort_c_lats.json").write_text(json.dumps(cc_lats), encoding="utf-8")
        cc_res = (cc_rows, cc_lats)

    rep = _build_report_and_save(cohort_a_fresh=ca_res, cohort_b_fresh=cb_res, cohort_c_fresh=cc_res)
    print("\n=== SAVED FINAL FROZEN BLIND HOLDOUT REPORT ===", flush=True)
    for cname, cdata in rep["cohorts"].items():
        v20s = cdata["summary"].get("V20_NAJS")
        if v20s:
            print(
                f"  {cname}: V20_NAJS mean_ja={v20s['mean_ja']} med={v20s['median_ja']} comp={v20s['comp100_count']}/120 "
                f"H1={v20s['pure_h1_all_mean']} p95={v20s['pooled_p95_ms']}ms p99={v20s['pooled_p99_ms']}ms",
                flush=True,
            )


if __name__ == "__main__":
    main()
