"""Post-v1.2.0 Quarantined VAL-72 Repair Verification, Single-Factor Ablations, and 2x2 Decoupling (`EXP-12`).

Evaluates strictly on the 72 quarantined VALIDATION episodes (`VAL_4TOPO_SEEDS = (16701..16704)` x 12 = 48,
`VAL_WORKSHOP_SEEDS = (1161..1163)` x 8 = 24), completely disjoint from Training, Gate G3 Cohort A, and
untouched Pristine Blind Cohort B.

Runs in a single uncontended wall-clock session (`clock_mode="wall"`, `include_decision_latencies=True`):
  Part A — Single-Factor Component Attribution & Repair Ablations (6 configs):
    1. `CoRe-Sched`            : Full NCSR (`V19_ADAPTIVE_RL`, NumPy AHG-RL + Causal Cone + JA-Unified + SLA Guard)
    2. `w/o Causal Cone`       : `enable_causal_cone_repair=False` (AHG-RL priors only)
    3. `w/o AHG-RL (Falsifier)`: `enable_neural_candidates=False` (pure Cone + Regret-2 surrogate without NN priors)
    4. `w/o Zero-Churn Proj`   : `enable_zero_churn_projection=False`
    5. `w/o JA-Unified Scorer` : `use_ja_aligned_scorer=False` (legacy surrogate evaluation)
    6. `V18-EA (Ref)`          : Strong heuristic + search baseline (`V18_EA_JOINT`)

  Part B — Exact Orthogonal 2x2 Trigger x Solver Decoupling Matrix:
    - Row/Controller: `{RuleV10PortfolioPolicy, V18AdaptiveController}`
    - Col/Solver:     `{CoRe-Sched, V18-EA}`
    - Diagonal cells reuse `CoRe-Sched` (`RuleV10 + CoRe-Sched`) and `V18-EA (Ref)` (`V18Ctrl + V18-EA`).
    - Off-diagonal cells execute `V18Ctrl + CoRe-Sched` and `RuleV10 + V18-EA` under equal wall-clock measurement.

Writes:
  - `results/g4_val72_repair_ablation.csv`
  - `out/v19_scientific_audit/exp12_val72_repair_and_ablation.{json,md}`
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Callable, Optional

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "_scratch"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import (
    _V17_CKPT,
    _V19_RL_CKPT,
    _V19_STAGED_RL_CKPT,
    build_external_proposer,
    create_discrete_policy_env,
    load_internal_cvcr_policy,
    resolve_policy_name,
    run_discrete_episode,
)
from amr_sched.cvcr.policy import LGCVCRPolicy
from amr_sched.cvcr.v12_proactive_guard import compute_detour_delta_signals
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from amr_sched.rl.v19_hetero_constructor import V19AdaptiveCouplingProposer
from experiments.train_v19_bc import (
    FORBIDDEN_BLIND_SEEDS,
    FORBIDDEN_DEV_SEEDS,
    TRAIN_4TOPO_SEEDS,
    TRAIN_WORKSHOP_SEEDS,
    VAL_4TOPO_SEEDS,
    VAL_WORKSHOP_SEEDS,
    build_val_specs,
)

OUT_DIR = ROOT / "out" / "v19_scientific_audit"
RESULTS_DIR = ROOT / "results"

COHORT_A_SEEDS = {27101, 27102, 27201, 27202, 27301, 27302, 2901, 2902, 2903, 2904, 2905, 2906}
COHORT_B_SEEDS = {37101, 37102, 37201, 37202, 37301, 37302, 3901, 3902, 3903, 3904, 3905, 3906}


def assert_val_quarantine() -> None:
    val_seeds = set(VAL_4TOPO_SEEDS) | set(VAL_WORKSHOP_SEEDS)
    bad = val_seeds & (
        set(TRAIN_4TOPO_SEEDS)
        | set(TRAIN_WORKSHOP_SEEDS)
        | set(FORBIDDEN_BLIND_SEEDS)
        | set(FORBIDDEN_DEV_SEEDS)
        | COHORT_A_SEEDS
        | COHORT_B_SEEDS
    )
    if bad:
        raise RuntimeError(f"VAL seed quarantine violation: {sorted(bad)}")


def _h1(r: dict[str, Any]) -> float:
    return float(r["cmax"] + r["task_tardiness_cost"] + r["op_tardiness_cost"])


def _comp100(r: dict[str, Any]) -> bool:
    return bool(float(r.get("task_comp", 0.0)) >= 0.9999 and float(r.get("prod_comp", 0.0)) >= 0.9999)


def _percentile(sorted_vals: list[float], q: float) -> float:
    if not sorted_vals:
        return 0.0
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    pos = float(q) * float(len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = min(len(sorted_vals) - 1, lo + 1)
    frac = pos - float(lo)
    return float(sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac)


import math  # used in _percentile


def _scenario_digest(sc: tuple[Any, ...], seed: int, p2_cfg: Any) -> str:
    payload = repr((sc[0], sc[1], sc[2], len(sc[3]), len(sc[4]), seed, p2_cfg)).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _run_custom_episode(
    *,
    variant_id: str,
    trigger_kind: str,
    solver_kind: str,
    spec: dict[str, Any],
    ext_proposer: Optional[Callable[..., Any]] = None,
    v18_pol: Any = None,
) -> dict[str, Any]:
    """Run one discrete episode with clean Trigger x Solver wiring and wall-clock telemetry."""
    scenario_copy = copy.deepcopy(spec["scenario_tuple"])
    p2_copy = copy.deepcopy(spec["p2_cfg"])
    seed = int(spec["seed"])
    max_steps = int(spec["max_steps"])
    episode_key = str(spec["episode_key"])
    topo = str(scenario_copy[0])

    if trigger_kind == "V18Ctrl" and solver_kind == "V18-EA":
        rec = run_discrete_episode(
            "V18_EA_JOINT",
            scenario_copy,
            seed=seed,
            max_steps=max_steps,
            use_rust=True,
            p2_cfg=p2_copy,
            episode_key=episode_key,
            clock_mode="wall",
            include_decision_latencies=True,
        )
        rec["variant"] = variant_id
        rec["trigger"] = trigger_kind
        rec["solver"] = solver_kind
        return rec

    if trigger_kind == "RuleV10" and solver_kind == "CoRe-Sched" and ext_proposer is None:
        rec = run_discrete_episode(
            "V19_ADAPTIVE_RL",
            scenario_copy,
            seed=seed,
            max_steps=max_steps,
            use_rust=True,
            p2_cfg=p2_copy,
            episode_key=episode_key,
            clock_mode="wall",
            include_decision_latencies=True,
        )
        rec["variant"] = variant_id
        rec["trigger"] = trigger_kind
        rec["solver"] = solver_kind
        return rec

    # Custom proposer (ablation) OR off-diagonal 2x2 cell
    if trigger_kind == "V18Ctrl":
        env_policy = "V18_EA_JOINT"
    else:
        env_policy = "V19_ADAPTIVE_RL"

    kind, ctrl, env = create_discrete_policy_env(
        env_policy,
        topology=topo,
        seed=seed,
        max_steps=max_steps,
        use_rust=True,
        p2_cfg=p2_copy,
        allow_missing_pgmpo=False,
    )
    if trigger_kind == "RuleV10" and solver_kind == "V18-EA":
        # Drive V18-EA solver from RuleV10PortfolioPolicy triggers
        pol = v18_pol if v18_pol is not None else load_internal_cvcr_policy("V18_EA_JOINT", allow_missing_pgmpo=False)

        def _v18_ext_proposer(snap: Any, network: Any, *, budget_ms: float = 35.0) -> Any:
            dd_sig = compute_detour_delta_signals(
                network,
                snap,
                simulator=env.sim,
                stall_threshold=env.loaded_repair_threshold,
            )
            pol.active_subscope = None
            p = LGCVCRPolicy(
                "V18_EA_JOINT",
                network,
                use_rust=True,
                bay_detour_delta=(dd_sig["bay_detour_delta"] if dd_sig is not None else None),
                detour_weights_q3=(dd_sig["weights_q3"] if dd_sig is not None else None),
                v15_policy=pol,
            )
            p._prev_ltm_weights = dict(getattr(env.sim, "_traffic_weights_q3", {}))
            return p.propose(snap, budget_ms=budget_ms)

        env.external_proposer = _v18_ext_proposer
        obs_backend = "numpy"
        ckpt_sha = getattr(pol, "_checkpoint_sha256", None)
    else:
        assert ext_proposer is not None
        if hasattr(ext_proposer, "_ensure_nets"):
            ext_proposer._ensure_nets()
        env.external_proposer = ext_proposer
        obs_backend = ext_proposer._effective_backend() if hasattr(ext_proposer, "_effective_backend") else "numpy"
        ckpt_sha = getattr(ext_proposer, "_checkpoint_sha256", None)

    wall_t0 = time.perf_counter()
    obs, info = env.reset(seed=seed, options={"scenario_tuple": scenario_copy})
    tm = False
    tr = False
    while not (tm or tr):
        if kind == "rule_v10":
            feats = compute_expert_v9_state(
                env.sim,
                env.tracker,
                last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks,
                decision_logs=env.decision_logs,
            )
            step_info = dict(
                info,
                action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick),
            )
        elif kind == "v18":
            feats = ctrl.observe(env, obs)
            step_info = info
        else:
            feats = env.get_expert_v12_obs(obs)
            step_info = info
        act = ctrl.select_action(feats, step_info, sim=env.sim)
        obs, _, tm, tr, info = env.step(act)

    wall_ms = (time.perf_counter() - wall_t0) * 1000.0
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
    rec = {
        "variant": variant_id,
        "trigger": trigger_kind,
        "solver": solver_kind,
        "policy": variant_id,
        "requested_policy": variant_id,
        "inference_backend": obs_backend,
        "checkpoint_sha256": ckpt_sha,
        "episode_key": episode_key,
        "seed": seed,
        "audit_passed": bool(aud.pass_audit),
        "ticks": int(aud.total_steps),
        "net_obj": round(float(m["net_obj"]), 4),
        "total_obj": round(float(m["total_obj"]), 4),
        "cmax": round(float(m["cmax"]), 4),
        "task_tardiness_cost": round(float(m.get("task_tardiness_cost", 0.0)), 4),
        "op_tardiness_cost": round(float(m.get("op_tardiness_cost", 0.0)), 4),
        "unfinished_task_penalty": round(float(m.get("unfinished_task_penalty", 0.0)), 4),
        "unfinished_op_penalty": round(float(m.get("unfinished_op_penalty", 0.0)), 4),
        "churn_cost": round(float(m.get("churn_cost", 15.0 * float(m.get("total_churn", 0)))), 4),
        "compute_cost": round(float(m.get("compute_cost", 0.05 * float(m.get("total_compute_ms", 0.0)))), 4),
        "churn": int(m["total_churn"]),
        "replans": int(m["total_replans"]),
        "task_comp": round(float(m["task_completion_rate"]), 6),
        "prod_comp": round(float(m["production_completion_rate"]), 6),
        "compute_ms": round(float(m["total_compute_ms"]), 3),
        "wall_ms": round(wall_ms, 2),
        "decision_count": len(decision_ms),
        "mean_decision_ms": round(sum(decision_ms) / len(decision_ms), 3) if decision_ms else 0.0,
        "decision_ms_p95": round(_percentile(decision_sorted, 0.95), 3),
        "decision_ms_p99": round(_percentile(decision_sorted, 0.99), 3),
        "expired_windows": int(env.total_window_expirations),
        "decision_latencies_ms": [round(float(v), 4) for v in decision_ms],
    }
    return rec


def _paired_vs(core_rows: list[dict[str, Any]], base_rows: list[dict[str, Any]]) -> dict[str, Any]:
    cb = {r["episode_key"]: r for r in core_rows}
    bb = {r["episode_key"]: r for r in base_rows}
    keys = sorted(set(cb) & set(bb))
    c_ja = np.array([float(cb[k]["net_obj"]) for k in keys], dtype=np.float64)
    b_ja = np.array([float(bb[k]["net_obj"]) for k in keys], dtype=np.float64)
    diff = b_ja - c_ja
    pct = 100.0 * float(np.mean(b_ja) - np.mean(c_ja)) / max(1e-9, float(np.mean(b_ja)))
    if np.all(np.abs(diff) < 1e-9):
        p_val = 1.0
    else:
        p_val = float(wilcoxon(b_ja, c_ja, zero_method="wilcox", alternative="two-sided").pvalue)
    rng = np.random.default_rng(20261124)
    idx = rng.integers(0, len(keys), size=(2000, len(keys)))
    boots = 100.0 * (np.mean(b_ja[idx], axis=1) - np.mean(c_ja[idx], axis=1)) / np.maximum(1e-9, np.mean(b_ja[idx], axis=1))
    ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
    wins = int(np.sum(diff > 1e-6))
    losses = int(np.sum(diff < -1e-6))
    ties = len(keys) - wins - losses

    mut_keys = [k for k in keys if _comp100(cb[k]) and _comp100(bb[k])]
    if mut_keys:
        ch1 = float(np.mean([_h1(cb[k]) for k in mut_keys]))
        bh1 = float(np.mean([_h1(bb[k]) for k in mut_keys]))
        h1_gain = 100.0 * (bh1 - ch1) / max(1e-9, bh1)
    else:
        ch1 = bh1 = h1_gain = 0.0
    return {
        "n": len(keys),
        "delta_ja_pct": round(pct, 2),
        "ci95_lo": round(ci_lo, 2),
        "ci95_hi": round(ci_hi, 2),
        "wilcoxon_p": float(f"{p_val:.4e}"),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "mutual_n": len(mut_keys),
        "core_h1_mutual": round(ch1, 1),
        "base_h1_mutual": round(bh1, 1),
        "h1_gain_mutual_pct": round(h1_gain, 2),
    }


def _summarize(rows: list[dict[str, Any]], lats: list[float]) -> dict[str, Any]:
    s_lats = sorted(float(x) for x in lats)
    return {
        "n": len(rows),
        "audit_pass": f"{sum(1 for r in rows if r['audit_passed'])}/{len(rows)}",
        "comp100": f"{sum(1 for r in rows if _comp100(r))}/{len(rows)}",
        "ja_mean": round(float(np.mean([r["net_obj"] for r in rows])), 1),
        "h1_mean": round(float(np.mean([_h1(r) for r in rows])), 1),
        "churn_mean": round(float(np.mean([r["churn"] for r in rows])), 1),
        "expired_windows": sum(int(r["expired_windows"]) for r in rows),
        "pooled_p95_ms": round(_percentile(s_lats, 0.95), 2),
        "pooled_p99_ms": round(_percentile(s_lats, 0.99), 2),
        "ep_p95_ms": round(float(np.mean([r["decision_ms_p95"] for r in rows])), 2),
    }


def main() -> None:
    assert_val_quarantine()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    specs = build_val_specs()
    assert len(specs) == 72

    # Warmup outside timing
    for pol in ("rule_v10", "V19_ADAPTIVE_RL", "V18_EA_JOINT"):
        _ = run_discrete_episode(
            pol,
            copy.deepcopy(specs[0]["scenario_tuple"]),
            seed=specs[0]["seed"],
            max_steps=specs[0]["max_steps"],
            use_rust=True,
            p2_cfg=specs[0]["p2_cfg"],
            episode_key=f"warmup_{pol}",
            clock_mode="wall",
            include_decision_latencies=False,
        )

    def _make_ablation_proposer(**overrides: Any) -> V19AdaptiveCouplingProposer:
        p = V19AdaptiveCouplingProposer(
            full_ckpt_path=_V19_RL_CKPT,
            staged_ckpt_path=_V19_STAGED_RL_CKPT,
            policy_id="V19_ADAPTIVE_RL",
            mode="bandit",
            backend="auto",
            use_rust=True,
            seed=0,
            evaluator_mode="B2_SEEDED",
            **overrides,
        )
        p._ensure_nets()
        return p

    v18_pol_shared = load_internal_cvcr_policy("V18_EA_JOINT", allow_missing_pgmpo=False)

    variants: list[tuple[str, str, str, Callable[[], Optional[Any]]]] = [
        ("CoRe-Sched", "RuleV10", "CoRe-Sched", lambda: None),
        ("CoRe w/o Cone", "RuleV10", "CoRe-Sched", lambda: _make_ablation_proposer(enable_causal_cone_repair=False)),
        ("CoRe w/o AHG-RL", "RuleV10", "CoRe-Sched", lambda: _make_ablation_proposer(enable_neural_candidates=False)),
        ("CoRe w/o ZeroChurn", "RuleV10", "CoRe-Sched", lambda: _make_ablation_proposer(enable_zero_churn_projection=False)),
        ("CoRe Legacy-Scorer", "RuleV10", "CoRe-Sched", lambda: _make_ablation_proposer(use_ja_aligned_scorer=False)),
        ("V18-EA", "V18Ctrl", "V18-EA", lambda: None),
        ("V18Ctrl + CoRe-Sched", "V18Ctrl", "CoRe-Sched", lambda: build_external_proposer("V19_ADAPTIVE_RL", seed=0, use_rust=True)),
        ("RuleV10 + V18-EA", "RuleV10", "V18-EA", lambda: None),
    ]

    by_var: dict[str, list[dict[str, Any]]] = {}
    lats_by_var: dict[str, list[float]] = {}
    csv_rows: list[dict[str, Any]] = []

    t_all = time.perf_counter()
    for vid, trig, solv, prop_factory in variants:
        t0 = time.perf_counter()
        vrows: list[dict[str, Any]] = []
        vlats: list[float] = []
        ext_prop = prop_factory()
        for sp in specs:
            rec = _run_custom_episode(
                variant_id=vid,
                trigger_kind=trig,
                solver_kind=solv,
                spec=sp,
                ext_proposer=ext_prop,
                v18_pol=v18_pol_shared,
            )
            rec["suite"] = sp["suite"]
            rec["layout"] = sp["layout"]
            rec["regime"] = sp["regime"]
            vlats.extend(rec["decision_latencies_ms"])
            row_slim = {k: v for k, v in rec.items() if k != "decision_latencies_ms"}
            vrows.append(row_slim)
            csv_rows.append(row_slim)
        by_var[vid] = vrows
        lats_by_var[vid] = vlats
        s = _summarize(vrows, vlats)
        dt = time.perf_counter() - t0
        print(
            f"[{vid:22s}] {dt:5.1f}s | comp={s['comp100']} aud={s['audit_pass']} | "
            f"JA={s['ja_mean']:7.1f} H1={s['h1_mean']:7.1f} ch={s['churn_mean']:4.1f} wE={s['expired_windows']} | "
            f"PoolP95={s['pooled_p95_ms']:5.2f} P99={s['pooled_p99_ms']:5.2f} EpP95={s['ep_p95_ms']:5.2f}",
            flush=True,
        )

    csv_path = RESULTS_DIR / "g4_val72_repair_ablation.csv"
    fieldnames = [
        "variant", "trigger", "solver", "suite", "layout", "regime", "seed", "episode_key",
        "audit_passed", "ticks", "task_comp", "prod_comp", "net_obj", "total_obj", "cmax",
        "task_tardiness_cost", "op_tardiness_cost", "unfinished_task_penalty", "unfinished_op_penalty",
        "churn", "churn_cost", "replans", "compute_ms", "compute_cost", "wall_ms", "decision_count",
        "mean_decision_ms", "decision_ms_p95", "decision_ms_p99", "expired_windows", "inference_backend",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in csv_rows:
            w.writerow({k: r.get(k) for k in fieldnames})

    core_rows = by_var["CoRe-Sched"]
    summary_all: dict[str, Any] = {}
    paired_all: dict[str, Any] = {}
    by_suite: dict[str, dict[str, Any]] = {}
    for vid, vrows in by_var.items():
        summary_all[vid] = _summarize(vrows, lats_by_var[vid])
        if vid != "CoRe-Sched":
            paired_all[vid] = _paired_vs(core_rows, vrows)
        suite_map: dict[str, Any] = {}
        for sk in ("topo4", "track1", "track2"):
            sub = [r for r in vrows if r["suite"] == sk]
            suite_map[sk] = {
                "n": len(sub),
                "comp100": f"{sum(1 for r in sub if _comp100(r))}/{len(sub)}",
                "ja_mean": round(float(np.mean([r["net_obj"] for r in sub])), 1),
                "h1_mean": round(float(np.mean([_h1(r) for r in sub])), 1),
                "churn_mean": round(float(np.mean([r["churn"] for r in sub])), 1),
                "expired_windows": sum(int(r["expired_windows"]) for r in sub),
            }
        by_suite[vid] = suite_map

    out_json = {
        "experiment": "EXP-12-VAL72-REPAIR-ABLATION-DECOUPLING",
        "val_seeds_4topo": list(VAL_4TOPO_SEEDS),
        "val_seeds_workshop": list(VAL_WORKSHOP_SEEDS),
        "episodes_per_variant": len(specs),
        "wall_seconds_total": round(time.perf_counter() - t_all, 1),
        "summary": summary_all,
        "paired_vs_core_sched": paired_all,
        "by_suite": by_suite,
    }
    json_path = OUT_DIR / "exp12_val72_repair_and_ablation.json"
    json_path.write_text(json.dumps(out_json, indent=2), encoding="utf-8")

    md_lines = [
        "# EXP-12: Quarantined VAL-72 Repair, Component Ablations & 2x2 Trigger x Solver Decoupling",
        "",
        "| Variant | Trigger | Solver | Audit | Comp 100% | Mean J_A | Mean H1 | 4-Topo J_A | Track 1 J_A | Track 2 J_A | Churn | wE | Pool P95 | Pool P99 | Ep P95 | vs CoRe-Sched (Delta J_A) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for vid, trig, solv, _ in variants:
        s = summary_all[vid]
        bs = by_suite[vid]
        if vid == "CoRe-Sched":
            vs_txt = "baseline"
        else:
            p = paired_all[vid]
            vs_txt = f"{p['delta_ja_pct']:+.2f}% (p={p['wilcoxon_p']}, W/T/L={p['wins']}/{p['ties']}/{p['losses']})"
        md_lines.append(
            f"| {vid} | {trig} | {solv} | {s['audit_pass']} | {s['comp100']} | {s['ja_mean']:.1f} | {s['h1_mean']:.1f} | "
            f"{bs['topo4']['ja_mean']:.1f} ({bs['topo4']['comp100']}) | "
            f"{bs['track1']['ja_mean']:.1f} ({bs['track1']['comp100']}) | "
            f"{bs['track2']['ja_mean']:.1f} ({bs['track2']['comp100']}) | "
            f"{s['churn_mean']:.1f} | {s['expired_windows']} | {s['pooled_p95_ms']:.2f} ms | {s['pooled_p99_ms']:.2f} ms | {s['ep_p95_ms']:.2f} ms | {vs_txt} |"
        )
    (OUT_DIR / "exp12_val72_repair_and_ablation.md").write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"Wrote {csv_path} and {json_path}")


if __name__ == "__main__":
    main()
