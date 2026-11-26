"""Phase V20-E: Gate E Verification — PDG Liveness on VAL-72, VAL-Static-35, Extreme Stress, & RQ4 Case Studies."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    assert_stage1_quarantine,
    build_task_c_outlier_specs,
    build_val72_specs,
    build_val_static35_specs,
    run_wrapped_episode,
    summarize_episode_rows,
)
from experiments.v20_audit.v20_e_liveness import (
    PDGLivenessTracker,
    V20EProposer,
    build_default_v20_proposer,
)


def _run_v20e_ep(vid: str, sp: dict[str, Any], *, hard_budget_ms: float = 35.0, enable_pdg: bool = True) -> dict[str, Any]:
    prop = build_default_v20_proposer(hard_budget_ms=hard_budget_ms, seed=0, enable_pdg_recovery=enable_pdg)
    tracker = PDGLivenessTracker(cooldown_ticks=12)

    def _hook(env, info, act, traces):
        return tracker.observe_and_maybe_escalate(env, info, act, traces, proposer=prop if enable_pdg else None)

    row, _ = run_wrapped_episode(
        variant_id=vid,
        spec=sp,
        base_env_policy="V19_ADAPTIVE_RL",
        external_proposer=prop,
        hard_budget_ms=hard_budget_ms,
        outer_action_hook=_hook,
    )
    row["pdg_alert_ticks"] = len(tracker.alert_ticks)
    row["pdg_first_alert_tick"] = tracker.alert_ticks[0] if tracker.alert_ticks else None
    row["pdg_escalations"] = len(tracker.escalation_ticks)
    row["pdg_inner_recoveries"] = int(prop.pdg_recoveries_triggered)
    return row


def _run_takeover_replay(sp: dict[str, Any], t_takeover: int) -> dict[str, Any]:
    """Run baseline V19_ADAPTIVE_RL prefix up to `t_takeover`, then switch to V20-E PDG recovery."""
    prop_v20e = build_default_v20_proposer(hard_budget_ms=35.0, seed=0, enable_pdg_recovery=True)
    tracker = PDGLivenessTracker(cooldown_ticks=10)
    orig_ext_ref: list[Any] = [None]

    def _dispatch(snap, net, *, budget_ms: float = 35.0):
        if int(snap.tick) < int(t_takeover):
            assert orig_ext_ref[0] is not None
            return orig_ext_ref[0](snap, net, budget_ms=budget_ms)
        return prop_v20e(snap, net, budget_ms=budget_ms)

    def _hook(env, info, act, traces):
        # Record underlying policy external proposer on first hook
        if orig_ext_ref[0] is None:
            # env.external_proposer was already wrapped by run_wrapped_episode; check base env creation cleanly
            pass
        esc = tracker.observe_and_maybe_escalate(env, info, act, traces, proposer=prop_v20e)
        if int(env.current_step) < int(t_takeover):
            return int(act)
        if int(env.current_step) == int(t_takeover) and int(esc) == 0:
            mask = info.get("action_mask")
            if mask is not None:
                legal = [i for i in range(1, len(mask)) if int(mask[i]) == 1]
                if legal:
                    return int(legal[-1])
            return 2
        return int(esc)

    # Instantiate native V19 proposer separately for strict prefix parity < t_takeover
    from amr_sched.baselines.policy_registry import create_discrete_policy_env

    _, _, tmp_env = create_discrete_policy_env(
        "V19_ADAPTIVE_RL", topology=str(sp["scenario_tuple"][0]), seed=int(sp["seed"]), max_steps=int(sp["max_steps"]), use_rust=True
    )
    orig_ext_ref[0] = tmp_env.external_proposer
    row, _ = run_wrapped_episode(
        variant_id=f"takeover_{t_takeover}",
        spec=sp,
        base_env_policy="V19_ADAPTIVE_RL",
        external_proposer=_dispatch,
        hard_budget_ms=35.0,
        outer_action_hook=_hook,
    )
    return {
        "episode_key": sp["episode_key"],
        "t_takeover": int(t_takeover),
        "net_obj": row["net_obj"],
        "h1": row["h1"],
        "cmax": row["cmax"],
        "comp100": row["comp100"],
        "task_comp": row["task_comp"],
        "prod_comp": row["prod_comp"],
        "audit_passed": row["audit_passed"],
        "pdg_first_alert": tracker.alert_ticks[0] if tracker.alert_ticks else None,
        "pdg_escalations": len(tracker.escalation_ticks),
    }


def _build_val_extreme_stress_12() -> list[dict[str, Any]]:
    """Build 12 extreme perturbation stress cases strictly on quarantined VAL seeds."""
    s72 = build_val72_specs()
    picks = [
        copy.deepcopy(s)
        for s in s72
        if s["regime"] in ("execution_blockage", "kitting_windows", "multi_wave_surge")
        and s["episode_key"]
        in (
            "topo4_offset_dual_lane_execution_blockage_16701",
            "topo4_offset_dual_lane_window_bottleneck_16702",
            "topo4_narrow_spine_execution_blockage_16703",
            "topo4_narrow_spine_execution_blockage_16704",
            "topo4_workshop_cells_execution_blockage_16702",
            "topo4_workshop_cells_window_bottleneck_16702",
            "track1_kitting_windows_1162",
            "track1_execution_blockage_1163",
            "track1_multi_wave_surge_1161",
            "track2_kitting_windows_1161",
            "track2_execution_blockage_1162",
            "track2_multi_wave_surge_1163",
        )
    ]
    # Also include 2 offset_dual_lane window_bottleneck if missing above
    if len(picks) < 12:
        for s in s72:
            if len(picks) >= 12:
                break
            if s["episode_key"] not in {p["episode_key"] for p in picks} and "window_bottleneck" in s["episode_key"]:
                picks.append(copy.deepcopy(s))
    for i, sp in enumerate(picks):
        sp["episode_key"] = f"stress_{sp['episode_key']}"
        sp["p2_cfg"] = P2PerturbationConfig(
            seed=int(sp["seed"]) + 500 + i,
            enable_travel_drift=True,
            drift_prob=0.52,
            early_max_steps=2,
            late_max_steps=4,
            enable_dynamic_obstacles=True,
            obstacle_count=5,
            obstacle_duration_steps=24,
            obstacle_window_range=(10, 95),
            enable_local_planner_fail=True,
            local_fail_count=4,
            local_fail_duration_steps=16,
            local_fail_window_range=(12, 95),
        )
    assert_stage1_quarantine([sp["seed"] for sp in picks], label="V20E_Extreme_VAL_Stress")
    return picks


def run_v20_e_gate_eval() -> dict[str, Any]:
    OUT_AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    specs72 = build_val72_specs()
    specs35 = build_val_static35_specs()
    assert_stage1_quarantine([sp["seed"] for sp in specs72 + specs35], label="V20E_GateE_VAL")

    print("=== [V20-E] Running VAL-72 (zero false positive PDG & regression check) ===", flush=True)
    rows72 = [_run_v20e_ep("V20E_VAL72", sp, hard_budget_ms=35.0, enable_pdg=True) for sp in specs72]
    ov72 = summarize_episode_rows(rows72)
    fpr72 = sum(1 for r in rows72 if r["pdg_alert_ticks"] > 0)
    by_suite72 = {g: summarize_episode_rows([r for r in rows72 if r["suite_group"] == g]) for g in ("4topo", "track1", "track2")}
    print(
        f"  V20-E VAL-72: net={ov72['mean_net_obj']} comp={ov72['comp100_count']}/72 audit={ov72['audit_pass_rate']} "
        f"p95={ov72['pooled_p95_ms']}ms PDG_FPR={fpr72}/72 | 4T={by_suite72['4topo']['mean_net_obj']} "
        f"T1={by_suite72['track1']['mean_net_obj']} T2={by_suite72['track2']['mean_net_obj']}",
        flush=True,
    )

    print("=== [V20-E] Running VAL-Static-35 ===", flush=True)
    rows35 = [_run_v20e_ep("V20E_Static35", sp, hard_budget_ms=35.0, enable_pdg=True) for sp in specs35]
    ov35 = summarize_episode_rows(rows35)
    fpr35 = sum(1 for r in rows35 if r["pdg_alert_ticks"] > 0)
    print(
        f"  V20-E VAL-Static-35: H1={ov35['mean_h1_all']} net={ov35['mean_net_obj']} comp={ov35['comp100_count']}/35 "
        f"p95={ov35['pooled_p95_ms']}ms PDG_FPR={fpr35}/35",
        flush=True,
    )

    print("=== [V20-E] Running VAL Extreme Disruption Stress (12 episodes: V19 vs V20-B vs V20-E) ===", flush=True)
    stress_specs = _build_val_extreme_stress_12()
    r_st_v19 = [run_wrapped_episode(variant_id="stress_v19", spec=sp, base_env_policy="V19_ADAPTIVE_RL", hard_budget_ms=35.0)[0] for sp in stress_specs]
    r_st_v20b = [_run_v20e_ep("stress_v20b", sp, hard_budget_ms=35.0, enable_pdg=False) for sp in stress_specs]
    r_st_v20e = [_run_v20e_ep("stress_v20e", sp, hard_budget_ms=35.0, enable_pdg=True) for sp in stress_specs]
    for nm, rs in (("V19_ADAPTIVE_RL", r_st_v19), ("V20B_NoPDG", r_st_v20b), ("V20E_WithPDG", r_st_v20e)):
        ovs = summarize_episode_rows(rs)
        pdg_trig = sum(1 for x in rs if x.get("pdg_inner_recoveries", 0) > 0)
        print(
            f"  Stress {nm:16s}: net={ovs['mean_net_obj']:8.2f} comp={ovs['comp100_count']}/12 "
            f"p95={ovs['pooled_p95_ms']:6.3f}ms pdg_recoveries={pdg_trig}",
            flush=True,
        )

    print("=== [V20-E / RQ4] Task-C Cohort B Outlier Liveness Case Studies (37302 & 3906) ===", flush=True)
    outliers = build_task_c_outlier_specs()
    rq4_cases: dict[str, Any] = {}
    for sp in outliers:
        ek = sp["episode_key"]
        r_full_v19 = run_wrapped_episode(variant_id="v19_out", spec=sp, base_env_policy="V19_ADAPTIVE_RL", hard_budget_ms=35.0)[0]
        r_full_v20e = _run_v20e_ep("v20e_out", sp, hard_budget_ms=35.0, enable_pdg=True)
        take_points = (102, 120, 127) if "37302" in ek else (325, 335, 345, 755)
        take_res = [_run_takeover_replay(sp, tk) for tk in take_points]
        rq4_cases[ek] = {
            "v19_full": {
                "net_obj": r_full_v19["net_obj"], "cmax": r_full_v19["cmax"],
                "comp100": r_full_v19["comp100"], "task_comp": r_full_v19["task_comp"], "prod_comp": r_full_v19["prod_comp"],
            },
            "v20e_full": {
                "net_obj": r_full_v20e["net_obj"], "cmax": r_full_v20e["cmax"],
                "comp100": r_full_v20e["comp100"], "task_comp": r_full_v20e["task_comp"], "prod_comp": r_full_v20e["prod_comp"],
                "pdg_alerts": r_full_v20e["pdg_alert_ticks"], "pdg_escalations": r_full_v20e["pdg_escalations"],
            },
            "v19_to_v20e_takeover_boundary": take_res,
        }
        print(f"  Outlier {ek}:", flush=True)
        print(f"    V19  full -> net={r_full_v19['net_obj']:.1f} comp={r_full_v19['comp100']} prod={r_full_v19['prod_comp']:.3f}", flush=True)
        print(f"    V20E full -> net={r_full_v20e['net_obj']:.1f} comp={r_full_v20e['comp100']} prod={r_full_v20e['prod_comp']:.3f}", flush=True)
        for tr in take_res:
            print(f"    Takeover t={tr['t_takeover']:3d} -> net={tr['net_obj']:8.1f} comp={tr['comp100']} prod={tr['prod_comp']:.3f} esc={tr['pdg_escalations']}", flush=True)

    gate_e_pass = bool(
        ov72["comp100_count"] == 72
        and ov72["audit_pass_rate"] >= 0.9999
        and ov72["pooled_p95_ms"] <= 28.0
        and fpr72 == 0
        and ov35["comp100_count"] == 35
        and ov35["pooled_p95_ms"] <= 28.0
        and fpr35 == 0
        and all(rq4_cases[k]["v20e_full"]["comp100"] for k in rq4_cases)
    )
    print(f"=== [V20-E / Gate E] PASS={gate_e_pass} (VAL72 FPR={fpr72}/72, Static35 FPR={fpr35}/35) ===", flush=True)

    report = {
        "schema": "v20_e.liveness_gate_eval.v1",
        "val72": ov72,
        "val72_by_suite": by_suite72,
        "val72_pdg_false_positive_episodes": fpr72,
        "val_static_35": ov35,
        "val_static_35_pdg_false_positive_episodes": fpr35,
        "val_extreme_stress_12": {
            "V19_ADAPTIVE_RL": summarize_episode_rows(r_st_v19),
            "V20B_NoPDG": summarize_episode_rows(r_st_v20b),
            "V20E_WithPDG": summarize_episode_rows(r_st_v20e),
        },
        "rq4_outlier_recovery": rq4_cases,
        "gate_e_pass": gate_e_pass,
    }
    (OUT_AUDIT_DIR / "v20_e_gate_eval.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    run_v20_e_gate_eval()
