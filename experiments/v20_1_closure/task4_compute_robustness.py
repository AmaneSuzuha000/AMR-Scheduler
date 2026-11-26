"""Task 4 (F4): Three-Protocol Compute Robustness Frontier (`10 / 20 / 35 / 50 / 100 ms`).

Evaluates `V18_EA_JOINT`, `V19_ADAPTIVE_RL`, and `V20_NAJS` (plus optional `V20_PDG2`) under:
  - Protocol P1 (`Deterministic Work Budget`): `virtual_clock_scope(poll_ms=0.002)` calibrated work clock
    across `10 / 20 / 35 / 50 / 100 ms`, verifying bit-identical repeatability.
  - Protocol P2 (`Wall-Clock Budget`): Uncontended single-process `clock_mode="wall"` across all 5 budgets
    plus 5-repeat variance runs on 4 high-volatility VAL instances.
  - Protocol P3 (`Compute-Jitter Stress`): Deterministic seeded compute-jitter injection (log-normal
    slowdown + 15% +6 ms preemption spikes inside hard deadline enforcement) measuring overrun rate,
    max latency, candidate truncation rate, CVaR_95(JA), and safe KEEP fallback integrity.

Writes Deliverable 4 (`out/v20_1_closure/deliverable4_compute_robustness.{json,md}`).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.baselines.policy_registry import create_discrete_policy_env
from amr_sched.cvcr.clock import now_s as _cvcr_now_s, record_work_ms as _cvcr_rec_work, virtual_clock_scope
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from experiments.v20_audit.common import (
    assert_stage1_quarantine,
    build_val72_specs,
    make_v18_external_proposer,
)
from experiments.v20_audit.v20_e_liveness import PDGLivenessTracker, build_default_v20_proposer

OUT_CLOSURE_DIR = ROOT / "out" / "v20_1_closure"
BUDGETS_MS: tuple[float, ...] = (10.0, 20.0, 35.0, 50.0, 100.0)


def _cvar(vals: list[float], alpha: float = 0.95) -> float:
    if not vals:
        return 0.0
    s = np.sort(np.asarray(vals, dtype=np.float64))
    k = max(1, int(np.ceil((1.0 - alpha) * s.size)))
    return float(np.mean(s[-k:]))


def _select_stratified_val36() -> list[dict[str, Any]]:
    """Balanced 36-episode VAL subset covering all 4-Topo layouts/regimes (24) + Track 1 (6) + Track 2 (6)."""
    s72 = build_val72_specs()
    picks = [s for s in s72 if s["suite_group"] == "4topo" and int(s["seed"]) in (16701, 16702)]
    picks += [s for s in s72 if s["suite_group"] == "track1"][:6]
    picks += [s for s in s72 if s["suite_group"] == "track2"][:6]
    assert len(picks) == 36
    assert_stage1_quarantine([s["seed"] for s in picks], label="Task4_VAL36")
    return picks


class _null_ctx:
    def __enter__(self):
        return None
    def __exit__(self, *args):
        return False


def _run_protocol_episode(
    method: str,
    sp: dict[str, Any],
    *,
    budget_ms: float,
    protocol: str,
    jitter_seed: int = 0,
) -> dict[str, Any]:
    """Execute one episode of `V18_EA_JOINT`, `V19_ADAPTIVE_RL`, or `V20_NAJS` under P1, P2, or P3."""
    sc_copy = copy.deepcopy(sp["scenario_tuple"])
    p2_copy = copy.deepcopy(sp["p2_cfg"])
    seed = int(sp["seed"])
    max_steps = int(sp["max_steps"])
    topo = str(sc_copy[0])

    base_pol = "V19_ADAPTIVE_RL" if method in ("V19_ADAPTIVE_RL", "V20_NAJS", "V18_EA_JOINT") else method
    _, ctrl, env = create_discrete_policy_env(
        base_pol, topology=topo, seed=seed, max_steps=max_steps, use_rust=True, p2_cfg=p2_copy
    )

    tracker = None
    if method == "V20_NAJS":
        soft_ms = min(15.0, max(4.5, budget_ms * 0.65)) if budget_ms <= 35.0 else min(22.0, budget_ms * 0.45)
        raw_prop = build_default_v20_proposer(hard_budget_ms=budget_ms, seed=0, enable_pdg_recovery=True)
        raw_prop.soft_deadline_ms = soft_ms
        tracker = PDGLivenessTracker(cooldown_ticks=12)
    elif method == "V18_EA_JOINT":
        raw_prop = make_v18_external_proposer(hard_budget_ms=budget_ms, seq_diff_b2=True, env_ref=env)
    else:
        # V19_ADAPTIVE_RL native external proposer capped cleanly to `budget_ms`
        nat_prop = env.external_proposer
        assert nat_prop is not None
        raw_prop = lambda snap, net, *, budget_ms=budget_ms, _np=nat_prop, _cap=budget_ms: _np(snap, net, budget_ms=min(float(budget_ms), float(_cap)))

    rng_j = np.random.default_rng(int(seed) * 1009 + int(jitter_seed) * 31 + int(budget_ms))
    trunc_calls = 0
    total_calls = 0
    keep_fallbacks = 0

    def _wrapped(snap: Any, net: Any, *, budget_ms: float = budget_ms):
        nonlocal trunc_calls, total_calls, keep_fallbacks
        total_calls += 1
        eff_b = min(float(budget_ms), float(cap_budget))
        if protocol == "P3_JitterStress":
            # Deterministic seeded OS compute jitter: slow effective search headroom + 15% +6ms preemption spike
            slow_mult = float(np.clip(rng_j.lognormal(mean=0.18, sigma=0.28), 1.0, 2.2))
            spike_ms = 6.0 if float(rng_j.random()) < 0.15 else 0.0
            # Shrink inner search cap to simulate contention, then inject preemption overhead before commit
            inner_b = max(3.5, (eff_b - spike_ms) / slow_mult)
            if method == "V20_NAJS":
                orig_soft = raw_prop.soft_deadline_ms
                raw_prop.soft_deadline_ms = min(float(orig_soft), max(3.5, inner_b * 0.72))
                try:
                    p = raw_prop(snap, net, budget_ms=inner_b)
                finally:
                    raw_prop.soft_deadline_ms = orig_soft
            else:
                p = raw_prop(snap, net, budget_ms=inner_b)
            # Inject realistic preemption delay (bounded so wall-clock reflects real jitter without hanging)
            if spike_ms > 0.0:
                time.sleep(spike_ms / 1000.0)
        elif protocol == "P1_WorkBudget":
            # Charge calibrated base work inside virtual clock so decisions advance deterministically
            t_v0 = _cvcr_now_s()
            p = raw_prop(snap, net, budget_ms=eff_b)
            # Ensure non-zero calibrated work charge proportional to evaluated candidates
            ec = int((p.telemetry or {}).get("eval_cnt", 4))
            _cvcr_rec_work(max(1.2, 1.45 * ec))
            if (_cvcr_now_s() - t_v0) * 1000.0 >= eff_b * 0.9:
                trunc_calls += 1
        else:
            p = raw_prop(snap, net, budget_ms=eff_b)

        telem = p.telemetry or {}
        if bool(telem.get("fallback_keep", False)):
            keep_fallbacks += 1
        if not bool(telem.get("deadline_met", True)) or (protocol == "P3_JitterStress" and int(telem.get("eval_cnt", 5)) <= 2):
            trunc_calls += 1
        return p

    cap_budget = float(budget_ms)
    env.external_proposer = _wrapped

    scope = virtual_clock_scope(poll_ms=0.002) if protocol == "P1_WorkBudget" else _null_ctx()
    with scope:
        obs, info = env.reset(seed=seed, options={"scenario_tuple": sc_copy})
        tm = tr = False
        while not (tm or tr):
            feats = compute_expert_v9_state(
                env.sim, env.tracker,
                last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks,
                decision_logs=env.decision_logs,
            )
            step_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
            act = int(ctrl.select_action(feats, step_info, sim=env.sim))
            if tracker is not None:
                act = tracker.observe_and_maybe_escalate(env, step_info, act, proposer=raw_prop)
            obs, _, tm, tr, info = env.step(act)

    m = env.compute_episode_metrics()
    aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
    decision_ms = [
        float(r["t_propose_ms"]) + float(r["t_validate_commit_ms"])
        for r in env.decision_logs
    ]
    return {
        "episode_key": sp["episode_key"],
        "suite_group": sp["suite_group"],
        "net_obj": round(float(m["net_obj"]), 2),
        "h1": round(float(m["cmax"]) + float(m.get("task_tardiness_cost", 0.0)) + float(m.get("op_tardiness_cost", 0.0)), 2),
        "comp100": bool(float(m["task_completion_rate"]) >= 0.9999 and float(m["production_completion_rate"]) >= 0.9999),
        "audit_passed": bool(aud.pass_audit),
        "expired_windows": int(env.total_window_expirations),
        "transaction_rejections": sum(not r["receipt_accepted"] for r in env.decision_logs),
        "decision_latencies_ms": decision_ms,
        "total_calls": total_calls,
        "trunc_calls": trunc_calls,
        "keep_fallbacks": keep_fallbacks,
    }


def _summarize_protocol(rows: list[dict[str, Any]], budget_ms: float) -> dict[str, Any]:
    ja = [r["net_obj"] for r in rows]
    lats = sorted(float(x) for r in rows for x in r["decision_latencies_ms"])
    tot_c = sum(int(r["total_calls"]) for r in rows)
    tr_c = sum(int(r["trunc_calls"]) for r in rows)
    fb_c = sum(int(r["keep_fallbacks"]) for r in rows)
    overruns = sum(1 for x in lats if x > budget_ms)
    return {
        "n": len(rows),
        "comp100_count": sum(1 for r in rows if r["comp100"]),
        "comp100_pct": round(100.0 * sum(1 for r in rows if r["comp100"]) / max(1, len(rows)), 2),
        "audit_pass": sum(1 for r in rows if r["audit_passed"]),
        "mean_ja": round(float(np.mean(ja)), 2),
        "median_ja": round(float(np.median(ja)), 2),
        "cvar95_ja": round(_cvar(ja, 0.95), 2),
        "mean_ms": round(float(np.mean(lats)), 3) if lats else 0.0,
        "pool_p95_ms": round(float(np.percentile(lats, 95)), 3) if lats else 0.0,
        "pool_p99_ms": round(float(np.percentile(lats, 99)), 3) if lats else 0.0,
        "max_ms": round(float(lats[-1]), 3) if lats else 0.0,
        "hard_overrun_rate": round(overruns / max(1, len(lats)), 5),
        "truncation_rate": round(tr_c / max(1, tot_c), 4),
        "keep_fallback_rate": round(fb_c / max(1, tot_c), 4),
        "expired_windows": sum(int(r["expired_windows"]) for r in rows),
        "transaction_rejections": sum(int(r["transaction_rejections"]) for r in rows),
    }


def run_task4_robustness() -> dict[str, Any]:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    val36 = _select_stratified_val36()
    methods = ("V18_EA_JOINT", "V19_ADAPTIVE_RL", "V20_NAJS")

    out: dict[str, Any] = {
        "schema": "v20_1_closure.deliverable4_compute_robustness.v1",
        "P1_WorkBudget": {},
        "P2_WallClock": {},
        "P3_JitterStress_35ms": {},
        "high_volatility_5repeats_35ms": {},
        "p1_determinism_verified": False,
    }

    # Verify P1 bit-exact determinism on 2 repeats of val36[0] & val36[-1]
    d_chk1 = [_run_protocol_episode("V20_NAJS", sp, budget_ms=35.0, protocol="P1_WorkBudget")["net_obj"] for sp in (val36[0], val36[-1])]
    d_chk2 = [_run_protocol_episode("V20_NAJS", sp, budget_ms=35.0, protocol="P1_WorkBudget")["net_obj"] for sp in (val36[0], val36[-1])]
    out["p1_determinism_verified"] = (d_chk1 == d_chk2)
    print(f"=== [Task 4] P1 Virtual-Clock Bit-Identical Check: {out['p1_determinism_verified']} ({d_chk1}) ===", flush=True)

    # P1: Deterministic Work Budget across 10/20/35/50/100 ms
    print("=== [Task 4.1] Protocol P1: Deterministic Work Budget (VAL-36 x 5 budgets) ===", flush=True)
    for b_ms in BUDGETS_MS:
        bk = f"{int(b_ms)}ms"
        out["P1_WorkBudget"][bk] = {}
        for m in methods:
            rows = [_run_protocol_episode(m, sp, budget_ms=b_ms, protocol="P1_WorkBudget") for sp in val36]
            sm = _summarize_protocol(rows, b_ms)
            out["P1_WorkBudget"][bk][m] = sm
            print(f"  P1 {bk:5s} {m:16s}: Comp={sm['comp100_count']:2d}/36 JA={sm['mean_ja']:7.1f} CVaR95={sm['cvar95_ja']:7.1f} P99={sm['pool_p99_ms']:5.2f}ms", flush=True)

    # P2: Real Wall-Clock Budget across 10/20/35/50/100 ms
    print("=== [Task 4.2] Protocol P2: Real Wall-Clock Budget (VAL-36 x 5 budgets) ===", flush=True)
    for b_ms in BUDGETS_MS:
        bk = f"{int(b_ms)}ms"
        out["P2_WallClock"][bk] = {}
        for m in methods:
            rows = [_run_protocol_episode(m, sp, budget_ms=b_ms, protocol="P2_WallClock") for sp in val36]
            sm = _summarize_protocol(rows, b_ms)
            out["P2_WallClock"][bk][m] = sm
            print(
                f"  P2 {bk:5s} {m:16s}: Comp={sm['comp100_count']:2d}/36 JA={sm['mean_ja']:7.1f} CVaR95={sm['cvar95_ja']:7.1f} "
                f"P95={sm['pool_p95_ms']:5.2f} P99={sm['pool_p99_ms']:5.2f} Max={sm['max_ms']:5.1f} Ovr={100*sm['hard_overrun_rate']:.2f}%",
                flush=True,
            )

    # P3: Compute-Jitter Stress at 35 ms
    print("=== [Task 4.3] Protocol P3: Compute-Jitter Stress @ 35 ms (VAL-36) ===", flush=True)
    for m in methods:
        rows = [_run_protocol_episode(m, sp, budget_ms=35.0, protocol="P3_JitterStress", jitter_seed=1) for sp in val36]
        sm = _summarize_protocol(rows, 35.0)
        out["P3_JitterStress_35ms"][m] = sm
        print(
            f"  P3 35ms  {m:16s}: Comp={sm['comp100_count']:2d}/36 Aud={sm['audit_pass']}/36 JA={sm['mean_ja']:7.1f} "
            f"CVaR95={sm['cvar95_ja']:7.1f} P99={sm['pool_p99_ms']:5.2f} Max={sm['max_ms']:5.1f} Ovr={100*sm['hard_overrun_rate']:.2f}% "
            f"Trunc={100*sm['truncation_rate']:.1f}% KeepFB={100*sm['keep_fallback_rate']:.1f}%",
            flush=True,
        )

    # 5-repeat wall-clock dispersion on 4 high-volatility VAL instances
    print("=== [Task 4.4] 5-Repeat Wall-Clock Dispersion on High-Volatility Instances ===", flush=True)
    s72 = build_val72_specs()
    vol_keys = (
        "topo4_workshop_cells_window_bottleneck_16702",
        "track1_multi_wave_surge_1163",
        "track2_execution_blockage_1163",
        "track2_kitting_windows_1163",
    )
    for vk in vol_keys:
        sp = next(s for s in s72 if s["episode_key"] == vk)
        out["high_volatility_5repeats_35ms"][vk] = {}
        for m in methods:
            vals = [_run_protocol_episode(m, sp, budget_ms=35.0, protocol="P2_WallClock")["net_obj"] for _ in range(5)]
            out["high_volatility_5repeats_35ms"][vk][m] = {
                "runs": vals,
                "mean": round(float(np.mean(vals)), 2),
                "std": round(float(np.std(vals)), 2),
                "iqr": round(float(np.percentile(vals, 75) - np.percentile(vals, 25)), 2),
            }

    (OUT_CLOSURE_DIR / "deliverable4_compute_robustness.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    # Markdown Deliverable 4
    md = [
        "# Deliverable 4: Three-Protocol Compute Robustness Frontier (`10 / 20 / 35 / 50 / 100 ms`)",
        "",
        f"- **P1 Deterministic Work-Clock Bit-Identical Check**: `{out['p1_determinism_verified']}`",
        "",
        "## 1. Protocol P1 (Deterministic Work Budget, `VAL-36`)",
        "",
        "| Budget | Method | Comp100 | Mean JA ↓ | Median JA ↓ | CVaR95 JA ↓ |",
        "| :--- | :--- | ---: | ---: | ---: | ---: |",
    ]
    for bk in (f"{int(b)}ms" for b in BUDGETS_MS):
        for m in methods:
            s = out["P1_WorkBudget"][bk][m]
            md.append(f"| {bk} | {m} | {s['comp100_count']}/36 | {s['mean_ja']:.1f} | {s['median_ja']:.1f} | {s['cvar95_ja']:.1f} |")

    md.extend([
        "",
        "## 2. Protocol P2 (Real Uncontended Wall-Clock Budget, `VAL-36`)",
        "",
        "| Budget | Method | Comp100 | Mean JA ↓ | CVaR95 JA ↓ | Pool P95 | Pool P99 | Max ms | Hard Overrun % |",
        "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for bk in (f"{int(b)}ms" for b in BUDGETS_MS):
        for m in methods:
            s = out["P2_WallClock"][bk][m]
            md.append(
                f"| {bk} | {m} | {s['comp100_count']}/36 | {s['mean_ja']:.1f} | {s['cvar95_ja']:.1f} | "
                f"{s['pool_p95_ms']:.2f} | {s['pool_p99_ms']:.2f} | {s['max_ms']:.1f} | {100*s['hard_overrun_rate']:.2f}% |"
            )

    md.extend([
        "",
        "## 3. Protocol P3 (Compute-Jitter Stress @ `35 ms`, `VAL-36`)",
        "",
        "| Method | Comp100 | Audit | Mean JA ↓ | CVaR95 JA ↓ | Pool P99 | Max ms | >35ms % | Trunc % | Safe KEEP FB % | ExpWin |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for m in methods:
        s = out["P3_JitterStress_35ms"][m]
        md.append(
            f"| {m} | {s['comp100_count']}/36 | {s['audit_pass']}/36 | {s['mean_ja']:.1f} | {s['cvar95_ja']:.1f} | "
            f"{s['pool_p99_ms']:.2f} | {s['max_ms']:.1f} | {100*s['hard_overrun_rate']:.2f}% | {100*s['truncation_rate']:.1f}% | "
            f"{100*s['keep_fallback_rate']:.1f}% | {s['expired_windows']} |"
        )

    md.extend([
        "",
        "## 4. High-Volatility 5-Repeat Wall-Clock Dispersion (`35 ms`)",
        "",
        "| Episode | `V18_EA_JOINT` mean ± std (IQR) | `V19_ADAPTIVE_RL` mean ± std (IQR) | `V20_NAJS` mean ± std (IQR) |",
        "| :--- | ---: | ---: | ---: |",
    ])
    for vk in vol_keys:
        cols = []
        for m in methods:
            d = out["high_volatility_5repeats_35ms"][vk][m]
            cols.append(f"{d['mean']:.1f} ± {d['std']:.1f} ({d['iqr']:.1f})")
        md.append(f"| {vk} | " + " | ".join(cols) + " |")

    (OUT_CLOSURE_DIR / "deliverable4_compute_robustness.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[TASK 4 DONE] Wrote Deliverable 4 -> {OUT_CLOSURE_DIR / 'deliverable4_compute_robustness.md'}", flush=True)
    return out


if __name__ == "__main__":
    run_task4_robustness()
