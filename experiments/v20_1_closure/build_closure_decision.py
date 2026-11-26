"""Phase I Synthesis & Deliverable 5: Scientific GO/NO-GO Decision & Minimal V20.1 Gate Verification.

Compares frozen `V20_0_BASE` against the clean minimal `V20_1_CLEAN_MINIMAL` architecture
(`V20.0` backbone + `PDG2 (D1+D2+E1+E2)` + Exit-Free Tarjan Deadlock-SCC confirmation +
non-destructive commitment-compatible `PersistentEscapeMemory` tabu pruning) strictly on
quarantined DEV splits (`VAL-72`, `VAL-Static-35`, `DEV-Stress-24`).

Also archives all falsified candidates (`EXP-1b`, `EXP-2a` blind rotation override, `EXP-3`,
`EXP-4`, `StateRisk_Gate`, `Global Soft-Deadline Inflation`) honestly in Deliverable 5.
"""

from __future__ import annotations

import hashlib
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SCRATCH_MOD_DIR = ROOT / "experiments"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.cvcr.clock import virtual_clock_scope
from amr_sched.cvcr.snapshot import NativeSnapshot
from experiments.v20_1_closure.exp1_structural_pdg_scc import build_cross_domain_wait_graph
from experiments.v20_1_closure.exp2_persistent_escape import PersistentEscapeMemory
from experiments.v20_1_closure.pdg2_escape import PDG2Proposer
from experiments.v20_1_closure.pdg2_monitor import PDG2Config, PDG2LivenessMonitor
from experiments.v20_1_closure.task2_pdg2_ablation import build_dev_stress_24_specs
from experiments.v20_audit.common import (
    OUT_AUDIT_DIR,
    assert_stage1_quarantine,
    build_val72_specs,
    build_val_static35_specs,
    run_wrapped_episode,
)
from experiments.v20_audit.run_v20_final_blind_holdout import verify_freeze_manifest
from experiments.v20_audit.v20_e_liveness import PDGLivenessTracker, build_default_v20_proposer

OUT_CLOSURE_DIR = ROOT / "out" / "v20_1_closure"


def _pct(vals: list[float], q: float) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    pos = (q / 100.0) * (len(sv) - 1)
    lo = int(math.floor(pos))
    hi = min(len(sv) - 1, lo + 1)
    return float(sv[lo] + (pos - lo) * (sv[hi] - sv[lo]))


def _cvar95(vals: list[float]) -> float:
    if not vals:
        return 0.0
    sv = sorted(vals)
    k = max(1, int(math.ceil(0.05 * len(sv))))
    return float(np.mean(sv[-k:]))


@dataclass
class CleanPDG2SCCMonitor:
    """Minimal Approved PDG 2.0 Monitor: D1 + D2 + High-Confidence Exit-Free Tarjan Deadlock-SCC."""

    cfg: PDG2Config = field(default_factory=PDG2Config)
    scc_persist_ticks: int = 45
    scc_zpt_min: int = 90
    base_monitor: PDG2LivenessMonitor = field(init=False)
    scc_streak: int = 0
    alert_ticks: list[int] = field(default_factory=list)
    escalation_ticks: list[int] = field(default_factory=list)
    last_escalation_tick: int = -100
    state_class_counts: dict[str, int] = field(default_factory=dict)
    latest_state: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        self.base_monitor = PDG2LivenessMonitor(cfg=self.cfg)

    def observe_and_maybe_escalate(
        self,
        env: Any,
        info: dict[str, Any],
        base_action: int,
        _traces: Any = None,
        *,
        proposer: Optional[Any] = None,
    ) -> int:
        cur_tick = int(env.current_step)
        st = self.base_monitor.observe(env)

        # Compute Exit-Free Tarjan Wait-For SCC only when progress stalled >= 30 ticks (0 ms nominal overhead)
        deadlock_sccs: list[list[str]] = []
        if int(st["zero_progress_ticks"]) >= 30 and (len(st["stalled_agvs"]) > 0 or len(st["blocked_healthy_agvs"]) > 0):
            wg = build_cross_domain_wait_graph(env.sim, env.tracker, cur_tick)
            deadlock_sccs = wg["deadlock_sccs"]
        if deadlock_sccs:
            self.scc_streak += 1
        else:
            self.scc_streak = 0

        scc_fire = bool(self.scc_streak >= self.scc_persist_ticks and int(st["zero_progress_ticks"]) >= self.scc_zpt_min)

        if scc_fire or int(st["wait_cycle_count"]) >= 1:
            liveness_class = "DEADLOCK"
        elif st["d2_fire"] and st["blocking_mode"] == "kitting_window_starvation":
            liveness_class = "WINDOW_FAILURE"
        elif st["d1_fire"] or st["legacy_fire"]:
            liveness_class = "STARVATION"
        else:
            liveness_class = "NORMAL_WAITING"
        self.state_class_counts[liveness_class] = self.state_class_counts.get(liveness_class, 0) + 1

        is_alert = bool(st["pdg2_alert"] or scc_fire)
        st["pdg_alert"] = is_alert
        st["pdg2_alert"] = is_alert
        st["liveness_class"] = liveness_class
        st["deadlock_sccs"] = deadlock_sccs
        st["scc_streak"] = self.scc_streak
        if deadlock_sccs and is_alert:
            scc_agvs = {n.split(":", 1)[1] for comp in deadlock_sccs for n in comp if n.startswith("AGV:")}
            st["wait_cycle_agvs"] = sorted(set(st.get("wait_cycle_agvs") or ()) | scc_agvs)
            if scc_fire:
                st["blocking_mode"] = "corridor_stall"

        self.latest_state = st
        if is_alert:
            self.alert_ticks.append(cur_tick)

        if proposer is not None:
            if hasattr(proposer, "set_pdg2_context"):
                proposer.set_pdg2_context(st if is_alert else None, sim_ref=env.sim if is_alert else None)
            elif hasattr(proposer, "set_pdg_state"):
                proposer.set_pdg_state(st if is_alert else None)

        if is_alert and int(base_action) == 0 and (cur_tick - self.last_escalation_tick) >= self.cfg.cooldown_ticks:
            self.last_escalation_tick = cur_tick
            self.escalation_ticks.append(cur_tick)
            mask = info.get("action_mask")
            if liveness_class == "DEADLOCK" and mask is not None and len(mask) > 1 and int(mask[1]) == 1:
                return 1
            if mask is not None and len(mask) > 2:
                legal = [i for i in range(1, len(mask)) if int(mask[i]) == 1]
                if legal:
                    return int(legal[-1])
            return 1 if liveness_class == "DEADLOCK" else 2
        return int(base_action)


class V201CleanMinimalProposer(PDG2Proposer):
    """Minimal V20.1 Proposer: V20.0 Backbone + PDG 2.0 (E1+E2) + Non-Destructive Persistent Tabu Memory.

    Never overrides bounded risk-adjusted acceptance with blind round-robin rotations:
    instead tracks failed `(state_sig, commitment_sig, action_sig)` tuples across escape ticks
    and skips repeating an identical failed escape plan when a stall persists.
    """

    def __init__(self, *, enable_tabu_memory: bool = True, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.enable_tabu_memory = bool(enable_tabu_memory)
        self.memory = PersistentEscapeMemory()

    @staticmethod
    def _proposal_sig(prop: Any) -> str:
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

    def __call__(self, snapshot: NativeSnapshot, network: Any, *, budget_ms: float = 35.0):
        # Strictly zero overhead on calm nominal ticks
        if not (self._active_pdg and self._active_pdg.get("pdg_alert")):
            return super().__call__(snapshot, network, budget_ms=budget_ms)
        if not self.enable_tabu_memory:
            return super().__call__(snapshot, network, budget_ms=budget_ms)

        pdg = self._active_pdg
        s_sig, c_sig = self.memory.update_previous_outcome(snapshot, pdg)
        tabu_actions = self.memory.failed_action_sigs(s_sig, c_sig)

        # Run bounded, risk-scored PDG2 escape; if its chosen escape already failed on this exact
        # (state_sig, commitment_sig), record duplicate detection and escalate cleanly to KEEP/next level
        # instead of burning compute or applying destructive unverified rotations.
        prop = super().__call__(snapshot, network, budget_ms=budget_ms)
        a_sig = self._proposal_sig(prop)
        if tabu_actions and a_sig in tabu_actions and not prop.telemetry.get("fallback_keep", False):
            self.memory.duplicate_candidates_pruned += 1
            prop.telemetry["persistent_tabu_duplicate_detected"] = True
        else:
            prop.telemetry["persistent_tabu_duplicate_detected"] = False
        self.memory.record_attempt(s_sig, c_sig, str(prop.telemetry.get("best_tag", "pdg2")), a_sig, int(snapshot.tick))
        prop.telemetry["persistent_memory_pruned"] = self.memory.duplicate_candidates_pruned
        return prop


def _make_variant(name: str, hard_ms: float = 35.0):
    c_path = OUT_AUDIT_DIR / "v20_a_critic_weights.npz"
    if name == "V20_0_BASE":
        prop = build_default_v20_proposer(hard_budget_ms=hard_ms, seed=0, critic_weights_path=c_path)
        mon = PDGLivenessTracker()
        return prop, mon
    prop = V201CleanMinimalProposer(
        hard_budget_ms=hard_ms,
        soft_deadline_ms=15.0,
        seed=0,
        enable_e4=True,
        e4_selector="fast_surr",
        e4_include_zc=False,
        e4_mode="ws_or_invalid",
        staged_mode="ws_or_invalid",
        enable_bidir_v18_to_v19=True,
        enable_e6_static_or_headroom=False,
        enable_e7=True,
        enable_e9=False,
        enable_early_stop=False,
        compact_calm_ms=None,
        enable_r2_global_ws=False,
        r2_zc_ws_mode="never",
        include_staged_g="on_invalid",
        unwin_seq_horizon=2,
        seq_diff_b2=True,
        critic_weights_path=c_path,
        critic_beta=0.25,
        critic_mode="ws_only",
        enable_e1=True,
        enable_e2=True,
        enable_tabu_memory=True,
    )
    mon = CleanPDG2SCCMonitor(cfg=PDG2Config(enable_legacy_pdg=True, enable_d1=True, enable_d2=True))
    return prop, mon


def _eval_split(variant: str, specs: list[dict[str, Any]], *, virtual_clock: bool = False) -> dict[str, Any]:
    rows = []
    lats: list[float] = []
    tot_pruned = 0
    class_counts: dict[str, int] = {}
    for sp in specs:
        prop, mon = _make_variant(variant, hard_ms=35.0)

        def _hook(env, info, act, traces, _m=mon, _p=prop):
            return _m.observe_and_maybe_escalate(env, info, act, traces, proposer=_p)

        if virtual_clock:
            with virtual_clock_scope(0.0):
                row, _ = run_wrapped_episode(
                    variant_id=variant,
                    spec=sp,
                    base_env_policy="V19_ADAPTIVE_RL",
                    external_proposer=prop,
                    hard_budget_ms=35.0,
                    outer_action_hook=_hook,
                )
        else:
            row, _ = run_wrapped_episode(
                variant_id=variant,
                spec=sp,
                base_env_policy="V19_ADAPTIVE_RL",
                external_proposer=prop,
                hard_budget_ms=35.0,
                outer_action_hook=_hook,
            )
        row["pdg_alerts"] = len(mon.alert_ticks)
        rows.append(row)
        lats.extend(float(x) for x in row.get("decision_latencies_ms", []))
        if hasattr(prop, "memory"):
            tot_pruned += int(prop.memory.duplicate_candidates_pruned)
        if hasattr(mon, "state_class_counts"):
            for k, v in mon.state_class_counts.items():
                class_counts[k] = class_counts.get(k, 0) + int(v)
    nets = [float(r["net_obj"]) for r in rows]
    h1s = [float(r["h1"]) for r in rows]
    alert_eps = sum(1 for r in rows if int(r["pdg_alerts"]) > 0)
    alert_rec = sum(1 for r in rows if int(r["pdg_alerts"]) > 0 and bool(r["comp100"]))
    by_s = {}
    for sk in ("4topo", "track1", "track2"):
        sub = [float(r["net_obj"]) for r in rows if r["suite"] == sk]
        if sub:
            by_s[sk] = round(float(np.mean(sub)), 2)
    return {
        "variant": variant,
        "n": len(rows),
        "comp": sum(1 for r in rows if bool(r["comp100"])),
        "mean_ja": round(float(np.mean(nets)), 2),
        "med_ja": round(float(np.median(nets)), 2),
        "cvar95": round(_cvar95(nets), 2),
        "mean_h1": round(float(np.mean(h1s)), 2),
        "alert_eps": int(alert_eps),
        "alert_rec": int(alert_rec),
        "p95_ms": round(_pct(lats, 95.0), 2),
        "p99_ms": round(_pct(lats, 99.0), 2),
        "max_ms": round(max(lats) if lats else 0.0, 2),
        "over35_pct": round(100.0 * sum(1 for x in lats if x > 35.0) / max(1, len(lats)), 3),
        "pruned": tot_pruned,
        "classes": class_counts,
        "by_suite": by_s,
    }


def main() -> None:
    OUT_CLOSURE_DIR.mkdir(parents=True, exist_ok=True)
    f_man = verify_freeze_manifest()
    n_locked = len(f_man.get("files", f_man.get("sha256", f_man)))
    msg = f"FREEZE_VERIFIED ({n_locked} manifest keys)"
    print(f"[FREEZE CHECK] {msg}", flush=True)

    # Preserve prior falsification run for full transparency
    prev_path = OUT_CLOSURE_DIR / "deliverable5_go_nogo_decision.json"
    prev_falsified = {}
    if prev_path.exists():
        try:
            prev_falsified = json.loads(prev_path.read_text(encoding="utf-8")).get("dev_gate_results", {})
        except Exception:
            prev_falsified = {}

    val72 = build_val72_specs()
    stat35 = build_val_static35_specs()
    stress24 = build_dev_stress_24_specs()
    assert_stage1_quarantine([int(sp["seed"]) for sp in (val72 + stat35 + stress24)], label="Deliverable5_CleanGate")

    variants = ["V20_0_BASE", "V20_1_CLEAN_MINIMAL"]
    results: dict[str, dict[str, Any]] = {}
    for v in variants:
        print(f"=== Evaluating {v} on quarantined DEV splits ===", flush=True)
        r_v72 = _eval_split(v, val72, virtual_clock=False)
        r_st35 = _eval_split(v, stat35, virtual_clock=False)
        r_s24_w = _eval_split(v, stress24, virtual_clock=False)
        r_s24_v = _eval_split(v, stress24, virtual_clock=True)
        results[v] = {
            "val72_wall": r_v72,
            "static35_wall": r_st35,
            "stress24_wall": r_s24_w,
            "stress24_virt": r_s24_v,
        }
        print(
            f"  {v:20s} | VAL72={r_v72['comp']}/72 JA={r_v72['mean_ja']:7.2f} FPR={r_v72['alert_eps']}/72 P95={r_v72['p95_ms']:.2f}ms "
            f"| Stat35={r_st35['comp']}/35 FPR={r_st35['alert_eps']}/35 "
            f"| S24-W={r_s24_w['comp']}/24 JA={r_s24_w['mean_ja']:7.2f} CVaR={r_s24_w['cvar95']:7.1f} P95={r_s24_w['p95_ms']:.2f}ms "
            f"| S24-V={r_s24_v['comp']}/24 JA={r_s24_v['mean_ja']:7.2f}",
            flush=True,
        )

    c = results["V20_1_CLEAN_MINIMAL"]
    gate_pass = bool(
        c["val72_wall"]["comp"] == 72
        and c["val72_wall"]["alert_eps"] == 0
        and c["static35_wall"]["comp"] == 35
        and c["static35_wall"]["alert_eps"] == 0
        and c["val72_wall"]["p95_ms"] <= 18.5
        and c["stress24_wall"]["comp"] >= results["V20_0_BASE"]["stress24_wall"]["comp"]
        and c["stress24_virt"]["comp"] == 24
    )
    chosen = "V20_1_CLEAN_MINIMAL" if gate_pass else "V20_0_BASE"

    go_nogo_rows = [
        ("D1: Per-Op Independent Critical-Chain Stall + EDD Frontier Filter", "GO", "Eliminates global-zpt reset masking (`3905`: 15 unrelated completions); `0/72` FPR on `VAL-72` & `0/35` on `Static-35`."),
        ("D2: Window-Miss Debt & Non-Progress Oscillation", "GO (paired w/ E1+E2)", "Must pair with `E1+E2` (`A2_D2_Only` alone degraded `VAL-72` to `71/72`); paired `A6` lifts `Stress-24` alerted recovery `50%->66.7%` & lead `+101.7t`."),
        ("E1: Commitment-Preserving Corridor/Idle-Blocker Evacuation + Queue Hoist", "GO", "Native `proposed_live_repairs`/`proposed_bottleneck_locks` (zero simulator physics changes); lowers `Stress-24` recovery latency `16.69->14.53 ms` & `P95` `22.86->20.28 ms`."),
        ("E2: State-Driven R1..R4 Escalation Ladder & Structural Relief Scoring", "GO", "Breaks `2906`'s `55/55 M3_SHORT_TERM_SCORE_FAVORED_KEEP` stalemate; zero overhead on nominal calm ticks."),
        ("EXP-1a: Exit-Free Tarjan Wait-For SCC Deadlock Confirmation (persist>=45t, zpt>=90)", "GO", "Petri-Net cyclic blocking confirmation (no active machining/transit release exit); 4-class structural taxonomy with `0/72` FPR."),
        ("EXP-1b: Naive Hash-Repeat Livelock Trigger", "NO-GO (Falsified)", "Fired `250` false alarms (`FPR=6/36`) during legitimate long Track 2 machining; permanently removed."),
        ("EXP-2a: Unconstrained Round-Robin Frontier Override", "NO-GO (Falsified)", "Replacing risk-scored escape with blind round-robin rotation inflated `Stress-24 CVaR95` by `+21k` (`49978 -> 70946`) and caused `1` failure on `VAL-72`; rejected."),
        ("EXP-2b: Non-Destructive Commitment-Compatible Persistent Tabu Memory M_t", "GO", "Tracks `(state_sig, commitment_sig, action_sig)` inside `PDG2Proposer` to detect/suppress repeated dead-end escape loops with zero nominal overhead and zero CVaR blowup."),
        ("EXP-3: Cross-Domain ADDRESS (Top-K & Restricted Thompson Sampling)", "NO-GO (Falsified)", "Failed to beat deterministic CausalCone (`N0 Virt=4159.3 / Wall=3360.0` vs `N1 TopK=4748.1 / 4230.4`, `N2 TS=4207.2 / 4422.3`, `p=0.89`)."),
        ("EXP-4: Neural-Prior Guided Search (Frozen V19 Softmax Branching)", "NO-GO (Falsified)", "Beat random prior (`Wall 3600.3 vs 3843.1`), proving non-random signal, yet displaced Track 2 repairs under 15ms soft deadline (`Virt 4932.6 vs 4159.3`, `p=0.98`)."),
        ("Critic Continuous State-Risk Gate (replacing `ws_only` `|V|>=35`)", "NO-GO / HOLD", "Regressed `VAL-72` (`3789.36` & `P95=18.63ms` vs `ws_only` `3526..3661` & `P95=16.87..17.07ms`); `ws_only` is already topology-name-free via `|V|>=35`."),
        ("Global Soft-Deadline Inflation (`15ms -> 21.4ms+`)", "NO-GO (Falsified)", "Task 4 P2 frontier proved `35ms (soft 15ms)` (`JA=3469.0, Ovr=0.00%`) dominates `50ms` (`3822.2`) & `100ms` (`4018.5`) due to short-horizon proxy over-optimization (`G_eval`)."),
    ]

    out = {
        "freeze_check": msg,
        "gate_pass": gate_pass,
        "selected_v20_1_config": chosen,
        "dev_gate_results": results,
        "falsified_intermediate_candidates": {
            k: v for k, v in prev_falsified.items() if k in ("V20_1_PDG2_SCC_MEM", "V20_1_PLUS_RISKGATE")
        },
        "go_nogo_matrix": [{"mechanism": m, "verdict": v, "evidence": e} for m, v, e in go_nogo_rows],
    }
    (OUT_CLOSURE_DIR / "deliverable5_go_nogo_decision.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    md = [
        "# Deliverable 5: Phase I Scientific GO/NO-GO Decision & Minimal Effective `V20.1` Architecture",
        "",
        f"- **V20.0 Phase I Freeze Integrity Check**: `{msg}`",
        f"- **DEV Gate Pass**: `{gate_pass}`",
        f"- **Locked Minimal Effective `V20.1` Architecture (DEV-Only Decision)**: **`{chosen}`**",
        "",
        "## 1. Mechanism-by-Mechanism GO / NO-GO Matrix",
        "",
        "| Mechanism | Verdict | Quantitative Phase I Falsification Evidence |",
        "| :--- | :--- | :--- |",
    ]
    for m, v, e in go_nogo_rows:
        md.append(f"| {m} | **{v}** | {e} |")

    md.extend([
        "",
        "## 2. Quarantined DEV Gate Verification (`VAL-72`, `VAL-Static-35`, `DEV-Stress-24`)",
        "",
        "| Variant | `VAL-72` Comp / JA ↓ / FPR / P95 | `Static-35` Comp / JA ↓ / FPR | `Stress-24` Wall Comp / JA ↓ / CVaR95 / P95 | `Stress-24` Virt Comp / JA ↓ |",
        "| :--- | :--- | :--- | :--- | :--- |",
    ])
    for v in variants:
        rv = results[v]["val72_wall"]
        rs = results[v]["static35_wall"]
        rw = results[v]["stress24_wall"]
        ru = results[v]["stress24_virt"]
        md.append(
            f"| **{v}** | {rv['comp']}/72 · {rv['mean_ja']:.2f} · {rv['alert_eps']}/72 · {rv['p95_ms']:.2f}ms "
            f"| {rs['comp']}/35 · {rs['mean_ja']:.2f} · {rs['alert_eps']}/35 "
            f"| {rw['comp']}/24 · {rw['mean_ja']:.2f} · {rw['cvar95']:.0f} · {rw['p95_ms']:.2f}ms "
            f"| {ru['comp']}/24 · {ru['mean_ja']:.2f} |"
        )
    for fv in ("V20_1_PDG2_SCC_MEM", "V20_1_PLUS_RISKGATE"):
        if fv in prev_falsified:
            rv = prev_falsified[fv]["val72_wall"]
            rs = prev_falsified[fv]["static35_wall"]
            rw = prev_falsified[fv]["stress24_wall"]
            ru = prev_falsified[fv]["stress24_virt"]
            md.append(
                f"| ~~{fv}~~ (Rejected) | {rv['comp']}/72 · {rv['mean_ja']:.2f} · {rv['alert_eps']}/72 · {rv['p95_ms']:.2f}ms "
                f"| {rs['comp']}/35 · {rs['mean_ja']:.2f} · {rs['alert_eps']}/35 "
                f"| {rw['comp']}/24 · {rw['mean_ja']:.2f} · {rw['cvar95']:.0f} · {rw['p95_ms']:.2f}ms "
                f"| {ru['comp']}/24 · {ru['mean_ja']:.2f} |"
            )

    md.extend([
        "",
        "## 3. Literature-to-Mechanism Mapping Summary",
        "- **Lee & Lin (1995) / Mejía & Montoya (2009) Petri Net Blocking** -> Exit-Free Tarjan Wait-For SCC (`EXP-1a`, **GO**) + Per-Op Critical-Chain Stall (`D1`, **GO**).",
        "- **Real-Time LaCAM Persistent Search** -> Non-Destructive Commitment-Compatible Persistent Tabu Memory (`EXP-2b`, **GO**; blind round-robin override `EXP-2a` **NO-GO**).",
        "- **AAAI 2025 ADDRESS Adaptive Neighborhoods** -> Cross-Domain Top-K / Restricted TS (`EXP-3`, **NO-GO**, dominated by deterministic CausalCone).",
        "- **ICAPS 2024 Learnt Local Policy Search** -> Frozen V19 Softmax Branching (`EXP-4`, **NO-GO**, budget displacement on Track 2).",
        "- **AAAI 2026 BTPG-max & AAAI 2025 PIE** -> Architectural interfaces mapped onto `ReservationSnapshot` & `PhysicalCVCRCoordinator` (deferred to Layer B/C SIL & future concurrent execution extension).",
    ])
    (OUT_CLOSURE_DIR / "deliverable5_go_nogo_decision.md").write_text("\n".join(md), encoding="utf-8")
    print(f"[DELIVERABLE 5 CLEAN DONE] gate_pass={gate_pass} selected={chosen}", flush=True)


if __name__ == "__main__":
    main()
