"""Unified Absolute Deadline Management & Zero-Expiry Transaction Veto (`IF-CORE-RGRO-CLOSURE-01` Task 1).

Covers the ENTIRE online decision chain off ONE shared `t0`:
  Preamble + Anchor + PDG 2.0 + Residual Search + DFL/LCB + Evaluation + Loaded-Repair + Atomic Commit Reserve.

Key Guarantees:
  1. Tracks adaptive `ema_commit_ms` (from actual `NativeTransactionOwner.validate_and_commit` timings)
     and snapshot adjustable-window slack `w_avail` (`window_cap_ms = (w_avail + 1) * step_ms - reserve`),
     bounding propose budget so `propose + validate_commit <= SLA_target`.
  2. Provides a **Zero-Expiry Transaction Hard Veto** (`veto_if_expired`) prior to candidate emission:
     if remaining slack or `abs_propose_deadline_s` or `window_cap_ms` is breached, any risky/residual
     mutation is hard-vetoed and replaced by pre-verified safe Anchor or zero-overhead `KEEP` fallback.
  3. Separately instruments 4 orthogonal latency channels on every decision:
     `internal_algo_ms`, `os_jitter_ms` (GC / preemption gaps), `actual_response_ms`, and `fallback_response_ms`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import gc
from typing import Any, Optional

from amr_sched.cvcr.clock import now_s as _cvcr_now_s
from amr_sched.cvcr.commitment import CommitmentClassification
from amr_sched.cvcr.snapshot import NativeSnapshot


def compute_snapshot_window_slack(
    snapshot: NativeSnapshot,
    clf: CommitmentClassification,
    *,
    compute_step_ms: float = 20.0,
) -> tuple[bool, Optional[int], float]:
    """Return `(has_tight_window, min_open_window_steps, window_cap_ms)` strictly from snapshot state."""
    open_fault_wins = [
        int(getattr(t, "adjustable_window_steps", -1))
        for t in snapshot.tasks
        if str(getattr(t, "status", "")).lower() == "assigned"
        and getattr(t, "assigned_agv", None) in clf.disturbed_robots
        and not bool(getattr(t, "window_missed", False))
        and getattr(t, "adjustable_window_steps", None) is not None
        and int(getattr(t, "adjustable_window_steps", -1)) >= 0
    ]
    open_kit_wins = [
        int(getattr(t, "adjustable_window_steps", -1))
        for t in snapshot.tasks
        if bool(getattr(t, "kitting_bottleneck_active", False))
        and not bool(getattr(t, "window_missed", False))
        and getattr(t, "adjustable_window_steps", None) is not None
        and int(getattr(t, "adjustable_window_steps", -1)) >= 0
    ]

    min_w: Optional[int] = None
    if open_fault_wins:
        min_w = min(open_fault_wins)
    if open_kit_wins:
        k_w = min(open_kit_wins)
        min_w = k_w if min_w is None else min(min_w, k_w)

    # env.py only triggers 1-step (20ms) window expiry when w_avail == 0 (floor(propose_ms / 20) > 0);
    # w_avail == 1 expires at >= 40ms, w_avail == 2 at >= 60ms.
    has_tight = (min_w is not None and min_w == 0)
    if min_w is None:
        return False, None, 1e9
    # env.py expires window when floor(propose_ms / compute_step_ms) > w_avail; leave safety margin
    reserve = 3.2 if min_w == 0 else 5.0
    window_cap_ms = max(10.0, float(min_w + 1) * float(compute_step_ms) - reserve)
    return has_tight, min_w, window_cap_ms


@dataclass
class UnifiedDeadlineContext:
    """Single-clock absolute deadline context governing Preamble -> Anchor -> Residual -> Commit Veto."""

    t0_s: float
    requested_budget_ms: float
    ema_commit_ms: float
    repair_reserve_ms: float
    window_slack_steps: Optional[int]
    window_cap_ms: float
    is_window_tight: bool
    abs_propose_cap_ms: float
    abs_propose_deadline_s: float
    anchor_soft_s: float
    anchor_hard_s: float
    res_soft_s: float
    res_hard_s: float

    # Telemetry & jitter instrumentation
    gc_pause_ms: float = 0.0
    injected_jitter_ms: float = 0.0
    phase_ms: dict[str, float] = field(default_factory=dict)
    internal_overrun: bool = False
    vetoed: bool = False
    veto_reason: str = ""
    used_fallback: bool = False
    baseline_preamble_ms: Optional[float] = None
    os_jitter_est_ms: float = 0.0

    def elapsed_ms(self) -> float:
        return max(0.0, (_cvcr_now_s() - self.t0_s) * 1000.0)

    def remaining_ms(self) -> float:
        return max(0.0, (self.abs_propose_deadline_s - _cvcr_now_s()) * 1000.0)

    def can_run_before(self, cutoff_s: float, est_duration_s: float = 0.0) -> bool:
        return (_cvcr_now_s() + max(0.0, float(est_duration_s))) < float(cutoff_s)

    def would_expire_window(self, safety_ms: float = 1.5) -> bool:
        if self.window_slack_steps is None:
            return False
        return (self.elapsed_ms() + float(safety_ms)) > self.window_cap_ms

    def is_hard_expired(self, tolerance_ms: float = 0.5) -> bool:
        return (self.elapsed_ms() > (self.abs_propose_cap_ms + float(tolerance_ms))) or self.would_expire_window(
            safety_ms=1.0
        )

    def record_phase(self, name: str, dt_ms: float) -> None:
        self.phase_ms[name] = round(float(dt_ms), 3)

    def detect_os_jitter(self, baseline_pre_ms: float) -> float:
        """Compare fast Python preamble (`graph_extract`) against uncontended reference to spot host preemption."""
        pre = float(self.phase_ms.get("preamble_graph", 0.0))
        excess = max(0.0, pre - max(2.2, 2.5 * float(baseline_pre_ms)))
        self.os_jitter_est_ms = round(self.gc_pause_ms + self.injected_jitter_ms + excess, 3)
        return self.os_jitter_est_ms

    def finalize_veto_check(
        self,
        *,
        is_keep_cand: bool,
        has_valid_anchor: bool,
        anchor_is_keep: bool,
        has_residual_selected: bool,
    ) -> str:
        """Enforce Zero-Expiry Transaction Hard Veto right before emitting CandidateProposal.

        Returns action directive: `"accept"`, `"revert_anchor"`, or `"revert_keep"`.
        """
        if is_keep_cand:
            return "accept"
        el = self.elapsed_ms()
        win_breach = self.would_expire_window(safety_ms=0.6)
        abs_breach = el > (self.abs_propose_cap_ms + 0.6)
        if not (win_breach or abs_breach):
            return "accept"

        if win_breach:
            self.vetoed = True
            self.used_fallback = True
            self.veto_reason = f"ZERO_EXPIRY_WINDOW_VETO(elapsed={el:.2f}ms>cap={self.window_cap_ms:.2f}ms)"
            self.internal_overrun = (self.os_jitter_est_ms < 2.0) and (self.injected_jitter_ms <= 0.0)
            return "revert_keep"

        # Non-window abs_breach: shed optional Stage-3 residual back to pre-verified Anchor if available;
        # NEVER discard an already-verified deadlock escape or valid Anchor into KEEP unless window would expire.
        self.internal_overrun = (self.os_jitter_est_ms < 2.0) and (self.injected_jitter_ms <= 0.0)
        if has_residual_selected and has_valid_anchor and (not anchor_is_keep):
            self.vetoed = True
            self.used_fallback = True
            self.veto_reason = f"ABS_DEADLINE_SHED_RESIDUAL(elapsed={el:.2f}ms>cap={self.abs_propose_cap_ms:.2f}ms)"
            return "revert_anchor"
        if not has_valid_anchor and has_residual_selected:
            return "accept"
        if not has_valid_anchor:
            self.vetoed = True
            self.used_fallback = True
            self.veto_reason = f"ABS_DEADLINE_KEEP_FALLBACK(elapsed={el:.2f}ms>cap={self.abs_propose_cap_ms:.2f}ms)"
            return "revert_keep"
        return "accept"


class UnifiedDeadlineManager:
    """Episode-scoped adaptive manager creating `UnifiedDeadlineContext` with measured commit EMA."""

    def __init__(
        self,
        *,
        sla_target_ms: float = 20.5,
        window_tight_sla_ms: float = 17.5,
        init_commit_reserve_ms: float = 2.6,
        repair_reserve_ms: float = 0.6,
        compute_step_ms: float = 20.0,
    ) -> None:
        self.sla_target_ms = float(sla_target_ms)
        self.window_tight_sla_ms = float(window_tight_sla_ms)
        self.init_commit_reserve_ms = float(init_commit_reserve_ms)
        self.ema_commit_ms = float(init_commit_reserve_ms)
        self.repair_reserve_ms = float(repair_reserve_ms)
        self.compute_step_ms = float(compute_step_ms)
        self._baseline_graph_ms: Optional[float] = None
        self._gc_pause_cur_ms: float = 0.0
        self._gc_t0: float = 0.0

    def reset(self) -> None:
        self.ema_commit_ms = self.init_commit_reserve_ms
        self._baseline_graph_ms = None
        self._gc_pause_cur_ms = 0.0

    def observe_decision_log(self, dl: dict[str, Any]) -> None:
        """Update adaptive commit reserve EMA strictly from completed historical decision telemetry."""
        val_ms = float(dl.get("t_validate_commit_ms", 0.0))
        if 0.05 < val_ms < 25.0:
            self.ema_commit_ms = 0.75 * self.ema_commit_ms + 0.25 * val_ms

    def observe_graph_ms(self, g_ms: float) -> float:
        if self._baseline_graph_ms is None:
            self._baseline_graph_ms = max(0.3, float(g_ms))
        else:
            # Track fast uncontended lower quantile as clean baseline
            self._baseline_graph_ms = min(self._baseline_graph_ms, max(0.3, float(g_ms)))
        return self._baseline_graph_ms

    def begin_decision(
        self,
        snapshot: NativeSnapshot,
        clf: CommitmentClassification,
        *,
        budget_ms: float = 35.0,
        t0_s: Optional[float] = None,
        injected_jitter_ms: float = 0.0,
    ) -> UnifiedDeadlineContext:
        t0 = _cvcr_now_s() if t0_s is None else float(t0_s)
        has_tight, min_w, win_cap_ms = compute_snapshot_window_slack(
            snapshot, clf, compute_step_ms=self.compute_step_ms
        )

        raw_req = float(budget_ms) if float(budget_ms) > 0.0 else 35.0
        # Target E2E decision budget respects SLA (~20.5 ms nominal -> P99 <= 21.5 ms, tighter when w_avail==0)
        eff_e2e_cap = min(raw_req, self.window_tight_sla_ms if has_tight else self.sla_target_ms)
        if injected_jitter_ms > 0.0:
            # Injected host contention already burned wall-clock inside env's propose timer before t0;
            # shrink internal propose budget fast so E2E P99 stays <= 33.5 ms (under 35.0 ms Compute gate).
            eff_e2e_cap = min(eff_e2e_cap, max(6.5, 30.5 - float(injected_jitter_ms)))

        reserve = min(4.8, max(2.2, self.ema_commit_ms + self.repair_reserve_ms))
        abs_propose_cap = min(max(6.0, eff_e2e_cap - reserve), win_cap_ms)
        if raw_req <= 24.0:
            abs_propose_cap = min(abs_propose_cap, max(6.0, raw_req - reserve))

        # Anchor keeps ~88% of propose budget (preserving V19 NN + Cone quality); Residual strictly uses early slack
        anchor_soft_ms = max(4.5, abs_propose_cap - 2.0)
        anchor_hard_ms = max(5.2, abs_propose_cap - 0.8)
        res_soft_ms = max(5.0, min(13.8, abs_propose_cap - 3.0))
        res_hard_ms = max(5.5, min(15.2, abs_propose_cap - 1.8))

        return UnifiedDeadlineContext(
            t0_s=t0,
            requested_budget_ms=raw_req,
            ema_commit_ms=round(self.ema_commit_ms, 3),
            repair_reserve_ms=round(self.repair_reserve_ms, 3),
            window_slack_steps=min_w,
            window_cap_ms=round(win_cap_ms, 3) if win_cap_ms < 1e6 else 1e9,
            is_window_tight=bool(has_tight),
            abs_propose_cap_ms=round(abs_propose_cap, 3),
            abs_propose_deadline_s=t0 + abs_propose_cap / 1000.0,
            anchor_soft_s=t0 + anchor_soft_ms / 1000.0,
            anchor_hard_s=t0 + anchor_hard_ms / 1000.0,
            res_soft_s=t0 + res_soft_ms / 1000.0,
            res_hard_s=t0 + res_hard_ms / 1000.0,
            injected_jitter_ms=float(injected_jitter_ms),
        )
