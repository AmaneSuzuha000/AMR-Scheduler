"""Module-isolated wall/work-calibrated monotonic clock for CVCR & deterministic benchmarks.

Outside virtual-clock benchmark runs, `now_s()` delegates directly to `time.perf_counter()`
and `record_work_ms()` is a no-op with near-zero overhead. Inside virtual-clock benchmark
workers, polling `now_s()` advances by tiny `poll_ms` (`0.002 ms`) while actual solver/evaluator
operations charge empirically calibrated `work_ms`, eliminating host CPU jitter without
prematurely truncating search budgets via global `time.perf_counter` monkeypatching.
"""

from __future__ import annotations

from contextlib import contextmanager
import time
from typing import Iterator

_VC_ACTIVE: bool = False
_VC_TIME_S: float = 1000.0
_VC_POLL_S: float = 0.000002  # 0.002 ms per budget poll check


def is_virtual_clock_active() -> bool:
    """Return True when deterministic work-calibrated virtual clock is active."""
    return _VC_ACTIVE


def enable_virtual_clock(poll_ms: float = 0.002, start_s: float = 1000.0) -> None:
    """Enable worker-local deterministic work-calibrated clock."""
    global _VC_ACTIVE, _VC_TIME_S, _VC_POLL_S
    _VC_ACTIVE = True
    _VC_TIME_S = float(start_s)
    _VC_POLL_S = max(1e-7, float(poll_ms) / 1000.0)


def disable_virtual_clock() -> None:
    """Restore standard `time.perf_counter()` wall-clock timing."""
    global _VC_ACTIVE
    _VC_ACTIVE = False


@contextmanager
def virtual_clock_scope(poll_ms: float = 0.002, start_s: float = 1000.0) -> Iterator[None]:
    """Context manager enabling work-calibrated virtual clock cleanly."""
    global _VC_ACTIVE, _VC_TIME_S, _VC_POLL_S
    prev_act = _VC_ACTIVE
    prev_t = _VC_TIME_S
    prev_p = _VC_POLL_S
    enable_virtual_clock(poll_ms=poll_ms, start_s=start_s)
    try:
        yield
    finally:
        _VC_ACTIVE = prev_act
        _VC_TIME_S = prev_t
        _VC_POLL_S = prev_p


def now_s() -> float:
    """Return monotonic clock seconds (wall clock or deterministic virtual work clock)."""
    if not _VC_ACTIVE:
        return time.perf_counter()
    global _VC_TIME_S
    _VC_TIME_S += _VC_POLL_S
    return _VC_TIME_S


def record_work_ms(work_ms: float) -> None:
    """Advance virtual clock by calibrated compute milliseconds when virtual clock is active."""
    if _VC_ACTIVE and work_ms > 0.0:
        global _VC_TIME_S
        _VC_TIME_S += float(work_ms) / 1000.0


def apply_cpsat_time_limit(params: object, max_time_s: float) -> None:
    """Configure OR-Tools CpSolver parameters with deterministic limit when virtual clock is on."""
    cap = max(0.004, float(max_time_s))
    setattr(params, "max_time_in_seconds", cap)
    if _VC_ACTIVE and hasattr(params, "max_deterministic_time"):
        setattr(params, "max_deterministic_time", cap)
