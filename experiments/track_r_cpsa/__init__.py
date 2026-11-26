"""Track R (`TRACK-R-CPSA-01`): Pure MAPF / Lifelong-MAPD & Execution-Aware Local Reservation Repair.

Strictly decoupled from complex FMS scheduling (V18-RGRO, DFL, CRC, production DAGs).
Anchored cleanly on CPS-A* (Guo et al., Nov 2026) and its Section 6 Future Work.
"""

from __future__ import annotations

from experiments.track_r_cpsa.cpsa_pure_runner import (
    LifelongResult,
    OneShotResult,
    reserve_mapf_path,
    run_lifelong_mapd,
    run_oneshot_mapf,
    verify_mapf_trace_safety,
)
from experiments.track_r_cpsa.execution_delay_sim import (
    DelayScenarioConfig,
    DelayedLifelongResult,
    simulate_lifelong_with_delays,
)
from experiments.track_r_cpsa.local_reservation_repair import (
    EALRConfig,
    detect_reservation_mismatches,
    local_reservation_repair,
    propagate_reservation_dependencies,
)
from experiments.track_r_cpsa.maps_and_workloads import (
    GridMapSpec,
    build_map_a,
    build_map_b,
    build_map_c,
    generate_fixed_bg_and_test_requests,
    generate_lifelong_requests,
    generate_oneshot_requests,
    load_movingai_map,
    load_movingai_scen,
)

__all__ = [
    "DelayScenarioConfig",
    "DelayedLifelongResult",
    "EALRConfig",
    "GridMapSpec",
    "LifelongResult",
    "OneShotResult",
    "build_map_a",
    "build_map_b",
    "build_map_c",
    "detect_reservation_mismatches",
    "generate_fixed_bg_and_test_requests",
    "generate_lifelong_requests",
    "generate_oneshot_requests",
    "load_movingai_map",
    "load_movingai_scen",
    "local_reservation_repair",
    "propagate_reservation_dependencies",
    "reserve_mapf_path",
    "run_lifelong_mapd",
    "run_oneshot_mapf",
    "simulate_lifelong_with_delays",
    "verify_mapf_trace_safety",
]
