"""External baseline solvers integrated under CVCR audit and transactional execution interfaces."""

from amr_sched.baselines.spga_cor2026 import (
    SPGAConfig,
    SPGADecodedPlan,
    SPGAProblem,
    SPGASolver,
    run_spga_policy_proposal,
)

__all__ = [
    "SPGAConfig",
    "SPGADecodedPlan",
    "SPGAProblem",
    "SPGASolver",
    "run_spga_policy_proposal",
]
