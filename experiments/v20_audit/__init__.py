"""Stage-1 Scientific Headroom & Mechanism Audit (`IF-CORE-V20-01`) package.

Strict non-invasive research harness operating on top of frozen `CoRe-Sched v1.2.1`.
NEVER mutates `amr_sched/`, checkpoints, `results/*.csv`, or external `J_A` definitions.
"""

from __future__ import annotations

from experiments.v20_audit.common import (
    COHORT_A_SEEDS,
    COHORT_B_SEEDS,
    COHORT_C_RESERVED_SEEDS,
    OUT_AUDIT_DIR,
    TASK_C_OUTLIER_WHITELIST,
    CandidateArtifact,
    RecordedStep,
    SeqDiffEvalContext,
    assert_git_amr_sched_clean_relative_to_baseline,
    assert_stage1_quarantine,
    build_stratified_train_val_specs,
    build_val72_specs,
    build_val_static35_specs,
    candidate_action_signature,
    make_v18_external_proposer,
    replay_fork_env_at_tick,
    rollout_48_and_optional_full,
    summarize_episode_rows,
)

__all__ = [
    "COHORT_A_SEEDS",
    "COHORT_B_SEEDS",
    "COHORT_C_RESERVED_SEEDS",
    "OUT_AUDIT_DIR",
    "TASK_C_OUTLIER_WHITELIST",
    "CandidateArtifact",
    "RecordedStep",
    "SeqDiffEvalContext",
    "assert_git_amr_sched_clean_relative_to_baseline",
    "assert_stage1_quarantine",
    "build_stratified_train_val_specs",
    "build_val72_specs",
    "build_val_static35_specs",
    "candidate_action_signature",
    "make_v18_external_proposer",
    "replay_fork_env_at_tick",
    "rollout_48_and_optional_full",
    "summarize_episode_rows",
]
