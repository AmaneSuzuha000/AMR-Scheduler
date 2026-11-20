"""LG-CVCR × AMR-Scheduler Scientific Research Brain Integration."""
from __future__ import annotations

from amr_sched.cvcr.auditor import AuditReport, IndependentAuditor
from amr_sched.cvcr.commitment import (
    TIER_CONDITIONAL,
    TIER_EXECUTING_HARD,
    TIER_IRREVOCABLE,
    TIER_SOFT,
    TIER_UNCOMMITTED,
    CommitmentClassification,
    classify_commitments,
    select_release_set,
)
from amr_sched.cvcr.evaluator import CVCREvaluator, EvalResult
from amr_sched.cvcr.execution_feedback import (
    EXEC_FEEDBACK_SCHEMA,
    AmrExecutionStatus,
    CorridorFeedback,
    ExecutionFeedbackPacket,
)
from amr_sched.cvcr.p2_perturbation import (
    ExecutionPerturbationModel,
    P2PerturbationConfig,
    ScheduledLocalFail,
    ScheduledObstacle,
)
from amr_sched.cvcr.p4_belief_choke import (
    BeliefDurationEstimate,
    CorridorDurationBeliefModel,
    corridor_detour_overhead,
    detect_choke_oscillation_agvs,
    plan_choke_passage_repairs,
    record_node_history_tick,
)
from amr_sched.cvcr.physical_bridge import (
    CorridorObstacleReport,
    Nav2RobotTelemetry,
    PhysicalCoordinationStepReport,
    PhysicalCVCRCoordinator,
)
from amr_sched.cvcr.features import (
    FEATURE_NAMES,
    extract_task_features,
    load_frozen_cie_models,
)
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy
from amr_sched.cvcr.v14_joint_rl import (
    V14JointPolicy,
    generate_v14_joint_candidates,
    load_v14_joint_policy,
)
from amr_sched.cvcr.v15_ea_joint import (
    V15AdaptiveController,
    V15ExecutionAwarePolicy,
    generate_v15_unified_candidates,
    load_v15_ea_policy,
)
from amr_sched.cvcr.v16_frontier import (
    V16AdaptiveController,
    V16FrontierPolicy,
    generate_v16_unified_candidates,
    load_v16_frontier_policy,
)
from amr_sched.cvcr.production import NativeProductionTracker, ProductionMetrics
from amr_sched.cvcr.snapshot import (
    NativeSnapshot,
    NativeSnapshotCapture,
    OperationSnapshot,
    ReservationSnapshot,
    RobotSnapshot,
    TaskSnapshot,
)
from amr_sched.cvcr.transaction import NativeTransactionOwner, TransactionReceipt

__all__ = [
    "NativeSnapshotCapture",
    "NativeSnapshot",
    "RobotSnapshot",
    "TaskSnapshot",
    "OperationSnapshot",
    "ReservationSnapshot",
    "CommitmentClassification",
    "classify_commitments",
    "select_release_set",
    "TIER_IRREVOCABLE",
    "TIER_EXECUTING_HARD",
    "TIER_CONDITIONAL",
    "TIER_SOFT",
    "TIER_UNCOMMITTED",
    "CVCREvaluator",
    "EvalResult",
    "EXEC_FEEDBACK_SCHEMA",
    "AmrExecutionStatus",
    "CorridorFeedback",
    "ExecutionFeedbackPacket",
    "P2PerturbationConfig",
    "ScheduledObstacle",
    "ScheduledLocalFail",
    "ExecutionPerturbationModel",
    "BeliefDurationEstimate",
    "CorridorDurationBeliefModel",
    "corridor_detour_overhead",
    "record_node_history_tick",
    "detect_choke_oscillation_agvs",
    "plan_choke_passage_repairs",
    "Nav2RobotTelemetry",
    "CorridorObstacleReport",
    "PhysicalCoordinationStepReport",
    "PhysicalCVCRCoordinator",
    "LGCVCRPolicy",
    "V14JointPolicy",
    "generate_v14_joint_candidates",
    "load_v14_joint_policy",
    "V15ExecutionAwarePolicy",
    "V15AdaptiveController",
    "generate_v15_unified_candidates",
    "load_v15_ea_policy",
    "V16AdaptiveController",
    "V16FrontierPolicy",
    "generate_v16_unified_candidates",
    "load_v16_frontier_policy",
    "CandidateProposal",
    "NativeTransactionOwner",
    "TransactionReceipt",
    "IndependentAuditor",
    "AuditReport",
    "NativeProductionTracker",
    "ProductionMetrics",
    "FEATURE_NAMES",
    "extract_task_features",
    "load_frozen_cie_models",
]
