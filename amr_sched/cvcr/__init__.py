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
from amr_sched.cvcr.features import (
    FEATURE_NAMES,
    extract_task_features,
    load_frozen_cie_models,
)
from amr_sched.cvcr.policy import CandidateProposal, LGCVCRPolicy
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
    "LGCVCRPolicy",
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
