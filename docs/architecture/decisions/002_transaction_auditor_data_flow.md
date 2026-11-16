# ADR 002: Single-Writer Simulator, Transactional Commit, and Independent Audit Data Flow

## Context
Online multi-AMR rescheduling in industrial workshops must satisfy strict physical safety constraints:
1. Vehicles carrying cargo cannot have their task revoked or destination mutated mid-flight.
2. Vehicles actively occupying physical road segments cannot have their reservation erased.
3. Rescheduling proposals must be verified against stale tick versions before committing.
4. Independent verification must inspect actual historical trajectories rather than merely trust planning returns.

## Decision
We establish a strict three-phase lifecycle:
$$\text{Capture Immutable Snapshot} \longrightarrow \text{Algorithmic Optimization} \longrightarrow \text{Transactional Atomic Commit} \longrightarrow \text{Step-by-step Audit}$$

```
 Simulator (Authoritative Owner)
     │
     ▼ capture(sim, ops, hide_future_releases=True)
 NativeSnapshot (Read-Only Point-in-Time Frozen View)
     │
     ▼ LGCVCRPolicy.propose(budget_ms=150.0)
 CandidateProposal (Proposed Task Pins, Bay Diversions, Fleet Gates)
     │
     ▼ NativeTransactionOwner.validate_and_commit(proposal)
 [Validation Checks: Tick Stale? Irrevocable Task Modified? Edge Valid?]
     ├── If Rejected ──> Rollback, keep existing schedule
     └── If Accepted ──> Atomic State Mutation & Task Pinning
     │
     ▼ sim.step_once() & tracker.step()
 StepRecord / _trajectory
     │
     ▼ IndependentAuditor.audit_simulation_run(sim, tracker.operations)
 Trajectory Verification (Node Capacity, Edge Validity, Precedence Mutexes)
```

## Key Invariants Enforced
1. **No Silent State Corruption**:
   `NativeTransactionOwner` validates every precondition before mutating `Simulator` fields. If any check fails, the transaction is rejected as a whole (`receipt.accepted = False`).
2. **True Trajectory Auditing**:
   `IndependentAuditor` iterates through `sim._trajectory` step-by-step. It evaluates actual positions and node occupancies at each timestamp $t$, checking against vertex capacity $C(v)$, edge directional connectivity $E$, and production mutex intervals.
3. **Commitment Preservation**:
   Any task with status `TaskStatus.PICKED`, `LOADING`, `UNLOADING`, or in active transit is classified as `FROZEN_COMMITMENT`. Any candidate mutating its assigned robot or dropoff station is rejected with code `VIOLATION_IRREVOCABLE_TASK_MUTATION`.
