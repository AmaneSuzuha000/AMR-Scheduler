"""Verification suite for RUST-04 Atomic Reservation Commit and Execution Commitment Ledger.

Verifies:
1. Multi-hop atomic transactions commit all staged hops or none at all (ACID Atomicity).
2. Clean rollback: zero orphan reservations remain if any intermediate hop conflicts.
3. Opposing edge head-on conflict detection within atomic staging.
4. Randomized multi-agent transaction concurrency test (1000 randomized transactions).
5. Execution commitment ledger lifecycle (commit_hop -> has_commitment -> complete_hop).
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

# Add project root to sys.path
repo_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(repo_root))

from amr_sched.reservation import Interval
from amr_sched.rust_reservation import (
    RustExecutionCommitmentLedger,
    RustReservationTable,
    RustReservationTransaction,
    is_rust_core_available,
)


def test_atomic_transaction_success():
    print("Testing basic atomic multi-hop transaction commit...")
    table = RustReservationTable()
    tx = table.begin_transaction("AGV_1")

    tx.stage_node("N1", Interval(0, 5))
    tx.stage_edge("N1", "N2", Interval(5, 10))
    tx.stage_node("N2", Interval(10, 15))
    tx.stage_edge("N2", "N3", Interval(15, 20))
    tx.stage_node("N3", Interval(20, 25))

    assert tx.staged_count == 5
    assert tx.can_commit(table) is True

    committed = tx.commit(table)
    assert committed is True
    assert table.total_reservations() == 5

    # Verify all resources are actually reserved
    assert table.is_node_free("N1", Interval(2, 4), capacity=1) is False
    assert table.is_edge_free("N1", "N2", Interval(6, 8), capacity=1) is False
    assert table.is_node_free("N2", Interval(12, 14), capacity=1) is False
    assert table.is_edge_free("N2", "N3", Interval(16, 18), capacity=1) is False
    assert table.is_node_free("N3", Interval(22, 24), capacity=1) is False
    print("  -> Passed successfully.")


def test_atomic_transaction_conflict_zero_orphans():
    print("Testing conflict rollback and zero orphan reservations...")
    table = RustReservationTable()

    # Pre-reserve N3 at [20, 25) with OBSTACLE_AGV
    table.reserve_node("N3", Interval(20, 25), "OBSTACLE_AGV")
    assert table.total_reservations() == 1

    tx = table.begin_transaction("AGV_1")
    tx.stage_node("N1", Interval(0, 5))
    tx.stage_edge("N1", "N2", Interval(5, 10))
    tx.stage_node("N2", Interval(10, 15))
    tx.stage_edge("N2", "N3", Interval(15, 20))
    tx.stage_node("N3", Interval(20, 25))  # <--- Conflicts with OBSTACLE_AGV!

    assert tx.staged_count == 5
    assert tx.can_commit(table) is False

    committed = tx.commit(table)
    assert committed is False, "Transaction should fail due to conflict on N3"

    # CRITICAL CHECK: Total reservations must still be exactly 1!
    # No orphan reservations on N1, N1->N2, N2, or N2->N3!
    assert table.total_reservations() == 1
    assert table.is_node_free("N1", Interval(0, 5), capacity=1) is True
    assert table.is_edge_free("N1", "N2", Interval(5, 10), capacity=1) is True
    assert table.is_node_free("N2", Interval(10, 15), capacity=1) is True
    assert table.is_edge_free("N2", "N3", Interval(15, 20), capacity=1) is True
    print("  -> Passed: 0 orphan reservations left on rollback.")


def test_atomic_transaction_opposing_edge_conflict():
    print("Testing opposing edge head-on conflict detection...")
    table = RustReservationTable()

    # AGV_2 reserves N2 -> N1 at [5, 10)
    table.reserve_edge("N2", "N1", Interval(5, 10), "AGV_2")

    # AGV_1 wants to traverse N1 -> N2 at [5, 10)
    tx = table.begin_transaction("AGV_1")
    tx.stage_node("N1", Interval(0, 5))
    tx.stage_edge("N1", "N2", Interval(5, 10))  # Head-on conflict!
    tx.stage_node("N2", Interval(10, 15))

    assert tx.can_commit(table) is False
    assert tx.commit(table) is False
    assert table.total_reservations() == 1
    print("  -> Passed: Opposing edge conflict blocked transaction.")


def test_randomized_transaction_stress():
    print("Running randomized multi-agent transaction stress test (1,000 transactions)...")
    rng = random.Random(42)
    nodes = [f"N_{i}" for i in range(20)]
    table = RustReservationTable()

    successes = 0
    rollbacks = 0

    for i in range(1000):
        agv_id = f"AGV_{i % 10}"
        tx = table.begin_transaction(agv_id)
        start_t = rng.randint(0, 200)

        # Build random 3-hop path
        p_nodes = rng.sample(nodes, 4)
        t = start_t
        for j in range(3):
            u, v = p_nodes[j], p_nodes[j + 1]
            tx.stage_node(u, Interval(t, t + 2))
            t += 2
            tx.stage_edge(u, v, Interval(t, t + 5))
            t += 5
            tx.stage_node(v, Interval(t, t + 2))

        before_count = table.total_reservations()
        can = tx.can_commit(table)
        committed = tx.commit(table)
        assert can == committed

        if committed:
            successes += 1
            assert table.total_reservations() > before_count
        else:
            rollbacks += 1
            assert table.total_reservations() == before_count

    print(f"  -> Stress test completed: {successes} committed, {rollbacks} safely rolled back.")
    assert successes > 0 and rollbacks > 0


def test_execution_commitment_ledger():
    print("Testing ExecutionCommitmentLedger...")
    ledger = RustExecutionCommitmentLedger()
    assert ledger.active_count == 0
    assert ledger.has_active_commitment("AGV_1") is False

    ledger.commit_hop("AGV_1", "N1", "N2", Interval(10, 20))
    assert ledger.active_count == 1
    assert ledger.has_active_commitment("AGV_1") is True
    assert ledger.has_active_commitment("AGV_2") is False

    ledger.commit_hop("AGV_2", "N3", "N4", Interval(15, 25))
    assert ledger.active_count == 2

    # Complete AGV_1 hop
    ledger.complete_hop("AGV_1", "N2")
    assert ledger.has_active_commitment("AGV_1") is False
    assert ledger.active_count == 1
    assert ledger.has_active_commitment("AGV_2") is True

    # Complete AGV_2 hop
    ledger.complete_hop("AGV_2", "N4")
    assert ledger.has_active_commitment("AGV_2") is False
    assert ledger.active_count == 0
    print("  -> Passed: Execution commitment ledger lifecycle verified.")


if __name__ == "__main__":
    assert is_rust_core_available(), "Rust reservation_core library is required"
    print("=== RUST-04 Atomic Reservation Commit Test Suite ===")
    test_atomic_transaction_success()
    test_atomic_transaction_conflict_zero_orphans()
    test_atomic_transaction_opposing_edge_conflict()
    test_randomized_transaction_stress()
    test_execution_commitment_ledger()
    print("=== All RUST-04 Atomic Commit Tests Passed! ===")
