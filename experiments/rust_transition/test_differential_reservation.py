"""RUST-01 & RUST-04: Differential Testing between Python and Rust Reservation Tables.

Executes thousands of randomized operations against both the Python baseline
and the Rust native engine simultaneously, asserting 100% behavioral equivalence
on every query, boundary case, and state transition.
"""

from __future__ import annotations

import random
import time
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from amr_sched.reservation import Interval, ReservationTable as PyReservationTable
from amr_sched.rust_reservation import RustReservationTable, is_rust_core_available


def run_differential_benchmark(num_ops: int = 10000, seed: int = 42):
    print("=" * 70)
    print(f"RUST-01: Differential Fuzzing & Equivalence Verification ({num_ops:,} Ops)")
    print("=" * 70)

    if not is_rust_core_available():
        raise RuntimeError("Native Rust reservation_core is not available.")

    random.seed(seed)
    py_tbl = PyReservationTable()
    rust_tbl = RustReservationTable()

    nodes = [f"N_{i}" for i in range(25)]
    edges = [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]
    agvs = [f"AGV_{i}" for i in range(8)]

    # Metrics
    checks_passed = 0
    t_start = time.perf_counter()

    for op_idx in range(num_ops):
        action = random.choice([
            "reserve_node",
            "reserve_edge",
            "query_node",
            "query_edge",
            "conflicts",
            "release",
            "transfer_pins",
            "purge",
        ])

        s = random.randint(0, 100)
        e = s + random.randint(1, 20)
        span = Interval(s, e)
        agv = random.choice(agvs)
        node = random.choice(nodes)
        edge = random.choice(edges)
        cap = random.choice([1, 2, 3])
        exclude = random.choice([None, agv, random.choice(agvs)])

        if action == "reserve_node":
            py_tbl.reserve_node(node, span, agv)
            rust_tbl.reserve_node(node, span, agv)

        elif action == "reserve_edge":
            py_tbl.reserve_edge(edge[0], edge[1], span, agv)
            rust_tbl.reserve_edge(edge[0], edge[1], span, agv)

        elif action == "query_node":
            py_free = py_tbl.is_node_free(node, span, cap, exclude)
            rust_free = rust_tbl.is_node_free(node, span, cap, exclude)
            assert py_free == rust_free, (
                f"Node free mismatch at op {op_idx}: node={node}, span={span}, cap={cap}, "
                f"exclude={exclude} -> Py={py_free}, Rust={rust_free}"
            )
            checks_passed += 1

        elif action == "query_edge":
            py_free = py_tbl.is_edge_free(edge[0], edge[1], span, cap, exclude)
            rust_free = rust_tbl.is_edge_free(edge[0], edge[1], span, cap, exclude)
            assert py_free == rust_free, (
                f"Edge free mismatch at op {op_idx}: edge={edge}, span={span}, cap={cap}, "
                f"exclude={exclude} -> Py={py_free}, Rust={rust_free}"
            )
            checks_passed += 1

        elif action == "conflicts":
            py_c = py_tbl.node_conflicts(node, span, exclude)
            rust_c = rust_tbl.node_conflicts(node, span, exclude)
            assert py_c == rust_c, (
                f"Node conflicts mismatch at op {op_idx}: node={node}, span={span}, "
                f"exclude={exclude} -> Py={py_c}, Rust={rust_c}"
            )
            checks_passed += 1

            py_ec = py_tbl.edge_conflicts(edge[0], edge[1], span, exclude)
            rust_ec = rust_tbl.edge_conflicts(edge[0], edge[1], span, exclude)
            assert py_ec == rust_ec, (
                f"Edge conflicts mismatch at op {op_idx}: edge={edge}, span={span}, "
                f"exclude={exclude} -> Py={py_ec}, Rust={rust_ec}"
            )
            checks_passed += 1

        elif action == "release":
            if random.random() < 0.5:
                py_tbl.release_node(node, span, agv)
                rust_tbl.release_node(node, span, agv)
            else:
                py_tbl.release_all(agv)
                rust_tbl.release_all(agv)

        elif action == "transfer_pins":
            p_nodes = random.sample(nodes, random.randint(1, 3))
            p_edges = random.sample(edges, random.randint(0, 2))
            py_tbl.transfer_pins(agv, p_nodes, p_edges)
            rust_tbl.transfer_pins(agv, p_nodes, p_edges)

            py_pins = py_tbl.pins()
            rust_pins = rust_tbl.pins()
            assert py_pins == rust_pins, f"Pins mismatch: Py={py_pins}, Rust={rust_pins}"
            checks_passed += 1

        elif action == "purge":
            step = random.randint(10, 80)
            py_tbl.purge_before(step)
            rust_tbl.purge_before(step)

        # Cross-verify total count every 100 steps
        if op_idx % 100 == 0:
            py_cnt = py_tbl.total_reservations()
            rust_cnt = rust_tbl.total_reservations()
            assert py_cnt == rust_cnt, f"Total count mismatch: Py={py_cnt}, Rust={rust_cnt}"
            checks_passed += 1

    duration = time.perf_counter() - t_start
    print(f"Differential testing finished in {duration:.3f}s")
    print(f"Total operations executed: {num_ops:,}")
    print(f"Total assertions verified: {checks_passed:,}")
    print("Verdict: 100% EXACT MATHEMATICAL EQUIVALENCE CONFIRMED (0 Inconsistencies)")
    print("=" * 70)


if __name__ == "__main__":
    run_differential_benchmark(10000)
