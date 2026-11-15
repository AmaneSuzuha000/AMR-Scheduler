"""RUST-01: Micro-Benchmark measuring pure query throughput between Python and Rust.
"""

from __future__ import annotations

import random
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from amr_sched.reservation import Interval, ReservationTable as PyReservationTable
from amr_sched.rust_reservation import RustReservationTable


def benchmark(num_queries: int = 100_000):
    print("=" * 70)
    print(f"RUST-01: Reservation Table Throughput Benchmark ({num_queries:,} Queries)")
    print("=" * 70)

    py_tbl = PyReservationTable()
    rust_tbl = RustReservationTable()

    # Prepopulate tables with 500 realistic reservations
    random.seed(12345)
    nodes = [f"N_{i}" for i in range(50)]
    edges = [(nodes[i], nodes[(i + 1) % len(nodes)]) for i in range(len(nodes))]
    agvs = [f"AGV_{i}" for i in range(16)]

    for _ in range(500):
        s = random.randint(0, 500)
        e = s + random.randint(2, 15)
        span = Interval(s, e)
        agv = random.choice(agvs)
        node = random.choice(nodes)
        edge = random.choice(edges)
        py_tbl.reserve_node(node, span, agv)
        rust_tbl.reserve_node(node, span, agv)
        py_tbl.reserve_edge(edge[0], edge[1], span, agv)
        rust_tbl.reserve_edge(edge[0], edge[1], span, agv)

    # Pre-generate query workloads so generation time is not included in measurement
    query_spans = [
        Interval(s, s + random.randint(1, 10))
        for s in (random.randint(0, 520) for _ in range(num_queries))
    ]
    query_nodes = [random.choice(nodes) for _ in range(num_queries)]
    query_caps = [random.choice([1, 2]) for _ in range(num_queries)]

    # 1. Benchmark Python
    t0 = time.perf_counter()
    for i in range(num_queries):
        py_tbl.is_node_free(query_nodes[i], query_spans[i], query_caps[i])
    t_py = time.perf_counter() - t0

    # 2. Benchmark Rust
    t0 = time.perf_counter()
    for i in range(num_queries):
        rust_tbl.is_node_free(query_nodes[i], query_spans[i], query_caps[i])
    t_rust = time.perf_counter() - t0

    py_qps = num_queries / t_py
    rust_qps = num_queries / t_rust
    speedup = t_py / t_rust

    print(f"Python: {t_py:.4f} s | {py_qps:,.0f} queries/sec")
    print(f"Rust:   {t_rust:.4f} s | {rust_qps:,.0f} queries/sec")
    print(f"Speedup: {speedup:.2f}x faster in native Rust!")
    print("=" * 70)


if __name__ == "__main__":
    benchmark(100_000)
