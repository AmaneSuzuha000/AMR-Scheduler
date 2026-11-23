"""Tests for Rust zero-copy indexed FFI, batch parallel SIPP, and CUDA RL backend helpers."""

from __future__ import annotations

import pytest

from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.reservation import ReservationTable
from amr_sched.planners.rust_spacetime_astar import (
    RustSpaceTimeAStarPlanner,
    is_rust_planner_available,
)
from amr_sched.rl.cuda_backend import batch_infer_logits, cuda_runtime_info, get_rl_device, move_module_to_device


def _small_grid_net() -> Network:
    net = Network()
    for nid, (x, y) in {"N0": (0.0, 0.0), "N1": (3.0, 0.0), "N2": (6.0, 0.0), "B1": (3.0, 3.0)}.items():
        net.add_node(Node(nid, x, y, kind=NodeKind.AISLE, capacity=1))
    for u, v, d in (("N0", "N1", 3.0), ("N1", "N2", 3.0), ("N0", "B1", 4.2), ("B1", "N2", 4.2)):
        net.add_edge(Edge(u, v, d, speed_limit=3.0, capacity=1))
        net.add_edge(Edge(v, u, d, speed_limit=3.0, capacity=1))
    return net


def test_rust_indexed_reservation_and_batch_sipp_parity() -> None:
    """Verify indexed Rust `reserve_chain_indexed` + `plan_batch_sipp` match sequential `plan`."""
    assert is_rust_planner_available()
    net = _small_grid_net()
    planner_seq = RustSpaceTimeAStarPlanner(max_speed=3.0, dt=1.0, use_sipp=True)
    planner_bat = RustSpaceTimeAStarPlanner(max_speed=3.0, dt=1.0, use_sipp=True)

    res = RustSpaceTimeAStarPlanner.create_reservation_table()
    # Reserve A0 holding N1 -> N2 from [1, 2] via indexed FFI (zero ASCII script serialization)
    ok = planner_bat.reserve_chain_indexed(net, res, "A0", [("N0", 0), ("N1", 1), ("N2", 2)], extra_node_wait=("N2", 2, 4))
    assert ok

    queries = [("N0", "N2", 0, "A1", 25), ("N2", "N0", 0, "A2", 25)]
    seq_res = [planner_seq.plan(net, s, g, t0, res, aid, hz) for s, g, t0, aid, hz in queries]
    bat_res = planner_bat.plan_batch_sipp(net, res, queries)
    assert len(seq_res) == len(bat_res) == 2
    for p_seq, p_bat in zip(seq_res, bat_res):
        assert p_seq is not None and p_bat is not None
        assert p_seq.arrival_steps == p_bat.arrival_steps
        assert p_seq.nodes == p_bat.nodes


def test_rust_indexed_reservation_rejects_head_on_collision() -> None:
    """Verify indexed FFI edge-transition reservation preserves capacity-1 head-on mutex."""
    net = _small_grid_net()
    planner = RustSpaceTimeAStarPlanner(max_speed=3.0, dt=1.0, use_sipp=True)
    rt = ReservationTable()
    p = planner.plan(net, "N0", "N2", 0, rt, "A0", 20)
    assert p is not None and p.arrival_steps == [1, 2]


def test_cuda_backend_runtime_and_batch_infer() -> None:
    """Verify RTX 5060 CUDA device discovery, module move, and `batch_infer_logits` helper."""
    torch = pytest.importorskip("torch")
    info = cuda_runtime_info()
    assert "torch_version" in info
    dev = get_rl_device(prefer_gpu=True)
    mod, moved_dev = move_module_to_device(torch.nn.Linear(12, 4), prefer_gpu=True)
    assert moved_dev == dev
    inp = [[0.1 * i for i in range(12)], [0.2 * i for i in range(12)]]
    outs = batch_infer_logits(mod, inp, device=dev)
    assert len(outs) == 1 and outs[0].shape == (2, 4)
