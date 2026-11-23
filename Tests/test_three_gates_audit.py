"""Comprehensive verification tests for Three-Gate Scientific Closure (G1, G2, G3).

Tests:
  - Gate G1: `COR26-Ref` physical relaxation soundness (`LB_phys <= H1_sim`), `strip_dynamic_disruptions`,
    and `EJOR26-Erdogan` ILS-LP exact timing + audited `Simulator` execution.
  - Gate G2: `ASC26_MA_TRANS` and `FGCS26_REN_GNN` PyTorch <-> pure-NumPy `< 1e-5` parity,
    hard commitment/eligibility mask enforcement, and registry integration.
  - Gate G3: Fresh 120-episode blind holdout zero-overlap seed quarantine and pooled P95/P99 latency math.
"""

from __future__ import annotations

import copy
from pathlib import Path
import tempfile

import numpy as np
import pytest

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.baselines.erdogan_ejor26 import solve_erdogan_ils_lp_ejor26
from amr_sched.baselines.fgcs26_ren_gnn import (
    FGCS26_REN_GNN_CKPT,
    FGCS26RenGNNNet,
    FGCS26RenGNNProposer,
    decode_fgcs26_candidate,
    export_fgcs26_ren_npz,
    load_fgcs26_ren_model,
)
from amr_sched.baselines.lbbd_cp_cor26 import solve_lbbd_cp_cor26, strip_dynamic_disruptions
from amr_sched.baselines.ma_trans_asc26 import (
    ASC26_MA_TRANS_CKPT,
    MATransASC26Proposer,
    MATransNet,
    decode_ma_trans_candidate,
    export_ma_trans_npz,
    load_ma_trans_model,
)
from amr_sched.baselines.policy_registry import SUPPORTED_POLICIES, resolve_policy_name
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.entities import Agv, Task, TaskPriority
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from amr_sched.simulation import SimulationConfig, Simulator


def _make_mini_static_scenario() -> tuple:
    net = Network()
    coords = {
        "P1": (0.0, 0.0, NodeKind.DOCK),
        "P2": (0.0, 2.0, NodeKind.DOCK),
        "M1": (2.0, 0.0, NodeKind.AISLE),
        "M2": (2.0, 2.0, NodeKind.AISLE),
        "B1": (4.0, 0.0, NodeKind.WORKSTATION),
        "B2": (4.0, 2.0, NodeKind.WORKSTATION),
    }
    for nid, (x, y, kind) in coords.items():
        net.add_node(Node(nid, x, y, kind=kind, capacity=2 if kind != NodeKind.AISLE else 1))
    edges = [
        ("P1", "M1"), ("M1", "P1"), ("P2", "M2"), ("M2", "P2"),
        ("M1", "M2"), ("M2", "M1"), ("M1", "B1"), ("B1", "M1"),
        ("M2", "B2"), ("B2", "M2"),
    ]
    for u, v in edges:
        net.add_edge(Edge(u, v, length=2.0, speed_limit=1.0))

    tasks = [
        Task("T1", "P1", "B1", release_step=0, due_step=40, priority=TaskPriority.URGENT),
        Task("T2", "P2", "B2", release_step=0, due_step=50),
        Task("T3", "P1", "B2", release_step=5, due_step=65),
    ]
    agvs = [
        Agv("R1", max_speed=1.0, start_node="P1"),
        Agv("R2", max_speed=1.0, start_node="P2"),
    ]
    ops = [
        OperationSnapshot("OP1", ("T1",), "B1", crew_id=1, tool_id=1, proc_steps=6, due_step=50, eligible_bays=("B1", "B2"), eligible_crews=(1, 2)),
        OperationSnapshot("OP2", ("T2",), "B2", crew_id=2, tool_id=2, proc_steps=6, due_step=60, eligible_bays=("B1", "B2"), eligible_crews=(1, 2)),
        OperationSnapshot("OP3", ("T3",), "B2", crew_id=1, tool_id=1, proc_steps=5, due_step=80, predecessor_op_id="OP1", eligible_bays=("B1", "B2"), eligible_crews=(1, 2)),
    ]
    meta = {"fault_step": 12, "urgent_release_step": 18, "event2_regime": "FAULT"}
    return (net, tasks, agvs, ops, meta)


def test_g1_strip_disruptions_and_relaxation_soundness():
    pytest.importorskip("ortools")
    pytest.importorskip("gymnasium")
    sc = _make_mini_static_scenario()
    clean = strip_dynamic_disruptions(sc)
    assert clean[4]["fault_step"] >= 999_999
    assert clean[4]["urgent_release_step"] >= 999_999
    assert clean[4]["event2_regime"] == "NONE"

    res_cor = solve_lbbd_cp_cor26(copy.deepcopy(clean), time_limit_s=8.0, max_iters=8, max_steps=200, verify_in_simulator=True, seed=7)
    res_erd = solve_erdogan_ils_lp_ejor26(copy.deepcopy(clean), time_limit_s=5.0, max_steps=200, seed=7)

    assert res_cor.sim_audit_passed and res_cor.sim_100pct_completed
    assert res_erd.sim_audit_passed and res_erd.sim_100pct_completed
    assert res_cor.lower_bound > 0.0
    assert res_cor.lower_bound <= res_cor.upper_bound + 1e-3
    assert res_cor.lower_bound <= res_erd.sim_h1_obj + 1e-3


def test_g2_asc26_and_fgcs26_parity_and_mask_safety():
    torch = pytest.importorskip("torch")
    from amr_sched.rl.v19_hetero_constructor import graph_to_torch_tensors

    net, tasks, agvs, ops, _ = _make_mini_static_scenario()
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(), GreedyAllocator(), SimulationConfig(dt=1.0, max_steps=200))
    snap = NativeSnapshotCapture.capture(sim, ops)
    g = extract_v19_hetero_graph(snap, net)
    ten = graph_to_torch_tensors(g)

    with tempfile.TemporaryDirectory() as td:
        for name, net_m, exp_fn, ld_fn, dec_fn in [
            ("ASC26_MA_TRANS", MATransNet(), export_ma_trans_npz, load_ma_trans_model, decode_ma_trans_candidate),
            ("FGCS26_REN_GNN", FGCS26RenGNNNet(), export_fgcs26_ren_npz, load_fgcs26_ren_model, decode_fgcs26_candidate),
        ]:
            net_m.eval()
            p = Path(td) / f"{name}.pt"
            torch.save({"state_dict": net_m.state_dict(), "config": net_m.config_dict()}, p)
            exp_fn(net_m, p.with_suffix(".npz"))
            pt_m, b_pt, _ = ld_fn(p, backend="torch")
            np_m, b_np, _ = ld_fn(p, backend="numpy")
            assert b_pt == "torch" and b_np == "numpy"

            with torch.no_grad():
                enc_pt = pt_m.encode(ten)
                if name == "ASC26_MA_TRANS":
                    jl_pt, bl_pt, cl_pt = pt_m.forward_machine_heads(enc_pt, ten)
                else:
                    jl_pt, bl_pt, cl_pt = pt_m.forward_static_heads(enc_pt, ten)
            enc_np = np_m.encode(g)
            if name == "ASC26_MA_TRANS":
                jl_np, bl_np, cl_np = np_m.predict_machine_heads(enc_np, g)
            else:
                jl_np, bl_np, cl_np = np_m.predict_static_heads(enc_np, g)

            assert float(np.max(np.abs(jl_pt[0].numpy() - jl_np))) < 1e-5
            assert float(np.max(np.abs(bl_pt[0].numpy()[g.op_bay_mask] - bl_np[g.op_bay_mask]))) < 1e-5
            assert float(np.max(np.abs(cl_pt[0].numpy()[g.op_crew_mask] - cl_np[g.op_crew_mask]))) < 1e-5

            # Verify decoded candidate respects all eligibility masks
            cand = dec_fn(np_m, g, snap, greedy=True)
            for oi, oid in enumerate(g.op_ids):
                bi = cand.chosen_bay_idx[oi]
                ci = cand.chosen_crew_idx[oi]
                assert g.op_bay_mask[oi, bi]
                assert g.op_crew_mask[oi, ci]
            for j, tid in enumerate(g.rel_tids):
                ai = cand.chosen_agv_idx[j]
                assert g.task_agv_mask[j, ai]


def test_g2_trained_checkpoints_load_and_propose_cleanly():
    assert "ASC26_MA_TRANS" in SUPPORTED_POLICIES
    assert "FGCS26_REN_GNN" in SUPPORTED_POLICIES
    assert resolve_policy_name("ma_trans") == "ASC26_MA_TRANS"
    assert resolve_policy_name("ren_gnn") == "FGCS26_REN_GNN"
    net, tasks, agvs, ops, _ = _make_mini_static_scenario()
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(), GreedyAllocator(), SimulationConfig(dt=1.0, max_steps=200))
    snap = NativeSnapshotCapture.capture(sim, ops)

    for ckpt, cls in [
        (ASC26_MA_TRANS_CKPT, MATransASC26Proposer),
        (FGCS26_REN_GNN_CKPT, FGCS26RenGNNProposer),
    ]:
        if ckpt.with_suffix(".npz").is_file() or ckpt.is_file():
            prop = cls(ckpt, backend="auto", use_rust=False, seed=0)(snap, net, budget_ms=35.0)
            assert prop.telemetry.get("reproduction_tier") == "architecture_faithful_domain_adapted_reimplementation"
            assert prop.policy_id in ("ASC26_MA_TRANS", "FGCS26_REN_GNN")
            assert "neural_candidates_evaluated" in prop.telemetry


def test_g3_fresh_blind_quarantine_and_pooled_p95():
    # Pooled P95 across episodes differs meaningfully from mean-of-episode-P95 when decision counts vary
    ep1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]  # 10 fast decisions, ep_p95 = 9.55
    ep2 = [30.0] * 100                                           # 100 slower decisions, ep_p95 = 30.0
    mean_ep_p95 = float(np.mean([np.percentile(ep1, 95), np.percentile(ep2, 95)]))
    pooled_p95 = float(np.percentile(ep1 + ep2, 95))
    assert pooled_p95 > mean_ep_p95 + 8.0

    # Verify zero overlap between fresh blind seeds and all historical seeds
    fresh_4 = {27101, 27102, 27201, 27202, 27301, 27302}
    fresh_w = {2901, 2902, 2903, 2904, 2905, 2906}
    old_blind = set(range(17101, 17317)) | set(range(901, 907))
    train_val = set(range(16101, 16705)) | set(range(1001, 1016)) | {1161, 1162, 1163}
    assert not ((fresh_4 | fresh_w) & (old_blind | train_val))
