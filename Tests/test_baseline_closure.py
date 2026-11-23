"""Unit tests for baseline closure additions: IJPR 2025 DFJSP-T H-MARL, COR 2026 SPGA, and COR 2026 LBBD/CP."""

from __future__ import annotations

import copy
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SCRATCH_MOD_DIR = ROOT / "_scratch"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRATCH_MOD_DIR) not in sys.path:
    sys.path.insert(0, str(SCRATCH_MOD_DIR))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.baselines.dfjspt_hmarl_ijpr25 import (
    DFJSPTHMARLNet,
    DFJSPTHMARLProposer,
    export_dfjspt_npz,
    load_dfjspt_model,
)
from amr_sched.baselines.lbbd_cp_cor26 import solve_lbbd_cp_cor26
from amr_sched.baselines.policy_registry import resolve_policy_name
from amr_sched.cvcr.commitment import classify_commitments
from amr_sched.cvcr.snapshot import NativeSnapshotCapture
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.rl.v19_hetero_constructor import graph_to_torch_tensors
from amr_sched.rl.v19_hetero_graph import extract_v19_hetero_graph
from amr_sched.simulation import Simulator
from tools.benchmarks.compare_workshop_algorithms import build_track1_scenario


def test_dfjspt_hmarl_pt_npz_parity_and_lock_safety(tmp_path: Path) -> None:
    """Verify exact PyTorch <-> pure-NumPy parity (<1e-5) and LOCKED commitment safety."""
    sc_tuple, _p2, _ms = build_track1_scenario(901, "nominal")
    net_map, tasks, agvs, ops, _ = copy.deepcopy(sc_tuple)
    sim = Simulator(net_map, tasks, agvs, StaticDijkstraPlanner(), GreedyAllocator())
    snap = NativeSnapshotCapture.capture(sim, ops)
    clf = classify_commitments(snap)
    g = extract_v19_hetero_graph(snap, net_map, clf=clf)

    torch.manual_seed(7)
    pt_net = DFJSPTHMARLNet(hidden_dim=64)
    pt_path = tmp_path / "dfjspt_test.pt"
    npz_path = tmp_path / "dfjspt_test.npz"
    torch.save({"model": pt_net.state_dict(), "config": pt_net.config_dict()}, pt_path)
    export_dfjspt_npz(pt_path, npz_path)

    m_pt, b_pt, _ = load_dfjspt_model(pt_path, backend="torch")
    m_np, b_np, _ = load_dfjspt_model(pt_path, backend="numpy")
    assert b_pt == "torch" and b_np == "numpy"

    with torch.no_grad():
        ten = graph_to_torch_tensors(g)
        gf_pt = m_pt.global_summary(ten)
        jl_pt, _ = m_pt.forward_job_agent(ten, gf_pt)
        bl_pt, cl_pt, _ = m_pt.forward_machine_agent(ten, gf_pt)

    gf_np = gf_pt[0].numpy()
    jl_np = m_np.predict_job_logits(g, gf_np)
    bl_np, cl_np = m_np.predict_machine_logits(g, gf_np)
    assert float(np.max(np.abs(jl_pt[0].numpy() - jl_np))) < 1e-5
    assert float(np.max(np.abs(bl_pt[0].numpy() - bl_np))) < 1e-5
    assert float(np.max(np.abs(cl_pt[0].numpy() - cl_np))) < 1e-5

    prop = DFJSPTHMARLProposer(pt_path, backend="numpy", use_rust=True, seed=0)(snap, net_map, budget_ms=35.0)
    assert prop.policy_id == "IJPR25_DFJSPT_HMARL"
    assert prop.telemetry["reproduction_tier"] == "paper_faithful_reproduction"
    for ftid in clf.frozen_tasks:
        orig = next((t.assigned_agv for t in snap.tasks if t.task_id == ftid), None)
        if orig is not None and ftid in prop.proposed_assignment:
            assert prop.proposed_assignment[ftid] == orig


def test_lbbd_cp_cor26_valid_bounds_and_conflict_free() -> None:
    """Verify COR 2026 LBBD/CP yields valid LB <= UB and conflict-free verification."""
    sc_tuple, _p2, ms = build_track1_scenario(901, "nominal")
    res = solve_lbbd_cp_cor26(copy.deepcopy(sc_tuple), time_limit_s=5.0, max_iters=4, num_workers=4, max_steps=ms)
    assert res.conflict_free_verified is True
    assert res.status in ("OPTIMAL", "FEASIBLE")
    assert res.lower_bound <= res.upper_bound + 1e-3
    assert res.upper_bound > 0.0
    assert len(res.best_assign) > 0


def test_closure_policy_aliases() -> None:
    """Verify CoRe-Sched, SPGA, and DFJSP-T canonical aliases resolve cleanly."""
    assert resolve_policy_name("core_sched") == "V19_ADAPTIVE_RL"
    assert resolve_policy_name("CoRe-Sched") == "V19_ADAPTIVE_RL"
    assert resolve_policy_name("spga") == "SPGA_AMR"
    assert resolve_policy_name("dfjspt") == "IJPR25_DFJSPT_HMARL"
    assert resolve_policy_name("IJPR25_DFJSPT") == "IJPR25_DFJSPT_HMARL"
