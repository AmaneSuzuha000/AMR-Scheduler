"""Unit tests for Track A: Wu et al. (EJOR 2026) 3D Warehouse VRP & Storage Policies."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from experiments.ejor26_warehouse_repro.models_and_solvers import (
    AttentionModelVRP,
    DiscreteSACVRPSolver,
    GeneticAlgorithmVRP,
    PPOVRPSolver,
    WuGRUAttentionVRP,
)
from experiments.ejor26_warehouse_repro.storage_policies import (
    STORAGE_POLICIES,
    assign_storage_slots,
    verify_storage_bijection,
)
from experiments.ejor26_warehouse_repro.warehouse_3d_env import Warehouse3DConfig, Warehouse3DEnv


def test_warehouse_3d_metric_and_tour_verifier() -> None:
    """3D travel metric is symmetric, non-negative, and capacity violations raise cleanly."""
    env = Warehouse3DEnv(Warehouse3DConfig())
    a = env.slot_to_xyz((0, 2, 1))
    b = env.slot_to_xyz((2, 6, 3))
    assert env.travel_time_3d(a, a) == 0.0
    assert env.travel_time_3d(a, b) == pytest.approx(env.travel_time_3d(b, a))
    assert env.travel_time_3d(env.depot_coord, a) > 0.0

    cat = env.build_sku_catalog(42)
    smap = assign_storage_slots(cat, env, "semi_circle", seed=42)
    inst = env.generate_instance(cat, smap, 12, 1001, "semi_circle", "chk_12")
    # Invalid route visiting only subset raises
    with pytest.raises(ValueError):
        env.evaluate_tour(inst, [0, 1, 2, 0])
    # Single route overloading capacity if all demands summed > 20 raises
    if float(inst.demands[1:].sum()) > inst.capacity:
        with pytest.raises(ValueError, match="Capacity exceeded"):
            env.evaluate_tour(inst, [0, *range(1, 13), 0])


def test_storage_policies_bijection_and_semicircle_proximity() -> None:
    """All 4 storage policies are strict 1-to-1 bijections and semi_circle minimizes 3D load-weighted depot distance."""
    env = Warehouse3DEnv(Warehouse3DConfig())
    cat = env.build_sku_catalog(7)
    pm_weighted_by_pol: dict[str, float] = {}
    turnover_weighted_by_pol: dict[str, float] = {}
    for pol in STORAGE_POLICIES:
        smap = assign_storage_slots(cat, env, pol, seed=7)
        verify_storage_bijection(smap, env.all_slots)
        pm_weighted_by_pol[pol] = float(
            sum(
                item.priority_pm * env.travel_time_3d(env.depot_coord, env.slot_to_xyz(smap[item.sku_id]))
                for item in cat
            )
        )
        turnover_weighted_by_pol[pol] = float(
            sum(
                item.turnover * env.travel_time_3d(env.depot_coord, env.slot_to_xyz(smap[item.sku_id]))
                for item in cat
            )
        )

    assert pm_weighted_by_pol["semi_circle"] < pm_weighted_by_pol["class_abc"]
    assert pm_weighted_by_pol["semi_circle"] < pm_weighted_by_pol["within_aisle"]
    assert pm_weighted_by_pol["semi_circle"] < pm_weighted_by_pol["random"]
    assert turnover_weighted_by_pol["semi_circle"] < turnover_weighted_by_pol["random"]
    assert turnover_weighted_by_pol["class_abc"] < turnover_weighted_by_pol["random"]


def test_all_five_track_a_solvers_produce_legal_routes() -> None:
    """All 5 solvers train cleanly and emit 100% capacity-feasible routes."""
    env = Warehouse3DEnv(Warehouse3DConfig())
    cat = env.build_sku_catalog(19)
    smap = assign_storage_slots(cat, env, "semi_circle", seed=19)
    inst = env.generate_instance(cat, smap, 12, 3001, "semi_circle", "smoke_12")

    cpu = torch.device("cpu")
    solvers = [
        WuGRUAttentionVRP(device=cpu, hidden_dim=32),
        AttentionModelVRP(device=cpu, hidden_dim=32),
        PPOVRPSolver(device=cpu, hidden_dim=32),
        DiscreteSACVRPSolver(device=cpu, hidden_dim=32),
        GeneticAlgorithmVRP(pop_size=12, generations=5, seed=19),
    ]
    for sol in solvers:
        if isinstance(sol, GeneticAlgorithmVRP):
            sol.fit(env, 12)
        else:
            sol.fit(env, 12, updates=2, batch_size=4, seed=19)
        route, total_t = sol.solve(inst)
        chk = env.evaluate_tour(inst, route)
        assert chk["valid"] is True
        assert np.isfinite(total_t) and total_t > 0.0
