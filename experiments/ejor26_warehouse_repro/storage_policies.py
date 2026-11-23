"""Storage location assignment policies for 3D warehouse VRP reproduction (Wu et al., EJOR 2026)."""

from __future__ import annotations

import random

from experiments.ejor26_warehouse_repro.warehouse_3d_env import SKUItem, Warehouse3DEnv

STORAGE_POLICIES: tuple[str, ...] = ("random", "within_aisle", "class_abc", "semi_circle")


def assign_storage_slots(
    catalog: list[SKUItem],
    env: Warehouse3DEnv,
    policy_name: str,
    *,
    seed: int = 0,
) -> dict[int, tuple[int, int, int]]:
    """Assign each SKU to a unique 3D rack slot under the specified storage policy."""
    slots = list(env.all_slots)
    if len(catalog) != len(slots):
        raise ValueError(f"Catalog size {len(catalog)} must match slot count {len(slots)}.")

    rng = random.Random(int(seed) * 7919 + sum(ord(ch) for ch in policy_name))

    if policy_name == "random":
        shuffled = list(slots)
        rng.shuffle(shuffled)
        mapping = {sku.sku_id: shuffled[i] for i, sku in enumerate(catalog)}
    elif policy_name == "within_aisle":
        # Fill aisle-by-aisle sequentially with highest-turnover SKUs in closest aisles/bays/tiers.
        skus_sorted = sorted(catalog, key=lambda s: (-s.turnover, s.sku_id))
        slots_sorted = sorted(slots, key=lambda sl: (sl[0], sl[1], sl[2]))
        mapping = {skus_sorted[i].sku_id: slots_sorted[i] for i in range(len(catalog))}
    elif policy_name == "class_abc":
        skus_sorted = sorted(catalog, key=lambda s: (-s.turnover, s.sku_id))
        slots_by_dist = sorted(
            slots,
            key=lambda sl: (env.travel_time_3d(env.depot_coord, env.slot_to_xyz(sl)), sl),
        )
        n = len(catalog)
        cut_a = int(round(0.20 * n))
        cut_b = int(round(0.50 * n))
        mapping = {}
        for lo, hi in ((0, cut_a), (cut_a, cut_b), (cut_b, n)):
            zone_slots = list(slots_by_dist[lo:hi])
            rng.shuffle(zone_slots)
            for idx, sku in enumerate(skus_sorted[lo:hi]):
                mapping[sku.sku_id] = zone_slots[idx]
    elif policy_name == "semi_circle":
        # Wu et al. (EJOR 2026): rank by composite priority p_m along semi-circular depot iso-time shells.
        skus_sorted = sorted(catalog, key=lambda s: (-s.priority_pm, -s.turnover, s.sku_id))
        slots_by_dist = sorted(
            slots,
            key=lambda sl: (env.travel_time_3d(env.depot_coord, env.slot_to_xyz(sl)), sl),
        )
        mapping = {skus_sorted[i].sku_id: slots_by_dist[i] for i in range(len(catalog))}
    else:
        raise ValueError(f"Unknown storage policy {policy_name!r}; expected one of {STORAGE_POLICIES}")

    verify_storage_bijection(mapping, slots)
    return mapping


def verify_storage_bijection(
    mapping: dict[int, tuple[int, int, int]],
    all_slots: list[tuple[int, int, int]] | tuple[tuple[int, int, int], ...],
) -> None:
    """Assert 1-to-1 bijection between SKUs and 3D storage slots."""
    expected_slots = set(all_slots)
    assigned_slots = list(mapping.values())
    if len(mapping) != len(expected_slots) or len(set(assigned_slots)) != len(expected_slots):
        raise ValueError("Storage mapping has duplicate or missing slot assignments.")
    if set(assigned_slots) != expected_slots:
        raise ValueError("Storage mapping does not cover the exact 3D slot set.")
