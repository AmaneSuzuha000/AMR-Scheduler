"""Gate 0: Execution Integrity Audit across 7 deterministic feasibility scenarios.

Diagnoses planner failures, discrete-vs-continuous timing mismatch, shield
conservativeness, right-of-way deadlocks, and moving-waypoint reward distortion
BEFORE any Gate 1 / Gate 2 RL work begins.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.network import Edge, Network, Node, NodeKind
from experiments.res_guided_avoidance.geometry_and_shield import BoxObstacle, clip_val, wrap_pi
from experiments.res_guided_avoidance.metric_world import (
    ContTask,
    CorridorCapsule,
    DynamicIntruder,
    MetricMap,
    MultiRobotUnicycleSim,
    build_occluded_junction_map,
    build_workshop_metric_map,
)


def _make_straight_corridor_map(length_m: float = 16.0, half_width_m: float = 1.3) -> MetricMap:
    net = Network()
    xs = [0.0, 4.0, 8.0, 12.0, length_m]
    for i, x in enumerate(xs):
        nid = f"N{i}"
        net.add_node(Node(nid, x, 0.0, NodeKind.AISLE))
    for i in range(len(xs) - 1):
        u, v = f"N{i}", f"N{i+1}"
        d = xs[i + 1] - xs[i]
        net.add_edge(Edge(u, v, d))
        net.add_edge(Edge(v, u, d))
    caps = tuple(CorridorCapsule(xs[i], 0.0, xs[i + 1], 0.0, half_width_m) for i in range(len(xs) - 1))
    ehw = {(f"N{i}", f"N{i+1}"): half_width_m for i in range(len(xs) - 1)}
    ehw.update({(f"N{i+1}", f"N{i}"): half_width_m for i in range(len(xs) - 1)})
    return MetricMap(
        name=f"Straight_{2*half_width_m:.1f}m",
        network=net,
        world_bounds=(-1.5, -half_width_m - 0.5, length_m + 1.5, half_width_m + 0.5),
        corridor_capsules=caps,
        edge_half_widths=ehw,
        pickup_nodes=("N1",),
        dropoff_nodes=("N4",),
        start_pool=("N0",),
        task_pairs=(("N1", "N4"),),
        macro_step_s=4.0,
    )


def _make_turn90_map(half_width_m: float = 1.3) -> MetricMap:
    net = Network()
    pts = [("N0", 0.0, 0.0), ("N1", 5.0, 0.0), ("N2", 10.0, 0.0), ("N3", 10.0, 5.0), ("N4", 10.0, 10.0)]
    for nid, x, y in pts:
        net.add_node(Node(nid, x, y, NodeKind.AISLE))
    caps = []
    ehw = {}
    for (u, ux, uy), (v, vx, vy) in zip(pts[:-1], pts[1:]):
        d = math.hypot(vx - ux, vy - uy)
        net.add_edge(Edge(u, v, d))
        net.add_edge(Edge(v, u, d))
        caps.append(CorridorCapsule(ux, uy, vx, vy, half_width_m))
        ehw[(u, v)] = half_width_m
        ehw[(v, u)] = half_width_m
    return MetricMap(
        name="Turn90_2.6m",
        network=net,
        world_bounds=(-2.0, -2.0, 12.0, 12.0),
        corridor_capsules=tuple(caps),
        edge_half_widths=ehw,
        pickup_nodes=("N1",),
        dropoff_nodes=("N4",),
        start_pool=("N0",),
        task_pairs=(("N1", "N4"),),
        macro_step_s=5.0,
    )


def _make_headon_bay_map() -> MetricMap:
    """2.4m narrow corridor with a 3.6m wide central passing bay + parallel bypass loop."""
    net = Network()
    coords = {
        "W0": (0.0, 0.0),
        "W1": (4.0, 0.0),
        "C0": (8.0, 0.0),
        "E1": (12.0, 0.0),
        "E0": (16.0, 0.0),
        "BAY": (8.0, 2.6),
    }
    for nid, (x, y) in coords.items():
        net.add_node(Node(nid, x, y, NodeKind.AISLE))
    edges = [("W0", "W1", 1.20), ("W1", "C0", 1.20), ("C0", "E1", 1.20), ("E1", "E0", 1.20),
             ("W1", "BAY", 1.30), ("BAY", "E1", 1.30), ("C0", "BAY", 1.30)]
    caps = []
    ehw = {}
    for u, v, hw in edges:
        nu, nv = net.node(u), net.node(v)
        d = math.hypot(nv.x - nu.x, nv.y - nu.y)
        net.add_edge(Edge(u, v, d))
        net.add_edge(Edge(v, u, d))
        caps.append(CorridorCapsule(nu.x, nu.y, nv.x, nv.y, hw))
        ehw[(u, v)] = hw
        ehw[(v, u)] = hw
    return MetricMap(
        name="HeadOn_PassingBay",
        network=net,
        world_bounds=(-2.0, -2.0, 18.0, 5.0),
        corridor_capsules=tuple(caps),
        edge_half_widths=ehw,
        pickup_nodes=("W0", "E0"),
        dropoff_nodes=("E0", "W0"),
        start_pool=("W0", "E0"),
        task_pairs=(("W0", "E0"), ("E0", "W0")),
        macro_step_s=3.5,
    )


def _deterministic_follower(sim: MultiRobotUnicycleSim, rid: str) -> tuple[float, float]:
    rb = sim.robots[rid]
    wx, wy = sim._advance_route_cursor(rb)
    # Shift local target laterally when an obstacle is strictly ahead in a corridor with bypass room
    obs_vis = sim.get_perceived_circles(rid)
    ct, st = math.cos(-rb.theta), math.sin(-rb.theta)
    lat_shift = 0.0
    v_cap = 0.95
    for o in obs_vis:
        lx = ct * (o.x - rb.x) - st * (o.y - rb.y)
        ly = st * (o.x - rb.x) + ct * (o.y - rb.y)
        if 0.05 < lx < 3.2 and abs(ly) < (sim.fp.radius_m + o.radius + 0.18):
            lat_shift += (-0.80 if ly >= 0.0 else 0.80)
            v_cap = min(v_cap, 0.72)
    if abs(lat_shift) > 1e-3:
        # Shift perpendicular to route segment tangent, not robot heading
        if rb.route_idx + 1 < len(rb.route_nodes):
            u_n = sim.mmap.network.node(rb.route_nodes[rb.route_idx])
            v_n = sim.mmap.network.node(rb.route_nodes[rb.route_idx + 1])
            tdx, tdy = v_n.x - u_n.x, v_n.y - u_n.y
            tln = max(1e-6, math.hypot(tdx, tdy))
            tx, ty = tdx / tln, tdy / tln
        else:
            tx, ty = math.cos(rb.theta), math.sin(rb.theta)
        wx_t = wx - ty * lat_shift
        wy_t = wy + tx * lat_shift
    else:
        wx_t, wy_t = wx, wy
    ang = math.atan2(wy_t - rb.y, wx_t - rb.x)
    err = wrap_pi(ang - rb.theta)
    v = v_cap if abs(err) < 0.95 else (0.35 if abs(err) < 1.35 else 0.10)
    return v, clip_val(2.2 * err, -1.65, 1.65)


def run_gate0_audit() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _run(label: str, sim: MultiRobotUnicycleSim, setup_fn=None) -> None:
        if setup_fn is not None:
            setup_fn(sim)
        res = sim.run_episode(_deterministic_follower, controller_name="Gate0_Follower")
        passed = bool(res.zero_collisions and res.completed_tasks >= sim.n_tasks)
        row = {
            "scenario": label,
            "passed": passed,
            "completed": f"{res.completed_tasks}/{sim.n_tasks}",
            "makespan_s": res.makespan_s,
            "zero_col": res.zero_collisions,
            "wall_col": res.wall_collisions,
            "peer_col": res.peer_collisions,
            "intr_col": res.intruder_collisions,
            "shield_interv": res.shield_interventions,
            "shield_infeas": res.shield_infeasible_count,
            "wait_ratio": res.wait_ratio,
            "replans": res.replans,
        }
        rows.append(row)
        print(
            f"[Gate0 | {label:20s}] pass={str(passed):5s} comp={row['completed']:5s} "
            f"ms={row['makespan_s']:6.2f}s col=({row['wall_col']},{row['peer_col']},{row['intr_col']}) "
            f"sh={row['shield_interv']:4d} inf={row['shield_infeas']:3d} wait={row['wait_ratio']:.3f} rep={row['replans']}",
            flush=True,
        )

    # 1. 1R_Straight
    _run("1R_Straight", MultiRobotUnicycleSim(_make_straight_corridor_map(), n_robots=1, n_tasks=1, horizon_s=35.0, seed=1, n_intruders=0))
    # 2. 1R_Turn90
    _run("1R_Turn90", MultiRobotUnicycleSim(_make_turn90_map(), n_robots=1, n_tasks=1, horizon_s=45.0, seed=1, n_intruders=0))
    # 3. 1R_StaticBypass
    def _add_static(sim: MultiRobotUnicycleSim) -> None:
        sim.intruders.append(DynamicIntruder(x=8.0, y=0.25, vx=0.0, vy=0.0, radius=0.35, x_lo=7.5, x_hi=8.5, y_lo=-0.5, y_hi=0.5))
    _run("1R_StaticBypass", MultiRobotUnicycleSim(_make_straight_corridor_map(half_width_m=1.65), n_robots=1, n_tasks=1, horizon_s=40.0, seed=1, n_intruders=0), _add_static)
    # 4. 1R_DynamicAvoid
    def _add_dyn(sim: MultiRobotUnicycleSim) -> None:
        sim.intruders.append(DynamicIntruder(x=10.5, y=0.0, vx=-0.35, vy=0.0, radius=0.32, x_lo=4.0, x_hi=13.0, y_lo=-0.5, y_hi=0.5))
    _run("1R_DynamicAvoid", MultiRobotUnicycleSim(_make_straight_corridor_map(half_width_m=1.65), n_robots=1, n_tasks=1, horizon_s=45.0, seed=1, n_intruders=0), _add_dyn)
    # 5. 2R_CrossJunction
    om = build_occluded_junction_map(cell_m=2.6)
    _run("2R_CrossJunction", MultiRobotUnicycleSim(om, n_robots=2, n_tasks=2, horizon_s=50.0, seed=7, n_intruders=0))
    # 6. 2R_HeadOnYield
    def _setup_headon(sim: MultiRobotUnicycleSim) -> None:
        sim.pending.clear()
        sim.ready.clear()
        sim.ready.extend([ContTask(0, "W0", "E0", release_s=0.0, due_s=60.0), ContTask(1, "E0", "W0", release_s=0.0, due_s=60.0)])
        r0, r1 = sim.robots["R_00"], sim.robots["R_01"]
        n_w, n_e = sim.mmap.network.node("W0"), sim.mmap.network.node("E0")
        r0.x, r0.y, r0.theta, r0.cur_node, r0.route_nodes = n_w.x, n_w.y, 0.0, "W0", ["W0"]
        r1.x, r1.y, r1.theta, r1.cur_node, r1.route_nodes = n_e.x, n_e.y, math.pi, "E0", ["E0"]
    _run("2R_HeadOnYield", MultiRobotUnicycleSim(_make_headon_bay_map(), n_robots=2, n_tasks=2, horizon_s=60.0, seed=1, n_intruders=0), _setup_headon)
    # 7. 4R_LifelongWorkshop
    wm = build_workshop_metric_map()
    _run("4R_LifelongWorkshop", MultiRobotUnicycleSim(wm, n_robots=4, n_tasks=4, horizon_s=300.0, seed=42, n_intruders=0))

    n_pass = sum(1 for r in rows if r["passed"])
    print(f"=== Gate 0 Summary: {n_pass}/{len(rows)} scenarios passed ===", flush=True)
    return rows


if __name__ == "__main__":
    out_p = ROOT / "out" / "ramp_study" / "gate0_audit.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    rows = run_gate0_audit()
    out_p.write_text(json.dumps(rows, indent=2), encoding="utf-8")
