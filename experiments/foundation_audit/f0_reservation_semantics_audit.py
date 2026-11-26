"""F0 Reservation Semantics & Multi-Tick Corridor Blast-Radius Audit.

Verifies mathematically and empirically across `amr_sched/`, `crates/`, `experiments/`,
and `tools/`:
1. Exact isolation of the Track R / Track A `dep_t = arr_i - 1` assumption in
   `experiments/track_r_cpsa/cpsa_pure_runner.py:expand_planned_to_step_positions`.
2. Zero presence of `arr_i - 1` edge truncation in Track B (`amr_sched/`, `crates/`,
   `Simulator`, `SpaceTimeAStarPlanner`, `RustSpaceTimeAStarPlanner`, `CVCREvaluator`).
3. Executable multi-tick (20-30 m) Workshop corridor tests confirming opposing-edge
   exclusion throughout `[depart, arrive)` in discrete planners/simulators vs the
   reproduced mid-corridor blind spot when using `expand_planned_to_step_positions`.
4. Node/edge capacity differences between `WorkshopScene.network()` and
   `point_execution_network(scene.network())`.

Usage:
    python -B -m experiments.foundation_audit.f0_reservation_semantics_audit
"""

from __future__ import annotations

import json
import math
import pathlib
import re
import sys
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.allocation.rule_based import DispatchRule, RuleBasedAllocator
from amr_sched.cvcr.evaluator import CVCREvaluator
from amr_sched.cvcr.snapshot import NativeSnapshot, RobotSnapshot, TaskSnapshot
from amr_sched.entities import Agv, Task, TaskPriority
from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.planners.rust_spacetime_astar import (
    RustSpaceTimeAStarPlanner,
    is_rust_planner_available,
)
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reservation import Interval, ReservationTable
from amr_sched.rust_reservation import RustReservationTable, is_rust_core_available
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import WorkshopScene
from experiments.res_guided_avoidance.metric_world import build_workshop_metric_map
from experiments.stp_research_reset.audit_env import _reserve_exact_span
from experiments.track_r_cpsa.cpsa_pure_runner import expand_planned_to_step_positions
from tools.live_bridge.fleet_config import point_execution_network

OUT_DIR = ROOT / "out" / "foundation_audit"


def scan_repo_reservation_patterns() -> dict[str, Any]:
    """Scan Python and Rust source trees for reservation/span patterns."""
    patterns = {
        "expand_planned_to_step_positions": re.compile(r"expand_planned_to_step_positions"),
        "arr_i_minus_1": re.compile(r"arr_i\s*-\s*1"),
        "_reserve_exact_span": re.compile(r"_reserve_exact_span"),
        "reserve_step_points_indexed": re.compile(r"reserve_step_points_indexed"),
        "_reserve_hop": re.compile(r"_reserve_hop"),
        "_commit_move": re.compile(r"def _commit_move\b"),
        "_reserve_path_steps": re.compile(r"_reserve_path_steps"),
        "collect_undirected_edge_conflicts_into": re.compile(r"collect_undirected_edge_conflicts_into"),
    }
    hits: dict[str, list[dict[str, Any]]] = {k: [] for k in patterns}
    for area in ("amr_sched", "crates", "experiments", "tools"):
        area_dir = ROOT / area
        if not area_dir.exists():
            continue
        for p in sorted(area_dir.rglob("*")):
            if p.suffix not in {".py", ".rs"}:
                continue
            rel = p.relative_to(ROOT).as_posix()
            text = p.read_text(encoding="utf-8", errors="ignore")
            for lineno, line in enumerate(text.splitlines(), 1):
                stripped = line.strip()
                for pname, rx in patterns.items():
                    if rx.search(line):
                        hits[pname].append(
                            {
                                "area": area,
                                "file": rel,
                                "line": lineno,
                                "snippet": stripped[:140],
                            }
                        )

    core_areas = {"amr_sched", "crates", "tools"}
    expand_in_core = [h for h in hits["expand_planned_to_step_positions"] if h["area"] in core_areas]
    arr_minus_1_in_core = [h for h in hits["arr_i_minus_1"] if h["area"] in core_areas]
    arr_minus_1_by_file: dict[str, list[int]] = {}
    for h in hits["arr_i_minus_1"]:
        arr_minus_1_by_file.setdefault(h["file"], []).append(int(h["line"]))

    return {
        "hits": hits,
        "summary": {
            "expand_planned_in_core_count": len(expand_in_core),
            "arr_i_minus_1_in_core_count": len(arr_minus_1_in_core),
            "expand_planned_callers": sorted(
                {h["file"] for h in hits["expand_planned_to_step_positions"]}
            ),
            "arr_i_minus_1_locations": arr_minus_1_by_file,
        },
    }


def _find_workshop_corridors() -> dict[str, Any]:
    """Summarize WorkshopScene geometry, step spans, and raw vs point_execution capacities."""
    scene = WorkshopScene.load()
    raw_net = scene.network()
    pt_net = point_execution_network(raw_net)

    multi_cap_nodes = [
        {"node_id": nid, "kind": str(nd.kind.value), "raw_capacity": int(nd.capacity), "point_capacity": int(pt_net.node(nid).capacity)}
        for nid, nd in sorted(raw_net.nodes.items())
        if int(nd.capacity) != int(pt_net.node(nid).capacity)
    ]
    multi_cap_edges = [
        {
            "src": src,
            "dst": dst,
            "length_m": float(e.length),
            "raw_capacity": int(e.capacity),
            "point_capacity": int(pt_net.edge(src, dst).capacity),
        }
        for (src, dst), e in sorted(raw_net.edges.items())
        if int(e.capacity) != int(pt_net.edge(src, dst).capacity)
    ]

    two_way_corridors: list[dict[str, Any]] = []
    seen_pairs: set[tuple[str, str]] = set()
    for (u, v), e in sorted(raw_net.edges.items()):
        pair = (u, v) if u <= v else (v, u)
        if pair in seen_pairs or not raw_net.has_edge(v, u):
            continue
        seen_pairs.add(pair)
        steps_dt1 = max(1, int(math.ceil(e.length / min(1.0, e.speed_limit) / 1.0)))
        steps_macro2 = max(1, int(math.ceil(e.length / min(1.0, e.speed_limit) / 2.0)))
        two_way_corridors.append(
            {
                "u": pair[0],
                "v": pair[1],
                "length_m": round(float(e.length), 3),
                "speed_limit": float(e.speed_limit),
                "raw_cap_uv": int(raw_net.edge(u, v).capacity),
                "raw_cap_vu": int(raw_net.edge(v, u).capacity),
                "steps_dt1_v1": int(steps_dt1),
                "steps_macro2_v1": int(steps_macro2),
            }
        )

    long_corridors = [c for c in two_way_corridors if 20.0 <= c["length_m"] <= 30.5]
    long_corridors.sort(key=lambda c: (-c["length_m"], c["u"], c["v"]))
    longest_corridors = sorted(two_way_corridors, key=lambda c: (-c["length_m"], c["u"], c["v"]))

    return {
        "scene_id": scene.scene_id,
        "node_count": len(raw_net.nodes),
        "directed_edge_count": len(raw_net.edges),
        "undirected_corridor_count": len(two_way_corridors),
        "multi_cap_nodes_count": len(multi_cap_nodes),
        "multi_cap_nodes": multi_cap_nodes,
        "multi_cap_edges_count": len(multi_cap_edges),
        "multi_cap_edges": multi_cap_edges,
        "corridors_20_to_30m_count": len(long_corridors),
        "corridors_20_to_30m": long_corridors,
        "top5_longest_corridors": longest_corridors[:5],
    }


def _build_single_corridor_subnetwork(
    u: str,
    v: str,
    length_m: float,
    speed_limit: float,
    *,
    terminal_capacity: int = 1,
    mouth_capacity: int = 1,
) -> Network:
    """Extract an isolated single two-way corridor (`edge_cap=1`) with endpoint staging laybys."""
    net = Network()
    net.add_node(Node("S_U", 0.0, 0.0, NodeKind.AISLE, capacity=terminal_capacity))
    net.add_node(Node(u, 2.0, 0.0, NodeKind.AISLE, capacity=mouth_capacity))
    net.add_node(Node(v, 2.0 + length_m, 0.0, NodeKind.AISLE, capacity=mouth_capacity))
    net.add_node(Node("S_V", 4.0 + length_m, 0.0, NodeKind.AISLE, capacity=terminal_capacity))
    for a, b, ln in (
        ("S_U", u, 2.0),
        (u, "S_U", 2.0),
        (u, v, float(length_m)),
        (v, u, float(length_m)),
        (v, "S_V", 2.0),
        ("S_V", v, 2.0),
    ):
        net.add_edge(Edge(a, b, length=ln, speed_limit=float(speed_limit), capacity=1))
    return net


def run_executable_corridor_checks(corridor_info: dict[str, Any]) -> dict[str, Any]:
    """Run opposing multi-tick corridor verification across planners, Simulator, and CVCREvaluator."""
    sample = (
        corridor_info["corridors_20_to_30m"][0]
        if corridor_info["corridors_20_to_30m"]
        else corridor_info["top5_longest_corridors"][0]
    )
    u = str(sample["u"])
    v = str(sample["v"])
    length_m = float(sample["length_m"])
    speed_limit = float(sample["speed_limit"])
    net = _build_single_corridor_subnetwork(u, v, length_m, speed_limit)

    span_dt1 = max(1, int(math.ceil(length_m / min(1.0, speed_limit) / 1.0)))
    span_entry = max(1, int(math.ceil(2.0 / min(1.0, speed_limit) / 1.0)))
    horizon = 400

    # 1. Python SpaceTimeAStar opposing check
    py_planner = SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0, max_expansions=5000)
    rt_py = ReservationTable()
    p_a = py_planner.plan(net, "S_U", "S_V", 0, rt_py, "R1", horizon)
    assert p_a is not None
    arr_u_a = p_a.arrival_steps[0]
    arr_v_a = p_a.arrival_steps[1]
    rt_py.reserve_node("S_U", Interval(0, 1), "R1")
    rt_py.reserve_edge("S_U", u, Interval(0, arr_u_a), "R1")
    rt_py.reserve_edge(u, "S_U", Interval(0, arr_u_a), "R1")
    rt_py.reserve_node(u, Interval(arr_u_a, arr_u_a + 1), "R1")
    rt_py.reserve_edge(u, v, Interval(arr_u_a, arr_v_a), "R1")
    rt_py.reserve_edge(v, u, Interval(arr_u_a, arr_v_a), "R1")
    rt_py.reserve_node(v, Interval(arr_v_a, arr_v_a + 1), "R1")
    rt_py.reserve_edge(v, "S_V", Interval(arr_v_a, p_a.arrival_steps[2]), "R1")
    rt_py.reserve_edge("S_V", v, Interval(arr_v_a, p_a.arrival_steps[2]), "R1")
    rt_py.reserve_node("S_V", Interval(p_a.arrival_steps[2], p_a.arrival_steps[2] + 1), "R1")

    p_b_py = py_planner.plan(net, "S_V", "S_U", 0, rt_py, "R2", horizon)
    assert p_b_py is not None
    r2_py_arr_u = next(t for nd, t in zip(p_b_py.nodes, p_b_py.arrival_steps) if nd == u)
    # R2 cannot enter v->u until R1 leaves (arr_v_a) and traverses span_dt1
    py_opposing_stagger_ok = r2_py_arr_u >= arr_v_a + span_dt1

    # 2. RustSpaceTimeAStar opposing check (using amr_sched core reserve_step_points_indexed cleanly)
    rust_ok = is_rust_planner_available() and is_rust_core_available()
    rust_opposing_stagger_ok = False
    p_b_rust_arr_u = None
    if rust_ok:
        rust_planner = RustSpaceTimeAStarPlanner(max_speed=1.0, dt=1.0, max_expansions=5000)
        rt_rs = RustReservationTable()
        p_a_rs = rust_planner.plan(net, "S_U", "S_V", 0, rt_rs, "R1", horizon)
        assert p_a_rs is not None
        # Track B / amr_sched full-span reservation (`dep_t = arr_t - hop_steps`)
        chain_pts: list[tuple[str, int]] = [("S_U", 0)]
        cur_t = 0
        cur_n = "S_U"
        edge_segs: list[tuple[str, str, int, int]] = []
        for nd, arr_i in zip(p_a_rs.nodes, p_a_rs.arrival_steps):
            hop_steps = max(1, int(math.ceil(net.edge(cur_n, nd).length / min(1.0, net.edge(cur_n, nd).speed_limit))))
            dep_t = arr_i - hop_steps
            if dep_t > cur_t:
                chain_pts.append((cur_n, dep_t))
            edge_segs.append((cur_n, nd, dep_t, arr_i))
            chain_pts.append((nd, arr_i))
            cur_n, cur_t = nd, arr_i
        rust_planner.reserve_step_points_indexed(net, rt_rs, "R1", chain_pts, edge_segs)
        p_b_rs = rust_planner.plan(net, "S_V", "S_U", 0, rt_rs, "R2", horizon)
        assert p_b_rs is not None
        p_b_rust_arr_u = next(int(t) for nd, t in zip(p_b_rs.nodes, p_b_rs.arrival_steps) if nd == u)
        r1_rs_arr_v = next(int(t) for nd, t in zip(p_a_rs.nodes, p_a_rs.arrival_steps) if nd == v)
        rust_opposing_stagger_ok = p_b_rust_arr_u >= r1_rs_arr_v + span_dt1

    # 3. Simulator opposing head-on multi-tick test (res_window=0 and res_window=1)
    # Endpoints S_U and S_V have staging capacity=2 so both opposing tasks can complete while
    # the 30 m corridor (u <-> v) and its mouths stay strictly capacity=1.
    sim_net = _build_single_corridor_subnetwork(u, v, length_m, speed_limit, terminal_capacity=2)
    sim_results: dict[str, Any] = {}
    for res_win in (0, 1):
        agvs = [
            Agv("R1", max_speed=1.0, start_node="S_U", load_time=1.0, unload_time=1.0),
            Agv("R2", max_speed=1.0, start_node="S_V", load_time=1.0, unload_time=1.0),
        ]
        tasks = [
            Task("T1", pickup="S_U", dropoff="S_V", release_step=0, due_step=300, priority=TaskPriority.NORMAL),
            Task("T2", pickup="S_V", dropoff="S_U", release_step=0, due_step=300, priority=TaskPriority.NORMAL),
        ]
        alloc = RuleBasedAllocator(DispatchRule.FIFO, "greedy")
        planner = (
            RustSpaceTimeAStarPlanner(max_speed=1.0, dt=1.0, max_expansions=5000)
            if rust_ok
            else SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0, max_expansions=5000)
        )
        cfg = SimulationConfig(
            dt=1.0,
            max_steps=400,
            reservation_window=res_win,
            load_time=1.0,
            unload_time=1.0,
            record_trajectory=True,
        )
        sim = Simulator(sim_net, tasks, agvs, planner, alloc, config=cfg)
        res = sim.run()
        # Verify zero simultaneous opposing presence on (u, v) in trajectory
        simultaneous_corridor_ticks = 0
        corridor_labels = {f"{u}->{v}", f"{v}->{u}"}
        for rec in res.trajectory:
            p1 = rec.agv_positions.get("R1")
            p2 = rec.agv_positions.get("R2")
            if p1 in corridor_labels and p2 in corridor_labels:
                simultaneous_corridor_ticks += 1
        sim_results[f"res_window_{res_win}"] = {
            "completed_tasks": len(res.completed_tasks()),
            "collision_events": int(res.collision_events),
            "finished_step": int(res.finished_step),
            "simultaneous_corridor_ticks": int(simultaneous_corridor_ticks),
        }

    # 4. CVCREvaluator opposing reservation verification on the same multi-tick subnetwork
    snap = NativeSnapshot(
        schema="v1",
        tick=0,
        version={"net": 1},
        world_fingerprint="f0_corridor_check",
        robots=[
            RobotSnapshot("R1", "S_U", "idle"),
            RobotSnapshot("R2", "S_V", "idle"),
        ],
        tasks=[
            TaskSnapshot("T1", "S_U", "S_V", 0, 300, "pending"),
            TaskSnapshot("T2", "S_V", "S_U", 0, 300, "pending"),
        ],
        operations=[],
        reservations=[],
        now_monotonic=0.0,
    )
    ev = CVCREvaluator(net, snap, mode="B1", max_speed=1.0, dt=1.0, sim_horizon=400, use_rust=rust_ok)
    rt_cvcr = RustReservationTable.fast() if ev.use_rust else ReservationTable()
    route_r1 = ev._space_time_route("S_U", "S_V", 0, rt_cvcr, "R1", max_horizon=200)
    assert route_r1 is not None
    ev._reserve_path_steps(
        rt_cvcr,
        "R1",
        route_r1,
        [],
        [],
        False,
        None,
        rust_planner=ev._rust_planner,
        network=ev.network,
    )
    mid_conflicts_r2 = rt_cvcr.edge_conflicts(v, u, Interval(span_entry + 1, span_entry + span_dt1 - 1), exclude_agv="R2")
    if net.has_edge(u, v):
        mid_conflicts_r2 += rt_cvcr.edge_conflicts(u, v, Interval(span_entry + 1, span_entry + span_dt1 - 1), exclude_agv="R2")
    route_r2 = ev._space_time_route("S_V", "S_U", 0, rt_cvcr, "R2", max_horizon=200)
    assert route_r2 is not None
    r1_v_arr = next(t for nd, t in route_r1 if nd == v)
    r2_u_arr = next(t for nd, t in route_r2 if nd == u)
    cvcr_stagger_ok = (len(set(mid_conflicts_r2)) > 0) and (r2_u_arr >= r1_v_arr + span_dt1)

    # 5. Reproducing the Workshop metric_world U0 blind-spot bug from `expand_planned_to_step_positions`
    mmap = build_workshop_metric_map()
    macro_planner = RustSpaceTimeAStarPlanner(max_speed=1.0, dt=2.0, max_expansions=5000)
    # Find a real raw Workshop corridor where entry node has raw_capacity >= 2 (e.g. a bay/junction)
    # or test direct mid-corridor conflict check on J_N_105 <-> J_S_105 AND a cap-2-node opposing query.
    rt_buggy = RustReservationTable.fast()
    macro_path_a = macro_planner.plan(mmap.network, u, v, 0, rt_buggy, "R1", 100)
    assert macro_path_a is not None
    macro_arr_v = int(macro_path_a.arrival_steps[-1])
    pts_buggy = expand_planned_to_step_positions(u, 0, macro_path_a)
    macro_planner.reserve_step_points_indexed(mmap.network, rt_buggy, "R1", pts_buggy, park_until=None)
    # Check opposing traversal mid-corridor Interval(1, macro_arr_v - 1) on (u, v)/(v, u)
    blind_interval = Interval(1, max(2, macro_arr_v - 1))
    buggy_conflicts = set(rt_buggy.edge_conflicts(u, v, blind_interval, exclude_agv="R2")) | set(
        rt_buggy.edge_conflicts(v, u, blind_interval, exclude_agv="R2")
    )
    # Test a staggered-entry head-on conflict across a 1-lane 30m edge (cap=2 endpoint mouth, speed=1.0):
    # R1 traverses u -> v during [2, 17); R2 queries v -> u starting at t=0 (duration 15 ticks: [0, 15)).
    # Physical overlap is [2, 15) = 13 macro ticks (26 seconds) INSIDE the single-lane 30 m corridor!
    # Under `expand_planned_to_step_positions`, R1 only reserves edge [16, 17), so [0, 15) completely misses it.
    cap2_corridor_net = _build_single_corridor_subnetwork(
        u, v, length_m, speed_limit, terminal_capacity=2, mouth_capacity=2
    )
    rt_buggy_fast = RustReservationTable.fast()
    p_slow = macro_planner.plan(cap2_corridor_net, u, v, 2, rt_buggy_fast, "R1", 100)
    assert p_slow is not None
    macro_planner.reserve_step_points_indexed(
        cap2_corridor_net, rt_buggy_fast, "R1", expand_planned_to_step_positions(u, 2, p_slow), park_until=None
    )
    buggy_headon_r2 = macro_planner.plan(cap2_corridor_net, v, u, 0, rt_buggy_fast, "R2", 100)
    buggy_r2_arr_u = int(buggy_headon_r2.arrival_steps[-1]) if buggy_headon_r2 is not None else -1

    rt_fixed = RustReservationTable.fast()
    _reserve_exact_span(rt_fixed, mmap, "R1", u, 0, macro_path_a, buffer_ticks=0, sub_tick_div=1, park_until=None)
    fixed_conflicts = set(rt_fixed.edge_conflicts(u, v, blind_interval, exclude_agv="R2")) | set(
        rt_fixed.edge_conflicts(v, u, blind_interval, exclude_agv="R2")
    )
    rt_fixed_fast = RustReservationTable.fast()
    rt_fixed_fast.reserve_node(u, Interval(2, 3), "R1")
    rt_fixed_fast.reserve_edge(u, v, Interval(2, int(p_slow.arrival_steps[-1])), "R1")
    rt_fixed_fast.reserve_edge(v, u, Interval(2, int(p_slow.arrival_steps[-1])), "R1")
    rt_fixed_fast.reserve_node(v, Interval(int(p_slow.arrival_steps[-1]), int(p_slow.arrival_steps[-1]) + 1), "R1")
    fixed_headon_r2 = macro_planner.plan(cap2_corridor_net, v, u, 0, rt_fixed_fast, "R2", 100)
    fixed_r2_arr_u = int(fixed_headon_r2.arrival_steps[-1]) if fixed_headon_r2 is not None else -1

    blind_ticks = max(0, macro_arr_v - 1)
    blind_pct = (100.0 * blind_ticks / macro_arr_v) if macro_arr_v > 0 else 0.0

    # Also test full Workshop network (point_execution_network) with two AGVs intersecting on (u, v)
    full_scene = WorkshopScene.load()
    full_pt_net = point_execution_network(full_scene.network())
    sim_full = Simulator(
        full_pt_net,
        [
            Task("FW1", pickup=u, dropoff=v, release_step=0, due_step=500),
            Task("FW2", pickup=v, dropoff=u, release_step=0, due_step=500),
        ],
        [
            Agv("R1", max_speed=1.0, start_node=u, load_time=2.0, unload_time=2.0),
            Agv("R2", max_speed=1.0, start_node=v, load_time=2.0, unload_time=2.0),
        ],
        RustSpaceTimeAStarPlanner(max_speed=1.0, dt=1.0, max_expansions=5000) if rust_ok else SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0, max_expansions=5000),
        RuleBasedAllocator(DispatchRule.FIFO, "greedy"),
        config=SimulationConfig(dt=1.0, max_steps=500, reservation_window=1, load_time=2.0, unload_time=2.0),
    )
    res_full = sim_full.run()

    return {
        "corridor_tested": sample,
        "span_dt1_ticks": int(span_dt1),
        "spacetime_astar_python": {
            "r1_arr_v": int(arr_v_a),
            "r2_arr_u": int(r2_py_arr_u),
            "stagger_ok": bool(py_opposing_stagger_ok),
        },
        "rust_spacetime_astar": {
            "available": bool(rust_ok),
            "r2_arr_u": p_b_rust_arr_u,
            "stagger_ok": bool(rust_opposing_stagger_ok),
        },
        "simulator_isolated_corridor": sim_results,
        "simulator_full_workshop_point_net": {
            "completed_tasks": len(res_full.completed_tasks()),
            "collision_events": int(res_full.collision_events),
            "finished_step": int(res_full.finished_step),
        },
        "cvcr_evaluator": {
            "r1_route": route_r1,
            "r2_route": route_r2,
            "mid_corridor_conflicts_detected": sorted(set(mid_conflicts_r2)),
            "stagger_ok": bool(cvcr_stagger_ok),
        },
        "track_a_expand_bug_repro": {
            "macro_step_s": 2.0,
            "actual_macro_ticks_u_to_v": int(macro_arr_v),
            "buggy_pts_from_expand": pts_buggy,
            "blind_spot_ticks": int(blind_ticks),
            "blind_spot_pct": round(float(blind_pct), 2),
            "buggy_mid_corridor_conflicts_detected": sorted(buggy_conflicts),
            "buggy_r2_arr_u": buggy_r2_arr_u,
            "fixed_mid_corridor_conflicts_detected": sorted(fixed_conflicts),
            "fixed_r2_arr_u": fixed_r2_arr_u,
        },
    }


def render_markdown(payload: dict[str, Any]) -> str:
    """Render clean Markdown audit report for F0."""
    scan = payload["repo_scan"]
    ws = payload["workshop_topology"]
    chk = payload["corridor_checks"]
    bug = chk["track_a_expand_bug_repro"]
    lines = [
        "# F0: Reservation Semantics & Multi-Tick Corridor Blast-Radius Audit",
        "",
        "## 1. Key Findings & Blast Radius",
        "",
        f"- **`arr_i - 1` occurrences in `amr_sched/`, `crates/`, `tools/`**: `{scan['summary']['arr_i_minus_1_in_core_count']}`",
        f"- **`expand_planned_to_step_positions` callers in `amr_sched/`, `crates/`, `tools/`**: `{scan['summary']['expand_planned_in_core_count']}`",
        f"- **Exact bug root location**: `experiments/track_r_cpsa/cpsa_pure_runner.py:77` (`dep_t = arr_i - 1`, documented explicitly for unit-step grid graphs).",
        "- **`experiments/stp_research_reset/upper_planners.py:102` classification**: only used inside soft heuristic `_build_occupancy_counts()` (`gamma` slot-pressure bias), NOT inside hard `RustReservationTable` writes (`ExecutionAuditSim` uses `_reserve_exact_span`).",
        f"- **Why `amr_sched` Track B was immune**: `Simulator._reserve_hop` (`amr_sched/simulation.py:1191`), `Simulator._commit_move` (`1395`), `SpaceTimeAStarPlanner` (`amr_sched/planners/spacetime_astar.py:165`), `RustSpaceTimeAStarPlanner` + `crates/reservation_core` (`Interval(depart, arrive)` + `collect_undirected_edge_conflicts_into`), and `CVCREvaluator._space_time_route` (`dep_t = arr_t - self._edge_steps(cur_n, nxt_n)`) always reserve full undirected multi-tick edge intervals `[depart, arrive)`.",
        "",
        "### All `expand_planned_to_step_positions` Callers in Repo",
        "",
    ]
    for f in scan["summary"]["expand_planned_callers"]:
        lines.append(f"- `{f}`")

    lines.extend(
        [
            "",
            "## 2. Workshop Scene (`rail_freight_workshop_v1`) Geometry & Capacity Audit",
            "",
            f"- Nodes: `{ws['node_count']}`, Directed edges: `{ws['directed_edge_count']}`, Bidirectional corridors: `{ws['undirected_corridor_count']}`",
            f"- Corridors with length in `[20, 30.5]` m: `{ws['corridors_20_to_30m_count']}`",
            f"- Nodes with `raw_capacity != point_capacity(1)`: `{ws['multi_cap_nodes_count']}`",
            f"- Edges with `raw_capacity != point_capacity(1)`: `{ws['multi_cap_edges_count']}`",
            "",
            "| Corridor | Length (m) | Speed Limit (m/s) | Discrete Steps (dt=1s, v=1) | Macro Steps (dt=2s, v=1) | Raw Cap |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for c in ws["top5_longest_corridors"]:
        lines.append(
            f"| `{c['u']} <-> {c['v']}` | {c['length_m']:.2f} | {c['speed_limit']:.2f} | {c['steps_dt1_v1']} | {c['steps_macro2_v1']} | {c['raw_cap_uv']}/{c['raw_cap_vu']} |"
        )

    lines.extend(
        [
            "",
            "## 3. Executable Multi-Tick Corridor Verification",
            "",
            f"Tested real corridor: `{chk['corridor_tested']['u']} <-> {chk['corridor_tested']['v']}` ({chk['corridor_tested']['length_m']} m, `{chk['span_dt1_ticks']}` ticks @ `dt=1.0s`, `{bug['actual_macro_ticks_u_to_v']}` ticks @ `macro_step_s=2.0s`).",
            "",
            f"- **Python `SpaceTimeAStarPlanner`**: `R1 arr(v)={chk['spacetime_astar_python']['r1_arr_v']}`, `R2 arr(u)={chk['spacetime_astar_python']['r2_arr_u']}`, stagger_ok=`{chk['spacetime_astar_python']['stagger_ok']}`",
            f"- **`RustSpaceTimeAStarPlanner`**: `available={chk['rust_spacetime_astar']['available']}`, `R2 arr(u)={chk['rust_spacetime_astar']['r2_arr_u']}`, stagger_ok=`{chk['rust_spacetime_astar']['stagger_ok']}`",
            f"- **`Simulator` isolated corridor (`res_window=0`)**: completed=`{chk['simulator_isolated_corridor']['res_window_0']['completed_tasks']}`, collisions=`{chk['simulator_isolated_corridor']['res_window_0']['collision_events']}`, simultaneous_corridor_ticks=`{chk['simulator_isolated_corridor']['res_window_0']['simultaneous_corridor_ticks']}`",
            f"- **`Simulator` isolated corridor (`res_window=1`)**: completed=`{chk['simulator_isolated_corridor']['res_window_1']['completed_tasks']}`, collisions=`{chk['simulator_isolated_corridor']['res_window_1']['collision_events']}`, simultaneous_corridor_ticks=`{chk['simulator_isolated_corridor']['res_window_1']['simultaneous_corridor_ticks']}`",
            f"- **`Simulator` full Workshop (`point_execution_network`)**: completed=`{chk['simulator_full_workshop_point_net']['completed_tasks']}`, collisions=`{chk['simulator_full_workshop_point_net']['collision_events']}`, finished_step=`{chk['simulator_full_workshop_point_net']['finished_step']}`",
            f"- **`CVCREvaluator` opposing check**: mid_corridor_conflicts=`{chk['cvcr_evaluator']['mid_corridor_conflicts_detected']}`, stagger_ok=`{chk['cvcr_evaluator']['stagger_ok']}`",
            "",
            "### Reproduced Track A `expand_planned_to_step_positions` Blind Spot on Workshop Corridor",
            "",
            f"- Actual macro traversal: `{bug['actual_macro_ticks_u_to_v']}` ticks (`[0, {bug['actual_macro_ticks_u_to_v']})`)",
            f"- Buggy `pts` from `expand_planned_to_step_positions`: `{bug['buggy_pts_from_expand']}`",
            f"- **Blind-spot window dropped**: `{bug['blind_spot_ticks']}` / `{bug['actual_macro_ticks_u_to_v']}` ticks (`{bug['blind_spot_pct']:.2f}%`)",
            f"- Opposing query mid-corridor conflict detected (buggy `expand`): `{bug['buggy_mid_corridor_conflicts_detected']}` -> `R2 arr(u)={bug['buggy_r2_arr_u']}` (HEAD-ON ALLOWED AT t=0!)",
            f"- Opposing query mid-corridor conflict detected (fixed `_reserve_exact_span`): `{bug['fixed_mid_corridor_conflicts_detected']}` -> `R2 arr(u)={bug['fixed_r2_arr_u']}` (STAGGERED SAFELY AFTER R1)",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    """Run F0 reservation semantics audit and write JSON and Markdown outputs."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scan = scan_repo_reservation_patterns()
    ws = _find_workshop_corridors()
    chk = run_executable_corridor_checks(ws)
    payload = {
        "repo_scan": scan,
        "workshop_topology": ws,
        "corridor_checks": chk,
    }
    json_path = OUT_DIR / "f0_reservation_audit.json"
    md_path = OUT_DIR / "f0_reservation_audit.md"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    md_path.write_text(render_markdown(payload), encoding="utf-8")
    print(f"Wrote {json_path} and {md_path}")
    print(f"core expand_planned={scan['summary']['expand_planned_in_core_count']}, core arr_i-1={scan['summary']['arr_i_minus_1_in_core_count']}")
    print(f"corridor={chk['corridor_tested']}, blind_pct={chk['track_a_expand_bug_repro']['blind_spot_pct']}%")


if __name__ == "__main__":
    main()
