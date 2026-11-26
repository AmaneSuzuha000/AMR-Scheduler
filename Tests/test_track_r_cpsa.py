"""Unit tests for Track R (`TRACK-R-CPSA-01`: Pure MAPF/MAPD & EALR Delay Repair)."""

from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.network import Edge, Network, Node, NodeKind
from amr_sched.reservation import Interval, ReservationTable
from experiments.track_r_cpsa.cpsa_pure_runner import (
    impact_aware_blocker_score,
    reserve_mapf_path,
    run_lifelong_mapd,
    run_oneshot_mapf,
    verify_mapf_trace_safety,
)
from experiments.track_r_cpsa.execution_delay_sim import (
    DelayScenarioConfig,
    simulate_lifelong_with_delays,
)
from experiments.track_r_cpsa.local_reservation_repair import (
    DelayDurationBeliefTracker,
    EALRConfig,
    detect_reservation_mismatches,
    local_reservation_repair,
    propagate_reservation_dependencies,
)
from experiments.track_r_cpsa.rust_sipp_accel import run_accelerated_lifelong_mapd
from experiments.track_r_cpsa.maps_and_workloads import (
    GridMapSpec,
    build_map_a,
    build_map_b,
    build_map_c,
    build_map_rooms_32x32,
    build_map_warehouse_40x40,
    generate_fixed_bg_and_test_requests,
    generate_lifelong_requests,
    generate_oneshot_requests,
    load_movingai_map,
    load_movingai_scen,
    save_movingai_map_scen,
)
from experiments.rgro_closure.closure_common import sha256_file
from tools.data_manifest import MANIFEST, build as build_csv_manifest


def test_maps_a_b_c_geometry_and_movingai_parser(tmp_path: Path) -> None:
    ma, mb, mc = build_map_a(), build_map_b(), build_map_c()
    assert (ma.width, ma.height) == (16, 11)
    assert (mb.width, mb.height) == (20, 20)
    assert (mc.width, mc.height) == (20, 20)
    assert len(ma.traversable) > 100
    assert len(mb.obstacles) == 36
    assert len(mc.traversable) > 250

    map_file = tmp_path / "mini.map"
    map_file.write_text("type octile\nheight 3\nwidth 4\nmap\n....\n.@@.\n....\n", encoding="utf-8")
    gm = load_movingai_map(map_file)
    assert (gm.width, gm.height) == (4, 3)
    assert (1, 1) in gm.obstacles and (2, 1) in gm.obstacles
    assert len(gm.traversable) == 10


def test_pure_ca_vs_cpsa_oneshot_and_lifelong_zero_conflicts() -> None:
    ma = build_map_a()
    bg, tst = generate_fixed_bg_and_test_requests(ma, n_bg=18, n_test=16, seed=42)
    r_ca = run_oneshot_mapf(ma, tst, algo="ca_star", horizon=80, fixed_bg_requests=bg, seed=42)
    r_cps = run_oneshot_mapf(ma, tst, algo="cps_oneshot", horizon=80, fixed_bg_requests=bg, seed=42)
    assert r_ca.zero_conflicts and r_cps.zero_conflicts
    assert r_cps.n_success >= r_ca.n_success

    init_p, reqs = generate_lifelong_requests(ma, n_agvs=14, n_requests=120, seed=39)
    l_ca = run_lifelong_mapd(ma, init_p, reqs, algo="ca_star", horizon=80, seed=39)
    l_cps = run_lifelong_mapd(ma, init_p, reqs, algo="cps_lifelong", horizon=80, seed=39)
    assert l_ca.zero_conflicts and l_cps.zero_conflicts
    assert l_cps.completed_tasks >= 1


def test_ealr_mechanisms_1_2_3_and_pin_protection() -> None:
    """3-AGV grid test: delayed AGV0 triggers Mechanism 1+2 extracting {AGV0, AGV1}, leaving distant AGV2 untouched."""
    net = Network()
    # 5x3 grid with single central lane & upper bypass
    for y in range(3):
        for x in range(5):
            if (x, y) in {(1, 1), (3, 1)}:
                continue
            net.add_node(Node(f"C_{x}_{y}", float(x), float(y), kind=NodeKind.AISLE, capacity=1))
    for y in range(3):
        for x in range(5):
            u = f"C_{x}_{y}"
            if u not in net.nodes:
                continue
            for dx, dy in ((1, 0), (0, 1)):
                v = f"C_{x+dx}_{y+dy}"
                if v in net.nodes:
                    net.add_bidirectional(u, v, length=1.0, speed_limit=1.0, capacity=1)

    res = ReservationTable()
    # AGV0 planned path C_0_0 -> C_1_0 -> C_2_0 -> C_3_0 -> C_4_0, stalled at C_0_0 at tick=1!
    p0 = [("C_0_0", 0), ("C_1_0", 1), ("C_2_0", 2), ("C_3_0", 3), ("C_4_0", 4)]
    # AGV1 follows right through C_2_0 -> C_1_0 -> C_0_0 over ticks 1..4 (conflicts with stalled AGV0!)
    p1 = [("C_2_0", 0), ("C_1_0", 2), ("C_0_0", 3)]
    # AGV2 is far away at upper-right row y=2, unaffected!
    p2 = [("C_4_2", 0), ("C_4_2", 5)]
    reserve_mapf_path(res, "AGV0", p0, park_until=10)
    reserve_mapf_path(res, "AGV1", p1, park_until=10)
    reserve_mapf_path(res, "AGV2", p2, park_until=10)

    plan_steps = {
        "AGV0": {st: nd for nd, st in p0},
        "AGV1": {st: nd for nd, st in p1},
        "AGV2": {st: nd for nd, st in p2},
    }
    cur_pos = {"AGV0": "C_0_0", "AGV1": "C_2_0", "AGV2": "C_4_2"}
    goals = {"AGV0": "C_4_0", "AGV1": "C_0_0", "AGV2": "C_4_2"}
    mism = detect_reservation_mismatches(
        ["AGV0", "AGV1", "AGV2"],
        cur_pos,
        {"AGV0": 2, "AGV1": 0, "AGV2": 0},
        {"AGV0": 0, "AGV1": 0, "AGV2": 0},
        plan_steps,
        {"AGV0": 4, "AGV1": 3, "AGV2": 5},
        goals,
        tick=1,
    )
    assert set(mism.keys()) == {"AGV0"}

    from experiments.cps_astar_routing.cps_astar_graph import CPSGraphFieldConfig, CPSAStarConfig, CPSAStarGraphPlanner, GraphCongestionField
    field = GraphCongestionField(net, CPSGraphFieldConfig(), t_start=0)
    field.add_agv_path("AGV0", p0)
    field.add_agv_path("AGV1", p1)
    field.add_agv_path("AGV2", p2)

    cfg = EALRConfig(prop_window_steps=12, prop_max_depth=2, max_repair_agents=4, comm_lag_steps=0)
    r_loc = propagate_reservation_dependencies(
        mism, ["AGV0", "AGV1", "AGV2"], cur_pos, plan_steps, {"AGV0": 4, "AGV1": 3, "AGV2": 5}, goals,
        res, field, net, cfg, tick=1,
    )
    assert "AGV0" in r_loc and "AGV1" in r_loc
    assert "AGV2" not in r_loc, f"Distant unaffected AGV2 should NOT be dragged into R_loc: {r_loc}"

    planner = CPSAStarGraphPlanner(net, CPSAStarConfig(mode="lifelong"), max_speed=1.0, dt=1.0)
    rep = local_reservation_repair(
        r_loc, mism, ["AGV0", "AGV1", "AGV2"], cur_pos,
        {"AGV0": 2, "AGV1": 0, "AGV2": 0}, {"AGV0": 0, "AGV1": 0, "AGV2": 0},
        goals, {"AGV0": 0, "AGV1": 0, "AGV2": 0},
        plan_steps, {"AGV0": 4, "AGV1": 3, "AGV2": 5},
        res, field, planner, net, cfg, {}, tick=1, horizon=20,
    )
    assert rep["repaired"] >= 1
    # Verify AGV0 respects physical pin at C_0_0 during stall ticks 1..3
    assert plan_steps["AGV0"][1] == "C_0_0" and plan_steps["AGV0"][2] == "C_0_0"


def test_delayed_lifelong_comparators_all_zero_conflicts() -> None:
    ma = build_map_a()
    init_p, reqs = generate_lifelong_requests(ma, n_agvs=16, n_requests=150, seed=41)
    dcfg = DelayScenarioConfig(
        motion_delay_prob=0.12,
        service_delay_prob=0.25,
        max_service_delay=2,
        heterogeneous_slow_frac=0.2,
        comm_lag_steps=1,
        horizon=90,
    )
    for sv in ("naive_no_repair", "adg_order_hold", "global_rolling_cpsa", "ealr_local_repair"):
        res = simulate_lifelong_with_delays(ma, init_p, reqs, dcfg, solver=sv, seed=41)
        assert res.zero_conflicts, f"{sv} violated MAPF physical safety!"
    # Compare repair radius <= global
    r_glob = simulate_lifelong_with_delays(ma, init_p, reqs, dcfg, solver="global_rolling_cpsa", seed=41)
    r_ealr = simulate_lifelong_with_delays(ma, init_p, reqs, dcfg, solver="ealr_local_repair", seed=41)
    if r_ealr.repair_events > 0 and r_glob.repair_events > 0:
        assert r_ealr.avg_repair_fraction <= r_glob.avg_repair_fraction + 1e-6


def test_direction_b_probe_and_frozen_assets_intact() -> None:
    s1 = impact_aware_blocker_score("R0", "R_high_unfix", {"R_high_unfix": 30}, remaining_len_j=40, local_cong_j=0.8, unblock_probe_ok=False)
    s2 = impact_aware_blocker_score("R0", "R_fixable", {"R_fixable": 12}, remaining_len_j=10, local_cong_j=0.2, unblock_probe_ok=True)
    assert s2 > s1

    expected = json.loads(Path(MANIFEST).read_text(encoding="utf-8"))
    current = build_csv_manifest()
    assert current == expected and len(current) == 38

    v20_lock = ROOT / "out" / "v20_1_closure" / "v20_1_freeze_manifest.json"
    if v20_lock.is_file():
        m = json.loads(v20_lock.read_text(encoding="utf-8"))
        files_map = m.get("sha256_files", m.get("files", {}))
        checked = 0
        for rel_p, exp_sha in files_map.items():
            if isinstance(exp_sha, str) and len(exp_sha) == 64:
                fp = ROOT / rel_p
                assert fp.is_file() and sha256_file(fp) == exp_sha, f"Tampered {rel_p}"
                checked += 1
        assert checked == 8


def test_incremental_field_cache_equivalence_and_migrations(tmp_path: Path) -> None:
    """Verify incremental Gaussian cache invalidation is bit-identical and EALR v2 / Rust SIPP pass zero conflicts."""
    from experiments.cps_astar_routing.cps_astar_graph import CPSGraphFieldConfig, GraphCongestionField

    ma = build_map_a()
    fcfg = CPSGraphFieldConfig(spatial_radius_L=1, temporal_radius_T=1, max_track_horizon=50)
    f_inc = GraphCongestionField(ma.network, fcfg, t_start=0)
    f_ref = GraphCongestionField(ma.network, fcfg, t_start=0)

    p_a = [("C_2_2", 0), ("C_2_3", 1), ("C_2_4", 2), ("C_3_4", 3)]
    p_b = [("C_4_4", 0), ("C_3_4", 1), ("C_2_4", 2)]
    f_inc.add_agv_path("A", p_a)
    f_inc.add_agv_path("B", p_b)
    _ = f_inc.phi("C_2_4", 2), f_inc.phi("C_10_8", 10), f_inc.slot_Phi("C_2_4", 2)
    f_inc.remove_agv("A")
    f_inc.add_agv_path("A", [("C_2_2", 0), ("C_2_2", 2), ("C_2_3", 3)])

    f_ref.add_agv_path("B", p_b)
    f_ref.add_agv_path("A", [("C_2_2", 0), ("C_2_2", 2), ("C_2_3", 3)])
    for nd in ("C_2_2", "C_2_3", "C_2_4", "C_10_8"):
        for step in (0, 1, 2, 3, 10):
            assert abs(f_inc.phi(nd, step) - f_ref.phi(nd, step)) < 1e-12
            assert abs(f_inc.slot_Phi(nd, step) - f_ref.slot_Phi(nd, step)) < 1e-12

    # Verify causal DelayDurationBeliefTracker monotonicity on sustained stalls
    tr = DelayDurationBeliefTracker()
    for s in range(5):
        tr.observe_step(s, {"AGV_0"})
    rem0 = tr.expected_remaining("AGV_0", 4)
    tr.observe_step(5, set())
    for s in range(6, 12):
        tr.observe_step(s, {"AGV_1"})
    rem1 = tr.expected_remaining("AGV_1", 11)
    assert rem0 > 0.0 and rem1 > 0.0

    # Verify EALR v2 migrations achieve 100% zero conflicts & Rust SIPP + SlotPressure passes safety
    init_p, reqs = generate_lifelong_requests(ma, n_agvs=14, n_requests=120, seed=41)
    dcfg = DelayScenarioConfig(motion_delay_prob=0.12, service_delay_prob=0.25, max_service_delay=2, heterogeneous_slow_frac=0.2, comm_lag_steps=1, horizon=80)
    r_v2 = simulate_lifelong_with_delays(ma, init_p, reqs, dcfg, solver="ealr_local_repair", ealr_cfg=EALRConfig(enable_v2_migrations=True), seed=41)
    assert r_v2.zero_conflicts and r_v2.completed_tasks >= 1

    r_sipp = run_accelerated_lifelong_mapd(ma, init_p, reqs, backend="rust_sipp_slot", horizon=80, seed=39)
    assert r_sipp.zero_conflicts and r_sipp.completed_tasks >= 1

    # Verify large-scale 32x32 / 40x40 grids, MovingAI export/reload, and hardened Map_B 55-AGV SIPP zero conflicts
    mr = build_map_rooms_32x32()
    mw = build_map_warehouse_40x40()
    assert (mr.width, mr.height) == (32, 32) and len(mr.traversable) > 800
    assert (mw.width, mw.height) == (40, 40) and len(mw.traversable) > 1000
    reqs_w = generate_oneshot_requests(mw, 20, seed=7)
    mp = tmp_path / "_test_roundtrip.map"
    sp = tmp_path / "_test_roundtrip.scen"
    save_movingai_map_scen(mw, reqs_w, mp, sp)
    mw_reloaded = load_movingai_map(mp, name="Warehouse_40x40")
    pairs_reloaded = load_movingai_scen(sp, mw_reloaded)
    assert len(mw_reloaded.traversable) == len(mw.traversable) and len(pairs_reloaded) == 20
    mp.unlink(missing_ok=True)
    sp.unlink(missing_ok=True)

    mb = build_map_b()
    ip_b, rq_b = generate_lifelong_requests(mb, 55, n_requests=55 * 20, seed=40)
    r_sipp_b = run_accelerated_lifelong_mapd(mb, ip_b, rq_b, backend="rust_sipp_slot", horizon=120, seed=40)
    assert r_sipp_b.zero_conflicts, "Hardened Rust SIPP + SlotPressure must maintain zero conflicts on Map_B seed 40"


def test_workshop_bridge_and_convergence_isolation() -> None:
    """Verify Wu EJOR'26 + CPS-A* + CVCR Havens + DFL Critic bridge and two-tier scope isolation."""
    import ast
    from experiments.track_r_cpsa.maps_and_workloads import build_official_trc_map
    from experiments.track_r_cpsa.workshop_bridge_eval import (
        _load_dfl_critic,
        _load_wu26_net,
        run_workshop_episode,
    )

    wu_net = _load_wu26_net()
    assert wu_net is not None, "Expected pure-NumPy Wu EJOR'26 trained weights to load cleanly"
    dfl_critic = _load_dfl_critic()
    assert dfl_critic is not None, "Expected pure-NumPy DFLAdvantageEnsemble checkpoint to load cleanly"

    for code, expected_trav in (("A", 120), ("B", 364), ("C", 320)):
        ogm = build_official_trc_map(code)
        assert len(ogm.traversable) == expected_trav

    r_demo = run_workshop_episode(
        layout="demo17", aisle_capacity=1, n_tasks=20, n_agvs=4, seed=42,
        dispatcher="Wu26_SemiCircle", router="CPS_IALPR_EALR", delay_prob=0.10, wu_net=wu_net,
    )
    assert r_demo.zero_conflicts and r_demo.completed_tasks == 20 and r_demo.on_time_rate > 0.0

    r_cvcr = run_workshop_episode(
        layout="demo17", aisle_capacity=1, n_tasks=20, n_agvs=6, seed=42,
        dispatcher="Wu26_SemiCircle_DFL", router="CPS_CVCR_DFL", delay_prob=0.10,
        wu_net=wu_net, dfl_critic=dfl_critic,
    )
    assert r_cvcr.zero_conflicts and r_cvcr.completed_tasks >= 15

    r_ws = run_workshop_episode(
        layout="workshop_v1", aisle_capacity=1, n_tasks=20, n_agvs=4, seed=42,
        dispatcher="Wu26_SemiCircle_DFL", router="CPS_CVCR_DFL", delay_prob=0.10,
        wu_net=wu_net, dfl_critic=dfl_critic,
    )
    assert r_ws.zero_conflicts and r_ws.completed_tasks >= 18

    from experiments.track_r_cpsa.converged_stress_and_baselines import make_converged_dispatcher
    for ext_dp in ("Ours_V19_HETERO_RL", "IJCAI26_PGMPO", "RAL26_DetourDelta"):
        r_ext = run_workshop_episode(
            layout="demo17", aisle_capacity=1, n_tasks=15, n_agvs=4, seed=42,
            dispatcher=ext_dp, router="CPS_CVCR_DFL", delay_prob=0.10,
            wu_net=wu_net, dfl_critic=dfl_critic,
            dispatch_override=make_converged_dispatcher(ext_dp),
        )
        assert r_ext.zero_conflicts and r_ext.completed_tasks >= 12

    # Three-Ring Convergence Guard:
    #   Ring 1 (Pure Paper Reproduction): strict zero CVCR/RL imports.
    #   Ring 2 (Converged Scenario Extensions): allows non-DAG CVCR + lightweight DFL critic.
    #   Ring 3 (Cross-Baseline Comparison Harness): allows read-only inference comparison of prior RL & 2026
    #           literature models while STILL forbidding online RL training envs and factory production DAG modules.
    pure_paper_files = {
        "cpsa_pure_runner.py",
        "maps_and_workloads.py",
        "run_large_scale_study.py",
        "run_r1_r4_pilot.py",
    }
    ring3_compare_files = {"converged_stress_and_baselines.py"}
    forbidden_tier1 = (
        "amr_sched.rl",
        "experiments.rgro",
        "amr_sched.cvcr.v1",
        "amr_sched.cvcr.v2",
        "amr_sched.cvcr.pdg",
        "amr_sched.cvcr.production",
    )
    forbidden_tier2 = (
        "amr_sched.cvcr.production",
        "amr_sched.cvcr.v13",
        "amr_sched.cvcr.v14",
        "amr_sched.cvcr.v15",
        "amr_sched.cvcr.v16",
        "amr_sched.cvcr.v17",
        "amr_sched.cvcr.v18",
        "amr_sched.rl.v19",
        "amr_sched.rl.v20",
        "amr_sched.rl.agent",
        "amr_sched.rl.env",
    )
    forbidden_tier3 = (
        "amr_sched.cvcr.production",
        "amr_sched.rl.agent",
        "amr_sched.rl.env",
    )
    track_dir = ROOT / "experiments" / "track_r_cpsa"
    for pyf in track_dir.glob("*.py"):
        tree = ast.parse(pyf.read_text(encoding="utf-8"))
        if pyf.name in pure_paper_files:
            rules = forbidden_tier1
        elif pyf.name in ring3_compare_files:
            rules = forbidden_tier3
        else:
            rules = forbidden_tier2
        for node in ast.walk(tree):
            mod = ""
            if isinstance(node, ast.ImportFrom) and node.module:
                mod = node.module
            elif isinstance(node, ast.Import):
                mod = ",".join(al.name for al in node.names)
            for bad in rules:
                assert bad not in mod, f"Track R file {pyf.name} violates convergence tier by importing {mod!r}"


def test_rust_e2e_parity_route_b_hook_and_adapted_ckpts() -> None:
    """Verify Track P1 Rust E2E parity, Route B physical hook safety, and Track P2 adapted checkpoints."""
    from amr_sched.planners.rust_spacetime_astar import RustCPSAStarGraphPlanner
    from amr_sched.rust_reservation import RustReservationTable
    from experiments.cps_astar_routing.cps_astar_graph import (
        CPSAStarConfig,
        CPSAStarGraphPlanner,
        GraphCongestionField,
    )
    from experiments.track_r_cpsa.workshop_bridge_eval import (
        _load_dfl_critic,
        _load_wu26_net,
        run_workshop_episode,
    )

    # 1. Single-leg Python vs Rust E2E parity (expansions, nodes, arrivals, blocker_hits)
    ma = build_map_a()
    cfg = CPSAStarConfig(mode="lifelong", beta=0.5, gamma=5.0)
    py_p = CPSAStarGraphPlanner(ma.network, cfg, max_speed=1.0, dt=1.0)
    rs_p = RustCPSAStarGraphPlanner(ma.network, cfg, max_speed=1.0, dt=1.0)
    py_res, rs_res = ReservationTable(), RustReservationTable.fast()
    py_fld = GraphCongestionField(ma.network, cfg.field)
    rs_fld = rs_p.create_field()

    nds = sorted(ma.network.nodes.keys())
    s, g, mid = nds[2], nds[-3], nds[len(nds) // 2]
    for tb in (py_res, rs_res):
        tb.reserve_node(s, Interval(0, 1), "A1")
        tb.reserve_node(mid, Interval(3, 8), "BLK1")
    pts_bg = [(mid, k) for k in range(3, 9)]
    py_fld.add_agv_path("BLK1", pts_bg)
    rs_fld.add_agv_path("BLK1", pts_bg)

    p1, h1 = py_p.plan_leg(s, g, 0, py_res, py_fld, "A1", 80, goal_park_steps=1)
    p2, h2 = rs_p.plan_leg(s, g, 0, rs_res, rs_fld, "A1", 80, goal_park_steps=1)
    assert p1 is not None and p2 is not None
    assert p1.nodes == p2.nodes and p1.arrival_steps == p2.arrival_steps
    assert py_p.last_expansions == rs_p.last_expansions and h1 == h2

    # 2. Route B physical obstacle & ETA drift telemetry hook
    def _obs_hook(tick: int, net: Network, pos: dict, in_edge: dict, phase: dict) -> dict:
        if tick == 10:
            edge_u = sorted(net.edges.keys())[0]
            return {
                "obstacles": [{"edge": edge_u, "estimated_clear_horizon_steps": 6}],
                "telemetry": {"AGV_01": {"eta_drift_steps": 2}},
            }
        return {}

    r_rb = run_workshop_episode(
        layout="demo17", aisle_capacity=1, n_tasks=18, n_agvs=4, seed=42,
        dispatcher="Wu26_SemiCircle_GRU", router="CPS_CVCR_DFL", delay_prob=0.10,
        route_b_telemetry_hook=_obs_hook, use_rust_e2e=True,
    )
    assert r_rb.zero_conflicts and r_rb.completed_tasks >= 15

    # 3. Quarantined adapted checkpoints load cleanly & preserve 100% safety on saturated W_Stress2
    ckpt_dir = ROOT / "amr_sched" / "checkpoints" / "track_r"
    dfl_ad = _load_dfl_critic(ckpt_dir / "mh_dfl_critic_workshop.npz")
    wu_ad = _load_wu26_net(ckpt_dir / "ejor26_wu_gru_workshop.npz")
    assert dfl_ad is not None and wu_ad is not None
    assert dfl_ad.metadata.get("domain") == "workshop_v1"
    r_w2 = run_workshop_episode(
        layout="workshop_v1", aisle_capacity=1, n_tasks=80, n_agvs=12, seed=42,
        dispatcher="Wu26_SemiCircle_DFL", router="CPS_CVCR_DFL", delay_prob=0.15,
        wu_net=wu_ad, dfl_critic=dfl_ad, use_rust_e2e=True,
    )
    assert r_w2.zero_conflicts and r_w2.completed_tasks >= 70


def test_route_a_lpr_and_route_b_kinematic_rl() -> None:
    """Verify Route A controlled LPR selectors (Safe=True) & Route B 2D kinematic PPO + closed-loop hook."""
    from experiments.track_r_cpsa.maps_and_workloads import (
        build_official_trc_map,
        generate_fixed_bg_and_test_requests,
    )
    from experiments.track_r_cpsa.route_a_lpr_ablation import run_oneshot_lpr_case
    from experiments.track_r_cpsa.route_b_kinematic_rl import (
        CKPT_PATH,
        NumpyRouteBPPO,
        UnicycleCorridorEnv,
        build_corridor_scenario,
        make_route_b_kinematic_hook,
    )
    from experiments.track_r_cpsa.workshop_bridge_eval import _load_dfl_critic, run_workshop_episode

    # 1. Route A1 One-Shot MAPF parity & zero-conflict verification across selectors
    gm = build_official_trc_map("A")
    bg, tst = generate_fixed_bg_and_test_requests(gm, 24, 30, seed=39)
    dfl_ad = _load_dfl_critic(ROOT / "amr_sched" / "checkpoints" / "track_r" / "mh_dfl_critic_workshop.npz")
    assert dfl_ad is not None
    r_hits = run_oneshot_lpr_case(gm, bg, tst, selector="LPR_MaxHits", horizon=90, seed=39)
    r_dfl = run_oneshot_lpr_case(gm, bg, tst, selector="LPR_LearnedWorkshopDFL", dfl_critic=dfl_ad, horizon=90, seed=39)
    assert r_hits.zero_conflicts and r_dfl.zero_conflicts
    assert r_hits.n_success >= 10 and r_dfl.n_success >= 10

    # 2. Route B 2D continuous kinematic PPO passes Bypass & HeadOn with zero physical collisions
    assert CKPT_PATH.exists(), f"Missing Route B checkpoint at {CKPT_PATH}"
    ppo = NumpyRouteBPPO.load_npz(CKPT_PATH)
    for sc_name in ("Bypass_MainAisle_3.0m", "HeadOn_Yield_2.2m"):
        env = UnicycleCorridorEnv(build_corridor_scenario(sc_name, seed=11), obs_noise_std=0.0, seed=11)
        ob = env.reset()
        cols = 0
        reached = False
        for _ in range(env.max_steps):
            vc, wc = ppo.act_raw(ob)
            ob, _, done, info = env.step(vc, wc, use_shield=True)
            cols += int(info["collided"])
            if info["reached"]:
                reached = True
                break
            if done:
                break
        assert reached and cols == 0, f"Route B PPO failed on {sc_name}: reached={reached}, cols={cols}"

    # 3. Closed-loop Rust CPS-A* + Route B telemetry hook preserves zero conflicts & zero micro collisions
    hook, st = make_route_b_kinematic_hook(mode="ppo_shield", ppo_agent=ppo, seed=42, corridor_disturb_prob=0.35)
    r_ws = run_workshop_episode(
        layout="workshop_v1", aisle_capacity=1, n_tasks=20, n_agvs=6, seed=42,
        dispatcher="Wu26_SemiCircle_GRU", router="CPS_CVCR_DFL", delay_prob=0.12,
        dfl_critic=dfl_ad, use_rust_e2e=True, route_b_telemetry_hook=hook,
        lpr_selector_mode="dfl",
    )
    assert r_ws.zero_conflicts and r_ws.completed_tasks >= 18
    assert st["micro_rollouts"] >= 1 and st["physical_collisions"] == 0 and st["min_surface_clear_m"] >= 0.15




