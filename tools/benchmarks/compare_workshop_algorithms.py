"""Decoupled multi-algorithm benchmark across real Workshop Track 1 & Track 2 topologies.

Evaluates V18_EA_JOINT, V17_EA_JOINT, V16_EA_JOINT, rule_v10, and literature SOTA
baselines on identical, SHA-256 paired, IndependentAuditor-verified discrete
scenarios built directly on:
  - Track 1 (`rail_freight_workshop_v1`, 62 nodes, 6 AGVs)
  - Track 2 (`wheelset_workshop_v2`, 50 nodes, 3 AGVs)

Using `amr_sched.baselines.policy_registry`, every policy tested here plugs
identically into the physical ROS 2 / Gazebo `PhysicalCVCRCoordinator`.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import platform
import random
import statistics
import sys
import time
from typing import Any, Optional

if sys.platform == "win32" and getattr(platform, "_wmi", None) is not None:
    platform._wmi = None  # type: ignore[attr-defined]

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import (
    DEFAULT_12_POLICIES,
    SUPPORTED_POLICIES,
    resolve_policy_name,
    run_discrete_episode,
)
from amr_sched.cvcr.p2_perturbation import P2PerturbationConfig
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.entities import Agv, MaterialType, Task, TaskPriority
from amr_sched.network import Network, NodeKind
from amr_sched.planners.rust_spacetime_astar import is_rust_planner_available
from amr_sched.workshop import WorkshopScene, legal_task_pairs
from tools.live_bridge.fleet_config import point_execution_network
from tools.wheelset_scene.workshop_v2_layout import build_improved_scene, build_v2_nav_graph

REGIMES: tuple[str, ...] = (
    "nominal",
    "kitting_windows",
    "execution_blockage",
    "multi_wave_surge",
)


def _regime_rng(seed: int, mult: int, regime: str) -> random.Random:
    # Avoid Python's randomized per-process hash() so worker pools produce bit-identical paired scenarios.
    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(str(regime)))
    return random.Random(int(seed) * mult + offset)


def _travel_estimate(net: Network, src: str, dst: str, speed: float) -> float:
    if src == dst:
        return 0.0
    tt = net.shortest_travel_times(src, max(0.2, float(speed))) if src in net.nodes else {}
    return float(tt.get(dst, 120.0))


def _load_track1_network() -> tuple[WorkshopScene, Network]:
    scene = WorkshopScene.load(ROOT / "scenes" / "rail_freight_workshop_v1.json")
    return scene, point_execution_network(scene.network())


def _load_track2_network() -> tuple[WorkshopScene, Network]:
    scene = WorkshopScene(build_v2_nav_graph(build_improved_scene()))
    return scene, point_execution_network(scene.network())


def build_track1_scenario(
    seed: int,
    regime: str,
) -> tuple[tuple[Any, ...], Optional[P2PerturbationConfig], int]:
    """Construct paired scenario on Track 1 (`rail_freight_workshop_v1`, 62 nodes, 6 AGVs)."""
    rng = _regime_rng(seed, 1009, regime)
    scene, net = _load_track1_network()
    starts = ["J_S_65", "J_N_45", "J_N_85", "W_DOOR", "W_COUPLER", "W_PARK"]
    if rng.random() < 0.5:
        starts[0], starts[2] = starts[2], starts[0]
    agvs = [
        Agv(f"R{i + 1}", max_speed=1.0, start_node=starts[i], load_time=2.0, unload_time=2.0)
        for i in range(6)
    ]

    all_pairs = list(legal_task_pairs(scene))
    rng.shuffle(all_pairs)
    core_pairs = [
        ("Q_BOGIE", "D_BOGIE", MaterialType.BOGIE, ("D_BOGIE", "D_WHEELSET")),
        ("J_S_65", "D_WHEELSET", MaterialType.WHEELSET, ("D_WHEELSET", "D_BOGIE")),
        ("J_N_85", "D_BRAKE", MaterialType.BRAKE_VALVE, ("D_BRAKE", "D_STORE")),
        ("R_COUPLER", "D_COUPLER", MaterialType.BRAKE_VALVE, ("D_COUPLER", "D_DOOR")),
        ("R_DOOR", "D_DOOR", MaterialType.BRAKE_VALVE, ("D_DOOR", "D_COUPLER")),
        ("J_S_125", "D_ASSEMBLY", MaterialType.WHEELSET, ("D_ASSEMBLY",)),
    ]
    n_extra = 4 if regime == "multi_wave_surge" else (2 if regime in ("kitting_windows", "execution_blockage") else 1)
    for idx in range(n_extra):
        pu, do = all_pairs[idx % len(all_pairs)]
        core_pairs.append((pu, do, MaterialType.BRAKE_VALVE, (do,)))

    tasks: list[Task] = []
    ops: list[OperationSnapshot] = []
    cum_op_est = 0.0
    slack_mult = 1.6 if regime in ("kitting_windows", "multi_wave_surge") else 2.1

    for idx, (pu, do, mat, alt_bays) in enumerate(core_pairs):
        op_id = f"OP_T1_{idx + 1}"
        tid_a = f"K_T1_{idx + 1}"
        est_t = _travel_estimate(net, pu, do, 0.8)
        rel = (idx // 2) * (12 if regime == "multi_wave_surge" else 18) + rng.randint(0, 8)
        if idx in (1, 5):
            cum_op_est = max(cum_op_est + 25.0, rel + est_t * slack_mult)
        t_due = int(round(rel + max(45.0, est_t * slack_mult)))
        is_urg = idx in (5, len(core_pairs) - 1)
        t_obj = Task(
            tid_a,
            pu,
            do,
            release_step=rel,
            due_step=t_due,
            priority=TaskPriority.URGENT if is_urg else TaskPriority.NORMAL,
            material_type=mat,
        )
        t_obj.kit_op_id = op_id
        tasks.append(t_obj)
        kit_ids = [tid_a]

        if regime == "kitting_windows" and idx in (1, 2, 3):
            tid_b = f"K_T1_{idx + 1}_B"
            src_b = rng.choice([n.node_id for n in net.nodes_of_kind(NodeKind.BUFFER, NodeKind.DISASSEMBLY) if n.node_id != do] or ["J_S_45"])
            est_b = _travel_estimate(net, src_b, do, 0.8)
            t_obj_b = Task(
                tid_b,
                src_b,
                do,
                release_step=rel + 6,
                due_step=int(round(rel + max(50.0, est_b * slack_mult))),
                material_type=mat,
            )
            t_obj_b.kit_op_id = op_id
            tasks.append(t_obj_b)
            kit_ids.append(tid_b)

        pred = "OP_T1_1" if idx == 1 else ("OP_T1_2" if idx == 5 else (f"OP_T1_{idx}" if idx >= 6 and idx % 2 == 0 else None))
        r_wins: tuple[tuple[int, int], ...] = ()
        if regime == "kitting_windows" and idx in (2, 3, 4):
            w_open = int(rel + est_t * 0.9)
            r_wins = ((0, w_open + 38), (w_open + 88, 1200))

        op_due = int(round(max(t_due + 25, (cum_op_est if pred else rel + est_t * slack_mult) + 35)))
        ops.append(
            OperationSnapshot(
                op_id=op_id,
                bay_id=do,
                crew_id=(idx % 3) + 1,
                tool_id=(idx % 3) + 1,
                proc_steps=10 + (idx % 3) * 3,
                due_step=op_due,
                predecessor_op_id=pred,
                predecessor_op_ids=(pred,) if pred else (),
                is_urgent=is_urg,
                kit_task_ids=tuple(kit_ids),
                eligible_bays=tuple(alt_bays),
                eligible_crews=(1, 2, 3),
                resource_windows=r_wins,
            )
        )

    disruptions: dict[str, Any] = {
        "fault_agv": agvs[rng.randrange(len(agvs))].agv_id,
        "fault_step": rng.randint(18, 35),
        "fault_window": 4,
        "event2_regime": "OPEN_WINDOW_KITTING_BOTTLENECK" if regime != "nominal" else "NONE",
        "kitting_bottleneck_window": 4,
        "kitting_bottleneck_delay": 35 if regime == "kitting_windows" else 20,
        "urgent_task_id": "T_URGENT_T1",
        "urgent_pickup": "Q_BOGIE",
        "urgent_dropoff": "D_ASSEMBLY",
        "urgent_release_step": rng.randint(45, 75),
        "urgent_due_step": rng.randint(145, 195),
        "post_pickup_fault_step": rng.randint(85, 130),
        "post_pickup_fault_duration": 20 if regime != "nominal" else 10,
    }

    p2_cfg: Optional[P2PerturbationConfig] = None
    if regime in ("execution_blockage", "multi_wave_surge"):
        p2_cfg = P2PerturbationConfig(
            seed=int(seed) * 29 + 11,
            enable_travel_drift=True,
            drift_prob=0.18,
            early_max_steps=1,
            late_max_steps=4,
            enable_dynamic_obstacles=True,
            obstacle_count=2,
            obstacle_duration_steps=28,
            obstacle_durations=(18, 42),
            obstacle_window_range=(25, 180),
            est_clear_horizon_steps=16,
            enable_local_planner_fail=False,
        )

    return (net, tasks, agvs, ops, disruptions), p2_cfg, 950


def build_track2_scenario(
    seed: int,
    regime: str,
) -> tuple[tuple[Any, ...], Optional[P2PerturbationConfig], int]:
    """Construct paired scenario on Track 2 (`wheelset_workshop_v2`, 50 nodes, 3 AGVs)."""
    rng = _regime_rng(seed, 1319, regime)
    _, net = _load_track2_network()
    starts = ["J_S_65", "J_S_90", "J_S_142"]
    if rng.random() < 0.5:
        starts = ["J_S_90", "J_S_142", "J_S_65"]
    agvs = [
        Agv(f"AGV_{i + 1}", max_speed=0.65, start_node=starts[i], load_time=2.0, unload_time=2.0)
        for i in range(3)
    ]

    flow_specs = [
        ("D_TEARDOWN_0", "D_INSPECTION_0", ("D_INSPECTION_0", "D_INSPECTION_1"), None),
        ("D_INSPECTION_1", "D_PREP_IN", ("D_PREP_IN",), "OP_V2_1"),
        ("D_PREP_OUT", "D_PRESS_IN", ("D_PRESS_IN",), "OP_V2_2"),
        ("D_PRESS_OUT", "D_CLOSING_IN", ("D_CLOSING_IN",), "OP_V2_3"),
        ("D_FINISHING_0", "D_STORAGE", ("D_STORAGE", "D_UNKNOWN"), "OP_V2_4"),
        ("D_FINISHING_1", "D_UNKNOWN", ("D_UNKNOWN", "D_STORAGE"), "OP_V2_5"),
    ]
    if regime in ("kitting_windows", "execution_blockage", "multi_wave_surge"):
        flow_specs.extend([
            ("D_TEARDOWN_0", "D_INSPECTION_1", ("D_INSPECTION_1", "D_INSPECTION_0"), None),
            ("D_CLOSING_IN", "D_FINISHING_0", ("D_FINISHING_0", "D_FINISHING_1"), "OP_V2_7"),
        ])
    if regime == "multi_wave_surge":
        flow_specs.append(("D_PREP_OUT", "D_STORAGE", ("D_STORAGE", "D_UNKNOWN"), None))

    tasks: list[Task] = []
    ops: list[OperationSnapshot] = []
    slack_mult = 1.65 if regime in ("kitting_windows", "multi_wave_surge") else 2.0

    for idx, (pu, do, alt_bays, pred) in enumerate(flow_specs):
        op_id = f"OP_V2_{idx + 1}"
        tid = f"T_V2_{idx + 1}"
        est_t = _travel_estimate(net, pu, do, 0.6)
        rel = (idx // 3) * (35 if regime == "multi_wave_surge" else 55) + rng.randint(0, 15)
        t_due = int(round(rel + max(120.0, est_t * slack_mult)))
        is_urg = idx in (3, len(flow_specs) - 1)
        t_obj = Task(
            tid,
            pickup=pu,
            dropoff=do,
            release_step=rel,
            due_step=t_due,
            priority=TaskPriority.URGENT if is_urg else TaskPriority.NORMAL,
        )
        t_obj.kit_op_id = op_id
        tasks.append(t_obj)
        kit_ids = [tid]

        if regime == "kitting_windows" and idx in (1, 3):
            tid_b = f"T_V2_{idx + 1}_B"
            alt_pu = "D_TEARDOWN_0" if idx == 3 else "D_STORAGE"
            est_b = _travel_estimate(net, alt_pu, do, 0.6)
            t_b = Task(
                tid_b,
                pickup=alt_pu,
                dropoff=do,
                release_step=rel + 12,
                due_step=int(round(rel + max(140.0, est_b * slack_mult))),
            )
            t_b.kit_op_id = op_id
            tasks.append(t_b)
            kit_ids.append(tid_b)

        r_wins: tuple[tuple[int, int], ...] = ()
        if regime == "kitting_windows" and idx in (2, 4):
            w_mark = int(rel + max(150.0, est_t * 1.1))
            r_wins = ((0, w_mark + 65), (w_mark + 145, 2200))

        ops.append(
            OperationSnapshot(
                op_id=op_id,
                kit_task_ids=tuple(kit_ids),
                bay_id=do,
                crew_id=(idx % 3) + 1,
                tool_id=(idx % 4) + 1,
                proc_steps=6 + (idx % 2) * 3,
                due_step=t_due + 60,
                predecessor_op_id=pred,
                predecessor_op_ids=(pred,) if pred else (),
                is_urgent=is_urg,
                eligible_bays=tuple(alt_bays),
                eligible_crews=(1, 2, 3),
                resource_windows=r_wins,
            )
        )

    disruptions: dict[str, Any] = {
        "fault_agv": agvs[rng.randrange(len(agvs))].agv_id,
        "fault_step": rng.randint(45, 95),
        "fault_window": 4,
        "event2_regime": "OPEN_WINDOW_KITTING_BOTTLENECK" if regime != "nominal" else "NONE",
        "kitting_bottleneck_window": 4,
        "kitting_bottleneck_delay": 45 if regime == "kitting_windows" else 25,
        "urgent_task_id": "T_URGENT_V2",
        "urgent_pickup": "D_TEARDOWN_0",
        "urgent_dropoff": "D_STORAGE",
        "urgent_release_step": rng.randint(120, 190),
        "urgent_due_step": rng.randint(360, 440),
        "post_pickup_fault_step": rng.randint(220, 320),
        "post_pickup_fault_duration": 24 if regime != "nominal" else 12,
    }

    p2_cfg: Optional[P2PerturbationConfig] = None
    if regime in ("execution_blockage", "multi_wave_surge"):
        p2_cfg = P2PerturbationConfig(
            seed=int(seed) * 37 + 19,
            enable_travel_drift=True,
            drift_prob=0.18,
            early_max_steps=1,
            late_max_steps=6,
            enable_dynamic_obstacles=True,
            obstacle_count=2,
            obstacle_duration_steps=38,
            obstacle_durations=(18, 48),
            obstacle_window_range=(60, 420),
            est_clear_horizon_steps=20,
            enable_local_planner_fail=False,
        )

    return (net, tasks, agvs, ops, disruptions), p2_cfg, 1800


def _enforce_single_thread_runtime() -> dict[str, str]:
    """Pin BLAS/OpenMP/PyTorch thread pools to 1 for real wall-clock timing."""
    enforced: dict[str, str] = {}
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[var] = "1"
        enforced[var] = "1"
    try:
        import torch

        if torch.get_num_threads() != 1:
            torch.set_num_threads(1)
        enforced["torch_num_threads"] = str(torch.get_num_threads())
    except Exception:
        pass
    return enforced


def _build_scenario_for_track(track: str, seed: int, regime: str) -> tuple[Any, P2PerturbationConfig, int]:
    if track == "track1":
        return build_track1_scenario(seed, regime)
    return build_track2_scenario(seed, regime)


def _run_job(job: tuple[Any, ...]) -> dict[str, Any]:
    if len(job) == 5:
        track, policy, regime, seed, use_rust = job
        clock_mode = "wall"
    else:
        track, policy, regime, seed, use_rust, clock_mode = job
    sc_tuple, p2_cfg, max_steps = _build_scenario_for_track(track, seed, regime)
    rec = run_discrete_episode(
        policy,
        sc_tuple,
        seed=seed,
        max_steps=max_steps,
        use_rust=use_rust,
        p2_cfg=p2_cfg,
        clock_mode=str(clock_mode),
        episode_key=f"{track}:{regime}:{seed}",
    )
    rec["track"] = track
    rec["regime"] = regime
    return rec


def _mean(xs: list[float]) -> float:
    return round(float(statistics.mean(xs)), 4) if xs else 0.0


def _paired_stats(ours: dict[str, dict[str, Any]], base: dict[str, dict[str, Any]]) -> dict[str, Any]:
    keys = sorted(set(ours) & set(base))
    if not keys:
        return {"n": 0}
    deltas = [float(ours[k]["net_obj"]) - float(base[k]["net_obj"]) for k in keys]
    wins = sum(1 for d in deltas if d < -1e-6)
    ties = sum(1 for d in deltas if abs(d) <= 1e-6)
    losses = sum(1 for d in deltas if d > 1e-6)
    out: dict[str, Any] = {
        "n": len(keys),
        "ours_mean_net_obj": _mean([float(ours[k]["net_obj"]) for k in keys]),
        "base_mean_net_obj": _mean([float(base[k]["net_obj"]) for k in keys]),
        "mean_delta_net_obj": _mean(deltas),
        "wins": wins,
        "ties": ties,
        "losses": losses,
    }
    try:
        from scipy.stats import wilcoxon
        if any(abs(d) > 1e-9 for d in deltas):
            stat = wilcoxon(deltas, zero_method="wilcox", alternative="less")
            out["wilcoxon_p_less"] = round(float(stat.pvalue), 6)
    except Exception:
        pass
    return out


def _render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Workshop Track 1 & Track 2 Multi-Algorithm Comparison",
        "",
        f"- Generated UTC: `{report['generated_at']}`",
        f"- Clock: `{report.get('clock_mode', 'wall')}` | Seed Order Rotated: `{report.get('seed_order_rotated', False)}` | Warmup: `{report.get('warmup_performed', False)}`",
        f"- Use Rust: `{report['use_rust']}` | Regimes: `{', '.join(report['regimes'])}` | Seeds: `{report['seeds']}`",
        (
            f"- SHA-256 Scenario Pairing Verified: `{report['sha_pairing_verified']}` | "
            f"Disruption Digest: `{report.get('scenario_digest_includes_disruptions', False)}` | "
            f"100% Audit Passed: `{report['all_audits_passed']}`"
        ),
        "",
    ]
    for track_id, track_title in (
        ("track1", "Track 1 (`rail_freight_workshop_v1`, 62 nodes, 6 AGVs)"),
        ("track2", "Track 2 (`wheelset_workshop_v2`, 50 nodes, 3 AGVs)"),
    ):
        rows = report["summaries"].get(track_id, [])
        if not rows:
            continue
        lines.extend([
            f"## {track_title}",
            "",
            "| Policy | Family | Net Obj | Cmax | Task Tard | Op Tard | Unfinished | Churn Cost | Compute Cost | Task Comp | Prod Comp | Audit | Mean ms | Dec p95 ms | Ckpt SHA | V18 Win/Tie/Loss |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
        ])
        for r in rows:
            p = r.get("vs_v18", {})
            wtl = f"{p.get('wins', 0)}/{p.get('ties', 0)}/{p.get('losses', 0)}" if p else "-"
            ckpt_short = str(r.get("checkpoint_sha256") or "-")[:12]
            lines.append(
                f"| `{r['policy']}` | {r['family']} | {r['net_obj']:.2f} | {r['cmax']:.1f} | "
                f"{r['task_tardiness_cost']:.1f} | {r['op_tardiness_cost']:.1f} | {r['unfinished']:.1f} | "
                f"{r['churn_cost']:.1f} | {r['compute_cost']:.2f} | {r['task_comp']*100:.1f}% | "
                f"{r['prod_comp']*100:.1f}% | {r['audit_pass_rate']*100:.0f}% | {r['wall_ms']:.0f} | "
                f"{r.get('decision_ms_p95', 0.0):.2f} | `{ckpt_short}` | {wtl} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", type=Path, default=ROOT / "out" / "workshop_benchmarks")
    p.add_argument("--tracks", choices=("track1", "track2", "all"), default="all")
    p.add_argument("--policies", type=str, default="12", help="'12', 'all13', or comma-separated names")
    p.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    p.add_argument("--regimes", nargs="+", default=list(REGIMES))
    p.add_argument("--workers", type=int, default=1)
    p.add_argument(
        "--clock",
        choices=("wall", "virtual"),
        default="wall",
        help="Timing clock: single-process serial real wall clock ('wall', default) or deterministic virtual poll ('virtual').",
    )
    p.add_argument(
        "--seed-order-rotate",
        action="store_true",
        help="Rotate policy evaluation order across episodes (enabled automatically when --clock wall).",
    )
    p.add_argument(
        "--no-warmup",
        action="store_true",
        help="Skip short per-policy warmup episodes before timing under --clock wall.",
    )
    p.add_argument("--quick", action="store_true", help="Run 2 regimes x 2 seeds smoke benchmark")
    p.add_argument("--no-rust", action="store_true")
    args = p.parse_args()

    if args.clock == "wall" and int(args.workers) != 1:
        p.error("--clock wall requires --workers 1 (single-process serial timing); pass --clock virtual for multi-worker runs.")

    thread_env = _enforce_single_thread_runtime() if args.clock == "wall" else {}

    if args.policies == "12":
        policies = list(DEFAULT_12_POLICIES)
    elif args.policies == "all13":
        policies = list(SUPPORTED_POLICIES)
    else:
        policies = [resolve_policy_name(x) for x in args.policies.split(",") if x.strip()]

    tracks = ["track1", "track2"] if args.tracks == "all" else [args.tracks]
    regimes = ["nominal", "kitting_windows"] if args.quick else list(args.regimes)
    seeds = list(args.seeds[:2] if args.quick else args.seeds)
    use_rust = (not args.no_rust) and is_rust_planner_available()
    rotate_order = bool(args.seed_order_rotate or args.clock == "wall")
    do_warmup = bool(args.clock == "wall" and not args.no_warmup and policies and tracks and regimes and seeds)

    if do_warmup:
        sc_w, p2_w, _ = _build_scenario_for_track(tracks[0], int(seeds[0]), regimes[0])
        for pol in policies:
            _ = run_discrete_episode(
                pol,
                sc_w,
                seed=int(seeds[0]),
                max_steps=40,
                use_rust=use_rust,
                p2_cfg=p2_w,
                clock_mode="wall",
                episode_key=f"warmup:{pol}",
            )

    jobs: list[tuple[str, str, str, int, bool, str]] = []
    ep_idx = 0
    for tr in tracks:
        for reg in regimes:
            for s in seeds:
                if rotate_order and len(policies) > 1:
                    shift = ep_idx % len(policies)
                    ep_policies = policies[shift:] + policies[:shift]
                else:
                    ep_policies = policies
                ep_idx += 1
                for pol in ep_policies:
                    jobs.append((tr, pol, reg, int(s), use_rust, str(args.clock)))

    t0 = time.perf_counter()
    results: list[dict[str, Any]] = []
    if args.workers <= 1 or args.clock == "wall":
        for idx, jb in enumerate(jobs, 1):
            rec = _run_job(jb)
            results.append(rec)
            print(
                f"[{idx}/{len(jobs)}] {rec['track']} {rec['regime']} s={rec['seed']} "
                f"{rec['policy']}: net_obj={rec['net_obj']:.2f} cmax={rec['cmax']:.1f} "
                f"tcr={rec['task_comp']:.2f} pcr={rec['prod_comp']:.2f} aud={rec['audit_passed']}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futs = [pool.submit(_run_job, jb) for jb in jobs]
            for idx, fut in enumerate(as_completed(futs), 1):
                rec = fut.result()
                results.append(rec)
                print(
                    f"[{idx}/{len(jobs)}] {rec['track']} {rec['regime']} s={rec['seed']} "
                    f"{rec['policy']}: net_obj={rec['net_obj']:.2f}",
                    flush=True,
                )

    sha_by_ep: dict[str, set[str]] = defaultdict(set)
    for r in results:
        sha_by_ep[r["episode_key"]].add(r["scenario_sha"])
    sha_ok = all(len(v) == 1 for v in sha_by_ep.values())
    aud_ok = all(r["audit_passed"] and r["cost_sum_err"] <= 0.01 for r in results)
    digest_has_disruptions = all(bool(r.get("scenario_digest_includes_disruptions")) for r in results)

    summaries: dict[str, list[dict[str, Any]]] = {}
    for tr in tracks:
        tr_rows = [r for r in results if r["track"] == tr]
        v18_by_ep = {r["episode_key"]: r for r in tr_rows if r["policy"] == "V18_EA_JOINT"}
        pol_rows: list[dict[str, Any]] = []
        for pol in policies:
            sub = [r for r in tr_rows if r["policy"] == pol]
            if not sub:
                continue
            by_ep = {r["episode_key"]: r for r in sub}
            ckpt_shas = sorted({str(r["checkpoint_sha256"]) for r in sub if r.get("checkpoint_sha256")})
            pol_rows.append({
                "policy": pol,
                "family": sub[0]["family"],
                "n": len(sub),
                "checkpoint_sha256": ckpt_shas[0] if len(ckpt_shas) == 1 else (",".join(ckpt_shas) if ckpt_shas else None),
                "inference_backend": sub[0].get("inference_backend", "algorithmic"),
                "net_obj": _mean([r["net_obj"] for r in sub]),
                "cmax": _mean([r["cmax"] for r in sub]),
                "task_tardiness_cost": _mean([r["task_tardiness_cost"] for r in sub]),
                "op_tardiness_cost": _mean([r["op_tardiness_cost"] for r in sub]),
                "tardiness": _mean([r["task_tardiness_cost"] + r["op_tardiness_cost"] for r in sub]),
                "unfinished": _mean([r["unfinished_task_penalty"] + r["unfinished_op_penalty"] for r in sub]),
                "churn_cost": _mean([r["churn_cost"] for r in sub]),
                "compute_cost": _mean([r["compute_cost"] for r in sub]),
                "task_comp": _mean([r["task_comp"] for r in sub]),
                "prod_comp": _mean([r["prod_comp"] for r in sub]),
                "audit_pass_rate": _mean([1.0 if r["audit_passed"] else 0.0 for r in sub]),
                "wall_ms": _mean([r["wall_ms"] for r in sub]),
                "decision_ms_p95": _mean([float(r.get("decision_ms_p95", 0.0)) for r in sub]),
                "vs_v18": _paired_stats(v18_by_ep, by_ep) if (v18_by_ep and pol != "V18_EA_JOINT") else {},
            })
        pol_rows.sort(key=lambda x: x["net_obj"])
        summaries[tr] = pol_rows

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_s": round(time.perf_counter() - t0, 2),
        "clock_mode": args.clock,
        "seed_order_rotated": rotate_order,
        "warmup_performed": do_warmup,
        "single_thread_runtime": thread_env,
        "use_rust": use_rust,
        "tracks": tracks,
        "regimes": regimes,
        "seeds": seeds,
        "policies": policies,
        "sha_pairing_verified": sha_ok,
        "scenario_digest_includes_disruptions": digest_has_disruptions,
        "all_audits_passed": aud_ok,
        "summaries": summaries,
        "episodes": results,
    }
    json_path = out_dir / "workshop_comparison.json"
    md_path = out_dir / "workshop_comparison.md"
    json_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_render_markdown(report) + "\n", encoding="utf-8")
    print(f"Saved {json_path} and {md_path} ({report['elapsed_s']}s)")
    if not sha_ok or not aud_ok:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
