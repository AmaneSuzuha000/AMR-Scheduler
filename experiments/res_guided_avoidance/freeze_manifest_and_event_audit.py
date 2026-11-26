"""Task 1: Frozen Evidence Manifest, Independent Collision Event Audit & Workshop T_min Check."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.res_guided_avoidance.geometry_and_shield import MAX_V_MPS
from experiments.res_guided_avoidance.gate1_headroom_audit import m2_res_margin_selector
from experiments.res_guided_avoidance.metric_world import (
    MultiRobotUnicycleSim,
    build_grid_metric_map,
    build_occluded_junction_map,
    build_workshop_metric_map,
)
from experiments.res_guided_avoidance.ramp_ppo import RAMP_CKPT_DIR, NumpyRampPolicy


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _shortest_metric_path_m(net, src: str, dst: str) -> float:
    import heapq
    pq = [(0.0, src)]
    best = {src: 0.0}
    while pq:
        d, u = heapq.heappop(pq)
        if u == dst:
            return d
        if d > best.get(u, 1e18) + 1e-9:
            continue
        for e in net.out_edges(u):
            nd = d + e.length
            if nd < best.get(e.dst, 1e18):
                best[e.dst] = nd
                heapq.heappush(pq, (nd, e.dst))
    return 1e9


def main() -> None:
    out_dir = ROOT / "out" / "decision_headroom"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Frozen Evidence Manifest
    assets = [
        ROOT / "experiments" / "res_guided_avoidance" / "geometry_and_shield.py",
        ROOT / "experiments" / "res_guided_avoidance" / "metric_world.py",
        ROOT / "experiments" / "res_guided_avoidance" / "motion_primitives.py",
        ROOT / "experiments" / "res_guided_avoidance" / "reservation_features.py",
        ROOT / "experiments" / "res_guided_avoidance" / "ramp_ppo.py",
        ROOT / "out" / "ramp_robust" / "robust_ramp_report.json",
        ROOT / "out" / "ramp_robust" / "factorial_and_blind_per_episode.csv",
    ]
    for sd in (42, 43, 44):
        assets.append(RAMP_CKPT_DIR / f"r5_ramp_ppo_s{sd}.npz")
        assets.append(RAMP_CKPT_DIR / f"r5_ramp_ppo_robust_s{sd}.npz")

    manifest = {
        "observed_dev_and_audit_seeds": {
            "gate1_seeds": [5001, 5002, 5003],
            "dev_factorial_seeds": [6001, 6002, 6003],
            "observed_confirmation_seeds": [7001, 7002, 7003, 7004, 7005, 7006],
            "reserved_unseen_blind_seeds": [8001, 8002, 8003, 8004, 8005, 8006, 8007, 8008, 8009, 8010],
        },
        "files": [
            {"path": str(p.relative_to(ROOT)).replace("\\", "/"), "bytes": p.stat().st_size, "sha256": _sha256(p)}
            for p in assets if p.is_file()
        ],
    }

    # 2. Independent collision events vs substep ticks on R5_Orig failing cases
    om = build_occluded_junction_map(cell_m=2.6)
    wm = build_workshop_metric_map()
    r5_s42 = NumpyRampPolicy.load_npz(RAMP_CKPT_DIR / "r5_ramp_ppo_s42.npz")
    event_audit = []
    for label, mp, nr, nt, hz, np_m, nv_m, rob, pol in (
        ("Workshop_8R_R5_Orig_s6001", wm, 8, 12, 160.0, 0.0, 0.0, False, r5_s42),
        ("Noise0.12_R5_Orig_s6001", om, 6, 12, 70.0, 0.12, 0.20, False, r5_s42),
        ("Noise0.12_R5_Robust_s6001", om, 6, 12, 70.0, 0.12, 0.20, True, r5_s42),
        ("Workshop_8R_R1_Robust_s6001", wm, 8, 12, 160.0, 0.0, 0.0, True, m2_res_margin_selector),
    ):
        sim = MultiRobotUnicycleSim(mp, n_robots=nr, n_tasks=nt, horizon_s=hz, seed=6001, n_intruders=(1 if mp is wm else 2), obs_noise_pos_m=np_m, obs_noise_vel_mps=nv_m, robust_safety=rob)
        res = sim.run_episode(pol, controller_name=label)
        substeps = res.wall_collisions + res.peer_collisions + res.intruder_collisions
        event_audit.append({
            "case": label,
            "zero_collisions": res.zero_collisions,
            "independent_collision_events": sim.independent_collision_events,
            "collision_substep_ticks": substeps,
            "wall_peer_intr_substeps": [res.wall_collisions, res.peer_collisions, res.intruder_collisions],
            "completed_tasks": res.completed_tasks,
        })
        print(f"[Task1 EventAudit | {label:28s}] indep_events={sim.independent_collision_events:3d} substep_ticks={substeps:5d} zcol={res.zero_collisions}", flush=True)

    # 3. Physical shortest-path transit time T_min for Gate 0 4R_LifelongWorkshop
    sim_ws = MultiRobotUnicycleSim(wm, n_robots=4, n_tasks=4, horizon_s=300.0, seed=42, n_intruders=0)
    # Match 4 tasks to 4 robots via exact initial greedy assignment & measure shortest path lower bound
    task_lbs = []
    for i, tk in enumerate(list(sim_ws.pending) + list(sim_ws.ready)):
        rb = sim_ws.robots[f"R_{i % 4:02d}"]
        d_pu = _shortest_metric_path_m(wm.network, rb.cur_node, tk.pickup)
        d_do = _shortest_metric_path_m(wm.network, tk.pickup, tk.dropoff)
        t_min_cruise = (d_pu + d_do) / 0.95 + tk.service_s + tk.release_s
        t_min_vmax = (d_pu + d_do) / MAX_V_MPS + tk.service_s + tk.release_s
        task_lbs.append({
            "task_id": tk.task_id,
            "dist_m": round(d_pu + d_do, 2),
            "release_s": tk.release_s,
            "t_lb_vmax_1.35": round(t_min_vmax, 2),
            "t_lb_cruise_0.95": round(t_min_cruise, 2),
        })
    ws_lb = {
        "tasks": task_lbs,
        "max_t_lb_vmax_1.35": max(t["t_lb_vmax_1.35"] for t in task_lbs),
        "max_t_lb_cruise_0.95": max(t["t_lb_cruise_0.95"] for t in task_lbs),
        "gate0_original_horizon_s": 130.0,
        "gate0_actual_4task_makespan_s": 252.25,
    }
    print(f"[Task1 Workshop T_min] max T_lb@vmax={ws_lb['max_t_lb_vmax_1.35']}s, max T_lb@0.95={ws_lb['max_t_lb_cruise_0.95']}s (vs 130.0s original horizon)", flush=True)

    payload = {"frozen_manifest": manifest, "event_vs_substep_audit": event_audit, "workshop_4r_t_min_check": ws_lb}
    (out_dir / "task1_frozen_manifest_and_audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
