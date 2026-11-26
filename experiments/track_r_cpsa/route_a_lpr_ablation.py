"""Route A (Layer 2/3) Controlled Ablation: CPS-A* Local Priority Repair (LPR) Selectors.

Strictly isolates the L2/L3 Local Priority Repair blocker-selection mechanism while
holding the spatiotemporal planner (Rust CPS-A*), congestion field (SlotCongestionField),
and upper dispatch policy (`Wu26_SemiCircle_GRU`, zero dispatch DFL) 100% constant:
  1. `LPR_MaxHits`: Classical CPS-A* hits-count priority repair rule.
  2. `LPR_ImpactHeuristic`: Spatiotemporal impact + remaining path + SlotPhi heuristic.
  3. `LPR_ZeroShotDFL`: CVCR commitment-tier protection + zero-shot `mh_dfl_critic.npz`.
  4. `LPR_LearnedWorkshopDFL`: CVCR commitment-tier protection + counterfactual-tuned
     `mh_dfl_critic_workshop.npz`.

Evaluated across both:
  - Regime A1: Pure One-Shot MAPF with fixed background traffic (`Official_Map_A/B/C`).
  - Regime A2: Asymmetric Workshop Lifelong (`demo17` & `rail_freight_workshop_v1`),
    plus `W_Stress1 (120T x 6 AGV)` recovery verification.
"""

from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any, Optional

import numpy as np

from amr_sched.planners.rust_spacetime_astar import RustCPSAStarGraphPlanner
from amr_sched.rust_reservation import RustReservationTable
from experiments.cps_astar_routing.cps_astar_graph import CPSAStarConfig
from experiments.rgro.dfl_critic import (
    CAND_RAW_DIM,
    DFLAdvantageEnsemble,
    HIST_DIM,
    build_anchor_relative_feature,
)
from experiments.track_r_cpsa.cpsa_pure_runner import (
    expand_planned_to_step_positions,
    impact_aware_blocker_score,
    verify_mapf_trace_safety,
)
from experiments.track_r_cpsa.maps_and_workloads import (
    GridMapSpec,
    build_official_trc_map,
    generate_fixed_bg_and_test_requests,
)
from experiments.track_r_cpsa.workshop_bridge_eval import (
    _build_workshop_history_vec,
    _dispatch_tasks_wu26,
    _load_dfl_critic,
    _load_wu26_net,
    run_workshop_episode,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "out" / "track_r_cpsa"
WS_DFL_CKPT = ROOT / "amr_sched" / "checkpoints" / "track_r" / "mh_dfl_critic_workshop.npz"
WS_WU_CKPT = ROOT / "amr_sched" / "checkpoints" / "track_r" / "ejor26_wu_gru_workshop.npz"


@dataclass(frozen=True)
class OneShotLPRResult:
    map_name: str
    selector: str
    seed: int
    n_test: int
    n_success: int
    success_rate: float
    avg_congestion: float
    total_expansions: int
    lpr_attempts: int
    lpr_wasted: int
    waste_rate: float
    zero_conflicts: bool


def _select_oneshot_blocker_dfl(
    aid: str,
    cands: list[str],
    hits: dict[str, int],
    uncommitted: dict[str, list[tuple[str, int]]],
    grid: GridMapSpec,
    field: Any,
    dfl_critic: DFLAdvantageEnsemble,
    n_total: int,
) -> Optional[str]:
    """Rank One-Shot LPR blockers via DFL Advantage Critic + CVCR choke awareness."""
    if not cands:
        return None
    is_large = 1.0 if len(grid.network.nodes) >= 180 else 0.0
    blocked_frac = min(1.0, float(len(cands)) / float(max(1, n_total // 4)))
    h_vec = _build_workshop_history_vec(
        blocked_frac=blocked_frac,
        choke_frac=blocked_frac * 0.6,
        urgent_pressure=0.5,
        comp_rate=0.4,
        horizon_frac=0.35,
    )
    fv_anchor = np.zeros(CAND_RAW_DIM, dtype=np.float32)
    fv_anchor[12] = 1.0
    fv_anchor[22] = is_large

    x_rows: list[np.ndarray] = []
    base_scores: list[tuple[float, float]] = []
    for bj in cands:
        pts_j = uncommitted[bj]
        start_nd = pts_j[0][0]
        on_choke = 1.0 if len(grid.network.out_edges(start_nd)) <= 2 else 0.0
        imp = impact_aware_blocker_score(aid, bj, hits, len(pts_j), field.slot_Phi(start_nd, 0))
        fv_c = np.zeros(CAND_RAW_DIM, dtype=np.float32)
        fv_c[0] = -0.15 * float(imp)
        fv_c[1] = fv_c[0]
        fv_c[2] = -0.1 * float(hits.get(bj, 1))
        fv_c[7] = -0.2 * float(hits.get(bj, 1))
        fv_c[8] = blocked_frac * 0.5
        fv_c[9] = -0.2 * on_choke
        fv_c[13] = 0.15
        fv_c[15] = 0.5 / float(max(4, n_total))
        fv_c[17] = on_choke * 0.5
        fv_c[22] = is_large
        x_rows.append(build_anchor_relative_feature(fv_c, fv_anchor, h_vec))
        base_scores.append((imp, on_choke))

    meta_trained_ws = str(getattr(dfl_critic, "metadata", {}).get("domain", "")) == "workshop_v1"
    dfl_out = dfl_critic.score_batch(np.stack(x_rows, axis=0), kappa=0.5)
    best_bj = cands[0]
    best_val = -1e18
    for idx, bj in enumerate(cands):
        imp, on_choke = base_scores[idx]
        lcb = max(-3.0, min(3.0, float(dfl_out[idx].lcb_advantage) / 100.0))
        harm = float(dfl_out[idx].harm_prob)
        ep_conf = 1.0 if meta_trained_ws else (1.0 / (1.0 + float(dfl_out[idx].epistemic_std) / 80.0))
        sc = imp + 1.0 * on_choke + (0.65 if meta_trained_ws else 0.45) * ep_conf * lcb - (1.8 if meta_trained_ws else 1.2) * ep_conf * harm
        if sc > best_val:
            best_val = sc
            best_bj = bj
    return best_bj


def run_oneshot_lpr_case(
    grid: GridMapSpec,
    bg_requests: list[tuple[str, str, str]],
    test_requests: list[tuple[str, str, str]],
    *,
    selector: str,
    dfl_critic: Optional[DFLAdvantageEnsemble] = None,
    horizon: int = 90,
    seed: int = 0,
) -> OneShotLPRResult:
    """Run controlled Rust E2E One-Shot MAPF with pluggable LPR blocker selector."""
    cfg = CPSAStarConfig(
        mode="oneshot",
        beta=0.5,
        gamma=5.0,
        lookahead_omega=50.0,
        batch_update_K=20,
        retry_limit_R=5,
        max_expansions=max(4000, 12 * len(grid.network.nodes)),
    )
    planner = RustCPSAStarGraphPlanner(grid.network, cfg, max_speed=1.0, dt=1.0)
    res_table = RustReservationTable.fast()
    field = planner.create_field(max_track_horizon=horizon + 10, t_start=0)

    all_traces: dict[str, list[tuple[str, int]]] = {}
    for aid, st, gl in bg_requests:
        pl, _ = planner.plan_leg(st, gl, 0, res_table, field, aid, horizon, goal_park_until=horizon)
        if pl is not None:
            pts = expand_planned_to_step_positions(st, 0, pl)
            planner.reserve_step_points_indexed(res_table, aid, pts, park_until=horizon)
            field.add_agv_path(aid, pts, horizon=horizon)
            all_traces[aid] = pts

    req_map = {aid: (st, gl) for aid, st, gl in test_requests}
    q: deque[str] = deque([aid for aid, _, _ in test_requests])
    uncommitted: dict[str, list[tuple[str, int]]] = {}
    accepted: dict[str, list[tuple[str, int]]] = {}
    retries: dict[str, int] = {aid: 0 for aid in req_map}
    total_exp = 0
    rb_attempts = 0
    rb_wasted = 0
    n_tot = len(bg_requests) + len(test_requests)

    while q:
        aid = q.popleft()
        st, gl = req_map[aid]
        planned, hits = planner.plan_leg(st, gl, 0, res_table, field, aid, horizon, goal_park_until=horizon)
        total_exp += int(planner.last_expansions)
        if planned is not None:
            pts = expand_planned_to_step_positions(st, 0, planned)
            planner.reserve_step_points_indexed(res_table, aid, pts, park_until=horizon)
            uncommitted[aid] = pts
            if len(uncommitted) >= cfg.batch_update_K:
                for caid, cpts in uncommitted.items():
                    accepted[caid] = cpts
                    field.add_agv_path(caid, cpts, horizon=horizon)
                uncommitted.clear()
            continue

        if retries[aid] >= cfg.retry_limit_R or not uncommitted:
            continue

        hit_blockers = [j for j in uncommitted if int(hits.get(j, 0)) > 0]
        if not hit_blockers and selector != "LPR_MaxHits":
            # Skip non-blocking rollbacks when only fixed background causes infeasibility
            continue
        cands = hit_blockers or list(uncommitted.keys())

        if selector == "LPR_MaxHits":
            blocker_j = max(cands, key=lambda j: (int(hits.get(j, 0)), len(uncommitted[j]), j))
        elif selector == "LPR_ImpactHeuristic":
            blocker_j = max(
                cands,
                key=lambda j: (
                    impact_aware_blocker_score(
                        aid, j, hits, len(uncommitted[j]), field.slot_Phi(uncommitted[j][0][0], 0)
                    ),
                    j,
                ),
            )
        else:
            assert dfl_critic is not None
            blocker_j = _select_oneshot_blocker_dfl(aid, cands, hits, uncommitted, grid, field, dfl_critic, n_tot)
            if blocker_j is None:
                continue

        rb_attempts += 1
        retries[aid] += 1
        saved_j = uncommitted.pop(blocker_j)
        res_table.release_all(blocker_j)
        field.remove_agv(blocker_j)
        p_chk, _ = planner.plan_leg(st, gl, 0, res_table, field, aid, horizon, goal_park_until=horizon)
        total_exp += int(planner.last_expansions)
        if p_chk is None:
            rb_wasted += 1
            planner.reserve_step_points_indexed(res_table, blocker_j, saved_j, park_until=horizon)
            uncommitted[blocker_j] = saved_j
            continue
        pts_aid = expand_planned_to_step_positions(st, 0, p_chk)
        planner.reserve_step_points_indexed(res_table, aid, pts_aid, park_until=horizon)
        uncommitted[aid] = pts_aid
        q.append(blocker_j)

    for caid, cpts in uncommitted.items():
        accepted[caid] = cpts
        field.add_agv_path(caid, cpts, horizon=horizon)

    all_traces.update(accepted)
    n_succ = len(accepted)
    n_t = len(test_requests)
    cong_vals = [field.slot_Phi(nd, t) for pts in accepted.values() for nd, t in pts]
    avg_cong = float(sum(cong_vals)) / max(1, len(cong_vals)) if cong_vals else 0.0
    trace_map = {aid: {t: nd for nd, t in pts} for aid, pts in all_traces.items()}
    safe_ok, _ = verify_mapf_trace_safety(grid.network, trace_map)
    return OneShotLPRResult(
        map_name=grid.name,
        selector=selector,
        seed=int(seed),
        n_test=n_t,
        n_success=n_succ,
        success_rate=round(float(n_succ) / max(1, n_t), 4),
        avg_congestion=round(avg_cong, 4),
        total_expansions=total_exp,
        lpr_attempts=rb_attempts,
        lpr_wasted=rb_wasted,
        waste_rate=round(float(rb_wasted) / max(1, rb_attempts), 4),
        zero_conflicts=bool(safe_ok),
    )


def main() -> None:
    t0 = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    zero_dfl = _load_dfl_critic()
    adapted_dfl = DFLAdvantageEnsemble.load_npz(WS_DFL_CKPT) if WS_DFL_CKPT.exists() else zero_dfl
    wu_net = _load_wu26_net(WS_WU_CKPT) if WS_WU_CKPT.exists() else _load_wu26_net()

    print("=== [Route A1] One-Shot Pure Traffic LPR Selector Ablation (Rust E2E) ===", flush=True)
    os_setup = [
        ("Official_Map_A", build_official_trc_map("A"), 36, 26, 90),
        ("Official_Map_B", build_official_trc_map("B"), 88, 56, 90),
        ("Official_Map_C", build_official_trc_map("C"), 68, 44, 90),
    ]
    selectors = [
        ("LPR_MaxHits", None, "max_hits"),
        ("LPR_ImpactHeuristic", None, "impact"),
        ("LPR_ZeroShotDFL", zero_dfl, "dfl"),
        ("LPR_LearnedWorkshopDFL", adapted_dfl, "dfl"),
    ]
    a1_summary: list[dict[str, Any]] = []
    for map_name, gm, n_tst, n_bg, hor in os_setup:
        for sel_name, cr_obj, _ in selectors:
            sub: list[OneShotLPRResult] = []
            for sd in (39, 40, 41, 42):
                bg_r, tst_r = generate_fixed_bg_and_test_requests(gm, n_bg, n_tst, seed=sd)
                sub.append(run_oneshot_lpr_case(gm, bg_r, tst_r, selector=sel_name, dfl_critic=cr_obj, horizon=hor, seed=sd))
            att_sum = sum(r.lpr_attempts for r in sub)
            wst_sum = sum(r.lpr_wasted for r in sub)
            rec = {
                "map": map_name,
                "selector": sel_name,
                "success_rate": round(sum(r.success_rate for r in sub) / len(sub), 4),
                "avg_congestion": round(sum(r.avg_congestion for r in sub) / len(sub), 4),
                "expansions": round(sum(r.total_expansions for r in sub) / len(sub), 1),
                "lpr_attempts": att_sum,
                "lpr_wasted": wst_sum,
                "waste_rate": round(wst_sum / max(1, att_sum), 4),
                "safe": all(r.zero_conflicts for r in sub),
            }
            a1_summary.append(rec)
            print(
                f"  [{map_name:14s} | {sel_name:22s}] SR={rec['success_rate']*100:5.1f}% "
                f"Attempts={att_sum:3d} Wasted={wst_sum:3d} ({rec['waste_rate']*100:4.1f}%) "
                f"Exp={rec['expansions']:7.1f} Safe={rec['safe']}",
                flush=True,
            )

    print("\n=== [Route A2] Weighted Workshop LPR Selector Ablation (Fixed Dispatch) ===", flush=True)

    def _fixed_dispatch(idle_agvs, ready_tasks, cur_pos, network, tt_cache, tick, ctx):
        return _dispatch_tasks_wu26(
            idle_agvs, ready_tasks, cur_pos, network, tt_cache, tick, wu_net,
            dfl_critic=None,
            haven_prof=ctx.get("haven_prof"),
            bay_load=ctx.get("bay_load"),
        )

    ws_cases = [
        ("demo17_40T_6AGV", "demo17", 1, 40, 6, 900, 0.12, (42, 43, 44)),
        ("workshop_v1_40T_6AGV", "workshop_v1", 1, 40, 6, 900, 0.12, (42, 43, 44)),
        ("workshop_v1_80T_12AGV", "workshop_v1", 1, 80, 12, 900, 0.15, (42, 43)),
        ("W_Stress1_120T_6AGV", "workshop_v1", 1, 120, 6, 1200, 0.12, (42, 43)),
    ]
    a2_summary: list[dict[str, Any]] = []
    for tag, lay, cap, nt, na, hor, dp, seeds in ws_cases:
        for sel_name, cr_obj, lpr_mode in selectors:
            sub_w = [
                run_workshop_episode(
                    layout=lay, aisle_capacity=cap, n_tasks=nt, n_agvs=na, seed=sd,
                    dispatcher="Wu26_SemiCircle_GRU", router="CPS_CVCR_DFL",
                    delay_prob=dp, horizon=hor, wu_net=wu_net, dfl_critic=cr_obj,
                    dispatch_override=_fixed_dispatch, use_rust_e2e=True,
                    lpr_selector_mode=lpr_mode,
                )
                for sd in seeds
            ]
            att_s = sum(r.lpr_attempts for r in sub_w)
            wst_s = sum(r.lpr_wasted for r in sub_w)
            rec = {
                "scenario": tag,
                "selector": sel_name,
                "n_tasks": nt,
                "completed": round(sum(r.completed_tasks for r in sub_w) / len(sub_w), 1),
                "makespan": round(sum(r.makespan for r in sub_w) / len(sub_w), 1),
                "on_time": round(sum(r.on_time_rate for r in sub_w) / len(sub_w), 4),
                "waits": round(sum(r.conflict_wait_count for r in sub_w) / len(sub_w), 1),
                "expansions": round(sum(r.total_expansions for r in sub_w) / len(sub_w), 1),
                "lpr_attempts": att_s,
                "lpr_wasted": wst_s,
                "waste_rate": round(wst_s / max(1, att_s), 4),
                "safe": all(r.zero_conflicts for r in sub_w),
            }
            a2_summary.append(rec)
            print(
                f"  [{tag:21s} | {sel_name:22s}] Done={rec['completed']:5.1f}/{nt} MS={rec['makespan']:6.1f} "
                f"OnTime={rec['on_time']*100:4.1f}% Waits={rec['waits']:6.1f} "
                f"LPR={att_s}/{wst_s} ({rec['waste_rate']*100:4.1f}%) Safe={rec['safe']}",
                flush=True,
            )

    elapsed = round(time.perf_counter() - t0, 2)
    payload = {"elapsed_s": elapsed, "oneshot_lpr": a1_summary, "workshop_lpr": a2_summary}
    (OUT_DIR / "route_a_ablation_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Route A (L2/L3) 局部优先级修复 (LPR) 严格控制变量消融实验报告",
        "",
        f"> 运行耗时：`{elapsed}s` · 全局底层统一使用 `Rust CPS-A* + SlotCongestionField`，上层派车严格锁定为微调后的 `Wu26_SemiCircle_GRU`（零派车 DFL 干扰），仅替换 L2/L3 局部回滚阻塞者选择器（Blocker Selector）。",
        "",
        "## 一、One-Shot 稠密交通图 LPR 消融（Official Map A/B/C 含背景交通, 4 种子均值）",
        "",
        "| 地图场景 | LPR 阻塞者选择器 | 成功率 | 尝试次数 | 无效回滚 | 浪费率 | 平均扩展数 | 零碰撞 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for r in a1_summary:
        md.append(
            f"| `{r['map']}` | `{r['selector']}` | {r['success_rate']*100:.1f}% | {r['lpr_attempts']} | "
            f"{r['lpr_wasted']} | {r['waste_rate']*100:.1f}% | {r['expansions']:.1f} | {r['safe']} |"
        )
    md.extend([
        "",
        "## 二、检修车间加权异步图 Lifelong LPR 消融（含 `W_Stress1 120T` 积压工况验证）",
        "",
        "| 车间场景 | LPR 阻塞者选择器 | 完工数 | Makespan | 准时率 | 冲突等待 | LPR 尝试/浪费 | 零碰撞 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for r in a2_summary:
        md.append(
            f"| `{r['scenario']}` | `{r['selector']}` | {r['completed']:.1f}/{r['n_tasks']} | {r['makespan']:.1f} | "
            f"{r['on_time']*100:.1f}% | {r['waits']:.1f} | {r['lpr_attempts']}/{r['lpr_wasted']} ({r['waste_rate']*100:.1f}%) | {r['safe']} |"
        )
    (OUT_DIR / "route_a_ablation_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"=== Wrote {OUT_DIR / 'route_a_ablation_report.md'} ({elapsed}s) ===", flush=True)


if __name__ == "__main__":
    main()
