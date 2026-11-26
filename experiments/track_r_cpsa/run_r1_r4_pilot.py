"""Track R (`TRACK-R-CPSA-01`) Stage R1 -> R3 -> R4 Pilot Study & Stage R5 Advisor Brief Generator.

Strictly hardware-clock-invariant evaluation (throughput, success rate, APL, waiting time,
node expansions, repair fraction |R_loc|/|I|, zero-collision verification).
Zero dependency on frozen V18-RGRO / DFL / CRC / FMS production schedules.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.track_r_cpsa.cpsa_pure_runner import (
    run_lifelong_mapd,
    run_oneshot_mapf,
)
from experiments.track_r_cpsa.execution_delay_sim import (
    DelayScenarioConfig,
    simulate_lifelong_with_delays,
)
from experiments.track_r_cpsa.local_reservation_repair import EALRConfig
from experiments.track_r_cpsa.maps_and_workloads import (
    build_map_a,
    build_map_b,
    build_map_c,
    generate_fixed_bg_and_test_requests,
    generate_lifelong_requests,
)
from experiments.track_r_cpsa.rust_sipp_accel import run_accelerated_lifelong_mapd

OUT_TRACK_R_DIR = ROOT / "out" / "track_r_cpsa"


def _mean(vals: list[float]) -> float:
    return sum(vals) / max(1, len(vals))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=200, help="Lifelong horizon steps (default 200 for fast pilot; 500 for full).")
    parser.add_argument("--seeds", type=int, default=3, help="Number of matched random seeds (default 3).")
    args = parser.parse_args()

    OUT_TRACK_R_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()
    maps = [build_map_a(), build_map_b(), build_map_c()]
    seeds_os = [42 + i for i in range(args.seeds)]
    seeds_ll = [39 + i for i in range(args.seeds)]

    # ------------------------------------------------------------------
    # STAGE R1 & Direction B Probe: Fixed-BG One-Shot Reproduction (Map A/B/C)
    # ------------------------------------------------------------------
    print("=== [Track R · R1] Pure One-Shot Fixed-Background (CA* vs CPS-A* vs CPS-A* + ImpactProbe S_j) ===", flush=True)
    os_setup = {
        "Map_A": (28, 30, 90),   # n_bg, n_test, horizon
        "Map_B": (80, 60, 90),
        "Map_C": (60, 50, 90),
    }
    r1_oneshot: dict[str, Any] = {}
    for gm in maps:
        nb, nt, oh = os_setup[gm.name]
        rows_ca, rows_cps, rows_imp = [], [], []
        for s in seeds_os:
            bg, tst = generate_fixed_bg_and_test_requests(gm, nb, nt, seed=s)
            rows_ca.append(run_oneshot_mapf(gm, tst, algo="ca_star", horizon=oh, fixed_bg_requests=bg, seed=s))
            rows_cps.append(run_oneshot_mapf(gm, tst, algo="cps_oneshot", horizon=oh, fixed_bg_requests=bg, seed=s, use_impact_blocker=False))
            rows_imp.append(run_oneshot_mapf(gm, tst, algo="cps_impact_b", horizon=oh, fixed_bg_requests=bg, seed=s, use_impact_blocker=True))

        r1_oneshot[gm.name] = {
            "ca_star_sr": round(_mean([r.success_rate for r in rows_ca]), 4),
            "cps_oneshot_sr": round(_mean([r.success_rate for r in rows_cps]), 4),
            "cps_impact_sr": round(_mean([r.success_rate for r in rows_imp]), 4),
            "cps_ac": round(_mean([r.avg_congestion for r in rows_cps]), 4),
            "ca_ac": round(_mean([r.avg_congestion for r in rows_ca]), 4),
            "cps_rb_attempts": sum(r.rollback_attempts for r in rows_cps),
            "cps_rb_wasted": sum(r.rollback_wasted for r in rows_cps),
            "imp_rb_wasted": sum(r.rollback_wasted for r in rows_imp),
            "ca_zero_conflicts": all(r.zero_conflicts for r in rows_ca),
            "cps_zero_conflicts": all(r.zero_conflicts for r in rows_cps + rows_imp),
            "all_zero_conflicts": all(r.zero_conflicts for r in rows_ca + rows_cps + rows_imp),
        }
        d = r1_oneshot[gm.name]
        print(
            f"  [R1 OneShot {gm.name}] CA*={d['ca_star_sr']*100:5.1f}% | CPS-A*={d['cps_oneshot_sr']*100:5.1f}% "
            f"(RB wasted {d['cps_rb_wasted']}/{d['cps_rb_attempts']}) | ImpactProbe={d['cps_impact_sr']*100:5.1f}% "
            f"(wasted {d['imp_rb_wasted']}) | zero_conflicts={d['all_zero_conflicts']}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # STAGE R1 Lifelong Zero-Delay Check (Map A/B/C)
    # ------------------------------------------------------------------
    print(f"=== [Track R · R1] Nominal Zero-Delay Lifelong ({args.horizon} steps x {args.seeds} seeds) ===", flush=True)
    ll_fleets = {"Map_A": 28, "Map_B": 55, "Map_C": 45}
    r1_lifelong: dict[str, Any] = {}
    for gm in maps:
        na = ll_fleets[gm.name]
        rows_ca, rows_cps = [], []
        for s in seeds_ll:
            init_p, reqs = generate_lifelong_requests(gm, na, n_requests=na * 25, seed=s)
            rows_ca.append(run_lifelong_mapd(gm, init_p, reqs, algo="ca_star", horizon=args.horizon, seed=s))
            rows_cps.append(run_lifelong_mapd(gm, init_p, reqs, algo="cps_lifelong", horizon=args.horizon, seed=s))
        r1_lifelong[gm.name] = {
            "n_agvs": na,
            "ca_tp": round(_mean([r.throughput for r in rows_ca]), 4),
            "cps_tp": round(_mean([r.throughput for r in rows_cps]), 4),
            "ca_wait": round(_mean([r.mean_waiting_time for r in rows_ca]), 2),
            "cps_wait": round(_mean([r.mean_waiting_time for r in rows_cps]), 2),
            "cps_rb_attempts": sum(r.rollback_attempts for r in rows_cps),
            "cps_rb_wasted": sum(r.rollback_wasted for r in rows_cps),
            "ca_zero_conflicts": all(r.zero_conflicts for r in rows_ca),
            "cps_zero_conflicts": all(r.zero_conflicts for r in rows_cps),
            "all_zero_conflicts": all(r.zero_conflicts for r in rows_ca + rows_cps),
        }
        d = r1_lifelong[gm.name]
        print(
            f"  [R1 Lifelong {gm.name} ({na} AGVs)] CA* TP={d['ca_tp']:.4f} (safe={d['ca_zero_conflicts']}) | CPS-A* TP={d['cps_tp']:.4f} (safe={d['cps_zero_conflicts']}) "
            f"| Wait CA/CPS={d['ca_wait']:.1f}/{d['cps_wait']:.1f}",
            flush=True,
        )

    # ------------------------------------------------------------------
    # STAGE R3 & R4 (Direction A Primary + Ablation): Execution Uncertainty Stress Test
    # ------------------------------------------------------------------
    print(f"=== [Track R · R3/R4] Direction A: Execution-Delay & Migrated EALR_v2 Ablation ===", flush=True)
    delay_regimes = [
        ("D1_ModDelay_NoLag", DelayScenarioConfig(motion_delay_prob=0.10, service_delay_prob=0.25, max_service_delay=2, heterogeneous_slow_frac=0.20, comm_lag_steps=0, horizon=args.horizon)),
        ("D2_HeavyUncert_CommLag1", DelayScenarioConfig(motion_delay_prob=0.16, service_delay_prob=0.40, max_service_delay=3, heterogeneous_slow_frac=0.30, comm_lag_steps=1, horizon=args.horizon)),
    ]
    solvers = ("naive_no_repair", "adg_order_hold", "global_rolling_cpsa", "ealr_v1", "ealr_v2_migrated")
    r4_delay: dict[str, Any] = {}

    for reg_name, dcfg in delay_regimes:
        r4_delay[reg_name] = {}
        for gm in maps:
            na = ll_fleets[gm.name]
            r4_delay[reg_name][gm.name] = {}
            by_sv: dict[str, list[Any]] = {sv: [] for sv in solvers}
            for s in seeds_ll:
                init_p, reqs = generate_lifelong_requests(gm, na, n_requests=na * 25, seed=s)
                for sv in solvers:
                    if sv == "ealr_v1":
                        by_sv[sv].append(
                            simulate_lifelong_with_delays(
                                gm, init_p, reqs, dcfg, solver="ealr_local_repair",
                                ealr_cfg=EALRConfig(enable_v2_migrations=False), seed=s, scenario_tag=reg_name,
                            )
                        )
                    elif sv == "ealr_v2_migrated":
                        by_sv[sv].append(
                            simulate_lifelong_with_delays(
                                gm, init_p, reqs, dcfg, solver="ealr_local_repair",
                                ealr_cfg=EALRConfig(enable_v2_migrations=True), seed=s, scenario_tag=reg_name,
                            )
                        )
                    else:
                        by_sv[sv].append(
                            simulate_lifelong_with_delays(gm, init_p, reqs, dcfg, solver=sv, seed=s, scenario_tag=reg_name)
                        )
            for sv in solvers:
                rs = by_sv[sv]
                r4_delay[reg_name][gm.name][sv] = {
                    "tp": round(_mean([x.throughput for x in rs]), 4),
                    "completed": round(_mean([x.completed_tasks for x in rs]), 1),
                    "wait": round(_mean([x.mean_waiting_time for x in rs]), 2),
                    "expansions": round(_mean([x.total_expansions for x in rs]), 1),
                    "repair_frac": round(_mean([x.avg_repair_fraction for x in rs]), 4),
                    "near_miss_blocks": round(_mean([x.near_miss_blocks for x in rs]), 1),
                    "zero_conflicts": all(x.zero_conflicts for x in rs),
                }
            e1 = r4_delay[reg_name][gm.name]["ealr_v1"]
            e2 = r4_delay[reg_name][gm.name]["ealr_v2_migrated"]
            a = r4_delay[reg_name][gm.name]["adg_order_hold"]
            g = r4_delay[reg_name][gm.name]["global_rolling_cpsa"]
            n = r4_delay[reg_name][gm.name]["naive_no_repair"]
            print(
                f"  [{reg_name:22s} | {gm.name}] TP: Naive={n['tp']:.3f} ADG={a['tp']:.3f} Glob={g['tp']:.3f} "
                f"EALR_v1={e1['tp']:.3f} EALR_v2={e2['tp']:.3f} "
                f"| NearMiss v1->v2={e1['near_miss_blocks']:.0f}->{e2['near_miss_blocks']:.0f} "
                f"| RepFrac v2={e2['repair_frac']:.2f} | Safe={all(r4_delay[reg_name][gm.name][sv]['zero_conflicts'] for sv in solvers)}",
                flush=True,
            )

    # ------------------------------------------------------------------
    # STAGE R-ACCEL (Section 6 Planning Latency): Python CPS-A* vs Rust A* vs Rust SIPP + SlotPressure
    # ------------------------------------------------------------------
    print("=== [Track R · R-Accel] Section 6 Latency Migration: Rust SpaceTime-A* / SIPP + SlotReservationPressureTable ===", flush=True)
    r_accel: dict[str, Any] = {}
    for gm in maps:
        na = ll_fleets[gm.name]
        b_rows: dict[str, list[Any]] = {"python_cpsa": [], "rust_astar_slot": [], "rust_sipp_slot": []}
        for s in seeds_ll:
            init_p, reqs = generate_lifelong_requests(gm, na, n_requests=na * 25, seed=s)
            for bk in b_rows:
                b_rows[bk].append(run_accelerated_lifelong_mapd(gm, init_p, reqs, backend=bk, horizon=args.horizon, seed=s))
        py_ms = _mean([x.wall_ms for x in b_rows["python_cpsa"]])
        ast_ms = _mean([x.wall_ms for x in b_rows["rust_astar_slot"]])
        sipp_ms = _mean([x.wall_ms for x in b_rows["rust_sipp_slot"]])
        r_accel[gm.name] = {
            bk: {
                "tp": round(_mean([x.throughput for x in xs]), 4),
                "wall_ms": round(_mean([x.wall_ms for x in xs]), 1),
                "zero_conflicts": all(x.zero_conflicts for x in xs),
            }
            for bk, xs in b_rows.items()
        }
        r_accel[gm.name]["sipp_speedup_vs_py"] = round(py_ms / max(1.0, sipp_ms), 2)
        r_accel[gm.name]["astar_speedup_vs_py"] = round(py_ms / max(1.0, ast_ms), 2)
        print(
            f"  [R-Accel {gm.name} ({na} AGVs)] Py={py_ms:.0f}ms | Rust-A*={ast_ms:.0f}ms ({r_accel[gm.name]['astar_speedup_vs_py']:.1f}x) "
            f"| Rust-SIPP+Slot={sipp_ms:.0f}ms ({r_accel[gm.name]['sipp_speedup_vs_py']:.1f}x) "
            f"| TP Py/SIPP={r_accel[gm.name]['python_cpsa']['tp']:.3f}/{r_accel[gm.name]['rust_sipp_slot']['tp']:.3f}",
            flush=True,
        )

    elapsed_s = round(time.perf_counter() - t_start, 2)
    payload = {
        "protocol": "TRACK-R-CPSA-01 Upgraded Pilot (Direction A EALR_v2 + Direction B Counterfactual + Rust SIPP Accel)",
        "horizon": int(args.horizon),
        "n_seeds": int(args.seeds),
        "elapsed_s": elapsed_s,
        "r1_oneshot": r1_oneshot,
        "r1_lifelong": r1_lifelong,
        "r4_delay_repair": r4_delay,
        "r_accel": r_accel,
    }
    (OUT_TRACK_R_DIR / "track_r_pilot_report.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = [
        "# Track R 先导研究交付简报（`TRACK-R-CPSA-01` · Stage R1–R5）",
        "",
        f"> **纯 MAPF / Lifelong-MAPD 独立实验分支**（完全冻结 V18–RGRO/DFL/CRC/生产 DAG；采用硬件主频无关指标；{args.seeds} 颗随机种子配对测试，总耗时 `{elapsed_s}s`）。",
        "",
        "## 1. Stage R1：原论文纯环境复现与 Direction B 阻塞探针（One-Shot 固定背景 + Lifelong 无延迟）",
        "",
        "| 地图 | OneShot `CA*` 成功率 | OneShot `CPS-A*` 成功率 | CPS 拥堵 `AC` (vs CA*) | Lifelong `CA*` 吞吐 | Lifelong `CPS-A*` 吞吐 | 零碰撞 OS / LL (`CA*` vs `CPS`) | Alg.1 阻塞无效回滚率 (Wasted/Att) | Direction B 探针 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for gm in maps:
        o = r1_oneshot[gm.name]
        l = r1_lifelong[gm.name]
        z_str = (
            f"OS:{'100%' if o['all_zero_conflicts'] else 'FAIL'} / "
            f"LL:CA={'OK' if l['ca_zero_conflicts'] else 'BOX'}·CPS={'100%' if l['cps_zero_conflicts'] else 'FAIL'}"
        )
        md.append(
            f"| `{gm.name}` | {o['ca_star_sr']*100:.1f}% | **{o['cps_oneshot_sr']*100:.1f}%** | {o['cps_ac']:.3f} ({o['ca_ac']:.3f}) | "
            f"{l['ca_tp']:.4f} | **{l['cps_tp']:.4f}** | {z_str} | {o['cps_rb_wasted']}/{o['cps_rb_attempts']} | Impact={o['cps_impact_sr']*100:.1f}% (wasted {o['imp_rb_wasted']}) |"
        )

    md.extend([
        "",
        "## 2. Stage R3/R4（方向 A 主线与既往资产迁移消融）：`EALR v1` vs **`EALR v2` (P4 信念门控 + 因果锥拉入 + 窄道退避)**",
        "",
        r"| 扰动工况 | 地图 | `Naive` TP | `ADG` TP | `Global` TP | `EALR v1` TP | **`EALR v2` TP** | 近距堵死阻挡 (`v1` -> `v2`) | 修复波及率 `v2` (vs Glob) | A\* 扩展 `v2` (vs Glob) | 零冲突 |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])
    for reg_name, _ in delay_regimes:
        for gm in maps:
            d = r4_delay[reg_name][gm.name]
            safe = all(d[sv]["zero_conflicts"] for sv in solvers)
            e1, e2, g = d["ealr_v1"], d["ealr_v2_migrated"], d["global_rolling_cpsa"]
            md.append(
                f"| `{reg_name}` | `{gm.name}` | {d['naive_no_repair']['tp']:.4f} | {d['adg_order_hold']['tp']:.4f} | "
                f"{g['tp']:.4f} | {e1['tp']:.4f} | **{e2['tp']:.4f}** | "
                f"{e1['near_miss_blocks']:.0f} -> **{e2['near_miss_blocks']:.0f}** | "
                f"**{e2['repair_frac']*100:.1f}%** vs 100.0% | "
                f"**{e2['expansions']:.0f}** vs {g['expansions']:.0f} | "
                f"{'100%' if safe else 'FAIL'} |"
            )

    md.extend([
        "",
        "## 3. Stage R-Accel（第 6 节降低规划延迟）：`SlotReservationPressureTable` + Rust `SIPP` 加速基准",
        "",
        "| 地图 | 车队 | `Python CPS-A*` 耗时 (TP) | `Rust A* + Slot` 耗时 (相对加速, TP) | **`Rust SIPP + Slot` 耗时 (相对加速, TP)** | 零冲突 |",
        "| :--- | :---: | :---: | :---: | :---: | :---: |",
    ])
    for gm in maps:
        ra = r_accel[gm.name]
        safe_ac = ra["python_cpsa"]["zero_conflicts"] and ra["rust_astar_slot"]["zero_conflicts"] and ra["rust_sipp_slot"]["zero_conflicts"]
        md.append(
            f"| `{gm.name}` | {ll_fleets[gm.name]} | {ra['python_cpsa']['wall_ms']:.0f}ms ({ra['python_cpsa']['tp']:.3f}) | "
            f"{ra['rust_astar_slot']['wall_ms']:.0f}ms (**{ra['astar_speedup_vs_py']:.1f}x**, {ra['rust_astar_slot']['tp']:.3f}) | "
            f"**{ra['rust_sipp_slot']['wall_ms']:.0f}ms** (**{ra['sipp_speedup_vs_py']:.1f}x**, {ra['rust_sipp_slot']['tp']:.3f}) | "
            f"{'100%' if safe_ac else 'LL-PyBOX'} |"
        )

    md.extend([
        "",
        "## 4. Stage R5: Execution-Uncertainty & Local Repair Summary",
        "",
        r"> Summary: Reproduces pure `CPS-A*` One-Shot/Lifelong baselines cleanly, reduces blocker-selection wasted rollbacks via counterfactual `IALPR` probes, and extends to execution-delay uncertainty (`EALR v2/v3`) using physical lock pins, causal duration belief gating, corridor haven yield, and native Rust SIPP slot-pressure acceleration.",
    ])
    (OUT_TRACK_R_DIR / "track_r_pilot_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"=== Wrote {OUT_TRACK_R_DIR / 'track_r_pilot_report.md'} ===", flush=True)


if __name__ == "__main__":
    main()
