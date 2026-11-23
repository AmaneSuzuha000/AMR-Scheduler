"""62-Node 4-Axis Stress & Scaling Benchmark (`out/scale_4axis/`).

Evaluates the 6 core policies (`V19_HETERO_RL`, `V18_EA_JOINT`, `IJCAI26_PGMPO`,
`rule_v10`, `EJOR26_Wu_GRU_Attn`, `RAL26_DetourDelta`) in single-process real wall-clock
mode (`--clock wall --workers 1`, seeds `1171..1173`) across 4 orthogonal scaling axes on
the 62-node Rail Freight Workshop (Track 1):
  - Axis 1 (Fleet Size |A|): 4, 6, 8, 10 AMRs
  - Axis 2 (Task/Op Horizon Load |T|): 1.0x (8 tasks), 1.5x (12 tasks), 2.0x (16 tasks)
  - Axis 3 (Disruption & Corridor Blockage Severity): nominal, kitting_windows,
    execution_blockage, multi_wave_surge
  - Axis 4 (Real-Time Decision Budget): 20 ms, 35 ms, 50 ms, 100 ms
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from amr_sched.baselines.policy_registry import create_discrete_policy_env, run_discrete_episode
from amr_sched.cvcr.auditor import IndependentAuditor
from amr_sched.cvcr.production import OperationSnapshot
from amr_sched.entities import Agv, Task
from amr_sched.rl import compute_expert_v10_action_mask, compute_expert_v9_state
from tools.benchmarks.compare_workshop_algorithms import build_track1_scenario

OUT_SCALE_DIR = ROOT / "out" / "scale_4axis"
SCALE_SEEDS: tuple[int, ...] = (1171, 1172, 1173)
EVAL_6_POLICIES: tuple[str, ...] = (
    "V19_HETERO_RL",
    "V18_EA_JOINT",
    "IJCAI26_PGMPO",
    "rule_v10",
    "EJOR26_Wu_GRU_Attn",
    "RAL26_DetourDelta",
)

EXTRA_START_NODES: tuple[str, ...] = (
    "J_S_25",
    "J_N_25",
    "J_S_105",
    "J_N_105",
    "J_S_125",
    "J_N_125",
)


def _scale_fleet(sc_tuple: tuple[Any, ...], n_agvs: int) -> tuple[Any, ...]:
    net, tasks, agvs, ops, meta = copy.deepcopy(sc_tuple)
    base_agvs: list[Agv] = list(agvs)
    if n_agvs <= len(base_agvs):
        scaled_agvs = base_agvs[:n_agvs]
    else:
        scaled_agvs = list(base_agvs)
        used_starts = {a.start_node for a in scaled_agvs}
        cand_nodes = [n for n in EXTRA_START_NODES if n in net.nodes and n not in used_starts] + [
            n for n in net.nodes if n not in used_starts
        ]
        while len(scaled_agvs) < n_agvs:
            idx = len(scaled_agvs) + 1
            st_node = cand_nodes[(idx - 1) % len(cand_nodes)]
            tmpl = base_agvs[0]
            scaled_agvs.append(
                Agv(
                    agv_id=f"R{idx}",
                    max_speed=tmpl.max_speed,
                    start_node=st_node,
                    load_time=tmpl.load_time,
                    unload_time=tmpl.unload_time,
                )
            )
    return (net, tasks, scaled_agvs, ops, meta)


def _scale_load(sc_tuple: tuple[Any, ...], mult: float, max_steps: int) -> tuple[tuple[Any, ...], int]:
    if mult <= 1.01:
        return copy.deepcopy(sc_tuple), int(max_steps)
    net, tasks, agvs, ops, meta = copy.deepcopy(sc_tuple)
    base_tasks: list[Task] = list(tasks)
    base_ops: list[OperationSnapshot] = list(ops)
    n_extra_ops = int(round((mult - 1.0) * len(base_ops)))
    wave_offset = 145

    new_tasks = list(base_tasks)
    new_ops = list(base_ops)
    task_by_id = {t.task_id: t for t in base_tasks}

    for w_i in range(n_extra_ops):
        src_op = base_ops[w_i % len(base_ops)]
        shift = wave_offset * (1 + (w_i // len(base_ops))) + (w_i % len(base_ops)) * 18
        new_oid = f"{src_op.op_id}_W{w_i + 2}"
        kit_ids: list[str] = []
        for ktid in src_op.kit_task_ids:
            if ktid not in task_by_id:
                continue
            st = task_by_id[ktid]
            ntid = f"{ktid}_W{w_i + 2}"
            new_tasks.append(
                Task(
                    task_id=ntid,
                    pickup=st.pickup,
                    dropoff=st.dropoff,
                    release_step=int(st.release_step) + shift,
                    due_step=int(st.due_step or 240) + shift + 80,
                    priority=st.priority,
                    material_type=st.material_type,
                )
            )
            kit_ids.append(ntid)
        if kit_ids:
            new_ops.append(
                replace(
                    src_op,
                    op_id=new_oid,
                    kit_task_ids=tuple(kit_ids),
                    due_step=int(src_op.due_step) + shift + 90,
                    predecessor_op_id=None,
                    predecessor_op_ids=(),
                    started=False,
                    start_step=None,
                    end_step=None,
                )
            )
    scaled_ms = int(round(max_steps * (1.0 + 0.55 * (mult - 1.0))))
    return (net, new_tasks, agvs, new_ops, meta), scaled_ms


def _run_budget_episode(
    policy_name: str,
    sc_tuple: tuple[Any, ...],
    p2_cfg: Any,
    *,
    seed: int,
    max_steps: int,
    budget_ms: float,
    episode_key: str,
) -> dict[str, Any]:
    """Run one real wall-clock episode overriding the proposer decision budget to `budget_ms`."""
    sc_copy = copy.deepcopy(sc_tuple)
    p2_copy = copy.deepcopy(p2_cfg)
    kind, ctrl, env = create_discrete_policy_env(
        policy_name,
        seed=seed,
        max_steps=max_steps,
        use_rust=True,
        p2_cfg=p2_copy,
        execution_aware=True,
        p4_choke_robust=True,
        backend="numpy",
    )
    if env.external_proposer is not None:
        raw_prop = env.external_proposer

        def _budget_wrapped(snap: Any, net: Any, budget_ms: float = budget_ms) -> Any:
            return raw_prop(snap, net, budget_ms=float(budget_ms))

        env.external_proposer = _budget_wrapped

    t0 = time.perf_counter()
    obs, info = env.reset(seed=int(seed), options={"scenario_tuple": sc_copy})
    tm = False
    tr = False
    while not (tm or tr):
        assert env.sim is not None and env.tracker is not None
        if kind == "rule_v10":
            feats = compute_expert_v9_state(
                env.sim, env.tracker,
                last_replan_tick=env.last_replan_tick,
                recent_conflict_ticks=env.recent_conflict_ticks,
                decision_logs=env.decision_logs,
            )
            step_info = dict(info, action_mask=compute_expert_v10_action_mask(env.sim, env.last_replan_tick))
        elif kind == "v18":
            feats = ctrl.observe(env, obs)
            step_info = info
        else:
            feats = env.get_expert_v12_obs(obs)
            step_info = info
        act = ctrl.select_action(feats, step_info, sim=env.sim)
        obs, _, tm, tr, info = env.step(act)
    wall_ms = (time.perf_counter() - t0) * 1000.0
    m = env.compute_episode_metrics()
    aud = IndependentAuditor.audit_simulation_run(env.sim, env.tracker.operations, require_trajectory=True)
    dec_ms = sorted(float(r["t_propose_ms"]) + float(r["t_validate_commit_ms"]) for r in env.decision_logs)
    overruns = sum(1 for v in dec_ms if v > float(budget_ms))
    return {
        "policy": policy_name,
        "episode_key": episode_key,
        "seed": seed,
        "budget_ms": float(budget_ms),
        "audit_passed": bool(aud.pass_audit),
        "net_obj": round(float(m["net_obj"]), 4),
        "cmax": round(float(m["cmax"]), 4),
        "tardiness": round(float(m["total_tardiness"]), 4),
        "churn": int(m["total_churn"]),
        "task_comp": round(float(m["task_completion_rate"]), 6),
        "prod_comp": round(float(m["production_completion_rate"]), 6),
        "mean_decision_ms": round(float(np.mean(dec_ms)) if dec_ms else 0.0, 3),
        "decision_ms_p95": round(float(np.percentile(dec_ms, 95)) if dec_ms else 0.0, 3),
        "decision_ms_p99": round(float(np.percentile(dec_ms, 99)) if dec_ms else 0.0, 3),
        "deadline_overrun_rate": round(float(overruns / max(1, len(dec_ms))), 6),
        "wall_ms": round(wall_ms, 2),
    }


def _agg(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": len(rows),
        "audit_pass_rate": round(float(np.mean([1.0 if r["audit_passed"] else 0.0 for r in rows])), 6),
        "mean_net_obj": round(float(np.mean([r["net_obj"] for r in rows])), 4),
        "mean_cmax": round(float(np.mean([r["cmax"] for r in rows])), 4),
        "mean_tardiness": round(float(np.mean([r["tardiness"] for r in rows])), 4),
        "mean_churn": round(float(np.mean([r["churn"] for r in rows])), 4),
        "mean_task_comp": round(float(np.mean([r["task_comp"] for r in rows])), 6),
        "mean_prod_comp": round(float(np.mean([r["prod_comp"] for r in rows])), 6),
        "mean_decision_ms": round(float(np.mean([r["mean_decision_ms"] for r in rows])), 3),
        "p95_decision_ms": round(float(np.mean([r.get("decision_ms_p95", r["mean_decision_ms"]) for r in rows])), 3),
        "p99_decision_ms": round(float(np.mean([r.get("decision_ms_p99", r["mean_decision_ms"]) for r in rows])), 3),
        "deadline_overrun_rate": round(float(np.mean([r.get("deadline_overrun_rate", 0.0) for r in rows])), 6),
    }


def run_4axis_scaling(out_dir: Path = OUT_SCALE_DIR) -> dict[str, Any]:
    """Execute the 62-node 4-axis scaling benchmark (`--clock wall --workers 1`)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    axes_results: dict[str, dict[str, dict[str, Any]]] = {
        "axis1_fleet_size": {},
        "axis2_task_load": {},
        "axis3_disruption_regime": {},
        "axis4_decision_budget_ms": {},
    }
    t0 = time.perf_counter()

    # Axis 1: Fleet size |A| in {4, 6, 8, 10}
    for n_a in (4, 6, 8, 10):
        lvl_key = f"agvs_{n_a}"
        axes_results["axis1_fleet_size"][lvl_key] = {}
        for pol in EVAL_6_POLICIES:
            rows = []
            for s in SCALE_SEEDS:
                sc_base, p2_cfg, ms = build_track1_scenario(s, "kitting_windows")
                sc_fl = _scale_fleet(sc_base, n_a)
                rec = run_discrete_episode(
                    pol, sc_fl, seed=s, max_steps=ms, p2_cfg=p2_cfg,
                    clock_mode="wall", episode_key=f"ax1_a{n_a}_{s}", backend="numpy",
                )
                rec["deadline_overrun_rate"] = round(
                    float(rec["proposer_stats"].get("deadline_overruns", 0)) / max(1, int(rec["decision_count"])), 6
                )
                rows.append(rec)
            axes_results["axis1_fleet_size"][lvl_key][pol] = _agg(rows)
        print(f"[Scale 4-Axis] Axis 1 {lvl_key} complete ({time.perf_counter() - t0:.1f}s)", flush=True)

    # Axis 2: Task/Op load multiplier in {1.0, 1.5, 2.0}
    for mult in (1.0, 1.5, 2.0):
        lvl_key = f"load_{mult:.1f}x"
        axes_results["axis2_task_load"][lvl_key] = {}
        for pol in EVAL_6_POLICIES:
            rows = []
            for s in SCALE_SEEDS:
                sc_base, p2_cfg, ms = build_track1_scenario(s, "nominal")
                sc_ld, ms_ld = _scale_load(sc_base, mult, ms)
                rec = run_discrete_episode(
                    pol, sc_ld, seed=s, max_steps=ms_ld, p2_cfg=p2_cfg,
                    clock_mode="wall", episode_key=f"ax2_m{mult:.1f}_{s}", backend="numpy",
                )
                rec["deadline_overrun_rate"] = round(
                    float(rec["proposer_stats"].get("deadline_overruns", 0)) / max(1, int(rec["decision_count"])), 6
                )
                rows.append(rec)
            axes_results["axis2_task_load"][lvl_key][pol] = _agg(rows)
        print(f"[Scale 4-Axis] Axis 2 {lvl_key} complete ({time.perf_counter() - t0:.1f}s)", flush=True)

    # Axis 3: Disruption & blockage severity across all 4 workshop regimes
    for reg in ("nominal", "kitting_windows", "execution_blockage", "multi_wave_surge"):
        axes_results["axis3_disruption_regime"][reg] = {}
        for pol in EVAL_6_POLICIES:
            rows = []
            for s in SCALE_SEEDS:
                sc_base, p2_cfg, ms = build_track1_scenario(s, reg)
                rec = run_discrete_episode(
                    pol, sc_base, seed=s, max_steps=ms, p2_cfg=p2_cfg,
                    clock_mode="wall", episode_key=f"ax3_{reg}_{s}", backend="numpy",
                )
                rec["deadline_overrun_rate"] = round(
                    float(rec["proposer_stats"].get("deadline_overruns", 0)) / max(1, int(rec["decision_count"])), 6
                )
                rows.append(rec)
            axes_results["axis3_disruption_regime"][reg][pol] = _agg(rows)
        print(f"[Scale 4-Axis] Axis 3 {reg} complete ({time.perf_counter() - t0:.1f}s)", flush=True)

    # Axis 4: Real-time decision budget in {20, 35, 50, 100} ms
    for b_ms in (20.0, 35.0, 50.0, 100.0):
        lvl_key = f"budget_{int(b_ms)}ms"
        axes_results["axis4_decision_budget_ms"][lvl_key] = {}
        for pol in EVAL_6_POLICIES:
            rows = []
            for s in SCALE_SEEDS:
                sc_base, p2_cfg, ms = build_track1_scenario(s, "execution_blockage")
                rows.append(
                    _run_budget_episode(
                        pol, sc_base, p2_cfg, seed=s, max_steps=ms,
                        budget_ms=b_ms, episode_key=f"ax4_b{int(b_ms)}_{s}",
                    )
                )
            axes_results["axis4_decision_budget_ms"][lvl_key][pol] = _agg(rows)
        print(f"[Scale 4-Axis] Axis 4 {lvl_key} complete ({time.perf_counter() - t0:.1f}s)", flush=True)

    report = {
        "gate": "SCALE_4AXIS_62NODE",
        "topology": "rail_freight_workshop_v1 (62 nodes)",
        "clock_mode": "wall",
        "workers": 1,
        "seeds": list(SCALE_SEEDS),
        "policies": list(EVAL_6_POLICIES),
        "axes": axes_results,
    }
    json_path = out_dir / "scale_4axis_report.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    md_lines = [
        "# 62-Node 4-Axis Scaling & Real-Time Budget Report",
        "",
        f"- **Topology**: 62-node Rail Freight Workshop (`seeds={list(SCALE_SEEDS)}`, `--clock wall --workers 1`)",
        "",
    ]
    for ax_name, levels in axes_results.items():
        md_lines.append(f"## {ax_name}")
        md_lines.append("")
        md_lines.append("| Level | Policy | Audit | Task Comp | Prod Comp | Mean `net_obj` | Mean `cmax` | Mean Decision (ms) | P95 (ms) | Overrun Rate |")
        md_lines.append("| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
        for lvl, pmap in levels.items():
            for pol in EVAL_6_POLICIES:
                s = pmap[pol]
                md_lines.append(
                    f"| `{lvl}` | **{pol}** | {s['audit_pass_rate']*100:.0f}% | {s['mean_task_comp']*100:.1f}% | "
                    f"{s['mean_prod_comp']*100:.1f}% | **{s['mean_net_obj']:.1f}** | {s['mean_cmax']:.1f} | "
                    f"{s['mean_decision_ms']:.2f} | {s['p95_decision_ms']:.2f} | {s['deadline_overrun_rate']*100:.2f}% |"
                )
        md_lines.append("")
    md_path = out_dir / "scale_4axis_report.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    print(f"[Scale 4-Axis] Saved {json_path} & {md_path}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 62-node 4-axis scaling benchmark.")
    parser.add_argument("--out-dir", type=Path, default=OUT_SCALE_DIR)
    args = parser.parse_args()
    run_4axis_scaling(args.out_dir)


if __name__ == "__main__":
    main()
