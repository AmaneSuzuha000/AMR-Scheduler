"""Run Track A: Wu et al. (EJOR 2026) Independent 3D Warehouse VRP & Storage Policy Study."""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "TORCH_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.ejor26_warehouse_repro.models_and_solvers import (
    AttentionModelVRP,
    DiscreteSACVRPSolver,
    GeneticAlgorithmVRP,
    PPOVRPSolver,
    WuGRUAttentionVRP,
)
from experiments.ejor26_warehouse_repro.storage_policies import STORAGE_POLICIES, assign_storage_slots
from experiments.ejor26_warehouse_repro.warehouse_3d_env import Warehouse3DConfig, Warehouse3DEnv


def _pct95(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(0.95 * len(s)))])


def _save_solver_weights(sol: Any, ckpt_path: Path) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    for attr in ("net", "actor"):
        mod = getattr(sol, attr, None)
        if isinstance(mod, torch.nn.Module):
            payload[attr] = {k: v.detach().cpu() for k, v in mod.state_dict().items()}
    if payload:
        torch.save(payload, ckpt_path)


def _load_solver_weights(sol: Any, ckpt_path: Path, cpu_device: torch.device) -> None:
    if not ckpt_path.is_file():
        return
    payload = torch.load(ckpt_path, map_location=cpu_device, weights_only=True)
    for attr, sd in payload.items():
        mod = getattr(sol, attr, None)
        if isinstance(mod, torch.nn.Module):
            mod.load_state_dict(sd)
            mod.to(cpu_device).eval()
    sol.device = cpu_device


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "out" / "ejor26_warehouse_repro")
    parser.add_argument("--stops", type=int, nargs="+", default=[20, 50])
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--instances-per-seed", type=int, default=12)
    parser.add_argument("--updates-scale", type=float, default=1.0)
    parser.add_argument("--mode", choices=("all", "train", "eval"), default="all")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    train_meta_path = ckpt_dir / "train_times.json"

    if args.mode == "all":
        cmd_base = [
            sys.executable,
            "-B",
            str(Path(__file__).resolve()),
            "--out-dir",
            str(out_dir),
            "--stops",
            *[str(s) for s in args.stops],
            "--seeds",
            *[str(s) for s in args.seeds],
            "--instances-per-seed",
            str(args.instances_per_seed),
            "--updates-scale",
            str(args.updates_scale),
        ]
        train_cmd = [*cmd_base, "--mode", "train"]
        if args.resume:
            train_cmd.append("--resume")
        subprocess.run(train_cmd, check=True)
        time.sleep(1.5)
        eval_env = dict(os.environ)
        eval_env["CUDA_VISIBLE_DEVICES"] = ""
        subprocess.run([*cmd_base, "--mode", "eval"], check=True, env=eval_env)
        return 0

    train_device = torch.device("cuda" if (args.mode == "train" and torch.cuda.is_available()) else "cpu")
    cpu_device = torch.device("cpu")
    env = Warehouse3DEnv(Warehouse3DConfig())

    if args.mode == "train":
        if train_device.type == "cuda":
            _ = torch.zeros((4, 4), device=train_device) @ torch.zeros((4, 4), device=train_device)
            torch.cuda.synchronize()
        train_times: dict[str, dict[str, float]] = {}
        for n_stops in args.stops:
            solvers = [
                WuGRUAttentionVRP(device=train_device),
                AttentionModelVRP(device=train_device),
                PPOVRPSolver(device=train_device),
                DiscreteSACVRPSolver(device=train_device),
                GeneticAlgorithmVRP(seed=7000 + int(n_stops)),
            ]
            n_key = str(n_stops)
            train_times[n_key] = {}
            u_scale = float(args.updates_scale)
            for sol in solvers:
                ckpt_p = ckpt_dir / f"n{n_stops}_{sol.name}.pt"
                meta_p = ckpt_dir / f"n{n_stops}_{sol.name}.meta.json"
                if args.resume and meta_p.is_file() and (ckpt_p.is_file() or isinstance(sol, GeneticAlgorithmVRP)):
                    tr_s = float(json.loads(meta_p.read_text(encoding="utf-8")).get("train_wall_s", 0.0))
                    train_times[n_key][sol.name] = round(tr_s, 3)
                    print(f"[N={n_stops}] resume {sol.name} -> {tr_s:.2f}s", flush=True)
                    continue
                if isinstance(sol, WuGRUAttentionVRP):
                    tr_s = sol.fit(env, n_stops, updates=max(40, int(240 * u_scale)), batch_size=64, seed=2100 + n_stops)
                elif isinstance(sol, AttentionModelVRP):
                    tr_s = sol.fit(env, n_stops, updates=max(40, int(180 * u_scale)), batch_size=64, seed=2200 + n_stops)
                elif isinstance(sol, PPOVRPSolver):
                    tr_s = sol.fit(env, n_stops, updates=max(30, int(160 * u_scale)), batch_size=48, seed=2300 + n_stops)
                elif isinstance(sol, DiscreteSACVRPSolver):
                    tr_s = sol.fit(env, n_stops, updates=max(40, int(160 * u_scale)), batch_size=128, seed=2400 + n_stops)
                else:
                    tr_s = sol.fit(env, n_stops)
                train_times[n_key][sol.name] = round(tr_s, 3)
                _save_solver_weights(sol, ckpt_p)
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                meta_p.write_text(
                    json.dumps({"train_wall_s": round(tr_s, 3), "train_device": str(train_device)}),
                    encoding="utf-8",
                )
                print(f"[N={n_stops}] fit {sol.name} -> {tr_s:.2f}s (device={train_device})", flush=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        train_meta_path.write_text(
            json.dumps({"train_device": str(train_device), "train_times": train_times}, indent=2) + "\n",
            encoding="utf-8",
        )
        return 0

    # args.mode == "eval": clean pure-CPU single-thread evaluation
    torch.set_num_threads(1)
    if train_meta_path.is_file():
        meta = json.loads(train_meta_path.read_text(encoding="utf-8"))
        recorded_train_device = str(meta.get("train_device", "cuda"))
        train_times = meta.get("train_times", {})
    else:
        recorded_train_device = "none"
        train_times = {str(n): {} for n in args.stops}

    all_rows: list[dict[str, Any]] = []
    for n_stops in args.stops:
        solvers = [
            WuGRUAttentionVRP(device=cpu_device),
            AttentionModelVRP(device=cpu_device),
            PPOVRPSolver(device=cpu_device),
            DiscreteSACVRPSolver(device=cpu_device),
            GeneticAlgorithmVRP(seed=7000 + int(n_stops)),
        ]
        n_key = str(n_stops)
        for sol in solvers:
            _load_solver_weights(sol, ckpt_dir / f"n{n_stops}_{sol.name}.pt", cpu_device)
            train_times.setdefault(n_key, {}).setdefault(sol.name, 0.0)

        w_cat = env.build_sku_catalog(999)
        w_smap = assign_storage_slots(w_cat, env, "semi_circle", seed=999)
        w_inst = env.generate_instance(w_cat, w_smap, n_stops, 9999, "semi_circle", "warmup")
        for sol in solvers:
            sol.solve(w_inst)

        for pol in STORAGE_POLICIES:
            for sd in args.seeds:
                catalog = env.build_sku_catalog(int(sd))
                smap = assign_storage_slots(catalog, env, pol, seed=int(sd))
                for inst_i in range(int(args.instances_per_seed)):
                    inst_seed = int(sd) * 10000 + inst_i + 1
                    inst = env.generate_instance(catalog, smap, n_stops, inst_seed, pol, f"N{n_stops}_{pol}_{sd}_{inst_i}")
                    mean_depot_t = float(np.mean(inst.dist_matrix[0, 1:]))

                    shift = (int(sd) + inst_i) % len(solvers)
                    rot_solvers = solvers[shift:] + solvers[:shift]
                    for sol in rot_solvers:
                        t0 = time.perf_counter()
                        route, tot_time = sol.solve(inst)
                        infer_ms = (time.perf_counter() - t0) * 1000.0
                        chk = env.evaluate_tour(inst, route)
                        all_rows.append({
                            "n_stops": int(n_stops),
                            "storage_policy": pol,
                            "seed": int(sd),
                            "instance_id": inst.instance_id,
                            "solver": sol.name,
                            "total_time_s": round(tot_time, 4),
                            "reloads": int(chk["reloads"]),
                            "valid": bool(chk["valid"]),
                            "mean_depot_time_s": round(mean_depot_t, 4),
                            "infer_ms": round(infer_ms, 3),
                        })

        print(f"Completed N={n_stops} (train={recorded_train_device}, eval=cpu_single_thread)", flush=True)

    train_device = torch.device(recorded_train_device if recorded_train_device != "none" else "cpu")

    # Summarize results
    summary: dict[str, Any] = {}
    for n_stops in args.stops:
        n_key = str(n_stops)
        summary[n_key] = {}
        for pol in STORAGE_POLICIES:
            summary[n_key][pol] = {}
            for sol_name in ("Wu_GRU_Attn", "AttentionModel", "PPO", "Discrete_SAC", "GA_OX_2opt"):
                sub = [
                    r for r in all_rows
                    if r["n_stops"] == n_stops and r["storage_policy"] == pol and r["solver"] == sol_name
                ]
                if not sub:
                    continue
                times = [r["total_time_s"] for r in sub]
                lat = [r["infer_ms"] for r in sub]
                summary[n_key][pol][sol_name] = {
                    "n": len(sub),
                    "valid_rate": round(float(statistics.mean(1.0 if r["valid"] else 0.0 for r in sub)), 4),
                    "mean_total_time_s": round(float(statistics.mean(times)), 3),
                    "std_total_time_s": round(float(statistics.pstdev(times)), 3),
                    "mean_reloads": round(float(statistics.mean(r["reloads"] for r in sub)), 3),
                    "mean_depot_time_s": round(float(statistics.mean(r["mean_depot_time_s"] for r in sub)), 3),
                    "infer_ms_mean": round(float(statistics.mean(lat)), 3),
                    "infer_ms_p95": round(_pct95(lat), 3),
                    "train_wall_s": train_times[n_key][sol_name],
                }

    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "data_provenance": "synthetic_benchmark_wu_ejor26_formulation",
        "isolated_from_amr_factory_tables": True,
        "train_device": str(train_device),
        "eval_device": "cpu_single_thread",
        "stops": list(args.stops),
        "seeds": list(args.seeds),
        "instances_per_seed": int(args.instances_per_seed),
        "all_valid": all(r["valid"] for r in all_rows),
        "train_times_s": train_times,
        "summary": summary,
        "rows": all_rows,
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    md: list[str] = [
        "# Track A: Wu et al. (EJOR 2026) Independent 3D Warehouse VRP Reproduction",
        "",
        f"- **Data Provenance**: `{report['data_provenance']}` (Synthetic 3D rack benchmark; isolated from factory AMR tables)",
        f"- **Offline Training Device**: `{train_device}` | **Online Inference Timing**: `{report['eval_device']}` (1 thread, rotated)",
        f"- **100% Tour Legality Verified**: `{report['all_valid']}`",
        "",
    ]
    for n_stops in args.stops:
        n_key = str(n_stops)
        md.extend([
            f"## N = {n_stops} Stops",
            "",
            "| Storage Policy | Mean Depot Time (s) | Solver | Mean Route Time (s) | Std (s) | Mean Reloads | Valid | CPU Infer Mean (ms) | CPU Infer p95 (ms) | GPU Train Time (s) |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ])
        for pol in STORAGE_POLICIES:
            for sol_name, m in sorted(summary[n_key][pol].items(), key=lambda kv: kv[1]["mean_total_time_s"]):
                md.append(
                    f"| `{pol}` | {m['mean_depot_time_s']:.2f} | `{sol_name}` | {m['mean_total_time_s']:.2f} | "
                    f"{m['std_total_time_s']:.2f} | {m['mean_reloads']:.2f} | {m['valid_rate']*100:.0f}% | "
                    f"{m['infer_ms_mean']:.2f} | {m['infer_ms_p95']:.2f} | {m['train_wall_s']:.2f} |"
                )
        md.append("")
    (out_dir / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"Saved Track A report -> {out_dir / 'report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
