"""Can a free 6-DOF refit rescue a camera whose stored pose misses its own landmarks?

Two cameras in the suite record a fit residual larger than the tolerance they are judged
against: exterior_check 25.36 px against 20, cross_transfer 24.28 px against 20. A stored
residual that high means either
    (a) the stored pose is stale or was fitted sloppily, and a free refit lands close, or
    (b) the landmark world points are mutually inconsistent with a pinhole camera at this
        focal length, and no pose can fit them.
Those have opposite remedies - refit the camera, or fix the landmark data - so guessing is
expensive. This tool settles it: it fits the pose freely from scratch, from many starts,
and reports the best achievable residual.

The camera model is fixed to the one the project uses everywhere: pinhole, horizontal field
of view 0.92 rad, 1280x720. Only the six pose parameters move.

A result far below the stored residual means the stored pose is simply bad. A result that
stays high means the data cannot be explained by any pose and must be re-read.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[2]
WIDTH, HEIGHT = 1280.0, 720.0
FOV = 0.92
FOCAL = (WIDTH / 2.0) / math.tan(FOV / 2.0)


def euler_to_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    mx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    my = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    mz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return mz @ my @ mx


def project(params: np.ndarray, world: np.ndarray) -> np.ndarray:
    """params = [px, py, pz, rx, ry, rz]; camera looks down its own -Z with +Y up."""
    forward = euler_to_matrix(*params[3:]) @ np.array([0.0, 0.0, -1.0])
    up = euler_to_matrix(*params[3:]) @ np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    delta = world - params[:3]
    z = delta @ forward
    if np.any(z <= 1e-3):
        return None
    u = (delta @ right) / z * FOCAL + WIDTH / 2.0
    v = HEIGHT / 2.0 - (delta @ up) / z * FOCAL
    return np.stack([u, v], axis=1)


def residuals(params: np.ndarray, world: np.ndarray, pixel: np.ndarray) -> np.ndarray:
    projected = project(params, world)
    if projected is None:
        return np.full(pixel.size, 1e5)
    return (projected - pixel).ravel()


def look_at_params(position: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = target - position
    forward = forward / np.linalg.norm(forward)
    up_world = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, up_world)
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    # Build rotation whose columns are (right, up, -forward) so that the local -Z looks forward.
    matrix = np.stack([right, up, -forward], axis=1)
    # Extract Euler XYZ that reproduces matrix under euler_to_matrix (Rz@Ry@Rx).
    sy = -matrix[2, 0]
    ry = math.asin(max(-1.0, min(1.0, sy)))
    rx = math.atan2(matrix[2, 1], matrix[2, 2])
    rz = math.atan2(matrix[1, 0], matrix[0, 0])
    return np.array([position[0], position[1], position[2], rx, ry, rz])


def refit(world: np.ndarray, pixel: np.ndarray, seeds: list[np.ndarray]) -> tuple[np.ndarray, float]:
    best_params, best_cost = None, 1e18
    for seed in seeds:
        try:
            result = least_squares(residuals, seed, args=(world, pixel),
                                   method="lm", xtol=1e-12, ftol=1e-12, max_nfev=20000)
        except Exception:
            continue
        rms = float(np.sqrt(np.mean(result.fun ** 2)))
        if rms < best_cost:
            best_params, best_cost = result.x, rms
    return best_params, best_cost


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", default="simulation/wheelset_workshop/cameras.json")
    parser.add_argument("--targets", default="exterior_check,cross_transfer")
    parser.add_argument("--output", default="out/visual_baseline_02/refit_probe.json")
    args = parser.parse_args()

    cameras = {c["id"]: c for c in json.loads((ROOT / args.cameras).read_text(encoding="utf-8"))}
    results = {}
    for name in [t.strip() for t in args.targets.split(",") if t.strip()]:
        camera = cameras[name]
        report = json.loads((ROOT / camera["registration_report"]).read_text(encoding="utf-8"))
        entry = next(v for v in report["views"] if v["camera"] == name)
        landmarks = entry["landmarks"]
        world = np.array([l["world_point"] for l in landmarks], dtype=np.float64)
        pixel = np.array([l["source_pixel"] for l in landmarks], dtype=np.float64)

        stored_rms = float(np.sqrt(np.mean(
            (project(np.array(look_at_params(np.array(camera["position"], dtype=np.float64),
                                             np.array(camera["target"], dtype=np.float64))), world)
             - pixel) ** 2)))

        seeds = [look_at_params(np.array(camera["position"], dtype=np.float64),
                                np.array(camera["target"], dtype=np.float64))]
        rng = np.random.default_rng(0)
        base = np.array(camera["position"], dtype=np.float64)
        for _ in range(60):
            offset = base + rng.uniform(-8, 8, 3)
            aim = np.array(camera["target"], dtype=np.float64) + rng.uniform(-6, 6, 3)
            seeds.append(look_at_params(offset, aim))

        params, rms = refit(world, pixel, seeds)
        per = None
        if params is not None:
            projected = project(params, world)
            per = [round(float(np.hypot(*(projected[i] - pixel[i]))), 2) for i in range(len(pixel))]

        results[name] = {
            "landmarks": len(landmarks),
            "stored_fit_rmse_px": round(entry.get("fit_rmse_px") or 0.0, 2),
            "recomputed_stored_rms_px": round(stored_rms, 2),
            "free_refit_rms_px": round(rms, 2),
            "per_landmark_px": per,
            "refit_pose": [round(float(v), 3) for v in params] if params is not None else None,
            "report_tolerance_px": report.get("pixel_reading_tolerance_estimate"),
        }
        print(f"{name:18s} landmarks={len(landmarks):2d}  stored={stored_rms:6.2f}  "
              f"free refit={rms:6.2f}  tol={report.get('pixel_reading_tolerance_estimate')}")
        print(f"                   per-landmark px: {per}")

    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    out = {
        "tool": "tools/wheelset_scene/refit_probe.py",
        "kind": "pose_identifiability_probe",
        "note_zh": (
            "对地标残差本身超容差的相机做自由 6 自由度重拟合。"
            "自由拟合仍高说明地标世界坐标与针孔模型不相容（数据问题）；"
            "自由拟合很低说明存储位姿陈旧（位姿问题）。"
        ),
        "focal_px": round(FOCAL, 2),
        "views": results,
    }
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
