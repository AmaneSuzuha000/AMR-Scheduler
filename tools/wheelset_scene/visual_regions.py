"""Localise visual mismatch in world coordinates.

Given a reference video frame and our render of the same registered camera, this
tool answers a concrete engineering question: *where in the workshop is our
geometry missing or oversized?*  It does not score fidelity; it produces a list of
ground-plane locations with a measured foreground deficit.

Method: both images are reduced to foreground masks (distance from the flat
backdrop colour, Otsu threshold) and then to a coarse cell grid.  For every cell
whose coverage differs by more than the tolerance, the cell centre is cast through
the pinhole camera and intersected with a horizontal plane, giving an approximate
world position.  Ray/plane intersection is only valid for cells that see the
ground; cells that look at the horizon or the roof are reported separately and
must be read as "elevated structure", not as a ground location.

A deficit means the reference shows material where our render shows backdrop.  It
can be missing equipment, equipment that is too small, or a camera that is aimed
away from the equipment.  The report cannot distinguish those three causes; the
caller must check the scene JSON at the reported coordinates before editing.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = ROOT / "visual_comparison" / "wheelset_workshop_v1"


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    prob = hist / total
    centres = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centres)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = (mu_total * omega - mu) ** 2 / denom
    sigma = np.nan_to_num(sigma, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return float(centres[int(np.argmax(sigma))])


def foreground(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate(
        [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0
    )
    bg = np.median(border, axis=0)
    distance = np.abs(rgb - bg).max(axis=2)
    return distance > otsu(distance.ravel())


def cell_coverage(mask: np.ndarray, cols: int, rows: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((rows, cols))
    for r in range(rows):
        for c in range(cols):
            block = mask[r * h // rows : (r + 1) * h // rows, c * w // cols : (c + 1) * w // cols]
            out[r, c] = block.mean()
    return out


def camera_basis(camera: dict):
    eye = np.array(camera["position"], dtype=np.float64)
    target = np.array(camera["target"], dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    norm = np.linalg.norm(right)
    if norm < 1e-9:  # camera looking straight down
        right = np.array([1.0, 0.0, 0.0])
    else:
        right /= norm
    up = np.cross(right, forward)
    return eye, forward, right, up


def ray_for_pixel(camera: dict, u: float, v: float, width: int, height: int) -> np.ndarray:
    eye, forward, right, up = camera_basis(camera)
    focal = (width / 2.0) / math.tan(camera["horizontal_fov"] / 2.0)
    x = (u - width / 2.0) / focal
    y = -(v - height / 2.0) / focal
    direction = forward + x * right + y * up
    return direction / np.linalg.norm(direction)


def ground_hit(camera: dict, u: float, v: float, width: int, height: int, plane_z: float):
    direction = ray_for_pixel(camera, u, v, width, height)
    if abs(direction[2]) < 1e-6:
        return None
    t = (plane_z - camera["position"][2]) / direction[2]
    if t <= 0 or t > 400.0:
        return None
    point = np.array(camera["position"], dtype=np.float64) + t * direction
    return [round(float(point[0]), 2), round(float(point[1]), 2), round(float(t), 2)]


def analyse_view(view_id: str, camera: dict, ref_path: Path, render_path: Path,
                 cols: int, rows: int, tolerance: float, plane_z: float) -> dict:
    ref = load_rgb(ref_path)
    render = load_rgb(render_path)
    if ref.shape != render.shape:
        render = np.asarray(
            Image.open(render_path).convert("RGB").resize((ref.shape[1], ref.shape[0]), Image.LANCZOS),
            dtype=np.float64,
        ) / 255.0
    height, width = ref.shape[:2]
    ref_cov = cell_coverage(foreground(ref), cols, rows)
    render_cov = cell_coverage(foreground(render), cols, rows)
    delta = ref_cov - render_cov

    deficits = []
    excesses = []
    for r in range(rows):
        for c in range(cols):
            d = float(delta[r, c])
            if abs(d) < tolerance:
                continue
            u = (c + 0.5) * width / cols
            v = (r + 0.5) * height / rows
            hit = ground_hit(camera, u, v, width, height, plane_z)
            entry = {
                "cell": [r, c],
                "px": [round(u, 1), round(v, 1)],
                "ref_coverage": round(float(ref_cov[r, c]), 4),
                "render_coverage": round(float(render_cov[r, c]), 4),
                "delta": round(d, 4),
                "ground_xy": hit[:2] if hit else None,
                "distance_m": hit[2] if hit else None,
            }
            if hit is None:
                entry["note"] = "ray misses ground plane: elevated structure or sky"
            (deficits if d > 0 else excesses).append(entry)

    deficits.sort(key=lambda e: -e["delta"])
    excesses.sort(key=lambda e: e["delta"])
    return {
        "id": view_id,
        "reference": str(ref_path),
        "render": str(render_path),
        "camera": {k: camera[k] for k in ("position", "target", "horizontal_fov")},
        "total_ref_coverage": round(float(ref_cov.mean()), 4),
        "total_render_coverage": round(float(render_cov.mean()), 4),
        "deficit_total": round(float(np.clip(delta, 0, None).sum()), 4),
        "excess_total": round(float(np.clip(-delta, 0, None).sum()), 4),
        "deficits": deficits,
        "excesses": excesses,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", default="out/visual_baseline_01/render")
    parser.add_argument("--cameras", default="simulation/wheelset_workshop/cameras.json")
    parser.add_argument("--output", default="out/visual_baseline_01/visual_regions.json")
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--tolerance", type=float, default=0.25)
    parser.add_argument("--plane-z", type=float, default=0.0)
    parser.add_argument("--top", type=int, default=12)
    args = parser.parse_args()

    cameras = {c["id"]: c for c in json.loads((ROOT / args.cameras).read_text(encoding="utf-8"))}
    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    render_dir = ROOT / args.render_dir

    views = []
    for view in index["views"]:
        view_id = view["id"]
        ref_path = COMPARE_ROOT / view["reference"]
        render_path = render_dir / f"{view_id}.png"
        if view_id not in cameras or not ref_path.exists() or not render_path.exists():
            continue
        result = analyse_view(
            view_id, cameras[view_id], ref_path, render_path,
            args.cols, args.rows, args.tolerance, args.plane_z,
        )
        result["name"] = view.get("name")
        result["second"] = view.get("second")
        result["deficits"] = result["deficits"][: args.top]
        result["excesses"] = result["excesses"][: args.top]
        views.append(result)

    views.sort(key=lambda v: -v["deficit_total"])
    report = {
        "tool": "tools/wheelset_scene/visual_regions.py",
        "kind": "mismatch_localisation_not_acceptance",
        "note_zh": (
            "负覆盖差表示参考画面有料、我们渲染是背景：可能是缺设备、设备偏小，"
            "也可能是相机没对准，三者本工具不能区分，须先查场景 JSON 对应坐标再改。"
            "ground_xy 为射线与水平面交点，仅对看向地面的格子有效。"
        ),
        "render_dir": args.render_dir,
        "grid": [args.rows, args.cols],
        "tolerance": args.tolerance,
        "plane_z": args.plane_z,
        "views": views,
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    summary = [
        {
            "id": v["id"],
            "deficit": v["deficit_total"],
            "ref_cov": v["total_ref_coverage"],
            "render_cov": v["total_render_coverage"],
            "worst_cells": [
                {"xy": d["ground_xy"], "delta": d["delta"]} for d in v["deficits"][:3]
            ],
        }
        for v in views[:12]
    ]
    print("REGIONS " + json.dumps({"views": len(views), "output": args.output, "worst": summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
