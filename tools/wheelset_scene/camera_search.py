"""Search camera pose only, to separate "wrong camera" from "wrong geometry".

The scene JSON stores every camera as an initial estimate.  When a view scores
badly we cannot tell whether the equipment is in the wrong place or the camera is
aimed at the wrong place.  This tool answers that question without touching any
geometry: it renders a small parameter sweep around the stored pose at low
resolution and reports which (if any) nearby pose scores materially better.

A large gain means the stored pose is the dominant error and geometry can stay.
A flat response means moving the camera does not help, so the deficit really is
missing or mis-sized geometry.

Results are ranking evidence only.  A pose that wins the sweep is NOT adopted
automatically: it has to be reviewed, because a pose can win by pointing at a
convenient coincidence (for example filling the frame with floor).
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = ROOT / "visual_comparison" / "wheelset_workshop_v1"
BLEND = ROOT / "assets" / "wheelset_workshop" / "source" / "wheelset_workshop.blend"


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def otsu(values: np.ndarray, bins: int = 256) -> float:
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    if hist.sum() == 0:
        return 0.5
    prob = hist / hist.sum()
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
    border = np.concatenate([rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0)
    bg = np.median(border, axis=0)
    distance = np.abs(rgb - bg).max(axis=2)
    return distance > otsu(distance.ravel())


def edges(rgb: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    grey = rgb @ np.array([0.299, 0.587, 0.114])
    gx = np.zeros_like(grey)
    gy = np.zeros_like(grey)
    gx[:, 1:-1] = grey[:, 2:] - grey[:, :-2]
    gy[1:-1, :] = grey[2:, :] - grey[:-2, :]
    magnitude = np.hypot(gx, gy)
    return magnitude > max(float(np.quantile(magnitude, quantile)), 1e-6)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    return 1.0 if union == 0 else float(np.logical_and(a, b).sum() / union)


def score(ref_path: Path, render_path: Path) -> dict:
    ref = load_rgb(ref_path)
    render = load_rgb(render_path)
    if ref.shape != render.shape:
        render = np.asarray(
            Image.open(render_path).convert("RGB").resize((ref.shape[1], ref.shape[0]), Image.LANCZOS),
            dtype=np.float64,
        ) / 255.0
    ref_mask = foreground(ref)
    render_mask = foreground(render)
    silhouette = iou(ref_mask, render_mask)
    edge = iou(edges(ref), edges(render))
    # Coverage guard. A camera can "win" an IoU sweep by pointing away from the
    # equipment: the intersection shrinks but so does the union, and a near-empty
    # render can score respectably. fg_frac is reported next to the score so that a
    # gain obtained by shrinking the subject is visible rather than silently trusted.
    ref_frac = float(ref_mask.mean())
    render_frac = float(render_mask.mean())
    coverage_ratio = render_frac / ref_frac if ref_frac > 1e-6 else None
    return {
        "silhouette_iou": round(silhouette, 4),
        "edge_iou": round(edge, 4),
        "score": round(0.6 * silhouette + 0.4 * edge, 4),
        "ref_fg_frac": round(ref_frac, 4),
        "render_fg_frac": round(render_frac, 4),
        "coverage_ratio": round(coverage_ratio, 4) if coverage_ratio is not None else None,
    }


def look_rotation(position, target):
    """Blender -Z forward, Y up, expressed as a unit basis (numpy)."""
    forward = np.array(target, dtype=np.float64) - np.array(position, dtype=np.float64)
    forward /= np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    if np.linalg.norm(right) < 1e-9:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    return forward, right


def sweep(camera: dict, radius: float, angle_deg: float, zoom: float) -> list[dict]:
    """Generate candidate poses around the stored one: orbit, dolly, and aim shifts."""
    position = np.array(camera["position"], dtype=np.float64)
    target = np.array(camera["target"], dtype=np.float64)
    forward = target - position
    distance = np.linalg.norm(forward)
    forward /= distance
    world_up = np.array([0.0, 0.0, 1.0])
    right = np.cross(forward, world_up)
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)

    candidates = []
    offsets = [0.0, 1.0, -1.0]
    for dz, dy, dx in itertools.product(offsets, offsets, offsets):
        if dz == 0 and dy == 0 and dx == 0:
            continue
        # dolly along the view direction, truck along right, pedestal along up
        moved = position + forward * (dz * radius) + right * (dx * radius) + up * (dy * radius)
        candidates.append({"kind": "translate", "delta": [dz, dx, dy], "position": moved.tolist(), "target": target.tolist()})

    for axis, name in ((right, "yaw"), (up, "pitch")):
        for sign in (1.0, -1.0):
            radians = math.radians(angle_deg) * sign
            rotated = (target - position) * math.cos(radians) + np.cross(axis, target - position) * math.sin(radians)
            candidates.append({"kind": name, "sign": sign, "position": position.tolist(), "target": (position + rotated).tolist()})

    for factor in (zoom, 1.0 / zoom):
        if abs(factor - 1.0) < 1e-6:
            continue
        candidates.append({"kind": "fov", "factor": factor, "position": position.tolist(),
                           "target": target.tolist(), "fov": camera["horizontal_fov"] * factor})
    return candidates


def run_blender(job_path: Path, out_dir: Path) -> dict:
    blender = Path(r"E:\Software\Blender\blender.exe")
    script = ROOT / "tools" / "wheelset_scene" / "render_cameras.py"
    completed = subprocess.run(
        [str(blender), "--background", "--factory-startup", "--python", str(script), "--", str(job_path)],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=1800,
    )
    manifest_path = out_dir / "manifest.json"
    if completed.returncode != 0 or not manifest_path.exists():
        tail = (completed.stderr or completed.stdout or "")[-1200:]
        raise RuntimeError(f"blender rc={completed.returncode} manifest={manifest_path.exists()}: {tail}")
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True)
    parser.add_argument("--radius", type=float, default=3.0)
    parser.add_argument("--angle", type=float, default=8.0)
    parser.add_argument("--zoom", type=float, default=1.25)
    parser.add_argument("--resolution", type=int, default=480)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    cameras = {c["id"]: c for c in json.loads(
        (ROOT / "simulation/wheelset_workshop/cameras.json").read_text(encoding="utf-8"))}
    if args.view not in cameras:
        raise SystemExit(f"unknown view {args.view}")
    camera = cameras[args.view]
    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    view = next(v for v in index["views"] if v["id"] == args.view)
    ref_path = COMPARE_ROOT / view["reference"]

    work = ROOT / "out" / "camera_search" / args.view
    work.mkdir(parents=True, exist_ok=True)
    candidates = sweep(camera, args.radius, args.angle, args.zoom)

    specs = [{"id": "base", "position": camera["position"], "target": camera["target"],
              "fov": camera["horizontal_fov"]}]
    for number, candidate in enumerate(candidates):
        specs.append({
            "id": f"c{number:02d}",
            "position": candidate["position"],
            "target": candidate["target"],
            "fov": candidate.get("fov", camera["horizontal_fov"]),
        })
    job = {
        "blend": str(BLEND),
        "out_dir": str(work / "render"),
        "engine": "BLENDER_EEVEE",
        "resolution": [args.resolution, int(round(args.resolution * 9 / 16))],
        "samples": 8,
        "cameras": specs,
    }
    job_path = work / "job.json"
    job_path.write_text(json.dumps(job, ensure_ascii=False, indent=1), encoding="utf-8")
    manifest = run_blender(job_path, work / "render")

    rows = []
    for spec, candidate in zip(specs, [None] + candidates):
        path = work / "render" / f"{spec['id']}.png"
        if not path.exists():
            continue
        entry = {"id": spec["id"], "pose": spec, "candidate": candidate, **score(ref_path, path)}
        rows.append(entry)
    rows.sort(key=lambda r: -r["score"])

    baseline = next((r for r in rows if r["id"] == "base"), None)
    report = {
        "tool": "tools/wheelset_scene/camera_search.py",
        "kind": "camera_pose_ranking_not_acceptance",
        "view": args.view,
        "reference": str(ref_path),
        "radius_m": args.radius,
        "angle_deg": args.angle,
        "zoom": args.zoom,
        "rendered": manifest["rendered"],
        "baseline": baseline,
        "best": rows[0] if rows else None,
        "coverage_note_zh": (
            "coverage_ratio 远小于 1 表示参考画面里前景占满、我们渲染几乎是空场："
            "此时分数高也不能采纳机位，缺的是设备尺寸或近景构图，不是相机。"
        ),
        "gain": round(rows[0]["score"] - baseline["score"], 4) if rows and baseline else None,
        "rows": rows,
    }
    out_path = ROOT / (args.output or f"out/camera_search/{args.view}_search.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print("CAMSEARCH " + json.dumps({
        "view": args.view,
        "baseline": baseline["score"] if baseline else None,
        "best": rows[0]["score"] if rows else None,
        "gain": report["gain"],
        "best_kind": rows[0]["candidate"]["kind"] if rows and rows[0]["candidate"] else "base",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
