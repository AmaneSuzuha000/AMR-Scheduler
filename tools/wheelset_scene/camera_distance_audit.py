"""Audit camera distance using true silhouettes instead of a brightness proxy.

This supersedes the first version of this tool. That version compared the reference
frame's foreground fraction (derived by border-colour difference + Otsu, the only
thing available for a video still) against the *render's* foreground fraction derived
the same way. The render side of that comparison was invalid: changing only the
colour management moved inspection's measured foreground from 0.222 to 0.819 with the
geometry untouched. It produced confident, wrong advice - it told us to dolly
press_front in from 10 m to 4 m, when the true silhouette shows the render already
covers 1.31x the reference's structure and should back off to about 11.4 m.

Now the render side is the alpha channel of a film_transparent pass with the ground
slab hidden (render_masks.py), which depends only on cameras and geometry.

For a subject seen against a floor, covered screen area scales roughly as 1/d^2, so:
    required_distance = stored_distance * sqrt(coverage_ratio)
with coverage_ratio = our silhouette fraction / reference foreground fraction.
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
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = (mu_total * omega - mu) ** 2 / (omega * (1.0 - omega))
    return float(centres[int(np.argmax(np.nan_to_num(sigma, nan=-1.0)))])


def reference_fg(path: Path) -> float:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    distance = np.abs(rgb - background).max(axis=2)
    return float((distance > otsu(distance.ravel())).mean())


def mask_fg(path: Path) -> float:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0
    return float((rgba[:, :, 3] > 0.5).mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", default="simulation/wheelset_workshop/cameras.json")
    parser.add_argument("--masks", default="out/visual_baseline_01/masks")
    parser.add_argument("--output", default="out/visual_baseline_01/camera_distance_audit.json")
    args = parser.parse_args()

    cameras = json.loads((ROOT / args.cameras).read_text(encoding="utf-8"))
    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    reference = {v["id"]: v["reference"] for v in index["views"]}
    mask_dir = ROOT / args.masks

    rows = []
    for camera in cameras:
        view_id = camera["id"]
        distance = math.dist(camera["position"], camera["target"])
        entry = {
            "id": view_id,
            "registration": camera.get("registration"),
            "stored_distance_m": round(distance, 3),
            "camera_z_m": round(camera["position"][2], 2),
        }
        if view_id in reference and (mask_dir / f"{view_id}.png").exists():
            ref = reference_fg(COMPARE_ROOT / reference[view_id])
            mask = mask_fg(mask_dir / f"{view_id}.png")
            entry["reference_fg"] = round(ref, 4)
            entry["mask_fg"] = round(mask, 4)
            if ref > 1e-6:
                ratio = mask / ref
                entry["coverage_ratio"] = round(ratio, 3)
                implied = distance * math.sqrt(ratio)
                entry["implied_distance_m"] = round(implied, 3)
                entry["distance_error_m"] = round(distance - implied, 3)
        rows.append(entry)

    distances = [r["stored_distance_m"] for r in rows]
    template = sum(1 for d in distances if abs(d - 10.0) < 0.05)
    report = {
        "tool": "tools/wheelset_scene/camera_distance_audit.py",
        "kind": "camera_distance_estimate_not_acceptance",
        "note_zh": (
            "implied_distance 是'让我们的轮廓面积等于参考前景面积'的一阶反解，"
            "不是配准结果。它受遮挡与透视影响，只能排序'哪个镜头明显站错距离'；"
            "写回相机前必须用 fit_camera.py 的地标拟合确认。"
        ),
        "camera_count": len(rows),
        "at_exactly_10m": template,
        "rows": rows,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    ranked = sorted(
        [r for r in rows if "distance_error_m" in r],
        key=lambda r: -abs(r["distance_error_m"]),
    )
    print(f"{'view':22s} {'stored':>7s} {'ref_fg':>7s} {'mask_fg':>8s} {'ratio':>6s} {'implied':>8s} {'err':>7s}")
    for row in ranked:
        print(
            f"{row['id']:22s} {row['stored_distance_m']:7.2f} {row['reference_fg']:7.3f} "
            f"{row['mask_fg']:8.3f} {row['coverage_ratio']:6.2f} {row['implied_distance_m']:8.2f} "
            f"{row['distance_error_m']:+7.2f}"
        )
    print(f"\ncameras at exactly 10.00 m: {template}/{len(rows)}")


if __name__ == "__main__":
    main()
