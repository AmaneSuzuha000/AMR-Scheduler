"""Recover camera pitch from the vanishing point of parallel floor markings.

Why this exists
---------------
Every camera in the scene is pitched down 26-90 degrees. The reference frame for the
"unknown" view shows the workshop floor filling most of the frame with a horizontal
wall band across the top, which is a shallow angle, not 48 degrees. So somewhere
between the stored pose and the video there is a large registration error, and moving
29 cameras on the strength of an impression is not acceptable.

Geometry, not impression: straight markings that are parallel in the world must meet at
a single vanishing point under perspective, and the row of that point is the horizon.

    focal px = (W/2) / tan(horizontal_fov/2)
    pitch    = atan((v_row - H/2) / focal)      downward positive

First attempt (kept in the git history, reported in the round-3 note) split segments by
a crude angle threshold and intersected all pairs. It produced residuals of thousands of
pixels: the two floor directions were mixed together, so most pairs were intersections
between near-parallel lines and the median snapped to garbage.

This version is deterministic and refuses to guess:

  1. reduce every segment to its line in normal form (theta, rho)
  2. cluster the thetas - the floor grid has a few dominant directions and the rest is
     noise from panel edges and reflections
  3. per cluster, the vanishing candidate is the total-least-squares point minimising
     distance to those lines, NOT the median of pairwise intersections
  4. only report a horizon when at least MIN_LINES lines support it and the RMS
     point-to-line distance is small; otherwise say "not measured"

The pitch it reports is geometric evidence, not a registration result. It is used to
say "this camera is tens of degrees off", never to write a pose.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]

MIN_LINES = 4
MAX_RMS_PX = 6.0
THETA_BIN_DEG = 6.0


def read_image(path: Path) -> np.ndarray:
    # cv2.imread hands the name to a narrow fopen and fails on non-ASCII paths.
    raw = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"cannot read {path}")
    return image


def yellow_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    # narrow band: the floor markings are strongly saturated yellow; wider bands pull in
    # warm-lit metal and the beige floor
    mask = cv2.inRange(hsv, np.array([20, 120, 120]), np.array([38, 255, 255]))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)


def segments(mask: np.ndarray, min_length: int) -> np.ndarray:
    lines = cv2.HoughLinesP(mask, 1, np.pi / 360, threshold=50,
                            minLineLength=min_length, maxLineGap=20)
    return np.empty((0, 4)) if lines is None else lines.reshape(-1, 4).astype(float)


def to_normal_form(group: np.ndarray) -> np.ndarray:
    """(theta, rho) with x*cos(theta) + y*sin(theta) = rho, theta in [0, pi)."""
    dx = group[:, 2] - group[:, 0]
    dy = group[:, 3] - group[:, 1]
    theta = np.mod(np.arctan2(dy, dx) + np.pi / 2.0, np.pi)
    rho = group[:, 0] * np.cos(theta) + group[:, 1] * np.sin(theta)
    return np.column_stack([theta, rho])


def vanish(lines: np.ndarray) -> dict:
    """Point minimising the sum of squared distances to all lines (total least squares).

    For line i: x cos(theta_i) + y sin(theta_i) = rho_i, so solve
        [cos, sin] @ [x, y] = rho
    in the least-squares sense. That is a 2-parameter linear system, closed form, no
    iteration and no pairwise intersections that can blow up.
    """
    design = np.column_stack([np.cos(lines[:, 0]), np.sin(lines[:, 0])])
    solution, _, rank, singular = np.linalg.lstsq(design, lines[:, 1], rcond=None)
    if rank < 2:
        return {"ok": False, "reason": "degenerate_rank"}
    residual = np.abs(design @ solution - lines[:, 1])
    rms = float(np.sqrt(np.mean(residual ** 2)))
    return {
        "ok": True,
        "x": round(float(solution[0]), 2),
        "y": round(float(solution[1]), 2),
        "lines": int(len(lines)),
        "rms_px": round(rms, 3),
        "max_residual_px": round(float(residual.max()), 2),
        "condition_number": round(float(singular[0] / singular[-1]), 1) if singular[-1] > 0 else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--horizontal-fov", type=float, default=0.92)
    parser.add_argument("--output", default="")
    parser.add_argument("--debug-mask", default="")
    args = parser.parse_args()

    path = Path(args.image) if Path(args.image).is_absolute() else ROOT / args.image
    bgr = read_image(path)
    height, width = bgr.shape[:2]
    mask = yellow_mask(bgr)
    found = segments(mask, min_length=max(30, width // 20))

    if args.debug_mask:
        target = Path(args.debug_mask) if Path(args.debug_mask).is_absolute() else ROOT / args.debug_mask
        target.parent.mkdir(parents=True, exist_ok=True)
        ok, buffer = cv2.imencode(".png", mask)
        if ok:
            buffer.tofile(str(target))

    focal = (width / 2.0) / math.tan(args.horizontal_fov / 2.0)
    report = {
        "tool": "tools/wheelset_scene/horizon_solve.py",
        "kind": "geometry_derived_pitch_not_registration",
        "note_zh": (
            "用地面平行标线的消失点求俯仰角。这是几何解算，不是配准；"
            "标线不平行、镜头畸变、标线被遮挡都会带偏，所以只在残差小且支撑线足够多时才报数。"
            "它的用途是判断'这台相机的俯仰角差了多少量级'，不能直接写回相机。"
        ),
        "image": path.name,
        "image_size": [width, height],
        "horizontal_fov_rad": args.horizontal_fov,
        "focal_px": round(focal, 2),
        "thresholds": {"min_lines": MIN_LINES, "max_rms_px": MAX_RMS_PX, "theta_bin_deg": THETA_BIN_DEG},
        "segments_found": int(len(found)),
        "clusters": [],
    }

    if len(found) >= 2:
        normal = to_normal_form(found)
        theta_deg = np.degrees(normal[:, 0])
        bins = np.floor(theta_deg / THETA_BIN_DEG).astype(int)
        for key in np.unique(bins):
            members = normal[bins == key]
            if len(members) < MIN_LINES:
                continue
            result = vanish(members)
            result["theta_deg_mean"] = round(float(np.degrees(members[:, 0].mean())), 2)
            report["clusters"].append(result)

    usable = [c for c in report["clusters"]
              if c.get("ok") and c["rms_px"] <= MAX_RMS_PX and c["lines"] >= MIN_LINES]
    # The horizontal family is the one whose vanishing point sits far to the side and
    # near the horizon; a vertical-ish family converges far above/below and is useless
    # for pitch. Prefer the cluster with the most support that is not near-vertical.
    horizontal = [c for c in usable if 20.0 <= c["theta_deg_mean"] <= 160.0]
    chosen = max(horizontal or usable, key=lambda c: c["lines"]) if usable else None

    report["vanishing_point_used"] = chosen
    if chosen:
        report["horizon_row"] = chosen["y"]
        report["implied_pitch_deg"] = round(
            math.degrees(math.atan2(chosen["y"] - height / 2.0, focal)), 2
        )
    else:
        report["horizon_row"] = None
        report["implied_pitch_deg"] = None
        report["reason"] = f"no cluster passed min_lines={MIN_LINES} and rms<={MAX_RMS_PX}px"

    if args.output:
        out = Path(args.output) if Path(args.output).is_absolute() else ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps({
        "image": report["image"],
        "segments": report["segments_found"],
        "clusters": len(report["clusters"]),
        "horizon_row": report["horizon_row"],
        "implied_pitch_deg": report["implied_pitch_deg"],
        "support": (chosen or {}).get("lines"),
        "rms_px": (chosen or {}).get("rms_px"),
        "reason": report.get("reason"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
