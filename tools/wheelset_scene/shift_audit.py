"""Is the silhouette vertically displaced? A shift test that names the camera, not the model.

IoU alone cannot tell "the structure is the wrong shape" from "the structure is in the
right shape but the camera is pointed slightly wrong". Those need opposite fixes, and the
attribution cross-tab can only rule out cameras whose fit report failed - it says nothing
when a camera's landmarks passed but its height or pitch is still off.

The cheap decisive test: build the row profile of the foreground (how much of each image
row is occupied) for the reference and for the alpha silhouette, then slide one against
the other and report the offset that maximises normalised correlation.

    shift of 0        the vertical mass is where the reference puts it
    shift of +n rows  our foreground sits n rows LOWER than the reference
    shift of -n rows  our foreground sits n rows HIGHER

For a camera looking down, a shifted horizon moves every ground feature the same way, so a
large consistent shift points at pitch or height. An inconsistent shift across views
points at the model instead.

Reported per view, plus the distribution across all views. No threshold decides anything
here: the number is the evidence, and it is only meaningful combined with the fit report's
own residual (registration_gate.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = ROOT / "visual_comparison" / "wheelset_workshop_v1"
MAX_SHIFT = 160


def reference_foreground(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    distance = np.abs(rgb - background).max(axis=2)
    hist, edges = np.histogram(distance.ravel(), bins=256, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    centres = (edges[:-1] + edges[1:]) / 2.0
    prob = hist / max(hist.sum(), 1)
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centres)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    threshold = float(centres[int(np.argmax(np.nan_to_num(sigma, nan=-1.0)))])
    return distance > threshold


def silhouette(path: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0
    return rgba[:, :, 3] > 0.5


def best_shift(ref_rows: np.ndarray, mask_rows: np.ndarray) -> tuple[int, float]:
    ref = ref_rows - ref_rows.mean()
    mask = mask_rows - mask_rows.mean()
    ref_norm = np.linalg.norm(ref)
    mask_norm = np.linalg.norm(mask)
    if ref_norm < 1e-9 or mask_norm < 1e-9:
        return 0, 0.0
    best = (0, -2.0)
    for shift in range(-MAX_SHIFT, MAX_SHIFT + 1):
        moved = np.roll(mask, shift)
        if shift > 0:
            moved[:shift] = 0.0
        elif shift < 0:
            moved[shift:] = 0.0
        denominator = ref_norm * np.linalg.norm(moved)
        if denominator < 1e-9:
            continue
        score = float(np.dot(ref, moved) / denominator)
        if score > best[1]:
            best = (shift, score)
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", default="out/visual_baseline_02/masks")
    parser.add_argument("--output", default="out/visual_baseline_02/shift_audit.json")
    args = parser.parse_args()

    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    mask_dir = ROOT / args.masks

    rows = {}
    for entry in index["views"]:
        view_id = entry["id"]
        mask_path = mask_dir / f"{view_id}.png"
        if not mask_path.exists():
            continue
        ref = reference_foreground(COMPARE_ROOT / entry["reference"])
        mask = silhouette(mask_path)
        shift, score = best_shift(ref.sum(axis=1).astype(np.float64),
                                  mask.sum(axis=1).astype(np.float64))
        rows[view_id] = {
            "best_shift_rows": shift,
            "correlation": round(score, 4),
            "implied_deg": round(float(np.degrees(np.arctan(shift / 1291.76))), 2),
            "ref_fg_frac": round(float(ref.mean()), 4),
            "mask_fg_frac": round(float(mask.mean()), 4),
        }

    shifts = np.array([r["best_shift_rows"] for r in rows.values()])
    report = {
        "tool": "tools/wheelset_scene/shift_audit.py",
        "kind": "vertical_displacement_diagnostic",
        "note_zh": (
            "把参考前景与 alpha 轮廓逐行投影后互相平移，找相关性最高的行偏移。"
            "正值表示我们比参考低、负值表示更高。位移一致指向相机俯仰/高度，"
            "位移不一致指向模型本身。这里不设阈值判定，只给出数值。"
        ),
        "focal_px": 1291.76,
        "max_shift_searched": MAX_SHIFT,
        "summary": {
            "views": len(rows),
            "mean_shift_rows": round(float(shifts.mean()), 2),
            "median_shift_rows": round(float(np.median(shifts)), 2),
            "abs_mean_rows": round(float(np.abs(shifts).mean()), 2),
            "positive": int((shifts > 3).sum()),
            "negative": int((shifts < -3).sum()),
            "near_zero": int((np.abs(shifts) <= 3).sum()),
            "implied_deg_mean": round(float(np.degrees(np.arctan(shifts.mean() / 1291.76))), 2),
        },
        "rows": rows,
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'view':22s} {'shift':>7s} {'deg':>7s} {'corr':>7s}")
    for view_id, row in sorted(rows.items(), key=lambda kv: kv[1]["best_shift_rows"]):
        print(f"{view_id:22s} {row['best_shift_rows']:7d} {row['implied_deg']:7.2f} "
              f"{row['correlation']:7.3f}")
    s = report["summary"]
    print()
    print(f"mean {s['mean_shift_rows']:+.2f} rows ({s['implied_deg_mean']:+.2f} deg)   "
          f"abs mean {s['abs_mean_rows']:.2f}   "
          f"down {s['positive']} / up {s['negative']} / flat {s['near_zero']}")


if __name__ == "__main__":
    main()
