"""Horizontal displacement audit: the partner of shift_audit's vertical test.

region_accumulated showed the deficit and the excess sit in ADJACENT columns of the same
rows - row 7 columns 1-4 short, rows 5-6 columns 9-11 over. Summed over a whole row the
two cancel, which is why the row totals looked healthy (row 7: reference 0.323 against our
alpha 0.319) while individual cells were 0.23 off. A pair of adjacent errors that cancel is
the signature of a horizontal displacement, not of missing or extra structure.

So run the same correlation-slide test along columns:

    shift of 0        horizontal mass is where the reference puts it
    shift of +n px    our foreground sits n px to the RIGHT of the reference
    shift of -n px    our foreground sits n px to the LEFT

Consistent shift across views points at yaw or layout translation; inconsistent shift
points at per-view registration. This tool does not decide which, it measures the shift.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = ROOT / "visual_comparison" / "wheelset_workshop_v1"
MAX_SHIFT = 200

from tools.wheelset_scene.geom_score import reference_foreground, silhouette  # noqa: E402


def best_shift(ref_cols: np.ndarray, mask_cols: np.ndarray) -> tuple[int, float]:
    ref = ref_cols - ref_cols.mean()
    mask = mask_cols - mask_cols.mean()
    ref_norm = np.linalg.norm(ref)
    if ref_norm < 1e-9:
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
    parser.add_argument("--output", default="out/visual_baseline_02/shift_audit_horizontal.json")
    args = parser.parse_args()

    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    mask_dir = ROOT / args.masks

    rows = {}
    for entry in index["views"]:
        mask_path = mask_dir / f"{entry['id']}.png"
        if not mask_path.exists():
            continue
        ref = reference_foreground(COMPARE_ROOT / entry["reference"])
        mask = silhouette(mask_path)
        shift, score = best_shift(ref.sum(axis=0).astype(np.float64),
                                  mask.sum(axis=0).astype(np.float64))
        rows[entry["id"]] = {
            "best_shift_px": shift,
            "correlation": round(score, 4),
            "implied_yaw_deg": round(float(np.degrees(np.arctan(shift / 1291.76))), 2),
        }

    shifts = np.array([r["best_shift_px"] for r in rows.values()])
    report = {
        "tool": "tools/wheelset_scene/shift_audit_horizontal.py",
        "kind": "horizontal_displacement_diagnostic",
        "note_zh": (
            "把参考前景与 alpha 轮廓逐列投影后互相平移，找相关性最高的列偏移。"
            "正值表示我们偏右、负值表示偏左。位移一致指向偏航角或整场景平移，"
            "位移不一致指向逐视角配准问题。"
        ),
        "focal_px": 1291.76,
        "max_shift_searched": MAX_SHIFT,
        "summary": {
            "views": len(rows),
            "mean_shift_px": round(float(shifts.mean()), 2),
            "median_shift_px": round(float(np.median(shifts)), 2),
            "abs_mean_px": round(float(np.abs(shifts).mean()), 2),
            "right": int((shifts > 4).sum()),
            "left": int((shifts < -4).sum()),
            "near_zero": int((np.abs(shifts) <= 4).sum()),
        },
        "rows": rows,
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'view':22s} {'shift':>7s} {'yaw_deg':>8s} {'corr':>7s}")
    for view_id, row in sorted(rows.items(), key=lambda kv: kv[1]["best_shift_px"]):
        print(f"{view_id:22s} {row['best_shift_px']:7d} {row['implied_yaw_deg']:8.2f} "
              f"{row['correlation']:7.3f}")
    s = report["summary"]
    print()
    print(f"mean {s['mean_shift_px']:+.2f} px  abs mean {s['abs_mean_px']:.2f}  "
          f"right {s['right']} / left {s['left']} / flat {s['near_zero']}")


if __name__ == "__main__":
    main()
