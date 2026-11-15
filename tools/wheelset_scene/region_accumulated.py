"""Accumulate the region diff over every view to find systematic, not local, gaps.

Single-view region_diff answers "where is this frame wrong". Twenty-nine of those answers
are mostly noise: each view has its own occlusion, its own caption, its own partial
equipment. Deciding a geometry change from one view is how a change improves one score and
wrecks eleven - which has already happened three times in this project.

Averaging the per-cell mismatch over all views answers a different and more useful
question: "is there anywhere in the frame that is wrong in the SAME way, in most views?"
A cell that is short of structure in 20 of 29 views is a systematic gap; a cell that is
short in 3 views is that view's occlusion.

Output is the per-cell mean signed foreground error and the count of views voting each
way. Signed, not absolute: mean_fg_delta < 0 means we are missing structure there.
"""
from __future__ import annotations

import argparse
import json
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
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = (mu[-1] * omega - mu) ** 2 / (omega * (1.0 - omega))
    return float(centres[int(np.argmax(np.nan_to_num(sigma, nan=-1.0)))])


def reference_foreground(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    distance = np.abs(rgb - background).max(axis=2)
    return distance > otsu(distance.ravel())


def silhouette(path: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0
    return rgba[:, :, 3] > 0.5


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", default="out/visual_baseline_02/masks")
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--output", default="out/visual_baseline_02/region_accumulated.json")
    args = parser.parse_args()

    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    mask_dir = ROOT / args.masks

    deltas = np.zeros((args.rows, args.cols))
    votes_deficit = np.zeros((args.rows, args.cols), dtype=int)
    votes_excess = np.zeros((args.rows, args.cols), dtype=int)
    used = 0
    for entry in index["views"]:
        mask_path = mask_dir / f"{entry['id']}.png"
        if not mask_path.exists():
            continue
        ref = reference_foreground(COMPARE_ROOT / entry["reference"])
        mask = silhouette(mask_path)
        height, width = ref.shape
        used += 1
        for row in range(args.rows):
            y0, y1 = int(row * height / args.rows), int((row + 1) * height / args.rows)
            for col in range(args.cols):
                x0, x1 = int(col * width / args.cols), int((col + 1) * width / args.cols)
                delta = float(mask[y0:y1, x0:x1].mean() - ref[y0:y1, x0:x1].mean())
                deltas[row, col] += delta
                if delta < -0.15:
                    votes_deficit[row, col] += 1
                elif delta > 0.15:
                    votes_excess[row, col] += 1

    mean_delta = deltas / max(used, 1)
    cells = []
    for row in range(args.rows):
        for col in range(args.cols):
            cells.append({
                "row": row,
                "col": col,
                "px_centre": [int((col + 0.5) * 1280 / args.cols),
                              int((row + 0.5) * 720 / args.rows)],
                "mean_fg_delta": round(float(mean_delta[row, col]), 4),
                "views_deficit": int(votes_deficit[row, col]),
                "views_excess": int(votes_excess[row, col]),
            })
    cells.sort(key=lambda c: c["mean_fg_delta"])
    report = {
        "tool": "tools/wheelset_scene/region_accumulated.py",
        "kind": "systematic_region_gap",
        "note_zh": (
            "把逐格前景差在 29 个视角上取平均。mean_fg_delta<0 表示这里普遍缺少结构；"
            "views_deficit 是投'缺'票的视角数，比单帧差异更能说明是否系统性问题。"
        ),
        "views_used": used,
        "grid": [args.cols, args.rows],
        "most_deficit": cells[:12],
        "most_excess": cells[-8:][::-1],
        "rows": cells,
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"views used {used}")
    print(f"{'cell':10s} {'px':14s} {'mean_dfg':>9s} {'deficit_v':>10s} {'excess_v':>9s}")
    for cell in cells[:12]:
        print(f"({cell['row']:2d},{cell['col']:2d})  {str(cell['px_centre']):14s} "
              f"{cell['mean_fg_delta']:+9.4f} {cell['views_deficit']:10d} {cell['views_excess']:9d}")
    print("  ... most excess ...")
    for cell in cells[-8:][::-1]:
        print(f"({cell['row']:2d},{cell['col']:2d})  {str(cell['px_centre']):14s} "
              f"{cell['mean_fg_delta']:+9.4f} {cell['views_deficit']:10d} {cell['views_excess']:9d}")


if __name__ == "__main__":
    main()
