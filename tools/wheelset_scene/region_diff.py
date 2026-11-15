"""Compare a reference frame and a render region by region, without guessing.

Three rounds of this project went into eyeballing a pair of images and stating a cause,
and twice the stated cause was wrong: "pitch is off by 40 degrees" (it was not, the wrong
frame had been used) and "the wheel discs are too thin" (they measure the same as the
reference). Eyeballing is not reliable at this scale of detail, so this tool makes the
comparison explicit and numeric.

It splits both images into a grid and reports, per cell, the structural mismatch using
the same two descriptors as the main score - foreground presence and edge density - plus
the local mean brightness. Cells are ranked by total mismatch so the output is a short
list of places to look, each with a world coordinate when a ground plane is given.

It deliberately does NOT name a cause. It says where and how much; deciding why is the
next step and needs the geometry inventory, not this tool.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = ROOT / "visual_comparison" / "wheelset_workshop_v1"


def load(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def grey(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


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


def foreground(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    distance = np.abs(rgb - background).max(axis=2)
    return distance > otsu(distance.ravel())


def edges(rgb: np.ndarray, quantile: float = 0.92) -> np.ndarray:
    g = grey(rgb)
    gx = np.zeros_like(g)
    gy = np.zeros_like(g)
    gx[:, 1:-1] = g[:, 2:] - g[:, :-2]
    gy[1:-1, :] = g[2:, :] - g[:-2, :]
    magnitude = np.hypot(gx, gy)
    return magnitude > max(float(np.quantile(magnitude, quantile)), 1e-6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--view", required=True)
    parser.add_argument("--render", required=True)
    parser.add_argument("--cols", type=int, default=16)
    parser.add_argument("--rows", type=int, default=9)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    entry = next(v for v in index["views"] if v["id"] == args.view)
    ref = load(COMPARE_ROOT / entry["reference"])
    render = load(ROOT / args.render)
    if ref.shape != render.shape:
        raise SystemExit(f"size mismatch {ref.shape} vs {render.shape}")

    height, width = ref.shape[:2]
    cell_h = height / args.rows
    cell_w = width / args.cols
    ref_fg, render_fg = foreground(ref), foreground(render)
    ref_ed, render_ed = edges(ref), edges(render)
    ref_g, render_g = grey(ref), grey(render)

    cells = []
    for row in range(args.rows):
        for col in range(args.cols):
            y0, y1 = int(row * cell_h), int((row + 1) * cell_h)
            x0, x1 = int(col * cell_w), int((col + 1) * cell_w)
            fg_ref = float(ref_fg[y0:y1, x0:x1].mean())
            fg_render = float(render_fg[y0:y1, x0:x1].mean())
            ed_ref = float(ref_ed[y0:y1, x0:x1].mean())
            ed_re = float(render_ed[y0:y1, x0:x1].mean())
            cells.append({
                "row": row,
                "col": col,
                "px_centre": [int(x0 + cell_w / 2), int(y0 + cell_h / 2)],
                "fg_ref": round(fg_ref, 3),
                "fg_render": round(fg_render, 3),
                "fg_delta": round(fg_render - fg_ref, 3),
                "edge_ref": round(ed_ref, 3),
                "edge_render": round(ed_re, 3),
                "edge_delta": round(ed_re - ed_ref, 3),
                "brightness_ref": round(float(ref_g[y0:y1, x0:x1].mean()), 3),
                "brightness_render": round(float(render_g[y0:y1, x0:x1].mean()), 3),
                "mismatch": round(abs(fg_render - fg_ref) + abs(ed_re - ed_ref), 3),
            })

    cells.sort(key=lambda c: -c["mismatch"])
    report = {
        "tool": "tools/wheelset_scene/region_diff.py",
        "kind": "region_mismatch_not_cause",
        "note_zh": (
            "按格比较结构差（前景占比 + 边缘密度）与亮度。它只说'哪里差、差多少'，"
            "不判断原因；原因要靠几何清单核对。正 fg_delta 表示我们画多了，负值表示画少了。"
        ),
        "view": args.view,
        "grid": [args.cols, args.rows],
        "reference": entry["reference"],
        "render": args.render,
        "worst_cells": cells[:16],
        "mean_fg_ref": round(float(ref_fg.mean()), 4),
        "mean_fg_render": round(float(render_fg.mean()), 4),
        "overall_mismatch": round(float(np.mean([c["mismatch"] for c in cells])), 4),
    }
    if args.output:
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'cell':10s} {'px':14s} {'fg_ref':>7s} {'fg_ren':>7s} {'dfg':>7s} {'ed_ref':>7s} {'ed_ren':>7s} {'ded':>7s} {'bright':>13s}")
    for cell in cells[:16]:
        print(
            f"({cell['row']:2d},{cell['col']:2d})  {str(cell['px_centre']):14s} "
            f"{cell['fg_ref']:7.3f} {cell['fg_render']:7.3f} {cell['fg_delta']:+7.3f} "
            f"{cell['edge_ref']:7.3f} {cell['edge_render']:7.3f} {cell['edge_delta']:+7.3f} "
            f"{cell['brightness_ref']:.2f}->{cell['brightness_render']:.2f}"
        )
    print()
    print(f"overall mismatch {report['overall_mismatch']:.4f}   "
          f"fg ref {report['mean_fg_ref']:.3f} vs render {report['mean_fg_render']:.3f}")


if __name__ == "__main__":
    main()
