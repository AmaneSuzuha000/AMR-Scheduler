"""Geometric fidelity score built on true silhouettes, not on a brightness proxy.

Why this replaces the earlier score
-----------------------------------
The first version of this project's visual score derived both the reference and the
render foreground by "distance from the median border colour, then Otsu". For the
reference frames that is the only thing available, so the reference side keeps it.
For the render side it was wrong, and measurably so: changing nothing but the colour
management moved inspection's render foreground from 0.222 to 0.819. The Otsu
threshold had slid onto a background gradient. A metric whose reading changes when
only the shading changes cannot be used to compare render settings or to claim that
geometry improved.

The render side now uses the alpha channel of a film_transparent pass (see
render_masks.py). Alpha is the exact set of pixels where geometry was hit; brightness
cannot move it. The ground slab is excluded so that the two masks mean the same
thing: built structure.

Reported per view
-----------------
  ref_fg, mask_fg   foreground fractions
  coverage_ratio    mask_fg / ref_fg; ~1 means "we have about as much structure"
  geom_iou          IoU(reference foreground, our silhouette)
  mask_recall       share of our silhouette that lands on reference structure
  ref_recall        share of reference structure we cover

The last two separate the two failure modes that IoU alone conflates: drawing
structure the video does not have (low mask_recall) versus missing structure the
video does have (low ref_recall).
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
    mu_total = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = (mu_total * omega - mu) ** 2 / (omega * (1.0 - omega))
    return float(centres[int(np.argmax(np.nan_to_num(sigma, nan=-1.0)))])


def production_graphics(rgb: np.ndarray) -> np.ndarray:
    """Pixels that come from the video's own post-production, not from the scene.

    25 of the 29 reference frames carry a caption banner ("轮对复探转轮器" and similar)
    and one carries a red magnifier inset. A camera can never reproduce these, so
    counting them as reference foreground charges the model for something no geometry
    change can fix. Measured share of reference foreground, before exclusion: 2% to 12%
    per view, so it is a constant tax on every IoU in the baseline.

    The colour test has to separate the banner from the cyan beds, which are real scene
    geometry and sit right where a loose threshold would delete them:

        banner  azure    b > g + 0.08   (blue clearly dominates green)
        beds    cyan     g ~ b          (0.71 vs 0.69 in the palette)

    Measured on frame_21s (sorting): 42132 banner px against 368 cyan px, and on
    frame_33s (reinspection) the detector boxes rows 570-659, exactly the caption.
    """
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    banner = (blue > 0.65) & (red < 0.45) & (blue > green + 0.08)
    # the magnifier ring in frame_33s: saturated red, far from any palette entry
    # (the only reds in the palette are 0.78/0.055/0.025 and 0.57/0.14/0.022, both
    # darker and less saturated than the graphic's outline)
    ring = (red > 0.80) & (green < 0.22) & (blue < 0.22)
    return banner | ring


def reference_foreground(path: Path, exclude_graphics: bool = True) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    distance = np.abs(rgb - background).max(axis=2)
    foreground = distance > otsu(distance.ravel())
    if exclude_graphics:
        foreground = np.logical_and(foreground, np.logical_not(production_graphics(rgb)))
    return foreground


def silhouette(path: Path) -> np.ndarray:
    rgba = np.asarray(Image.open(path).convert("RGBA"), dtype=np.float64) / 255.0
    return rgba[:, :, 3] > 0.5


def score(ref_path: Path, mask_path: Path, exclude_graphics: bool = True) -> dict:
    ref = reference_foreground(ref_path, exclude_graphics=exclude_graphics)
    mask = silhouette(mask_path)
    intersection = np.logical_and(ref, mask).sum()
    union = np.logical_or(ref, mask).sum()
    ref_fg = float(ref.mean())
    mask_fg = float(mask.mean())
    return {
        "ref_fg": round(ref_fg, 4),
        "mask_fg": round(mask_fg, 4),
        "coverage_ratio": round(mask_fg / ref_fg, 3) if ref_fg > 1e-6 else None,
        "geom_iou": round(float(intersection / union), 4) if union else 0.0,
        "mask_recall": round(float(intersection / mask.sum()), 4) if mask.sum() else 0.0,
        "ref_recall": round(float(intersection / ref.sum()), 4) if ref.sum() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--masks", default="out/visual_baseline_01/masks")
    parser.add_argument("--output", default="out/visual_baseline_01/geom_score.json")
    parser.add_argument("--views", default="")
    args = parser.parse_args()

    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    wanted = {v.strip() for v in args.views.split(",") if v.strip()}
    mask_dir = ROOT / args.masks

    rows = {}
    for entry in index["views"]:
        view_id = entry["id"]
        if wanted and view_id not in wanted:
            continue
        mask_path = mask_dir / f"{view_id}.png"
        if not mask_path.exists():
            rows[view_id] = {"missing": True}
            continue
        rows[view_id] = score(COMPARE_ROOT / entry["reference"], mask_path)

    scored = {k: v for k, v in rows.items() if "geom_iou" in v}
    ious = np.array([v["geom_iou"] for v in scored.values()])
    ratios = np.array([v["coverage_ratio"] for v in scored.values() if v["coverage_ratio"]])
    report = {
        "tool": "tools/wheelset_scene/geom_score.py",
        "kind": "geometric_fidelity_not_photometric",
        "note_zh": (
            "geom_iou 用真实 alpha 轮廓，不受亮度/色彩管理影响，因此能用来判断几何是否真的改好了。"
            "它仍然只衡量'结构落在画面的哪里'，不衡量尺寸是否正确、机构是否可动。"
        ),
        "mask_dir": str(mask_dir),
        "views_scored": len(scored),
        "geom_iou_mean": round(float(ious.mean()), 4) if ious.size else None,
        "geom_iou_median": round(float(np.median(ious)), 4) if ious.size else None,
        "coverage_ratio_mean": round(float(ratios.mean()), 3) if ratios.size else None,
        "rows": rows,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'view':22s} {'ref':>6s} {'mask':>6s} {'ratio':>6s} {'iou':>7s} {'maskRec':>8s} {'refRec':>7s}")
    for view_id, row in sorted(scored.items(), key=lambda kv: kv[1]["geom_iou"]):
        print(
            f"{view_id:22s} {row['ref_fg']:6.3f} {row['mask_fg']:6.3f} "
            f"{row['coverage_ratio']:6.2f} {row['geom_iou']:7.4f} "
            f"{row['mask_recall']:8.3f} {row['ref_recall']:7.3f}"
        )
    print()
    print(f"views {report['views_scored']}  geom_iou mean {report['geom_iou_mean']:.4f}  "
          f"median {report['geom_iou_median']:.4f}  coverage ratio {report['coverage_ratio_mean']:.2f}")


if __name__ == "__main__":
    main()
