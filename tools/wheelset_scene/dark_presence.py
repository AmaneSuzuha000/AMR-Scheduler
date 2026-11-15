"""Is the reference-dark structure ABSENT from our render, or merely mis-shaded there?

Every attempt to close the tonal gap by re-shading has failed under full-suite A/B:
albedo (-5% on the dark error), ambient fill (w1 worse in all five variants), tone curve
(w1 worse in all five), material specular (w1 worse, and the dark share did not move at
all: 1.7% -> 1.8% even at metallic 1.0).

That pattern - shading changes cannot create dark area - is what you would expect if the
dark area is not mis-shaded but MISSING. The two are trivially distinguishable, and the
test costs one render, not a sweep:

    at the pixels where the reference is dark, how different is our render from our own
    neighbourhood?

If our value there matches the local background, we drew nothing there: the structure is
missing. If our value there differs from the background but is too bright, the structure
exists and is mis-shaded, and shading is the right lever after all.

Reported per view, plus a whole-suite summary. No threshold decides the verdict; the
comparison to the local background is the evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = ROOT / "visual_comparison" / "wheelset_workshop_v1"
DARK = 0.28


def grey(rgb: np.ndarray) -> np.ndarray:
    return rgb @ np.array([0.299, 0.587, 0.114])


def graphics_mask(rgb: np.ndarray) -> np.ndarray:
    """Production graphics: blue caption banner and red magnifier ring."""
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    banner = (b > 0.65) & (r < 0.45) & (b > g + 0.08)
    ring = (r > 0.80) & (g < 0.22) & (b < 0.22)
    return banner | ring


def local_background(our: np.ndarray, window: int = 41) -> np.ndarray:
    """Median of our render in a window around each pixel.

    A 41 px median filter over 29 full-resolution frames takes minutes, which is long
    enough that the first version of this tool timed out and told nobody anything. The
    background estimate does not need full resolution - it only has to be smooth on the
    scale of the structure - so it is computed on a 4x downsample and mapped back up.
    The dark-pixel statistics themselves still use the original pixels.
    """
    from scipy.ndimage import median_filter, zoom

    small = our[::4, ::4]
    background = median_filter(small, size=max(3, window // 4), mode="nearest")
    expanded = zoom(background, 4, order=1)
    return expanded[: our.shape[0], : our.shape[1]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shaded", default="out/visual_baseline_02/shaded")
    parser.add_argument("--output", default="out/visual_baseline_02/dark_presence.json")
    args = parser.parse_args()

    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    shaded = ROOT / args.shaded

    rows = {}
    for entry in index["views"]:
        path = shaded / f"{entry['id']}.png"
        if not path.exists():
            continue
        ref = np.asarray(Image.open(COMPARE_ROOT / entry["reference"]).convert("RGB"),
                         dtype=np.float64) / 255.0
        ours = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
        ref_grey, our_grey = grey(ref), grey(ours)
        dark = (ref_grey < DARK) & ~graphics_mask(ref)
        if dark.sum() < 500:
            continue
        background = local_background(our_grey)
        contrast_at_dark = float(np.abs(our_grey[dark] - background[dark]).mean())
        elsewhere = ~dark
        contrast_elsewhere = float(np.abs(our_grey[elsewhere] - background[elsewhere]).mean())
        rows[entry["id"]] = {
            "dark_frac_ref": round(float(dark.mean()), 4),
            "our_grey_at_dark": round(float(our_grey[dark].mean()), 4),
            "our_local_background": round(float(background[dark].mean()), 4),
            "our_contrast_at_dark": round(contrast_at_dark, 4),
            "our_contrast_elsewhere": round(contrast_elsewhere, 4),
            "presence_ratio": round(contrast_at_dark / max(contrast_elsewhere, 1e-6), 3),
        }

    ratios = np.array([r["presence_ratio"] for r in rows.values()])
    report = {
        "tool": "tools/wheelset_scene/dark_presence.py",
        "kind": "missing_versus_misshaded_diagnostic",
        "note_zh": (
            "在参考帧暗像素处，比较我们的灰度与我们自己的局部背景。"
            "presence_ratio 接近 0 表示那里什么都没有画（几何缺失）；"
            "接近或超过 1 表示结构在、只是明暗不对（着色问题）。"
        ),
        "dark_threshold": DARK,
        "summary": {
            "views": len(rows),
            "mean_presence_ratio": round(float(ratios.mean()), 3),
            "median_presence_ratio": round(float(np.median(ratios)), 3),
            "views_below_0.25": int((ratios < 0.25).sum()),
            "views_above_0.75": int((ratios > 0.75).sum()),
        },
        "views": rows,
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'view':22s} {'dark%':>7s} {'ours@dark':>10s} {'bg':>7s} {'presence':>9s}")
    for view_id, row in sorted(rows.items(), key=lambda kv: kv[1]["presence_ratio"]):
        print(f"{view_id:22s} {row['dark_frac_ref'] * 100:6.1f}% {row['our_grey_at_dark']:10.3f} "
              f"{row['our_local_background']:7.3f} {row['presence_ratio']:9.3f}")
    s = report["summary"]
    print()
    print(f"median presence {s['median_presence_ratio']:.3f}   "
          f"views with almost nothing drawn: {s['views_below_0.25']} / {s['views']}")


if __name__ == "__main__":
    main()
