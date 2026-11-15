"""Photometric distance that cannot be fooled by a shifting threshold.

The geometric score (geom_score.py) is blind to lighting by construction: alpha does not
change when the sun changes. So lighting needs its own metric, and the old one -
"border-colour difference, then Otsu" - is not usable: it moved inspection's measured
foreground from 0.222 to 0.819 when only the colour management changed.

This compares the full grey-level distribution instead of a binary mask. The
1-dimensional Wasserstein distance (equivalently, the L1 distance between cumulative
histograms) is stable, has units of brightness, and has no threshold to slide:

    W1 = integral | CDF_reference(g) - CDF_render(g) | dg   over g in [0,1]

It reports, per view and on average:
  w1           0 = identical tonal distribution, 1 = disjoint
  dark_share   share of pixels below 72% of the frame median, a crude shadow proxy
  dark_error   render dark share minus reference dark share (signed)

Measured baselines on 2026-11-11 over 29 views: reference dark share 0.2478,
legacy 0.1237 (too flat), contrast 0.2535 (matches the mean but is bimodal - far too
dark on run_in, still too flat on storage).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = ROOT / "visual_comparison" / "wheelset_workshop_v1"
BINS = 256


def grey(path: Path) -> np.ndarray:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return rgb @ np.array([0.299, 0.587, 0.114])


def cdf(values: np.ndarray) -> np.ndarray:
    hist, _ = np.histogram(values, bins=BINS, range=(0.0, 1.0))
    return np.cumsum(hist.astype(np.float64)) / max(values.size, 1)


def dark_share(values: np.ndarray) -> float:
    return float((values < np.median(values) * 0.72).mean())


def compare(ref_path: Path, render_path: Path) -> dict:
    ref = grey(ref_path)
    render = grey(render_path)
    # W1 in brightness units: sum of |CDF difference| times the bin width
    w1 = float(np.abs(cdf(ref) - cdf(render)).sum() / BINS)
    return {
        "w1": round(w1, 4),
        "ref_dark_share": round(dark_share(ref), 4),
        "render_dark_share": round(dark_share(render), 4),
        "dark_error": round(dark_share(render) - dark_share(ref), 4),
        "ref_mean": round(float(ref.mean()), 4),
        "render_mean": round(float(render.mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", dest="passes", action="append", required=True, help="label=dir")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    reference = {v["id"]: v["reference"] for v in index["views"]}

    report = {"tool": "tools/wheelset_scene/photo_score.py",
              "kind": "tonal_distribution_distance_not_acceptance",
              "note_zh": "Wasserstein 距离衡量整幅灰度分布差异，没有可滑动的阈值；"
                         "它只说明'明暗分布像不像'，不说明几何对不对，两者必须分开看。",
              "passes": {}}
    for spec in args.passes:
        label, _, directory = spec.partition("=")
        rows = {}
        for view_id, ref_rel in reference.items():
            render_path = ROOT / directory / f"{view_id}.png"
            if not render_path.exists():
                continue
            rows[view_id] = compare(COMPARE_ROOT / ref_rel, render_path)
        if not rows:
            continue
        w1 = np.array([r["w1"] for r in rows.values()])
        err = np.array([r["dark_error"] for r in rows.values()])
        report["passes"][label] = {
            "dir": directory,
            "views": len(rows),
            "w1_mean": round(float(w1.mean()), 4),
            "w1_median": round(float(np.median(w1)), 4),
            "dark_error_mean": round(float(err.mean()), 4),
            "dark_error_abs_mean": round(float(np.abs(err).mean()), 4),
            "rows": rows,
        }
        print(f"{label:20s} views={len(rows):3d}  w1 mean={w1.mean():.4f} "
              f"median={np.median(w1):.4f}  dark_error mean={err.mean():+.4f} "
              f"abs={np.abs(err).mean():.4f}")

    if args.output:
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
