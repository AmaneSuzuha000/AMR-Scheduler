"""Score one or more render passes against the reference frames.

Thin wrapper around the same metric used by visual_score.py, but it takes an
explicit list of (label, render_dir) pairs so a sweep can be scored in one call and
printed as a table. Kept separate from visual_score.py because that one writes the
canonical baseline report and must stay stable.
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
    return float(centres[int(np.argmax(np.nan_to_num(sigma, nan=-1.0)))])


def foreground(rgb: np.ndarray) -> np.ndarray:
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    bg = np.median(border, axis=0)
    distance = np.abs(rgb - bg).max(axis=2)
    return distance > otsu(distance.ravel())


def edge_mask(rgb: np.ndarray, quantile: float = 0.90) -> np.ndarray:
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


def score_pair(ref_path: Path, render_path: Path) -> dict:
    ref = load(ref_path)
    render = load(render_path)
    if ref.shape != render.shape:
        render = np.asarray(
            Image.open(render_path).convert("RGB").resize((ref.shape[1], ref.shape[0]), Image.LANCZOS),
            dtype=np.float64,
        ) / 255.0
    ref_mask = foreground(ref)
    render_mask = foreground(render)
    silhouette = iou(ref_mask, render_mask)
    edge = iou(edge_mask(ref), edge_mask(render))
    return {
        "silhouette_iou": round(silhouette, 4),
        "edge_iou": round(edge, 4),
        "score": round(0.6 * silhouette + 0.4 * edge, 4),
        "ref_fg_frac": round(float(ref_mask.mean()), 4),
        "render_fg_frac": round(float(render_mask.mean()), 4),
        "render_std": round(float(render.std()), 4),
        "render_mean": round(float(render.mean()), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pass", dest="passes", action="append", required=True,
                        help="label=dir, repeatable")
    parser.add_argument("--views", default="", help="comma separated view ids, default all")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    views = [v["id"] for v in index["views"]]
    if args.views:
        wanted = {v.strip() for v in args.views.split(",") if v.strip()}
        views = [v for v in views if v in wanted]
    reference = {v["id"]: v["reference"] for v in index["views"]}

    passes = []
    for spec in args.passes:
        label, _, directory = spec.partition("=")
        passes.append((label, ROOT / directory))

    report = {"views": views, "passes": {}}
    for label, directory in passes:
        rows = {}
        for view_id in views:
            ref_path = COMPARE_ROOT / reference[view_id]
            render_path = directory / f"{view_id}.png"
            if not render_path.exists():
                rows[view_id] = {"missing": True}
                continue
            rows[view_id] = score_pair(ref_path, render_path)
        scored = [r["score"] for r in rows.values() if "score" in r]
        report["passes"][label] = {
            "dir": str(directory),
            "mean": round(float(np.mean(scored)), 4) if scored else None,
            "median": round(float(np.median(scored)), 4) if scored else None,
            "scored": len(scored),
            "rows": rows,
        }

    print(f"{'view':22s} " + " ".join(f"{label:>10s}" for label, _ in passes))
    for view_id in views:
        cells = []
        for label, _ in passes:
            row = report["passes"][label]["rows"].get(view_id, {})
            cells.append(f"{row['score']:10.4f}" if "score" in row else f"{'--':>10s}")
        print(f"{view_id:22s} " + " ".join(cells))
    print(f"{'MEAN':22s} " + " ".join(
        f"{report['passes'][label]['mean']:10.4f}" if report["passes"][label]["mean"] is not None else f"{'--':>10s}"
        for label, _ in passes))

    if args.output:
        out = ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
