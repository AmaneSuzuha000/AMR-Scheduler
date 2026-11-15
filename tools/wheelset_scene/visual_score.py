"""Quantitative visual comparison between reference video frames and our renders.

This is a *baseline ranking* tool, not an acceptance gate. It reports silhouette
IoU, edge IoU, and a coarse spatial mismatch grid so that low-scoring views can be
ranked and inspected. Camera registration in the scene is still an initial
estimate, so a low score can mean "wrong camera" just as easily as "wrong
geometry"; the report keeps those two hypotheses separate by also emitting the
per-view camera residual when one is available.

No score in this file may be quoted as scene fidelity. It only ranks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
COMPARE_ROOT = ROOT / "visual_comparison" / "wheelset_workshop_v1"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_rgb(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0


def otsu(values: np.ndarray, bins: int = 256) -> float:
    """Otsu threshold over a 1-D sample; deterministic and dependency-free."""
    hist, edges = np.histogram(values, bins=bins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.5
    prob = hist / total
    centres = (edges[:-1] + edges[1:]) / 2.0
    omega = np.cumsum(prob)
    mu = np.cumsum(prob * centres)
    mu_total = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma = (mu_total * omega - mu) ** 2 / denom
    sigma = np.nan_to_num(sigma, nan=-1.0, posinf=-1.0, neginf=-1.0)
    return float(centres[int(np.argmax(sigma))])


def background(rgb: np.ndarray) -> np.ndarray:
    """Estimate the flat backdrop colour from the image border."""
    border = np.concatenate(
        [rgb[0, :, :], rgb[-1, :, :], rgb[:, 0, :], rgb[:, -1, :]], axis=0
    )
    return np.median(border, axis=0)


def foreground_mask(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    bg = background(rgb)
    distance = np.abs(rgb - bg).max(axis=2)
    threshold = otsu(distance.ravel())
    mask = distance > threshold
    info = {
        "background_rgb": [round(float(v), 4) for v in bg],
        "threshold": round(threshold, 4),
        "fg_frac": round(float(mask.mean()), 4),
    }
    return mask, info


def gradient(rgb: np.ndarray) -> np.ndarray:
    grey = rgb @ np.array([0.299, 0.587, 0.114])
    gx = np.zeros_like(grey)
    gy = np.zeros_like(grey)
    gx[:, 1:-1] = grey[:, 2:] - grey[:, :-2]
    gy[1:-1, :] = grey[2:, :] - grey[:-2, :]
    return np.hypot(gx, gy)


def edge_mask(rgb: np.ndarray, quantile: float = 0.90) -> np.ndarray:
    magnitude = gradient(rgb)
    threshold = float(np.quantile(magnitude, quantile))
    return magnitude > max(threshold, 1e-6)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(a, b).sum() / union)


def grid_mismatch(ref: np.ndarray, render: np.ndarray, cells: int = 8) -> list[list[float]]:
    diff = np.abs(ref - render).mean(axis=2)
    h, w = diff.shape
    out = []
    for row in range(cells):
        line = []
        for col in range(cells):
            block = diff[
                row * h // cells : (row + 1) * h // cells,
                col * w // cells : (col + 1) * w // cells,
            ]
            line.append(round(float(block.mean()), 4))
        out.append(line)
    return out


def compare(ref_path: Path, render_path: Path, cells: int = 8) -> dict:
    ref = load_rgb(ref_path)
    render = load_rgb(render_path)
    if ref.shape != render.shape:
        render_img = Image.open(render_path).convert("RGB").resize(
            (ref.shape[1], ref.shape[0]), Image.LANCZOS
        )
        render = np.asarray(render_img, dtype=np.float64) / 255.0
        resized = True
    else:
        resized = False
    ref_mask, ref_info = foreground_mask(ref)
    render_mask, render_info = foreground_mask(render)
    ref_edges = edge_mask(ref)
    render_edges = edge_mask(render)
    silhouette = iou(ref_mask, render_mask)
    edges = iou(ref_edges, render_edges)
    mad = float(np.abs(ref - render).mean())
    return {
        "reference": str(ref_path),
        "render": str(render_path),
        "reference_sha256": sha256(ref_path),
        "render_sha256": sha256(render_path),
        "resized": resized,
        "silhouette_iou": round(silhouette, 4),
        "edge_iou": round(edges, 4),
        "mean_abs_diff": round(mad, 4),
        "reference_mask": ref_info,
        "render_mask": render_info,
        "mismatch_grid": grid_mismatch(ref, render, cells),
        "score": round(0.6 * silhouette + 0.4 * edges, 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-dir", default="out/visual_baseline_01/render")
    parser.add_argument("--output", default="out/visual_baseline_01/visual_score.json")
    parser.add_argument("--cells", type=int, default=8)
    args = parser.parse_args()

    render_dir = ROOT / args.render_dir
    index = json.loads((COMPARE_ROOT / "reference_index.json").read_text(encoding="utf-8"))
    rows = []
    missing = []
    for view in index["views"]:
        view_id = view["id"]
        ref_path = COMPARE_ROOT / view["reference"]
        render_path = render_dir / f"{view_id}.png"
        if not ref_path.exists():
            missing.append({"id": view_id, "reason": "reference missing"})
            continue
        if not render_path.exists():
            missing.append({"id": view_id, "reason": "render missing"})
            continue
        row = compare(ref_path, render_path, args.cells)
        row.update(
            {
                "id": view_id,
                "name": view.get("name"),
                "second": view.get("second"),
            }
        )
        rows.append(row)

    rows.sort(key=lambda r: r["score"])
    scored = [r["score"] for r in rows]
    report = {
        "tool": "tools/wheelset_scene/visual_score.py",
        "kind": "baseline_ranking_not_acceptance",
        "note_zh": (
            "本分数只用于排序与定位不匹配区域，不能当作场景保真度验收。"
            "相机配准仍为初始估计，低分既可能是几何错误也可能是机位错误。"
        ),
        "render_dir": args.render_dir,
        "views_compared": len(rows),
        "views_missing": missing,
        "score_mean": round(float(np.mean(scored)), 4) if scored else None,
        "score_median": round(float(np.median(scored)), 4) if scored else None,
        "score_min": round(float(np.min(scored)), 4) if scored else None,
        "score_max": round(float(np.max(scored)), 4) if scored else None,
        "rows": rows,
    }
    out_path = ROOT / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        "VISUALSCORE "
        + json.dumps(
            {
                "compared": len(rows),
                "missing": len(missing),
                "mean": report["score_mean"],
                "median": report["score_median"],
                "worst": [r["id"] for r in rows[:5]],
                "best": [r["id"] for r in rows[-3:]],
                "output": args.output,
            }
        )
    )


if __name__ == "__main__":
    main()
