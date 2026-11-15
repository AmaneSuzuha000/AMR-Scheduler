"""Report a render pass against the project's frozen visual floor.

The project freezes a visual baseline of 0.26 and requires new work to sit on it.
A mean over a hand-picked subset can be moved by choosing the subset, so this tool
reports the things that do not depend on that choice:

  * mean / median / min over every view with a reference frame
  * how many views are below the frozen floor
  * the absolute margin, i.e. mean - floor
  * per-view delta against a named reference pass

It refuses to print a single headline number without the view count next to it.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FROZEN_FLOOR = 0.26


def summarise(scores: dict[str, float]) -> dict:
    values = np.array(list(scores.values()), dtype=float)
    return {
        "views": int(values.size),
        "mean": round(float(values.mean()), 4),
        "median": round(float(np.median(values)), 4),
        "min": round(float(values.min()), 4),
        "max": round(float(values.max()), 4),
        "below_floor": int((values < FROZEN_FLOOR).sum()),
        "margin_to_floor": round(float(values.mean() - FROZEN_FLOOR), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="output of score_passes.py")
    parser.add_argument("--baseline", default="", help="label inside the same report to diff against")
    args = parser.parse_args()

    report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    passes = report["passes"]
    base_scores = None
    if args.baseline:
        base_scores = {k: v["score"] for k, v in passes[args.baseline]["rows"].items() if "score" in v}

    for label, data in passes.items():
        scores = {k: v["score"] for k, v in data["rows"].items() if "score" in v}
        stats = summarise(scores)
        print(f"== {label}")
        print(
            f"   views={stats['views']} mean={stats['mean']:.4f} median={stats['median']:.4f} "
            f"min={stats['min']:.4f} max={stats['max']:.4f}"
        )
        print(
            f"   below frozen floor {FROZEN_FLOOR}: {stats['below_floor']}/{stats['views']}  "
            f"margin to floor: {stats['margin_to_floor']:+.4f}"
        )
        if base_scores and label != args.baseline:
            deltas = {
                view: scores[view] - base_scores[view]
                for view in scores
                if view in base_scores
            }
            improved = sum(1 for d in deltas.values() if d > 0.002)
            regressed = sum(1 for d in deltas.values() if d < -0.002)
            print(
                f"   vs {args.baseline}: mean delta {np.mean(list(deltas.values())):+.4f}  "
                f"improved {improved}  regressed {regressed}  unchanged {len(deltas) - improved - regressed}"
            )
            worst = sorted(deltas.items(), key=lambda kv: kv[1])[:3]
            best = sorted(deltas.items(), key=lambda kv: -kv[1])[:3]
            print("   worst: " + ", ".join(f"{k} {v:+.4f}" for k, v in worst))
            print("   best : " + ", ".join(f"{k} {v:+.4f}" for k, v in best))


if __name__ == "__main__":
    main()
