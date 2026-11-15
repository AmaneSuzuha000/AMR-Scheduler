"""Classify each view's failure mode so the next fix targets the right thing.

A single IoU number cannot say what is wrong, and the round-2 evidence showed two
different failures were being treated as one:

  * "we draw far more structure than the video shows" - mask_recall low
  * "the structure lands where the video has none"   - both recalls low

This tool turns the geom_score report into a verdict per view, using fixed
thresholds stated up front so the classification is auditable rather than a matter
of taste. It does not decide what to change; it only says which lever applies.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# Fixed, stated thresholds. Chosen before looking at the result so the buckets are
# not tuned to make the table look tidy.
EXCESS_RATIO = 1.15     # silhouette covers >15% more of the frame than the reference
DEFICIT_RATIO = 0.85    # ... or >15% less
PLACEMENT_IOU = 0.30    # below this, what we draw is not where the video has it

VERDICTS = {
    "placement": "结构量大体对但位置错：优先用地标定相机，不要加减设备",
    "excess": "画多了：镜头里有多余实例或缺少遮挡，先查实例清单",
    "deficit": "画少了：确实缺结构，可以补几何",
    "both": "既多又位错：先定相机，再重估实例",
    "ok": "量级与位置都相对接近：留作对照，不动",
}


def classify(row: dict) -> str:
    ratio = row.get("coverage_ratio")
    iou = row.get("geom_iou")
    if ratio is None or iou is None:
        return "unscored"
    if ratio > EXCESS_RATIO and iou < PLACEMENT_IOU:
        return "excess"
    if ratio < DEFICIT_RATIO and iou < PLACEMENT_IOU:
        return "deficit"
    if iou < PLACEMENT_IOU:
        return "placement"
    if ratio > EXCESS_RATIO or ratio < DEFICIT_RATIO:
        return "both"
    return "ok"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="out/visual_baseline_01/geom_score.json")
    parser.add_argument("--output", default="out/visual_baseline_01/defect_classes.json")
    args = parser.parse_args()

    report = json.loads((ROOT / args.report).read_text(encoding="utf-8"))
    rows = {k: v for k, v in report["rows"].items() if "geom_iou" in v}
    buckets: dict[str, list] = {}
    for view_id, row in sorted(rows.items(), key=lambda kv: kv[1]["geom_iou"]):
        verdict = classify(row)
        row["verdict"] = verdict
        buckets.setdefault(verdict, []).append(view_id)

    out = {
        "tool": "tools/wheelset_scene/defect_classify.py",
        "thresholds": {
            "excess_coverage_ratio_above": EXCESS_RATIO,
            "deficit_coverage_ratio_below": DEFICIT_RATIO,
            "placement_iou_below": PLACEMENT_IOU,
        },
        "verdicts_zh": VERDICTS,
        "counts": {k: len(v) for k, v in buckets.items()},
        "buckets": buckets,
        "rows": rows,
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'view':22s} {'iou':>6s} {'ratio':>6s} {'maskRec':>8s} {'refRec':>7s}  verdict")
    for view_id, row in sorted(rows.items(), key=lambda kv: kv[1]["geom_iou"]):
        print(
            f"{view_id:22s} {row['geom_iou']:6.4f} {row['coverage_ratio']:6.2f} "
            f"{row['mask_recall']:8.3f} {row['ref_recall']:7.3f}  {row['verdict']}"
        )
    print()
    for verdict, views in sorted(buckets.items(), key=lambda kv: -len(kv[1])):
        print(f"{verdict:10s} {len(views):2d}  {VERDICTS.get(verdict, '')}")
        print(f"           {', '.join(views)}")


if __name__ == "__main__":
    main()
