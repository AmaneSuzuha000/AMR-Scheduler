"""Cross-tabulate geometric score against registration trust.

Two separate defects look identical in a single visual score, and they need opposite
fixes:

  * the geometry is wrong            -> change the scene
  * the camera is not registered     -> change the camera (or accept the view is unusable)

This joins geom_score.json (true-silhouette IoU) with registration_gate.json (held-out
landmark residual against the report's own tolerance) so a low score can be attributed
instead of guessed at. Views whose registration is 'fail' should not be used to justify
geometry changes: you cannot tell missing structure from a misplaced camera.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GOOD_GEOM = 0.35   # at or above this the silhouette is broadly in the right place


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geom", default="out/visual_baseline_01/geom_score.json")
    parser.add_argument("--gate", default="out/visual_baseline_01/registration_gate.json")
    parser.add_argument("--output", default="out/visual_baseline_01/attribution.json")
    args = parser.parse_args()

    geom = json.loads((ROOT / args.geom).read_text(encoding="utf-8"))["rows"]
    gate = {r["id"]: r for r in json.loads((ROOT / args.gate).read_text(encoding="utf-8"))["views"]}

    table = []
    for view_id, row in geom.items():
        if "geom_iou" not in row:
            continue
        trust = gate.get(view_id, {})
        status = trust.get("status", "unknown")
        iou = row["geom_iou"]
        if status == "fail":
            verdict = "配准不可信：不能据此判断几何，先修相机或排除该视角"
            action = "registration_first"
        elif iou < GOOD_GEOM and status in {"pass", "marginal"}:
            verdict = "配准可信但几何差：这是真的几何问题，可以改场景"
            action = "geometry"
        elif iou < GOOD_GEOM:
            verdict = "几何差但配准未验证：先补留出地标再判断"
            action = "verify_first"
        else:
            verdict = "几何与配准都相对可信：留作对照"
            action = "keep"
        table.append({
            "id": view_id,
            "geom_iou": iou,
            "coverage_ratio": row["coverage_ratio"],
            "registration_status": status,
            "registration_ratio": trust.get("ratio"),
            "fit_rmse_px": trust.get("fit_rmse_px"),
            "check_rmse_px": trust.get("check_rmse_px"),
            "action": action,
            "verdict_zh": verdict,
        })

    table.sort(key=lambda r: (r["action"], r["geom_iou"]))
    groups: dict[str, list] = {}
    for row in table:
        groups.setdefault(row["action"], []).append(row)

    out = {
        "tool": "tools/wheelset_scene/attribution.py",
        "good_geometry_threshold": GOOD_GEOM,
        "note_zh": "把几何得分与配准可信度交叉，避免用'配准坏的视角'去证明'几何差'。",
        "counts": {k: len(v) for k, v in groups.items()},
        "groups": {k: [r["id"] for r in v] for k, v in groups.items()},
        "rows": table,
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"{'view':22s} {'iou':>6s} {'ratio':>6s} {'reg':>10s} {'regRatio':>9s}  action")
    for row in table:
        print(
            f"{row['id']:22s} {row['geom_iou']:6.4f} {row['coverage_ratio']:6.2f} "
            f"{row['registration_status']:>10s} "
            f"{row['registration_ratio'] if row['registration_ratio'] is not None else '-':>9}  "
            f"{row['action']}"
        )
    print()
    for action, rows in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        print(f"{action:18s} {len(rows):2d}  {rows[0]['verdict_zh']}")
        print(f"                   {', '.join(r['id'] for r in rows)}")
    print()
    usable = [r for r in table if r["action"] == "geometry"]
    if usable:
        print("几何问题可以被证实（配准 pass/marginal 且 iou < %.2f）的视角：" % GOOD_GEOM)
        for row in usable:
            print(f"  {row['id']:22s} iou={row['geom_iou']:.4f} coverage={row['coverage_ratio']:.2f} "
                  f"reg={row['registration_status']}({row['registration_ratio']})")


if __name__ == "__main__":
    main()
