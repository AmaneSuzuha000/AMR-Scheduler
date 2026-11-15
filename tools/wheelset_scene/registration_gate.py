"""Turn the existing camera-fit reports into a gate: which registrations fail their own test?

Every camera except three points at a camera_fit_*.json. Those reports already record
what they were fitted on, how well the fit landed, and how well it held up on landmarks
that were deliberately excluded from the fit. Nobody was reading those numbers.

The 'unknown' report is the demonstration case:

    pixel_reading_tolerance_estimate: 15 px
    fit_rmse_px:   12.85 px   (8 landmarks used for fitting)
    check_rmse_px: 29.38 px   (4 landmarks held out)
    claim_boundary: "Cyan bed outlines and snapshot bridge endpoints only;
                     not independent full-image, count, dimension or dynamics acceptance."

A held-out residual of nearly twice the stated tolerance means the camera did not
generalise to the landmarks it was not fitted to. That is not a matter of taste; it is
the report contradicting itself.

This tool does NOT move any camera. It produces the list of views whose registration is
not trustworthy, so that:
  * a visual comparison can exclude them instead of blaming the geometry, and
  * re-fixing them is prioritised by evidence rather than by whichever view scores worst.

Decision rule, stated before use:
    ratio = check_rmse_px / pixel_reading_tolerance_estimate
    ratio <= 1.0   pass        - fit generalised within the stated reading tolerance
    ratio <= 1.5   marginal    - usable for rough framing, not for measurement
    ratio >  1.5   fail        - held-out landmarks miss by more than 50% of tolerance
Views with no check landmarks are 'unverified', which is not the same as 'good'.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PASS = 1.0
MARGINAL = 1.5
# Used only when a report records a held-out residual but no tolerance to judge it
# against. 15 px is the median declared tolerance across camera_fit_*.json, so the
# fallback sits in the middle of the range the project actually writes down.
DEFAULT_TOLERANCE_PX = 15.0


def load_report(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cameras", default="simulation/wheelset_workshop/cameras.json")
    parser.add_argument("--output", default="out/visual_baseline_01/registration_gate.json")
    args = parser.parse_args()

    cameras = json.loads((ROOT / args.cameras).read_text(encoding="utf-8"))
    cache: dict[str, dict | None] = {}
    rows = []

    for camera in cameras:
        view_id = camera["id"]
        rel = camera.get("registration_report")
        row = {
            "id": view_id,
            "registration": camera.get("registration"),
            "report": rel,
        }
        if not rel:
            row["status"] = "no_report"
            rows.append(row)
            continue
        if rel not in cache:
            cache[rel] = load_report(ROOT / rel)
        report = cache[rel]
        if report is None:
            row["status"] = "report_missing"
            rows.append(row)
            continue

        tolerance = report.get("pixel_reading_tolerance_estimate")
        entries = [v for v in report.get("views", []) if v.get("camera") == view_id]
        if not entries:
            row["status"] = "view_not_in_report"
            row["report_tolerance_px"] = tolerance
            rows.append(row)
            continue
        entry = entries[0]
        fit = entry.get("fit_rmse_px")
        check = entry.get("check_rmse_px")
        row.update({
            "report_tolerance_px": tolerance,
            "fit_rmse_px": round(fit, 2) if isinstance(fit, (int, float)) else fit,
            "check_rmse_px": round(check, 2) if isinstance(check, (int, float)) else check,
            "fit_count": entry.get("fit_count"),
            "check_count": entry.get("check_count"),
            "claim_boundary": (report.get("claim_boundary") or "")[:160],
        })
        # A report that omits its tolerance still gives us a held-out residual, and that
        # residual is evidence whether or not a tolerance was written down. Treating
        # "no tolerance declared" as "unverified" let inspection_exit through with a
        # check residual of 78.43 px - the second worst in the whole suite - simply
        # because camera_fit_*.json left the field empty. Falling back to a declared
        # default keeps such cases visible instead of silently lumping them with the
        # cameras that genuinely have no check landmarks.
        if not isinstance(check, (int, float)):
            row["status"] = "unverified"
        else:
            if not isinstance(tolerance, (int, float)) or tolerance <= 0:
                tolerance = DEFAULT_TOLERANCE_PX
                row["tolerance_source"] = "fallback_default"
            else:
                row["tolerance_source"] = "report"
            row["effective_tolerance_px"] = tolerance
            ratio = check / tolerance
            row["ratio"] = round(ratio, 2)
            if ratio <= PASS:
                row["status"] = "pass"
            elif ratio <= MARGINAL:
                row["status"] = "marginal"
            else:
                row["status"] = "fail"
        rows.append(row)

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1

    out = {
        "tool": "tools/wheelset_scene/registration_gate.py",
        "kind": "registration_trust_gate",
        "rule": {
            "ratio": "check_rmse_px / pixel_reading_tolerance_estimate",
            "pass_at_or_below": PASS,
            "marginal_at_or_below": MARGINAL,
            "fail_above": MARGINAL,
        },
        "note_zh": (
            "本工具不移动任何相机。它只把已有配准报告里的'留出地标残差'与报告自述的读数容差相比，"
            "列出不可信的视角。unverified（无留出地标）不等于合格。"
        ),
        "counts": counts,
        "views": rows,
    }
    path = ROOT / args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")

    order = {"fail": 0, "marginal": 1, "unverified": 2, "view_not_in_report": 3,
             "no_report": 4, "report_missing": 5, "pass": 6}
    for row in sorted(rows, key=lambda r: (order.get(r["status"], 9), -(r.get("ratio") or 0))):
        def cell(value, width, digits=2):
            if isinstance(value, (int, float)):
                return f"{value:>{width}.{digits}f}"
            return f"{'-':>{width}}"

        ratio = row.get("ratio")
        print(
            f"{row['id']:22s} {row['status']:18s} "
            f"fit={cell(row.get('fit_rmse_px'), 7)} check={cell(row.get('check_rmse_px'), 7)} "
            f"tol={cell(row.get('report_tolerance_px'), 4, 0)} ratio={cell(ratio, 6)}"
        )
    print()
    print("counts: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
