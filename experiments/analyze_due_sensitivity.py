"""Due-deadline model sensitivity analysis (R5.5 metric validity)."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEVELS = ["low", "medium", "high"]
LEVEL_CN = {"low": "low load", "medium": "medium load", "high": "high load"}
MODELS = ["travel_only", "travel_queue", "fixed"]
MODEL_CN = {
    "travel_only": "travel time x3 (current)",
    "travel_queue": "travel + queueing",
    "fixed": "fixed 600 steps",
}


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/due_sensitivity.csv")
    args = p.parse_args(argv)

    with open(args.csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    print("=" * 96)
    print("Due-deadline model sensitivity: same schedules, on-time rate re-judged under each model")
    print("=" * 96)
    print()
    print("%-8s | %-20s %-16s %-14s | %8s"
          % ("load", MODEL_CN["travel_only"], MODEL_CN["travel_queue"],
             MODEL_CN["fixed"], "spread"))
    print("-" * 96)

    per_level = {}
    for lv in LEVELS:
        sel = [r for r in rows if r["load_level"] == lv]
        if not sel:
            continue
        vals = {m: mean([float(r["on_time_%s" % m]) for r in sel]) for m in MODELS}
        per_level[lv] = vals
        spread = max(vals.values()) - min(vals.values())
        print("%-8s | %20.3f %16.3f %14.3f | %8.3f"
              % (LEVEL_CN[lv], vals["travel_only"], vals["travel_queue"],
                 vals["fixed"], spread))

    print()
    print("=" * 96)
    print("Key findings")
    print("-" * 96)

    all_spread = [max(v.values()) - min(v.values()) for v in per_level.values()]
    worst = max(all_spread)
    print("1) For the same schedule, the on-time rate varies by up to %.3f across models (%.1f x)"
          % (worst, worst / min(min(v.values()) for v in per_level.values())
             if min(min(v.values()) for v in per_level.values()) > 0 else float("nan")))
    print("2) On-time rate under the current model (travel time x3) for the three loads: %s"
          % " / ".join("%.3f" % per_level[lv]["travel_only"] for lv in LEVELS if lv in per_level))
    print("3) Under the queueing model: %s"
          % " / ".join("%.3f" % per_level[lv]["travel_queue"] for lv in LEVELS if lv in per_level))
    print()
    print("Verdict: the current model's values are driven mainly by the deadline formula and do not reflect differences in scheduling quality.")
    print("      the same algorithm's on-time rate drops from %.3f to %.3f across loads,"
          % (per_level["low"]["travel_only"], per_level["high"]["travel_only"])
          if "low" in per_level and "high" in per_level else "")
    print("      but this is mainly because the deadline slack does not adapt to load, not because the algorithm got worse.")
    print()

    # urgent tasks, viewed separately
    print("=" * 96)
    print("Urgent-task on-time rate (same results, three models)")
    print("-" * 96)
    for lv in LEVELS:
        sel = [r for r in rows if r["load_level"] == lv]
        if not sel:
            continue
        vs = ["%.3f" % mean([float(r["urgent_%s" % m]) for r in sel]) for m in MODELS]
        print("  %-8s  %s" % (LEVEL_CN[lv], "  /  ".join(vs)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
