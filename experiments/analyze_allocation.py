"""Compare allocation strategies against scheduling rules.

Usage::

    python experiments/analyze_allocation.py
    python experiments/analyze_allocation.py --baseline priority+greedy
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_reschedule import exact_wilcoxon_p  # reuse the same test implementation

#: metrics where lower is better
LOWER_BETTER = {"makespan_s", "conflict_wait_steps", "mean_urgent_response_steps"}

METRICS = [
    ("makespan_s", "makespan(s)"),
    ("on_time_rate", "on-time rate"),
    ("urgent_on_time_rate", "urgent on-time rate"),
    ("agv_utilization", "AGV utilization"),
    ("conflict_wait_steps", "conflict wait (steps)"),
]


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/allocation.csv")
    p.add_argument("--baseline", default="priority+greedy")
    args = p.parse_args(argv)

    rows = load(args.csv)
    strategies = sorted({r["strategy"] for r in rows})
    cells = sorted({(int(r["n_tasks"]), int(r["n_agvs"])) for r in rows})

    # (strategy, n_tasks, n_agvs) -> metric -> [values]
    agg: dict[tuple[str, int, int], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        key = (r["strategy"], int(r["n_tasks"]), int(r["n_agvs"]))
        for m, _ in METRICS:
            agg[key][m].append(float(r[m]))

    def mean(strategy: str, nt: int, na: int, metric: str) -> float:
        vals = agg[(strategy, nt, na)][metric]
        return sum(vals) / len(vals) if vals else float("nan")

    print("baseline = %s" % args.baseline)
    print("=" * 100)
    print("%-22s | %14s %8s | %10s | %10s | %10s"
          % ("strategy", "makespan(s)", "vs baseline", "on-time", "util", "conflict wait"))
    print("-" * 100)

    # overall mean across cells + percentage relative to the baseline
    for st in strategies:
        mk = [mean(st, nt, na, "makespan_s") for nt, na in cells]
        ot = [mean(st, nt, na, "on_time_rate") for nt, na in cells]
        ut = [mean(st, nt, na, "agv_utilization") for nt, na in cells]
        wt = [mean(st, nt, na, "conflict_wait_steps") for nt, na in cells]
        bmk = [mean(args.baseline, nt, na, "makespan_s") for nt, na in cells]
        rel = 100.0 * (sum(mk) / len(mk) - sum(bmk) / len(bmk)) / (sum(bmk) / len(bmk))
        flag = "  <-- baseline" if st == args.baseline else ""
        print("%-22s | %14.1f %+7.2f%% | %10.4f | %10.4f | %10.1f%s"
              % (st, sum(mk) / len(mk), rel, sum(ot) / len(ot),
                 sum(ut) / len(ut), sum(wt) / len(wt), flag))

    # best per cell
    print()
    print("best strategy per cell by makespan:")
    print("-" * 100)
    for nt, na in cells:
        best = min(strategies, key=lambda s: mean(s, nt, na, "makespan_s"))
        base_v = mean(args.baseline, nt, na, "makespan_s")
        best_v = mean(best, nt, na, "makespan_s")
        print("  %3d tasks / %d AGVs: %-22s %8.1fs (baseline %8.1fs, %+.2f%%)"
              % (nt, na, best, best_v, base_v, 100.0 * (best_v - base_v) / base_v))

    # paired significance: each strategy vs the baseline (same instance, same seed)
    print()
    print("paired significance (same instance & seed, exact Wilcoxon):")
    print("-" * 100)
    for st in strategies:
        if st == args.baseline:
            continue
        lines = []
        for metric, label in METRICS:
            base_vals: dict[tuple[int, int, int], float] = {}
            st_vals: dict[tuple[int, int, int], float] = {}
            for r in rows:
                key = (int(r["n_tasks"]), int(r["n_agvs"]), int(r["seed"]))
                if r["strategy"] == args.baseline:
                    base_vals[key] = float(r[metric])
                elif r["strategy"] == st:
                    st_vals[key] = float(r[metric])
            diffs = []
            for key, av in base_vals.items():
                bv = st_vals.get(key)
                if bv is None:
                    continue
                d = bv - av
                if metric in LOWER_BETTER:
                    d = -d          # flip so that larger is always better
                diffs.append(d)
            if not diffs:
                continue
            better = sum(1 for d in diffs if d > 0)
            worse = sum(1 for d in diffs if d < 0)
            pv = exact_wilcoxon_p(diffs)
            star = " *sig" if pv < 0.05 else ""
            lines.append("    %-16s better/worse = %3d/%3d  p=%.4f%s"
                         % (label, better, worse, pv, star))
        if lines:
            print("  %s  (n=%d pairs)" % (st, len(diffs)))
            for ln in lines:
                print(ln)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
