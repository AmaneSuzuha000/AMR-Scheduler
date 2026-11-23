"""Matrix result analysis: paired comparison + aggregation by scale.

It does three things:

1. within each (task scale, #AGVs) cell, compare the two planners in a
   **paired** way (same seed = one pair, cancelling instance randomness);
2. report the relative change, the same-direction count, and the exact
   Wilcoxon signed-rank p value;
3. discuss how many AGVs it makes sense to deploy ().

Usage::

    python experiments/analyze_matrix.py --csv results/matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The test lives in analyze_reschedule, which has the n > 20 normal-approximation
# branch. This module used to carry its own copy holding only the exact
# enumeration: once n went past 20 -- e.g. handing it 90 paired results at once --
# it enumerated 2^n sign combinations and hung the process. Reuse the one
# implementation, as analyze_allocation and analyze_layout_paired already do.
from experiments.analyze_reschedule import exact_wilcoxon_p  # noqa: E402

METRICS = ["makespan_s", "on_time_rate", "agv_utilization", "conflict_wait_steps"]


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default="results/matrix.csv")
    p.add_argument("--baseline", default="static_dijkstra")
    p.add_argument("--treatment", default="spacetime_astar")
    args = p.parse_args(argv)

    rows = load(args.csv)
    for r in rows:
        for m in METRICS:
            r[m] = float(r[m])
        r["n_tasks"] = int(r["n_tasks"])
        r["n_agvs"] = int(r["n_agvs"])
        r["seed"] = int(r["seed"])

    cells: dict[tuple[int, int], dict[str, dict[int, dict]]] = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        cells[(r["n_tasks"], r["n_agvs"])][r["planner"]][r["seed"]] = r

    print(f"baseline = {args.baseline}    treatment = {args.treatment}")
    print("=" * 112)
    header = (
        f"{'tasks':>5} {'AGV':>4} | {'base ms(s)':>12} {'trt ms(s)':>12} {'change':>9} {'same':>7} {'p':>8}"
        f" | {'base wait':>9} {'trt wait':>9} {'change':>9}"
    )
    print(header)
    print("-" * 112)

    summary_rows = []
    for (nt, na) in sorted(cells):
        b = cells[(nt, na)].get(args.baseline, {})
        t = cells[(nt, na)].get(args.treatment, {})
        common = sorted(set(b) & set(t))
        if not common:
            continue

        def mean(store, key):
            return sum(store[s][key] for s in common) / len(common)

        b_ms, t_ms = mean(b, "makespan_s"), mean(t, "makespan_s")
        b_w, t_w = mean(b, "conflict_wait_steps"), mean(t, "conflict_wait_steps")
        diffs_ms = [t[s]["makespan_s"] - b[s]["makespan_s"] for s in common]
        better = sum(1 for d in diffs_ms if d < 0)
        pval = exact_wilcoxon_p(diffs_ms)
        pct = (t_ms - b_ms) / b_ms * 100 if b_ms else float("nan")
        wpct = (t_w - b_w) / b_w * 100 if b_w else float("nan")
        print(
            f"{nt:>5} {na:>4} | {b_ms:12.1f} {t_ms:12.1f} {pct:+8.2f}% "
            f"{better:>3}/{len(common):<3} {pval:8.4f} | {b_w:9.1f} {t_w:9.1f} {wpct:+8.1f}%"
        )
        summary_rows.append((nt, na, b_ms, t_ms, pct, better, len(common), pval, b_w, t_w, wpct))

    print("=" * 112)
    # aggregate by #AGVs, to discuss how many AGVs to deploy ()
    print("\nAggregated by #AGVs (across task scales):")
    print(f"{'AGV':>4} | {'base ms mean':>12} {'trt ms mean':>12} {'change':>9} | {'base util':>10} {'trt util':>10}")
    for na in sorted({k[1] for k in cells}):
        bs, ts, bu, tu = [], [], [], []
        for nt in sorted({k[0] for k in cells}):
            b = cells[(nt, na)].get(args.baseline, {})
            t = cells[(nt, na)].get(args.treatment, {})
            common = sorted(set(b) & set(t))
            if not common:
                continue
            bs.append(sum(b[s]["makespan_s"] for s in common) / len(common))
            ts.append(sum(t[s]["makespan_s"] for s in common) / len(common))
            bu.append(sum(b[s]["agv_utilization"] for s in common) / len(common))
            tu.append(sum(t[s]["agv_utilization"] for s in common) / len(common))
        if not bs:
            continue
        bm, tm = sum(bs) / len(bs), sum(ts) / len(ts)
        print(
            f"{na:>4} | {bm:12.1f} {tm:12.1f} {(tm-bm)/bm*100:+8.2f}% | "
            f"{sum(bu)/len(bu):10.3f} {sum(tu)/len(tu):10.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
