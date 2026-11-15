"""Paired comparison across the three preemption tiers (thesis 6.4).

``results/preempt_tiers.csv`` holds 90 (task scale, fleet size, seed) cells run
under each of three policies. This script turns that file into the three tables
of section 6.4: the tier overview, NONE -> PREEMPT_UNPICKED, and
PREEMPT_UNPICKED -> PREEMPT_ANY.

Every p value is a paired Wilcoxon signed-rank test on the 90 paired
differences, using the same implementation as the rest of the repository
(``analyze_reschedule.exact_wilcoxon_p``: exact enumeration up to n = 20,
normal approximation with tie correction past it).

The "better / worse" counts depend on the metric's direction: higher is better
for the two on-time rates, lower is better for response time, makespan and
material waste.

Usage::

    python experiments/analyze_preempt.py
    python experiments/analyze_preempt.py --csv results/preempt_tiers.csv
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.analyze_reschedule import exact_wilcoxon_p  # noqa: E402

#: (column, label, direction, decimals); direction +1 means higher is better
METRICS = [
    ("urgent_on_time_rate", "urgent on-time rate", +1, 3),
    ("mean_urgent_response_steps", "urgent response (steps)", -1, 1),
    ("on_time_rate", "on-time rate", +1, 3),
    ("makespan_s", "makespan (s)", -1, 1),
    ("material_waste_steps", "material waste (s)", -1, 1),
]

PAIRS = [
    ("none", "preempt_unpicked", "NONE -> PREEMPT_UNPICKED"),
    ("preempt_unpicked", "preempt_any", "PREEMPT_UNPICKED -> PREEMPT_ANY"),
]


def load(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def keyed(rows, policy):
    return {(r["n_tasks"], r["n_agvs"], r["seed"]): r
            for r in rows if r["policy"] == policy}


def cells(rows):
    """The paired cells, in a stable order."""
    return sorted({(r["n_tasks"], r["n_agvs"], r["seed"]) for r in rows})


def compare(a, b, keys, column, direction):
    """Mean change, better/worse/same counts and the paired p value."""
    xs = [float(a[k][column]) for k in keys]
    ys = [float(b[k][column]) for k in keys]
    diffs = [y - x for x, y in zip(xs, ys)]
    nonzero = [d for d in diffs if abs(d) > 1e-12]
    better = sum(1 for d in nonzero if direction * d > 0)
    worse = sum(1 for d in nonzero if direction * d < 0)
    same = len(diffs) - len(nonzero)
    return {
        "mean_a": statistics.fmean(xs),
        "mean_b": statistics.fmean(ys),
        "better": better,
        "worse": worse,
        "same": same,
        "p": exact_wilcoxon_p(diffs),
        "pct": (statistics.fmean(ys) - statistics.fmean(xs)) / statistics.fmean(xs) * 100
        if statistics.fmean(xs) else 0.0,
    }


def fmt_p(p):
    return "<0.0001" if p < 0.0001 else "%.4f" % p


def noninferiority(diffs, deltas=(3.0, 5.0, 10.0)):
    """One-sided t test of H0: relative loss >= delta against H1: loss < delta."""
    n = len(diffs)
    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs)
    se = sd / math.sqrt(n)
    out = []
    for delta in deltas:
        # H0 boundary is +delta% of the baseline; shift the sample to it.
        t = (mean - delta) / se if se else float("-inf")
        p = 0.5 * math.erfc(-t / math.sqrt(2.0))
        out.append((delta, mean, t, p))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="results/preempt_tiers.csv")
    args = parser.parse_args(argv)

    rows = load(args.csv)
    keys = cells(rows)
    print("cells: %d x %d policies = %d rows"
          % (len(keys), len({r['policy'] for r in rows}), len(rows)))
    print()

    by_policy = {p: keyed(rows, p) for p in ("none", "preempt_unpicked", "preempt_any")}

    # ---- 6.4.1 overview ----
    print("=== 6.4.1 tier overview (mean over %d cells) ===" % len(keys))
    overview = [METRICS[0], METRICS[1], ("preemption_count", "preemptions", +1, 2), METRICS[4]]
    print("%-18s%14s%14s%14s%14s" % (("policy",) + tuple(m[1] for m in overview)))
    for name in ("none", "preempt_unpicked", "preempt_any"):
        line = "%-18s" % name
        for column, _label, _direction, digits in overview:
            vals = [float(by_policy[name][k][column]) for k in keys]
            line += "%14.*f" % (digits, statistics.fmean(vals))
        print(line)
    print()

    # ---- 6.4.2 / 6.4.3 paired tables ----
    for lo, hi, title in PAIRS:
        a, b = by_policy[lo], by_policy[hi]
        print("=== %s ===" % title)
        print("%-24s %-20s %-22s %-9s %s"
              % ("metric", "mean", "better/worse/same", "p", "verdict"))
        for column, label, direction, digits in METRICS:
            r = compare(a, b, keys, column, direction)
            if r["better"] == 0 and r["worse"] == 0:
                verdict = "no change"
            elif r["p"] >= 0.05:
                verdict = "not significant"
            else:
                verdict = "better" if direction * (r["mean_b"] - r["mean_a"]) > 0 else "worse"
            print("%-24s %8.*f -> %-8.*f %3d/%3d/%-3d          %-9s %s"
                  % (label, digits, r["mean_a"], digits, r["mean_b"],
                     r["better"], r["worse"], r["same"], fmt_p(r["p"]), verdict))
        print()

    # ---- 6.4.2 non-inferiority on makespan ----
    a, b = by_policy["none"], by_policy["preempt_unpicked"]
    diffs = [float(b[k]["makespan_s"]) - float(a[k]["makespan_s"]) for k in keys]
    base = statistics.fmean(float(a[k]["makespan_s"]) for k in keys)
    rel = [d / base * 100 for d in diffs]
    print("=== 6.4.2 non-inferiority, makespan (n=%d pairs) ===" % len(keys))
    print("mean change: %+.1f s (%+.2f%%)" % (statistics.fmean(diffs), statistics.fmean(rel)))
    print("%-6s %-12s %-12s %s" % ("delta", "t", "one-sided p", "verdict"))
    for delta, _mean, t, p in noninferiority(rel):
        print("%-6.0f%% %-12.2f %-12.4f %s"
              % (delta, t, p, "non-inferior" if p < 0.05 else "inconclusive"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
