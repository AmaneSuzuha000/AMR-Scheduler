"""6.9.2 Stratified paired test (only pairs where both sides fully completed).

Why not use the output of analyze_layout_portability.py
-----------------------------------------------------
That script pairs runs whenever both sides have tasks_completed > 0, i.e. as
soon as a run moved at all it is paired.
But makespan is defined as "the completion time of the last of the completed
tasks" — **the fewer tasks completed, the smaller this number looks**.
Mixing in unfinished runs yields, on hub, the reversed conclusion that "the
main method is 23.8% worse than space-time A*", while the truth is that the
main method simply leaves more runs unfinished on hub.

Section 6.9.2 of the paper uses "both sides 100% complete"; this script
recomputes the numbers under that rule.
"""

from __future__ import annotations

import csv
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.analyze_reschedule import exact_wilcoxon_p  # noqa: E402


def main() -> int:
    with open(ROOT / "results" / "layout_portability.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    cell = defaultdict(dict)
    for r in rows:
        k = (r["layout"], int(r["n_tasks"]), int(r["n_agvs"]), r["seed"])
        cell[k][r["strategy"]] = (
            float(r["makespan_s"]),
            int(r["tasks_completed"]),
            int(r["n_tasks"]),
        )

    print("6.9.2 Stratified paired test (a pair is valid only if both sides are 100% complete)")
    print("=" * 74)
    print(f"{'layout':<8}{'valid':>8}{'better/worse':>14}{'p':>12}{'mean diff':>12}{'median diff':>13}")
    print("-" * 74)
    for lay in ("ring", "grid", "hub"):
        xs, ys = [], []
        for k, v in cell.items():
            if k[0] != lay:
                continue
            a = v.get("dijkstra+greedy")
            b = v.get("spacetime+greedy")
            if not a or not b:
                continue
            if a[1] != a[2] or b[1] != b[2]:
                continue  # either side incomplete -> not comparable
            xs.append(a[0])
            ys.append(b[0])
        if not xs:
            print(f"{lay:<8}  no valid pairs")
            continue
        diffs = [x - y for x, y in zip(xs, ys)]
        good = sum(1 for d in diffs if d < 0)
        bad = sum(1 for d in diffs if d > 0)
        p = exact_wilcoxon_p(diffs)
        md = (st.mean(xs) - st.mean(ys)) / st.mean(ys) * 100
        medd = (st.median(xs) - st.median(ys)) / st.median(ys) * 100
        print(f"{lay:<8}{len(xs):>8}{f'{good}better/{bad}worse':>14}"
              f"{p:>12.4f}{md:>11.1f}%{medd:>11.1f}%")
    print("=" * 74)
    print("Note: a negative mean diff means the main method is faster. p is the exact Wilcoxon signed-rank (enumerated for n<=20).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())