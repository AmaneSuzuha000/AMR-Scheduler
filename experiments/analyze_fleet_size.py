#!/usr/bin/env python3
"""How many AGVs are actually worth buying (domain specification, section 2).

The specification asks for a discussion of "the reasonableness of the AGV configur-
ation". That is not answered by showing that more vehicles finish sooner -- they do, and a
table saying so is not an argument. It is answered by the marginal value of each vehicle,
and by what the fleet costs in interference as it grows.

So this script reports, per task scale:

* the makespan saved per vehicle added, for each fleet step;
* the conflict waiting added over the same step;
* the ratio between successive steps, which is what shows the knee.

The ratio is the honest form of the claim. "4 to 6 gives back about a third of what 2 to 4
did" is checkable; "4 vehicles is optimal" would need an objective function nobody has
defined, because makespan and conflict waiting point in opposite directions and the
specification does not weight them.

Output goes to results/fleet_size.csv so the numbers in the thesis can be traced back here,
and this script re-reads results/baseline.csv rather than re-running the simulation --
the sweep already exists and a second copy of it would be a second thing to keep in sync.
"""
from __future__ import annotations

import argparse
import csv
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")

#: The main method, as used everywhere else. Mixing strategies here would confound the
#: fleet-size effect with the planner effect, which is the mistake the capacity-regime
#: tests record having made once already.
MAIN = "dijkstra+greedy"

FIELDS = ["n_tasks", "fleet_step", "makespan_mean_s", "on_time_mean", "util_mean",
          "conflict_wait_mean", "s_per_vehicle_saved", "conflict_per_vehicle_added",
          "step_ratio"]


def load_baseline(path=None):
    path = path or os.path.join(RESULTS, "baseline.csv")
    with open(path, encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r["strategy"] == MAIN]


def means_by_fleet(rows, n_tasks):
    """Mean metrics for each fleet size at one task scale."""
    out = {}
    for na in sorted({r["n_agvs"] for r in rows}, key=int):
        sub = [r for r in rows if r["n_tasks"] == n_tasks and r["n_agvs"] == na]
        if not sub:
            continue
        out[int(na)] = {
            "makespan": st.fmean(float(r["makespan_s"]) for r in sub),
            "on_time": st.fmean(float(r["on_time_rate"]) for r in sub),
            "util": st.fmean(float(r["agv_utilization"]) for r in sub),
            "conflict": st.fmean(float(r["conflict_wait_steps"]) for r in sub),
            "n": len(sub),
        }
    return out


def fleet_rows(rows):
    """One row per (task scale, fleet step).

    ``step_ratio`` is the honest headline: it compares the saving per vehicle at this step
    with the saving per vehicle at the previous one. A ratio well below 1 means the step is
    buying less than the last one did, which is what "diminishing returns" actually means.
    """
    out = []
    for n_tasks in sorted({r["n_tasks"] for r in rows}, key=int):
        m = means_by_fleet(rows, n_tasks)
        sizes = sorted(m)
        prev_saving = None
        for i in range(1, len(sizes)):
            lo, hi = sizes[i - 1], sizes[i]
            step = (lo, hi)
            n_added = hi - lo
            saving = (m[lo]["makespan"] - m[hi]["makespan"]) / n_added
            conflict = (m[hi]["conflict"] - m[lo]["conflict"]) / n_added
            ratio = "" if prev_saving in (None, 0) else round(saving / prev_saving, 4)
            out.append({
                "n_tasks": int(n_tasks),
                "fleet_step": "%d->%d" % step,
                "makespan_mean_s": round(m[hi]["makespan"], 2),
                "on_time_mean": round(m[hi]["on_time"], 4),
                "util_mean": round(m[hi]["util"], 4),
                "conflict_wait_mean": round(m[hi]["conflict"], 2),
                "s_per_vehicle_saved": round(saving, 2),
                "conflict_per_vehicle_added": round(conflict, 2),
                "step_ratio": ratio,
            })
            prev_saving = saving
    return out


def write_csv(rows, path):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(RESULTS, "fleet_size.csv"))
    args = ap.parse_args(argv)

    rows = load_baseline()
    if not rows:
        print("baseline.csv has no %s rows; nothing to analyse" % MAIN)
        return 1

    fleet = fleet_rows(rows)
    write_csv(fleet, args.out)

    print("%-7s %-8s %-12s %-12s %-13s %-10s"
          % ("tasks", "step", "makespan", "s/veh saved", "conflict/veh", "ratio"))
    for r in fleet:
        print("%-7d %-8s %-12.1f %-12.1f %-13.1f %-10s"
              % (r["n_tasks"], r["fleet_step"], r["makespan_mean_s"],
                 r["s_per_vehicle_saved"], r["conflict_per_vehicle_added"],
                 r["step_ratio"]))

    # The claim the data supports is about the ratios, so state it from the data rather than
    # asserting it separately in prose that can drift away from the CSV.
    ratios = [r["step_ratio"] for r in fleet if r["step_ratio"] != ""]
    if ratios:
        lo, hi = min(ratios), max(ratios)
        print()
        print("Each fleet step after the first returns %.0f%%-%.0f%% of the previous one."
              % (lo * 100, hi * 100))
        print("That range, not a single optimum, is what this data supports: the "
              "specification")
        print("sets no weighting between makespan and conflict waiting, so no single fleet "
              "size")
        print("can be called best without inventing one.")

    print()
    print("wrote %s" % os.path.relpath(args.out, ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
