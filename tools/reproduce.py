"""One-command reproduction: re-run the experiment scripts and diff the output against the,official CSVs checked into the repository.,,Why this exists,---------------,The thesis claims that all of the code and all of the experimental data are reproducible.,That is a *checkable assertion*, not a rhetorical flourish, and this script is the thing that,checks it. A reproduction claim with no tool behind it is worth nothing.,,Why the scripts cannot simply be re-run,---------------------------------------,Every experiment script hardcodes its output path as `results/xxx.csv`, so running one,directly *overwrites the official data*. The sequence here is:,,1. move `results/*.csv` into `results/_backup/`;,2. run the script, letting it write into `results/`;,3. move the freshly generated files into `results/reproduce/`;,4. move the backup back.,,That way not a single byte of the official data changes, and `results/reproduce/` holds,files that were *computed from scratch*.,,How the comparison is done,--------------------------,Floats are compared with a 1e-6 relative tolerance. Without that the whole thing founders on,`12.3` versus `12.300000000000001`, which is a formatting difference rather than a,disagreement.,,Usage::,,    python tools/reproduce.py --stage quick     # main experiments, about a minute,    python tools/reproduce.py --stage full      # everything, about fifteen minutes,    python tools/reproduce.py --list            # print the plan without running it,"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
REPRO = os.path.join(RESULTS, "reproduce")
BACKUP = os.path.join(RESULTS, "_backup")
EXPS = os.path.join(ROOT, "experiments")

# stage -> [(script, extra_args, description)]
STAGES = {
    "quick": [
        ("run_matrix.py", [], "two planners x 9 cells x 10 seeds (thesis 6.2)"),
        ("run_baseline.py", [], "four baseline strategies compared (thesis 6.8.2)"),
    ],
    "full": [
        ("run_matrix.py", [], "two planners x 9 cells x 10 seeds (6.2)"),
        ("run_baseline.py", [], "four baseline strategies compared (6.8.2)"),
        ("run_baseline_events.py", [], "conflict-wait count definition (6.8.1)"),
        ("run_preempt_tiers.py", [], "three preemption tiers (6.4)"),
        ("run_reschedule.py", [], "urgent-triggered rescheduling (6.4)"),
        ("run_allocation.py", [], "allocation algorithms x dispatch rules (6.8.3)"),
        ("run_diagnosis.py", [], "spacetime A* mechanism diagnosis (6.6)"),
        ("run_load.py", [], "corridor load (6.8.4)"),
        ("run_due_sensitivity.py", [], "due-date definition sensitivity (6.5)"),
        ("run_capacity_regime.py", ["--seeds", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"], "capacity regimes (6.7)"),
        ("run_deadlock.py", [], "deadlock and one-way cycles (6.7)"),
        ("run_layout_portability.py", [], "three layouts (6.9)"),
        ("run_process_time.py", [], "station processing time (6.10)"),
        ("run_baseline_events.py", [], "conflict-wait event definition (6.8.1)"),
        ("run_hub_capacity.py", [], "hub capacity (6.9.5)"),
        ("run_deadlock_predictor.py", [], "deadlock predictor metrics (6.9.6)"),
        ("run_deadlock_threshold.py", [], "deadlock threshold (6.9.6)"),
        ("run_deadlock_transition.py", [], "deadlock transition band (6.9.6)"),
        ("run_due_alpha_policy.py", [], "due-date policy (6.5.2)"),
        ("run_due_alpha_sweep.py", [], "due-date sweep (6.5.1)"),
    ],
    # These seven CSVs were generated before the last change to simulation.py (the
    # round-34 livelock fix, 13:35). They are re-run to establish whether that change
    # moved any of the numbers the thesis cites.
    "stale": [
        ("run_baseline.py", [], "baselines (6.8.2)"),
        ("run_baseline_events.py", [], "conflict-wait count definition (6.8.1)"),
        ("run_diagnosis.py", [], "spacetime A* mechanism diagnosis (6.6)"),
        ("run_allocation.py", [], "allocation algorithms x dispatch rules (6.8.3)"),
        ("run_due_sensitivity.py", [], "due-date definition sensitivity (6.5)"),
        ("run_layout_portability.py", [], "three layouts (6.9)"),
        ("run_process_time.py", [], "station processing time (6.10)"),
    ],
}


def csv_rows(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def same_value(a: str, b: str) -> bool:
    """Compare numerically where possible, as strings otherwise."""
    if a == b:
        return True
    try:
        x, y = float(a), float(b)
    except (TypeError, ValueError):
        return False
    if x == y:
        return True
    scale = max(abs(x), abs(y), 1.0)
    return abs(x - y) / scale < 1e-6


# Columns that depend on the machine, which can never reproduce, so they are skipped.
#
# wall_clock_s is wall-clock duration: the same code on the same machine differs by a few
# milliseconds between runs. Counting it as "reproduction consistency" would make the check
# impossible to pass, and it influences no conclusion either way.
NONDET = {"wall_clock_s"}


def compare(old_path: str, new_path: str):
    """Return (state, message). State is one of same / diff / missing."""
    if not os.path.exists(new_path):
        return "missing", "the re-run produced no file"
    old, new = csv_rows(old_path), csv_rows(new_path)
    if len(old) != len(new):
        return "diff", "row count %d -> %d" % (len(old), len(new))
    for i, (ro, rn) in enumerate(zip(old, new)):
        if set(ro) != set(rn):
            return "diff", "line %d has different columns" % (i + 2)
        for k in ro:
            if k in NONDET:
                continue
            if not same_value(ro[k], rn[k]):
                return "diff", "line %d, %s: %s -> %s" % (i + 2, k, ro[k], rn[k])
    return "same", "%d rows identical" % len(old)


LOCK = os.path.join(RESULTS, ".reproduce.lock")


def acquire_lock():
    """Refuse to let two reproduction runs proceed at once.

    This is not fussiness. The script moves the official CSVs away, runs the scripts, and
    moves them back. With two processes running, the second stashes the files the first has
    just restored, both then fail halfway through, and results/ is left holding a partial
    set. That race actually happened.
    """
    if os.path.exists(LOCK):
        print("Another reproduction run is in progress (%s exists)." % LOCK)
        print("If you are sure nothing is running, delete that lock file and retry.")
        return False
    with open(LOCK, "w", encoding="utf-8") as fh:
        fh.write(str(os.getpid()))
    return True


def release_lock():
    if os.path.exists(LOCK):
        os.remove(LOCK)


def stash_official():
    """Move the official CSVs aside so a script cannot overwrite them."""
    os.makedirs(BACKUP, exist_ok=True)
    moved = []
    for name in os.listdir(RESULTS):
        if not name.endswith(".csv"):
            continue
        src = os.path.join(RESULTS, name)
        dst = os.path.join(BACKUP, name)
        shutil.move(src, dst)
        moved.append(name)
    return moved


def restore_official(moved):
    for name in moved:
        src = os.path.join(BACKUP, name)
        dst = os.path.join(RESULTS, name)
        if os.path.exists(src):
            shutil.move(src, dst)
    if os.path.isdir(BACKUP) and not os.listdir(BACKUP):
        os.rmdir(BACKUP)


def run_stage(items, keep: bool) -> int:
    os.makedirs(REPRO, exist_ok=True)
    moved = stash_official()
    print("Stashed %d official CSVs -> results/_backup/" % len(moved))
    bad = 0
    try:
        for script, extra, desc in items:
            cmd = [sys.executable, os.path.join(EXPS, script)] + extra
            t0 = time.time()
            env = dict(os.environ)
            env["PYTHONIOENCODING"] = "utf-8"
            proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", env=env)
            dt = time.time() - t0
            if proc.returncode != 0:
                print("  [FAILED] %-26s %5.1fs  %s" % (script, dt, desc))
                tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
                for line in tail:
                    print("         " + line)
                bad += 1
                continue
            print("  [done]   %-26s %5.1fs  %s" % (script, dt, desc))
    finally:
        # Move the freshly generated files to reproduce/, then put the official data back.
        for name in list(os.listdir(RESULTS)):
            if name.endswith(".csv"):
                shutil.move(os.path.join(RESULTS, name), os.path.join(REPRO, name))
        restore_official(moved)
        print("Official CSVs restored -> results/")
    return bad


def report(names):
    """Compare file by file and print a table."""
    print("")
    print("%-28s %-8s %s" % ("file", "result", "detail"))
    print("-" * 78)
    same = diff = missing = 0
    for name in sorted(names):
        old = os.path.join(RESULTS, name)
        new = os.path.join(REPRO, name)
        if not os.path.exists(old):
            continue
        state, msg = compare(old, new)
        tag = {"same": "same", "diff": "DIFF", "missing": "MISSING"}[state]
        if state == "same":
            same += 1
        elif state == "diff":
            diff += 1
        else:
            missing += 1
        print("%-28s %-8s %s" % (name, tag, msg))
    print("-" * 78)
    print("%d same, %d differ, %d missing" % (same, diff, missing))
    return diff + missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-run experiments and diff against the official CSVs")
    ap.add_argument("--stage", default="quick", choices=sorted(STAGES),
                    help="quick = main experiments, about a minute; full = everything, about fifteen")
    ap.add_argument("--list", action="store_true", help="print the plan without running it")
    args = ap.parse_args()

    items = STAGES[args.stage]
    print("Stage: %s (%d scripts)" % (args.stage, len(items)))
    for script, extra, desc in items:
        print("  %-26s %s" % (script, desc))
    if args.list:
        return 0

    if not acquire_lock():
        return 2
    t0 = time.time()
    try:
        failed = run_stage(items, keep=True)
    finally:
        release_lock()
    names = [n for n in os.listdir(REPRO) if n.endswith(".csv")]
    problems = report(names)
    print("Took %.1f minutes" % ((time.time() - t0) / 60))

    if failed:
        print("")
        print("%d scripts failed, so the comparison below cannot be trusted." % failed)
        return 1
    if problems:
        print("")
        print("Reproduction incomplete: %d files do not match. The official data was not "
              "modified; the new output is in results/reproduce/." % problems)
        return 1
    print("")
    print("Reproduction passed: every re-run file matches the committed CSV field by field.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
