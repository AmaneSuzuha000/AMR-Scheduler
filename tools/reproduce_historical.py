"""Reproduce historical CSVs in a new directory using their frozen source version."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "experiments/reference/amr-study-f62c064.zip"
ARCHIVE_SHA256 = "66888c0db6a7cf746f6b38df6d3a42d3caef46ed6f0843bf54bb3bbd09f8ca88"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main(argv=None):
    # Import the existing stage list and comparison rule; no experiments run on import.
    sys.path.insert(0, str(ROOT))
    from tools.reproduce import STAGES, compare

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="quick", choices=sorted(STAGES))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    items = STAGES[args.stage]
    if args.list:
        for script, _, description in items:
            print(f"{script}: {description}")
        return 0
    if sha256(ARCHIVE) != ARCHIVE_SHA256:
        raise RuntimeError("historical study source archive mismatch")
    output = args.output or ROOT / "results/reproduce" / time.strftime("legacy_%Y%m%d_%H%M%S")
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    before = {path.name: sha256(path) for path in (ROOT / "results").glob("*.csv")}
    failures = []
    with tempfile.TemporaryDirectory(prefix="amr_legacy_replay_") as temporary:
        workspace = Path(temporary)
        with zipfile.ZipFile(ARCHIVE) as archive:
            archive.extractall(workspace)
        (workspace / "results").mkdir()
        for index, (script, extra, description) in enumerate(items, 1):
            print(f"{index}/{len(items)}: {script}: {description}", flush=True)
            with (output / f"{index:02d}_{Path(script).stem}.log").open("x", encoding="utf-8") as log:
                result = subprocess.run(
                    [sys.executable, "-X", "utf8", str(workspace / "experiments" / script), *extra],
                    cwd=workspace, stdout=log, stderr=subprocess.STDOUT,
                )
            if result.returncode:
                failures.append({"script": script, "exit_code": result.returncode})
        comparisons = []
        for new in sorted((workspace / "results").glob("*.csv")):
            destination = output / new.name
            destination.write_bytes(new.read_bytes())
            original = ROOT / "results" / new.name
            if original.exists():
                state, detail = compare(str(original), str(destination))
            else:
                state, detail = "missing", "original reference CSV is absent"
            comparisons.append({"file": new.name, "result": state, "detail": detail,
                                "sha256": sha256(destination)})
    after = {path.name: sha256(path) for path in (ROOT / "results").glob("*.csv")}
    report = {
        "source_commit": "f62c0641bd0d56c86ddcf21d06e725c75d5b6a0f",
        "archive_sha256": ARCHIVE_SHA256, "stage": args.stage,
        "original_csv_before": before, "original_csv_after": after,
        "original_csv_unchanged": before == after,
        "script_failures": failures, "comparisons": comparisons,
        "passed": bool(comparisons) and not failures and before == after
                  and all(row["result"] == "same" for row in comparisons),
        "scope": "Historical source and CSV reproduction. No current solver or physical acceptance.",
    }
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "passed": report["passed"],
                      "comparisons": comparisons}, ensure_ascii=False), flush=True)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())