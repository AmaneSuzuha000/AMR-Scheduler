"""The delivery status sheet must not silently drift out of date.

`docs/交付状态总表.md` is the one document that answers "is it finished", and it had
rotted badly: it still claimed 22 CSVs / 4560 rows / 363 Python tests / 293 verified
numbers long after the repo had moved to 26 / 4577 / 640 / 535. Nothing was watching it,
so the numbers went stale exactly like the hand-typed Gazebo figures did.

The counters are cheap to compute, so they are checked here rather than trusted:

* CSV count and row total come from `results/MANIFEST.json`;
* the Python test count comes from `pytest --collect-only` (collecting is enough to
  know the number of items and does not recursively run the suite again);
* the retired figures are listed so that a copy-paste cannot bring them back.

The document is a local deliverable (like the thesis markdown) and is absent from a bare
checkout; the tests skip rather than fail in that case.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC = os.path.join(ROOT, "docs", "交付状态总表.md")


def _text() -> str:
    if not os.path.exists(DOC):
        pytest.skip("交付状态总表不在本 checkout 中")
    with open(DOC, encoding="utf-8") as fh:
        return fh.read()


def test_csv_counts_match_the_manifest():
    text = _text()
    with open(os.path.join(ROOT, "results", "MANIFEST.json"), encoding="utf-8") as fh:
        manifest = json.load(fh)

    count = len(manifest)
    rows = sum(int(entry.get("rows", 0)) for entry in manifest.values())

    stated = re.findall(r"(\d+) 份 CSV / (\d+) 行", text)
    assert stated, "表里找不到「N 份 CSV / M 行」的说法"
    assert stated == [(str(count), str(rows))], (
        "表里写的是 %s 份 / %s 行，MANIFEST 实际是 %d 份 / %d 行"
        % (stated[0][0], stated[0][1], count, rows))


def test_python_test_count_matches_the_suite():
    text = _text()
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        stdin=subprocess.DEVNULL,
    )
    total = sum(int(m.group(1)) for m in
                re.finditer(r"^\S+\.py: (\d+)$", proc.stdout, re.MULTILINE))
    assert total > 0, "没能从 pytest --collect-only 读出项数：\n" + proc.stdout[-2000:]

    stated = re.findall(r"\| Python 测试 \| \*\*(\d+)\*\* \|", text)
    assert stated, "表里找不到「| Python 测试 | **N** |」这一行"
    assert stated == [str(total)], (
        "表里写的是 %s 项，实际收集到 %d 项" % (stated[0], total))


# Figures that were true once and are not any more. Keeping them here means a
# copy-paste from an old revision fails immediately instead of shipping.
_STALE = [
    ("22 份 CSV", "已增至 26 份"),
    ("4560 行", "已增至 4577 行"),
    ("363 项测试", "已增至 640 项"),
    ("293 条", "已增至 535 条"),
    ("C# 124", "已增至 150"),
]


def test_status_sheet_has_no_retired_numbers():
    text = _text()
    hits = [(stale, why) for stale, why in _STALE if stale in text]
    assert not hits, "交付状态总表里还留着过期数字：\n" + "\n".join(
        "  「%s」——%s" % (stale, why) for stale, why in hits)
