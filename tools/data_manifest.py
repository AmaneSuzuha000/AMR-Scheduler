"""给 results/*.csv 建立指纹清单，用于发现「数据被静默改动」。

存在理由
--------
论文里的每个结论都挂在 13 份 CSV 上。但这 13 份文件可以被任何一次
误操作覆盖——本项目实际发生过一次：两个复现进程并发，
`results/matrix.csv` 一度消失，`baseline.csv` 混进了重跑值。
当时没有任何自动告警，是靠人工比对发现的。

做法
----
记录每份 CSV 的 sha256、行数、列名，以及若干关键统计量，写进
`results/MANIFEST.json`。之后任何时候跑 `--check`，都能立刻知道
「哪些文件变了、变在哪一列」。

清单本身不含时间戳，这样同样内容重复生成时文件内容一致，便于入库。

用法::

    python tools/data_manifest.py            # 生成/更新清单
    python tools/data_manifest.py --check    # 校验，发现漂移则非零退出
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
MANIFEST = os.path.join(RESULTS, "MANIFEST.json")

#: 墙钟耗时随机器负载变化，不进指纹
NONDET = {"wall_clock_s"}


def digest(path: str) -> str:
    """Compute sha256 normalized for canonical LF line endings across all platforms."""
    with open(path, "rb") as fh:
        raw = fh.read()
    lf_bytes = raw.replace(b"\r\n", b"\n")
    return hashlib.sha256(lf_bytes).hexdigest()[:16]


def describe(path: str) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    cols = list(rows[0].keys()) if rows else []
    stats = {}
    for c in cols:
        if c in NONDET:
            continue
        # Non-finite values are dropped rather than allowed to propagate. A route that did
        # not complete has no accuracy to report and records nan, which is honest per row;
        # but statistics.fmean([0.15, nan]) is nan, so one failed route would turn the whole
        # column's mean into nan and hide any drift in the rows that did complete.
        vals = []
        for r in rows:
            try:
                v = float(r[c])
            except (TypeError, ValueError):
                vals = []
                break
            if math.isfinite(v):
                vals.append(v)
        if vals:
            stats[c] = round(st.fmean(vals), 4)
    return {"sha256": digest(path), "rows": len(rows), "columns": cols, "means": stats}


def build() -> dict:
    out = {}
    for name in sorted(os.listdir(RESULTS)):
        if not name.endswith(".csv") or "v0.3_backup" in name or name.startswith("_"):
            continue
        out[name] = describe(os.path.join(RESULTS, name))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="校验而不是重写")
    args = ap.parse_args()

    cur = build()
    if not args.check:
        with open(MANIFEST, "w", encoding="utf-8", newline="\n") as fh:
            json.dump(cur, fh, ensure_ascii=False, indent=1, sort_keys=True)
            fh.write("\n")
        total = sum(v["rows"] for v in cur.values())
        print("已写入 results/MANIFEST.json：%d 份 CSV，共 %d 行" % (len(cur), total))
        return 0

    if not os.path.exists(MANIFEST):
        print("清单不存在，先跑一次不带 --check 的。")
        return 2
    old = json.load(open(MANIFEST, encoding="utf-8"))
    bad = 0
    for name in sorted(set(old) | set(cur)):
        if name not in cur:
            print("  缺失  %s" % name)
            bad += 1
            continue
        if name not in old:
            print("  新增  %s" % name)
            bad += 1
            continue
        a, b = old[name], cur[name]
        if a["sha256"] == b["sha256"]:
            continue
        bad += 1
        if a["rows"] != b["rows"]:
            print("  改动  %-26s 行数 %d -> %d" % (name, a["rows"], b["rows"]))
            continue
        moved = [k for k in a["means"] if k in b["means"]
                 and abs(a["means"][k] - b["means"][k]) > 1e-4]
        detail = ", ".join("%s %.2f->%.2f" % (k, a["means"][k], b["means"][k])
                             for k in moved[:4])
        print("  改动  %-26s %s" % (name, detail or "内容变了但均值未变"))
    print("")
    if bad:
        print("有 %d 份文件与清单不符。" % bad)
        return 1
    print("全部 %d 份 CSV 与清单一致。" % len(cur))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
