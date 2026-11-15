"""docs/ 与关键源文件的快照工具。

为什么需要它
--------------
2026-11-03 第 52 轮，重写论文某节时用 `splice(start, end-start)` 定终点，
而查找下一个锚点返回了 -1（那节之后没有兄弟节点）。
`end - start = -1010`，splice 一路删到文件末尾。

论文从约 300 KB 变成 59 KB。没有 git、没有 .md 备份，
靠 1.5 小时前恰好导出的 docx 才救回来——**纯属运气**。

这个脚本就是为了不再靠运气。

用法
----
    python tools/snapshot.py            # 建快照
    python tools/snapshot.py --list     # 列出全部快照
    python tools/snapshot.py --restore 20261103_153000   # 恢复
    python tools/snapshot.py --diff 20261103_153000      # 与当前比较

设计取舍
--------
* 只快照**文本源**（.md/.py/.json/.csv/.txt），不碰 docx/pdf/图片——
  它们体积大且能由源重建。
* 用 zip 单文件存储，一个快照一个文件，便于拷走。
* 不做增量：全量简单可靠，本仓库文本总量只有几 MB。
* 保留最近 N 个（默认 30），避免无限增长。
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import zipfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SNAP_DIR = ROOT / ".snapshots"
KEEP = 30

# 只收这些后缀；docx/pdf/png 等能由源重建的不收
SUFFIXES = {".md", ".py", ".json", ".csv", ".txt", ".yaml", ".yml",
            ".toml", ".cfg", ".ini", ".html", ".js", ".css", ".cs"}

# 这些目录不进快照
SKIP_DIRS = {
    ".snapshots", "__pycache__", ".git", ".pytest_cache", "node_modules",
    ".venv", "results", "out",
    # 第三方 / 移植来的 C# 源码，不是本文档的编辑对象
    "RAWSimO.Core", "RAWSimO.Visualization", "RAWSimO.Playground",
    "RAWSimO.MultiAgentPathFinding", "RAWSimO.DataPreparation",
    "RAWSimO.Live", "RAWSimO.Toolbox", "RAWSimO.CLI", "Tests",
    "MaTiTiny-jenkinssetting1-jenkinsconfig1-0",
    # 2895 个文件的大证据目录：是**采集产物**，不是编辑对象。
    # 它们靠 MANIFEST.json 的指纹来保证不变，不需要进快照。
    "docs/证据",
}

# results/ 单独处理：CSV 要收（是证据），但只收 .csv
KEEP_ALWAYS = {"results"}


def iter_files():
    """遍历需要快照的文件。"""
    for dirpath, dirnames, filenames in os.walk(ROOT):
        rel = Path(dirpath).relative_to(ROOT)
        rel_s = str(rel).replace("\\", "/")
        # 目录剪枝
        dirnames[:] = [
            d for d in dirnames
            if not any((rel_s + "/" + d).lstrip("/").startswith(s)
                       for s in SKIP_DIRS)
            and d not in ("__pycache__", ".git", ".pytest_cache")
        ]
        in_results = rel_s.startswith("results")
        for fn in filenames:
            p = Path(dirpath) / fn
            suf = p.suffix.lower()
            if in_results:
                if suf != ".csv":
                    continue
            elif suf not in SUFFIXES:
                continue
            if p.stat().st_size > 8 * 1024 * 1024:
                continue  # 单个超 8 MB 的不收
            yield p, str(p.relative_to(ROOT)).replace("\\", "/")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


def make_snapshot(tag: str = "") -> Path:
    SNAP_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}{('_' + tag) if tag else ''}.zip"
    out = SNAP_DIR / name
    files = list(iter_files())
    manifest = []
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p, rel in files:
            data = p.read_bytes()
            zf.writestr(rel, data)
            manifest.append(f"{digest(data)}  {len(data):>9}  {rel}")
        zf.writestr("__MANIFEST__.txt",
                    "\n".join(manifest) + f"\n\n共 {len(files)} 个文件\n"
                    + f"生成时间 {stamp}\n")
    size_kb = out.stat().st_size / 1024
    print(f"快照已建：{out.name}  {len(files)} 个文件  {size_kb:.0f} KB")
    prune()
    return out


def prune() -> None:
    snaps = sorted(SNAP_DIR.glob("*.zip"))
    for old in snaps[:-KEEP]:
        old.unlink()
        print(f"  已清理旧快照 {old.name}")


def list_snapshots() -> int:
    snaps = sorted(SNAP_DIR.glob("*.zip"))
    if not snaps:
        print("还没有任何快照。")
        return 0
    print(f"{len(snaps)} 个快照：")
    for s in snaps:
        st = s.stat()
        when = datetime.fromtimestamp(st.st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {s.stem:<28} {when}  {st.st_size / 1024:>7.0f} KB")
    return 0


def restore(stamp: str) -> int:
    """把某个快照的文件写回工作区。**不删除**快照之后新增的文件。"""
    cand = list(SNAP_DIR.glob(f"{stamp}*.zip"))
    if not cand:
        print(f"找不到快照 {stamp}", file=sys.stderr)
        return 1
    snap = cand[0]
    # 恢复前先给当前状态留一份，万一恢复错了还能回来
    make_snapshot(tag="pre-restore")
    n = 0
    with zipfile.ZipFile(snap) as zf:
        for info in zf.infolist():
            if info.filename == "__MANIFEST__.txt":
                continue
            target = ROOT / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(info.filename))
            n += 1
    print(f"已从 {snap.name} 恢复 {n} 个文件。")
    print("注意：快照之后**新建**的文件不会被删除。")
    return 0


def diff(stamp: str) -> int:
    """列出快照与当前工作区的差异（新增 / 删除 / 修改）。"""
    cand = list(SNAP_DIR.glob(f"{stamp}*.zip"))
    if not cand:
        print(f"找不到快照 {stamp}", file=sys.stderr)
        return 1
    snap = cand[0]
    with zipfile.ZipFile(snap) as zf:
        names = [i.filename for i in zf.infolist()
                 if i.filename != "__MANIFEST__.txt"]
        old_hash = {n: digest(zf.read(n)) for n in names}
    now = {}
    for fp, rel in iter_files():
        now[rel] = digest(fp.read_bytes())

    added = sorted(set(now) - set(old_hash))
    removed = sorted(set(old_hash) - set(now))
    changed = sorted(k for k in set(now) & set(old_hash)
                     if now[k] != old_hash[k])

    print(f"对比 {snap.name}：")
    print(f"  新增 {len(added)}  删除 {len(removed)}  修改 {len(changed)}")
    for k in added[:40]:
        print(f"    + {k}")
    for k in removed[:40]:
        print(f"    - {k}")
    for k in changed[:40]:
        print(f"    ~ {k}")
    if len(added) + len(removed) + len(changed) > 120:
        print("    ...（只显示前 40 条）")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="docs/ 与源文件快照")
    ap.add_argument("--list", action="store_true", help="列出全部快照")
    ap.add_argument("--restore", metavar="STAMP", help="恢复到指定快照")
    ap.add_argument("--diff", metavar="STAMP", help="与指定快照比较")
    ap.add_argument("--tag", default="", help="给快照加标签")
    a = ap.parse_args()
    if a.list:
        return list_snapshots()
    if a.restore:
        return restore(a.restore)
    if a.diff:
        return diff(a.diff)
    make_snapshot(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
