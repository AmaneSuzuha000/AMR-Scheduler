"""论文内部重复自检（查重前的自查，不等于知网查重）。

要回答的问题
--------------
同一份文稿内部有多少字是**自我重复**的？
这类重复有两个后果：
  1. 部分学校的校内查重会把跨章节自我重复计入复制比；
  2. 就算不计入，读起来也是车轱辘话。

方法
----
对清洗后的正文取字符级 13-gram（知网较早版本的检测单元就是「连续 13 字」），
把命中 >=2 次的 n-gram 合并成极大重复段，再按下式估算自重复率：

    自重复率 = 落在重复段内的字符数 / 正文字符数

只统计正文，排除：代码块、表格、图片行、参考文献（按引用规范本来就会重复题名）、
以及标题行。

用法::

    python experiments/check_self_similarity.py
    python experiments/check_self_similarity.py --min-len 10
"""

from __future__ import annotations

import argparse
import os
import re
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER = os.path.join(ROOT, "docs", "论文", "论文-主文档.md")

CJK = re.compile(r"[\u4e00-\u9fff]")
STATUS_EMOJI = ("✅", "🟡", "⬜", "❌")
SEP = "\x00"  # 段落分隔符：n-gram 不跨段，避免把两段拼起来造成假重复


def load_prose(path: str) -> list[tuple[int, str]]:
    """返回 [(起始行号, 段落文本)]。

    段落 = 连续的非空正文行（Markdown 里一段会折成多行）。
    剔除：代码块、表格、图片、标题、引用块、图注、以及**状态标记行**
    （`✅ 定稿（数据：...）` 是写作过程标记，不会进交付稿，
    留着会与正文粘连、制造假的重复段）。
    """
    raw = open(path, encoding="utf-8").read().splitlines()
    out: list[tuple[int, str]] = []
    buf: list[str] = []
    start = 0
    in_fence = False
    in_refs = False

    def flush() -> None:
        nonlocal buf, start
        if buf:
            out.append((start, " ".join(buf)))
            buf = []

    for i, line in enumerate(raw, 1):
        s = line.strip()
        if s.startswith("```"):
            in_fence = not in_fence
            flush()
            continue
        if in_fence:
            continue
        if s.startswith("# 参考文献"):
            in_refs = True
            flush()
            continue
        if s.startswith("# 致谢"):
            in_refs = False
        if in_refs:
            continue
        skip = (not s or s.startswith("#") or s.startswith("|") or s.startswith("![")
                or s.startswith(">") or s.startswith("<p align") or s.startswith(STATUS_EMOJI))
        if skip:
            flush()
            continue
        if not buf:
            start = i
        buf.append(s)
    flush()
    return out

def normalize(text: str) -> str:
    """去掉 Markdown 标记与空白，只留可比较的字符。"""
    text = re.sub(r"\*\*|`|\$", "", text)
    text = re.sub(r"\s+", "", text)
    text = re.sub(r"[\*\-]{2,}", "", text)
    return text



def build_text(paras: list[tuple[int, str]]) -> tuple[str, list[int]]:
    """拼成一条长串，段落之间插 SEP；同时记录每个字符来自 Markdown 的第几行。"""
    parts: list[str] = []
    owner: list[int] = []
    for ln, text in paras:
        n = normalize(text)
        if not n:
            continue
        if parts:
            parts.append(SEP)
            owner.append(ln)
        parts.append(n)
        owner.extend([ln] * len(n))
    return "".join(parts), owner

def find_duplicated(full: str, min_len: int) -> list[bool]:
    """标记每个字符是否落在「至少出现两次的 min_len 字窗口」内。"""
    covered = [False] * len(full)
    index: dict[str, list[int]] = defaultdict(list)
    for i in range(len(full) - min_len + 1):
        index[full[i:i + min_len]].append(i)
    for positions in index.values():
        if len(positions) < 2:
            continue
        for start in positions:
            for k in range(start, start + min_len):
                covered[k] = True
    return covered


def merge_runs(covered: list[bool]) -> list[tuple[int, int]]:
    """把连续的已覆盖字符合并成区间。"""
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(covered):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i));
            start = None
    if start is not None:
        runs.append((start, len(covered)))
    return runs


def count_cjk(s: str) -> int:
    return len(CJK.findall(s))


def main() -> int:
    ap = argparse.ArgumentParser(description="论文内部重复自检")
    ap.add_argument("--min-len", type=int, default=13, help="重复窗口长度（字），默认 13")
    ap.add_argument("--top", type=int, default=25, help="打印最长的多少段")
    ap.add_argument("--min-report", type=int, default=20, help="只报告不短于此长度的重复段")
    args = ap.parse_args()

    lines = load_prose(PAPER)
    full, owner = build_text(lines)
    total = len(full.replace(SEP, ""))
    if total == 0:
        print("正文为空")
        return 1

    covered = find_duplicated(full, args.min_len)
    runs = merge_runs(covered)
    long_runs = [r for r in runs if r[1] - r[0] >= args.min_report]
    dup_chars = sum(len(full[r[0]:r[1]].replace(SEP, "")) for r in runs)

    print("=" * 72)
    print("论文内部重复自检（窗口 %d 字，报告阈值 %d 字）" % (args.min_len, args.min_report))
    print("=" * 72)
    print("正文总字数（去标记、去空白）: %d" % total)
    print("落入重复段的字数: %d" % dup_chars)
    print("自重复率: %.2f%%" % (100.0 * dup_chars / total))
    print("重复段数: %d（其中 >= %d 字的 %d 段）" % (len(runs), args.min_report, len(long_runs)))
    print()

    # 按长度排序，输出最长的若干段
    long_runs.sort(key=lambda r: r[1] - r[0], reverse=True)
    seen: set[str] = set()
    shown = 0
    for start, end in long_runs:
        if shown >= args.top:
            break
        snippet = full[start:end]
        key = snippet[:40]
        if key in seen:
            continue
        seen.add(key)

        # 找出该段在全文的所有出现位置
        places: list[int] = []
        cursor = full.find(snippet)
        while cursor != -1:
            places.append(cursor)
            cursor = full.find(snippet, cursor + 1)
        where = sorted({owner[min(q, total - 1)] for q in places})

        shown += 1
        print("%2d. %d 字 × %d 处，行号 %s" % (shown, end - start, len(places), where[:6]))
        print("    %s" % snippet[:120])
        print()

    # 按章节汇总，定位问题最集中的地方
    print("-" * 72)
    print("按重复字数排前 10 的章节（用最近的 Markdown 一级/二级标题归属）:")
    heads: dict[int, str] = {}
    cur = "（前言）"
    for i, raw in enumerate(open(PAPER, encoding="utf-8").read().splitlines(), 1):
        if re.match(r"^#{1,2} ", raw.strip()):
            cur = raw.strip().lstrip("# ").strip()
        heads[i] = cur
    per_section: dict[str, int] = defaultdict(int)
    for start, end in runs:
        per_section[heads.get(owner[start], "（未知）")] += end - start
    for name, cnt in sorted(per_section.items(), key=lambda kv: -kv[1])[:10]:
        print("  %6d 字  %s" % (cnt, name))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
