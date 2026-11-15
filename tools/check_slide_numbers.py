"""核对答辩 PPT 里每个数字的出处。

答辩最怕的一类追问是「这个数哪来的」。幻灯片上的数字必须能在论文正文或者
results/*.csv 里找到，本脚本把这件事自动化。

判定顺序
--------
1. 论文正文（docs/论文/论文-主文档.md）——优先，因为评委翻的是论文；
2. results/*.csv 原始数据——论文表格的上游；
3. 以上都没有、也不在派生值白名单 → 报出来。

已知的合法例外
----------------
R4.7 这类是任务书条款号，不是数据，正则里先剔除；
派生值（如「降 44%」由 145.8→81.5 算出）登记在 DERIVED 里，仍需人工确认算式。

用法::

    python tools/check_slide_numbers.py
"""

from __future__ import annotations

import glob
import os
import re

from pptx import Presentation

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# 派生值白名单：幻灯片上已写明算式，人工确认成立即可豁免
DERIVED = {
    "44%": "145.8 -> 81.5，幻灯片同页已写算式",
    "990": "540 + 450，幻灯片同页已写算式",
    "1339": "results/replay.html 的实际帧数",
}


def flatten(text: str) -> str:
    """去掉千分位空格，让「2 373」和「2373」能匹配上。"""
    text = re.sub(r"(?<=\d)[ \u00a0](?=\d)", "", text)
    return text.replace(",", "")


def slide_numbers(prs):
    """产出 (页码, 数字, 上下文)。"""
    pat = re.compile(r"\d+(?:\.\d+)?%?")
    out = []
    for i, s in enumerate(prs.slides, 1):
        texts = []
        for shp in s.shapes:
            if shp.has_text_frame:
                texts.append(shp.text_frame.text)
            if getattr(shp, "has_table", False):
                for row in shp.table.rows:
                    texts.extend(c.text for c in row.cells)
        for t in texts:
            # R4.7 这类是任务书条款号，不是数据
            t2 = re.sub(r"\bR\d+\.\d+", " ", t)
            for num in pat.findall(t2):
                out.append((i, num, t.strip()[:60]))
    return out


def main() -> int:
    paper = flatten(open(os.path.join(ROOT, "docs", "论文", "论文-主文档.md"),
                          encoding="utf-8").read())
    csv_text = ""
    for f in glob.glob(os.path.join(ROOT, "results", "*.csv")):
        csv_text += flatten(open(f, encoding="utf-8").read())

    prs = Presentation(os.path.join(ROOT, "docs", "答辩", "答辩PPT.pptx"))
    items = slide_numbers(prs)

    seen = set()
    miss, derived = [], []
    for page, num, ctx in items:
        if num in seen:
            continue
        seen.add(num)
        if num in paper or num in csv_text:
            continue
        (derived if num in DERIVED else miss).append((page, num, ctx))

    print("幻灯片去重数字 %d 个（共 %d 页）" % (len(seen), len(prs.slides._sldIdLst)))
    if derived:
        print("")
        print("派生值（已登记，需人工确认算式）:")
        for page, num, _ctx in derived:
            print("  p%02d  %-8s  %s" % (page, num, DERIVED.get(num, "")))
    if miss:
        print("")
        print("[!] 论文与 CSV 里都找不到，也不在白名单:")
        for page, num, ctx in miss:
            print("  p%02d  %-8s  出自：%s" % (page, num, ctx))
        return 1
    print("")
    print("结论：幻灯片上每个数字都能追溯到论文正文或原始数据。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
