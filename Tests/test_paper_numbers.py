"""论文数字与 CSV 的一致性测试。

这是论文数字核查脚本 verify_numbers.py 的 pytest 包装（该脚本不随公开仓库发布）：让「论文里写的数字」
进入 CI，而不是靠人记得去跑一次核查脚本。

为什么值得做成测试
------------------
第 45 轮发现过 6.9.5 整张表用的是旧种子数据——所有代码测试都绿，
但论文是错的。代码正确 ≠ 论文正确，两者必须分别验证。

为什么直接 import 而不是 subprocess
----------------------------------
最初用 subprocess.run 调脚本，在 Windows 上偶发
`OSError: [WinError 6] 句柄无效`（pytest 捕获输出时与子进程句柄冲突）。
改成把核查逻辑 import 进来跑，绕开子进程，也快得多。
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def test_test_summary_sync_preserves_experiment_decimals_and_bibliography_pages():
    path=ROOT/'docs/证据/G05-传感器定位/synchronize_test_counts.py'
    if not path.exists():
        pytest.skip("synchronize_test_counts.py 不在本 checkout 中")
    spec=importlib.util.spec_from_file_location('test_summary_sync',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text=('1000 项测试全部通过。\n**1000 项测试全部通过**，分 1 个文件：\n'
          '| `test_external_execution.py` | 1000 | 契约 |\n'
          '| **合计** | **1000** | 1 个文件：单元 |\n'
          '测试（1000 项：1 个文件）\n'
          '实验1000.1秒，完成率0.1000，参考1000–1050。另一个值939。\n')
    result=module.synchronize_paper_text(text,{'test_external_execution.py':1005})
    assert result==text.replace('1000 项','1005 项').replace('| 1000 |','| 1005 |').replace('**1000**','**1005**')
    assert module.synchronize_paper_text(result,{'test_external_execution.py':1005})==result


def _load_verifier():
    """按文件路径加载论文数字核查脚本（它不是包）。"""
    path = ROOT / "tools" / "verify_numbers.py"
    spec = importlib.util.spec_from_file_location("_verify_numbers", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(mod, argv):
    """跑一次核查并自己接住输出。

    正文核对是 main() 里的第二阶段，走 capsys 抓不全；
    自己接管 stdout，断言失败时才能看到到底是哪一条缺失。
    """
    buf = io.StringIO()
    old = sys.argv
    sys.argv = ["verify_numbers"] + list(argv)
    try:
        with contextlib.redirect_stdout(buf):
            rc = mod.main()
    finally:
        sys.argv = old
    return rc, buf.getvalue()


def test_paper_numbers_match_csv(capsys):
    """论文正文里登记过的每个数字，都能由 CSV 重算出来。

    公开仓库的精简版不含论文与 tools/verify_numbers.py，
    此时跳过而不是失败。
    """
    if not (ROOT / "tools" / "verify_numbers.py").exists():
        pytest.skip("论文数字核查脚本不在公开仓库中")

    mod = _load_verifier()
    rc = mod.main()
    out = capsys.readouterr().out

    assert rc == 0, "论文数字与 CSV 不一致：\n" + out

    # 至少要有相当数量的断言，防止登记表被误删成空壳还显示通过
    assert "共 " in out, "核查脚本没有输出断言总数：\n" + out
    n = int(out.split("共 ")[1].split(" 条")[0])
    assert n >= 120, f"断言数只有 {n} 条，疑似登记表被截断"


def test_paper_text_contains_the_verified_numbers():
    """正文里必须真的写着这些数字——闭环缺掉的那一半。

    这个测试是一次真实失误换来的。verify_numbers.py 曾长期报「0 条不一致」，
    因为它拿 CSV 去比自己的覆盖表（UPDATED），从来没读过论文正文。
    结果论文 6.10 节整张表、6.3.1 节两行、6.4.3 与 6.6.1 各一个数，
    共 15 个数字已经落后于数据，工具却一直显示通过——
    它在校验自己，不在校验论文。
    """
    if not (ROOT / "tools" / "verify_numbers.py").exists():
        pytest.skip("论文数字核查脚本不在公开仓库中")
    if not (ROOT / "docs" / "论文" / "论文-主文档.md").exists():
        pytest.skip("论文正文不在本 checkout 中")

    mod = _load_verifier()
    rc, out = _run(mod, ["--paper"])

    assert "正文核对" in out, "核查脚本没有执行正文那一半：\n" + out
    assert rc == 0, "论文正文与 CSV 不一致：\n" + out


def test_paper_check_rounds_half_up_like_the_thesis():
    """正文用四舍五入，内置 format() 用四舍六入五取偶，两者在 .5 上会吵起来。

    2800.5 按论文的写法是 2801；用 format(x, ".0f") 会得到 2800。
    搞错这一点，曾经让 23 个正确的数字看起来像过期的。
    """
    if not (ROOT / "tools" / "verify_numbers.py").exists():
        pytest.skip("论文数字核查脚本不在公开仓库中")
    mod = _load_verifier()
    if not hasattr(mod, "_half_up"):
        pytest.skip("该版本核查脚本还没有正文核对功能")
    assert mod._half_up(2800.5, 0) == "2801"
    assert mod._half_up(1.6615, 3) == "1.662"
    assert mod._half_up(3893.7777, 0) == "3894"


def test_thesis_test_count_matches_the_suite():
    """论文里写的测试项数必须等于真实收集到的项数。

    这个数每加一个测试就会变，靠人记必错：本轮修论文时它已经落后两次
    （291 -> 633 -> 635 -> 638）。既然论文声称「每个数字都能回溯」，
    那这个数也得有东西盯着。

    用 --collect-only 而不是真跑一遍：收集阶段就足以确定项数，
    而且不会递归地再跑一次全套测试。
    """
    md_path = ROOT / "docs" / "论文" / "论文-主文档.md"
    if not md_path.exists():
        pytest.skip("论文正文不在本 checkout 中")

    import re
    import subprocess

    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q"],
        cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8",
        stdin=subprocess.DEVNULL,
    )
    # --collect-only -q 每行形如 "Tests/test_x.py: 7"，把每文件项数加起来。
    total = sum(int(m.group(1)) for m in
                re.finditer(r"^\S+\.py: (\d+)$", proc.stdout, re.MULTILINE))
    assert total > 0, "没能从 pytest --collect-only 读出项数：\n" + proc.stdout[-2000:]

    text = md_path.read_text(encoding="utf-8")
    stated = set(int(n) for n in re.findall(r"(\d+) 项测试全部通过", text))
    assert stated, "论文里找不到「N 项测试全部通过」的说法"
    assert stated == {total}, (
        "论文写的测试项数 %s 与实际收集到的 %d 不符" % (sorted(stated), total))

    # 附录 A.3 的逐文件表也要对。它比总数更容易烂：总数只有一处，表有五十多行，
    # 加文件时忘了补一行，总数对得上、表却是旧的（曾落后到 33 行 / 291 项）。
    actual = {m.group(1): int(m.group(2)) for m in
              re.finditer(r"^\S+[\\/](test_\w+\.py): (\d+)$", proc.stdout, re.MULTILINE)}
    assert actual, "没能从 --collect-only 解析出逐文件项数"

    table = {m.group(1): int(m.group(2)) for m in
             re.finditer(r"^\| `(test_\w+\.py)` \| (\d+) \|", text, re.MULTILINE)}
    assert table, "论文里找不到逐文件测试表"

    missing = sorted(set(actual) - set(table))
    extra = sorted(set(table) - set(actual))
    wrong = ["%s: 表中 %d，实际 %d" % (f, table[f], actual[f])
             for f in sorted(set(actual) & set(table)) if table[f] != actual[f]]
    assert not (missing or extra or wrong), (
        "逐文件测试表与实际不符："
        + ("\n  缺行: " + ", ".join(missing) if missing else "")
        + ("\n  多行: " + ", ".join(extra) if extra else "")
        + ("\n  项数不符: " + "; ".join(wrong) if wrong else ""))

    assert sum(table.values()) == total, (
        "逐文件表合计 %d，与实际 %d 不符" % (sum(table.values()), total))

def _strip_blockquotes(text):
    """去掉引用块，只留下「当前有效」的正文。

    这篇论文里 `>` 引用块一律是方法论备注、口径说明与修正记录，从不承载结果——
    结果都在表格和正文里。而修正记录必须保留旧值才讲得清错在哪（6.3.1 记了
    −12.2%/−14.1%/−17.3% 的来历，6.2 记了符号检验的 0.1094 与 Wilcoxon 的差别）。
    把这些块挖掉再查，护栏才不会逼人删掉最有价值的复盘。
    """
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith(">"))


#: 已经修掉的过期字面量。它们每一个都曾经真实出现在论文里，
#: 并且因为登记表只认识「代码算出来的值」而长期无人发现。
#: 这里存的是原文片段而不是裸数字：像 0.0020 这种值在别处是合法的
#: （100/4 的 Wilcoxon 恰好就是 0.0020），裸数字黑名单会误伤。
_STALE = [
    # --- 6.10 加工时间表：整张表落后一轮数据 ---
    ("3887", "6.10.2 200s 完成时间，应为 3894"),
    ("3892", "6.10.2 200s 完成时间，应为 3901"),
    ("4013", "6.10.2 200s 完成时间，应为 4012"),
    ("5181", "6.10.2 300s 完成时间，应为 5198"),
    ("8476", "6.10.2 600s 完成时间，应为 8500"),
    # --- 6.3.1 枢纽：用了「全部运行」中位数，违反本节自己声明的「只算全完成」 ---
    ("5958", "6.3.1 hub/50/6 完成时间，应为 6269"),
    ("11713", "6.3.1 hub/100/6 完成时间，应为 13076"),
    ("−14.1%", "6.3.1 hub 边际，应为 −9.6%"),
    ("−17.3%", "6.3.1 hub 边际，应为 −7.6%"),
    # --- 6.4 抢占：p 值来自旧数据 ---
    ("0.2402", "6.4.3 总准时率，应为 0.2383"),
    ("0.529 → 0.725", "6.4 加急准时率，应为 0.531 → 0.712"),
    ("145.8 → 81.5", "6.4 加急响应，应为 145.7 → 85.5"),
    ("0.238 → 0.257", "6.4 总准时率，应为 0.229 → 0.253"),
    ("0.5104", "6.4.2 完成时间 p，应为 0.5719"),
    ("0.7590", "6.4.2 完成时间 p，应为 0.5719"),
    ("0.0091", "6.4.3 总准时率 p，应为 0.0018"),
    ("（p = 0.0021）", "6.4.3 完成时间 p，应为 <0.0001"),
    ("（p = 0.0006）", "6.4.3 总准时率 p，应为 0.0018"),
    ("（p = 0.0003）", "6.4.3 加急准时率 p，应为 <0.0001"),
    ("68 / 0（其余 22 相同）", "6.4.3 物料浪费好坏写反，应为 0 / 68"),
    # --- 6.2 每格 p：这一列原来是精确符号检验，与声明的 Wilcoxon 不是一个检验 ---
    ("0.7539", "6.2 p 列旧符号检验值"),
    ("0.1094", "6.2 p 列旧符号检验值"),
    ("0.0215", "6.2 p 列旧符号检验值"),
    # --- 6.2 / 6.9.2 内部矛盾 ---
    ("6.34%", "6.2 最大变慢，应为 5.84%"),
    ("−14.6%", "6.9.2 网格差距，与本节表格的 −16.9% 自相矛盾"),
    ("8 格更慢（最多 5.58%）", "摘要引用 6.2 时用了 6.8.2 的数"),
    # --- 机理解释 ---
    ("冲突等待增加 48%", "应为 35.7%"),
    ("冲突等待增加 33%", "应为 35.7%"),
    ("里程差仅 0.3%", "应为 0.01%"),
    # --- 规模计数 ---
    ("363 项测试", "应为 835 项测试"),
    ("291 项测试", "应为 835 项测试"),
    ("638 项测试", "应为 835 项测试"),
    ("640 项测试", "应为 835 项测试"),
    ("645 项测试", "应为 835 项测试"),
    ("674 项测试", "应为 835 项测试"),
    ("710 项测试", "应为 835 项测试"),
    ("714 项测试", "应为 835 项测试"),
    ("715 项测试", "应为 835 项测试"),
    ("716 项测试", "应为 835 项测试"),
    ("719 项测试", "应为 835 项测试"),
    ("720 项测试", "应为 835 项测试"),
    ("721 项测试", "应为 835 项测试"),
    ("722 项测试", "应为 835 项测试"),
    ("723 项测试", "应为 835 项测试"),
    ("725 项测试", "应为 835 项测试"),
    ("726 项测试", "应为 835 项测试"),
    ("724 项测试", "应为 835 项测试"),
    ("713 项测试", "应为 835 项测试"),
    ("52 个文件", "应为 57 个文件"),
    ("55 个文件", "应为 57 个文件"),
    ("4560 行", "应为 4872 行"),
    ("4280 行", "应为 4872 行"),
    ("4577 行", "应为 4872 行"),
    ("4625 行", "应为 4872 行"),
    ("4643 行", "应为 4872 行"),
    ("4637 行", "应为 4872 行"),
    ("4689 行", "应为 4872 行"),
    ("4761 行", "应为 4872 行"),
    ("4791 行", "应为 4872 行"),
    ("4836 行", "应为 4872 行"),
    ("4860 行", "应为 4872 行"),
    ("4749 行", "应为 4872 行"),
    ("22 份 CSV", "应为 34 份 CSV"),
    ("18 份 CSV", "应为 34 份 CSV"),
    ("24 份 CSV", "应为 34 份 CSV"),
    ("26 份 CSV", "应为 34 份 CSV"),
    ("27 份 CSV", "应为 34 份 CSV"),
    ("28 份 CSV", "应为 34 份 CSV"),
    ("29 份 CSV", "应为 34 份 CSV"),
    ("30 份 CSV", "应为 34 份 CSV"),
    ("31 份 CSV", "应为 34 份 CSV"),
    ("32 份 CSV", "应为 34 份 CSV"),
    ("33 个文件", "应为 53 个文件"),
]


def test_thesis_has_no_known_stale_numbers():
    """修掉过的过期数字不许再回来。

    这一条是补前一条的漏洞。test_paper_text_contains_the_verified_numbers 只保证
    「登记过的值确实写在正文里」，它管不到那些根本没登记的句子——第七章的结论段
    和摘要曾经各有一份独立的数字副本，正文改了、副本没改，两处都还写着旧值。
    黑名单把「已经判定为错的值」本身变成断言，副本再漂移就会红。
    """
    if not (ROOT / "tools" / "verify_numbers.py").exists():
        pytest.skip("论文数字核查脚本不在公开仓库中")
    mod = _load_verifier()
    text = mod._paper_text()
    if not text:
        pytest.skip("论文正文不在本 checkout")

    # 引用块会**故意**引用旧值来讲清楚错在哪（6.3.1 与 6.2 各有一段）。
    # 那是记录，不是残留。
    body = _strip_blockquotes(text)

    # 答辩 PPT 的正文与备注来自 tools/build_slides.py，讲稿又由 build_script.py
    # 从备注导出。同一批数字在这里有第三、第四份副本，一并盯住。
    extra = [ROOT / "tools" / "build_slides.py", ROOT / "docs" / "答辩" / "答辩讲稿.md"]
    sources = [("论文", body)]
    for path in extra:
        if path.exists():
            sources.append((path.name, path.read_text(encoding="utf-8")))

    hits = []
    for label, raw in sources:
        normalised = _strip_blockquotes(raw).replace("\u2212", "-")
        for stale, why in _STALE:
            if stale in normalised or stale.replace("−", "-") in normalised:
                hits.append(f"[{label}] {stale!r} —— {why}")
    assert not hits, "还有已判定过期的数字：\n  " + "\n  ".join(hits)
