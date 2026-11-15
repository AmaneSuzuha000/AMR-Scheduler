"""把 docs/论文/论文-主文档.md 转成可提交的 .docx（中文毕业论文常用格式）。

设计原则
--------
1. **不改源文件**。Markdown 仍是唯一真源，docx 是派生产物。
2. 学校模板尚未拿到，所以这里只上「通用中文学位论文格式」：
   A4、宋体小四、1.5 倍行距、首行缩进 2 字符、黑体各级标题、三线表。
   模板到了以后只需改本文件的 STYLE 常量，正文不用重写。
3. 状态标记（✅/🟡/⬜）在 docx 里去掉——那是写作过程标记，不该出现在交付稿。

用法::

    python tools/build_docx.py
"""

from __future__ import annotations

import os
import re
import sys

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "docs", "论文", "论文-主文档.md")
OUT = os.path.join(ROOT, "docs", "论文", "论文-主文档.docx")

# ---- 格式常量：学校模板下来后只改这里 ----
BODY_FONT = "宋体"
HEAD_FONT = "黑体"
CODE_FONT = "Consolas"
BODY_SIZE = Pt(12)        # 小四
TABLE_SIZE = Pt(10.5)     # 五号
CAPTION_SIZE = Pt(10.5)


def _set_cjk(run, font_name: str) -> None:
    """python-docx 只设 latin 字体，中文要单独写 eastAsia。"""
    run.font.name = font_name
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:eastAsia"), font_name)
    rfonts.set(qn("w:ascii"), font_name)
    rfonts.set(qn("w:hAnsi"), font_name)


def _first_line_indent_chars(p, chars: float = 2.0, size_pt: float = 12.0) -> None:
    """首行缩进按「字符」计（w:firstLineChars），这是中文学位论文的通行做法。"""
    ppr = p._p.get_or_add_pPr()
    ind = ppr.find(qn("w:ind"))
    if ind is None:
        ind = OxmlElement("w:ind")
        ppr.append(ind)
    ind.set(qn("w:firstLineChars"), str(int(chars * 100)))
    ind.set(qn("w:firstLine"), str(int(chars * size_pt * 20)))


def _add_field(run, instr: str) -> None:
    """插入 Word 域（用于自动目录与页码）。"""
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr_el = OxmlElement("w:instrText")
    instr_el.set(qn("xml:space"), "preserve")
    instr_el.text = instr
    fld_sep = OxmlElement("w:fldChar")
    fld_sep.set(qn("w:fldCharType"), "separate")
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr_el)
    run._r.append(fld_sep)
    run._r.append(fld_end)


def _page_number_footer(section) -> None:
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    _set_cjk(run, BODY_FONT)
    run.font.size = Pt(10.5)
    _add_field(run, "PAGE")


def _repeat_header_and_keep_rows(table) -> None:
    """表头跨页重复（tblHeader），且单行不被页面切成两半（cantSplit）。

    不做这两件事，长表格跨页时表头会丢，读者看到的是「值）」这种残缺行。
    """
    if not table.rows:
        return
    trpr = table.rows[0]._tr.get_or_add_trPr()
    hdr = OxmlElement("w:tblHeader")
    hdr.set(qn("w:val"), "true")
    trpr.append(hdr)
    for row in table.rows:
        rpr = row._tr.get_or_add_trPr()
        cant = OxmlElement("w:cantSplit")
        cant.set(qn("w:val"), "true")
        rpr.append(cant)


def _three_line_borders(table) -> None:
    """三线表：只有顶线、表头下横线、底线。"""
    tbl = table._tbl
    tblpr = tbl.tblPr
    borders = OxmlElement("w:tblBorders")
    for edge, sz in (("top", "12"), ("bottom", "12")):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "single")
        el.set(qn("w:sz"), sz)
        el.set(qn("w:color"), "000000")
        borders.append(el)
    for edge in ("left", "right", "insideH", "insideV"):
        el = OxmlElement("w:" + edge)
        el.set(qn("w:val"), "none")
        borders.append(el)
    tblpr.append(borders)
    # 表头行下加一条细线
    if table.rows:
        for cell in table.rows[0].cells:
            tcpr = cell._tc.get_or_add_tcPr()
            tcb = OxmlElement("w:tcBorders")
            bot = OxmlElement("w:bottom")
            bot.set(qn("w:val"), "single")
            bot.set(qn("w:sz"), "6")
            bot.set(qn("w:color"), "000000")
            tcb.append(bot)
            tcpr.append(tcb)


MATH_SYMBOLS = {
    r"\le": "≤", r"\leq": "≤", r"\ge": "≥", r"\geq": "≥",
    r"\neq": "≠", r"\ne": "≠", r"\approx": "≈",
    r"\in": "∈", r"\notin": "∉", r"\forall": "∀", r"\exists": "∃",
    r"\Longrightarrow": "⟹", r"\Rightarrow": "⇒", r"\rightarrow": "→", r"\to": "→",
    r"\times": "×", r"\cdot": "·", r"\pm": "±", r"\infty": "∞",
    r"\sum": "Σ", r"\prod": "Π", r"\int": "∫",
    r"\max": "max", r"\min": "min", r"\log": "log", r"\exp": "exp", r"\arg": "arg",
    r"\equiv": "≡", r"\Delta": "Δ", r"\delta": "δ", r"\Omega": "Ω",
    r"\sim": "~", r"\propto": "∝", r"\cup": "∪", r"\cap": "∩",
    r"\subset": "⊂", r"\emptyset": "∅", r"\dots": "…", r"\ldots": "…",
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\sigma": "σ",
    r"\mu": "μ", r"\lambda": "λ", r"\theta": "θ", r"\pi": "π",
    r"\{": "\x01", r"\}": "\x02",   # 占位，最后再还原，免得被当成分组花括号删掉
    r"\_": "_", r"\%": "%", r"\*": "*",
    r"\quad": "    ", r"\qquad": "        ", r"\,": " ", r"\;": " ", r"\ ": " ",
}

# 允许一层嵌套的花括号组，用于 \text{中文} 与 \underbrace{a}_{b} 这类参数
G = r"\{((?:[^{}]|\{[^{}]*\})*)\}"

# 结构性命令：必须在符号表之前处理，否则 \le 会把 \left 改成「≤ft」
STRUCT_RE = [
    (re.compile(r"\\left\s*\.?"), ""),
    (re.compile(r"\\right\s*\.?"), ""),
    (re.compile(r"\\(?:text|mathrm|mathbb|operatorname|textrm)\s*" + G), r"\1"),
    (re.compile(r"\\underbrace\s*" + G + r"\s*_\s*" + G), r"\1（\2）"),
    (re.compile(r"\\frac\s*" + G + r"\s*" + G), r"(\1)/(\2)"),
    (re.compile(r"\\sqrt\s*" + G), r"√(\1)"),
    (re.compile(r"\\overline\s*" + G), "\\1\u0304"),
    (re.compile(r"\\bar\s*" + G), "\\1\u0304"),
]

def latex_to_text(expr: str) -> str:
    """把 LaTeX 线性化成可读的 Unicode 文本。

    不追求排版精度，只求「读者能看懂、不出现反斜杠」。
    复杂公式的最终排版仍需 Word 公式编辑器，这一点在文档里写明。
    """
    for pattern, repl in STRUCT_RE:
        expr = pattern.sub(repl, expr)
    # 长的命令先替换，避免 \le 抢走 \leq 的前缀
    for key in sorted(MATH_SYMBOLS, key=len, reverse=True):
        # 只有以字母结尾的命令才需要「后面不是字母」的前瞻，
        # 否则 \{a 这种会被误判成更长的命令
        suffix = r"(?![a-zA-Z])" if key[-1].isalpha() else ""
        expr = re.sub(re.escape(key) + suffix, lambda _m, v=MATH_SYMBOLS[key]: v, expr)
    # 上下标的大括号保留成括号，避免 Σ_i ∈ T 这种歧义
    expr = re.sub(r"_\{([^{}]*)\}", r"_(\1)", expr)
    expr = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", expr)
    expr = re.sub(r"\\[a-zA-Z]+", "", expr)
    expr = expr.replace("{", "").replace("}", "")
    expr = expr.replace("\x01", "{").replace("\x02", "}")   # 还原 \{ \} 的字面花括号
    expr = re.sub(r"[ \t]+", " ", expr)
    return expr.strip()

STATUS_EMOJI = ("✅", "🟡", "⬜", "❌", "🔴")


def _strip_status(text: str) -> str:
    """去掉写作过程用的状态标记，只留正文。"""
    for e in STATUS_EMOJI:
        text = text.replace(e, "")
    text = re.sub(r"\*\*\s*\*\*", "", text)
    return text.strip()


INLINE_RE = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\$[^$]+\$)")


CJK_RE = re.compile(r"[\u3000-\u303f\u4e00-\u9fff\uff00-\uffef]")


def smart_join(parts) -> str:
    """拼接折行时按中英文分别处理：

    中文之间不插空格（插了会在行尾悬挂、还会让 PDF 文本抽取多出空格），
    英文/数字之间插一个空格。
    """
    out = ""
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if not out:
            out = p
            continue
        prev, nxt = out[-1], p[0]
        if CJK_RE.match(prev) or CJK_RE.match(nxt):
            out += p
        else:
            out += " " + p
    return out


def add_runs(p, text: str, size, font: str = BODY_FONT, bold_all: bool = False) -> None:
    text = _strip_status(text)
    if not text:
        return
    # 转义的 \* \_ \% 先换成哨兵，否则 `**时空 A\***` 会被 ** 分割器从中间切断
    for raw, sentinel in (("\\*", "\x03"), ("\\_", "\x04"), ("\\%", "\x05")):
        text = text.replace(raw, sentinel)
    for part in INLINE_RE.split(text):
        if not part:
            continue
        bold = bold_all
        italic = False
        mono = False
        if part.startswith("**") and part.endswith("**") and len(part) > 4:
            part = part[2:-2]; bold = True
        elif part.startswith("`") and part.endswith("`") and len(part) > 2:
            part = part[1:-1]; mono = True
        elif part.startswith("$") and part.endswith("$") and len(part) > 2:
            part = latex_to_text(part[1:-1]); italic = True
        part = part.replace("\x03", "*").replace("\x04", "_").replace("\x05", "%")
        run = p.add_run(part)
        _set_cjk(run, CODE_FONT if mono else font)
        run.font.size = size
        run.bold = bold
        run.italic = italic


def _setup_styles(doc: Document) -> None:
    st = doc.styles["Normal"]
    st.font.name = BODY_FONT
    st.font.size = BODY_SIZE
    st.element.rPr.rFonts.set(qn("w:eastAsia"), BODY_FONT)
    pf = st.paragraph_format
    pf.line_spacing = 1.5
    pf.space_before = Pt(0)
    pf.space_after = Pt(0)

    spec = {
        "Heading 1": (16, WD_ALIGN_PARAGRAPH.CENTER, 18, 12),
        "Heading 2": (14, WD_ALIGN_PARAGRAPH.LEFT, 12, 6),
        "Heading 3": (12, WD_ALIGN_PARAGRAPH.LEFT, 10, 6),
        "Heading 4": (12, WD_ALIGN_PARAGRAPH.LEFT, 6, 6),
    }
    for name, (size, align, before, after) in spec.items():
        s = doc.styles[name]
        s.font.name = HEAD_FONT
        s.font.size = Pt(size)
        s.font.bold = True
        s.font.color.rgb = RGBColor(0, 0, 0)
        s.element.rPr.rFonts.set(qn("w:eastAsia"), HEAD_FONT)
        s.paragraph_format.alignment = align
        s.paragraph_format.space_before = Pt(before)
        s.paragraph_format.space_after = Pt(after)
        s.paragraph_format.line_spacing = 1.5
        s.paragraph_format.first_line_indent = None


def _build_cover(doc: Document, title: str) -> None:
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("本科毕业设计（论文）")
    _set_cjk(r, HEAD_FONT); r.font.size = Pt(22); r.bold = True

    for _ in range(2):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(title)
    _set_cjk(r, HEAD_FONT); r.font.size = Pt(26); r.bold = True

    for _ in range(6):
        doc.add_paragraph()

    rows = [
        ("学　　院", ""), ("专　　业", ""), ("班　　级", ""),
        ("学　　号", ""), ("学生姓名", ""), ("指导教师", "郭　鹏"),
        ("完成日期", "2027 年 5 月 7 日"),
    ]
    for k, v in rows:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(k + "：")
        _set_cjk(r, BODY_FONT); r.font.size = Pt(14)
        r2 = p.add_run(v if v else "　" * 12)
        _set_cjk(r2, BODY_FONT); r2.font.size = Pt(14)
        r2.underline = True

    doc.add_page_break()


def _build_toc(doc: Document) -> None:
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("目　录")
    _set_cjk(r, HEAD_FONT); r.font.size = Pt(16); r.bold = True
    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run()
    _set_cjk(run, BODY_FONT); run.font.size = BODY_SIZE
    _add_field(run, r'TOC \o "1-3" \h \z \u')
    doc.add_page_break()


TABLE_SEP_RE = re.compile(r"^\|[\s:\-\|]+\|\s*$")
IMG_RE = re.compile(r"^!\[[^\]]*\]\(([^)]+)\)\s*$")
CAPTION_RE = re.compile(r'^<p align="center">(.*?)</p>\s*$')
LIST_RE = re.compile(r"^(\s*)([-*]|\d+\.)\s+(.*)$")
DISPLAY_MATH_RE = re.compile(r"^\$\$(.+)\$\$$")


def _split_row(line: str) -> list[str]:
    cells = line.strip().strip("|").split("|")
    return [c.strip() for c in cells]


def _add_table(doc: Document, rows: list[list[str]]) -> None:
    ncol = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=ncol)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = table.cell(i, j)
            cell.text = ""
            p = cell.paragraphs[0]
            p.paragraph_format.line_spacing = 1.15
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            txt = row[j] if j < len(row) else ""
            # 对齐：纯数字列右对齐，其余居中
            numeric = re.fullmatch(r"[\d\.,%\s\-–~+]*", txt or "") and any(ch.isdigit() for ch in txt)
            p.alignment = WD_ALIGN_PARAGRAPH.RIGHT if (numeric and j > 0) else WD_ALIGN_PARAGRAPH.CENTER
            add_runs(p, txt, TABLE_SIZE, bold_all=(i == 0))
    _three_line_borders(table)
    _repeat_header_and_keep_rows(table)
    doc.add_paragraph().paragraph_format.space_after = Pt(6)


def convert(md_path: str, out_path: str) -> None:
    raw = open(md_path, encoding="utf-8").read().splitlines()

    # 跳过前半部分（封面表 / 状态说明 / 手写目录），从「# 摘要」开始
    start = next(i for i, l in enumerate(raw) if l.strip() == "# 摘要")
    lines = raw[start:]
    title = raw[0].lstrip("# ").strip()

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.top_margin, sec.bottom_margin = Cm(2.5), Cm(2.5)
    sec.left_margin, sec.right_margin = Cm(3.0), Cm(2.5)
    _page_number_footer(sec)
    _setup_styles(doc)

    _build_cover(doc, title)
    _build_toc(doc)

    i = 0
    first_chapter_seen = False
    while i < len(lines):
        line = lines[i]
        s = line.strip()

        if not s or s == "---":
            i += 1
            continue

        # ---- 标题 ----
        m = re.match(r"^(#{1,4})\s+(.*)$", s)
        if m:
            level, text = len(m.group(1)), m.group(2).strip()
            text = _strip_status(text)
            if level == 1 and text.startswith("第") and "章" in text:
                if first_chapter_seen:
                    doc.add_page_break()
                first_chapter_seen = True
            elif level == 1 and text in ("摘要", "参考文献", "致谢"):
                doc.add_page_break()
            elif level == 1 and text.startswith("附录"):
                doc.add_page_break()
            h = doc.add_heading(level=min(level, 4))
            add_runs(h, text, {1: Pt(16), 2: Pt(14), 3: Pt(12), 4: Pt(12)}[min(level, 4)],
                     font=HEAD_FONT, bold_all=True)
            i += 1
            continue

        # ---- 代码块 ----
        if s.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1
            for cl in buf:
                p = doc.add_paragraph()
                p.paragraph_format.line_spacing = 1.0
                p.paragraph_format.left_indent = Cm(0.75)
                p.paragraph_format.space_after = Pt(0)
                r = p.add_run(cl if cl else " ")
                _set_cjk(r, CODE_FONT); r.font.size = Pt(9)
            doc.add_paragraph()
            continue

        # ---- 图片 ----
        m = IMG_RE.match(s)
        if m:
            rel = m.group(1)
            abs_p = os.path.normpath(os.path.join(os.path.dirname(md_path), rel))
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            if os.path.exists(abs_p):
                p.add_run().add_picture(abs_p, width=Cm(15.0))
            else:
                add_runs(p, "[缺图: " + rel + "]", CAPTION_SIZE)
            i += 1
            continue

        # ---- 图注 ----
        m = CAPTION_RE.match(s)
        if m:
            p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_after = Pt(10)
            add_runs(p, m.group(1), CAPTION_SIZE, font=HEAD_FONT)
            i += 1
            continue

        # ---- 独立公式 $...$ ----
        m = DISPLAY_MATH_RE.match(s)
        if m:
            p = doc.add_paragraph()
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            p.paragraph_format.line_spacing = 1.5
            r = p.add_run(latex_to_text(m.group(1)))
            _set_cjk(r, BODY_FONT)
            r.font.size = BODY_SIZE
            # 这里不设斜体：斜体会让 | 看起来像 / ，绝对值线读不出来
            i += 1
            continue

        # ---- 表格 ----
        if s.startswith("|") and i + 1 < len(lines) and TABLE_SEP_RE.match(lines[i + 1].strip()):
            rows = [_split_row(s)]
            i += 2
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(_split_row(lines[i]))
                i += 1
            _add_table(doc, rows)
            continue

        # ---- 引用块 ----
        if s.startswith(">"):
            buf = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                buf.append(lines[i].strip().lstrip(">").strip())
                i += 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.75)
            p.paragraph_format.right_indent = Cm(0.75)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(4)
            add_runs(p, smart_join(buf), Pt(10.5), font="楷体")
            continue

        # ---- 列表 ----
        m = LIST_RE.match(line)
        if m:
            indent, marker, text = m.group(1), m.group(2), m.group(3)
            p = doc.add_paragraph()
            depth = len(indent) // 2
            p.paragraph_format.left_indent = Cm(0.75 + 0.6 * depth)
            p.paragraph_format.first_line_indent = Cm(-0.5)
            p.paragraph_format.space_after = Pt(2)
            p.paragraph_format.line_spacing = 1.5
            bullet = "• " if marker in ("-", "*") else marker + " "
            add_runs(p, bullet + text, BODY_SIZE)
            i += 1
            continue

        # ---- 状态行整行丢弃（"✅ **定稿**（数据：...）" 之类） ----
        if line.lstrip().startswith(STATUS_EMOJI):
            i += 1
            continue

        # ---- 普通段落 ----
        buf = [s]
        i += 1
        while i < len(lines):
            nxt = lines[i].strip()
            if (not nxt or nxt == "---" or nxt.startswith("#") or nxt.startswith("|")
                    or nxt.startswith(">") or nxt.startswith("```") or IMG_RE.match(nxt)
                    or CAPTION_RE.match(nxt) or LIST_RE.match(lines[i])):
                break
            buf.append(nxt)
            i += 1
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(3)
        _first_line_indent_chars(p, 2.0)
        add_runs(p, smart_join(buf), BODY_SIZE)

    doc.save(out_path)




def audit_math_commands(path: str) -> list[str]:
    """列出公式里出现的、但符号表与结构化规则都不认识的 LaTeX 命令。

    不认识不代表出错——会被静默删除。但静默删除会让公式少一块（本文就漏过 \\max、\\equiv），
    所以每次构建都报出来。
    """
    text = open(path, encoding="utf-8").read()
    known = set(MATH_SYMBOLS)
    struct = {"\\left", "\\right", "\\text", "\\mathrm", "\\mathbb", "\\operatorname",
              "\\textrm", "\\underbrace", "\\frac", "\\sqrt", "\\overline", "\\bar"}
    seen: dict[str, int] = {}
    for m in re.finditer(r"\$[^$\n]+\$", text):
        for c in re.finditer(r"\\[a-zA-Z]+", m.group(0)):
            cmd = c.group(0)
            if cmd not in known and cmd not in struct:
                seen[cmd] = seen.get(cmd, 0) + 1
    return [k + "×" + str(v) for k, v in sorted(seen.items(), key=lambda kv: -kv[1])]

def main() -> int:
    if not os.path.exists(SRC):
        print("找不到源文件:", SRC)
        return 1
    unhandled = audit_math_commands(SRC)
    if unhandled:
        print("⚠ 公式里有未识别的 LaTeX 命令（转换时会被删掉，请补进 MATH_SYMBOLS）:")
        print("   " + "  ".join(unhandled))
    else:
        print("公式命令审计: 全部已识别")
    convert(SRC, OUT)
    print("已生成", OUT, "(%.1f KB)" % (os.path.getsize(OUT) / 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
