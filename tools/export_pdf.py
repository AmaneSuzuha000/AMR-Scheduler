"""用 Word / PowerPoint COM 导出 PDF。

为什么需要它
------------
本机没有安装真正的 LibreOffice：`C:\\Tools\\codex-office-shim\\soffice.exe`
是一个用 reportlab **以单一字体重排**的 shim，导出的 PDF 会：
  - 把宋体/黑体全部换成 SimHei，行高被压扁；
  - 页数从 77 页掉到 31 页。
也就是说 soffice 这条路**不能用于交付物**。

但 Office COM 接口是真实可用的，导出的 PDF 页数与字体都正确。

用法::

    python tools/export_pdf.py                  # 论文 + PPT 全导
    python tools/export_pdf.py --only paper
    python tools/export_pdf.py --only slides
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_DOCX = os.path.join(ROOT, "docs", "论文", "论文-主文档.docx")
PAPER_PDF = os.path.join(ROOT, "docs", "论文", "论文-主文档.pdf")
SLIDES_PPTX = os.path.join(ROOT, "docs", "答辩", "答辩PPT.pptx")
SLIDES_PDF = os.path.join(ROOT, "docs", "答辩", "答辩PPT.pdf")

# Word: wdExportFormatPDF = 17
WORD_PS = r"""
$ErrorActionPreference = "Stop"
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {{
    $d = $word.Documents.Open("{src}", $false, $true)
    $d.Repaginate()
    $pages = $d.ComputeStatistics(2)
    $d.ExportAsFixedFormat("{dst}", 17)
    $d.Close($false)
    Write-Output "PAGES=$pages"
}} finally {{
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}}
"""

# PowerPoint: ppSaveAsPDF = 32
PPT_PS = r"""
$ErrorActionPreference = "Stop"
$ppt = New-Object -ComObject PowerPoint.Application
try {{
    $p = $ppt.Presentations.Open("{src}", $true, $false, $false)
    $p.SaveAs("{dst}", 32)
    $p.Close()
    Write-Output "SLIDES=$($p.Slides.Count)"
}} finally {{
    $ppt.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt) | Out-Null
}}
"""


def _run_ps(script: str) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".ps1", delete=False,
                                     encoding="utf-8-sig") as fh:
        fh.write(script)
        path = fh.name
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", path],
            capture_output=True, text=True, timeout=900,
        )
        if out.returncode != 0:
            raise RuntimeError(out.stderr.strip() or out.stdout.strip())
        return out.stdout.strip()
    finally:
        os.unlink(path)


def export_paper() -> int:
    if not os.path.exists(PAPER_DOCX):
        print(f"缺文件：{PAPER_DOCX}")
        return 1
    if os.path.exists(PAPER_PDF):
        backup = PAPER_PDF.replace(".pdf", "_prev.pdf")
        os.replace(PAPER_PDF, backup)
        print(f"旧 PDF 已备份为 {os.path.basename(backup)}")
    out = _run_ps(WORD_PS.format(src=PAPER_DOCX, dst=PAPER_PDF))
    size = os.path.getsize(PAPER_PDF) // 1024
    print(f"论文 PDF 已导出：{PAPER_PDF}（{size} KB）{out}")
    return 0


def export_slides() -> int:
    if not os.path.exists(SLIDES_PPTX):
        print(f"缺文件：{SLIDES_PPTX}")
        return 1
    if os.path.exists(SLIDES_PDF):
        backup = SLIDES_PDF.replace(".pdf", "_prev.pdf")
        os.replace(SLIDES_PDF, backup)
        print(f"旧 PDF 已备份为 {os.path.basename(backup)}")
    out = _run_ps(PPT_PS.format(src=SLIDES_PPTX, dst=SLIDES_PDF))
    size = os.path.getsize(SLIDES_PDF) // 1024
    print(f"PPT PDF 已导出：{SLIDES_PDF}（{size} KB）{out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", choices=["paper", "slides"], default=None)
    args = ap.parse_args(argv)

    rc = 0
    if args.only in (None, "paper"):
        rc |= export_paper()
    if args.only in (None, "slides"):
        rc |= export_slides()
    return rc


if __name__ == "__main__":
    sys.exit(main())
