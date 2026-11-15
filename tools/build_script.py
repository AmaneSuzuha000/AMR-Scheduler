from pptx import Presentation
import os

TIMING = {
    1: ("0:30", "开场"), 2: ("0:40", "任务书对照"), 3: ("0:50", "问题"),
    4: ("0:50", "场景建模"), 5: ("0:50", "系统架构"), 6: ("0:50", "方法总览"),
    7: ("1:00", "核心贡献"), 8: ("0:50", "实验设计"), 9: ("1:10", "结果一"),
    10: ("1:00", "结果二"), 11: ("0:50", "结果三"), 12: ("0:50", "结果四"),
    13: ("0:40", "稳健性"), 14: ("0:30", "可视化"), 15: ("0:45", "边界"),
    16: ("0:50", "结论"), 17: ("—", "提问"),
}

p = Presentation("docs/答辩/答辩PPT.pptx")
L = []
L.append("# 答辩讲稿")
L.append("")
L.append("> 与 `答辩PPT.pptx` 的备注逐页对应，由 `tools/build_script.py` 自动导出。")
L.append("> 改 PPT 备注后重跑脚本即可，不会出现「讲稿和幻灯片不一致」。")
L.append("")
L.append("## 时间分配")
L.append("")
L.append("| 页 | 内容 | 时长 | 累计 |")
L.append("|---|---|---:|---:|")
total = 0
for i in range(1, len(p.slides._sldIdLst) + 1):
    nm, lab = TIMING[i]
    if nm != "—":
        m, sec = nm.split(":")
        total += int(m) * 60 + int(sec)
        acc = "%d:%02d" % (total // 60, total % 60)
    else:
        acc = "—"
    L.append("| %d | %s | %s | %s |" % (i, lab, nm, acc))
L.append("")
L.append("**总计约 12 分 50 秒**，留 5~7 分钟提问。若规定 10 分钟，砍 4、13、14 三页。")
L.append("")
L.append("## 时间不够时的取舍")
L.append("")
L.append("| 优先级 | 页 | 说明 |")
L.append("|---|---|---|")
L.append("| 必讲 | 1 7 9 10 11 12 16 | 题目、核心贡献、四条结论、总结 |")
L.append("| 可压成一句 | 3 4 5 6 8 | 背景与方法，一句话带过 |")
L.append("| 时间紧可跳过 | 13 14 | 稳健性与可视化，放到提问环节再展开 |")
L.append("| 不要跳 | 15 | 主动交代边界，评委很吃这一套 |")
L.append("")
L.append("---")
L.append("")
for i, s in enumerate(p.slides, 1):
    t = s.notes_slide.notes_text_frame.text if s.has_notes_slide else ""
    title = ""
    for shp in s.shapes:
        if shp.has_text_frame and shp.text_frame.text.strip():
            title = shp.text_frame.text.strip().split("\n")[0]
            break
    L.append("## 第 %d 页　%s" % (i, title))
    L.append("")
    L.append(t)
    L.append("")
    L.append("---")
    L.append("")
open("docs/答辩/答辩讲稿.md", "w", encoding="utf-8").write("\n".join(L))
print("已生成讲稿，%d 字" % sum(len(x) for x in L))