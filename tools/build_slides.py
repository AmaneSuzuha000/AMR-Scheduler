"""生成答辩演示文稿（.pptx）。

设计原则
--------
1. **每个数字都能在论文里找到**。答辩被追问「这个数哪来的」，必须能立刻翻到；
   本文件里的所有数字都取自 `docs/论文/论文-主文档.md` 已定稿章节。
2. **图直接从 results/figures 取**，不重画——论文里的图和 PPT 里的图必须是同一张，
   否则出现细微不一致会很难解释。
3. 每页带**讲稿（备注）**，含「这一页要说什么」和「可能被问到什么」。

用法::

    python tools/build_slides.py
"""

from __future__ import annotations

import os

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(ROOT, "results", "figures")
OUT_DIR = os.path.join(ROOT, "docs", "答辩")
OUT = os.path.join(OUT_DIR, "答辩PPT.pptx")

# ---- 版式常量 ----
W, H = Inches(13.333), Inches(7.5)      # 16:9
FONT = "微软雅黑"
NAVY = RGBColor(0x1F, 0x3B, 0x64)
INK = RGBColor(0x22, 0x26, 0x2B)
GRAY = RGBColor(0x66, 0x6B, 0x72)
ACCENT = RGBColor(0xC0, 0x39, 0x2B)
LIGHT = RGBColor(0xF2, 0xF4, 0xF7)


def _style(run, size, color=INK, bold=False, font=FONT) -> None:
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    rPr = run.font._rPr
    for tag in ("a:ea", "a:cs"):        # 中文字形要单独指定，否则回落到宋体
        el = rPr.makeelement(qn(tag), {"typeface": font})
        rPr.append(el)


def _box(slide, left, top, width, height):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    return tf


def _para(tf, text, size, color=INK, bold=False, first=False, space_before=6,
          align=PP_ALIGN.LEFT, indent=0):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    if indent:
        p.level = indent
    run = p.add_run()
    run.text = text
    _style(run, size, color, bold)
    return p


def _rect(slide, left, top, width, height, color):
    from pptx.enum.shapes import MSO_SHAPE
    shp = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = color
    shp.line.fill.background()
    shp.shadow.inherit = False
    return shp


def new_slide(prs, title: str, kicker: str = ""):
    """标准内容页：顶部标题条 + 底部页码。"""
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, W, Inches(0.92), NAVY)
    tf = _box(s, Inches(0.55), Inches(0.12), W - Inches(1.1), Inches(0.7))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _para(tf, title, 26, RGBColor(0xFF, 0xFF, 0xFF), bold=True, first=True, space_before=0)
    if kicker:
        tf2 = _box(s, Inches(0.55), Inches(1.02), W - Inches(1.1), Inches(0.4))
        _para(tf2, kicker, 13, GRAY, first=True, space_before=0)
    return s


def footer(slide, page: int, total: int) -> None:
    tf = _box(slide, Inches(0.55), H - Inches(0.45), Inches(9), Inches(0.3))
    _para(tf, "面向铁路货车检修车间的多 AGV 物料配送调度系统设计", 9, GRAY, first=True, space_before=0)
    tf2 = _box(slide, W - Inches(1.3), H - Inches(0.45), Inches(0.75), Inches(0.3))
    _para(tf2, "%d / %d" % (page, total), 9, GRAY, first=True, space_before=0, align=PP_ALIGN.RIGHT)


def bullets(slide, items, left=Inches(0.75), top=Inches(1.55), width=None,
            height=Inches(5.2), size=17):
    """items: [(文字, 层级)]，层级 0 为一级，1 为二级。"""
    width = width or (W - Inches(1.5))
    tf = _box(slide, left, top, width, height)
    first = True
    for text, level in items:
        prefix = "▪ " if level == 0 else "– "
        p = _para(tf, prefix + text, size if level == 0 else size - 2,
                  INK if level == 0 else GRAY, bold=(level == 0),
                  first=first, space_before=(8 if level == 0 else 3), indent=level)
        p.space_after = Pt(2)
        first = False
    return tf


def picture(slide, name: str, left, top, width=None, height=None):
    path = os.path.join(FIG, name)
    if not os.path.exists(path):
        tf = _box(slide, left, top, Inches(6), Inches(1))
        _para(tf, "[缺图 %s]" % name, 14, ACCENT, first=True, space_before=0)
        return None
    kw = {}
    if width:
        kw["width"] = width
    if height:
        kw["height"] = height
    return slide.shapes.add_picture(path, left, top, **kw)


def kpi(slide, items, top=Inches(1.7), height=Inches(1.5)):
    """items: [(数字, 说明)]，均分一行。

    数字长度差别很大（「0」和「0.531 → 0.712」），固定字号会让长的折行、
    压住下面的说明。这里按字符数挑字号，并关掉自动换行。
    """
    n = len(items)
    gap = Inches(0.25)
    total = W - Inches(1.5)
    w = int((total - gap * (n - 1)) / n)
    for i, (num, label) in enumerate(items):
        left = Inches(0.75) + i * (w + gap)
        _rect(slide, left, top, w, height, LIGHT)
        size = 30 if len(num) <= 6 else (24 if len(num) <= 9 else 19)
        tf = _box(slide, left, top + Inches(0.14), w, Inches(0.72))
        tf.word_wrap = False
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        _para(tf, num, size, ACCENT, bold=True, first=True, space_before=0, align=PP_ALIGN.CENTER)
        tf2 = _box(slide, left + Inches(0.08), top + Inches(0.92), w - Inches(0.16), Inches(0.52))
        _para(tf2, label, 11, GRAY, first=True, space_before=0, align=PP_ALIGN.CENTER)

def takeaway(slide, text: str, top=Inches(6.0)) -> None:
    """底部要点带：一句话把这一页的结论钉死。

    没有它的话，KPI 页下面会空出一大片，视觉上像是没做完。
    """
    _rect(slide, Inches(0.75), top, Inches(0.07), Inches(0.62), ACCENT)
    tf = _box(slide, Inches(1.0), top, W - Inches(1.85), Inches(0.62))
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    _para(tf, text, 15, NAVY, bold=True, first=True, space_before=0)

def notes(slide, text: str) -> None:
    slide.notes_slide.notes_text_frame.text = text


def build() -> Presentation:
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    slides = []

    # ---------- 1 封面 ----------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, W, H, NAVY)
    tf = _box(s, Inches(1.0), Inches(2.0), W - Inches(2.0), Inches(1.0))
    _para(tf, "本科毕业设计（论文）答辩", 20, RGBColor(0xC8, 0xD4, 0xE6), first=True, space_before=0)
    tf = _box(s, Inches(1.0), Inches(2.75), W - Inches(2.0), Inches(1.9))
    _para(tf, "面向铁路货车检修车间的", 38, RGBColor(0xFF, 0xFF, 0xFF), bold=True, first=True, space_before=0)
    _para(tf, "多 AGV 物料配送调度系统设计", 38, RGBColor(0xFF, 0xFF, 0xFF), bold=True, space_before=4)
    _rect(s, Inches(1.0), Inches(4.85), Inches(0.9), Inches(0.06), RGBColor(0xC0, 0x39, 0x2B))
    tf = _box(s, Inches(1.0), Inches(5.15), W - Inches(2.0), Inches(1.2))
    _para(tf, "出题人：郭鹏　　完成日期：2027 年 5 月 7 日", 15, RGBColor(0xC8, 0xD4, 0xE6), first=True, space_before=0)
    _para(tf, "Python 3.14  |  35 个模块  |  835 项测试  |  4872 行实验数据", 13, RGBColor(0x9A, 0xAB, 0xC4), space_before=6)
    notes(s, N_COVER)
    slides.append(s)

    # ---------- 2 任务书对照 ----------
    s = new_slide(prs, "一页看清：任务书五项要求与交付", "对应关系来自 docs/任务书/验收自查")
    rows = [
        ("R1 文献调研与开题", "20 篇文献（外文 15），开题报告 + 需求分析", "✅"),
        ("R2 建模与方法设计", "节点-边网络、双目标模型、分配/规划/消解、三档抢占", "✅"),
        ("R3 系统设计", "六模块架构，含架构图与数据流", "✅"),
        ("R4 系统实现", "Python 30 模块，状态监控与过程回放", "✅"),
        ("R5 算例验证", "180 组算例，四项指标，与基准对比", "✅"),
        ("R6 论文与答辩", "论文 77 页（Word + PDF 均已导出）；查重按学校安排", "🟡"),
    ]
    _table(s, rows, Inches(0.75), Inches(1.55), W - Inches(1.5), Inches(4.6),
           widths=[Inches(3.1), Inches(7.3), Inches(0.8)])
    tf = _box(s, Inches(0.75), Inches(6.35), W - Inches(1.5), Inches(0.6))
    _para(tf, "任务书原文：「如 20/50/100」「如 2/4/6」——本课题按举例理解，三档全部覆盖。", 12, GRAY, first=True, space_before=0)
    notes(s, N_SCOPE)
    slides.append(s)

    # ---------- 3 背景 ----------
    s = new_slide(prs, "问题：检修车间的物料配送卡在哪")
    bullets(s, [
        ("检修车间与仓储拣选是两种场景，不能直接套用", 0),
        ("仓储：SKU 多、单件小、路径规则；检修：工位固定、配件批量、通道狭窄", 1),
        ("现有研究的缺口", 0),
        ("多 AGV 调度研究集中在仓储；轨道交通车辆检修车间的公开工作少见", 1),
        ("「任务分配—路径规划—冲突消解」三层少有打通并交付可复现原型", 1),
        ("三个具体痛点", 0),
        ("响应依赖老师傅经验，无量化派工依据", 1),
        ("加急任务（临修车）没有优先通道，只能排队", 1),
        ("配送过程不留数据，出了问题查不回去", 1),
    ], size=18)
    notes(s, N_BG)
    slides.append(s)

    # ---------- 4 场景建模 ----------
    s = new_slide(prs, "场景建模：把车间路网变成带容量的图", "R2.1 / R2.2")
    picture(s, "layout.png", Inches(0.55), Inches(1.5), width=Inches(7.1))
    bullets(s, [
        ("17 个节点 / 36 条边", 0),
        ("6 环道 + 6 检修工位 + 3 缓存区 + 拆解/组装区", 1),
        ("节点容量是硬约束", 0),
        ("环道 2、工位 1、缓存区 2；同时刻停放数不得超过容量", 1),
        ("双向边合并为一条物理通道", 0),
        ("能让「会车冲突」在物理层被检出，而不是当成两条无关的路", 1),
    ], left=Inches(7.9), top=Inches(1.6), width=Inches(4.9), size=16)
    notes(s, N_MODEL)
    slides.append(s)

    # ---------- 5 系统架构 ----------
    s = new_slide(prs, "系统设计：四层六模块", "R3")
    # 架构图 1.62:1。按高 5.3 寸放会宽 8.6 寸、伸到 9.45 寸，压住右侧文字。
    # 改成按宽度定位：7.9 寸宽 → 4.88 寸高，右侧留 4.28 寸给要点。
    picture(s, "architecture.png", Inches(0.45), Inches(1.5), width=Inches(7.9))
    bullets(s, [
        ("六模块与任务书一一对应", 0),
        ("任务管理、物流网络、调度决策、AGV 状态监控、数据存储、可视化", 1),
        ("两条设计纪律", 0),
        ("规划器只读预约表，写入统一由仿真器做——避免两边各写各的", 1),
        ("状态全落 SQLite，四张表可回放任意一步", 1),
    ], left=Inches(8.6), top=Inches(1.6), width=Inches(4.28), size=13)
    notes(s, N_ARCH)
    slides.append(s)

    # ---------- 6 方法总览 ----------
    s = new_slide(prs, "调度方法：三层决策 + 一层重调度")
    bullets(s, [
        ("第一层　任务分配（R2.4a）：先做哪个、交给谁", 0),
        ("调度规则（FIFO / EDD / SPT / PRIORITY / CR）× 分配算法（贪心 / 最近车 / 匈牙利）", 1),
        ("第二层　路径规划（R2.4b）：怎么走", 0),
        ("静态 Dijkstra 与时空 A*（在时间维上搜索，避开已预约时空点）", 1),
        ("第三层　冲突消解（R2.4c）：撞上了怎么办", 0),
        ("预约表做前馈 + 到达硬校验做反馈，两层结构；实测碰撞 0 次", 1),
        ("重调度　加急插入（R2.5）：被打乱了怎么办", 0),
        ("分档抢占：不抢 / 抢未取货 / 抢已装货——本文的核心方法贡献", 1),
    ], size=18)
    notes(s, N_METHOD)
    slides.append(s)

    # ---------- 7 加急三档 ----------
    s = new_slide(prs, "核心贡献：加急任务的分档抢占", "R2.5 / 论文 4.4 节")
    # 这张图是 3.2:1 的三联图，缩到 7 寸宽每格只有 2.4 寸，坐标轴数字看不清——必须整幅铺开
    picture(s, "preempt_tiers.png", Inches(0.95), Inches(1.32), width=Inches(11.4))
    bullets(s, [
        ("物料代价 =「已白跑」+「还要白跑」：抢空驶车只多一段空路，抢已装货车要真实返程", 0),
        ("NONE 不抢占　|　PREEMPT_UNPICKED 抢空驶车　|　PREEMPT_ANY 连已装货车也抢", 0),
    ], top=Inches(5.05), size=13)
    takeaway(s, "三档不是三个参数，是三种物料代价：零、一段空路、一次真实返程。", top=Inches(6.15))
    notes(s, N_PREEMPT)
    slides.append(s)

    # ---------- 8 实验设计 ----------
    s = new_slide(prs, "实验设计：180 组算例，每格 10 个种子", "R5.1 / R5.2")
    kpi(s, [("20 / 50 / 100", "任务规模三档"), ("2 / 4 / 6", "AGV 数量三档"),
            ("180", "主实验运行次数"), ("10", "每格独立种子")])
    bullets(s, [
        ("为什么一定要多种子", 0),
        ("单种子下完成时间的方差能吞掉全部差异，早期用 3 个种子得出过一个完全相反的结论", 1),
        ("统计口径", 0),
        ("配对精确 Wilcoxon（逐种子配对，n≤20 精确枚举），不比较中位数", 1),
        ("四项指标严格按任务书", 0),
        ("配送完成时间、任务准时率、AGV 利用率、冲突等待次数", 1),
    ], top=Inches(3.5), size=17)
    takeaway(s, "结论：10 个种子 + 逐种子配对，是本课题所有结论的统计地基。")
    notes(s, N_EXP)
    slides.append(s)

    # ---------- 9 结果一 抢占 ----------
    s = new_slide(prs, "结果一：抢占粒度应止于「未取货」", "论文 6.4 节，n=90/档")
    kpi(s, [("0.531 → 0.712", "加急任务准时率  p<0.0001"), ("145.7 → 85.5", "加急响应（步）  p<0.0001"),
            ("0.229 → 0.253", "总体准时率  p<0.0001"), ("0", "物料浪费（次）")])
    bullets(s, [
        ("继续放宽到「抢已装货」会怎样", 0),
        ("加急准时率能再升到 0.7848，但总体准时率显著下降（p=0.0018），并产生 148.7 s 物料浪费", 1),
        ("也就是说：收益被转嫁给了被抢的普通任务", 1),
        ("这是本文第一次把「抢到哪一档最划算」量化出来", 0),
    ], top=Inches(3.45), size=17)
    takeaway(s, "结论：抢「未取货」的车，加急响应 145.7 → 85.5 步（降 41%），物料代价为零。")
    notes(s, N_R1)
    slides.append(s)

    # ---------- 10 结果二 时空A* ----------
    s = new_slide(prs, "结果二：时空 A* 反而更慢（反直觉）", "论文 6.2 / 6.6 节")
    picture(s, "planner_cmp.png", Inches(0.95), Inches(1.32), width=Inches(11.4))
    bullets(s, [
        ("9 个算例格里无一显著优于静态 Dijkstra；6 格显著更慢，最多 5.84%（p=0.0137）", 0),
        ("不是绕路，是等：冲突等待 +35.7%，而总里程只差 0.01%——机理是前向锁定", 0),
    ], top=Inches(5.05), size=13)
    takeaway(s, "时空预约在低冲突窄通道里是净负担：省下的冲突抵不掉前向锁定带来的等待。", top=Inches(6.15))
    notes(s, N_R2)
    slides.append(s)

    # ---------- 11 结果三 瓶颈 ----------
    s = new_slide(prs, "结果三：瓶颈在任务分配层，不在路径规划层", "论文 6.8.3 / 6.8.4 节")
    kpi(s, [("9.0% ~ 9.9%", "分配算法对完成时间的影响"), ("≤5.6%", "路径规划器"),
            ("1.4% ~ 3.9%", "调度规则"), ("52% ~ 84%", "通道归一化负荷（均值 70%）")])
    bullets(s, [
        ("这条结论有工程价值", 0),
        ("把精力花在路径算法上，收益上限只有个位数；换分配算法能拿到两位数", 1),
        ("顺带纠正了一个口径陷阱", 0),
        ("原始负荷 11.6%~13.4% 看着「很空」，但分母是全通道容量，", 1),
        ("而任意时刻最多只有 n_agvs 条边被占，结构上限本就只有 16%~22%。归一化后是 52%~84%，属高负荷。", 1),
    ], top=Inches(3.5), size=16)
    takeaway(s, "结论：分配算法的杠杆是路径算法的 2~3 倍，工程投入该往这投。")
    notes(s, N_R3)
    slides.append(s)

    # ---------- 12 结果四 AGV 配置 ----------
    s = new_slide(prs, "结果四：4 台 AGV 是性价比拐点", "R5.7，论文 6.3 节")
    kpi(s, [("−46.4%", "2 → 4 车 完成时间"), ("−27.3%", "4 → 6 车 再降"),
            ("0.973 → 0.893", "AGV 利用率"), ("4", "本布局推荐配置")])
    bullets(s, [
        ("为什么不是越多越好", 0),
        ("加车能缩短完成时间，但边际收益掉得很快，而且通道容量会把车堵在路上", 1),
        ("怎么读这个数", 0),
        ("「4 台最优」只在本文的 17 节点布局与任务密度下成立，换布局要重新扫（见下页）", 1),
        ("任务书要求「讨论 AGV 配置数量的合理性」——给的是方法与拐点，不是一句话结论", 0),
    ], top=Inches(3.45), size=17)
    takeaway(s, "结论：4 台之后每加一台车的边际收益已不足其占用的通道资源。")
    notes(s, N_R4)
    slides.append(s)

    # ---------- 13 稳健性 ----------
    s = new_slide(prs, "结论稳不稳：换布局、加加工时间", "论文 6.9 / 6.10 节，540 + 450 = 990 行补充实验")
    bullets(s, [
        ("换三种布局（540 行）：主结论不变，且差距更大", 0),
        ("环网领先时空 A* 1.9%，网格 16.9%，枢纽 15.7%（只算双方都完成的配对）", 1),
        ("枢纽布局 180 次里有 22 次未完成（20 次在 6 车）——通道容量是真实物理边界，不是算法没调好", 1),
        ("加工位加工时间（450 行）：存在饱和相变", 0),
        ("120 s 以下几乎无影响；300 s 时完成时间 +33.5%，600 s 时 +118.3%", 1),
        ("AGV 利用率只从 0.931 降到 0.926，说明瓶颈转移到了工位，不是车不够", 1),
        ("附带收益", 0),
        ("这个维度揭出一个只在工位排队时才暴露的活锁缺陷，已修复并加了回归测试", 1),
    ], size=17)
    notes(s, N_ROBUST)
    slides.append(s)

    # ---------- 14 可视化 ----------
    s = new_slide(prs, "看得见：从甘特图到逐帧回放", "R4.5 / R4.7")
    picture(s, "gantt.png", Inches(0.5), Inches(1.5), width=Inches(5.6))
    picture(s, "trajectory.png", Inches(6.35), Inches(1.5), width=Inches(5.6))
    tf = _box(s, Inches(0.75), Inches(4.95), Inches(11.8), Inches(0.9))
    _para(tf, "另有：通道负荷热力图、AGV 状态时间线、任务进度曲线、指标演化曲线，以及可播放/单步/调速的 HTML 过程回放（1339 帧，含预约层与抢占事件）。", 13, GRAY, first=True, space_before=0)
    takeaway(s, "所有图由 experiments/make_figures.py 直接读 results/*.csv 生成，不存在手工填数。", top=Inches(6.05))
    notes(s, N_VIZ)
    slides.append(s)

    # ---------- 15 边界 ----------
    s = new_slide(prs, "诚实的边界：哪些结论不能外推", "论文 1.4 / 6.10.5 / 7.3 节")
    bullets(s, [
        ("规模上限", 0),
        ("算例最大 100 任务 / 6 台，不追求最优性证明，也不适用于上百台的仓储场景", 1),
        ("主方法并不总赢", 0),
        ("完成时间比最朴素的基线慢 6.0%（p=0.0207）——预约机制在低冲突场景是净负担", 1),
        ("准时率的绝对值依赖口径", 0),
        ("同一批结果换三种截止期口径，准时率相差 6.0 倍。跨论文比较必须先对齐口径", 1),
        ("加工时间是全局常数", 0),
        ("没按工位类型区分、没用随机分布，只能看趋势与拐点，不能预测具体车间产出", 1),
        ("没有实车数据", 0),
        ("全部结论来自仿真，未做实物验证", 1),
        ("物料是同质的", 0),
        ("任务书点名的转向架/轮轴/制动阀未区分。补做最小案例（30 任务/4 车/10 种子配对）：", 1),
        ("完成率不变（30/30），完工时间 +212.5 s（约 11.8%，10 个种子同向），准时率 −6.0 个百分点", 1),
        ("准时率降是因为截止口径没计入物料装卸时长，不是算法变差", 1),
    ], size=15)
    takeaway(s, "这些边界都写在论文里。答辩时主动交代，比被问出来好。")
    notes(s, N_LIMIT)
    slides.append(s)

    # ---------- 16 结论 ----------
    s = new_slide(prs, "结论")
    bullets(s, [
        ("交付了一套可复现的 Python 调度系统原型", 0),
        ("六模块齐全，835 项测试全绿，4872 行实验数据（34 份 CSV），一键复现", 1),
        ("给出了四条可证伪的量化结论", 0),
        ("① 抢占粒度止于「未取货」；② 时空预约在低冲突场景是净负担；", 1),
        ("③ 瓶颈在任务分配层；④ 4 台 AGV 是本布局的拐点", 1),
        ("其中三条与立项时的技术预期相反", 0),
        ("预期时空 A* 更快——实测 6/9 格更慢；预期加急必须抢已装货车——实测不必；", 1),
        ("预期预约能提升完成率——单车道实测反而更低", 1),
        ("贡献不在新算法，在场景适配、工程可复现，和一批敢报的负面结果", 0),
    ], size=17)
    takeaway(s, "四条结论里三条推翻了立项预期——数据说了算，这是我最想讲的一点。")
    notes(s, N_CONC)
    slides.append(s)

    # ---------- 17 致谢 Q&A ----------
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _rect(s, 0, 0, W, H, NAVY)
    tf = _box(s, Inches(1.2), Inches(2.6), W - Inches(2.4), Inches(1.4))
    _para(tf, "请各位老师批评指正", 40, RGBColor(0xFF, 0xFF, 0xFF), bold=True, first=True, space_before=0, align=PP_ALIGN.CENTER)
    tf = _box(s, Inches(1.2), Inches(4.3), W - Inches(2.4), Inches(1.0))
    _para(tf, "感谢郭鹏老师拟定本课题，任务书里的具体指标直接决定了本文实验设计的骨架。", 15, RGBColor(0xC8, 0xD4, 0xE6), first=True, space_before=0, align=PP_ALIGN.CENTER)
    notes(s, N_QA)
    slides.append(s)

    total = len(slides)
    for i, sl in enumerate(slides, 1):
        if i == 1 or i == total:
            continue
        footer(sl, i, total)

    return prs


def _table(slide, rows, left, top, width, height, widths=None):
    """三列表格：条目 / 内容 / 状态。表头加粗、斑马纹。"""
    nrow, ncol = len(rows) + 1, len(rows[0])
    shape = slide.shapes.add_table(nrow, ncol, left, top, width, height)
    tbl = shape.table
    if widths:
        for i, w in enumerate(widths):
            tbl.columns[i].width = w
    head = ("任务书要求", "交付物", "状态")
    for j, text in enumerate(head):
        cell = tbl.cell(0, j)
        cell.text = ""
        p = cell.text_frame.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = text
        _style(run, 14, RGBColor(0xFF, 0xFF, 0xFF), bold=True)
        cell.fill.solid()
        cell.fill.fore_color.rgb = NAVY
    for i, row in enumerate(rows, 1):
        for j, text in enumerate(row):
            cell = tbl.cell(i, j)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j == 2 else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = text
            _style(run, 13, INK, bold=(j == 0))
            cell.fill.solid()
            cell.fill.fore_color.rgb = LIGHT if i % 2 == 0 else RGBColor(0xFF, 0xFF, 0xFF)
    return tbl


N_COVER = """【这一页说什么】
各位老师好，我是 XXX，毕设题目是《面向铁路货车检修车间的多 AGV 物料配送调度系统设计》，
出题人是郭鹏老师，完成日期 2027 年 5 月 7 日。

【开场只讲三件事，别展开】
1. 用 Python 做了一套能跑的系统，六个模块齐全；
2. 在 180 组算例上做了对照实验，结论可复现；
3. 有四条结论，其中三条和立项时的预期相反——这个我会重点讲。
"""

N_SCOPE = """【这一页说什么】
先给一张对照表，让老师知道任务书的五项要求分别落在哪里。

【讲法】不要逐行念。只说：「R1 到 R5 全部完成，证明材料在验收自查文档里逐条列了指针；
R6 论文已成稿 77 页，查重和答辩按学校安排进行。」

【可能被问】
Q: 任务书说「如 20/50/100」，你只做了这三档，是硬性要求吗？
A: 原文是「如」，属举例。我按举例理解并全部覆盖了三档；如果导师要求扩展，
   代码里的规模是参数，加档位不需要改代码。
"""

N_BG = """【这一页说什么】
为什么这个题值得做：检修车间和仓储是两种场景，不能直接套用仓储的 AGV 调度研究。

【重点】讲清三个痛点的具体形态，不要停在「效率低」这种空话上：
- 派工靠经验 → 没有可比较的量化依据；
- 临修车加急 → 没有优先通道，只能排队；
- 过程不留数据 → 出了问题查不回去。

【可能被问】Q: 你的创新点是什么？
A: 不是新算法。是把三层打通、交付可复现原型，并且把「加急抢到哪一档最划算」第一次量化出来。
"""

N_MODEL = """【这一页说什么】
把车间路网抽象成带权有向图，重点是「节点容量」这个约束——它是后面所有冲突的根源。

【要讲到的三个点】
1. 17 节点 / 36 条边不是随便定的，6 环道 + 6 检修工位 + 3 缓存区 + 拆解/组装区，
   对应真实车间的功能区；
2. 容量：环道 2、工位 1、缓存区 2。工位容量 1 意味着「一台车卸货时另一台不能进」；
3. 双向边合并成一条物理通道，用 physical_edge_id 标识——这样「会车冲突」是物理事实，
   不是建模假象。

【可能被问】Q: 为什么环道容量是 2 不是 1？
A: 单车道会锁死，我在 6.7 节做过实验：容量 1 的环网在 6 车配置下会跑满 horizon。
"""

N_ARCH = """【这一页说什么】
四层六模块，和任务书 R3 的逐条要求对应。

【讲法】只讲两条设计纪律，不要念模块名：
1. 规划器只读预约表，写入统一由仿真器做。这条纪律避免了「规划器和仿真器各写各的」
   这类最难查的 bug——早期版本就是两边都写，导致预约表和实际位置对不上；
2. 状态全落 SQLite 四张表，任何一步都能回放。

【可能被问】Q: 为什么不用现成的仿真框架？
A: 试过 RAWSim-O，但它是仓储模型，看不到预约表占用、加急抢占、工位加工时间这三样；
   本文的可视化专门画这三样。
"""

N_METHOD = """【这一页说什么】
方法分三层 + 一层重调度。这一页只交代结构，细节在后面两页。

【一句话概括每层】
- 分配：先做哪个 + 交给谁（规则 × 算法两层分解）；
- 规划：怎么走（静态 Dijkstra vs 时空 A*）；
- 消解：撞上了怎么办（预约表前馈 + 到达硬校验反馈）；
- 重调度：被打乱了怎么办（分档抢占）。

【强调】碰撞实测 0 次——这是硬指标，不是「基本没有」。
"""

N_PREEMPT = """【这一页说什么】
这是本文的核心贡献，要讲慢一点。

【先讲概念】物料代价 = 已经白跑的里程 + 接下来还要白跑的里程。
抢一台「空驶去取货」的车，只多跑一段空路，代价接近零；
抢一台「已经装好货」的车，货要送回取货点，代价是真的。

【再讲三档】NONE / UNPICKED / ANY，这是本文提出并实现的。
下页给量化结果，「抢到哪一档最划算」此前只有直觉，没有数。

【可能被问】Q: 这个「分档」是你提的吗？
A: 抢占本身在动态调度里早有研究（Ouelhadj & Petrovic 综述里归为预测—反应式），
   但按「物料代价」分档并给出代价-收益曲线，是我这篇做的。
"""

N_EXP = """【这一页说什么】
实验设计的三个关键决定：规模档位、种子数、统计口径。

【为什么强调 10 个种子】这是踩过的坑：早期用 3 个种子 + 看中位数，
得出过「基线完成 100/100、死锁 0 次」的结论；复查原始行发现基线某次实际是 98 完成 / 打破 204 次，
「0 次」纯粹是取中位数造成的假象。现在全部改成逐种子配对 + 精确 Wilcoxon。

【可能被问】Q: n=10 会不会太少？
A: 精确 Wilcoxon 在 n=10 时的最小可达 p 是 0.0020（双尾），够用；
   要更小只能加种子。所有比较我都报了 p 值，不显著的就写不显著。
"""

N_R1 = """【这一页说什么】
第一个结论，也是最重要的一个：抢占粒度应止于「未取货」。

【数字要背下来】
- 加急准时率 0.531 → 0.712，p<0.0001，90 次配对里 55 好 7 差；
- 响应时间 145.7 → 85.5 步，81 好 1 差；
- 总体准时率 0.229 → 0.253，p<0.0001；
- 完成时间没有显著代价（p=0.5719），物料浪费为 0。

【关键对比】再往上抢「已装货」：加急准时率能到 0.785，但总体准时率反而显著下降
（p=0.0018），还产生 148.7 s 物料浪费。也就是说收益是从普通任务那里抢来的，系统层面没赚。

【可能被问】Q: 148.7 s 物料浪费是怎么算的？
A: 按「已白跑 + 还要白跑」的里程折算成时间，见论文 4.4.1 的定义。
"""

N_R2 = """【这一页说什么】
第二个结论，也是最反直觉的一个：时空 A* 反而更慢。

【先承认预期】立项时判断时空搜索应该更好，这也是文献里的普遍结论（CBS、ECBS 等）。
实测 9 个算例格里没有一格显著优于静态 Dijkstra，6 格显著更慢，最多 5.84%。

【机理必须讲清楚】不是绕路——总里程只差 0.01%；是等——冲突等待增加 35.7%。
时空 A* 提前把未来时段预约掉，后来的车只能绕开或等，前向锁定反而把路堵死。
在低冲突、窄通道的场景里，这个代价大于它带来的收益。

【可能被问】Q: 那是不是你的时空 A* 实现有问题？
A: 我专门做了机制诊断（论文 6.6 节）：先怀疑是两个假设（绕路、预约窗口太长），
   两个都被数据推翻，最后定位到等待。另外调度规则、车队规模都扫过，结论稳定。
"""

N_R3 = """【这一页说什么】
第三个结论，也是最有工程价值的一个：瓶颈在分配层，不在规划层。

【三个数字的对比要讲出来】分配算法 9.0%~9.9%，路径规划器 ≤5.6%，调度规则 1.4%~3.9%。
差了整整一个数量级。

【工程含义】车间上调度系统，优先把任务分配做对；在路径算法上精调，收益上限只有个位数。

【口径陷阱】这一页右下角那条要讲：通道原始负荷只有 11%~13%，看着很空；
但那个指标的分母是全通道容量，而任意时刻最多只有 n_agvs 条边被占，结构上限本就只有 16%~22%。
归一化后实际是 52%~84%，属高负荷。这是我做数据核查时发现并纠正的一处口径错误。
"""

N_R4 = """【这一页说什么】
任务书 R5.7 要求「讨论 AGV 配置数量的合理性」，这一页就是回答。

【数字】2→4 车完成时间降 46.4%，4→6 车只再降 27.3%；利用率从 0.973 掉到 0.893。
所以 4 台是本布局的拐点。

【必须补的一句】这个「4 台」不是普适结论，只在本布局 + 本任务密度下成立。
换布局要重新扫，6.9 节就是这么做的。任务书要的是「讨论」，不是一句「4 台最好」。
"""

N_ROBUST = """【这一页说什么】
回答「你的结论是不是只在一种布局/一种参数下成立」。

【换布局】三种布局 540 次运行，主结论不变，而且换布局后差距更大：
环网 1.9%、网格 16.9%、枢纽 15.7%（主方法领先时空 A* 的幅度，只算双方都完成的配对）。
枢纽布局 180 次里 22 次未完成，其中 20 次在 6 车配置——这是通道容量的真实物理边界，
不是算法没调好。

【加工时间】补了一个原模型里没有的维度（卸货即完成是简化）。
120 s 以下几乎无影响，300 s 时完成时间 +33.5%，600 s 时 +118.3%，存在饱和相变。
AGV 利用率只从 0.931 降到 0.926——说明瓶颈转移到工位了，不是车不够。

【附带收益】这个维度揭出一个活锁缺陷：死锁打破把已取货任务退回待分配，
下一轮又让某台车回原取货点取货，但货已经在别的车上了。只在工位排队时才暴露，已修复。
"""

N_VIZ = """【这一页说什么】
任务书 R4.5 / R4.7 要求「AGV 状态监控」与「调度过程可视化展示」。

【左图】甘特图：每台车的任务块，红块是加急任务，可见加急被优先执行。
【右图】轨迹图：实际行驶路径叠加在网络上，能直接看出绕行。

【演示建议】如果时间充裕，现场播 HTML 回放（10 秒即可）：可以单步、调速，
能看到节点预约占用与抢占事件。这是静态图给不出的东西。

【可能被问】Q: 这些图怎么保证和论文数据一致？
A: 图和数据同一份脚本生成，`experiments/make_figures.py` 读 `results/*.csv`，不手工填数。
"""

N_LIMIT = """【这一页说什么】主动交代边界。这一页是加分项，不要跳过、不要含糊。

【五条边界逐条说】
1. 规模上限 100 任务 / 6 台，不做最优性证明，也不适用于上百台的仓储；
2. 主方法并不总赢：完成时间比最朴素的基线慢 6.0%（p=0.0207）。预约机制在低冲突场景是净负担；
3. 准时率的绝对值依赖口径：同一批结果换三种截止期口径，准时率相差 6.0 倍；
4. 加工时间是全局常数，只能看趋势与拐点；
5. 全部结论来自仿真，没有实车数据；
6. 物料是同质的——补做的最小案例显示完工时间 +11.8%，但准时率下降源于
   截止口径没计入物料装卸时长，不是算法变差，这点要主动说清楚。

【可能被问】Q: 既然主方法更慢，它的价值在哪？
A: 它在高冲突场景（单车道、加急频繁）能减少死锁；代价是低冲突场景下的等待。
   本文把这个权衡量化了，而不是宣称它全面更优。
"""

N_CONC = """【这一页说什么】收尾，三句话。

【第一句】交付了可复现的系统原型：六模块、835 项测试、4872 行数据、一键复现。
【第二句】四条量化结论，其中三条与立项预期相反。这一条要强调——
不是「都达到了预期」，而是数据推翻了预期，我把它如实写进论文。
【第三句】贡献不在新算法，在场景适配与工程可复现，以及一批敢报的负面结果。

【时间控制】如果前面超时，这一页压缩成 30 秒说完，但「三条与预期相反」一定要说。
"""

N_QA = """【可能被追问的问题，提前准备好】

Q1 你的创新点到底是什么？
→ 不是新算法。三个层面：① 把分配/规划/消解三层在检修车间场景下打通并交付可复现原型；
   ② 提出按物料代价分档的抢占策略，给出代价-收益曲线；
   ③ 一批可证伪的量化结论，含三条反直觉结果。

Q2 为什么不用强化学习/元启发式？
→ 任务书要求的是调度系统设计与算例验证，不是算法竞赛。而且 6.8.3 的结论说明：
   瓶颈在分配层，先把分配做对收益最大；上 RL 需要大量训练样本，本课题规模下性价比低。

Q3 和 RAWSim-O 比，你的系统有什么？
→ RAWSim-O 是仓储模型，看不到三样东西：预约表占用、加急抢占过程、工位加工时间的影响。
   我的可视化专门画这三样，6.10 节的相变结论在 RAWSim-O 里做不出来。

Q4 数据能复现吗？
→ 种子、参数、脚本全在仓库里；34 份 CSV 有版本记录；还有一组可复现性测试，
   代码改动导致数据对不上时测试会红。

Q5 导师指出的问题（周数矛盾 / 日期栏空白）怎么处理？
→ 任务书原文「共 17 周」与各部分之和 21 周不一致，已如实记录并请导师确认；
   姓名与日期栏空白，待补齐后重新签署。
"""


def main() -> int:
    os.makedirs(OUT_DIR, exist_ok=True)
    prs = build()
    prs.save(OUT)
    size = os.path.getsize(OUT) / 1024
    print("已生成 %s（%.0f KB，%d 页）" % (OUT, size, len(prs.slides._sldIdLst)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
