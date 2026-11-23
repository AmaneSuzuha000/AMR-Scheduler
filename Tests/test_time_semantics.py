"""时间表示与浮点容差的回归测试。

背景
----
C# 版 RAWSim-O 的预约表用 double 存时间，容差是绝对值 1e-12，
在长仿真时刻（>12000 s）失效，导致 CBS 崩溃。
详见上游 CBS 规划器的崩溃与可复现性分析（内部记录）。

Python 侧的规避方式是时间一律用整数步。本模块守住这条约束不被破坏。
"""

from __future__ import annotations

import pytest

from amr_sched.reservation import Interval, ReservationTable


def test_interval_rejects_fractional_floats():
    """非整值浮点必须被拒绝，否则误差会在 len() 里被静默截断。

    回归：Interval(1, 2.0000000001) 曾被接受，len() 返回 1，
    误差被藏起来——这正是 C# 侧 1-ULP 区间问题的弱化版本。
    """
    with pytest.raises(TypeError):
        Interval(1, 2.0000000001)
    with pytest.raises(TypeError):
        Interval(1.5, 2.5)
    with pytest.raises(TypeError):
        Interval(0.1, 1.9)


def test_interval_accepts_integral_floats_and_normalizes():
    """3.0 这类整值 float 放行，但必须转成 int 存进去。"""
    iv = Interval(1.0, 3.0)
    assert isinstance(iv.start, int) and isinstance(iv.end, int)
    assert not isinstance(iv.start, bool)
    assert len(iv) == 2


def test_interval_rejects_bool():
    """bool 是 int 的子类，但 True/False 当步号是错误用法。"""
    with pytest.raises(TypeError):
        Interval(True, False)


def test_many_adjacent_intervals_do_not_drift():
    """模拟 C# 的崩溃场景：在很大的时间尺度上排布大量相邻区间。

    整数步下，无论时刻多大，"相邻不冲突"都必须严格成立，
    不需要任何容差常数。
    """
    table = ReservationTable()
    base_t = 12000 * 1000        # 对应 C# 崩溃点 12062 s 的量级
    n = 500
    for i in range(n):
        table.reserve_node("N", Interval(base_t + i, base_t + i + 1), "AGV%03d" % i)
    # 每个区间只与自己重合，不与邻居冲突
    for i in (0, 1, n // 2, n - 1):
        t = base_t + i
        span = Interval(t, t + 1)
        owners = table.node_conflicts("N", span)
        assert owners == ["AGV%03d" % i], \
            "在 t=%d 处冲突判定错误：%r" % (t, owners)


def test_adjacent_intervals_never_conflict_at_any_magnitude():
    """相邻区间在任何数量级都不冲突（左闭右开的核心保证）。"""
    for magnitude in (0, 10 ** 3, 10 ** 6, 10 ** 9, 10 ** 12):
        a = Interval(magnitude, magnitude + 1)
        b = Interval(magnitude + 1, magnitude + 2)
        assert not a.overlaps(b), "量级 %d 下相邻区间被判冲突" % magnitude
        assert not b.overlaps(a)


def test_overlap_detected_at_large_magnitude():
    """真实重叠在任何数量级都必须被检出（不能为了不崩就放过重叠）。"""
    m = 10 ** 12
    assert Interval(m, m + 5).overlaps(Interval(m + 4, m + 9))
    assert Interval(m, m + 1).overlaps(Interval(m, m + 1))


def test_reservation_table_has_no_float_tolerance_constant():
    """守护设计决策：预约表不应该引入任何浮点容差常数。

    如果有人为了"修"某个边界问题加一个 TOLERANCE，说明时间表示出了偏差，
    应该回头查整数步是否被破坏，而不是加容差把问题盖住。
    """
    import inspect
    import re

    from amr_sched import reservation as mod
    src = inspect.getsource(mod)
    # 只查**代码行**，不查注释/docstring —— 模块文档里本来就会提到 C# 的 TOLERANCE
    code_lines = []
    in_doc = False
    for line in src.splitlines():
        s = line.strip()
        if s.startswith('"""') or s.startswith("'''"):
            in_doc = not in_doc if s.count('"""') % 2 else in_doc
            continue
        if in_doc or s.startswith("#"):
            continue
        code_lines.append(line)
    code = "\n".join(code_lines)
    for bad in ("TOLERANCE", "EPSILON", "isclose"):
        assert bad not in code, \
            "预约表代码里出现了浮点容差痕迹 %r，请检查时间是否被写成浮点" % bad
    # 数值形态的容差常数
    assert not re.search(r"\b[1-9]\.?\d*e-0?[6-9]\b", code), \
        "预约表代码里出现了小量级浮点常数，疑似容差"
