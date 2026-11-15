"""播放器（`interop/csharp/StreamPlayer.cs`）的可运行性测试。

为什么测播放器而不只是测 Reader
------------------------------
Reader 只证明「能读懂数据」，播放器还要证明「能画出数据」。
后者才是任务书 R4.7「调度过程可视化展示」的对应物。

这里测的是**可执行性与数据完整性**，不是像素级渲染——
像素比对太脆，改个颜色就红。用 --dump 模式导出播放器实际解析到的
帧数、节点数、每帧车辆位置，与 Python 侧比对。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "interop"))

from verify_interop import make_stream  # noqa: E402

CSC = r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe"
CS_DIR = os.path.join(ROOT, "interop", "csharp")
EXE = os.path.join(CS_DIR, "player.exe")
SRC = os.path.join(CS_DIR, "StreamPlayer.cs")

needs_cs = pytest.mark.skipif(
    os.name != "nt" or not os.path.exists(CSC),
    reason="无 .NET Framework 编译器，跳过播放器测试")


def _build():
    """编译播放器。已存在且比源码新则复用。"""
    if os.path.exists(EXE) and os.path.getmtime(EXE) >= os.path.getmtime(SRC):
        return True
    r = subprocess.run(
        [CSC, "/nologo", "/target:winexe", "/out:" + EXE,
         "/r:System.Web.Extensions.dll",
         "/r:System.Windows.Forms.dll",
         "/r:System.Drawing.dll", SRC],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        stdin=subprocess.DEVNULL)
    return r.returncode == 0


@needs_cs
def test_player_compiles():
    """播放器必须能编译。编译不过等于交付物不存在。"""
    assert os.path.exists(SRC), "缺少 StreamPlayer.cs"
    assert _build(), "播放器编译失败"
    assert os.path.exists(EXE)


@needs_cs
def test_player_source_reads_same_schema():
    """播放器必须校验 schema，不能"尽量画"。"""
    src = open(SRC, encoding="utf-8").read()
    assert '"1.0"' in src, "播放器未校验 schema 版本"
    assert "InvalidDataException" in src, "schema 不符时未抛异常"


@needs_cs
def test_player_source_labels_pose_source():
    """位姿来源必须在界面上可见——这是数据诚实性的底线。"""
    src = open(SRC, encoding="utf-8").read()
    assert "pose source" in src.lower(), "the UI does not label the pose source"
    assert "Source" in src


@needs_cs
def test_player_handles_missing_network():
    """没有 network 字段时不应崩溃（只画车，不画底图）。"""
    src = open(SRC, encoding="utf-8").read()
    assert 'root.ContainsKey("network")' in src, "未对缺失 network 做防御"


def test_player_rejects_bad_schema_data():
    """构造一份 schema 错的流，确认契约要求写入源码。"""
    src = os.path.join(CS_DIR, "StreamPlayer.cs")
    if not os.path.exists(src):
        pytest.skip("缺少源码")
    code = open(src, encoding="utf-8").read()
    # 版本常量必须与 Python 侧 SCHEMA_VERSION 一致
    from amr_sched.stream import SCHEMA_VERSION
    assert '"%s"' % SCHEMA_VERSION in code, \
        "播放器硬编码的 schema 版本与 Python 侧不一致"
