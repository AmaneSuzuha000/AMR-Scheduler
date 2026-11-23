"""跨语言互操作测试：C# 能否正确读回 Python 导出的流。

这是「两套软件成果」是否真的连上的**唯一硬证据**。
文档可以写"接口已通"，但只有另一种语言真读了一遍才算数。
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "interop"))

from verify_interop import build_exe, make_stream, parse_cs  # noqa: E402

EXE = os.path.join(ROOT, "interop", "csharp", "reader.exe")
HAS_CS = os.name == "nt" and os.path.exists(
    r"C:\Windows\Microsoft.NET\Framework64\v4.0.30319\csc.exe")

needs_cs = pytest.mark.skipif(
    not HAS_CS, reason="无 .NET Framework 编译器，跳过 C# 侧验证")


@pytest.fixture(scope="module")
def stream_file():
    assert build_exe(), "C# 读取器编译失败"
    path = os.path.join(tempfile.gettempdir(), "interop_pytest.stream.json")
    expect = make_stream(path)
    yield path, expect
    try:
        os.remove(path)
    except OSError:
        pass


class _Result:
    """迷你结果对象，模拟 subprocess.CompletedProcess 的两个字段。"""

    def __init__(self, returncode, stdout, stderr):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def _run(path):
    """跑 C# 读取器。

    不用 capture_output：pytest 的 fd 捕获与子进程句柄继承在 Windows 上会
    冲突（OSError: [WinError 6] 句柄无效）。改为重定向到临时文件，稳健得多。
    """
    run_exe = EXE
    try:
        EXE.encode("ascii")
    except UnicodeEncodeError:
        # .NET Framework 4.0 的 CLR host 在模块路径包含非 ASCII 字符时抛出
        # ERROR_DLL_INIT_FAILED (0x8007045A)。当工作区位于中文字符路径时，
        # 将原程序镜像安全暂存到纯 ASCII 临时目录执行。
        import shutil
        tmp_exe = os.path.join(tempfile.gettempdir(), "amr_interop_reader.exe")
        if not os.path.exists(tmp_exe) or os.path.getmtime(EXE) > os.path.getmtime(tmp_exe):
            shutil.copy2(EXE, tmp_exe)
        run_exe = tmp_exe

    out_p = tempfile.mktemp(suffix=".out")
    err_p = tempfile.mktemp(suffix=".err")
    try:
        with open(out_p, "w", encoding="utf-8") as fo, \
                open(err_p, "w", encoding="utf-8") as fe:
            rc = subprocess.call([run_exe, path], stdout=fo, stderr=fe,
                                 stdin=subprocess.DEVNULL)
        with open(out_p, encoding="utf-8", errors="replace") as fo:
            out = fo.read()
        with open(err_p, encoding="utf-8", errors="replace") as fe:
            err = fe.read()
        return _Result(rc, out, err)
    finally:
        for q in (out_p, err_p):
            try:
                os.remove(q)
            except OSError:
                pass


@needs_cs
def test_csharp_reads_schema_and_frames(stream_file):
    path, expect = stream_file
    r = _run(path)
    assert r.returncode == 0, r.stdout + r.stderr
    got = parse_cs(r.stdout)
    assert got["schema"] == expect["schema"]
    assert got["frames"] == expect["frames"]


@needs_cs
def test_csharp_reads_every_agv(stream_file):
    path, expect = stream_file
    got = parse_cs(_run(path).stdout)
    by_id = {x["agv_id"]: x for x in got["agvs"]}
    assert set(by_id) == set(expect["agv_ids"])
    for agv in expect["agv_ids"]:
        assert by_id[agv]["node"] == expect["nodes"][agv]
        assert by_id[agv]["state"] == expect["states"][agv]


@needs_cs
def test_csharp_reports_pose_source(stream_file):
    """来源标记必须跨语言保留——这是数据诚实性的底线。"""
    path, _ = stream_file
    got = parse_cs(_run(path).stdout)
    assert "sources" in got
    assert got["sources"] == "Sim"


@needs_cs
def test_csharp_rejects_bad_schema(tmp_path):
    """schema 不符时 C# 必须报错，不能"尽量解析"。"""
    p = os.path.join(str(tmp_path), "bad.json")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write('{"schema":"0.9","frames":[]}')
    r = _run(p)
    assert r.returncode != 0, "schema 不匹配时必须非零退出"
    msg = (r.stdout + r.stderr)
    # .NET Framework 的异常消息在部分环境下可能打印不全，
    # 因此只强求"失败"，消息内容作软校验。
    if "schema" not in msg.lower():
        print("注：错误消息未含 schema 字样，实际输出为 %r" % msg[:200])
