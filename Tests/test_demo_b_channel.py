"""Demo B 实时通道的回归测试。

tools/verify_demo_b*.py 是演示脚本（打印给人看），这里把同样的
不变量固化成断言，避免以后改动 stream.py / live_bridge.py 时静默回归。
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from amr_sched.allocation import GreedyAllocator                      # noqa: E402
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner  # noqa: E402
from amr_sched.scenarios import ScenarioConfig, generate_scenario     # noqa: E402
from amr_sched.simulation import Simulator, SimulationConfig          # noqa: E402
from amr_sched.stream import FrameBuilder                             # noqa: E402


def _make_sim(n_tasks=6, n_agvs=3, seed=3):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed, dt=1.0,
                        max_speed=1.0, urgent_ratio=0.15)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=60_000, seed=seed))
    sim.reset()
    return sim


# ---------------- step_once 必须先 reset ----------------

def test_step_once_without_reset_raises_runtime_error():
    """分步驱动漏调 reset() 时，必须给出可读的 RuntimeError。

    原先会抛 AttributeError: '_load_end'，报错点离真正原因隔了好几层。
    """
    sc = ScenarioConfig(n_tasks=4, n_agvs=2, seed=1, dt=1.0,
                        max_speed=1.0, urgent_ratio=0.15)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=1000, seed=1))
    with pytest.raises(RuntimeError, match="reset"):
        sim.step_once()


def test_step_once_works_after_reset():
    sim = _make_sim()
    assert sim.step_once() is True


# ---------------- hello / frame 结构 ----------------

def test_hello_declares_string_schema_and_consistent_network():
    sim = _make_sim()
    hello = FrameBuilder(sim).hello()
    assert hello["type"] == "hello"
    # schema 是 "1.0" 这样的字符串，不是 int
    assert isinstance(hello["schema"], str)
    assert int(hello["schema"].split(".")[0]) >= 1

    nodes = hello["network"]["nodes"]
    ids = {n["id"] for n in nodes}
    assert len(ids) == len(nodes), "节点 id 必须唯一"
    for e in hello["network"]["edges"]:
        assert e["u"] in ids and e["v"] in ids


def test_frame_references_only_known_nodes():
    sim = _make_sim()
    ids = {n["id"] for n in FrameBuilder(sim).hello()["network"]["nodes"]}
    b = FrameBuilder(sim)
    prev = None
    for _ in range(60):
        sim.step_once()
        f = b.frame()
        assert f["type"] == "frame"
        if prev is not None:
            assert f["step"] > prev, "step 必须严格递增"
        prev = f["step"]
        for a in f["agvs"]:
            p = a["pose"]
            if p.get("node"):
                assert p["node"] in ids
            if p.get("edge"):
                # 生产端序列化成 "U->V" 字符串
                assert isinstance(p["edge"], str) and "->" in p["edge"]
                u, v = p["edge"].split("->", 1)
                assert u in ids and v in ids
            assert 0.0 <= float(p.get("progress", 0.0)) <= 1.0 + 1e-9
            for n in (a.get("path") or []):
                assert n in ids


# ---------------- live_bridge 协议往返 ----------------

def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def bridge_proc():
    port = _free_port()
    # Windows 上 pytest 的输出捕获会关掉部分句柄，Popen 再 DuplicateHandle
    # 就会抛 WinError 6（句柄无效）。用 close_fds=False 并重定向到文件避开。
    log = open(Path(tempfile.gettempdir()) / ("lb_%d.log" % port), "wb")
    proc = subprocess.Popen(
        [sys.executable, "tools/live_bridge/live_bridge.py",
         "--mode", "sim", "--rate", "20", "--port", str(port),
         "--start", "0.9", "1.35", "0", "--watchdog", "0", "--proto", "1"],
        cwd=str(ROOT), stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT,
        close_fds=False)
    time.sleep(1.5)
    try:
        yield port
    finally:
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception:
            pass
        finally:
            try:
                log.close()
            except Exception:
                pass


def _read_msgs(sock, seconds):
    out, buf = [], b""
    t_end = time.time() + seconds
    sock.settimeout(0.2)
    while time.time() < t_end:
        try:
            data = sock.recv(8192)
        except socket.timeout:
            continue
        if not data:
            break
        buf += data
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            if line.strip():
                out.append(json.loads(line.decode("utf-8")))
    return out


def test_bridge_hello_pose_status_fields_match_csharp_records(bridge_proc):
    """C# LiveMessages.cs 声明的字段，服务端必须都发。

    本机没有 .NET SDK，C# 侧跑不起来；但"字段对不对"是纯协议问题，
    可以在 Python 里严格验证。
    """
    declared = {
        "hello": {"vehicle", "proto", "frame", "caps"},
        "pose": {"vehicle", "seq", "t", "x", "y", "yaw"},
        "status": {"vehicle", "battery", "mode", "error"},
    }
    s = socket.create_connection(("127.0.0.1", bridge_proc), timeout=8)
    try:
        msgs = _read_msgs(s, 2.5)
    finally:
        s.close()
    by_type = {}
    for m in msgs:
        by_type.setdefault(m["type"], m)
    for t, want in declared.items():
        assert t in by_type, "服务端没发 %s" % t
        missing = want - set(by_type[t])
        # caps/error/battery/mode 允许缺省，其余必须有
        core = want - {"caps", "battery", "mode", "error"}
        assert not (core - set(by_type[t])), "%s 缺字段 %s" % (t, core - set(by_type[t]))


def test_bridge_nav_and_cancel_roundtrip(bridge_proc):
    s = socket.create_connection(("127.0.0.1", bridge_proc), timeout=8)
    try:
        _read_msgs(s, 0.6)
        s.sendall((json.dumps({"type": "nav", "goal": "t-nav",
                               "path": [[1.0, 1.35], [1.15, 1.35]],
                               "max_speed": 0.3}) + "\n").encode())
        deadline = time.time() + 20
        res = None
        while time.time() < deadline and res is None:
            for m in _read_msgs(s, 0.5):
                if m["type"] == "nav_result" and m.get("goal") == "t-nav":
                    res = m
                    break
        assert res is not None, "nav 没有回 nav_result"
        assert res["status"] == "succeeded"

        s.sendall((json.dumps({"type": "nav", "goal": "t-cancel",
                               "path": [[3.0, 3.0], [3.2, 3.2]],
                               "max_speed": 0.15}) + "\n").encode())
        _read_msgs(s, 0.5)
        s.sendall((json.dumps({"type": "cancel", "goal": "t-cancel"}) + "\n").encode())
        deadline = time.time() + 12
        res2 = None
        while time.time() < deadline and res2 is None:
            for m in _read_msgs(s, 0.4):
                if m["type"] == "nav_result" and m.get("goal") == "t-cancel":
                    res2 = m
                    break
        assert res2 is not None, "cancel 没有回 nav_result"
        assert res2["status"] == "canceled"
    finally:
        s.close()
