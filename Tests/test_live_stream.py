"""实时状态通道的测试。

这层测的是"直播"，不是"录播"
--------------------------
离线导出已在 test_stream.py 里覆盖。这里必须证明的是另外几件事：

1. ``FrameBuilder`` 在读**运行中**的 Simulator 时能出帧，
   不依赖 ``record_trajectory`` 攒好的结果；
2. 实时帧与离线导出**逐字段同构**，否则 C# 那边要写两套解析；
3. 服务端会**等客户端接入**才开始推进——否则人还没开界面就跑完了；
4. 位置随时间真的变化，不是把同一个点重复发。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from amr_sched.stream import (
    SCHEMA_VERSION,
    AgvFrame,
    Frame,
    FrameBuilder,
    Pose,
    PoseSource,
    frame_to_line,
)
from amr_sched.live import build_sim


@pytest.fixture()
def sim():
    s = build_sim(n_tasks=12, n_agvs=3, seed=5)
    s.reset()
    return s


def test_hello_has_full_network(sim):
    b = FrameBuilder(sim)
    h = b.hello()
    assert h["type"] == "hello"
    assert h["schema"] == SCHEMA_VERSION
    assert h["source"] == "sim"
    assert len(h["network"]["nodes"]) == 17
    assert len(h["network"]["edges"]) == 36
    assert h["meta"]["n_agvs"] == 3
    assert h["meta"]["agv_ids"] == ["AGV00", "AGV01", "AGV02"]


def test_frame_from_running_simulator(sim):
    """不需要 record_trajectory，直接读运行中的状态就能出帧。"""
    assert not sim.config.record_trajectory, "本测试前提：不记录轨迹"
    b = FrameBuilder(sim)
    for _ in range(40):
        sim.step_once()
    f = b.frame()
    assert f["type"] == "frame"
    assert f["step"] == 40
    assert f["t"] == pytest.approx(40.0)
    assert len(f["agvs"]) == 3
    for a in f["agvs"]:
        assert set(a) >= {"agv_id", "pose", "state", "task_id", "path"}
        assert set(a["pose"]) >= {"agv_id", "node", "edge", "progress",
                                  "source", "timestamp"}
        assert a["pose"]["source"] == "sim"


def test_live_frame_is_isomorphic_to_offline(sim):
    """实时帧与离线导出的 AgvFrame 必须逐字段一致。

    两套格式一旦漂移，C# 侧就得写两份解析器，
    而"两份解析器"迟早会不一致。
    """
    live = FrameBuilder(sim).frame()
    offline = Frame(
        step=live["step"], t=live["t"],
        agvs=[AgvFrame(agv_id=a["agv_id"],
                      pose=Pose(agv_id=a["pose"]["agv_id"],
                                node=a["pose"]["node"],
                                edge=a["pose"]["edge"],
                                progress=a["pose"]["progress"],
                                source=PoseSource.SIM,
                                timestamp=a["pose"]["timestamp"]),
                      state=a["state"], task_id=a["task_id"],
                      path=a["path"])
              for a in live["agvs"]],
    ).to_dict()
    assert live["agvs"] == offline["agvs"]


def test_positions_actually_change(sim):
    """位置必须真的在动——这条区分"实时"与"把同一帧发 600 遍"。"""
    b = FrameBuilder(sim)
    seen = set()
    for _ in range(60):
        for _ in range(5):
            sim.step_once()
        f = b.frame()
        seen.add(tuple(sorted(
            (a["agv_id"], a["pose"]["node"], a["pose"]["edge"])
            for a in f["agvs"])))
    assert len(seen) >= 5, "60 帧里位置组合只有 %d 种，不像实时" % len(seen)


def test_frame_line_is_single_line_utf8(sim):
    """一行一条 JSON：不能含裸换行，否则按行解析的 C# 会错位。"""
    b = FrameBuilder(sim)
    for obj in (b.hello(), b.frame()):
        line = frame_to_line(obj)
        assert isinstance(line, bytes)
        assert line.endswith(b"\n")
        assert line.count(b"\n") == 1, "一条消息里出现了多个换行"
        back = json.loads(line.decode("utf-8"))
        assert back["type"] == obj["type"]


def test_waits_for_client_before_running():
    """没有客户端接入时，服务端不应把仿真跑完。

    这条来自实测事故：初版不等客户端，476 帧约 8 秒推完，
    C# 连上时只看到"连接被拒绝"。实时联动的语义是"有人在看才跑"。
    """
    from amr_sched.live import LiveServer

    s = build_sim(n_tasks=20, n_agvs=3, seed=2)
    srv = LiveServer(s, port=0, rate=1000.0, steps_per_frame=5)
    srv.run(wait_client=0.4)          # 等 0.4 秒，无人接入
    assert s._step_cursor == 0, "没有客户端却推进了 %d 步" % s._step_cursor
    assert srv._stop.is_set(), "放弃等待后应置停止位"


def test_step_is_strictly_increasing(sim):
    """step 必须严格递增：C# 侧靠它判断有没有掉帧/重连。"""
    b = FrameBuilder(sim)
    steps = []
    for _ in range(20):
        for _ in range(3):
            sim.step_once()
        steps.append(b.frame()["step"])
    assert steps == sorted(steps)
    assert len(set(steps)) == len(steps)
