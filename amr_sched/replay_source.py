"""实车数据回放适配器：在没有硬件时先跑通上层链路。

为什么需要它
------------
P3 实车接入要求 Ubuntu 侧按 ``amr_sched.stream`` 的 schema 回传位姿，
Windows 侧接收并渲染。但硬件未到，这条链路**无法端到端验证**。

若等硬件到了再写接收端，会出现「硬件就绪但软件没准备好」的空转期；
更糟的是，解析器与渲染器里的小错误要到那时才暴露，排查成本高得多。

这个模块的作用是：**把一份预先录制的位姿序列当成本地"实车"喂给接收端**。
录制格式与实车完全一致（同样的 schema、同样的 ``ODOM`` 来源标记），
因此硬件到位后只需把数据源从文件换成 CAN 解析器，上层无需改动。

它不做什么
------------
不生成数据、不模拟物理、不猜测量。只读取已有的 JSON。
任何「实车数据」都必须来自真实录制，本模块不提供伪造能力。
"""

from __future__ import annotations

import json
from typing import Iterator

from .stream import AgvFrame, Frame, Pose, PoseSource, SCHEMA_VERSION


class SchemaMismatch(ValueError):
    """录制文件与当前 schema 不兼容。"""


def _frame_from_dict(d: dict) -> Frame:
    agvs = []
    for a in d.get("agvs", []):
        p = a["pose"]
        agvs.append(
            AgvFrame(
                agv_id=a["agv_id"],
                pose=Pose(
                    agv_id=p["agv_id"],
                    node=p.get("node"),
                    edge=p.get("edge"),
                    progress=float(p.get("progress", 0.0)),
                    source=PoseSource(p.get("source", "sim")),
                    timestamp=p.get("timestamp"),
                ),
                state=a.get("state", "idle"),
                task_id=a.get("task_id"),
                path=list(a.get("path", [])),
            )
        )
    return Frame(
        step=int(d["step"]),
        t=float(d.get("t", 0.0)),
        agvs=agvs,
        task_counts=dict(d.get("task_counts", {})),
        n_blocked=int(d.get("n_blocked", 0)),
        preemptions=list(d.get("preemptions", [])),
    )


def load_recording(path: str) -> list[Frame]:
    """读取一份录制文件，返回 :class:`Frame` 序列。

    录制文件与 :func:`amr_sched.stream.export_stream` 的输出同构，
    唯一区别是 ``pose.source`` 通常为 ``odom`` 或 ``localized``。
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    got = data.get("schema")
    if got != SCHEMA_VERSION:
        raise SchemaMismatch(
            f"录制文件 schema={got!r}，当前要求 {SCHEMA_VERSION!r}。"
            "请用对应版本的工具重新导出。"
        )
    return [_frame_from_dict(d) for d in data.get("frames", [])]


def replay(path: str) -> Iterator[Frame]:
    """逐帧迭代录制文件，模拟实车的持续推送。"""
    yield from load_recording(path)


def compare_sources(sim_frames: list[Frame],
                    real_frames: list[Frame]) -> dict[str, int]:
    """统计仿真帧与实车帧的步号对齐情况。

    **先对齐，再谈误差。** 两类数据的时间轴若对不上，
    任何位姿偏差的比较都没有意义。
    """
    sim_steps = {f.step for f in sim_frames}
    real_steps = {f.step for f in real_frames}
    return {
        "sim_frames": len(sim_frames),
        "real_frames": len(real_frames),
        "matched_steps": len(sim_steps & real_steps),
        "unmatched_sim_steps": len(sim_steps - real_steps),
        "unmatched_real_steps": len(real_steps - sim_steps),
    }


def pose_sources(frames: list[Frame]) -> set[str]:
    """收集出现过的位姿来源。用于**拒绝混用**。

    若一份「实车录制」里出现了 ``sim``，那它不是实车数据。
    """
    return {a.pose.source.value for f in frames for a in f.agvs}
