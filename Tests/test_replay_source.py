"""实车回放适配器（``amr_sched.replay_source``）的契约测试。

这些测试守住的不是"功能"，而是**数据诚实性**：
仿真数据不得被标成实车数据，实车数据也不得被悄悄当成仿真真值。
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from amr_sched.replay_source import (
    SchemaMismatch,
    compare_sources,
    load_recording,
    pose_sources,
    replay,
)
from amr_sched.stream import Frame, Pose, PoseSource, AgvFrame, SCHEMA_VERSION


def _write(tmp_path, frames, schema=SCHEMA_VERSION):
    path = os.path.join(str(tmp_path), "rec.json")
    payload = {
        "schema": schema,
        "meta": {"source": "test"},
        "network": None,
        "frames": [f.to_dict() for f in frames],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)
    return path


def _frames(source, steps=(0, 1, 2), agv="SC01"):
    out = []
    for s in steps:
        out.append(Frame(
            step=s, t=float(s),
            agvs=[AgvFrame(agv_id=agv,
                          pose=Pose(agv_id=agv, node="A", source=source),
                          state="to_dropoff")],
            task_counts={"completed": s},
        ))
    return out


def test_odom_recording_roundtrip(tmp_path):
    path = _write(tmp_path, _frames(PoseSource.ODOM))
    frames = load_recording(path)
    assert len(frames) == 3
    assert pose_sources(frames) == {"odom"}


def test_localized_recording_keeps_source(tmp_path):
    path = _write(tmp_path, _frames(PoseSource.LOCALIZED))
    frames = load_recording(path)
    assert pose_sources(frames) == {"localized"}


def test_schema_mismatch_is_rejected(tmp_path):
    """旧版录制必须报错，不能"尽量解析"——那会读出错位的数。"""
    path = _write(tmp_path, _frames(PoseSource.ODOM), schema="0.9")
    with pytest.raises(SchemaMismatch, match="0.9"):
        load_recording(path)


def test_missing_schema_is_rejected(tmp_path):
    path = os.path.join(str(tmp_path), "noschema.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"frames": []}, fh)
    with pytest.raises(SchemaMismatch):
        load_recording(path)


def test_replay_yields_in_order(tmp_path):
    path = _write(tmp_path, _frames(PoseSource.ODOM, steps=(0, 5, 10)))
    steps = [f.step for f in replay(path)]
    assert steps == [0, 5, 10]


def test_compare_sources_counts_alignment(tmp_path):
    """先对齐再谈误差：步号重合数必须能被算出来。"""
    sim = _frames(PoseSource.SIM, steps=(0, 1, 2, 3))
    real = _frames(PoseSource.ODOM, steps=(0, 1, 2, 9))
    p = _write(tmp_path, real)
    stats = compare_sources(sim, load_recording(p))
    assert stats["sim_frames"] == 4
    assert stats["real_frames"] == 4
    assert stats["matched_steps"] == 3
    assert stats["unmatched_sim_steps"] == 1
    assert stats["unmatched_real_steps"] == 1


def test_pose_sources_detects_contamination():
    """一份自称"实车"的录制里若混入 sim，必须能被发现。"""
    mixed = _frames(PoseSource.ODOM) + _frames(PoseSource.SIM, steps=(3, 4))
    assert pose_sources(mixed) == {"odom", "sim"}


def test_sim_recording_is_not_silently_accepted_as_real(tmp_path):
    """仿真录制可以读，但来源标记必须保留，不允许被"洗白"。"""
    path = _write(tmp_path, _frames(PoseSource.SIM))
    frames = load_recording(path)
    assert pose_sources(frames) == {"sim"}
    assert "odom" not in pose_sources(frames)
