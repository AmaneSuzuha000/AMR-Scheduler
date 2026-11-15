"""回放演示的数据完整性测试（G2 导师演示需求）。

为什么单独立一个测试
------------------
回放是答辩现场唯一的「看得见」的东西，但它最容易悄悄退化：
少一个字段，页面照样打开，只是那层信息没了——不报错、不崩，只是缺。

第 61 轮发现 `steps` 只导出 `pos` 没导出 `agv_paths`，
结果界面上看不到「这台车准备往哪走」，而这正是任务书 R4.7 与
导师录音「把路径规划算法接进去」都明确要求的东西。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _build_payload():
    """跑一次小规模仿真并生成回放负载。"""
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    from export_replay import build_payload
    return build_payload(12, 3, 0, "spacetime_astar", 0.0, "preempt_unpicked", 1)


def test_payload_has_path_field():
    """每帧必须带规划路径，否则界面上画不出「车要往哪走」。"""
    p = _build_payload()
    steps = p["steps"]
    assert steps, "轨迹为空"

    # 至少有一帧存在非空 path，否则说明字段没被填
    nonempty = [s for s in steps if s.get("path")]
    assert nonempty, "所有帧的 path 都为空，规划路径没有导出"

    # path 的值必须是节点 id 列表
    sample = nonempty[len(nonempty) // 2]
    node_ids = {n["id"] for n in p["nodes"]}
    for agv_id, seq in sample["path"].items():
        assert isinstance(seq, list), f"{agv_id} 的 path 不是列表"
        for nd in seq:
            assert nd in node_ids, f"{agv_id} 的 path 含未知节点 {nd}"


def test_payload_meta_complete():
    """顶栏要显示的元信息一个都不能少。"""
    m = _build_payload()["meta"]
    for k in ("n_tasks", "n_agvs", "seed", "planner", "policy",
              "makespan", "on_time", "util", "completed"):
        assert k in m, f"meta 缺少 {k}"


def test_template_renders_path():
    """模板里必须有绘制 path 的代码，否则数据导出了也没人画。"""
    html = (ROOT / "experiments" / "replay_template.html").read_text(
        encoding="utf-8")
    assert "__DATA__" in html, "模板缺少数据注入占位符"
    assert "s.path" in html, "模板没有使用 path 字段"
    # 必须从当前位置起画：agv_paths 是「剩余节点」，单个节点连不成线
    assert "parts[parts.length-1]" in html or "fromN" in html, \
        "模板没有把路径起点接到车辆当前位置"


def test_template_supports_a_to_b():
    """G2：界面必须支持 A→B 点选。

    导师原话是「把路径规划算法接进去」——只在回放里看既定路线不够，
    要能现场指定任两点、当场算出路线。
    """
    html = (ROOT / "experiments" / "replay_template.html").read_text(
        encoding="utf-8")
    assert "shortestPath" in html, "模板没有最短路计算"
    assert "nodeAt" in html, "模板没有把点击坐标换算成节点"
    assert 'cv.addEventListener("click"' in html, "画布没有点击事件"
    # 算出来的路线必须真的被画出来，否则能点但看不见
    assert "drawAB" in html, "模板没有绘制 A→B 路线"
    assert 'if(typeof drawAB==="function") drawAB();' in html, \
        "drawAB 没有被 draw() 调用，路线画不出来"


def test_data_injection_leaves_no_placeholder(tmp_path):
    """导出的 HTML 里不能残留 __DATA__，否则页面打不开数据。

    为什么不走 subprocess
    --------------------
    本环境下 pytest 里起子进程会偶发 `OSError: [WinError 6] 句柄无效`
    （pytest 捕获 stdout 时与子进程句柄冲突），单独跑却正常。
    改为直接调 main()，绕开子进程。
    """
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    import export_replay

    out = tmp_path / "r.html"
    rc = export_replay.main(["--tasks", "10", "--agvs", "2", "--seed", "0",
                             "--out", str(out)])
    assert rc == 0
    html = out.read_text(encoding="utf-8")
    assert "__DATA__" not in html, "导出的 HTML 残留 __DATA__ 占位符"
    assert "const DATA = {" in html, "数据没有正确注入"
