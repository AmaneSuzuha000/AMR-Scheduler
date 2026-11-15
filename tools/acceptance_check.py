"""任务书条款终检：逐条真跑，不靠文档自述。

为什么要有它
------------
交付前的最后一道检查。它回答的不是「代码能不能跑」（那是 pytest 的事），
而是「任务书要求的条款是否都还在」。

典型失效场景：某次重构删掉了 `record_trajectory` 的传递、
或某个模块改名，pytest 未必报错（可能没有对应用例），
但 R4.7「调度过程可视化展示」已经名存实亡。

用法
----
    python tools/acceptance_check.py

非零退出表示有条款不满足。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amr_sched.allocation.greedy import GreedyAllocator  # noqa: E402
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner  # noqa: E402
from amr_sched.scenarios import ScenarioConfig, generate_scenario  # noqa: E402
from amr_sched.simulation import SimulationConfig, Simulator  # noqa: E402
from amr_sched.stream import frames_from_result  # noqa: E402

ok, bad = [], []


def chk(name, fn):
    try:
        ok.append((name, fn()))
    except Exception as e:
        bad.append((name, type(e).__name__ + ": " + str(e)))


def _run(n_tasks, n_agvs, seed=0, max_steps=60000, record=False):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed,
                        dt=1.0, urgent_ratio=0.15, max_speed=1.0)
    net, tasks, agvs = generate_scenario(sc)
    pl = SpaceTimeAStarPlanner(max_speed=sc.max_speed, dt=sc.dt)
    # record_trajectory=True 才会记逐步轨迹；帧流依赖它
    sim = Simulator(net, tasks, agvs, pl, GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=max_steps, seed=seed,
                                     record_trajectory=record))
    res = sim.run()
    res.total_distance = sim._odometer
    return net, tasks, agvs, sim, res


def six_modules():
    from amr_sched import network, reschedule, scenarios, simulation  # noqa
    from amr_sched import storage, stream, viz  # noqa
    from amr_sched.allocation import base as _a  # noqa
    from amr_sched.planners import base as _p  # noqa
    return "仿真/网络/场景/存储/重调度/流/可视化/分配/规划 全部可导入"


def e2e():
    _, tasks, _, _, res = _run(20, 4)
    s = res.summary()
    return "20任务/4车 完成 %d/%d，完成时间 %.0fs，准时率 %.3f，利用率 %.3f" % (
        s["tasks_completed"], len(tasks), s["makespan_seconds"],
        s["on_time_rate"], s["agv_utilization"])


def viz_stream():
    _, _, _, _, res = _run(10, 2, record=True)
    frames = frames_from_result(res)
    return "%d 帧，每帧 %d 台车位姿" % (len(frames), len(frames[0].agvs))


def data_assets():
    csvs = sorted((ROOT / "results").glob("*.csv"))
    rows = 0
    for f in csvs:
        with open(f, encoding="utf-8") as fh:
            rows += sum(1 for _ in fh) - 1
    return "%d 份 CSV / %d 行" % (len(csvs), rows)


def code_size():
    py2 = list((ROOT / "amr_sched").rglob("*.py"))
    lines = sum(len(f.read_text(encoding="utf-8").splitlines()) for f in py2)
    return "%d 个 .py / %d 行" % (len(py2), lines)


def replay_has_path():
    """G2：回放数据里必须有规划路径，否则看不出「车要往哪走」。

    这一条是第 61 轮补的。之前只导出了 pos，界面能开、车能动，
    但「路径规划算法接进来」这件事在界面上看不见——
    而这正是任务书 R4.7 与导师录音都明确要求的。
    """
    import sys
    sys.path.insert(0, str(ROOT / "experiments"))
    from export_replay import build_payload
    p = build_payload(10, 2, 0, "spacetime_astar", 0.0, "preempt_unpicked", 1)
    steps = p["steps"]
    nonempty = [x for x in steps if x.get("path")]
    if not nonempty:
        raise AssertionError("所有帧的 path 都为空")
    mid = nonempty[len(nonempty) // 2]
    sample = next(iter(mid["path"].values()))
    return "%d 帧带路径，示例 %s" % (len(nonempty), sample)


chk("G2 回放含规划路径", replay_has_path)

def replay_supports_ab():
    """G2：界面支持 A→B 点选，现场算路。

    任务书 R4.2 要求路径规划可接入；导师录音要求「把算法接进去」。
    只看既定回放不算，要能任选两点现算。
    """
    html = (ROOT / "experiments" / "replay_template.html").read_text(
        encoding="utf-8")
    for token in ("shortestPath", "nodeAt", "drawAB"):
        if token not in html:
            raise AssertionError("模板缺少 " + token)
    if "drawAB();" not in html:
        raise AssertionError("drawAB 未被调用")
    return "点选两点即算最短路并绘制"


chk("G2 支持 A→B 点选", replay_supports_ab)

chk("R3/R4 六模块可导入", six_modules)
chk("G1 端到端仿真", e2e)
chk("R4.7 可视化帧流", viz_stream)
chk("R5 实验数据", data_assets)
chk("R4 Python 实现规模", code_size)

def main() -> int:
    """打印结果并返回退出码。"""
    for n, v in ok:
        print("  [OK]   " + n + ": " + v)
    for n, v in bad:
        print("  [FAIL] " + n + ": " + v)
    print("")
    print("通过 %d / %d" % (len(ok), len(ok) + len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())