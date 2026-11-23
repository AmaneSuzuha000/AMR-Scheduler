"""回归测试：LOADING 等待态不得形成活锁。

背景
----
AGV 到达取货点、装完货之后要现场规划第二段（取货点→卸货点）。
如果第二段规划失败，车会停在取货点上以 ``LOADING`` 状态等下一步重试。

问题在于：这个分支**不写** ``_blocked_since``，
于是 ``_break_deadlock`` 看不见它——车既不进"阻塞计时"，也就永远不会被解锁。

实测触发场景：grid 布局、50 任务、6 车、seed 0。
四个任务的取货点都是 B3，B3 容量 2。两台车各自停在 B3 上反复重规划失败，
另外两个任务永远分不到车位。完成数死在 46/50；
主循环空转到 ``max_steps``（12 万步），整个实验脚本挂死 50 分钟。

与死锁的区别
--------------
死锁是"大家都在等，谁也不动"；这是"系统整体在动，但特定任务永久饥饿"。
CPU 一直满载，进度条不动，日志也不报错——最难查的一类。
"""

from __future__ import annotations

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.layouts import build_layout
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def _run(layout: str, n_tasks: int, n_agvs: int, seed: int, max_steps: int):
    net = build_layout(layout, aisle_capacity=2)
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed,
                        dt=1.0, max_speed=1.0, urgent_ratio=0.15)
    net, tasks, agvs = generate_scenario(sc, network=net)
    sim = Simulator(
        net, tasks, agvs,
        SpaceTimeAStarPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=max_steps, seed=seed),
    )
    return sim.run(), tasks


def test_grid_50_6_all_tasks_complete():
    """这个算例以前会卡在 46/50。修复后应当全部完成。"""
    result, tasks = _run("grid", 50, 6, 0, 30_000)
    done = sum(1 for t in tasks if t.complete_step is not None)
    assert done == 50, "只完成 %d/50，活锁复发" % done


def test_grid_50_6_finishes_early():
    """不仅要跑完，还要**早早**跑完——不是拖到 horizon 才凑齐。"""
    result, tasks = _run("grid", 50, 6, 0, 30_000)
    assert result.finished_step < 10_000, (
        "finished_step=%d，说明主循环在空转" % result.finished_step)


def test_loading_waits_are_counted_as_blocked():
    """LOADING 等待必须计入阻塞统计，否则死锁打破看不见它。"""
    result, _ = _run("grid", 50, 6, 0, 30_000)
    # 修复前 blocked_steps 只有真正 BLOCKED 的那部分，
    # 修复后装载等待也计入，因此这个数应当大于 0 且有实质量级。
    assert result.blocked_steps > 0


def test_no_hung_state_across_layouts():
    """三种布局、6 车、50 任务都不能"空转挂死"。

    ring / grid 必须全部完成；hub 允许少量未完成——
    那是 6.9.4 记录的**枢纽物理饱和**（容量 2 时 6 台车争一个中心），
    不是代码缺陷，也**不该**被这条测试判红。

    真正要抓的是：完成数停在某个值之后再无进展、
    主循环一路空转到 horizon。判据就是 ``finished_step``。
    """
    expect = {"ring": 50, "grid": 50}
    for layout in ("ring", "grid", "hub"):
        result, tasks = _run(layout, 50, 6, 0, 60_000)
        done = sum(1 for t in tasks if t.complete_step is not None)
        if layout in expect:
            assert done == expect[layout], "%s 布局只完成 %d/50" % (layout, done)
            # 这两个布局缓存与环道充足，饿死就是 bug。
            assert result.finished_step < 60_000, (
                "%s 跑满了 horizon，说明主循环空转" % layout)
        else:
            assert done >= 40, "hub 完成 %d/50，比已知的饱和水平还低" % done
    # hub 是个例外：它的可停靠节点只有 B1 / B2 / H 三个（容量 2），
    # 6 台车时物理上就无处让位——``_nearest_park`` 会返回 None，
    # 空车只能占着 B2 不动。这是**布局的资源上限**，不是活锁：
    # ring/grid 有足够的缓存与环道节点，同样的代码不会卡住。
    # 因此这里只对 ring/grid 断言「早退出」。
