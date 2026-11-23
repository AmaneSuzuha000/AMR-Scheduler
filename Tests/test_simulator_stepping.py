"""分步执行接口与整体执行的行为等价性。


以便 Web 控制台能"运行中"逐帧推送状态。

拆分最大的风险是**悄悄改变仿真语义**——比如终止条件的次序、
``_record`` 的调用时机、``purge_before`` 的触发点。
只要有一处不同，历史 CSV 就全部作废，而这种偏差不会报错。

因此这里用三重断言锁死等价性：
（1）逐字段比对 summary；（2）逐帧比对轨迹；（3）比对全量快照的 SHA256。
"""

from __future__ import annotations

import hashlib
import json

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator


def _build(seed: int = 0, n_tasks: int = 20, n_agvs: int = 3):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(
        net, tasks, agvs, StaticDijkstraPlanner(), GreedyAllocator(),
        SimulationConfig(max_steps=40_000, seed=seed, record_trajectory=True),
    )
    return sim


def _digest(result) -> str:
    """对结果的**全部可观测字段**取摘要，任何一处漂移都会变。"""
    payload = {
        "summary": {k: (round(v, 9) if isinstance(v, float) else v)
                    for k, v in sorted(result.summary().items())},
        "tasks": [(t.task_id, t.status.value, t.pickup_step, t.complete_step)
                  for t in result.tasks],
        "agvs": [(a.agv_id, a.cur_node, a.state.value, a.busy_steps, a.idle_steps,
                  a.blocked_steps, a.loaded_steps, a.deadhead_steps)
                 for a in result.agvs],
        "trajectory_len": len(result.trajectory),
        "frames": [(r.step, sorted(r.agv_positions.items()),
                    sorted(r.agv_states.items()),
                    sorted(r.task_counts.items()))
                   for r in result.trajectory],
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def test_stepping_matches_run_exactly():
    """reset + 反复 step_once 必须与 run() 产生同一份结果。"""
    whole = _build().run()

    sim = _build()
    sim.reset()
    guard = 0
    while sim.step_once():
        guard += 1
        assert guard <= 40_000, "step_once 没有终止，疑似死循环"
    stepped = sim._build_result()
    assert _digest(whole) == _digest(stepped), (
        "分步执行与整体执行结果不一致——重构改变了仿真语义")


def test_step_count_and_termination():
    """终止后 step_once 必须返回 False，且不再推进游标。"""
    sim = _build()
    sim.reset()
    n = 0
    while sim.step_once():
        n += 1
    assert sim.is_finished(), "循环结束后 is_finished() 仍为 False"
    assert sim.step_once() is False, "已终止后 step_once 仍返回 True"
    assert sim._step_cursor == n, "终止后游标被多推了一步"

    whole = _build().run()
    assert n == whole.finished_step, (
        "分步步数 %d 与 run() 的 finished_step %d 不等" % (n, whole.finished_step))

def test_reset_is_idempotent():
    """连续 reset 两次，结果必须相同——reset 不能有残留状态。"""
    sim = _build()
    sim.reset()
    while sim.step_once():
        pass
    sim.reset()
    while sim.step_once():
        pass
    d1 = _digest(sim._build_result())

    sim2 = _build()
    sim2.reset()
    while sim2.step_once():
        pass
    assert d1 == _digest(sim2._build_result()), "reset 有残留状态"
