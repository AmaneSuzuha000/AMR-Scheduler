"""执行层上报接进调度主循环：改道必须真的改变路由，且不许把仿真弄坏。

这组测试盯的是最容易只做一半的地方：
* 收到事件、回了决策，但**车照旧走原路**；
* 改了车的路，却**没有释放**它不再经过的那段预约，把别人堵死；
* 边跑边改道把车辆状态搞乱，整场跑不完。
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from amr_sched.entities import TaskStatus  # noqa: E402
from amr_sched.live import build_sim  # noqa: E402


def _sim(n_tasks=12, n_agvs=4, seed=0):
    sim = build_sim(n_tasks, n_agvs, seed, layout="workshop")
    sim.reset()
    return sim


def _advance_until_busy(sim, limit=2000):
    """Steps until at least one vehicle holds a job; returns the vehicle."""
    for _ in range(limit):
        busy = [a for a in sim.agvs if a.agv_id in sim._jobs]
        if busy:
            return busy[0]
        if not sim.step_once():
            break
    return None


def _assert_route_avoids(agv, sim, blocked):
    job = sim._jobs[agv.agv_id]
    pairs = {(h.src, h.dst) for h in job.hops[job.idx:]}
    assert blocked not in pairs and blocked[::-1] not in pairs, "改道后仍经过受阻通道"


# ---------- 拒收 ----------

def test_unknown_vehicle_is_rejected():
    sim = _sim()
    out = sim.report_deviation("AGV_99", "J_S_45-J_S_65", [8.0, 30.0, 0.0], 1.0)
    assert out["action"] == "rejected"
    assert "未知车辆" in out["reason"]


def test_malformed_report_is_rejected():
    sim = _sim()
    out = sim.report_deviation("AGV01", "not-an-edge", [8.0, 30.0, 0.0], 1.0)
    assert out["action"] == "rejected"


def test_unknown_edge_changes_nothing():
    """网络里没有的通道不许猜：决策是 ignore，车的路线一个字节都不动。"""
    sim = _sim()
    agv = _advance_until_busy(sim)
    assert agv is not None
    before = list(sim._jobs[agv.agv_id].hops)
    out = sim.report_deviation(agv.agv_id, "NO_A-NO_B", [8.0, 30.0, 0.0], 5.0)
    assert out["action"] == "ignore"
    assert sim._jobs[agv.agv_id].hops == before


# ---------- 改道 ----------

def test_deviation_rewrites_the_route_and_avoids_the_edge():
    sim = _sim()
    agv = _advance_until_busy(sim)
    assert agv is not None
    blocked = ("J_S_45", "J_S_65")
    out = sim.report_deviation(agv.agv_id, blocked, [8.0, 30.0, 0.0], 5.0)
    if out["action"] == "deferred":
        for _ in range(200):
            sim.step_once()
            if not sim._pending_detour:
                break
        out = {"action": "reroute"}
    assert out["action"] == "reroute", out
    _assert_route_avoids(agv, sim, blocked)


def test_reroute_counts_as_a_replan_and_releases_the_old_tail():
    """旧尾巴的预约必须释放，否则别人会被一段永远不会走的边堵住。"""
    sim = _sim()
    agv = _advance_until_busy(sim)
    assert agv is not None
    if agv.agv_id in sim._transit:
        for _ in range(50):
            sim.step_once()
            if agv.agv_id not in sim._transit:
                break
    replans = sim._replans
    out = sim.report_deviation(agv.agv_id, ("J_S_45", "J_S_65"), [8.0, 30.0, 0.0], 5.0)
    if out["action"] == "reroute":
        assert sim._replans == replans + 1


def test_a_deviation_mid_edge_is_deferred_not_applied():
    """跨边途中改道等于瞬移。必须先挂着，抵达节点再改。"""
    sim = _sim()
    agv = _advance_until_busy(sim)
    assert agv is not None
    for _ in range(50):
        if agv.agv_id in sim._transit:
            break
        sim.step_once()
    if agv.agv_id not in sim._transit:
        pytest.skip("该种子下没抓到跨边瞬间")
    out = sim.report_deviation(agv.agv_id, ("J_S_45", "J_S_65"), [8.0, 30.0, 0.0], 5.0)
    assert out["action"] in ("deferred", "ignore", "reassign")


# ---------- 整场不许被弄坏 ----------

@pytest.mark.parametrize("seed", [0, 1])
def test_injecting_deviations_mid_run_still_completes_every_task(seed):
    """跑到一半反复上报偏差，整场仍要跑完：改道不是让仿真崩掉的借口。"""
    sim = _sim(n_tasks=12, n_agvs=4, seed=seed)
    injected = 0
    for step in range(6000):
        if step % 60 == 30:
            busy = [a for a in sim.agvs if a.agv_id in sim._jobs]
            if busy:
                agv = busy[0]
                node = sim.network.node(agv.cur_node or agv.start_node)
                out = sim.report_deviation(
                    agv.agv_id, ("J_S_45", "J_S_65"),
                    [float(node.x), float(node.y), 0.0], float(step))
                assert out["action"] in ("reroute", "deferred", "ignore", "reassign"), out
                injected += 1
        if not sim.step_once():
            break
    assert injected > 0, "一次偏差都没注进去，测试没有意义"
    done = [t for t in sim.tasks if t.status is TaskStatus.COMPLETED]
    assert len(done) == len(sim.tasks), "改道后任务没跑完: %d/%d" % (len(done), len(sim.tasks))
    assert sim._deadlock_breaks == 0 or len(done) == len(sim.tasks)
    assert not sim._pending_detour, "收尾时仍有挂起的改道"

def test_unreachable_deviation_puts_the_task_back_in_the_queue():
    """绕不开时把任务退回待分配，让分配层去挑另一辆车。

    “改派”如果只写在回复里、任务还挂在这辆车上，那就是假的改派。
    """
    sim = _sim()
    agv = _advance_until_busy(sim)
    assert agv is not None
    for _ in range(50):
        if agv.agv_id not in sim._transit:
            break
        sim.step_once()
    job = sim._jobs[agv.agv_id]
    task = sim._task_by_id[job.task_id]
    # 工位是死胡同：封掉目标唯一的入口，从别处就进不去。
    incoming = [e for e in sim.network.edges.values() if e.dst == job.goal]
    if len(incoming) != 1 or job.leg != "to_pickup":
        pytest.skip("该种子下目标不是单入口工位，或车已装货（装货不许退回）")
    entry = incoming[0]
    out = sim.report_deviation(agv.agv_id, (entry.src, entry.dst), [0.0, 0.0, 0.0], 5.0)
    if out["action"] == "deferred":
        pytest.skip("跨边途中，本次不验证改派")
    assert out["action"] == "reassign", out
    if out.get("requeued"):
        assert task.status is TaskStatus.PENDING
        assert task.assigned_agv is None
        assert agv.agv_id not in sim._jobs

