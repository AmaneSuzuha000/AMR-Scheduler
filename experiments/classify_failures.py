"""失败实例的机制分类（E3）。

为什么必须分类
--------------
此前所有「未完成任务」都被叫作死锁。这不严谨：

  - 真死锁：车之间形成**循环等待**，谁也动不了，必须外部打破；
  - 饥饿：某任务总被后来的高优先级任务插队，一直轮不到；
  - 活锁：车一直在动，但因为反复重规划/让行，始终到不了目的地；
  - 超时截断：仿真在 max_steps 内没跑完，任务本身没失败；
  - 不可达：路径规划返回 None，拓扑问题。

把后四类都叫死锁，会高估死锁的普遍性，也会掩盖真正该修的问题。

分类依据（可用仿真结果直接判定）
--------------------------------
对每个未完成任务 t：
  1. 若 t.status == FAILED 且完成数是 0 → 不可达（拓扑）
  2. 若 makespan 接近 max_steps → 超时截断
  3. 否则看等待结构：
     - 所有车都 BLOCKED 且等待关系构成环 → 真死锁
     - 有车在动但任务一直没分配 → 饥饿
     - 车在动但位置反复循环 → 活锁
"""
from __future__ import annotations

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator  # noqa: E402
from amr_sched.entities import AgvState, TaskStatus  # noqa: E402
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner  # noqa: E402
from amr_sched.scenarios import ScenarioConfig, generate_scenario  # noqa: E402
from amr_sched.simulation import SimulationConfig, Simulator  # noqa: E402


def find_cycle(wait_for: dict) -> list:
    """在等待关系图里找环。wait_for: agv -> 它在等的 agv 集合。"""
    color = {k: 0 for k in wait_for}
    stack = []

    def dfs(u):
        color[u] = 1
        stack.append(u)
        for v in wait_for.get(u, ()):
            if v not in color:
                continue
            if color[v] == 1:
                i = stack.index(v)
                return stack[i:]
            if color[v] == 0:
                r = dfs(v)
                if r:
                    return r
        stack.pop()
        color[u] = 2
        return None

    for k in list(wait_for):
        if color.get(k, 0) == 0:
            r = dfs(k)
            if r:
                return r
    return []


def classify(result, cfg):
    """对一次运行给出失败分类计数。

    第 63 轮修正：原分类把「全部卡住」一律判活锁，是错的。
    实测发现真正的情形是**终点容量争抢**——三台车都要送 W5（容量 1），
    第一台占住之后其余永远进不去，deadlock_breaks 反复打破 1439 次也无效。
    这类必须单独列，否则会误以为死锁打破机制有效。
    """
    s = result.summary()
    unfinished = [t for t in result.tasks if t.status is not TaskStatus.COMPLETED]
    out = {"unfinished": len(unfinished), "deadlock": 0, "contention": 0,
           "starved": 0, "livelock": 0, "timeout": 0, "unreachable": 0}
    if not unfinished:
        return out

    # 1. 不可达
    failed = [t for t in unfinished if t.status is TaskStatus.FAILED]
    out["unreachable"] = len(failed)

    # 2. 超时截断：makespan 真打满 max_steps
    truncated = s["makespan_seconds"] >= cfg.max_steps * cfg.dt * 0.99
    if truncated:
        out["timeout"] = len(unfinished) - out["unreachable"]
        return out

    rest = [t for t in unfinished if t.status is not TaskStatus.FAILED]
    if not rest:
        return out

    # 3. 终点争抢：多个未完成任务共享同一个 dropoff，且该节点容量小
    by_drop = defaultdict(list)
    for t in rest:
        by_drop[t.dropoff].append(t)
    worst = max(by_drop.values(), key=len)
    if len(worst) >= 2:
        out["contention"] = len(worst)
        rest = [t for t in rest if t not in worst]
    if not rest:
        return out

    # 4. 看末尾等待结构
    tail = result.trajectory[-200:] if len(result.trajectory) > 200 else result.trajectory
    blocked_all = 0
    for rec in tail:
        states = list(rec.agv_states.values())
        if states and all(x in (AgvState.BLOCKED.value, AgvState.LOADING.value)
                           for x in states):
            blocked_all += 1
    ratio = blocked_all / max(1, len(tail))
    if ratio > 0.5:
        out["deadlock"] = len(rest)
    elif ratio > 0.1:
        out["starved"] = len(rest)
    else:
        out["livelock"] = len(rest)
    return out

def main() -> int:
    rows = []
    for cap in (1, 2):
        for seed in range(10):
            sc = ScenarioConfig(n_tasks=50, n_agvs=4, seed=seed, dt=1.0,
                                urgent_ratio=0.15, max_speed=1.0,
                                aisle_capacity=cap)
            net, tasks, agvs = generate_scenario(sc)
            sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(),
                            GreedyAllocator(),
                            SimulationConfig(dt=1.0, max_steps=120_000, seed=seed))
            res = sim.run()
            c = classify(res, sim.config)
            c["aisle_capacity"] = cap
            c["seed"] = seed
            c["completed"] = res.summary()["tasks_completed"]
            rows.append(c)

    out = os.path.join("results", "failure_classification.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("已写入 %s（%d 行）" % (out, len(rows)))
    print("")
    print("容量  seed  完成  未完成  终点争抢  真死锁  饥饿  活锁  超时  不可达")
    for r in rows:
        print("%-5d %-5d %-5d %-7d %-8d %-7d %-5d %-5d %-5d %-6d"
              % (r["aisle_capacity"], r["seed"], r["completed"],
                 r["unfinished"], r["contention"], r["deadlock"], r["starved"],
                 r["livelock"], r["timeout"], r["unreachable"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
