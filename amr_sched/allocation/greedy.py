"""贪心分配：优先加急，再按"取货+送达"总时间最短。

规则
----
1. **加急任务优先**（任务书 R2.5 的加急语义）。
2. 同一优先级内，按 ``(AGV 到取货点时间 + 取货点到卸货点时间)`` 从小到大配对。
3. 每轮只做确定性配对——同样输入必得同样输出，保证实验可复现。
"""

from __future__ import annotations

from amr_sched.allocation.base import AllocationStrategy
from amr_sched.entities import Agv, Task, TaskPriority
from amr_sched.network import Network


class GreedyAllocator(AllocationStrategy):
    name = "greedy"

    def allocate(
        self,
        network: Network,
        idle_agvs: list[Agv],
        pending_tasks: list[Task],
        current_step: int,
        max_speed: float,
        travel_time_cache: dict[tuple[str, str], float],
    ) -> list[tuple[Agv, Task]]:
        if not idle_agvs or not pending_tasks:
            return []

        def tt(a: str, b: str) -> float:
            if a == b:
                # 车已经在取货点：代价是 0。
                # 早期版本直接查最短路表，而该表**不含源点自身**，
                # 于是返回 inf 把"最优配对"整对丢掉——实测会让
                # 两台车停在取货点却一个任务都分不出去。
                return 0.0
            key = (a, b)
            if key not in travel_time_cache:
                d = network.shortest_travel_times(a, max_speed)
                for k, v in d.items():
                    travel_time_cache[(a, k)] = v
                travel_time_cache[key] = d.get(b, float("inf"))
            return travel_time_cache[key]

        # 可分配的任务：已下达且仍待分配
        ready = [
            t for t in pending_tasks
            if t.release_step <= current_step
        ]
        if not ready:
            return []

        # 候选配对
        cand: list[tuple[int, float, str, str, Agv, Task]] = []
        for agv in idle_agvs:
            here = agv.cur_node if agv.cur_node is not None else agv.start_node
            for task in ready:
                to_pick = tt(here, task.pickup)
                if to_pick == float("inf"):
                    continue
                to_drop = tt(task.pickup, task.dropoff)
                if to_drop == float("inf"):
                    continue
                cost = to_pick + to_drop
                # 加急优先用负优先级参与排序：urgent(1) 排在 normal(0) 前
                cand.append((-int(task.priority), cost, agv.agv_id, task.task_id, agv, task))

        cand.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

        used_agv: set[str] = set()
        used_task: set[str] = set()
        out: list[tuple[Agv, Task]] = []
        for _, _, aid, tid, agv, task in cand:
            if aid in used_agv or tid in used_task:
                continue
            used_agv.add(aid)
            used_task.add(tid)
            out.append((agv, task))
        return out
