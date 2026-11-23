"""朴素基线策略：代表"不做研究的做法"，用于 基准对比。

为什么需要它
------------
此前的对比都是"自己 vs 自己"：时空 A* 比静态 Dijkstra、
匈牙利比贪婪。这只能说明"我的两个设计里哪个更好"，
不能说明"我的方法比不做研究强多少"。

基准策略代表**朴素做法**：

1. ``nearest_shortest``：空闲车选最近的，路径走静态最短路，
   **完全不做冲突消解**——车与车之间靠运气，撞了就等。
2. ``fifo_shortest``：任务按先来后到，车按编号顺序，静态最短路。

这两条对应实际车间里"上了 AGV 但没上调度系统"的状态，
作为朴素对照基线。

实现说明
--------
不复用 ``GreedyAllocator``，因为它已经做了"取货点+送达点"综合代价评估；
基线要更笨，只看当前距离。故意写简单，避免"基线被偷偷加强"。
"""

from __future__ import annotations

from amr_sched.entities import AgvState, Task
from amr_sched.network import Network


class NearestShortestAllocator:
    """最笨的分配：每台空闲车拿离它最近的那个待办任务。

    不考虑送达点、不考虑负载均衡、不考虑截止期。
    对应"车自己看到啥就干啥"的朴素做法。
    """

    name = "nearest_shortest"

    def allocate(self, network: Network, idle_agvs, pending_tasks,
                 current_step: int, max_speed: float, travel_time_cache):
        if not idle_agvs or not pending_tasks:
            return []

        def tt(a: str | None, b: str) -> float:
            # 车还没动过时 cur_node 是 None（起点只存在 start_node 里）。
            # 早期版本没处理，直接 KeyError 崩掉。
            key = (a, b)
            if key in travel_time_cache:
                return travel_time_cache[key]
            if a is None:
                return float("inf")
            dist = network.shortest_travel_times(a, max_speed)
            for k, v in dist.items():
                travel_time_cache[(a, k)] = v
            return dist.get(b, float("inf"))

        pairs = []
        taken_tasks = set()
        # 车按 id 顺序，各取离自己最近的未分配任务（会抢，后到的只能退而求其次）
        for agv in sorted(idle_agvs, key=lambda a: a.agv_id):
            best = None
            best_d = float("inf")
            for task in pending_tasks:
                if task.task_id in taken_tasks:
                    continue
                # 只看取货距离，完全无视送达点与截止期
                here = agv.cur_node or agv.start_node
                d = tt(here, task.pickup)
                if d < best_d:
                    best_d = d
                    best = task
            if best is not None and best_d < float("inf"):
                pairs.append((agv, best))
                taken_tasks.add(best.task_id)
        return pairs


class FifoShortestAllocator:
    """先来后到：任务按下达顺序，车按编号顺序，一一配对。

    比 ``NearestShortestAllocator`` 更机械，代表"有队列但没优化"的做法。
    """

    name = "fifo_shortest"

    def allocate(self, network: Network, idle_agvs, pending_tasks,
                 current_step: int, max_speed: float, travel_time_cache):
        if not idle_agvs or not pending_tasks:
            return []
        agvs = sorted(idle_agvs, key=lambda a: a.agv_id)
        tasks = sorted(pending_tasks,
                       key=lambda t: (t.release_step, t.task_id))
        n = min(len(agvs), len(tasks))
        return [(agvs[i], tasks[i]) for i in range(n)]
