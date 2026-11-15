"""静态最短路规划器——**不做冲突消解**的基线。

用途：
1. 对照基线，量化"冲突消解"带来多少收益（任务书 R5.6 要求对比基准策略）。
2. 无冲突场景下的最优路径下界。
"""

from __future__ import annotations

import heapq

from amr_sched.network import Network
from amr_sched.planners.base import PathPlanner, PlannedPath
from amr_sched.reservation import ReservationTable


class StaticDijkstraPlanner(PathPlanner):
    """按行驶时间最短选路，**完全忽略**其他 AGV。"""

    name = "static_dijkstra"
    #: 基线：不理解预约，也不写预约
    reservations_aware = False

    def plan(
        self,
        network: Network,
        start: str,
        goal: str,
        start_step: int,
        reservations: ReservationTable,
        agv_id: str,
        horizon: int,
    ) -> PlannedPath | None:
        if start == goal:
            return PlannedPath(nodes=[], arrival_steps=[], wait_steps=0)

        best: dict[str, float] = {start: 0.0}
        prev: dict[str, str] = {}
        heap: list[tuple[float, str]] = [(0.0, start)]
        done: set[str] = set()

        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            if u == goal:
                break
            for e in network.out_edges(u):
                if e.dst in done:
                    continue
                nd = d + e.traversal_time(self.max_speed)
                if nd < best.get(e.dst, float("inf")):
                    best[e.dst] = nd
                    prev[e.dst] = u
                    heapq.heappush(heap, (nd, e.dst))

        if goal not in best:
            return None

        chain = [goal]
        while chain[-1] != start:
            chain.append(prev[chain[-1]])
        chain.reverse()

        nodes = chain[1:]
        arrival_steps: list[int] = []
        t = start_step
        for i in range(len(chain) - 1):
            e = network.edge(chain[i], chain[i + 1])
            t += self.steps_for(e.traversal_time(self.max_speed))
            arrival_steps.append(t)

        return PlannedPath(nodes=nodes, arrival_steps=arrival_steps, wait_steps=0)
