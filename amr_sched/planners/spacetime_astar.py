"""时空 A* 规划器——带冲突消解。

算法
----
在 ``(节点, 时间步)`` 的扩展图上做 A*：

* **等待**动作：留在当前节点，代价 1 步。
* **移动**动作：沿边 u→v，代价 ``n`` 步（n = 边长 / 速度 / dt 向上取整）。
* 约束来自 :class:`ReservationTable`：节点按容量、边按容量。

对向相遇
--------
真实的车间单车道是**一条物理通道、两个方向**。如果只在有向边上做预约，
两车对向相遇时各自的边预约都不重叠，会规划出"迎面撞上"的路径。

因此每条边的物理资源 id 取 ``frozenset({src, dst})`` 的无序规范形式，
预约与查重都按物理通道做。这样 u→v 与 v→u 自动互斥。
"""

from __future__ import annotations

import heapq
import itertools
import math

from amr_sched.network import Network
from amr_sched.planners.base import PathPlanner, PlannedPath
from amr_sched.reservation import Interval, ReservationTable


def physical_edge_id(src: str, dst: str) -> tuple[str, str]:
    """把有向边归一成物理通道 id，使 u→v 与 v→u 映射到同一资源。"""
    return (src, dst) if src <= dst else (dst, src)


class SpaceTimeAStarPlanner(PathPlanner):
    """时空 A*，逐条边对齐容量约束。"""

    name = "spacetime_astar"
    #: 会读取并写入时空预约
    reservations_aware = True

    def __init__(
        self,
        max_speed: float = 1.0,
        dt: float = 1.0,
        tie_break_noise: float = 0.0,
        seed: int = 0,
        max_expansions: int = 200_000,
    ) -> None:
        super().__init__(max_speed=max_speed, dt=dt)
        self.tie_break_noise = tie_break_noise
        self._rng = None
        self._seed = seed
        #: 单次 ``plan`` 允许扩展的状态数上限。
        #:
        #: 状态是 ``(节点, 时刻)``；等待动作让状态空间随时长线性增长，
        #: 因此**极端拥塞 + 大 horizon** 时 A* 会退化成准穷举。
        #: 实测 grid 布局 50 任务 / 6 车 / seed 0 单次 plan 跑满 CPU 数分钟不返回，
        #: 整个实验脚本挂死。超限后返回 ``None``——调用方本来就会把「规划失败」
        #: 当成「本步不动、下一步重试」，语义与现有 ``_plan_and_reserve`` 一致。
        self.max_expansions = max_expansions

    # ---------- 启发式 ----------

    def _heuristic_table(self, network: Network, goal: str) -> dict[str, float]:
        """各节点到 goal 的静态最短时间（秒），用于 A* 启发。"""
        rev: dict[str, list[tuple[str, float]]] = {n: [] for n in network.nodes}
        for (src, dst), e in network.edges.items():
            rev[dst].append((src, e.traversal_time(self.max_speed)))
        dist: dict[str, float] = {goal: 0.0}
        heap: list[tuple[float, str]] = [(0.0, goal)]
        done: set[str] = set()
        while heap:
            d, u = heapq.heappop(heap)
            if u in done:
                continue
            done.add(u)
            for v, w in rev[u]:
                nd = d + w
                if nd < dist.get(v, float("inf")):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        return dist

    # ---------- 主逻辑 ----------

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

        h_tbl = self._heuristic_table(network, goal)
        if goal not in h_tbl:
            return None

        def h(node: str) -> int:
            v = h_tbl.get(node)
            if v is None:
                return 10**9
            return int(math.floor(v / self.dt))

        # (f, g, node, tiebreak)
        counter = itertools.count()
        start_node_capacity = network.node(start).capacity
        # 起点：若起点被别的车占满，允许等待
        frontier: list[tuple[int, int, int, str]] = [
            (h(start), 0, next(counter), start)
        ]
        best_g: dict[tuple[str, int], int] = {(start, start_step): 0}
        parent: dict[tuple[str, int], tuple[str, int] | None] = {(start, start_step): None}
        wait_count: dict[tuple[str, int], int] = {(start, start_step): 0}

        goal_state: tuple[str, int] | None = None
        expansions = 0

        while frontier:
            # 状态爆炸保护：见 ``max_expansions`` 的说明。
            expansions += 1
            if expansions > self.max_expansions:
                return None
            f, g, _, node = heapq.heappop(frontier)
            root = f - g
            t = start_step + g
            if t > horizon:
                continue
            state = (node, t)
            if best_g.get(state, 10**18) < g:
                continue
            if node == goal:
                goal_state = state
                break

            # 动作 1：等待
            if t + 1 <= horizon:
                span = Interval(t, t + 1)
                if reservations.is_node_free(
                    node, span, network.node(node).capacity, exclude_agv=agv_id
                ):
                    ns = (node, t + 1)
                    if g + 1 < best_g.get(ns, 10**18):
                        best_g[ns] = g + 1
                        parent[ns] = state
                        wait_count[ns] = wait_count[state] + 1
                        heapq.heappush(frontier, (g + 1 + h(node), g + 1, next(counter), node))

            # 动作 2：沿边移动
            for e in network.out_edges(node):
                n = self.steps_for(e.traversal_time(self.max_speed))
                arrive = t + n
                if arrive > horizon:
                    continue
                phys = physical_edge_id(e.src, e.dst)
                edge_span = Interval(t, arrive)
                edge_cap = min(e.capacity, network.edge(e.dst, e.src).capacity)                     if network.has_edge(e.dst, e.src) else e.capacity
                # 物理通道：两个方向都要查
                occ = reservations.edge_conflicts(phys[0], phys[1], edge_span, exclude_agv=agv_id)
                if network.has_edge(phys[1], phys[0]) and phys[1] != phys[0]:
                    occ += reservations.edge_conflicts(phys[1], phys[0], edge_span, exclude_agv=agv_id)
                if len(set(occ)) >= edge_cap:
                    continue
                arrive_span = Interval(arrive, arrive + 1)
                if not reservations.is_node_free(
                    e.dst, arrive_span, network.node(e.dst).capacity, exclude_agv=agv_id
                ):
                    continue
                ns = (e.dst, arrive)
                ng = g + n
                if ng < best_g.get(ns, 10**18):
                    best_g[ns] = ng
                    parent[ns] = state
                    wait_count[ns] = wait_count[state]
                    heapq.heappush(frontier, (ng + h(e.dst), ng, next(counter), e.dst))

        if goal_state is None:
            return None

        # 回溯
        chain: list[tuple[str, int]] = []
        cur: tuple[str, int] | None = goal_state
        while cur is not None:
            chain.append(cur)
            cur = parent[cur]
        chain.reverse()

        # 折叠连续相同的节点：等待动作会生成 (u,t),(u,t+1)... 这类状态，
        # 直接取 node 序列会出现自环 u->u。保留每段连续相同节点的**最后一个**
        # （即真正离开该节点的时刻）。
        collapsed: list[tuple[str, int]] = []
        for node, t in chain:
            if collapsed and collapsed[-1][0] == node:
                collapsed[-1] = (node, t)
            else:
                collapsed.append((node, t))

        nodes = [c[0] for c in collapsed[1:]]
        arrivals = [c[1] for c in collapsed[1:]]
        return PlannedPath(
            nodes=nodes, arrival_steps=arrivals, wait_steps=wait_count[goal_state]
        )
