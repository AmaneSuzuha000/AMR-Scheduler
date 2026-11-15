"""可组合的分配器：调度规则 × 分配算法（任务书 R2.4a）。

为什么要拆成两层
----------------
"任务分配"在文献里其实是两个决策叠在一起：

* **调度规则**决定先处理哪个任务（FIFO / EDD / SPT / 优先级 / 临界比）；
* **分配算法**决定这个任务交给哪台车（贪心 / 最近车 / 匈牙利最优匹配）。

如果只拿一个整体策略跟另一个整体策略比，说不清收益来自"排序"还是"配对"。
分开之后就可以做消融：固定配对法只换顺序，或固定顺序只换配对法。
"""

from __future__ import annotations

from amr_sched.allocation.base import AllocationStrategy
from amr_sched.allocation.dispatch import DispatchRule, order_tasks
from amr_sched.allocation.hungarian import hungarian
from amr_sched.entities import Agv, Task
from amr_sched.network import Network

# 匈牙利算法不接受 inf，用一个足够大的数代替
_BIG = 1e9


class RuleBasedAllocator(AllocationStrategy):
    """按 ``dispatch`` 排序任务，按 ``assign`` 配对 AGV。

    Parameters
    ----------
    dispatch:
        FIFO / EDD / SPT / PRIORITY / CR，见 ``dispatch.py``。
    assign:
        * ``greedy``    —— 全局按代价从小到大配对（局部贪心，快）
        * ``nearest``   —— 按任务顺序，每个任务取最近的车（任务驱动）
        * ``hungarian`` —— 一步内最小总代价的完整匹配（最慢，作为上界参照）
    """

    def __init__(self, dispatch: DispatchRule = DispatchRule.PRIORITY,
                 assign: str = "greedy") -> None:
        if assign not in ("greedy", "nearest", "hungarian"):
            raise ValueError("未知分配算法: %r" % (assign,))
        self.dispatch = DispatchRule(dispatch)
        self.assign = assign
        self.name = "%s+%s" % (self.dispatch.value, assign)

    # ---------- 内部工具 ----------

    @staticmethod
    def _tt(network: Network, a: str, b: str, max_speed: float,
            cache: dict[tuple[str, str], float]) -> float:
        if a == b:
            return 0.0   # 已在原地：最短时间必须是 0，不能落到 inf
        key = (a, b)
        if key not in cache:
            d = network.shortest_travel_times(a, max_speed)
            for k, v in d.items():
                cache[(a, k)] = v
            cache[key] = d.get(b, float("inf"))
        return cache[key]

    def _cost(self, network: Network, agv: Agv, task: Task, max_speed: float,
              cache: dict[tuple[str, str], float]) -> float:
        here = agv.cur_node if agv.cur_node is not None else agv.start_node
        to_pick = self._tt(network, here, task.pickup, max_speed, cache)
        to_drop = self._tt(network, task.pickup, task.dropoff, max_speed, cache)
        return to_pick + to_drop

    # ---------- 主入口 ----------

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

        ready = [t for t in pending_tasks if t.release_step <= current_step]
        if not ready:
            return []

        ordered = order_tasks(
            ready, self.dispatch, current_step, network, max_speed,
            travel_time_cache,
        )

        if self.assign == "greedy":
            return self._assign_greedy(network, idle_agvs, ordered, max_speed,
                                       travel_time_cache)
        if self.assign == "nearest":
            return self._assign_nearest(network, idle_agvs, ordered, max_speed,
                                        travel_time_cache)
        return self._assign_hungarian(network, idle_agvs, ordered, max_speed,
                                      travel_time_cache)

    def _assign_greedy(self, network, idle_agvs, ordered, max_speed, cache):
        cand = []
        for agv in idle_agvs:
            for task in ordered:
                c = self._cost(network, agv, task, max_speed, cache)
                if c == float("inf"):
                    continue
                cand.append((c, agv.agv_id, task.task_id, agv, task))
        cand.sort(key=lambda x: (x[0], x[1], x[2]))
        used_a, used_t, out = set(), set(), []
        for _, aid, tid, agv, task in cand:
            if aid in used_a or tid in used_t:
                continue
            used_a.add(aid)
            used_t.add(tid)
            out.append((agv, task))
        return out

    def _assign_nearest(self, network, idle_agvs, ordered, max_speed, cache):
        """按任务顺序逐个处理，每个任务挑当前最近的一台空闲车。"""
        free = list(idle_agvs)
        out = []
        for task in ordered:
            if not free:
                break
            best = None
            for agv in free:
                c = self._cost(network, agv, task, max_speed, cache)
                if c == float("inf"):
                    continue
                key = (c, agv.agv_id)
                if best is None or key < best[0]:
                    best = (key, agv)
            if best is None:
                continue
            agv = best[1]
            free.remove(agv)
            out.append((agv, task))
        return out

    def _assign_hungarian(self, network, idle_agvs, ordered, max_speed, cache):
        """一步内最小总代价的完整匹配。

        代价矩阵取 ``min(len(idle_agvs), len(ordered))`` 规模，不足补大数，
        解出来后丢弃补位配对。
        """
        n = min(len(idle_agvs), len(ordered))
        if n == 0:
            return []

        # 公平性：先按"该任务能被任何空闲车以最低代价完成"挑出 n 个候选任务，
        # 再做方阵求解。
        #
        # 为什么不能直接取 ordered[:n]：那样匈牙利只看得到调度顺序里的前 n 个
        # 任务，而贪心法遍历的是**全部**任务对。两者候选集不同，比较就没意义
        # （初版实测显示"每步最优"反而比贪心慢 9.8%，纯粹是候选集被人为缩小）。
        def best_cost(t: Task) -> float:
            return min(self._cost(network, a, t, max_speed, cache) for a in idle_agvs)

        # 两阶段：
        # 1) 先按"最便宜的车能多快完成"取 2n 个候选，保证不比贪心吃亏
        #    （贪心遍历全部任务；这里至少给到两倍车数的候选池）；
        # 2) 在这 2n 个候选里**按调度规则排序**取前 n 个。
        #
        # 只做第 1 步会让调度规则完全失效——实测 4 种规则得出完全相同的结果，
        # 消融就失去意义了。加上第 2 步，顺序才有影响。
        pool_size = max(n, 2 * n)
        pool = sorted(
            (t for t in ordered
             if best_cost(t) < float("inf")),
            key=lambda t: (best_cost(t), t.task_id),
        )[:pool_size]
        pool_ids = {t.task_id for t in pool}
        tasks = [t for t in ordered if t.task_id in pool_ids][:n]
        if len(tasks) < n:
            chosen = {t.task_id for t in tasks}
            for t in pool:
                if len(tasks) >= n:
                    break
                if t.task_id not in chosen:
                    tasks.append(t)
                    chosen.add(t.task_id)
        agvs = idle_agvs[:n]
        cost: list[list[float]] = []
        for agv in agvs:
            row = []
            for task in tasks:
                c = self._cost(network, agv, task, max_speed, cache)
                row.append(_BIG if c == float("inf") else c)
            cost.append(row)
        pairs = hungarian(cost)
        out = []
        for i, j in sorted(pairs):
            if cost[i][j] < _BIG / 2:
                out.append((agvs[i], tasks[j]))
        return out
