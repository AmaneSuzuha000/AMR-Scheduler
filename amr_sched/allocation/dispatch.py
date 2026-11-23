"""调度规则：决定"先做哪个任务"（系统规范 的前半部分）。

任务分配其实是两个正交的决策
--------------------------
1. **调度规则（dispatch rule）**：待分配任务按什么顺序处理；
2. **分配算法（assignment）**：给定顺序后，任务交给哪台车。

本模块只管第 1 件事，第 2 件在 ``rule_based.py``。
这样可以把"顺序"与"配对"分开做消融，才知道收益来自哪里。
"""

from __future__ import annotations

from enum import Enum

from amr_sched.entities import Task
from amr_sched.network import Network


class DispatchRule(str, Enum):
    """经典调度规则，取自通用生产调度（排序）理论，不是 AGV 专有规则。

    文献归属要写准：FIFO/EDD/SPT/CR 这套排序规则来自**单机与车间调度理论**，
    在 AGV 场景的适用性有实证支持（如 Le-Anh & De Koster 2006 综述、
    胡晓阳等 2022 在解码中嵌入 FIFO）。
    注意 **不要** 把它们归到 Egbelu & Tanchoco (1984) 名下——该文的规则是
    面向"工位请求取货"排优先级的，与这里"任务排序"不是同一类东西。
    """

    FIFO = "fifo"
    """先到先服务：按下达步数排序。最朴素，也最公平。"""

    EDD = "edd"
    """最早截止优先（Earliest Due Date）：按截止步升序。
    单机下可最小化最大拖期，是多机调度的常用启发式。"""

    SPT = "spt"
    """最短作业优先（Shortest Processing Time）：按"取货到送达"时间升序。
    能压低平均流程时间，但会让长任务饿死。"""

    PRIORITY = "priority"
    """优先级 + 先到先服务：加急任务一律排前面，同级按 FIFO。"""

    CR = "cr"
    """临界比（Critical Ratio）：(截止步 - 当前步) / 剩余作业时间。
    比值越小越紧急。兼顾"还剩多少时间"与"还要干多久"。"""


def _travel(network: Network, a: str, b: str, max_speed: float,
            cache: dict[tuple[str, str], float]) -> float:
    key = (a, b)
    if key not in cache:
        d = network.shortest_travel_times(a, max_speed)
        for k, v in d.items():
            cache[(a, k)] = v
        cache[key] = d.get(b, float("inf"))
    return cache[key]


def order_tasks(
    tasks: list[Task],
    rule: DispatchRule,
    current_step: int,
    network: Network,
    max_speed: float,
    cache: dict[tuple[str, str], float],
) -> list[Task]:
    """按调度规则给待分配任务排序，返回新列表（不改动入参）。"""

    def remaining_time(t: Task) -> float:
        """完成该任务还需要多少秒（取货点行驶 + 装卸 + 送达）。"""
        return _travel(network, t.pickup, t.dropoff, max_speed, cache)

    if rule is DispatchRule.FIFO:
        return sorted(tasks, key=lambda t: (t.release_step, t.task_id))

    if rule is DispatchRule.EDD:
        return sorted(
            tasks,
            key=lambda t: (t.due_step if t.due_step is not None else float("inf"),
                           t.release_step, t.task_id),
        )

    if rule is DispatchRule.SPT:
        return sorted(
            tasks,
            key=lambda t: (remaining_time(t), t.release_step, t.task_id),
        )

    if rule is DispatchRule.PRIORITY:
        return sorted(
            tasks,
            key=lambda t: (-int(t.priority), t.release_step, t.task_id),
        )

    if rule is DispatchRule.CR:
        def critical_ratio(t: Task) -> float:
            if t.due_step is None:
                return float("inf")
            slack = t.due_step - current_step
            work = max(1e-6, remaining_time(t))
            return slack / work
        return sorted(
            tasks,
            key=lambda t: (critical_ratio(t), t.release_step, t.task_id),
        )

    raise ValueError("未知调度规则: %r" % (rule,))
