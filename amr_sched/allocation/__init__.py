"""任务分配策略（任务书 R2.4a）。

分成两层：
* ``dispatch`` —— 调度规则，决定先做哪个任务；
* ``rule_based`` —— 把调度规则与分配算法组合起来。
"""

from amr_sched.allocation.base import AllocationStrategy
from amr_sched.allocation.dispatch import DispatchRule, order_tasks
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.allocation.hungarian import hungarian
from amr_sched.allocation.rule_based import RuleBasedAllocator

__all__ = [
    "AllocationStrategy",
    "DispatchRule",
    "order_tasks",
    "GreedyAllocator",
    "hungarian",
    "RuleBasedAllocator",
]
