"""任务分配策略接口。

分配决策与路径规划分离：换分配策略不影响规划器，反之亦然。
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from amr_sched.entities import Agv, Task
from amr_sched.network import Network


class AllocationStrategy(ABC):
    """把待分配任务指派给空闲 AGV。"""

    name: str = "base"

    @abstractmethod
    def allocate(
        self,
        network: Network,
        idle_agvs: list[Agv],
        pending_tasks: list[Task],
        current_step: int,
        max_speed: float,
        travel_time_cache: dict[tuple[str, str], float],
    ) -> list[tuple[Agv, Task]]:
        """返回 ``(agv, task)`` 配对列表。

        Parameters
        ----------
        travel_time_cache:
            可复用的 ``(from, to) -> 秒`` 缓存，避免重复 Dijkstra。
        """
        raise NotImplementedError
