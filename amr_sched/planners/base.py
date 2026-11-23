"""路径规划器接口。

所有规划器实现同一签名，换算法只换一个类，不动仿真器——这是"可插拔"的含义。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from amr_sched.network import Network
from amr_sched.reservation import ReservationTable


@dataclass
class PlannedPath:
    """一条规划结果。

    Attributes
    ----------
    nodes:
        依次经过的节点 id，**不含起点**，末元素是目标。
    arrival_steps:
        与 ``nodes`` 等长，第 i 个元素是到达 ``nodes[i]`` 的时间步。
    wait_steps:
        因冲突而额外等待的步数（用于"冲突等待次数/时间"指标）。
    """

    nodes: list[str]
    arrival_steps: list[int]
    wait_steps: int = 0

    def __post_init__(self) -> None:
        if len(self.nodes) != len(self.arrival_steps):
            raise ValueError("nodes 与 arrival_steps 长度必须一致")

    @property
    def finish_step(self) -> int:
        return self.arrival_steps[-1] if self.arrival_steps else 0

    def __len__(self) -> int:
        return len(self.nodes)


class PathPlanner(ABC):
    """路径规划器基类。

    Parameters
    ----------
    max_speed:
        AGV 最大速度（米/秒）。
    dt:
        一个时间步对应多少秒。必须 > 0。
    """

    name: str = "base"

    def __init__(self, max_speed: float = 1.0, dt: float = 1.0) -> None:
        if max_speed <= 0:
            raise ValueError("max_speed 必须 > 0")
        if dt <= 0:
            raise ValueError("dt 必须 > 0")
        self.max_speed = max_speed
        self.dt = dt

    def steps_for(self, seconds: float) -> int:
        """把秒数换算成**至少 1** 步。"""
        import math

        return max(1, math.ceil(seconds / self.dt - 1e-9))

    @abstractmethod
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
        """规划一条从 ``start`` 到 ``goal`` 的路径。

        Parameters
        ----------
        start_step:
            AGV 最早可在何时出发。
        reservations:
            其他 AGV 已占用的时空资源。实现方**不应**修改它。
        horizon:
            允许的最晚到达步。超过即失败，防止无限搜索。

        Returns
        -------
        PlannedPath | None
            找不到路径时返回 None。
        """
        raise NotImplementedError
