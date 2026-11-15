"""执行层监控：把 L3 上报的异常翻译成调度决策。

为什么必须有这一层
------------------
Python 决定车**经过哪些节点**，Nav2 决定**怎么开过去**。局部绕行一旦占用了别的
路段，就产生了一个只有 Python 有资格处理的问题：那条路段可能已经被别的车预约了。
Nav2 自己去改全局路线，等于在调度核心之外又长出一个决策者——这正是本项目要避免的。

所以协议是：Nav2 上报，Python 决定，Nav2 执行。

本模块只做三件事：
1. **校验**上报的事件（字段缺一不可，事件类型只认已定义的）；
2. **判定**阻断的边是否真的存在于网络中（不认识的边不猜）；
3. **决定**改走哪条路，或改派任务。

它不发送任何消息，也不读时钟——给定输入就给出输出，这样可以在 Windows 上
用单元测试钉住判断，不需要 ROS。

尚未做的
--------
* 还没有和桥接服务连起来：`route_deviation` 消息尚未在 `live_bridge` 上收发。
* 只知道某个 AGV 的目标节点，不知道它当前任务的完整上下文（是否需要改派给别的车）。
  改派结论目前是"交给上层重新分配"，而不是自己挑车——挑车是 Task Allocation 的职责。
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Any

#: 唯一已定义的事件类型。新增类型必须先写进协议文档，否则这里拒绝。
ROUTE_DEVIATION = "ROUTE_DEVIATION"

#: 允许的事件类型集合。刻意做成白名单：未知事件被拒绝，而不是被当成偏差处理。
KNOWN_EVENTS = frozenset({ROUTE_DEVIATION})


class DeviationEventError(ValueError):
    """上报的事件本身不合法。调用方应当把这条消息丢弃并记录原因。"""


@dataclass(frozen=True)
class DeviationEvent:
    """L3 上报的一次局部绕行。

    字段名与协议文档一致，一个都不能少：少了就无法判断它到底阻断了哪条边。
    """

    robot_id: str
    event: str
    blocked_edge: tuple[str, str]
    current_pose: tuple[float, float, float]
    timestamp: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "DeviationEvent":
        missing = [k for k in ("robot_id", "event", "blocked_edge", "current_pose", "timestamp")
                   if k not in raw]
        if missing:
            raise DeviationEventError("缺少字段: %s" % ", ".join(sorted(missing)))
        event = str(raw["event"])
        if event not in KNOWN_EVENTS:
            raise DeviationEventError("未知事件类型: %s" % event)
        edge = _parse_edge(raw["blocked_edge"])
        pose = _parse_pose(raw["current_pose"])
        robot = str(raw["robot_id"]).strip()
        if not robot:
            raise DeviationEventError("robot_id 为空")
        try:
            stamp = float(raw["timestamp"])
        except (TypeError, ValueError):
            raise DeviationEventError("timestamp 不是数字: %r" % (raw["timestamp"],))
        return cls(robot, event, edge, pose, stamp)


def _parse_edge(value: Any) -> tuple[str, str]:
    """接受 "A|B"、"A-B" 或 ["A","B"],统一成排序后的二元组。

    排序是为了让 (A,B) 与 (B,A) 表示同一条受阻通道——物理上它就是同一条路。
    """
    if isinstance(value, (list, tuple)):
        parts = [str(v) for v in value]
    elif isinstance(value, str):
        text = value.strip()
        for sep in ("|", "-", "->", "—"):
            if sep in text:
                parts = [p.strip() for p in text.split(sep)]
                break
        else:
            parts = []
    else:
        parts = []
    parts = [p for p in parts if p]
    if len(parts) != 2:
        raise DeviationEventError("blocked_edge 无法解析: %r" % (value,))
    if parts[0] == parts[1]:
        raise DeviationEventError("blocked_edge 两端相同: %r" % (value,))
    return tuple(sorted(parts))  # type: ignore[return-value]


def _parse_pose(value: Any) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise DeviationEventError("current_pose 需要至少 (x, y): %r" % (value,))
    try:
        nums = [float(v) for v in value]
    except (TypeError, ValueError):
        raise DeviationEventError("current_pose 含非数字: %r" % (value,))
    if len(nums) == 2:
        nums.append(0.0)
    return (nums[0], nums[1], nums[2])


@dataclass(frozen=True)
class Decision:
    """监控层给出的结论。

    `action` 取值：
      * `reroute`  —— 绕开受阻通道后仍有路，按 `nodes` 走，并重新预约这些节点；
      * `reassign` —— 绕不开，这一趟交给上层重新分配车辆/任务；
      * `ignore`   —— 上报的边不在网络里，不做任何改变（也绝不猜测）。
    """

    action: str
    robot_id: str
    blocked_edge: tuple[str, str]
    nodes: tuple[str, ...] = ()
    meters: float = 0.0
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "robot_id": self.robot_id,
            "blocked_edge": list(self.blocked_edge),
            "nodes": list(self.nodes),
            "meters": round(self.meters, 3),
            "reason": self.reason,
        }


class ExecutionMonitor:
    """把偏差事件变成决策。持有网络，不持有仿真状态。"""

    def __init__(self, network: Any) -> None:
        self.network = network
        #: 最近若干条决定，供显示层读到"Python 收到了什么、怎么处理的"。
        self.history: list[Decision] = []

    # ---------- 查询 ----------

    def edge_exists(self, edge: tuple[str, str]) -> bool:
        """受阻通道是否真的在网络里。

        有向网络可能只存了一个方向；物理通道两个方向都该被阻断，所以两个方向任一存在即算存在。
        """
        src, dst = edge
        edges = getattr(self.network, "edges", {})
        return (src, dst) in edges or (dst, src) in edges

    # ---------- 决策 ----------

    def handle(self, event: DeviationEvent, *, goal: str) -> Decision:
        """处理一条偏差事件。

        `goal` 是这台车当前要去的节点。监控层不去猜目标——猜错就会把车派到别处。
        """
        if not self.edge_exists(event.blocked_edge):
            return self._remember(Decision(
                "ignore", event.robot_id, event.blocked_edge,
                reason="受阻通道不在网络中，不做任何改变: %s" % (event.blocked_edge,)))

        start = self._nearest_node(event.current_pose)
        if start is None:
            return self._remember(Decision(
                "ignore", event.robot_id, event.blocked_edge,
                reason="当前位姿无法吸附到任何节点"))

        if start == goal:
            return self._remember(Decision(
                "reassign", event.robot_id, event.blocked_edge,
                reason="已到达目标 %s，无需改道" % goal))

        nodes, meters = self._detour(start, goal, event.blocked_edge)
        if not nodes:
            return self._remember(Decision(
                "reassign", event.robot_id, event.blocked_edge,
                reason="绕开 %s 后 %s 到 %s 不可达" % (event.blocked_edge, start, goal)))

        return self._remember(Decision(
            "reroute", event.robot_id, event.blocked_edge,
            nodes=tuple(nodes), meters=meters,
            reason="绕开 %s" % (event.blocked_edge,)))

    def _remember(self, decision: Decision) -> Decision:
        self.history.append(decision)
        # 只留最近一批，避免长跑时无限增长。
        if len(self.history) > 256:
            del self.history[:-256]
        return decision

    def _nearest_node(self, pose: tuple[float, float, float]) -> str | None:
        best, best_d = None, float("inf")
        for node_id, node in getattr(self.network, "nodes", {}).items():
            x, y = getattr(node, "x", None), getattr(node, "y", None)
            if x is None or y is None:
                continue
            d = (x - pose[0]) ** 2 + (y - pose[1]) ** 2
            if d < best_d:
                best, best_d = node_id, d
        return best

    def detour_from(self, start: str, goal: str,
                    blocked: tuple[str, str]) -> tuple[list[str], float]:
        """Public form of the forbidden-edge search, for callers that already know
        the node the vehicle is standing on (the simulator does)."""
        return self._detour(start, goal, blocked)

    def _detour(self, start: str, goal: str,
                blocked: tuple[str, str]) -> tuple[list[str], float]:
        """禁止一条通道后的最短路径（Dijkstra）。

        受阻通道**两个方向都禁止**：Nav2 报上来的是"这条路走不通"，不是"这个方向走不通"。
        不让它经过任何第三方的规划器，是因为这里多了一个全局规划器不会有的约束
        （禁止边），混进去会让"规划结果来自哪个规划器"变得说不清。
        """
        nodes = getattr(self.network, "nodes", {})
        if start not in nodes or goal not in nodes:
            return [], 0.0
        blocked_pair = set(blocked)

        dist = {start: 0.0}
        prev: dict[str, str] = {}
        heap: list[tuple[float, str]] = [(0.0, start)]
        seen = set()
        while heap:
            d, node = heapq.heappop(heap)
            if node in seen:
                continue
            seen.add(node)
            if node == goal:
                break
            for edge in self.network.out_edges(node):
                if {edge.src, edge.dst} == blocked_pair:
                    continue          # 受阻通道，两个方向都不走
                nd = d + float(getattr(edge, "length", 0.0))
                if nd < dist.get(edge.dst, float("inf")):
                    dist[edge.dst] = nd
                    prev[edge.dst] = node
                    heapq.heappush(heap, (nd, edge.dst))

        if goal not in dist:
            return [], 0.0
        path = [goal]
        while path[-1] != start:
            path.append(prev[path[-1]])
        path.reverse()
        return path, dist[goal]
