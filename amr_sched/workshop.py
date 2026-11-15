"""铁路货车检修车间场景：把唯一场景源文件编译成 Python 调度网络（MAP-01C）。

为什么单独一层
--------------
在 MAP-01 之前，"车间"就是 build_demo_network 里的抽象环网：节点叫 W1..W6，
边只有长度与容量。用它做算法对照是合适的；用它声称"铁路货车检修车间"不合适——
它既没有"拆解区 -> 专业检修 -> 组装区"的物流关系，也没有决定调度能否真正执行的
资源语义（等待区、装卸交接位、工位库存、共享单车道）。

这一层做三件事：

1. **编译**：场景 JSON -> Network。节点坐标、边长度、限速与容量全部来自文件，
   代码里不藏任何几何常量。
2. **校验**：把"画得出来但跑不起来"的问题变成一份可读报告，而不是留给仿真去撞。
3. **任务合法性**：按物料-工艺路线给出合法的取货/卸货节点对，而不是从全体节点里
   随机挑两个点。

只读，不改场景
--------------
load 只读文件，validate 只返回问题列表。校验发现几何有问题时不会去调整几何让它通过
——那正是"为了让结论好看而保留已知缺陷"的镜像做法。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from amr_sched.network import Edge, Network, Node, NodeKind

SCENE_DIR = Path(__file__).resolve().parent.parent / "scenes"
DEFAULT_SCENE = "rail_freight_workshop_v1"

#: 场景 node_kind 字符串 -> 调度网络的节点种类。显式写成一张表，编译时才没有隐藏规则：
#: 节点属于哪一层、算不算工位（要不要计工序时间）、能不能被空车退避选中，都由场景文件说了算。
NODE_KINDS: dict[str, NodeKind] = {
    "aisle": NodeKind.AISLE,
    "disassembly": NodeKind.DISASSEMBLY,
    "workstation": NodeKind.WORKSTATION,
    "assembly": NodeKind.ASSEMBLY,
    "buffer": NodeKind.BUFFER,
    "dock": NodeKind.DOCK,
}

EVIDENCE_LEVELS = (
    "real_cad",
    "process_referenced_synthetic",
    "abstract_benchmark",
    "scaled_testbed",
)

_GEOM_TOL = 1e-6


class SceneError(ValueError):
    """场景无法编译成网络（与"校验发现问题"不同：这是文件本身不成立）。"""


@dataclass(frozen=True)
class SceneNode:
    node_id: str
    zone: str
    layer: str
    type: str
    node_kind: str
    x: float
    y: float
    agv_capacity: int = 1
    queue_slots: int = 0
    service_capacity: int = 0
    stock_capacity: int = 0
    agv_access: bool = True
    note_zh: str = ""

    @property
    def kind(self) -> NodeKind:
        return NODE_KINDS[self.node_kind]


@dataclass(frozen=True)
class SceneLane:
    lane_id: str
    src: str
    dst: str
    direction: str
    lane_count: int
    width_m: float
    speed_limit_mps: float
    physical_channel: str
    centerline: tuple[tuple[float, float], ...]

    @property
    def two_way(self) -> bool:
        return self.direction == "two_way"

    @property
    def polyline_length(self) -> float:
        total = 0.0
        for (x0, y0), (x1, y1) in zip(self.centerline, self.centerline[1:]):
            total += math.hypot(x1 - x0, y1 - y0)
        return total

    @property
    def edge_capacity(self) -> int:
        """每个方向允许同时在通道上的车辆数。

        单车道双向（lane_count == 1 且双向）时是 1：两个方向共用同一条物理路面，
        仿真层的 _reserve_hop 会把两个方向一起登记、_can_enter 取两个方向的最小容量，
        因此这里必须也是 1，两边语义才一致。
        """
        if self.two_way and self.lane_count == 1:
            return 1
        return self.lane_count


@dataclass(frozen=True)
class SceneZone:
    zone_id: str
    name_zh: str
    kind: str
    polygon: tuple[tuple[float, float], ...]
    note_zh: str = ""
    #: 该区域的主体节点种类（workstation / disassembly / assembly / buffer 或 traffic）。
    #: 由场景文件给出，而不是在代码里按区域名字猜。
    node_kind: str = ""

    @property
    def scope(self) -> str:
        """ZONE_BRAKE -> BRAKE。物料路线用后者指称区域。"""
        return self.zone_id[5:] if self.zone_id.startswith("ZONE_") else self.zone_id


@dataclass(frozen=True)
class MaterialRoute:
    material_id: str
    name_zh: str
    from_zones: tuple[str, ...]
    to_zones: tuple[str, ...]
    unit: str = ""
    note_zh: str = ""


@dataclass(frozen=True)
class SceneIssue:
    """一条校验发现。code 稳定可断言，detail 给人看。"""

    code: str
    where: str
    detail: str

    def __str__(self) -> str:
        return f"[{self.code}] {self.where}: {self.detail}"


class WorkshopScene:
    """一份车间场景。用 load 读文件，不要直接构造。"""

    def __init__(self, raw: dict, path: Path | None = None) -> None:
        self.raw = raw
        self.path = path
        self.scene_id: str = raw.get("scene_id", "")
        self.schema_version: int = int(raw.get("schema_version", 0))
        self.frame_id: str = raw.get("frame_id", "")
        self.units: dict = raw.get("units", {})
        self.bounds: tuple[float, float] = tuple(raw.get("bounds_m", (0.0, 0.0)))  # type: ignore[assignment]
        self.envelope: dict = raw.get("agv_envelope", {})
        self.evidence: dict = raw.get("evidence", {})
        self.nodes: dict[str, SceneNode] = {
            n["id"]: SceneNode(
                node_id=n["id"],
                zone=n.get("zone", ""),
                layer=n.get("layer", "road"),
                type=n.get("type", ""),
                node_kind=n.get("node_kind", ""),
                x=float(n["x"]),
                y=float(n["y"]),
                agv_capacity=int(n.get("agv_capacity", 1)),
                queue_slots=int(n.get("queue_slots", 0)),
                service_capacity=int(n.get("service_capacity", 0)),
                stock_capacity=int(n.get("stock_capacity", 0)),
                agv_access=bool(n.get("agv_access", True)),
                note_zh=n.get("note_zh", ""),
            )
            for n in raw.get("nodes", [])
        }
        self.lanes: tuple[SceneLane, ...] = tuple(
            SceneLane(
                lane_id=l["id"],
                src=l["from"],
                dst=l["to"],
                direction=l.get("direction", "two_way"),
                lane_count=int(l.get("lane_count", 1)),
                width_m=float(l.get("width_m", 0.0)),
                speed_limit_mps=float(l.get("speed_limit_mps", 0.0)),
                physical_channel=l.get("physical_channel", ""),
                centerline=tuple((float(a), float(b)) for a, b in l.get("centerline", [])),
            )
            for l in raw.get("lanes", [])
        )
        self.zones: dict[str, SceneZone] = {
            z["id"]: SceneZone(
                zone_id=z["id"],
                name_zh=z.get("name_zh", ""),
                kind=z.get("kind", ""),
                polygon=tuple((float(a), float(b)) for a, b in z.get("polygon", [])),
                note_zh=z.get("note_zh", ""),
                node_kind=z.get("node_kind", ""),
            )
            for z in raw.get("zones", [])
        }
        self.materials: dict[str, MaterialRoute] = {
            m["id"]: MaterialRoute(
                material_id=m["id"],
                name_zh=m.get("name_zh", ""),
                from_zones=tuple(m.get("from_zones", [])),
                to_zones=tuple(m.get("to_zones", [])),
                unit=m.get("unit", ""),
                note_zh=m.get("note_zh", ""),
            )
            for m in raw.get("materials", [])
        }

    # ---------- loading ----------

    @classmethod
    def load(cls, scene: str | Path = DEFAULT_SCENE) -> "WorkshopScene":
        """按场景名或路径读取。场景名会在 scenes/ 下找同名 .json。"""
        path = Path(scene)
        if not path.suffix:
            path = SCENE_DIR / f"{scene}.json"
        elif not path.is_absolute():
            path = (Path.cwd() / path) if path.exists() else SCENE_DIR / path.name
        if not path.exists():
            raise SceneError(f"场景文件不存在: {path}")
        with path.open(encoding="utf-8") as fh:
            raw = json.load(fh)
        if not isinstance(raw, dict):
            raise SceneError(f"场景文件的顶层必须是对象: {path}")
        return cls(raw, path)

    # ---------- queries ----------

    @property
    def agv_nodes(self) -> dict[str, SceneNode]:
        return {k: v for k, v in self.nodes.items() if v.agv_access}

    def zone_of(self, node_id: str) -> SceneZone | None:
        node = self.nodes.get(node_id)
        return self.zones.get(node.zone) if node else None

    def dock_node(self, scope: str) -> str | None:
        """区域的装卸交接位节点 id（ZONE_BRAKE / BRAKE 都接受）。"""
        scope = scope[5:] if scope.startswith("ZONE_") else scope
        candidate = f"D_{scope}"
        return candidate if candidate in self.nodes else None

    def station_node(self, scope: str) -> str | None:
        """区域的工位/库存节点 id。物料从这里取出，在装卸位交接。"""
        scope = scope[5:] if scope.startswith("ZONE_") else scope
        candidate = f"W_{scope}"
        return candidate if candidate in self.nodes else None

    def task_endpoints(self, material_id: str) -> list[tuple[str, str]]:
        """一种物料的合法取货/卸货节点对。

        规则统一：**从取货区域的工位/库存 W 取，在卸货区域的装卸交接位 D 交接**。
        物料总要从一个存放或加工的位置拿出来，也总要在某个交接位上交给对方；
        车辆不必为了交货开进工位，但取货时要开进去——这正是把 W 与 D 分开的意义。
        """
        try:
            material = self.materials[material_id]
        except KeyError:
            raise SceneError(f"未知物料: {material_id}") from None
        pairs: list[tuple[str, str]] = []
        for src_zone in material.from_zones:
            pickup = self.station_node(src_zone)
            if pickup is None:
                raise SceneError(f"物料 {material_id}: 取货区域 {src_zone} 没有工位节点")
            for dst_zone in material.to_zones:
                if dst_zone == src_zone:
                    continue
                dropoff = self.dock_node(dst_zone)
                if dropoff is None:
                    raise SceneError(f"物料 {material_id}: 卸货区域 {dst_zone} 没有装卸位节点")
                pairs.append((pickup, dropoff))
        return pairs

    # ---------- compilation ----------

    def network(self, aisle_capacity: int | None = None,
                lane_capacity: int | None = None) -> Network:
        """编译成调度网络。

        aisle_capacity 覆盖**路口节点**的车辆容量，与 build_layout 的同名参数一致，
        让"通道负载"这类既有实验能在本场景上重跑。

        lane_capacity 覆盖**通道边**的容量，只用于敏感性分析。
        场景声明的几何是 lane_count=1（单车道双向、两个方向共用同一块路面）；
        把它调到 2 等于假设"这条通道能同时容纳两辆车"，那不是 v1 的几何，
        结果只能用来回答"容量这一项值多少"，不能当成场景的真实配置引用。
        """
        for lane in self.lanes:
            if lane.two_way and lane.lane_count != 1:
                raise SceneError(
                    f"通道 {lane.lane_id} 是双向 {lane.lane_count} 车道；v1 不支持："
                    "预约层目前对存在反向边的通道一律按共享物理资源处理，无法区分"
                    "两个方向是否共用路面。要么在场景里写成两条单向通道，"
                    "要么先让预约层读到路面是否共享。"
                )
        net = Network()
        for node in self.nodes.values():
            if not node.agv_access:
                continue
            capacity = node.agv_capacity
            if aisle_capacity is not None and node.kind is NodeKind.AISLE:
                capacity = aisle_capacity
            net.add_node(Node(node.node_id, node.x, node.y, node.kind, capacity))
        for lane in self.lanes:
            length = lane.polyline_length
            if lane.two_way:
                net.add_bidirectional(
                    lane.src, lane.dst, length,
                    speed_limit=lane.speed_limit_mps,
                    capacity=lane.edge_capacity if lane_capacity is None else lane_capacity,
                    centerline=lane.centerline,
                )
            else:
                net.add_edge(Edge(lane.src, lane.dst, length,
                                  speed_limit=lane.speed_limit_mps,
                                  capacity=lane.edge_capacity if lane_capacity is None else lane_capacity,
                                  centerline=lane.centerline))
        return net

    # ---------- validation ----------

    def validate(self) -> list[SceneIssue]:
        """返回问题列表。空列表表示这份场景可以编译，并且任务起讫合法。"""
        issues: list[SceneIssue] = []

        def add(code: str, where: str, detail: str) -> None:
            issues.append(SceneIssue(code, where, detail))

        # --- 场景元信息：版本与假设参数必须可追溯 ---
        if self.schema_version < 1:
            add("scene_schema_version", self.scene_id, "缺少 schema_version")
        if not self.frame_id:
            add("scene_frame_id", self.scene_id, "缺少 frame_id，坐标没有声明所在坐标系")
        level = self.evidence.get("level")
        if level not in EVIDENCE_LEVELS:
            add("evidence_level_unknown", self.scene_id,
                f"证据等级 {level!r} 不在 {EVIDENCE_LEVELS}；"
                "地图来源必须标注，否则演示数据会被当成实测")
        if not self.evidence.get("assumptions_zh"):
            add("evidence_assumptions_missing", self.scene_id, "没有列出假设参数")
        if not self.evidence.get("not_measured_zh"):
            add("evidence_limits_missing", self.scene_id,
                "没有写明本场景不是什么（未实测清单），容易被读成实测车间")
        if not self.envelope:
            add("agv_envelope_missing", self.scene_id, "缺少 AGV 外廓，无法校验通道净宽")
        if len(self.bounds) != 2 or min(self.bounds) <= 0:
            add("scene_bounds", self.scene_id, f"bounds_m 非法: {self.bounds}")

        # --- 节点 ---
        for node in self.nodes.values():
            if node.node_kind not in NODE_KINDS:
                add("unknown_node_kind", node.node_id,
                    f"node_kind={node.node_kind!r} 不在 {sorted(NODE_KINDS)}")
            if node.zone and node.zone not in self.zones:
                add("node_zone_missing", node.node_id, f"引用了不存在的区域 {node.zone}")
            if self.bounds != (0.0, 0.0) and not (
                0.0 <= node.x <= self.bounds[0] and 0.0 <= node.y <= self.bounds[1]
            ):
                add("node_out_of_bounds", node.node_id,
                    f"({node.x}, {node.y}) 超出车间范围 {self.bounds}")
            if node.agv_capacity < 1:
                add("node_capacity", node.node_id, f"agv_capacity={node.agv_capacity} < 1")
            if node.type == "wait_area" and node.queue_slots < 1:
                add("wait_area_no_slots", node.node_id,
                    "等待区没有等待槽位，等于把等待又推回主通道")
            if node.type == "service_dock":
                if node.service_capacity < 1:
                    add("dock_no_service", node.node_id, "装卸位没有服务容量")
                if node.agv_capacity != 1:
                    add("dock_capacity", node.node_id,
                        f"装卸位 agv_capacity={node.agv_capacity}，应为 1："
                        "同一交接位同时只应服务一辆车")
                if node.stock_capacity:
                    add("dock_holds_stock", node.node_id,
                        "装卸位带了库存容量；库存属于工位，交接位只负责交接")
            if node.type == "station" and node.stock_capacity < 1:
                add("station_no_stock", node.node_id,
                    "工位没有库存容量；物料占用与车辆占用是两个资源，不能只有一个")

        # --- 通道几何与物理共享 ---
        channels: dict[str, list[SceneLane]] = {}
        for lane in self.lanes:
            for endpoint in (lane.src, lane.dst):
                if endpoint not in self.nodes:
                    add("lane_endpoint_missing", lane.lane_id, f"端点 {endpoint} 不存在")
            if lane.width_m <= 0:
                add("lane_width", lane.lane_id, f"width_m={lane.width_m} 必须 > 0")
            if lane.speed_limit_mps <= 0:
                add("lane_speed", lane.lane_id,
                    f"speed_limit_mps={lane.speed_limit_mps} 必须 > 0")
            if lane.src in self.nodes and lane.dst in self.nodes:
                a, b = self.nodes[lane.src], self.nodes[lane.dst]
                if len(lane.centerline) < 2:
                    add("lane_centerline", lane.lane_id, "中心线至少要有两个点")
                else:
                    if math.dist(lane.centerline[0], (a.x, a.y)) > _GEOM_TOL:
                        add("lane_centerline_detached", lane.lane_id,
                            f"中心线起点 {lane.centerline[0]} 与节点 {lane.src} "
                            f"坐标 ({a.x}, {a.y}) 不一致")
                    if math.dist(lane.centerline[-1], (b.x, b.y)) > _GEOM_TOL:
                        add("lane_centerline_detached", lane.lane_id,
                            f"中心线终点 {lane.centerline[-1]} 与节点 {lane.dst} "
                            f"坐标 ({b.x}, {b.y}) 不一致")
                car_w = self.envelope.get("width_m")
                margin = self.envelope.get("min_clearance_m")
                if car_w and margin and lane.width_m < car_w + 2 * margin:
                    add("lane_too_narrow", lane.lane_id,
                        f"净宽 {lane.width_m} m 小于车宽 {car_w} + 两侧余量 "
                        f"{2 * margin} m，AGV 进不去")
            channels.setdefault(lane.physical_channel, []).append(lane)

        # 物理通道编号必须与预约层**实际**的共享方式一致。
        #
        # 预约层是按结构判断的：只要一条通道存在反向边，_reserve_hop 就把两个方向
        # 一起登记，_can_enter 取两个方向容量的最小值。它读不到"这两条边是不是同一块
        # 路面"。所以场景能表达的自由度只有"是不是同一对节点"：
        #   * 同一对节点由多条路段连接，却给了不同的通道编号 -> 场景在说这是两条独立
        #     的路，而调度器会按一条共享路处理；两个说法必须统一，否则"地图"和
        #     "调度器眼里的地图"不是一张图。
        #   * 声明 two_way 就等于两个方向都存在，不需要（也不允许）文件里再写一条反向
        #     路段；真的写了会在编译时报重复边，那是响的失败，不是静默的错误。
        by_pair: dict[frozenset[str], list[SceneLane]] = {}
        for lane in self.lanes:
            if lane.direction not in ("two_way", "one_way"):
                add("lane_direction_unknown", lane.lane_id,
                    f"方向 {lane.direction!r} 不是 two_way / one_way")
            by_pair.setdefault(frozenset((lane.src, lane.dst)), []).append(lane)
        for channel, members in sorted(channels.items()):
            if len({frozenset((l.src, l.dst)) for l in members}) > 1:
                add("channel_ambiguous", channel,
                    "同一个物理通道编号被用在不同路段上，预约时会互相顶替："
                    + ", ".join(sorted(l.lane_id for l in members)))
        for pair, members in sorted(by_pair.items(), key=lambda kv: sorted(kv[0])):
            if len(members) < 2:
                continue
            channel_ids = {l.physical_channel for l in members}
            if len(channel_ids) > 1:
                add("channel_split", " / ".join(sorted(pair)),
                    "同一对节点之间有多条路段，却给了不同的物理通道编号 "
                    + ", ".join(sorted(channel_ids))
                    + "：场景在说它们是各自独立的路面，而预约层对存在反向边的通道"
                    "一律按共享路面处理。要么统一通道编号，要么先让预约层读到"
                    "两个方向是否共用路面。")

        # --- 连通性与资源释放（要编译成网络才能查） ---
        try:
            net = self.network()
        except (SceneError, ValueError) as exc:
            # ValueError 来自 Network.add_edge：同一条有向边被写了两遍。
            # 这是场景的问题，不该让校验器自己崩掉——崩掉就等于没有报告。
            add("network_compile", self.scene_id, str(exc))
            return sorted(issues, key=lambda i: (i.code, i.where))

        agv_ids = sorted(self.agv_nodes)
        if agv_ids:
            forward = _reachable(net, agv_ids[0])
            backward = _reachable(net, agv_ids[0], reverse=True)
            for nid in agv_ids:
                if nid not in forward:
                    add("unreachable_node", nid,
                        f"从 {agv_ids[0]} 出发到不了；存在只能出不能进的死角")
                elif nid not in backward:
                    add("unescapable_node", nid,
                        f"从 {nid} 出发回不到 {agv_ids[0]}；进去就出不来")
        for node in self.nodes.values():
            if node.type == "service_dock" and node.agv_access and not net.out_edges(node.node_id):
                add("dock_without_exit", node.node_id,
                    "装卸位没有出边；卸完货的车没有路可以离开")

        # --- 任务起讫的业务合法性 ---
        known_scopes = {z.scope for z in self.zones.values()}
        for material in self.materials.values():
            for zone in material.from_zones + material.to_zones:
                if zone not in known_scopes:
                    add("material_unknown_zone", material.material_id,
                        f"物料路线引用了不存在的区域 {zone}")
            for zone in material.from_zones:
                if zone in material.to_zones:
                    add("material_self_loop", material.material_id,
                        f"取货与卸货都在 {zone}")
            try:
                pairs = self.task_endpoints(material.material_id)
            except SceneError as exc:
                add("material_endpoint_missing", material.material_id, str(exc))
                continue
            if not pairs:
                add("material_no_route", material.material_id, "没有产生任何合法的取送节点对")
            for pickup, dropoff in pairs:
                if not _has_path(net, pickup, dropoff):
                    add("material_unreachable", material.material_id,
                        f"{pickup} -> {dropoff} 在路网上不可达")

        covered = {z for m in self.materials.values() for z in m.from_zones + m.to_zones}
        for zone in self.zones.values():
            if zone.node_kind == "workstation" and zone.scope not in covered:
                add("workstation_without_material", zone.zone_id,
                    "这个检修工位没有任何物料进出，等于图上有一个不参与生产的点")

        # --- 坐标正反变换精度 ---
        err = self.coordinate_roundtrip_error()
        if err > 1e-9:
            add("coordinate_roundtrip", self.scene_id,
                f"场景坐标 -> 网络坐标 -> 场景坐标 的最大误差 {err:.3e} m")

        return sorted(issues, key=lambda i: (i.code, i.where))

    def coordinate_roundtrip_error(self) -> float:
        """场景坐标与编译后网络坐标的最大偏差。

        这不是形式检查：Web 控制台、RAWSim-O 与 Gazebo 都吃同一份坐标，任何一个环节
        自行做一次投影或单位换算，三个界面就会各说各话，而且每张图单独看都对。
        """
        net = self.network()
        worst = 0.0
        for node in self.nodes.values():
            if not node.agv_access:
                continue
            compiled = net.node(node.node_id)
            worst = max(worst, math.dist((node.x, node.y), (compiled.x, compiled.y)))
        return worst

    def summary(self) -> str:
        kinds: dict[str, int] = {}
        for node in self.nodes.values():
            kinds[node.node_kind] = kinds.get(node.node_kind, 0) + 1
        directed = (
            2 * sum(1 for l in self.lanes if l.two_way)
            + sum(1 for l in self.lanes if not l.two_way)
        )
        return (
            f"{self.scene_id} v{self.schema_version}  "
            f"{len(self.nodes)} 节点 / {len(self.lanes)} 通道 / {directed} 有向边 / "
            f"{len(self.zones)} 区域 / {len(self.materials)} 物料  "
            f"范围 {self.bounds[0]:.0f}x{self.bounds[1]:.0f} m  "
            f"证据等级 {self.evidence.get('level')}  节点种类 "
            + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items()))
        )


def _reachable(net: Network, start: str, reverse: bool = False) -> set[str]:
    seen = {start}
    stack = [start]
    while stack:
        node = stack.pop()
        edges = net.in_edges(node) if reverse else net.out_edges(node)
        for edge in edges:
            nxt = edge.src if reverse else edge.dst
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def _has_path(net: Network, src: str, dst: str) -> bool:
    if src not in net.nodes or dst not in net.nodes:
        return False
    return dst in _reachable(net, src)


def legal_task_pairs(scene: "WorkshopScene | None" = None) -> list[tuple[str, str]]:
    """Every pickup/drop-off pair the process actually allows, over all materials."""
    sc = scene or WorkshopScene.load()
    pairs: set[tuple[str, str]] = set()
    for material in sc.materials.values():
        pairs.update(sc.task_endpoints(material.material_id))
    return sorted(pairs)


def parking_slots(scene: "WorkshopScene | None" = None) -> list[str]:
    """Where the fleet waits between jobs: the PARK area, expanded by capacity.

    Queue slots and docking positions are deliberately excluded -- a vehicle waiting for
    a delivery is not the same thing as a vehicle parked for the shift.
    """
    sc = scene or WorkshopScene.load()
    park = [n for n in sc.nodes.values() if n.zone == "ZONE_PARK"]
    # Charging/stock position first, then the handover position, then the waiting slots,
    # then the junction: the fleet fills the best place first and only spills over.
    order = {"W_": 0, "D_": 1, "Q_": 2, "R_": 3}
    park.sort(key=lambda n: (order.get(n.node_id[:2], 9), n.node_id))
    slots: list[str] = []
    for node in park:
        cap = node.queue_slots if node.queue_slots else node.agv_capacity
        slots.extend([node.node_id] * max(1, cap))
    return slots


def build_scenario(
    scene: "WorkshopScene | None" = None,
    n_tasks: int = 20,
    n_agvs: int = 2,
    seed: int = 0,
    **kwargs,
):
    """Compile the scene and generate tasks that only use legal material routes.

    This is the workshop counterpart of scenarios.generate_scenario: same due-time
    definitions, same AGV placement rule, but the pickup/drop-off pairs come from the
    scene instead of from node kinds.
    """
    from amr_sched.scenarios import ScenarioConfig, generate_scenario

    sc = scene or WorkshopScene.load()
    network = sc.network()
    config = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed, **kwargs)
    return generate_scenario(
        config,
        network=network,
        pairs=legal_task_pairs(sc),
        start_pool=parking_slots(sc),
    )

def load_and_validate(scene: str | Path = DEFAULT_SCENE) -> tuple[WorkshopScene, list[SceneIssue]]:
    """读一份场景并校验。返回 (场景, 问题列表)，不抛异常。"""
    loaded = WorkshopScene.load(scene)
    return loaded, loaded.validate()


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    name = args[0] if args else DEFAULT_SCENE
    scene, issues = load_and_validate(name)
    print(scene.summary())
    print(f"场景文件: {scene.path}")
    if not issues:
        net = scene.network()
        print(f"校验通过：0 个问题；编译得到 {len(net.nodes)} 节点 / {len(net.edges)} 有向边")
        return 0
    print(f"校验发现 {len(issues)} 个问题：")
    for issue in issues:
        print("  " + str(issue))
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
