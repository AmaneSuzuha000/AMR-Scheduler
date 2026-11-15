"""替代车间布局：检验四条结论是否依赖单一 17 节点环网。

论文 7.2 节把「单一布局」列为局限。本模块补充两种结构差异明显的布局，
用于检验结论的普适性——如果换布局后排序翻转，那结论就只对本布局成立。

两种布局的设计意图
------------------
* :func:`build_grid_network` —— **网格型**（3x3 格 + 外围工位）。
  格子状通道提供大量等长绕行选择，路径层面的自由度比环形更高，
  是检验「预约机制价值」的更有利场景（理论上预约应当更有用）。

* :func:`build_hub_network` —— **枢纽型**（单一中央枢纽 + 放射状支线）。
  几乎无绕行可能，任何两车去不同方向都要经过枢纽，
  冲突集中在一个点上，是对冲突消解的极端压力测试。

两者都保留「拆解区→工位→组装区」的业务语义与缓存区，
因此可直接替换 :func:`build_demo_network` 用于对比实验。
"""

from __future__ import annotations

import math

from amr_sched.network import Edge, Network, Node, NodeKind


def build_grid_network(aisle_capacity: int = 2) -> Network:
    """3x3 网格型布局：9 个路口 + 6 工位 + 3 缓存 + 拆解/组装区。

    网格的特点是**等长替代路径多**：从左上到右下有 6 条等长最短路，
    规划器有充分的选择余地，预约机制理论上应能发挥作用。
    """
    net = Network()

    # 3x3 路口，间距 40
    for r in range(3):
        for c in range(3):
            nid = "G%d%d" % (r, c)
            net.add_node(Node(nid, 40.0 * c, 40.0 * r,
                              NodeKind.AISLE, capacity=aisle_capacity))

    # 横向边
    for r in range(3):
        for c in range(2):
            net.add_bidirectional("G%d%d" % (r, c), "G%d%d" % (r, c + 1),
                                  40.0, speed_limit=1.0,
                                  capacity=aisle_capacity)
    # 纵向边
    for c in range(3):
        for r in range(2):
            net.add_bidirectional("G%d%d" % (r, c), "G%d%d" % (r + 1, c),
                                  40.0, speed_limit=1.0,
                                  capacity=aisle_capacity)

    # 工位挂在网格外围
    stations = [
        ("W1", "G00", -30.0, 0.0),
        ("W2", "G01", 40.0, -30.0),
        ("W3", "G02", 110.0, 0.0),
        ("W4", "G20", -30.0, 80.0),
        ("W5", "G21", 40.0, 110.0),
        ("W6", "G22", 110.0, 80.0),
    ]
    coords = {}
    for r in range(3):
        for c in range(3):
            coords["G%d%d" % (r, c)] = (40.0 * c, 40.0 * r)
    for nid, hook, x, y in stations:
        net.add_node(Node(nid, x, y, NodeKind.WORKSTATION, capacity=1))
        hx, hy = coords[hook]
        net.add_bidirectional(nid, hook, math.hypot(x - hx, y - hy),
                              speed_limit=1.0)

    # 拆解区 / 组装区
    net.add_node(Node("D", -30.0, -30.0, NodeKind.DISASSEMBLY, capacity=2))
    net.add_node(Node("S", 110.0, 110.0, NodeKind.ASSEMBLY, capacity=2))
    net.add_bidirectional("D", "G00", math.hypot(-30 - 0, -30 - 0),
                          speed_limit=1.0)
    net.add_bidirectional("S", "G22", math.hypot(110 - 80, 110 - 80),
                          speed_limit=1.0)

    # 缓存区
    buffers = [
        ("B1", "G10", -30.0, 40.0),
        ("B2", "G12", 110.0, 40.0),
        ("B3", "G01", 40.0, 70.0),
    ]
    for nid, hook, x, y in buffers:
        net.add_node(Node(nid, x, y, NodeKind.BUFFER, capacity=2))
        hx, hy = coords[hook]
        net.add_bidirectional(nid, hook, math.hypot(x - hx, y - hy),
                              speed_limit=1.0)

    return net


def build_hub_network(aisle_capacity: int = 2) -> Network:
    """枢纽型布局：单一中央枢纽 H，所有工位与功能区放射状挂在 H 上。

    特点是**几乎无绕行可能**：任意两车去往不同方向都必须经过 H，
    冲突高度集中。这是对冲突消解机制的极端压力测试，
    也是检验「预约在高冲突场景是否更有价值」的理想算例。
    """
    net = Network()
    net.add_node(Node("H", 0.0, 0.0, NodeKind.AISLE,
                      capacity=aisle_capacity))

    # 放射状支线：角度均分
    spokes = [
        ("W1", NodeKind.WORKSTATION, 1, 60.0, 0.0),
        ("W2", NodeKind.WORKSTATION, 1, 30.0, 52.0),
        ("W3", NodeKind.WORKSTATION, 1, -30.0, 52.0),
        ("W4", NodeKind.WORKSTATION, 1, -60.0, 0.0),
        ("W5", NodeKind.WORKSTATION, 1, -30.0, -52.0),
        ("W6", NodeKind.WORKSTATION, 1, 30.0, -52.0),
    ]
    for nid, kind, cap, x, y in spokes:
        net.add_node(Node(nid, x, y, kind, capacity=cap))
        net.add_bidirectional(nid, "H", math.hypot(x, y), speed_limit=1.0)

    # 拆解区、组装区
    net.add_node(Node("D", 0.0, 95.0, NodeKind.DISASSEMBLY, capacity=2))
    net.add_bidirectional("D", "H", 95.0, speed_limit=1.0)
    net.add_node(Node("S", 0.0, -95.0, NodeKind.ASSEMBLY, capacity=2))
    net.add_bidirectional("S", "H", 95.0, speed_limit=1.0)

    # 缓存区
    for nid, x, y in [("B1", 95.0, 45.0), ("B2", -95.0, 45.0),
                      ("B3", 0.0, 0.0)]:
        if nid == "B3":
            continue  # 枢纽位置不重复占位
        net.add_node(Node(nid, x, y, NodeKind.BUFFER, capacity=2))
        net.add_bidirectional(nid, "H", math.hypot(x, y), speed_limit=1.0)

    return net


def build_workshop_network(aisle_capacity: int = 2) -> Network:
    """铁路货车检修车间场景（MAP-01）。

    几何、通道宽度与资源语义全部来自 scenes/rail_freight_workshop_v1.json；
    这个函数只做编译。要改车间就改场景文件，不要在这里加坐标——
    否则 Python、RAWSim-O 与 Gazebo 又会各有一张地图。
    """
    from amr_sched.workshop import DEFAULT_SCENE, WorkshopScene
    return WorkshopScene.load(DEFAULT_SCENE).network(aisle_capacity=aisle_capacity)


LAYOUTS = {
    "ring": None,          # 由 build_demo_network 提供，避免循环导入
    "grid": build_grid_network,
    "hub": build_hub_network,
    # MAP-01 主场景。ring/grid/hub 保持原样不动：历史实验要能原样重跑，
    # 新场景是**新增**的对照，不是替换。
    "workshop": build_workshop_network,
}


def build_layout(name: str, aisle_capacity: int = 2) -> Network:
    """按名字构造布局。``ring`` 为标准 17 节点环网。"""
    if name == "ring":
        from amr_sched.network import build_demo_network
        return build_demo_network(aisle_capacity=aisle_capacity)
    try:
        fn = LAYOUTS[name]
    except KeyError:
        raise ValueError("未知布局: %r（可选 %s）"
                         % (name, list(LAYOUTS))) from None
    return fn(aisle_capacity=aisle_capacity)