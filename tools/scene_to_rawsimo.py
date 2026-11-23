"""把车间场景导出成 RAWSim-O 的实例，让两边共用同一张地图。

为什么存在
----------
调度器、Web 控制台与 Gazebo 缩比台架都读 scenes/rail_freight_workshop_v1.json，
而 RAWSim-O 读自己的 .xinst / .xsett / .xconf 三件套。此前 RAWSim-O 里那张图是
生成器造的 3x3 m 玩具布局，与车间没有任何关系——这就是 MAP-01D 记录的空白：
"RAWSim-O 与其它三端不是同一个场景来源"。本导出器把第二个来源去掉：下面每一个
waypoint、path、station 都由场景文件推出，并且导出的文件必须能通过 RAWSim-O 自己的
sanity 规则（死胡同、可被堵死的存储位、休息位不足），否则直接报错不产出。

这里主张什么、不主张什么
------------------------
这是一次**几何**导出，不是行为等价。场景建模的是 AGV 配送调度，RAWSim-O 建模的是
pod 挑选系统。因此映射是：

    场景 station (W)        -> InputStation   物料库存：AGV 在这里取货
    场景 service_dock (D)   -> OutputStation  交接位：AGV 在这里卸货
    场景 wait_area (Q)      -> 普通 waypoint
    场景 intersection       -> 普通 waypoint

**不**主张 RAWSim-O 的 pod 流程复现了调度器的配送逻辑。主张只有一条：操作员看到的
那张地图是同一个车间、来自同一个文件。

尺度
----
--scale 乘在每一个场景坐标上。默认 1/25 让地图与 Gazebo 缩比台架同框
（GzFrame(scale=1/25)），于是桥接报来的位姿可以靠一个**测出来的**标定画上去，
而不是靠假设一个变换。

用法
----
    python3 tools/scene_to_rawsimo.py --out-dir out/archive/map01_dual_demo
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from xml.sax.saxutils import escape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_SCENE = os.path.join(ROOT, "scenes", "rail_freight_workshop_v1.json")
BENCH_SCALE = 1.0 / 25.0

#: 场景节点类型 -> RAWSim-O 角色。取值只有三种：input / output / plain。
NODE_ROLE = {
    "station": "input",
    "service_dock": "output",
    "wait_area": "plain",
    "intersection": "plain",
}


class ExportError(ValueError):
    """场景无法导出成 RAWSim-O 实例（与"导出后不合法"不同：这是输入本身不成立）。"""


def load_scene(path):
    """读场景 JSON，并检查导出需要的字段都在。"""
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    for key in ("scene_id", "nodes", "lanes", "bounds_m"):
        if key not in raw:
            raise ExportError("场景缺少字段 %s" % key)
    node_ids = [n["id"] for n in raw["nodes"]]
    if len(set(node_ids)) != len(node_ids):
        raise ExportError("场景存在重名节点")
    known = set(node_ids)
    for lane in raw["lanes"]:
        for end in (lane["from"], lane["to"]):
            if end not in known:
                raise ExportError("通道 %s 的端点 %s 不在节点表里" % (lane["id"], end))
    return raw


def directed_edges(lanes):
    """通道 -> 有向边列表。双向通道产生两条反向边。"""
    edges = []
    for lane in lanes:
        edges.append((lane["from"], lane["to"], lane["id"]))
        if lane.get("direction") == "two_way":
            edges.append((lane["to"], lane["from"], lane["id"]))
    return edges


def waypoint_ids(scene):
    """场景节点 ID -> RAWSim-O waypoint ID（从 1 开始）。

    RAWSim-O 的 waypoint 只有整数 ID，场景用的是 J_S_125 这样的名字，所以两套编号
    必须有一个唯一来源。此前这个映射是 build_model 里的局部变量，于是控制台把
    waypoint 7 发给 Python 时，Python 找不到名为 "7" 的节点——同一张地图被叫成两个
    名字。把它提出来，导出和路线服务都从这里取，才不会各推一套。
    """
    return {node["id"]: index + 1 for index, node in enumerate(scene["nodes"])}


def node_ids_by_waypoint(scene):
    """waypoint ID -> 场景节点 ID，waypoint_ids 的反向表。"""
    return {index: node_id for node_id, index in waypoint_ids(scene).items()}


def build_model(scene, scale=BENCH_SCALE, n_bots=4, bot_radius=0.45,
                station_radius=0.30, max_velocity=1.0):
    """场景 -> 中间模型（纯函数，便于测试：不碰文件系统）。"""
    if scale <= 0:
        raise ExportError("scale 必须为正")
    nodes = scene["nodes"]
    waypoint_of = waypoint_ids(scene)

    role_of = {}
    inputs, outputs = [], []
    for node in nodes:
        role = NODE_ROLE.get(node.get("type"))
        if role is None:
            raise ExportError("未知节点类型 %r（节点 %s）" % (node.get("type"), node["id"]))
        role_of[node["id"]] = role
        if role == "input":
            inputs.append(node["id"])
        elif role == "output":
            outputs.append(node["id"])
    if not inputs or not outputs:
        raise ExportError("实例至少需要一个 InputStation 与一个 OutputStation")

    input_index = {node_id: i for i, node_id in enumerate(inputs)}
    output_index = {node_id: i for i, node_id in enumerate(outputs)}

    paths = {node["id"]: [] for node in nodes}
    for src, dst, _lane in directed_edges(scene["lanes"]):
        if waypoint_of[dst] not in paths[src]:
            paths[src].append(waypoint_of[dst])

    # 休息位：RAWSim-O 要求 存储位数 - pod 数 >= bot 数。场景没有 pod，
    # 于是把等待区整片当作休息位（语义上就是"车停在这儿等活"）。
    resting = [n["id"] for n in nodes if n.get("type") == "wait_area"]
    if len(resting) < n_bots:
        raise ExportError("休息位 %d 个，不足以放下 %d 台车" % (len(resting), n_bots))

    bots = []
    for i in range(n_bots):
        host = resting[i % len(resting)]
        node = next(n for n in nodes if n["id"] == host)
        bots.append({"id": i, "x": node["x"] * scale, "y": node["y"] * scale,
                     "radius": bot_radius, "max_velocity": max_velocity})

    return {
        "scene_id": scene["scene_id"],
        "scale": scale,
        "waypoint_of": waypoint_of,
        "nodes": nodes,
        "paths": paths,
        "role_of": role_of,
        "inputs": inputs,
        "outputs": outputs,
        "input_index": input_index,
        "output_index": output_index,
        "bots": bots,
        "resting": resting,
        "bounds": scene["bounds_m"],
        "station_radius": station_radius,
    }


def check_model(model):
    """按 RAWSim-O InstanceSanity.SanityCheck 的规则自检，返回问题列表。"""
    problems = []
    for node_id, out in model["paths"].items():
        if not out:
            problems.append("DeadEnd: %s 没有任何出边" % node_id)
    for node_id in model["resting"]:
        if not model["paths"][node_id]:
            problems.append("BlockableStorageLocation: %s 是死胡同" % node_id)
    if len(model["resting"]) < len(model["bots"]):
        problems.append("InsufficientRestingPositions: %d - 0 < %d"
                        % (len(model["resting"]), len(model["bots"])))
    # 可达性：从任一休息位出发应当能走到每个工位与每个交接位。
    by_waypoint = {wid: nid for nid, wid in model["waypoint_of"].items()}
    seen, stack = set(), [model["resting"][0]]
    while stack:
        current = stack.pop()
        if current in seen:
            continue
        seen.add(current)
        for neighbour in model["paths"][current]:
            stack.append(by_waypoint[neighbour])
    for node_id in list(model["inputs"]) + list(model["outputs"]):
        if node_id not in seen:
            problems.append("Unreachable: %s 从休息位不可达" % node_id)
    return problems


def _f(value):
    """固定小数格式，避免 locale 与浮点噪声影响 diff。"""
    return ("%.6f" % float(value)).rstrip("0").rstrip(".") or "0"


def render_instance(model, name):
    """中间模型 -> .xinst 文本。"""
    scale = model["scale"]
    node_of = {n["id"]: n for n in model["nodes"]}
    lines = []
    add = lines.append
    add('<?xml version="1.0" encoding="utf-8"?>')
    add('<Instance xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema" Name="%s">' % escape(name))

    add("  <Bots>")
    for bot in model["bots"]:
        add('    <Bot ID="%d" PodTransferTime="3" MaxAcceleration="0.5" '
            'MaxDeceleration="0.5" MaxVelocity="%s" TurnSpeed="5.5" '
            'CollisionPenaltyTime="0.5" X="%s" Y="%s" Radius="%s" Orientation="0" Tier="0" />'
            % (bot["id"], _f(bot["max_velocity"]), _f(bot["x"]), _f(bot["y"]),
               _f(bot["radius"])))
    add("  </Bots>")

    # 场景没有 pod：如实导出 0 个，而不是为了"像 RAWSim-O"就编几个出来。
    add("  <Pods>")
    add("  </Pods>")
    add("  <Elevators />")

    add("  <InputStations>")
    for node_id, index in model["input_index"].items():
        node = node_of[node_id]
        add('    <InputStation ID="%d" X="%s" Y="%s" Radius="%s" Tier="0" '
            'Capacity="100" ItemBundleTransferTime="2">'
            % (index, _f(node["x"] * scale), _f(node["y"] * scale),
               _f(model["station_radius"])))
        add("      <Queues>")
        add("      </Queues>")
        add("    </InputStation>")
    add("  </InputStations>")

    add("  <OutputStations>")
    for node_id, index in model["output_index"].items():
        node = node_of[node_id]
        add('    <OutputStation ID="%d" X="%s" Y="%s" Radius="%s" Tier="0" '
            'Capacity="20" ItemTransferTime="2">'
            % (index, _f(node["x"] * scale), _f(node["y"] * scale),
               _f(model["station_radius"])))
        add("      <Queues>")
        add("      </Queues>")
        add("    </OutputStation>")
    add("  </OutputStations>")

    add("  <Tiers>")
    add('    <Tier ID="0" Length="%s" Width="%s" RelativePositionX="0" '
        'RelativePositionY="0" RelativePositionZ="0" />'
        % (_f(model["bounds"][0] * scale), _f(model["bounds"][1] * scale)))
    add("  </Tiers>")

    add("  <Waypoints>")
    for node in model["nodes"]:
        node_id = node["id"]
        add('    <Waypoint ID="%d" X="%s" Y="%s" Tier="0" OutputStation="%d" '
            'InputStation="%d" Elevator="-1" Pod="-1" PodStorageLocation="%s" '
            'IsQueueWaypoint="false">'
            % (model["waypoint_of"][node_id], _f(node["x"] * scale), _f(node["y"] * scale),
               model["output_index"].get(node_id, -1),
               model["input_index"].get(node_id, -1),
               "true" if node_id in model["resting"] else "false"))
        add("      <Paths>")
        for neighbour in model["paths"][node_id]:
            add("        <Waypoint>%d</Waypoint>" % neighbour)
        add("      </Paths>")
        add("    </Waypoint>")
    add("  </Waypoints>")
    add("  <Semaphores />")
    add("</Instance>")
    return "\n".join(lines) + "\n"


def render_settings(name):
    """最小可用 .xsett：够 GUI 加载这个实例即可。"""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<SettingConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema">\n'
        "  <Name>" + escape(name) + "</Name>\n"
        "  <SimulationWarmupTime>0</SimulationWarmupTime>\n"
        "  <SimulationDuration>600</SimulationDuration>\n"
        "  <Tolerance>0.2</Tolerance>\n"
        "  <UseAcceleration>false</UseAcceleration>\n"
        "  <UseTurnDelay>false</UseTurnDelay>\n"
        "  <QueueHandlingEnabled>true</QueueHandlingEnabled>\n"
        "  <Seed>0</Seed>\n"
        "  <LogLevel>Default</LogLevel>\n"
        "  <HeatMode>NumItemsHandled</HeatMode>\n"
        "  <InventoryConfiguration>\n"
        "    <ItemType>SimpleItem</ItemType>\n"
        "    <OrderMode>Fill</OrderMode>\n"
        "    <InitialInventory>0.7</InitialInventory>\n"
        "    <IgnoreCapacityForBundleGeneration>false</IgnoreCapacityForBundleGeneration>\n"
        "    <BufferBundlesUntilInventoryLoad>1.1</BufferBundlesUntilInventoryLoad>\n"
        "    <WarmupOrderCount>5000</WarmupOrderCount>\n"
        "    <SimpleItemConfiguration>\n"
        "      <GeneratorConfigFile>ItemGenMinimalistic.xgenc</GeneratorConfigFile>\n"
        "    </SimpleItemConfiguration>\n"
        "  </InventoryConfiguration>\n"
        "</SettingConfiguration>\n")


def render_control(name):
    """最小可用 .xconf：路径规划用 FAR，任务分配用 Balanced。"""
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<ControlConfiguration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
        'xmlns:xsd="http://www.w3.org/2001/XMLSchema">\n'
        "  <Name>" + escape(name) + "-FAR</Name>\n"
        "  <PathPlanningType>FAR</PathPlanningType>\n"
        '  <PathPlanningConfig xsi:type="FARPathPlanningConfiguration">\n'
        "    <AutoSetParameter>false</AutoSetParameter>\n"
        "    <LengthOfAWaitStep>2</LengthOfAWaitStep>\n"
        "    <RuntimeLimitPerAgent>0.25</RuntimeLimitPerAgent>\n"
        "    <Clocking>3</Clocking>\n"
        "    <MaximumNumberOfBreakingManeuverTries>4</MaximumNumberOfBreakingManeuverTries>\n"
        "    <evadingStrategy>EvadeByRerouting</evadingStrategy>\n"
        "    <NoBackEvading>true</NoBackEvading>\n"
        "  </PathPlanningConfig>\n"
        "  <TaskAllocationType>Balanced</TaskAllocationType>\n"
        '  <TaskAllocationConfig xsi:type="BalancedTaskAllocationConfiguration">\n'
        "    <BotReallocationTimeout>60</BotReallocationTimeout>\n"
        "    <WeightInventoryFeedback>false</WeightInventoryFeedback>\n"
        "    <InventoryFillDesired>false</InventoryFillDesired>\n"
        "    <WeightInputStations>1</WeightInputStations>\n"
        "    <WeightOutputStations>1</WeightOutputStations>\n"
        "    <PreferSameTier>false</PreferSameTier>\n"
        "    <ExtendSearch>false</ExtendSearch>\n"
        "    <ExtendedSearchRadius>2</ExtendedSearchRadius>\n"
        "    <OutputPodOrderType>Nearest</OutputPodOrderType>\n"
        "    <InputPodOrderType>Nearest</InputPodOrderType>\n"
        "    <RestLocationOrderType>Nearest</RestLocationOrderType>\n"
        "  </TaskAllocationConfig>\n"
        "  <ItemStorageType>Random</ItemStorageType>\n"
        '  <ItemStorageConfig xsi:type="RandomItemStorageConfiguration" />\n'
        "  <PodStorageType>Random</PodStorageType>\n"
        '  <PodStorageConfig xsi:type="RandomPodStorageConfiguration">\n'
        "    <PreferSameTier>true</PreferSameTier>\n"
        "  </PodStorageConfig>\n"
        "  <OrderBatchingType>Random</OrderBatchingType>\n"
        '  <OrderBatchingConfig xsi:type="RandomOrderBatchingConfiguration" />\n'
        "  <ReplenishmentBatchingType>Random</ReplenishmentBatchingType>\n"
        '  <ReplenishmentBatchingConfig xsi:type="RandomReplenishmentBatchingConfiguration" />\n'
        "</ControlConfiguration>\n")


def export(scene_path=DEFAULT_SCENE, out_dir=None, name=None, scale=BENCH_SCALE,
           n_bots=4, verbose=True):
    """导出并返回写出的文件路径。任何自检问题都会抛 ExportError。"""
    scene = load_scene(scene_path)
    model = build_model(scene, scale=scale, n_bots=n_bots)
    problems = check_model(model)
    if problems:
        raise ExportError("导出的实例过不了 RAWSim-O 的 sanity 检查: " + "; ".join(problems))
    out_dir = out_dir or os.path.dirname(os.path.abspath(scene_path))
    name = name or "%s_bench25" % scene["scene_id"]
    written = {}
    payloads = {
        ".xinst": render_instance(model, name),
        ".xsett": render_settings(name),
        ".xconf": render_control(name),
    }
    for suffix, text in payloads.items():
        path = os.path.join(out_dir, name + suffix)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(text)
        written[suffix] = path
    if verbose:
        print("场景 %s -> %s" % (scene["scene_id"], out_dir))
        print("  waypoint %d 个（其中休息位 %d），有向边 %d 条"
              % (len(model["nodes"]), len(model["resting"]),
                 sum(len(v) for v in model["paths"].values())))
        print("  InputStation %d（工位 W），OutputStation %d（交接位 D），Bot %d"
              % (len(model["inputs"]), len(model["outputs"]), len(model["bots"])))
        print("  尺度 %.4f，台面 %.3f x %.3f m"
              % (scale, model["bounds"][0] * scale, model["bounds"][1] * scale))
        for suffix, path in written.items():
            print("  写出 %s" % path)
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="车间场景 -> RAWSim-O 实例")
    ap.add_argument("--scene", default=DEFAULT_SCENE)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--scale", type=float, default=BENCH_SCALE,
                    help="坐标倍率；默认 1/25，与 Gazebo 缩比台架同框")
    ap.add_argument("--bots", type=int, default=4)
    args = ap.parse_args(argv)
    export(scene_path=args.scene, out_dir=args.out_dir, name=args.name,
           scale=args.scale, n_bots=args.bots)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
