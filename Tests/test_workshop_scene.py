"""MAP-01C：铁路货车检修车间场景的编译与校验。

这些测试盯的是"地图作为调度器的输入"这件事本身：几何对不对、等待区与工位库存
有没有分开、任务起讫合不合法。**其中一半是反向用例**——把场景故意弄坏，校验器
必须报出来。一个从不失败的校验器等于没有校验器，而"地图看起来对"正是这类项目
最容易自我欺骗的地方。
"""

from __future__ import annotations

import copy

import pytest

from amr_sched.layouts import LAYOUTS, build_layout
from amr_sched.network import NodeKind
from amr_sched.workshop import (
    DEFAULT_SCENE,
    EVIDENCE_LEVELS,
    NODE_KINDS,
    SceneError,
    WorkshopScene,
    load_and_validate,
)


@pytest.fixture(scope="module")
def scene() -> WorkshopScene:
    return WorkshopScene.load(DEFAULT_SCENE)


@pytest.fixture()
def raw(scene) -> dict:
    return copy.deepcopy(scene.raw)


def codes_after(raw: dict) -> list[str]:
    return [issue.code for issue in WorkshopScene(raw).validate()]


# ---------------------------------------------------------------- 正向：场景成立


def test_scene_validates_with_no_issues(scene):
    issues = scene.validate()
    assert issues == [], "\n".join(str(i) for i in issues)


def test_scene_file_is_the_single_source_of_geometry(scene):
    """网络的节点坐标必须逐字来自场景文件，编译不许再做一次投影或换算。"""
    net = scene.network()
    for node in scene.nodes.values():
        if not node.agv_access:
            continue
        compiled = net.node(node.node_id)
        assert (compiled.x, compiled.y) == (node.x, node.y)
        assert compiled.node_id == node.node_id
    assert scene.coordinate_roundtrip_error() == 0.0


def test_lane_length_comes_from_its_centerline(scene):
    net = scene.network()
    for lane in scene.lanes:
        assert net.edge(lane.src, lane.dst).length == pytest.approx(lane.polyline_length)
        if lane.two_way:
            assert net.edge(lane.dst, lane.src).length == pytest.approx(lane.polyline_length)


def test_agv_network_is_strongly_connected(scene):
    """每个 AGV 可通行节点都要能到别人、也要能回来。只通一半的路网会在仿真里变成
    永远出不来的死角，而在 2D 图上完全看不出来。"""
    net = scene.network()
    ids = sorted(scene.agv_nodes)
    reach = _forward(net, ids[0])
    assert set(ids) <= reach
    back = _forward(net, ids[0], reverse=True)
    assert set(ids) <= back


def _forward(net, start, reverse=False):
    seen, stack = {start}, [start]
    while stack:
        node = stack.pop()
        edges = net.in_edges(node) if reverse else net.out_edges(node)
        for edge in edges:
            nxt = edge.src if reverse else edge.dst
            if nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return seen


def test_every_material_route_is_legal_and_reachable(scene):
    """任务起讫必须来自物料-工艺路线，而且真的走得通。"""
    net = scene.network()
    checked = 0
    for material in scene.materials.values():
        pairs = scene.task_endpoints(material.material_id)
        assert pairs, f"{material.material_id} 没有合法的取送节点对"
        for pickup, dropoff in pairs:
            assert pickup != dropoff
            assert pickup in net.nodes and dropoff in net.nodes
            assert dropoff in _forward(net, pickup), f"{pickup} -> {dropoff} 不可达"
            checked += 1
    # 44 这个数同时写在 MAP-01A 文档里。钉住它，免得改了物料路线却忘了改文档。
    assert checked == 44


def test_pickup_is_at_stock_and_dropoff_at_the_dock(scene):
    """统一规则：从工位/库存取货，在装卸交接位交货。"""
    for material in scene.materials.values():
        for pickup, dropoff in scene.task_endpoints(material.material_id):
            assert pickup.startswith("W_")
            assert dropoff.startswith("D_")


def test_waiting_is_not_modelled_as_a_workstation(scene):
    """W5 类阻塞的建模修法：路口/等待/装卸/工位各是一个节点，容量语义各不相同。"""
    for scope in ("BRAKE", "BOGIE", "WHEELSET", "BODY", "COUPLER", "DOOR"):
        ring = [f"R_{scope}", f"Q_{scope}", f"D_{scope}", f"W_{scope}"]
        assert all(n in scene.nodes for n in ring)
        assert len(set(ring)) == 4

    q = scene.nodes["Q_BRAKE"]
    d = scene.nodes["D_BRAKE"]
    w = scene.nodes["W_BRAKE"]
    assert q.queue_slots == 2 and q.stock_capacity == 0
    assert d.service_capacity == 1 and d.agv_capacity == 1 and d.stock_capacity == 0
    assert w.stock_capacity >= 1
    # 物料占用（stock_capacity）与车辆占用（agv_capacity）是两个资源。
    assert (w.stock_capacity, w.agv_capacity) == (4, 1)


def test_every_wait_area_has_slots(scene):
    areas = [n for n in scene.nodes.values() if n.type == "wait_area"]
    assert len(areas) == 12
    assert all(n.queue_slots >= 1 for n in areas)


def test_every_dock_can_be_left(scene):
    """卸完必须能走：装卸位必须有出边。没有出边的装卸位就是永久堵点。"""
    net = scene.network()
    docks = [n for n in scene.nodes.values() if n.type == "service_dock"]
    assert len(docks) == 12
    for dock in docks:
        assert net.out_edges(dock.node_id), f"{dock.node_id} 有进无出"


def test_single_lane_two_way_is_one_shared_physical_resource(scene):
    """单车道双向：两个方向同属一个 physical_channel，容量都是 1。

    仿真层的 _reserve_hop 会把两个方向一起登记，_can_enter 取两个方向容量的最小值，
    所以这里必须是 1；如果场景写成 2，两个方向就会各自预约成功而在路上会车。
    """
    net = scene.network()
    two_way = [l for l in scene.lanes if l.two_way]
    assert two_way, "场景里应当有双向通道"
    for lane in two_way:
        assert lane.lane_count == 1
        assert lane.edge_capacity == 1
        fwd, rev = net.edge(lane.src, lane.dst), net.edge(lane.dst, lane.src)
        assert fwd.capacity == 1 and rev.capacity == 1
        assert min(fwd.capacity, rev.capacity) == lane.edge_capacity
    channels = {l.physical_channel for l in two_way}
    assert len(channels) == len(two_way), "每条双向通道应当有自己的物理通道编号"


def test_node_kinds_come_from_the_scene_file(scene):
    net = scene.network()
    expected = {
        "D_BRAKE": NodeKind.DOCK,
        "Q_BRAKE": NodeKind.BUFFER,
        "R_BRAKE": NodeKind.AISLE,
        "W_BRAKE": NodeKind.WORKSTATION,
        "W_DISASSEMBLY": NodeKind.DISASSEMBLY,
        "W_ASSEMBLY": NodeKind.ASSEMBLY,
    }
    for node_id, kind in expected.items():
        assert net.node(node_id).kind is kind
    for node in scene.nodes.values():
        assert node.node_kind in NODE_KINDS


def test_aisle_capacity_can_be_overridden_like_the_other_layouts(scene):
    net = scene.network(aisle_capacity=1)
    assert net.node("J_S_45").capacity == 1
    assert net.node("W_BRAKE").capacity == 1  # 非路口节点不受影响


def test_evidence_block_states_what_the_scene_is_not(scene):
    """尺寸是研究假设，必须写清楚不是什么。演示画面最容易被读成实测车间。"""
    assert scene.evidence["level"] in EVIDENCE_LEVELS
    assert scene.evidence["level"] == "process_referenced_synthetic"
    text = " ".join(scene.evidence["not_measured_zh"])
    assert "没有" in text and "实测" in text
    assert scene.evidence["assumptions_zh"]
    assert "假设" in " ".join(scene.evidence["assumptions_zh"])


def test_historical_layouts_are_untouched():
    """MAP-01 是新增场景，不是替换：ring/grid/hub 必须逐字保持原样。"""
    assert len(build_layout("ring").nodes) == 17
    assert len(build_layout("ring").edges) == 36
    assert len(build_layout("grid").nodes) == 20
    assert len(build_layout("grid").edges) == 46
    assert len(build_layout("hub").nodes) == 11
    assert len(build_layout("hub").edges) == 20


def test_workshop_is_registered_and_compiles_through_build_layout():
    assert "workshop" in LAYOUTS
    net = build_layout("workshop", aisle_capacity=2)
    assert len(net.nodes) == 62
    assert len(net.edges) == 144


# ------------------------------------------------- 反向：坏场景必须被抓住


def test_validator_rejects_a_cut_off_area(raw):
    raw["lanes"] = [l for l in raw["lanes"] if l["id"] != "L_Q_BODY__D_BODY"]
    assert "unreachable_node" in codes_after(raw)


def test_validator_rejects_a_dock_with_no_way_out(raw):
    """装卸位必须是"进得来、出得去"。把它仅有的两条支线都拿掉，它就成了死胡同。"""
    raw["lanes"] = [
        l for l in raw["lanes"]
        if l["id"] not in ("L_D_BODY__W_BODY", "L_Q_BODY__D_BODY")
    ]
    assert "dock_without_exit" in codes_after(raw)


def test_validator_rejects_a_station_without_stock(raw):
    for node in raw["nodes"]:
        if node["id"] == "W_BODY":
            node["stock_capacity"] = 0
    assert "station_no_stock" in codes_after(raw)


def test_validator_rejects_a_dock_that_holds_stock(raw):
    for node in raw["nodes"]:
        if node["id"] == "D_BODY":
            node["stock_capacity"] = 3
    assert "dock_holds_stock" in codes_after(raw)


def test_validator_rejects_a_dock_that_serves_two_vehicles(raw):
    for node in raw["nodes"]:
        if node["id"] == "D_BODY":
            node["agv_capacity"] = 2
    assert "dock_capacity" in codes_after(raw)


def test_validator_rejects_a_wait_area_without_slots(raw):
    for node in raw["nodes"]:
        if node["id"] == "Q_BODY":
            node["queue_slots"] = 0
    assert "wait_area_no_slots" in codes_after(raw)


def test_validator_rejects_a_lane_narrower_than_the_vehicle(raw):
    for lane in raw["lanes"]:
        if lane["id"] == "L_Q_BODY__D_BODY":
            lane["width_m"] = 0.8
    assert "lane_too_narrow" in codes_after(raw)


def test_validator_rejects_a_centerline_that_does_not_reach_its_node(raw):
    for node in raw["nodes"]:
        if node["id"] == "D_BODY":
            node["y"] = node["y"] + 1.0
    assert "lane_centerline_detached" in codes_after(raw)


def test_validator_rejects_two_roads_where_the_reservation_layer_sees_one(raw):
    """把一条双向通道拆成两条方向各异的单向通道，并给它们不同的物理通道编号：
    场景在说"这是两块独立的路面"，而预约层对存在反向边的通道一律按共享路面处理。
    两个说法必须统一，否则"地图"和"调度器眼里的地图"不是一张图。"""
    for lane in raw["lanes"]:
        if lane["id"] == "L_J_S_25__J_S_45":
            lane["direction"] = "one_way"
            lane["physical_channel"] = "CH_S_SPLIT_A"
    raw["lanes"].append({
        "id": "L_J_S_45__J_S_25", "from": "J_S_45", "to": "J_S_25",
        "direction": "one_way", "lane_count": 1, "width_m": 3.0,
        "speed_limit_mps": 1.0, "physical_channel": "CH_S_SPLIT_B",
        "centerline": [[45, 30], [25, 30]],
    })
    codes = codes_after(raw)
    assert "channel_split" in codes
    assert "network_compile" not in codes, "这种场景本身是能编译的，问题在语义"


def test_validator_rejects_one_channel_id_on_two_different_segments(raw):
    """同一个物理通道编号被用在两条不同路段上：预约时它们会互相顶替。"""
    for lane in raw["lanes"]:
        if lane["id"] == "L_J_S_105__J_S_125":
            lane["physical_channel"] = "CH_S_0"
    assert "channel_ambiguous" in codes_after(raw)


def test_validator_rejects_an_unknown_material_zone(raw):
    for material in raw["materials"]:
        if material["id"] == "brake_parts":
            material["to_zones"] = ["NO_SUCH_ZONE"]
    codes = codes_after(raw)
    assert "material_unknown_zone" in codes
    assert "material_endpoint_missing" in codes


def test_validator_rejects_an_unlabelled_evidence_level(raw):
    raw["evidence"]["level"] = "we_drew_it_in_powerpoint"
    assert "evidence_level_unknown" in codes_after(raw)


def test_validator_rejects_a_scene_without_an_assumption_list(raw):
    raw["evidence"]["assumptions_zh"] = []
    assert "evidence_assumptions_missing" in codes_after(raw)


def test_validator_rejects_a_node_outside_the_workshop(raw):
    for node in raw["nodes"]:
        if node["id"] == "W_BODY":
            node["x"] = 999.0
    assert "node_out_of_bounds" in codes_after(raw)


def test_validator_reports_a_duplicate_directed_edge_instead_of_crashing(raw):
    """同一条有向边写两遍：Network.add_edge 会抛 ValueError。
    校验器必须把它变成一条报告，而不是让 --help 之外的调用方看到 traceback。"""
    base = next(l for l in raw["lanes"] if l["id"] == "L_J_S_25__J_S_45")
    dup = dict(base)
    dup["id"] = "L_J_S_25__J_S_45_TWICE"
    dup["direction"] = "one_way"
    raw["lanes"].append(dup)
    codes = codes_after(raw)
    assert "network_compile" in codes


def test_two_way_two_lane_is_refused_loudly(scene):
    """v1 不支持双向双车道；必须报错，而不是安静地按单车道算。"""
    raw = copy.deepcopy(scene.raw)
    for lane in raw["lanes"]:
        if lane["id"] == "L_J_S_25__J_S_45":
            lane["lane_count"] = 2
    with pytest.raises(SceneError, match="双向 2 车道"):
        WorkshopScene(raw).network()
    assert "network_compile" in codes_after(raw)


def test_load_and_validate_does_not_raise(scene):
    loaded, issues = load_and_validate(DEFAULT_SCENE)
    assert loaded.scene_id == scene.scene_id
    assert issues == []


def test_loading_a_missing_scene_says_so():
    with pytest.raises(SceneError, match="场景文件不存在"):
        WorkshopScene.load("no_such_scene")


def test_task_endpoints_rejects_an_unknown_material(scene):
    with pytest.raises(SceneError, match="未知物料"):
        scene.task_endpoints("no_such_material")

def test_generated_tasks_only_use_legal_material_routes(scene):
    """任务生成必须落在物料-工艺路线上，不能从全体节点随机挑两点。"""
    from amr_sched.workshop import build_scenario, legal_task_pairs

    legal = set(legal_task_pairs(scene))
    _net, tasks, _agvs = build_scenario(scene=scene, n_tasks=40, n_agvs=3, seed=7)
    assert len(tasks) == 40
    for task in tasks:
        assert (task.pickup, task.dropoff) in legal, (task.pickup, task.dropoff)
        assert task.pickup.startswith("W_"), task.pickup
        assert task.dropoff.startswith("D_"), task.dropoff


def test_fleet_starts_in_the_parking_area_not_in_a_queue_slot(scene):
    """车辆从停车充电区出发；Q_ 是近站等待槽位，不是车队待命位。"""
    from amr_sched.workshop import build_scenario, parking_slots

    slots = parking_slots(scene)
    assert slots, "停车区没有可用车位"
    assert all(n.split("_")[1] == "PARK" for n in slots)
    _net, _tasks, agvs = build_scenario(scene=scene, n_tasks=5, n_agvs=3, seed=0)
    assert [a.start_node for a in agvs] == slots[:3]


def test_a_small_fleet_finishes_on_the_workshop_network(scene):
    """场景不仅要能编译，还要能真的跑完：6 个任务 / 2 台车必须全部送达且不死锁。"""
    from amr_sched.allocation.greedy import GreedyAllocator
    from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
    from amr_sched.simulation import SimulationConfig, Simulator
    from amr_sched.workshop import build_scenario

    net, tasks, agvs = build_scenario(scene=scene, n_tasks=6, n_agvs=2, seed=0)
    result = Simulator(
        net, tasks, agvs,
        StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
        GreedyAllocator(),
        SimulationConfig(dt=1.0, max_steps=200_000),
    ).run()
    summary = result.summary()
    assert summary["tasks_completed"] == 6
    assert summary["collision_events"] == 0
    assert summary["deadlock_breaks"] == 0


def test_the_scene_keeps_charging_and_forbidden_zones_open():
    """两份清单必须诚实：哪些做了、哪些没做。

    MAP-01 的硬约束列了"充电位"和"禁行区"两类资源，v1 都没有建模。
    这一条不是检查功能，而是检查**场景文件自己承认了没做**——
    从而显式标注当前场景覆盖边界。
    """
    from amr_sched.workshop import WorkshopScene

    scene = WorkshopScene.load()
    ids = set(scene.nodes)
    assert not any(nid.startswith(("C_", "CHG_", "CHARGE_")) for nid in ids), (
        "场景里出现了充电位节点；若确实建模了，请同步改 open_items_zh 与 MAP-01A")
    assert not any(z.kind == "forbidden" for z in scene.zones.values()), (
        "场景里出现了禁行区；若确实建模了，请同步改 open_items_zh 与 MAP-01A")

    text = " ".join(scene.raw.get("open_items_zh", []))
    assert "禁行区" in text, "未建模的禁行区必须在 open_items_zh 里写出来"
    assert "充电" in text or "双车道" in text, "open_items_zh 至少要写出两项未覆盖的资源"
