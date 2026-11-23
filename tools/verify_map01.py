"""MAP-01 车间场景几何、连通性与起讫业务规则校验。

为什么单独一个脚本
------------------
车间场景的核心几何与连通性校验项：路网连通性、任务起讫业务合法性、物理通道共享约束、
装卸后车辆资源释放、AGV 几何可通行性、坐标正反变换精度、场景版本与假设参数可追溯。
这些散落在 WorkshopScene.validate、GzFrame、仿真里，逐条去点很容易漏。
这里一次跑完，非零退出即有不满足项。

用法::

    python tools/verify_map01.py
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from amr_sched.workshop import WorkshopScene, legal_task_pairs, parking_slots
from tools.live_bridge.gz_frame import GzFrame

ok: list = []
bad: list = []


def chk(name, fn):
    try:
        ok.append((name, fn()))
    except Exception as exc:  # noqa: BLE001 - 验收脚本要把失败原因打出来，不是抛栈
        bad.append((name, "%s: %s" % (type(exc).__name__, exc)))


def scene() -> WorkshopScene:
    return WorkshopScene.load()


def connectivity() -> str:
    s = scene()
    net = s.network()
    nodes = set(net.nodes)
    starts = parking_slots(s)
    reached = set()
    for start in starts:
        stack = [start]
        while stack:
            cur = stack.pop()
            if cur in reached:
                continue
            reached.add(cur)
            for e in net.out_edges(cur):
                nxt = e.dst
                if nxt not in reached:
                    stack.append(nxt)
    missing = nodes - reached
    assert not missing, "从停车区出发到达不了 %d 个节点：%s" % (len(missing), sorted(missing)[:5])
    return "%d 节点全部可从 %d 个停车位出发到达" % (len(nodes), len(starts))


def task_legality() -> str:
    sc = scene()
    pairs = set(legal_task_pairs(sc))
    assert pairs, "合法任务对为空"

    # 从场景自己重新推一遍，而不是相信 legal_task_pairs 的实现：
    # 每种物料从取货区域取货（W 工位/库存），在本物料允许的卸货区域交接（D 装卸位）。
    expected = set()
    for material in sc.materials.values():
        for src in material.from_zones:
            pickup = sc.station_node(src)
            for dst in material.to_zones:
                if dst == src:
                    continue
                dropoff = sc.dock_node(dst)
                assert pickup and dropoff, "%s: %s -> %s 缺端点" % (material.material_id, src, dst)
                expected.add((pickup, dropoff))

    assert pairs == expected, "合法任务对与物料工艺重推结果不一致：%s" % (sorted(pairs ^ expected)[:4],)
    for p, d in pairs:
        assert p.startswith("W_"), "起点不是工位/库存节点：%s" % p
        assert d.startswith("D_"), "终点不是装卸交接位：%s" % d
    return "%d 个合法任务对，与 %d 种物料的工艺路线逐条一致" % (len(pairs), len(sc.materials))


def physical_channel() -> str:
    s = scene()
    two_way = [l for l in s.lanes if getattr(l, "two_way", False)]
    assert two_way, "没有双向通道，无法验证共享语义"
    for lane in two_way:
        assert lane.lane_count == 1, "%s 双向但 lane_count=%s，预约层会按共享路面处理" % (lane.lane_id, lane.lane_count)
        assert lane.physical_channel, "%s 缺 physical_channel，两个方向会被当成两条独立路面" % lane.lane_id
    return "%d 条双向通道全部 lane_count=1 且带 physical_channel" % len(two_way)


def geometry() -> str:
    s = scene()
    env = s.envelope
    width = float(env["width_m"])
    margin = float(env["min_clearance_m"])
    need = width + 2 * margin
    narrow = [(l.lane_id, l.width_m) for l in s.lanes if l.width_m < need]
    assert not narrow, "净宽不足：%s（需要 %.2f m）" % (narrow[:3], need)
    narrowest = min(l.width_m for l in s.lanes)
    return "最窄通道 %.2f m >= 车宽 %.2f + 2×%.2f = %.2f m" % (narrowest, width, margin, need)


def transform() -> str:
    f = GzFrame(scale=1.0 / 25.0)
    worst = 0.0
    for x, y in ((0.0, 0.0), (117.0, 55.0), (-40.0, 88.5)):
        wx, wy = f.to_world(x, y)
        bx, by = f.to_network(wx, wy)
        worst = max(worst, abs(bx - x), abs(by - y))
    assert worst < 1e-9, "正反变换误差 %g" % worst
    return "3 个点往返最大误差 %.2e m" % worst


def release() -> str:
    s = scene()
    net = s.network()
    cfg = ScenarioConfig(n_tasks=12, n_agvs=6, seed=0)
    _n, tasks, agvs = generate_scenario(
        cfg, network=net, pairs=legal_task_pairs(s), start_pool=parking_slots(s))
    sim = Simulator(net, tasks, agvs, StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(), SimulationConfig(dt=1.0, max_steps=200_000))

    # 逐步推进，边跑边数每个装卸交接位上同时有几辆车。
    # 服务容量 1 的含义是"同一时刻只允许一辆车占用"；如果卸完不走，
    # 后面的车就只能一直等——那样任务也跑不完，下面两条断言会同时挂。
    sim.reset()
    users: dict = {}
    worst = 0
    steps = 0
    while not sim.is_finished() and steps < 200_000:
        sim.step_once()
        steps += 1
        occupied: dict = {}
        for agv in sim.agvs:
            node = str(agv.cur_node)
            if node.startswith("D_"):
                occupied[node] = occupied.get(node, 0) + 1
                users.setdefault(node, set()).add(agv.agv_id)
        worst = max([worst] + list(occupied.values()))

    assert worst <= 1, "有装卸交接位同时停了 %d 辆车（服务容量应为 1）" % worst
    done = sum(1 for t in sim.tasks if t.complete_step)
    assert done == 12, "12 个任务只完成 %d 个，说明有车卡在交接位上没让出来" % done
    shared = {k: v for k, v in users.items() if len(v) > 1}
    assert shared, "没有任何一个交接位被两辆车先后使用过，无法证明卸完会释放"
    example = sorted(shared)[0]
    return ("%d 个任务全部完成；同一交接位同时最多 1 辆车；"
            "%d 个交接位被多车复用（如 %s 被 %d 辆车用过）"
            % (done, len(shared), example, len(shared[example])))


def traceability() -> str:
    s = scene()
    raw = s.raw
    assert raw.get("schema_version"), "场景缺 schema_version"
    assert raw.get("evidence"), "场景缺 evidence（证据等级）"
    assert raw.get("open_items_zh"), "场景缺 open_items_zh（未覆盖项）"
    digest = hashlib.sha256(Path(s.path).read_bytes()).hexdigest()[:12]
    return "schema %s / 证据等级 %d 项 / 未覆盖项 %d 条 / sha256 %s" % (
        raw["schema_version"], len(raw["evidence"]), len(raw["open_items_zh"]), digest)


def main() -> int:
    chk("M1.1 路网连通性", connectivity)
    chk("M1.2 任务起讫业务合法性", task_legality)
    chk("M1.3 物理通道共享约束", physical_channel)
    chk("M1.4 装卸后车辆资源释放", release)
    chk("M1.5 AGV 几何可通行性", geometry)
    chk("M1.6 坐标正反变换精度", transform)
    chk("M1.7 场景版本与假设可追溯", traceability)

    for name, msg in ok:
        print("  [OK]   %s：%s" % (name, msg))
    for name, msg in bad:
        print("  [FAIL] %s：%s" % (name, msg))
    print("")
    print("MAP-01 验收通过 %d / %d" % (len(ok), len(ok) + len(bad)))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())