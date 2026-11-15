#!/usr/bin/env python3
"""Demo B 端到端验证：Python 实时通道 -> TCP -> 独立消费端解析。

验的不是"代码看起来对"，而是：
  1. 真起 TCP 服务、真连、真收 hello 与连续 frame；
  2. 每帧用**独立实现的解析器**校验 schema（不复用生产端 dataclass，
     否则等于自己验自己）；
  3. step 必须单调递增；pose.node / pose.edge / path 引用的节点必须
     存在于 hello 下发的路网里；progress 必须在 [0,1]；
  4. 至少要观察到状态迁移，避免只推了几帧全空闲就宣称成功。
"""
import sys, json, socket, threading, time
sys.path.insert(0, ".")

from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig
from amr_sched.stream import FrameBuilder
from amr_sched.allocation import GreedyAllocator
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner


def serve(port, builder, stop_evt, errors, n_sent):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", port))
    srv.listen(1)
    srv.settimeout(15)
    try:
        conn, _ = srv.accept()
    except socket.timeout:
        errors.append("没人连上来")
        srv.close()
        return
    try:
        conn.sendall((json.dumps(builder.hello()) + "\n").encode())
        while not stop_evt.is_set():
            conn.sendall((json.dumps(builder.frame()) + "\n").encode())
            n_sent.append(1)
            time.sleep(0.05)
    except Exception as e:
        errors.append("发送失败: %r" % (e,))
    finally:
        try: conn.close()
        except Exception: pass
        srv.close()


def parse_hello(obj):
    assert obj["type"] == "hello", "首条不是 hello"
    assert isinstance(obj["schema"], str) and obj["schema"], "schema 缺失或非字符串"
    major = int(str(obj["schema"]).split(".")[0])
    assert major >= 1, "schema 主版本不兼容: %r" % (obj["schema"],)
    nodes, edges = obj["network"]["nodes"], obj["network"]["edges"]
    assert nodes, "路网没有节点"
    ids = {n["id"] for n in nodes}
    assert len(ids) == len(nodes), "节点 id 重复"
    for e in edges:
        assert e["u"] in ids and e["v"] in ids, "边引用未知节点: %r" % (e,)
    return ids


def parse_frame(obj, node_ids, prev_step):
    assert obj["type"] == "frame", "不是 frame"
    step = obj["step"]
    assert isinstance(step, int), "step 非整数"
    if prev_step is not None:
        assert step > prev_step, "step 未单调递增 %d -> %d" % (prev_step, step)
    agvs = obj["agvs"]
    assert agvs, "帧里没有车"
    for a in agvs:
        assert a["agv_id"], "agv_id 空"
        p = a["pose"]
        if p.get("node"):
            assert p["node"] in node_ids, "pose.node 未知: %r" % (p["node"],)
        if p.get("edge"):
            # 生产端把边序列化成 "U->V" 字符串（见 stream._split_edge），不是二元组
            e = p["edge"]
            assert isinstance(e, str) and "->" in e, "pose.edge 格式非法: %r" % (e,)
            u, v = e.split("->", 1)
            assert u in node_ids and v in node_ids, "pose.edge 未知: %r" % (e,)
        pr = float(p.get("progress", 0.0))
        assert -1e-9 <= pr <= 1.0 + 1e-9, "progress 越界: %r" % (pr,)
        assert a["state"], "state 空"
        for n in (a.get("path") or []):
            assert n in node_ids, "path 引用未知节点: %r" % (n,)
    assert isinstance(obj["task_counts"], dict), "task_counts 非字典"
    assert isinstance(obj["n_blocked"], int), "n_blocked 非整数"
    return step, {a["agv_id"]: a["state"] for a in agvs}


def main():
    sc = ScenarioConfig(n_tasks=12, n_agvs=4, seed=7, dt=1.0,
                        max_speed=1.0, urgent_ratio=0.15)
    net, tasks, agvs = generate_scenario(sc)
    sim = Simulator(net, tasks, agvs,
                    StaticDijkstraPlanner(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=60_000, seed=7))
    # reset() 才会建立 _load_end / _drop_end 等分步驱动所需的状态表；
    # run() 内部会自己调，但分步驱动必须显式调，否则 step_once 会 AttributeError。
    sim.reset()
    builder = FrameBuilder(sim)

    port = 39311
    stop_evt = threading.Event()
    errors, n_sent = [], []
    th = threading.Thread(target=serve,
                          args=(port, builder, stop_evt, errors, n_sent),
                          daemon=True)
    th.start()
    time.sleep(0.4)

    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.settimeout(10)
    f = sock.makefile("r", encoding="utf-8")

    hello = json.loads(f.readline())
    node_ids = parse_hello(hello)
    print("hello OK: schema=%s 节点=%d 边=%d 车=%d"
          % (hello["schema"], len(node_ids),
             len(hello["network"]["edges"]), len(hello["meta"]["agv_ids"])))

    frames, prev, states_seen, transitions = 0, None, set(), 0
    last = None
    for _ in range(200):
        sim.step_once()
        time.sleep(0.02)
        line = f.readline()
        if not line:
            break
        step, states = parse_frame(json.loads(line), node_ids, prev)
        prev = step
        frames += 1
        states_seen.update(states.values())
        if last is not None and states != last:
            transitions += 1
        last = states

    stop_evt.set()
    sock.close()
    time.sleep(0.3)

    print("收到帧数: %d (服务端发送 %d)" % (frames, len(n_sent)))
    print("step 区间: %s .. %s" % (None if frames == 0 else "(见上)", prev))
    print("观察到状态: %s" % sorted(states_seen))
    print("状态迁移次数: %d" % transitions)
    if errors:
        print("服务端错误: %s" % errors)

    ok = frames >= 30 and transitions >= 3 and not errors
    print("")
    print("判定:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
