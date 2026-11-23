"""导出调度过程为自包含 HTML 回放。

和 RAWSim-O 可视化有什么不同
----------------------------
RAWSim-O 自带 WPF 俯视地图，能看到车在哪、货架在哪。这里**不重复**那个视角
。

本工具专门画 RAWSim-O 画不出来的三样东西——它们是时空预约调度的核心机制：

1. **预约表占用**：冲突消解的全部机理在预约表里。只看车辆位置，
   永远解释不清"为什么这台车在这儿等了 700 步"；看预约才知道
   是哪个节点的哪个时间窗被别人提前订走了。
2. **抢占事件**：的核心结论是"抢占应止于未取货"。
   抢占发生的瞬间，界面上会标出谁抢了谁、对方是否已装货。
3. **工位加工占用**：6.11 节的工位节拍，工位饱和是相变点。

产物是单个 HTML 文件，内含全部数据与渲染逻辑：不联网、不需服务器、
双击浏览器即可离线交互查看。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.scenarios import ScenarioConfig, generate_scenario
from amr_sched.simulation import Simulator, SimulationConfig

HERE = os.path.dirname(os.path.abspath(__file__))


def build_payload(n_tasks, n_agvs, seed, planner="spacetime_astar",
                  process_time=0.0, policy="none", step_stride=1):
    sc = ScenarioConfig(n_tasks=n_tasks, n_agvs=n_agvs, seed=seed,
                        dt=1.0, max_speed=1.0, urgent_ratio=0.15)
    net, tasks, agvs = generate_scenario(sc)
    pcls = (SpaceTimeAStarPlanner if planner == "spacetime_astar"
            else StaticDijkstraPlanner)
    sim = Simulator(net, tasks, agvs, pcls(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=60_000, seed=seed,
                                     process_time=process_time,
                                     reschedule_policy=ReschedulePolicy(policy),
                                     record_trajectory=True))
    result = sim.run()

    nodes = [{"id": n.node_id, "x": round(n.x, 3), "y": round(n.y, 3),
              "kind": n.kind.value, "cap": n.capacity}
             for n in net.nodes.values()]
    edges = []
    seen = set()
    for e in net.edges.values():
        key = tuple(sorted((e.src, e.dst)))
        if key in seen:
            continue
        seen.add(key)
        edges.append({"a": key[0], "b": key[1]})

    steps = []
    for idx, r in enumerate(result.trajectory):
        # 跳帧取样时必须**保留有抢占事件的帧**：
        # 抢占是瞬时的，一旦被 stride 跳过就永远看不到了。
        # 实测 stride=2 时 4 次抢占只剩 1 次。
        if (step_stride > 1 and idx % step_stride != 0
                and not r.preemptions):
            continue
        steps.append({"t": r.step, "pos": r.agv_positions,
                      "st": r.agv_states, "job": r.agv_tasks,
                      # path 是规划器给出的完整既定路线（不是走过的历史）。
                      # 只导 pos 能看到车在哪，但看不出它接下来打算怎么走。
                      "path": {k: list(v) for k, v in r.agv_paths.items()},
                      "cnt": r.task_counts, "nb": r.n_blocked,
                      "hold": {k: v for k, v in r.node_holds.items()},
                      "pre": [list(x) for x in r.preemptions]})

    s = result.summary()
    return {
        "meta": {"n_tasks": n_tasks, "n_agvs": n_agvs, "seed": seed,
                 "planner": planner, "policy": policy,
                 "process_time": process_time,
                 "makespan": round(s["makespan_seconds"], 1),
                 "on_time": round(s["on_time_rate"], 4),
                 "util": round(s["agv_utilization"], 4),
                 "conflict_events": s["conflict_wait_events"],
                 "conflict_steps": s["conflict_wait_steps"],
                 "preemptions": s["preemption_count"],
            "completed": s["tasks_completed"]},
        "nodes": nodes,
        "edges": edges,
        "agv_ids": [a.agv_id for a in sim.agvs],
        "steps": steps,
    }


def _payload_via_stream(args):
    """经统一接口生成 payload，并顺带写出一份 .json 交换文件。"""
    from amr_sched.reschedule import ReschedulePolicy as _RP
    from amr_sched.stream import export_stream, frames_from_result

    sc = ScenarioConfig(n_tasks=args.tasks, n_agvs=args.agvs, seed=args.seed,
                        dt=1.0, max_speed=1.0, urgent_ratio=0.15)
    net, tasks, agvs = generate_scenario(sc)
    pcls = (SpaceTimeAStarPlanner if args.planner == "spacetime_astar"
            else StaticDijkstraPlanner)
    sim = Simulator(net, tasks, agvs, pcls(max_speed=1.0, dt=1.0),
                    GreedyAllocator(),
                    SimulationConfig(dt=1.0, max_steps=60_000, seed=args.seed,
                                     process_time=args.process_time,
                                     reschedule_policy=_RP(args.policy),
                                     record_trajectory=True))
    result = sim.run()
    s = result.summary()
    meta = {"n_tasks": args.tasks, "n_agvs": args.agvs, "seed": args.seed,
            "planner": args.planner, "policy": args.policy,
            "via": "stream",
            "makespan": round(s["makespan_seconds"], 1),
            "completed": s["tasks_completed"]}
    frames = frames_from_result(result, dt=1.0, stride=args.stride)
    # 顺带落一份交换文件：这正是给 C# 适配层 / 实车监控端消费的东西
    json_out = os.path.splitext(args.out)[0] + ".stream.json"
    export_stream(result, json_out, dt=1.0, stride=args.stride,
                  network=net, meta=meta)
    print("已写出交换文件 %s" % json_out, flush=True)
    return payload_from_stream(frames, net, meta=meta,
                               agv_ids=[a.agv_id for a in sim.agvs])


def payload_from_stream(frames, net, *, meta=None, agv_ids=None):
    """把 ``amr_sched.stream`` 的帧序列转成回放模板要的 payload。

    为什么不各写一套
    ----------------
    回放模板（``replay_template.html``）提供完整的轨迹与路径可视化，
    蓄意重写一遍只会让两套渲染逻辑慢慢分叉。
    这里只做**格式适配**：stream 是语言无关的交换格式，
    payload 是模板的输入格式，二者字段几乎一一对应。

    一个重要差别：``StepRecord.node_holds``（预约表）**不在** stream 里，
    因为它是仿真器内部机理，实车端没有对应物。
    因此经 stream 走的回放**看不到预约表**——这是有意的取舍，
    要看预约表请用 :func:`build_payload` 直接导出仿真轨迹。
    """
    nodes = [{"id": n.node_id, "x": round(n.x, 3), "y": round(n.y, 3),
              "kind": n.kind.value, "cap": n.capacity}
             for n in net.nodes.values()]
    edges = []
    seen = set()
    for e in net.edges.values():
        key = tuple(sorted((e.src, e.dst)))
        if key in seen:
            continue
        seen.add(key)
        edges.append({"a": key[0], "b": key[1]})

    steps = []
    for fr in frames:
        pos = {}
        st_ = {}
        job = {}
        for a in fr.agvs:
            p_ = a.pose
            pos[a.agv_id] = p_.edge if p_.edge else (p_.node or "")
            st_[a.agv_id] = a.state
            job[a.agv_id] = a.task_id
        steps.append({
            "t": fr.step,
            "pos": pos,
            "st": st_,
            "job": job,
            "cnt": dict(fr.task_counts),
            "nb": fr.n_blocked,
            "hold": {},  # 见 docstring：stream 不携带预约表
            "pre": [[x["urgent"], x["victim_agv"], x["victim_task"],
                     x["was_picked"]] for x in fr.preemptions],
        })

    if agv_ids is None:
        agv_ids = sorted({a.agv_id for fr in frames for a in fr.agvs})

    return {
        "meta": meta or {},
        "nodes": nodes,
        "edges": edges,
        "agv_ids": agv_ids,
        "steps": steps,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="导出调度过程 HTML 回放（单文件，双击即看）")
    ap.add_argument("--tasks", type=int, default=50)
    ap.add_argument("--agvs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--planner", default="spacetime_astar",
                    choices=["spacetime_astar", "static_dijkstra"])
    ap.add_argument("--policy", default="none",
                    choices=["none", "preempt_unpicked", "preempt_any"])
    ap.add_argument("--process-time", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1,
                    help="每 N 步记一帧，任务多时可减小文件")
    ap.add_argument("--out", default="results/replay.html")
    ap.add_argument("--via-stream", action="store_true",
                    help="经 amr_sched.stream 统一接口生成（与 C#/实车共用同一格式）")
    args = ap.parse_args(argv)

    print("正在生成回放（%d 任务 / %d 车 / seed %d / %s / %s）..."
          % (args.tasks, args.agvs, args.seed, args.planner, args.policy),
          flush=True)
    if args.via_stream:
        payload = _payload_via_stream(args)
    else:
        payload = build_payload(args.tasks, args.agvs, args.seed, args.planner,
                                args.process_time, args.policy, args.stride)
    data_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    tpl_path = os.path.join(HERE, "replay_template.html")
    with open(tpl_path, encoding="utf-8") as f:
        tpl = f.read()
    html = tpl.replace("__DATA__", data_json)

    d = os.path.dirname(args.out)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(html)
    size = os.path.getsize(args.out) / 1024.0
    n_pre = payload["meta"].get("preemptions", "—")
    print("已写入 %s（%.0f KB，%d 帧，抢占 %s 次）"
          % (args.out, size, len(payload["steps"]), n_pre))
    print("直接用浏览器打开即可回放，无需服务器。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
