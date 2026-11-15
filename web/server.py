"""Multi-AGV scheduling — web console server.

Design constraints
------------------
1. **The scheduling core is not rewritten here.** This file calls the
   ``amr_sched`` Simulator directly and only handles drive, transport, and
   persistence. It contains no scheduling logic.
2. **Live, not playback.** Advancement goes through
   ``Simulator.step_once()`` one step at a time, pushing a frame to the
   WebSocket every N steps rather than sending everything after the run ends.
3. **Interruptible.** Urgent tasks can be inserted, the run paused, and the
   advance rate changed while it is running.

Endpoints
---------
GET  /                 console page
GET  /api/meta         available layouts / policies / planners
POST /api/run          create and start a run
POST /api/pause        pause / resume
POST /api/speed        change advance rate
POST /api/urgent       insert an urgent task mid-run
POST /api/stop         stop
GET  /api/runs         run history
GET  /api/runs/{id}    full record of one run
GET  /api/runs/{id}/export  export CSV
WS   /ws               live frame stream
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import math
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from amr_sched.allocation.baseline import (FifoShortestAllocator,
                                           NearestShortestAllocator)
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Agv, Task, TaskPriority, TaskStatus
from amr_sched.layouts import LAYOUTS, build_layout
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.planners.static_dijkstra import StaticDijkstraPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.scenarios import DueModel, ScenarioConfig, generate_scenario
from amr_sched.simulation import SimulationConfig, Simulator
from tools.live_bridge.gz_frame import DEMO_FRAME
from web.live_source import LiveRegistry

#: The one live bridge overlay. Empty until the operator attaches one -- the console has
#: always been able to draw its own Simulation; this is the only channel by which a pose
#: from outside this process reaches the canvas.
LIVE = LiveRegistry()


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "results" / "web_runs"
RUNS_DIR.mkdir(parents=True, exist_ok=True)
STATIC_DIR = Path(__file__).resolve().parent / "static"

ALLOCATORS = {
    "greedy": GreedyAllocator,
    "nearest": NearestShortestAllocator,
    "fifo": FifoShortestAllocator,
}
PLANNERS = {
    "static": StaticDijkstraPlanner,
    "spacetime": SpaceTimeAStarPlanner,
}
POLICIES = {
    "none": ReschedulePolicy.NONE,
    "preempt_unpicked": ReschedulePolicy.PREEMPT_UNPICKED,
    "preempt_any": ReschedulePolicy.PREEMPT_ANY,
}

@dataclass
class RunHandle:
    """Full state of one run. Owned by the server, not stored inside the Simulator."""

    run_id: str
    sim: Simulator
    cfg: dict[str, Any]
    created: str
    frames: list[dict[str, Any]] = field(default_factory=list)
    state: str = "running"          # running / paused / finished / stopped
    speed: float = 0.0              # steps/second; 0 means unlimited (finish as fast as possible)
    speed_credit: float = 0.0       # fractional step budget carried between iterations
    finished_step: int | None = None
    summary: dict[str, Any] | None = None
    #: 每步一条的轻量记录，专供回放。frames 是每 200 步一个快照（给实时推送用，
    #: 体积大），拿来回放会只剩十几帧、车一跳一跳；回放要的是逐帧连续。
    tape: list[dict[str, Any]] = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def now_frame(self) -> dict[str, Any]:
        """Return the current frame without advancing the simulation."""
        return self.frames[-1] if self.frames else {"step": 0}


RUNS: dict[str, RunHandle] = {}
SUBS: set[WebSocket] = set()
#: Whether to start the background advance task automatically. Disabled in tests so
#: _drive_sync has full control of the pace; otherwise every TestClient request
#: lets the background task creep one step and step-count assertions become flaky.
AUTO_DRIVE = True


def _yaw_deg(sim, transit) -> float | None:
    """Heading in degrees along the current edge, or None when standing on a node.

    Purely a display aid: the simulator carries no orientation state, so this is
    derived from the edge direction and must not be read as a measured heading.
    """
    if not transit:
        return None
    a = sim.network.nodes.get(transit[0])
    b = sim.network.nodes.get(transit[1])
    if a is None or b is None:
        return None
    dx, dy = b.x - a.x, b.y - a.y
    if dx == 0 and dy == 0:
        return None
    return round(math.degrees(math.atan2(dy, dx)), 1)


def _frame(h: RunHandle) -> dict[str, Any]:
    """Pack the simulator's current state into one frame for the front end to draw."""
    sim = h.sim
    step = sim._step_cursor
    agvs = []
    for a in sim.agvs:
        tr = sim._transit.get(a.agv_id)
        pos = "%s->%s" % (tr[0], tr[1]) if tr else (a.cur_node or "")
        # Continuous position along the current edge.
        # "pos" alone only says which edge the vehicle is on; the front end needs
        # "progress" (0..1) to place the marker between the two endpoints.
        # "xy" is computed here from the simulator node coordinates so the browser
        # never has to guess or carry a second geometry model.
        progress = 0.0
        xy = None
        if tr:
            src_n = sim.network.nodes.get(tr[0])
            dst_n = sim.network.nodes.get(tr[1])
            if src_n is not None and dst_n is not None:
                dep, arr = int(tr[2]), int(tr[3])
                span = max(1, arr - dep)
                progress = min(1.0, max(0.0, (step - dep) / span))
                xy = [round(src_n.x + (dst_n.x - src_n.x) * progress, 3),
                      round(src_n.y + (dst_n.y - src_n.y) * progress, 3)]
        elif a.cur_node:
            n = sim.network.nodes.get(a.cur_node)
            if n is not None:
                xy = [round(n.x, 3), round(n.y, 3)]
        job = sim._jobs.get(a.agv_id)
        rest: list[str] = []
        if job is not None and job.hops:
            rest = [x.dst for x in job.hops[job.idx:]]
        agvs.append({
            "id": a.agv_id,
            "node": a.cur_node,
            "pos": pos,
            "progress": round(progress, 4),
            "xy": xy,
            "yaw": _yaw_deg(sim, tr),
            "source": "sim",
            "t": step,
            "state": a.state.value,
            "task": job.task_id if job is not None else None,
            "path": rest,
            "busy": a.busy_steps,
            "blocked": a.blocked_steps,
        })
    counts = {"pending": 0, "assigned": 0, "picked": 0, "completed": 0,
              "failed": 0}
    for t in sim.tasks:
        k = t.status.value
        if k in counts:
            counts[k] += 1
    return {
        "run_id": h.run_id,
        "step": step,
        "state": h.state,
        "speed": h.speed,
        "agvs": agvs,
        "counts": counts,
        "n_tasks": len(sim.tasks),
        "n_agvs": len(sim.agvs),
        "deadlock_breaks": sim._deadlock_breaks,
        "preemptions": sim._preemptions,
        "finished_step": h.finished_step,
        "summary": h.summary,
    }


def _nodes_of(net) -> dict[str, Any]:
    """Nodes and edges for drawing a network (sent once, never changes)."""
    out = []
    for n in net.nodes.values():
        out.append({"id": n.node_id, "kind": n.kind.value,
                    "x": round(n.x, 3), "y": round(n.y, 3),
                    "capacity": n.capacity})
    edges = []
    for u in net.nodes:
        for e in net.out_edges(u):
            edges.append({"src": e.src, "dst": e.dst,
                          "length": round(e.length, 3),
                          "capacity": e.capacity})
    return {"nodes": out, "edges": edges}


def _nodes(h: RunHandle) -> list[dict[str, Any]]:
    """The map payload for a run, kept as a one-element list for the existing callers."""
    return [_nodes_of(h.sim.network)]

def _broadcast(frame: dict[str, Any]) -> None:
    """Broadcast one frame to all subscribers. Disconnected clients do not block the rest."""
    dead = []
    for ws in list(SUBS):
        try:
            asyncio.create_task(ws.send_json(frame))
        except Exception:
            dead.append(ws)
    for ws in dead:
        SUBS.discard(ws)


async def _driver(h: RunHandle) -> None:
    """Background advance entry point: push frame by frame at the given speed,
    broadcasting as it goes.

    Under TestClient, asyncio.create_task is not driven by the event loop, so a
    synchronous ``_drive_sync()`` is provided that reuses the same advance logic
    for tests to call directly.
    """
    sim = h.sim
    h._started = True
    sim.reset()
    last = time.monotonic()
    try:
        while not sim.is_finished():
            if h.state == "stopped":
                break
            if h.state == "paused":
                await asyncio.sleep(0.05)
                last = time.monotonic()
                continue
            # speed = simulation steps per second; 0 means unlimited
            if h.speed and h.speed > 0:
                # Accumulate the fractional step budget across iterations.
                # Clamping to at least one step per iteration (the previous
                # behaviour) silently enforced a floor of ~20 steps/s at a 50 ms
                # loop period, so any requested speed below that ran too fast.
                # Carrying the remainder makes slow speeds actually slow.
                h.speed_credit += h.speed * (time.monotonic() - last)
                budget = int(h.speed_credit)
                h.speed_credit -= budget
                budget = min(budget, 2000)
            else:
                budget = 200
            advanced = _advance_block(h, budget)
            if advanced:
                _broadcast(h.frames[-1])
            last = time.monotonic()
            if h.speed and h.speed > 0:
                await asyncio.sleep(max(0.01, 0.05))
            else:
                await asyncio.sleep(0)
        if h.state != "stopped":
            h.state = "finished"
            h.finished_step = sim._step_cursor
            h.summary = sim._build_result().summary()
            _persist(h)
            _broadcast(_frame(h))
    except Exception as exc:  # pragma: no cover - defensive
        h.state = "error"
        h.summary = {"error": str(exc)}
        _broadcast(_frame(h))


def _advance_block(h: RunHandle, budget: int) -> int:
    """Advance N steps and record one frame, shared by the async and sync drivers.

    reset() is called automatically on the first invocation so callers need not
    remember to initialize first; otherwise session state such as _load_end does
    not exist yet and blows up inside _try_assign.
    """
    sim = h.sim
    if not getattr(h, "_started", False):
        sim.reset()
        h._started = True
    advanced = 0
    for _ in range(budget):
        if not sim.step_once():
            break
        advanced += 1
        # 回放要逐帧，所以 tape 在**步循环里**记，不是每批记一条：一次 _advance_block
        # 可能推进 200 步，按批记就把 200 步压成一个点，回放会一跳一跳。
        h.tape.append(_tape_entry(h))
        if len(h.tape) > 40_000:
            del h.tape[:10_000]
    if advanced:
        h.frames.append(_frame(h))
        if len(h.frames) > 4000:
            del h.frames[:1000]
    return advanced


def _tape_entry(h: RunHandle) -> dict[str, Any]:
    """One replay record per step: where each vehicle was and what it was doing.

    Deliberately holds only what the map draws. A full frame carries the task list,
    counters and route geometry, which is ~40x larger and would make a 10-minute run
    too big to keep in memory or serve over HTTP.
    """
    sim = h.sim
    step = sim._step_cursor
    items = []
    for a in sim.agvs:
        tr = sim._transit.get(a.agv_id)
        xy = _xy_of(sim, a, tr, step)
        items.append({
            "i": a.agv_id,
            "s": a.state.value,
            "x": None if xy is None else xy[0],
            "y": None if xy is None else xy[1],
            "yaw": _yaw_deg(sim, tr),
            "n": a.cur_node,
        })
    return {"t": step, "a": items}


def _xy_of(sim, agv, transit, step: int) -> list[float] | None:
    """Interpolated position along the current edge, or the node it is standing on."""
    if transit:
        src = sim.network.nodes.get(transit[0])
        dst = sim.network.nodes.get(transit[1])
        if src is not None and dst is not None:
            dep, arr = int(transit[2]), int(transit[3])
            span = max(1, arr - dep)
            p = min(1.0, max(0.0, (step - dep) / span))
            return [round(src.x + (dst.x - src.x) * p, 3),
                    round(src.y + (dst.y - src.y) * p, 3)]
    if agv.cur_node:
        n = sim.network.nodes.get(agv.cur_node)
        if n is not None:
            return [round(n.x, 3), round(n.y, 3)]
    return None


def _drive_sync(h: RunHandle, block: int = 200) -> None:
    """Synchronously drive a run to its final state. For tests and batch use."""
    h.sim.reset()
    h._started = True
    guard = 0
    while not h.sim.is_finished():
        if h.state in ("stopped", "paused"):
            return
        _advance_block(h, block)
        guard += 1
        if guard > 100_000:
            raise RuntimeError("synchronous driver did not terminate")
    h.state = "finished"
    h.finished_step = h.sim._step_cursor
    h.summary = h.sim._build_result().summary()
    _persist(h)


def _persist(h: RunHandle) -> None:
    """Persist one complete run to results/web_runs/<id>/."""
    d = RUNS_DIR / h.run_id
    d.mkdir(parents=True, exist_ok=True)
    (d / "config.json").write_text(
        json.dumps(h.cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    (d / "summary.json").write_text(
        json.dumps(h.summary or {}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (d / "frames.json").write_text(
        json.dumps(h.frames[-2000:], ensure_ascii=False), encoding="utf-8")
    (d / "tape.json").write_text(
        json.dumps(h.tape, ensure_ascii=False), encoding="utf-8")
    sim = h.sim
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["task_id", "pickup", "dropoff", "priority", "status",
                "assigned_agv", "release_step", "due_step",
                "pickup_step", "complete_step"])
    for t in sim.tasks:
        w.writerow([t.task_id, t.pickup, t.dropoff, t.priority.value,
                    t.status.value, t.assigned_agv or "", t.release_step,
                    t.due_step if t.due_step is not None else "",
                    t.pickup_step if t.pickup_step is not None else "",
                    t.complete_step if t.complete_step is not None else ""])
    (d / "tasks.csv").write_text(buf.getvalue(), encoding="utf-8-sig")
    buf2 = io.StringIO()
    w2 = csv.writer(buf2)
    w2.writerow(["agv_id", "start_node", "busy_steps", "idle_steps",
                 "blocked_steps", "loaded_steps", "deadhead_steps"])
    for a in sim.agvs:
        w2.writerow([a.agv_id, a.start_node, a.busy_steps, a.idle_steps,
                     a.blocked_steps, a.loaded_steps, a.deadhead_steps])
    (d / "agvs.csv").write_text(buf2.getvalue(), encoding="utf-8-sig")

app = FastAPI(title="Multi-AGV Scheduling Console")


class RunRequest(BaseModel):
    layout: str = Field(default="ring")
    n_tasks: int = Field(default=50, ge=1, le=300)
    n_agvs: int = Field(default=4, ge=1, le=12)
    seed: int = Field(default=0, ge=0, le=99999)
    aisle_capacity: int = Field(default=2, ge=1, le=4)
    urgent_ratio: float = Field(default=0.15, ge=0.0, le=1.0)
    policy: str = Field(default="preempt_unpicked")
    planner: str = Field(default="static")
    allocator: str = Field(default="greedy")
    max_steps: int = Field(default=120_000, ge=100, le=400_000)
    speed: float = Field(default=0.0, ge=0.0, le=50_000.0)


class UrgentRequest(BaseModel):
    pickup: str
    dropoff: str
    due_slack: int = Field(default=300, ge=1, le=100_000)


class SpeedRequest(BaseModel):
    speed: float = Field(ge=0.0, le=50_000.0)


@app.get("/api/meta")
def meta() -> dict[str, Any]:
    return {
        "layouts": sorted(LAYOUTS),
        "planners": sorted(PLANNERS),
        "allocators": sorted(ALLOCATORS),
        "policies": sorted(POLICIES),
        "defaults": RunRequest().model_dump(),
    }


@app.get("/api/map")
def map_only(layout: str = "ring") -> dict[str, Any]:
    """The workshop network on its own, with no run behind it.

    Why this exists: the live bridge overlay is useful *without* starting a scheduling
    run -- watching the vehicle in Gazebo move across the workshop is its own check. But
    the canvas has always taken its geometry from /api/run, so before this endpoint the
    only way to see the map was to launch a job that had nothing to do with the question.
    """
    if layout not in LAYOUTS:
        raise HTTPException(400, "Unknown layout: %s" % layout)
    return _nodes_of(build_layout(layout))


@app.post("/api/run")
async def start_run(req: RunRequest) -> dict[str, Any]:
    if req.layout not in LAYOUTS:
        raise HTTPException(400, "Unknown layout: %s" % req.layout)
    if req.planner not in PLANNERS:
        raise HTTPException(400, "Unknown planner: %s" % req.planner)
    if req.allocator not in ALLOCATORS:
        raise HTTPException(400, "Unknown allocator: %s" % req.allocator)
    if req.policy not in POLICIES:
        raise HTTPException(400, "Unknown policy: %s" % req.policy)

    # The ring layout is the main study instance; grid/hub are used for portability checks
    if req.layout == "workshop":
        # The workshop has real process semantics: a task is a material moving along a
        # material route, from a station/stock node to a handover node. Remapping tasks
        # onto arbitrary nodes (what the grid/hub branch below does) would produce
        # deliveries no process would ever order, so it is not allowed here.
        from amr_sched.workshop import build_scenario

        net, tasks, agvs = build_scenario(
            n_tasks=req.n_tasks,
            n_agvs=req.n_agvs,
            seed=req.seed,
            urgent_ratio=req.urgent_ratio,
        )
    elif req.layout == "ring":
        sc = ScenarioConfig(
            n_tasks=req.n_tasks, n_agvs=req.n_agvs, seed=req.seed,
            aisle_capacity=req.aisle_capacity,
            urgent_ratio=req.urgent_ratio)
        net, tasks, agvs = generate_scenario(sc)
    else:
        net = build_layout(req.layout, aisle_capacity=req.aisle_capacity)
        sc = ScenarioConfig(
            n_tasks=req.n_tasks, n_agvs=req.n_agvs, seed=req.seed,
            aisle_capacity=req.aisle_capacity,
            urgent_ratio=req.urgent_ratio)
        _n, tasks, agvs = generate_scenario(sc)
        # Remap tasks and vehicles onto the nodes of this layout
        nodes = [n.node_id for n in net.nodes.values()]
        for i, t in enumerate(tasks):
            t.pickup = nodes[i % len(nodes)]
            t.dropoff = nodes[(i * 7 + 3) % len(nodes)]
        for i, a in enumerate(agvs):
            a.start_node = nodes[i % len(nodes)]

    planner = PLANNERS[req.planner]()
    sim = Simulator(
        net, tasks, agvs, planner, ALLOCATORS[req.allocator](),
        SimulationConfig(
            max_steps=req.max_steps, seed=req.seed,
            record_trajectory=False,
            reschedule_policy=POLICIES[req.policy]),
    )
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "-" + uuid.uuid4().hex[:6]
    h = RunHandle(run_id=run_id, sim=sim, cfg=req.model_dump(),
                  created=datetime.now(timezone.utc).isoformat(),
                  speed=req.speed)
    RUNS[run_id] = h
    if AUTO_DRIVE:
        asyncio.create_task(_driver(h))
    return {"run_id": run_id, "map": _nodes(h)[0],
            "frame": _frame(h)}

@app.post("/api/pause")
async def pause(run_id: str) -> dict[str, Any]:
    h = RUNS.get(run_id)
    if h is None:
        raise HTTPException(404, "No such run")
    h.state = "running" if h.state == "paused" else "paused"
    return {"run_id": run_id, "state": h.state}


@app.post("/api/speed")
async def speed(run_id: str, req: SpeedRequest) -> dict[str, Any]:
    h = RUNS.get(run_id)
    if h is None:
        raise HTTPException(404, "No such run")
    h.speed = req.speed
    return {"run_id": run_id, "speed": h.speed}


@app.post("/api/stop")
async def stop(run_id: str) -> dict[str, Any]:
    h = RUNS.get(run_id)
    if h is None:
        raise HTTPException(404, "No such run")
    h.state = "stopped"
    h.finished_step = h.sim._step_cursor
    h.summary = h.sim._build_result().summary()
    _persist(h)
    return {"run_id": run_id, "state": h.state, "step": h.finished_step}


@app.post("/api/urgent")
async def insert_urgent(run_id: str, req: UrgentRequest) -> dict[str, Any]:
    """Insert one urgent task during a run.

    This is the most direct demonstration of dynamic rescheduling: a task appears
    from nowhere at step N and, if the policy allows preemption, a vehicle that is
    already working gets interrupted.
    """
    h = RUNS.get(run_id)
    if h is None:
        raise HTTPException(404, "No such run")
    sim = h.sim
    now = sim._step_cursor
    tid = "U%04d" % (len([t for t in sim.tasks if t.priority is TaskPriority.URGENT]) + 1)
    task = Task(
        task_id=tid, pickup=req.pickup, dropoff=req.dropoff,
        release_step=now, due_step=now + req.due_slack,
        priority=TaskPriority.URGENT)
    sim.tasks.append(task)
    sim._task_by_id[tid] = task
    _broadcast(_frame(h))
    return {"run_id": run_id, "task_id": tid, "step": now,
            "due_step": task.due_step}



class OrderRequest(BaseModel):
    station: str = Field(description="被点击的工位节点 id")


#: 工艺允许的取送对，只读一次。每次点击都去读一遍场景文件没有必要。
_LEGAL_PAIRS: list[tuple[str, str]] | None = None


def _legal_pairs() -> list[tuple[str, str]]:
    global _LEGAL_PAIRS
    if _LEGAL_PAIRS is None:
        from amr_sched.workshop import legal_task_pairs
        _LEGAL_PAIRS = legal_task_pairs()
    return _LEGAL_PAIRS


@app.post("/api/order")
async def place_order(run_id: str, req: OrderRequest) -> dict[str, Any]:
    """点击工位下单：把点到的节点当配送目标，取货点按工艺允许的路线自动配对。

    为什么不是"点一下就从料库送到这里"：车间里每种物料允许的取送对是工艺决定的，
    随手编一个取货点可能造出一条现实中不存在的运输路线。所以只在**工艺允许的对**
    里挑，并把挑中的那一对原样回给调用方——点了 W_BODY 却从别处取货，界面上必须
    看得见，不能靠猜。
    """
    h = RUNS.get(run_id)
    if h is None:
        raise HTTPException(404, "No such run")
    sim = h.sim
    station = (req.station or "").strip()
    if not station:
        raise HTTPException(400, "station 不能为空")
    if station not in getattr(sim.network, "nodes", {}):
        raise HTTPException(400, "网络里没有这个节点: %s" % station)

    pairs = [p for p in _legal_pairs()
             if p[0] in sim.network.nodes and p[1] in sim.network.nodes]
    if not pairs:
        raise HTTPException(400, "当前场景没有工艺路线，不能按工位下单")

    as_dropoff = sorted({p[0] for p in pairs if p[1] == station})
    as_pickup = sorted({p[1] for p in pairs if p[0] == station})
    clicked_role = ""
    if as_dropoff:
        pickup, dropoff, clicked_role = as_dropoff[0], station, "dropoff"
    elif as_pickup:
        pickup, dropoff, clicked_role = station, as_pickup[0], "pickup"
    else:
        raise HTTPException(
            400, "该节点不在任何物料路线上，不能下单: %s" % station)

    now = sim._step_cursor
    tid = "O%04d" % (len([t for t in sim.tasks if t.task_id.startswith("O")]) + 1)
    task = Task(
        task_id=tid, pickup=pickup, dropoff=dropoff,
        release_step=now, due_step=now + 600,
        priority=TaskPriority.NORMAL)
    sim.tasks.append(task)
    sim._task_by_id[tid] = task
    _broadcast(_frame(h))
    return {"run_id": run_id, "task_id": tid, "step": now,
            "pickup": pickup, "dropoff": dropoff,
            "clicked": station,
            "clicked_role": clicked_role,
            "network_nodes": len(sim.network.nodes)}

class LiveAttachRequest(BaseModel):
    host: str = "127.0.0.1"
    port: int = 9300


@app.post("/api/live/attach")
def live_attach(req: LiveAttachRequest) -> dict[str, Any]:
    """Attach a read-only bridge connection and start drawing it.

    Deliberately refuses to paper over a dead socket: if the bridge is not listening the
    connect raises and the caller gets a 502 with the reason, rather than an attached-but-
    empty overlay that looks like a vehicle which never moved.
    """
    try:
        link = LIVE.attach(req.host, req.port, frame=DEMO_FRAME)
    except OSError as exc:
        raise HTTPException(502, "cannot reach %s:%d -- %s" % (req.host, req.port, exc))
    return {"attached": True, "host": link.host, "port": link.port,
            "frame_in_use": "demo", "live": LIVE.snapshot()}


@app.delete("/api/live")
def live_detach() -> dict[str, Any]:
    LIVE.detach()
    return {"attached": False}


@app.get("/api/live")
def live_state() -> dict[str, Any]:
    snap = LIVE.snapshot()
    return {"attached": snap is not None, "live": snap}


@app.get("/api/runs")
def list_runs() -> list[dict[str, Any]]:
    """Run history: in-memory runs plus those already persisted to disk."""
    out = []
    seen = set()
    for rid, h in RUNS.items():
        seen.add(rid)
        out.append({"run_id": rid, "config": h.cfg, "state": h.state,
                    "step": h.finished_step, "summary": h.summary,
                    "created": h.created})
    for d in sorted(RUNS_DIR.iterdir() if RUNS_DIR.exists() else []):
        if not d.is_dir() or d.name in seen:
            continue
        try:
            cfg = json.loads((d / "config.json").read_text(encoding="utf-8"))
            sm = json.loads((d / "summary.json").read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({"run_id": d.name, "config": cfg, "state": "finished",
                    "step": sm.get("makespan_seconds"), "summary": sm,
                    "created": d.name[:15]})
    out.sort(key=lambda r: r["run_id"], reverse=True)
    return out


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    h = RUNS.get(run_id)
    if h is not None:
        return {"run_id": run_id, "config": h.cfg, "summary": h.summary,
                "frames": h.frames[-400:], "state": h.state}
    d = RUNS_DIR / run_id
    if not d.exists():
        raise HTTPException(404, "No such run")
    return {
        "run_id": run_id,
        "config": json.loads((d / "config.json").read_text(encoding="utf-8")),
        "summary": json.loads((d / "summary.json").read_text(encoding="utf-8")),
        "frames": json.loads((d / "frames.json").read_text(encoding="utf-8")),
        "state": "finished",
    }


@app.get("/api/runs/{run_id}/replay")
def replay_run(run_id: str, start: int = 0, limit: int = 2000) -> dict[str, Any]:
    """回放：按步号取一段逐帧位置记录。

    分页而不是整包返回：一轮 40 分钟的运行有一万多个 step，一次全发会把浏览器
    卡住，而且回放本来就是"一段一段看"。`total` 让前端知道还有多少。
    """
    h = RUNS.get(run_id)
    if h is not None:
        tape = h.tape
        meta = {"config": h.cfg, "state": h.state, "summary": h.summary}
    else:
        d = RUNS_DIR / run_id
        if not d.exists():
            raise HTTPException(404, "No such run")
        f = d / "tape.json"
        if not f.exists():
            raise HTTPException(
                404, "这一轮没有回放记录（tape.json 不存在）。回放记录是后加的，"
                     "早于它的运行只能看 summary 和 csv 导出。")
        tape = json.loads(f.read_text(encoding="utf-8"))
        meta = {"config": json.loads((d / "config.json").read_text(encoding="utf-8")),
                "state": "finished",
                "summary": json.loads((d / "summary.json").read_text(encoding="utf-8"))}
    start = max(0, start)
    limit = max(1, min(limit, 5000))
    return {"run_id": run_id, "total": len(tape), "start": start,
            "frames": tape[start:start + limit], **meta}


@app.get("/api/runs/{run_id}/export", response_class=PlainTextResponse)
def export_run(run_id: str, kind: str = "tasks") -> PlainTextResponse:
    d = RUNS_DIR / run_id
    f = d / ("%s.csv" % kind)
    if not f.exists():
        raise HTTPException(404, "No such export file")
    return PlainTextResponse(f.read_text(encoding="utf-8-sig"),
                             media_type="text/csv; charset=utf-8")


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket) -> None:
    await ws.accept()
    SUBS.add(ws)
    try:
        while True:
            await ws.receive_text()   # heartbeat; the client may stay silent
    except WebSocketDisconnect:
        pass
    finally:
        SUBS.discard(ws)


if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def index() -> Any:
    idx = STATIC_DIR / "index.html"
    if idx.exists():
        return FileResponse(str(idx))
    return JSONResponse({"error": "missing static/index.html"}, status_code=500)


def _self_check(port: int) -> int:
    """Boot the real server, run one job over real HTTP, and report.

    Why this exists
    ---------------
    ``Tests/test_web_console.py`` uses FastAPI's TestClient, which never drives the
    asyncio task that ``start_run`` creates. Those tests therefore exercise the state
    machine but **not** the path a browser actually takes: ``create_task(_driver)``.
    If the background driver stopped advancing, every one of them would still pass --
    the console would simply sit at step 0 in front of the user.

    So this mode runs the thing the browser runs: a real uvicorn server on a real
    socket, a real POST, a real background task, and nobody calling ``_drive_sync``.
    """
    import threading
    import urllib.request

    import uvicorn

    global AUTO_DRIVE
    AUTO_DRIVE = True
    RUNS.clear()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    deadline = time.monotonic() + 30
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.05)
    if not server.started:
        print("SELF-CHECK FAIL: server did not start on port %d" % port)
        return 1

    base_url = "http://127.0.0.1:%d" % port
    body = json.dumps({"layout": "ring", "n_tasks": 4, "n_agvs": 2, "seed": 1,
                       "speed": 0, "max_steps": 60000}).encode("utf-8")
    request = urllib.request.Request(base_url + "/api/run", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=30) as fh:
        run_id = json.loads(fh.read())["run_id"]
    print("SELF-CHECK run_id=%s" % run_id)

    started = time.monotonic()
    record: dict[str, Any] = {}
    while time.monotonic() - started < 120:
        with urllib.request.urlopen(base_url + "/api/runs/" + run_id,
                                    timeout=30) as fh:
            record = json.loads(fh.read())
        if record.get("state") in ("finished", "error", "stopped"):
            break
        time.sleep(0.2)

    elapsed = time.monotonic() - started
    server.should_exit = True
    thread.join(timeout=10)

    summary = record.get("summary") or {}
    state = record.get("state")
    completed = summary.get("tasks_completed")
    print("SELF-CHECK state=%s completed=%s elapsed=%.1fs frames=%d"
          % (state, completed, elapsed, len(record.get("frames") or [])))
    ok = state == "finished" and completed == 4
    print("SELF-CHECK %s" % ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Multi-AGV scheduling console")
    parser.add_argument("--self-check", action="store_true",
                        help="start a real server, drive one job over HTTP, exit")
    parser.add_argument("--port", type=int,
                        default=int(os.environ.get("AMR_WEB_PORT", "8077")))
    args = parser.parse_args()

    if args.self_check:
        raise SystemExit(_self_check(args.port))

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="info")


if __name__ == "__main__":
    main()