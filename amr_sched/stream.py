"""The data interface: turns the scheduler's output into a format both the visualiser
and a real vehicle can consume.

Why a separate layer
--------------------
Before this existed, the Python scheduling core (``amr_sched``) produced Python objects
and the C# RAWSim-O adapter read its own data model,

with **no interface** between the two.

The result was two separate software artefacts: the simulation ran and the interface
animated, but they had nothing to do with each other.


This layer defines a **stable, language-neutral** exchange format:


    Simulator ──► FrameStream ──► JSON ──► visualiser / RAWSim-O / vehicle monitor

Three design constraints
--------------
**1. A pose must carry its provenance.**
Simulation truth, integrated odometry, and map localisation are three kinds of data with
entirely different accuracy,

and mixing them makes "does the real vehicle agree with the simulation" impossible to
discuss.
:class:`Pose` therefore always carries a :class:`PoseSource`, defaulting to ``SIM``;
the real-vehicle side must declare ``ODOM`` or ``LOCALIZED`` explicitly.

**2. A frame is self-contained.**
Any single frame can be read without its neighbours: who is where, what they are doing,
what state each task is in.
A monitor can then join at any moment, skip frames, or replay without starting from the
beginning.

**3. The format is versioned.**
``SCHEMA_VERSION`` is exported so a consumer can refuse an incompatible stream.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Iterable

from .simulation import SimulationResult, StepRecord

#: Exchange format version. Any incompatible change must bump it, and consumers should
#: refuse a mismatch.
SCHEMA_VERSION = "1.0"


class PoseSource(str, Enum):
    """Pose provenance. **Never mix these**, and in particular never present ODOM as an
    absolute position."""

    #: The simulator's own truth: no noise, no drift. Used for algorithm comparison and
    #: experiment statistics.
    SIM = "sim"
    #: Pose estimated by integrating wheel speed and IMU. It **drifts**, so it is fit only
    #: for drawing a track.
    ODOM = "odom"
    #: Pose after map matching or loop closure. Used to assess real-vehicle localisation
    #: accuracy.
    LOCALIZED = "localized"
    #: Ground truth reported by Gazebo's official get_entity_state service (NOT /odom, which
    #: drifts and is a different physical quantity). This is the source a 三维验证模式 frame
    #: must declare, so a trail drawn on the console is never mistaken for a simulation pose.
    GAZEBO = "gazebo"


@dataclass
class Pose:
    """Where one vehicle is at one moment.

At least one of ``node`` and ``edge`` is set:

- on a node: ``node`` holds its id;
- in transit: ``edge`` is ``"A->B"`` and ``progress`` is the fraction travelled, 0..1.
    """

    agv_id: str
    node: str | None = None
    edge: str | None = None
    progress: float = 0.0
    source: PoseSource = PoseSource.SIM
    #: Filled with a CAN timestamp on a real vehicle; left empty in simulation.
    timestamp: float | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["source"] = self.source.value
        return d


@dataclass
class AgvFrame:
    """One vehicle's complete state within a frame."""

    agv_id: str
    pose: Pose
    #: idle / to_pickup / loading / to_dropoff / unloading / blocked / parking
    state: str
    task_id: str | None = None
    #: Remaining node sequence for this leg, including the goal. Used to draw the path.
    #: An empty list when unavailable -- do **not** infer from that that the vehicle is
    #: standing still.
    path: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to the per-vehicle mapping inside a frame.

        Serializes vehicle state consistently across
        ``Frame.to_dict()`` only, and reusing it from the live channel
        (``FrameBuilder``) is what revealed that ``AgvFrame`` had no matching method.
        Adding it and making ``Frame.to_dict()`` call it keeps the offline export and
        the live push **field-for-field identical**.
        """
        return {
            "agv_id": self.agv_id,
            "pose": self.pose.to_dict(),
            "state": self.state,
            "task_id": self.task_id,
            "path": list(self.path),
        }


@dataclass
class Frame:
    """A complete, **self-contained** snapshot of one time step."""

    step: int
    t: float
    agvs: list[AgvFrame] = field(default_factory=list)
    #: Task counts by state: pending / assigned / picked / completed
    task_counts: dict[str, int] = field(default_factory=dict)
    #: How many vehicles are blocked this step
    n_blocked: int = 0
    #: Preemption events this step: [{urgent, victim_agv, victim_task, was_picked}]
    preemptions: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": "frame",
            "schema": SCHEMA_VERSION,
            "step": self.step,
            "t": round(self.t, 3),
            "agvs": [a.to_dict() for a in self.agvs],
            "task_counts": dict(self.task_counts),
            "n_blocked": self.n_blocked,
            "preemptions": list(self.preemptions),
        }


@dataclass
class NetworkSpec:
    """Network topology, so a consumer can draw the base map without rebuilding it."""

    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_network(cls, net: Any) -> "NetworkSpec":
        nodes = [
            {
                "id": n.node_id,
                "kind": n.kind.value,
                "capacity": n.capacity,
                "x": getattr(n, "x", None),
                "y": getattr(n, "y", None),
            }
            for n in net.nodes.values()
        ]
        # ``Network.edges`` returns a dict keyed by (src, dst) whose values are
        # Edge objects -- iterating the dict alone yields only the keys, which is
        # how the edge length used to end up as a constant None on the wire.
        # Read the values instead and publish the length under both names so early
        # consumers (``u``/``v``/``w``) and the C# side (``src``/``dst``/``length``)
        # keep working.
        edges = []
        for (u, v), edge in net.edges.items():
            edges.append({
                "u": u, "v": v, "w": edge.length,
                "src": u, "dst": v, "length": edge.length,
                "capacity": edge.capacity,
                "centerline": [list(point) for point in net.edge_centerline(u, v)],
            })
        return cls(nodes=nodes, edges=edges)


def _split_edge(pos: str, progress: float = 0.0) -> tuple[str | None, str | None, float]:
    """Split an "A->B" or "A" position into (node, edge, progress).

    ``progress`` must be supplied by the caller: the position string alone only
    says *which* edge the vehicle is on, never how far along it. It used to
    default to 0.0 here, which made every in-transit pose look like the vehicle
    was sitting exactly on the source node.
    """
    if "->" in pos:
        u, v = pos.split("->", 1)
        return None, f"{u}->{v}", float(progress)
    return (pos or None), None, 0.0


def frames_from_result(
    result: SimulationResult,
    *,
    dt: float = 1.0,
    stride: int = 1,
    source: PoseSource = PoseSource.SIM,
) -> list[Frame]:
    """Turn a :class:`SimulationResult` trajectory into a sequence of :class:`Frame`.

    Args:
        stride: frame interval. 1 means one frame per step, 10 means one per ten steps.
            Replay usually uses 5..20, trading file size against smoothness.
        source: pose provenance. Simulation results should keep the ``SIM`` default.
    """
    frames: list[Frame] = [];
    for rec in result.trajectory:
        if rec.step % stride:
            continue;
        frames.append(_frame_from_record(rec, dt=dt, source=source))
    return frames


def _frame_from_record(
    rec: StepRecord, *, dt: float, source: PoseSource
) -> Frame:
    agvs: list[AgvFrame] = [];
    for agv_id, pos in rec.agv_positions.items():
        node, edge, progress = _split_edge(pos)
        agvs.append(
            AgvFrame(
                agv_id=agv_id,
                pose=Pose(
                    agv_id=agv_id, node=node, edge=edge,
                    progress=progress, source=source,
                ),
                state=rec.agv_states.get(agv_id, "idle"),
                task_id=rec.agv_tasks.get(agv_id),
                path=list(rec.agv_paths.get(agv_id, [])),
            )
        )
    pre = [
        {"urgent": p[0], "victim_agv": p[1],
         "victim_task": p[2], "was_picked": bool(p[3])}
        for p in rec.preemptions
    ]
    return Frame(
        step=rec.step,
        t=rec.step * dt,
        agvs=agvs,
        task_counts=dict(rec.task_counts),
        n_blocked=rec.n_blocked,
        preemptions=pre,
    )


def export_stream(
    result: SimulationResult,
    path: str,
    *,
    dt: float = 1.0,
    stride: int = 1,
    network: Any | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Export to a single JSON file for the visualiser, the RAWSim-O adapter, or a
    real-vehicle monitor.

    File structure::

        {
          "schema": "1.0",
          "meta": {...},          // 场景与实验标识
          "network": {...},       // 拓扑，画底图用
          "frames": [...]         // 逐帧状态
        }

Returns a summary (frame count, byte count, and so on) so the caller can record it.
    """
    frames = frames_from_result(result, dt=dt, stride=stride)
    if not frames:
        raise ValueError(
            "there are no frames at all. The most common cause is "
            "record_trajectory=False in SimulationConfig, the memory-saving default."
            "Turn it on explicitly before exporting: "
            "SimulationConfig(record_trajectory=True)."
        )
    payload: dict[str, Any] = {
        "schema": SCHEMA_VERSION,
        "meta": {
            "dt": dt,
            "stride": stride,
            "n_frames": len(frames),
            "finished_step": result.finished_step,
            **(meta or {}),
        },
        "network": (
            asdict(NetworkSpec.from_network(network)) if network is not None else None
        ),
        "frames": [f.to_dict() for f in frames],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    import os
    return {
        "path": path,
        "n_frames": len(frames),
        "bytes": os.path.getsize(path),
        "schema": SCHEMA_VERSION,
    }


# ---------------------------------------------------------------------------
# Live NDJSON TCP channel
# ---------------------------------------------------------------------------
#
# Before this the C# side could only call `StreamDocument.Load(path)` on a finished JSON
# file, that is, read results offline. Live trajectory monitoring requires **streaming frames while the simulation is running**.
#
# The exchange format is unchanged (schema is still 1.0); the same Frame is simply pushed
# frame by frame. The transport is **newline-delimited JSON over TCP**:
#   * the C# side needs no WebSocket library, just one StreamReader.ReadLine per line;
#   * it can be inspected by eye with `nc` or `telnet`;
#   * reconnection realigns naturally: discard the partial line and resynchronise on the
#
# next one. Contract: once a connection is up the server sends one `hello` line
# (schema, network, meta), then one `frame` line per frame. A client only parses lines;
# there is **no handshake**.


class FrameBuilder:
    """Build Frames straight from a **running Simulator**, without a SimulationResult.

    Why not call ``frames_from_result`` afterwards: that needs `record_trajectory`
    to accumulate the whole trajectory in memory, which in a live setting both wastes
    memory and still cannot give you *now*. Reading the simulator's current state gives
    the actual current frame.
    """

    #: How many past positions per vehicle the actual-trail keeps. Long enough to see a
    #: lap of the workshop, short enough that a 20 Hz stream does not grow without bound.
    TRAIL_POINTS = 300

    def __init__(self, sim: Any, source: PoseSource = PoseSource.SIM) -> None:
        self.sim = sim
        gates = getattr(sim, 'executions', {})
        self.source = PoseSource(next(iter(gates.values())).identity['source']) if gates else source
        self.network = sim.network
        #: Where each vehicle actually was, in scene metres, oldest first. Built here and
        #: not in the display layer: a trail is a claim about the past, and only the side
        #: that owns the poses may make it.
        self._trails: dict[str, list[list[float]]] = {}
        #: 三维验证模式里那台**物理车**的最近位姿与轨迹（场景米）。
        #:
        #: 它是独立实体，不是把某台仿真车的位姿"换成"真值。覆盖的做法会让同一辆车同时
        #: 存在两个位置——引擎以为它在这、Gazebo 说它在那——而目标明令禁止两处独立计算
        #: 同一辆车的位置。物理车单独画，比较的是"计划"与"实走"，不是两张嘴各说一个坐标。
        #: 节点 -> 它从哪一步开始一直满员。拥堵时长由此算出：一次堵 1 步和堵 30 步
        #: 在画面上本来是同一个圈，看不出严重程度差别。
        self._congested_since: dict[str, int] = {}
        self._last_congestion_step: int = -1
        self.physical_trail: list[list[float]] = []
        self.physical_pose: tuple[float, float, float] | None = None
        self.physical_id: str = "scout_mini"

    def _xy(self, agv: Any, step: int) -> list[float] | None:
        """Scene coordinates of one vehicle right now, interpolated across a transit.

        A vehicle on an edge is *between* two nodes; snapping it to either endpoint would
        make the trail look like a staircase of jumps and hide how long each corridor
        really takes.
        """
        net = self.network
        execution = self.sim.execution_for(agv.agv_id) if hasattr(self.sim,'execution_for') else None
        if execution is not None:
            return [execution.pose["x"], execution.pose["y"]] if execution.pose else None
        tr = self.sim._transit.get(agv.agv_id)
        try:
            if tr:
                src, dst, depart, arrive = tr[0], tr[1], int(tr[2]), int(tr[3])
                span = max(1, arrive - depart)
                f = min(1.0, max(0.0, (step - depart) / span))
                from amr_sched.geometry import point_along_polyline
                return list(point_along_polyline(net.edge_centerline(src,dst), f))
            node = net.node(agv.cur_node) if agv.cur_node else None
            return [node.x, node.y] if node is not None else None
        except (KeyError, AttributeError):
            return None

    def set_physical(self, x: float, y: float, yaw: float = 0.0,
                     name: str = "scout_mini") -> None:
        """记一次物理车真值位姿。连续相同的点不重复入列，轨迹才不会被静止抖动撑满。"""
        self.physical_id = name
        self.physical_pose = (float(x), float(y), float(yaw))
        if not self.physical_trail or self.physical_trail[-1] != [float(x), float(y)]:
            self.physical_trail.append([float(x), float(y)])
            if len(self.physical_trail) > self.TRAIL_POINTS:
                del self.physical_trail[: len(self.physical_trail) - self.TRAIL_POINTS]

    def clear_physical(self) -> None:
        """真值断了：位姿清掉，但**轨迹保留**——已经走过的那段路是事实，不该被抹掉。"""
        self.physical_pose = None

    def _trail(self, agv: Any, step: int) -> list[list[float]]:
        xy = self._xy(agv, step)
        if xy is None:
            return self._trails.get(agv.agv_id, [])
        points = self._trails.setdefault(agv.agv_id, [])
        if not points or points[-1] != xy:
            points.append(xy)
            if len(points) > self.TRAIL_POINTS:
                del points[: len(points) - self.TRAIL_POINTS]
        return points

    def _holds(self, step: int) -> dict[str, list[str]]:
        return _holds_from_reservations(self.sim, step)

    def _congestion(self, step: int) -> dict[str, dict[str, Any]]:
        stuck = {a.agv_id for a in self.sim.agvs if a.state.value == "blocked"}
        out = _congestion_from_holds(self._holds(step), self.network, stuck)
        # 时间倒流说明换了一轮仿真（同一台服务重跑），计数必须清空，否则新的第 1 步
        # 会继承上一轮的历史，报出一个从没发生过的"已堵 400 步"。
        if step < self._last_congestion_step:
            self._congested_since.clear()
        self._last_congestion_step = step
        for node_id in list(self._congested_since):
            if node_id not in out:
                del self._congested_since[node_id]
        for node_id, info in out.items():
            since = self._congested_since.setdefault(node_id, step)
            info["steps"] = step - since + 1
        return out

    def hello(self) -> dict[str, Any]:
        """The first message on connect: network plus metadata."""
        spec = NetworkSpec.from_network(self.network)
        return {
            "type": "hello",
            "schema": SCHEMA_VERSION,
            "source": self.source.value,
            "network": {"nodes": spec.nodes, "edges": spec.edges},
            "meta": {
                "n_tasks": len(self.sim.tasks),
                "n_agvs": len(self.sim.agvs),
                "dt": self.sim.config.dt,
                "agv_ids": [a.agv_id for a in self.sim.agvs],
            },
        }

    def frame(self) -> dict[str, Any]:
        """The complete frame for the current step, field-compatible with the offline format."""
        sim = self.sim
        dt = sim.config.dt
        step = sim._step_cursor
        agvs = []
        n_blocked = 0
        for a in sim.agvs:
            tr = sim._transit.get(a.agv_id)
            progress = 0.0
            if tr:
                pos = "%s->%s" % (tr[0], tr[1])
                # _transit is (src, dst, depart, arrive); interpolate between the
                # two so the pose reflects how far the vehicle actually got.
                depart, arrive = int(tr[2]), int(tr[3])
                span = max(1, arrive - depart)
                progress = min(1.0, max(0.0, (step - depart) / span))
            else:
                pos = a.cur_node or ""
            node, edge, prog = _split_edge(pos, progress)
            pose = Pose(agv_id=a.agv_id, node=node, edge=edge,
                        progress=prog, source=self.source,
                        timestamp=step * dt)
            job = sim._jobs.get(a.agv_id)
            rest: list[str] = []
            if job is not None and job.hops:
                rest = [h.dst for h in job.hops[job.idx:]]
            agvs.append(AgvFrame(
                agv_id=a.agv_id, pose=pose, state=a.state.value,
                task_id=job.task_id if job is not None else None,
                path=rest,
            ).to_dict())
            execution = sim.execution_for(a.agv_id) if hasattr(sim,'execution_for') else None
            if execution is not None:
                actual = execution.pose
                agvs[-1]["pose"] = {
                    "agv_id": a.agv_id, "source": execution.identity["source"],
                    "available": actual is not None,
                    "x": actual["x"] if actual else None,
                    "y": actual["y"] if actual else None,
                    "yaw": actual["yaw"] if actual else None,
                    "timestamp": actual["timestamp"] if actual else None,
                    "frame_id": execution.identity["frame_id"],
                    "time_base": execution.identity["time_base"],
                }
                if tr:
                    agvs[-1]["path"] = [tr[0], tr[1]] + rest
            if a.state.value == "blocked":
                n_blocked += 1
        counts: dict[str, int] = {}
        for t in sim.tasks:
            counts[t.status.value] = counts.get(t.status.value, 0) + 1
        return {
            "type": "frame",
            "schema": SCHEMA_VERSION,
            "step": step,
            "t": round(step * dt, 3),
            "agvs": agvs,
            "task_counts": counts,
            **({"execution": sim.execution.snapshot(), "physical_holds": sim._res.pins()}
               if getattr(sim, "execution", None) else {}),
            **({'executions': {robot: gate.snapshot() for robot, gate in sim.executions.items()},
                'physical_holds': sim._res.pins()}
               if getattr(sim, 'executions', {}) and sim.execution is None else {}),
            "n_blocked": n_blocked,
            # Who holds which node *right now*, straight from the reservation table. The
            # display layer needs this to show junction reservations and congestion: a
            # vehicle's position does not reveal that it has booked the next junction, and
            # two vehicles queued at one node look identical to two vehicles just passing.
            "holds": self._holds(step),
            # Where each vehicle actually was, in scene metres. The console draws this as
            # a thin dashed line under the planned route, so "what Python decided" and
            # "what really happened" can be compared on one screen. In 三维验证模式 the
            # poses come from Gazebo; in algorithm research mode from this engine. Either
            # way the display layer only projects what it is given.
            "trails": {a.agv_id: self._trail(a, step) for a in sim.agvs},
            # Which authority produced each trail. Per-vehicle, not per-frame: in 三维验证模式
            # the vehicle Gazebo actually drives reports Gazebo truth while the other vehicles
            # are still the engine's own. One frame-level label would be a lie about one of them.
            "trail_sources": {a.agv_id: self.source.value for a in sim.agvs},
            # 物理车：Gazebo 真值，独立实体。没有真值就没有这个键，不会凭空造一台。
            **({"physical": {
                "id": self.physical_id,
                "source": PoseSource.GAZEBO.value,
                "pose": list(self.physical_pose) if self.physical_pose else None,
                "trail": self.physical_trail,
            }} if self.physical_pose is not None or self.physical_trail else {}),
            # Nodes whose current claim count reaches their capacity: the queues the
            # operator actually needs to see.
            "congestion": self._congestion(step),
            "preemptions": [
                {"urgent": u, "victim_agv": va, "victim_task": vt,
                 "was_picked": bool(wp)}
                for (u, va, vt, wp) in getattr(sim, "_step_preemptions", [])
            ],
        }


def _holds_from_reservations(sim: Any, step: int) -> dict[str, list[str]]:
    """Current node claims from the simulator's reservation table, owner ids listed."""
    table = getattr(sim, "_res", None)
    if table is None:
        return {}
    snapshot = table.snapshot(step, step + 1)
    out: dict[str, list[str]] = {}
    for node, entries in snapshot.items():
        owners = sorted({owner for (_start, _end, owner) in entries})
        if owners:
            out[node] = owners
    return out


def _congestion_from_holds(holds: dict[str, list[str]], network: Any,
                           blocked: set | None = None) -> dict[str, dict[str, Any]]:
    """Nodes at or over capacity, with how many claims they carry and how many are waiting.

    Capacity comes from the network definition rather than being assumed, so a node with a
    capacity of 2 is not reported as congested at two claims.

    Why "blocked" is counted
    -------------------------
    A full node is not automatically a queue. Vehicles parked in a bay sit at capacity all
    night without anyone waiting, and reporting that as congestion would overstate what the
    picture shows. The blocked count is how many claimants are actually stuck this step, so
    the display can tell "full but flowing" from "full and waiting".
    """
    out: dict[str, dict[str, Any]] = {}
    stuck = blocked or set()
    for node_id, owners in holds.items():
        node = None
        try:
            node = network.nodes.get(node_id)
        except AttributeError:
            node = None
        capacity = int(getattr(node, "capacity", 1) or 1)
        if len(owners) >= capacity and len(owners) > 1:
            waiting = [o for o in owners if o in stuck]
            out[node_id] = {"claims": len(owners), "capacity": capacity, "agvs": owners,
                            "blocked": len(waiting)}
    return out


def frame_to_line(obj: dict[str, Any]) -> bytes:
    """Encode one message as a line of UTF-8 JSON, including the newline."""
    return (json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
