"""Discrete-time simulation main loop.

Time model
----------
Fixed step ``dt`` seconds. Event-driven execution is not used: conflict
resolution happens on a space-time grid and must share the same discretization
as execution. Under a fixed step, identical input always yields identical
output, which is what the paired comparisons require.

Where the two planners differ
-----------------------------
The difference lives in :meth:`Simulator._plan_and_reserve`:

* **Planner A writes the whole path into the reservation table at planning
  time**, including per-step edge and node occupancy. Later vehicles can "see"
  the intentions of every vehicle ahead of them while planning.
* **Space-time A*** steers around those reservations; **static Dijkstra** does
  not and simply takes the shortest path. The static planner writes no
  reservation (it has no notion of conflict), so its paths overlap with others
  and are stopped at execution time by :meth:`Simulator._can_enter`, which
  shows up as waiting.

This is the fair comparison of "conflict resolution vs none": both use the same
executor, the same allocator, and the same instances. The only variable is
whether the planner understands space-time occupancy.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, field

from amr_sched.allocation.base import AllocationStrategy
from amr_sched.entities import Agv, AgvState, MaterialType, Task, TaskPriority, TaskStatus
from amr_sched.external_execution import ExternalExecution
from amr_sched.monitor import DeviationEvent, DeviationEventError, ExecutionMonitor
from amr_sched.network import Network, NodeKind
from amr_sched.planners.base import PathPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.reservation import Interval, ReservationTable


@dataclass
class Hop:
    """One movement segment."""

    src: str
    dst: str
    depart: int
    arrive: int


@dataclass
class _Job:
    """Execution state of one AGV's current transport task."""

    task_id: str
    leg: str                      # pickup, delivery, return, yield, or parking
    hops: list[Hop] = field(default_factory=list)
    idx: int = 0
    dwell_until: int = 0
    #: Target node of the current leg (pickup or drop-off point)
    goal: str = ""
    #: Exclusive upper bound of the hops index already written to the reservation table
    reserved_upto: int = 0
    #: Urgent task held while this vehicle carries its original load back.
    return_urgent_id: str = ""
    #: Actual loaded movement, including a return if preempted, in seconds.
    loaded_travel_seconds: float = 0.0


@dataclass
class SimulationConfig:
    dt: float = 1.0
    max_steps: int = 200_000
    planning_horizon: int = 20_000
    record_trajectory: bool = False
    seed: int = 0
    load_time: float = 5.0
    unload_time: float = 5.0
    #: Workstation **service time** in seconds: how long the workpiece keeps the
    #: workstation occupied after drop-off completes.
    #: 0 = the task is finished at drop-off (legacy behavior, equivalent to a
    #: workstation with no service time).
    #:
    #: Why model it separately
    #: -----------------------
    #: A maintenance workstation does not become free the moment material is
    #: dropped off — the workpiece is serviced there for a while. Without this,
    #: workstations stay forever idle, **workstation queueing is underestimated**,
    #: and system throughput is overestimated.
    #:
    #: Why the AGV does not wait alongside
    #: -----------------------------------
    #: In a real workshop the AGV leaves to do other work after drop-off and the
    #: workpiece stays at the workstation. Making the AGV wait turns the vehicle
    #: into an appendage of the workstation, so what gets measured is the
    #: workstation cycle rather than scheduling quality. This implementation
    #: therefore occupies the **workstation only**, never the AGV.
    process_time: float = 0.0
    #: Number of path segments **reserved ahead** at planning time. 0 reserves the
    #: entire path, which locks up the network.
    #: Measured sweep (20/50/100 tasks x 4/6 vehicles x 5 seeds, versus the static
    #: baseline): w=1 is best (+1.1% to +6.9%); w>=2 degrades sharply (+15% to
    #: +27%); w=0 (reserve everything) is worst. In other words, **over-coordination
    #: is worse than no coordination**. Note that w=1 still trails the baseline by a
    #: small margin — see the results report.
    reservation_window: int = 1
    #: Rescheduling policy for urgent tasks
    reschedule_policy: ReschedulePolicy = ReschedulePolicy.NONE
    #: Whether an idle vehicle yields its spot to an incoming loaded vehicle (deadlock prevention)
    idle_retreat: bool = True
    #: Whether to re-check node capacity against actual occupancy (strict, prone to livelock, off by default)
    strict_node_check: bool = False
    #: Dwell reservation window at a station, in steps. Renewed while dwelling, released on departure.
    node_hold_steps: int = 60
    #: Dwell window on ring aisles, in steps. Must be short, otherwise transit nodes get locked.
    aisle_hold_steps: int = 6
    #: Steps of continuous blocking before deadlock is declared and broken (0 = disabled)
    deadlock_timeout: int = 200


def _shortest_distance_static(network, a: str, b: str) -> float:
    """Static shortest **distance** between two nodes (ignores capacity and reservations)."""
    import heapq
    if a == b:
        return 0.0
    dist = {a: 0.0}
    pq = [(0.0, a)]
    seen = set()
    while pq:
        d, u = heapq.heappop(pq)
        if u in seen:
            continue
        seen.add(u)
        if u == b:
            return d
        for e in network.out_edges(u):
            nd = d + e.length
            if nd < dist.get(e.dst, float("inf")):
                dist[e.dst] = nd
                heapq.heappush(pq, (nd, e.dst))
    return dist.get(b, 0.0)


@dataclass
class StepRecord:
    """A complete snapshot of one time step, used for state monitoring and
    process visualization.

    Recording positions alone is not enough: without seeing what each vehicle is
    doing (idle / loading / blocked / parking) and how far each task has
    progressed (pending / assigned / picked / completed), there is no real
    process monitoring.
    """

    step: int
    agv_positions: dict[str, str]
    #: State name of each vehicle (idle / to_pickup / loading / to_dropoff / unloading / blocked / parking)
    agv_states: dict[str, str] = field(default_factory=dict)
    #: Task id currently executed by each vehicle (None when idle)
    agv_tasks: dict[str, str | None] = field(default_factory=dict)
    #: Counts per task state: pending / assigned / picked / completed
    task_counts: dict[str, int] = field(default_factory=dict)
    #: Number of vehicles blocked during this step
    n_blocked: int = 0
    #: Node occupancy summary: {node_id: [agv_id, ...]}.
    #: This is the contents of the **reservation table**, not where vehicles
    #: physically are. The whole mechanism of conflict resolution lives here, and
    #: a vehicle position plot cannot show who reserved a spot out from under whom.
    node_holds: dict[str, list[str]] = field(default_factory=dict)
    #: Remaining node sequence of each vehicle's **current leg** (including the
    #: destination, excluding nodes already passed).
    #:
    #: Why record it
    #: -------------
    #: Without it, the visualization side can only draw where a vehicle is, not
    #: where it intends to go. The requirement to show the scheduling process is
    #: met directly by these path lines.
    #:
    #: Scope: only the **current leg** (to pickup or to drop-off) is recorded, not
    #: the whole task. Cross-leg paths are replanned on reaching the switch point,
    #: so recording the old leg ahead of time would be meaningless.
    agv_paths: dict[str, list[str]] = field(default_factory=dict)
    #: Preemption events during this step: [(urgent_task_id, victim_agv_id, victim_task_id, already_picked)]
    preemptions: list[tuple[str, str, str, bool]] = field(default_factory=list)


@dataclass
class SimulationResult:
    tasks: list[Task]
    agvs: list[Agv]
    config: SimulationConfig
    trajectory: list[StepRecord] = field(default_factory=list)
    finished_step: int = 0
    total_wait_steps: int = 0
    blocked_steps: int = 0
    #: Number of blocking **events** (one continuous block counts once, distinct
    #: from the step count in blocked_steps). The acceptance metric asks for the
    #: **count** of conflict waits; only steps were recorded before, so it is added
    #: here.
    conflict_wait_events: int = 0
    replan_count: int = 0
    #: Planning retries (the task did not fail, this step simply found no route)
    planning_retries: int = 0
    #: Real failures: a task is voided because the execution position diverged from the plan
    hard_failures: int = 0
    collision_events: int = 0
    #: Number of times an urgent task preempted a normal one
    preemption_count: int = 0
    #: Preemption detail: [(step, urgent_task, victim_agv, victim_task, already_picked)]
    preemption_events: list[tuple[int, str, str, str, bool]] = field(
        default_factory=list)
    #: Deadlock breaks (the task is abandoned after a blocking timeout)
    deadlock_breaks: int = 0
    #: Material waste in seconds caused by preempting a loaded vehicle. Non-zero only under PREEMPT_ANY.
    material_waste_steps: float = 0.0
    #: Cumulative distance travelled by the whole fleet, in meters. Used to test the
    #: "detour" hypothesis: if a planner has a longer completion time but also a
    #: longer distance, it really is taking detours.
    total_distance: float = 0.0
    #: Cumulative **ideal** distance for the whole fleet (static shortest-path
    #: distance per leg). Detour ratio = total_distance / ideal_distance; 1.0 means
    #: not a single extra step was taken.
    ideal_distance: float = 0.0

    # ---------- Detour diagnostics (explains why a planner is slower) ----------

    def detour_ratio(self, network) -> float:
        """Detour ratio = actual distance / sum of static shortest distances.

        = 1.0 means not a single extra step; > 1.0 means the planner took a longer
        route. This is direct evidence for the hypothesis that space-time A*
        detours in order to yield.
        """
        ideal = 0.0
        for task in self.tasks:
            if task.complete_step is None:
                continue
            ideal += _shortest_distance_static(network, task.pickup, task.dropoff)
        if ideal <= 0:
            return 1.0
        return self.total_distance / ideal

    # ---------- The four acceptance metrics ----------

    def completed_tasks(self) -> list[Task]:
        return [t for t in self.tasks if t.status is TaskStatus.COMPLETED]

    def makespan_seconds(self) -> float:
        done = [t.complete_step for t in self.completed_tasks() if t.complete_step is not None]
        if not done:
            return float("nan")
        return max(done) * self.config.dt

    def on_time_rate(self) -> float:
        """Fraction of tasks with a due date that finished on time. Tasks without a due date are excluded."""
        with_due = [t for t in self.tasks if t.due_step is not None]
        if not with_due:
            return float("nan")
        on_time = [
            t for t in with_due
            if t.status is TaskStatus.COMPLETED
            and t.complete_step is not None
            and t.complete_step <= t.due_step
        ]
        return len(on_time) / len(with_due)

    def agv_utilization(self) -> float:
        if not self.agvs:
            return float("nan")
        ratios = []
        for a in self.agvs:
            total = a.busy_steps + a.idle_steps
            ratios.append(a.busy_steps / total if total > 0 else 0.0)
        return sum(ratios) / len(self.agvs)

    def conflict_wait_steps(self) -> int:
        """Total **steps** of conflict wait (accumulated duration, not event count)."""
        return self.blocked_steps

    def conflict_wait_count(self) -> int:
        """Number of conflict wait **events**.

        One continuous block counts once regardless of how many steps it lasts.
        Complementary to :meth:`conflict_wait_steps`: the count reflects how often
        a vehicle was interrupted, the step count reflects how long it waited in
        total. The same dataset can rank configurations differently under the two.
        """
        return self.conflict_wait_events

    def urgent_tasks(self) -> list[Task]:
        from amr_sched.entities import TaskPriority
        return [t for t in self.tasks if t.priority is TaskPriority.URGENT]

    def urgent_on_time_rate(self) -> float:
        """On-time rate of urgent tasks (only urgent tasks with a due date are counted)."""
        u = [t for t in self.urgent_tasks() if t.due_step is not None]
        if not u:
            return float("nan")
        ok = [
            t for t in u
            if t.status is TaskStatus.COMPLETED
            and t.complete_step is not None
            and t.complete_step <= t.due_step
        ]
        return len(ok) / len(u)

    def mean_response_steps(self, urgent_only: bool = False) -> float:
        """Response time: mean steps from task release to being taken up by an AGV."""
        pool = self.urgent_tasks() if urgent_only else self.tasks
        vals = [
            t.assigned_step - t.release_step
            for t in pool
            if t.assigned_step is not None
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    def rate_against_due(self, dues: dict[str, int]) -> dict[str, float]:
        """Recompute the on-time rate against **externally supplied** due steps, for
        due-date definition sensitivity analysis.

        Why it is needed: the same schedule produces a completely different
        on-time rate under a different due-date definition. Judging whether the
        metric is valid requires re-judging the same completion times under several
        definitions **without re-running the simulation**. This changes the metric
        definition, not the results.
        """
        done = [t for t in self.tasks if t.complete_step is not None]
        if not done:
            return {"on_time_rate": float("nan"),
                    "urgent_on_time_rate": float("nan"), "evaluated": 0}
        ok = [t for t in done if t.complete_step <= dues[t.task_id]]
        urg = [t for t in done if t.is_urgent]
        urg_ok = [t for t in urg if t.complete_step <= dues[t.task_id]]
        return {
            "on_time_rate": len(ok) / len(done),
            "urgent_on_time_rate": (len(urg_ok) / len(urg)) if urg else float("nan"),
            "evaluated": len(done),
        }

    def summary(self) -> dict[str, float]:
        return {
            "tasks_total": len(self.tasks),
            "tasks_completed": len(self.completed_tasks()),
            "makespan_seconds": self.makespan_seconds(),
            "on_time_rate": self.on_time_rate(),
            "agv_utilization": self.agv_utilization(),
            "conflict_wait_steps": self.conflict_wait_steps(),
            "conflict_wait_events": self.conflict_wait_count(),
            "planner_reported_wait": self.total_wait_steps,
            "replan_count": self.replan_count,
            "planning_retries": self.planning_retries,
            "hard_failures": self.hard_failures,
            "collision_events": self.collision_events,
            "urgent_on_time_rate": self.urgent_on_time_rate(),
            "mean_response_steps": self.mean_response_steps(),
            "mean_urgent_response_steps": self.mean_response_steps(urgent_only=True),
            "preemption_count": self.preemption_count,
            "deadlock_breaks": self.deadlock_breaks,
            "material_waste_steps": self.material_waste_steps,
            "finished_step": self.finished_step,
        }


class Simulator:
    """Wire up network, tasks, AGVs, planner, and allocator, then run one full simulation."""

    def __init__(
        self,
        network: Network,
        tasks: list[Task],
        agvs: list[Agv],
        planner: PathPlanner,
        allocator: AllocationStrategy,
        config: SimulationConfig | None = None,
        execution: ExternalExecution | None = None,
        executions: dict[str, ExternalExecution] | None = None,
        handlings: dict | None = None,
        inventory=None,
    ) -> None:
        self.network = network
        self.tasks = tasks
        self.agvs = agvs
        self.planner = planner
        self.allocator = allocator
        self.config = config or SimulationConfig()
        self.execution = execution
        if execution is not None and executions is not None:
            raise ValueError("select single or fleet execution, not both")
        if executions is not None and not executions:
            raise ValueError("physical fleet execution mapping must not be empty")
        self.executions = (dict(executions) if executions is not None else
                           {execution.identity['robot_id']: execution} if execution else {})
        self.handlings = dict(handlings or {})
        self.inventory = inventory
        if handlings is not None:
            from .external_execution import ExternalHandling
            if (not self.executions or set(self.handlings) != set(self.executions)
                    or any(not isinstance(h, ExternalHandling) or h.execution is not self.executions[r]
                           for r, h in self.handlings.items())):
                raise ValueError("handling requires the exact physical execution for every robot")
        if inventory is not None:
            from .inventory import DeliveryInventory
            if (not self.handlings or not isinstance(inventory, DeliveryInventory)
                    or not set(inventory.capacities) <= set(network.nodes)
                    or any(not inventory.validate_task(t) for t in tasks)):
                raise ValueError('finite inventory requires measured handling and matching delivery recipes')
        if execution is not None and (len(agvs) != 1 or agvs[0].agv_id != execution.identity['robot_id']):
            raise ValueError("external mode requires one matching AGV")
        if self.executions:
            if (self.config.idle_retreat or self.config.deadlock_timeout != 0
                    or self.config.reschedule_policy is not ReschedulePolicy.NONE):
                raise ValueError("external mode requires no retreat, "
                                 "deadlock recovery or automatic preemption")
            if set(self.executions) != {a.agv_id for a in agvs} or len(self.executions) != len(agvs):
                raise ValueError("every physical AGV requires exactly one matching execution")
            identities = set()
            starts = {}
            for agv in agvs:
                gate = self.executions[agv.agv_id]
                if not isinstance(gate, ExternalExecution):
                    raise ValueError("physical execution mapping requires feedback gates")
                if gate.identity['robot_id'] != agv.agv_id:
                    raise ValueError("execution robot identity mismatch")
                start = network.node(agv.start_node)
                if gate.start_xy != (start.x, start.y):
                    raise ValueError("execution bootstrap must match the scheduler start")
                starts[agv.start_node] = starts.get(agv.start_node, 0) + 1
                if starts[agv.start_node] > start.capacity:
                    raise ValueError("physical starts exceed node capacity")
                identities.add(tuple(gate.identity[k] for k in ('scene_id','run_id','frame_id','time_base','source')))
                if len(agvs) > 1 and not gate.namespace_routes:
                    raise ValueError("physical fleet requires robot-namespaced route identities")
            if len(identities) != 1:
                raise ValueError("physical fleet must share scene, run, frame, time and source")

        table_factory = getattr(self.planner, "create_reservation_table", ReservationTable)
        self._res = table_factory()
        self._task_by_id = {t.task_id: t for t in tasks}
        self._travel_cache: dict[tuple[str, str], float] = {}
        self._jobs: dict[str, _Job] = {}
        self._transit: dict[str, tuple[str, str, int, int]] = {}

        self._total_wait = 0
        self._blocked = 0
        self._replans = 0
        self._planning_retries = 0
        self._hard_failures = 0
        self._collisions = 0
        self._preemptions = 0
        self._preemption_events: list[tuple[int, str, str, str, bool]] = []
        self._step_preemptions: list[tuple[str, str, str, bool]] = []
        #: Dwell reservation per vehicle at its destination node (node, Interval), released on departure
        self._node_hold: dict[str, tuple[str, Interval]] = {}
        #: Arrival reservation for in-transit vehicles (node, Interval), released on arrival
        self._node_claim: dict[str, tuple[str, Interval]] = {}
        #: First step of each vehicle's current continuous block (for deadlock timeout)
        self._blocked_since: dict[str, int] = {}
        self._conflict_wait_events: int = 0
        # Distance accumulation (extra metric + detour diagnostics)
        self._odometer = 0.0
        # Material waste from preempting loaded vehicles, in seconds. Accumulated only under PREEMPT_ANY.
        self._material_waste_steps = 0.0
        self._ideal_odometer = 0.0
        self._deadlock_breaks = 0
        self._trajectory: list[StepRecord] = []
        #: Turns execution-layer reports into routing decisions. Lives here, next to the
        #: reservations it has to change, so a deviation is handled in the same place as
        #: every other scheduling decision.
        self._monitor = ExecutionMonitor(self.network)
        #: Detours held because the vehicle was mid-edge when it reported.
        self._pending_detour: dict[str, tuple[str, tuple[str, str]]] = {}
        #: Every deviation decision taken, in order. Read by the display layer.
        self._deviation_log: list[dict] = []
        # Physical road closures are owned here, beside the shared reservations.
        # Pending changes never release the corridor a vehicle still occupies.
        self._closed_corridors: set[tuple[str, str]] = set()
        self._corridor_revisions: dict[tuple[str, str], int] = {}
        self._physical_detours: dict[str, dict] = {}
        self._physical_preemptions: dict[str, dict] = {}
        self._submitted_tasks: dict[str, Task] = {}
        # Durable operator latch; physical feedback/watchdogs keep running.
        self._operator_paused = False
        #: Step cursor. Initialized at construction so external reads before reset() do not fail.
        self._step_cursor = 0

    # ---------- Helpers ----------

    def execution_for(self, robot_id: str) -> ExternalExecution | None:
        """Return only this robot's feedback gate; algorithm mode has none."""
        return self.executions.get(robot_id)

    def execution_feedback(self, message: dict, received_at: float) -> dict:
        """The only feedback entry point for this scheduler's physical action."""
        if not self.executions:
            raise RuntimeError("not in external execution mode")
        gate = self.execution_for(message.get('robot_id')) if isinstance(message, dict) else None
        if gate is None:
            return {'accepted': False, 'reason': 'unknown physical robot'}
        answer = gate.feedback(message, received_at)
        self.poll_execution(received_at)
        return answer

    def poll_execution(self, now: float) -> None:
        """Runtime calls this even while ROS time is paused or tasks are blocked."""
        for agv in self.agvs:
            gate = self.execution_for(agv.agv_id)
            if gate is not None:
                gate.poll(now)
                if gate.fault:
                    agv.state = AgvState.BLOCKED

    def handling_feedback(self, message: dict, received_at: float, source_now: float) -> dict:
        """Equipment receipts acknowledge handling; they never release road pins."""
        handling = self.handlings.get(message.get('robot_id')) if isinstance(message, dict) else None
        if handling is None:
            return dict(accepted=False, reason='unknown physical handling endpoint')
        if self.inventory is not None and not self.inventory.receipt_allowed(message):
            return dict(accepted=False, reason='retained receiving capacity and unique support proof required')
        result = handling.feedback(message, received_at, source_now)
        if result['accepted'] and self.inventory is not None:
            self.inventory._commit_accepted_receipt(message)
        return result

    def _handling_ready(self, agv, job, operation, step):
        """Hold the actual dock until the equipment adapter confirms transfer."""
        handling = self.handlings.get(agv.agv_id)
        if handling is None:
            return True
        node = self.network.node(agv.cur_node)
        if (operation == 'UNLOAD' and self.inventory is not None
                and job.task_id not in self.inventory.committed
                and not self.inventory.reserve(job.task_id, agv.agv_id, agv.cur_node,
                                               handling.execution.pose['timestamp'])):
            agv.state = AgvState.UNLOADING
            agv.busy_steps += 1
            agv.blocked_steps += 1
            return False
        ready = handling.require(job.task_id, operation, agv.cur_node, (node.x, node.y),
                                 handling.execution.pose['timestamp'])
        if not ready:
            agv.state = AgvState.LOADING if operation == 'LOAD' else AgvState.UNLOADING
            agv.busy_steps += 1
        return ready

    def resume_execution(self, expected_route_id: str, now: float, robot_id: str | None = None) -> dict:
        """Continue the same pinned hop from its actual pose, without snapping to a node.

        The native adapter additionally confirms that the old action terminated
        and its guard reports a fresh stop. This API never releases occupancy,
        cancels a task, changes the load, or chooses another global corridor.
        """
        gate = self.execution_for(robot_id) if robot_id is not None else self.execution
        if gate is None:
            return {"accepted": False, "reason": "not in external execution mode"}
        self.poll_execution(now)
        agv = next(a for a in self.agvs if a.agv_id == gate.identity['robot_id'])
        transit = self._transit.get(agv.agv_id)
        command, pose = gate.command, gate.pose
        job = self._jobs.get(agv.agv_id)
        if agv.agv_id in self._physical_detours:
            if transit is None:
                return {"accepted": False, "reason": "pending physical detour requires node reauthorization"}
            if tuple(sorted(transit[:2])) in self._closed_corridors:
                return {"accepted": False, "reason": "retained corridor is closed; keep physical occupancy"}
        if agv.agv_id in self._physical_preemptions and transit is None:
            return {"accepted": False, "reason": "pending urgent task requires node reauthorization"}
        if transit is None and command is None and pose is not None:
            node = self.network.node(agv.cur_node)
            if (agv.agv_id not in self._res.pins()["nodes"].get(agv.cur_node, [])
                    or math.dist((pose["x"], pose["y"]), (node.x, node.y)) > gate.tolerance):
                return {"accepted": False, "reason": "actual pose and retained node occupancy required"}
            if ((job is None) != (agv.cur_task is None)
                    or job is not None and (job.task_id != agv.cur_task or job.leg not in {"to_pickup", "to_dropoff"})):
                return {"accepted": False, "reason": "retained stationary task inconsistent"}
            answer = gate.resume(expected_route_id, now)
            if answer["accepted"]:
                agv.state = (AgvState.IDLE if job is None else
                             AgvState.LOADING if job.leg == "to_pickup" and job.idx >= len(job.hops) else
                             AgvState.UNLOADING if job.leg == "to_dropoff" and job.idx >= len(job.hops) else
                             AgvState.TO_PICKUP if job.leg == "to_pickup" else AgvState.TO_DROPOFF)
            return answer
        if (transit is None or command is None or pose is None or job is None
                or list(transit[:2]) != command["nodes"] or agv.cur_task != command["task_id"]
                or job.task_id != agv.cur_task or job.leg not in {"to_pickup", "to_dropoff"}):
            return {"accepted": False, "reason": "retained task and transit required"}
        src, dst = transit[:2]
        pins = self._res.pins()
        if (any(agv.agv_id not in pins["nodes"].get(n, []) for n in (src, dst))
                or any(agv.agv_id not in pins["edges"].get(f"{a}->{b}", []) for a, b in ((src, dst), (dst, src)))):
            return {"accepted": False, "reason": "retained corridor occupancy required"}
        from amr_sched.geometry import remaining_polyline
        try:
            path = remaining_polyline((pose['x'],pose['y']), self.network.edge_centerline(src,dst), gate.tolerance)
        except ValueError as exc:
            return {"accepted": False, "reason": str(exc)}
        answer = gate.resume(expected_route_id, now, resume_path_xy=path)
        if answer["accepted"]:
            agv.state = AgvState.TO_PICKUP if job.leg == "to_pickup" else AgvState.TO_DROPOFF
        return answer

    def request_physical_detour(self, robot_id: str, expected_route_id: str,
                               blocked_edge, now: float) -> dict:
        """Close a known route corridor and latch its affected physical vehicles.

        This trusted owner API uses only the gate's accepted pose and identity.
        It does not turn a reported coordinate into a new graph location.
        """
        gate = self.execution_for(robot_id)
        if gate is None or isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            return {"accepted": False, "reason": "physical robot and finite receipt time required"}
        self.poll_execution(now)
        if (not gate.bound or gate.pose is None or gate.last_receipt is None or now < gate.last_receipt
                or gate.recovery_token() != expected_route_id):
            return {"accepted": False, "reason": "fresh physical pose and current action version required"}
        if (not isinstance(blocked_edge, (tuple, list)) or len(blocked_edge) != 2
                or any(not isinstance(n, str) for n in blocked_edge)):
            return {"accepted": False, "reason": "two named corridor endpoints required"}
        blocked = tuple(sorted(blocked_edge))
        if blocked[0] == blocked[1] or not self._monitor.edge_exists(blocked):
            return {"accepted": False, "reason": "unknown physical corridor"}

        def affected(robot):
            candidate = self.execution_for(robot)
            job = self._jobs.get(robot)
            return bool(candidate.command and tuple(sorted(candidate.command['nodes'])) == blocked
                        or job and any(tuple(sorted((h.src, h.dst))) == blocked for h in job.hops[job.idx:]))

        if not affected(robot_id):
            return {"accepted": False, "reason": "corridor is outside the retained route"}
        return self.close_physical_corridor(blocked_edge, now, robot_id=robot_id)

    def close_physical_corridor(self, blocked_edge, now: float, *, robot_id: str | None = None) -> dict:
        """Trusted operator/map closure, including roads before task release.

        This is an explicit global traffic input, not a sensor obstacle claim.
        Existing routes stop and retain resources; reopening never resumes them.
        """
        if (not self.executions or type(now) not in (int,float) or not math.isfinite(now)
                or not isinstance(blocked_edge,(tuple,list)) or len(blocked_edge)!=2
                or any(not isinstance(n,str) for n in blocked_edge)):
            return dict(accepted=False,reason='physical corridor and finite receipt time required')
        blocked = tuple(sorted(blocked_edge))
        if blocked[0]==blocked[1] or not self._monitor.edge_exists(blocked):
            return dict(accepted=False,reason='unknown physical corridor')
        if blocked in self._closed_corridors:
            return {"accepted": False, "reason": "corridor is already closed"}

        def affected(robot):
            candidate = self.execution_for(robot)
            job = self._jobs.get(robot)
            return bool(candidate.command and tuple(sorted(candidate.command['nodes'])) == blocked
                        or job and any(tuple(sorted((h.src,h.dst))) == blocked for h in job.hops[job.idx:]))

        self.poll_execution(now)
        self._closed_corridors.add(blocked)
        revision = self._corridor_revisions.get(blocked, 0) + 1
        self._corridor_revisions[blocked] = revision
        stopped = []
        for robot, candidate in self.executions.items():
            if affected(robot):
                request = dict(blocked_edge=list(blocked), closure_revision=revision, reported_route_id=candidate.recovery_token(),
                               timestamp=candidate.pose['timestamp'] if candidate.pose else None)
                self._physical_detours[robot] = request
                candidate.fail('physical corridor closed; stop and replan at an actual node')
                stopped.append(robot)
        answer = dict(accepted=True, action='stop_then_replan',
                      blocked_edge=list(blocked), closure_revision=revision, stopped_robots=stopped)
        if robot_id is not None:
            answer['robot_id'] = robot_id
        self._deviation_log.append(dict(answer, step=self._step_cursor))
        return answer

    def reopen_physical_corridor(self, blocked_edge, expected_revision: int) -> dict:
        """Apply an explicit clear event; never resume a robot or discard its pins.

        The trusted caller confirms obstacle clearance separately. The closure
        revision prevents a delayed clear event from reopening a newer closure.
        """
        if (not self.executions or not isinstance(blocked_edge, (list, tuple)) or len(blocked_edge) != 2
                or any(not isinstance(n, str) for n in blocked_edge)
                or type(expected_revision) is not int):
            return dict(accepted=False, reason='physical corridor and integer closure revision required')
        edge = tuple(sorted(blocked_edge))
        if edge not in self._closed_corridors or self._corridor_revisions.get(edge) != expected_revision:
            return dict(accepted=False, reason='current closed corridor revision required')
        self._closed_corridors.remove(edge)
        self._corridor_revisions[edge] = expected_revision + 1
        answer = dict(accepted=True, action='corridor_reopened', blocked_edge=list(edge),
                      closure_revision=expected_revision, revision=expected_revision + 1)
        self._deviation_log.append(dict(answer, step=self._step_cursor))
        return answer

    def submit_physical_task(self, task: Task) -> dict:
        """Accept an immutable dynamic request once; task lifecycle stays here."""
        if (not self.executions or type(task) is not Task or not isinstance(task.task_id, str) or not task.task_id
                or not isinstance(task.pickup, str) or task.pickup not in self.network.nodes
                or not isinstance(task.dropoff, str) or task.dropoff not in self.network.nodes
                or type(task.release_step) is not int or task.release_step < 0
                or task.due_step is not None and (type(task.due_step) is not int or task.due_step < task.release_step)
                or type(task.priority) is not TaskPriority
                or task.material_type is not None and type(task.material_type) is not MaterialType
                or task.status is not TaskStatus.PENDING
                or any(getattr(task, key) is not None for key in
                       ('assigned_agv', 'assigned_step', 'pickup_step', 'complete_step'))):
            return dict(accepted=False, reason='new valid physical transport task required')
        previous = self._submitted_tasks.get(task.task_id)
        if previous is not None:
            return dict(accepted=previous == task, duplicate=previous == task,
                        reason='same immutable task request required')
        if task.task_id in self._task_by_id or task.release_step < self._step_cursor:
            return dict(accepted=False, reason='unique task ID and current or future release required')
        if self.inventory is not None and not self.inventory.validate_task(task):
            return dict(accepted=False, reason='explicit matching receiving buffer recipe required')
        self._submitted_tasks[task.task_id] = deepcopy(task)
        owned = deepcopy(task)
        self.tasks.append(owned)
        self._task_by_id[owned.task_id] = owned
        return dict(accepted=True, duplicate=False, task_id=owned.task_id, release_step=owned.release_step)

    def request_physical_preemption(self, urgent_task_id: str, now: float) -> dict:
        """Select an unloaded normal task; stop now and retain its current hop."""
        if not self.executions or type(urgent_task_id) is not str or type(now) not in (int, float) or not math.isfinite(now):
            return dict(accepted=False, reason='physical urgent task and finite receipt time required')
        self.poll_execution(now)
        urgent = self._task_by_id.get(urgent_task_id)
        if (urgent is None or not urgent.is_urgent or urgent.status is not TaskStatus.PENDING
                or urgent.release_step > self._step_cursor
                or any(p['urgent_task_id'] == urgent_task_id for p in self._physical_preemptions.values())):
            return dict(accepted=False, reason='released pending urgent task without a prior request required')
        choices = []
        for agv in self.agvs:
            robot = agv.agv_id
            gate = self.execution_for(robot)
            job = self._jobs.get(robot)
            task = self._task_by_id.get(job.task_id) if job else None
            if (robot in self._physical_detours or robot in self._physical_preemptions
                    or robot in self.handlings and self.handlings[robot].command is not None
                       and self.handlings[robot].command['task_id'] == (task.task_id if task else None)
                    or not gate.bound or gate.fault or gate.pose is None or gate.last_receipt is None
                    or now < gate.last_receipt or job is None or job.leg != 'to_pickup'
                    or task is None or task.is_urgent or task.status is not TaskStatus.ASSIGNED
                    or task.assigned_agv != robot):
                continue
            # A physical vehicle must finish its pinned hop before reassignment.
            node = self._transit[robot][1] if robot in self._transit else agv.cur_node
            times = self._routing_network().shortest_travel_times(node, self._max_speed())
            eta = 0. if node == urgent.pickup else times.get(urgent.pickup, math.inf)
            if math.isfinite(eta):
                choices.append((eta, -(len(job.hops) - job.idx), robot))
        if not choices:
            return dict(accepted=False, reason='no eligible unloaded normal task')
        robot = min(choices)[2]
        gate = self.execution_for(robot)
        request = dict(urgent_task_id=urgent_task_id, victim_task_id=self._jobs[robot].task_id,
                       reported_route_id=gate.recovery_token())
        self._physical_preemptions[robot] = request
        gate.fail('physical urgent task; retain hop until actual node reauthorization')
        return dict(accepted=True, action='stop_then_preempt_at_node', robot_id=robot, **request)

    def preempt_execution(self, expected_route_id: str, now: float, robot_id: str | None = None) -> dict:
        """Atomically replace an empty job at its proven node, keeping physical pins."""
        gate = self.execution_for(robot_id) if robot_id is not None else self.execution
        if gate is None or type(now) not in (int, float) or not math.isfinite(now):
            return dict(accepted=False, reason='physical robot and finite recovery time required')
        self.poll_execution(now)
        robot = gate.identity['robot_id']
        request = self._physical_preemptions.get(robot)
        if (request is None or not gate.fault or not gate.stopped or gate.pose is None
                or gate.last_receipt is None or now < gate.last_receipt
                or gate.recovery_token() != expected_route_id or gate._outbox
                or robot in self._transit or gate.command is not None):
            return dict(accepted=False, reason='current stopped action at an actual retained node required')
        agv = next(a for a in self.agvs if a.agv_id == robot)
        pins = self._res.pins()
        job = self._jobs.get(robot)
        victim = self._task_by_id.get(request['victim_task_id'])
        urgent = self._task_by_id.get(request['urgent_task_id'])
        if (agv.cur_node is None or {n for n, owners in pins['nodes'].items() if robot in owners} != {agv.cur_node}
                or any(robot in owners for owners in pins['edges'].values())
                or math.dist((gate.pose['x'], gate.pose['y']),
                             (self.network.node(agv.cur_node).x, self.network.node(agv.cur_node).y)) > gate.tolerance
                or job is None or job.leg != 'to_pickup' or job.task_id != request['victim_task_id']
                or agv.cur_task != job.task_id or victim is None or victim.is_urgent
                or victim.status is not TaskStatus.ASSIGNED or victim.assigned_agv != robot
                or urgent is None or not urgent.is_urgent
                or urgent.status is not TaskStatus.PENDING or urgent.release_step > self._step_cursor):
            return dict(accepted=False, reason='unloaded normal task, pending urgent task and retained node required')
        previous_res, previous_wait = self._res, self._total_wait
        candidate = deepcopy(previous_res)
        candidate.release_all(robot)  # Finite predictions only; permanent pins remain.
        self._res = candidate
        try:
            planned = self._plan_and_reserve(agv, agv.cur_node, urgent.pickup, self._step_cursor,
                                            getattr(self.planner, 'reservations_aware', False))
        finally:
            self._res = previous_res
        if planned is None:
            self._total_wait = previous_wait
            return dict(accepted=False, reason='no reserved urgent route; original task and pins retained')
        answer = gate.resume(expected_route_id, now)
        if not answer['accepted']:
            self._total_wait = previous_wait
            return answer
        hops, pickup_step, reserved = planned
        self._res = candidate
        self._jobs[robot] = _Job(task_id=urgent.task_id, leg='to_pickup', hops=hops,
                                 goal=urgent.pickup, reserved_upto=reserved)
        self._load_end[robot] = pickup_step + self._load_steps(urgent)
        self._drop_end.pop(robot, None)
        self._drop_arrive.pop(robot, None)
        self._node_hold.pop(robot, None)
        self._node_claim.pop(robot, None)
        victim.status = TaskStatus.PENDING
        victim.assigned_agv = victim.assigned_step = None
        agv.cur_task, agv.state = urgent.task_id, AgvState.TO_PICKUP
        urgent.status, urgent.assigned_agv, urgent.assigned_step = TaskStatus.ASSIGNED, robot, self._step_cursor
        self._physical_preemptions.pop(robot)
        self._physical_detours.pop(robot, None)
        event = (self._step_cursor, urgent.task_id, robot, victim.task_id, False)
        self._preemptions += 1
        self._preemption_events.append(event)
        self._step_preemptions.append(event[1:])
        answer = dict(accepted=True, robot_id=robot, urgent_task_id=urgent.task_id, victim_task_id=victim.task_id,
                      recovery_token=gate.recovery_token(), nodes=[agv.cur_node] + [h.dst for h in hops])
        self._deviation_log.append(dict(answer, action='physical_preemption', step=self._step_cursor))
        return answer

    def replan_execution(self, expected_route_id: str, now: float, robot_id: str | None = None) -> dict:
        """Commit a new reserved tail only at a fresh, actually occupied stopped node.

        The native adapter must also establish termination and a fresh guard stop.
        Failed planning leaves the live reservation table and the latched gate intact.
        """
        gate = self.execution_for(robot_id) if robot_id is not None else self.execution
        if gate is None or isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            return {"accepted": False, "reason": "physical robot and finite recovery time required"}
        self.poll_execution(now)
        robot = gate.identity['robot_id']
        if robot in self._physical_preemptions:
            return dict(accepted=False, reason='pending urgent task requires explicit preemption authorization')
        if (robot not in self._physical_detours or not gate.fault or not gate.stopped
                or gate.pose is None or gate.last_receipt is None or now < gate.last_receipt
                or gate.recovery_token() != expected_route_id or gate._outbox):
            return {"accepted": False, "reason": "pending detour, current version and fresh stopped fault required"}
        if robot in self._transit or gate.command is not None:
            return {"accepted": False, "reason": "actual node arrival required; mid-edge detour remains stopped"}
        agv = next(a for a in self.agvs if a.agv_id == robot)
        job = self._jobs.get(robot)
        pins = self._res.pins()
        owned_nodes = {n for n, owners in pins['nodes'].items() if robot in owners}
        if (agv.cur_node is None or owned_nodes != {agv.cur_node}
                or any(robot in owners for owners in pins['edges'].values())
                or math.dist((gate.pose['x'], gate.pose['y']),
                             (self.network.node(agv.cur_node).x, self.network.node(agv.cur_node).y)) > gate.tolerance
                or job is None or job.task_id != agv.cur_task or job.leg not in {'to_pickup', 'to_dropoff'}):
            return {"accepted": False, "reason": "actual pose, retained node and original task required"}
        previous_res, previous_wait = self._res, self._total_wait
        candidate = deepcopy(previous_res)
        candidate.release_all(robot)  # Finite predictions only; physical pins survive.
        self._res = candidate
        try:
            planned = self._plan_and_reserve(agv, agv.cur_node, job.goal, self._step_cursor,
                                            getattr(self.planner, 'reservations_aware', False))
        finally:
            self._res = previous_res
        if planned is None:
            self._total_wait = previous_wait
            return {"accepted": False, "reason": "no feasible reserved detour; task and occupancy retained"}
        answer = gate.resume(expected_route_id, now)
        if not answer['accepted']:
            self._total_wait = previous_wait
            return answer
        hops, _arrival, reserved = planned
        job.hops = job.hops[:job.idx] + hops
        job.reserved_upto = job.idx + reserved
        self._res = candidate
        self._node_hold.pop(robot, None)
        self._node_claim.pop(robot, None)
        request = self._physical_detours.pop(robot)
        agv.state = AgvState.TO_PICKUP if job.leg == 'to_pickup' else AgvState.TO_DROPOFF
        self._replans += 1
        answer = dict(accepted=True, recovery_token=gate.recovery_token(),
                      nodes=[agv.cur_node] + [h.dst for h in hops], request=request)
        self._deviation_log.append(dict(answer, action='physical_reroute', robot_id=robot, step=self._step_cursor))
        return answer

    def _routing_network(self):
        """Keep canonical geometry unchanged while excluding closed global roads."""
        if not self.executions or not self._closed_corridors:
            return self.network
        network = Network()
        for node in self.network.nodes.values():
            network.add_node(node)
        for edge in self.network.edges.values():
            if tuple(sorted((edge.src, edge.dst))) not in self._closed_corridors:
                network.add_edge(edge)
        return network

    def _max_speed(self) -> float:
        return getattr(self.planner, "max_speed", 1.0)

    def _hop_steps(self, src: str, dst: str) -> int:
        return self.planner.steps_for(
            self.network.edge(src, dst).traversal_time(self._max_speed())
        )

    def _load_steps(self, task: "Task | None" = None) -> int:
        """Load duration in steps. Uses the per-material value when material typing
        is on, otherwise a global constant.

        Three material types are specified: bogie, wheelset, and brake valve. Their
        weight and handling difficulty differ by an order of magnitude.
        With ``material_mix=False`` (the default) behavior matches historical data
        exactly; when enabled, :meth:`MaterialType.typical_load_steps` decides.
        """
        mt = getattr(task, "material_type", None) if task is not None else None
        if mt is not None:
            return max(1, int(mt.typical_load_steps()))
        return max(1, int(math.ceil(self.config.load_time / self.config.dt)))

    def _unload_steps(self, task: "Task | None" = None) -> int:
        """Unload duration in steps. Same rules as :meth:`_load_steps`."""
        mt = getattr(task, "material_type", None) if task is not None else None
        if mt is not None:
            # Unloading is one notch faster than loading: no spreader alignment is
            # needed, the load simply drops into place
            return max(1, int(math.ceil(mt.typical_load_steps() * 0.6)))
        return max(1, int(math.ceil(self.config.unload_time / self.config.dt)))

    def _has_processing(self, node_id: str) -> bool:
        """Whether this node has a processing operation.

        Only **workstations** and **assembly areas** perform processing; buffers are
        transit staging, where a load is dropped off and leaves immediately.
        Adding processing time to buffers as well would turn each buffer into a
        second workstation, which makes no physical sense.
        """
        kind = self.network.node(node_id).kind
        return kind in (NodeKind.WORKSTATION, NodeKind.ASSEMBLY)

    def _process_steps(self) -> int:
        """Workstation processing time converted to steps. 0 means not modelled and returns 0 (not 1)."""
        if self.config.process_time <= 0:
            return 0
        return max(1, int(math.ceil(self.config.process_time / self.config.dt)))

    # ---------- Plan and reserve ----------

    def _physical_route_possible(self, agv_id: str, src: str, goal: str) -> bool:
        """Reject topology cut by physical pins before searching future times.

        Pins have no ETA: only a later physical observation can release them.
        This check does not reserve, release, cache failure or replace planning;
        a reachable route still goes through the original space-time planner.
        """
        frontier=[src]
        seen={src}
        while frontier:
            node=frontier.pop()
            if node==goal:
                return True
            for edge in self.network.out_edges(node):
                if tuple(sorted((edge.src, edge.dst))) in self._closed_corridors:
                    continue
                if edge.dst in seen:
                    continue
                if len(self._res.pinned_node_conflicts(edge.dst,agv_id))>=self.network.node(edge.dst).capacity:
                    continue
                owners=self._res.pinned_edge_conflicts(edge.src,edge.dst,agv_id)
                capacity=edge.capacity
                if self.network.has_edge(edge.dst,edge.src):
                    capacity=min(capacity,self.network.edge(edge.dst,edge.src).capacity)
                    owners|=self._res.pinned_edge_conflicts(edge.dst,edge.src,agv_id)
                if len(owners)>=capacity:
                    continue
                seen.add(edge.dst)
                frontier.append(edge.dst)
        return False

    def _plan_and_reserve(
        self, agv: Agv, src: str, goal: str, start_step: int, reserve: bool
    ) -> tuple[list[Hop], int, int] | None:
        """Plan one leg; with ``reserve=True`` the first N hops are written into the reservation table.

        Returns ``(hops, end step, number of hops reserved)``.
        

        The static planner (``reservations_aware = False``) writes no reservations —
        it is inherently "blind" to the others, and precisely that is what defines
        it as the baseline.
        """
        if src == goal:
            # The vehicle already stands on the goal: no movement is needed.
            # Three elements must be returned to match the caller's unpacking (this
            # used to be a bug source: once "zero distance must record 0" was fixed,
            # this branch was taken for the first time and the defect surfaced at once).
            return [], start_step, 0

        if (self.executions and getattr(self.planner,'reservations_aware',False)
                and not self._physical_route_possible(agv.agv_id,src,goal)):
            return None

        path = self.planner.plan(
            network=self._routing_network(),
            start=src,
            goal=goal,
            start_step=start_step,
            reservations=self._res,
            agv_id=agv.agv_id,
            horizon=min(start_step + self.config.planning_horizon, self.config.max_steps),
        )
        if path is None:
            return None
        self._total_wait += path.wait_steps
        # The plan must fall inside the simulation horizon.
        #
        # Under heavy congestion the planner emits plans that "depart only after a
        # long wait", whose arrival step may **exceed max_steps**. Once such a plan
        # is committed, the vehicle enters ``_transit`` and waits for an arrival
        # step that never comes — the simulation then never finishes even after
        # running to the horizon (measured with a 450 s processing time: AGV00 got
        # stuck at arrive=200021 > max_steps=200000 and was blocked for 100k steps).
        # Reject it here up front and leave the vehicle in place to replan next step.
        if path.arrival_steps and path.arrival_steps[-1] >= self.config.max_steps:
            return None

        hops: list[Hop] = []
        cur = src
        t = start_step
        for node, arrive in zip(path.nodes, path.arrival_steps):
            if node == cur:
                continue
            hops.append(Hop(cur, node, t, arrive))
            cur = node
            t = arrive

        if reserve:
            win = self.config.reservation_window
            upto = len(hops) if win <= 0 else min(len(hops), win)
            for h in hops[:upto]:
                self._reserve_hop(agv, h)
            return hops, t, upto
        return hops, t, 0

    def _reserve_hop(self, agv: Agv, hop: Hop) -> None:
        """Write one movement segment into the reservation table (both directions are
        taken, modelling a single-lane aisle)."""
        self._res.reserve_edge(hop.src, hop.dst, Interval(hop.depart, hop.arrive), agv.agv_id)
        if self.network.has_edge(hop.dst, hop.src):
            self._res.reserve_edge(hop.dst, hop.src, Interval(hop.depart, hop.arrive), agv.agv_id)
        self._res.reserve_node(
            hop.dst, Interval(hop.arrive, hop.arrive + 1), agv.agv_id
        )

    def _extend_reservations(self, agv: Agv, job: _Job, now: int) -> None:
        """Rolling top-up: write the following hops inside the window into the
        reservation table, with their times re-laid out from the current progress."""
        # The static baseline takes no part in the reservation system: it neither
        # reads nor writes reservations.
        if not getattr(self.planner, "reservations_aware", False):
            return
        win = self.config.reservation_window
        if win <= 0:
            return
        target = min(len(job.hops), job.idx + win)
        if target <= job.reserved_upto:
            return
        # Re-lay out the times of the following hops starting from the current actual time
        tt = now
        for h in job.hops[job.idx:target]:
            n = self._hop_steps(h.src, h.dst)
            h.depart = max(h.depart, tt)
            h.arrive = h.depart + n
            self._reserve_hop(agv, h)
            tt = h.arrive
        job.reserved_upto = target

    def _refresh_node_holds(self, step: int) -> None:
        """Refresh the "dwell reservation": wherever a vehicle stands, it holds that node
        until it leaves.

        Why this is mandatory
        ---------------------
        Reserving only the "edges" is not enough: after arriving, a vehicle loads,
        unloads and waits for its next task, and throughout that time it
        **physically stands on the node**. Without holding the node, other vehicles
        see it as free in the reservation table and drive in — measured to cause
        hundreds of node capacity violations (two vehicles overlapping on the same
        workstation).

        Implementation notes
        --------------------
        * Each step **first releases** the previous dwell reservation unconditionally,
          so a vehicle that has driven away does not keep holding the old node (leak);
        * then the node is reserved again as ``[step, step+HOLD)`` if the vehicle is
          currently standing on it;
        * HOLD is a finite value, so even if the logic is wrong it expires by itself
          and never locks the network up forever.
        """
        hold = self.config.node_hold_steps
        for agv in self.agvs:
            prev = self._node_hold.pop(agv.agv_id, None)
            if prev is not None:
                self._res.release_node(prev[0], prev[1], agv.agv_id)
            # Only vehicles that have **already arrived** may release their arrival
            # claim; a vehicle still en route must keep holding its target node,
            # otherwise the following vehicle assumes the slot is free.
            if agv.agv_id not in self._transit:
                claim = self._node_claim.pop(agv.agv_id, None)
                if claim is not None:
                    self._res.release_node(claim[0], claim[1], agv.agv_id)
            if agv.agv_id in self._transit:
                continue
            node = agv.cur_node
            if node is None:
                continue
            # A station (workstation/buffer) gets a long dwell: the vehicle loads,
            # unloads and waits for its next task here;
            # a ring aisle gets only a **short window**: the vehicle is just passing
            # through, and holding it too long deadlocks the whole network
            # (measured: with a long dwell window on aisles too, 50/6 and 100/6 got
            # stuck at 500k steps).
            kind = self.network.node(node).kind
            if kind is NodeKind.AISLE:
                # A short dwell is enough for "I am still here at this step" to be
                # visible, without locking a transit node into a parking lot.
                span = self.config.aisle_hold_steps
            else:
                span = hold
            iv = Interval(step, step + span)
            self._res.reserve_node(node, iv, agv.agv_id)
            self._node_hold[agv.agv_id] = (node, iv)

    def _can_enter(self, agv: Agv, src: str, dst: str, t: int,
                   dwell: int = 1) -> bool:
        if self.executions and tuple(sorted((src, dst))) in self._closed_corridors:
            return False
        n = self._hop_steps(src, dst)
        arrive = t + n
        # Never step outside the simulation horizon.
        #
        # ``_plan_and_reserve`` performs the same check, but it uses the start_step
        # of the **planning instant**; a blocked vehicle actually departs at a later
        # step, and ``_commit_move`` recomputes the arrival from the actual step,
        # which can cross the horizon. Measured on the hub layout with 100 tasks /
        # 6 vehicles: arrive=119980 at planning time was legal, but in reality it
        # was 120072 > 120000, hitting the fallback assertion in ``_commit_move``
        # head-on.
        #
        # Stopping the vehicle at the boundary is right: it could not finish this
        # segment anyway, and waiting on would only conjure extra movement into the
        # statistics after the horizon.
        if arrive >= self.config.max_steps:
            return False
        edge_cap = self.network.edge(src, dst).capacity
        if self.network.has_edge(dst, src):
            edge_cap = min(edge_cap, self.network.edge(dst, src).capacity)
        occ = set(self._res.edge_conflicts(src, dst, Interval(t, arrive), exclude_agv=agv.agv_id))
        if self.network.has_edge(dst, src):
            occ |= set(self._res.edge_conflicts(dst, src, Interval(t, arrive), exclude_agv=agv.agv_id))
        if not self.executions:
            # A vehicle can remain on a channel after its predicted arrival.
            # Count the observed occupant once, even if a reservation also exists.
            occ.update(owner for owner, tr in self._transit.items()
                       if owner != agv.agv_id and (tr[:2] == (src, dst) or tr[:2] == (dst, src)))
            # Do not use the channel as a queue for a full destination. A vehicle
            # en route owns its destination slot until it reaches the node.
            claims = sum((tr[1] if tr else other.cur_node) == dst
                         for other in self.agvs if other.agv_id != agv.agv_id
                         for tr in (self._transit.get(other.agv_id),))
            if claims >= self.network.node(dst).capacity:
                return False
        if len(occ) >= edge_cap:
            return False
        # The destination node must be free across the whole **dwell window**, not
        # just at the arrival step. This is the key to fixing "two vehicles parked on
        # the same workstation": the vehicle loads/unloads after arriving, and that
        # time also occupies node capacity.
        if not self._res.is_node_free(
            dst, Interval(arrive, arrive + max(1, dwell)),
            self.network.node(dst).capacity, exclude_agv=agv.agv_id
        ):
            return False
        # Workstation processing occupancy: while the workpiece is still at the
        # workstation, no other vehicle may drive in.
        # Note that this uses the **arrive instant**, not the whole dwell window —
        # processing routinely ends after arrival, and requiring the entire dwell
        # window to be free would produce "processing not finished yet + I can stop
        # but not fit", an over-conservative decision.
        if self._process_holds.get(dst):
            live = [h for h in self._process_holds[dst] if h[1] > arrive]
            if live:
                # Drop expired records so the list does not grow without bound
                self._process_holds[dst] = live
                if len(live) >= self.network.node(dst).capacity:
                    return False
        # Optional fallback: check once more against "the number of vehicles actually
        # standing on the node".
        # Note: this check makes two vehicles see each other and both stand still,
        # which easily causes **livelock** under high load, so it is off by default
        # and enabled only when strict verification is needed.
        if self.config.strict_node_check:
            here = [
                a for a in self.agvs
                if a.agv_id != agv.agv_id
                and a.agv_id not in self._transit
                and a.cur_node == dst
            ]
            if len(here) >= self.network.node(dst).capacity:
                return False
        return True

    def _commit_move(self, agv: Agv, src: str, dst: str, t: int,
                     dwell: int = 1) -> None:
        n = self._hop_steps(src, dst)
        arrive = t + n
        self._res.reserve_edge(src, dst, Interval(t, arrive), agv.agv_id)
        if self.network.has_edge(dst, src):
            self._res.reserve_edge(dst, src, Interval(t, arrive), agv.agv_id)
        # Key point: claim the **destination node** already at commit time, otherwise
        # two vehicles can both pass the admission check at the same instant and both
        # drive toward the same capacity-1 workstation (measured close to a hundred
        # overlaps). It is released on arrival and taken over by the dwell reservation
        # in ``_refresh_node_holds``.
        claim = Interval(arrive, arrive + max(1, dwell))
        self._res.reserve_node(dst, claim, agv.agv_id)
        self._node_claim[agv.agv_id] = (dst, claim)
        if arrive > self.config.max_steps:
            # The last gate: never allow a commit to an instant that cannot be
            # reached. Reaching this point means an upstream check was missed; raise
            # rather than hang silently.
            raise RuntimeError(
                "commit_move arrival step %d exceeds max_steps %d (%s: %s->%s)"
                % (arrive, self.config.max_steps, agv.agv_id, src, dst))
        # (src, dst, depart, arrive): depart is kept so consumers can interpolate a
        # continuous position along the edge instead of snapping to an endpoint.
        self._transit[agv.agv_id] = (src, dst, t, arrive)
        # Odometer accumulation: the length of the edge actually travelled.
        # Compared with "the static shortest-path distance from task origin to
        # destination", it reveals how much of a detour the planner took.
        gate = self.execution_for(agv.agv_id)
        if gate is None:
            self._odometer += self.network.edge(src, dst).length
        else:
            edge = self.network.edge(src, dst)
            target = self.network.node(dst)
            gate.begin(agv.cur_task, src, dst, (target.x, target.y),
                                 min(agv.max_speed, self._max_speed(), edge.speed_limit),
                                 path_xy=self.network.edge_centerline(src,dst))
            self._res.transfer_pins(agv.agv_id, [src, dst], [(src, dst), (dst, src)])

    # ---------- Allocation ----------

    def _try_assign(self, agv: Agv, task: Task, t: int) -> bool:
        roster = getattr(self, '_fleet_admission', None)
        if roster is not None and not roster.eligible(agv.agv_id):
            return False
        here = agv.cur_node or agv.start_node
        reserve = getattr(self.planner, "reservations_aware", False)

        # Plan **only the first leg**. The second leg is planned after reaching the
        # pickup point: reserving both legs at once locks up the whole network for a
        # long time, which was measured to make space-time A* worse than the baseline.
        leg1 = self._plan_and_reserve(agv, here, task.pickup, t, reserve)
        if leg1 is None:
            return False
        self._accept_assignment(agv, task, t, leg1)
        return True

    def _accept_assignment(self, agv: Agv, task: Task, t: int,
                           leg1: tuple[list[Hop], int, int]) -> None:
        """Install a feasible pickup leg; callers commit its reservations first."""
        hops1, t_pick, upto = leg1

        self._jobs[agv.agv_id] = _Job(
            task_id=task.task_id, leg="to_pickup", hops=hops1,
            goal=task.pickup, reserved_upto=upto,
        )
        self._load_end[agv.agv_id] = t_pick + self._load_steps(self._task_by_id.get(agv.cur_task))

        agv.cur_task = task.task_id
        agv.state = AgvState.TO_PICKUP
        task.status = TaskStatus.ASSIGNED
        task.assigned_agv = agv.agv_id
        task.assigned_step = t

    # ---------- Main loop ----------

    # ---- Step-by-step execution interface (added in an earlier round, round 64,
    # ---- to let the Web console drive the simulation live) ----
    #
    # Motivation for the split: ``run()`` is an uninterruptible for loop, so the Web
    # side cannot show a run in progress and can only wait for it to finish and then
    # look at the results — exactly the "live run vs. result replay" gap the advisor
    # pointed out.
    #
    # Principle of the split: **bit-for-bit equivalence**. ``reset()`` plus repeated
    # calls to ``step_once()`` must produce exactly the same SimulationResult as
    # ``run()``.
    # This is asserted by tests/test_simulator_stepping.py (including a SHA comparison).

    def reset(self) -> None:
        """Restore the simulator to the "not started" state; may be called repeatedly.

        Why the reset must be exhaustive
        --------------------------------
        The Web console allows "rerun with the same configuration". If ``reset()``
        misses resetting any single item, the second run carries leftovers from the
        first — most typically tasks already marked COMPLETED, so the second run
        reports "all done" right after starting and looks like it finished in 0
        steps. Such a defect throws no exception; it only yields silently wrong
        results.

        An earlier round (round 64) filled in the 8 state items that had been
        missed: task status, reservation table, job table, transit table, node
        occupancy, the various counters, the event list, and the trajectory.
        ``test_reset_is_idempotent`` locks this invariant down.
        """
        if getattr(self, '_fleet_admission', None) is not None:
            raise RuntimeError('fleet membership is bound; create a fresh Simulator before resetting')
        if any(gate.bound for gate in self.executions.values()):
            raise RuntimeError("cannot reset a bound physical execution; reconcile occupancy first")
        if any(h.command is not None for h in self.handlings.values()):
            raise RuntimeError("cannot reset an issued material transfer; reconcile its load first")
        if self.inventory is not None and (self.inventory.stock != self.inventory.initial
                or self.inventory.reservations or self.inventory.committed or self.inventory.consumed):
            raise RuntimeError('cannot reset committed material or retained receiving slots')
        self._load_end: dict[str, int] = {}
        self._drop_arrive: dict[str, int] = {}
        #: Workstation processing occupancy: node_id -> (start step, end step, agv_id)
        self._process_holds: dict[str, list[tuple[int, int, str]]] = {}
        self._drop_end: dict[str, int] = {}
        self._step_cursor = 0

        # --- The items missed before an earlier round (round 64) filled them in ---
        for task in self.tasks:
            task.status = TaskStatus.PENDING
            task.assigned_agv = None
            task.assigned_step = None
            task.pickup_step = None
            task.complete_step = None
        table_factory = getattr(self.planner, "create_reservation_table", ReservationTable)
        self._res = table_factory()
        for agv in self.agvs:
            if self.execution_for(agv.agv_id) is not None:
                self._res.transfer_pins(agv.agv_id, [agv.start_node])
        self._jobs.clear()
        self._transit.clear()
        self._pending_detour.clear()
        self._closed_corridors.clear()
        self._corridor_revisions.clear()
        self._physical_detours.clear()
        self._physical_preemptions.clear()
        self._submitted_tasks.clear()
        self._operator_paused = False
        self._deviation_log.clear()
        self._node_hold.clear()
        self._node_claim.clear()
        self._blocked_since.clear()
        self._trajectory.clear()
        self._preemption_events.clear()
        self._step_preemptions.clear()
        self._total_wait = 0
        self._blocked = 0
        self._replans = 0
        self._planning_retries = 0
        self._hard_failures = 0
        self._collisions = 0
        self._preemptions = 0
        self._conflict_wait_events = 0
        self._odometer = 0.0
        self._ideal_odometer = 0.0
        self._material_waste_steps = 0.0
        self._deadlock_breaks = 0

        for agv in self.agvs:
            agv.cur_node = agv.start_node
            agv.cur_task = None
            agv.state = AgvState.IDLE
            agv.route = []
            agv.depart_step = None
            agv.reset_stats()

    def step_once(self) -> bool:
        """Advance one simulation step. Returns True if it should continue, False if
        it has terminated.

        The termination conditions correspond one to one with the loop condition of
        ``run()``:
        (1) all tasks are completed/failed; (2) ``max_steps`` is reached.
        """
        cfg = self.config
        step = self._step_cursor
        # Step-by-step driving (Web console / Demo B live channel) must reset() first.
        #
        # run() calls reset() itself, but an external direct call to step_once() does
        # not, and the state tables _load_end / _drop_arrive / _drop_end are only
        # created inside reset().
        # If that is missed, the first task assignment raises AttributeError:
        # '_load_end', with the error sighted several layers away from the real
        # cause, which is very hard to trace.
        #
        # Check explicitly here and put the reason into the exception, instead of
        # letting it surface as an AttributeError.
        if not hasattr(self, "_load_end"):
            raise RuntimeError(
                "reset() must be called before step_once(): "
                "the state tables _load_end/_drop_arrive/_drop_end are created by reset()."
                " (run() calls reset() internally; a step-by-step driver must call it itself.)"
            )
        if step >= cfg.max_steps:
            return False
        if self._all_done():
            return False
        if self.executions and self._operator_paused:
            return True
        self._occupancy_check()
        # Detours held mid-edge are applied at the start of a step, when the previous
        # step's transit has already settled and the vehicle is standing on a node.
        self._apply_pending_detours()
        self._reschedule_step(step)
        self._coordinate_passing(step)
        self._break_deadlock(step)
        self._park_step(step)
        self._assign_step(step)
        self._refresh_node_holds(step)
        self._advance_step(step)
        self._record(step)
        if step % 500 == 0 and step > 0:
            self._res.purge_before(step)
        self._step_cursor = step + 1
        return True

    def is_finished(self) -> bool:
        """Whether the termination condition is already met (without advancing state)."""
        return self._all_done() or self._step_cursor >= self.config.max_steps

    def run(self) -> SimulationResult:
        cfg = self.config
        self.reset()
        while self.step_once():
            pass
        step = self._step_cursor
        return self._build_result(step)

    def _build_result(self, step: int | None = None) -> SimulationResult:
        """Assemble the current state into a SimulationResult.

        Extracted so that the step-by-step driver can also produce exactly the same
        object as ``run()``, instead of writing the assembly logic twice and letting
        the two drift apart over time.
        """
        if step is None:
            step = self._step_cursor
        cfg = self.config
        return SimulationResult(
            tasks=self.tasks, agvs=self.agvs, config=cfg, trajectory=self._trajectory,
            finished_step=step, total_wait_steps=self._total_wait,
            blocked_steps=self._blocked, replan_count=self._replans,
            conflict_wait_events=self._conflict_wait_events,
            planning_retries=self._planning_retries,
            hard_failures=self._hard_failures,
            collision_events=self._collisions,
            preemption_count=self._preemptions,
        preemption_events=list(self._preemption_events),
            deadlock_breaks=self._deadlock_breaks,
            material_waste_steps=self._material_waste_steps,
        )



    def _all_done(self) -> bool:
        return all(t.status in (TaskStatus.COMPLETED, TaskStatus.FAILED) for t in self.tasks)

    def _occupancy_check(self) -> None:
        seen: dict[str, list[str]] = {}
        for agv in self.agvs:
            if agv.agv_id in self._transit:
                continue
            if agv.cur_node is None:
                continue
            seen.setdefault(agv.cur_node, []).append(agv.agv_id)
        for node, ids in seen.items():
            cap = self.network.node(node).capacity
            if len(ids) > cap:
                self._collisions += len(ids) - cap

    def _coordinate_passing(self, step: int) -> None:
        """Divert a vehicle before a saturated wait cycle reaches timeout."""
        if self.executions or self.config.deadlock_timeout <= 0:
            return
        claimed = {}
        ready = {}
        for agv in self.agvs:
            owner = agv.agv_id
            transit = self._transit.get(owner)
            node = transit[1] if transit else agv.cur_node
            claimed.setdefault(node, []).append(owner)
            job = self._jobs.get(owner)
            if not transit and job and job.idx < len(job.hops) and job.leg in ("to_pickup", "to_dropoff", "to_return", "park"):
                hop = job.hops[job.idx]
                if hop.src == node:
                    ready[owner] = (agv, job, hop)
        blocked = {}
        for owner, (_, _, hop) in ready.items():
            occupants = claimed.get(hop.dst, ())
            if len(occupants) >= self.network.node(hop.dst).capacity:
                blocked[owner] = tuple(o for o in occupants if o in ready)
        cycles = set()
        for start in blocked:
            todo = list(blocked[start])
            seen = set()
            while todo:
                owner = todo.pop()
                if owner == start:
                    cycles.add(start)
                    break
                if owner not in seen:
                    seen.add(owner)
                    todo.extend(blocked.get(owner, ()))
        if not cycles:
            return
        options = []
        for owner in sorted(cycles):
            agv, job, hop = ready[owner]
            task = self._task_by_id.get(job.task_id)
            peers = [triple for key, triple in ready.items() if key != owner]
            claimed_goals = {j.goal for _, j, _ in peers}
            claimed_hops = {p.dst for _, _, p in peers}
            for edge in self.network.out_edges(hop.src):
                target = edge.dst
                if target == hop.dst or not self._can_enter(agv, hop.src, target, step):
                    continue
                distance = self._travel_time(target, job.goal)
                if not math.isfinite(distance):
                    continue
                remaining = [e.dst for e in self.network.out_edges(target) if e.dst != hop.src]
                open_exits = sum(len(claimed.get(n, ())) < self.network.node(n).capacity
                                 for n in remaining)
                releases = sum(p.dst == hop.src for _, _, p in peers)
                score = (target in claimed_hops, target in claimed_goals, not open_exits,
                         -releases, task is not None and task.status is TaskStatus.PICKED,
                         self._hop_steps(hop.src, target) + distance, owner, target)
                options.append((score, agv, job, hop.src, target))
        if not options:
            return
        _, agv, job, src, target = min(options, key=lambda option: option[0])
        owner = agv.agv_id
        task = self._task_by_id.get(job.task_id)
        arrive = step + self._hop_steps(src, target)
        self._res.release_all(owner)
        self._node_hold.pop(owner, None)
        self._node_claim.pop(owner, None)
        job.hops = [Hop(src, target, step, arrive)]
        job.idx = job.reserved_upto = 0
        if task is not None:
            job.leg = "loaded_yield" if task.status is TaskStatus.PICKED else "empty_yield"
        job.goal = target
        self._load_end[owner] = arrive
        self._drop_end.pop(owner, None)
        self._blocked_since.pop(owner, None)
        self._replans += 1

    def _break_deadlock(self, step: int) -> None:
        """Try a legal yield from a node after continuous blocking.

        A vehicle keeps its active channel until arrival. A loaded job keeps its
        owner, endpoints and pickup history. Only an unpicked job may be returned
        to the queue if it cannot yield. Physical recovery uses its own stop and
        authorization protocol; physical configurations disable this heuristic.
        """
        timeout = self.config.deadlock_timeout
        if timeout <= 0:
            return
        for agv in self.agvs:
            since = self._blocked_since.get(agv.agv_id)
            if since is None or step - since < timeout:
                continue
            owner = agv.agv_id
            job = self._jobs.get(owner)
            if job is None:
                self._blocked_since.pop(agv.agv_id, None)
                continue
            # Even an empty vehicle cannot give up a channel it still occupies.
            if owner in self._transit:
                continue
            task = self._task_by_id.get(job.task_id)
            if task is not None and task.status in (TaskStatus.PICKED, TaskStatus.ASSIGNED):
                if self._yield_at_node(agv, job, task, step):
                    self._deadlock_breaks += 1
                    continue
                if task.status is TaskStatus.PICKED:
                    continue
            del self._jobs[owner]
            if task is not None and task.status is TaskStatus.ASSIGNED:
                task.status = TaskStatus.PENDING
                task.assigned_agv = None
                task.assigned_step = None
                task.pickup_step = None
            agv.cur_task = None
            agv.state = AgvState.IDLE
            for d in (self._load_end, self._drop_end, self._node_hold,
                      self._node_claim, self._transit):
                d.pop(agv.agv_id, None)
            self._res.release_all(agv.agv_id)
            self._blocked_since.pop(agv.agv_id, None)
            self._deadlock_breaks += 1

    def _yield_at_node(self, agv: Agv, job: _Job, task: Task, step: int) -> bool:
        """Yield one legal hop, then replan from its actual arrival node."""
        here = agv.cur_node
        loaded = task.status is TaskStatus.PICKED
        goal = task.dropoff if loaded and not job.return_urgent_id else task.pickup
        if here is None or here == goal:
            return False
        old_next = job.hops[job.idx].dst if job.idx < len(job.hops) else None
        usable = []
        for edge in self.network.out_edges(here):
            if edge.dst != old_next and self._can_enter(agv, here, edge.dst, step):
                present = sum(other.cur_node == edge.dst and other.agv_id not in self._transit
                              for other in self.agvs)
                if present < self.network.node(edge.dst).capacity:
                    distance = self._travel_time(edge.dst, goal)
                    if math.isfinite(distance):
                        usable.append((distance, edge.dst))
        if not usable:
            return False
        _, target = min(usable)
        arrive = step + self._hop_steps(here, target)
        owner = agv.agv_id
        self._res.release_all(owner)
        self._node_hold.pop(owner, None)
        self._node_claim.pop(owner, None)
        self._load_end.pop(owner, None)
        job.hops = [Hop(here, target, step, arrive)]
        job.idx = job.reserved_upto = 0
        job.leg = 'loaded_yield' if loaded else 'empty_yield'
        job.goal = target
        self._load_end[owner] = arrive
        self._drop_end.pop(owner, None)
        self._blocked_since.pop(owner, None)
        return True

    def _park_step(self, step: int) -> None:
        """Idle vehicles yield: they must not occupy "the node someone else is heading
        for".

        Why it is needed
        ----------------
        Station capacity is limited (workstation 1, buffer 2). If an empty vehicle
        just stays where it unloaded, the next loaded vehicle coming to deliver can
        never get in — measured, 2 loaded vehicles were each blocked for 170k steps
        and the whole simulation ran to max_steps without finishing.

        Yielding rules
        --------------
        If the node an idle vehicle stands on is **the destination of any in-transit
        task**, move it to the nearest buffer/aisle slot that is unused and not
        claimed by any task.
        """
        if not self.config.idle_retreat:
            return

        # Destinations of all in-transit tasks — these nodes must stay clear
        wanted: set[str] = set()
        for agv in self.agvs:
            job = self._jobs.get(agv.agv_id)
            if job is not None and job.goal:
                wanted.add(job.goal)
        if not wanted:
            return

        # Waypoints must yield too (fixed in an earlier round, round 65).
        #
        # The original rule looked only at "destinations" and missed choke points.
        # Measured (50 tasks / 4 vehicles / capacity 1 / seed 1): AGV00 stopped on R5
        # after unloading and stayed put, and R5 is the only entrance to W5 (the only
        # incoming edge of W5 is R5->W5). The other three vehicles all had to pass
        # through R5 to reach W5 and were all blocked — AGV00 held R5 for 117337 steps
        # (about 32 hours), deadlock_breaks reached 2501, and only 48/50 tasks
        # completed.
        #
        # A vehicle not occupying a critical node "legitimately" squatted on the choke
        # point because its own destination was in nobody's target set — yet other
        # vehicles' **waypoints** included it.
        # So the remaining path nodes of in-transit vehicles must also be added to the
        # set that has to stay clear.
        #
        # Blast radius: only extreme configurations with aisle_capacity=1 are affected;
        # instances with capacity>=2 are bit-for-bit unchanged (verified with 10 paired
        # runs).
        for agv in self.agvs:
            job = self._jobs.get(agv.agv_id)
            if job is None or not job.hops:
                continue
            for k in range(job.idx, len(job.hops)):
                wanted.add(job.hops[k].dst)

        for agv in self.agvs:
            roster = getattr(self, '_fleet_admission', None)
            if roster is not None and not roster.eligible(agv.agv_id):
                continue  # inactive parked vehicles must not be moved by idle retreat
            if agv.agv_id in self._jobs or agv.agv_id in self._transit:
                continue
            node = agv.cur_node or agv.start_node
            if node not in wanted:
                continue
            target = self._nearest_park(node, avoid=wanted)
            if target is None:
                continue
            planned = self._plan_and_reserve(
                agv, node, target, step,
                getattr(self.planner, "reservations_aware", False),
            )
            if planned is None:
                continue
            hops, _t_arr, upto = planned
            self._jobs[agv.agv_id] = _Job(
                task_id=None, leg="park", hops=hops,
                goal=target, reserved_upto=upto,
            )
            agv.state = AgvState.PARKING

    def _nearest_park(self, from_node: str,
                      avoid: set[str] | None = None) -> str | None:
        """Find the nearest buffer/aisle node that has spare capacity and is not claimed
        by anyone else."""
        avoid = avoid or set()
        best: tuple[float, str] | None = None
        for n in self.network.nodes.values():
            if n.node_id == from_node:
                continue
            if n.kind not in (NodeKind.BUFFER, NodeKind.AISLE):
                continue
            if n.node_id in avoid:
                continue
            occ = sum(
                1 for a in self.agvs
                if a.agv_id not in self._transit and a.cur_node == n.node_id
            )
            if occ >= n.capacity:
                continue
            tt = self._travel_time(from_node, n.node_id)
            if tt == float("inf"):
                continue
            if best is None or tt < best[0]:
                best = (tt, n.node_id)
        return best[1] if best else None

    @property
    def deviation_events(self) -> list[dict]:
        """Every execution-layer deviation decision taken so far, in order.

        Read-only by convention: the display layer reads this to show what Python did
        with a report, and must not edit it.
        """
        return list(self._deviation_log)

    # ---------- 执行层上报（ROUTE_DEVIATION）----------

    def report_deviation(self, robot_id: str, blocked_edge, current_pose,
                         timestamp: float, goal: str | None = None,
                         event: str = "ROUTE_DEVIATION") -> dict:
        """Handles one deviation report from the execution layer.

        This is where the architecture promise becomes code: the local layer says
        "I could not get through here", and **Python** decides what happens next.
        The local layer never picks its own global route.

        Returns the decision as a plain dict so a bridge can put it on the wire as-is.
        """
        if self.executions:
            return {"action": "rejected", "reason": "physical recovery requires stop and reconciliation"}
        agv = next((a for a in self.agvs if a.agv_id == robot_id), None)
        if agv is None:
            return {"action": "rejected", "reason": "未知车辆: %s" % robot_id}

        job = self._jobs.get(agv.agv_id)
        target = goal or (job.goal if job is not None and job.goal else None) \
            or agv.cur_node or agv.start_node

        try:
            reported = DeviationEvent.from_dict({
                "robot_id": robot_id, "event": event,
                "blocked_edge": blocked_edge, "current_pose": current_pose,
                "timestamp": timestamp,
            })
        except DeviationEventError as exc:
            return {"action": "rejected", "reason": str(exc)}

        decision = self._monitor.handle(reported, goal=target)
        answer = decision.to_dict()
        answer["goal"] = target

        if decision.action == "reroute":
            if job is None:
                answer["action"] = "ignore"
                answer["reason"] = "该车当前没有任务，无需改道"
            elif agv.agv_id in self._transit:
                # Mid-edge: switching the route now would teleport the vehicle. Hold it
                # and apply once the vehicle is standing on a node.
                self._pending_detour[agv.agv_id] = (target, decision.blocked_edge)
                answer["action"] = "deferred"
                answer["reason"] = "正在跨边行驶，抵达节点后再改道"
            elif not self._apply_detour(agv, target, decision.blocked_edge):
                # Unreachable from here: hand the task back so that allocation picks
                # another vehicle. _cancel_job already encodes when that is allowed
                # (an empty run may be abandoned, a loaded one may not), so reuse it
                # instead of writing a second, subtly different rule.
                answer["action"] = "reassign"
                if self._cancel_job(agv):
                    answer["reason"] = "从当前节点绕不开受阻通道，任务已退回待分配"
                    answer["requeued"] = True
                else:
                    answer["reason"] = "绕不开受阻通道，但任务不能退回（已装货或策略不允许）"
                    answer["requeued"] = False
        self._deviation_log.append(dict(answer, step=self._step_cursor))
        return answer

    def _apply_detour(self, agv: Agv, goal: str,
                      blocked: tuple[str, str]) -> bool:
        """Replaces the not-yet-traversed part of the current leg with a detour.

        Reservations of the abandoned tail are released: keeping them would block other
        vehicles on segments this vehicle will never drive.
        """
        job = self._jobs.get(agv.agv_id)
        here = agv.cur_node or agv.start_node
        if job is None or here is None:
            return False
        nodes, _meters = self._monitor.detour_from(here, goal, blocked)
        if not nodes:
            return False
        hops: list[Hop] = []
        t = self._step_cursor
        for src, dst in zip(nodes, nodes[1:]):
            span = self._hop_steps(src, dst)
            hops.append(Hop(src, dst, t, t + span))
            t += span
        job.hops = job.hops[:job.idx] + hops
        job.goal = goal
        # The new tail reserves lazily, from the first hop nobody has entered yet.
        job.reserved_upto = job.idx
        self._res.release_all(agv.agv_id)
        self._replans += 1
        return True

    def _apply_pending_detours(self) -> None:
        """Applies detours that were held because the vehicle was mid-edge."""
        if not self._pending_detour:
            return
        for agv_id, (goal, blocked) in list(self._pending_detour.items()):
            if agv_id in self._transit:
                continue
            agv = next((a for a in self.agvs if a.agv_id == agv_id), None)
            del self._pending_detour[agv_id]
            if agv is not None and self._jobs.get(agv_id) is not None:
                self._apply_detour(agv, goal, blocked)

    def _reschedule_step(self, step: int) -> None:
        """Retry pending urgent work at nodes, committing only feasible replacements.

        Physical executions keep their separate stop/reconciliation protocol.
        A loaded return holds its urgent request until unloading actually finishes.
        """
        if self.executions or self.config.reschedule_policy is ReschedulePolicy.NONE:
            return
        returning = {j.return_urgent_id for j in self._jobs.values() if j.return_urgent_id}
        preempted = {event[1] for event in self._preemption_events}
        urgent_waiting = [
            t for t in self.tasks
            if t.release_step <= step
            and t.is_urgent
            and t.status is TaskStatus.PENDING
            and t.task_id not in returning
            and t.task_id not in preempted
        ]
        if not urgent_waiting:
            return
        # Use available vehicles before interrupting a normal assignment.
        self._assign_step(step, urgent_only=True)
        for urgent in urgent_waiting:
            if urgent.status is not TaskStatus.PENDING:
                continue
            attempted: set[str] = set()
            while (victim := self._pick_preemption_victim(step, urgent, attempted)) is not None:
                attempted.add(victim.agv_id)
                victim_task = self._task_by_id[victim.cur_task]
                victim_picked = victim_task.status is TaskStatus.PICKED
                if not self._preempt_at_node(victim, urgent, step):
                    continue
                event = (step, urgent.task_id, victim.agv_id, victim_task.task_id, victim_picked)
                self._preemptions += 1
                self._preemption_events.append(event)
                self._step_preemptions.append(event[1:])
                break

    def _pick_preemption_victim(self, step: int, urgent: Task,
                                excluded: set[str] | None = None) -> Agv | None:
        """Choose a normal task at an actual node; loaded returns include their delay.

        Waiting urgent tasks are retried after arrival. The former mid-edge
        cancellation treated the departure node as the current position and was
        invalid: deleting a transit never models pulling over or returning.
        """
        best: Agv | None = None
        best_key: tuple[float, int] = (float("inf"), 0)

        for agv in self.agvs:
            if agv.agv_id in self._transit or (excluded and agv.agv_id in excluded):
                continue
            job = self._jobs.get(agv.agv_id)
            if job is None or job.return_urgent_id:
                continue
            # Not picked yet (to_pickup): preemptable under any policy;
            # loaded (to_dropoff): only PREEMPT_ANY preempts — that setting exists
            # specifically to quantify the material cost of "preempting a loaded
            # vehicle".
            if job.leg == "to_dropoff":
                if self.config.reschedule_policy is not ReschedulePolicy.PREEMPT_ANY:
                    continue
            elif job.leg != "to_pickup":
                continue
            task = self._task_by_id.get(job.task_id)
            if task is None or task.is_urgent:
                continue
            if job.leg == "to_pickup" and task.status is not TaskStatus.ASSIGNED:
                continue
            here = agv.cur_node or agv.start_node
            eta = self._travel_time(here, urgent.pickup)
            if job.leg == "to_dropoff":
                if task.status is not TaskStatus.PICKED or here == task.dropoff:
                    continue
                eta = (self._travel_time(here, task.pickup)
                       + self._unload_steps(task) * self.config.dt
                       + self._travel_time(task.pickup, urgent.pickup))
            if eta == float("inf"):
                continue
            remaining = len(job.hops) - job.idx   # the more, the less far it has gone and the smaller the waste
            key = (eta, -remaining)
            if key < best_key:
                best_key = key
                best = agv
        return best

    def _reserve_preemption_leg(self, agv: Agv,
                                plan: tuple[list[Hop], int, int]) -> tuple[list[Hop], int, int]:
        hops, arrival, _ = plan
        upto = 0
        if getattr(self.planner, "reservations_aware", False):
            win = self.config.reservation_window
            upto = len(hops) if win <= 0 else min(win, len(hops))
            for hop in hops[:upto]:
                self._reserve_hop(agv, hop)
        return hops, arrival, upto

    def _preempt_at_node(self, agv: Agv, urgent: Task, step: int) -> bool:
        """Prepare a feasible leg before changing any task, load or reservation."""
        job = self._jobs[agv.agv_id]
        task = self._task_by_id[job.task_id]
        loaded = task.status is TaskStatus.PICKED
        target = task.pickup if loaded else urgent.pickup
        wait_before = self._total_wait
        plan = self._plan_and_reserve(agv, agv.cur_node or agv.start_node, target, step, False)
        if plan is None:
            self._total_wait = wait_before
            return False
        if loaded:
            # Also prove the onward pickup is currently reachable before taking
            # this vehicle away from its original delivery. Neither leg is committed
            # by this probe; traffic is checked again after actual unloading.
            wait_after_return = self._total_wait
            onward = self._plan_and_reserve(agv, task.pickup, urgent.pickup,
                                            plan[1] + self._unload_steps(task), False)
            if onward is None:
                self._total_wait = wait_before
                return False
            self._total_wait = wait_after_return
            self._release_future_at_node(agv.agv_id)
            job.hops, arrival, job.reserved_upto = self._reserve_preemption_leg(agv, plan)
            job.idx = 0
            job.leg, job.goal, job.return_urgent_id = "to_return", task.pickup, urgent.task_id
            self._drop_end[agv.agv_id] = arrival + self._unload_steps(task)
            self._load_end.pop(agv.agv_id, None)
        else:
            if not self._cancel_job(agv):
                self._total_wait = wait_before
                return False
            self._accept_assignment(agv, urgent, step, self._reserve_preemption_leg(agv, plan))
        self._pending_detour.pop(agv.agv_id, None)
        self._blocked_since.pop(agv.agv_id, None)
        return True

    def _travel_time(self, a: str, b: str) -> float:
        """Cached static shortest travel time (in seconds)."""
        if a == b:
            return 0.0
        key = (a, b)
        if key not in self._travel_cache:
            d = self.network.shortest_travel_times(a, self._max_speed())
            for k, v in d.items():
                self._travel_cache[(a, k)] = v
            self._travel_cache[key] = d.get(b, float("inf"))
        return self._travel_cache[key]

    def _cancel_job(self, agv: Agv) -> bool:
        """Requeue an unpicked task at a node; never erase movement or a load.

        PREEMPT_ANY uses an executed return and unloading before requeueing.
        This guard also protects callers such as failed deviation recovery.
        """
        if self.executions or agv.agv_id in self._transit:
            return False
        job = self._jobs.get(agv.agv_id)
        if job is None:
            return False
        task = self._task_by_id[job.task_id]

        if job.leg != "to_pickup" or task.status is not TaskStatus.ASSIGNED:
            return False
        self._requeue_at_node(agv, task)
        return True

    def _requeue_at_node(self, agv: Agv, task: Task) -> None:
        """Release an empty vehicle, after cancellation or verified return unloading."""
        task.status = TaskStatus.PENDING
        task.assigned_agv = None
        task.assigned_step = None
        task.pickup_step = None
        agv.cur_task = None
        agv.state = AgvState.IDLE
        self._jobs.pop(agv.agv_id, None)
        self._load_end.pop(agv.agv_id, None)
        self._drop_end.pop(agv.agv_id, None)
        self._drop_arrive.pop(agv.agv_id, None)
        self._blocked_since.pop(agv.agv_id, None)
        self._release_future_at_node(agv.agv_id)

    def _release_future_at_node(self, owner: str) -> None:
        """Release the old route while preserving occupancy of the actual node."""
        self._res.release_all(owner)
        for hold in (self._node_hold.get(owner), self._node_claim.get(owner)):
            if hold is not None:
                self._res.reserve_node(*hold, owner)

    def _assign_step(self, step: int, *, urgent_only: bool = False) -> None:
        roster = getattr(self, '_fleet_admission', None)
        idle = [a for a in self.agvs if a.cur_task is None and a.agv_id not in self._transit
                and (roster is None or roster.eligible(a.agv_id))
                and (not self.executions or (self.execution_for(a.agv_id).bound
                                             and not self.execution_for(a.agv_id).fault))]
        if not idle:
            return
        returning = {j.return_urgent_id for j in self._jobs.values() if j.return_urgent_id}
        pending = [
            t for t in self.tasks
            if t.status is TaskStatus.PENDING and t.release_step <= step
            and (not urgent_only or t.is_urgent)
            and t.task_id not in returning
            and (not self.executions or not any(p['urgent_task_id'] == t.task_id
                                               for p in self._physical_preemptions.values()))
        ]
        if not pending:
            return
        pairs = self.allocator.allocate(
            self.network, idle, pending, step, self._max_speed(), self._travel_cache
        )
        for agv, task in pairs:
            if not self._try_assign(agv, task, step):
                # No feasible path can be planned this step; the task stays PENDING
                # and is retried next step.
                # This is a **retry counter**, not a task failure.
                self._planning_retries += 1

    def _advance_step(self, step: int) -> None:
        for agv in self.agvs:
            gate = self.execution_for(agv.agv_id)
            if gate is not None:
                if not gate.bound or gate.fault:
                    # Handling is an explicitly simulated dwell, not a hardware
                    # handshake. A fault must not silently consume its duration.
                    if gate.fault and agv.agv_id not in self._transit:
                        job = self._jobs.get(agv.agv_id)
                        if job is not None and job.idx >= len(job.hops):
                            timers = self._load_end if job.leg == "to_pickup" else self._drop_end
                            if timers.get(agv.agv_id, 0) >= step:
                                timers[agv.agv_id] += 1
                    agv.state = AgvState.BLOCKED
                    agv.blocked_steps += 1
                    continue
            # 1) In transit
            tr = self._transit.get(agv.agv_id)
            if tr is not None:
                src, dst, _depart, arrive = tr
                agv.busy_steps += 1
                arrived = (step + 1 >= arrive if gate is None else gate.state == "arrived")
                if arrived:
                    # Hard guarantee: verify once more against the **actual occupancy**
                    # before really entering the node. The reservation table can have
                    # blind spots due to concurrent subscription/cleanup, and this is
                    # the last gate.
                    occ = sum(
                        1 for b in self.agvs
                        if b.agv_id != agv.agv_id
                        and b.agv_id not in self._transit
                        and b.cur_node == dst
                    )
                    if occ >= self.network.node(dst).capacity:
                        # Target full: wait one more step on the edge and do not
                        # complete this transfer.
                        #
                        # This **must** also be counted as blocking: the deadlock on a
                        # single-lane ring network is stuck exactly on this branch (the
                        # vehicle has reached the target edge, but the target node is
                        # held by another vehicle). If blocking were only recorded on
                        # the "no path can be planned" branch, _break_deadlock would
                        # never see a timeout, and the task would hang forever
                        # (measured breaks=0, only 48 of 50 tasks completed).
                        # Transit already contributed one busy tick above. This
                        # tick also waits for traffic, without a second busy tick.
                        self._blocked += 1
                        agv.blocked_steps += 1
                        agv.state = AgvState.BLOCKED
                        if agv.agv_id not in self._blocked_since:
                            # A new blocking episode starts (it was not blocked before):
                            # count 1 event
                            self._conflict_wait_events += 1
                        self._blocked_since.setdefault(agv.agv_id, step)
                        continue
                    agv.cur_node = dst
                    if gate is None and self.config.reschedule_policy is ReschedulePolicy.PREEMPT_ANY:
                        job = self._jobs[agv.agv_id]
                        task = self._task_by_id.get(job.task_id)
                        if task is not None and task.status is TaskStatus.PICKED:
                            job.loaded_travel_seconds += self._hop_steps(src, dst) * self.config.dt
                        if job.leg == "to_return" and job.idx >= len(job.hops):
                            self._drop_end[agv.agv_id] = step + 1 + self._unload_steps(task)
                    # Note: do **not** release the arrival claim here.
                    # Later vehicles within the same step still rely on it to see "this
                    # node is already taken"; otherwise three vehicles cram into a
                    # capacity-2 aisle node. Releasing is left to ``_refresh_node_holds``
                    # on the next step, which also converts it into a dwell reservation.
                    del self._transit[agv.agv_id]
                    if gate is not None:
                        assert gate.consume_arrival()
                        self._res.transfer_pins(agv.agv_id, [dst])
                        if agv.agv_id in self._physical_detours:
                            gate.fail('actual node reached; explicit physical detour authorization required')
                        elif agv.agv_id in self._physical_preemptions:
                            gate.fail('actual node reached; explicit urgent task authorization required')
                        self._odometer += self.network.edge(src, dst).length
                        job = self._jobs[agv.agv_id]
                        if job.idx >= len(job.hops):
                            task = self._task_by_id[job.task_id]
                            if job.leg == "to_pickup":
                                self._load_end[agv.agv_id] = step + 1 + self._load_steps(task)
                            else:
                                self._drop_end[agv.agv_id] = step + 1 + self._unload_steps(task)
                continue

            job = self._jobs.get(agv.agv_id)
            if job is None:
                agv.idle_steps += 1
                agv.state = AgvState.IDLE
                continue

            if job.leg == "to_return" and job.idx >= len(job.hops):
                task = self._task_by_id[job.task_id]
                if agv.cur_node != task.pickup:
                    raise RuntimeError("loaded return finished away from original pickup")
                agv.busy_steps += 1
                agv.state = AgvState.UNLOADING
                if step < self._drop_end[agv.agv_id]:
                    continue
                urgent = self._task_by_id[job.return_urgent_id]
                self._material_waste_steps += job.loaded_travel_seconds
                self._requeue_at_node(agv, task)
                if urgent.status is TaskStatus.PENDING:
                    self._try_assign(agv, urgent, step)
                continue

            # 1.5) Park task: it ends once the route is finished and involves no task
            if job.leg == "park" and job.idx >= len(job.hops):
                agv.cur_task = None
                agv.state = AgvState.IDLE
                del self._jobs[agv.agv_id]
                agv.idle_steps += 1
                continue

            # 2) Loading dwell after reaching the pickup point
            load_end = self._load_end.get(agv.agv_id, 0)
            if (job.leg == "to_pickup" and job.idx >= len(job.hops)
                    and not self._handling_ready(agv, job, 'LOAD', step)):
                continue
            if job.leg == 'empty_yield' and job.idx >= len(job.hops) and step >= load_end:
                task = self._task_by_id[job.task_id]
                leg = self._plan_and_reserve(agv, agv.cur_node or agv.start_node, task.pickup, step,
                    getattr(self.planner, 'reservations_aware', False))
                if leg is None:
                    self._blocked += 1
                    agv.busy_steps += 1
                    agv.blocked_steps += 1
                    agv.state = AgvState.BLOCKED
                    if agv.agv_id not in self._blocked_since:
                        self._conflict_wait_events += 1
                        self._blocked_since[agv.agv_id] = step
                    continue
                job.hops, arrival, job.reserved_upto = leg
                job.leg, job.goal, job.idx = 'to_pickup', task.pickup, 0
                self._load_end[agv.agv_id] = arrival + self._load_steps(task)
            if job.leg == "to_pickup" and job.idx >= len(job.hops) and step < load_end:
                # The vehicle is already standing at the pickup point and loading has
                # begun: from this moment the load counts as being on board.
                # PICKED must be set here, otherwise the task stays ASSIGNED during
                # this period and ``_break_deadlock`` returns it to PENDING — but the
                # load has in fact been moved onto the vehicle, and the next vehicle
                # going to the pickup point finds nothing.
                task = self._task_by_id[job.task_id]
                if task.status is TaskStatus.ASSIGNED:
                    task.status = TaskStatus.PICKED
                    task.pickup_step = step
                agv.busy_steps += 1
                agv.state = AgvState.LOADING
                continue
            if job.leg in ("to_pickup", "loaded_yield") and job.idx >= len(job.hops) and step >= load_end:
                task = self._task_by_id[job.task_id]
                target = task.pickup if job.return_urgent_id else task.dropoff
                # Plan the second leg on the spot: reserve only now, to avoid locking
                # the drop-off area ahead of time
                leg2 = self._plan_and_reserve(
                    agv, agv.cur_node or agv.start_node, target, step,
                    getattr(self.planner, "reservations_aware", False),
                )
                if leg2 is None:
                    # The second leg cannot be planned: the vehicle can only stand at
                    # the pickup point and wait.
                    #
                    # This **must count toward the blocking timer**, otherwise it
                    # forms a real livelock: the vehicle occupies the workstation for a
                    # long time in the LOADING state, never enters _blocked_since, so
                    # _break_deadlock never sees it and the task is never delivered.
                    #
                    # Measured on the grid layout with 50 tasks / 6 vehicles / seed 0:
                    # four tasks all had B3 as their pickup point, B3 has capacity 2,
                    # two vehicles each sat on B3 failing to replan again and again, the
                    # other two tasks could never get a slot, completions froze at 46/50,
                    # the simulation idled on to max_steps (120k steps), and the whole
                    # script hung for 50 minutes.
                    agv.busy_steps += 1
                    agv.state = AgvState.LOADING
                    agv.blocked_steps += 1
                    self._blocked += 1
                    if agv.agv_id not in self._blocked_since:
                        self._conflict_wait_events += 1
                        self._blocked_since[agv.agv_id] = step
                    continue
                hops2, t_drop, upto2 = leg2
                if getattr(self.planner, "reservations_aware", False):
                    self._res.reserve_node(
                        target, Interval(t_drop, t_drop + self._unload_steps(task)), agv.agv_id
                    )
                job.hops = hops2
                job.idx = 0
                job.goal = target
                if job.leg == "to_pickup":
                    task.pickup_step = step
                job.leg = "to_return" if job.return_urgent_id else "to_dropoff"
                job.reserved_upto = upto2
                self._drop_end[agv.agv_id] = t_drop + self._unload_steps(task)
                task.status = TaskStatus.PICKED
                agv.state = AgvState.TO_DROPOFF

            # 3) Unloading dwell after reaching the drop-off point
            if job.leg == "to_dropoff" and job.idx >= len(job.hops):
                if not self._handling_ready(agv, job, 'UNLOAD', step):
                    continue
                drop_end = self._drop_end.get(agv.agv_id, 0)
                if agv.agv_id in self.handlings:
                    drop_end = max(drop_end, step)
                if step < drop_end:
                    agv.busy_steps += 1
                    agv.state = AgvState.UNLOADING
                    continue
                task = self._task_by_id[job.task_id]
                task.status = TaskStatus.COMPLETED
                task.complete_step = drop_end
                # Workstation processing: after unloading, the workstation is still
                # occupied by the workpiece for process_time.
                # The AGV does not wait along — it leaves once unloaded, and the
                # occupancy is borne by the workstation itself.
                psteps = self._process_steps()
                if psteps > 0 and self._has_processing(task.dropoff):
                    self._process_holds.setdefault(task.dropoff, []).append(
                        (drop_end, drop_end + psteps, agv.agv_id))
                agv.cur_task = None
                agv.state = AgvState.IDLE
                del self._jobs[agv.agv_id]
                for d in (self._load_end, self._drop_arrive, self._drop_end):
                    d.pop(agv.agv_id, None)
                agv.busy_steps += 1
                continue

            # 4) Take the next hop along the plan
            if job.idx >= len(job.hops):
                agv.idle_steps += 1
                continue
            self._extend_reservations(agv, job, step)
            hop = job.hops[job.idx]
            src = agv.cur_node or agv.start_node
            if src != hop.src:
                # The actual position does not match the first planned hop — this is a
                # **genuine fault** and is counted separately.
                self._hard_failures += 1
                self._task_by_id[job.task_id].status = TaskStatus.FAILED
                agv.cur_task = None
                agv.state = AgvState.IDLE
                del self._jobs[agv.agv_id]
                continue
            # Dwell time at the destination node: the final station loads/unloads,
            # intermediate stations stop for only 1 step
            is_final = job.idx == len(job.hops) - 1
            if is_final and job.leg == "to_pickup":
                dwell = self._load_steps()
            elif is_final and job.leg == "to_return":
                dwell = self._unload_steps(self._task_by_id[job.task_id])
            elif is_final and job.leg == "to_dropoff":
                dwell = self._unload_steps()
            else:
                dwell = 1

            if self._can_enter(agv, hop.src, hop.dst, step, dwell):
                # Early departure: migrate this hop's original reservations to the
                # actual instants, so a phantom time window is not held any longer
                if step != hop.depart:
                    self._res.release_edge(
                        hop.src, hop.dst, Interval(hop.depart, hop.arrive), agv.agv_id
                    )
                    if self.network.has_edge(hop.dst, hop.src):
                        self._res.release_edge(
                            hop.dst, hop.src, Interval(hop.depart, hop.arrive), agv.agv_id
                        )
                    self._res.release_node(
                        hop.dst, Interval(hop.arrive, hop.arrive + 1), agv.agv_id
                    )
                    hop.depart = step
                    hop.arrive = step + self._hop_steps(hop.src, hop.dst)
                    self._reserve_hop(agv, hop)
                self._commit_move(agv, hop.src, hop.dst, step, dwell)
                job.idx += 1
                agv.busy_steps += 1
                self._blocked_since.pop(agv.agv_id, None)
                agv.state = AgvState.TO_PICKUP if job.leg == "to_pickup" else AgvState.TO_DROPOFF
            else:
                self._blocked += 1
                agv.blocked_steps += 1
                agv.busy_steps += 1
                agv.state = AgvState.BLOCKED
                if agv.agv_id not in self._blocked_since:
                    self._conflict_wait_events += 1
                self._blocked_since.setdefault(agv.agv_id, step)

    def _record(self, step: int) -> None:
        if not self.config.record_trajectory:
            return
        pos: dict[str, str] = {}
        states: dict[str, str] = {}
        jobs: dict[str, str | None] = {}
        paths: dict[str, list[str]] = {}
        n_blocked = 0
        for a in self.agvs:
            tr = self._transit.get(a.agv_id)
            pos[a.agv_id] = f"{tr[0]}->{tr[1]}" if tr else (a.cur_node or "")
            states[a.agv_id] = a.state.value
            job = self._jobs.get(a.agv_id)
            jobs[a.agv_id] = job.task_id if job is not None else None
            # Remaining path = the sequence of hop destinations not yet covered by the
            # current leg.
            #
            # Note the semantics of idx: _Job.idx points at "the hop currently being
            # travelled".
            # The remaining path must therefore include the destination of the hop at
            # idx, otherwise the drawn line falls one segment short of what the vehicle
            # actually has to travel (this off-by-one only shows up when drawing).
            if job is not None and job.hops:
                rest = [h.dst for h in job.hops[job.idx:]]
                if rest:
                    paths[a.agv_id] = rest
            if a.state is AgvState.BLOCKED:
                n_blocked += 1
        counts = {"pending": 0, "assigned": 0, "picked": 0, "completed": 0}
        for task in self._task_by_id.values():
            key = task.status.value
            if key in counts:
                counts[key] += 1
        holds: dict[str, list[str]] = {}
        for node, items in self._res.snapshot(step, step + 1).items():
            holds[node] = [owner for _, _, owner in items]
        self._trajectory.append(
            StepRecord(
                step=step,
                agv_positions=pos,
                agv_states=states,
                agv_tasks=jobs,
                task_counts=counts,
                n_blocked=n_blocked,
                node_holds=holds,
                agv_paths=paths,
                preemptions=list(self._step_preemptions),
            )
        )
        self._step_preemptions.clear()
