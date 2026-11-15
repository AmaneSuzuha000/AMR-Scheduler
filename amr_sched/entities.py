"""Core domain objects: transport tasks and AGVs (task book R2.2).

**Cleanly separated** from RAWSim-O's warehouse objects (Order/Pod/Bot):
there are no pods and no picking stations here, only "move material from A to B".

Time convention
--------
Simulation time is an **integer step count**, one step being ``SimulationConfig.dt``
seconds.
Space-time search, the reservation table, and conflict checks are then exact integer
arithmetic, immune to accumulated floating-point error.
The result is multiplied back by dt at the end to get wall-clock seconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class TaskPriority(int, Enum):
    """Task priority. The task book requires urgent tasks to be inserted dynamically."""

    NORMAL = 0
    URGENT = 1


class TaskStatus(str, Enum):
    PENDING = "pending"        # released, not yet assigned
    ASSIGNED = "assigned"      # an AGV has it, not yet picked up
    PICKED = "picked"          # picked up, in transit
    COMPLETED = "completed"
    FAILED = "failed"          # cannot be completed (unreachable, etc.)




class MaterialType(str, Enum):
    """Railway maintenance material kinds, matching the three the task book names.

    Why distinguish them
    ------------
Bogies, wheelsets, and brake valves differ in weight, handling time, and which
workstations can hold them.
Task used to carry no such field, so every experiment treated material as
homogeneous.
That keeps the scheduling conclusions valid but leaves a visible gap against the task
book
(see the business-mapping audit, an internal document).

    Why the default is None
    ---------------
All 4280 existing rows were produced under the "homogeneous material" assumption.
Adding the field with a None default changes no existing run's behaviour,
so historical data stays comparable. It is set explicitly only when wanted.
    """

    BOGIE = "bogie"              # heaviest, slowest to handle
    WHEELSET = "wheelset"        # medium
    BRAKE_VALVE = "brake_valve"  # lightest, fastest to handle

    def typical_load_steps(self) -> int:
        """Typical loading steps at dt=1s. A business parameter for reference only; it
        takes no part in scheduling decisions."""
        return {"bogie": 120, "wheelset": 60, "brake_valve": 30}[self.value]
@dataclass
class Task:
    """One material transport task.

    Attributes
    ----------
    task_id:
        Unique identifier.
    pickup:
        Pick-up node id.
    dropoff:
        Drop-off node id.
    release_step:
        Earliest step at which the task may start. Urgent tasks appear mid-run with a
        larger release_step.
    due_step:
        Step by which completion is required; used for the on-time rate.
        ``None`` means no deadline, and it is excluded from the on-time rate.
    priority:
        See :class:`TaskPriority`.
    status:
        Lifecycle state.
    assigned_agv:
        Which vehicle took it.
    """

    task_id: str
    pickup: str
    dropoff: str
    release_step: int = 0
    due_step: Optional[int] = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    assigned_agv: Optional[str] = None

    #: Material kind. A None default means "homogeneous material", which is the basis of
    #: the existing 4280 rows.
    #: The field exists to match the bogies/wheelsets/brake valves the task book names,
    #: while the default keeps every existing run unchanged.
    material_type: Optional["MaterialType"] = None

    # Observed timestamps, filled in by the simulator
    assigned_step: Optional[int] = None
    pickup_step: Optional[int] = None
    complete_step: Optional[int] = None

    def __post_init__(self) -> None:
        if self.release_step < 0:
            raise ValueError(f"task {self.task_id}: release_step cannot be negative")
        if self.due_step is not None and self.due_step < self.release_step:
            raise ValueError(
                f"task {self.task_id}: due_step({self.due_step}) is earlier than "
                f"release_step({self.release_step})"
            )

    @property
    def is_urgent(self) -> bool:
        return self.priority is TaskPriority.URGENT

    def lead_time(self) -> Optional[int]:
        """Time allowed from pick-up to the deadline; None when there is no deadline."""
        if self.due_step is None:
            return None
        return self.due_step - self.release_step


class AgvState(str, Enum):
    IDLE = "idle"              # waiting for work
    TO_PICKUP = "to_pickup"    # running empty to the pick-up
    LOADING = "loading"        # picking up
    TO_DROPOFF = "to_dropoff"  # carrying it to the drop-off
    UNLOADING = "unloading"    # dropping off
    BLOCKED = "blocked"
    PARKING = "parking"        # held back by a conflict this step


@dataclass
class Agv:
    """One AGV.

    Attributes
    ----------
    agv_id:
        Unique identifier.
    max_speed:
        Maximum speed in m/s. The effective limit on an edge is
        ``min(max_speed, edge.speed_limit)``.
    start_node:
        Node it starts on.
    load_time / unload_time:
        Seconds needed to pick up and to drop off (converted to steps).
    """

    agv_id: str
    max_speed: float = 1.0
    start_node: str = ""
    load_time: float = 5.0
    unload_time: float = 5.0

    # Runtime state
    state: AgvState = AgvState.IDLE
    cur_node: Optional[str] = None
    cur_task: Optional[str] = None
    #: Nodes still to visit, excluding the current one
    route: list[str] = field(default_factory=list)
    #: Step at which the current node is left
    depart_step: Optional[int] = None
    #: Remaining handling steps (loading or unloading)
    remaining_job_steps: int = 0

    # Statistics, in steps
    busy_steps: int = 0
    idle_steps: int = 0
    blocked_steps: int = 0
    loaded_steps: int = 0
    deadhead_steps: int = 0

    def __post_init__(self) -> None:
        if self.max_speed <= 0:
            raise ValueError(f"AGV {self.agv_id}: max_speed must be > 0")
        if self.load_time < 0 or self.unload_time < 0:
            raise ValueError(f"AGV {self.agv_id}: handling times cannot be negative")

    def reset_stats(self) -> None:
        self.busy_steps = 0
        self.idle_steps = 0
        self.blocked_steps = 0
        self.loaded_steps = 0
        self.deadhead_steps = 0
