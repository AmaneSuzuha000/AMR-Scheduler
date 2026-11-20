"""Instance generation (task specification R5.1/R5.2/R5.3).

The specification says "different task scales (**e.g.** 20/50/100 tasks) and AGV counts
(**e.g.** 2/4/6)".
The word "e.g." makes those examples, so scale is a parameter here rather than a hardcoded
3x3 grid.

Determinism
-------------
* Uses a private ``random.Random(seed)`` and never touches global random state.
* Tasks are emitted sorted by task_id.
* The same (config, network) always yields the same result, which is what makes paired
experiments possible at all.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum

from amr_sched.entities import Agv, MaterialType, Task, TaskPriority
from amr_sched.network import Network, NodeKind, build_demo_network


class DueModel(str, Enum):
    """How due times are generated -- the validity precondition for R5.5, on-time rate.

   The same schedule, under a different due-time definition, gives a completely different
   "on-time rate".
   If the definition itself is unreasonable -- one that omits queueing, say -- then the
   metric no longer measures scheduling quality,
   but how loose the due-date formula is. It therefore has to be made explicit and
   comparable.
    """

    #: Travel time only: due = release + static shortest travel time x slack.
    TRAVEL_ONLY = "travel_only"
    #: Travel time plus queueing estimated from fleet size. Closer to a real workshop's
    #: delivery dates.
    TRAVEL_QUEUE = "travel_queue"
    #: Fixed allowance: due = release + constant x slack, independent of the task itself.
    FIXED = "fixed"


@dataclass
class ScenarioConfig:
    """The scale settings for one experiment."""

    n_tasks: int = 20
    n_agvs: int = 2
    seed: int = 0
    #: Fraction of urgent tasks (task specification R2.5).
    urgent_ratio: float = 0.15
    #: Time window over which tasks are released, in steps.
    release_window: int = 600
    #: Due-date slack multiplier: due = release + round(static shortest time / dt * slack).
    deadline_slack: float = 3.0
    #: Due-time definition; see DueModel.
    due_model: DueModel = DueModel.TRAVEL_ONLY
    #: Under TRAVEL_QUEUE, converts each vehicle's share of the queue into waiting time.
    queue_factor: float = 1.0
    max_speed: float = 1.0
    dt: float = 1.0
    #: Main corridor capacity, the structural dimension of R5.3 corridor load.
    #: 2 = two lanes (default); 1 = single lane, where any head-on meeting can deadlock. Used
    #: for stress testing.
    aisle_capacity: int = 2
    #: Number of maintenance stations (1..6).
    n_workstations: int = 6
    #: Load and unload time in seconds, used only by TRAVEL_QUEUE to estimate queueing.
    load_time: float = 5.0
    unload_time: float = 5.0
    #: Whether to tag tasks with a material type -- the bogie / wheelset / brake valve the
    #: specification names.
    #:
    #: Defaults to False. The 4564 rows of existing experiments were all produced under
    #: material homogeneity;
    #: turning it on makes Task.material_type non-null, which changes load/unload durations
    #: and due times,
    #: so the historical data is no longer comparable. It must be **switched on
    #: explicitly**; the definition is never changed silently.
    material_mix: bool = False
    #: Mix of the three material types (bogie / wheelset / brake valve). Applies only when
    #: material_mix=True.
    material_weights: tuple[float, float, float] = (0.2, 0.45, 0.35)


def _pickup_nodes(network: Network) -> list[str]:
    """Nodes a load can be picked up from: the dismantling area and the buffers."""
    return [
        n.node_id
        for n in network.nodes_of_kind(NodeKind.DISASSEMBLY, NodeKind.BUFFER)
    ]


def _dropoff_nodes(network: Network) -> list[str]:
    """Nodes a load can be dropped at: maintenance stations, the assembly area, buffers."""
    return [
        n.node_id
        for n in network.nodes_of_kind(
            NodeKind.WORKSTATION, NodeKind.ASSEMBLY, NodeKind.BUFFER
        )
    ]


def generate_scenario(
    config: ScenarioConfig,
    network: Network | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    start_pool: Sequence[str] | None = None,
) -> tuple[Network, list[Task], list[Agv]]:
    """Generate (network, tasks, AGVs).

    pairs supplies an explicit set of **legal** pickup/drop-off pairs -- for the
    workshop scene these are the material-route endpoints (see
    WorkshopScene.task_endpoints). When it is given, tasks are sampled only from that set.

    The old behaviour (pick a node by kind, then any other node by kind) is fine on an
    abstract ring, but on a workshop it manufactures tasks that no process would ever
    create, such as "pick up at a workstation, deliver to the dismantling area". The
    task-endpoint legality of the scene is only meaningful if the generator respects it.
    """
    net = network or build_demo_network(
        n_workstations=config.n_workstations,
        aisle_capacity=config.aisle_capacity,
    )
    rng = random.Random(config.seed)
    # Dedicated to material sampling, isolated from the task-generation rng. See below.
    mat_rng = random.Random(config.seed * 7919 + 104729)

    legal: list[tuple[str, str]] | None = None
    if pairs is not None:
        legal = sorted(set(pairs))
        if not legal:
            raise ValueError("pairs is empty: no legal pickup/drop-off pair to sample from")
        for _p, _d in legal:
            if _p not in net.nodes or _d not in net.nodes:
                raise ValueError(f"pair {_p} -> {_d} names a node that is not in the network")

    pickups = sorted(_pickup_nodes(net))
    dropoffs = sorted(_dropoff_nodes(net))
    if legal is None and (not pickups or not dropoffs):
        raise ValueError("the network has no pickup or no dropoff nodes")

    # Cache the static travel times, used to set sensible due dates.
    dist_cache: dict[tuple[str, str], float] = {}

    def travel(a: str, b: str) -> float:
        if a == b:
            return 0.0
        key = (a, b)
        if key not in dist_cache:
            d = net.shortest_travel_times(a, config.max_speed)
            for k, v in d.items():
                dist_cache[(a, k)] = v
            dist_cache[key] = d.get(b, float("inf"))
        return dist_cache[key]

    tasks: list[Task] = []
    for i in range(config.n_tasks):
        if legal is not None:
            # One draw either way, so the release/due sequence stays a function of the seed.
            pickup, dropoff = legal[rng.randrange(len(legal))]
        else:
            pickup = rng.choice(pickups)
            # Avoid a self-loop: dropoff must differ from pickup.
            candidates = [d for d in dropoffs if d != pickup]
            dropoff = rng.choice(candidates)
        release = int(rng.random() * config.release_window)

        tt = travel(pickup, dropoff)
        if tt == float("inf"):
            raise ValueError(f"pickup {pickup} cannot reach dropoff {dropoff}")
        travel_steps = tt / config.dt
        if config.due_model is DueModel.TRAVEL_ONLY:
            lead = travel_steps * config.deadline_slack
        elif config.due_model is DueModel.TRAVEL_QUEUE:
            # Each vehicle takes n_tasks/n_agvs of the work, and each task costs roughly
            # "average travel time + load/unload time".
            # Note that queue_factor plays the role of slack: slack has to act monotonically
            # under all three definitions, otherwise within one experiment batch slack becomes
            # a knob that only works for some of them.
            per_agv = max(1.0, config.n_tasks / max(1, config.n_agvs))
            opp = travel_steps + (config.load_time + config.unload_time) / config.dt
            queue = per_agv * opp * config.queue_factor * config.deadline_slack
            lead = travel_steps * config.deadline_slack + queue
        elif config.due_model is DueModel.FIXED:
            lead = 600.0 * config.deadline_slack / 3.0
        else:  # pragma: no cover - defensive
            raise ValueError(f"unknown due-time definition: {config.due_model}")
        lead = max(1, int(round(lead)))
        due = release + lead

        urgent = rng.random() < config.urgent_ratio
        material = None
        if config.material_mix:
            # Materials are drawn from a **separate** rng, never the one above.
            #
            # Sharing it would mean material_mix=True consumes one extra random draw, which
            # shifts the
            # pickup/dropoff/release of every later task -- the A and B arms would no longer
            # be running the same tasks,
            # and the paired comparison collapses. The first version hit exactly this, and
            # the difference it reported contained the task difference.
            # A separate rng guarantees that under one seed the two arms get field-for-field
            # identical task sequences, differing only in material.
            material = mat_rng.choices(
                [MaterialType.BOGIE, MaterialType.WHEELSET, MaterialType.BRAKE_VALVE],
                weights=list(config.material_weights), k=1,
            )[0]
        tasks.append(
            Task(
                task_id=f"T{i:04d}",
                pickup=pickup,
                dropoff=dropoff,
                release_step=release,
                due_step=due,
                priority=TaskPriority.URGENT if urgent else TaskPriority.NORMAL,
                material_type=material,
            )
        )

    tasks.sort(key=lambda t: (t.release_step, t.task_id))

    # AGVs start parked in the buffers or the dismantling area, expanded by capacity when
    # there are not enough of them.
    if start_pool is not None:
        # An explicit parking area beats "any buffer node": queue slots and docking
        # positions are not where a fleet waits between jobs.
        starts = list(start_pool)
        if not starts:
            raise ValueError("start_pool is empty")
    else:
        pool: list[str] = []
        primary_kinds = {NodeKind.BUFFER, NodeKind.DISASSEMBLY, NodeKind.ASSEMBLY}
        for n in sorted(net.nodes.values(), key=lambda x: x.node_id):
            if n.kind in primary_kinds:
                pool.extend([n.node_id] * max(1, n.capacity))
        if len(pool) < config.n_agvs:
            for n in sorted(net.nodes.values(), key=lambda x: x.node_id):
                if n.kind not in primary_kinds:
                    pool.extend([n.node_id] * max(1, n.capacity))
        starts = pool
    agvs: list[Agv] = []
    for i in range(config.n_agvs):
        start = starts[i % len(starts)]
        agvs.append(
            Agv(agv_id=f"AGV{i:02d}", max_speed=config.max_speed, start_node=start)
        )

    return net, tasks, agvs
