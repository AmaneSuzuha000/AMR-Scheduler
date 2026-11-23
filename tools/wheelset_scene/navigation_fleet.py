"""Use the existing shared scheduler for this workshop's sensor/Nav2 execution."""
import math

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Agv, Task
from amr_sched.external_execution import ExternalExecution, ExternalHandling
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import Simulator, SimulationConfig

from .light_material_network import proposed_network
from .light_material_scheduler import compile_network


def build_sensor_fleet(layout=None, *, run_id, case='delivery', speed_mps=.2,
                       handling_source=None, inventory=None):
    """Keep the truth-feedback runner unchanged; sensor gates explicitly use map."""
    if case not in ('short', 'delivery'):
        raise ValueError('short or delivery case required')
    if not math.isfinite(speed_mps) or not 0 < speed_mps <= .2:
        raise ValueError('positive speed within the current Nav2 .20m/s bound required')
    layout = proposed_network() if layout is None else layout
    network = compile_network(layout)
    nodes = {n['id']: n for n in layout['nodes']}
    gates = {robot['id']: ExternalExecution(layout['id'], robot['id'],
        (nodes[robot['start']]['x'], nodes[robot['start']]['y']), run_id=run_id,
        source='localized', frame_id='map', namespace_routes=True)
        for robot in layout['robots']}
    robots = [Agv(robot['id'], max_speed=speed_mps, start_node=robot['start'],
                  load_time=.4, unload_time=.4) for robot in layout['robots']]
    tasks = ([Task('navigation_short_a', 'bay_west', 'west'),
              Task('navigation_short_b', 'bay_east', 'east')]
             if case == 'short' else
             [Task('inspection_consumables', 'bay_west', 'receiving'),
              Task('small_tools', 'bay_east', 'supply')])
    handlings = (None if handling_source is None else
                 {robot: ExternalHandling(gate, source=handling_source) for robot, gate in gates.items()})
    sim = Simulator(network, tasks, robots,
        SpaceTimeAStarPlanner(max_speed=speed_mps, dt=.2), GreedyAllocator(),
        SimulationConfig(dt=.2, max_steps=6000, planning_horizon=6000, reservation_window=0,
            load_time=.4, unload_time=.4, idle_retreat=False, deadlock_timeout=0,
            reschedule_policy=ReschedulePolicy.NONE), executions=gates, handlings=handlings,
        inventory=inventory)
    sim.reset()
    return sim


def build_line_side_fleet(layout=None, *, run_id, handling_source, initial=None, speed_mps=.2):
    """Wire the declared light-parts buffers and process supply to one fleet owner.

    Capacities and quantities are scenario units, not measured payloads. The
    caller must provide actual handling feedback; constructing this factory
    supplies neither motion observations nor a physical handoff mechanism.
    """
    from amr_sched.inventory import DeliveryInventory
    from .inventory_supply import InventorySupply

    inventory = DeliveryInventory({'receiving': 4, 'supply': 2}, {
        'inspection_consumables': dict(node_id='receiving', quantity=4),
        'small_tools': dict(node_id='supply', quantity=2)}, initial=initial)
    sim = build_sensor_fleet(layout, run_id=run_id, speed_mps=speed_mps,
                             handling_source=handling_source, inventory=inventory)
    return sim, InventorySupply(sim, {'magnetic': 'receiving', 'derust': 'supply'})
