"""Compile this scene's dedicated road into the existing AMR scheduler.

The optional reference executor is synthetic and deliberately late. It checks
adapter wiring and feedback-owned occupancy before physical robot integration.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.entities import Agv,Task
from amr_sched.external_execution import ExternalExecution
from amr_sched.network import Network,Node,NodeKind
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.reschedule import ReschedulePolicy
from amr_sched.simulation import Simulator,SimulationConfig
from .light_material_network import proposed_network


def compile_network(layout):
    net=Network();kinds={'handoff':NodeKind.DOCK,'aisle':NodeKind.AISLE,'buffer':NodeKind.BUFFER}
    nodes={n['id']:n for n in layout['nodes']}
    for n in nodes.values():net.add_node(Node(n['id'],n['x'],n['y'],kinds[n['kind']],capacity=1))
    for lane in layout['lanes']:
        a,b=nodes[lane['src']],nodes[lane['dst']];length=math.hypot(a['x']-b['x'],a['y']-b['y'])
        if not lane['two_way']:raise ValueError('This road declares two-way physical lanes')
        net.add_bidirectional(lane['src'],lane['dst'],length,speed_limit=lane['speed_limit_mps'],capacity=lane['capacity'])
    return net


def build_scheduler(layout=None,*,source='synthetic_reference',run_id='wheelset-light-reference'):
    layout=proposed_network() if layout is None else layout;net=compile_network(layout)
    nodes={n['id']:n for n in layout['nodes']}
    gates={r['id']:ExternalExecution(layout['id'],r['id'],(nodes[r['start']]['x'],nodes[r['start']]['y']),
        frame_id='wheelset_world',run_id=run_id,namespace_routes=True,source=source) for r in layout['robots']}
    robots=[Agv(r['id'],max_speed=.5,start_node=r['start'],load_time=.4,unload_time=.4) for r in layout['robots']]
    # First delivery starts at each private bay; endpoints remain separate after
    # unloading. This reference case tests opposing access to the shared road.
    tasks=[Task('inspection_consumables','bay_west','receiving'),Task('small_tools','bay_east','supply')]
    sim=Simulator(net,tasks,robots,SpaceTimeAStarPlanner(max_speed=.5,dt=.2),GreedyAllocator(),
        SimulationConfig(dt=.2,max_steps=6000,planning_horizon=6000,reservation_window=0,
            load_time=.4,unload_time=.4,idle_retreat=False,deadlock_timeout=0,reschedule_policy=ReschedulePolicy.NONE),executions=gates)
    sim.reset();return sim


def reference(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False);layout=proposed_network();sim=build_scheduler(layout)
    motions={};positions={r:gate.start_xy for r,gate in sim.executions.items()};trace=[];eta_overruns=0
    for step in range(6000):
        now=step*.2
        for robot,gate in sim.executions.items():
            command=gate.command;event='POSE';xy=positions[robot]
            if command:
                token=command['route_id']
                if token not in motions:
                    distance=math.dist(xy,command['target_xy'])
                    motions[token]=dict(start=now,source=xy,duration=distance/command['speed']*1.4+.2)
                action=motions[token];fraction=min(1.,(now-action['start'])/action['duration'])
                xy=tuple(a+(b-a)*fraction for a,b in zip(action['source'],command['target_xy']));positions[robot]=xy
                if fraction==1:event='ARRIVED'
                elif now-action['start']>action['duration']/1.4:eta_overruns+=1
            result=sim.execution_feedback({**gate.envelope(),'event':event,'pose':[*xy,0.],
                'seq':gate.last_seq+1,'timestamp':now},now)
            if not result['accepted']:raise RuntimeError(str(result))
        sim.step_once();pins=sim._res.pins()
        assert all(len(owners)==1 for group in pins.values() for owners in group.values())
        active=[g.command for g in sim.executions.values() if g.command]
        assert len({tuple(sorted(c['nodes'])) for c in active})==len(active)
        if any(g.fault for g in sim.executions.values()):raise RuntimeError('Reference feedback fault')
        trace.append(dict(step=step,time_s=now,poses=dict(positions),pins=pins,
            commands={r:g.command for r,g in sim.executions.items()},
            tasks={t.task_id:t.status.value for t in sim.tasks}))
        if sim._all_done():break
    passed=sim._all_done() and eta_overruns>0
    content=''.join(json.dumps(row)+'\n' for row in trace).encode();(output/'trace.jsonl').write_bytes(content)
    result=dict(passed=passed,completed_tasks=[t.task_id for t in sim.tasks if t.status.value=='completed'],
        elapsed_s=now,eta_overrun_observations=eta_overruns,routes_executed=len(motions),network=layout,
        trace_sha256=hashlib.sha256(content).hexdigest(),
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['light_material_scheduler.py','light_material_network.py']},
        claim_boundary='Existing AMR scheduler, two identity-gated execution adapters, synthetic late movement feedback on this scene road. No ROS/Gazebo robot execution, physical traffic, localization or equipment delivery acceptance.')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    r=reference(args.output);print(json.dumps({k:r[k] for k in ['passed','completed_tasks','elapsed_s','eta_overrun_observations','routes_executed']}))
    if not r['passed']:raise SystemExit(1)


if __name__=='__main__':main()
