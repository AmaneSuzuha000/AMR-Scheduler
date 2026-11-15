"""SYSTEM12 small, non-publication, native event-driven experiment.

Original scenario / plan is identical across policies within each release
profile. Robot A1 becomes available only at release_tick, so an always-on A1
'static' arm is physically INELIGIBLE and must not be reported as a legal rival.
"""
from __future__ import annotations
import argparse,json,math,statistics,time
from pathlib import Path
import sys
# Running this directly as python tools/research/... must find the repository root.
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from amr_sched.entities import Agv,Task,TaskStatus
from amr_sched.network import build_demo_network
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import Simulator,SimulationConfig
from amr_sched.native_fleet_admission import NativeFleetAdmission
from amr_sched.native_joint_production import (NativeJointProduction,ProductionOperation,ResourceChoice)

def one(release_tick, policy):
    net=build_demo_network()
    tasks=[Task('K0','D','W1',release_step=0),
           Task('K1','B2','W5',release_step=release_tick)]
    agvs=[Agv('A0',start_node='B1',load_time=0.,unload_time=0.),
          Agv('A1',start_node='B2',load_time=0.,unload_time=0.)]
    sim=Simulator(net,tasks,agvs,SpaceTimeAStarPlanner(),GreedyAllocator(),
             SimulationConfig(max_steps=3500,idle_retreat=False,process_time=0.,
                              record_trajectory=False))
    sim.reset()
    fleet=NativeFleetAdmission(sim,initial_active={'A0'})
    prod=NativeJointProduction(sim,[
        ProductionOperation('P0',('K0',),(ResourceChoice('W1','C0','T0'),),90),
        ProductionOperation('P1',('K1',),(ResourceChoice('W5','C0','T1'),
                              ResourceChoice('W4','C1','T1')),50)],fleet=fleet)
    # Clairvoyant static production selection is a stronger, explicitly
    # labelled reference. It does not activate the unavailable A1 early.
    if policy=='PRECOMMITTED_FLEX':
        prod.commit(prod.prepare(op_id='P1',choice_index=1,expected_tick=0,expected_revision=0))
    step_ms=[];receipt=None;transaction_ms=None;observations=[]
    while True:
        t_step0=time.perf_counter_ns()
        advanced=sim.step_once()
        step_ms.append((time.perf_counter_ns()-t_step0)/1e6)
        if not advanced:break
        positions={a.agv_id:a.cur_node for a in sim.agvs if a.agv_id not in sim._transit}
        observations.append(dict(tick=sim._step_cursor,node_positions=positions,
              transit={rid:list(tr) for rid,tr in sim._transit.items()}))
        # The fixture models *exogenous* robot availability: A1 is unavailable
        # until release_tick, independent of policy and of future disturbances.
        if sim._step_cursor==release_tick and policy != 'STATIC_ONE':
            t_commit0=time.perf_counter_ns()
            if policy in ('EVENT_FLEET','PRECOMMITTED_FLEX'):
                candidate=fleet.prepare(robot_id='A1',action='ACTIVATE',
                         expected_tick=sim._step_cursor,expected_revision=fleet.revision)
                receipt=fleet.commit(candidate)
            elif policy=='EVENT_JOINT':
                candidate=prod.prepare(op_id='P1',choice_index=1,
                      expected_tick=sim._step_cursor,expected_revision=prod.revision,
                      fleet_action=dict(robot_id='A1',action='ACTIVATE',
                            expected_tick=sim._step_cursor,expected_revision=fleet.revision))
                receipt=prod.commit(candidate)
            else:raise ValueError('unsupported policy')
            transaction_ms=(time.perf_counter_ns()-t_commit0)/1e6
        # A single step timer below is advisory, and does not certify hard
        # real-time. Measured separate from candidate transactions above.
        # No sleep/timer loop: these are wall measurements on an offline CI runner.
    result=sim._build_result(sim._step_cursor)
    task_records={t.task_id:dict(status=t.status.value,assigned_agv=t.assigned_agv,
          pickup_step=t.pickup_step,complete_step=t.complete_step,dropoff=t.dropoff)
          for t in sim.tasks}
    complete=(result.summary()['tasks_completed']==len(tasks) and prod.all_finished())
    resource_valid=True
    for a in prod.operations:
        for b in prod.operations:
            if a>=b or prod.started[a] is None or prod.started[b] is None:continue
            x=prod.operations[a].choices[prod.choice[a]]
            y=prod.operations[b].choices[prod.choice[b]]
            sharing=x.crew==y.crew or x.tool==y.tool or x.bay==y.bay
            overlapping=(prod.started[a]<prod.finished[b] and
                         prod.started[b]<prod.finished[a]) if prod.all_finished() else False
            if sharing and overlapping:resource_valid=False
    node_cap={node_id:node.capacity for node_id,node in sim.network.nodes.items()}
    independent_node_capacity_violations=[]
    for obs in observations:
        occ={}
        for node in obs['node_positions'].values():
            occ[node]=occ.get(node,0)+1
        for node,n in occ.items():
            if n>node_cap[node]:
                independent_node_capacity_violations.append((obs['tick'],node,n))
    valid=(resource_valid and not independent_node_capacity_violations and result.collision_events==0 and
           all((prod.started[k] is None or
                all(task_records[kit]['complete_step'] is not None and
                task_records[kit]['complete_step']<=prod.started[k]
                    for kit in op.kits)) for k,op in prod.operations.items()))
    if not valid:raise AssertionError('transport/production invariant failure')
    return dict(release_tick=release_tick,policy=policy,finished=complete,
        makespan_step=sim._step_cursor,transport_completed=result.summary()['tasks_completed'],
        production_completed=sum(v is not None for v in prod.finished.values()),
        collision_events=result.collision_events,source='native_software_simulation',
        outcome_task_states=task_records,production_started=prod.started,
        production_finished=prod.finished,production_events=prod.events,
        fleet_events=fleet.events,commit=receipt,
        observations=observations,node_capacity_violations=independent_node_capacity_violations,
        node_capacities=node_cap,
        step_p99_ms=sorted(step_ms)[min(len(step_ms)-1,math.ceil(len(step_ms)*.99)-1)],
        max_step_ms=max(step_ms),transaction_gate_ms=transaction_ms,
        timing_scope='software simulator and eligibility+commit only; no physical control or full candidate search')

def matrix():
    rows=[one(t,policy) for t in (40,60,80)
          for policy in ('STATIC_ONE','EVENT_FLEET','EVENT_JOINT','PRECOMMITTED_FLEX')]
    pairs=[]
    for t in (40,60,80):
        cells={r['policy']:r for r in rows if r['release_tick']==t}
        pairs.append(dict(release_tick=t,
            task_and_production_completion={k:v['finished'] for k,v in cells.items()},
            delta_event_fleet_vs_static=cells['EVENT_FLEET']['makespan_step']-
                                        cells['STATIC_ONE']['makespan_step'],
            delta_event_joint_vs_fleet=cells['EVENT_JOINT']['makespan_step']-
                                        cells['EVENT_FLEET']['makespan_step'],
            delta_joint_vs_clairvoyant_static=cells['EVENT_JOINT']['makespan_step']-
                                        cells['PRECOMMITTED_FLEX']['makespan_step']))
    return dict(schema='system12-native-dynamic-research-v1',
      origin='two-kit synthetic workshop; NOT real industry, not holdout, not a hardware test',
      availability='A1 unavailable before release_tick; no physical motor dispatch',
      methods={'STATIC_ONE':'one eligible vehicle for entire run',
               'EVENT_FLEET':'A1 accepted when availability event occurs',
               'EVENT_JOINT':'same activation + qualified P1 bay/crew change',
               'PRECOMMITTED_FLEX':'qualified P1 choice known at tick0, A1 still activates only after event; clairvoyant benchmark'},
      paired_profiles=pairs,episodes=rows,
      scientific_claim='mechanism and confounder diagnosis only; no SOTA or statistical claims')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();out=matrix();a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(out,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({'profiles':out['paired_profiles'],'rows':len(out['episodes'])}))
