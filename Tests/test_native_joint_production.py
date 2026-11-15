"""SYSTEM12 software-only cross-domain, production-transport atomicity checks."""
import pytest
from amr_sched.entities import Agv, Task, TaskStatus
from amr_sched.network import build_demo_network
from amr_sched.allocation.greedy import GreedyAllocator
from amr_sched.planners.spacetime_astar import SpaceTimeAStarPlanner
from amr_sched.simulation import Simulator, SimulationConfig
from amr_sched.native_fleet_admission import NativeFleetAdmission
from amr_sched.native_joint_production import (NativeJointProduction,
                                                 ProductionOperation,ResourceChoice)

def make_scene(*, common_crew=False, predecessor=False):
    net=build_demo_network()
    tasks=[Task('K0','D','W1',release_step=15),
           Task('K1','B2','W5',release_step=15)]
    agvs=[Agv('A0',start_node='B1',load_time=0.,unload_time=0.),
          Agv('A1',start_node='B2',load_time=0.,unload_time=0.)]
    sim=Simulator(net,tasks,agvs,SpaceTimeAStarPlanner(),GreedyAllocator(),
                  SimulationConfig(max_steps=8000,idle_retreat=False,process_time=0.))
    sim.reset()
    fleet=NativeFleetAdmission(sim,initial_active={'A0'})
    operations=[
      ProductionOperation('P0',('K0',),
         (ResourceChoice('W1','crew0','tool0'),ResourceChoice('W2','crew1','tool1')),7),
      ProductionOperation('P1',('K1',),
         (ResourceChoice('W5','crew0' if common_crew else 'crew2','tool2'),
          ResourceChoice('W4','crew3','tool3')),11,
         'P0' if predecessor else None)]
    prod=NativeJointProduction(sim,operations,fleet=fleet)
    return sim,fleet,prod

def run_to_end(sim):
    while sim.step_once():
        pass
    return sim._build_result(sim._step_cursor)

def test_joint_change_at_t4_completes_transport_and_production():
    sim,fleet,prod=make_scene()
    for _ in range(4):assert sim.step_once()
    candidate=prod.prepare(op_id='P0',choice_index=1,expected_tick=4,expected_revision=0,
         fleet_action=dict(robot_id='A1',action='ACTIVATE',expected_tick=4,expected_revision=0))
    receipt=prod.commit(candidate)
    assert receipt['future_destination_changes']==1
    assert receipt['fleet']['action']=='ACTIVATE'
    assert sim._task_by_id['K0'].dropoff=='W2'
    result=run_to_end(sim)
    assert result.summary()['tasks_completed']==2
    assert result.collision_events==0
    assert prod.all_finished()
    assert sim._task_by_id['K0'].assigned_agv in ('A0','A1')
    assert any(e['event']=='PRODUCTION_STARTED' for e in prod.events)
    assert any(e['event']=='PRODUCTION_FINISHED' for e in prod.events)
    assert receipt['physical_authorization'] is False

def test_production_never_starts_before_real_kit_delivery():
    sim,fleet,prod=make_scene()
    prod.advance(100)
    assert all(x is None for x in prod.started.values())
    assert prod.events==[]

def test_kit_assignment_blocks_destination_change():
    sim,_,prod=make_scene()
    kit=sim._task_by_id['K0']
    kit.status=TaskStatus.ASSIGNED
    kit.assigned_agv='A0'
    with pytest.raises(ValueError,match='KIT_PICKUP_OR_ASSIGNMENT_COMMITTED'):
        prod.prepare(op_id='P0',choice_index=1,expected_tick=0,expected_revision=0)
    assert kit.dropoff=='W1'

def test_loaded_kit_immutable():
    sim,_,prod=make_scene()
    sim._task_by_id['K0'].status=TaskStatus.PICKED
    with pytest.raises(ValueError,match='KIT_PICKUP_OR_ASSIGNMENT_COMMITTED'):
        prod.prepare(op_id='P0',choice_index=1,expected_tick=0,expected_revision=0)

def test_cycle_and_unqualified_bay_rejected():
    sim,_,prod=make_scene()
    del sim._native_production  # construct an alternate malformed calendar for negative test only
    with pytest.raises(ValueError,match='cyclic production precedence'):
        NativeJointProduction(sim,[
            ProductionOperation('A',('K0',),(ResourceChoice('W1','C','T'),),2,'B'),
            ProductionOperation('B',('K1',),(ResourceChoice('W5','C','T'),),2,'A')])
    with pytest.raises(ValueError,match='UNQUALIFIED_RESOURCE_COMBINATION'):
        prod.prepare(op_id='P0',choice_index=5,expected_tick=0,expected_revision=0)

def test_stale_global_tick_refuses_both_changes():
    sim,fleet,prod=make_scene()
    candidate=prod.prepare(op_id='P0',choice_index=1,expected_tick=0,expected_revision=0,
         fleet_action=dict(robot_id='A1',action='ACTIVATE',expected_tick=0,expected_revision=0))
    sim.step_once()
    with pytest.raises(ValueError,match='STALE_OWNER_TICK'):
        prod.commit(candidate)
    assert fleet.active==frozenset({'A0'}) and prod.choice['P0']==0
    assert sim._task_by_id['K0'].dropoff=='W1'

def test_stale_fleet_version_refuses_production_update():
    sim,fleet,prod=make_scene()
    c=prod.prepare(op_id='P0',choice_index=1,expected_tick=0,expected_revision=0,
      fleet_action=dict(robot_id='A1',action='ACTIVATE',expected_tick=0,expected_revision=0))
    fleet.commit(fleet.prepare(robot_id='A1',action='ACTIVATE',expected_tick=0,expected_revision=0))
    with pytest.raises(ValueError,match='STALE_FLEET_REVISION'):
        prod.commit(c)
    assert sim._task_by_id['K0'].dropoff=='W1' and prod.revision==0

def test_rollback_when_admission_commit_fails(monkeypatch):
    sim,fleet,prod=make_scene()
    c=prod.prepare(op_id='P0',choice_index=1,expected_tick=0,expected_revision=0,
       fleet_action=dict(robot_id='A1',action='ACTIVATE',expected_tick=0,expected_revision=0))
    def fail(_):
        raise RuntimeError('injected_commit_failure')
    monkeypatch.setattr(fleet,'commit',fail)
    with pytest.raises(RuntimeError,match='injected_commit_failure'):
        prod.commit(c)
    assert sim._task_by_id['K0'].dropoff=='W1'
    assert prod.choice['P0']==0 and prod.revision==0 and prod.events==[]
    assert fleet.active==frozenset({'A0'})

def test_shared_crew_is_never_parallel_in_production():
    sim,fleet,prod=make_scene(common_crew=True)
    fleet.commit(fleet.prepare(robot_id='A1',action='ACTIVATE',expected_tick=0,expected_revision=0))
    result=run_to_end(sim)
    assert result.summary()['tasks_completed']==2 and prod.all_finished()
    a,b=prod.started['P0'],prod.started['P1']
    assert prod.finished['P0']<=b or prod.finished['P1']<=a
    assert result.collision_events==0

def test_predecessor_is_respected():
    sim,fleet,prod=make_scene(predecessor=True)
    fleet.commit(fleet.prepare(robot_id='A1',action='ACTIVATE',expected_tick=0,expected_revision=0))
    result=run_to_end(sim)
    assert result.summary()['tasks_completed']==2 and prod.all_finished()
    assert prod.started['P1']>=prod.finished['P0']

def test_reject_any_physical_execution():
    sim,_,_=make_scene()
    sim.executions={'A0':object()}
    with pytest.raises(ValueError,match='PHYSICAL_CROSS_DOMAIN_ATOMICITY_NOT_VERIFIED'):
        NativeJointProduction(sim,[])

def test_reset_refuses_to_erase_running_production():
    sim,_,_=make_scene()
    with pytest.raises(RuntimeError,match='production commitments are bound'):
        sim.reset()
