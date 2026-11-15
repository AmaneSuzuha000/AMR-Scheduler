"""Real spawned-process inventory protocol, with synthetic motion observations."""
import multiprocessing

from tools.wheelset_scene.whole_line_ipc import SupplyManager,SharedSupply


def test_spawned_supply_manager_keeps_handoff_gates_and_atomic_single_consumption():
    ctx=multiprocessing.get_context('spawn')
    with SupplyManager(ctx=ctx) as manager:
        supply=SharedSupply(manager.MaterialSupply('r',initial={'derust':0,'magnetic':0},
            destinations={'derust':(21.,53.,1.),'magnetic':(35.45,53.,1.)}))
        c=dict(run_id='r',task_id='inspection_consumables',kind='MOVE',nodes=['east','receiving'],
            source='gazebo_truth',route_id='r:scout_a:3',robot_id='scout_a',target_xy=[60.,69.2])
        o=dict(pose=[60.,69.2,0.],linear_speed=0.,angular_speed=0.,sim_time_s=10.)
        assert supply.arrived(c['task_id'],'magnetic',4,c,o,now=10.)
        assert not supply.can_start('a','magnetic',10.)
        handoff=supply.begin_handoff(c['task_id'],now=10.);assert handoff
        target=supply.destinations['magnetic'];o=dict(time_s=11.,xyz=target,velocity=[0.,0.,0.])
        assert not supply.commit('old',o,target,now=11.)
        assert supply.stock['magnetic']==0
        assert supply.commit(handoff['token'],o,target,now=11.)
        assert not supply.commit(handoff['token'],o,target,now=11.)
        assert supply.can_start('a','magnetic',11.)
        supply.consume('a','magnetic','r:a:device',11.)
        assert supply.stock['magnetic']==3 and ('a','magnetic') in supply.consumed
        # Callers cannot modify the manager's inventory through returned copies.
        snapshot=supply.stock;snapshot['magnetic']=99
        assert supply.stock['magnetic']==3
        assert len([e for e in supply.events if e['event']=='MATERIAL_CONSUMED'])==1


def test_localized_receiving_proof_survives_process_boundary_without_committing_stock():
    from Tests.test_wheelset_navigation_dock import receipt
    ctx=multiprocessing.get_context('spawn');c,o,p=receipt()
    with SupplyManager(ctx=ctx) as manager:
        supply=SharedSupply(manager.MaterialSupply('test',initial={'derust':0,'magnetic':0},dock_source='localized'))
        assert supply.arrived(c['task_id'],'magnetic',4,c,o,now=10.,dock_proof=p)
        assert supply.stock['magnetic']==0 and not supply.can_start('ws001','magnetic',10.)
        assert supply.events[0]['dock_proof']==p
        p['state_query']['request']['name']='wrong'
        assert supply.events[0]['dock_proof']['state_query']['request']['name']=='scout_a'
        assert not supply.arrived(c['task_id'],'magnetic',4,c,o,now=10.,dock_proof=p)
