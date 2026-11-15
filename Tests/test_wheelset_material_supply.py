from tools.wheelset_scene.material_supply import MaterialSupply,SuppliedRuntime
from tools.wheelset_scene.resource_runtime import Place


def receipt():
    command=dict(run_id='test',task_id='inspection_consumables',kind='MOVE',nodes=['east','receiving'],
        source='gazebo_truth',route_id='test:scout_a:3',robot_id='scout_a',target_xy=[60.,69.2])
    observation=dict(pose=[60.,69.2,0.],linear_speed=0.,angular_speed=0.,sim_time_s=10.)
    return command,observation


def test_supply_does_not_unlock_at_dock_or_on_wrong_handoff_and_deduplicates():
    supply=MaterialSupply('test');c,o=receipt()
    assert not supply.arrived(c['task_id'],'magnetic',4,c,{**o,'linear_speed':.2},now=10.)
    assert not supply.arrived(c['task_id'],'magnetic',4,c,o,now=11.)
    assert not supply.arrived(c['task_id'],'derust',4,c,o,now=10.)
    assert supply.arrived(c['task_id'],'magnetic',4,c,o,now=10.)
    assert not supply.arrived(c['task_id'],'magnetic',4,c,o,now=10.)
    assert not supply.can_start('ws001','magnetic',10.)
    handoff=supply.begin_handoff(c['task_id'],now=10.)
    target=(35.45,50.,1.)
    observed=dict(time_s=20.,xyz=list(target),velocity=[0.,0.,0.])
    assert not supply.commit('old',observed,target,now=20.)
    assert not supply.commit(handoff['token'],{**observed,'velocity':[.1,0.,0.]},target,now=20.)
    assert not supply.commit(handoff['token'],observed,(0.,0.,0.),now=20.)
    assert supply.stock['magnetic']==0
    assert supply.commit(handoff['token'],observed,target,now=20.)
    assert not supply.commit(handoff['token'],observed,target,now=20.)
    assert supply.stock['magnetic']==4


def test_device_waits_for_supply_and_recovery_does_not_charge_twice():
    supply=MaterialSupply('test',initial=dict(derust=0,magnetic=0))
    runtime=SuppliedRuntime([Place('magnetic',1,(0.,0.,0.),True),Place('out',1,(1.,0.,0.))],supply=supply,run_id='process')
    assert runtime.admit('ws001',0.)
    assert runtime.dispatch(0.)==[] and runtime.jobs['ws001']['state']=='ready'
    c,o=receipt();assert supply.arrived(c['task_id'],'magnetic',4,c,o,now=10.)
    h=supply.begin_handoff(c['task_id'],now=10.)
    assert runtime.dispatch(10.)==[]
    assert supply.commit(h['token'],dict(time_s=20.,xyz=[35.45,50.,1.],velocity=[0.,0.,0.]),(35.45,50.,1.),now=20.)
    action,=runtime.dispatch(20.);assert supply.stock['magnetic']==3
    assert not runtime.observe(action['id'],now=21.,observed_at=21.,success=False)
    assert runtime.confirm_stopped(action['id'],now=22.,observed_at=22.,stopped=True)
    replacement=runtime.recover(action['id'],22.)
    assert replacement and supply.stock['magnetic']==3
    assert runtime.observe(replacement,now=23.,observed_at=23.,device_complete=True)
    assert not runtime.observe(action['id'],now=23.,observed_at=23.,device_complete=True)


def test_equipment_side_handoff_uses_configured_target_and_keeps_old_default():
    target=(35.45,53.,1.)
    supply=MaterialSupply('test',destinations={'magnetic':target,'derust':(21.,53.,1.)})
    c,o=receipt();assert supply.arrived(c['task_id'],'magnetic',4,c,o,now=10.)
    handoff=supply.begin_handoff(c['task_id'],now=10.)
    old=(35.45,50.,1.);observed=dict(time_s=20.,xyz=list(old),velocity=[0.,0.,0.])
    assert not supply.commit(handoff['token'],observed,old,now=20.)
    assert not supply.commit(handoff['token'],observed,target,now=20.)
    assert supply.stock['magnetic']==0
    observed['xyz']=list(target)
    assert supply.commit(handoff['token'],observed,target,now=20.)
    assert MaterialSupply('old').destinations['magnetic']==old
