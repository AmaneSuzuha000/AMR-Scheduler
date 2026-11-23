"""Causal feedback/failure checks for the first wheelset process capability."""
from tools.wheelset_scene.process_runtime import Operation,ProcessController


def controller():
    return ProcessController([Operation('a','machine_a',(1,2,0),2),
        Operation('b','machine_b',(3,2,0),1)],run_id='run',transfer_timeout_s=10)


def arrive(c,now,**changes):
    data=dict(run_id='run',command_id=c.command['command_id'],pose=c.command['target'],
        observed_at=now,received_at=now)
    data.update(changes)
    return c.observe(**data)


def test_no_service_before_confirmed_arrival_and_no_early_completion():
    c=controller();c.start(0);c.tick(1)
    assert c.state=='transferring'
    assert not arrive(c,2,pose=(0,0,0))
    assert not any(e['event']=='DEVICE_BUSY' for e in c.events)
    assert arrive(c,3);c.tick(4.99)
    assert c.state=='processing' and c.locked_station=='a'
    c.tick(5);assert c.state=='transferring' and c.command['operation']=='b'
    assert arrive(c,6);c.tick(7)
    assert c.state=='complete' and c.report()['completed_operations']==['a','b']
    c.tick(8);assert sum(e['event']=='CYCLE_COMPLETE' for e in c.events)==1


def test_wrong_run_old_command_and_stale_feedback_cannot_start_work():
    c=controller();c.start(1)
    assert not arrive(c,2,run_id='old')
    assert not arrive(c,3,observed_at=0)
    old=c.command['command_id'];assert arrive(c,4);c.tick(6)
    assert not arrive(c,7,command_id=old)
    assert c.state=='transferring' and c.locked_station=='b'


def test_failure_retains_station_and_never_reports_completion():
    c=controller();c.start(0)
    assert not arrive(c,1,success=False)
    c.tick(100)
    assert c.state=='failed' and c.locked_station=='a'
    assert c.report()['completed_operations']==[]
    assert not any(e['event']=='CYCLE_COMPLETE' for e in c.events)


def test_late_arrival_cannot_outrun_transfer_timeout():
    c=controller();c.start(0)
    assert not arrive(c,10)
    assert c.state=='failed' and c.fault=='transfer_timeout'
