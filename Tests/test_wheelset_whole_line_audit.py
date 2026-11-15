"""Synthetic evidence corruption checks; these are not Gazebo run results."""
import copy
import json

import pytest

from tools.wheelset_scene.resource_runtime import ResourceRuntime
from tools.wheelset_scene.whole_line_runtime import whole_line_places
from tools.wheelset_scene.whole_line_evidence import Evidence
from tools.wheelset_scene.verify_whole_line import verify_resources
from tools.wheelset_scene.whole_line_worker_audit import bind_press_samples


def test_global_press_samples_bind_both_workers_before_incremental_replay():
    events=[dict(event='COMMAND_ISSUED',command=dict(id=job+':press',job=job,kind='device',station='press'))
            for job in ['ws001','ws002']]
    samples=[dict(command_id=job+':press',fixture=dict(entity=job),side='left') for job in ['ws002','ws001']]
    groups=bind_press_samples(events,samples,['ws001','ws002'])
    assert groups['ws001']==[samples[1]] and groups['ws002']==[samples[0]]


@pytest.mark.parametrize('corruption',['unknown_command','wrong_fixture','nonpress_command','duplicate_command'])
def test_press_partition_rejects_unknown_or_misbound_global_records(corruption):
    events=[dict(event='COMMAND_ISSUED',command=dict(id='press',job='ws001',kind='device',station='press'))]
    samples=[dict(command_id='press',fixture=dict(entity='ws001'),side='right')]
    if corruption=='unknown_command':samples[0]['command_id']='other'
    elif corruption=='wrong_fixture':samples[0]['fixture']['entity']='ws002'
    elif corruption=='nonpress_command':events[0]['command']['kind']='transfer'
    else:events.append(copy.deepcopy(events[0]))
    with pytest.raises(AssertionError):bind_press_samples(events,samples,['ws001','ws002'])


def sample(name,t):
    return dict(entity=name,time_s=t,xyz=[0.,0.,0.],velocity=[0.,0.,0.],
        spin=0.,omega=0.,probe=0.,probe_velocity=0.,servo_sequence=1,
        servo_fault=False,applied_effort={'x':0.})


def raw_fixture(directory):
    definitions={'tote':dict(axes=['x'],origin=[0.,0.,0.],kind='part')}
    samples=[]; commands=[]
    for seq in range(1,131):
        t=seq*.05; obs=sample('tote',t); obs['servo_sequence']=max(1,seq-1)
        samples.append(dict(time_s=t,observations={'tote':obs},targets={'tote':dict(xyz=[0.,0.,0.])},commands={}))
        commands.append(dict(entity='tote',sequence=seq,time_s=t,axis_values=[0.,1.],execution_id=None,disabled_axes=[]))
    return definitions,samples,commands


def save(directory,name,values):
    (directory/(name+'.jsonl')).write_text(''.join(json.dumps(v)+'\n' for v in values))


def test_streaming_raw_accepts_consistent_feedback_and_stopped_event(tmp_path):
    definitions,samples,commands=raw_fixture(tmp_path)
    save(tmp_path,'samples',samples); save(tmp_path,'native_commands',commands)
    evidence=Evidence(tmp_path,definitions)
    evidence.require('tote',samples[20]['observations']['tote'],now=samples[20]['time_s'],target=[0.,0.,0.])
    result,disabled=evidence.read([],{})
    assert result['joint_sample_groups']==130 and result['native_command_records']==130 and not disabled


@pytest.mark.parametrize('corruption',['duplicate_sequence','false_target','wrong_execution','force_on_disabled_axis','future_ack','unexplained_retirement'])
def test_streaming_raw_rejects_semantic_corruption(tmp_path,corruption):
    definitions,samples,commands=raw_fixture(tmp_path)
    if corruption=='duplicate_sequence': commands[20]['sequence']=commands[19]['sequence']
    elif corruption=='false_target': commands[20]['axis_values'][0]=1.
    elif corruption=='wrong_execution': commands[20]['execution_id']='other-job'
    elif corruption=='force_on_disabled_axis':
        commands[19]['disabled_axes']=['x']; commands[19]['axis_values'][1]=0.
        samples[20]['observations']['tote']['applied_effort']['x']=2.
    elif corruption=='future_ack': samples[20]['observations']['tote']['servo_sequence']=100
    else: samples[20]['observations']['ghost']=copy.deepcopy(samples[20]['observations']['tote'])
    save(tmp_path,'samples',samples); save(tmp_path,'native_commands',commands)
    with pytest.raises(AssertionError): Evidence(tmp_path,definitions).read([],{})


def resource_fixture():
    runtime=ResourceRuntime(whole_line_places(),run_id='synthetic',timeout_s=10.)
    workers={}; stamp=0.
    for job in ['ws001','ws002']:
        runtime.admit(job,stamp)
        while runtime.jobs[job]['index']!=18 or not runtime.jobs[job]['device_done']:
            issued=runtime.dispatch(stamp)
            assert len(issued)==1
            c=issued[0]; execution=c['id']+':worker'
            runtime.events[-1]['command']['execution_id']=execution
            target=runtime.places[next(i for i,p in enumerate(runtime.places) if p.id==c['target'])].target
            stamp+=.1
            runtime.observe(c['id'],now=stamp,observed_at=stamp,pose=target,device_complete=c['kind']=='device')
            workers[execution]=dict(id=execution,job=job,kind=c['kind'],station=c['target'],issued_at=c['issued_at'],
                completed_at=stamp,observation={'time_s':stamp},status='done')
        runtime.release_outfeed(job,stamp); stamp+=.1
    class SyntheticAdmission:
        retained_checks=[]
        def window(self,*args): return []
    return runtime.events,workers,SyntheticAdmission()


def test_resource_audit_reconstructs_two_sequential_synthetic_workers():
    events,workers,evidence=resource_fixture()
    commands,result=verify_resources(events,['ws001','ws002'],workers,evidence)
    assert len(commands)==72 and result['faults']==0


@pytest.mark.parametrize('corruption',['early_release','occupied_destination','wrong_worker_binding','lost_source'])
def test_resource_audit_rejects_forged_ownership_with_otherwise_valid_events(corruption):
    events,workers,evidence=resource_fixture()
    if corruption=='early_release':
        next(e for e in events if e['event']=='ARRIVAL_CONFIRMED')['observed_at']-=1.
    elif corruption=='occupied_destination':
        e=next(e for e in events if e['event']=='COMMAND_ISSUED'); e['snapshot']['stock']['identification']=['ws002']
    elif corruption=='wrong_worker_binding':
        e=next(e for e in events if e['event']=='COMMAND_ISSUED'); workers[e['command']['execution_id']]['job']='ws002'
    else:
        e=next(e for e in events if e['event']=='COMMAND_ISSUED'); e['snapshot']['stock']['infeed']=[]
    with pytest.raises(AssertionError): verify_resources(events,['ws001','ws002'],workers,evidence)
