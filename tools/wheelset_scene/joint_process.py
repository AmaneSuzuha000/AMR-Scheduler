"""Finite-capacity multi-wheelset flow with effort-driven device and transfer axes."""
import argparse
from collections import deque
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time

from .actuated_cell import JointSample,efforts
from .build import ROOT,save_xml
from .device_action import DeviceAction,stopped_and_home
from .evidence import build_fingerprint
from .multi_actuated_world import build_multi_world
from .resource_runtime import Place,inspection_flow
from .material_supply import SuppliedRuntime
from .supply_bridge import BridgeSample,BridgeAction,DESTINATIONS,bridge_efforts


def run_worker(output,supply,case,server,stop_event,metadata):
    """Run effort control in a separate thread against the fleet's same server."""
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from gazebo_msgs.srv import ApplyJointEffort,JointRequest,GetEntityState,DeleteEntity
    from sensor_msgs.msg import JointState
    output=Path(output);output.mkdir(exist_ok=False)
    jobs=metadata['jobs'];origins=metadata['origins'];removed=metadata['removed'];assumptions=metadata['assumptions']
    world=output.parent/'light_material.world';fingerprint=build_fingerprint()
    bridge=dict(sample=None,receipt=0.,action=None,events=[],samples=[],efforts=[])
    runtime=SuppliedRuntime(places=[Place(q.id,1,q.target,q.device) for q in inspection_flow()],supply=supply,run_id=output.name,timeout_s=60)
    latest={};receipts={};history={job:deque(maxlen=400) for job in jobs};images=[];samples=[];requests=[];poses=[];action_samples=[]
    targets={job:(0.,0.,0.) for job in jobs}
    actors={};deleted=set();admitting=None;fault=None;stop_evidence=None;recovery=None;late_rejected=False
    output_saturated=False;runtime_error=None;final_stop=None;start=time.monotonic();last_control=-math.inf;last_pose=-math.inf
    node=Node('wheelset_joint_process')
    # Sensor streams favour fresh samples over retransmission of old ones.
    # Reliable streams showed multi-second head-of-line delays in four-model
    # runs; age and position-consistency gates remain unchanged.
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    def feedback(job,message):
        data={name:(position,velocity) for name,position,velocity in zip(message.name,message.position,message.velocity)}
        if not {job+'_slide',job+'_probe',job+'_spin'}<=data.keys():return
        stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        s=JointSample(stamp,*data[job+'_slide'],*data[job+'_probe'],*data[job+'_spin'])
        latest[job]=s;receipts[job]=time.monotonic();history[job].append(s)
    subscriptions=[node.create_subscription(JointState,'/'+job+'/joint_states',
        lambda message,job=job:feedback(job,message),qos) for job in jobs]
    def bridge_feedback(message):
        data={n:(p,v) for n,p,v in zip(message.name,message.position,message.velocity)}
        if all('supply_'+axis in data for axis in 'xyz'):
            stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
            bridge['sample']=BridgeSample(stamp,tuple(data['supply_'+a][0] for a in 'xyz'),tuple(data['supply_'+a][1] for a in 'xyz'))
            bridge['receipt']=time.monotonic()
    subscriptions.append(node.create_subscription(JointState,'/supply_bridge/joint_states',bridge_feedback,qos))
    apply=node.create_client(ApplyJointEffort,'/apply_joint_effort');clear=node.create_client(JointRequest,'/clear_joint_efforts')
    getter=node.create_client(GetEntityState,'/wheelset_review/get_entity_state');deleter=node.create_client(DeleteEntity,'/delete_entity')
    executor=SingleThreadedExecutor();executor.add_node(node)
    spin_thread=threading.Thread(target=executor.spin,daemon=True);spin_thread.start()
    def wait_future(future,seconds):
        ready=threading.Event();future.add_done_callback(lambda unused:ready.set())
        ready.wait(seconds)
    def call(client,req):
        future=client.call_async(req);wait_future(future,5)
        if not future.done() or future.result() is None:raise RuntimeError('Multi-actuation service timeout')
        result=future.result()
        if hasattr(result,'success') and not result.success:raise RuntimeError(result.status_message)
        return result
    def sim_now():return max([runtime.last_time]+[s.time_s for job,s in latest.items() if job not in deleted])
    def snapshot_sample(job):
        s=latest[job];state=runtime.jobs.get(job,{});token=state.get('action');actor=actors.get(token)
        phase=actor.state if actor else state.get('state','admission' if admitting==job else 'staged')
        return dict(s.__dict__,job=job,command_id=token,phase=phase,target=list(targets[job]),world_x=origins[job]+s.slide)
    def drive(stopping=False):
        live=[job for job in jobs if job not in deleted and job in latest]
        # Keep each concurrent service batch below the Gazebo service's default
        # history depth (10). Sending all 12 axes at once dropped responses.
        for job in live:
            pending=[]
            for axis in ['slide','probe','spin']:
                req=JointRequest.Request();req.joint_name=job+'_'+axis;pending.append(clear.call_async(req))
            for future in pending:
                wait_future(future,3)
                if not future.done() or future.result() is None:raise RuntimeError('Effort clear timed out')
            pending=[];batch=[]
            s=latest[job];values=efforts(s,targets[job]);token=runtime.jobs.get(job,{}).get('action');actor=actors.get(token)
            stalled=(not stopping and fault is None and case=='stalled_recovery' and job=='ws002'
                and actor and actor.station=='magnetic' and actor.state=='rotate')
            if stalled:values['wsd_spin']=0.
            row=snapshot_sample(job)
            if stopping:row['phase']='stopping'
            samples.append(row)
            for name,value in values.items():
                name=name.replace('wsd_',job+'_');req=ApplyJointEffort.Request();req.joint_name=name;req.effort=float(value)
                ns=int(round(s.time_s*1e9));req.start_time.sec,req.start_time.nanosec=divmod(ns,1000000000)
                req.duration.nanosec=250000000;pending.append(apply.call_async(req))
                batch.append(dict(job=job,command_id=token,sample_time_s=s.time_s,joint=name,effort=value,
                    start_sim_time_s=s.time_s,duration_s=.25,stopping=stopping,stall_injected=bool(stalled)))
            for future in pending:
                wait_future(future,3)
                if not future.done() or future.result() is None or not future.result().success:
                    raise RuntimeError('Effort application rejected')
            for row in batch:row['accepted']=True
            requests.extend(batch)
        s=bridge.get('sample')
        if s is not None:
            target=bridge.get('target',s.xyz)
            if stopping:target=s.xyz
            pending=[]
            for axis in 'xyz':
                req=JointRequest.Request();req.joint_name='supply_'+axis;pending.append(clear.call_async(req))
            for future in pending:
                wait_future(future,3)
                if not future.done() or future.result() is None:raise RuntimeError('Bridge clear timeout')
            pending=[]
            for name,value in bridge_efforts(s,target).items():
                req=ApplyJointEffort.Request();req.joint_name=name;req.effort=float(value)
                ns=int(round(s.time_s*1e9));req.start_time.sec,req.start_time.nanosec=divmod(ns,1000000000);req.duration.nanosec=250000000
                pending.append(apply.call_async(req))
                bridge['efforts'].append(dict(time_s=s.time_s,joint=name,effort=value,target=list(target),stopping=stopping))
            for future in pending:
                wait_future(future,3)
                if not future.done() or future.result() is None or not future.result().success:raise RuntimeError('Bridge effort rejected')
            bridge['samples'].append(dict(time_s=s.time_s,xyz=list(s.xyz),velocity=list(s.velocity),target=list(target),
                token=bridge['action'].command['token'] if bridge['action'] else None))
    def measured_pose(job):
        req=GetEntityState.Request();req.name='fixture_'+job+'::wheelset';req.reference_frame='world'
        reply=call(getter,req);pos=reply.state.pose.position
        stamp=reply.header.stamp.sec+reply.header.stamp.nanosec*1e-9;deadline=time.monotonic()+1
        while latest[job].time_s<stamp and time.monotonic()<deadline:time.sleep(.002)
        match=min(list(history[job]),key=lambda s:abs(s.time_s-stamp));lag=abs(match.time_s-stamp)
        error=abs(pos.x-origins[job]-match.slide)
        poses.append(dict(job=job,sim_time_s=stamp,joint_time_s=match.time_s,sample_time_error_s=lag,
            world_pose=[pos.x,pos.y,pos.z],joint_slide=match.slide,x_consistency_error_m=error,
            command_id=runtime.jobs.get(job,{}).get('action')))
        if lag>.05 or error>.15 or abs(pos.y-50)>.02 or abs(pos.z-.56)>.02:
            raise RuntimeError('Multi-fixture position consistency failed')
        return [pos.x,pos.y,pos.z-.56],stamp
    try:
        for client in [apply,clear,getter,deleter]:
            if not client.wait_for_service(timeout_sec=45):raise RuntimeError('Multi-actuation services unavailable')
        deadline=time.monotonic()+30
        while (len(latest)<len(jobs) or bridge.get('sample') is None) and time.monotonic()<deadline:time.sleep(.01)
        if len(latest)<len(jobs) or bridge.get('sample') is None:raise RuntimeError('Missing initial per-job joint feedback')
        while len(runtime.completed)<len(jobs):
            time.sleep(.002);now=sim_now()
            if stop_event.is_set() or server.poll() is not None:raise RuntimeError('Joint run stopping')
            if time.monotonic()-start>900:raise RuntimeError('Multi-actuation wall timeout')
            if any(time.monotonic()-receipts[job]>1 for job in jobs if job not in deleted):
                raise RuntimeError('Per-job feedback stale')
            if now-last_control<.05:continue
            if time.monotonic()-bridge['receipt']>1:raise RuntimeError('Bridge feedback stale')
            bs=bridge['sample'];actor=bridge['action']
            if actor is None:
                for task_id,d in list(supply.deliveries.items()):
                    if d['status']=='at_dock':
                        command=supply.begin_handoff(task_id,now=now)
                        if command:
                            actor=BridgeAction(command,bs);bridge['action']=actor
                            break
            if actor:
                actor.update(bs);bridge['target']=actor.target
                if actor.fault:raise RuntimeError(actor.fault)
                if actor.complete:
                    if not supply.commit(actor.command['token'],dict(time_s=bs.time_s,xyz=list(bs.xyz),velocity=list(bs.velocity)),
                        DESTINATIONS[actor.command['station']],now=max(sim_now(),bs.time_s)):raise RuntimeError('Supply commit rejected')
                    bridge['events'].extend(actor.events);bridge['action']=None
                    print(json.dumps(dict(progress=True,event='SUPPLY_COMMITTED',task_id=actor.command['task_id'],sim_time_s=sim_now())),flush=True)
            if admitting is None and len(runtime.jobs)<len(jobs) and not runtime.stock['infeed'] and not runtime.reserved['infeed']:
                admitting=jobs[len(runtime.jobs)];targets[admitting]=(10-origins[admitting],0.,0.)
            if admitting:
                s=latest[admitting]
                if abs(origins[admitting]+s.slide-10)<.025 and stopped_and_home(s):
                    measured_pose(admitting);assert runtime.admit(admitting,sim_now());admitting=None
            for job,state in list(runtime.jobs.items()):
                token=state.get('action')
                if token is None:continue
                command=runtime.commands[token];s=latest[job]
                if command['status']=='active' and command['kind']=='transfer':
                    target=next(q.target for q in runtime.places if q.id==command['target'])
                    if abs(origins[job]+s.slide-target[0])<=.025 and stopped_and_home(s):
                        pose,stamp=measured_pose(job)
                        assert runtime.observe(token,now=sim_now(),observed_at=stamp,pose=pose)
                elif command['status']=='active':
                    actor=actors[token]
                    action_samples.append(dict(s.__dict__,job=job,command_id=token,phase_before=actor.state))
                    actor.update(s);targets[job]=actor.targets()
                    # Explicit degraded-mode workload: after the spindle fault,
                    # magnetic processing resumes at half nominal speed. The
                    # measured full-turn, stop/home and 12s rotation timeout
                    # gates are unchanged. This is an assumed demonstration
                    # recipe, not a recovered manufacturer fault policy.
                    if recovery is not None and actor.station=='magnetic' and actor.state=='rotate':
                        targets[job]=(targets[job][0],targets[job][1],.8)
                    if actor.state=='complete':
                        measured_pose(job)
                        assert runtime.observe(token,now=sim_now(),observed_at=s.time_s,device_complete=True)
                    elif actor.state=='failed':
                        stamp=sim_now();runtime.observe(token,now=stamp,observed_at=s.time_s,success=False)
                        targets[job]=(s.slide,0.,0.)
                        fault=dict(job=job,token=token,sim_time_s=stamp,reason=actor.fault,hold_slide=s.slide)
                        print(json.dumps(dict(progress=True,event='FAULT_LATCHED',**fault)),flush=True)
                elif command['status']=='failed':
                    if fault is None or fault['token']!=token:raise RuntimeError('Unexpected resource fault')
                    if s.time_s-fault['sim_time_s']>.5 and stopped_and_home(s):
                        measured_pose(job)
                        stamp=sim_now();assert runtime.confirm_stopped(token,now=stamp,observed_at=s.time_s,stopped=True)
                        stop_evidence=dict(s.__dict__,job=job,command_id=token)
                        recovery=runtime.recover(token,sim_now());assert recovery
                        late_rejected=not runtime.observe(token,now=sim_now(),observed_at=s.time_s,device_complete=True)
                        print(json.dumps(dict(progress=True,event='RECOVERY_REISSUED',job=job,command_id=recovery)),flush=True)
            if case=='stalled_recovery' and sim_now()<300:
                output_saturated |= bool(runtime.stock['outfeed'])
            else:
                for job in sorted(runtime.stock['outfeed']):
                    s=latest[job]
                    if not stopped_and_home(s):continue
                    measured_pose(job)
                    for axis in ['slide','probe','spin']:
                        req=JointRequest.Request();req.joint_name=job+'_'+axis;call(clear,req)
                    req=DeleteEntity.Request();req.name='fixture_'+job;call(deleter,req);deleted.add(job)
                    assert runtime.release_outfeed(job,sim_now())
                    print(json.dumps(dict(progress=True,event='OUTFEED_RELEASED',job=job,completed=runtime.completed,sim_time_s=sim_now())),flush=True)
            runtime.expire(sim_now())
            for command in runtime.dispatch(sim_now()):
                job=command['job'];target=next(q.target for q in runtime.places if q.id==command['target'])
                targets[job]=(target[0]-origins[job],0.,0.)
                if command['kind']=='device':
                    actors[command['id']]=DeviceAction(command['id'],job,command['target'],targets[job][0])
                print(json.dumps(dict(progress=True,event='COMMAND_ISSUED',job=job,kind=command['kind'],target=command['target'],sim_time_s=sim_now())),flush=True)
            if sim_now()-last_pose>=1:
                for job in jobs:
                    if job not in deleted:measured_pose(job)
                last_pose=sim_now()
            xs=sorted(origins[job]+latest[job].slide for job in jobs if job not in deleted)
            if any(b-a<1.2 for a,b in zip(xs,xs[1:])):raise RuntimeError('Measured wheelset envelopes overlap')
            drive();last_control=now
    except Exception as exc:runtime_error=str(exc)
    finally:
        live=[job for job in jobs if job not in deleted and job in latest]
        if live and server.poll() is None and apply.service_is_ready() and clear.service_is_ready():
            for job in live:targets[job]=(latest[job].slide,0.,0.)
            deadline=time.monotonic()+20;stop_start=sim_now()
            try:
                while time.monotonic()<deadline:
                    time.sleep(.002)
                    if sim_now()-last_control<.05:continue
                    drive(stopping=True);last_control=sim_now()
                    if (sim_now()-stop_start>.5 and all(stopped_and_home(latest[job]) for job in live)
                        and max(abs(v) for v in bridge['sample'].velocity)<.035):
                        final_stop={job:dict(latest[job].__dict__) for job in live};break
                if final_stop is None:raise RuntimeError('Remaining axes did not confirm stop/home')
            except Exception as exc:runtime_error='; '.join(filter(None,[runtime_error,str(exc)]))
        else:final_stop={}
        if clear.service_is_ready():
            for job in live:
                for axis in ['slide','probe','spin']:
                    try:
                        req=JointRequest.Request();req.joint_name=job+'_'+axis;call(clear,req)
                    except Exception:pass
            for axis in 'xyz':
                try:
                    req=JointRequest.Request();req.joint_name='supply_'+axis;call(clear,req)
                except Exception:pass
        executor.shutdown(timeout_sec=5);spin_thread.join(timeout=5)
        for subscription in subscriptions:node.destroy_subscription(subscription)
        node.destroy_node()
    passed=runtime_error is None and len(runtime.completed)==4
    if case=='normal':passed=passed and fault is None
    if case=='stalled_recovery':passed=passed and bool(fault and recovery and stop_evidence and late_rejected and output_saturated)
    events=''.join(json.dumps(e)+'\n' for e in runtime.events).encode();(output/'events.jsonl').write_bytes(events)
    device_events=[e for actor in actors.values() for e in actor.events]
    data_sha256={}
    for name,rows in [('joint_samples',samples),('effort_commands',requests),('pose_checks',poses),('device_events',device_events),('action_samples',action_samples)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        data_sha256[name+'.jsonl']=hashlib.sha256((output/(name+'.jsonl')).read_bytes()).hexdigest()
    source_names=['joint_process.py','material_supply.py','supply_bridge.py','multi_actuated_world.py','device_action.py','resource_runtime.py',
        'actuated_cell.py','actuated_world.py','build.py','evidence.py','run_process_demo.py']
    report=dict(passed=passed,case=case,runtime_error=runtime_error,job_count=4,completed=runtime.completed,
        flow=[dict(q.__dict__) for q in runtime.places],device_durations_s={q.id:None for q in runtime.places if q.device},
        event_count=len(runtime.events),events_sha256=hashlib.sha256(events).hexdigest(),
        source_sha256={name:hashlib.sha256((output.parent/'executed_sources'/name).read_bytes()).hexdigest() for name in source_names},
        build_fingerprint=fingerprint,world_sha256=hashlib.sha256(world.read_bytes()).hexdigest(),world='../'+world.name,
        joint_sample_count=len(samples),effort_request_count=len(requests),device_event_count=len(device_events),images=images,
        data_sha256=data_sha256,
        baseline_session_preserved=True,owned_server_pid=server.pid,owned_server_stopped=False,
        elapsed_wall_s=time.monotonic()-start,assumptions=assumptions,removed_static_snapshot_ids=removed,
        fault=fault,stop_evidence=stop_evidence,recovered_command=recovery,superseded_completion_rejected=late_rejected,
        output_saturation_observed=output_saturated,deleted_after_measured_output=sorted(deleted),final_stop=final_stop,
        claim_boundary='Four effort-driven constrained fixtures and finite-capacity ownership in one Gazebo run. Joint feedback gates transfer, full-turn device work, braking, probe retraction and retry. Assumed mechanisms/masses, gravity/contact excluded; staging and output deletion abstract external boundaries. Same server as two AGVs and an assumed driven consumable shuttle; no full-line or industrial acceptance.')
    for name,rows in [('bridge_samples',bridge['samples']),('bridge_efforts',bridge['efforts']),('bridge_events',bridge['events'])]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
        report['data_sha256'][name+'.jsonl']=hashlib.sha256((output/(name+'.jsonl')).read_bytes()).hexdigest()
    report['bridge_assumptions']=metadata['bridge_assumptions']
    report['bridge_final_sample']=dict(bridge['sample'].__dict__) if bridge['sample'] else None
    report['magnetic_spin_recipe']=dict(nominal_rad_s=1.6,after_recovery_rad_s=.8,
        assumption='Reduced magnetic spindle speed after explicit recovery; full turn and rotation timeout unchanged.')
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    return report
