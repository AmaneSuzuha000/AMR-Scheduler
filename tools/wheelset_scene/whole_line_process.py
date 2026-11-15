"""Whole-line process worker sharing the existing two-SCOUT Gazebo runner."""
from collections import deque
import gc
import hashlib
import json
import math
from pathlib import Path
import time

from .build import ROOT
from .evidence import build_fingerprint
from .finishing_flow import STORED_JOB,STORAGE_APPROACH,STORAGE_UNLOAD
from .front_flow import ORIGIN,Waypoints
from .whole_line_actions import WholeLineAction
from .whole_line_runtime import WholeLineRuntime,stopped_sample


def run_worker(output,supply,case,server,stop_event,metadata,*,supply_mode='legacy_handoff'):
    if supply_mode not in ('legacy_handoff','fleet_inventory'):
        raise ValueError('explicit supported process supply mode required')
    import rclpy
    from rclpy.node import Node
    from rclpy.context import Context
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    from gazebo_msgs.srv import GetEntityState,DeleteEntity
    import threading

    # This worker now owns an isolated Python process. Defer cyclic collection
    # until its exit, keeping large acyclic raw evidence off callback pauses.
    gc.collect();gc.disable()
    output=Path(output);output.mkdir(exist_ok=False);meta=metadata['whole_line']
    definitions=meta['definitions'];jobs=meta['jobs']
    runtime=WholeLineRuntime(output.parent.name,supply=supply,worker_geometry=meta['worker_geometry'])
    latest={};receipts={};history={n:deque(maxlen=400) for n in definitions};deleted=set()
    targets={n:dict(xyz=list(d['origin']),probe=0.,spin=0.) for n,d in definitions.items()}
    samples=[];native=[];poses=[];actions={};all_actions=[];faults=[];retired=[]
    stage_events=[];tote_events=[];tote_program=None;tote_command=None;unload=None;admitting=None;shutting_down=False
    last_tick=-math.inf;last_pose=-math.inf;last_progress=-math.inf;last_bridge_task=None
    # Two executors must not build/wait on wait sets in the same ROS context.
    # Keep the same domain and server, but isolate process/fleet wait handles.
    context=Context();context.init(args=None)
    node=Node('wheelset_whole_line_process',context=context);executor=SingleThreadedExecutor(context=context);executor.add_node(node)
    start=time.monotonic();error=None;final_stop=None;last_time=0.;retry_tokens={};recovery_started={}
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    def callback(name,message):
        d=definitions[name];data={n:(p,v) for n,p,v in zip(message.name,message.position,message.velocity)}
        if not all(name+'_'+a in data for a in d['axes']):return
        force=dict(zip(message.name,message.effort));stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        row=dict(entity=name,time_s=stamp,xyz=[d['origin'][i]+data.get(name+'_'+a,(0.,0.))[0] for i,a in enumerate('xyz')],
            velocity=[data.get(name+'_'+a,(0.,0.))[1] for a in 'xyz'],spin=data.get(name+'_spin',(0.,0.))[0],
            omega=data.get(name+'_spin',(0.,0.))[1],probe=data.get(name+'_probe',(0.,0.))[0],
            probe_velocity=data.get(name+'_probe',(0.,0.))[1],applied_effort={a:force[name+'_'+a] for a in d['axes']},
            servo_sequence=int(data[name+'_servo_sequence'][0]),servo_fault=bool(data[name+'_servo_fault'][0]))
        latest[name]=row;receipts[name]=time.monotonic();history[name].append(row)
    subscriptions=[node.create_subscription(JointState,'/'+name+'/joint_states',lambda msg,name=name:callback(name,msg),qos) for name in definitions]
    publishers={n:node.create_publisher(Float64MultiArray,'/'+n+'/servo_targets',1) for n in definitions}
    getter=node.create_client(GetEntityState,'/wheelset_review/get_entity_state');deleter=node.create_client(DeleteEntity,'/delete_entity')
    spin_errors=[]
    def spin():
        try:executor.spin()
        except Exception as exc:spin_errors.append(repr(exc))
    thread=threading.Thread(target=spin,daemon=True);thread.start()
    def now():return max([last_time]+[s['time_s'] for n,s in latest.items() if n not in deleted])
    def sync_mounted():
        for job,w in runtime.workers.items():
            if job in deleted:continue
            detached=actions[job].detached if job in actions else set()
            for n,i in w.items.items():
                if n in deleted:continue
                if w.part_state[n] in {'mounted','refitted'} and n not in detached:
                    targets[n]['xyz']=[a+b for a,b in zip(targets[job]['xyz'],i['offset'])]
            for n,i in w.components.items():
                if n not in deleted and w.component_state[n]=='fitted':targets[n]['xyz']=[a+b for a,b in zip(targets[job]['xyz'],i['offset'])]
    def tick():
        nonlocal last_tick,last_time
        if spin_errors:raise RuntimeError('whole-line executor: '+spin_errors[0])
        if server.poll() is not None:raise RuntimeError('owned whole-line server exited')
        if not shutting_down and time.monotonic()-start>7200:raise RuntimeError('whole-line wall timeout')
        active=[n for n in definitions if n not in deleted]
        if any(time.monotonic()-receipts.get(n,start)>1 for n in active):raise RuntimeError('whole-line native feedback stale')
        deadline=time.monotonic()+.25
        while max(latest[n]['time_s'] for n in active)-min(latest[n]['time_s'] for n in active)>.1:
            if time.monotonic()>deadline:raise RuntimeError('whole-line source time skew')
            time.sleep(.003)
        observation={n:dict(latest[n]) for n in active};stamp=now();last_time=stamp
        if any(s['servo_fault'] for s in observation.values()):raise RuntimeError('native servo lease fault')
        sync_mounted();sequence=len(samples)+1
        samples.append(dict(time_s=stamp,observations=observation,targets={n:dict(targets[n]) for n in active},
            commands={j:a.command['id'] for j,a in actions.items()},phases={j:a.phase for j,a in actions.items()}))
        disabled={a.disabled for a in actions.values() if a.disabled}
        for n in active:
            d=definitions[n];t=targets[n];values=[float(sequence)]
            for axis in d['axes']:
                value=t['xyz']['xyz'.index(axis)]-d['origin']['xyz'.index(axis)] if axis in 'xyz' else t[axis]
                values.extend([float(value),0. if (n,axis) in disabled else 1.])
            message=Float64MultiArray();message.data=values;publishers[n].publish(message)
            native.append(dict(entity=n,sequence=sequence,time_s=stamp,axis_values=values[1:],
                execution_id=actions[n].token if n in actions else None,disabled_axes=[a for a in d['axes'] if (n,a) in disabled]))
        last_tick=stamp;return observation
    def call(client,request):
        future=client.call_async(request);deadline=time.monotonic()+5
        while not future.done() and time.monotonic()<deadline:
            if now()-last_tick>=.05:tick()
            else:time.sleep(.002)
        if not future.done() or future.result() is None or not future.result().success:raise RuntimeError('whole-line world service failed')
        return future.result()
    def measured(name):
        req=GetEntityState.Request();d=definitions[name];req.name=name+('::probe' if d['kind']=='ram' else '::body');req.reference_frame='world'
        reply=call(getter,req);stamp=reply.header.stamp.sec+reply.header.stamp.nanosec*1e-9
        deadline=time.monotonic()+1
        while latest[name]['time_s']<stamp and time.monotonic()<deadline:time.sleep(.002)
        s=min(list(history[name]),key=lambda row:abs(row['time_s']-stamp));p=reply.state.pose.position;expected=list(s['xyz'])
        if d['kind']=='ram':expected[1]+=d['direction_y']*s['probe']
        error=math.dist(expected,[p.x,p.y,p.z]);lag=abs(stamp-s['time_s'])
        poses.append(dict(entity=name,time_s=stamp,joint_sample=s,world_xyz=[p.x,p.y,p.z],expected_xyz=expected,error_m=error,source_time_error_s=lag))
        if error>.03 or lag>.05:raise RuntimeError('whole-line world/joint mismatch')
    def remove(names):
        for n in names:
            req=DeleteEntity.Request();req.name=n;call(deleter,req);deleted.add(n)
            retired.append(dict(entity=n,time_s=now(),observation=dict(latest[n])))
    def hold_job(job):
        w=runtime.workers[job]
        names={job,*w.items,*w.components}
        if w.station=='press':names.update({'press_ram_left','press_ram_right'})
        for n in names:
            if n not in deleted:targets[n].update(xyz=list(latest[n]['xyz']),probe=0.,spin=0.)
    try:
        for client in [getter,deleter]:
            if not client.wait_for_service(timeout_sec=45):raise RuntimeError('whole-line services unavailable')
        deadline=time.monotonic()+30
        while len(latest)<len(definitions) and time.monotonic()<deadline:time.sleep(.01)
        if len(latest)!=len(definitions):raise RuntimeError('whole-line initial feedback missing')
        tick()
        while len(runtime.completed)<len(jobs):
            if stop_event.is_set():raise RuntimeError('fleet runner requested process stop')
            if now()-last_tick<.05:time.sleep(.002);continue
            observation=tick();stamp=now()
            # A second external arrival moves physically into the single infeed
            # only after the preceding workpiece's measured departure.
            if admitting is None and len(runtime.workers)<len(jobs) and not runtime.stock['infeed']:
                job=jobs[len(runtime.workers)];admitting=job;targets[job]['xyz']=list(ORIGIN)
            if admitting:
                job=admitting;delta=[ORIGIN[i]-definitions[job]['origin'][i] for i in range(3)]
                for n,i in meta['worker_geometry'][job]['parts'].items():
                    targets[n]['xyz']=[definitions[n]['origin'][k]+delta[k] for k in range(3)]
                ready=math.dist(observation[job]['xyz'],ORIGIN)<.025 and stopped_sample(observation[job],stamp)
                ready &= all(math.dist(observation[n]['xyz'],targets[n]['xyz'])<.035 and stopped_sample(observation[n],stamp)
                    for n in meta['worker_geometry'][job]['parts'])
                if ready:
                    measured(job);assert runtime.admit(job,now(),observation=dict(latest[job]));admitting=None
            for job,a in list(actions.items()):
                c=runtime.commands[a.command['id']]
                if c['status']=='failed':
                    hold_job(job)
                    if stamp-recovery_started[job]>.5 and runtime.confirm_stopped(c['id'],now=stamp,observations=observation):
                        new=runtime.recover(c['id'],stamp);assert new
                        assert not runtime.complete(c['id'],stamp,observation[job]);retry_tokens[job]=new
                        del actions[job]
                    continue
                a.advance(observation,stamp)
                if a.fault:
                    if case!='stalled_recovery' or job in recovery_started or c['target']!='press':raise RuntimeError(a.fault)
                    assert runtime.fail(c['id'],stamp,a.fault);recovery_started[job]=stamp
                    faults.append(dict(command_id=c['id'],execution_id=a.token,job=job,time_s=stamp,reason=a.fault))
                    hold_job(job)
                elif a.complete:
                    measured(job);assert runtime.complete(c['id'],now(),dict(latest[job]))
                    del actions[job]
            # One finite slot includes the pre-existing physical wheelset. Its
            # unload path is independently driven before the next reservation.
            if unload is None and runtime.storage.occupant is not None:
                waiting=any(w.station=='storage_wait' for w in runtime.workers.values())
                finished=runtime.storage.occupant in runtime.workers and runtime.workers[runtime.storage.occupant].passed
                if waiting or finished:
                    token=runtime.begin_unload(now())
                    if token:
                        job=runtime.storage.occupant;p=STORAGE_APPROACH;u=STORAGE_UNLOAD
                        unload=dict(token=token,job=job,program=Waypoints(token,[p,(*p[:2],6.),(*u[:2],6.),u],now()))
            if unload:
                job=unload['job'];program=unload['program'];targets[job]['xyz']=list(program.target)
                s=latest[job];index=program.index
                if not program.complete and program.update(s['xyz'],s['velocity'],s['time_s']):
                    stage_events.append(dict(event='UNLOAD_WAYPOINT',job=job,command_id=unload['token'],index=index,
                        time_s=now(),target=list(program.points[index]),observation=dict(s)))
                attached=[]
                if job in runtime.workers:
                    w=runtime.workers[job]
                    attached=[n for n in w.items if w.part_state[n]=='refitted']+[n for n in w.components if w.component_state[n]=='fitted']
                settled=all(stopped_sample(latest[n],now()) and math.dist(latest[n]['xyz'],targets[n]['xyz'])<.035 for n in attached)
                if program.complete and settled:
                    measured(job);assert runtime.complete_unload(unload['token'],now(),dict(latest[job]))
                    # Outgoing items have measured reached the declared external
                    # boundary. Retire them before another unload uses it.
                    names=[job]
                    if job in runtime.workers:
                        w=runtime.workers[job];names += [n for n,i in w.items.items() if w.part_state[n]=='refitted']
                        names += [n for n in w.components if w.component_state[n]=='fitted']
                    remove(names);unload=None
            # Consumables are committed only after the existing fleet receipt
            # and an independent native driven tote completes six waypoints.
            if supply_mode == 'legacy_handoff' and tote_program is None:
                for task in list(supply.deliveries):
                    command=supply.begin_handoff(task,now=now())
                    if command:
                        from .supply_bridge import SOURCES
                        source=SOURCES[task];dest=supply.destinations[command['station']];p=latest['line_supply_tote']['xyz']
                        points=[(*p[:2],6.),(*source[:2],6.),source,(*source[:2],6.),(*dest[:2],6.),dest]
                        tote_program=Waypoints(command['token'],points,now(),75);tote_command=command;break
            if tote_program:
                name='line_supply_tote';targets[name]['xyz']=list(tote_program.target);s=latest[name];index=tote_program.index
                if tote_program.update(s['xyz'],s['velocity'],s['time_s']):
                    tote_events.append(dict(token=tote_command['token'],task_id=tote_command['task_id'],waypoint=index,
                        time_s=now(),target=list(tote_program.points[index]),observation=dict(s)))
                if tote_program.complete:
                    measured(name);assert supply.commit(tote_command['token'],dict(latest[name]),supply.destinations[tote_command['station']],now=now())
                    tote_program=None;tote_command=None
            for command in runtime.dispatch(now()):
                job=command['job'];actions[job]=WholeLineAction(command,runtime.workers[job],targets,
                    case='press_recovery' if case=='stalled_recovery' and job==jobs[-1] else 'normal',recovered=job in retry_tokens)
                all_actions.append(actions[job])
            runtime.expire(now())
            if now()-last_pose>=1:
                live=[n for n in definitions if n not in deleted];measured(live[int(now())%len(live)]);last_pose=now()
            active_jobs=[j for j in jobs if j not in deleted]
            if len(active_jobs)==2 and math.dist(latest[active_jobs[0]]['xyz'],latest[active_jobs[1]]['xyz'])<2.6:
                raise RuntimeError('measured main wheelset envelopes too close')
            if now()-last_progress>=20:
                print(json.dumps(dict(progress=True,whole_line=True,time_s=now(),completed=runtime.completed,
                    stations={j:w.station for j,w in runtime.workers.items()},phases={j:a.phase for j,a in actions.items()},
                    supply=dict(supply.stock),storage=runtime.storage.snapshot())),flush=True);last_progress=now()
    except Exception as exc:error=repr(exc)
    finally:
        shutting_down=True
        for n,s in latest.items():
            if n not in deleted:targets[n].update(xyz=list(s['xyz']),probe=0.,spin=0.)
        actions.clear();deadline=time.monotonic()+20;began=now()
        try:
            while server.poll() is None and time.monotonic()<deadline:
                if now()-last_tick<.05:time.sleep(.002);continue
                observation=tick()
                if now()-began>.5 and all(stopped_sample(s,now()) for s in observation.values()):final_stop=observation;break
            if final_stop is None:raise RuntimeError('whole-line final stop unconfirmed')
        except Exception as exc:error='; '.join(filter(None,[error,repr(exc)]))
        executor.shutdown(timeout_sec=5);thread.join(timeout=5)
        for sub in subscriptions:node.destroy_subscription(sub)
        node.destroy_node();context.shutdown()
    # Preserve each exact worker's raw events and protocol/stock state.
    hashes={}
    rows=dict(samples=samples,native_commands=native,pose_checks=poses,events=runtime.events,
        storage_events=runtime.storage.events,tote_events=tote_events,retired=retired,unload_waypoints=stage_events,
        worker_events=[dict(job=j,**e) if 'job' not in e else e for j,w in runtime.workers.items() for e in w.events],
        stock_events=[dict(job=j,**e) for j,w in runtime.workers.items() for e in w.stock_events],
        device_events=[e for a in all_actions for e in a.events],
        waypoints=[e for a in all_actions for e in a.waypoints],press_samples=[e for a in all_actions for e in a.press_samples])
    for name,records in rows.items():
        path=output/(name+'.jsonl')
        with path.open('w') as stream:
            for row in records:stream.write(json.dumps(row)+'\n')
        hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    report=dict(passed=error is None and len(runtime.completed)==len(jobs) and final_stop is not None,
        runtime_error=error,mode='whole_line',case=case,completed=runtime.completed,job_count=len(jobs),faults=faults,
        supply_mode=supply_mode,
        recovered_commands=retry_tokens,final_stop=final_stop,data_sha256=hashes,images=[],metadata=meta,
        final_snapshot=runtime.snapshot(),worker_snapshots={j:w.snapshot() for j,w in runtime.workers.items()},
        worker_stock={j:w.stock_snapshot() for j,w in runtime.workers.items()},storage_snapshot=runtime.storage.snapshot(),
        build_fingerprint=build_fingerprint(),elapsed_wall_s=time.monotonic()-start,
        claim_boundary='Two identified eighteen-stage constrained wheelset workers, shared measured finite station/carrier/bin/storage ownership, same-world fleet dock receipt and native driven tote supply. Assumed motion/mass/layout; gravity/contact excluded for wheelset/parts/tote. No process-quality/contact attachment, Nav2/localization or industrial acceptance.')
    if supply_mode == 'fleet_inventory':
        report['claim_boundary']='Two constrained wheelset workers request process-service units from the fleet owner after its accepted unloading. No second receiving ledger or native tote transfer. Buffer-to-machine service is declared software demand, not observed physical transfer. Other wheelset motion/mass/layout assumptions and excluded gravity/contact remain; no industrial acceptance.'
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n');return report
