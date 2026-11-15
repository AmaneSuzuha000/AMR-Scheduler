"""Multi-wheelset finite-capacity flow with Gazebo kinematic pose feedback.

Device work here is explicitly software timed. The separate actuated demo
validates joint-driven actions; this adapter validates shared logical resources
and multiple model positions, without claiming a combined dynamic factory.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from .build import ROOT,SCENE,make_world,save_xml
from .evidence import build_fingerprint
from .resource_runtime import Place,ResourceRuntime,inspection_flow
from .run_process_demo import save_camera


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=['normal','blocked_recovery'],default='normal')
    parser.add_argument('--port',type=int,default=11363);parser.add_argument('--domain',type=int,default=77)
    args=parser.parse_args()
    if os.name=='nt':raise SystemExit('Run on the Ubuntu ROS2/Gazebo host.')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',args.port))==0:raise SystemExit('Resource-demo port occupied; preserved.')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    os.environ['ROS_DOMAIN_ID']=str(args.domain)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from gazebo_msgs.srv import GetEntityState,SetEntityState
    from sensor_msgs.msg import Image
    scene=json.loads(SCENE.read_text());derived=copy.deepcopy(scene);fingerprint=build_fingerprint()
    removed=[i['id'] for i in derived['instances'] if i['asset']=='wheelset' and abs(i['pose'][1]-50)<1e-6]
    template=copy.deepcopy(next(i for i in derived['instances'] if i['id'] in removed))
    derived['instances']=[i for i in derived['instances'] if i['id'] not in removed]
    jobs=[f'ws{n:03d}' for n in range(1,5)]
    for job in jobs:
        item=copy.deepcopy(template);item.update(id='resource_'+job,pose=[10,50,-10,0,0,0]);derived['instances'].append(item)
    camera=dict(id='resource_overview',position=[31.5,20,23],target=[31.5,50,.56],
        horizontal_fov=1.15,width=1280,height=720,reference_second=None)
    tree=make_world(derived,camera);tree.find("world/model[@name='review_camera']/link/sensor/update_rate").text='3'
    world=output/'resources.world';save_xml(tree,world)
    # One physical model per slot in this adapter. Two-slot logical cases are
    # exercised separately, rather than placing two wheelsets at one coordinate.
    runtime=ResourceRuntime([Place(p.id,1,p.target,p.device) for p in inspection_flow()],timeout_s=60)
    device_durations={'derust':2.,'magnetic':6.,'ultrasonic':3.}
    baseline=ROOT/'simulation/wheelset_workshop/runtime/session.json';before=hashlib.sha256(baseline.read_bytes()).hexdigest()
    env=dict(os.environ,GAZEBO_MASTER_URI=f'http://127.0.0.1:{args.port}',DISPLAY=':0',
        GAZEBO_MODEL_PATH=str(ROOT/'assets/wheelset_workshop/models'),GAZEBO_MODEL_DATABASE_URI='',
        LIBGL_ALWAYS_SOFTWARE='1',OMP_NUM_THREADS='4')
    log=(output/'gzserver.log').open('w');server=subprocess.Popen(['gzserver','--verbose',
        '-s','libgazebo_ros_init.so','-s','libgazebo_ros_factory.so',str(world)],
        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rclpy.init();node=Node('wheelset_resource_demo');latest={'image':None,'stamp':None}
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
    def camera_callback(message):
        latest.update(image=message,stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9)
    subscription=node.create_subscription(Image,'/wheelset_review/reference/image_raw',camera_callback,qos)
    setter=node.create_client(SetEntityState,'/wheelset_review/set_entity_state')
    getter=node.create_client(GetEntityState,'/wheelset_review/get_entity_state')
    motions={};positions={};trajectory=[];images=[];error=None;admitted=0;fault=None;recovery=None;stopped=None
    output_saturated=False;late_rejected=False;start=time.monotonic();last_image=-math.inf;last_progress=-1
    def now():return time.monotonic()-start
    def call(client,request):
        future=client.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=5)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError('Gazebo resource service failed')
        return future.result()
    def read(job):
        req=GetEntityState.Request();req.name='resource_'+job;req.reference_frame='world';reply=call(getter,req)
        p=reply.state.pose.position;t=reply.state.twist
        return [p.x,p.y,p.z],max(abs(t.linear.x),abs(t.linear.y),abs(t.linear.z),abs(t.angular.x),abs(t.angular.y),abs(t.angular.z))
    def write(job,pose,token,kind):
        req=SetEntityState.Request();req.state.name='resource_'+job;req.state.reference_frame='world'
        req.state.pose.position.x,req.state.pose.position.y,req.state.pose.position.z=map(float,pose)
        req.state.pose.orientation.w=1.;call(setter,req);observed,speed=read(job)
        distance=math.dist(pose,observed)
        trajectory.append(dict(time_s=now(),job=job,command_id=token,kind=kind,requested=pose,
            observed=observed,position_error_m=distance,reported_twist_norm_max=speed))
        if distance>.03:raise RuntimeError('Multiple-workpiece pose readback mismatch')
        positions[job]=observed;return observed
    def capture():
        nonlocal last_image
        if latest['image'] is None or now()-last_image<.5:return
        if images and images[-1]['sim_time_s']>=latest['stamp']:return
        path=output/'frames'/f'{len(images):05d}.png';path.parent.mkdir(exist_ok=True)
        save_camera(latest['image'],path);images.append(dict(time_s=now(),sim_time_s=latest['stamp'],
            file='frames/'+path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()));last_image=now()
    try:
        for client in [setter,getter]:
            if not client.wait_for_service(timeout_sec=45):raise RuntimeError('Resource service unavailable')
        while len(runtime.completed)<len(jobs):
            loop_start=time.monotonic()
            if server.poll() is not None:raise RuntimeError('Owned Gazebo server exited')
            if now()>240:raise RuntimeError('Resource-run wall timeout')
            if admitted<len(jobs) and runtime.admit(jobs[admitted],now()):
                write(jobs[admitted],list(runtime.places[0].target),None,'admission');admitted+=1
            for token,action in list(motions.items()):
                c=runtime.commands[token];target=next(p.target for p in runtime.places if p.id==c['target'])
                if c['kind']=='transfer':
                    ratio=min(1,(now()-action['start'])/max(action['duration'],.001))
                    pose=[a+(b-a)*ratio for a,b in zip(action['source'],target)]
                    observed=write(c['job'],pose,token,'transfer')
                    if (args.case=='blocked_recovery' and fault is None and c['job']=='ws002'
                            and c['target']=='magnetic' and ratio>=.45):
                        stamp=now();runtime.observe(token,now=stamp,observed_at=stamp,success=False)
                        fault=dict(token=token,time_s=now(),pose=observed);del motions[token]
                    elif ratio==1:
                        stamp=now();assert runtime.observe(token,now=stamp,observed_at=stamp,pose=observed);del motions[token]
                elif now()-action['start']>=device_durations[c['target']]:
                    observed,_=read(c['job']);stamp=now()
                    if math.dist(observed,target)>.03:raise RuntimeError('Software device left its position')
                    trajectory.append(dict(time_s=stamp,job=c['job'],command_id=token,kind='software_device',
                        observed=observed,position_error_m=math.dist(observed,target)))
                    assert runtime.observe(token,now=stamp,observed_at=stamp,device_complete=True);del motions[token]
            if fault and recovery is None:
                observed,speed=read(runtime.commands[fault['token']]['job']);stamp=now()
                trajectory.append(dict(time_s=stamp,job='ws002',command_id=fault['token'],kind='fault_hold',
                    observed=observed,position_error_m=math.dist(observed,fault['pose']),reported_twist_norm_max=speed))
                if stamp-fault['time_s']>=.75 and stopped is None:
                    assert math.dist(observed,fault['pose'])<.001 and speed<.001
                    assert runtime.confirm_stopped(fault['token'],now=stamp,observed_at=stamp,stopped=True)
                    stopped=dict(time_s=stamp,pose=observed,reported_twist_norm_max=speed)
                if stamp-fault['time_s']>=2 and stopped:
                    recovery=runtime.recover(fault['token'],now())
                    stamp=now();late_rejected=not runtime.observe(fault['token'],now=stamp,observed_at=stamp,pose=target)
            if args.case=='blocked_recovery' and now()<75:
                output_saturated |= bool(runtime.stock['outfeed'])
            else:
                for job in sorted(runtime.stock['outfeed']):
                    assert runtime.release_outfeed(job,now());write(job,[10,50,-10],None,'unload');positions.pop(job,None)
            runtime.expire(now())
            for c in runtime.dispatch(now()):
                source,_=read(c['job']);target=next(p.target for p in runtime.places if p.id==c['target'])
                motions[c['id']]=dict(start=now(),source=source,duration=math.dist(source,target)/3.)
            if len(positions)>1:
                separation=min(math.dist(positions[a],positions[b]) for n,a in enumerate(sorted(positions)) for b in sorted(positions)[n+1:])
                if separation<1.2:raise RuntimeError('Wheelset visual envelopes overlap')
            capture();progress=len(runtime.completed)
            if progress!=last_progress:
                print(json.dumps(dict(progress=True,completed=runtime.completed,admitted=admitted,
                    elapsed_wall_s=now(),fault=bool(fault),recovered=bool(recovery))),flush=True);last_progress=progress
            rclpy.spin_once(node,timeout_sec=.04)
            time.sleep(max(0,.05-(time.monotonic()-loop_start)))
        capture()
    except Exception as exc:error=str(exc)
    finally:
        node.destroy_subscription(subscription);node.destroy_node();rclpy.shutdown()
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:server.wait(timeout=8)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        log.close()
    preserved=hashlib.sha256(baseline.read_bytes()).hexdigest()==before
    passed=error is None and len(runtime.completed)==len(jobs) and preserved and server.poll() is not None
    if args.case=='blocked_recovery':passed=passed and bool(fault and recovery and stopped and late_rejected and output_saturated)
    contents=''.join(json.dumps(e)+'\n' for e in runtime.events).encode();(output/'events.jsonl').write_bytes(contents)
    (output/'trajectory.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in trajectory))
    report=dict(passed=passed,case=args.case,runtime_error=error,job_count=len(jobs),completed=runtime.completed,
        flow=[dict(p.__dict__) for p in runtime.places],device_durations_s=device_durations,
        event_count=len(runtime.events),events_sha256=hashlib.sha256(contents).hexdigest(),
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['resource_runtime.py','run_resource_gazebo.py','build.py','run_process_demo.py','evidence.py']},
        world_sha256=hashlib.sha256(world.read_bytes()).hexdigest(),world_file=world.name,build_fingerprint=fingerprint,
        motion_sample_count=len(trajectory),images=images,baseline_session_preserved=preserved,
        owned_server_pid=server.pid,owned_server_stopped=server.poll() is not None,elapsed_wall_s=now(),
        fault=fault,recovered_command=recovery,stop_feedback=stopped,superseded_completion_rejected=late_rejected,
        output_saturation_observed=output_saturated,final_snapshot=runtime.snapshot(),
        claim_boundary='Four Gazebo wheelset models, kinematic SetEntityState with GetEntityState feedback, one-slot places and one exclusive dedicated carrier. Software-timed devices, wall-clock demo parameters, hidden offstage models before admission/after unloading. No driven multi-workpiece dynamics, contact, full-line machinery or AGV integration.')
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','case','completed','runtime_error','motion_sample_count']}),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
