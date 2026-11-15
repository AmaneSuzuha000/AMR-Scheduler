"""Run the first causal process slice on a separate, temporary Gazebo master.

Kinematic SetEntityState movement is followed by GetEntityState readback. The
selected lane's static wheel snapshots are removed in a derived world only.
The visual baseline and its live Gazebo session are never changed.
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
import struct
import subprocess
import time
import zlib

from .build import ROOT,SCENE,make_world,save_xml
from .evidence import build_fingerprint
from .process_runtime import ProcessController,inspection_plan


def save_camera(message,path):
    if message.encoding.lower()!='rgb8':raise RuntimeError('Expected RGB8 review camera')
    rows=[b'\0'+bytes(message.data[y*message.step:y*message.step+message.width*3]) for y in range(message.height)]
    def chunk(kind,data):return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    path.write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',message.width,message.height,8,2,0,0,0))+
        chunk(b'IDAT',zlib.compress(b''.join(rows)))+chunk(b'IEND',b''))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=11361)
    parser.add_argument('--domain',type=int,default=75)
    parser.add_argument('--speed',type=float,default=2)
    parser.add_argument('--service-seconds',type=float,default=2)
    parser.add_argument('--inject-failure',choices=['derust_hood','magnetic_test_hood','ultrasonic_test_hood'])
    args=parser.parse_args()
    if os.name=='nt':raise SystemExit('Run on the existing Ubuntu ROS2/Gazebo host.')
    if not math.isfinite(args.speed) or args.speed<=0:raise SystemExit('Positive speed required.')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',args.port))==0:raise SystemExit('Process-demo port already occupied; preserved.')
    os.environ['ROS_DOMAIN_ID']=str(args.domain)
    import rclpy
    from rclpy.node import Node
    from gazebo_msgs.srv import GetEntityState,SetEntityState
    from sensor_msgs.msg import Image
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    scene=json.loads(SCENE.read_text());fingerprint=build_fingerprint()
    plan=inspection_plan(scene,args.service_seconds);lane=plan[0].target[1]
    derived=copy.deepcopy(scene)
    snapshots=[i for i in derived['instances'] if i['asset']=='wheelset' and abs(i['pose'][1]-lane)<1e-6]
    # The pre-existing snapshot positions are not operating workpieces.
    derived['instances']=[i for i in derived['instances'] if i['id'] not in {s['id'] for s in snapshots}]
    workpiece=copy.deepcopy(snapshots[0]);workpiece.update(id='process_demo_wheelset',pose=[10,lane,0,0,0,0])
    derived['instances'].append(workpiece)
    camera=next(c for c in derived['cameras'] if c['id']=='inspection')
    world=output/'process_demo.world';save_xml(make_world(derived,camera),world)
    controller=ProcessController(plan)
    env=dict(os.environ,GAZEBO_MASTER_URI=f'http://127.0.0.1:{args.port}',DISPLAY=':0',
        GAZEBO_MODEL_PATH=str(ROOT/'assets/wheelset_workshop/models'),GAZEBO_MODEL_DATABASE_URI='',
        LIBGL_ALWAYS_SOFTWARE='1',OMP_NUM_THREADS='4')
    baseline_session=ROOT/'simulation/wheelset_workshop/runtime/session.json'
    baseline_before=hashlib.sha256(baseline_session.read_bytes()).hexdigest()
    log=(output/'gzserver.log').open('w')
    server=subprocess.Popen(['gzserver','--verbose','-s','libgazebo_ros_init.so','-s','libgazebo_ros_factory.so',str(world)],
        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rclpy.init();node=Node('wheelset_process_demo');trajectory=[];images={};cleanup={};runtime_error=None
    frame={'message':None,'count':0}
    def on_image(message):frame.update(message=message,count=frame['count']+1)
    image_qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
    subscription=node.create_subscription(Image,'/wheelset_review/reference/image_raw',on_image,image_qos)
    setter=node.create_client(SetEntityState,'/wheelset_review/set_entity_state')
    getter=node.create_client(GetEntityState,'/wheelset_review/get_entity_state')
    def call(client,request):
        future=client.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=10)
        if not future.done() or future.result() is None or not future.result().success:
            raise RuntimeError('Gazebo entity-state service failed')
        return future.result()
    def read_pose():
        request=GetEntityState.Request();request.name=workpiece['id'];request.reference_frame='world'
        p=call(getter,request).state.pose.position
        return [p.x,p.y,p.z]
    def write_pose(pose):
        request=SetEntityState.Request();request.state.name=workpiece['id'];request.state.reference_frame='world'
        p=request.state.pose.position;p.x,p.y,p.z=pose
        request.state.pose.orientation.w=1.0
        call(setter,request)
    def capture(name):
        wanted=frame['count']+2;deadline=time.monotonic()+45
        while frame['count']<wanted and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.2)
        if frame['count']<wanted:raise RuntimeError('Process camera timed out')
        path=output/(name+'.png');save_camera(frame['message'],path)
        images[name]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    start=time.monotonic()
    try:
        for client in [setter,getter]:
            if not client.wait_for_service(timeout_sec=45):raise RuntimeError('Process Gazebo startup timed out')
        controller.start(time.monotonic());capture('start')
        while controller.state not in {'complete','failed'}:
            if controller.state=='transferring':
                command=dict(controller.command)
                if command['operation']==args.inject_failure:
                    now=time.monotonic()
                    controller.observe(run_id=controller.run_id,command_id=command['command_id'],pose=read_pose(),
                        observed_at=now,received_at=time.monotonic(),success=False)
                    continue
                source=read_pose();target=command['target'];distance=math.dist(source,target)
                move_start=time.monotonic();duration=distance/args.speed
                while True:
                    ratio=min(1,(time.monotonic()-move_start)/max(duration,.001))
                    requested=[a+(b-a)*ratio for a,b in zip(source,target)]
                    write_pose(requested);observed=read_pose();now=time.monotonic()
                    trajectory.append(dict(command_id=command['command_id'],time_s=now,
                        requested=requested,observed=observed,position_error_m=math.dist(requested,observed)))
                    if math.dist(requested,observed)>.03:raise RuntimeError('Kinematic movement readback mismatch')
                    if ratio==1:break
                    controller.tick(now)
                    if controller.state=='failed':break
                    rclpy.spin_once(node,timeout_sec=.1)
                if controller.state=='failed':continue
                accepted=controller.observe(run_id=controller.run_id,command_id=command['command_id'],
                    pose=observed,observed_at=now,received_at=time.monotonic())
                if not accepted:raise RuntimeError('Final movement feedback rejected')
                capture(command['operation'])
            else:
                rclpy.spin_once(node,timeout_sec=.1);controller.tick(time.monotonic())
        capture('final')
    except Exception as error:
        runtime_error=str(error)
        controller.fail(str(error),time.monotonic())
    finally:
        node.destroy_subscription(subscription);node.destroy_node();rclpy.shutdown()
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:server.wait(timeout=8)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        log.close();cleanup={'owned_server_pid':server.pid,'owned_server_stopped':server.poll() is not None}
    report=controller.report()
    expected_failure=args.inject_failure is not None
    passed=(controller.state=='failed' and controller.fault=='execution_failed' and
        controller.locked_station==args.inject_failure) if expected_failure else controller.state=='complete'
    preserved=hashlib.sha256(baseline_session.read_bytes()).hexdigest()==baseline_before
    report.update(passed=passed and preserved and runtime_error is None and cleanup['owned_server_stopped'],runtime_error=runtime_error,
        execution='gazebo_kinematic_service_readback',
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__),Path(__file__).with_name('process_runtime.py')]},
        build_fingerprint=fingerprint,elapsed_wall_s=time.monotonic()-start,speed_m_per_wall_s=args.speed,
        service_seconds=args.service_seconds,world_sha256=hashlib.sha256(world.read_bytes()).hexdigest(),
        baseline_session_preserved=preserved,removed_static_snapshot_ids=[s['id'] for s in snapshots],
        motion_sample_count=len(trajectory),max_readback_error_m=max((p['position_error_m'] for p in trajectory),default=None),
        injected_failure=args.inject_failure,images=images,cleanup=cleanup,
        claim_boundary='Single workpiece, three software-timed operations and outfeed with Gazebo kinematic pose feedback. No joint actuation/contact/drive physics, production cycle time, multi-workpiece coordination or AGV integration.')
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    (output/'trajectory.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in trajectory))
    print(json.dumps({k:report[k] for k in ['passed','state','completed_operations','motion_sample_count','max_readback_error_m','baseline_session_preserved']}))
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
