"""Effort-driven inspection fixture, joint-feedback process and stalled-axis case."""
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
import time

from .actuated_cell import JointSample,InspectionCycle,efforts
from .actuated_world import build_actuated_world
from .build import ROOT,save_xml
from .evidence import build_fingerprint
from .run_process_demo import save_camera


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--case',choices=['normal','stalled_spindle','feedback_loss'],default='normal')
    parser.add_argument('--port',type=int,default=11362)
    parser.add_argument('--domain',type=int,default=76)
    args=parser.parse_args()
    if os.name=='nt':raise SystemExit('Run on the Ubuntu ROS2/Gazebo host.')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',args.port))==0:raise SystemExit('Actuated-demo port occupied; preserved.')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    os.environ['ROS_DOMAIN_ID']=str(args.domain)
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from gazebo_msgs.srv import ApplyJointEffort,JointRequest,GetEntityState
    from sensor_msgs.msg import JointState,Image
    tree,removed,assumptions=build_actuated_world()
    world=output/'actuated.world';save_xml(tree,world)
    fingerprint=build_fingerprint()
    baseline=ROOT/'simulation/wheelset_workshop/runtime/session.json'
    before=hashlib.sha256(baseline.read_bytes()).hexdigest()
    env=dict(os.environ,GAZEBO_MASTER_URI=f'http://127.0.0.1:{args.port}',DISPLAY=':0',
        GAZEBO_MODEL_PATH=str(ROOT/'assets/wheelset_workshop/models'),GAZEBO_MODEL_DATABASE_URI='',
        LIBGL_ALWAYS_SOFTWARE='1',OMP_NUM_THREADS='4')
    log=(output/'gzserver.log').open('w')
    server=subprocess.Popen(['gzserver','--verbose','-s','libgazebo_ros_init.so','-s','libgazebo_ros_factory.so',
        '-s','libgazebo_ros_force_system.so',str(world)],
        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    cycle=InspectionCycle();samples=[];commands=[];images=[];pose_checks=[];runtime_error=None
    latest={'sample':None,'receipt':None,'image':None,'image_stamp':None}
    feedback_history=deque(maxlen=400)
    rclpy.init();node=Node('wheelset_actuated_demo')
    def joints(message):
        data={name:(p,v) for name,p,v in zip(message.name,message.position,message.velocity)}
        if not {'wsd_slide','wsd_probe','wsd_spin'}<=data.keys():return
        stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        latest.update(sample=JointSample(stamp,*data['wsd_slide'],*data['wsd_probe'],*data['wsd_spin']),receipt=time.monotonic())
        feedback_history.append(latest['sample'])
    def camera(message):
        latest['image']=message
        latest['image_stamp']=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.RELIABLE,durability=DurabilityPolicy.VOLATILE)
    joint_sub=node.create_subscription(JointState,'/wheelset_actuation/joint_states',joints,qos)
    image_sub=node.create_subscription(Image,'/wheelset_review/reference/image_raw',camera,qos)
    apply=node.create_client(ApplyJointEffort,'/apply_joint_effort')
    clear=node.create_client(JointRequest,'/clear_joint_efforts')
    getter=node.create_client(GetEntityState,'/wheelset_review/get_entity_state')
    def call(client,request):
        future=client.call_async(request);rclpy.spin_until_future_complete(node,future,timeout_sec=5)
        if not future.done() or future.result() is None:raise RuntimeError('Actuation service timed out')
        result=future.result()
        if hasattr(result,'success') and not result.success:raise RuntimeError(result.status_message)
        return result
    def actuate(values,sample):
        windows=[]
        for name,value in values.items():
            request=JointRequest.Request();request.joint_name=name;call(clear,request)
            request=ApplyJointEffort.Request();request.joint_name=name;request.effort=float(value)
            stamp=latest['sample'].time_s
            request.start_time.sec=int(stamp);request.start_time.nanosec=int(round((stamp-int(stamp))*1e9))
            request.duration.sec=0;request.duration.nanosec=250000000
            call(apply,request)
            windows.append(dict(joint=name,start_sim_time_s=stamp,duration_s=.25,accepted=True))
        commands.append(dict(sim_time_s=sample.time_s,phase=cycle.state,station=cycle.station,efforts=values,
            request_windows=windows))
    def save_frame():
        stamp=latest['image_stamp']
        if latest['image'] is None or images and stamp<=images[-1]['sim_time_s']:return
        path=output/'frames'/f'{len(images):05d}.png';path.parent.mkdir(exist_ok=True)
        save_camera(latest['image'],path)
        images.append(dict(sim_time_s=stamp,file='frames/'+path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    started=time.monotonic();last_control=-math.inf;last_pose_check=-math.inf;injected=False
    final_stop=None;last_feedback_used=None
    def stop_axes():
        nonlocal last_control
        if latest['sample'] is None:return None
        hold=latest['sample'].slide;stop_start=latest['sample'].time_s;deadline=time.monotonic()+15
        while time.monotonic()<deadline:
            rclpy.spin_once(node,timeout_sec=.03);s=latest['sample']
            if time.monotonic()-latest['receipt']>1:raise RuntimeError('Stop feedback unavailable')
            if s.time_s-last_control<.05:continue
            samples.append(dict(s.__dict__,phase='stopping',station=cycle.station))
            actuate(efforts(s,(hold,0,0)),s);last_control=s.time_s;save_frame()
            if (s.time_s-stop_start>.5 and abs(s.slide_velocity)<.04 and abs(s.angular_velocity)<.025
                    and abs(s.probe)<.005 and abs(s.probe_velocity)<.015):
                return dict(s.__dict__)
        raise RuntimeError('Measured axes did not stop/home')
    try:
        for client in [apply,clear,getter]:
            if not client.wait_for_service(timeout_sec=45):raise RuntimeError('Actuation service unavailable')
        deadline=time.monotonic()+30
        while latest['sample'] is None and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.1)
        if latest['sample'] is None:raise RuntimeError('Joint feedback unavailable')
        while cycle.state not in {'complete','failed'}:
            rclpy.spin_once(node,timeout_sec=.02)
            if server.poll() is not None:raise RuntimeError('Owned Gazebo server exited')
            if time.monotonic()-started>240:raise RuntimeError('Whole-cycle wall timeout')
            s=latest['sample']
            if args.case=='feedback_loss' and cycle.station=='magnetic' and cycle.state=='rotate':injected=True
            if time.monotonic()-latest['receipt']>1 or injected and args.case=='feedback_loss':
                # Injection discards input at the controller boundary, while logging
                # the still-live observer for the final braking check.
                if args.case=='feedback_loss' and injected:
                    if last_feedback_used is None:last_feedback_used=time.monotonic()
                    if time.monotonic()-last_feedback_used<1:continue
                cycle.fail('joint_feedback_stale',s.time_s);break
            if s.time_s-last_control<.05:continue
            old_events=len(cycle.events);cycle.update(s)
            if len(cycle.events)>old_events:
                print(json.dumps(dict(progress=True,phase=cycle.state,station=cycle.station,
                    sim_time_s=s.time_s,completed=cycle.completed)),flush=True)
            samples.append(dict(s.__dict__,phase=cycle.state,station=cycle.station))
            values=efforts(s,cycle.targets())
            if args.case=='stalled_spindle' and cycle.station=='magnetic' and cycle.state=='rotate':
                values['wsd_spin']=0.;injected=True
            actuate(values,s);last_control=s.time_s
            if s.time_s-last_pose_check>=.5:
                request=GetEntityState.Request();request.name='actuated_fixture::wheelset';request.reference_frame='world'
                response=call(getter,request);p=response.state.pose.position
                stamp=response.header.stamp.sec+response.header.stamp.nanosec*1e-9
                deadline=time.monotonic()+1
                while latest['sample'].time_s<stamp and time.monotonic()<deadline:
                    rclpy.spin_once(node,timeout_sec=.01)
                current=min(feedback_history,key=lambda item:abs(item.time_s-stamp))
                lag=abs(current.time_s-stamp);error=abs(p.x-(10+current.slide))
                pose_checks.append(dict(sim_time_s=stamp,world_pose=[p.x,p.y,p.z],
                    joint_time_s=current.time_s,sample_time_error_s=lag,
                    joint_slide=current.slide,x_consistency_error_m=error))
                if lag>.05:raise RuntimeError('No time-aligned joint sample for link position')
                if error>.15 or abs(p.y-50)>.02 or abs(p.z-.56)>.02:
                    raise RuntimeError('Link position inconsistent with constrained joint')
                last_pose_check=s.time_s
            save_frame()
    except Exception as error:
        runtime_error=str(error)
        cycle.fail(str(error),latest['sample'].time_s if latest['sample'] else 0)
    finally:
        # Runtime exceptions must take the same measured stopping path as a
        # normal finish or a latched process fault, before clearing commands.
        if latest['sample'] is not None and server.poll() is None:
            try:final_stop=stop_axes()
            except Exception as error:
                runtime_error='; '.join(filter(None,[runtime_error,str(error)]))
        if clear.service_is_ready():
            for name in ['wsd_slide','wsd_probe','wsd_spin']:
                try:
                    request=JointRequest.Request();request.joint_name=name;call(clear,request)
                except Exception:pass
        node.destroy_subscription(joint_sub);node.destroy_subscription(image_sub);node.destroy_node();rclpy.shutdown()
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:server.wait(timeout=8)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        log.close()
    passed=(cycle.state=='complete' and cycle.completed==['derust','magnetic','ultrasonic','outfeed']) if args.case=='normal' else (
        cycle.state=='failed' and cycle.completed==['derust'] and injected and
        cycle.fault==('rotate_timeout' if args.case=='stalled_spindle' else 'joint_feedback_stale'))
    preserved=hashlib.sha256(baseline.read_bytes()).hexdigest()==before
    source_names=['actuated_cell.py','actuated_world.py','run_actuated_demo.py','run_process_demo.py','build.py','evidence.py']
    report=dict(passed=passed and runtime_error is None and final_stop is not None and preserved,
        case=args.case,state=cycle.state,fault=cycle.fault,completed_operations=cycle.completed,
        retained_station=cycle.station if cycle.state=='failed' else None,events=cycle.events,
        runtime_error=runtime_error,final_stop=final_stop,build_fingerprint=fingerprint,
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in source_names},
        world_sha256=hashlib.sha256(world.read_bytes()).hexdigest(),assumptions=assumptions,
        physics_engine='Gazebo Classic ODE, force/torque commands and joint-state feedback',
        removed_static_snapshot_ids=removed,joint_sample_count=len(samples),effort_command_count=len(commands),
        pose_checks=pose_checks,images=images,baseline_session_preserved=preserved,
        elapsed_wall_s=time.monotonic()-started,owned_server_pid=server.pid,owned_server_stopped=server.poll() is not None,
        claim_boundary='Constrained dynamic fixture with assumed mass/inertia, no gravity or contact geometry. Accepted effort requests and measured joint motion; no force sensor, recovered machinery, wheel/rail contact, full-line execution, production timing, multi-job or AGV integration.')
    for name,rows in [('joint_samples',samples),('effort_commands',commands)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:report[key] for key in ['passed','case','state','fault','completed_operations','runtime_error','joint_sample_count']}),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
