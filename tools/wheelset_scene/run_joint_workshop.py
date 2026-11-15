"""Two SCOUT derivatives execute this scene's scheduler routes using Gazebo truth.

Wheel-driven contact simulation, not SetEntityState. This first adapter uses
Gazebo pose feedback rather than localization/Nav2; simulated handling only.
"""
import argparse
from collections import deque
import gc
import faulthandler
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
import math
import multiprocessing
import os
from pathlib import Path
import signal
import socket
import subprocess
import threading
import time
import uuid
import xml.etree.ElementTree as ET

from .build import ROOT,SCENE,make_world,save_xml,box_model
from .light_material_network import audit,proposed_network,point_segment
from .light_material_scheduler import build_scheduler
from .run_process_demo import save_camera
from .multi_actuated_world import build_multi_world
from .supply_bridge import add_bridge
from .material_supply import MaterialSupply
from .joint_process import run_worker


def wrap(a):return math.atan2(math.sin(a),math.cos(a))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--port',type=int,default=11366);parser.add_argument('--domain',type=int,default=80)
    parser.add_argument('--case',choices=['normal','feedback_recovery'],default='normal')
    parser.add_argument('--whole-line',action='store_true',help='Two identified eighteen-stage workers sharing this fleet and server')
    args=parser.parse_args()
    if os.name=='nt':raise SystemExit('Run on the Ubuntu ROS2/Gazebo host.')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',args.port))==0:raise SystemExit('AGV-demo port occupied; preserved.')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False);os.environ['ROS_DOMAIN_ID']=str(args.domain)
    source_names=['run_joint_workshop.py','joint_process.py','material_supply.py','supply_bridge.py',
        'light_material_network.py','light_material_scheduler.py','multi_actuated_world.py','device_action.py',
        'resource_runtime.py','actuated_cell.py','actuated_world.py','build.py','evidence.py','run_process_demo.py']
    if args.whole_line:
        source_names += ['whole_line_runtime.py','whole_line_actions.py','whole_line_process.py','whole_line_world.py','whole_line_ipc.py','fleet_truth.py',
            'front_flow.py','front_world.py','assembly_flow.py','assembly_world.py','finishing_flow.py','finishing_world.py',
            'finishing_action.py','press_action.py','assets.py','geometry.py']
    snapshot=output/'executed_sources';snapshot.mkdir()
    source_hashes={}
    for name in source_names:
        data=Path(__file__).with_name(name).read_bytes();(snapshot/name).write_bytes(data)
        source_hashes[name]=hashlib.sha256(data).hexdigest()
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from gazebo_msgs.srv import SpawnEntity,GetEntityState
    from geometry_msgs.msg import Twist
    from sensor_msgs.msg import Image,MultiDOFJointState
    import xacro
    from ament_index_python.packages import get_package_share_directory
    layout=proposed_network();scene=json.loads(SCENE.read_text(encoding='utf-8'));clearance=audit(layout,scene)
    if not clearance['passed']:raise RuntimeError('Road clearance failed')
    camera=dict(id='light_material',position=[32,53,22],target=[32,69.5,.15],horizontal_fov=1.6,width=1280,height=720,reference_second=None)
    jobs=[f'ws{n:03d}' for n in range(1,3 if args.whole_line else 5)]
    whole_meta=None;plugin_build=ROOT/'out/wheelset_axis_servo_build'
    if args.whole_line:
        from .whole_line_world import build_whole_line_world,whole_line_clearance
        whole_meta=build_whole_line_world(output,jobs);whole_clearance=whole_line_clearance(whole_meta)
        (output/'whole_line_metadata.json').write_text(json.dumps(whole_meta,indent=2)+'\n')
        (output/'whole_line_clearance.json').write_text(json.dumps(whole_clearance,indent=2)+'\n')
        if not whole_clearance['passed']:raise RuntimeError('Whole-line cargo sweep preflight failed')
        native_build=json.loads((plugin_build/'build_manifest.json').read_text());assert native_build['passed']
        for relative,digest in native_build['source_sha256'].items():
            data=(ROOT/'tools/ros2/wheelset_axis_servo'/relative).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
            target=snapshot/'native_plugin'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
            source_hashes['native_plugin/'+relative]=digest
        for name in ['libwheelset_axis_servo.so','build_manifest.json','build.log']:
            data=(plugin_build/name).read_bytes();(snapshot/name).write_bytes(data);source_hashes[name]=hashlib.sha256(data).hexdigest()
        assert source_hashes['libwheelset_axis_servo.so']==native_build['library_sha256']
        truth_library='libwheelset_fleet_truth.so'
        data=(plugin_build/truth_library).read_bytes()
        assert hashlib.sha256(data).hexdigest()==native_build['auxiliary_libraries'][truth_library]
        (snapshot/truth_library).write_bytes(data);source_hashes[truth_library]=hashlib.sha256(data).hexdigest()
        tree=ET.parse(output/'front.world').getroot();world=tree.find('world')
        plugin=ET.SubElement(world,'plugin',name='wheelset_fleet_truth',filename=truth_library)
        ET.SubElement(ET.SubElement(plugin,'ros'),'namespace').text='/wheelset_review'
        removed=whole_meta['removed_static_snapshot_ids'];assumptions=whole_meta['assumptions'];origins=whole_meta['job_origins']
        bridge_assumptions=dict(design='Native XYZ tote at 6m travel height and measured equipment-side handoffs; no contact gripping')
    else:
        tree,removed,assumptions,origins=build_multi_world(jobs);world=tree.find('world')
        bridge_assumptions=add_bridge(world)
    world.find('physics/max_step_size').text='.002'
    world.find('physics/real_time_update_rate').text='500';world.find("model[@name='review_camera']/link/sensor/update_rate").text='2'
    nodes={n['id']:n for n in layout['nodes']}
    for lane in layout['lanes']:
        a,b=nodes[lane['src']],nodes[lane['dst']];dx,dy=b['x']-a['x'],b['y']-a['y'];length=math.hypot(dx,dy)
        for side in [-1,1]:
            box_model(world,dict(id='light_'+lane['id']+str(side),center=[(a['x']+b['x'])/2-side*dy/length*.9,
                (a['y']+b['y'])/2+side*dx/length*.9,.012],size=[length,.04,.008],yaw=math.atan2(dy,dx),material='cyan'),collision=False)
    world_path=output/'light_material.world';save_xml(tree,world_path)
    helper=ROOT/'tools/ros2/amr_gazebo_guard/launch/metric_model.py'
    spec=importlib.util.spec_from_file_location('wheelset_metric_model',helper);geometry=importlib.util.module_from_spec(spec);spec.loader.exec_module(geometry)
    share=Path(get_package_share_directory('scout_gazebo_description'));robot_xacro=share/'urdf/scout_mini.urdf.xacro'
    model_sources={str(helper):hashlib.sha256(helper.read_bytes()).hexdigest(),
        str(robot_xacro):hashlib.sha256(robot_xacro.read_bytes()).hexdigest()}
    for path in [share/'urdf/scout_mini.gazebo.xacro',*sorted((share/'meshes').glob('*.dae'))]:
        model_sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    descriptions={}
    for r in layout['robots']:
        robot=ET.fromstring(xacro.process_file(str(robot_xacro),mappings={'drive_mode':'diff','use_lidar':'true'}).toxml())
        geometry.augment(robot,wheel_mu=1.,lateral_mu=.05,contact_scale=10.)
        geometry.namespace_robot(robot,r['id'])
        # Gazebo converts package URIs to model://; this isolated server's
        # model path need not include the external colcon package. Resolve
        # visual meshes explicitly instead of silently losing robot visuals.
        for mesh in robot.findall('.//mesh'):
            prefix='package://scout_gazebo_description/'
            if not mesh.get('filename','').startswith(prefix):raise RuntimeError('Uncovered robot mesh URI')
            path=share/mesh.get('filename')[len(prefix):]
            if not path.is_file():raise RuntimeError('Robot visual mesh missing')
            mesh.set('filename',path.as_uri())
        # This truth-feedback adapter does not consume the laser. Retain its
        # model frames but avoid running an unused ray sensor on a shared host.
        for parent in robot.iter():
            for child in list(parent):
                if child.tag=='sensor':parent.remove(child)
        descriptions[r['id']]=ET.tostring(robot,encoding='unicode');(output/(r['id']+'.urdf')).write_text(descriptions[r['id']])
    sim=build_scheduler(layout,source='gazebo_truth',run_id=uuid.uuid4().hex)
    supply_manager=None;result_queue=None
    if args.whole_line:
        from .whole_line_world import SUPPLY_HANDOFFS
        from .whole_line_ipc import SupplyManager,SharedSupply,run_child,receive_results
        process_context=multiprocessing.get_context('spawn')
        supply_manager=SupplyManager(ctx=process_context);supply_manager.start()
        supply=SharedSupply(supply_manager.MaterialSupply(next(iter(sim.executions.values())).identity['run_id'],
            initial={'derust':0,'magnetic':0},destinations=SUPPLY_HANDOFFS))
    else:supply=MaterialSupply(next(iter(sim.executions.values())).identity['run_id'])
    process_result={};stop_process=process_context.Event() if args.whole_line else threading.Event()
    metadata=dict(jobs=jobs,removed=removed,assumptions=assumptions,origins=origins,bridge_assumptions=bridge_assumptions)
    if args.whole_line:metadata['whole_line']=whole_meta
    def process_worker():
        try:
            worker=run_worker
            if args.whole_line:
                from .whole_line_process import run_worker as worker
            process_result['report']=worker(output/'process',supply,
                'stalled_recovery' if args.case=='feedback_recovery' else 'normal',server,stop_process,metadata)
        except Exception as exc:process_result['error']=repr(exc)
    process_thread=threading.Thread(target=process_worker,daemon=True)
    baseline=ROOT/'simulation/wheelset_workshop/runtime/session.json';before=hashlib.sha256(baseline.read_bytes()).hexdigest()
    env=dict(os.environ,GAZEBO_MASTER_URI=f'http://127.0.0.1:{args.port}',DISPLAY=':0',
        GAZEBO_MODEL_PATH=str(ROOT/'assets/wheelset_workshop/models'),GAZEBO_MODEL_DATABASE_URI='',
        LIBGL_ALWAYS_SOFTWARE='1',OMP_NUM_THREADS='4')
    if args.whole_line:env['GAZEBO_PLUGIN_PATH']=str(plugin_build)+':'+os.environ.get('GAZEBO_PLUGIN_PATH','')
    gc_was_enabled=gc.isenabled()
    if args.whole_line:gc.collect();gc.disable()
    log=(output/'gzserver.log').open('w');server=subprocess.Popen(['gzserver','--verbose',
        '-s','libgazebo_ros_init.so','-s','libgazebo_ros_factory.so','-s','libgazebo_ros_force_system.so',str(world_path)],
        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    if args.whole_line:
        result_queue=process_context.Queue()
        process_thread=process_context.Process(target=run_child,args=(output/'process',supply,
            'stalled_recovery' if args.case=='feedback_recovery' else 'normal',server.pid,stop_process,metadata,result_queue),daemon=True)
    rclpy.init();node=Node('wheelset_joint_fleet');executor=SingleThreadedExecutor();executor.add_node(node)
    fleet_spin_errors=[]
    def fleet_spin():
        try:executor.spin()
        except Exception as exc:fleet_spin_errors.append(repr(exc))
    thread=threading.Thread(target=fleet_spin,daemon=True);thread.start()
    spawn=node.create_client(SpawnEntity,'/spawn_entity');getter=node.create_client(GetEntityState,'/wheelset_review/get_entity_state')
    publishers={r['id']:node.create_publisher(Twist,'/'+r['id']+'/cmd_vel',10) for r in layout['robots']}
    image={'message':None,'stamp':None};frames=[];trace=[];controls=[];routes={};error=None;stops={};start=time.monotonic();epoch=None
    feedback=[];dispatches=[];recovery=None;last_progress=-math.inf;service_waits=[]
    watchdog=None
    if args.whole_line:
        watchdog=(output/'thread_watchdog.log').open('w')
        faulthandler.enable(file=watchdog)
    frame_writer=ThreadPoolExecutor(max_workers=1);frame_futures=[]
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    def on_image(message):image.update(message=message,stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9)
    sub=node.create_subscription(Image,'/wheelset_review/reference/image_raw',on_image,qos)
    native_truth=[];fleet_pose_checks=[];truth_latest={'row':None};truth_errors=[]
    truth_history=deque(maxlen=400);truth_subscription=None;pending_checks={};last_truth_check=-math.inf
    if args.whole_line:
        from .fleet_truth import decode as decode_truth
        def on_truth(message):
            try:
                observation=decode_truth(message)
                row=dict(sequence=len(native_truth)+1,received_at=time.monotonic()-start,
                    source_time_s=observation['scout_a']['sim_time_s'],observations=observation)
                native_truth.append(row);truth_history.append(row);truth_latest['row']=row
            except Exception as exc:truth_errors.append(repr(exc))
        truth_subscription=node.create_subscription(MultiDOFJointState,'/wheelset_review/fleet_truth',on_truth,qos)
    def call(client,req):
        began=time.monotonic()
        future=client.call_async(req);ready=threading.Event();future.add_done_callback(lambda unused:ready.set());ready.wait(5)
        if args.whole_line:service_waits.append(dict(service=client.srv_name,entity=getattr(req,'name',None),wall_s=time.monotonic()-start,duration_s=time.monotonic()-began))
        if not future.done() or future.result() is None or not future.result().success:raise RuntimeError('Robot state/spawn service failed')
        return future.result()
    def read(robot):
        req=GetEntityState.Request();req.name=robot;req.reference_frame='world';reply=call(getter,req)
        return decode(reply)
    def decode(reply):
        p=reply.state.pose.position;q=reply.state.pose.orientation;t=reply.state.twist
        yaw=math.atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z))
        return dict(pose=[p.x,p.y,yaw],z=p.z,linear_speed=math.hypot(t.linear.x,t.linear.y),
            angular_speed=abs(t.angular.z),sim_time_s=reply.header.stamp.sec+reply.header.stamp.nanosec*1e-9)
    def read_fleet():
        if args.whole_line:
            deadline=time.monotonic()+30
            while truth_latest['row'] is None:
                if truth_errors:raise RuntimeError('Native fleet truth: '+truth_errors[0])
                if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Native fleet truth unavailable')
                time.sleep(.003)
            if truth_errors:raise RuntimeError('Native fleet truth: '+truth_errors[0])
            row=truth_latest['row']
            if time.monotonic()-start-row['received_at']>2.:raise RuntimeError('Native fleet truth wall feedback stale')
            return row['observations']
        # Submit both requests before waiting, so host scheduling cannot put an
        # entire service round trip between the two physical observations.
        for attempt in range(3):
            futures={};began=time.monotonic()
            for robot in sim.executions:
                req=GetEntityState.Request();req.name=robot;req.reference_frame='world'
                futures[robot]=getter.call_async(req)
            values={};deadline=time.monotonic()+5
            for robot,future in futures.items():
                ready=threading.Event();future.add_done_callback(lambda unused,event=ready:event.set())
                ready.wait(max(0,deadline-time.monotonic()))
                if not future.done() or future.result() is None or not future.result().success:
                    raise RuntimeError('Fleet state service failed')
                values[robot]=decode(future.result())
            if args.whole_line:service_waits.append(dict(service='coherent_fleet',wall_s=time.monotonic()-start,
                duration_s=time.monotonic()-began,attempt=attempt,source_stamps={r:o['sim_time_s'] for r,o in values.items()},gc_enabled=gc.isenabled()))
            stamps=[o['sim_time_s'] for o in values.values()]
            if max(stamps)-min(stamps)<=.05:return values
        raise RuntimeError('No coherent fleet truth sample in three attempts')
    def cross_check_truth():
        nonlocal last_truth_check
        for robot,(future,began) in list(pending_checks.items()):
            if not future.done():
                if time.monotonic()-began>5:raise RuntimeError('Native fleet independent service check timeout')
                continue
            reply=future.result()
            if reply is None or not reply.success:raise RuntimeError('Native fleet independent service check failed')
            observed=decode(reply);stamp=observed['sim_time_s']
            source=min(list(truth_history),key=lambda row:abs(row['source_time_s']-stamp))
            native=source['observations'][robot];lag=abs(native['sim_time_s']-stamp)
            distance=math.dist(native['pose'][:2],observed['pose'][:2]);yaw=abs(wrap(native['pose'][2]-observed['pose'][2]))
            fleet_pose_checks.append(dict(robot=robot,service_observation=observed,native_observation=native,
                native_sequence=source['sequence'],source_time_error_s=lag,error_m=distance,yaw_error_rad=yaw,
                service_duration_s=time.monotonic()-began))
            if lag>.05 or distance>.03 or yaw>.04 or abs(native['z']-observed['z'])>.03:
                raise RuntimeError('Native fleet/world-service disagreement')
            del pending_checks[robot]
        row=truth_latest['row']
        if not pending_checks and row and row['source_time_s']-last_truth_check>=2:
            for robot in sim.executions:
                req=GetEntityState.Request();req.name=robot;req.reference_frame='world'
                pending_checks[robot]=(getter.call_async(req),time.monotonic())
            last_truth_check=row['source_time_s']
    def publish(robot,v=0.,w=0.):
        message=Twist();message.linear.x=float(v);message.angular.z=float(w);publishers[robot].publish(message)
        controls.append(dict(wall_s=time.monotonic()-start,robot=robot,v=v,w=w,route_id=sim.executions[robot].envelope()['route_id']))
    def observe(gate,event,obs,receipt,*,envelope=None,probe=False):
        message={**(gate.envelope() if envelope is None else envelope),'event':event,'pose':obs['pose'],
            'seq':gate.last_seq+1,'timestamp':obs['sim_time_s']}
        answer=sim.execution_feedback(message,receipt)
        feedback.append(dict(message=message,received_at=receipt,answer=dict(answer),
            measured_state=dict(obs),probe_expected_rejection=probe))
        if not probe and not answer['accepted'] and answer['reason']!='duplicate source timestamp':raise RuntimeError(str(answer))
        return answer
    def write_frame(message,path,stamp,source_stamp):
        save_camera(message,path)
        return dict(time_s=stamp,sim_time_s=source_stamp,file='frames/'+path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    try:
        for client in [spawn,getter]:
            if not client.wait_for_service(timeout_sec=45):raise RuntimeError('AGV services unavailable')
        for r in layout['robots']:
            req=SpawnEntity.Request();req.name=r['id'];req.xml=descriptions[r['id']];req.reference_frame='world'
            req.initial_pose.position.x=nodes[r['start']]['x'];req.initial_pose.position.y=nodes[r['start']]['y']
            req.initial_pose.position.z=.03;req.initial_pose.orientation.z=math.sin(-math.pi/4);req.initial_pose.orientation.w=math.cos(-math.pi/4)
            call(spawn,req)
        process_thread.start()
        last_step=-math.inf;last_frame=-math.inf;previous_completed=0
        accepted_arrivals={};received_tasks=set()
        while not (sim._all_done() and process_result.get('report',{}).get('passed',False)):
            loop=time.monotonic()
            if args.whole_line:
                receive_results(result_queue,process_result)
                if process_thread.exitcode is not None and not process_result:
                    raise RuntimeError('Whole-line worker exited without report: '+str(process_thread.exitcode))
            if fleet_spin_errors:raise RuntimeError('Fleet executor: '+fleet_spin_errors[0])
            if server.poll() is not None:raise RuntimeError('Owned AGV server exited')
            if process_result.get('error'):raise RuntimeError(process_result['error'])
            if process_result.get('report') and not process_result['report']['passed']:raise RuntimeError('Process worker failed: '+str(process_result['report']['runtime_error']))
            if time.monotonic()-start>(7200 if args.whole_line else 900):raise RuntimeError('AGV-run wall timeout')
            observations=read_fleet()
            if args.whole_line:cross_check_truth()
            stamp=min(v['sim_time_s'] for v in observations.values())
            if epoch is None:epoch=stamp
            now=stamp-epoch;receipt=time.monotonic()-start
            candidate=sim.executions['scout_a'].command
            if args.case=='feedback_recovery' and recovery is None and candidate and candidate['nodes']==['west','east']:
                if observations['scout_a']['pose'][0]>22:
                    recovery=dict(robot='scout_a',old_command=dict(candidate),injected_at=receipt,
                        injected_sim_time_s=stamp,retained_pins=json.loads(json.dumps(sim._res.pins())),
                        old_envelope=sim.executions['scout_a'].envelope(),resumed=False)
            sim.poll_execution(receipt)
            for robot,gate in sim.executions.items():
                obs=observations[robot];command=gate.command;event='POSE';v=w=0.
                interrupted=bool(recovery and robot==recovery['robot'] and not recovery['resumed'])
                if command and gate.state=='moving' and not interrupted:
                    routes.setdefault(command['route_id'],dict(command))
                    dx,dy=command['target_xy'][0]-obs['pose'][0],command['target_xy'][1]-obs['pose'][1]
                    distance=math.hypot(dx,dy);heading=wrap(math.atan2(dy,dx)-obs['pose'][2])
                    if distance<.10:
                        if obs['linear_speed']<.025 and obs['angular_speed']<.05:event='ARRIVED'
                    else:
                        w=max(-.6,min(.6,2.2*heading))
                        if abs(heading)<.15:v=min(.4,.8*distance)
                publish(robot,v,w)
                pending=gate.drain_commands()
                dispatches.extend(dict(received_at=receipt,command=dict(c)) for c in pending)
                if interrupted:
                    if gate.fault and 'fault_at' not in recovery:
                        recovery.update(fault_at=receipt,fault=gate.fault,fault_pins=json.loads(json.dumps(sim._res.pins())))
                    if receipt-recovery['injected_at']>=3.2:
                        if not gate.fault:raise RuntimeError('Injected feedback loss did not latch')
                        if obs['linear_speed']<.025 and obs['angular_speed']<.05:
                            observe(gate,'STOPPED',obs,receipt)
                            recovery.update(stopped_at=receipt,stop_evidence=dict(obs))
                            answer=sim.resume_execution(command['route_id'],receipt,robot_id=robot)
                            if not answer['accepted']:raise RuntimeError(str(answer))
                            recovery.update(resumed=True,resumed_at=receipt,new_command=dict(gate.command),
                                resumed_pins=json.loads(json.dumps(sim._res.pins())))
                            old=observe(gate,'ARRIVED',obs,receipt,envelope=recovery['old_envelope'],probe=True)
                            if old['accepted'] or old['reason']!='identity mismatch':raise RuntimeError('Superseded arrival accepted')
                            recovery['old_arrival_rejected']=True
                else:
                    envelope=gate.envelope();answer=observe(gate,event,obs,receipt)
                    if event=='ARRIVED' and answer['accepted']:
                        accepted_arrivals[command['task_id']]=dict(command=dict(command),observation=dict(obs))
            if now-last_step>=.2:sim.step_once();last_step=now
            for task in sim.tasks:
                if task.status.value=='completed' and task.task_id not in received_tasks:
                    arrival=accepted_arrivals[task.task_id];obs=observations[arrival['command']['robot_id']]
                    station,quantity=('magnetic',4) if task.task_id=='inspection_consumables' else ('derust',2)
                    if not supply.arrived(task.task_id,station,quantity,arrival['command'],obs,now=max(v['sim_time_s'] for v in observations.values())):
                        raise RuntimeError('Measured material dock receipt rejected')
                    received_tasks.add(task.task_id)
            sim.poll_execution(time.monotonic()-start)
            if any(g.fault and not (recovery and r==recovery['robot'] and not recovery['resumed'])
                for r,g in sim.executions.items()):raise RuntimeError('Robot execution gate latched')
            pins=sim._res.pins();assert all(len(owners)==1 for group in pins.values() for owners in group.values())
            separation=math.dist(*(v['pose'][:2] for v in observations.values()))
            state_skew=abs(observations['scout_a']['sim_time_s']-observations['scout_b']['sim_time_s'])
            if state_skew>.15:raise RuntimeError('Robot truth samples too far apart in time')
            if separation-.65*state_skew<2*(layout['robot_enclosing_radius_m']+layout['safety_margin_m']):
                raise RuntimeError('Observed robot safety margins overlap')
            for robot,gate in sim.executions.items():
                if gate.command:
                    a,b=[(nodes[n]['x'],nodes[n]['y']) for n in gate.command['nodes']]
                    if point_segment(observations[robot]['pose'][:2],a,b)>.15:
                        raise RuntimeError('Robot left its reserved road segment')
            trace.append(dict(time_s=now,wall_s=receipt,observations=observations,pins=pins,
                commands={r:g.command for r,g in sim.executions.items()},tasks={t.task_id:t.status.value for t in sim.tasks},
                gates={r:g.snapshot() for r,g in sim.executions.items()}))
            if args.whole_line:
                trace[-1]['native_truth_sequence']=next(row['sequence'] for row in reversed(truth_history) if row['observations'] is observations)
            if image['message'] is not None and now-last_frame>=1. and sum(not f.done() for f in frame_futures)<2:
                path=output/'frames'/f'{len(frame_futures):05d}.png';path.parent.mkdir(exist_ok=True)
                frame_futures.append(frame_writer.submit(write_frame,image['message'],path,now,image['stamp']));last_frame=now
            completed=sum(t.status.value=='completed' for t in sim.tasks)
            if completed!=previous_completed:
                print(json.dumps(dict(progress=True,completed=completed,routes=len(routes),elapsed_sim_s=now)),flush=True);previous_completed=completed
            if now-last_progress>=20:
                print(json.dumps(dict(progress=True,elapsed_sim_s=now,completed=completed,
                    poses={r:o['pose'][:2] for r,o in observations.items()},recovered=bool(recovery and recovery['resumed']))),flush=True)
                last_progress=now
            time.sleep(max(0,.05-(time.monotonic()-loop)))
    except Exception as exc:error=str(exc)
    finally:
        deadline=time.monotonic()+8
        try:
            while time.monotonic()<deadline:
                for robot in publishers:publish(robot)
                observations={robot:read(robot) for robot in publishers}
                if all(o['linear_speed']<.025 and o['angular_speed']<.05 for o in observations.values()):stops=observations;break
                time.sleep(.1)
        except Exception as exc:error='; '.join(filter(None,[error,str(exc)]))
        stop_process.set()
        if process_thread.ident is not None:process_thread.join(timeout=35)
        if args.whole_line:receive_results(result_queue,process_result)
        if process_thread.is_alive():error='; '.join(filter(None,[error,'Process worker did not stop']))
        executor.shutdown(timeout_sec=5);thread.join(timeout=5);node.destroy_subscription(sub)
        if truth_subscription is not None:node.destroy_subscription(truth_subscription)
        node.destroy_node();rclpy.shutdown()
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:server.wait(timeout=8)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        log.close()
        if watchdog is not None:faulthandler.disable();watchdog.close()
        if args.whole_line and gc_was_enabled:gc.enable()
        frame_writer.shutdown(wait=True)
        try:frames=[future.result() for future in frame_futures]
        except Exception as exc:error='; '.join(filter(None,[error,'Frame writer: '+str(exc)]))
    content=''.join(json.dumps(row)+'\n' for row in trace).encode();(output/'trace.jsonl').write_bytes(content)
    (output/'cmd_vel.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in controls))
    for name,rows in [('feedback',feedback),('dispatches',dispatches)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    preserved=hashlib.sha256(baseline.read_bytes()).hexdigest()==before
    if 'report' in process_result:
        # One asynchronous recorder owns the shared-world image stream.
        # Avoid duplicate ROS image deserialization and synchronous PNG work
        # on the process/control path. Reuse the exact saved bytes and their
        # camera source timestamps, not the process polling time.
        process_result['report']['images']=[dict(frame,time_s=frame['sim_time_s'],file='../'+frame['file']) for frame in frames]
        process_result['report']['image_source']='Shared same-world fleet recorder; camera source timestamps and original bytes.'
        process_result['report'].update(owned_server_stopped=server.poll() is not None,baseline_session_preserved=preserved)
        (output/'process/result.json').write_text(json.dumps(process_result['report'],indent=2)+'\n')
    (output/'supply_events.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in supply.events))
    case_passed=args.case=='normal' or bool(recovery and recovery['resumed'] and recovery['old_arrival_rejected'])
    report=dict(passed=error is None and sim._all_done() and bool(stops) and preserved and case_passed and process_result.get('report',{}).get('passed',False),runtime_error=error,
        case=args.case,recovery=recovery,process_report=process_result.get('report'),supply_initial=supply.initial,
        supply_final=supply.stock,supply_deliveries=supply.deliveries,supply_capacity=supply.capacity,
        completed_tasks=[t.task_id for t in sim.tasks if t.status.value=='completed'],routes=list(routes.values()),
        observations=len(trace),trace_sha256=hashlib.sha256(content).hexdigest(),images=frames,final_stop=stops,
        network=layout,clearance=clearance,model_source_sha256=model_sources,baseline_session_preserved=preserved,
        owned_server_pid=server.pid,owned_server_stopped=server.poll() is not None,elapsed_wall_s=time.monotonic()-start,
        source_sha256={name:source_hashes[name]
            for name in ['run_joint_workshop.py','joint_process.py','material_supply.py','supply_bridge.py','light_material_network.py','light_material_scheduler.py']},
        claim_boundary='Two local AgileX/community-derived SCOUT Mini surrogates with chassis/wheel contact, differential-wheel drive, existing AMR scheduler and Gazebo truth pose feedback on a new light-material road. Same Gazebo world also runs four effort-driven wheelsets and an assumed contact-free XYZ material shuttle; measured dock and shuttle arrival commit finite supplies that gate device starts. No localization/Nav2, calibrated hardware, recovered factory handoff, contact gripping, full-line or industrial acceptance.')
    report['data_sha256']={name:hashlib.sha256((output/name).read_bytes()).hexdigest()
        for name in ['trace.jsonl','cmd_vel.jsonl','feedback.jsonl','dispatches.jsonl','light_material.world','scout_a.urdf','scout_b.urdf','supply_events.jsonl']}
    if args.whole_line:
        report.update(mode='whole_line',source_sha256=source_hashes,supply_destinations=supply.destinations,
            fleet_feedback_method='native_world_truth',
            claim_boundary='Two SCOUT derivatives with wheel contact and Gazebo truth feedback on independent light-material roads, existing AMR scheduler, and two identified eighteen-stage constrained wheelset workers in the same Gazebo. Native XYZ tote and measured finite supplies/stations/bins/storage. Assumed process/masses/layout; wheelset/parts/tote contact and gravity excluded. No process-quality, scene localization/Nav2 or industrial acceptance.')
        for name in ['whole_line_metadata.json','whole_line_clearance.json','bare_wheelset.dae']:
            report['data_sha256'][name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
        (output/'service_waits.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in service_waits))
        for name in ['service_waits.jsonl','thread_watchdog.log']:
            report['data_sha256'][name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
        for name,records in [('native_truth',native_truth),('fleet_pose_checks',fleet_pose_checks)]:
            path=output/(name+'.jsonl');path.write_text(''.join(json.dumps(row)+'\n' for row in records))
            report['data_sha256'][path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n')
    if supply_manager is not None:supply_manager.shutdown()
    print(json.dumps({k:report[k] for k in ['passed','runtime_error','completed_tasks','observations']}),flush=True)
    if not report['passed']:raise SystemExit(1)


if __name__=='__main__':main()
