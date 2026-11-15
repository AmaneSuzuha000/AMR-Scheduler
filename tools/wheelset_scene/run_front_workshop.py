"""Execute front-end dismantling and inspection with XYZ/part feedback."""
import argparse
import gc
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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
from .build import ROOT
from .device_action import DeviceAction
from .evidence import build_fingerprint
from .front_flow import FrontRuntime,ORIGIN,STAGES,TRAVEL_Z,Waypoints,parts,transfer_points
from .front_world import build_front_world,planned_clearance
from .assembly_flow import AssemblyRuntime,components
from .assembly_world import build_assembly_world,assembly_clearance
from .press_action import PressAction,PressSample
from .finishing_flow import FinishingRuntime,STORAGE_APPROACH,STORAGE_UNLOAD,STORED_JOB,PAINT
from .finishing_world import build_finishing_world,finishing_clearance
from .finishing_action import RunInAction,RunInSample
from .run_process_demo import save_camera


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['normal','demount_recovery','press_recovery'],default='normal')
    p.add_argument('--through',choices=['front','assembly','storage'],default='front')
    p.add_argument('--port',type=int,default=11368);p.add_argument('--domain',type=int,default=82)
    p.add_argument('--wall-timeout-s',type=float,default=None,
        help='Overall host-time budget; feedback, lease and motion deadlines stay unchanged')
    args=p.parse_args()
    assembly=args.through in {'assembly','storage'};finishing=args.through=='storage'
    wall_timeout=args.wall_timeout_s if args.wall_timeout_s is not None else (7200 if finishing else 2400 if assembly else 1200)
    if not math.isfinite(wall_timeout) or wall_timeout<=0:p.error('positive finite wall timeout required')
    shutting_down=False
    if args.case=='press_recovery' and not assembly:p.error('press_recovery requires --through assembly or storage')
    if os.name=='nt':raise SystemExit('Run on Ubuntu with ROS2/Gazebo.')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',args.port))==0:raise SystemExit('Front-demo port occupied; preserved.')
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    names=['run_front_workshop.py','front_flow.py','front_world.py','actuated_cell.py','actuated_world.py',
        'device_action.py','assets.py','geometry.py','build.py','evidence.py','run_process_demo.py','light_material_network.py',
        'assembly_flow.py','assembly_world.py','press_action.py','finishing_flow.py','finishing_world.py','finishing_action.py']
    saved=output/'executed_sources';saved.mkdir();source_hashes={}
    for name in names:
        data=Path(__file__).with_name(name).read_bytes();(saved/name).write_bytes(data)
        source_hashes[name]=hashlib.sha256(data).hexdigest()
    plugin_source=ROOT/'tools/ros2/wheelset_axis_servo';plugin_build=ROOT/'out/wheelset_axis_servo_build'
    native_build=json.loads((plugin_build/'build_manifest.json').read_text())
    assert native_build['passed']
    for relative,digest in native_build['source_sha256'].items():
        data=(plugin_source/relative).read_bytes();assert hashlib.sha256(data).hexdigest()==digest
        path=saved/'native_plugin'/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(data)
        source_hashes['native_plugin/'+relative]=digest
    for name in ['libwheelset_axis_servo.so','build_manifest.json','build.log']:
        data=(plugin_build/name).read_bytes();(saved/name).write_bytes(data);source_hashes[name]=hashlib.sha256(data).hexdigest()
    assert source_hashes['libwheelset_axis_servo.so']==native_build['library_sha256']
    meta=build_finishing_world(output) if finishing else build_assembly_world(output) if assembly else build_front_world(output)
    clearance=finishing_clearance(meta) if finishing else assembly_clearance(meta) if assembly else planned_clearance(meta)
    (output/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n')
    (output/'planned_clearance.json').write_text(json.dumps(clearance,indent=2)+'\n')
    if not clearance['passed']:raise SystemExit('Cargo sweep preflight failed; see planned_clearance.json')
    os.environ['ROS_DOMAIN_ID']=str(args.domain)
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.qos import QoSProfile,ReliabilityPolicy,DurabilityPolicy
    from gazebo_msgs.srv import GetEntityState
    from sensor_msgs.msg import JointState,Image
    from std_msgs.msg import Float64MultiArray
    baseline=ROOT/'simulation/wheelset_workshop/runtime/session.json';before=hashlib.sha256(baseline.read_bytes()).hexdigest()
    env=dict(os.environ,GAZEBO_MASTER_URI=f'http://127.0.0.1:{args.port}',DISPLAY=':0',
        GAZEBO_MODEL_PATH=str(ROOT/'assets/wheelset_workshop/models'),GAZEBO_MODEL_DATABASE_URI='',
        LIBGL_ALWAYS_SOFTWARE='1',OMP_NUM_THREADS='4',
        GAZEBO_PLUGIN_PATH=str(plugin_build)+':'+os.environ.get('GAZEBO_PLUGIN_PATH',''))
    log=(output/'gzserver.log').open('w');server=subprocess.Popen(['gzserver','--verbose',
        '-s','libgazebo_ros_init.so','-s','libgazebo_ros_factory.so','-s','libgazebo_ros_force_system.so',str(output/'front.world')],
        env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    # All retained samples/commands are acyclic dictionaries. Full cyclic-GC
    # scans of this growing evidence store stopped Python callbacks for >1s
    # in assembly03. Collect before arming; defer cyclic collection until
    # after our server stops, leaving feedback/lease gates unchanged.
    gc.collect();gc_was_enabled=gc.isenabled();gc.disable()
    rclpy.init();node=Node('wheelset_front_workshop');executor=SingleThreadedExecutor();executor.add_node(node)
    latest={};receipts={};history={name:deque(maxlen=400) for name in meta['definitions']}
    runtime=FinishingRuntime(output.name) if finishing else AssemblyRuntime(output.name) if assembly else FrontRuntime(output.name)
    recipe=runtime.recipe;samples=[];requests=[];waypoints=[];pose_checks=[];device_events=[];press_samples=[]
    native_commands=[];command_lookup={};service_waits=[];lease_fault_observation=None
    control_context=dict(inject=False,disabled=None);gc_events=[];gc_began={}
    def gc_timing(phase,info):
        generation=info['generation']
        if phase=='start':gc_began[generation]=time.monotonic()
        else:gc_events.append(dict(generation=generation,wall_s=time.monotonic()-gc_began.get(generation,time.monotonic()),
            source_time_s=max([0.]+[s['time_s'] for s in latest.values()])))
    gc.callbacks.append(gc_timing)
    targets={name:dict(xyz=list(d['origin']),spin=0.,probe=0.) for name,d in meta['definitions'].items()}
    held_parts=set();phase='initial';last_tick=-math.inf;last_frame=-math.inf;last_progress=-math.inf
    images=[];frame_futures=[];image={'message':None,'stamp':None};writer=ThreadPoolExecutor(max_workers=1)
    last_world=-math.inf;world_index=0
    failure=None;fault=None;recovered=False;old_completion_rejected=False;final_stop=None;start=time.monotonic()
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT,durability=DurabilityPolicy.VOLATILE)
    def callback(name,message):
        d=meta['definitions'][name];data={n:(p,v) for n,p,v in zip(message.name,message.position,message.velocity)}
        if not all(name+'_'+a in data for a in d['axes']):return
        stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9
        force=dict(zip(message.name,message.effort))
        row=dict(time_s=stamp,xyz=[d['origin'][i]+data.get(name+'_'+a,(0.,0.))[0] for i,a in enumerate('xyz')],
            velocity=[data.get(name+'_'+a,(0.,0.))[1] for a in 'xyz'],
            spin=data.get(name+'_spin',(0.,0.))[0],omega=data.get(name+'_spin',(0.,0.))[1],
            probe=data.get(name+'_probe',(0.,0.))[0],probe_velocity=data.get(name+'_probe',(0.,0.))[1],
            applied_effort={a:force[name+'_'+a] for a in d['axes']},
            servo_sequence=int(data[name+'_servo_sequence'][0]),servo_fault=bool(data[name+'_servo_fault'][0]))
        latest[name]=row;receipts[name]=time.monotonic();history[name].append(row)
    subscriptions=[node.create_subscription(JointState,'/'+name+'/joint_states',
        lambda message,name=name:callback(name,message),qos) for name in meta['definitions']]
    def on_image(message):image.update(message=message,stamp=message.header.stamp.sec+message.header.stamp.nanosec*1e-9)
    subscriptions.append(node.create_subscription(Image,'/wheelset_review/reference/image_raw',on_image,qos))
    publishers={name:node.create_publisher(Float64MultiArray,'/'+name+'/servo_targets',1) for name in targets}
    getter=node.create_client(GetEntityState,'/wheelset_review/get_entity_state')
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start()
    def wait(future):
        # Entity queries can return a source stamp from before a delayed RPC
        # response. Keep holding the current measured command while awaiting
        # the response, rather than starving the native one-second lease.
        began=time.monotonic();source_begin=now();deadline=began+5
        while not future.done() and time.monotonic()<deadline:
            if now()-last_tick>=.05:tick(allow_world=False,**control_context)
            else:time.sleep(.002)
        service_waits.append(dict(wall_s=time.monotonic()-began,source_begin_s=source_begin,
            source_end_s=now(),command_id=runtime.current))
        if not future.done() or future.result() is None:raise RuntimeError('Front service timeout')
        result=future.result()
        if hasattr(result,'success') and not result.success:raise RuntimeError(result.status_message)
        return result
    def now():return max([runtime.last]+[s['time_s'] for s in latest.values()])
    def capture(message,path,stamp):
        save_camera(message,path)
        return dict(time_s=stamp,file='frames/'+path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    def tick(*,inject=False,disabled=None,allow_world=True):
        nonlocal last_tick,last_frame,last_progress,last_world,world_index,lease_fault_observation
        if allow_world:control_context.update(inject=inject,disabled=disabled)
        while True:
            if server.poll() is not None:raise RuntimeError('Owned front server exited')
            if not shutting_down and time.monotonic()-start>wall_timeout:raise RuntimeError('Front run wall timeout')
            if any(time.monotonic()-receipts.get(name,start)>1 for name in targets):raise RuntimeError('Front joint feedback stale')
            stamp=now()
            if stamp-last_tick>=.05:break
            time.sleep(.002)
        # Independent model publishers may arrive on adjacent callbacks. Allow
        # a bounded resample before latching, retaining the .1s skew gate.
        deadline=time.monotonic()+.25
        while True:
            observation={name:dict(latest[name]) for name in targets}
            skew=max(s['time_s'] for s in observation.values())-min(s['time_s'] for s in observation.values())
            if skew<=.1:break
            if time.monotonic()>=deadline:
                raise RuntimeError('Front joint source times diverged: '+str({n:s['time_s'] for n,s in observation.items()}))
            time.sleep(.003)
        stamp=now()
        if any(s['servo_fault'] for s in observation.values()):
            lease_fault_observation=observation
            raise RuntimeError('Native servo command lease latched stop')
        samples.append(dict(time_s=stamp,command_id=runtime.current,phase=phase,observations=observation,
            targets={name:dict(t) for name,t in targets.items()},part_state=dict(runtime.part_state)))
        for name,d in meta['definitions'].items():
            s=observation[name];t=targets[name];sequence=len(samples)
            ack=command_lookup.get((name,s['servo_sequence']))
            for axis,value in s['applied_effort'].items():
                requests.append(dict(time_s=s['time_s'],command_id=runtime.current,entity=name,axis=axis,
                    effort=value,injected=bool(ack and axis in ack['disabled_axes']),
                    acknowledged_sequence=s['servo_sequence'],source='native_step_servo'))
            values=[float(sequence)];disabled_axes=[]
            for axis in d['axes']:
                value=t['xyz']['xyz'.index(axis)]-d['origin']['xyz'.index(axis)] if axis in 'xyz' else t[axis]
                injected=(inject and name=='ws001_bearing_right' and axis=='y') or (name,axis)==disabled
                values.extend([float(value),0. if injected else 1.])
                if injected:disabled_axes.append(axis)
            command=dict(entity=name,sequence=sequence,time_s=stamp,command_id=runtime.current,
                target=json.loads(json.dumps(t)),axis_values=values[1:],disabled_axes=disabled_axes)
            native_commands.append(command);command_lookup[name,sequence]=command
            message=Float64MultiArray();message.data=values;publishers[name].publish(message)
        last_tick=stamp
        if allow_world and stamp-last_world>=(.6 if finishing else .8 if assembly else 2):
            measured(list(targets)[world_index%len(targets)],reason='periodic_in_motion')
            world_index+=1;last_world=stamp
        if image['message'] is not None and image['stamp']-last_frame>=1 and sum(not f.done() for f in frame_futures)<2:
            path=output/'frames'/f'{len(frame_futures):05d}.png';path.parent.mkdir(exist_ok=True)
            frame_futures.append(writer.submit(capture,image['message'],path,image['stamp']));last_frame=image['stamp']
        if stamp-last_progress>=20:
            print(json.dumps(dict(progress=True,stage=runtime.station,phase=phase,sim_time_s=stamp,
                xyz=observation['ws001']['xyz'],bins=runtime.bins)),flush=True);last_progress=stamp
        return observation
    def measured(name,reason='waypoint_or_operation'):
        req=GetEntityState.Request();req.name=name+'::body';req.reference_frame='world';r=wait(getter.call_async(req))
        stamp=r.header.stamp.sec+r.header.stamp.nanosec*1e-9;deadline=time.monotonic()+1
        while latest[name]['time_s']<stamp and time.monotonic()<deadline:time.sleep(.002)
        s=min(list(history[name]),key=lambda row:abs(row['time_s']-stamp));p=r.state.pose.position
        xyz=[p.x,p.y,p.z];error=math.dist(xyz,s['xyz']);lag=abs(stamp-s['time_s'])
        pose_checks.append(dict(entity=name,command_id=runtime.current,time_s=stamp,joint_sample=s,reason=reason,
            world_xyz=xyz,error_m=error,source_time_error_s=lag))
        if error>.03 or lag>.05:raise RuntimeError('Front world/joint position mismatch')
        return s
    def sync_mounted():
        for name,item in runtime.items.items():
            if runtime.part_state[name] in {'mounted','refitted'} and name not in held_parts:
                targets[name]['xyz']=[a+b for a,b in zip(targets['ws001']['xyz'],item['offset'])]
        if assembly:
            for name,item in runtime.components.items():
                if runtime.component_state[name]=='fitted':
                    targets[name]['xyz']=[a+b for a,b in zip(targets['ws001']['xyz'],item['offset'])]
    def follow(name,points,token,*,label,timeout=45,inject=False):
        nonlocal phase
        phase=label;program=Waypoints(token,points,now(),timeout)
        while not program.complete:
            targets[name]['xyz']=list(program.target);sync_mounted();observation=tick(inject=inject)
            s=observation[name]
            if name=='ws001':
                attached=[p for p in runtime.items if runtime.part_state[p] in {'mounted','refitted'}]
                if assembly:attached += [p for p in runtime.components if runtime.component_state[p]=='fitted']
                if any(math.dist(observation[p]['xyz'],targets[p]['xyz'])>.035 or
                    max(abs(v) for v in observation[p]['velocity'])>=.035 for p in attached):continue
            index=program.index
            if program.update(s['xyz'],s['velocity'],s['time_s']):
                measured(name)
                s=dict(latest[name])
                if math.dist(s['xyz'],program.points[index])>=.025 or max(abs(v) for v in s['velocity'])>=.035:
                    raise RuntimeError('Waypoint moved while querying world position')
                waypoints.append(dict(command_id=token,entity=name,label=label,index=index,
                    target=list(program.points[index]),observation=s,time_s=now()))
    def rotate(token,stage):
        nonlocal phase
        phase='measured_rotation';a=DeviceAction(token,'ws001',stage['id'],stage['target'][0]-ORIGIN[0])
        while a.state not in {'complete','failed'}:
            s=tick()['ws001'];q=JointSample(s['time_s'],s['xyz'][0]-ORIGIN[0],s['velocity'][0],
                s['probe'],s['probe_velocity'],s['spin'],s['omega'])
            if math.dist(s['xyz'],stage['target'])>.03:raise RuntimeError('Front workpiece left device')
            a.update(q);_,targets['ws001']['probe'],targets['ws001']['spin']=a.targets()
        device_events.extend(a.events)
        if a.state=='failed':raise RuntimeError(a.fault)
    def scan(token,stage):
        nonlocal phase
        for extension in [.22,0.]:
            phase='scan_extend' if extension else 'scan_retract';targets['ws001']['probe']=extension;began=now()
            while True:
                s=tick()['ws001']
                if now()-began>8:raise TimeoutError('scanner_timeout')
                if abs(s['probe']-extension)<.005 and abs(s['probe_velocity'])<.015:
                    if extension and stage['id']=='identification':assert runtime.tag_read(token,'DEMO-ws001',now())
                    device_events.append(dict(command_id=token,event=phase,time_s=s['time_s'],sample=s));break
    def dismantle(token,kind):
        for name,item in runtime.items.items():
            if item['kind']!=kind or runtime.part_state[name]=='binned':continue
            held_parts.add(name)
            if runtime.part_state[name]=='mounted':
                main=recipe[runtime.stage]['target'];released=[a+b for a,b in zip(main,item['offset'])]
                released[1]+=item['sign']*item['separation_m']
                injected=args.case=='demount_recovery' and name=='ws001_bearing_right' and not recovered
                follow(name,[released],token,label='separate_'+name,timeout=12,inject=injected)
                assert runtime.separate(token,name,now(),dict(latest[name]))
            s=latest[name]['xyz'];dest=item['destination']
            follow(name,[(s[0],s[1],TRAVEL_Z),(dest[0],dest[1],TRAVEL_Z),dest],token,label='bin_'+name)
            assert runtime.deposit(token,name,now(),dict(latest[name]))
    def ram_world(name):
        req=GetEntityState.Request();req.name=name+'::probe';req.reference_frame='world';r=wait(getter.call_async(req))
        stamp=r.header.stamp.sec+r.header.stamp.nanosec*1e-9;deadline=time.monotonic()+1
        while latest[name]['time_s']<stamp and time.monotonic()<deadline:time.sleep(.002)
        s=min(list(history[name]),key=lambda row:abs(row['time_s']-stamp));d=meta['definitions'][name]
        expected=list(d['origin']);expected[1]+=d['direction_y']*s['probe'];p=r.state.pose.position
        xyz=[p.x,p.y,p.z];error=math.dist(expected,xyz);lag=abs(stamp-s['time_s'])
        pose_checks.append(dict(entity=name,command_id=runtime.current,time_s=stamp,joint_sample=s,
            reason='ram_head',world_xyz=xyz,expected_xyz=expected,error_m=error,source_time_error_s=lag))
        if error>.03 or lag>.05:raise RuntimeError('Ram head world/joint mismatch')
    def prepare_stock(token):
        nonlocal phase
        for name,item in runtime.components.items():
            if runtime.component_state[name]=='unreceived' and name!='ws001_new_ring_right':
                measured(name);assert runtime.receive(name,now(),dict(latest[name]))
        if runtime.stock_owner is None:
            assert not runtime.reserve_stock(token,now());phase='material_wait';began=now()
            while now()-began<3:tick()
            name='ws001_new_ring_right';dest=runtime.components[name]['receiving']
            follow(name,[dest],token,label='missing_ring_delivery')
            assert runtime.receive(name,now(),dict(latest[name]));assert runtime.reserve_stock(token,now())
        else:assert runtime.reserve_stock(token,now())
        for name,item in runtime.components.items():
            if runtime.component_state[name]!='reserved':continue
            r=item['receiving'];s=item['staged'];gauge=[]
            for point in [(*r[:2],r[2]+.12),r]:
                follow(name,[point],token,label='component_gauge_'+name);gauge.append(dict(latest[name]))
            follow(name,[(*r[:2],TRAVEL_Z),(*s[:2],TRAVEL_Z),s],token,label='component_stage_'+name)
            assert runtime.prepared(token,name,now(),dict(latest[name]),gauge)
    def press(token,stage):
        nonlocal phase
        prepare_stock(token)
        for side in ['left','right']:
            if side in runtime.fitted_sides:continue
            names=[n for n,i in runtime.components.items() if i['side']==side]
            for name in names:follow(name,[runtime.components[name]['staged']],token,label='press_restage_'+name)
            ram='press_ram_'+side
            bearing=next(n for n in names if runtime.components[n]['kind']=='bearing')
            ring=next(n for n in names if runtime.components[n]['kind']=='ring')
            action=PressAction(token,'ws001',bearing,ring)
            while action.state not in {'failed','complete'}:
                phase='press_'+side+'_'+action.state;targets[ram]['probe']=action.target()
                injected=args.case=='press_recovery' and side=='right' and action.state=='insert' and not recovered
                observation=tick(disabled=(ram,'probe') if injected else None);s=observation[ram];fixture=observation['ws001']
                q=PressSample(s['time_s'],token,s['probe'],s['probe_velocity'],s['applied_effort']['probe'],
                    math.dist(fixture['xyz'],stage['target']),max(abs(v) for v in fixture['velocity']))
                press_samples.append(dict(side=side,**q.__dict__,fixture=fixture,ram=s))
                count=len(action.events);action.update(q)
                if len(action.events)!=count:ram_world(ram)
                if action.state in {'insert','hold'}:
                    fraction=max(0.,min(1.,(s['probe']-.04)/.18))
                    for name in names:
                        item=runtime.components[name]
                        targets[name]['xyz']=[a+fraction*(b-a) for a,b in zip(item['staged'],item['fitted'])]
            device_events.extend(action.events);ram_world(ram)
            if action.state=='failed':raise TimeoutError(action.fault)
            for name in names:follow(name,[runtime.components[name]['fitted']],token,label='press_fit_'+name)
            observation={name:dict(latest[name]) for name in names}
            assert runtime.fitted(token,side,now(),observation,action.events)
    def stopped(observation):
        return all(max(abs(v) for v in s['velocity'])<.035 for s in observation.values()) and (
            abs(observation['ws001']['omega'])<.025 and abs(observation['ws001']['probe'])<.005
            and abs(observation['ws001']['probe_velocity'])<.015) and all(
                abs(s['probe'])<.005 and abs(s['probe_velocity'])<.015 for n,s in observation.items() if n.startswith('press_ram_'))
    def close_caps(token,stage):
        for name,item in runtime.items.items():
            if item['kind']!='cap' or runtime.part_state[name]=='refitted':continue
            if runtime.part_state[name]=='binned':
                measured(name);assert runtime.retrieve_cap(token,name,now(),dict(latest[name]))
            r=latest[name]['xyz'];f=[a+b for a,b in zip(stage['target'],item['offset'])]
            s=[*f];s[1]+=item['sign']*.35
            follow(name,[(*r[:2],6.),(*s[:2],6.),s,f],token,label='closing_'+name,timeout=160)
            assert runtime.refit_cap(token,name,now(),dict(latest[name]));held_parts.discard(name)
    def run_in(token,stage):
        nonlocal phase
        phase='bearing_run_in';action=RunInAction(token)
        while action.state not in {'failed','complete'}:
            targets['ws001']['spin']=action.target();s=tick()['ws001']
            action.update(RunInSample(s['time_s'],token,s['spin'],s['omega'],math.dist(s['xyz'],stage['target']),
                max(abs(v) for v in s['velocity']),s['probe'],s['probe_velocity']))
        device_events.extend(action.events)
        if action.state=='failed':raise RuntimeError(action.fault)
        assert runtime.finish_measured(token,now(),dict(kind='run_in',events=action.events))
    def paint(token,stage):
        points=[(PAINT[0]-.7,PAINT[1],PAINT[2]+.85),(PAINT[0]+.7,PAINT[1],PAINT[2]+.85),
                (PAINT[0]-.7,PAINT[1],PAINT[2]+.85)]
        evidence=[]
        for point in points:
            follow('spray_tool',[point],token,label='spray_traverse')
            fixture=dict(latest['ws001'])
            if math.dist(fixture['xyz'],stage['target'])>.025 or max(abs(v) for v in fixture['velocity'])>.035:raise RuntimeError('Spray fixture departed')
            evidence.append(dict(tool=dict(latest['spray_tool']),fixture=fixture,target=point))
        assert runtime.finish_measured(token,now(),dict(kind='paint',traverse=evidence,
            coating='simulated coating state, no fluid/coverage-quality measurement'))
    def storage_gate(token):
        nonlocal phase
        ledger=runtime.storage;assert not ledger.reserve(runtime.job,token,now())
        phase='storage_blocked';began=now()
        while now()-began<3:tick()
        release=ledger.begin_release(now());assert release
        p=STORAGE_APPROACH;u=STORAGE_UNLOAD
        follow(STORED_JOB,[p,(*p[:2],6.),(*u[:2],6.),u],token,label='storage_existing_unload')
        assert ledger.released(release,now(),dict(latest[STORED_JOB]));assert ledger.reserve(runtime.job,token,now())
    def stop_all():
        nonlocal phase
        phase='stopping'
        for name,s in latest.items():targets[name].update(xyz=list(s['xyz']),spin=0.,probe=0.)
        began=now();deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            observation=tick()
            if now()-began>.5 and stopped(observation):
                for name in targets:measured(name)
                return observation
        raise RuntimeError('Front stop not measured')
    try:
        for client in [getter]:
            if not client.wait_for_service(timeout_sec=45):raise RuntimeError('Front services unavailable')
        deadline=time.monotonic()+30
        while len(latest)<len(targets) and time.monotonic()<deadline:time.sleep(.01)
        if len(latest)!=len(targets):raise RuntimeError('Front initial feedback missing')
        tick()
        for index,stage in enumerate(recipe):
            token=runtime.issue(index,'transfer',now())
            follow('ws001',transfer_points(latest['ws001']['xyz'],stage),token,label='transfer_'+stage['id'])
            assert runtime.complete(token,now(),dict(latest['ws001']))
            token=runtime.issue(index,'device',now())
            while True:
                try:
                    if stage['action']=='scan':scan(token,stage)
                    elif stage['action']=='rotate':rotate(token,stage)
                    elif stage['action'] in {'caps','bearings'}:dismantle(token,'cap' if stage['action']=='caps' else 'bearing')
                    elif stage['action']=='press':press(token,stage)
                    elif stage['action']=='close':close_caps(token,stage)
                    elif stage['action']=='run_in':run_in(token,stage)
                    elif stage['action']=='paint':paint(token,stage)
                    elif stage['action']=='storage_gate':storage_gate(token)
                    elif stage['action']=='store':
                        assert runtime.storage.committed(runtime.job,runtime.storage.reserved['command_id'],now(),dict(latest['ws001']))
                    else:tick()
                    measured('ws001');assert runtime.complete(token,now(),dict(latest['ws001']));break
                except TimeoutError as exc:
                    recoverable=(args.case=='demount_recovery' and stage['id']=='demount') or (
                        args.case=='press_recovery' and stage['id']=='press')
                    if not recoverable or recovered:raise
                    assert runtime.fail(token,now(),str(exc));fault=dict(token=token,time_s=now(),reason=str(exc))
                    observation=stop_all();assert runtime.stopped(token,now(),observation)
                    old=token;token=runtime.issue(index,'device',now(),replacement=old);recovered=True
                    old_completion_rejected=not runtime.complete(old,now(),dict(latest['ws001']))
        final_stop=stop_all()
    except Exception as exc:failure=repr(exc)
    finally:
        # A run deadline must not prevent the bounded measured-stop attempt.
        shutting_down=True
        if latest and server.poll() is None:
            try:final_stop=stop_all()
            except Exception as exc:failure='; '.join(filter(None,[failure,repr(exc)]))
        executor.shutdown(timeout_sec=5);thread.join(timeout=5)
        for subscription in subscriptions:node.destroy_subscription(subscription)
        node.destroy_node();rclpy.shutdown()
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:server.wait(timeout=8)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        log.close();writer.shutdown(wait=True)
        gc.callbacks.remove(gc_timing)
        if gc_was_enabled:gc.enable()
        try:images=[f.result() for f in frame_futures]
        except Exception as exc:failure='; '.join(filter(None,[failure,repr(exc)]))
    data_hashes={}
    for name,rows in [('samples',samples),('efforts',requests),('events',runtime.events),('waypoints',waypoints),
        ('pose_checks',pose_checks),('device_events',device_events),('native_commands',native_commands),
        ('stock_events',getattr(runtime,'stock_events',[])),('press_samples',press_samples),('service_waits',service_waits),('gc_events',gc_events),
        ('storage_events',runtime.storage.events if finishing else [])]:
        path=output/(name+'.jsonl');path.write_text(''.join(json.dumps(row)+'\n' for row in rows));data_hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ['metadata.json','planned_clearance.json','front.world','bare_wheelset.dae']:
        data_hashes[name]=hashlib.sha256((output/name).read_bytes()).hexdigest()
    preserved=hashlib.sha256(baseline.read_bytes()).hexdigest()==before
    passed=failure is None and runtime.passed and preserved and final_stop is not None
    if args.case in {'demount_recovery','press_recovery'}:passed=passed and recovered and old_completion_rejected
    result=dict(passed=passed,runtime_error=failure,case=args.case,through=args.through,completed=runtime.completed,
        stock_snapshot=runtime.stock_snapshot() if assembly else None,
        storage_snapshot=runtime.storage.snapshot() if finishing else None,
        lease_fault_observation=lease_fault_observation,
        control_method='native_step_servo',native_build=native_build,
        final_snapshot=runtime.snapshot(),fault=fault,recovered=recovered,superseded_completion_rejected=old_completion_rejected,
        final_stop=final_stop,images=images,sample_count=len(samples),effort_count=len(requests),data_sha256=data_hashes,
        source_sha256=source_hashes,owned_server_pid=server.pid,owned_server_stopped=server.poll() is not None,
        baseline_session_preserved=preserved,build_fingerprint=build_fingerprint(),elapsed_wall_s=time.monotonic()-start,
        wall_timeout_s=wall_timeout,
        claim_boundary='One identified wheelset through front/inspection'+(' and preparation/two dedicated linear rams with finite replacement stock' if assembly else '')+(' then caps/run-in/spray-traverse/finite storage with measured prior content removal' if finishing else '')+'. Native force-driven constrained assumed rigs, gravity/contact excluded. Simulated identities/coating; no dimensional/press-fit/bearing quality. No multi-job/AGV integration for this prefix, carrier/tool/dynamic-pair sweep or industrial acceptance.')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['passed','runtime_error','completed','sample_count','effort_count']}),flush=True)
    if not passed:raise SystemExit(1)


if __name__=='__main__':main()
