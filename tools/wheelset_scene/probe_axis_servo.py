"""Actual Gazebo check of native command identity and local lease braking."""
import argparse
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
import xml.etree.ElementTree as ET

from .build import ROOT,e,save_xml
from .front_world import add_body


def verify(directory):
    directory=Path(directory);r=json.loads((directory/'result.json').read_text())
    assert r['runtime_error'] is None and r['owned_server_stopped']
    for name,digest in r['sha256'].items():assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest
    trace=[json.loads(line) for line in (directory/'trace.jsonl').read_text().splitlines()]
    commands=[json.loads(line) for line in (directory/'commands.jsonl').read_text().splitlines()]
    assert len(trace)>80 and all(math.isfinite(v) for row in trace for v in [*row['position'],*row['velocity'],*row['effort']])
    assert all(max(abs(v) for v in row['effort'])<=400 for row in trace)
    assert all(b['time_s']>a['time_s'] for a,b in zip(trace,trace[1:]))
    invalid=next(c for c in commands if c['case']=='out_of_bounds')
    assert not any(row['sequence']==invalid['sequence'] for row in trace)
    last=next(c for c in commands if c['case']=='last_valid')
    duplicate=next(c for c in commands if c['case']=='old_sequence')
    assert duplicate['sequence']<last['sequence']
    after=[row for row in trace if row['time_s']>=duplicate['time_s']+.1]
    assert all(row['sequence']==last['sequence'] for row in after)
    fault=next(row for row in trace if row['fault'])
    assert .95<fault['time_s']-last['time_s']<1.15
    moving=[row for row in trace if last['time_s']-.1<row['time_s']<last['time_s']+.1]
    assert moving and max(row['velocity'][0] for row in moving)>.2
    # A finite lease halts before the distant 5m goal; the latched controller
    # rejects a fresh sequence as well as an older one until explicit reset.
    final=trace[-1];assert final['fault'] and max(abs(v) for v in final['velocity'])<.02
    stopped=[row for row in trace if row['time_s']>=r['stopped_at_s']]
    assert final['time_s']-stopped[0]['time_s']>.4
    assert all(row['fault'] and max(abs(v) for v in row['velocity'])<.025 for row in stopped)
    assert max(math.dist(row['position'],stopped[0]['position']) for row in stopped)<.015
    assert max(row['position'][0] for row in trace)<3. and final['position'][0]<4.9
    result=dict(passed=True,raw_joint_samples=len(trace),out_of_bounds_rejected=True,
        old_sequence_rejected=True,lease_latched_after_s=fault['time_s']-last['time_s'],
        stopped_before_goal=True,new_target_after_latch_rejected=True,final_position=final['position'],
        boundary='Assumed no-contact 3-axis part rig; native command validation and simulation-time local brake, not a hardware emergency stop.')
    (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


def run(output,port,domain):
    if os.name=='nt':raise SystemExit('Run on Ubuntu after sourcing ROS2 Humble.')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:raise SystemExit('Probe port occupied; preserved.')
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    tree=ET.Element('sdf',version='1.7');world=e(tree,'world',name='native_servo_probe')
    physics=e(world,'physics',name='ode',type='ode');e(physics,'max_step_size',.002);e(physics,'real_time_update_rate',500)
    add_body(world,'probe',(0.,0.,1.),'cap');save_xml(tree,output/'probe.world')
    build=ROOT/'out/wheelset_axis_servo_build'
    for name in ['libwheelset_axis_servo.so','build_manifest.json']:(output/name).write_bytes((build/name).read_bytes())
    os.environ['ROS_DOMAIN_ID']=str(domain)
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64MultiArray
    env=dict(os.environ,GAZEBO_MASTER_URI=f'http://127.0.0.1:{port}',GAZEBO_MODEL_DATABASE_URI='',
        GAZEBO_PLUGIN_PATH=str(build)+':'+os.environ.get('GAZEBO_PLUGIN_PATH',''))
    log=(output/'gzserver.log').open('w');server=subprocess.Popen(['gzserver','--verbose','-s','libgazebo_ros_init.so',
        str(output/'probe.world')],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    rclpy.init();node=Node('native_servo_probe');executor=SingleThreadedExecutor();executor.add_node(node)
    trace=[];commands=[];failure=None;stopped_at=None
    def callback(msg):
        values=dict(zip(msg.name,zip(msg.position,msg.velocity,msg.effort)))
        trace.append(dict(time_s=msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9,
            position=[values['probe_'+a][0] for a in 'xyz'],velocity=[values['probe_'+a][1] for a in 'xyz'],
            effort=[values['probe_'+a][2] for a in 'xyz'],sequence=int(values['probe_servo_sequence'][0]),
            fault=bool(values['probe_servo_fault'][0])))
    sub=node.create_subscription(JointState,'/probe/joint_states',callback,qos_profile_sensor_data)
    pub=node.create_publisher(Float64MultiArray,'/probe/servo_targets',1)
    thread=threading.Thread(target=executor.spin,daemon=True);thread.start();deadline=time.monotonic()+50
    def latest():
        if server.poll() is not None or time.monotonic()>deadline:raise RuntimeError('Native servo probe timed out/exited')
        if not trace:time.sleep(.005);return None
        return trace[-1]
    def send(seq,x,case):
        s=latest();msg=Float64MultiArray();msg.data=[float(seq),float(x),1.,0.,1.,0.,1.]
        commands.append(dict(sequence=seq,target_x=x,time_s=s['time_s'],case=case));pub.publish(msg)
    try:
        while not trace or pub.get_subscription_count()==0:latest();time.sleep(.005)
        send(10000,300,'out_of_bounds');begin=latest()['time_s']
        while latest()['time_s']-begin<.2:time.sleep(.002)
        seq=0;last=-math.inf
        while True:
            s=latest()
            if s['position'][0]>.35 and s['velocity'][0]>.2:break
            if s['time_s']-last>.04:seq+=1;send(seq,5.,'valid');last=s['time_s']
            time.sleep(.002)
        seq+=1;send(seq,5.,'last_valid');time.sleep(.03);send(seq-1,100.,'old_sequence')
        quiet_since=None
        while True:
            s=latest()
            if s['fault'] and max(abs(v) for v in s['velocity'])<.015:
                if quiet_since is None:quiet_since=s['time_s']
                if s['time_s']-quiet_since>.4:stopped_at=s['time_s'];break
            else:quiet_since=None
            time.sleep(.002)
        send(seq+100,10.,'after_latch')
        while latest()['time_s']-stopped_at<.6:time.sleep(.002)
    except Exception as exc:failure=repr(exc)
    finally:
        executor.shutdown(timeout_sec=5);thread.join(timeout=5);node.destroy_subscription(sub);node.destroy_node();rclpy.shutdown()
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:server.wait(timeout=8)
            except subprocess.TimeoutExpired:server.kill();server.wait(timeout=5)
        log.close()
    for name,rows in [('trace',trace),('commands',commands)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
    source=Path(__file__);(output/'executed_probe.py').write_bytes(source.read_bytes())
    result=dict(runtime_error=failure,stopped_at_s=stopped_at,owned_server_pid=server.pid,owned_server_stopped=server.poll() is not None,
        sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.iterdir() if p.is_file()})
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    if failure:raise RuntimeError(failure)
    return verify(output)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--verify',action='store_true');p.add_argument('--port',type=int,default=11369);p.add_argument('--domain',type=int,default=83)
    args=p.parse_args();print(json.dumps(verify(args.directory) if args.verify else run(args.directory,args.port,args.domain)))


if __name__=='__main__':main()
