"""Drive conveyor rollers only; observe a free wheelset through state queries."""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import time

from .build import ROOT
from .contact_conveyor_world import build, AXLE_Z, DISTANCE, ROLLER_RADIUS, RADIUS, MASS
from .evidence import build_fingerprint


def json_sequence(value):
    """Preserve ROS numeric sequence values in raw JSON records."""
    if isinstance(value, array):
        return value.tolist()
    raise TypeError(f'Unsupported record value: {type(value).__name__}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--axis-build', type=Path, required=True)
    parser.add_argument('--port', type=int, default=11369); parser.add_argument('--domain', type=int, default=86)
    parser.add_argument('--mechanism', choices=['rollers', 'chain'], default='rollers')
    args = parser.parse_args(); assert os.name != 'nt'
    with socket.socket() as port:
        assert port.connect_ex(('127.0.0.1', args.port)) != 0, 'selected port occupied'
    axis = args.axis_build.resolve(); library = axis/'libwheelset_axis_servo.so'
    native = json.loads((axis/'build_manifest.json').read_text())
    assert native['passed'] and hashlib.sha256(library.read_bytes()).hexdigest() == native['library_sha256']
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    builder = build
    if args.mechanism == 'chain':
        from .contact_chain_conveyor_world import build as builder
    frozen = build_fingerprint(); manifest = builder(output/'world', library)
    actuators = manifest.get('actuator_joints', manifest.get('roller_joints'))
    source = output/'executed_sources'; source.mkdir()
    for name in ('run_contact_conveyor.py','contact_conveyor_world.py','contact_chain_conveyor_world.py',
                 'actuated_world.py','assets.py','geometry.py','build.py'):
        (source/name).write_bytes(Path(__file__).with_name(name).read_bytes())
    (output/'native_build_manifest.json').write_text(json.dumps(native,indent=2)+'\n')
    native_copy = output/'native'; native_copy.mkdir()
    (native_copy/library.name).write_bytes(library.read_bytes())
    build_command = native['commands'][0]
    native_source = Path(build_command[build_command.index('-S')+1])
    for name, digest in native['source_sha256'].items():
        data = (native_source/name).read_bytes()
        assert hashlib.sha256(data).hexdigest() == digest
        target = native_copy/'source'/name; target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
    os.environ.update(ROS_DOMAIN_ID=str(args.domain), GAZEBO_MASTER_URI='http://127.0.0.1:'+str(args.port))
    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from gazebo_msgs.msg import ContactsState
    from gazebo_msgs.srv import GetEntityState
    from sensor_msgs.msg import JointState, Image
    from std_msgs.msg import Float64MultiArray
    from rosgraph_msgs.msg import Clock
    from rosidl_runtime_py.convert import message_to_ordereddict
    from .run_process_demo import save_camera
    from .run_navigation_readiness import stop_launch_group
    rclpy.init(); node = Node('wheelset_contact_conveyor_driver')
    started = time.monotonic(); clock = 0.; latest = {}; pending = None; query_id = 0
    phase = 'settling'; phase_start = None; stable = None; target_w = 0.; seq = 0
    next_command = next_query = 0.; process = None; cleanup = None; error = None; passed = False
    log = (output/'records.jsonl').open('x'); frames = output/'frames'; frames.mkdir()
    frame_count = 0; final_pose = None

    def stamp(msg): return msg.header.stamp.sec+msg.header.stamp.nanosec*1e-9
    def record(kind, **values):
        row = dict(kind=kind, wall_s=time.monotonic()-started, clock_s=clock, **values)
        log.write(json.dumps(row, default=json_sequence)+'\n'); log.flush(); return row
    def on_clock(msg):
        nonlocal clock
        clock = msg.clock.sec+msg.clock.nanosec*1e-9
    def on_joint(msg):
        latest['joint'] = msg; record('roller_joint_state', message=message_to_ordereddict(msg))
    def on_contact(kind, msg):
        latest[kind] = (clock, msg); record(kind, message=message_to_ordereddict(msg))
    def on_image(msg):
        nonlocal frame_count
        name = f'{frame_count:05d}.png'; save_camera(msg, frames/name)
        record('camera_frame', file='frames/'+name, source_s=stamp(msg)); frame_count += 1
    subscriptions = [node.create_subscription(Clock, '/clock', on_clock, qos_profile_sensor_data),
        node.create_subscription(JointState, '/contact_conveyor/drive/joint_states', on_joint, qos_profile_sensor_data),
        node.create_subscription(Image, '/contact_conveyor/camera/image_raw', on_image, qos_profile_sensor_data)]
    for kind in ('tread_left_contacts','tread_right_contacts','stop_contacts'):
        subscriptions.append(node.create_subscription(ContactsState, '/contact_conveyor/'+kind,
            lambda msg, kind=kind: on_contact(kind,msg), qos_profile_sensor_data))
    publisher = node.create_publisher(Float64MultiArray, '/contact_conveyor/drive/servo_targets', 1)
    getter = node.create_client(GetEntityState, '/contact_conveyor/get_entity_state')
    try:
        env = dict(os.environ, DISPLAY=':0', GAZEBO_MODEL_DATABASE_URI='')
        with (output/'gazebo.log').open('x') as stream:
            command = ['gzserver','--verbose',str(output/'world/conveyor.world'),
                       '-s','libgazebo_ros_init.so','-s','libgazebo_ros_factory.so']
            process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record('launch', pid=process.pid, command=command,
                   start_ticks=int(Path('/proc',str(process.pid),'stat').read_text().split(') ',1)[1].split()[19]))
            while time.monotonic()-started < 180.:
                assert process.poll() is None, 'owned server exited'
                rclpy.spin_once(node, timeout_sec=.002)
                if pending is not None and pending[0].done():
                    future, query = pending; reply = future.result()
                    assert reply is not None and reply.success
                    latest['pose'] = reply; latest['pose_wall'] = time.monotonic()
                    record('wheelset_state_reply', query=query, response=message_to_ordereddict(reply)); pending = None
                if pending is None and clock >= next_query and getter.service_is_ready():
                    request = GetEntityState.Request(); request.name = 'contact_wheelset'; request.reference_frame = 'world'
                    query = dict(id=query_id, service=getter.srv_name, request=message_to_ordereddict(request), issued_clock_s=clock)
                    record('wheelset_state_request', query=query); pending = (getter.call_async(request), query)
                    query_id += 1; next_query = clock+.05
                pose = latest.get('pose'); joint = latest.get('joint')
                if pose is not None and joint is not None and phase == 'settling' and clock >= 2.:
                    phase = 'conveying'; phase_start = clock; record('phase', value=phase)
                desired = 0.
                if phase == 'conveying':
                    assert time.monotonic()-latest['pose_wall'] < 1., 'independent wheelset feedback stale'
                    p = pose.state.pose.position; v = pose.state.twist.linear; w = pose.state.twist.angular
                    assert abs(p.y) < .05 and .765 < p.z < .825, 'wheelset left supported conveyor envelope'
                    assert p.x < DISTANCE+.10, 'wheelset overran station'
                    # Gain includes approximate free-wheel inertia; position comes
                    # exclusively from the independent measured state, never a time path.
                    desired = (manifest['maximum_target'] if args.mechanism == 'chain'
                               else max(-2.5,min(2.5,22.*(DISTANCE-p.x))))
                    speed = math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z)
                    angular = math.sqrt(w.x*w.x+w.y*w.y+w.z*w.z)
                    supported = all(k in latest and clock-latest[k][0] < .15 and latest[k][1].states
                                    for k in ('tread_left_contacts','tread_right_contacts'))
                    arrived = abs(p.x-DISTANCE) < .015 and speed < .01 and angular < .03 and supported
                    stable = (clock if stable is None else stable) if arrived else None
                    if stable is not None and clock-stable >= 1.:
                        phase = 'stopped'; phase_start = clock
                        desired = manifest['maximum_target'] if args.mechanism == 'chain' else 0.
                        record('phase', value=phase)
                    assert clock-phase_start < 100., 'measured conveyor arrival timeout'
                if phase == 'stopped':
                    desired = manifest['maximum_target'] if args.mechanism == 'chain' else 0.
                    if clock-phase_start >= 2.:
                        p = pose.state.pose.position; v = pose.state.twist.linear; w = pose.state.twist.angular
                        assert abs(p.x-DISTANCE) < .025 and math.sqrt(v.x*v.x+v.y*v.y+v.z*v.z) < .01
                        assert math.sqrt(w.x*w.x+w.y*w.y+w.z*w.z) < .03
                        final_pose = message_to_ordereddict(pose); passed = True; break
                if clock >= next_command and joint is not None:
                    step = .15*.05 if args.mechanism == 'chain' else .75*.05
                    target_w += max(-step,min(step,desired-target_w))
                    seq += 1; msg = Float64MultiArray(); msg.data = [float(seq)]+[value for _ in actuators for value in (target_w,1.)]
                    publisher.publish(msg); record('roller_command', sequence=seq, angular_target=target_w, values=msg.data)
                    next_command = clock+.05
            else:
                raise TimeoutError('contact conveyor wall timeout')
    except Exception as exc:
        error = repr(exc); record('runtime_error', error=error)
    finally:
        # Stop and continue observing before ending the exact owned process group.
        for _ in range(60):
            hold = target_w if args.mechanism == 'chain' else 0.
            seq += 1; msg = Float64MultiArray(); msg.data = [float(seq)]+[value for _ in actuators for value in (hold,1.)]
            publisher.publish(msg); record('shutdown_zero_command', sequence=seq)
            rclpy.spin_once(node, timeout_sec=.01)
        if process is not None: cleanup = stop_launch_group(process)
        node.destroy_node(); rclpy.shutdown(); log.close()
    assert build_fingerprint() == frozen
    result = dict(passed=passed and error is None, runtime_error=error, elapsed_wall_s=time.monotonic()-started,
        scope='isolated_contact_conveyor_runner_only', mechanism=args.mechanism, final_state=final_pose, camera_frames=frame_count,
        owned_process_group_stopped=cleanup is not None and cleanup['passed'], cleanup=cleanup,
        build_fingerprint=frozen, world_manifest=manifest,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in source.iterdir()},
        files_sha256={p.relative_to(output).as_posix():hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in output.rglob('*') if p.is_file()},
        claim_boundary='Isolated assumed conveyor. Runner success requires independent raw contact/motion audit; '
            'full video conveyor identity, workshop connections, workpiece quality and industrial acceptance are not established.')
    (output/'result.json').write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result))
    if not result['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
