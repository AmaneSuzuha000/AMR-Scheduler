"""Fine station measurements use physical scans, causal priors and one pose."""
import ast
from copy import deepcopy
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest

from tools.live_bridge.station_refinement import (
    FitSpec, ScanMatcher, StationRefiner, laser_pose, scan_snapshot,
)
from tools.live_bridge.terminal_docking import DockRegulator, DockSpec


def raster(boxes):
    image = np.full((1000, 1000), 254, dtype=np.uint8)
    for x0, y0, x1, y1 in boxes:
        c0, c1 = [round((x+10.)/.02) for x in (x0, x1)]
        r0, r1 = [round((y+10.)/.02) for y in (y0, y1)]
        image[1000-r1:1000-r0, c0:c1] = 0
    return image


def rays(boxes, pose):
    """Analytic slab intersections, independent of map distance interpolation."""
    angles = np.linspace(-3.14159, 3.14159, 1080)
    origin = (pose[0]+.15*math.cos(pose[2]), pose[1]+.15*math.sin(pose[2]))
    ranges = []
    for angle in angles+pose[2]:
        nearest = math.inf
        for x0, y0, x1, y1 in boxes:
            entry, leave = -math.inf, math.inf
            for value, direction, low, high in zip(
                    origin, (math.cos(angle), math.sin(angle)), (x0, y0), (x1, y1)):
                if abs(direction) < 1e-12:
                    if not low <= value <= high:
                        entry, leave = 1., 0.
                        break
                else:
                    a, b = sorted(((low-value)/direction, (high-value)/direction))
                    entry, leave = max(entry, a), min(leave, b)
            if entry <= leave and leave >= 0:
                nearest = min(nearest, entry if entry >= 0 else leave)
        ranges.append(nearest if .05 <= nearest < 8. else math.inf)
    return ranges


BOXES = [(3., 1., 3.8, 2.5), (-2.3, 3., -1.3, 4.6),
         (-8., -8., 8., -7.96), (-8., 7.96, 8., 8.),
         (-8., -8., -7.96, 8.), (7.96, -8., 8., 8.)]


def descriptor(tmp_path):
    image = raster(BOXES)
    path = tmp_path/'station.pgm'
    path.write_bytes(b'P5\n1000 1000\n255\n'+image.tobytes())
    return dict(version='station_scan_v1', map_image=str(path),
        map_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        resolution_m=.02, origin=[-10., -10., 0.], stations={'dock': [0., 0.]},
        fit=asdict(FitSpec()),
        laser=dict(frame_id='laser_link', beams=1080, angle_min=-3.14159,
            angle_max=3.14159, angle_increment=6.28318/1079,
            range_min=.05, range_max=8., xyzyaw=[.15, 0., .4, 0.]))


def scan():
    return dict(source_s=10., received_wall_s=100.,
        ranges=rays(BOXES, [.013, -.027, .013]), frame_id='laser_link',
        angle_min=-3.14159, angle_max=3.14159, angle_increment=6.28318/1079,
        range_min=.05, range_max=8.)


def admitted(tmp_path):
    refiner = StationRefiner(descriptor(tmp_path))
    refiner.note_prior([.08, -.08, .04], 9.95)
    return refiner


@pytest.mark.parametrize('pose', [[.313, -.227, .113], [-.283, .191, -.173], [.127, .239, .047]])
def test_independent_rectangle_rays_recover_pose_without_truth_or_target_argument(pose):
    matcher = ScanMatcher(raster(BOXES), .02, [-10., -10., 0.])
    fit = matcher.fit(rays(BOXES, pose), -3.14159, 6.28318/1079,
        [.15, 0., 0.], [pose[0]+.07, pose[1]-.055, pose[2]+.03], 8.)
    assert fit['valid'] and math.dist(fit['pose'][:2], pose[:2]) < .015
    assert abs(math.atan2(math.sin(fit['pose'][2]-pose[2]),
                          math.cos(fit['pose'][2]-pose[2]))) < .01


def test_one_wall_has_no_longitudinal_localization_authority():
    boxes = [(-10., 5., 10., 5.04)]
    matcher = ScanMatcher(raster(boxes), .02, [-10., -10., 0.])
    fit = matcher.fit(rays(boxes, [0., 0., 0.]), -3.14159, 6.28318/1079,
                      [.15, 0., 0.], [.06, .04, .01], 8.)
    assert not fit['valid'] and not fit['checks']['bounded_profile']


def test_bilinear_jacobian_and_outside_map_support():
    matcher = ScanMatcher(raster(BOXES), .02, [-10., -10., 0.])
    points = np.array([[2.993, 1.053], [3.213, .993], [-1.295, 3.373]])
    _, gradient, inside = matcher.sample(points)
    assert inside.all()
    for axis in (0, 1):
        offset = np.zeros(2); offset[axis] = 1e-6
        finite = (matcher.sample(points+offset)[0]-matcher.sample(points-offset)[0])/2e-6
        assert np.allclose(finite, gradient[:, axis], atol=1e-7)
    assert not matcher.sample(np.array([[100., 100.]]))[2].any()


def test_live_metadata_causal_seed_provenance_and_duplicate_receipt_are_bound(tmp_path):
    refiner = admitted(tmp_path)
    refiner.note_prior([9., 9., 0.], 10.05)  # Future relative to the measured scan.
    row = refiner.measure(scan(), [.15, 0., .4, 0.], 10.05, 100.05, 9.9)
    assert row['valid'] and row['prior_source_s'] == 9.95
    assert row['prior'] == [.08, -.08, .04] and row['received_wall_s'] == 100.
    assert refiner.fresh(10.1, 100.1)
    duplicate = refiner.measure(scan(), [.15, 0., .4, 0.], 10.1, 100.1, 10.)
    assert duplicate['duplicate'] and not duplicate['valid']
    assert refiner.latest is row and row['received_wall_s'] == 100.
    assert not refiner.fresh(10.16, 100.16)


@pytest.mark.parametrize('corruption', [
    'old_source', 'future_source', 'old_receipt', 'future_receipt',
    'wrong_frame', 'wrong_count', 'wrong_angle', 'wrong_range',
    'missing_tf', 'wrong_tf', 'old_seed', 'old_execution', 'invalid_returns',
])
def test_rejected_input_never_produces_station_pose_or_fit_credit(tmp_path, corruption):
    refiner = admitted(tmp_path)
    item = scan(); extrinsic = [.15, 0., .4, 0.]; minimum = 9.9
    if corruption == 'old_source': item['source_s'] = 9.
    if corruption == 'future_source': item['source_s'] = 10.1
    if corruption == 'old_receipt': item['received_wall_s'] = 99.
    if corruption == 'future_receipt': item['received_wall_s'] = 100.1
    if corruption == 'wrong_frame': item['frame_id'] = 'invented'
    if corruption == 'wrong_count': item['ranges'].pop()
    if corruption == 'wrong_angle': item['angle_increment'] *= 1.01
    if corruption == 'wrong_range': item['range_max'] = 20.
    if corruption == 'missing_tf': extrinsic = None
    if corruption == 'wrong_tf': extrinsic[0] += .01
    if corruption == 'old_seed': refiner.history.clear(); refiner.note_prior([0., 0., 0.], 9.)
    if corruption == 'old_execution': minimum = 10.
    if corruption == 'invalid_returns': item['ranges'] = [math.inf]*1080
    row = refiner.measure(item, extrinsic, 10.05, 100.05, minimum)
    assert not row['valid'] and not refiner.fresh(10.05, 100.05)
    assert 'pose' not in row


@pytest.mark.parametrize('corruption', ['hash', 'resolution', 'rotation', 'stations', 'fit'])
def test_unbound_map_or_invalid_configuration_is_refused_before_use(tmp_path, corruption):
    config = descriptor(tmp_path)
    if corruption == 'hash': config['map_sha256'] = '0'*64
    if corruption == 'resolution': config['resolution_m'] = .05
    if corruption == 'rotation': config['origin'][2] = .1
    if corruption == 'stations': config['stations'] = {}
    if corruption == 'fit': config['fit']['noise_floor_m'] = -1.
    with pytest.raises(ValueError): StationRefiner(config)


def test_station_eligibility_cannot_attract_or_snap_the_measured_fit(tmp_path):
    refiner = admitted(tmp_path)
    assert refiner.station([.34, 0., 0.]) == 'dock'
    assert refiner.station([.351, 0., 0.]) is None
    first = refiner.measure(scan(), [.15, 0., .4, 0.], 10.05, 100.05, 9.9)
    other = admitted(tmp_path)
    other.stations = {'irrelevant_target': [900., 900.]}
    second = other.measure(scan(), [.15, 0., .4, 0.], 10.05, 100.05, 9.9)
    assert first['valid'] and second['pose'] == first['pose']


def test_scan_snapshot_retains_original_receipt_and_static_laser_tilt_is_rejected():
    message = NS(header=NS(stamp=NS(sec=10, nanosec=0), frame_id='laser_link'),
        angle_min=-3.14159, angle_max=3.14159, angle_increment=6.28318/1079,
        range_min=.05, range_max=8., ranges=[1., 2.])
    stored = scan_snapshot(message, 100.)
    message.ranges[0] = -1.
    assert stored['ranges'] == (1., 2.) and stored['received_wall_s'] == 100.
    transform = NS(transform=NS(translation=NS(x=.15, y=0., z=.4),
        rotation=NS(x=0., y=0., z=0., w=1.)))
    assert laser_pose(transform) == [.15, 0., .4, 0.]
    transform.transform.rotation.x = .02
    with pytest.raises(ValueError): laser_pose(transform)


def test_repeated_source_time_cannot_accumulate_regulator_settle_credit():
    regulator = DockRegulator(DockSpec(.35, .005, .04), [0., 0.], 0.)
    for _ in range(30):
        row = regulator.update([0., 0., 0.], 0., 0., 10.)
        assert not row['done'] and row['settled_s'] == 0.
    assert regulator.update([0., 0., 0.], 0., 0., 10.31)['done']


def test_owner_docking_uses_the_accepted_common_pose_and_not_a_parallel_fit(tmp_path):
    # Evaluate the actual sensor adapter method without importing ROS packages.
    path = Path(__file__).resolve().parents[1]/'tools/live_bridge/sensor_scheduler.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == 'docking_measurement_time')
    scope = {}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    refiner = admitted(tmp_path)
    row = refiner.measure(scan(), [.15, 0., .4, 0.], 10.05, 100.05, 9.9)
    assert row['valid']
    owner = NS(station_refiner=refiner, clock_s=10.05, fine_pose_source=10.,
               sim=NS(execution=NS(pose={'timestamp': 10.})))
    measure_time = scope['docking_measurement_time']
    assert measure_time(owner, 100.05) == 10.
    owner.sim.execution.pose = {'timestamp': 10.01}  # Global feedback replaced it.
    assert measure_time(owner, 100.05) is None
    owner.sim.execution.pose = {'timestamp': 10.}
    assert measure_time(owner, 100.16) is None


@pytest.mark.parametrize('valid', [True, False])
def test_actual_sensor_query_selects_only_one_common_pose_and_preserves_health_gates(tmp_path, valid):
    path = Path(__file__).resolve().parents[1]/'tools/live_bridge/sensor_scheduler.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = dict(SchedulerNode=object, math=math, time=NS(monotonic=lambda: 100.05),
                 rclpy=NS(time=NS(Time=lambda: 0)))
    exec(compile(ast.Module(body=[klass], type_ignores=[]), str(path), 'exec'), scope)
    node = scope['SensorSchedulerNode'].__new__(scope['SensorSchedulerNode'])
    node.station_refiner = admitted(tmp_path)
    node.latest_scan = scan()
    node.laser_extrinsic = [.15, 0., .4, 0.] if valid else None
    node.sensor_ingress = None; node.clock_s = 10.05; node.localization_ready = True
    node.filtered_received = 100.
    node.filtered = NS(header=NS(stamp=NS(sec=10, nanosec=30000000)),
        twist=NS(twist=NS(linear=NS(x=0., y=0.), angular=NS(z=0.))))
    node.health = NS(samples={}, reason=lambda *a: None)
    node.last_health = None; node.last_feedback_unavailable = None
    node.request_map_update = lambda now: None
    node.tf = NS(lookup_transform=lambda *a: NS(header=NS(stamp=NS(sec=9, nanosec=950000000)),
        transform=NS(translation=NS(x=.08, y=-.08),
                     rotation=NS(w=math.cos(.02), z=math.sin(.02), x=0., y=0.))))
    node.sim = NS(execution=NS(last_timestamp=9.9, envelope=lambda: {'identity': 'unchanged'}))
    records = []; feedback = []
    node.record = lambda kind, **row: records.append((kind, row))
    node.process_observation = lambda *args: feedback.append(args)
    node.query_feedback(); node.query_feedback()
    if valid:
        assert len(feedback) == 1
        assert feedback[0][:3] == tuple(node.station_refiner.latest['pose'])
        assert feedback[0][6:] == (10., {'identity': 'unchanged'})
        assert node.fine_pose_source == 10. and node.fine_pose_received == 100.
    else:
        assert feedback == []  # Never fall back to the inaccurate global pose at a station.
    assert sum(kind == 'station_measurement' for kind, _ in records) == 1
    # Original sensor-health rejection still runs before fitting.
    node.health.reason = lambda *a: 'scan stale'
    node.sim.execution.bound = False
    node.note_feedback_unavailable = lambda *a, **k: None
    before = len(feedback); node.query_feedback()
    assert len(feedback) == before and node.last_health == 'scan stale'


def test_actual_dock_step_holds_zero_and_erases_settle_credit_when_fine_pose_expires():
    path = Path(__file__).resolve().parents[1]/'tools/live_bridge/nav2_scheduler.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    method = next(n for n in ast.walk(tree)
                  if isinstance(n, ast.FunctionDef) and n.name == 'dock_step')
    scope = dict(math=math, Twist=lambda: NS(linear=NS(x=0.), angular=NS(z=0.)))
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), 'exec'), scope)
    regulator = DockRegulator(DockSpec(.35, .005, .04), [0., 0.], 0.)
    regulator.update([0., 0., 0.], 0., 0., 10.)
    regulator.update([0., 0., 0.], 0., 0., 10.2)
    assert regulator.settled_s > 0.
    published = []
    node = NS(dock_spec=regulator.spec, dock_pending=None,
        sim=NS(execution=NS(fault=None, pose={'x': 0., 'y': 0., 'yaw': 0.})),
        dock={'route_id': 'same-action', 'started': 100., 'handover': False, 'regulator': regulator},
        persistence_failed=False, guard_authorized=True,
        last_motion={'speed': 0., 'wz': 0.}, docking_measurement_time=lambda now: None,
        dock_pub=NS(publish=published.append), record=lambda *a, **k: None)
    scope['dock_step'](node, 100.1)
    assert len(published) == 1 and published[0].linear.x == published[0].angular.z == 0.
    assert regulator.settled_s == 0. and regulator.last_now is None


def slow_numerical_peer(connection, descriptor):
    """Test a delayed numerical service with no ROS or control APIs."""
    import time
    connection.send(dict(kind='ready'))
    try:
        while True:
            request=connection.recv()
            if request is None:
                connection.send(dict(kind='closed'));return
            token,inputs=request
            time.sleep(.15)
            if descriptor.get('forged_token'):token+=1
            connection.send(dict(kind='fit',token=token,
                fit=dict(valid=True,pose=[.013,-.027,.013])))
    finally:connection.close()


def wait_worker_ready(worker):
    import time
    deadline=time.monotonic()+10.
    while not worker.ready:
        assert time.monotonic()<deadline
        worker.take()
        time.sleep(.002)


def test_numerical_process_returns_same_fit_without_blocking_or_renewing_receipts(tmp_path):
    import time
    from tools.live_bridge.station_fit_worker import StationFitWorker
    config=descriptor(tmp_path)
    worker=StationFitWorker(config)
    refiner=admitted(tmp_path)
    try:
        wait_worker_ready(worker)
        started=time.monotonic()
        first=refiner.measure(scan(),[.15,0.,.4,0.],10.05,100.05,9.9,fit_worker=worker)
        assert time.monotonic()-started<.05 and not first['valid'] and worker.busy
        deadline=time.monotonic()+5.
        while True:
            row=refiner.measure(scan(),[.15,0.,.4,0.],10.05,100.05,9.9,fit_worker=worker)
            if row['valid']:break
            assert time.monotonic()<deadline
            time.sleep(.002)
        assert row['source_s']==10. and row['received_wall_s']==100.
        expected=admitted(tmp_path).measure(scan(),[.15,0.,.4,0.],10.05,100.05,9.9)
        assert row['pose']==pytest.approx(expected['pose'],abs=1e-12)
        assert row['worker_token']==1
    finally:
        assert worker.close()['passed']
    assert not worker.process.is_alive() and not worker.transport.is_alive()


def test_slow_peer_never_blocks_owner_and_stale_completion_has_no_pose_credit(tmp_path):
    import time
    from tools.live_bridge.station_fit_worker import StationFitWorker
    worker=StationFitWorker(descriptor(tmp_path),_target=slow_numerical_peer)
    refiner=admitted(tmp_path)
    try:
        wait_worker_ready(worker)
        first=refiner.measure(scan(),[.15,0.,.4,0.],10.05,100.05,9.9,fit_worker=worker)
        assert not first['valid']
        for _ in range(10):
            started=time.monotonic()
            assert worker.take() is None
            assert time.monotonic()-started<.03
        time.sleep(.2)
        # A new wrapper does not change the original scan's expired receipt.
        row=refiner.measure(scan(),[.15,0.,.4,0.],10.3,100.3,9.9,fit_worker=worker)
        assert not row['valid'] and not refiner.fresh(10.3,100.3)
        assert refiner.pending_fit is None
    finally:assert worker.close()['passed']


def test_numerical_reply_token_is_owner_bound_and_foreign_result_fails(tmp_path):
    import time
    from tools.live_bridge.station_fit_worker import StationFitWorker
    config=descriptor(tmp_path);config['forged_token']=True
    worker=StationFitWorker(config,_target=slow_numerical_peer)
    try:
        wait_worker_ready(worker)
        assert worker.submit(([1.]*1080,0.,.01,[0.,0.,0.],[0.,0.,0.],8.))
        deadline=time.monotonic()+3.
        with pytest.raises(RuntimeError,match='unexpected numerical reply'):
            while True:
                worker.take()
                assert time.monotonic()<deadline
                time.sleep(.002)
    finally:
        summary=worker.close()
        assert not summary['passed'] and 'unexpected' in summary['error']
    assert not worker.process.is_alive()


def test_invalid_new_scan_cannot_use_cached_fit_while_peer_is_busy(tmp_path):
    from tools.live_bridge.station_fit_worker import StationFitWorker
    worker=StationFitWorker(descriptor(tmp_path),_target=slow_numerical_peer)
    refiner=admitted(tmp_path)
    try:
        wait_worker_ready(worker)
        refiner.measure(scan(),[.15,0.,.4,0.],10.05,100.05,9.9,fit_worker=worker)
        refiner.latest=dict(valid=True,source_s=9.98,received_wall_s=100.)
        corrupt=scan();corrupt['source_s']=10.01;corrupt['frame_id']='unbound'
        row=refiner.measure(corrupt,[.15,0.,.4,0.],10.05,100.05,9.9,fit_worker=worker)
        assert not row['valid'] and refiner.latest is None
    finally:assert worker.close()['passed']


def test_numerical_peer_loss_and_foreign_thread_fail_before_any_result_credit(tmp_path):
    import threading
    from tools.live_bridge.station_fit_worker import StationFitWorker
    worker=StationFitWorker(descriptor(tmp_path),_target=slow_numerical_peer)
    failures=[]
    try:
        wait_worker_ready(worker)
        def foreign():
            try:worker.take()
            except Exception as exc:failures.append(str(exc))
        thread=threading.Thread(target=foreign);thread.start();thread.join(timeout=1.)
        assert not thread.is_alive() and failures==['only the decision owner may consume station fits']
        worker.process.terminate();worker.process.join(timeout=2.)
        with pytest.raises(RuntimeError,match='worker failed'):worker.take()
    finally:
        assert not worker.close()['passed']
    assert not worker.transport.is_alive() and not worker.process.is_alive()


def test_future_scan_cannot_erase_a_still_fresh_accepted_measurement_or_renew_it(tmp_path):
    refiner=admitted(tmp_path)
    row=refiner.measure(scan(),[.15,0.,.4,0.],10.05,100.05,9.9)
    assert row['valid']
    later=scan();later['source_s']=10.06
    rejected=refiner.measure(later,[.15,0.,.4,0.],10.05,100.05,10.)
    assert not rejected['valid'] and rejected['duplicate']
    assert refiner.latest is row and row['source_s']==10. and row['received_wall_s']==100.
    assert refiner.fresh(10.05,100.05)
    assert not refiner.fresh(10.16,100.16)


def test_causal_completed_fit_survives_a_newer_not_yet_causal_scan(tmp_path):
    import time
    from tools.live_bridge.station_fit_worker import StationFitWorker
    worker=StationFitWorker(descriptor(tmp_path))
    refiner=admitted(tmp_path)
    try:
        wait_worker_ready(worker)
        refiner.measure(scan(),[.15,0.,.4,0.],10.05,100.05,9.9,fit_worker=worker)
        future=scan();future['source_s']=10.1
        deadline=time.monotonic()+5.
        while True:
            row=refiner.measure(future,[.15,0.,.4,0.],10.05,100.05,9.9,fit_worker=worker)
            if row['valid']:break
            assert time.monotonic()<deadline
            time.sleep(.002)
        assert row['source_s']==10. and row['received_wall_s']==100.
        assert row['quarantined_input_source_s']==10.1
        assert refiner.consumed_source==10.  # No credit from the future header.
    finally:assert worker.close()['passed']


def test_station_idle_parked_vehicle_emits_watchdog_fallback_when_fit_invalid(tmp_path):
    path = Path(__file__).resolve().parents[1]/'tools/live_bridge/sensor_scheduler.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = dict(SchedulerNode=object, math=math, time=NS(monotonic=lambda: 100.05),
                 rclpy=NS(time=NS(Time=lambda: 0)))
    exec(compile(ast.Module(body=[klass], type_ignores=[]), str(path), 'exec'), scope)
    node = scope['SensorSchedulerNode'].__new__(scope['SensorSchedulerNode'])
    node.station_refiner = admitted(tmp_path)
    node.latest_scan = scan()
    node.laser_extrinsic = None  # Force invalid fit
    node.sensor_ingress = None; node.clock_s = 10.05; node.localization_ready = True
    node.filtered_received = 100.
    node.filtered = NS(header=NS(stamp=NS(sec=10, nanosec=30000000)),
        twist=NS(twist=NS(linear=NS(x=0., y=0.), angular=NS(z=0.))))
    node.health = NS(samples={}, reason=lambda *a: None)
    node.last_health = None; node.last_feedback_unavailable = None
    node.request_map_update = lambda now: None
    node.tf = NS(lookup_transform=lambda *a: NS(header=NS(stamp=NS(sec=9, nanosec=950000000)),
        transform=NS(translation=NS(x=.08, y=-.08),
                     rotation=NS(w=math.cos(.02), z=math.sin(.02), x=0., y=0.))))
    node.dock = None; node.dock_candidate = None; node.active = None
    node.sim = NS(execution=NS(last_timestamp=9.9, state="idle", command=None,
                              last_receipt=100.0, envelope=lambda: {'identity': 'parked'}))
    records = []; feedback = []
    node.record = lambda kind, **row: records.append((kind, row))
    node.process_observation = lambda *args: feedback.append(args)
    node.query_feedback()
    # Idle parked vehicle MUST emit fallback observation to keep watchdog alive
    assert len(feedback) == 1
    assert feedback[0][0] == .08 and feedback[0][1] == -.08
    assert node.fine_pose_source is None  # Never claim inaccurate global pose as fine pose


def test_station_watchdog_keepalive_respects_active_dock_and_safeguards_watchdog(tmp_path):
    path = Path(__file__).resolve().parents[1]/'tools/live_bridge/sensor_scheduler.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef))
    scope = dict(SchedulerNode=object, math=math, time=NS(monotonic=lambda: 102.0),
                 rclpy=NS(time=NS(Time=lambda: 0)))
    exec(compile(ast.Module(body=[klass], type_ignores=[]), str(path), 'exec'), scope)
    node = scope['SensorSchedulerNode'].__new__(scope['SensorSchedulerNode'])
    node.station_refiner = admitted(tmp_path)
    node.latest_scan = scan()
    node.laser_extrinsic = None  # Force invalid fit
    node.sensor_ingress = None; node.clock_s = 12.0; node.localization_ready = True
    node.filtered_received = 101.9
    node.filtered = NS(header=NS(stamp=NS(sec=11, nanosec=950000000)),
        twist=NS(twist=NS(linear=NS(x=0., y=0.), angular=NS(z=0.))))
    node.health = NS(samples={}, reason=lambda *a: None)
    node.last_health = None; node.last_feedback_unavailable = None
    node.request_map_update = lambda now: None
    node.tf = NS(lookup_transform=lambda *a: NS(header=NS(stamp=NS(sec=11, nanosec=900000000)),
        transform=NS(translation=NS(x=.08, y=-.08),
                     rotation=NS(w=math.cos(.02), z=math.sin(.02), x=0., y=0.))))
    # Case 1: Actively docking -> must NOT fallback even if watchdog is at risk!
    node.dock = {'route_id': 'active-dock'}; node.active = {'route_id': 'active-dock'}
    node.sim = NS(execution=NS(last_timestamp=11.8, state="moving", command={'route_id': 'active-dock'},
                              last_receipt=100.5, envelope=lambda: {'identity': 'docking'}))
    feedback = []
    node.record = lambda *a, **k: None
    node.process_observation = lambda *args: feedback.append(args)
    node.query_feedback()
    assert feedback == []

    # Case 2: Not docking, but watchdog expiring (now=102.0, last_receipt=100.5, gap=1.5s >= 1.0s) -> fallback!
    node.dock = None
    node.query_feedback()
    assert len(feedback) == 1
    assert node.fine_pose_source is None

