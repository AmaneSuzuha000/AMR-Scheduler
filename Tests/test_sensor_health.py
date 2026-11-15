"""Lost sensors and repeated samples must not masquerade as fresh localization."""
import math
import ast
from pathlib import Path
import sys
from concurrent.futures import Future
from types import SimpleNamespace

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/live_bridge'))
from sensor_health import SensorHealth,pose_quality


def healthy():
    health=SensorHealth()
    for source in health.limits:
        health.observe(source,10.,100.)
    return health


def test_no_pose_authority_until_all_four_sensor_sources_exist():
    health=SensorHealth()
    for source in ('scan','imu','odom'):
        health.observe(source,10.,100.)
    assert health.reason(10.,100.)=='amcl unavailable'
    health.observe('amcl',10.,100.)
    assert health.reason(10.,100.) is None


@pytest.mark.parametrize('source',['scan','imu','odom','amcl'])
def test_each_indispensable_input_has_independent_source_and_receipt_watchdog(source):
    health=healthy()
    limit=health.limits[source]
    for key in health.limits:
        if key!=source:
            health.observe(key,10.+limit+.01,100.+limit+.01)
    assert source in health.reason(10.+limit+.01,100.+limit+.01)


def test_replaying_a_scan_does_not_extend_its_receipt_deadline():
    health=healthy()
    health.observe('scan',10.,100.29)
    for source in ('imu','odom','amcl'):
        health.observe(source,10.31,100.31)
    assert health.reason(10.31,100.31)=='scan stale'


def test_invalid_new_scan_latches_quality_failure_until_a_new_valid_source_sample():
    health=healthy()
    health.observe('scan',10.01,100.01,False,'no returns')
    assert health.reason(10.01,100.01)=='scan: no returns'
    health.observe('scan',10.01,100.02,True)
    assert health.reason(10.02,100.02)=='scan: no returns'
    health.observe('scan',10.02,100.02,True)
    assert health.reason(10.02,100.02) is None


@pytest.mark.parametrize('index,value',[(0,-.01),(0,.063),(7,.063),(35,.041),(35,math.nan)])
def test_map_pose_quality_rejects_uncertainty_and_invalid_covariance(index,value):
    covariance=[0.]*36
    covariance[0]=covariance[7]=.01
    covariance[35]=.005
    assert pose_quality(covariance)
    covariance[index]=value
    assert not pose_quality(covariance)


def test_actual_large_map_roundoff_is_tolerated_without_mutating_the_reported_covariance():
    # Recorded first collapsed posterior at (25.007,67.964); keep this exact
    # failed-run value portable rather than depending on file ordering.
    covariance=[0.]*36
    covariance[0],covariance[7],covariance[35]=(4.0153054214897566e-9,-2.7557689463719726e-10,3.9575904814702484e-9)
    original=list(covariance)
    assert not pose_quality(covariance)
    assert pose_quality(covariance,(25.007,67.964))
    assert covariance==original
    covariance[7]=-1.01e-9
    assert not pose_quality(covariance,(25.,68.))
    assert not pose_quality(covariance,(1e8,1e8))
    covariance[7]=.25**2+1e-10
    assert not pose_quality(covariance,(25.,68.))


def test_roundoff_allowance_stays_bounded_for_extreme_finite_coordinates():
    covariance=[0.]*36
    covariance[7]=-1.01e-9
    assert not pose_quality(covariance,(1e308,-1e308))
    covariance[7]=0.
    assert pose_quality(covariance,(1e308,-1e308))
    assert not pose_quality(covariance,(math.inf,68.))
    assert not pose_quality(covariance,(25.,math.nan))


def test_lost_nomotion_service_response_is_bounded_and_does_not_block_later_map_updates():
    path=Path(__file__).resolve().parents[1]/'tools/live_bridge/sensor_scheduler.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    klass=next(n for n in tree.body if isinstance(n,ast.ClassDef))
    scope={'SchedulerNode':object,'Empty':SimpleNamespace(Request=lambda:object())}
    exec(compile(ast.Module(body=[klass],type_ignores=[]),str(path),'exec'),scope)
    node=scope['SensorSchedulerNode'].__new__(scope['SensorSchedulerNode'])
    old=Future()
    replacement=Future()
    removed=[]
    requests=[]
    node.nomotion=SimpleNamespace(remove_pending_request=removed.append,service_is_ready=lambda:True,
                                  call_async=lambda request:requests.append(request) or replacement)
    node.nomotion_pending=old
    node.next_nomotion=1.
    events=[]
    node.record=lambda *args,**kwargs:events.append(args)
    node.request_map_update(.9)
    assert not requests and not old.cancelled()
    node.request_map_update(1.)
    assert old.cancelled() and removed==[old] and len(requests)==1
    assert node.nomotion_pending is replacement and node.next_nomotion==2.
    assert events==[('localization_update_timeout',)]
    node.request_map_update(1.5)
    assert len(requests)==1
