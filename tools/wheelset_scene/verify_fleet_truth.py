"""Corroborate the streamed dual-model truth with independent service replies."""
from bisect import bisect_left
import json
import math
import xml.etree.ElementTree as ET

from .whole_line_evidence import rows


def verify(directory,trace,report):
    world=ET.parse(directory/'light_material.world').getroot().find('world')
    assert len(world.findall("plugin[@filename='libwheelset_fleet_truth.so']"))==1
    manifest=json.loads((directory/'executed_sources/build_manifest.json').read_text())
    assert manifest['auxiliary_libraries']['libwheelset_fleet_truth.so']==report['source_sha256']['libwheelset_fleet_truth.so']
    raw=list(rows(directory,'native_truth')); checks=list(rows(directory,'fleet_pose_checks'))
    assert len(raw)>100 and len(checks)>30
    times=[]
    for seq,e in enumerate(raw,1):
        assert e['sequence']==seq and set(e['observations'])=={'scout_a','scout_b'}
        if seq>1:
            assert e['source_time_s']>raw[seq-2]['source_time_s'] and e['received_at']>=raw[seq-2]['received_at']
        assert all(o['sim_time_s']==e['source_time_s'] for o in e['observations'].values())
        times.append(e['source_time_s'])
    for e in trace:
        source=raw[e['native_truth_sequence']-1]
        assert e['observations']==source['observations'] and 0<=e['wall_s']-source['received_at']<=2.
    assert {e['robot'] for e in checks}=={'scout_a','scout_b'}
    for e in checks:
        source=raw[e['native_sequence']-1];native=source['observations'][e['robot']]; observed=e['service_observation']
        assert native==e['native_observation']
        assert e['source_time_error_s']==abs(native['sim_time_s']-observed['sim_time_s'])<=.05
        assert e['error_m']==math.dist(native['pose'][:2],observed['pose'][:2])<=.03
        yaw=abs(math.atan2(math.sin(native['pose'][2]-observed['pose'][2]),math.cos(native['pose'][2]-observed['pose'][2])))
        assert yaw==e['yaw_error_rad']<=.04 and abs(native['z']-observed['z'])<=.03
    for robot,stop in report['final_stop'].items():
        index=bisect_left(times,stop['sim_time_s'])
        source=min(raw[max(0,index-1):index+2],key=lambda e:abs(e['source_time_s']-stop['sim_time_s']))
        native=source['observations'][robot]
        assert abs(native['sim_time_s']-stop['sim_time_s'])<=.05
        assert math.dist(native['pose'][:2],stop['pose'][:2])<=.02 and native['linear_speed']<.025 and native['angular_speed']<.05
    return dict(native_world_truth_samples=len(raw),independent_fleet_service_checks=len(checks))
