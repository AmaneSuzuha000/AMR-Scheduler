"""Synthetic messages for decode gates, not a Gazebo or navigation claim."""
import copy
from types import SimpleNamespace as N

import pytest

from tools.wheelset_scene.fleet_truth import decode


def message():
    transform=N(translation=N(x=1.,y=2.,z=.01),rotation=N(x=0.,y=0.,z=0.,w=1.))
    twist=N(linear=N(x=.3,y=.4,z=0.),angular=N(x=0.,y=0.,z=-.2))
    return N(header=N(frame_id='wheelset_world',stamp=N(sec=7,nanosec=25000000)),
        joint_names=['scout_a','scout_b'],transforms=[transform,copy.deepcopy(transform)],twist=[twist,copy.deepcopy(twist)])


def test_decode_measured_body_twists_preserves_one_source_stamp_for_both_robots():
    result=decode(message())
    assert result['scout_a']==dict(pose=[1.,2.,0.],z=.01,linear_speed=.5,angular_speed=.2,sim_time_s=7.025)
    assert result['scout_a']==result['scout_b']


@pytest.mark.parametrize('corruption',['frame','duplicate_identity','missing_twist','nonfinite','bad_quaternion'])
def test_decode_rejects_invalid_source_identity_and_pose(corruption):
    m=message()
    if corruption=='frame': m.header.frame_id='other_scene'
    elif corruption=='duplicate_identity': m.joint_names[1]='scout_a'
    elif corruption=='missing_twist': m.twist.pop()
    elif corruption=='nonfinite': m.transforms[0].translation.x=float('nan')
    else: m.transforms[0].rotation.w=2.
    with pytest.raises(AssertionError): decode(m)
