from copy import deepcopy

import pytest
import yaml

from tools.wheelset_scene.navigation_parameters import (audit_parameter_records, expected_parameters,
                                                         parameter_mismatches)


def records():
    values = dict(laser_model_type='likelihood_field_prob', max_beams=1080, do_beamskip=False)
    expected = {'scout_a': {'amcl': values}, 'scout_b': {'amcl': values}}
    rows = [dict(kind='runtime_parameters', robot=robot, node='amcl', wall_s=1.,
                 received_wall_s=.9, expected=deepcopy(values), actual=deepcopy(values), mismatches={})
            for robot in expected]
    return rows, expected


def test_live_parameters_match_before_readiness():
    rows, expected = records()
    assert audit_parameter_records(rows, expected, 2.) == 2


@pytest.mark.parametrize('failure', ['launch_override', 'missing_node', 'after_readiness', 'missing_value', 'wrong_type'])
def test_false_configured_success_does_not_replace_live_parameters(failure):
    rows, expected = records()
    if failure == 'launch_override':
        rows[0]['actual']['laser_model_type'] = 'likelihood_field'
    elif failure == 'missing_node':
        rows.pop()
    elif failure == 'after_readiness':
        rows[0]['wall_s'] = rows[0]['received_wall_s'] = 3.
    elif failure == 'missing_value':
        del rows[0]['actual']['max_beams']
    elif failure == 'wrong_type':
        rows[0]['actual']['do_beamskip'] = 0
    with pytest.raises(AssertionError):
        audit_parameter_records(rows, expected, 2.)


def test_parameter_values_may_not_be_reinterpreted_as_another_type():
    assert parameter_mismatches({'max_beams': 1080}, {'max_beams': '1080'})


def test_expected_runtime_parameters_include_both_transform_tolerances(tmp_path):
    robot = 'scout_a'
    localization = {robot: {
        'amcl': {'ros__parameters': dict(laser_model_type='likelihood_field_prob', max_beams=540,
            do_beamskip=False, robot_model_type='nav2_amcl::DifferentialMotionModel',
            laser_max_range=8.0, sigma_hit=.1, update_min_d=.005, update_min_a=.005,
            min_particles=5000, max_particles=10000, transform_tolerance=.3)},
        'ekf_filter_node': {'ros__parameters': dict(world_frame='odom', odom0='/wheel/odometry',
            imu0='/imu/data', odom0_config=[False,False,False,False,False,False,True,False,False,
                False,False,False,False,False,False], imu0_config=[False,False,False,False,False,
                False,False,False,False,False,False,True,False,False,False])},
    }}
    controller = {robot: {'controller_server': {'ros__parameters': {
        'FollowPath': dict(plugin='amr_execution::BoundedPursuit', rotate_to_heading_min_angle=.02,
                           max_command_linear=.2, max_command_angular=.8, transform_tolerance=.2),
        'general_goal_checker': dict(stateful=True, xy_goal_tolerance=.05),
    }}}}
    (tmp_path/'localization.yaml').write_text(yaml.safe_dump(localization))
    (tmp_path/'controller.yaml').write_text(yaml.safe_dump(controller))

    actual = expected_parameters(tmp_path, robot,
        dict(localization='localization.yaml', controller='controller.yaml'))
    assert actual['amcl']['transform_tolerance'] == .3
    assert actual['controller_server']['FollowPath.transform_tolerance'] == .2
