"""Compare configured navigation parameters with each live ROS node."""
from pathlib import Path

import yaml


def expected_parameters(bundle, robot, config):
    bundle = Path(bundle)
    loc = yaml.safe_load((bundle/config['localization']).read_text())[robot]
    nav = yaml.safe_load((bundle/config['controller']).read_text())[robot]
    keys = {
        'amcl': ('laser_model_type', 'max_beams', 'do_beamskip', 'robot_model_type',
                 'laser_max_range', 'sigma_hit', 'update_min_d', 'update_min_a',
                 'min_particles', 'max_particles', 'transform_tolerance'),
        'ekf_filter_node': ('world_frame', 'odom0', 'imu0', 'odom0_config', 'imu0_config'),
        'controller_server': ('FollowPath.plugin', 'FollowPath.rotate_to_heading_min_angle',
                              'FollowPath.max_command_linear', 'FollowPath.max_command_angular',
                              'general_goal_checker.stateful', 'general_goal_checker.xy_goal_tolerance',
                              'FollowPath.transform_tolerance'),
    }
    result = {}
    for node, names in keys.items():
        values = (loc if node in loc else nav)[node]['ros__parameters']
        result[node] = {}
        for name in names:
            value = values
            for part in name.split('.'):
                value = value[part]
            result[node][name] = value
    if config.get('contact_bin_footprint'):
        result['local_costmap/local_costmap'] = dict(
            footprint=nav['local_costmap']['local_costmap']['ros__parameters']['footprint'])
    return result


def parameter_mismatches(expected, actual):
    """Missing values and type changes are errors, including bool versus int."""
    return {name: dict(expected=value, actual=actual.get(name))
            for name, value in expected.items()
            if name not in actual or type(value) is not type(actual[name]) or value != actual[name]}


def audit_parameter_records(rows, expected, ready_start_wall):
    seen = set()
    for row in rows:
        if row['kind'] != 'runtime_parameters':
            continue
        robot, node = row['robot'], row['node']
        assert (robot, node) not in seen
        values = expected[robot][node]
        assert 0 <= row['received_wall_s'] <= row['wall_s'] <= ready_start_wall
        assert row['expected'] == values and row['mismatches'] == {}
        assert not parameter_mismatches(values, row['actual']), (robot, node, row['actual'])
        seen.add((robot, node))
    assert seen == {(robot, node) for robot, nodes in expected.items() for node in nodes}
    return len(seen)
