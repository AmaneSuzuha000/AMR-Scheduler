"""Namespace the existing sensor/controller parameters without changing tuning."""
from copy import deepcopy
from dataclasses import replace
import re

from amr_sched.network import Network


def point_execution_network(network):
    """Copy an abstract network for adapters that expose one physical pose per node.

    Multiple capacity slots need distinct target geometry. Until that mapping
    exists, native fleet execution explicitly uses one slot per node and edge;
    the algorithm network and its capacity experiments remain independent.
    """
    result = Network()
    for node in network.nodes.values():
        result.add_node(replace(node, capacity=1))
    for edge in network.edges.values():
        result.add_edge(replace(edge, capacity=1))
    return result


def robot_configs(namespace, localization, controller, *,
                  dock_tolerance_m=None, dock_yaw_tolerance_rad=None):
    """Isolate absolute sensor topics and parameter roots for one physical robot.

    A station terminal has to satisfy the handoff tolerance, which is tighter
    than the transit goal tolerance: Nav2 may legally stop anywhere inside the
    transit window, and a station that is handed material cannot. When the
    caller declares a docking tolerance, a second goal checker is registered so
    the owner can select it for those hops only. The declared value must be
    strictly tighter than the transit checker, because it is not allowed to
    widen an acceptance gate.
    """
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*',namespace):
        raise ValueError('one simple ROS robot namespace required')
    localization=deepcopy(localization)
    controller=deepcopy(controller)
    prefix='/'+namespace
    ekf=localization['ekf_filter_node']['ros__parameters']
    ekf['odom0']=prefix+'/wheel/odometry'
    ekf['imu0']=prefix+'/imu/data'
    localization['amcl']['ros__parameters']['scan_topic']=prefix+'/scan'
    controller['controller_server']['ros__parameters']['odom_topic']=prefix+'/odometry/filtered'
    costmap=controller['local_costmap']['local_costmap']['ros__parameters']
    costmap['obstacle_layer']['scan']['topic']=prefix+'/scan'
    if dock_tolerance_m is not None or dock_yaw_tolerance_rad is not None:
        parameters=controller['controller_server']['ros__parameters']
        general=parameters['general_goal_checker']
        xy=general['xy_goal_tolerance'] if dock_tolerance_m is None else float(dock_tolerance_m)
        yaw=general['yaw_goal_tolerance'] if dock_yaw_tolerance_rad is None else float(dock_yaw_tolerance_rad)
        if not (0 < xy < general['xy_goal_tolerance']) or not (0 < yaw <= general['yaw_goal_tolerance']):
            raise ValueError('dock goal tolerance must be strictly tighter than the transit checker')
        parameters['goal_checker_plugins']=list(parameters['goal_checker_plugins'])+['dock_goal_checker']
        parameters['dock_goal_checker']=dict(plugin='nav2_controller::SimpleGoalChecker',
            stateful=True, xy_goal_tolerance=xy, yaw_goal_tolerance=yaw)
        # Regulated pure pursuit only rotates while the heading error exceeds
        # its rotate-to-heading band. A band wider than the docking tolerance
        # makes that tolerance unreachable: the checker keeps refusing a goal
        # the controller has already stopped correcting, the robot circles and
        # the progress checker aborts the action. Keep the band strictly
        # inside the docking tolerance so a normally reachable station hop
        # stays reachable.
        follow=parameters['FollowPath']
        band=yaw*.5
        assert 0 < band < yaw
        follow['rotate_to_heading_min_angle']=band
    return {namespace:localization},{namespace:controller}
