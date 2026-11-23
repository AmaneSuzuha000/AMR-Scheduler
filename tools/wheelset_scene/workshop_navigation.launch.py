"""Two isolated SCOUT sensor/localization/Nav2 stacks; startup motion stays locked."""
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
import xml.etree.ElementTree as ET
import yaml
import xacro

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def start(context):
    bundle = Path(LaunchConfiguration('bundle').perform(context)).resolve()
    model_output = Path(LaunchConfiguration('model_output').perform(context)).resolve()
    model_output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((bundle/'manifest.json').read_text())
    for name, digest in manifest['files_sha256'].items():
        if hashlib.sha256((bundle/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('navigation bundle changed: '+name)
    for name, digest in manifest['source_sha256'].items():
        source = Path(name)
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != digest:
            raise RuntimeError('navigation source changed; rebuild the bundle: '+name)
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.wheelset_scene.navigation_model import build_model
    guard_share = Path(get_package_share_directory('amr_gazebo_guard'))
    scout_share = Path(get_package_share_directory('scout_gazebo_description'))
    gazebo_share = Path(get_package_share_directory('gazebo_ros'))
    launch_dir = guard_share/'launch'
    spec = importlib.util.spec_from_file_location('workshop_metric_model', launch_dir/'metric_model.py')
    geometry = importlib.util.module_from_spec(spec); spec.loader.exec_module(geometry)
    def include(path, **arguments):
        return IncludeLaunchDescription(PythonLaunchDescriptionSource(str(path)),
            launch_arguments={k: str(v) for k, v in arguments.items()}.items())
    actions = [
        SetEnvironmentVariable('GAZEBO_PLUGIN_PATH', str(bundle/'native_axis')+os.pathsep
                               +str(bundle/'bin_axis')+os.pathsep
                               +str(Path(get_package_prefix('amr_gazebo_guard'))/'lib')
                               +os.pathsep+os.environ.get('GAZEBO_PLUGIN_PATH', '')),
        SetEnvironmentVariable('GAZEBO_MODEL_PATH', os.pathsep.join([
            str(root/'assets/wheelset_workshop/models'), str(scout_share.parent),
            os.environ.get('GAZEBO_MODEL_PATH', '')])),
        SetEnvironmentVariable('GAZEBO_MODEL_DATABASE_URI', ''),
        include(gazebo_share/'launch/gzserver.launch.py', world=bundle/'navigation.world',
                verbose='false', params_file=launch_dir/'metric_clock.yaml'),
    ]
    layout = json.loads((bundle/'network.json').read_text())
    nodes = {n['id']: n for n in layout['nodes']}
    for robot, config in manifest['robot_configs'].items():
        position = nodes[config['start_node']]
        spawn = next(item for item in layout['robots'] if item['id'] == robot)
        localization = yaml.safe_load((bundle/config['localization']).read_text())[robot]['amcl']['ros__parameters']
        base = ET.fromstring(xacro.process_file(str(scout_share/'urdf/scout_mini.urdf.xacro'),
                                                 mappings={'drive_mode': 'diff'}).toxml())
        description, sdf_path = build_model(base, geometry, robot, model_output, manifest)
        if manifest.get('contact_bins'):
            from tools.wheelset_scene.scout_bin_hardware import audit_contact_clearance
            clearance=audit_contact_clearance(ET.parse(sdf_path).getroot(),
                ET.parse(bundle/'navigation.world').getroot().find('world'),manifest['contact_bins'])
            (model_output/(robot+'_handoff_clearance.json')).write_text(json.dumps(clearance,indent=2)+'\n')
        if manifest.get('encoder_source') == 'joint_positions':
            actions.append(ExecuteProcess(cmd=[
                sys.executable, str(root/'tools/live_bridge/position_encoder.py'),
                '--ros-args', '-r', '__ns:=/'+robot, '-p', 'use_sim_time:=true'],
                output='screen'))
        actions.extend([
            Node(package='robot_state_publisher', executable='robot_state_publisher', namespace=robot,
                remappings=[('/tf', '/'+robot+'/tf'), ('/tf_static', '/'+robot+'/tf_static')],
                parameters=[{'robot_description': description, 'use_sim_time': True}], output='screen'),
            Node(package='gazebo_ros', executable='spawn_entity.py', namespace=robot, output='screen',
                arguments=['-file', str(sdf_path), '-entity', robot, '-x', str(position['x']),
                           '-y', str(position['y']), '-z', '.02', '-Y', str(spawn.get('start_yaw_rad', 0.))]),
            include(launch_dir/'sensor_localization.launch.py', robot_namespace=robot,
                config=bundle/config['localization'], map=bundle/config.get('map', 'map.yaml'),
                laser_max_range=manifest['laser_range_max_m'],
                laser_model_type=localization['laser_model_type']),
            include(launch_dir/'metric_controller.launch.py', robot_namespace=robot,
                config=bundle/config['controller']),
        ])
    return actions


def generate_launch_description():
    return LaunchDescription([DeclareLaunchArgument('bundle'), DeclareLaunchArgument('model_output'),
                              OpaqueFunction(function=start)])
