"""Metric validation surrogate: chassis collision, encoder odometry, LiDAR/IMU."""
import importlib.util
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
import xacro


def start(context):
    source = Path(__file__).with_name("metric_model.py")
    spec = importlib.util.spec_from_file_location("amr_metric_geometry", source)
    geometry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(geometry)
    share = Path(get_package_share_directory("scout_gazebo_description"))
    base_urdf = LaunchConfiguration("base_urdf").perform(context)
    if base_urdf:
        robot = ET.parse(base_urdf).getroot()
    else:
        robot = ET.fromstring(xacro.process_file(str(share / "urdf/scout_mini.urdf.xacro"),
                                               mappings={"drive_mode": "diff"}).toxml())
    lateral_mu = LaunchConfiguration("lateral_mu").perform(context)
    laser_range = LaunchConfiguration("laser_range_max").perform(context)
    laser_fov = LaunchConfiguration('laser_fov_radians').perform(context)
    geometry.augment(robot, wheel_mu=float(LaunchConfiguration("wheel_mu").perform(context)),
                     lateral_mu=float(lateral_mu) if lateral_mu else None,
                     contact_scale=float(LaunchConfiguration("contact_scale").perform(context)),
                     imu_gyro_noise=float(LaunchConfiguration("imu_gyro_noise").perform(context)),
                     laser_range_max=float(laser_range) if laser_range else None,
                     laser_fov_radians=float(laser_fov) if laser_fov else None)
    if LaunchConfiguration('front_tracking').perform(context).lower()=='true':
        geometry.add_tracking_reference(robot)
    drive = robot.find(".//plugin[@filename='libgazebo_ros_diff_drive.so']")
    ET.SubElement(drive.find("ros"), "remapping").text = "cmd_vel:=guarded_cmd_vel"
    ET.SubElement(drive.find("ros"), "remapping").text = "odom:="+LaunchConfiguration("odom_topic").perform(context)
    drive.find("publish_odom_tf").text = LaunchConfiguration("odom_tf").perform(context)
    drive.find("publish_wheel_tf").text = "false"  # robot_state_publisher owns wheel TF
    ET.SubElement(ET.SubElement(robot, "gazebo"), "plugin", name="amr_motion_guard", filename="libamr_motion_guard.so")
    namespace = LaunchConfiguration('robot_namespace').perform(context)
    if namespace:
        geometry.namespace_robot(robot, namespace)
    description = ET.tostring(robot, encoding="unicode")
    model_out = LaunchConfiguration("model_out").perform(context)
    if model_out:
        Path(model_out).write_text(description)
    actions = [
        SetEnvironmentVariable("GAZEBO_PLUGIN_PATH", str(Path(get_package_prefix("amr_gazebo_guard")) / "lib")
                               + os.pathsep + os.environ.get("GAZEBO_PLUGIN_PATH", "")),
    ]
    if LaunchConfiguration('start_world').perform(context).lower() == 'true':
        actions.append(IncludeLaunchDescription(
            PythonLaunchDescriptionSource(str(Path(get_package_share_directory('gazebo_ros')) / 'launch/gzserver.launch.py')),
            launch_arguments={'world':LaunchConfiguration('world'),'verbose':'false',
                              'params_file':LaunchConfiguration('clock_config')}.items()))
    remappings = [('/tf','/'+namespace+'/tf'),('/tf_static','/'+namespace+'/tf_static')] if namespace else []
    actions += [
        Node(package="robot_state_publisher", executable="robot_state_publisher",namespace=namespace,
             remappings=remappings,
             parameters=[{"robot_description": description, "use_sim_time": True}], output="screen"),
        Node(package="gazebo_ros", executable="spawn_entity.py", output="screen",namespace=namespace,
             arguments=["-topic", "robot_description", "-entity", namespace or "scout_mini", "-x", LaunchConfiguration("x"),
                        "-y", LaunchConfiguration("y"), "-z", LaunchConfiguration("z"),
                        "-Y", LaunchConfiguration("yaw")]),
    ]
    return actions


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world"),
        DeclareLaunchArgument('start_world',default_value='true'),
        DeclareLaunchArgument('robot_namespace',default_value=''),
        DeclareLaunchArgument('front_tracking',default_value='false'),
        DeclareLaunchArgument("clock_config", default_value=str(Path(__file__).with_name("metric_clock.yaml"))),
        DeclareLaunchArgument("x", default_value="0"),
        DeclareLaunchArgument("y", default_value="0"),
        DeclareLaunchArgument("z", default_value="0.02"),
        DeclareLaunchArgument("yaw", default_value="0"),
        DeclareLaunchArgument("wheel_mu", default_value="1"),
        DeclareLaunchArgument("lateral_mu", default_value="0.05"),
        DeclareLaunchArgument("contact_scale", default_value="10"),
        DeclareLaunchArgument("base_urdf", default_value=""),
        DeclareLaunchArgument("odom_topic", default_value="odom"),
        DeclareLaunchArgument("odom_tf", default_value="true"),
        DeclareLaunchArgument("imu_gyro_noise", default_value="0"),
        DeclareLaunchArgument("laser_range_max", default_value=""),
        DeclareLaunchArgument('laser_fov_radians', default_value=''),
        DeclareLaunchArgument("model_out", default_value=""),
        OpaqueFunction(function=start),
    ])
