"""Metric ROS 2 Gazebo launcher for actuated black-orange wheelset_agv carriers."""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction, SetEnvironmentVariable
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _load_agv_model_module():
    source = Path(__file__).with_name("wheelset_v2_agv_model.py")
    spec = importlib.util.spec_from_file_location("wheelset_v2_agv_model", source)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def start(context):
    agv_mod = _load_agv_model_module()
    namespace = LaunchConfiguration("robot_namespace").perform(context)
    description = agv_mod.build_wheelset_agv_urdf(
        namespace=namespace,
        wheel_mu=float(LaunchConfiguration("wheel_mu").perform(context)),
        lateral_mu=float(LaunchConfiguration("lateral_mu").perform(context)),
        contact_scale=float(LaunchConfiguration("contact_scale").perform(context)),
        imu_gyro_noise=float(LaunchConfiguration("imu_gyro_noise").perform(context)),
        laser_range_max=float(LaunchConfiguration("laser_range_max").perform(context)),
        laser_fov_radians=float(LaunchConfiguration("laser_fov_radians").perform(context)),
        add_tracking=LaunchConfiguration("front_tracking").perform(context).lower() == "true",
    )
    model_out = LaunchConfiguration("model_out").perform(context)
    if model_out:
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        Path(model_out).write_text(description, encoding="utf-8")

    models_dir = str(Path(__file__).resolve().parents[2] / "assets" / "wheelset_workshop" / "models")
    existing_models = os.environ.get("GAZEBO_MODEL_PATH", "")
    model_path = models_dir if not existing_models else f"{models_dir}{os.pathsep}{existing_models}"
    guard_lib = str(Path(get_package_prefix("amr_gazebo_guard")) / "lib")
    existing_plugins = os.environ.get("GAZEBO_PLUGIN_PATH", "")
    plugin_path = guard_lib if not existing_plugins else f"{guard_lib}{os.pathsep}{existing_plugins}"

    actions = [
        SetEnvironmentVariable("GAZEBO_MODEL_PATH", model_path),
        SetEnvironmentVariable("GAZEBO_PLUGIN_PATH", plugin_path),
    ]
    if LaunchConfiguration("start_world").perform(context).lower() == "true":
        actions.append(
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(Path(get_package_share_directory("gazebo_ros")) / "launch/gzserver.launch.py")
                ),
                launch_arguments={
                    "world": LaunchConfiguration("world"),
                    "verbose": "false",
                    "lockstep": LaunchConfiguration("lockstep"),
                    "params_file": LaunchConfiguration("clock_config"),
                }.items(),
            )
        )
    remappings = [("/tf", "/" + namespace + "/tf"), ("/tf_static", "/" + namespace + "/tf_static")] if namespace else []
    actions += [
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            namespace=namespace,
            remappings=remappings,
            parameters=[{"robot_description": description, "use_sim_time": True}],
            output="screen",
        ),
        Node(
            package="gazebo_ros",
            executable="spawn_entity.py",
            output="screen",
            namespace=namespace,
            arguments=[
                "-topic",
                "robot_description",
                "-entity",
                namespace or "wheelset_agv_1",
                "-x",
                LaunchConfiguration("x"),
                "-y",
                LaunchConfiguration("y"),
                "-z",
                LaunchConfiguration("z"),
                "-Y",
                LaunchConfiguration("yaw"),
                "-timeout",
                LaunchConfiguration("spawn_timeout"),
            ],
        ),
    ]
    return actions


def generate_launch_description():
    default_clock = str(
        Path(__file__).resolve().parents[1] / "ros2" / "amr_gazebo_guard" / "launch" / "metric_clock.yaml"
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("world", default_value=""),
            DeclareLaunchArgument("lockstep", default_value="false"),
            DeclareLaunchArgument("spawn_timeout", default_value="180"),
            DeclareLaunchArgument("start_world", default_value="true"),
            DeclareLaunchArgument("robot_namespace", default_value=""),
            DeclareLaunchArgument("front_tracking", default_value="true"),
            DeclareLaunchArgument("clock_config", default_value=default_clock),
            DeclareLaunchArgument("x", default_value="4.0"),
            DeclareLaunchArgument("y", default_value="2.5"),
            DeclareLaunchArgument("z", default_value="0.02"),
            DeclareLaunchArgument("yaw", default_value="0.0"),
            DeclareLaunchArgument("wheel_mu", default_value="1.0"),
            DeclareLaunchArgument("lateral_mu", default_value="0.05"),
            DeclareLaunchArgument("contact_scale", default_value="10.0"),
            DeclareLaunchArgument("imu_gyro_noise", default_value="0.002"),
            DeclareLaunchArgument("laser_range_max", default_value="60.0"),
            DeclareLaunchArgument("laser_fov_radians", default_value="4.18879008"),
            DeclareLaunchArgument("model_out", default_value=""),
            OpaqueFunction(function=start),
        ]
    )
