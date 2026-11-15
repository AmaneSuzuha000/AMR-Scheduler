"""Original SCOUT geometry/drive, corrected friction basis, independent guard."""
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
    share = Path(get_package_share_directory("scout_gazebo_description"))
    robot = ET.fromstring(xacro.process_file(str(share / "urdf/scout_mini.urdf.xacro"),
                                           mappings={"drive_mode": "diff"}).toxml())
    for name in ("front_left_wheel_link", "front_right_wheel_link", "rear_left_wheel_link", "rear_right_wheel_link"):
        direction = robot.find(f"gazebo[@reference='{name}']/fdir1")
        if direction is None:
            raise RuntimeError("wheel friction configuration missing: " + name)
        # A spinning wheel-local X axis is not a contact tangent. For the
        # existing isotropic mu1 == mu2, let ODE choose a valid tangent basis.
        direction.text = "0 0 0"
    drive = robot.find(".//plugin[@filename='libgazebo_ros_diff_drive.so']")
    if drive is None or robot.find(".//plugin[@filename='libgazebo_ros_planar_move.so']") is not None:
        raise RuntimeError("exactly the native differential-drive model required")
    ros = drive.find("ros")
    if ros is None:
        ros = ET.SubElement(drive, "ros")
    for remap in list(ros.findall("remapping")):
        if (remap.text or "").split(":=")[0].lstrip("/") == "cmd_vel":
            ros.remove(remap)
    ET.SubElement(ros, "remapping").text = "cmd_vel:=guarded_cmd_vel"
    gazebo = ET.SubElement(robot, "gazebo")
    ET.SubElement(gazebo, "plugin", name="amr_motion_guard", filename="libamr_motion_guard.so")
    description = ET.tostring(robot, encoding="unicode")
    gazebo_share = Path(get_package_share_directory("gazebo_ros"))
    return [
        SetEnvironmentVariable("GAZEBO_PLUGIN_PATH", str(Path(get_package_prefix("amr_gazebo_guard")) / "lib")
                               + os.pathsep + os.environ.get("GAZEBO_PLUGIN_PATH", "")),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(str(gazebo_share / "launch/gzserver.launch.py")),
                                 launch_arguments={"world": LaunchConfiguration("world"), "verbose": "false"}.items()),
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": description, "use_sim_time": True}], output="screen"),
        Node(package="gazebo_ros", executable="spawn_entity.py", output="screen",
             arguments=["-topic", "robot_description", "-entity", "scout_mini",
                        "-x", LaunchConfiguration("x"), "-y", LaunchConfiguration("y"), "-z", LaunchConfiguration("z")]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("world", description="Explicit collision world; the integration fixture is scaled 1:25"),
        DeclareLaunchArgument("x", default_value="0.32"),
        DeclareLaunchArgument("y", default_value="1.2"),
        DeclareLaunchArgument("z", default_value="0.06"),
        OpaqueFunction(function=start),
    ])
