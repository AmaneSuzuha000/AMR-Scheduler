"""Metric actuated black-orange wheelset AGV carrier & dynamic workpiece models for wheelset_workshop_v2."""
from __future__ import annotations

import math
import re
import xml.etree.ElementTree as ET

BODY_SIZE = (1.70, 1.50, 0.14)
BODY_CENTER_BASE = (0.0, 0.0, 0.065)
WHEEL_RADIUS = 0.08
WHEEL_WIDTH = 0.06
WHEELBASE = 1.16
TRACK = 1.24
FOOTPRINT = ((-0.85, -0.75), (0.85, -0.75), (0.85, 0.75), (-0.85, 0.75))
TRACKING_REFERENCE_X = 0.58
CARRIER_DECK_WHEELSET_Z = 0.17  # wheelset tread bottom 0.095 + 0.17 = 0.265 m (seated on orange cradles)


def _tag(parent: ET.Element, tag_name: str, /, text: str | None = None, **attrs: str) -> ET.Element:
    elem = ET.SubElement(parent, tag_name, **attrs)
    if text is not None:
        elem.text = text
    return elem


def namespace_robot(robot: ET.Element, namespace: str) -> ET.Element:
    """Isolate every Gazebo ROS plugin and TF stream under /<namespace>."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", namespace):
        raise ValueError("simple ROS robot namespace required")
    for plugin in robot.findall(".//plugin"):
        ros = plugin.find("ros")
        if ros is None:
            ros = _tag(plugin, "ros")
        ns = ros.find("namespace")
        if ns is None:
            ns = _tag(ros, "namespace")
        ns.text = "/" + namespace
        _tag(ros, "remapping", "/tf:=/" + namespace + "/tf")
        _tag(ros, "remapping", "/tf_static:=/" + namespace + "/tf_static")
    return robot


def build_wheelset_agv_urdf(
    namespace: str = "",
    *,
    wheel_mu: float = 1.0,
    lateral_mu: float = 0.05,
    contact_scale: float = 10.0,
    imu_gyro_noise: float = 0.002,
    laser_range_max: float = 60.0,
    laser_fov_radians: float = 4.18879008,  # +-120 deg; clears carried wheelset & deck
    add_tracking: bool = True,
) -> str:
    """Build metric URDF XML for an actuated black-orange wheelset carrier AGV with motion guard."""
    if not math.isfinite(wheel_mu) or wheel_mu <= 0:
        raise ValueError("positive finite wheel_mu required")
    if not math.isfinite(lateral_mu) or lateral_mu <= 0:
        raise ValueError("positive finite lateral_mu required")
    if not math.isfinite(contact_scale) or contact_scale <= 0:
        raise ValueError("positive finite contact_scale required")
    if not math.isfinite(imu_gyro_noise) or imu_gyro_noise < 0:
        raise ValueError("nonnegative finite imu_gyro_noise required")
    if not math.isfinite(laser_range_max) or laser_range_max <= 0.15:
        raise ValueError("laser_range_max must exceed 0.15 m")
    if not math.isfinite(laser_fov_radians) or not 0 < laser_fov_radians <= 2.0 * math.pi:
        raise ValueError("laser_fov_radians must be in (0, 2pi]")

    robot = ET.Element("robot", name="wheelset_agv_v2")
    _tag(robot, "link", name="base_footprint")

    base_joint = _tag(robot, "joint", name="base_joint", type="fixed")
    _tag(base_joint, "parent", link="base_footprint")
    _tag(base_joint, "child", link="base_link")
    _tag(base_joint, "origin", xyz=f"0 0 {WHEEL_RADIUS}", rpy="0 0 0")

    base = _tag(robot, "link", name="base_link")
    vis = _tag(base, "visual", name="carrier_mesh")
    _tag(vis, "origin", xyz=f"0 0 {-WHEEL_RADIUS}", rpy="0 0 0")
    geom = _tag(vis, "geometry")
    _tag(geom, "mesh", filename="model://wsv1_wheelset_agv/meshes/wheelset_agv.dae", scale="1 1 1")

    col = _tag(base, "collision", name="metric_chassis")
    _tag(col, "origin", xyz=" ".join(map(str, BODY_CENTER_BASE)), rpy="0 0 0")
    _tag(_tag(col, "geometry"), "box", size=" ".join(map(str, BODY_SIZE)))

    inert = _tag(base, "inertial")
    _tag(inert, "origin", xyz=" ".join(map(str, BODY_CENTER_BASE)), rpy="0 0 0")
    _tag(inert, "mass", value="180.0")
    _tag(inert, "inertia", ixx="34.0", ixy="0", ixz="0", iyy="44.0", iyz="0", izz="75.0")

    wheel_specs = (
        ("front_left", WHEELBASE / 2.0, TRACK / 2.0),
        ("front_right", WHEELBASE / 2.0, -TRACK / 2.0),
        ("rear_left", -WHEELBASE / 2.0, TRACK / 2.0),
        ("rear_right", -WHEELBASE / 2.0, -TRACK / 2.0),
    )
    kp_val = str(1_000_000.0 * contact_scale)
    kd_val = str(100.0 * contact_scale)
    for wname, wx, wy in wheel_specs:
        link_name = f"{wname}_wheel_link"
        wlink = _tag(robot, "link", name=link_name)
        wcol = _tag(wlink, "collision")
        _tag(wcol, "origin", xyz="0 0 0", rpy="0 0 0")
        _tag(_tag(wcol, "geometry"), "cylinder", length=str(WHEEL_WIDTH), radius=str(WHEEL_RADIUS))
        winert = _tag(wlink, "inertial")
        _tag(winert, "origin", xyz="0 0 0", rpy="0 0 0")
        _tag(winert, "mass", value="5.0")
        _tag(winert, "inertia", ixx="0.01", ixy="0", ixz="0", iyy="0.01", iyz="0", izz="0.016")

        wjoint = _tag(robot, "joint", name=f"{wname}_wheel", type="continuous")
        _tag(wjoint, "parent", link="base_link")
        _tag(wjoint, "child", link=link_name)
        _tag(wjoint, "origin", xyz=f"{wx} {wy} 0", rpy=f"{-math.pi / 2:.17g} 0 0")
        _tag(wjoint, "axis", xyz="0 0 1")
        _tag(wjoint, "dynamics", damping="0.1", friction="0.1")

        gsurf = _tag(robot, "gazebo", reference=link_name)
        _tag(gsurf, "mu1", str(lateral_mu))
        _tag(gsurf, "mu2", str(wheel_mu))
        _tag(gsurf, "fdir1", "0 0 1")
        _tag(gsurf, "kp", kp_val)
        _tag(gsurf, "kd", kd_val)
        _tag(gsurf, "minDepth", "0.001")
        _tag(gsurf, "maxContacts", "8")

    if add_tracking:
        _tag(robot, "link", name="front_tracking")
        tjoint = _tag(robot, "joint", name="front_tracking_joint", type="fixed")
        _tag(tjoint, "parent", link="base_footprint")
        _tag(tjoint, "child", link="front_tracking")
        _tag(tjoint, "origin", xyz=f"{TRACKING_REFERENCE_X} 0 0", rpy="0 0 0")

    imu = _tag(robot, "link", name="imu_link")
    imu_inert = _tag(imu, "inertial")
    _tag(imu_inert, "mass", value="0.001")
    _tag(imu_inert, "inertia", ixx="0.000001", iyy="0.000001", izz="0.000001", ixy="0", ixz="0", iyz="0")
    imu_joint = _tag(robot, "joint", name="imu_joint", type="fixed")
    _tag(imu_joint, "parent", link="base_link")
    _tag(imu_joint, "child", link="imu_link")
    _tag(imu_joint, "origin", xyz="0 0 0.065", rpy="0 0 0")

    imu_sensor = _tag(_tag(robot, "gazebo", reference="imu_link"), "sensor", name="metric_imu", type="imu")
    _tag(imu_sensor, "always_on", "true")
    _tag(imu_sensor, "update_rate", "50")
    imu_cfg = _tag(imu_sensor, "imu")
    if imu_gyro_noise:
        for kind, stddev in (("angular_velocity", imu_gyro_noise), ("linear_acceleration", 0.02)):
            axes = _tag(imu_cfg, kind)
            for axis in ("x", "y", "z"):
                noise = _tag(_tag(axes, axis), "noise", type="gaussian")
                _tag(noise, "mean", "0")
                _tag(noise, "stddev", str(stddev))
    imu_plug = _tag(imu_sensor, "plugin", name="metric_imu_ros", filename="libgazebo_ros_imu_sensor.so")
    _tag(_tag(imu_plug, "ros"), "remapping", "~/out:=imu/data")
    _tag(imu_plug, "frame_name", "imu_link")
    _tag(imu_plug, "initial_orientation_as_reference", "false")

    # Front bumper LiDAR at x=+0.88, z_world=0.23 (z_base=0.15) above the 0.205 m deck
    laser = _tag(robot, "link", name="laser_link")
    lvis = _tag(laser, "visual")
    _tag(lvis, "origin", xyz="0 0 0", rpy="0 0 0")
    _tag(_tag(lvis, "geometry"), "cylinder", length="0.04", radius="0.045")
    lmat = _tag(lvis, "material", name="lidar_dark")
    _tag(lmat, "color", rgba="0.18 0.18 0.18 1")
    linert = _tag(laser, "inertial")
    _tag(linert, "mass", value="0.1")
    _tag(linert, "inertia", ixx="0.0001", ixy="0", ixz="0", iyy="0.0001", iyz="0", izz="0.0001")
    ljoint = _tag(robot, "joint", name="laser_joint", type="fixed")
    _tag(ljoint, "parent", link="base_link")
    _tag(ljoint, "child", link="laser_link")
    _tag(ljoint, "origin", xyz="0.88 0 0.15", rpy="0 0 0")

    lsensor = _tag(_tag(robot, "gazebo", reference="laser_link"), "sensor", name="wheelset_agv_lidar", type="ray")
    _tag(lsensor, "always_on", "true")
    _tag(lsensor, "visualize", "false")
    _tag(lsensor, "update_rate", "20")
    _tag(lsensor, "pose", "0 0 0 0 0 0")
    ray = _tag(lsensor, "ray")
    horiz = _tag(_tag(ray, "scan"), "horizontal")
    samples = max(181, int(math.ceil(laser_fov_radians / (4.18879008 / 240.0))) + 1)
    _tag(horiz, "samples", str(samples))
    _tag(horiz, "resolution", "1")
    _tag(horiz, "min_angle", str(-laser_fov_radians / 2.0))
    _tag(horiz, "max_angle", str(laser_fov_radians / 2.0))
    rrange = _tag(ray, "range")
    _tag(rrange, "min", "0.10")
    _tag(rrange, "max", str(laser_range_max))
    _tag(rrange, "resolution", "0.01")
    rnoise = _tag(ray, "noise")
    _tag(rnoise, "type", "gaussian")
    _tag(rnoise, "mean", "0.0")
    _tag(rnoise, "stddev", "0.01")
    rplug = _tag(lsensor, "plugin", name="wheelset_agv_ray", filename="libgazebo_ros_ray_sensor.so")
    _tag(_tag(rplug, "ros"), "remapping", "~/out:=scan")
    _tag(rplug, "output_type", "sensor_msgs/LaserScan")
    _tag(rplug, "frame_name", "laser_link")

    # Bumper contact plugin on metric_chassis
    bumper = _tag(_tag(robot, "gazebo", reference="base_link"), "sensor", name="chassis_contacts", type="contact")
    _tag(bumper, "always_on", "true")
    _tag(bumper, "update_rate", "10")
    _tag(_tag(bumper, "contact"), "collision", "base_footprint_fixed_joint_lump__metric_chassis_collision")
    bplug = _tag(bumper, "plugin", name="metric_contacts_ros", filename="libgazebo_ros_bumper.so")
    _tag(_tag(bplug, "ros"), "remapping", "bumper_states:=chassis_contacts")
    _tag(bplug, "frame_name", "base_link")

    # Differential drive + joint states + motion guard
    dgaz = _tag(robot, "gazebo")
    drive = _tag(dgaz, "plugin", name="wheelset_agv_diff_drive", filename="libgazebo_ros_diff_drive.so")
    dros = _tag(drive, "ros")
    _tag(dros, "remapping", "cmd_vel:=guarded_cmd_vel")
    _tag(dros, "remapping", "odom:=wheel/odometry")
    _tag(drive, "update_rate", "50")
    _tag(drive, "num_wheel_pairs", "2")
    _tag(drive, "left_joint", "front_left_wheel")
    _tag(drive, "right_joint", "front_right_wheel")
    _tag(drive, "left_joint", "rear_left_wheel")
    _tag(drive, "right_joint", "rear_right_wheel")
    _tag(drive, "wheel_separation", str(TRACK))
    _tag(drive, "wheel_separation", str(TRACK))
    _tag(drive, "wheel_diameter", str(2.0 * WHEEL_RADIUS))
    _tag(drive, "wheel_diameter", str(2.0 * WHEEL_RADIUS))
    _tag(drive, "max_wheel_torque", "400")
    _tag(drive, "max_wheel_acceleration", "4.0")
    _tag(drive, "odometry_source", "0")
    _tag(drive, "publish_odom", "true")
    _tag(drive, "publish_odom_tf", "false")
    _tag(drive, "publish_wheel_tf", "false")
    _tag(drive, "odometry_frame", "odom")
    _tag(drive, "robot_base_frame", "base_footprint")

    jgaz = _tag(robot, "gazebo")
    js = _tag(jgaz, "plugin", name="wheelset_agv_joint_states", filename="libgazebo_ros_joint_state_publisher.so")
    _tag(js, "ros")
    _tag(js, "update_rate", "50")
    for wname, _, _ in wheel_specs:
        _tag(js, "joint_name", f"{wname}_wheel")

    mgaz = _tag(robot, "gazebo")
    _tag(mgaz, "plugin", name="amr_motion_guard", filename="libamr_motion_guard.so")

    if namespace:
        namespace_robot(robot, namespace)
    return ET.tostring(robot, encoding="unicode")


def build_dynamic_wheelset_sdf(name: str) -> str:
    """Visual-only dynamic wheelset workpiece SDF seated accurately without airborne flight."""
    root = ET.Element("sdf", version="1.6")
    model = _tag(root, "model", name=name)
    _tag(model, "static", "true")
    link = _tag(model, "link", name="workpiece")
    visual = _tag(link, "visual", name="visual")
    _tag(_tag(_tag(visual, "geometry"), "mesh"), "uri", "model://wsv1_wheelset/meshes/wheelset.dae")
    return ET.tostring(root, encoding="unicode")
