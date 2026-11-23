"""Explicit synthetic SCOUT surrogate geometry for metric physical validation.

These dimensions derive from the archived local mesh, not measured hardware.
Keep the G02/G03 integration model unchanged; this is a separate model variant.
"""
import math
import re
import xml.etree.ElementTree as ET

BODY_SIZE = (.562, .392, .184)
BODY_CENTER = (.0177, 0., .093)
VISUAL_Z_OFFSET = .115
WHEEL_RADIUS = .08
WHEEL_WIDTH = .084
WHEELBASE = .463951
TRACK = .416503
# A conservative axis-aligned footprint includes the wheels and body.
FOOTPRINT = ((-.315, -.252), (.315, -.252), (.315, .252), (-.315, .252))
TRACKING_REFERENCE_X = WHEELBASE / 2.


def tag(parent, element_name, text=None, **attributes):
    element = ET.SubElement(parent, element_name, **attributes)
    element.text = text
    return element


def add_tracking_reference(robot):
    """Add a massless front-axle coordinate frame; physical geometry is unchanged."""
    if robot.find("link[@name='front_tracking']") is not None:
        raise ValueError('tracking frame already exists')
    tag(robot,'link',name='front_tracking')
    joint=tag(robot,'joint',name='front_tracking_joint',type='fixed')
    tag(joint,'parent',link='base_footprint')
    tag(joint,'child',link='front_tracking')
    tag(joint,'origin',xyz=f'{TRACKING_REFERENCE_X} 0 0',rpy='0 0 0')
    return robot


def namespace_robot(robot, namespace):
    """Isolate every robot plugin's commands, sensors and TF on its ROS namespace."""
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', namespace):
        raise ValueError('one simple ROS robot namespace required')
    for plugin in robot.findall('.//plugin'):
        ros = plugin.find('ros')
        if ros is None:
            ros = tag(plugin, 'ros')
        existing = ros.find('namespace')
        if existing is None:
            existing = tag(ros, 'namespace')
        existing.text = '/' + namespace
        tag(ros, 'remapping', '/tf:=/' + namespace + '/tf')
        tag(ros, 'remapping', '/tf_static:=/' + namespace + '/tf_static')
    return robot


def augment(robot, *, wheel_mu=50., lateral_mu=None, contact_scale=1., imu_gyro_noise=0.,
            laser_range_max=None, laser_fov_radians=None):
    """Add body collision and IMU, choose encoder odometry, align wheel axes."""
    if not math.isfinite(wheel_mu) or wheel_mu <= 0:
        raise ValueError("positive finite friction required")
    if not math.isfinite(contact_scale) or contact_scale <= 0:
        raise ValueError("positive finite contact scale required")
    if lateral_mu is not None and (not math.isfinite(lateral_mu) or lateral_mu <= 0):
        raise ValueError("positive finite lateral friction required")
    if not math.isfinite(imu_gyro_noise) or imu_gyro_noise < 0:
        raise ValueError("nonnegative finite IMU gyro noise required")
    if laser_fov_radians is not None:
        if not math.isfinite(laser_fov_radians) or not 0<laser_fov_radians<=2*math.pi:
            raise ValueError('laser field of view must be finite and within (0, 2pi]')
        horizontal=robot.find(".//sensor[@type='ray']/ray/scan/horizontal")
        if horizontal is None:
            raise ValueError('laser horizontal scan required')
        old_span=float(horizontal.findtext('max_angle'))-float(horizontal.findtext('min_angle'))
        spacing=old_span/(int(horizontal.findtext('samples'))-1)
        horizontal.find('min_angle').text=str(-laser_fov_radians/2)
        horizontal.find('max_angle').text=str(laser_fov_radians/2)
        horizontal.find('samples').text=str(math.ceil(laser_fov_radians/spacing)+1)
    if laser_range_max is not None:
        laser_range = robot.find(".//sensor[@type='ray']/ray/range")
        if laser_range is None:
            raise ValueError("configured laser requires an existing ray sensor")
        if not math.isfinite(laser_range_max) or laser_range_max <= float(laser_range.findtext('min')):
            raise ValueError("laser maximum range must be finite and above its minimum")
        laser_range.find("max").text = str(laser_range_max)
    base = robot.find("link[@name='base_link']")
    if base is None or base.find("collision") is not None:
        raise ValueError("expected the archived base model without chassis collision")
    visual = base.find("visual/origin")
    if visual is None or visual.attrib["rpy"] != "1.57 0 -1.57":
        raise ValueError("body visual no longer matches audited geometry")
    # The archived visual penetrated ground by 33mm. Translate the body above
    # the axle, then bound the translated mesh instead of levitating the wheels.
    visual.set("xyz", f"0 0 {VISUAL_Z_OFFSET}")
    collision = tag(base, "collision", name="metric_chassis")
    tag(collision, "origin", xyz=" ".join(map(str, BODY_CENTER)), rpy="0 0 0")
    tag(tag(collision, "geometry"), "box", size=" ".join(map(str, BODY_SIZE)))
    # Retain base mass23kg and four2.5kg wheels; this 33.1kg surrogate is not
    # calibrated hardware. Shift its COM consistently with the raised chassis.
    base.find("inertial/origin").set("xyz", " ".join(map(str, BODY_CENTER)))
    names = ("front_left", "front_right", "rear_left", "rear_right")
    for name in names:
        joint = robot.find(f"joint[@name='{name}_wheel']")
        joint.find("origin").set("rpy", f"{-math.pi/2:.17g} 0 0")
        surface = robot.find(f"gazebo[@reference='{name}_wheel_link']")
        # Cylinder-local z is the axle: invariant under wheel spin and tangent
        # to a level floor. Using the radial x direction here is invalid.
        surface.find("fdir1").text = "0 0 0" if lateral_mu is None else "0 0 1"
        surface.find("mu1").text = str(wheel_mu if lateral_mu is None else lateral_mu)
        surface.find("mu2").text = str(wheel_mu)
        if contact_scale != 1.:
            for key in ("kp", "kd"):
                parameter = surface.find(key)
                parameter.text = str(float(parameter.text)*contact_scale)
    # Raise virtual LiDAR above the translated chassis; preserve ray noise.
    robot.find("joint[@name='laser_joint']/origin").set("xyz", "0.15 0 0.32")
    # The synthetic sensing housing is not an obstacle. CPU rays begin at the
    # old cylinder's radius (the configured minimum range), causing self hits.
    laser = robot.find("link[@name='laser_link']")
    if laser is not None:
        for shape in laser.findall("collision"):
            laser.remove(shape)
    imu = tag(robot, "link", name="imu_link")
    inertial = tag(imu, "inertial")
    tag(inertial, "mass", value="0.001")
    tag(inertial, "inertia", ixx="0.000001", iyy="0.000001", izz="0.000001", ixy="0", ixz="0", iyz="0")
    joint = tag(robot, "joint", name="imu_joint", type="fixed")
    tag(joint, "parent", link="base_link")
    tag(joint, "child", link="imu_link")
    tag(joint, "origin", xyz="0 0 .093", rpy="0 0 0")
    sensor = tag(tag(robot, "gazebo", reference="imu_link"), "sensor", name="metric_imu", type="imu")
    tag(sensor, "always_on", "true")
    tag(sensor, "update_rate", "50")
    imu_settings = tag(sensor, "imu")
    if imu_gyro_noise:
        for kind,stddev in (("angular_velocity",imu_gyro_noise),("linear_acceleration",.02)):
            axes = tag(imu_settings,kind)
            for axis in ("x","y","z"):
                noise = tag(tag(axes,axis),"noise",type="gaussian")
                tag(noise,"mean","0")
                tag(noise,"stddev",str(stddev))
    plugin = tag(sensor, "plugin", name="metric_imu_ros", filename="libgazebo_ros_imu_sensor.so")
    tag(tag(plugin, "ros"), "remapping", "~/out:=imu/data")
    tag(plugin, "frame_name", "imu_link")
    tag(plugin, "initial_orientation_as_reference", "false")
    drive = robot.find(".//plugin[@filename='libgazebo_ros_diff_drive.so']")
    tag(drive, "odometry_source", "0")  # ENCODER; plugin default1 is WORLD truth
    drive.find("publish_odom_tf").text = "true"
    # Contacts are observable, including chassis contacts; do not use a
    # Boolean 'no collision' inferred from path geometry alone.
    bumper = tag(tag(robot, "gazebo", reference="base_link"), "sensor", name="chassis_contacts", type="contact")
    tag(bumper, "always_on", "true")
    tag(bumper, "update_rate", "10")
    tag(tag(bumper, "contact"), "collision", "base_footprint_fixed_joint_lump__metric_chassis_collision")
    plugin = tag(bumper, "plugin", name="metric_contacts_ros", filename="libgazebo_ros_bumper.so")
    tag(tag(plugin, "ros"), "remapping", "bumper_states:=chassis_contacts")
    tag(plugin, "frame_name", "base_link")
    return robot
