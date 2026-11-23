"""Build the navigation surrogate and apply contact parameters in actual SDF."""
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET


WHEELS = ('front_left', 'front_right', 'rear_left', 'rear_right')


def set_lateral_compliance(sdf, slip):
    if not math.isfinite(slip) or slip <= 0:
        raise ValueError('positive finite lateral contact compliance required')
    for name in WHEELS:
        link = sdf.find(f"model/link[@name='{name}_wheel_link']")
        if link is None or len(link.findall('collision')) != 1:
            raise ValueError('one wheel collision required: '+name)
        ode = link.find('collision/surface/friction/ode')
        if ode is None or ode.findtext('fdir1') != '0 0 1':
            raise ValueError('audited lateral wheel direction required: '+name)
        element = ode.find('slip1')
        if element is None:
            element = ET.SubElement(ode, 'slip1')
        element.text = str(slip)
    return sdf


def build_model(base, geometry, namespace, output, manifest):
    output = Path(output)
    geometry.augment(base, wheel_mu=1., lateral_mu=.05, contact_scale=10.,
        laser_range_max=manifest['laser_range_max_m'], laser_fov_radians=manifest['laser_fov_radians'])
    lift=(manifest.get('contact_bins') or {}).get('geometry',{}).get('platform_lift_m',0.)
    if lift:
        # Raise the actual URDF joint as well as the map/fit descriptor. The
        # original 20mm clearance above a loaded bin is preserved.
        base.find("joint[@name='laser_joint']/origin").set("xyz",f"0.15 0 {0.32+lift:.17g}")
    rate = manifest.get('laser_update_rate_hz')
    if rate is not None:
        if not math.isfinite(rate) or rate <= 0:
            raise ValueError('positive finite laser update rate required')
        lasers = base.findall(".//sensor[@type='ray']")
        if len(lasers) != 1:
            raise ValueError('one actual laser required for explicit update rate')
        lasers[0].find('update_rate').text = str(rate)
    drive = base.find(".//plugin[@filename='libgazebo_ros_diff_drive.so']")
    ET.SubElement(drive.find('ros'), 'remapping').text = 'cmd_vel:=guarded_cmd_vel'
    ET.SubElement(drive.find('ros'), 'remapping').text = 'odom:=wheel/odometry'
    drive.find('publish_odom_tf').text = 'false'
    drive.find('publish_wheel_tf').text = 'false'
    guard = ET.SubElement(ET.SubElement(base, 'gazebo'), 'plugin', name='amr_motion_guard',
                          filename='libamr_motion_guard.so')
    geometry.namespace_robot(base, namespace)
    if manifest['model_parameters'].get('guard_command_relay'):
        ET.SubElement(guard.find('ros'), 'remapping').text = 'cmd_vel:=guard_input_cmd_vel'
    description = ET.tostring(base, encoding='unicode')
    urdf_path = output/(namespace+'.urdf'); urdf_path.write_text(description)
    converted = subprocess.run(['gz', 'sdf', '-p', str(urdf_path)], check=True,
                               text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output/(namespace+'_conversion.log')).write_text(converted.stderr)
    sdf = ET.fromstring(converted.stdout)
    # URDF extension <gazebo reference=wheel><slip1> is copied to the link,
    # where the physics engine ignores it. Set the real collision surface.
    set_lateral_compliance(sdf, manifest['model_parameters']['lateral_slip_m_per_ns'])
    if manifest.get('contact_bins'):
        from .scout_bin_hardware import add_carrier
        add_carrier(sdf, namespace, manifest['contact_bins']['axis_library'],
                    integral_gain=manifest['contact_bins'].get('pusher_integral_gain_n_per_m_s', 0.),
                    direction=manifest['contact_bins'].get('transfer_direction', 'lateral'),
                    platform_lift_m=lift)
    sdf_path = output/(namespace+'.sdf')
    ET.ElementTree(sdf).write(sdf_path, encoding='unicode')
    return description, sdf_path
