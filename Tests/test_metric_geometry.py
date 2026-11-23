"""Physical surrogate geometry and scene identity must agree before Gazebo runs."""
import importlib.util
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import pytest

from tools.live_bridge.metric_world import build_scene, calibration_world

ROOT = Path(__file__).resolve().parents[1]


def test_physical_point_network_blocks_second_owner_without_changing_algorithm_scene():
    from amr_sched.reservation import Interval, ReservationTable
    from amr_sched.workshop import WorkshopScene
    from tools.live_bridge.fleet_config import point_execution_network
    scene = ROOT/'scenes/rail_freight_workshop_v1.json'
    before = scene.read_bytes()
    abstract = WorkshopScene.load(scene).network()
    physical = point_execution_network(abstract)
    assert set(physical.nodes) == set(abstract.nodes)
    assert set(physical.edges) == set(abstract.edges)
    for key, node in abstract.nodes.items():
        other = physical.node(key)
        assert (other.node_id, other.x, other.y, other.kind) == (node.node_id, node.x, node.y, node.kind)
        assert other.capacity == 1
    assert physical.edges == abstract.edges  # canonical roads already have one slot
    junction = 'J_N_65'
    assert abstract.node(junction).capacity == 2
    resources = ReservationTable()
    resources.transfer_pins('R2', [junction])
    for step in (0, 1_000_000):
        span = Interval(step, step+1)
        assert resources.is_node_free(junction, span, abstract.node(junction).capacity, 'R1')
        assert not resources.is_node_free(junction, span, physical.node(junction).capacity, 'R1')
    resources.transfer_pins('R2', ['J_N_85'])
    assert resources.is_node_free(junction, Interval(1_000_000, 1_000_001), 1, 'R1')
    assert scene.read_bytes() == before and abstract.node(junction).capacity == 2


def test_native_fleet_rejects_abstract_multiple_slots_before_opening_output(tmp_path):
    import ast
    from types import SimpleNamespace
    from amr_sched.network import Edge, Network, Node
    tree = ast.parse((ROOT/'tools/live_bridge/fleet_scheduler.py').read_text(encoding='utf-8'))
    owner_class = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'FleetRuntime')
    scope = {}
    exec(compile(ast.Module(body=[owner_class], type_ignores=[]), 'fleet_scheduler.py', 'exec'), scope)
    for node_capacity, edge_capacity in ((2, 1), (1, 2)):
        network = Network()
        network.add_node(Node('A', 0., 0., capacity=node_capacity))
        network.add_node(Node('B', 1., 0.))
        network.add_edge(Edge('A', 'B', 1., capacity=edge_capacity))
        sim = SimpleNamespace(execution=None, executions={'R1': object(), 'R2': object()}, network=network)
        output = tmp_path/f'capacity-{node_capacity}-{edge_capacity}'
        with pytest.raises(ValueError, match='single-point native fleet execution requires capacity 1'):
            scope['FleetRuntime'](sim, output, {'R1': 'r1', 'R2': 'r2'}, None)
        assert not output.exists()


def test_fleet_configs_isolate_sensor_topics_without_changing_navigation_or_health_tuning():
    import yaml
    from tools.live_bridge.fleet_config import robot_configs
    localization=yaml.safe_load((ROOT/'tools/live_bridge/sensor_localization.yaml').read_text())
    controller=yaml.safe_load((ROOT/'tools/live_bridge/nav2_metric.yaml').read_text())
    loc,nav=robot_configs('r2',localization,controller)
    assert loc['r2']['ekf_filter_node']['ros__parameters']['odom0']=='/r2/wheel/odometry'
    assert loc['r2']['ekf_filter_node']['ros__parameters']['imu0']=='/r2/imu/data'
    assert loc['r2']['amcl']['ros__parameters']['scan_topic']=='/r2/scan'
    assert nav['r2']['controller_server']['ros__parameters']['odom_topic']=='/r2/odometry/filtered'
    costmap=nav['r2']['local_costmap']['local_costmap']['ros__parameters']
    assert costmap['obstacle_layer']['scan']['topic']=='/r2/scan'
    assert costmap['obstacle_layer']['scan']['expected_update_rate']==.3
    assert nav['r2']['controller_server']['ros__parameters']['general_goal_checker']['xy_goal_tolerance']==.05
    assert localization['amcl']['ros__parameters']['scan_topic']=='/scan'
    assert controller['controller_server']['ros__parameters']['odom_topic']=='/odometry/filtered'
    with pytest.raises(ValueError):
        robot_configs('../other',localization,controller)


def test_a_declared_station_tolerance_adds_a_tighter_goal_checker_only_then():
    """A station hop must converge inside the handoff gate, a junction hop need not."""
    import yaml
    from tools.live_bridge.fleet_config import robot_configs
    localization=yaml.safe_load((ROOT/'tools/live_bridge/sensor_localization.yaml').read_text())
    controller=yaml.safe_load((ROOT/'tools/live_bridge/nav2_metric.yaml').read_text())
    _,plain=robot_configs('r2',localization,controller)
    parameters=plain['r2']['controller_server']['ros__parameters']
    assert parameters['goal_checker_plugins']==['general_goal_checker']
    assert 'dock_goal_checker' not in parameters
    assert parameters['FollowPath']['rotate_to_heading_min_angle']==.5
    _,nav=robot_configs('r2',localization,controller,
                        dock_tolerance_m=.015,dock_yaw_tolerance_rad=.04)
    parameters=nav['r2']['controller_server']['ros__parameters']
    assert parameters['goal_checker_plugins']==['general_goal_checker','dock_goal_checker']
    assert parameters['dock_goal_checker']==dict(plugin='nav2_controller::SimpleGoalChecker',
        stateful=True,xy_goal_tolerance=.015,yaw_goal_tolerance=.04)
    assert parameters['general_goal_checker']['xy_goal_tolerance']==.05
    # The controller only rotates outside its band, so the band has to sit
    # inside the docking tolerance or a station hop can never be accepted.
    band=parameters['FollowPath']['rotate_to_heading_min_angle']
    assert 0<band<parameters['dock_goal_checker']['yaw_goal_tolerance']<.05
    assert parameters['dock_goal_checker']['xy_goal_tolerance']<.025
    # A docking checker may never widen the gate it is meant to tighten.
    for xy,yaw in ((.05,.04),(.06,.04),(.015,.25),(.0,.04),(.015,.0)):
        with pytest.raises(ValueError):
            robot_configs('r2',localization,controller,
                          dock_tolerance_m=xy,dock_yaw_tolerance_rad=yaw)

spec = importlib.util.spec_from_file_location("metric_model", ROOT / "tools/ros2/amr_gazebo_guard/launch/metric_model.py")
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)


def original():
    # Minimal expanded original for portable tests; no installed ROS dependency.
    robot = ET.Element("robot", name="scout_mini")
    base = ET.SubElement(robot, "link", name="base_link")
    ET.SubElement(ET.SubElement(base, "visual"), "origin", xyz="0 0 0", rpy="1.57 0 -1.57")
    ET.SubElement(ET.SubElement(base, "inertial"), "origin", xyz="0 0 0")
    for prefix in ("front_left", "front_right", "rear_left", "rear_right"):
        ET.SubElement(ET.SubElement(robot, "joint", name=prefix+"_wheel"), "origin", rpy="-1.57 0 0")
        surface = ET.SubElement(robot, "gazebo", reference=prefix+"_wheel_link")
        for key, value in (("mu1", "50"), ("mu2", "50"), ("fdir1", "1 0 0"), ("kp","1000000"), ("kd","100")):
            ET.SubElement(surface, key).text = value
    ET.SubElement(ET.SubElement(robot, "joint", name="laser_joint"), "origin", xyz=".15 0 .18")
    plugin = ET.SubElement(ET.SubElement(robot, "gazebo"), "plugin", filename="libgazebo_ros_diff_drive.so")
    ET.SubElement(plugin, "publish_odom_tf").text = "false"
    return robot


def test_chassis_box_is_above_ground_and_footprint_contains_wheel_envelope():
    robot = model.augment(original())
    center = tuple(map(float, robot.find("link[@name='base_link']/collision/origin").attrib["xyz"].split()))
    size = tuple(map(float, robot.find("link[@name='base_link']/collision/geometry/box").attrib["size"].split()))
    assert model.WHEEL_RADIUS + center[2] - size[2]/2 > .075
    assert center[0] - size[0]/2 >= -.315 and center[0]+size[0]/2 <= .315
    assert model.WHEELBASE/2 + model.WHEEL_RADIUS < .315
    assert model.TRACK/2 + model.WHEEL_WIDTH/2 < .252
    assert .08 + .32 > .08 + center[2] + size[2]/2


def test_optional_tracking_frame_does_not_change_mass_sensors_contacts_or_collisions():
    robot=model.augment(original())
    before=ET.tostring(robot)
    count=len(robot)
    model.add_tracking_reference(robot)
    joint=robot.find("joint[@name='front_tracking_joint']")
    assert joint.find('parent').get('link')=='base_footprint'
    assert joint.find('child').get('link')=='front_tracking'
    assert tuple(map(float,joint.find('origin').get('xyz').split()))==(model.WHEELBASE/2,0.,0.)
    assert len(robot.find("link[@name='front_tracking']"))==0
    with pytest.raises(ValueError,match='already exists'):
        model.add_tracking_reference(robot)
    for child in list(robot)[count:]:
        robot.remove(child)
    assert ET.tostring(robot)==before


def test_encoder_odometry_imu_and_valid_friction_basis_are_explicit():
    robot = model.augment(original())
    assert robot.find(".//plugin[@filename='libgazebo_ros_diff_drive.so']/odometry_source").text == "0"
    assert robot.find(".//sensor[@type='imu']/plugin").attrib["filename"] == "libgazebo_ros_imu_sensor.so"
    assert robot.find(".//sensor[@type='contact']") is not None
    for surface in robot.findall("gazebo"):
        if surface.find("fdir1") is not None:
            assert surface.find("fdir1").text == "0 0 0"
            assert surface.find("mu1").text == surface.find("mu2").text == "50.0"
    for joint in robot.findall("joint"):
        if joint.attrib["name"].endswith("_wheel"):
            assert math.isclose(float(joint.find("origin").attrib["rpy"].split()[0]), -math.pi/2)


@pytest.mark.parametrize("mu", [0, -1, math.nan, math.inf])
def test_invalid_contact_parameters_are_rejected(mu):
    with pytest.raises(ValueError):
        model.augment(original(), wheel_mu=mu)


def test_metric_world_preserves_scene_identity_units_and_real_obstacles():
    scene = json.loads((ROOT / "scenes/rail_freight_workshop_v1.json").read_text(encoding="utf-8"))
    root, manifest = build_scene(scene)
    assert manifest["scene_id"] == scene["scene_id"] and manifest["scale"] == 1
    assert len(manifest["obstacles"]) == 28
    solid = root.findall(".//collision")
    assert len(solid) == len(manifest["obstacles"])
    assert all(m.find(".//collision") is None for m in root.findall(".//model") if m.attrib["name"].startswith("paint_"))
    assert float(root.find("world/model[@name='boundary_e']/pose").text.split()[0]) > 140


def test_equipment_boxes_leave_a_clear_metric_swept_corridor_for_every_canonical_lane():
    scene = json.loads((ROOT / "scenes/rail_freight_workshop_v1.json").read_text(encoding="utf-8"))
    _, manifest = build_scene(scene)
    radius = math.hypot(.315, .252) + .3
    # Rectangle-to-polyline distance, sampling <=5cm. This checks conservative
    # swept discs at all headings against every solid equipment/boundary box.
    for lane in scene["lanes"]:
        for a,b in zip(lane["centerline"], lane["centerline"][1:]):
            n = math.ceil(math.dist(a,b)/.05)
            for index in range(n+1):
                x,y = [a[k]+(b[k]-a[k])*index/n for k in (0,1)]
                for obstacle in manifest["obstacles"]:
                    center,size = obstacle["center"], obstacle["size"]
                    dx = max(abs(x-center[0])-size[0]/2, 0)
                    dy = max(abs(y-center[1])-size[1]/2, 0)
                    assert math.hypot(dx,dy) >= radius, (lane["id"], obstacle["id"], (x,y))


def test_sensor_wall_uses_actual_collision_at_three_metres():
    root = calibration_world()
    wall = root.find("world/model[@name='front_wall']")
    x = float(wall.find("pose").text.split()[0])
    width = float(wall.find("link/collision/geometry/box/size").text.split()[0])
    assert x-width/2 == 3


def test_optional_long_range_lidar_preserves_archived_view_and_noise():
    archived = ROOT / 'tools/ros2/wheelset_dependencies/scout_gazebo_description/urdf/scout_mini_expanded.urdf'
    if not archived.exists():
        pytest.skip("Optional archived URDF model not present in checkout")
    default = model.augment(ET.parse(archived).getroot())
    wide = model.augment(ET.parse(archived).getroot(), laser_range_max=30.)
    sensor_path = ".//sensor[@type='ray']/ray"
    original_ray, wide_ray = default.find(sensor_path), wide.find(sensor_path)
    assert float(original_ray.findtext('scan/horizontal/max_angle')) == pytest.approx(2.09439504)
    assert float(original_ray.findtext('range/max')) == 8.
    assert float(wide_ray.findtext('range/max')) == 30.
    for key in ('scan', 'noise'):
        assert ET.tostring(original_ray.find(key)) == ET.tostring(wide_ray.find(key))
    assert original_ray.findtext('range/min') == wide_ray.findtext('range/min')


def test_optional_full_laser_view_preserves_sampling_noise_and_all_physical_geometry():
    archived=ROOT/'tools/ros2/wheelset_dependencies/scout_gazebo_description/urdf/scout_mini_expanded.urdf'
    if not archived.exists():
        pytest.skip("Optional archived URDF model not present in checkout")
    baseline=model.augment(ET.parse(archived).getroot())
    candidate=model.augment(ET.parse(archived).getroot(),laser_fov_radians=2*math.pi)
    path=".//sensor[@type='ray']/ray/scan/horizontal"
    old,new=baseline.find(path),candidate.find(path)
    spacing=lambda h:(float(h.findtext('max_angle'))-float(h.findtext('min_angle')))/(int(h.findtext('samples'))-1)
    assert float(new.findtext('min_angle'))==-math.pi and float(new.findtext('max_angle'))==math.pi
    assert spacing(new)<=spacing(old)
    # Replacing only the horizontal scan restores the whole original model,
    # including collisions, mass, sensor noise/range/extrinsics and contacts.
    scan=candidate.find(".//sensor[@type='ray']/ray/scan")
    scan.remove(new)
    scan.append(old)
    assert ET.tostring(candidate)==ET.tostring(baseline)
    for invalid in (0.,-1.,math.nan,math.inf,2*math.pi+.001):
        with pytest.raises(ValueError,match='field of view'):
            model.augment(ET.parse(archived).getroot(),laser_fov_radians=invalid)


@pytest.mark.parametrize('maximum', [0., .01, math.inf, math.nan])
def test_invalid_laser_range_is_rejected(maximum):
    archived = ROOT / 'tools/ros2/wheelset_dependencies/scout_gazebo_description/urdf/scout_mini_expanded.urdf'
    if not archived.exists():
        pytest.skip("Optional archived URDF model not present in checkout")
    with pytest.raises(ValueError, match='maximum range'):
        model.augment(ET.parse(archived).getroot(), laser_range_max=maximum)


def test_two_robot_plugins_isolate_guard_sensor_and_tf_names_without_changing_geometry():
    archived = ROOT / 'tools/ros2/wheelset_dependencies/scout_gazebo_description/urdf/scout_mini_expanded.urdf'
    if not archived.exists():
        pytest.skip("Optional archived URDF model not present in checkout")
    robots=[]
    for name in ('agv_01','agv_02'):
        robot=model.augment(ET.parse(archived).getroot(),laser_range_max=30.)
        guard=ET.SubElement(ET.SubElement(robot,'gazebo'),'plugin',name='guard',filename='libamr_motion_guard.so')
        assert guard is not None
        model.namespace_robot(robot,name)
        for plugin in robot.findall('.//plugin'):
            assert plugin.findtext('ros/namespace')=='/'+name
            assert '/tf:=/'+name+'/tf' in [r.text for r in plugin.findall('ros/remapping')]
            assert '/tf_static:=/'+name+'/tf_static' in [r.text for r in plugin.findall('ros/remapping')]
        robots.append(robot)
    for path in ("link[@name='base_link']/collision",'joint',".//sensor[@type='ray']/ray"):
        assert [ET.tostring(e) for e in robots[0].findall(path)]==[ET.tostring(e) for e in robots[1].findall(path)]


def test_skid_friction_direction_is_the_spin_invariant_axle():
    robot = model.augment(original(),wheel_mu=1.,lateral_mu=.2,contact_scale=10.)
    for prefix in ("front_left","front_right","rear_left","rear_right"):
        surface = robot.find(f"gazebo[@reference='{prefix}_wheel_link']")
        assert surface.find("fdir1").text == "0 0 1"
        assert float(surface.find("mu1").text) == .2
        assert float(surface.find("mu2").text) == 1.
        assert float(surface.find("kp").text) == 1e7
        assert float(surface.find("kd").text) == 1000.
        # Rotation around the cylinder's local z cannot rotate its z axis.
        for angle in (0.,.7,2.3):
            dx,dy,dz = map(float,surface.find("fdir1").text.split())
            sx,sy = dx*math.cos(angle)-dy*math.sin(angle),dx*math.sin(angle)+dy*math.cos(angle)
            roll = float(robot.find(f"joint[@name='{prefix}_wheel']/origin").attrib['rpy'].split()[0])
            world_axis = (sx,math.cos(roll)*sy-math.sin(roll)*dz,math.sin(roll)*sy+math.cos(roll)*dz)
            assert abs(world_axis[2]) < 1e-12  # tangent to the ground
            assert abs(world_axis[0]) < 1e-12  # lateral, not rolling direction
            assert world_axis[1] == pytest.approx(1.)
