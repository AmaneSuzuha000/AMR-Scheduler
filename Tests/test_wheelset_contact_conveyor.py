"""Guard the mechanical distinction between free payload and driven conveyor."""
import xml.etree.ElementTree as ET
from array import array
import json

from tools.wheelset_scene.contact_conveyor_world import build
from tools.wheelset_scene.contact_chain_conveyor_world import build as build_chain
from tools.wheelset_scene.run_contact_conveyor import json_sequence
from tools.wheelset_scene.verify_contact_conveyor import world_vertical_force


def test_raw_ros_numeric_sequences_retain_values():
    sample = dict(position=array('d', [0., .125, -2.5]), velocity=array('d', []))
    assert json.loads(json.dumps(sample, default=json_sequence)) == dict(position=[0., .125, -2.5], velocity=[])


def test_body_wrench_rotation_preserves_upward_support_after_wheel_turn():
    force = dict(x=0.,y=0.,z=-4905.)
    assert world_vertical_force(force,dict(x=0.,y=1.,z=0.,w=0.)) == 4905.


def test_chain_force_is_applied_to_dogs_and_payload_is_free(tmp_path):
    library = tmp_path/'native.so'; library.write_bytes(b'unit-fixture-not-an-executable-library')
    manifest = build_chain(tmp_path/'world', library)
    world = ET.parse(tmp_path/'world/conveyor.world').getroot().find('world')
    payload = world.find("model[@name='contact_wheelset']")
    assert payload.findtext('link/gravity') == 'true'
    assert not payload.findall('joint') and not payload.findall('plugin')
    conveyor = world.find("model[@name='powered_conveyor']")
    assert manifest['actuator_joints'] == ['chain_straight_stroke']
    assert conveyor.findtext("joint[@type='prismatic']/child") == 'chain_dogs'
    assert conveyor.findtext('plugin/axis/mode') == 'position'
    assert {c.get('name') for c in conveyor.findall("link[@name='base']/collision")}.issuperset(
        {'support_rail_-1','support_rail_1','station_stop_-1','station_stop_1'})
    assert len(conveyor.findall("link[@name='chain_dogs']/collision")) == 4


def test_payload_has_gravity_and_contacts_and_only_conveyor_joints_are_driven(tmp_path):
    library = tmp_path/'native.so'; library.write_bytes(b'unit-fixture-not-an-executable-library')
    manifest = build(tmp_path/'world', library)
    world = ET.parse(tmp_path/'world/conveyor.world').getroot().find('world')
    payload = world.find("model[@name='contact_wheelset']")
    assert payload.findtext('link/gravity') == 'true'
    assert not payload.findall('joint') and not payload.findall('plugin')
    assert len(payload.findall('link/collision')) == 3
    assert len(payload.findall("link/sensor[@type='contact']")) == 2
    conveyor = world.find("model[@name='powered_conveyor']")
    assert len(manifest['roller_joints']) == 36
    assert {j.get('name') for j in conveyor.findall("joint[@type='revolute']")} == set(manifest['roller_joints'])
    axes = conveyor.findall('plugin/axis')
    assert {a.findtext('joint') for a in axes} == set(manifest['roller_joints'])
    assert all(a.findtext('mode') == 'velocity' for a in axes)
    assert not world.findall("model[@name='contact_pusher']")
