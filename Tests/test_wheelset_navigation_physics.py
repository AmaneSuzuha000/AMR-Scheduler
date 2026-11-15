import xml.etree.ElementTree as ET

import pytest

from tools.wheelset_scene.workshop_navigation import configure_island_solver


def test_parallel_islands_preserve_every_other_world_and_accuracy_setting():
    world = ET.fromstring('''<world><physics><max_step_size>.002</max_step_size>
      <ode><solver><type>quick</type><iters>400</iters><sor>1</sor></solver>
      <constraints><cfm>1e-09</cfm><erp>0.8</erp></constraints></ode></physics>
      <model name="payload"><link name="body"/></model></world>''')
    baseline = ET.tostring(world)
    configure_island_solver(world, 4)
    solver = world.find('physics/ode/solver'); assert solver.findtext('island_threads') == '4'
    solver.remove(solver.find('island_threads'))
    assert ET.tostring(world) == baseline
    configure_island_solver(world, 0)
    assert ET.tostring(world) == baseline


@pytest.mark.parametrize('threads', [-1, 17, True, 1.5])
def test_invalid_parallelism_is_rejected(threads):
    with pytest.raises(ValueError): configure_island_solver(ET.Element('world'), threads)


def test_parallelism_requires_unchanged_explicit_solver():
    world = ET.fromstring('<world><physics><ode><solver><type>quick</type><iters>50</iters></solver></ode></physics></world>')
    with pytest.raises(ValueError): configure_island_solver(world, 4)
