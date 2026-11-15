"""Derived world of independent driven wheelset fixtures on a shared lane."""
import copy
from .actuated_world import build_actuated_world
from .build import e


def build_multi_world(jobs):
    tree,removed,assumptions=build_actuated_world();world=tree.find('world')
    original=world.find("model[@name='actuated_fixture']");world.remove(original)
    origins={job:-4.*(n+1) for n,job in enumerate(jobs)}
    for job,x in origins.items():
        model=copy.deepcopy(original);model.set('name','fixture_'+job);model.find('pose').text=f'{x} 50 0 0 0 0'
        for joint in model.findall('joint'):
            joint.set('name',joint.get('name').replace('wsd_',job+'_'))
        model.find(f"joint[@name='{job}_slide']/axis/limit/upper").text='80'
        plugin=model.find("plugin[@name='actuated_joint_feedback']")
        plugin.find('ros/namespace').text='/'+job
        for name in plugin.findall('joint_name'):name.text=name.text.replace('wsd_',job+'_')
        # Distinct colours distinguish the moving fixtures in the actual camera.
        colour=['.1 .6 .9 1','.9 .3 .1 1','.2 .8 .3 1','.7 .3 .9 1'][list(origins).index(job)%4]
        material=model.find("link[@name='carriage']/visual/material")
        material.find('ambient').text=colour;material.find('diffuse').text=colour
        world.append(model)
    assumptions.update(fixture_count=len(jobs),fixture_world_x_origins=origins,
        admission='Driven motion from explicit upstream staging; slot admitted after measured arrival and stop',
        unloading='Delete stopped output fixture after simulated unload acknowledgement; external unload mechanism unmodelled',
        lane_safety='Logical one-slot places and exclusive transfer; measured X envelopes checked; contact excluded')
    return tree,removed,assumptions,origins
