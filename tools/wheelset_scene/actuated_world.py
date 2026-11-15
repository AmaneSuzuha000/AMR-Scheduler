"""Build a derived inspection fixture; never rewrite the frozen scene export."""
import copy
import json
from .build import e,make_world,SCENE


def inertia(link,mass,diagonal):
    inertial=e(link,'inertial');e(inertial,'mass',mass);tensor=e(inertial,'inertia')
    for key,value in zip(['ixx','iyy','izz'],diagonal):e(tensor,key,value)
    for key in ['ixy','ixz','iyz']:e(tensor,key,0)
    e(link,'gravity','false')


def box_visual(link,name,size,color,pose='0 0 0 0 0 0'):
    visual=e(link,'visual',name=name);e(visual,'pose',pose)
    e(e(e(visual,'geometry'),'box'),'size',size)
    material=e(visual,'material');e(material,'ambient',color);e(material,'diffuse',color)


def joint(model,name,kind,parent,child,xyz,limits=None,damping=0):
    item=e(model,'joint',name=name,type=kind);e(item,'parent',parent);e(item,'child',child)
    if kind=='fixed':return item
    axis=e(item,'axis');e(axis,'xyz',xyz)
    if limits:
        limit=e(axis,'limit');e(limit,'lower',limits[0]);e(limit,'upper',limits[1])
        e(limit,'effort',limits[2]);e(limit,'velocity',limits[3])
    dynamics=e(axis,'dynamics');e(dynamics,'damping',damping);e(dynamics,'friction',0)
    return item


def build_actuated_world():
    scene=json.loads(SCENE.read_text(encoding='utf-8'));derived=copy.deepcopy(scene)
    removed=[i['id'] for i in derived['instances'] if i['asset']=='wheelset' and abs(i['pose'][1]-50)<1e-6]
    derived['instances']=[i for i in derived['instances'] if i['id'] not in removed]
    camera=dict(id='actuated_overview',position=[31.5,20,23],target=[31.5,50,.56],
        horizontal_fov=1.15,width=1280,height=720,reference_second=None)
    tree=make_world(derived,camera);world=tree.find('world')
    world.find('physics/max_step_size').text='0.002'
    world.find('physics/real_time_update_rate').text='500'
    # Improve convergence of the heavy, world-anchored chain without relaxing
    # the link-position checks. The default iteration budget allowed drift;
    # a direct solve stalled near a quarter turn in the recorded trial.
    ode=e(world.find('physics'),'ode');solver=e(ode,'solver');e(solver,'type','quick')
    e(solver,'iters',400);e(solver,'sor',1.0)
    constraints=e(ode,'constraints');e(constraints,'cfm',1e-9);e(constraints,'erp',.8)
    world.find("model[@name='review_camera']/link/sensor/update_rate").text='5'
    model=e(world,'model',name='actuated_fixture');e(model,'pose','10 50 0 0 0 0');e(model,'static','false')
    base=e(model,'link',name='base');inertia(base,10,[1,1,1])
    joint(model,'wsd_anchor','fixed','world','base',None)
    carriage=e(model,'link',name='carriage');inertia(carriage,20,[4,4,4])
    box_visual(carriage,'fixture_bed','1.2 2.5 .08','.04 .25 .40 1','0 0 .06 0 0 0')
    joint(model,'wsd_slide','prismatic','base','carriage','1 0 0',[-3,48,7000,2.5],50)
    axle=e(model,'link',name='wheelset');e(axle,'pose','0 0 .56 0 0 0')
    inertia(axle,1000,[400,160,400])
    visual=e(axle,'visual',name='wheelset_mesh');e(visual,'pose','0 0 -.56 0 0 0')
    e(e(e(visual,'geometry'),'mesh'),'uri','model://wsv1_wheelset/meshes/wheelset.dae')
    box_visual(axle,'rotation_reference',' .065 .025 .26','1 .35 .03 1','0 -.90 .24 0 0 0')
    joint(model,'wsd_spin','revolute','carriage','wheelset','0 1 0',[-1e16,1e16,1000,2.5],3)
    probe=e(model,'link',name='probe');e(probe,'pose','0 -1.57 .56 0 0 0');inertia(probe,5,[.1,.1,.1])
    box_visual(probe,'assumed_probe',' .16 .30 .18','1 .3 .02 1')
    joint(model,'wsd_probe','prismatic','carriage','probe','0 1 0',[0,.25,200,.5],4)
    plugin=e(model,'plugin',name='actuated_joint_feedback',filename='libgazebo_ros_joint_state_publisher.so')
    e(e(plugin,'ros'),'namespace','/wheelset_actuation');e(plugin,'update_rate',40)
    for name in ['wsd_slide','wsd_probe','wsd_spin']:e(plugin,'joint_name',name)
    assumptions=dict(fixture_design='Assumed linear fixture and probe, not a recovered manufacturer mechanism',
        wheelset_mass_kg=1000,carriage_mass_kg=20,probe_mass_kg=5,
        wheelset_inertia_kg_m2=[400,160,400],gravity_enabled=False,contact_geometry=False,
        max_slide_effort_n=7000,max_probe_effort_n=200,max_spin_torque_nm=1000,
        completion='Measured one-turn rotation, measured stopped spindle and home probe before release',
        axis_constraints='World-anchored carriage with X prismatic, wheelset Y revolute, probe Y prismatic',
        constraint_solver='ODE quick solver, 400 iterations, SOR=1, 0.002 s step, CFM=1e-9, ERP=0.8')
    return tree,removed,assumptions
