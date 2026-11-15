"""Back-ring station visible at 55-57 s. Estimated dimensions; static snapshot."""
from .geometry import Mesh

BED_LENGTH=4.7450389
BED_WIDTH=BED_LENGTH*.22

# Fixed 55 s camera, approximate visible corners; all dimensions are estimates.
COLUMN_CENTRE=[130.44853060579896,26.17064515279176]
COLUMN_SIZE=[.5873764203986621,.4939964796231201,1.5288242335840423]
COLUMN_PIXELS=[[495,81],[437,101],[471,127],[528,106],[440,264],[474,290],[529,265]]
COLUMN_CORNER_INDICES=[0,1,2,3,5,6,7]
COVER_CENTRE=[128.26444455739295,27.48349212060477]
COVER_SIZE=[.6495371278338825,.3827708668277896,1.5898009087492557]
COVER_PIXELS=[[796,56],[740,78],[766,94],[822,72],[745,237],[767,266],[830,240]]
COVER_CORNER_INDICES=[0,1,2,3,5,6,7]
FIXTURE_ANCHOR=[129.04469182038457,27.53454415435687,1.25]
ROBOT_CARRIAGE=[131.52340966125155,29.14088156617039,.85]
ROBOT_JOINTS_WORLD=[[131.46327065043184,29.022852401626032,1.16],
    [131.38620507395518,29.216884786973715,1.96],
    [131.48432596767665,29.003503778902463,1.99],
    [131.47667600178545,29.106543469482826,2.27]]


def rings_x(mesh, center, count=10, pitch=.075, radius=.22, material='polished'):
    """Closed hollow ring profiles, with their axes parallel to the storage bed."""
    x,y,z=center
    for index in range(count):
        ring=Mesh('ring')
        ring.lathe_y([(-.025,radius*.70),(-.025,radius),(.025,radius),
                      (.025,radius*.70),(-.025,radius*.70)],(0,0,0),material,'back_ring',32)
        mesh.append([(x+index*pitch+b,y+a,z+c) for a,b,c in ring.vertices],
                    ring.faces,material,'stored_back_rings')


def bed(name, length, width, legs=True):
    m=Mesh(name)
    for y in [-width/2,width/2]:
        m.box((0,y,.78),(length,.09,.16),'blue','edge_frames')
        if legs:
            for x in [-length/2+.2,length/2-.2]:
                m.box((x,y,.38),(.10,.10,.76),'blue','legs')
                m.box((x,y,.035),(.26,.24,.07),'steel','feet')
    count=round((length-.15)/.10)
    for k in range(count+1):
        x=-length/2+.10+k*(length-.20)/count
        m.box((x,0,.785),(.065,width,.075),'blue','cross_slats')
        for j in range(14 if width>1 else 6):
            y=-width/2+.10+j*(width-.20)/(13 if width>1 else 5)
            m.cylinder((x-.040,y,.84),(x+.040,y,.84),.032,'blue','roller_beads',8)
    return m


def storage_bed():
    m=bed('back_ring_storage_bed',BED_LENGTH,BED_WIDTH)
    # Two occupied banks. Counts and pitch are appearance estimates, not a capacity claim.
    for xs,material in [([-2.08,-1.80,-1.52],'polished'),([-.08,.20],'dark_steel')]:
        for x in xs:
            for n in range(21):
                m.lathe_y([(-.012,.075),(-.012,.11),(.012,.11),(.012,.075),(-.012,.075)],
                          (x,-.42+n*.042,.96),material,'stored_back_rings',32)
    return m


def short_table():
    m=bed('back_ring_short_table',.82,.47)
    rings_x(m,(-.31,-.04,.96),12,.035,.11)
    return m


def electrical_cabinet():
    m=Mesh('back_ring_electrical_cabinet')
    m.box((0,0,1.47),(1.60,.90,2.94),'light_gray','cabinet')
    m.box((0,0,2.97),(1.48,.79,.06),'gray','roof_inset')
    m.box((0,.456,1.5),(1.45,.012,2.76),'light_gray','door')
    m.box((0,.466,1.5),(.012,.015,2.76),'gray','door_split')
    for x in [-.08,.08]:m.box((x,.48,1.57),(.025,.025,.19),'steel','door_handles')
    m.box((.48,.48,2.42),(.27,.014,.22),'gray','vent')
    for x in [.24,.34,.44,.54,.64]:
        for z in [1.83,1.94,2.05,2.16]:m.box((x,.48,z),(.05,.014,.025),'gray','vents')
    return m


def measuring_column():
    m=Mesh('back_ring_measuring_column')
    dx,dy,h=COLUMN_SIZE
    m.box((0,0,h/2),(dx,dy,h),'light_gray','plain_white_tower')
    m.box((0,0,.025),(dx+.025,dy+.025,.05),'gray','flush_base')
    return m


def measuring_booth():
    import math
    m=Mesh('back_ring_measuring_booth')
    dx,dy,h=COVER_SIZE;r=.10
    # Solid rounded cover envelope with real top fillets; the unseen interior is
    # deliberately not assigned an invented process mechanism.
    profile=[(-dx/2,0),(dx/2,0),(dx/2,h-r)]
    profile += [(dx/2-r+r*math.cos(t),h-r+r*math.sin(t))
        for t in [math.pi*n/16 for n in range(1,9)]]
    profile += [(-dx/2+r,h)]
    profile += [(-dx/2+r+r*math.cos(t),h-r+r*math.sin(t))
        for t in [math.pi/2+math.pi*n/16 for n in range(1,9)]]
    for y,flip in [(-dy/2,False),(dy/2,True)]:
        vertices=[(x,y,z) for x,z in profile]
        faces=[(0,n+1,n) if flip else (0,n,n+1) for n in range(1,len(vertices)-1)]
        m.append(vertices,faces,'black','rounded_black_cover')
    for a,b in zip(profile,profile[1:]+profile[:1]):
        m.append([(a[0],-dy/2,a[1]),(b[0],-dy/2,b[1]),
            (b[0],dy/2,b[1]),(a[0],dy/2,a[1])],[(0,2,1),(0,3,2)],'black','cover_edges')
    x=FIXTURE_ANCHOR[0]-COVER_CENTRE[0];y=FIXTURE_ANCHOR[1]-COVER_CENTRE[1]
    m.box((.36,.10,.20),(1.40,.75,.40),'light_gray','low_cabinet')
    m.box((.36,.10,.42),(1.44,.79,.055),'black','black_countertop')
    m.box((.36,.495,.20),(1.32,.018,.33),'gray','cabinet_front')
    m.box((.36,.507,.20),(.012,.012,.31),'machine','cabinet_door_split')
    # The fixture stands beside the cover, rather than in its middle.
    m.cylinder((x,y,.45),(x,y,.59),.19,'orange','lower_rotary_chuck',32)
    m.cylinder((x,y,.59),(x,y,.64),.17,'black','chuck_recess',32)
    for sx in [-.12,.12]:
        m.box((x+sx,y-.12,.86),(.045,.055,.50),'steel','measurement_frame_posts')
    m.box((x,y,1.25),(.42,.30,.045),'light_gray','probe_bridge')
    m.cylinder((x,y,1.02),(x,y,1.20),.145,'orange','upper_chuck',32)
    m.cylinder((x,y,.96),(x,y,1.02),.11,'orange_dark','upper_tool_tip',24)
    for sx in [-.095,.095]:
        m.cylinder((x+sx,y-.02,1.27),(x+sx,y-.02,1.47),.025,'black','top_guide_rods',16)
    return m


def travelling_robot():
    m=Mesh('back_ring_travelling_robot')
    m.box((0,0,.045),(.55,.47,.09),'orange','travelling_carriage')
    for x in [-.22,.22]:m.box((x,0,.02),(.08,.56,.04),'orange_dark','carriage_side_slides')
    m.cylinder((0,0,.09),(0,0,.25),.16,'orange_dark','turntable',32)
    joints=[tuple(v-b for v,b in zip(point,ROBOT_CARRIAGE)) for point in ROBOT_JOINTS_WORLD]
    shoulder,elbow,wrist,tip=joints
    m.beam((0,0,.22),shoulder,.19,.20,'orange_dark','shoulder_body')
    m.beam(shoulder,elbow,.085,.12,'orange_dark','upright_arm')
    m.beam(elbow,wrist,.11,.13,'orange','short_forearm')
    for k,(x,y,z) in enumerate(joints[:3]):
        r=.13 if k==0 else .075
        m.cylinder((x-.075,y,z),(x+.075,y,z),r,'orange_dark','joint_housings',24)
        m.cylinder((x+.076,y,z),(x+.086,y,z),r*.75,'orange','joint_caps',24)
    m.box((shoulder[0]-.10,shoulder[1],shoulder[2]+.04),(.12,.20,.27),'orange_dark','rear_motor')
    m.box(wrist,(.20,.14,.09),'orange','tool_flange')
    m.beam(wrist,(tip[0],tip[1],tip[2]-.10),.055,.07,'orange','gripper_stem')
    for dx in [-.065,.065]:
        m.box((tip[0]+dx,tip[1],tip[2]-.045),(.025,.04,.15),'steel','upright_gripper_fingers')
        m.box((tip[0]+dx,tip[1],tip[2]-.13),(.055,.07,.035),'orange','finger_sliders')
    # Static cable segments follow the visible arm; no driven-joint claim.
    points=[(shoulder[0]-.09,shoulder[1]-.09,shoulder[2]-.10),
        (shoulder[0]-.10,shoulder[1]-.12,shoulder[2]+.22),
        (elbow[0]-.08,elbow[1]-.09,elbow[2]),(wrist[0],wrist[1]-.08,wrist[2])]
    for a,b in zip(points,points[1:]):m.cylinder(a,b,.012,'black','arm_cable',12)
    return m


def operator():
    m=Mesh('workshop_operator')
    for x in [-.12,.12]:
        m.box((x,.06,.07),(.18,.34,.14),'black','shoes')
        m.beam((x,0,.15),(x,0,.89),.15,.19,'screen','trousers')
    m.box((0,0,1.15),(.46,.27,.59),'gray','work_jacket')
    for x in [-.28,.28]:
        m.beam((x,0,1.38),(x,.03,.92),.12,.14,'gray','sleeves')
        m.cylinder((x,.03,.90),(x,.03,.82),.055,'light_gray','hands',16)
    m.cylinder((0,0,1.43),(0,0,1.62),.10,'light_gray','head',24)
    m.cylinder((0,0,1.62),(0,0,1.71),.15,'yellow','helmet',24)
    m.box((0,.03,1.63),(.36,.34,.025),'yellow','helmet_brim')
    return m


def make_metrology_assets():
    assets=[storage_bed(),short_table(),electrical_cabinet(),measuring_column(),
            measuring_booth(),travelling_robot(),operator()]
    # Baked geometry estimates. The schematic human is assumed to be 1.7 m tall.
    # These ratios align cabinet/robot height in this view; none is a measurement.
    for mesh,scale in [(assets[2],.62)]:
        mesh.vertices=[tuple(v*scale for v in vertex) for vertex in mesh.vertices]
    return assets
