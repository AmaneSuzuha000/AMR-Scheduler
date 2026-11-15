"""Visible preparation details, estimated from source43/45/48; not metrology."""
WIPING_TRIANGLE_XZ = [(-.38865188,1.50495044),(.65411672,1.40449956),(-.25826257,.68236678)]
WIPING_TRIANGLE_SPAN = 1.33181939
WIPING_TRIANGLE_PIXELS_45 = [[475,321],[554,384],[488,408],[697,214],[762,264],[704,283]]
INFEED_X = 74.50206023
INFEED_POST_Y = [24.77868396-26.54513306,27.91731598-26.54513306]
INFEED_POST_HEIGHT = 1.28732226
INFEED_POST_PIXELS_48 = [[84,145],[129,328],[546,20],[553,179]]
INFEED_POST_PIXELS_43 = [[136,198],[149,229],[194,130],[204,157]]
INFEED_BENCH_END_X = 74.2
INFEED_BENCH_Y = [25.19-26.54513306,27.67-26.54513306]
INFEED_BENCH_HEIGHT = .72
INFEED_BENCH_PIXELS_48 = [[156,190],[493,82]]
FIN_UNIT_CENTER = [75.6597799,30.20081823,0]


def perforated_plate(mesh, triangle, y, thickness, hole_center, radius):
    """An XZ triangle with a true circular opening, extruded in Y."""
    import math
    vertices=list(triangle);center=hole_center
    angles=[2*math.pi*n/32 for n in range(32)]
    angles.extend(math.atan2(v[1]-center[1],v[0]-center[0])%(2*math.pi) for v in vertices)
    angles=sorted(set(angles));outer=[];inner=[]
    def cross(a,b):return a[0]*b[1]-a[1]*b[0]
    for angle in angles:
        ray=(math.cos(angle),math.sin(angle));distances=[]
        for a,b in zip(vertices,vertices[1:]+vertices[:1]):
            edge=(b[0]-a[0],b[1]-a[1]);delta=(a[0]-center[0],a[1]-center[1]);denom=cross(ray,edge)
            if abs(denom)<1e-10:continue
            t=cross(delta,edge)/denom;u=cross(delta,ray)/denom
            if t>0 and -1e-8<=u<=1+1e-8:distances.append(t)
        assert distances and min(distances)>radius
        outer.append(tuple(center[k]+min(distances)*ray[k] for k in range(2)))
        inner.append(tuple(center[k]+radius*ray[k] for k in range(2)))
    for n in range(len(angles)):
        j=(n+1)%len(angles)
        ring=[outer[n],outer[j],inner[j],inner[n]]
        points=[(x,y+sign*thickness/2,z) for sign in [-1,1] for x,z in ring]
        # Separate planar faces from hole/edge walls so exported vertex normals
        # do not round the flat plate into a radial fan under Gazebo lighting.
        for indices,faces in [([0,1,2,3],[(0,2,1),(0,3,2)]),
                ([4,5,6,7],[(0,1,2),(0,2,3)]),
                ([0,1,5,4],[(0,1,2),(0,2,3)]),
                ([3,7,6,2],[(0,1,2),(0,2,3)])]:
            mesh.append([points[k] for k in indices],[(a,c,b) for a,b,c in faces],
                'orange','perforated_wiping_bracket')


def wiping_asset():
    from .geometry import Mesh
    m=Mesh('wiping_station')
    for y in [-.77,.77]:
        for x in [-.25,.25]:
            m.box((x,y,.09),(.18,.44,.18),'machine','roller_support')
            m.cylinder((x,y-.17,.16),(x,y+.17,.16),.13,'polished','rotation_rollers',28)
    for x in [-.68,.75]:m.box((x,0,.10),(.12,3.9,.16),'steel','cross_ties')
    for y in [-.67,.67]:m.box((0,y,.07),(1.9,.10,.12),'gray','longitudinal_bed')
    for side in [-1,1]:
        for y in [side*.99,side*1.92]:
            m.box((-.57,y,.69),(.12,.14,1.38),'machine','frame_posts')
            m.box((-.57,y,.045),(.38,.35,.09),'steel','frame_feet')
        for z in [.31,.65,1.04]:
            m.box((-.57,side*1.455,z),(.09,.94,.07),'gray','frame_crossmembers')
        m.box((-.47,side*1.46,1.48),(.30,1.2,.19),'orange_dark','top_crosshead')
        for x in [-.59,-.36]:m.box((x,side*1.46,1.59),(.035,1.14,.025),'steel','top_guides')
        y=side*WIPING_TRIANGLE_SPAN
        perforated_plate(m,WIPING_TRIANGLE_XZ,y,.035,(-.025,1.335),.07)
        m.beam((-.39,y,1.50),(.65,y,1.40),.075,.12,'orange_dark','bracket_top_rib')
        m.beam((-.26,y,.68),(.65,y,1.40),.075,.12,'orange_dark','bracket_lower_rib')
        m.cylinder((-.50,y,1.43),(-.28,y,1.43),.09,'steel','pivot_pin',24)
        m.box((.62,y,1.32),(.24,.30,.18),'orange','head_carriage')
        m.cylinder((.62,y-side*.19,1.18),(.62,y+side*.20,1.18),.155,'machine','tool_drive',32)
        m.cylinder((.62,y-side*.205,1.18),(.62,y-side*.18,1.18),.13,'black','tool_face',32)
        for z in [.47,.59]:
            m.cylinder((-.50,side*1.04,z),(-.50,side*1.92,z),.025,'polished','lower_guides',16)
        m.box((-.45,side*1.08,.48),(.28,.17,.18),'orange','lower_slide')
    m.box((.02,-1.94,.29),(.47,.39,.50),'machine','low_drive_box')
    m.box((.02,-1.94,.55),(.49,.41,.035),'light_gray','low_drive_lid')
    m.cylinder((.45,0,.25),(1.03,0,.25),.12,'machine','rotation_motor',32)
    # One separate low floor control cabinet is visible at45s, replacing the
    # unrelated tall operator-console asset from the initial scene proposal.
    m.box((2.0,2.30,.35),(.47,.32,.66),'light_gray','floor_control_cabinet')
    m.box((2.0,2.127,.35),(.35,.015,.47),'machine','floor_cabinet_door')
    m.box((2.0,2.114,.35),(.30,.012,.42),'gray','floor_cabinet_panel')
    for x in [1.78,2.22]:m.box((x,2.30,.035),(.11,.42,.07),'gray','floor_cabinet_feet')
    return m


def infeed_asset():
    from .geometry import Mesh
    m=Mesh('preparation_infeed_bench')
    for y in INFEED_POST_Y:
        h=INFEED_POST_HEIGHT
        m.box((0,y,h/2),(.30,.32,h),'machine','boundary_upright')
        m.box((0,y,h+.016),(.32,.34,.032),'light_gray','upright_cap')
        m.box((.157,y,.94),(.015,.12,.15),'gray','upright_panel')
        m.box((.167,y,.95),(.008,.035,.035),'steel','upright_button')
        m.box((0,y,.025),(.40,.42,.05),'orange_dark','upright_foot')
    end=INFEED_BENCH_END_X-INFEED_X;length=3.40
    for y in INFEED_BENCH_Y:
        m.box((end-length/2,y,.69),(length,.25,.06),'blue','narrow_blue_bench')
        for dy in [-.13,.13]:
            m.box((end-length/2,y+dy,.735),(length,.025,.06),'blue','bench_edge_guides')
        for x in [end-.25,end-length+.25]:
            for dy in [-.1,.1]:m.box((x,y+dy,.32),(.05,.05,.64),'blue','bench_legs')
        for x in [end-2.1,end-2.45]:
            m.lathe_y([(-.08,.11),(-.08,.20),(.08,.20),(.08,.11),(-.08,.11)],
                (x,y,.95),'polished','visible_ring_stock',32)
    m.box((-1.70,0,.012),(4.4,2.4,.024),'gray','flush_infeed_plate')
    for y in [-.77,.77]:
        m.box((-1.70,y,.06),(4.4,.10,.08),'dark_steel','infeed_rails')
        m.box((-1.70,y,.105),(4.4,.035,.025),'steel','rail_heads')
        for x in [-3.65,-2.9,-2.15,-1.4,-.65,.05]:
            for dx in [-.10,.10]:
                m.cylinder((x+dx,y-.11,.12),(x+dx,y+.11,.12),.07,'polished','indexing_rollers',16)
                m.box((x+dx,y-.15,.09),(.13,.08,.10),'machine','roller_bearing_blocks')
    return m


def fin_unit_asset():
    from .geometry import Mesh
    m=Mesh('preparation_fin_unit')
    m.box((0,0,.43),(.90,1.1,.86),'gray','main_body')
    m.box((0,0,.87),(.96,1.16,.06),'light_gray','body_roof')
    m.box((0,-.60,.49),(.72,.12,.55),'gray','end_cabinet')
    m.box((0,-.67,.49),(.60,.03,.42),'light_gray','end_panel')
    for n in range(13):
        y=-.48+n*.08
        m.cylinder((.54,y,.34),(.54,y,1.18),.017,'polished','vertical_cooling_tubes',12)
        m.beam((.54,y,1.18),(.38,y,1.29),.035,.035,'polished','upper_tube_bends')
        m.cylinder((.38,y,1.29),(-.18,y,1.29),.017,'polished','top_tube_runs',12)
    m.cylinder((.05,.10,.92),(.05,.10,1.55),.19,'gray','top_cylindrical_vessel',32)
    m.cylinder((.05,.10,1.55),(.05,.10,1.59),.20,'light_gray','vessel_cap',32)
    for y in [-.32,.32]:m.box((.15,y,1.02),(.30,.08,.22),'gray','top_fittings')
    return m
