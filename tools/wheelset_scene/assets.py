"""Reusable visual assets reconstructed from video appearances, not OEM CAD."""
from __future__ import annotations
import math
from .geometry import Mesh

INSPECTION_SHELL_WIDTH=3.9  # visual estimate shared by both inspection viewpoints


def wheelset():
    m=Mesh("wheelset")
    m.lathe_y([(-1.14,0),(-1.14,.075),(-.98,.075),(-.92,.11),(-.62,.11),
               (-.5,.085),(.5,.085),(.62,.11),(.92,.11),(.98,.075),(1.14,.075),(1.14,0)],
              (0,0,.56),"polished","axle")
    profile=[(-.09,.10),(-.09,.20),(-.065,.24),(-.052,.34),(-.07,.46),(-.045,.49),
             (-.022,.49),(-.012,.46),(.085,.45),(.10,.42),(.09,.35),(.05,.28),(.045,.18),(.08,.13),(.08,.10)]
    for side in [-1,1]:
        p=[(side*y+side*.77,r) for y,r in profile]
        if side<0:p.reverse()
        m.lathe_y(p,(0,0,.56),"wheel_steel",f"wheel_{side}")
        m.lathe_y([(side*.69,.43),(side*.72,.465),(side*.81,.46),(side*.84,.435)] if side>0 else
                  [(-.84,.435),(-.81,.46),(-.72,.465),(-.69,.43)],(0,0,.56),"polished",f"tread_{side}")
        m.cylinder((0,side*.95,.56),(0,side*1.11,.56),.14,"dark_steel",f"bearing_{side}")
        m.torus_y((0,side*1.115,.56),.108,.018,"polished",f"endcap_{side}")
    return m


def robot():
    m=Mesh("industrial_arm")
    m.box((0,0,.055),(.68,.64,.11),"orange","base")
    for x in [-.25,.25]:
        for y in [-.24,.24]:m.cylinder((x,y,.10),(x,y,.145),.035,"dark_steel","base_bolts",12)
    m.cylinder((0,0,.11),(0,0,.32),.25,"orange","base")
    m.cylinder((0,0,.32),(0,0,.43),.23,"orange_dark","axis_1")
    points=[(0,0,.51),(.12,0,1.27),(.82,0,1.65),(1.28,0,1.34),(1.47,0,1.22)]
    for i,(a,b) in enumerate(zip(points,points[1:])):
        m.beam(a,b,.26 if i<2 else .17,.28 if i<2 else .18,"orange",f"link_{i+1}")
    for i,(x,y,z) in enumerate(points):
        r=.20 if i<2 else .14
        m.cylinder((x,-.20,z),(x,.20,z),r,"orange",f"joint_{i+2}")
        m.cylinder((x,-.225,z),(x,-.205,z),r*.69,"orange_dark",f"joint_cap_{i+2}")
    m.cylinder((1.45,0,1.21),(1.60,0,1.08),.09,"black","wrist")
    m.box((1.61,0,1.03),(.14,.30,.10),"steel","gripper")
    for y in [-.13,.13]:m.box((1.63,y,.91),(.07,.055,.24),"orange","gripper_fingers")
    for a,b in zip([(0,.19,.55),(.12,.19,1.29),(.82,.19,1.70),(1.26,.16,1.4)],
                   [(.12,.19,1.29),(.82,.19,1.70),(1.26,.16,1.4),(1.43,.13,1.2)]):
        m.cylinder(a,b,.025,"black","cable",12)
    return m


def overhead_robot():
    """4 s labelling-arm pose with gearbox, counterbalance and extended probe."""
    from .front_geometry import IDENTIFICATION_ROBOT_JOINTS
    m=Mesh("overhead_robot")
    m.box((0,0,.06),(.88,.72,.12),"orange","carriage_base")
    m.cylinder((0,0,.12),(0,0,.32),.25,"orange","axis_1")
    # Source 3/4 s carriages and arms move; retain one explicit 4 s pose.
    # Round gear housings and the broad shoulder distinguish the arm from a
    # chain of constant-width bars. Joint depths are visual assumptions.
    for y in [-.20,.20]:
        m.box((.12,y,.30),(.68,.19,.42),'orange','base_gearbox')
        m.cylinder((.12,y-.08,.34),(.12,y+.08,.34),.18,'orange_dark','base_gear_cover',32)
    m.cylinder((.50,0,.20),(.50,0,.49),.23,'orange','shoulder_pedestal',32)
    points=IDENTIFICATION_ROBOT_JOINTS
    for i,(a,b) in enumerate(zip(points[:3],points[1:4])):
        m.beam(a,b,.34 if i<2 else .20,.36 if i<2 else .24,"orange",f"link_{i+1}")
    m.beam((.05,-.25,.30),(.75,-.25,.38),.12,.13,'orange_dark','counterbalance_rod')
    m.cylinder((.03,0,.38),(.03,0,.67),.12,'orange_dark','rear_motor',24)
    for i,(x,y,z) in enumerate(points[:4]):
        radius=.23 if i<3 else .15
        m.cylinder((x,y-.21,z),(x,y+.21,z),radius,"orange",f"joint_{i+1}")
        m.cylinder((x,y-.235,z),(x,y-.21,z),radius*.70,"orange_dark",f"joint_cover_{i+1}")
    # The source's final landmark is a thin probe: stop the broad orange
    # housing above it, then model the fork and probe as separate sections.
    def tool_point(t):return tuple(a+(b-a)*t for a,b in zip(points[3],points[4]))
    wrist=tool_point(.43)
    m.beam(points[3],wrist,.20,.24,'orange','wrist_housing')
    m.cylinder((wrist[0],wrist[1]-.13,wrist[2]),
               (wrist[0],wrist[1]+.13,wrist[2]),.11,'orange_dark','wrist_flange',24)
    for side in [-1,1]:
        a,b=tool_point(.43),tool_point(.78)
        m.beam((a[0]+side*.075,a[1],a[2]),(b[0]+side*.075,b[1],b[2]),
               .045,.045,'orange','tool_fingers')
    m.cylinder(tool_point(.76),points[4],.014,'steel','label_probe',12)
    for a,b in zip(points[1:3],points[2:4]):
        m.cylinder((a[0],a[1]+.19,a[2]),(b[0],b[1]+.19,b[2]),.022,"black","cable",12)
    return m


def identification_bridge():
    """Low twin-rail bridge over the two observed identification lanes."""
    m=Mesh("identification_bridge")
    for y in [-5.1,0,5.1]:
        for x in [-.34,.34]:
            m.box((x,y,.65),(.14,.16,1.30),"gray","posts")
            m.box((x,y,.035),(.32,.55,.07),"gray","footplate")
            m.beam((x,y-.20,.08),(x,y,.40),.05,material="steel",part="foot_brace")
        for z in [.28,.64,1.0]:m.box((0,y,z),(.68,.09,.07),"steel","ladder_rungs")
    for x in [-.40,.40]:
        m.box((x,0,1.32),(.13,10.6,.20),"gray","bridge_beams")
        m.box((x,0,1.435),(.045,10.6,.03),"polished","robot_rails")
    for i in range(16):m.box((0,-5.1+i*.68,1.32),(.80,.08,.09),"gray","crossbars")
    return m


def identification_end_covers():
    """Two short enclosures outside the wheel ends, with inward-facing working openings."""
    m=Mesh("identification_hood")
    for side in [-1,1]:
        for x in [-.60,.60]:
            m.box((x,side*1.55,.60),(.10,.90,1.20),"machine","side_walls")
        m.box((0,side*1.98,.60),(1.30,.08,1.20),"machine","closed_outer_panel")
        m.box((0,side*1.55,.045),(1.34,.98,.09),"light_gray","interior_floor")
        m.box((0,side*1.68,1.29),(1.40,.66,.07),"light_gray","flat_cover")
        m.beam((0,side*1.10,1.12),(0,side*1.35,1.29),1.40,.07,"light_gray","chamfered_cover")
        for x in [-.54,.54]:m.box((x,side*1.10,.56),(.045,.045,1.10),"steel","opening_trim")
        m.box((0,side*1.73,.68),(.24,.18,.70),"orange","working_head")
        m.cylinder((0,side*1.71,.56),(0,side*1.12,.56),.055,"orange","axle_end_tool",20)
        for x in [-.54,.54]:
            for y in [1.19,1.90]:m.cylinder((x,side*y,1.32),(x,side*y,1.35),.017,"steel","cover_screws",10)
    return m


def console():
    m=Mesh("operator_console")
    m.box((0,0,.48),(.54,.43,.96),"gray","cabinet")
    m.box((0,-.215,.85),(.46,.025,.22),"black","panel")
    m.box((0,-.232,.86),(.37,.013,.15),"screen","display")
    for i in range(5):m.box((-.16+i*.08,-.234,.68),(.035,.015,.025),"light_gray","buttons")
    m.cylinder((.18,0,.96),(.18,0,1.15),.03,"dark_steel","light")
    m.cylinder((.18,0,1.13),(.18,0,1.19),.05,"green","light")
    return m


def opening_tool_station():
    """Paired low cylindrical cap tools, distinct from the adjacent grey portal at 8 s."""
    m=Mesh('opening_tool_station')
    m.box((0,0,.045),(1.45,2.75,.09),'dark_steel','workpiece_bed')
    m.box((0,0,.33),(.25,1.20,.18),'orange','axle_support')
    for y in [-.77,.77]:
        for x in [-.21,.21]:m.cylinder((x,y-.13,.13),(x,y+.13,.13),.12,'steel','workpiece_rollers',24)
    m.cylinder((.60,0,.20),(1.13,0,.20),.10,'machine','floor_motor',24)
    for side in [-1,1]:
        y=side*1.95
        m.box((0,y,.035),(1.7,1.70,.07),'light_gray','tool_floor_plate')
        m.box((0,y,.16),(1.14,1.42,.19),'orange','tool_slide')
        for x in [-.43,.43]:m.box((x,y,.28),(.10,1.37,.08),'steel','slide_guides')
        a,b=sorted([side*1.43,side*2.47])
        m.cylinder((0,a,.67),(0,b,.67),.32,'gray','cylindrical_shell',48)
        for yy in [a,b]:
            m.cylinder((0,yy-.025,.67),(0,yy+.025,.67),.34,'dark_steel','shell_end_flange',48)
            for n in range(8):
                angle=n*math.pi/4;xx=.275*math.cos(angle);zz=.67+.275*math.sin(angle)
                m.cylinder((xx,yy-.032,zz),(xx,yy+.032,zz),.024,'steel','flange_bolts',10)
        for yy in [a+.13+j*.15 for j in range(6)]:
            for x in [-.12,.12]:m.box((x,yy,.979),(.065,.04,.012),'black','shell_vents')
        m.cylinder((0,side*1.17,.67),(0,side*1.43,.67),.10,'orange_dark','end_tool')
        for x in [-.15,.15]:
            m.cylinder((x,side*1.23,.67),(x,side*1.47,.67),.028,'red','tool_connectors',12)
        for x in [-.37,.37]:m.beam((x,side*2.35,.29),(x,side*2.18,.69),.045,material='steel',part='shell_brackets')
        # Small white control box on an open pedestal beside each cylindrical tool.
        cx=.80;cy=side*1.70
        for dx in [-.22,.22]:
            for dy in [-.18,.18]:m.box((cx+dx,cy+dy,.28),(.035,.035,.56),'dark_steel','console_legs')
        m.box((cx,cy,.67),(.52,.48,.46),'light_gray','control_box')
        m.box((cx-.266,cy,.67),(.012,.40,.31),'cyan','control_face')
        m.box((cx-.276,cy,.69),(.012,.25,.17),'screen','control_display')
    return m


def opening_robot():
    """Generic static arm with the wider cap-handling tool visible in the opening sequence."""
    m=Mesh('opening_robot')
    m.box((0,0,.07),(.66,.66,.14),'orange','base')
    m.cylinder((0,0,.14),(0,0,.46),.25,'orange_dark','base_axis',32)
    pts=[(0,0,.52),(-.20,0,1.55),(.68,0,1.63),(1.02,0,1.13)]
    for n,(a,b) in enumerate(zip(pts,pts[1:])):
        m.beam(a,b,.26,.30,'orange',f'arm_link_{n}')
    for n,(x,y,z) in enumerate(pts):
        m.cylinder((x,-.18,z),(x,.18,z),.19 if n<2 else .14,'orange',f'joint_{n}',32)
        m.cylinder((x,-.205,z),(x,-.18,z),.13,'black',f'joint_cover_{n}',24)
    m.box((1.02,0,.95),(.32,.37,.22),'dark_steel','tool_chassis')
    for x in [.90,1.14]:
        for y in [-.17,.17]:
            m.cylinder((x,y,.78),(x,y,1.04),.027,'steel','tool_rods',12)
            m.box((x,y,.90),(.06,.06,.11),'green','tool_sensors')
    for y in [-.11,.11]:m.box((1.06,y,.72),(.07,.035,.18),'orange_dark','tool_fingers')
    return m


def opening_parts_bin():
    """Adjacent blue open bins with visible grooved cap stacks and roller dividers."""
    m=Mesh('opening_parts_bin')
    for cy in [-.65,.65]:
        m.box((0,cy,.14),(2.0,1.2,.12),'blue','bin_floor')
        for x in [-.97,.97]:m.box((x,cy,.36),(.065,1.2,.43),'blue','end_walls')
        for dy in [-.57,.57]:m.box((0,cy+dy,.36),(2.0,.065,.43),'blue','side_walls')
        for x in [-.85,.85]:
            for dy in [-.48,.48]:m.box((x,cy+dy,.08),(.10,.10,.16),'blue','feet')
    for y in [.36,.66,.96]:
        for x in [-.78+j*.10 for j in range(16)]:
            m.cylinder((x-.025,y,.40),(x+.025,y,.40),.145,'dark_steel','stacked_caps',24)
    for y in [-1.03,-.65,-.27]:
        for x in [-.78+j*.12 for j in range(14)]:
            m.cylinder((x,y-.12,.23),(x,y+.12,.23),.035,'steel','empty_bin_rollers',12)
    return m


def opening_portal():
    """Preparation portal shared by 43/45/48 s; roof and columns are estimated."""
    from .component_geometry import (PREPARATION_PORTAL_TOP as top,
        PREPARATION_PORTAL_HALF_X as hx,PREPARATION_PORTAL_HALF_Y as hy,
        PREPARATION_PORTAL_TOOL_X)
    m=Mesh('opening_portal');column_y=hy-.34
    for side in [-1,1]:
        y=side*column_y
        m.box((0,y,.05),(1.10,.78,.10),'dark_steel','feet')
        m.box((0,y,(top-.28)/2),(.58,.55,top-.28),'machine','columns')
        m.box((.296,y,.71),(.014,.44,.83),'gray','access_panels')
        for z in [.26,1.48]:m.box((.303,y,z),(.02,.59,.025),'steel','panel_seams')
        # Working tools retain the wheel axle height instead of inheriting a
        # uniform height scale from an unrelated press asset.
        x=PREPARATION_PORTAL_TOOL_X
        m.cylinder((x,side*1.50,.67),(x,side*1.0,.67),.18,'orange','end_tool_housing',32)
        m.cylinder((x,side*1.0,.67),(x,side*.87,.67),.085,'polished','end_tool_rod',24)
        m.box((x,side*1.45,.86),(.33,.42,.50),'orange_dark','vertical_end_drive')
    m.box((0,0,top-.18),(2*hx-.04,2*hy-.06,.32),'machine','header')
    m.box((0,0,top-.02),(2*hx,2*hy,.04),'light_gray','header_cover')
    for y in [-hy+.22,hy-.22]:
        m.cylinder((0,y,top+.001),(0,y,top+.006),.063,'dark_steel','service_holes',24)
        for x in [-hx+.08,hx-.08]:m.torus_y((x,y,top+.035),.035,.011,'steel','lifting_eyes',24)
    for y in [-1.30,-.65,0,.65,1.30]:m.box((0,y,top+.002),(2*hx-.14,.012,.003),'gray','cover_seams')
    m.box((PREPARATION_PORTAL_TOOL_X,0,.12),(1.05,3.2,.12),'gray','lower_bed')
    m.box((PREPARATION_PORTAL_TOOL_X,0,.24),(.23,2.7,.16),'orange','lower_drive')
    return m


def opening_frontage_portal():
    """Thin gray gate at 8 s; distinct from the preparation-area portal."""
    m=Mesh('opening_frontage_portal')
    for side in [-1,1]:
        y=side*1.91
        m.box((0,y,.045),(.95,.70,.09),'dark_steel','feet')
        m.box((0,y,.87),(.46,.48,1.65),'gray','columns')
        m.box((.237,y,.82),(.014,.32,.78),'dark_steel','column_recess')
        m.box((.25,y,.39),(.028,.38,.48),'gray','lower_panels')
        m.box((0,y,.51),(.50,.54,.022),'light_gray','panel_seams')
        m.cylinder((-.347,side*1.65,.67),(-.347,side*1.04,.67),.19,'orange','end_tools',32)
        m.cylinder((-.347,side*1.04,.67),(-.347,side*.88,.67),.09,'polished','tool_rods',24)
    m.box((0,0,1.775),(.48,4.28,.22),'gray','header')
    m.box((0,0,1.888),(.51253944,4.34349907,.04),'light_gray','header_cover')
    for y in [-1.85,-.95,0,.95,1.85]:
        m.box((0,y,1.910),(.44,.012,.004),'gray','cover_seams')
    for y in [-1.91,1.91]:
        m.cylinder((0,y,1.911),(0,y,1.915),.065,'dark_steel','service_holes',24)
        for x in [-.19,.19]:m.torus_y((x,y,1.94),.030,.010,'steel','lifting_eyes',20)
    for y in [-1.25,0,1.25]:
        m.box((-.13,y,1.54),(.15,.18,.26),'orange','suspended_tools')
    m.box((-.347,0,.12),(.90,3.05,.12),'gray','lower_bed')
    for y in [-.77,.77]:
        m.cylinder((-.58,y-.13,.18),(-.58,y+.13,.18),.13,'polished','support_rollers',24)
    return m


def parts_gantry():
    """Two longitudinal beams across the lanes and a compact bin trolley, seen at 14-16 s."""
    m=Mesh('parts_gantry')
    for x in [-2.1,2.1]:
        for y in [-8.7,8.7]:
            m.box((x,y,1.47),(.20,.23,2.94),'machine','columns')
            m.box((x,y,.045),(.54,.58,.09),'gray','feet')
        m.box((x,0,3.0),(.20,18.0,.30),'gray','runway_web')
        for z in [2.84,3.16]:m.box((x,0,z),(.34,18,.04),'steel','runway_flanges')
        m.box((x,0,3.20),(.055,18,.045),'polished','travel_rails')
    for y in [-8.7,8.7]:m.box((0,y,3.0),(4.5,.17,.23),'gray','end_ties')
    for y in [6.1,8.3]:m.box((0,y,3.40),(4.7,.20,.40),'orange','trolley_end_beams')
    for x in [-2.1,2.1]:m.box((x,7.2,3.40),(.20,2.4,.40),'orange_dark','trolley_side_beams')
    m.box((0,7.2,3.37),(1.5,1.6,.13),'orange','trolley_centre')
    for x in [-.70,.70]:m.box((x,7.2,2.23),(.17,.23,2.2),'orange','lift_posts')
    m.box((0,7.2,1.18),(1.7,1.0,.18),'orange','bin_gripper')
    from .front_geometry import GANTRY_X_SCALE, GANTRY_Y_SCALE
    m.vertices=[(x*GANTRY_X_SCALE,y*GANTRY_Y_SCALE,z) for x,y,z in m.vertices]
    return m


def exterior_inspection_rollers():
    m=Mesh('exterior_inspection_rollers')
    for y in [-.77,.77]:
        m.box((0,y,.045),(.80,.50,.09),'light_gray','floor_pad')
        for x in [-.22,.22]:
            m.box((x,y,.12),(.18,.32,.17),'machine','bearing_blocks')
            m.cylinder((x,y-.15,.18),(x,y+.15,.18),.10,'polished','rotation_rollers',24)
    return m


def axle_end_check_table():
    """Low motor-and-probe table visible at 17 s; static external form only."""
    m=Mesh('axle_end_check_table')
    m.box((0,0,.32),(.91,1.02,.64),'machine','cabinet')
    m.box((0,0,.66),(.96,1.08,.07),'gray','table_top')
    m.box((-.10,.03,.73),(.39,.70,.07),'steel','motor_mount')
    m.cylinder((-.10,-.25,.88),(-.10,.28,.88),.16,'gray','motor_housing',32)
    for y in [-.22,-.12,-.02,.08,.18]:m.torus_y((-.10,y,.88),.16,.012,'steel','motor_fins',24)
    m.cylinder((-.10,.28,.88),(-.10,.40,.88),.10,'machine','motor_end',24)
    for x in [.24,.36]:
        for y in [-.35,-.12,.11,.34]:
            m.cylinder((x,y,.70),(x,y,.82),.045,'steel','probe_bases',16)
            m.cylinder((x,y,.82),(x,y,.90),.027,'gray','probe_tips',16)
    for y in [-.38,.38]:m.box((-.38,y,.31),(.025,.14,.18),'black','vent')
    return m


def transfer_auxiliary_cell():
    """Small cabinet and narrow roller chute seen in the 42/50 s side enclosure; process unknown."""
    m=Mesh('transfer_auxiliary_cell')
    m.box((0,0,.37),(.48,.54,.74),'machine','cabinet')
    m.box((0,0,.76),(.54,.58,.045),'light_gray','top')
    m.box((-.17,0,.80),(.12,.23,.03),'orange','top_tool')
    m.box((0,-.28,.42),(.30,.025,.40),'black','front_panel')
    for x in [.44,.70,.96]:
        m.cylinder((x,-.20,.78),(x,.20,.78),.13,'polished','visible_rollers',24)
    for y in [-.23,.23]:
        m.box((1.33,y,.66),(2.15,.045,.09),'cyan','chute_sides')
        for x in [.50,2.28]:m.box((x,y,.32),(.055,.055,.64),'machine','legs')
    m.box((1.7,0,.70),(1.40,.40,.025),'light_gray','chute_bed')
    return m


def bearing_sorting_system():
    """Continuous blue travel base, open component bins and end tooling at 18-28 s."""
    m=Mesh('bearing_sorting_system')
    for y in [-.48,.48]:
        m.box((0,y,.38),(15.6,.13,.34),'blue','long_travel_beams')
        m.box((0,y,.57),(15.6,.06,.045),'polished','robot_travel_rails')
        for x in [-7.5,-4.5,-1.5,1.5,4.5,7.5]:m.box((x,y,.17),(.12,.12,.34),'blue','rail_feet')
    for x in [-7.5+j*1.0 for j in range(16)]:m.box((x,0,.30),(.10,1.0,.14),'blue','cross_ties')
    m.box((0,0,.66),(1.70,1.15,.15),'orange','robot_carriage')
    m.box((-1.15,0,1.02),(1.05,.85,.64),'orange','carriage_drive_box')
    for n,x in enumerate([-5.3,-3.05,-.8,1.45]):
        cy=-1.52
        m.box((x,cy,.18),(2.12,1.80,.13),'blue','bin_floor')
        for xx in [x-1.02,x+1.02]:m.box((xx,cy,.46),(.06,1.80,.50),'blue','bin_end_walls')
        for yy in [cy-.87,cy+.87]:m.box((x,yy,.46),(2.12,.06,.50),'blue','bin_side_walls')
        if n==0:
            for xx in [x-.88+j*.13 for j in range(14)]:m.cylinder((xx,cy-.75,.29),(xx,cy+.75,.29),.045,'steel','empty_bin_rollers',12)
        elif n==1:
            for xx in [x-.65,x,x+.65]:bearing_shell(m,xx,cy,.44,'loose_bearing_shells')
        elif n==2:
            for xx in [x-.80+j*.26 for j in range(7)]:
                for yy in [cy-.62+j*.20 for j in range(7)]:m.cylinder((xx,yy,.25),(xx,yy,.92),.022,'black','bin_location_pins',12)
        else:
            for xx in [x-.65,x,x+.65]:
                for yy in [cy-.42,cy+.42]:
                    # Convert a hollow Y-axis bearing to a vertical stack.
                    ring=Mesh('ring');bearing_shell(ring,0,0,0)
                    for z in [.30+i*.07 for i in range(9)]:
                        m.append([(xx+a,yy+c,z+b) for a,b,c in ring.vertices],ring.faces,'polished','stacked_vertical_shells')
                    for z in [.31+i*.07 for i in range(9)]:
                        ring=Mesh('groove');ring.torus_y((0,0,0),.228,.008,'dark_steel','groove',24)
                        m.append([(xx+a,yy+c,z+b) for a,b,c in ring.vertices],ring.faces,'dark_steel','stack_grooves')
    # End worktable lies across the long travel bed, with visible orange workholders.
    for x in [3.3,6.4]:
        for y in [-2.6,.7]:m.box((x,y,.45),(.10,.10,.90),'blue','table_legs')
    for x in [3.3,6.4]:
        for y in [-2.6,.7]:m.box((x,y,.95),(.13,.13,.13),'cyan','table_corners')
        m.box((x,-.95,.92),(.12,3.5,.20),'cyan','table_cross_beams')
    for y in [-2.55,-1.7,-.5,.65]:
        m.box((4.85,y,.94),(3.5,.10,.16),'cyan','fixture_rails')
        m.box((4.85,y,1.04),(3.5,.035,.035),'polished','fixture_guides')
    for x,y in [(3.7,-1.9),(6.1,-1.9),(4.9,.2)]:
        m.box((x,y,1.10),(.50,.58,.13),'orange','fixture_base')
        for side in [-1,1]:m.box((x+side*.18,y,1.32),(.10,.30,.35),'orange_dark','fixture_posts')
        m.cylinder((x,y-.25,1.37),(x,y+.25,1.37),.11,'dark_steel','fixture_jaw',24)
        m.box((x+.15,y,1.50),(.10,.12,.16),'green','fixture_sensor')
    # White / blue-violet casing, with an open front and an orange manipulator mast.
    m.box((4.9,.15,1.38),(1.25,1.15,.70),'machine','machine_cabinet')
    m.box((4.9,.15,1.76),(1.37,1.25,.08),'light_gray','cabinet_top')
    m.box((4.9,.15,1.81),(1.13,1.01,.025),'screen','violet_top_inset')
    m.box((4.9,-.44,1.40),(.95,.025,.43),'screen','front_opening')
    m.box((5.75,.22,1.94),(.20,.24,2.16),'orange_dark','transfer_mast')
    m.box((5.25,.22,1.08),(1.17,.24,.18),'orange','transfer_arm')
    for y in [.06,.38]:m.box((4.8,y,.93),(.06,.05,.30),'orange_dark','transfer_fingers')
    # Grey transverse rails seen behind the end machine, with no moving trolley claim.
    for x in [6.10,6.70]:
        m.box((x,9.0,2.80),(.18,22.0,.24),'gray','overhead_beams')
        m.box((x,9.0,2.95),(.045,22.0,.035),'polished','overhead_rails')
        for y in [-1.6,8.0,19.6]:m.box((x,y,1.34),(.16,.18,2.68),'machine','overhead_posts')
    # The end fixture table is narrower than the travel base plus its front bins.
    table_parts={'table_legs','table_corners','table_cross_beams','fixture_rails','fixture_guides',
        'fixture_base','fixture_posts','fixture_jaw','fixture_sensor'}
    ids={v for face,part in zip(m.faces,m.parts) if part in table_parts for v in face}
    m.vertices=[(2.7+(x-3.1)*1.2,y*.50-.60,z) if n in ids else (x,y,z) for n,(x,y,z) in enumerate(m.vertices)]
    # The near demount cabinet is shared with this table, rather than duplicated.
    casing={'machine_cabinet','cabinet_top','violet_top_inset','front_opening','transfer_mast','transfer_arm','transfer_fingers'}
    ids={v for face,part in zip(m.faces,m.parts) if part in casing for v in face}
    m.vertices=[(x,y+.85,z) if n in ids else (x,y,z) for n,(x,y,z) in enumerate(m.vertices)]
    return m


def sorting_robot():
    m=overhead_robot();m.name='sorting_robot'
    return m


def sorting_pick_robot():
    """21 s snapshot on a travel carriage parallel to the wheelset tracks."""
    m=Mesh('sorting_pick_robot')
    m.box((0,0,.06),(.88,.76,.12),'orange','base')
    m.cylinder((0,0,.12),(0,0,.40),.29,'orange_dark','axis_1',32)
    points=[(0,0,.30),(.206,.563,.56),(.924,2.583,1.66),(3.514,1.834,1.26),(2.899,1.386,.46)]
    for i,(a,b) in enumerate(zip(points,points[1:])):
        m.beam(a,b,.36 if i<2 else .25,.40 if i<2 else .29,'orange',f'arm_link_{i}')
    for i,(x,y,z) in enumerate(points[:-1]):
        r=.27 if i<2 else .18
        m.cylinder((x,y-.24,z),(x,y+.24,z),r,'orange',f'joint_{i}',32)
        m.cylinder((x,y-.27,z),(x,y-.24,z),r*.7,'orange_dark',f'joint_cover_{i}',24)
    m.box((2.9,1.39,.42),(.58,.43,.12),'orange','gripper_frame')
    for x in [2.69,3.11]:m.box((x,1.39,.19),(.045,.045,.42),'orange_dark','gripper_fingers')
    for a,b in zip(points[1:-1],points[2:]):
        m.cylinder((a[0],a[1]+.20,a[2]),(b[0],b[1]+.20,b[2]),.024,'black','cable',12)
    return m


def bearing_demount_station(name='bearing_demount_station',near_cabinet=True):
    """Paired white/violet low cabinets and orange axle-end tooling at 18-19 s."""
    m=Mesh(name)
    m.box((0,0,.045),(1.65,3.2,.09),'blue','floor_base')
    m.box((0,0,.33),(.25,1.2,.17),'orange','workpiece_slide')
    for side in [-1,1]:
        y=side*1.75
        if side<0 or near_cabinet:
            m.box((0,y,.49),(1.1,.90,.92),'light_gray','cabinet_shell')
            m.box((0,y,.99),(1.18,.98,.065),'light_gray','cabinet_rim')
            m.box((0,y,1.03),(.98,.78,.028),'screen','cabinet_violet_top')
            m.box((.56,y,.48),(.025,.68,.42),'screen','side_panel')
        m.box((0,side*1.21,.55),(.48,.33,.56),'orange_dark','tool_carriage')
        m.cylinder((0,side*.98,.67),(0,side*1.32,.67),.16,'orange','axle_end_tool',32)
        for x in [-.23,.23]:m.cylinder((x,side*1.0,.82),(x,side*1.58,.82),.035,'steel','tool_guides',16)
    if near_cabinet:
        m.box((-.45,2.6,1.20),(.26,.28,2.4),'orange_dark','transfer_column')
        m.box((-.45,2.16,.70),(.28,1.05,.18),'orange','transfer_arm')
    return m


def wiping_station(greasing=False):
    """Distinct tall wiping frame / low grease fixture seen either side of the portal."""
    if not greasing:
        from .preparation_details import wiping_asset
        return wiping_asset()
    m=Mesh('greasing_station' if greasing else 'wiping_station')
    from .component_geometry import WIPING_BOX_X, WIPING_BOX_Y, WIPING_FIXTURE_TOP
    height=.67 if greasing else WIPING_FIXTURE_TOP
    for y in [-.77,.77]:
        for x in [-.25,.25]:
            m.box((x,y,.08),(.18,.45,.16),'machine','roller_supports')
            m.cylinder((x,y-.18,.135),(x,y+.18,.135),.13,'polished','rotation_rollers',32)
    for y in [-.62,.62]:m.box((0,y,.08),(1.8,.12,.13),'gray','base_rails')
    for x in [-.58,.58]:m.box((x,0,.10),(.12,3.6,.17),'steel','cross_ties')
    for side in [-1,1]:
        y=side*1.40
        m.box((0,y,.06),(1.6,.90,.12),'light_gray','fixture_floor')
        for yy in [side*1.1,side*1.85]:
            post_height=.66 if greasing else height-.08
            m.box((-.42,yy,post_height/2+.06),(.13,.16,post_height),'gray','uprights')
            m.box((-.42,yy,.11),(.32,.35,.07),'steel','upright_feet')
        m.box((-.32,side*1.52,height),(.19,1.15,.16),'orange_dark','crosshead')
        m.box((-.25,side*1.20,.55),(.30,.38,.25),'orange','tool_slide')
        m.cylinder((0,side*.94,.67),(0,side*1.33,.67),.17,'machine','tool_drive',32)
        m.box((-.4,side*1.45,.64),(.08,.12,.10),'green','tool_sensor')
        if greasing:
            m.beam((-.32,side*1.15,.25),(-.32,side*1.80,.78),.15,material='orange',part='angled_tool_arm')
        else:
            triangle=[(side*1.12,.55),(side*1.12,height-.06),(side*1.86,height-.06)]
            vertices=[(x,yy,zz) for x in [-.40,-.24] for yy,zz in triangle]
            m.append(vertices,[(0,2,1),(3,4,5),(0,1,4),(0,4,3),(1,2,5),
                (1,5,4),(2,0,3),(2,3,5)],'orange','triangular_tool_bracket')
            m.beam((-.32,side*1.12,.55),(-.32,side*1.86,height-.06),
                   .08,.20,material='orange_dark',part='bracket_diagonal_rib')
            for yy,zz in [(side*1.28,1.12),(side*1.52,1.42)]:
                m.cylinder((-.414,yy,zz),(-.401,yy,zz),.05,'orange_dark','bracket_bolt_recess',24)
                m.cylinder((-.423,yy,zz),(-.415,yy,zz),.023,'steel','bracket_bolt',16)
        for x in [-.36,.36]:
            m.cylinder((x,side*1.1,.47),(x,side*1.95,.47),.032,'polished','axial_guides',16)
        bx,by=WIPING_BOX_X,WIPING_BOX_Y[side]
        m.box((bx,by,.48),(.38,.38,.64),'light_gray','side_box')
        m.box((bx,by,.805),(.18,.16,.012),'black','box_roof_display')
        m.box((bx,by,.812),(.125,.105,.006),'screen','box_roof_screen')
        for z in [.23,.70]:
            m.box((bx-.196,by-.15,z),(.014,.028,.09),'black','box_door_hinges')
        m.box((bx-.198,by+.12,.49),(.015,.025,.055),'steel','box_door_handle')
    m.cylinder((.45,0,.27),(1.08,0,.27),.12,'black','side_motor',32)
    return m


def closing_transfer_cell():
    """Separate cabinet, crossbeam, suspended orange tool and pinned tray at 67-69 s."""
    m=Mesh('closing_transfer_cell')
    # Source 67-69 s: cabinet behind the arms, plate and bellows along the track direction.
    m.box((5.47,2.72,1.24),(1.40,.78,2.48),'machine','machine_cabinet')
    for x,z in [(5.10,2.54),(5.82,2.42)]:
        m.box((x,2.72,z),(.70,.82,.09),'light_gray','cabinet_top')
    m.box((5.47,2.72,.06),(1.55,.94,.12),'gray','cabinet_foot')
    m.box((6.27,2.72,.75),(.27,.70,1.5),'machine','cabinet_side_table')
    for z in [.85,1.03,1.21]:m.box((6.43,2.68,z),(.08,.45,.035),'black','cabinet_fixture')
    for x in [5.35,10.65]:
        m.box((x,3.65,1.12),(.15,.20,2.24),'gray','transfer_posts')
        m.box((x,3.65,.035),(.48,.52,.07),'gray','transfer_feet')
    for y in [3.50,3.80]:
        m.box((8,y,2.30),(5.6,.12,.16),'gray','transfer_beams')
        m.box((8,y,2.40),(5.6,.035,.025),'polished','transfer_rails')
    m.box((10.35,3.65,2.48),(.66,.60,.16),'orange','hanging_carriage')
    m.box((10.35,3.65,2.72),(.25,.22,1.55),'orange_dark','hanging_mast')
    m.box((10.35,3.38,1.72),(.65,.25,.20),'orange','hanging_wrist')
    for x in [10.12,10.58]:m.box((x,3.32,1.39),(.045,.045,.52),'orange_dark','hanging_fingers')
    m.box((9.30,1.62,.14),(4.0,1.8,.16),'blue','table_runway')
    for y in [.77,2.47]:m.box((9.30,y,.24),(4.0,.045,.045),'polished','table_rails')
    m.box((8.45,1.62,.43),(1.72,1.51,.50),'screen','tray_carriage')
    m.box((8.45,1.62,.68),(1.8,1.6,.09),'light_gray','tray_top')
    for x in [7.82+j*.17 for j in range(8)]:
        for y in [1.64+j*.10 for j in range(7)]:
            m.cylinder((x,y,.73),(x,y,.77),.045,'dark_steel','cover_matrix',12)
            m.cylinder((x,y,.77),(x,y,.79),.018,'polished','cover_matrix',12)
    for x in [7.77,8.04,8.31,8.84,9.12]:
        m.cylinder((x,.96,.73),(x,.96,1.10),.019,'black','tray_pins',12)
    for y in [1.20,1.44,1.68]:m.cylinder((9.17,y,.73),(9.17,y,1.10),.019,'black','tray_pins',12)
    for z in [.77,.82,.87,.92]:
        ring=Mesh('cover');ring.torus_y((0,0,0),.20,.018,'steel','cover',32)
        m.append([(8.35+a,1.24+c,z+b) for a,b,c in ring.vertices],ring.faces,'steel','cover_stack')
    m.box((10.35,1.62,.44),(1.85,1.55,.48),'black','bellows_table')
    for x in [9.47+j*.075 for j in range(25)]:m.box((x,1.62,.71),(.035,1.55,.10),'dark_steel','bellows_ridges')
    for x,y in [(5.0,1.0),(6.7,1.35)]:
        for dx in [-.29,.29]:
            for dy in [-.25,.25]:m.box((x+dx,y+dy,.32),(.07,.07,.64),'gray','fixture_legs')
        m.box((x,y,.05),(.70,.62,.06),'gray','fixture_foot')
        m.box((x,y,.67),(.78,.68,.08),'light_gray','fixture_plate')
        for dx in [-.22,.22]:
            for dy in [-.18,.18]:m.cylinder((x+dx,y+dy,.71),(x+dx,y+dy,.72),.026,'black','fixture_holes',12)
    # Joint 67/68 s estimate: the fixed rails sit above the cabinet, not behind its roof.
    # The orange travelling mast is deliberately excluded from fixed-rail correspondences.
    beam_start,beam_end=5.07300334368132,10.63258700399617
    beam_shift=.5778215181231977
    tray_ratio=1.8674724519341654/1.6
    table_parts={'table_runway','table_rails','tray_carriage','tray_top','cover_matrix',
                 'tray_pins','cover_stack','bellows_table','bellows_ridges'}
    used={}
    for face,part in zip(m.faces,m.parts):
        for idx in face:used[idx]=part
    for idx,(x,y,z) in enumerate(m.vertices):
        part=used[idx]
        if part in {'transfer_posts','transfer_feet','transfer_beams','transfer_rails'} or part.startswith('hanging_'):
            x=beam_start+(x-5.2)*(beam_end-beam_start)/5.6;y-=.9
            if part=='transfer_posts':z*=2.80/2.24
            elif part not in {'transfer_feet'}:z+=beam_shift
        elif part in table_parts:y=1.62+(y-1.62)*tray_ratio
        m.vertices[idx]=(x,y,z)
    return m


def closing_primary_robot():
    m=Mesh('closing_primary_robot')
    m.box((0,0,.07),(.88,.82,.14),'orange','base')
    m.cylinder((0,0,.14),(0,0,.38),.30,'orange_dark','axis_1')
    points=[(0,0,.38),(.282,.631,2.0),(1.1012,-.8601,.68),(1.8804,-.7267,.88),(2.7202,-.9912,.82)]
    for i,(a,b) in enumerate(zip(points,points[1:])):
        m.beam(a,b,.37 if i<2 else .24,.42 if i<2 else .28,'orange',f'link_{i+1}')
    for i,(x,y,z) in enumerate(points[:-1]):
        r=.27 if i<2 else .18
        m.cylinder((x,y-.25,z),(x,y+.25,z),r,'orange',f'joint_{i+1}')
        m.cylinder((x,y-.28,z),(x,y-.25,z),r*.7,'orange_dark',f'joint_cover_{i+1}')
    m.box((2.70,-.99,.84),(.65,.48,.11),'orange','gripper_plate')
    for x in [2.48,2.9]:m.box((x,-.99,.96),(.06,.31,.17),'orange_dark','gripper_fingers')
    for a,b in zip(points[1:-1],points[2:]):
        m.cylinder((a[0],a[1]+.22,a[2]),(b[0],b[1]+.22,b[2]),.025,'black','cable',12)
    return m


def closing_secondary_robot():
    m=Mesh('closing_secondary_robot')
    m.cylinder((0,0,.03),(0,0,.22),.24,'orange_dark','base')
    points=[(0,0,.22),(.20,.20,.60),(.34,.7253,1.60),(1.1637,.0188,1.03),(1.1717,-.258,.80)]
    for i,(a,b) in enumerate(zip(points,points[1:])):
        m.cylinder(a,b,.18 if i<2 else .11,'orange',f'link_{i+1}',24)
    for i,(x,y,z) in enumerate(points[:-1]):
        m.cylinder((x,y-.17,z),(x,y+.17,z),.23 if i<3 else .13,'orange',f'joint_{i+1}',24)
    m.box((1.17,-.258,.76),(.16,.18,.10),'orange_dark','tool')
    return m


def roller_stand(name="roller_stand",color="blue"):
    m=Mesh(name)
    m.box((0,0,.09),(1.45,2.15,.18),color,"base")
    for y in [-.77,.77]:
        for x in [-.31,.31]:
            m.box((x,y,.17),(.20,.38,.23),"dark_steel","bearing_blocks")
            m.cylinder((x,y-.18,.24),(x,y+.18,.24),.115,"polished","rollers",32)
    for y in [-1,1]:m.box((.42,y,.25),(.55,.20,.20),"orange","actuators")
    return m


def portal(name="portal_press",mode="press"):
    m=Mesh(name)
    for y in [-1.65,1.65]:
        m.box((0,y,.10),(1.65,.85,.20),"dark_steel","feet")
        m.box((0,y,1.58),(.68,.66,2.96),"gray",f"column_{y}")
        m.box((.39,y,1.20),(.08,.58,.85),"light_gray","access_panels")
    m.box((0,0,3.0),(.90,3.94,.48),"gray","header")
    m.box((0,0,3.255),(1.03,4.05,.05),"light_gray","header_cover")
    for y in [-1.15,1.15]:
        side=1 if y>0 else -1
        m.cylinder((0,side*1.49,.56),(0,side*.98,.56),.20,"orange","ram_housing")
        m.cylinder((0,side*.98,.56),(0,side*.84,.56),.105,"polished","ram")
        for x in [-.32,.32]:m.cylinder((x,side*1.5,.8),(x,side*.95,.8),.045,"polished","guide_rods",16)
    m.box((0,0,.14),(1.40,2.70,.12),"gray","lower_bed")
    m.box((.40,0,.30),(.23,2.8,.17),"orange","lower_drive")
    if mode=="press":
        for y in [-1.15,1.15]:
            for x in [-.27,0,.27]:m.cylinder((x,y,3.26),(x,y,3.52),.10,"light_gray","hydraulics",20)
    return m


def hood(name="identification_hood",color="machine",length=2.2,height=1.75):
    """Box machine with chamfered light top, closed side walls and a lit working opening.

    The 2.62 m span between side walls keeps the wheelset axle passage clear; both ends stay open
    because the workpiece travels through the machine along the rail.
    """
    if name=="identification_hood":return identification_end_covers()
    m=Mesh(name)
    half=length/2
    for y in [-1.18,1.18]:
        m.box((0,y,height/2),(length,.30,height),color,f"side_{y}")
        m.box((-.06,y+(.155 if y>0 else -.155),height*.46),(.58,.02,.56),"dark_steel","service_vent")
        m.box((half*.18,y+(.155 if y>0 else -.155),height*.30),(.30,.02,.30),"steel","access_hatch")
        for x in [-half+.10,half-.10]:m.box((x,y,.055),(.16,.52,.11),"dark_steel","feet")
    # Light top plate, sloping slightly to the front, with an overhanging front lip.
    m.beam((half+.10,0,height-.04),(-half-.04,0,height+.20),2.66,.12,"light_gray","roof")
    m.box((0,0,height+.10),(length+.16,2.70,.07),"light_gray","roof_cap")
    for y in [-1.33,1.33]:m.box((0,y,height+.02),(length+.14,.07,.05),"steel","roof_trim")
    m.box((-half+.05,0,height-.10),(.10,2.62,.10),"steel","drip_edge")
    # Working opening: bright interior floor, orange tool head and a small status lamp.
    m.box((0,0,.16),(length*.82,2.30,.06),"light_gray","interior_floor")
    m.box((-.08,0,height-.30),(.34,2.10,.16),"steel","interior_header")
    m.box((-.05,0,height-.62),(.20,.34,.36),"orange","tool_body")
    m.cylinder((0,0,height-.80),(0,0,height-1.10),.055,"orange","tool_spindle",20)
    m.box((.02,0,height-1.14),(.16,.16,.09),"dark_steel","tool_tip")
    m.box((.10,.86,height-.72),(.14,.12,.20),"red","tool_lamp")
    m.cylinder((half*.55,-1.05,height+.10),(half*.55,-1.05,height+.28),.035,"black","light")
    m.cylinder((half*.55,-1.05,height+.28),(half*.55,-1.05,height+.37),.055,"green","light")
    return m


def machine_foundation(name="machine_foundation",length=4.6,width=3.4):
    """Raised light-grey floor pad with a machined rail slot, visible under each inspection machine."""
    m=Mesh(name)
    m.box((0,0,.045),(length,width,.09),"light_gray","pad")
    for y in [-.77,.77]:
        m.box((0,y,.10),(length-.24,.19,.02),"dark_steel","rail_bed")
    for y in [-width/2+.22,width/2-.22]:
        m.box((0,y,.093),(length-.5,.10,.012),"gray","edge_strip")
    for x in [-length/2+.22,length/2-.22]:
        for y in [-width/2+.22,width/2-.22]:
            m.box((x,y,.10),(.14,.14,.02),"dark_steel","anchor_plates")
    return m


def drive_unit():
    """Track drive motor and gearbox bolted beside the rail, seen at the derusting station."""
    m=Mesh("track_drive_unit")
    m.box((0,0,.10),(.90,.52,.20),"dark_steel","base_plate")
    m.cylinder((-.18,.10,.30),(.42,.10,.30),.15,"steel","motor_body",28)
    for i in range(6):
        a=-.14+i*.11
        m.cylinder((a,-.04,.30),(a,.24,.30),.155,"dark_steel","motor_fins",20)
    m.box((.52,.10,.30),(.22,.30,.30),"gray","gearbox")
    m.cylinder((.62,.10,.42),(.62,.10,.52),.045,"steel","oil_plug",16)
    for x in [-.32,.30]:
        for y in [-.18,.36]:m.cylinder((x,y,.20),(x,y,.235),.022,"steel","bolts",12)
    return m


def inspection_stand():
    """Paired wheel cradles at one axle station; assumed roller contact, static only."""
    m=Mesh("inspection_stand")
    for y in [-.77,.77]:
        for x in [-.60,.60]:
            m.box((x,y,.49),(.075,.10,.98),"blue","posts")
            m.box((x,y,.035),(.20,.28,.07),"steel","feet")
        m.box((0,y,1.00),(1.30,.10,.10),"cyan","upper_beam")
        # Wheel disks sit just inside the frame, leaving the axle visible.
        for x in [-.22,.22]:
            m.box((x,y-.15,.32),(.15,.18,.22),"blue","roller_bearings")
            m.cylinder((x,y-.23,.42),(x,y+.23,.42),.15,"polished","support_rollers",28)
        m.box((0,y,.23),(1.20,.12,.10),"blue","lower_tie")
    return m


def inspection_hood(name,length,height):
    """Separate pass-through stations visible in the 32-second side view."""
    m=Mesh(name)
    for y in [-INSPECTION_SHELL_WIDTH/2+.15,INSPECTION_SHELL_WIDTH/2-.15]:
        m.box((0,y,height/2),(length,.30,height),"machine","side_panel")
        m.box((0,y,.035),(length+.12,.42,.07),"steel","plinth")
    m.box((0,0,height),(length+.06,INSPECTION_SHELL_WIDTH,.12),"light_gray","flat_roof")
    for x in [-length/2,length/2]:
        for y in [-INSPECTION_SHELL_WIDTH/2+.30,INSPECTION_SHELL_WIDTH/2-.30]:m.box((x,y,height/2),(.06,.06,height),"steel","opening_trim")
    if name=="derust_hood":
        for y in [-.77,.77]:
            for x in [-.38,.38]:m.cylinder((x,y-.20,.22),(x,y+.20,.22),.12,"steel","internal_rollers",24)
        for x in [length/2+.12+i*.12 for i in range(10)]:
            m.box((x,0,.10),(.035,2.15,.045),"dark_steel","exit_grating")
        for y in [-1.03,1.03]:m.box((length/2+.65,y,.075),(1.35,.08,.10),"steel","grating_frame")
    else:
        for y in [-1.03,1.03]:
            m.cylinder((0,y,.61),(0,y*.82,.61),.12,"steel","side_probe")
    return m


def inspection_track():
    """Flush rail strip with one floor plate and low indexing rollers per module."""
    m=Mesh("inspection_track")
    for y in [-.77,.77]:
        m.box((0,y,.06),(3,.10,.08),"dark_steel","rail_bed")
        m.box((0,y,.105),(3,.035,.025),"steel","running_edge")
        for x in [-.16,.16]:
            m.cylinder((x,y-.10,.10),(x,y+.10,.10),.065,"polished","indexing_roller",16)
            m.box((x,y-.14,.075),(.15,.07,.10),"dark_steel","bearing_block")
    # The floor plate is split around the rails; it does not bury the rail head.
    for y,width in [(0,1.30),(-1.04,.34),(1.04,.34)]:
        m.box((0,y,.012),(1.5,width,.024),"gray","flush_floor_plate")
    return m


def inspection_crossbeam():
    """Visible transverse rail bridge; its hidden function is left unspecified."""
    m=Mesh("inspection_crossbeam")
    for x in [-.32,.32]:
        m.box((x,0,2.65),(.13,17,.22),"gray","bridge_beam")
        m.box((x,0,2.78),(.045,17,.04),"polished","running_rail")
        for y in [-8.3,8.3]:m.box((x,y,1.3),(.13,.16,2.6),"gray","end_post")
    for y in range(-8,9):m.box((0,y,2.64),(.76,.08,.08),"steel","cross_tie")
    return m


def floor_control_box():
    m=Mesh("floor_control_box")
    m.box((0,0,.18),(.40,.32,.36),"gray","enclosure")
    m.box((0,-.166,.20),(.29,.014,.15),"black","panel")
    m.box((0,-.177,.21),(.23,.014,.10),"light_gray","display")
    return m


def conveyor(name="roller_conveyor",length=4.0):
    m=Mesh(name)
    for y in [-.49,.49]:
        m.box((0,y,.66),(length,.10,.20),"blue","rails")
        for x in [-length/2+.18,length/2-.18]:m.box((x,y,.32),(.09,.09,.64),"blue","legs")
    for i in range(int(length/.17)):
        x=-length/2+.12+i*.17
        m.cylinder((x,-.44,.70),(x,.44,.70),.055,"polished","rollers",16)
    return m


def tray(name="parts_tray",length=2.0):
    m=Mesh(name)
    m.box((0,0,.24),(length,1.05,.12),"blue","bed")
    for y in [-.51,.51]:m.box((0,y,.38),(length,.065,.28),"blue","edges")
    for x in [-length/2+.04,length/2-.04]:m.box((x,0,.38),(.08,1.05,.28),"blue","edges")
    for x in [-length/2+.12,length/2-.12]:
        for y in [-.42,.42]:m.box((x,y,.12),(.08,.08,.24),"blue","feet")
    for i in range(8):
        x=-length/2+.2+i*(length-.4)/7
        for y in [-.24,.24]:m.torus_y((x,y,.43),.13,.038,"dark_steel","components",24)
    return m


def bearing_rack():
    m=Mesh("bearing_rack")
    for x in [-2.5,2.5]:
        for y in [-.48,.48]:m.box((x,y,.56),(.09,.09,1.12),"blue","legs")
    for z in [.32,.85]:
        m.box((0,0,z),(5.2,1.10,.08),"blue","shelves")
        for i in range(17):
            x=-2.4+i*.30
            for y in [-.24,.24]:m.torus_y((x,y,z+.20),.16,.045,"polished","stored_bearings",24)
    return m


def gantry(name="overhead_gantry",length=12.0,span=16.0):
    m=Mesh(name)
    for x in [-length/2,length/2]:
        for y in [-span/2,span/2]:
            m.box((x,y,1.9),(.23,.23,3.8),"gray","columns")
            m.box((x,y,.06),(.65,.65,.12),"gray","footplates")
    for y in [-span/2,span/2]:
        m.box((0,y,3.8),(length+.5,.25,.35),"gray","longitudinal_beams")
        m.box((0,y,4.00),(length+.5,.07,.045),"polished","rails")
    m.box((1,0,4.05),(.65,span+.65,.35),"orange","traverse")
    for x in [.65,1.35]:m.box((x,0,4.36),(.10,2.35,.26),"orange","trolley")
    m.box((1,0,4.25),(1.0,1.35,.18),"orange","trolley")
    for y in [-.45,.45]:m.cylinder((1,y,1.8),(1,y,4.2),.055,"polished","lift_rods",16)
    m.box((1,0,1.8),(.75,1.7,.16),"orange","gripper_frame")
    return m


def unknown_cyan_cell():
    """Low turquoise bed and exposed front tooling visible at 34-39 s; process unknown."""
    m=Mesh("unknown_cyan_cell")
    m.box((0,0,.55),(6.8,2.0,1.10),"cyan","low_bed")
    m.box((0,.92,1.27),(6.8,.16,.34),"cyan","rear_lip")
    for x in [-3.30,3.30]:m.box((x,0,1.20),(.20,2.0,.20),"cyan","end_lips")
    m.box((0,-1.20,.32),(2.10,.60,.38),"dark_steel","front_tool_bed")
    for x in [-.90,.90]:
        m.box((x,-1.35,.53),(.12,1.30,.12),"orange_dark","tool_frame")
        m.box((x,-1.95,.52),(.045,2.40,.055),"steel","floor_guides")
    for y in [-1.08,-1.78]:m.box((0,y,.51),(2.0,.10,.12),"orange","tool_crossmember")
    for side in [-1,1]:
        m.box((side*1.65,-1.42,.40),(.76,.64,.48),"light_gray","drive_cover")
        m.cylinder((side*.82,-1.42,.57),(side*1.25,-1.42,.57),.20,"yellow","drive_flange",24)
        m.cylinder((side*.58,-1.42,.57),(side*.91,-1.42,.57),.095,"steel","exposed_drive",24)
    for x in [-.48,.48]:
        m.cylinder((x,-1.55,.55),(x,-1.25,.55),.13,"black","tool_rollers",24)
    return m


def unknown_runway():
    """One continuous pair of low I-beam runways, shared by the three orange bridges."""
    m=Mesh("unknown_runway")
    for y in [-7.25,7.25]:
        for x in [-27,-18,-9,0,9,18,27]:
            m.box((x,y,1.06),(.20,.24,2.12),"machine","columns")
            m.box((x,y,.045),(.52,.58,.09),"gray","feet")
        m.box((0,y,2.23),(58,.09,.24),"machine","beam_web")
        for z in [2.10,2.36]:m.box((0,y,z),(58,.30,.045),"gray","beam_flanges")
        m.box((0,y,2.405),(58,.055,.045),"polished","travel_rail")
    return m


def unknown_bridge():
    """Static twin-girder bridge snapshot; no inferred joints or hidden drive mechanism."""
    m=Mesh("unknown_bridge")
    for x in [-.68,.68]:
        m.box((x,0,2.50),(.18,15.10,.26),"orange","bridge_girders")
        m.box((x,0,2.65),(.055,14.8,.035),"steel","trolley_rail")
    for y in [-7.25,7.25]:
        m.box((0,y,2.48),(1.90,.50,.20),"orange_dark","end_carriage")
        for x in [-.73,.73]:m.cylinder((x,y-.17,2.43),(x,y+.17,2.43),.10,"dark_steel","travel_wheels",20)
    for y in [-7+i*.75 for i in range(20)]:
        m.box((.95,y,2.67),(.52,.065,.06),"orange_dark","walkway_rungs")
        m.box((1.18,y,2.96),(.035,.035,.60),"orange_dark","walkway_posts")
    m.box((1.18,0,3.26),(.045,14.5,.045),"orange_dark","handrail")
    m.box((0,-1.7,2.76),(1.70,1.80,.20),"orange_dark","trolley")
    m.box((0,-1.7,3.17),(.30,.35,.72),"orange","lift_mast")
    for x in [-.40,.40]:m.cylinder((x,-1.7,1.10),(x,-1.7,2.72),.035,"steel","lift_rods",12)
    m.box((0,-1.7,1.10),(1.40,.50,.12),"orange","tool_crossbar")
    return m


def bearing_shell(m,x,y,z,part="bearing_shell"):
    # Closed hollow sleeve, rather than the ring-only placeholder used in 0.1.
    profile=[(-.20,.135),(-.20,.21),(-.17,.235),(.17,.235),(.20,.21),(.20,.135),(-.20,.135)]
    m.lathe_y(profile,(x,y,z),"polished",part,40)
    for end in [-.175,.175]:m.torus_y((x,y+end,z),.201,.009,"dark_steel",part+"_seam",32)


def bearing_feed_module(side=1):
    """Mirrored side feeders: both bearing rows approach the press from the same end."""
    m=Mesh('bearing_feed_module' if side>0 else 'bearing_feed_mirrored')
    for y in [-.20,.20]:
        m.box((-.05,y,.60),(4.5,.065,.14),'blue','feed_rails')
        m.box((-.05,y,.68),(4.5,.032,.020),'polished','feed_guides')
        for x in [-2.2,0,2.1]:m.box((x,y,.28),(.065,.065,.56),'blue','legs')
    m.box((-.05,0,.65),(4.5,.35,.025),'cyan','conveyor_bed')
    for x in [.85,1.10,1.35,1.60]:
        for y in [-.13,.13]:m.box((x,y,.69),(.18,.06,.045),'dark_steel','bearing_cradles')
        m.lathe_y([(-.13,.08),(-.13,.126),(-.11,.141),(.11,.141),(.13,.126),
                   (.13,.08),(-.13,.08)],(x,0,.85),'polished','bearing_shell',40)
        for y in [-.115,.115]:m.torus_y((x,y,.85),.121,.006,'dark_steel','bearing_seams',32)
    # Ring magazine is beside the conveyor, as in the frontal 62 s view.
    m.box((1.8134601,.43881241,.43292097),(.35584701,.38928885,.21919705),'blue','ring_magazine')
    for x in [1.66,1.97]:
        for y in [.27,.61]:m.cylinder((x,y,.54252),(x,y,1.08),.021,'dark_steel','magazine_posts',16)
    for level in range(5):
        ring=Mesh('ring')
        ring.lathe_y([(-.018,.105),(-.018,.16),(.018,.16),(.018,.105),(-.018,.105)],
                     material='dark_steel',segments=40)
        m.append([(1.8134601+a,.43881241+c,.59252+level*.052+b) for a,b,c in ring.vertices],
                 ring.faces,'dark_steel','stacked_rings')
    for x in [-2.3,2.4]:
        m.box((x,.80,.75),(.16,.19,1.50),'gray','loader_columns')
        m.box((x,.80,.045),(.40,.43,.09),'gray','loader_feet')
    m.box((.05,.80,1.55),(4.95,.20,.24),'gray','loader_beam')
    for y in [.67,.93]:m.box((.05,y,1.685),(4.95,.04,.025),'polished','loader_slides')
    m.box((1.81346,.80,1.73),(.48,.48,.10),'orange','loader_carriage')
    m.box((1.81346,.80,1.515),(.19,.23,1.07),'orange_dark','vertical_loader')
    for z in [1.05+i*.10 for i in range(10)]:
        m.box((1.70846,.80,z),(.025,.12,.045),'orange','mast_slots')
    m.box((1.81346,.56,1.02),(.32,.70,.14),'orange','loader_arm')
    m.cylinder((1.81346,.43881,.88),(1.81346,.43881,1.04),.10,'orange','loader_wrist',24)
    for x in [1.60,2.03]:m.box((x,.43881,.78),(.045,.07,.30),'orange_dark','loader_gripper')
    # Gauge table beside the final bearing, plus the upstream ring-check pedestal.
    m.box((2.0,-.22,.94),(.65,.65,.09),'light_gray','gauge_table')
    for x in [1.76,2.24]:
        for y in [-.46,.02]:m.box((x,y,.45),(.065,.065,.90),'gray','gauge_legs')
    m.box((1.85,.20,1.19),(.10,.12,.60),'orange','gauge_column')
    m.cylinder((1.85,.20,1.43),(1.85,-.25,1.43),.025,'polished','gauge_probe',16)
    m.box((1.85,-.27,1.43),(.12,.10,.11),'green','gauge_head')
    m.box((.55,.50,.46),(.26,.28,.92),'machine','ring_check_pedestal')
    m.box((.55,.50,.95),(.42,.40,.065),'orange','ring_check_tray')
    ring=Mesh('ring')
    ring.lathe_y([(-.015,.105),(-.015,.14),(.015,.14),(.015,.105),(-.015,.105)],
                 material='polished',segments=40)
    m.append([(.55+a,.50+c,1.0+b) for a,b,c in ring.vertices],ring.faces,'polished','gauge_ring')
    if side<0:
        m.vertices=[(x,-y,z) for x,y,z in m.vertices]
        m.faces=[(a,c,b) for a,b,c in m.faces]
    return m


def bearing_press_cell():
    """Paired low press housings and central wheel support, observed at 60–62 seconds."""
    m=Mesh("bearing_press_cell")
    m.box((0,0,.105),(2.15,4.2,.21),"gray","foundation")
    for side in [-1,1]:
        y=side*1.78
        m.box((0,y,.47),(1.48,.78,.62),"black","hydraulic_lower_housing")
        for x in [-.748,.748]:
            m.box((x,y,.53),(.018,.79,.055),"red","housing_red_band")
        # An enclosed faceted hood replaces the former floating inclined slab.
        profile=[(-.74,.78),(.74,.78),(.74,.96),(.25,1.35),(-.40,1.35),(-.74,1.02)]
        for yy,flip in [(y-.39,False),(y+.39,True)]:
            m.append([(x,yy,z) for x,z in profile],
                [(0,n+1,n) if flip else (0,n,n+1) for n in range(1,len(profile)-1)],
                'light_gray','faceted_white_hood')
        for a,b in zip(profile,profile[1:]+profile[:1]):
            m.append([(a[0],y-.39,a[1]),(b[0],y-.39,b[1]),
                (b[0],y+.39,b[1]),(a[0],y+.39,a[1])],[(0,2,1),(0,3,2)],
                'light_gray','hood_faces')
        m.cylinder((0,side*1.46,.78),(0,side*1.08,.78),.22,"steel","press_cylinder",40)
        m.cylinder((0,side*1.08,.78),(0,side*.91,.78),.145,"polished","press_ram",40)
        for x in [-.34,.34]:
            m.cylinder((x,side*1.48,.83),(x,side*1.11,.83),.105,"machine","guide_barrels",28)
            m.cylinder((x,side*1.11,.83),(x,side*.97,.83),.058,"polished","guide_shafts",20)
            for yy in [1.13,1.40]:
                m.cylinder((x,side*(yy+.025),.83),(x,side*(yy-.025),.83),.117,"steel","barrel_collars",24)
        for x in [-.34,.34]:
            for z in [.54,1.02]:m.cylinder((x,side*1.55,z),(x,side*.98,z),.032,"polished","ram_guides",16)
        m.box((-.55,y,1.02),(.21,.72,.12),"dark_steel","tool_slide")
    m.box((.3,0,.26),(1.2,1.3,.17),"black","indexing_bed")
    for x in [-.65,.65]:
        for y in [-.46,.46]:m.cylinder((x,y,.35),(x,y,.41),.17,"dark_steel","indexing_bearings",24)
    m.box((0,0,.405),(1.40,1.26,.05),"dark_steel","indexing_top_plate")
    for x in [-.51,.51]:m.box((x,0,.46),(.12,1.04,.12),"orange","wheel_carriage_sides")
    for y in [-.49,.49]:m.box((0,y,.46),(1.14,.11,.12),"orange","wheel_carriage_crossmembers")
    for x in [-.57,0,.57]:
        for y in [-.49,.49]:m.cylinder((x,y,.433),(x,y,.44),.055,"black","indexing_recesses",20)
    for y in [-.70,.70]:
        for x in [-.24,.24]:m.cylinder((x,y-.12,.22),(x,y+.12,.22),.12,"polished","support_rollers",24)
    m.cylinder((-.05,0,.35),(-.65,0,.35),.115,"steel","drive_motor",24)
    # Rear transverse manipulator, separate from the side feeder bridges.
    for y in [-2.15,2.15]:
        m.box((.45,y,.67),(.22,.22,1.34),"gray","bridge_columns")
        m.box((.45,y,.05),(.50,.50,.10),"gray","bridge_feet")
    m.box((.45,0,1.45),(.26,4.5,.24),"gray","manipulator_beam")
    for x in [.31,.59]:m.box((x,0,1.59),(.04,4.5,.04),"polished","manipulator_rails")
    m.box((.45,0,1.67),(.44,.62,.10),"orange","manipulator_carriage")
    m.box((.45,0,1.46),(.16,.17,1.08),"orange","manipulator_mast")
    for z in [.99,1.10,1.21,1.32,1.43,1.54,1.65,1.76,1.87,1.98]:
        m.box((.355,0,z),(.035,.15,.025),"orange_dark","mast_rack_teeth")
    m.box((.45,.17,1.55),(.22,.19,.32),"orange_dark","mast_drive")
    m.cylinder((.45,.18,1.55),(.45,.32,1.55),.13,"orange","mast_drive_cap",28)
    m.box((.08,0,1.10),(.86,.14,.13),"orange","manipulator_arm")
    m.cylinder((-.33,0,.90),(-.33,0,1.10),.13,"orange","manipulator_wrist",24)
    for y in [-.29,.29]:
        m.box((-.33,y,.72),(.08,.07,.30),"orange","manipulator_fingers")
        m.box((-.33,y*.8,.57),(.12,.16,.045),"orange_dark","finger_pads")
    return m


def storage_gantry():
    """Twin-girder moving bridge with trolley, lifting mast and a visual clamp."""
    m=Mesh("storage_gantry")
    for x in [-9,9]:
        for y in [-4,4]:
            m.box((x,y,1.9),(.22,.25,3.8),"gray","columns")
            m.box((x,y,.06),(.60,.65,.12),"gray","footplates")
            for dx in [-.20,.20]:
                for dy in [-.21,.21]:m.cylinder((x+dx,y+dy,.12),(x+dx,y+dy,.16),.025,"dark_steel","foot_bolts",12)
    for y in [-4,4]:
        m.box((0,y,3.8),(18.5,.23,.32),"gray","runway_web")
        for z in [3.64,3.96]:m.box((0,y,z),(18.5,.38,.055),"steel","runway_flange")
        m.box((0,y,4.02),(18.5,.065,.055),"polished","runway_rail")
    # End ties complete the rectangular stationary frame.
    for x in [-9,9]:m.box((x,0,3.81),(.14,8,.18),"steel","end_ties")
    bridge_x=0.0
    for x in [bridge_x-1.05,bridge_x+1.05]:
        m.box((x,0,4.14),(.18,8.6,.30),"orange","bridge_girders")
        for z in [3.99,4.29]:m.box((x,0,z),(.32,8.6,.055),"orange_dark","bridge_flanges")
        m.box((x,0,4.34),(.06,8.5,.045),"polished","trolley_rails")
    for y in [-4.12,4.12]:
        m.box((bridge_x,y,4.16),(2.75,.32,.30),"orange","end_trucks")
        for x in [bridge_x-1.05,bridge_x+1.05]:
            m.cylinder((x,y-.13,4.01),(x,y+.13,4.01),.16,"dark_steel","travel_wheels",24)
        m.cylinder((bridge_x+.70,y,4.32),(bridge_x+.70,y,4.73),.10,"orange","travel_motor",24)
    for x in [bridge_x-.88,bridge_x+.88]:m.box((x,0,4.49),(.15,1.7,.22),"orange","trolley_frame")
    for y in [-.78,.78]:m.box((bridge_x,y,4.49),(1.9,.15,.22),"orange","trolley_frame")
    m.box((bridge_x,0,5.12),(.24,.28,2.2),"orange","lift_mast")
    m.box((bridge_x+.18,0,4.05),(.12,.16,2.7),"steel","lift_slide")
    for z in [4.4,4.8,5.2,5.6,6.0]:m.box((bridge_x-.14,0,z),(.035,.18,.07),"dark_steel","mast_slots")
    for y in [-.44,.44]:m.beam((bridge_x,y,4.45),(bridge_x,y,3.68),.10,.10,"orange","mast_brace")
    m.box((bridge_x,0,2.69),(2.2,.65,.16),"orange","clamp_crossbar")
    for x in [-.95,.95]:
        m.box((bridge_x+x,0,2.32),(.15,.16,.70),"orange","clamp_fingers")
        m.box((bridge_x+x*.84,0,2.02),(.42,.30,.10),"dark_steel","clamp_pads")
    for y in [-4.52,-3.97]:m.beam((-9.45,y,.15),(-9.12,y,4.10),.035,material="steel",part="ladder_sides")
    for i in range(14):
        z=.20+i*.28;x=-9.45+(z-.15)/3.95*.33
        m.cylinder((x,-4.52,z),(x,-3.97,z),.022,"steel","ladder_rungs",12)
    return m


def storage_buffer():
    m=Mesh("storage_buffer")
    for y in [-.77,.77]:
        m.box((0,y,1.07),(11.2,.10,.12),"blue","buffer_rails")
        m.box((0,y,1.145),(11.2,.055,.04),"polished","buffer_tops")
        for x in [-5.4,-1.8,1.8,5.4]:m.box((x,y,.53),(.09,.09,1.06),"blue","buffer_legs")
    for x in [-5.4,-1.8,1.8,5.4]:m.box((x,0,.96),(.09,1.6,.09),"blue","buffer_crossbars")
    # Orange transfer cradle visible on the blue rails; internal drive is unknown.
    for y in [-.82,.82]:m.box((-.8,y,1.39),(.10,.10,.55),"orange","cradle_posts")
    for z in [1.17,1.60]:m.box((-.8,0,z),(.10,1.72,.10),"orange","cradle_crossbars")
    return m


def rack_bay():
    """Paired upright supports and open arms visible at 75-78 s; dimensions estimated."""
    m=Mesh("wheelset_storage_bay")
    for y in [-1.1,1.1]:
        m.box((0,y,2.6),(.24,.24,5.2),"gray","uprights")
        m.box((0,y,.10),(1.45,.46,.20),"gray","column_feet")
        for z in [1.10,2.35,3.60,4.85]:
            m.box((0,y*.70,z),(1.85,.13,.12),"steel","support_arms")
            m.box((0,y*.85,z),(.18,.40,.15),"gray","arm_bracket")
        m.box((0,y,5.20),(2.5,.08,.10),"steel","top_stringers")
    m.box((0,0,5.20),(.08,2.3,.10),"steel","top_tie")
    return m


def storage_stacker():
    """Static raised transfer frame seen at 77 s; no drive/kinematic claim."""
    m=Mesh("storage_stacker")
    for y in [-1.0,1.0]:
        m.box((0,y,2.6),(.18,.18,5.0),"orange","lift_columns")
        m.box((0,y,.22),(.85,.38,.30),"orange_dark","carriages")
    for z in [.50,5.10]:m.box((0,0,z),(.20,2.2,.20),"orange","frame_ties")
    for y in [-.77,.77]:m.box((0,y,1.12),(1.60,.14,.14),"orange","lift_forks")
    m.box((0,0,1.04),(.20,2.0,.18),"orange","lift_crossbar")
    return m


def bearing_storage_module():
    """Dense component storage at 43-54 s, separate from wheelset racks; tier count estimated."""
    m=Mesh('bearing_storage_module')
    for y in [-1.16,1.16]:
        m.box((0,y,2.70),(.075,.075,5.40),'gray','uprights')
        m.box((0,y,.04),(.24,.24,.08),'gray','feet')
        for z in [.62+i*.64 for i in range(8)]:
            m.box((0,y*.76,z),(.94,.065,.075),'steel','shelf_arms')
            for x in [-.40,.40]:m.box((x,y*.62,z+.045),(.055,.82,.055),'polished','tray_guides')
        for z in [.62,5.37]:m.box((0,y,z),(1.0,.065,.065),'steel','stringers')
    m.box((0,0,5.37),(.055,2.38,.06),'gray','top_tie')
    from .component_geometry import RACK_BAY_PITCH, RACK_HEIGHT
    m.vertices=[(x*RACK_BAY_PITCH,y,z*RACK_HEIGHT/5.4) for x,y,z in m.vertices]
    return m


def bearing_store_lift():
    """Orange vertical component transfer at the end of the dense rack."""
    m=Mesh('bearing_store_lift')
    from .component_geometry import RACK_HEIGHT
    for y in [-1.05,1.05]:
        m.box((0,y,RACK_HEIGHT/2),(.16,.18,RACK_HEIGHT),'orange','lift_columns')
        m.box((0,y,.17),(.55,.36,.28),'orange_dark','drive_carts')
        m.cylinder((-.22,y,.23),(.22,y,.23),.15,'orange_dark','drive_motors',24)
    for z in [.40,RACK_HEIGHT-.06]:m.box((0,0,z),(.16,2.25,.17),'orange','frame_ties')
    m.box((.15,0,.90),(.90,1.88,.12),'orange','transfer_bed')
    for y in [-.77,.77]:m.box((.50,y,.99),(1.3,.09,.09),'orange','transfer_forks')
    for y in [-1.05,1.05]:m.box((0,y,.025),(.85,.07,.05),'dark_steel','floor_guides')
    return m


def bearing_identification_line():
    """Narrow blue line and paired hoods visible beside the component store at 54 s."""
    m=Mesh('bearing_identification_line')
    m.box((0,0,.925),(7,.43,.035),'light_gray','conveyor_bed')
    for y in [-.25,.25]:
        m.box((0,y,.88),(7.0,.065,.10),'cyan','conveyor_rails')
        m.box((0,y,.95),(7.0,.035,.035),'polished','guide_tops')
        for x in [-3.35,-2,0,2,3.35]:
            m.box((x,y,.43),(.055,.055,.86),'blue','legs')
            m.cylinder((x,y,.01),(x,y,.045),.075,'gray','feet',16)
    for x in [-.30,.30]:m.box((-.6,x,.63),(1.20,.20,1.26),'blue','hood_base')
    for y in [-.52,.52]:
        m.box((-.6,y,1.20),(1.32,.50,.58),'blue','scanner_hood')
        m.beam((-1.26,y,1.48),(.06,y,1.38),.48,.08,'cyan','hood_top')
    for x in [.50,1.15,2.1]:bearing_shell(m,x,0,1.19,'visible_bearings')
    return m


def bearing_loading_robot():
    """Folded upright snapshot at 54 s; visual links, not driven joints."""
    m=Mesh('bearing_loading_robot')
    m.box((0,0,.06),(.66,.62,.12),'orange','base')
    m.cylinder((0,0,.12),(0,0,.38),.23,'orange_dark','axis_1')
    points=[(0,0,.40),(-.30,0,.98),(-.12,0,1.45),(.16,0,2.40),(.55,0,2.35),(.48,0,2.12)]
    for i,(a,b) in enumerate(zip(points,points[1:])):
        m.beam(a,b,.25 if i<3 else .17,.28 if i<3 else .19,'orange',f'link_{i+1}')
    for i,(x,y,z) in enumerate(points[:-1]):
        radius=.20 if i<3 else .15
        m.cylinder((x,y-.19,z),(x,y+.19,z),radius,'orange',f'joint_{i+1}')
        m.cylinder((x,y-.215,z),(x,y-.19,z),radius*.68,'orange_dark',f'joint_cover_{i+1}')
    for a,b in zip(points[1:-1],points[2:]):
        m.cylinder((a[0],.17,a[2]),(b[0],.17,b[2]),.022,'black','cable',12)
    m.box((.48,0,2.06),(.16,.22,.12),'dark_steel','tool')
    return m


def bearing_loading_fixture():
    """Independent gray cabinet and orange rotary clamping plate at 54 s."""
    m=Mesh('bearing_loading_fixture')
    m.box((0,0,.47),(.82,.92,.94),'machine','cabinet')
    for y in [-.35,.35]:
        m.box((0,y,.98),(.88,.055,.08),'polished','table_guides')
        m.box((0,y,1.045),(.58,.07,.06),'steel','carriage_slides')
    m.cylinder((0,0,.98),(0,0,1.13),.22,'dark_steel','rotary_axis')
    m.box((0,0,1.15),(.80,.74,.10),'orange','rotating_plate')
    m.box((0,0,1.235),(.16,.15,.10),'orange_dark','clamp')
    return m


def bearing_loading_tool_stand():
    m=Mesh('bearing_loading_tool_stand')
    m.box((0,0,.045),(.58,.48,.09),'machine','foot')
    m.box((0,0,.66),(.42,.36,1.23),'machine','upright')
    m.box((0,0,1.30),(.48,.44,.08),'orange','tool_tray')
    m.box((0,-.04,1.35),(.43,.32,.035),'light_gray','tool_pad')
    m.cylinder((-.12,0,1.38),(-.40,0,1.38),.025,'polished','tool_shaft',12)
    return m


def carrier():
    from .transfer_details import carrier_asset
    return carrier_asset()
    # Historical block-shaped proposal retained below for source history.
    m=Mesh("wheelset_agv")
    m.box((0,0,.19),(2.1,1.85,.26),"black","chassis")
    m.box((0,0,.34),(1.85,1.65,.055),"black","deck")
    for x in [-.78,.78]:
        for y in [-.75,.75]:
            m.cylinder((x,y-.075,.13),(x,y+.075,.13),.13,"black","running_wheels",24)
    for x in [-.26,.26]:
        for y in [-.76,.76]:m.box((x,y,.46),(.16,.30,.20),"orange","wheel_cradles")
    for x in [-.92,.92]:m.box((x,0,.34),(.12,.4,.04),"orange","cover")
    m.box((1.06,0,.22),(.035,1.5,.09),"dark_steel","bumper")
    m.cylinder((.85,.60,.33),(.85,.60,.42),.06,"yellow","beacon",16)
    return m


def treatment_head(name="wipe_head",color="orange"):
    m=Mesh(name)
    for side in [-1,1]:
        y=side*1.12
        m.box((0,y,.10),(.85,.6,.2),"gray","base")
        m.box((0,y,.52),(.18,.25,.85),"gray","column")
        m.cylinder((0,y,.70),(0,side*.86,.70),.13,color,"tool_housing")
        m.cylinder((0,side*.86,.70),(0,side*.77,.70),.17,"dark_steel","tool")
    return m


def measurement_cabinet():
    m=Mesh("measurement_cabinet")
    for y in [-.65,.65]:m.box((0,y,1.1),(1.3,.16,2.2),"light_gray","walls")
    m.box((.56,0,1.1),(.18,1.3,2.2),"light_gray","back")
    m.box((0,0,2.15),(1.3,1.45,.16),"light_gray","roof")
    m.box((.25,0,1.0),(.3,1.0,1.8),"black","interior")
    m.box((-.12,0,.7),(.75,1.0,.12),"steel","measurement_bed")
    m.cylinder((.2,0,.77),(.2,0,1.52),.06,"orange","probe")
    return m


def bearing_run_in_stand():
    """One of the two low end fixtures visible at 71 s; no driven axle or contact model."""
    m=Mesh('bearing_run_in_stand')
    for side in [-1,1]:
        y=side*1.22
        m.box((0,y,.035),(.66,.52,.07),'gray','base_plates')
        m.box((0,y,.115),(.46,.33,.10),'orange','end_drive_housings')
        m.box((0,y,.275),(.12,.15,.26),'orange_dark','bearing_end_supports')
        m.cylinder((0,side*1.10,.56),(0,side*1.32,.56),.065,'steel','end_spindles',24)
        m.cylinder((0,side*1.34,.07),(0,side*1.34,.84),.023,'black','sensor_masts',16)
        m.box((0,side*1.34,.79),(.07,.08,.19),'black','sensor_heads')
        m.cylinder((0,side*1.34,.87),(0,side*1.34,1.02),.012,'steel','sensor_probes',12)
    return m


def spray_portal():
    """Low transverse blue frame, side tooling and partial screens at 72-73 s."""
    m=Mesh('spray_portal')
    for y in [-1.9,1.9]:
        m.box((0,y,.03),(.48,.55,.06),'cyan','foot_plates')
        m.box((0,y,.65),(.19,.22,1.24),'blue','uprights')
        m.box((0,y,1.28),(.29,.32,.14),'blue','header_supports')
    m.box((0,0,1.38),(.22,4.1,.18),'blue','header')
    m.box((-.12,0,1.45),(.04,3.95,.045),'cyan','header_rail')
    for x in [-.17,.17]:
        m.box((x,-1.93,1.55),(.22,.38,.19),'black','end_drive_motors')
    for side in [-1,1]:
        y=side*1.20
        m.box((.8,y,.04),(.74,.60,.08),'gray','tool_bases')
        m.box((.8,y,.55),(.11,.12,1.02),'blue','side_tool_posts')
        m.box((.8,y,1.08),(.23,.25,.07),'cyan','side_tool_covers')
        m.cylinder((.80,side*.96,.56),(.80,side*1.28,.56),.09,'steel','tool_spindles',28)
        m.box((.80,side*1.40,.66),(.42,.30,.19),'orange' if side<0 else 'machine','tool_drive_motors')
        m.beam((.72,side*1.18,.84),(.88,side*.95,.69),.12,.13,'orange' if side<0 else 'blue','tool_carriages')
        m.cylinder((.8,side*.94,.67),(.8,side*.86,.59),.022,'black','spray_nozzles',16)
    for y in [-1.52,1.52]:
        m.box((1.15,y,1.0),(3.15,.025,2.0),'spray_screen','partial_screens')
    return m


def track_segment():
    m=Mesh("track_segment")
    for y in [-.77,.77]:
        m.box((0,y,.072),(3.0,.08,.06),"steel","rail_head")
        m.box((0,y,.035),(3.0,.035,.045),"dark_steel","rail_web")
    for x in [-1.0,0,1.0]:
        m.box((x,0,.012),(.16,1.9,.024),"dark_steel","ties")
        for y in [-.84,-.70,.70,.84]:m.box((x,y,.040),(.12,.05,.025),"gray","fasteners")
    return m


def make_assets():
    from .metrology_assets import make_metrology_assets
    assets=[wheelset(),robot(),overhead_robot(),identification_bridge(),console(),opening_tool_station(),opening_robot(),opening_parts_bin(),opening_portal(),parts_gantry(),exterior_inspection_rollers(),bearing_sorting_system(),sorting_robot(),sorting_pick_robot(),bearing_demount_station(),bearing_demount_station('bearing_demount_frontage',False),wiping_station(),closing_transfer_cell(),closing_primary_robot(),closing_secondary_robot(),roller_stand(),portal(),portal("cap_portal","cap"),
            hood(),inspection_hood("derust_hood",2.0,2.3),inspection_hood("magnetic_test_hood",1.55,2.4),
            inspection_hood("ultrasonic_test_hood",1.55,2.4),unknown_cyan_cell(),unknown_runway(),unknown_bridge(),
            conveyor(),conveyor("long_conveyor",6.0),tray(),bearing_rack(),gantry(),
            storage_gantry(),storage_buffer(),storage_stacker(),bearing_storage_module(),bearing_store_lift(),bearing_identification_line(),bearing_loading_robot(),bearing_loading_fixture(),bearing_loading_tool_stand(),bearing_feed_module(),bearing_press_cell(),rack_bay(),carrier(),treatment_head(),
            treatment_head("grease_head","light_gray"),measurement_cabinet(),spray_portal(),track_segment(),
            machine_foundation(),machine_foundation("inspection_foundation",4.2,3.2),drive_unit(),inspection_stand(),
            inspection_track(),inspection_crossbeam(),floor_control_box()]
    assets.extend([opening_frontage_portal(),bearing_feed_module(-1),bearing_run_in_stand(),axle_end_check_table(),transfer_auxiliary_cell(),wiping_station(True)])
    assets.extend(make_metrology_assets())
    from .preparation_details import infeed_asset,fin_unit_asset
    assets.extend([infeed_asset(),fin_unit_asset()])
    from .transfer_details import fin_bank_asset,boundary_post_asset
    assets.extend([fin_bank_asset(),boundary_post_asset()])
    return {m.name:m for m in assets}


def collision_boxes(name):
    """Conservative compound geometry; portals explicitly leave the workpiece opening clear."""
    if name=='opening_tool_station':
        return [([0,y,.52],[1.14,1.42,1.04]) for y in [-1.95,1.95]]+[
            ([0,0,.045],[1.45,2.75,.09])]
    if name=='opening_portal':
        from .component_geometry import PREPARATION_PORTAL_TOP as top,PREPARATION_PORTAL_HALF_X as hx,PREPARATION_PORTAL_HALF_Y as hy
        return [([0,y,(top-.28)/2],[.58,.55,top-.28]) for y in [-hy+.34,hy-.34]]+[
            ([0,0,top-.18],[2*hx-.04,2*hy-.06,.32])]
    if name=='opening_frontage_portal':
        return [([0,y,.87],[.46,.48,1.65]) for y in [-1.91,1.91]]+[
            ([0,0,1.775],[.48,4.28,.22]),([-.347,0,.12],[.90,3.05,.12])]
    if name=='parts_gantry':
        from .front_geometry import GANTRY_X_SCALE as sx, GANTRY_Y_SCALE as sy
        boxes=[([x,y,1.47],[.20,.23,2.94]) for x in [-2.1,2.1] for y in [-8.7,8.7]]+[
            ([x,0,3.0],[.34,18,.36]) for x in [-2.1,2.1]]
        return [([p[0]*sx,p[1]*sy,p[2]],[s[0]*sx,s[1]*sy,s[2]]) for p,s in boxes]
    if name=='bearing_sorting_system':
        return [([0,y,.38],[15.6,.13,.34]) for y in [-.48,.48]]+[
            ([x,-1.52,.46],[2.12,1.80,.50]) for x in [-5.3,-3.05,-.8,1.45]]+[
            ([x,y,1.34],[.16,.18,2.68]) for x in [6.1,6.7] for y in [-1.6,8,19.6]]
    if name in {'bearing_demount_station','bearing_demount_frontage'}:
        return [([0,y,.49],[1.1,.90,.92]) for y in ([-1.75,1.75] if name=='bearing_demount_station' else [-1.75])]+[
            ([0,0,.045],[1.65,3.2,.09])]
    if name=='preparation_infeed_bench':
        from .preparation_details import INFEED_POST_Y,INFEED_POST_HEIGHT,INFEED_BENCH_Y
        return [([0,y,INFEED_POST_HEIGHT/2],[.30,.32,INFEED_POST_HEIGHT]) for y in INFEED_POST_Y]+[
            ([-2,y,.36],[3.4,.25,.72]) for y in INFEED_BENCH_Y]
    if name=='wiping_station':
        return [([-.57,y,.69],[.12,.14,1.38]) for y in [-1.92,-.99,.99,1.92]]+[
            ([.62,y,1.18],[.30,.44,.32]) for y in [-1.33,1.33]]+[
            ([.02,-1.94,.29],[.47,.39,.50]),([2.0,2.30,.35],[.47,.32,.66])]
    if name=='greasing_station':
        from .component_geometry import WIPING_BOX_X,WIPING_BOX_Y,WIPING_FIXTURE_TOP
        post_height=.66 if name=='greasing_station' else WIPING_FIXTURE_TOP-.08
        return [([-.42,side*y,post_height/2+.06],[.13,.16,post_height]) for side in [-1,1] for y in [1.1,1.85]]+[
            ([-.25,y,.55],[.30,.38,.25]) for y in [-1.2,1.2]]+[
            ([WIPING_BOX_X,y,.48],[.38,.38,.64]) for y in WIPING_BOX_Y.values()]
    if name=='closing_transfer_cell':
        return [([5.47,2.72,1.24],[1.4,.78,2.48]),([8.45,1.62,.43],[1.8,1.8674724519341654,.59]),
                ([10.35,1.62,.44],[1.85,1.81,.58])]+[
                ([x,2.75,1.4],[.15,.20,2.8]) for x in [5.222,10.484]]
    if name in {"portal_press","cap_portal"}:
        return [([0,y,1.58],[.68,.66,2.96]) for y in [-1.65,1.65]]+[
            ([0,0,3.0],[.90,3.94,.48]),([0,0,.14],[1.4,2.7,.12])]
    if name=="bearing_press_cell":
        return [([0,y,.755],[1.48,.78,1.19]) for y in [-1.78,1.78]]+[
            ([.45,y,.67],[.22,.22,1.34]) for y in [-2.15,2.15]]+[
            ([.45,0,1.45],[.26,4.5,.24]),([.45,0,1.46],[.16,.17,1.08]),
            ([0,0,.105],[2.15,4.2,.21])]
    if name in {"derust_hood","magnetic_test_hood","ultrasonic_test_hood"}:
        length,height=(2.0,2.3) if name=="derust_hood" else (1.55,2.4)
        return [([0,y,height/2],[length,.30,height]) for y in [-INSPECTION_SHELL_WIDTH/2+.15,INSPECTION_SHELL_WIDTH/2-.15]]+[
            ([0,0,height],[length+.06,INSPECTION_SHELL_WIDTH,.12])]
    if name=="identification_hood":
        return [([0,y,.64],[1.4,.98,1.32]) for y in [-1.56,1.56]]
    if name=="unknown_cyan_cell":
        return [([0,0,.55],[6.8,2.0,1.10]),([0,-1.4,.32],[2.1,1.0,.38])]
    if name=="unknown_runway":
        return [([x,y,1.06],[.20,.24,2.12]) for x in [-27,-18,-9,0,9,18,27] for y in [-7.25,7.25]]+[
            ([0,y,2.23],[58,.30,.30]) for y in [-7.25,7.25]]
    if name=="unknown_bridge":
        return [([x,0,2.50],[.18,15.1,.26]) for x in [-.68,.68]]
    if "hood" in name:
        length,height={"identification_hood":(2.2,1.75),"derust_hood":(2.6,2.3),
            "magnetic_test_hood":(2.1,2.4),"ultrasonic_test_hood":(1.9,2.15),"unknown_cyan_cell":(3.7,2.15)}[name]
        # Side walls and the roof are solid; the interior stays open so the wheelset axle passes through.
        return [([0,y,height/2],[length,.30,height]) for y in [-1.18,1.18]]+[
            ([0,0,height+.05],[length+.06,2.66,.20])]
    if name in {"track_segment","inspection_track"}:return []
    if name=="inspection_stand":
        return [([x,y,.49],[.075,.10,.98]) for y in [-.77,.77] for x in [-.60,.60]]
    if name=="inspection_crossbeam":
        return [([x,y,1.3],[.13,.16,2.6]) for x in [-.32,.32] for y in [-8.3,8.3]]+[
            ([x,0,2.65],[.13,17,.22]) for x in [-.32,.32]]
    if name=="identification_bridge":
        return [([x,y,.65],[.14,.16,1.30]) for y in [-5.1,0,5.1] for x in [-.34,.34]]+[
            ([x,0,1.32],[.13,10.6,.20]) for x in [-.40,.40]]
    if name in {"overhead_gantry","storage_gantry"}:
        length,span=(12,16) if name=="overhead_gantry" else (18,8)
        return [([x,y,1.9],[.23,.23,3.8]) for x in [-length/2,length/2] for y in [-span/2,span/2]]
    if name=='bearing_run_in_stand':
        return [([0,y,.20],[.66,.52,.40]) for y in [-1.22,1.22]]
    if name=='spray_portal':
        return [([0,y,.65],[.19,.22,1.24]) for y in [-1.9,1.9]]+[
            ([0,0,1.38],[.22,4.1,.18])]+[
            ([1.15,y,1.0],[3.15,.025,2.0]) for y in [-1.52,1.52]]
    return None  # use the full asset's bounding box for static occupied footprints
