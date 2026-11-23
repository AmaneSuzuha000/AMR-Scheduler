"""Visible transfer envelopes; source65/66 estimates and room-orientation assumption."""
import math

FIN_BANK_X=136.7795227447896
FIN_BANK_Y=8.410809882891638
FIN_BANK_PIXELS=[[230,372],[579,664]]  # far bank, near bank approximate centres at65s
ROOM_PIVOT=[78.3,45.9]
WEST_CROSS_X=140.4009081700507
WEST_CROSS_PIXELS={
    'cache_transfer':[[812,459],[645,354],[438,236],[261,133]],
    'cache_reverse':[[203,479],[286,582],[365,681]],
}


def room_point(point):
    return [2*ROOM_PIVOT[0]-point[0],2*ROOM_PIVOT[1]-point[1],*point[2:]]


def carrier_asset():
    from .geometry import Mesh
    m=Mesh('wheelset_agv')
    hx,hy,r=.85,.75,.10
    outline=[]
    for cx,cy,start in [(hx-r,hy-r,0),(-hx+r,hy-r,90),(-hx+r,-hy+r,180),(hx-r,-hy+r,270)]:
        for n in range(5):
            angle=math.radians(start+n*22.5)
            outline.append((cx+r*math.cos(angle),cy+r*math.sin(angle)))
    # CCW closed footprint, outward caps/walls (Gazebo uses backface culling).
    for z,top in [(.08,False),(.205,True)]:
        m.append([(x,y,z) for x,y in outline],
            [(0,n,n+1) if top else (0,n+1,n) for n in range(1,len(outline)-1)],'black','rounded_deck')
    for a,b in zip(outline,outline[1:]+outline[:1]):
        m.append([(a[0],a[1],.08),(b[0],b[1],.08),(b[0],b[1],.205),(a[0],a[1],.205)],
            [(0,1,2),(0,2,3)],'black','deck_edges')
    for x in [-.64,.64]:
        for y in [-.63,.63]:m.cylinder((x,y-.045,.09),(x,y+.045,.09),.085,'black','running_wheels',20)
    for side in [-1,1]:
        for x in [-.19,.19]:
            m.box((x,side*.59,.232),(.13,.22,.054),'orange_dark','low_wheel_cradles')
            m.box((x,side*.59,.264),(.10,.17,.014),'orange','cradle_top')
    m.cylinder((0,0,.207),(0,0,.214),.09,'orange','circular_deck_insert',24)
    for x in [-.78,.78]:m.box((x,0,.14),(.035,.50,.035),'orange_dark','side_trim')
    # No tall beacon is visible in65/66. Sensors/drive are hidden assumptions.
    return m


def fin_bank_asset():
    from .geometry import Mesh
    from .preparation_details import fin_unit_asset
    m=Mesh('press_exit_fin_bank');unit=fin_unit_asset()
    for offset in [-.55,.55]:
        m.append([(-x*.75,-y*.75+offset,z*.65) for x,y,z in unit.vertices],
            unit.faces,'gray','placeholder')
        # Preserve unit materials and part names after the append.
        m.materials[-len(unit.faces):]=unit.materials
        m.parts[-len(unit.faces):]=['exit_'+part for part in unit.parts]
    m.box((0,0,.035),(.94,2.10,.07),'gray','bank_foundation')
    return m


def boundary_post_asset():
    from .geometry import Mesh
    m=Mesh('transfer_boundary_post')
    for x in [-.16,.16]:m.box((x,0,.75),(.045,.28,1.50),'machine','post_flanges')
    m.box((0,0,.75),(.32,.045,1.50),'machine','post_web')
    m.box((0,0,.025),(.43,.35,.05),'machine','post_foot')
    return m
