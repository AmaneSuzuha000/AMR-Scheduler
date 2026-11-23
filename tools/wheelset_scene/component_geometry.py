"""Shared video estimates for the preparation cache and component rack, not metrology."""
LONG_CACHE_FIRST_X = 96.4
LONG_CACHE_FIRST_Y = 26.0
LONG_CACHE_PITCH = 1.1543897700783718
LONG_CACHE_ROW_PITCH = 3.8108862933083625
LONG_CACHE_COUNTS = (18, 18, 18, 18)  # End counts outside the frame are assumed.
RACK_START = 74.83042617773256
RACK_END = 99.25679042381147
RACK_FRONT_Y = 31.298079224930344
RACK_HEIGHT = 4.360260530052982
RACK_BAY_PITCH = (RACK_END - RACK_START) / 33
ROAD_X = (90.16650300759805, 92.8515545637323)
COMPONENT_CAMERA = [104.12620992219188, 2.473151580483464, 30.505403941687806,
                    2.0271907307412174, .8560324404079062]
PREPARATION_DX = 2.96953174
PREPARATION_TOP_Y = 26.54513306
PREPARATION_LANE_PITCH = 7.97380424
COMPONENT_END_DX = -1.8
COMPONENT_END_DY = -2.7
# Historical 0.21 camera seed. 0.23 supersedes the same-station interpretation:
# 43 s shows separate wiping/greasing fixtures either side of the portal.
GREASE_CAMERA = [75.45318258542827, 28.70753126149, 5.201986014098565,
                 -.5882213220268174, .7064347772444157]
WIPING_BOX_X = .35279660125136303
WIPING_BOX_Y = {1: 1.2149186639288725, -1: -1.6478121459180706}
GREASE_HUB_PIXELS = [[459,418], [709,313]]
GREASE_ROOF_PIXELS = [[310,390],[367,367],[394,395],[330,421],
                      [753,215],[796,198],[826,220],[785,236]]
GREASE_RIM_BOUNDS = [[402,330,521,501],[650,241,772,385]]
# Keep all metrology equipment and its camera together; place it beside the
# existing third press lane rather than inventing a second background machine.
METROLOGY_DX = 6.36012180964423
METROLOGY_DY = -4.36690713598658
PREPARATION_PORTAL_DX = .70757538043566
PREPARATION_PORTAL_DY = -.50113658952461
PREPARATION_PORTAL_TOP = 1.926837398478177
PREPARATION_PORTAL_HALF_X = .39269864249966924
PREPARATION_PORTAL_HALF_Y = 1.9869496728597509
PREPARATION_PORTAL_ROOF_PIXELS = [[214,185],[253,209],[588,96],[543,80]]
# 43 s shows distinct wiping/greasing stations on opposite sides of the roof.
# Jointly use 43/45/48 s fixed roof points; keep camera43 and scale fixed.
WIPING_CAMERA_23 = [90.87395168072321,20.265759364130524,7.837656987264342,
                    2.4507139678888303,.6206102040038287]
GREASING_CAMERA_23 = [81.69989528285265,23.715499838350695,5.068877578075942,
                       2.5352500908851043,.6869846061859107]
GREASING_CENTER_X_23 = 77.162664853673
PREPARATION_PORTAL_CENTER_23 = [79.9585073644458,26.225545052689395]
PREPARATION_PORTAL_SNAPSHOT_X = 76.67788543186032
PREPARATION_FRONT_SNAPSHOT_X = 81.7251443993849
PREPARATION_PORTAL_TOOL_X = PREPARATION_PORTAL_SNAPSHOT_X+PREPARATION_DX-PREPARATION_PORTAL_CENTER_23[0]
WIPING_CENTRAL_HUB_PIXELS = [[550,395],[681,327]]
WIPING_FIXTURE_TOP = 1.65  # tall wiping frame; grease fixture is a separate low asset
PREPARATION_ROOF_PIXELS_43 = [[247,228],[271,241],[347,164],[327,160]]
GREASING_EXIT_WHEEL = [78.68592226514475,PREPARATION_TOP_Y,.3133172123351975]
HUB_READINGS = [
    (1,0,-1,[785,427]), (1,0,1,[803,388]),
    (2,0,-1,[821,443]), (2,0,1,[839,402]),
    (3,0,-1,[855,456]), (3,0,1,[874,417]),
    (4,0,-1,[891,472]), (4,0,1,[910,431]),
    (0,1,-1,[688,515]), (0,1,1,[711,477]),
    (1,1,-1,[724,533]), (1,1,1,[747,493]),
    (0,2,-1,[628,618]), (0,2,1,[655,575]),
]


def preparation_y(y):
    """Three shared lane centres; preserve each item's local transverse offset."""
    lane = min((8,17,26), key=lambda old: abs(y-old))
    return PREPARATION_TOP_Y - (26-lane)/9*PREPARATION_LANE_PITCH + y-lane


def preparation_portal_roof(scene):
    """Four corners of the shared top-lane roof, in 45 s reading order."""
    import math
    item=max((i for i in scene['instances'] if i['asset']=='opening_portal'
              and i['zone']=='preparation'),key=lambda i:i['pose'][1])
    x,y,z,_,_,yaw=item['pose'];c,s=math.cos(yaw),math.sin(yaw)
    hx,hy=PREPARATION_PORTAL_HALF_X,PREPARATION_PORTAL_HALF_Y
    return [[x+c*dx-s*dy,y+s*dx+c*dy,z+PREPARATION_PORTAL_TOP]
            for dx,dy in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]]


def greasing_landmarks(scene):
    """48 s hubs and cabinet roof corners on the actual separate machine."""
    import math
    item=max((i for i in scene['instances'] if i['asset']=='greasing_station'),
             key=lambda i:i['pose'][1])
    x,y,z,_,_,yaw=item['pose'];c,s=math.cos(yaw),math.sin(yaw)
    local=[[0,.77,.67],[0,-.77,.67]]
    for side in [1,-1]:
        for dx,dy in [(-.19,-.19),(-.19,.19),(.19,.19),(.19,-.19)]:
            local.append([WIPING_BOX_X-dx,WIPING_BOX_Y[side]-dy,.8])
    return [[x+c*a-s*b,y+s*a+c*b,z+h] for a,b,h in local]
