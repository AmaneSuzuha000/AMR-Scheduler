"""Initial explicit scene proposal. Subsequent builds read the saved JSON unchanged."""
from __future__ import annotations
import math


SEGMENTS=[
    ("S01",[0,2],"AGV上线","teardown"),("S02",[3,5],"标志板清理识别贴码","teardown"),
    ("S03",[6,7],"轮对外观检查","teardown"),("S04",[8,13],"轴承智能开盖","teardown"),
    ("S05",[14,16],"附件料筐搬运","teardown"),("S06",[17,17],"轴端识别与油封检查","teardown"),
    ("S07",[18,28],"轴承退卸分拣","sorting"),("S08",[29,32],"除锈与探伤","inspection"),
    ("S09",[33,33],"复探转轮","inspection"),("S10",[34,42],"字幕缺失设备区","unknown"),
    ("S11",[43,47],"轴颈与防尘板座擦拭","assembly"),("S12",[48,49],"自动涂脂","assembly"),
    ("S13",[50,51],"跨区AGV转运","logistics"),("S14",[52,54],"轴承识别上料","assembly"),
    ("S15",[55,57],"后挡测量与存取","metrology"),("S16",[58,64],"轴承压装","assembly"),
    ("S17",[65,66],"轮对缓存转运","logistics"),("S18",[67,70],"轴承智能关盖","assembly"),
    ("S19",[71,73],"磨合与涂漆","finishing"),("S20",[74,78],"轮对自动入库","storage"),
]


def initial_scene():
    scene={
        "scene_id":"wheelset_workshop_video_v1","schema_version":"wheelset-visual-1",
        "artifact_type":"visual_reconstruction","visual_revision":"0.3","units":{"length":"m","angle":"rad"},
        "global_scene_scale":1.0,
        "status":{"geometry":"video_referenced_estimate","camera_registration":"initial_estimate",
                  "physics":"static_collision_geometry_only","process_execution":"not_implemented",
                  "scheduler_integration":"not_implemented"},
        "reference_video":"车辆段轮轴检修线方案.mp4",
        "scale_anchor":{"kind":"assumed","wheel_tread_diameter_m":.92,
                        "note_zh":"仅作初始比例基准，未获用户实测尺寸。全局尺度可调整。"},
        "assumptions":[
            "总平面为跨镜头拼接估计；区域相对位置和设备实例数量需要进一步同镜头校准。",
            "机械臂为可编辑的通用外形，不声称恢复原厂运动学。",
            "34至42秒区域工序名未知，保留UNKNOWN标签。",
            "重载轮对AGV仅有外形和静态占位，未验证载荷。",
            "0.3版按2至6秒与29至33秒画面重建识别罩体、贴码桁架、探伤地基板与超探蓝色托架；尺寸仍为估计。",
            "轮对和部件使用视觉快照位置，尚无输送或作业状态机。",
            "通行路线只记录几何意图，未接入导航或验证转弯扫掠。"],
        "bounds_m":[138,72],
        "zones":[
            {"id":"teardown","name_zh":"识别 / 开盖 / 退卸","bounds":[7,6,55,27]},
            {"id":"sorting","name_zh":"轴承分拣与料筐","bounds":[42,27,63,34]},
            {"id":"inspection","name_zh":"除锈 / 探伤 / 复探","bounds":[7,44,62,63]},
            {"id":"unknown","name_zh":"字幕缺失设备区","bounds":[76,49,130,64]},
            {"id":"assembly","name_zh":"擦拭 / 涂脂 / 压装 / 关盖","bounds":[76,4,116,31]},
            {"id":"metrology","name_zh":"后挡测量 / 轴承上料","bounds":[76,31,94,37]},
            {"id":"finishing","name_zh":"磨合 / 涂漆","bounds":[116,4,130,31]},
            {"id":"storage","name_zh":"自动立体库","bounds":[72,38,130,48]},
            {"id":"logistics","name_zh":"转运 / 交接 / 缓存","bounds":[63,4,74,68]}],
        "video_segments":[{"id":a,"seconds":b,"name_zh":c,"zone":d} for a,b,c,d in SEGMENTS],
        "instances":[],"boxes":[],"rail_lines":[],"agv_routes":[],"cameras":[]}
    serial={}
    def put(asset,x,y,z=0,yaw=0,zone="teardown",segments=(),label=None):
        serial[asset]=serial.get(asset,0)+1
        key=f"{asset}_{serial[asset]:03d}"
        scene["instances"].append({"id":key,"asset":asset,"pose":[x,y,z,0,0,yaw],"zone":zone,
            "source_segments":list(segments),"geometry_source":"video_observed_form",
            "pose_source":"estimated","label_zh":label or asset})
        return key
    def box(key,center,size,material,zone="environment",source="assumed",yaw=0):
        scene["boxes"].append({"id":key,"center":list(center),"size":list(size),"material":material,
                               "zone":zone,"source":source,"yaw":yaw})
    def rails(key,x0,x1,y,zone):
        scene["rail_lines"].append({"id":key,"start":[x0,y],"end":[x1,y],"gauge_m":1.54,
                                     "source":"estimated","zone":zone})
        length=x1-x0
        for i in range(int(length/3)):
            put("track_segment",x0+1.5+i*3,y,zone=zone)
    def wheels(xs,y,zone,segments,z=0):
        for x in xs:put("wheelset",x,y,z,zone=zone,segments=segments)

    box("floor",[69,36,-.08],[138,72,.16],"floor")
    box("north_wall",[69,71,1],[138,.35,2],"wall",source="estimated")
    box("west_wall",[1,36,1],[.35,70,2],"wall",source="estimated")
    box("east_wall",[137,36,1],[.35,70,2],"wall",source="estimated")
    for x0,x1 in [(1,61),(74,137)]:box(f"south_wall_{x0}",[(x0+x1)/2,1,.55],[x1-x0,.25,1.1],"wall",source="estimated")
    for y0,y1 in [(1,5),(11,14),(20,23),(29,32),(42,62)]:box(f"partition_{y0}",[71,(y0+y1)/2,1],[.3,y1-y0,2],"wall",source="estimated")
    box("unknown_partition",[106,47.7,1],[58,.25,2],"wall",source="estimated")
    for i,(start,end) in enumerate([([4,4],[134,4]),([4,37.5],[134,37.5]),([4,67.5],[134,67.5]),
                                    ([4,4],[4,67.5]),([67,4],[67,67.5]),([134,4],[134,67.5])]):
        scene["agv_routes"].append({"id":f"visual_route_{i}","start":start,"end":end,"width_m":3.5,
                                    "kind":"wheelset_carrier","source":"estimated","navigation_validated":False})
    for y in [9,15,21,50,58]:
        scene["agv_routes"].append({"id":f"dock_west_{y}","start":[61,y],"end":[67,y],"width_m":3.5,"kind":"handoff","source":"estimated","navigation_validated":False})
    for y in [8,17,26,53,61]:
        scene["agv_routes"].append({"id":f"dock_east_{y}","start":[67,y],"end":[76,y],"width_m":3.5,"kind":"handoff","source":"estimated","navigation_validated":False})

    # Three parallel dismantling lanes, as seen in the first sequence.
    for lane,y in enumerate([9,15,21]):
        rails(f"dismantling_{lane}",7,61,y,"teardown")
        put("identification_hood",14,y,segments=["S02"])
        put("roller_stand",21,y,segments=["S03"])
        put("cap_portal",28,y,segments=["S04"])
        put("roller_stand",35,y,segments=["S06"])
        put("portal_press",43,y,segments=["S07"],zone="sorting")
        for x in [13,28,43]:
            if x==13:
                put("overhead_robot",12.1,y,z=2.83,segments=["S02"])
            else:
                put("industrial_arm",x-1.4,y+2.35,yaw=-math.pi/2,segments=["S04" if x==28 else "S07"])
            put("operator_console",x-1,y-2.3,yaw=math.pi/2,segments=["S04"])
        for x in [30.5,38]:put("parts_tray",x,y+2.5,segments=["S04","S05"])
        wheels([9.5,14,18,21,24.5,28,32,35,39,43,48,52,56,59],y,"teardown",["S01","S04","S07"])
    put("identification_bridge",12.1,15,zone="teardown",segments=["S02"])
    put("overhead_gantry",34,15,zone="teardown",segments=["S05"])
    for x in [46,53,60]:
        put("bearing_rack",x,31,zone="sorting",segments=["S07"])
        put("industrial_arm",x-1.3,29,zone="sorting",segments=["S07"],yaw=math.pi/2)
        put("roller_conveyor",x,28,zone="sorting",segments=["S07"])
    # Twin inspection lines; each machine sits on its own raised floor pad, as at 30-32 seconds.
    for lane,y in enumerate([50,58]):
        rails(f"inspection_{lane}",7,61,y,"inspection")
        for x,asset in [(21,"derust_hood"),(36,"magnetic_test_hood"),(50,"ultrasonic_test_hood")]:
            put("inspection_foundation",x,y,zone="inspection",segments=["S08"])
            put(asset,x,y,zone="inspection",segments=["S08"])
            put("operator_console",x-1,y-2.2,zone="inspection",segments=["S08"])
        # Rail drive motor and gearbox beside the derusting machine inlet.
        put("track_drive_unit",19.2,y+1.04,zone="inspection",segments=["S08"])
        # Two blue cradles carry the wheelsets out of the ultrasonic test machine.
        for x in [54.4,57.0]:
            put("inspection_stand",x,y,zone="inspection",segments=["S09"])
        wheels([10,13,16,21,27,30,33,36,42,45,50,54.4,57],y,"inspection",["S08","S09"])
    # Unknown cyan equipment is retained as unknown, with explicit initial placement.
    for i,x in enumerate([80,89,98,107,116,125]):
        put("unknown_cyan_cell",x,59,yaw=math.pi/2,zone="unknown",segments=["S10"])
        for y in [50.5,53.0,55.5]:
            put("wheelset",x,y,yaw=math.pi/2,zone="unknown",segments=["S10"])
        put("roller_stand",x,57,yaw=math.pi/2,zone="unknown",segments=["S10"])
    put("overhead_gantry",100,55,zone="unknown",segments=["S10"])
    # Reassembly and finishing; all geometry remains static in this visual release.
    for lane,y in enumerate([8,17,26]):
        rails(f"assembly_{lane}",76,130,y,"assembly")
        for x,asset,seg in [(80,"wipe_head","S11"),(86,"grease_head","S12"),(98,"bearing_press_cell","S16"),
                           (110,"cap_portal","S18"),(119,"roller_stand","S19"),(125,"spray_portal","S19")]:
            put(asset,x,y,zone="assembly" if x<116 else "finishing",segments=[seg])
        for side in [-1,1]:
            put("bearing_feed_module",95,y+side*1.6,yaw=math.pi if side<0 else 0,zone="assembly",segments=["S14","S16"])
        for x in [109]:
            put("industrial_arm",x,y+2.25,yaw=-math.pi/2,zone="assembly",segments=["S14","S16","S18"])
            put("roller_conveyor",x,y-2.4,zone="assembly",segments=["S14","S16"])
        for x in [82,102,112]:put("operator_console",x,y+2.3,zone="assembly",segments=["S16"])
        wheels([77,80,83,86,90,94,98,103,106,110,115,119,125,128],y,"assembly",["S11","S12","S16","S18","S19"])
    for x in [80,87]:
        put("bearing_rack",x,34.2,zone="metrology",segments=["S15"])
        put("measurement_cabinet",x,32,zone="metrology",segments=["S15"],yaw=-math.pi/2)
        put("industrial_arm",x-1.4,32,zone="metrology",segments=["S15"],yaw=math.pi/2)
    # Modular storage bays with intentionally partial occupancy.
    for row,y in enumerate([42,45]):
        for col in range(17):
            x=87+2.5*col
            put("wheelset_storage_bay",x,y,zone="storage",segments=["S20"])
            for level in range(3):
                if (col+2*level+row)%4==0:
                    put("wheelset",x,y,z=.17+level*1.25,zone="storage",segments=["S20"])
    put("storage_gantry",78,42,zone="storage",segments=["S20"])
    for y in [40.3,43.7]:
        put("storage_buffer",82.5,y,zone="storage",segments=["S20"])
        put("wheelset",83.5,y,z=1.09,zone="storage",segments=["S20"])
    put("wheelset",76.1,42,z=1.64,zone="storage",segments=["S20"],label="入库夹具中的轮对静态快照")
    for i,(x,y,yaw) in enumerate([(64,9,0),(67,33,math.pi/2),(67,61,math.pi/2),(72,17,0),(134,23,math.pi/2),(129,37.5,0)]):
        put("wheelset_agv",x,y,yaw=yaw,zone="logistics",segments=["S01","S13","S17","S20"])
        put("wheelset",x,y,z=.34,yaw=yaw,zone="logistics",segments=["S01","S17","S20"])
    for y in [40,43]:wheels([10,13,16,19,22],y,"logistics",["S17"])

    # These are explicitly initial camera estimates. Registration must be measured separately.
    camera_specs=[
        ("overview",0,[35,-36,64],[68,31,0],.92),
        ("identification",3,[27,-1,16],[14,15,.8],.92),
        ("opening",10,[37,3,10],[29,15,1],.90),
        ("sorting",21,[58,20,7.5],[53,30,1],.90),
        ("inspection",30,[38,29,23],[36,54,.7],1.12),
        ("unknown",36,[126,34,22],[102,56,.8],1.00),
        ("wiping",45,[89,2,9],[81,15,1],.90),
        ("press",58,[90,11.2,7.2],[97,16,1.35],.92),
        ("press_front",62,[88,17,7.2],[98,17,1.3],.92),
        ("closing",68,[118,9,7],[110,17,1.1],.88),
        ("storage",76,[64,28,17],[78,42,2.4],.92),
        ("storage_reverse",78,[66,57,18],[79,42,2.4],.92),
        ("overview_east",None,[193,-75,116],[69,34,0],1.05),
        ("top",None,[69,36,120],[69,36,0],1.11)]
    scene["cameras"]=[{"id":a,"reference_second":b,"position":c,"target":d,"horizontal_fov":e,
                        "width":1280,"height":720,"registration":"initial_estimate"} for a,b,c,d,e in camera_specs]
    for camera in scene["cameras"]:
        ratio=math.tan(camera["horizontal_fov"]/2)/math.tan(.92/2)
        camera["position"]=[b+(a-b)*ratio for a,b in zip(camera["position"],camera["target"])]
        camera["horizontal_fov"]=.92
    # Supports raise wheels above the rails at processing stations; no dynamics claimed.
    supports={(round(i["pose"][0],3),round(i["pose"][1],3)):i["asset"] for i in scene["instances"]
              if i["asset"] in {"roller_stand","portal_press","cap_portal","bearing_press_cell"}}
    for item in scene["instances"]:
        if item["asset"]=="wheelset" and item["pose"][2]==0:
            support=supports.get(tuple(round(v,3) for v in item["pose"][:2]))
            if support:item["pose"][2]=.164 if support=="roller_stand" else .22 if support=="bearing_press_cell" else .11
    refine_inspection_scene(scene)
    refine_identification_scene(scene)
    refine_visual_environment(scene)
    refine_storage_scene(scene)
    refine_unknown_scene(scene)
    refine_opening_scene(scene)
    refine_sorting_scene(scene)
    refine_assembly_service_scene(scene)
    refine_component_storage_scene(scene)
    refine_transfer_blocks_scene(scene)
    refine_component_frontage_scene(scene)
    refine_closing_side_scene(scene)
    refine_sorting_parallel_scene(scene)
    refine_opening_frontage_scene(scene)
    refine_press_feed_scene(scene)
    refine_finishing_scene(scene)
    refine_closing_fixed_scene(scene)
    refine_metrology_scene(scene)
    refine_front_detail_scene(scene)
    refine_reinspection_scene(scene)
    refine_front_shared_scene(scene)
    refine_transfer_detail_scene(scene)
    refine_preparation_scale_scene(scene)
    refine_greasing_silhouette_scene(scene)
    refine_metrology_press_connection_scene(scene)
    refine_preparation_portal_and_robot_scene(scene)
    refine_preparation_fixed_scene(scene)
    refine_preparation_details_scene(scene)
    refine_metrology_details_scene(scene)
    refine_transfer_context_scene(scene)
    return scene


def refine_storage_scene(scene):
    """Revision 0.7: shared storage geometry constrained by the 75-78 s sequence."""
    scene['visual_revision']='0.7'
    scene.setdefault('layout_offsets',{})['storage_y']=0
    for camera in scene['cameras']:
        if camera['id']=='top':
            camera.update(position=[69.0,36.0,165.0],target=[69,36,0],yaw_rad=math.pi/2)
    scene['instances']=[i for i in scene['instances'] if i['zone']!='storage']
    serial={}
    def put(asset,x,y,z=0,yaw=0):
        serial[asset]=serial.get(asset,0)+1
        scene['instances'].append({'id':f'storage_r07_{asset}_{serial[asset]:03d}',
            'asset':asset,'pose':[x,y,z,0,0,yaw],'zone':'storage','source_segments':['S20'],
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':asset})
    for y in [36.7,43.5,48.5]:
        for col in range(16):put('wheelset_storage_bay',89.1+2.5*col,y)
        put('storage_buffer',82.3,y)
    put('storage_gantry',78,42.4,yaw=math.pi/2)
    put('storage_stacker',88.1,48.5)
    put('wheelset',78,42.4,1.64)
    # REJECTED 2026-11-11: extra lattice gantries over the rack rows were tried as a
    # candidate (env WS_STORAGE_GANTRY_VARIANT) at x=104/115 and x=206/218 and in both
    # cases the added geometry produced ZERO new silhouette pixels:
    #     coverage 1x 0.2088  2x 0.2088  3x 0.2088   added px 0 / 0
    # because storage_reverse looks north-east from (191.6, 21.0) and the rack rows at
    # y 36.7-48.5 project to screen x 1200-2500, i.e. off the right edge. Back-projecting
    # the reference's dense-rack region lands at negative z under the camera, so the
    # reference structure there is closer and taller than anything in the rack rows, and
    # its position cannot be inferred from this view. Rejected under ACCEPTANCE_RULE.md
    # (mean gain 0.0000 against a +0.0050 bar). Do not re-add without new evidence.
    scene['assumptions']=[a for a in scene['assumptions'] if not a.startswith('0.7入库区')]
    scene['assumptions'].append('0.7入库区按75至78秒改为桥梁与交接轨道平行、三条交接轨道对齐三列开式立柱货架；升降转运框架按77秒外形补入。尺寸、16仓位长度与静态停放位置仍为估计，内部驱动未知。')


def refine_inspection_scene(scene):
    """Revision 0.4: common estimated inspection layout for the 30 s and 32 s cameras."""
    scene['visual_revision']='0.4'
    scene['instances']=[i for i in scene['instances'] if i['zone']!='inspection']
    scene['rail_lines']=[r for r in scene['rail_lines'] if r['zone']!='inspection']
    serial={}
    def put(asset,x,y,z=0):
        serial[asset]=serial.get(asset,0)+1
        scene['instances'].append({'id':f'inspection_r04_{asset}_{serial[asset]:03d}',
            'asset':asset,'pose':[x,y,z,0,0,0],'zone':'inspection',
            'source_segments':['S08','S09'],'geometry_source':'video_observed_form',
            'pose_source':'estimated','label_zh':asset})
    for lane,y in enumerate([50,56.5]):
        scene['rail_lines'].append({'id':f'inspection_{lane}','start':[7,y],'end':[55,y],
            'gauge_m':1.54,'source':'estimated','zone':'inspection'})
        for x in [8.5+3*i for i in range(16)]:put('inspection_track',x,y)
        for x,asset in [(21,'derust_hood'),(36,'magnetic_test_hood'),(44.5,'ultrasonic_test_hood')]:
            put(asset,x,y)
        put('inspection_stand',46.5,y)
        put('floor_control_box',52.5,y+2.1)
        for x in [10,13,16,27,30,31,32,35,42]:put('wheelset',x,y)
        # Roller radius .15 at z .42 and offset .22 => wheel hub about .99 m.
        put('wheelset',46.5,y,.43)
    put('inspection_crossbeam',24,49)
    for road in scene['agv_routes']:
        if road['id']=='dock_west_58':
            road['start']=[55,56.5];road['end']=[67,56.5]
        elif road['id']=='dock_west_50':road['start']=[55,50]
    if not any(c['id']=='inspection_exit' for c in scene['cameras']):
        scene['cameras'].insert(5,{'id':'inspection_exit','reference_second':32,
            'position':[48,22,27],'target':[48,53,.7],'horizontal_fov':.92,
            'width':1280,'height':720,'registration':'initial_estimate'})
    note='0.4探伤区按30与32秒共同调整：两线间距6.5m、超探X44.5m、出口托架X46.5m均为视觉估计；两台磁粉设备保持独立，不据投影重叠推断跨线连接。'
    if note not in scene['assumptions']:scene['assumptions'].append(note)


def refine_identification_scene(scene):
    scene['visual_revision']='0.5'
    scene['instances']=[i for i in scene['instances'] if not (
        (i['asset'] in {'identification_hood','overhead_robot'} and i['pose'][1]==21)
        or (i['asset']=='operator_console' and i['zone']=='teardown' and i['pose'][0]==12))]
    for item in scene['instances']:
        if item['asset']=='overhead_robot':
            lane=9 if item['pose'][1]<12 else 15
            item['pose']=[13.2,lane+1,1.46,0,0,math.pi/2 if lane==9 else -math.pi/2]
        elif item['asset']=='identification_bridge':item['pose'][:2]=[13.2,12]
        elif item['asset']=='track_segment' and item['zone']=='teardown' and item['pose'][0]<=19:
            item['asset']='inspection_track'
    if not any(c['id']=='identification_close' for c in scene['cameras']):
        scene['cameras'].insert(2,{'id':'identification_close','reference_second':4,
            'position':[21.5,0,8.5],'target':[13.8,12,1.0],'horizontal_fov':.92,
            'width':1280,'height':720,'registration':'initial_estimate'})
    note='0.5识别区依据2至5秒画面采用两套端罩与两台机械臂，第三条并行轨道保留；端罩位于轮对两侧，低位双轨桥架与机械臂为静态外形估计。'
    if note not in scene['assumptions']:scene['assumptions'].append(note)


def refine_visual_environment(scene):
    """Avoid invented tall walls obscuring cropped video views; preserve the estimated floor slab."""
    scene['visual_revision']='0.6'
    scene['boxes']=[box for box in scene['boxes'] if box['material']!='wall']
    note='当前视频帧未显示闭合厂房墙体或高隔断；这些构件此前属于场景估计，本版不再放入可见区域。厂房建筑边界仍未知，地面范围也仍为估计。'
    if note not in scene['assumptions']:scene['assumptions'].append(note)


def refine_unknown_scene(scene):
    """Revision 0.8: common low-machine / runway layout from the 34-42 s sequence."""
    scene['visual_revision']='0.8'
    offsets=scene.setdefault('layout_offsets',{})
    delta=-6.0-offsets.get('storage_y',0)
    for item in scene['instances']:
        if item['zone']=='storage':item['pose'][1]+=delta
    for camera in scene['cameras']:
        if camera['id'] in {'storage','storage_reverse'}:
            camera['position'][1]+=delta;camera['target'][1]+=delta
    offsets['storage_y']=-6.0
    for item in scene['instances']:
        if item['zone']=='inspection' and item['asset']=='magnetic_test_hood':item['pose'][0]=35.45
    scene['instances']=[i for i in scene['instances'] if i['zone']!='unknown']
    serial={}
    def put(asset,x,y,z=0,yaw=0):
        serial[asset]=serial.get(asset,0)+1
        scene['instances'].append({'id':f'unknown_r08_{asset}_{serial[asset]:03d}',
            'asset':asset,'pose':[x,y,z,0,0,yaw],'zone':'unknown','source_segments':['S10'],
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':asset})
    for x in [80,89,98,107,116,125]:
        put('unknown_cyan_cell',x,63)
        # Repeated groups are reconstructed as a static 3 x 3 queue; occupancy is estimated.
        for dx in [0,2.7,5.4]:
            for y in [54.8,56.25,57.7]:put('wheelset',x+dx,y,yaw=math.pi/2)
    put('unknown_runway',101,58.25)
    for x in [85,98.5,112]:put('unknown_bridge',x,58.25)
    for zone in scene['zones']:
        if zone['id']=='unknown':zone['bounds']=[72,50,131,67]
        elif zone['id']=='storage':zone['bounds']=[72,29,130,44]
    # Only low boundaries actually seen in wide shots; no inferred closed factory shell.
    scene['boxes']=[b for b in scene['boxes'] if not b['id'].startswith('r08_')]
    def box(key,center,size,material='wall'):
        scene['boxes'].append({'id':'r08_'+key,'center':center,'size':size,'material':material,
            'zone':'environment','source':'video_observed_form_dimensions_estimated','yaw':0})
    box('north_low_boundary',[69,67.1,.80],[126,.22,1.60])
    box('unknown_west_return',[72,62.1,.80],[.22,10,1.60])
    # Adjacent open enclosure at 32-35 s. Its purpose remains unknown.
    box('enclosure_front',[66.5,60,.75],[15,.18,1.5],'black')
    for x in [59,74]:box('enclosure_side_'+str(x),[x,63.3,.75],[.18,6.6,1.5],'black')
    for x,length in [(61.5,5),(70.5,7)]:box('enclosure_back_'+str(x),[x,66.6,.75],[length,.18,1.5],'black')
    for x,length in [(80,8),(105,38)]:box('unknown_front_'+str(x),[x,49.9,.75],[length,.18,1.5])
    # Low returns visible around the assembly/storage block; openings follow transfer lanes.
    for y0,y1 in [(3,5),(11,14),(20,23),(29,34)]:
        box('assembly_return_'+str(y0),[74.3,(y0+y1)/2,.75],[.18,y1-y0,1.5])
    scene['assumptions']=[a for a in scene['assumptions'] if not a.startswith(('当前视频帧未显示','0.8字幕缺失','34至42秒广角','0.8探伤','0.8跨区域'))]
    scene['assumptions'].append('34至42秒广角画面可见低矮深色边界和开口；仅补入可见围挡，厂房闭合建筑边界与地面真实尺寸仍未知。')
    scene['assumptions'].append('0.8字幕缺失设备区采用六台低矮青色机座、一对连续行走梁和三座橙色双梁桥架；轮对按六组3×3静态缓存估计，桥架采用36秒附近的一组位置快照。低围挡、机座、缓存数量与全部尺寸仍含估计，工序名称和内部驱动未确认。')
    scene['assumptions'].append('0.8探伤区两台磁粉机共同沿输送方向移动-0.55m，以减小32秒屋顶的共同偏移；该帧已用于几何调整，原屋顶检查不再视作未参与调整的检查点。')
    scene['assumptions'].append('0.8跨区域按36与43秒背景关系，将整个入库区沿Y平移-6m，并同步平移其近景相机；内部连接保持一致。此跨区域距离仍为人工估计，尚非全局多视图标定。')


def refine_opening_scene(scene):
    """Revision 0.9: low cap tools and neighbouring grey portals share one layout."""
    scene['visual_revision']='0.9'
    scene['instances']=[i for i in scene['instances'] if not i['id'].startswith('opening_r09_')]
    remove=[]
    for item in scene['instances']:
        if item['zone']!='teardown':continue
        x,y=item['pose'][:2]
        if item['asset'] in {'cap_portal','opening_portal'}:item['asset']='opening_portal';item['pose'][0]=31.15
        elif item['asset'] in {'overhead_gantry','parts_gantry'}:item['asset']='parts_gantry';item['pose'][:2]=[22.5,15]
        elif item['asset']=='industrial_arm' and abs(x-26.6)<.01:remove.append(item['id'])
        elif item['asset']=='parts_tray' and x==30.5:remove.append(item['id'])
        elif item['asset']=='operator_console' and x==27:remove.append(item['id'])
        elif item['asset'] in {'roller_stand','wheelset'} and x in [21,24.5]:remove.append(item['id'])
        elif item['asset']=='track_segment' and 19<x<39:item['asset']='inspection_track'
        elif item['asset']=='wheelset' and x==28:item['pose'][2]=.11
        elif item['asset']=='wheelset' and x==32:item['pose'][0]=30.1
        elif item['asset']=='wheelset' and x in [35,33.5]:item['pose'][0]=31.15;item['pose'][2]=.11
    scene['instances']=[i for i in scene['instances'] if i['id'] not in remove]
    serial={}
    def put(asset,x,y,yaw=0):
        serial[asset]=serial.get(asset,0)+1
        scene['instances'].append({'id':f'opening_r09_{asset}_{serial[asset]:03d}',
            'asset':asset,'pose':[x,y,0,0,0,yaw],'zone':'teardown','source_segments':['S04'],
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':asset})
    for y in [9,15,21]:
        put('opening_tool_station',28,y)
        for dy in [-1.45,1.45]:put('opening_robot',26,y+dy)
        put('opening_parts_bin',23.6,y-1.8)
        put('wheelset',33.25,y)
        put('exterior_inspection_rollers',36.5,y)
        scene['instances'][-1]['source_segments']=['S03']
        put('wheelset',36.5,y)
        scene['instances'][-1]['source_segments']=['S03']
        scene['instances'][-1]['pose'][2]=.135
    if not any(c['id']=='opening_wide' for c in scene['cameras']):
        index=next(n for n,c in enumerate(scene['cameras']) if c['id']=='opening')
        scene['cameras'].insert(index,{'id':'opening_wide','reference_second':8,
            'position':[41,0,15],'target':[30,12,1],'horizontal_fov':.92,
            'width':1280,'height':720,'registration':'initial_estimate'})
    scene['assumptions']=[a for a in scene['assumptions'] if not a.startswith('0.9开盖')]
    scene['assumptions'].append('0.9开盖区按8至13秒保留相邻灰色门架，另建轮对两端低位圆筒工具、两台机械臂和蓝色料筐；快照轮对间距2.1m、门架X31.15m及工具内部均为视觉估计。机械臂使用通用可编辑外形，无原厂运动学或连续动作。')


def refine_sorting_scene(scene):
    """Revision 0.9: replace independent shelves by the observed integrated sorting structure."""
    scene['visual_revision']='0.9'
    scene['instances']=[i for i in scene['instances'] if not i['id'].startswith('sorting_r09_') and not (
        i['zone']=='sorting' and i['asset'] in {'bearing_rack','industrial_arm','roller_conveyor','portal_press','bearing_demount_station'}) and not (
        i['zone']=='teardown' and i['asset']=='industrial_arm' and abs(i['pose'][0]-41.6)<.01)]
    scene['instances'].append({'id':'sorting_r09_system','asset':'bearing_sorting_system',
        'pose':[45.5,16,0,0,0,math.pi/2],'zone':'sorting','source_segments':['S07'],
        'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':'轴承退卸分拣整体结构'})
    scene['instances'].append({'id':'sorting_r09_robot','asset':'sorting_robot',
        'pose':[45.5,16,.74,0,0,math.pi/2],'zone':'sorting','source_segments':['S07'],
        'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':'分拣机械臂静态快照'})
    for n,y in enumerate([9,15,21]):
        scene['instances'].append({'id':f'sorting_r09_demount_{n}','asset':'bearing_demount_station',
            'pose':[43,y,0,0,0,0],'zone':'sorting','source_segments':['S07'],
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':'低位退卸站'})
    for item in scene['instances']:
        if item['zone']=='teardown' and item['asset']=='wheelset' and item['pose'][0]==43:item['pose'][2]=.11
    scene['assumptions']=[a for a in scene['assumptions'] if not a.startswith('0.9分拣')]
    scene['assumptions'].append('0.9分拣区按18至28秒将三个独立料架改为连续蓝色行走底架、开放料箱、端部双轨工作台、罩体与转运柱；机械臂停在底架中段。零件数量、长轴方向、与三条退卸线的连接、内部运动学均含估计。')


def refine_assembly_service_scene(scene):
    """Revision 0.9: separate visible portal / open wiping fixtures / cap-handling cell."""
    scene['visual_revision']='0.9'
    scene['instances']=[i for i in scene['instances'] if not i['id'].startswith('service_r09_') and not (
        i['zone']=='assembly' and (i['asset'] in {'wipe_head','grease_head','cap_portal'} or
        (i['asset']=='industrial_arm' and i['pose'][0]==109) or
        (i['asset']=='roller_conveyor' and i['pose'][0]==109)))]
    serial={}
    def put(asset,x,y,z=0,yaw=0,segments=('S11',)):
        serial[asset]=serial.get(asset,0)+1
        scene['instances'].append({'id':f'service_r09_{asset}_{serial[asset]:03d}',
            'asset':asset,'pose':[x,y,z,0,0,yaw],'zone':'assembly','source_segments':list(segments),
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':asset})
    for item in scene['instances']:
        if item['zone']!='assembly':continue
        x,y=item['pose'][:2]
        if item['asset']=='wheelset':
            if x==77:item['pose'][0]=76.5;item['pose'][2]=.11
            elif x==80:item['pose'][2]=.11
            elif x==83:item['pose'][0]=82.2
            elif x==110:item['pose'][2]=.11
        elif item['asset']=='track_segment' and (x<93 or 106<x<116):item['asset']='inspection_track'
    for y in [8,17,26]:
        put('opening_portal',76.5,y)
        put('wiping_station',80,y)
        put('wiping_station',86,y,segments=('S12',))
        put('opening_tool_station',110,y,segments=('S18',))
        put('closing_transfer_cell',108.2,y+.2,segments=('S18',))
        put('sorting_robot',109.9,y-.55,yaw=math.pi/2,segments=('S18',))
        put('industrial_arm',107.9,y+2.67,segments=('S18',))
    for camera in scene['cameras']:
        if camera['id']=='wiping' and camera['registration']=='initial_estimate':
            camera.update(position=[87,2,8],target=[80,9,.7],registration='manual_initial_estimate')
        elif camera['id']=='closing' and camera['registration']=='initial_estimate':
            camera.update(position=[113.5,10.5,6.8],target=[109.8,17,.8],registration='manual_initial_estimate')
    scene['assumptions']=[a for a in scene['assumptions'] if not a.startswith('0.9擦拭关盖')]
    scene['assumptions'].append('0.9擦拭关盖区按44至49、67至69秒拆分可见结构：擦拭门架与开放工具站分开；关盖使用低位圆筒工具、灰色独立柱梁、悬挂橙色工具、双机械臂及带定位杆料盘。复用通用工具外形，精确尺寸、内部机构、动作和跨区连接尚未恢复。')


def refine_component_storage_scene(scene):
    """Revision 0.10: distinguish bearing storage (43-54 s) from wheelset storage (75-78 s)."""
    if (scene.get('layout_offsets',{}).get('assembly_process_x')==35 and
            any(i['asset']=='bearing_storage_module' for i in scene['instances'])):return
    scene['visual_revision']='0.10'
    offsets=scene.setdefault('layout_offsets',{})
    dx=70-offsets.get('storage_x',0)
    for item in scene['instances']:
        if item['zone']=='storage':item['pose'][0]+=dx
    for camera in scene['cameras']:
        if camera['id'] in {'storage','storage_reverse'}:
            camera['position'][0]+=dx;camera['target'][0]+=dx
        elif camera['id']=='wiping':
            dy=26-offsets.get('wiping_lane_y',8)
            camera['position'][1]+=dy;camera['target'][1]+=dy
    offsets.update(storage_x=70,wiping_lane_y=26)
    # The 54 s close-up shows the dense rack's orange lift feeding a narrow blue bearing line.
    # It is not the large-pitch open wheelset rack shown at 75-78 s.
    scene['instances']=[i for i in scene['instances'] if not i['id'].startswith('component_r10_')]
    mx=42-offsets.get('metrology_x',0)
    for item in scene['instances']:
        if item['zone']=='metrology':item['pose'][0]+=mx
    offsets['metrology_x']=42
    serial={}
    def put(asset,x,y,z=0,yaw=0,segments=('S11','S14')):
        serial[asset]=serial.get(asset,0)+1
        scene['instances'].append({'id':f'component_r10_{asset}_{serial[asset]:03d}',
            'asset':asset,'pose':[x,y,z,0,0,yaw],'zone':'component_storage',
            'source_segments':list(segments),'geometry_source':'video_observed_form',
            'pose_source':'estimated','label_zh':asset})
    for y in [34,36.7,39.4]:
        for col in range(37):put('bearing_storage_module',68+col,y)
    put('bearing_store_lift',105.3,36.7)
    put('bearing_identification_line',110.5,36.7)
    put('industrial_arm',116,35.7,yaw=math.pi)
    put('floor_control_box',115.9,34.2,yaw=math.pi/2)
    # Outside the finishing lines, wheelsets wait on the floor before the separate wheelset store.
    for y in [8,17,26]:
        for x in [135,137.7,140.4]:put('wheelset',x,y,segments=('S17','S20'))
    # Enclosure frontage remains low and open at the lift/transfer points.
    scene['boxes']=[b for b in scene['boxes'] if not b['id'].startswith('component_r10_')]
    scene['boxes'].append({'id':'component_r10_back_boundary','center':[94,44,.75],
        'size':[56,.18,1.5],'material':'wall','zone':'environment',
        'source':'video_observed_form_dimensions_estimated','yaw':0})
    for b in scene['boxes']:
        if b['id']=='floor':b.update(center=[104,36,-.08],size=[208,72,.16])
    scene['bounds_m']=[208,72]
    scene['zones']=[z for z in scene['zones'] if z['id']!='component_storage']
    scene['zones'].append({'id':'component_storage','name_zh':'轴承料库 / 识别上料','bounds':[66,32.5,118,42]})
    for z in scene['zones']:
        if z['id']=='storage':z['bounds']=[142,27,200,45]
        elif z['id']=='metrology':z['bounds']=[118,31,136,37]
        elif z['id']=='sorting':z['bounds']=[42,6,50,27]
    for camera in scene['cameras']:
        if camera['id']=='top':camera.update(position=[104,36,245],target=[104,36,0])
        elif camera['id']=='overview_east':camera.update(position=[276,-133,175],target=[104,36,0])
    existing={c['id'] for c in scene['cameras']}
    for cid,second,eye,target in [
        ('component_overview',43,[107,-5,32],[87,27,0]),
        ('bearing_loading',54,[122,26,9],[110,36,1])]:
        if cid not in existing:scene['cameras'].append({'id':cid,'reference_second':second,
            'position':eye,'target':target,'horizontal_fov':.92,'width':1280,'height':720,
            'registration':'manual_initial_estimate'})
    # Remove the road through the unclassified enclosure, and keep the long store clear of paint.
    scene['agv_routes']=[r for r in scene['agv_routes'] if not r['id'].startswith('component_r10_')]
    for road in scene['agv_routes']:
        if road['id']=='visual_route_0':road['end']=[202,4]
        elif road['id']=='visual_route_1':road['end']=[64,37.5]
        elif road['id']=='visual_route_2':road['start']=[4,47.5];road['end']=[202,47.5]
        elif road['id']=='visual_route_4':road['end']=[67,58]
        elif road['id']=='visual_route_5':road['start']=[202,4];road['end']=[202,47.5]
        elif road['id']=='dock_east_53':road.update(start=[67,48],end=[85,48])
        elif road['id']=='dock_east_61':road.update(start=[67,56],end=[72,56])
    for cid,a,b in [('transfer_spine',[133,4],[133,47.5]),
                    ('store_front',[140,4],[140,47.5]),
                    ('store_cross',[133,36.4],[148,36.4])]:
        scene['agv_routes'].append({'id':'component_r10_'+cid,'start':a,'end':b,
            'width_m':3.5,'kind':'wheelset_carrier','source':'estimated','navigation_validated':False})
    # Vehicle and carried wheel must be translated together.
    moves={}
    for item in scene['instances']:
        if item['asset']=='wheelset_agv':
            p=item['pose'];old=tuple(p[:2]);new=None
            if p[0]==67 and p[1]==61:new=[67,56]
            elif p[0]==134:new=[133,23]
            elif p[0]==129 and p[1]==37.5:new=[140,36.4]
            if new:moves[old]=new;p[:2]=new
    for item in scene['instances']:
        if item['zone']=='logistics' and item['asset']=='wheelset' and item['pose'][2]>.3:
            new=moves.get(tuple(item['pose'][:2]))
            if new:item['pose'][:2]=new
    scene['assumptions']=[a for a in scene['assumptions'] if not a.startswith('0.10料库')]
    # The empty foreground rail pair and nearby yellow transfer boundaries are visible at 30-32 s.
    for col in range(16):
        put('inspection_track',8.5+3*col,43.5,segments=('S08',))
        scene['instances'][-1]['zone']='inspection'
    scene['rail_lines']=[r for r in scene['rail_lines'] if r['id']!='inspection_foreground_r10']
    scene['rail_lines'].append({'id':'inspection_foreground_r10','start':[7,43.5],'end':[55,43.5],
        'gauge_m':1.54,'source':'estimated','zone':'inspection'})
    for suffix,x in [('left',56.1),('right',59.6)]:
        scene['agv_routes'].append({'id':'component_r10_inspection_'+suffix,'start':[x,38],
            'end':[x,56.4],'width_m':3.5,'kind':'visible_transfer_boundary',
            'source':'video_observed_form_dimensions_estimated','navigation_validated':False})
    scene['assumptions'].append('0.10料库按43至54秒辨认为密集轴承部件架及末端升降/识别上料线，与75至78秒轮对立柱仓架分开。擦拭相机改取邻料库的Y=26线；轮对库移至涂漆后侧，新增末端缓存和转运开口。37格×3列、八层、全部跨区域距离与208×72m地面均为估计，未做全局标定、通行或动力学验收。')


def refine_transfer_blocks_scene(scene):
    """43 s shows short preparation lanes ending at a transfer area, not one continuous assembly track."""
    offsets=scene.setdefault('layout_offsets',{})
    if offsets.get('assembly_process_x')==35:return
    scene['instances']=[i for i in scene['instances'] if not (i['id'].startswith('component_r10_') and i['asset']=='wheelset')]
    updated=[]
    for item in scene['instances']:
        if item['zone'] in {'assembly','finishing'}:
            x=item['pose'][0]
            if item['asset'] in {'track_segment','inspection_track'}:continue
            if x<90:
                if item['asset']=='wiping_station' and x==86:continue
                if item['asset']=='wheelset' and x==86:continue
                item['zone']='preparation'
                if item['asset']=='wiping_station':item['source_segments']=['S11','S12']
            else:item['pose'][0]+=35
        updated.append(item)
    scene['instances']=updated
    serial={}
    def put(asset,x,y,z=0,yaw=0,zone='logistics',segments=('S17',)):
        serial[asset]=serial.get(asset,0)+1
        scene['instances'].append({'id':f'transfer_r10_{asset}_{serial[asset]:03d}',
            'asset':asset,'pose':[x,y,z,0,0,yaw],'zone':zone,'source_segments':list(segments),
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':asset})
    scene['rail_lines']=[r for r in scene['rail_lines'] if r['zone']!='assembly']
    for y in [8,17,26]:
        for col in range(4):put('inspection_track',73.5+3*col,y,zone='preparation',segments=('S11','S12'))
        put('wheelset',74,y,zone='preparation',segments=('S11','S12'))
        for col in range(13):put('inspection_track',126.5+3*col,y,zone='assembly',segments=('S16','S18','S19'))
        scene['rail_lines'].extend([
            {'id':f'preparation_r10_{y}','start':[72,y],'end':[84,y],'gauge_m':1.54,'source':'estimated','zone':'preparation'},
            {'id':f'assembly_r10_{y}','start':[125,y],'end':[164,y],'gauge_m':1.54,'source':'estimated','zone':'assembly'}])
        for col in range(11):put('wheelset',96+2.45*col,y,segments=('S13','S17'))
        for col in range(3):put('wheelset',179+2.7*col,y,segments=('S17','S20'))
    for camera in scene['cameras']:
        if camera['id'] in {'press','press_front','closing'}:
            camera['position'][0]+=35;camera['target'][0]+=35
    # Keep wheelset-store frame and its two camera fits together, away from the bearing store.
    for item in scene['instances']:
        if item['zone']=='storage':item['pose'][0]+=50
    for camera in scene['cameras']:
        if camera['id'] in {'storage','storage_reverse'}:
            camera['position'][0]+=50;camera['target'][0]+=50
        elif camera['id']=='top':camera.update(position=[129,36,295],target=[129,36,0])
        elif camera['id']=='overview_east':camera.update(position=[325,-155,206],target=[129,36,0])
    offsets.update(assembly_process_x=35,storage_x=120)
    scene['zones']=[z for z in scene['zones'] if z['id']!='preparation']
    scene['zones'].append({'id':'preparation','name_zh':'擦拭 / 涂脂 / 短轨道','bounds':[72,4,85,31]})
    for zone in scene['zones']:
        if zone['id']=='assembly':zone['bounds']=[125,4,151,31]
        elif zone['id']=='finishing':zone['bounds']=[151,4,165,31]
        elif zone['id']=='storage':zone['bounds']=[192,27,250,45]
    # Visual boundaries of preparation/cache transfer space from 43 s; dimensions still estimated.
    scene['agv_routes']=[r for r in scene['agv_routes'] if not r['id'].startswith('component_r10_store') and r['id']!='component_r10_transfer_spine']
    for road in scene['agv_routes']:
        if road['id']=='visual_route_0':road['end']=[252,4]
        elif road['id']=='visual_route_2':road['end']=[252,47.5]
        elif road['id']=='visual_route_4':road.update(start=[64.5,4],end=[64.5,58])
        elif road['id']=='visual_route_5':road.update(start=[252,4],end=[252,47.5])
        elif road['id'].startswith('dock_west_'):road['end'][0]=64.5
        elif road['id'].startswith('dock_east_'):road['start'][0]=64.5
    for key,a,b in [('prep_left',[87,4],[87,29]),('prep_right',[91,4],[91,29]),
                    ('finish_left',[170,4],[170,47.5]),('finish_right',[174,4],[174,47.5]),
                    ('store_left',[186,4],[186,47.5]),('store_right',[190,4],[190,47.5])]:
        scene['agv_routes'].append({'id':'transfer_r10_'+key,'start':a,'end':b,'width_m':3.5,
            'kind':'visual_transfer_boundary','source':'video_observed_form_dimensions_estimated','navigation_validated':False})
    for y in [8,17,26]:
        scene['agv_routes'].extend([
            {'id':f'transfer_r10_preparation_{y}','start':[84,y],'end':[123,y],'width_m':3.5,
             'kind':'handoff','source':'estimated','navigation_validated':False},
            {'id':f'transfer_r10_finishing_{y}','start':[164,y],'end':[198,y],'width_m':3.5,
             'kind':'handoff','source':'estimated','navigation_validated':False}])
    moves={}
    for item in scene['instances']:
        if item['asset']=='wheelset_agv':
            p=item['pose'];old=tuple(p[:2]);new=None
            if p[0]==67:new=[64.5,p[1]]
            elif p[0]==133:new=[172,23]
            elif p[0]==140:new=[188,36.4]
            elif p[0]==72:new=[89,17]
            if new:moves[old]=new;p[:2]=new
    for item in scene['instances']:
        if item['zone']=='logistics' and item['asset']=='wheelset' and item['pose'][2]>.3:
            new=moves.get(tuple(item['pose'][:2]))
            if new:item['pose'][:2]=new
    for b in scene['boxes']:
        if b['id']=='floor':b.update(center=[129,36,-.08],size=[258,72,.16])
    scene['bounds_m']=[258,72]
    scene['assumptions']=[a for a in scene['assumptions'] if not a.startswith('0.10料库')]
    scene['assumptions'].append('0.10料库：43至54秒的密集轴承部件架/末端升降/识别上料线与75至78秒轮对立柱仓架分开。密集架暂用37格×3列×八层；擦拭近景改取邻料库的Y=26线。全部尺寸仍为估计。')
    scene['assumptions'].append('0.10跨区：43秒可见擦拭/涂脂短轨道在黄色转运区前结束，旁边为轮对缓存；因此拆开此前连续到压装/关盖的长轨道。压装、关盖、涂漆与轮对库的跨区绝对坐标仍为设计估计；地面258×72m是容纳当前模型的画布，不代表从视频恢复了实测总平面。')


def refine_component_frontage_scene(scene):
    """0.11: correct front/rear rack landmark correspondence and the visible cache rows."""
    if scene.get('layout_offsets',{}).get('component_frontage_revision')==11:return
    scene['visual_revision']='0.11'
    # The 43 s rack top rear corner is not the top of its front upright.
    # One visible rack depth is retained; neither bay count nor dimensions are measured.
    scene['instances']=[i for i in scene['instances'] if not (
        (i['asset']=='bearing_storage_module' and (i['pose'][1]!=34 or i['pose'][0]>100)) or
        (i['id'].startswith('component_r10_') and i['asset'] in {'industrial_arm','floor_control_box'}) or
        (i['id'].startswith('transfer_r10_wheelset_') and i['pose'][0]<125))]
    for i in scene['instances']:
        if i['asset'] in {'bearing_store_lift','bearing_identification_line'}:
            i['pose'][0]-=1;i['pose'][1]-=1.7
    def put(asset,x,y,z=0,yaw=0,zone='component_storage',segments=('S14',)):
        scene['instances'].append({'id':f'frontage_r11_{asset}_{len(scene["instances"]):03d}',
            'asset':asset,'pose':[x,y,z,0,0,yaw],'zone':zone,'source_segments':list(segments),
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':asset})
    put('bearing_loading_robot',114.27335,34.91797,yaw=math.pi)
    put('bearing_loading_fixture',113.98987,33.32737)
    put('bearing_loading_tool_stand',116.09176,34.25827)
    for y in [26,21,16,11]:
        for col in range(11):put('wheelset',96.4+2*col,y,zone='logistics',segments=('S13','S17'))
    for i in scene['instances']:
        if (i['zone']=='logistics' and i['pose'][:2]==[89,17] and
                (i['asset']=='wheelset_agv' or (i['asset']=='wheelset' and i['pose'][2]>.3))):
            i['pose'][1]=11
    for b in scene['boxes']:
        if b['id']=='component_r10_back_boundary':b['center'][1]=38.5
    for x in [70,78,86,94,102,110,118]:
        scene['boxes'].append({'id':f'frontage_r11_wall_post_{x}','center':[x,38.36,.75],
            'size':[.22,.30,1.5],'material':'wall','zone':'environment',
            'source':'video_observed_form_dimensions_estimated','yaw':0})
    for c in scene['cameras']:
        if c['id']=='bearing_loading':
            c['position'][0]-=1;c['target'][0]-=1
            c['position'][1]-=1.7;c['target'][1]-=1.7
    # Four cache rows share the preparation area's transverse transfer markings.
    scene['agv_routes']=[r for r in scene['agv_routes'] if not r['id'].startswith('transfer_r10_preparation_')]
    for y in [26,21,16,11]:
        scene['agv_routes'].append({'id':f'frontage_r11_cache_{y}','start':[84,y],'end':[119,y],
            'width_m':3.5,'kind':'visual_transfer_boundary','source':'estimated','navigation_validated':False})
    for r in scene['agv_routes']:
        if r['id'] in {'transfer_r10_prep_left','transfer_r10_prep_right'}:
            r['start'][1]=11;r['end'][1]=26
    scene['layout_offsets']['component_frontage_revision']=11
    scene['assumptions'].append('0.11：43秒料架前侧立柱顶/底与后侧顶角重新区分，密集架改为33格、单排、八层估计外形；缓存按可见四行收密，节距2m和行距5m为同序列估计。54秒新增直立折叠臂、独立回转夹持柜和工具座，均为静态视觉快照，未实现关节控制。')


def refine_closing_side_scene(scene):
    """0.12: track is transverse in 67-69 s; correct the plate's axis correspondence."""
    if scene.get('layout_offsets',{}).get('closing_side_revision')==12:return
    scene['visual_revision']='0.12'
    for item in scene['instances']:
        if item['id'].startswith('service_r09_') and item['asset']=='sorting_robot':
            lane=item['pose'][1]+.55
            item['asset']='closing_primary_robot';item['pose']=[146.683,lane+2.4573,0,0,0,0]
        elif item['id'].startswith('service_r09_') and item['asset']=='industrial_arm':
            lane=item['pose'][1]-2.67
            item['asset']='closing_secondary_robot';item['pose']=[150.0842,lane+2.4001,0,0,0,0]
    scene['instances']=[i for i in scene['instances'] if not (
        i['zone']=='assembly' and ((i['asset']=='operator_console' and i['pose'][0]==147) or
        (i['asset']=='wheelset' and i['pose'][0]==150)))]
    for item in scene['instances']:
        if item['zone'] in {'assembly','finishing'} and item['pose'][0] in {154,160,163}:
            item['pose'][0]+=4
    for y in [8,17,26]:
        scene['instances'].append({'id':f'closing_r12_track_{y}','asset':'inspection_track',
            'pose':[165.5,y,0,0,0,0],'zone':'assembly','source_segments':['S18','S19'],
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':'后段轨道'})
    for rail in scene['rail_lines']:
        if rail['id'].startswith('assembly_r10_'):rail['end'][0]=168
    for road in scene['agv_routes']:
        if road['id'].startswith('transfer_r10_finishing_'):road['start'][0]=168
    for zone in scene['zones']:
        if zone['id']=='finishing':zone['bounds'][2]=169
    for y in [8,17,26]:
        scene['boxes'].append({'id':f'closing_r12_side_boundary_{y}',
            'center':[154.57,y+5, .75],'size':[.18,7.9,1.5],'yaw':.218,
            'material':'wall','zone':'environment','source':'video_observed_form_dimensions_estimated'})
    camera=next(c for c in scene['cameras'] if c['id']=='closing')
    x,y,z,yaw,pitch=[147.46847694335636,10.461616353873232,9.158545014929599,1.3424246575163494,.7097917439494226]
    camera.update(position=[x,y,z],target=[x+10*math.cos(yaw)*math.cos(pitch),
        y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)],
        registration='manual_sparse_landmark_fit')
    if not any(c['id']=='closing_wide' for c in scene['cameras']):
        scene['cameras'].append(dict(camera,id='closing_wide',reference_second=67))
    scene['layout_offsets']['closing_side_revision']=12
    scene['assumptions'].append('0.12关盖：67至69秒轨道横穿画面，原沿轨道观察及料盘长短轴对应错误已纠正。改为侧面观察，料盘后接黑色伸缩台；主柜、双臂和悬挂转运梁按同序列估计重建。机械臂姿态为68秒静态快照，零件尺寸、复制线数与跨区位置仍为估计，未实现动作或关节控制。')


def refine_sorting_parallel_scene(scene):
    """0.13: the robot base runs parallel to the tracks, with transverse overhead rails."""
    if scene.get('layout_offsets',{}).get('sorting_parallel_revision')==13:return
    scene['visual_revision']='0.13'
    for i in scene['instances']:
        if i['id']=='sorting_r09_system':i['pose']=[48,23.75,0,0,0,math.pi]
        elif i['id']=='sorting_r09_robot':
            i['asset']='sorting_pick_robot';i['pose']=[48.0536,23.3112,.74,0,0,math.pi]
        elif i['id']=='sorting_r09_demount_2':i['asset']='bearing_demount_frontage'
    # The 19-22 s sequence shows empty demount lanes after extraction; avoid invented queues.
    scene['instances']=[i for i in scene['instances'] if not (
        i['zone']=='teardown' and ((i['asset']=='wheelset' and i['pose'][0] in {43,48,52}) or
        (i['asset']=='parts_tray' and i['pose'][0]==38)))]
    for i in scene['instances']:
        if i['zone']=='teardown' and i['asset']=='track_segment' and 40<i['pose'][0]<51:
            i['asset']='inspection_track'
    for z in scene['zones']:
        if z['id']=='sorting':z['bounds']=[40,6,57,27]
    camera=next(c for c in scene['cameras'] if c['id']=='sorting')
    x,y,z,yaw,pitch=[52.80993884,35.82104038,15.06886290,-2.05075690,.78652996]
    camera.update(position=[x,y,z],target=[x+10*math.cos(yaw)*math.cos(pitch),
        y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)],registration='manual_sparse_landmark_fit',
        )
    scene['layout_offsets']['sorting_parallel_revision']=13
    scene['assumptions'].append('0.13分拣：18至24秒蓝色机器人底座与轮对轨道平行，原转90度跨线方向已纠正；灰色双轨梁单独跨线。料箱按可见箱口加宽，最近退卸柜与分拣台共用，移除原布局重叠柜和无依据的轮对队列。机器人采用21秒静态抓取姿态；料箱内容数量、跨区绝对坐标和尺寸仍为估计。')


def refine_opening_frontage_scene(scene):
    """Joint wheel/gate comparison at 8 s; retain the preparation-area asset."""
    if scene.get('layout_offsets',{}).get('opening_frontage_revision')==13:return
    for item in scene['instances']:
        if item['zone']=='teardown' and item['asset']=='opening_portal':
            item['asset']='opening_frontage_portal';item['pose'][0]=31.4971653
    camera=next(c for c in scene['cameras'] if c['id']=='opening_wide')
    x,y,z,yaw,pitch=[38.60028627,3.44107779,5.98758749,2.46615201,.541014]
    camera.update(position=[x,y,z],target=[x+10*math.cos(yaw)*math.cos(pitch),
        y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)])
    scene['layout_offsets']['opening_frontage_revision']=13
    scene['assumptions'].append('0.13开盖：8秒门架顶角与六轮心共同核对，降低门架并缩窄沿轨道方向的厚度；开盖采用独立资产，擦拭区不随之缩放。四个顶角人工读数误差约15px，几何和相机使用同一帧调整，无独立验证。8秒与10秒共用静态门架，轮对运动位置仍有差异。')


def refine_press_feed_scene(scene):
    """0.14: same longitudinal flow on both mirrored feeders, no 180-degree reversal."""
    if scene.get('layout_offsets',{}).get('press_feed_revision')==14:return
    for item in scene['instances']:
        if item['asset']=='bearing_feed_module':
            side=-1 if abs(item['pose'][5])>1 else 1
            lane=item['pose'][1]-side*1.6
            item['pose'][1]=lane+side*1.55;item['pose'][5]=0
            if side<0:item['asset']='bearing_feed_mirrored'
    camera=next(c for c in scene['cameras'] if c['id']=='press')
    camera.update(position=[128.8000437525813, 16.242380315087892, 3.563565482444545],target=[135.483830122578, 20.817871437818845, -2.300893635501169],registration='manual_sparse_landmark_fit')
    scene['visual_revision']='0.14'
    scene['layout_offsets']['press_feed_revision']=14
    scene['assumptions'].append('0.14压装上料：62秒两侧轴承输送线为镜像，原一侧旋转180度导致前后顺序相反已纠正。环件料仓移到窄输送线旁，补独立前端环件检查台；轴承直径、节距和位置按62秒估计调整。58秒移动轮对不能直接对应62秒静态位置，需与固定设备一起检查；全部尺寸为视频估计。')


def refine_finishing_scene(scene):
    """0.15: paired run-in fixtures and the low blue painting frame, shared by two views."""
    if scene.get('layout_offsets',{}).get('finishing_revision')==15:return
    scene['instances']=[i for i in scene['instances'] if not (
        i['zone']=='finishing' and i['asset']=='roller_stand')]
    for item in scene['instances']:
        if item['asset']=='spray_portal':item['pose'][0]=163.56491403612551
        elif item['asset']=='wheelset' and item['zone']=='assembly':
            if item['pose'][0]==158:item['pose'][2]=0
            elif item['pose'][0]==164:item['pose'][0]=162.2
    for y in [8,17,26]:
        for n,x in enumerate([158,156.04703562179583]):
            scene['instances'].append({'id':f'finishing_r15_run_in_{y}_{n}',
                'asset':'bearing_run_in_stand','pose':[x,y,0,0,0,0],'zone':'finishing',
                'source_segments':['S19'],'geometry_source':'video_observed_form',
                'pose_source':'estimated','label_zh':'轴承磨合端部工位'})
        scene['instances'].append({'id':f'finishing_r15_wheel_{y}',
            'asset':'wheelset','pose':[156.04703562179583,y,0,0,0,0],'zone':'finishing',
            'source_segments':['S19'],'geometry_source':'video_observed_form',
            'pose_source':'estimated','label_zh':'71秒磨合轮对静态快照'})
        scene['instances'].append({'id':f'finishing_r15_spray_wheel_{y}',
            'asset':'wheelset','pose':[164.3,y,0,0,0,0],'zone':'finishing',
            'source_segments':['S19'],'geometry_source':'video_observed_form',
            'pose_source':'estimated','label_zh':'涂漆工位轮对静态占位'})
    parameters={
        'run_in':(71,[154.6252942373212,29.409124224713796,10.801638562202715,-1.4075324584528015,.6763431529002795]),
        'painting':(73,[155.1110644945855,29.724702165688765,14.656683583366542,-.9152223447851617,.7099596407680488])}
    for cid,(sec,p) in parameters.items():
        x,y,z,yaw,pitch=p
        camera={'id':cid,'reference_second':sec,'position':[x,y,z],
            'target':[x+10*math.cos(yaw)*math.cos(pitch),y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)],
            'horizontal_fov':.92,'width':1280,'height':720,
            'registration':'manual_sparse_landmark_fit'}
        if not any(c['id']==cid for c in scene['cameras']):scene['cameras'].append(camera)
    scene['visual_revision']='0.15'
    scene['layout_offsets']['finishing_revision']=15
    scene['assumptions'].append('0.15磨合涂漆：71秒低位端驱/传感器为两个独立磨合工位，73秒为低蓝色横向门架、双侧工具与半透明挡板。以两个门架脚及四个磨合底座共同估计两相机、门架位置与磨合间距，非独立验收。两帧轮对在运动，当前保留一套静态占位，73秒磨合占用和门架前轮对数量仍与原片不同；全部尺寸、三条线复制和相邻黄色路网仍为估计。')


def refine_closing_fixed_scene(scene):
    """0.16: two supported closing/finishing lanes; shared fixed rail/tray alignment."""
    if scene.get('layout_offsets',{}).get('closing_fixed_revision')==16:return
    removed=[]
    def discard(item):
        x,y=item['pose'][:2]
        remove=(item['zone'] in {'assembly','finishing'} and x>=141 and y==26) or (
            item['id'].startswith('service_r09_') and item['asset']=='closing_transfer_cell' and y==26.2) or (
            item['id'].startswith('service_r09_') and item['asset'] in {'closing_primary_robot','closing_secondary_robot'} and y>27)
        if remove:removed.append(item['id'])
        return remove
    scene['instances']=[i for i in scene['instances'] if not discard(i)]
    scene['boxes']=[b for b in scene['boxes'] if b['id']!='closing_r12_side_boundary_26']
    for r in scene['rail_lines']:
        if r['id']=='assembly_r10_26':r['end'][0]=140
    for road in scene['agv_routes']:
        if road['id']=='transfer_r10_finishing_26':
            road['start'][0]=141;road['kind']='visible_transfer_boundary'
    for zone in scene['zones']:
        if zone['id']=='finishing':zone['bounds']=[151,4,169,22]
    cameras={
        'closing_wide':[146.73054808741148,9.020911254814834,9.946194030367428,1.222271360163813,.678215332855163],
        'closing':[147.6432963234173,9.424554087705413,8.3473271207142,1.386556238193035,.6159661825896456]}
    for c in scene['cameras']:
        if c['id'] in cameras:
            x,y,z,yaw,pitch=cameras[c['id']]
            c.update(position=[x,y,z],target=[x+10*math.cos(yaw)*math.cos(pitch),
                y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)])
        elif c['id'] in {'run_in','painting'}:
            c['position'][1]-=9;c['target'][1]-=9
    scene['visual_revision']='0.16'
    scene['layout_offsets']['closing_fixed_revision']=16
    scene['layout_offsets']['closing_removed_unsupported_instances']=removed
    scene['assumptions'].append('0.16末段：67至73秒可见两条关盖/磨合/涂漆线，移除无对应画面的Y=26第三复制线和其低隔墙，保留前段压装线；空区补接现有黄色转运边界。关盖看Y=17线、71/73秒从相反方向看Y=8线，两个区域共用Y=8/17两条线，不为单帧变更设备。关盖固定横梁端原混用了移动吊具位置，改读可见固定轨端，并与两帧料盘共同估计横梁高/前后位置、料盘宽度和两相机。全部为同序列估计；机器人运动姿态、柜体细节和跨区距离仍无独立验收。')


def refine_metrology_scene(scene):
    """0.17: replace the six generic S15 placeholders with the observed back-ring station."""
    if scene.get('layout_offsets',{}).get('metrology_revision')==17:return
    removed=[i['id'] for i in scene['instances'] if i['zone']=='metrology']
    scene['instances']=[i for i in scene['instances'] if i['zone']!='metrology']
    def put(asset,x,y,z=0,yaw=0):
        scene['instances'].append({'id':f'metrology_r17_{len(scene["instances"]):03d}',
            'asset':asset,'pose':[x,y,z,0,0,yaw],'zone':'metrology','source_segments':['S15'],
            'geometry_source':'video_observed_form','pose_source':'estimated',
            'label_zh':'后挡测量存取区 / 视频估计静态模型'})
    put('back_ring_storage_bed',125,34)
    for x,y in [(125.2175414,32.6094200),(124.2005888,32.5907251)]:put('back_ring_short_table',x,y)
    put('back_ring_electrical_cabinet',122.4362181,30.5648331)
    put('back_ring_measuring_column',124.1872668,30.7161367)
    put('back_ring_measuring_booth',122.4490212,31.9356813)
    put('back_ring_travelling_robot',125.1398406,33.32,.85)
    put('workshop_operator',121.9486917,32.5415040,yaw=.4)
    scene['boxes'].extend([
        {'id':'metrology_r17_robot_rail','center':[125,33.32,.79],'size':[4.7450389,.10,.16],
         'material':'blue','zone':'metrology','source':'video_observed_form_dimensions_estimated'},
        {'id':'metrology_r17_front_boundary','center':[124.6,35.0,.75],'size':[6.5,.18,1.5],
         'material':'wall','zone':'environment','source':'video_observed_form_dimensions_estimated'},
        {'id':'metrology_r17_end_boundary','center':[120.6,32,.75],'size':[.18,6.0,1.5],
         'material':'wall','zone':'environment','source':'video_observed_form_dimensions_estimated'}])
    for zone in scene['zones']:
        if zone['id']=='metrology':zone['bounds']=[120.7,29.8,127.7,35.4]
    x,y,z,yaw,pitch=[129.24540741,38.6508757,6.126769,-2.27245787,.64544979]
    scene['cameras'].append({'id':'back_ring','reference_second':55,'position':[x,y,z],
        'target':[x+10*math.cos(yaw)*math.cos(pitch),y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)],
        'horizontal_fov':.92,'width':1280,'height':720,'registration':'manual_sparse_landmark_fit'})
    scene['visual_revision']='0.17'
    scene['layout_offsets']['metrology_revision']=17
    scene['layout_offsets']['metrology_removed_placeholder_ids']=removed
    scene['assumptions'].append('0.17后挡区：55至57秒可见长蓝色储料床、双短料台、两台白柜、黑色回转测量罩和橙色沿床移动机械臂，替换六个通用占位。55秒镜头由储料床四角及假定1.7m人员共同估计，床约4.75×1.04m、环件数量/节距、柜高、人员和跨区关系均为外观估计；一角被字幕部分遮挡。相机与几何同序列调整，非独立验收；机械臂为55秒附近静态姿态，未实现测量/取放/移动。')


def refine_front_detail_scene(scene):
    """0.18: expose three missing early views and replace generic axle-check supports."""
    if scene.get('layout_offsets',{}).get('front_detail_revision')==18:return
    removed=[i['id'] for i in scene['instances'] if i['asset']=='roller_stand' and i['source_segments']==['S06']]
    scene['instances']=[i for i in scene['instances'] if i['id'] not in removed]
    for n,(asset,x,y) in enumerate([('axle_end_check_table',18.27072584,10.90453343),
        ('axle_end_check_table',18.12276634,12.56839902),('wheelset',20,9)]):
        scene['instances'].append({'id':f'front_r18_{n+1:03d}','asset':asset,'pose':[x,y,0,0,0,0],
            'zone':'teardown','source_segments':['S06'],'geometry_source':'video_observed_form',
            'pose_source':'estimated','label_zh':'轴端检查静态快照 / 尺寸位置估计'})
    poses={
        'exterior_check':(6,[43.41378382370463,.02599022007333815,6.138131966353262,2.2823428331324993,.4500985839254172]),
        'parts_transfer':(14,[42.125895271286346,5.324642466311754,17.22694556993202,2.6086837912957286,.6482246068215304]),
        'axle_check':(17,[15.91918417,2.57064849,7.6059029,.98994356,.73892899])}
    for cid,(sec,p) in poses.items():
        x,y,z,yaw,pitch=p
        scene['cameras'].append({'id':cid,'reference_second':sec,'position':[x,y,z],
            'target':[x+10*math.cos(yaw)*math.cos(pitch),y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)],
            'horizontal_fov':.92,'width':1280,'height':720,'registration':'manual_sparse_landmark_fit'})
    scene['visual_revision']='0.18'
    scene['layout_offsets']['front_detail_revision']=18
    scene['layout_offsets']['front_detail_removed_placeholder_ids']=removed
    scene['assumptions'].append('0.18前段：新增6秒外观检查、14秒料筐搬运和17秒轴端检查专用参考相机；以手读轮心、料筐中心和门架角点估计相机，轴端检查两台低矮电机/探头台替换三个通用轮托。工件为一个静态快照，前段搬运架、料筐与检查位置的跨镜头对应仍需联合核对，6秒放大圈为原视频叠加，不建成三维实体。')


def refine_reinspection_scene(scene):
    """33 s exit view and the visible exit workpieces, with source callout excluded."""
    if scene.get('layout_offsets',{}).get('reinspection_detail_revision')==18:return
    for n,(asset,x,y) in enumerate([('wheelset',52.5,50),('wheelset',52.5,56.5),('wheelset_agv',55.4,56.5)]):
        scene['instances'].append({'id':f'reinspection_r18_{n+1:03d}','asset':asset,'pose':[x,y,0,0,0,0],
            'zone':'inspection' if asset=='wheelset' else 'logistics','source_segments':['S09'],
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':'复探出口 / 静态工件快照'})
    x,y,z,yaw,pitch=[49.66488609,28.40662851,35.37372935,1.61171071,.94085082]
    scene['cameras'].append({'id':'reinspection','reference_second':33,'position':[x,y,z],
        'target':[x+10*math.cos(yaw)*math.cos(pitch),y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)],
        'horizontal_fov':.92,'width':1280,'height':720,'registration':'manual_sparse_landmark_fit'})
    scene['layout_offsets']['reinspection_detail_revision']=18
    scene['assumptions'].append('0.18复探：33秒两台超探机屋顶八角点估计相机，补入出口两轮对与上侧转运车的静态快照。放大圈为视频二维叠加，不复制为场景设备。出口轮对正在移动，30/32/33秒占用位置无法由同一静态快照逐帧复现；固定设备和道路仍共用同一布局。')


def refine_front_shared_scene(scene):
    """0.19: correct the shared crane/bin layout and cross-region occlusions."""
    if scene.get('layout_offsets',{}).get('front_shared_revision')==19:return
    from .front_geometry import (GANTRY_CENTER, BIN_FIRST_CENTER, BIN_PITCH,
        AXLE_CHECK_X, IDENTIFICATION_X_OFFSET, SORTING_X_OFFSET)
    removed=[]
    for item in scene['instances']:
        p=item['pose']
        if item['asset']=='parts_gantry':p[:2]=GANTRY_CENTER
        elif item['asset']=='opening_parts_bin':
            n=int(item['id'].rsplit('_',1)[1])-1
            p[:2]=[BIN_FIRST_CENTER[0],BIN_FIRST_CENTER[1]+n*BIN_PITCH]
        elif item['id']=='front_r18_001':p[:2]=[17.65590993,11.56629733]
        elif item['id']=='front_r18_002':p[:2]=[17.64091448,13.65443556]
        elif item['id']=='front_r18_003':p[0]=AXLE_CHECK_X
        elif item['source_segments']==['S02'] or (
            item['zone']=='teardown' and item['asset']=='wheelset' and p[0]==14):
            p[0]+=IDENTIFICATION_X_OFFSET
        elif item['zone']=='sorting' or (
            item['zone']=='teardown' and item['asset']=='operator_console' and p[0]==42):
            p[0]+=SORTING_X_OFFSET
        elif item['zone']=='teardown' and item['asset']=='wheelset' and p[0] in {18,56,59}:
            removed.append(item['id'])
    scene['instances']=[i for i in scene['instances'] if i['id'] not in removed]
    for n,(x,y) in enumerate([(21.78390848,11.22795469),(22.05321824,7.41395224),(19.25652796,6.76655155)]):
        scene['instances'].append({'id':f'front_r19_control_{n+1}','asset':'floor_control_box',
            'pose':[x,y,0,0,0,0],'zone':'teardown','source_segments':['S06'],
            'geometry_source':'video_observed_form','pose_source':'estimated',
            'label_zh':'轴端检查地面控制箱 / 位置为视频估计'})
    for zone in scene['zones']:
        if zone['id']=='sorting':zone['bounds']=[48,6,65,27]
    poses={
        'parts_transfer':[37.36852241636513,2.670973918092781,16.472154425787235,2.4898409070215117,.7074301339128543],
        'axle_check':[13.872068171440011,1.4147318835092249,9.610916990588843,.9067995619506829,.7440991600624467]}
    for camera in scene['cameras']:
        cid=camera['id']
        if cid in poses:
            x,y,z,yaw,pitch=poses[cid]
            camera.update(position=[x,y,z],target=[x+10*math.cos(yaw)*math.cos(pitch),
                y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)])
        elif cid in {'identification','identification_close','sorting'}:
            dx=SORTING_X_OFFSET if cid=='sorting' else IDENTIFICATION_X_OFFSET
            camera['position'][0]+=dx;camera['target'][0]+=dx
    scene['visual_revision']='0.19'
    scene['layout_offsets'].update(front_shared_revision=19,
        identification_x=IDENTIFICATION_X_OFFSET,sorting_x=SORTING_X_OFFSET,
        front_shared_removed_queue_ids=removed)
    scene['assumptions'].append('0.19前段共享布局：14秒原近端顶角混用了端横撑，改用四个固定纵梁端角和四个轮心共同估计，搬运架横向半跨约1.09m、纵向半跨约7.94m。料筐沿原三条线保持6m共同节距；17秒改读近筐四个口沿角点，与轮心估计同一场景相机，补三只地面控制箱。识别组连同其两相机平移2.4m，分拣组连同相机平移8m，去除X18、56、59的无对应静态队列；这些跨区域距离均为视频外观估计，非实测或独立验收。17秒左梁身份仍待确认，当前识别梁与机械臂未完全吻合。所有相机共用固定布局，不逐镜头移动设备；工件快照及机器人外形仍有差异。')


def refine_transfer_detail_scene(scene):
    """0.20: distinguish the short press-exit cache from long preparation queues."""
    if scene.get('layout_offsets',{}).get('transfer_detail_revision')==20:return
    from .transfer_geometry import (CLOSING_BLOCK_X_OFFSET as dx, CACHE_FIRST_X,
        CACHE_FIRST_Y, CACHE_PITCH, CACHE_ROW_PITCH, CACHE_COUNTS, CACHE_CAMERAS,
        GREASE_CAMERA, ROOM_CAMERA, ROOM_Y_OFFSET as room_dy, PRESS_LANE_Y_OFFSETS,
        CLOSING_LANE_Y_OFFSETS)
    removed=[]
    for item in scene['instances']:
        p=item['pose']
        # A transfer area separates press exits from the closing/finishing block.
        if item['zone']=='assembly' and item['asset']=='inspection_track' and p[0]>=140:
            removed.append(item['id'])
        elif item['zone']=='logistics' and item['asset']=='wheelset' and 179<=p[0]<=185:
            removed.append(item['id'])
        elif (item['zone']=='assembly' and p[0]>=141) or item['zone']=='finishing':
            p[0]+=dx
            lane=min(CLOSING_LANE_Y_OFFSETS,key=lambda y:abs(p[1]-y))
            p[1]+=CLOSING_LANE_Y_OFFSETS[lane]
        elif item['zone']=='assembly':
            lane=min(PRESS_LANE_Y_OFFSETS,key=lambda y:abs(p[1]-y))
            p[1]+=PRESS_LANE_Y_OFFSETS[lane]
    scene['instances']=[i for i in scene['instances'] if i['id'] not in removed]
    def put(asset,x,y,z=0,zone='logistics',segments=('S17',),yaw=0):
        scene['instances'].append({'id':f'transfer_r20_{asset}_{len(scene["instances"]):03d}',
            'asset':asset,'pose':[x,y,z,0,0,yaw],'zone':zone,'source_segments':list(segments),
            'geometry_source':'video_observed_form','pose_source':'estimated',
            'label_zh':'视频估计静态外观 / '+asset})
    for row,count in enumerate(CACHE_COUNTS):
        y=CACHE_FIRST_Y+row*CACHE_ROW_PITCH
        for col in range(count):put('wheelset',CACHE_FIRST_X+col*CACHE_PITCH,y)
    # Reuse the upstream press, rails and exit wheel at X138; a second portal here
    # would duplicate the same visible machine and intersect its body.
    for y in [11.3,17.3]:
        put('wheelset',136.3,y,zone='press_exit')
        put('axle_end_check_table',136.4,y-3.2,zone='press_exit')
    for y in [8,17]:
        for n in range(10):put('inspection_track',139.5+dx+3*n,y+CLOSING_LANE_Y_OFFSETS[y],zone='assembly',segments=('S18','S19'))
    put('wheelset_agv',139.4,11.2)
    put('wheelset_agv',140.4,17.0)
    put('wheelset_agv',150.92,10.74)
    put('wheelset',150.92,10.74,z=.34)
    put('transfer_auxiliary_cell',75.97,40.73+room_dy,zone='auxiliary_transfer',segments=('S10','S13'))
    scene['boxes']=[b for b in scene['boxes'] if b['id']!='r08_unknown_front_80']
    for b in scene['boxes']:
        if b['id'].startswith('closing_r12_'):
            b['center'][0]+=dx
            b['center'][1]+=CLOSING_LANE_Y_OFFSETS[int(b['id'].rsplit('_',1)[-1])]
    def box(name,center,size,material='wall'):
        scene['boxes'].append({'id':'transfer_r20_'+name,'center':center,'size':size,
            'material':material,'zone':'environment','source':'video_observed_form_dimensions_estimated'})
    box('cache_wall_near',[140,5.1,.75],[.18,4.3,1.5])
    box('cache_wall_mid',[140,14.05,.75],[.18,3.1,1.5])
    box('cache_wall_far',[140,22.0,.75],[.18,6.5,1.5])
    box('cache_back',[143.6,22.2,.75],[11.6,.18,1.5])
    for x in [138,144,149.4]:box('cache_back_post_'+str(x),[x,22.2,.75],[.26,.30,1.5])
    box('room_left',[74.3,39+room_dy,.75],[.18,8,1.5])
    box('room_back',[78.3,43+room_dy,.75],[8,.18,1.5])
    for suffix,point in [('a',[74.3,35,.75]),('b',[74.3,43,.75]),('c',[82.3,43,.75])]:
        point[1]+=room_dy
        box('room_post_'+suffix,point,[.26,.30,1.5])
    for r in scene['rail_lines']:
        if r['id'].startswith('assembly_r10_'):
            r['end'][0]=140
            lane=min(PRESS_LANE_Y_OFFSETS,key=lambda y:abs(r['start'][1]-y))
            r['start'][1]+=PRESS_LANE_Y_OFFSETS[lane];r['end'][1]+=PRESS_LANE_Y_OFFSETS[lane]
    for y in [8,17]:scene['rail_lines'].append({'id':f'closing_r20_{y}',
        'start':[138+dx,y+CLOSING_LANE_Y_OFFSETS[y]],'end':[168+dx,y+CLOSING_LANE_Y_OFFSETS[y]],'gauge_m':1.54,'zone':'assembly','source':'estimated'})
    for road in scene['agv_routes']:
        if road['id'] in {'transfer_r10_finish_left','transfer_r10_finish_right'}:
            road['start'][0]+=dx;road['end'][0]+=dx
        elif road['id'] in {'transfer_r10_finishing_8','transfer_r10_finishing_17'}:
            road['start'][0]+=dx
            lane=int(road['id'].rsplit('_',1)[-1])
            road['start'][1]+=CLOSING_LANE_Y_OFFSETS[lane];road['end'][1]+=CLOSING_LANE_Y_OFFSETS[lane]
        elif road['id']=='visual_route_2':road['end'][0]=74.0
        elif road['id']=='dock_east_53':road['end'][0]=74.0
    def road(name,start,end):
        scene['agv_routes'].append({'id':'transfer_r20_'+name,'start':start,'end':end,'width_m':3.5,
            'kind':'visible_transfer_boundary','source':'video_observed_form_dimensions_estimated','navigation_validated':False})
    road('north_route_after_room',[83,47.5],[252,47.5])
    for row in range(4):
        y=CACHE_FIRST_Y+row*CACHE_ROW_PITCH;road('cache_row_'+str(row),[140.4,y],[155.5,y])
    for x in [141.8,151.2]:road('cache_cross_'+str(x),[x,5.5],[x,19])
    road('room_spine_a',[67,34+room_dy],[67,47+room_dy])
    road('room_spine_b',[70.5,34+room_dy],[70.5,47+room_dy])
    for y in [35,39,43,47]:road('room_cross_'+str(y),[64.5,y+room_dy],[70.5,y+room_dy])
    for c in scene['cameras']:
        if c['id'] in {'closing','closing_wide','run_in','painting'}:
            c['position'][0]+=dx;c['target'][0]+=dx
            lane=17 if c['id'].startswith('closing') else 8
            c['position'][1]+=CLOSING_LANE_Y_OFFSETS[lane];c['target'][1]+=CLOSING_LANE_Y_OFFSETS[lane]
        elif c['id'] in {'press','press_front'}:
            c['position'][1]+=PRESS_LANE_Y_OFFSETS[17];c['target'][1]+=PRESS_LANE_Y_OFFSETS[17]
    for cid,(sec,p) in dict(CACHE_CAMERAS,greasing=(48,GREASE_CAMERA),cross_transfer=(50,ROOM_CAMERA)).items():
        x,y,z,yaw,pitch=p
        scene['cameras'].append({'id':cid,'reference_second':sec,'position':[x,y,z],
            'target':[x+10*math.cos(yaw)*math.cos(pitch),y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)],
            'horizontal_fov':.92,'width':1280,'height':720,'registration':'manual_sparse_landmark_fit'})
    for z in scene['zones']:
        if z['id']=='finishing':z['bounds']=[151+dx,4,169+dx,22]
        elif z['id']=='assembly':z['bounds']=[125,4,151+dx,31]
    scene['zones'].append({'id':'press_exit','name_zh':'压装出口 / 独立短缓存','bounds':[133,4,155.8,25.5]})
    scene['zones'].append({'id':'auxiliary_transfer','name_zh':'跨区侧室 / 未知小设备','bounds':[74,34.5+room_dy,83,43.5+room_dy]})
    scene['layout_offsets'].update(transfer_detail_revision=20,closing_block_x=dx,
        press_reference_y=PRESS_LANE_Y_OFFSETS[17],closing_reference_y=CLOSING_LANE_Y_OFFSETS[17],
        finishing_reference_y=CLOSING_LANE_Y_OFFSETS[8],auxiliary_room_y=room_dy,transfer_detail_removed_ids=removed)
    scene['visual_revision']='0.20'
    scene['assumptions'].append('0.20转运区：43秒料库旁长队列与65/66秒压装出口短队列并非同一缓存，新增独立四行短缓存，5/5/5/4轮对为静态占用估计。两视角轮心共同估计节距约1.068m、行距约2.696m，轮径沿用假定0.92m基准；未获实测。关盖/磨合/涂漆整体连同相机后移15m，为可见转运区留空；压装线共同改为约6m行距，并复用原主机，取消重复门架和轨道。电机台暂用已有外形，数量和连接待校核。48秒从擦拭工位反侧观察，未将前景原门架重复建成新机。50秒侧室位于料库北侧，新增小柜与窄辊道，工序未知；围挡高1.5m、房间8m尺度和全部跨区坐标为估计，不将该设备称为54秒轴承识别线。所有镜头使用同一固定布局；新增专用对照不表示工序视觉通过。')
    scene['assumptions'].append('0.20末段两线由9m间距改为估计6.8m，关盖与磨合/涂漆对应相机同步平移，保留局部投影关系；用于66秒短缓存反向背景的共享连接检查，仍非实测尺寸。')


def refine_preparation_scale_scene(scene):
    """0.21: use paired wheel hubs as scale constraints in the 43 s wide view."""
    if scene.get('layout_offsets',{}).get('preparation_scale_revision')==21:return
    from .component_geometry import (LONG_CACHE_FIRST_X, LONG_CACHE_FIRST_Y,
        LONG_CACHE_PITCH, LONG_CACHE_ROW_PITCH, LONG_CACHE_COUNTS, RACK_START,
        RACK_BAY_PITCH, RACK_FRONT_Y, ROAD_X, COMPONENT_CAMERA, PREPARATION_DX,
        PREPARATION_TOP_Y, COMPONENT_END_DX, COMPONENT_END_DY, preparation_y)
    racks=sorted([i for i in scene['instances'] if i['asset']=='bearing_storage_module'],key=lambda i:i['pose'][0])
    for n,item in enumerate(racks):
        item['pose'][:2]=[RACK_START+(n+.5)*RACK_BAY_PITCH,RACK_FRONT_Y+1.16]
    # Preserve the locally fitted 45/48 s geometry while moving the three lanes
    # together into the wide view. Their separate camera poses move with them.
    for i in scene['instances']:
        p=i['pose']
        if i['zone']=='preparation':
            p[0]+=PREPARATION_DX;p[1]=preparation_y(p[1])
        elif i['zone']=='component_storage' and i['asset']!='bearing_storage_module':
            p[0]+=COMPONENT_END_DX;p[1]+=COMPONENT_END_DY
    removed=[i['id'] for i in scene['instances'] if i['id'].startswith('frontage_r11_wheelset_')]
    scene['instances']=[i for i in scene['instances'] if i['id'] not in removed]
    def put(asset,x,y,z=0):
        scene['instances'].append({'id':f'component_r21_{asset}_{len(scene["instances"]):03d}',
            'asset':asset,'pose':[x,y,z,0,0,0],'zone':'logistics','source_segments':['S13'],
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':'准备区长缓存 / '+asset})
    for row,count in enumerate(LONG_CACHE_COUNTS):
        y=LONG_CACHE_FIRST_Y-row*LONG_CACHE_ROW_PITCH
        for col in range(count):put('wheelset',LONG_CACHE_FIRST_X+col*LONG_CACHE_PITCH,y,.34 if row==col==0 else 0)
    put('wheelset_agv',LONG_CACHE_FIRST_X,LONG_CACHE_FIRST_Y)
    for i in scene['instances']:
        if i['zone']=='logistics' and i['pose'][:2]==[89,11] and i['asset'] in {'wheelset','wheelset_agv'}:
            i['pose'][:2]=[91.5,LONG_CACHE_FIRST_Y-3*LONG_CACHE_ROW_PITCH]
    for b in scene['boxes']:
        if b['id']=='component_r10_back_boundary' or b['id'].startswith('frontage_r11_wall_post_'):
            b['center'][1]+=COMPONENT_END_DY
    for rail in scene['rail_lines']:
        if rail['id'].startswith('preparation_r10_'):
            for key in ['start','end']:
                rail[key][0]+=PREPARATION_DX;rail[key][1]=preparation_y(rail[key][1])
    for r in scene['agv_routes']:
        if r['id'].startswith('frontage_r11_cache_'):
            old=int(r['id'].rsplit('_',1)[-1]);y=LONG_CACHE_FIRST_Y-(26-old)/5*LONG_CACHE_ROW_PITCH
            r['start']=[84+PREPARATION_DX,y];r['end']=[119,y]
        elif r['id'] in {'transfer_r10_prep_left','transfer_r10_prep_right'}:
            x=ROAD_X[0 if r['id'].endswith('left') else 1]
            r['start']=[x,LONG_CACHE_FIRST_Y-3*LONG_CACHE_ROW_PITCH];r['end']=[x,LONG_CACHE_FIRST_Y]
    for c in scene['cameras']:
        if c['id'] in {'wiping','greasing'}:
            for key in ['position','target']:
                c[key][0]+=PREPARATION_DX;c[key][1]+=PREPARATION_TOP_Y-26
        elif c['id']=='bearing_loading':
            for key in ['position','target']:
                c[key][0]+=COMPONENT_END_DX;c[key][1]+=COMPONENT_END_DY
        elif c['id']=='component_overview':
            x,y,z,yaw,pitch=COMPONENT_CAMERA
            c.update(position=[x,y,z],target=[x+10*math.cos(yaw)*math.cos(pitch),
                y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)])
    for z in scene['zones']:
        if z['id']=='preparation':z['bounds']=[74.5,6,88.5,30]
        elif z['id']=='component_storage':z['bounds']=[74,30,117,37]
    scene['layout_offsets'].update(preparation_scale_revision=21,
        preparation_x=PREPARATION_DX,preparation_top_y=PREPARATION_TOP_Y-26,
        component_end_x=COMPONENT_END_DX,component_end_y=COMPONENT_END_DY,
        preparation_scale_removed_ids=removed)
    scene['visual_revision']='0.21'
    scene['assumptions'].append('0.21准备区：43秒原有稀疏中心检查未约束轮径/轴距，改用14个成对轮心、6个地面交点和3个料架立柱点共同估计相机及固定布局。轮对跨距1.54m和轮面0.92m仍是假定尺度；长缓存节距约1.154m、行距约3.811m，料架长约24.43m、高约4.36m、33格八层为估计。每行18轮对用于覆盖画面可见队列，画外总数未知；首行首轮对位于静态转运车上。三条准备线及45/48秒相机共同移动，料库末端和54秒相机共同移动，不按单张图片隐藏设备。所有读数来自同一方案动画，非独立视觉验收，工件运动快照和机械臂/料架内部结构仍简化。')


def refine_greasing_silhouette_scene(scene):
    """Correct the off-axis 48 s hub reading; also constrain wheel silhouettes."""
    if scene.get('layout_offsets',{}).get('greasing_silhouette_revision')==21:return
    from .component_geometry import GREASE_CAMERA, PREPARATION_DX, PREPARATION_TOP_Y
    x,y,z,yaw,pitch=GREASE_CAMERA
    x+=PREPARATION_DX;y+=PREPARATION_TOP_Y-26
    camera=next(c for c in scene['cameras'] if c['id']=='greasing')
    camera.update(position=[x,y,z],target=[x+10*math.cos(yaw)*math.cos(pitch),
        y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)],
        registration='same_video_hubs_rims_and_roofs_joint_estimate')
    scene['layout_offsets']['greasing_silhouette_revision']=21
    scene['assumptions'].append('0.21涂脂镜头：旧48秒近轮心[475,438]偏离轴心，改读[459,418]，远轮心[709,313]；连同两轮外轮廓和柜顶共同估计相机及左右柜偏置。柜体并非强制镜像，45/48秒仍复用同一固定工位；轮廓边界读数、轮径和柜高来自估计，旧点误差与新轮廓加权残差不可直接比较，没有独立检查点。')


def refine_metrology_press_connection_scene(scene):
    """Place the existing back-ring cell beside the existing press lanes (55 s)."""
    if scene.get('layout_offsets',{}).get('metrology_press_revision')==21:return
    from .component_geometry import METROLOGY_DX as dx, METROLOGY_DY as dy
    for i in scene['instances']:
        if i['zone']=='metrology':i['pose'][0]+=dx;i['pose'][1]+=dy
    for b in scene['boxes']:
        if b['id'].startswith('metrology_r17_'):b['center'][0]+=dx;b['center'][1]+=dy
    for c in scene['cameras']:
        if c['id']=='back_ring':
            for key in ['position','target']:c[key][0]+=dx;c[key][1]+=dy
    for z in scene['zones']:
        if z['id']=='metrology':
            x0,y0,x1,y1=z['bounds'];z['bounds']=[x0+dx,y0+dy,x1+dx,y1+dy]
    scene['layout_offsets'].update(metrology_press_revision=21,metrology_press_x=dx,metrology_press_y=dy)
    scene['assumptions'].append('0.21后挡与压装衔接：后挡整组设备、低墙及55秒相机同步平移约(+6.36,-4.37)m，保留局部料床/人员投影，让已有第三条压装线出现在左后方，不复制背景主机。相邻压装轮心约[55,155]只是同视频跨区位置估计；压装背景数量、主机外形和支承仍需核对，非实测总平面。17秒左侧梁架按方向和3/4秒连续结构对应既有识别桥架，运动中的识别臂仍采用早期静态姿态，未为17秒单独移动或隐藏机械臂。')


def refine_preparation_portal_and_robot_scene(scene):
    """0.22 roof-based preparation portal and explicit 4 s robot snapshot."""
    if scene.get('layout_offsets',{}).get('preparation_portal_revision')==22:return
    from .component_geometry import PREPARATION_PORTAL_DX as dx,PREPARATION_PORTAL_DY as dy
    for i in scene['instances']:
        if i['zone']=='preparation' and i['asset']=='opening_portal':
            i['pose'][0]+=dx;i['pose'][1]+=dy
    scene['layout_offsets']['preparation_portal_revision']=22
    scene['visual_revision']='0.22'
    scene['assumptions'].append('0.22识别臂末端分段：4秒最后读点是细探针尖端，粗腕壳在约43%处结束，叉指延至78%，余段为细探针。探针尖端高1.38m，相对1.325m罩顶高55mm；仅为统一静态网格的几何间距检查，不代表运动、接触或动力学验证。')
    scene['assumptions'].append('0.22固定门架：45秒四个可见顶盖角点估计约2.19m顶高、4.05m跨度和0.94m厚度，准备区三座门架共同移动约(+0.708,-0.501)m，替代从压机统一缩放的外形。端驱保持0.67m轴心高，增加顶盖检修孔、吊环、面板和接缝。45秒顶角仍有约20px残差，48秒固定边缘和运动中的前景轮对仍不同时匹配；两镜头共同检查的候选未获一致拟合，不作为通过。识别机器人使用4秒静态姿态补后部齿轮箱、宽肩、关节和长探针；3/4/17秒机器人在移动，一套静态姿态不代表动作复现。')


def refine_preparation_fixed_scene(scene):
    """0.23: distinct stations either side of the portal, tied by three views."""
    if scene.get('layout_offsets',{}).get('preparation_fixed_revision')==23:return
    from .component_geometry import (WIPING_CAMERA_23,PREPARATION_PORTAL_CENTER_23,
        PREPARATION_PORTAL_SNAPSHOT_X,PREPARATION_FRONT_SNAPSHOT_X,GREASING_CAMERA_23,
        GREASING_CENTER_X_23,GREASING_EXIT_WHEEL)
    portals=[i for i in scene['instances'] if i['zone']=='preparation' and i['asset']=='opening_portal']
    target_x,target_y=PREPARATION_PORTAL_CENTER_23
    shift_y=target_y-max(i['pose'][1] for i in portals)
    offset=scene['layout_offsets']['preparation_x']
    for i in portals:i['pose'][0]=target_x;i['pose'][1]+=shift_y
    for i in scene['instances']:
        if i['zone']!='preparation' or i['asset']!='wheelset':continue
        for old,new in [(76.5,PREPARATION_PORTAL_SNAPSHOT_X),(82.2,PREPARATION_FRONT_SNAPSHOT_X)]:
            if abs(i['pose'][0]-old-offset)<1e-6:
                i['pose'][0]=new+offset
                i['static_snapshot_reference_second']=45
    wipes=sorted((i for i in scene['instances'] if i['zone']=='preparation' and i['asset']=='wiping_station'),key=lambda i:i['pose'][1])
    for n,wipe in enumerate(wipes,1):
        y=wipe['pose'][1]
        for asset,z,yaw in [('greasing_station',0,math.pi),('wheelset',.11,0)]:
            scene['instances'].append({'id':f'greasing_r23_{asset}_{n}','asset':asset,
                'pose':[GREASING_CENTER_X_23,y,z,0,0,yaw],'zone':'preparation',
                'source_segments':['S11','S12'],'geometry_source':'video_observed_form',
                'pose_source':'estimated','label_zh':'独立低位涂脂工位' if asset=='greasing_station' else '涂脂工位轮对静态快照'})
    scene['instances'].append({'id':'greasing_r23_portal_exit_wheel','asset':'wheelset',
        'pose':GREASING_EXIT_WHEEL+[0,0,0],'zone':'preparation','source_segments':['S12'],
        'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':'48秒门架出口可见轮对静态快照'})
    for cid,pose in [('wiping',WIPING_CAMERA_23),('greasing',GREASING_CAMERA_23)]:
        x,y,z,yaw,pitch=pose
        camera=next(c for c in scene['cameras'] if c['id']==cid)
        camera.update(position=[x,y,z],target=[x+10*math.cos(yaw)*math.cos(pitch),
            y+10*math.sin(yaw)*math.cos(pitch),z-10*math.sin(pitch)],
            registration='same_video_three_view_fixed_roof_and_separate_stations_joint_estimate')
    scene['layout_offsets']['preparation_fixed_revision']=23
    scene['visual_revision']='0.23'
    scene['assumptions'].append('0.23准备区结构纠正：43秒全景显示每条线在灰门架两侧各有开放工位；45秒为高位擦拭支架，48秒为另一侧独立低位涂脂工位。撤销两镜头对应同一机器的旧假设，新增三套低位涂脂设备和静态轮对，擦拭横梁暂估高1.65m并补三角侧板。联合43/45秒顶盖四角、45秒中央两轮心、48秒顶盖远角/后边缘及涂脂两轮心/柜顶估计共享门架和45/48相机；43秒相机保持不动。门架暂估顶高1.927m、跨度3.974m、深0.785m。三条线共用位置规则；48秒前景出口轮对另有一个可见静态快照，轴高约0.873m为估计，未还原升降或接触机构。所有相机共用全部设备与工件，不为单镜头隐藏/换形；移动轮对与3/4/17秒机械臂仍有时序差异。全部约束来自同一视频，没有独立视觉、尺寸、动力学或工艺验收。')


def refine_preparation_details_scene(scene):
    """0.24 visible infeed details, perforated wiping brackets and floor cabinets."""
    if scene.get('layout_offsets',{}).get('preparation_details_revision')==24:return
    from .preparation_details import INFEED_X,FIN_UNIT_CENTER
    removed=[i['id'] for i in scene['instances'] if i['zone']=='preparation' and i['asset']=='operator_console']
    scene['instances']=[i for i in scene['instances'] if i['id'] not in removed]
    wipes=sorted((i for i in scene['instances'] if i['asset']=='wiping_station'),key=lambda i:i['pose'][1])
    for n,wipe in enumerate(wipes,1):
        scene['instances'].append({'id':f'preparation_r24_infeed_{n}','asset':'preparation_infeed_bench',
            'pose':[INFEED_X,wipe['pose'][1],0,0,0,0],'zone':'preparation','source_segments':['S11','S12'],
            'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':'准备区窄蓝输送台与灰立柱'})
    scene['instances'].append({'id':'preparation_r24_fin_unit','asset':'preparation_fin_unit',
        'pose':FIN_UNIT_CENTER+[0,0,0],'zone':'preparation','source_segments':['S11','S12'],
        'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':'准备区可见散热管设备，工用未确认'})
    for rail in scene['rail_lines']:
        if rail['id'].startswith('preparation_r10_'):rail['start'][0]=INFEED_X-3.90
    removed_routes=[r['id'] for r in scene['agv_routes'] if r['id'] in {'dock_east_8','dock_east_17','dock_east_26'}]
    scene['agv_routes']=[r for r in scene['agv_routes'] if r['id'] not in removed_routes]
    scene['layout_offsets'].update(preparation_details_revision=24,preparation_removed_tall_consoles=removed,
        preparation_removed_unsupported_routes=removed_routes)
    scene['visual_revision']='0.24'
    scene['assumptions'].append('0.24准备区细节：擦拭三角板改为沿轨道X-Z平面、带真实圆孔的侧板，不再用错误横向Y-Z整板；45秒两侧六个手读角点参与同帧外形估计。高位工具与底座/导杆重建；移除原三台无对应的高操作台，在擦拭资产中加入低落地柜，白色涂脂柜只保留在独立涂脂资产。48秒补每线两条窄蓝台和两根灰柱，并补料架前可见散热管设备；柱高约1.287m、蓝台高0.72m、长度3.4m和部件数量为估计，工用未确认。蓝台/立柱按三个准备线复用，43秒对位保留剩余偏差；相机与0.23门架不改，所有视角共享同一模型。未恢复工艺动作、运动学或接触机构。')


def refine_transfer_context_scene(scene):
    """0.26: rotate the whole 50 s enclosure; share revised carriers and exits."""
    if scene.get('layout_offsets',{}).get('transfer_context_revision')==26:return
    from .transfer_details import room_point,FIN_BANK_X,FIN_BANK_Y,WEST_CROSS_X
    for item in scene['instances']:
        if item['zone']=='auxiliary_transfer':
            item['pose'][:3]=room_point(item['pose'][:3]);item['pose'][5]+=math.pi
        elif item['zone']=='press_exit' and item['asset']=='axle_end_check_table':
            item['asset']='press_exit_fin_bank';item['pose'][:2]=[FIN_BANK_X,FIN_BANK_Y+(6 if item['pose'][1]>11 else 0)]
            item['label_zh']='压装出口散热管组，内部功能未确认'
    for b in scene['boxes']:
        if b['id'].startswith('transfer_r20_room_'):b['center']=room_point(b['center'])
    for camera in scene['cameras']:
        if camera['id']=='cross_transfer':
            for key in ['position','target']:camera[key]=room_point(camera[key])
    for road in scene['agv_routes']:
        if road['id'].startswith(('transfer_r20_room_spine_','transfer_r20_room_cross_')):
            for key in ['start','end']:road[key]=room_point(road[key])
        elif road['id']=='transfer_r20_cache_cross_141.8':
            road['start'][0]=road['end'][0]=WEST_CROSS_X
    # Replace rectangular corner posts with visible I-section envelopes.
    post_ids=[b['id'] for b in scene['boxes'] if b['id'].startswith(('transfer_r20_room_post_','transfer_r20_cache_back_post_'))]
    for b in scene['boxes']:
        if b['id'] in post_ids:
            scene['instances'].append({'id':b['id']+'_section','asset':'transfer_boundary_post',
                'pose':b['center'][:2]+[0,0,0,0],
                'zone':'auxiliary_transfer' if 'room' in b['id'] else 'press_exit',
                'source_segments':['S13'] if 'room' in b['id'] else ['S17'],
                'geometry_source':'video_observed_form','pose_source':'estimated','label_zh':'围挡工字立柱，截面估计'})
    scene['boxes']=[b for b in scene['boxes'] if b['id'] not in post_ids]
    scene['layout_offsets'].update(transfer_context_revision=26,side_room_absolute_orientation='assumed_180deg_about_78.3_45.9')
    scene['visual_revision']='0.26'
    scene['assumptions'].append('0.26跨区连接：50秒侧室、室内小设备、相机和局部地面标线绕(78.3,45.9)m共同转180度，避免旧绝对朝向将青色区高梁投影到参考帧中不存在的位置；局部墙角对应保留，绝对朝向仍缺少跨镜头依据，为布局假设。压装出口两台通用检测桌替换为散热管组，65秒两个近似中心、0.45m假定读点高度和共用6m行距用于位置估计；西侧横向标线由两帧地面交点估计为X140.401m。转运车共用圆角低底盘、低托块、中央圆盘，撤销无可见依据的高信标；未重排工件快照或恢复驱动/承载。围挡角柱改工字截面。缓存轮心相机和队列沿用0.25，未应用out中的探索性联合候选。工件快照、出口轨道、关盖设备、东侧标线和整幅遮挡仍有差异，未作整体视觉、接触或调度验收。')


def refine_metrology_details_scene(scene):
    """0.25 plain tower, adjacent rotary fixture and slender static 55 s arm."""
    if scene.get('layout_offsets',{}).get('metrology_details_revision')==25:return
    from .metrology_assets import COLUMN_CENTRE,COVER_CENTRE,ROBOT_CARRIAGE
    for i in scene['instances']:
        if i['asset']=='back_ring_measuring_column':i['pose'][:2]=COLUMN_CENTRE
        elif i['asset']=='back_ring_measuring_booth':i['pose'][:2]=COVER_CENTRE
        elif i['asset']=='back_ring_travelling_robot':i['pose'][:3]=ROBOT_CARRIAGE
    scene['layout_offsets']['metrology_details_revision']=25
    scene['visual_revision']='0.25'
    scene['assumptions'].append('0.25后挡外形：55秒白色立柜恢复为素面立柜，撤销旧版无依据的三个黑色长工具。固定相机下七个手读柜角估计约0.587×0.494×1.529m；黑罩七个近似包络角估计约0.650×0.383×1.590m，圆角处不冒充精确物理角点。黑罩和回转夹具改为相邻布置，并补低柜、白色顶桥与朝上导杆；55秒机械臂采用细长竖臂、短前臂和朝上指爪，四个手读中心对应的高度1.16/1.96/1.99/2.27m为假定值，二维射线回投不作为独立误差证据。床面辊珠改蓝色。压装三台主机统一改为黑红下壳、实体斜白罩、成组缸体/导杆、薄橙色夹台和带齿条的窄升降桅杆，未增加单镜头副本。全部相机不动；局部外形和相邻主机间距仍需整幅检查，未恢复内部机构、接触、连续运动或调度。')
