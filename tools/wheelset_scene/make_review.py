"""Generate an offline review page and a source-derived floor plan. No external web assets."""
import gzip
import hashlib
import html
import json
import math
from pathlib import Path
import cv2
from PIL import Image,ImageDraw,ImageFont
from .build import ROOT,SCENE

OUT=ROOT/"visual_comparison/wheelset_workshop_v1"
NAMES={"overview":"初始区域总览","identification":"识别与贴码","opening":"轴承开盖",
       "sorting":"退卸与分拣","inspection":"除锈与探伤","unknown":"字幕缺失设备区",
       "wiping":"轴颈擦拭","press":"轴承压装","closing":"轴承关盖","storage":"轮对入库",
       "overview_east":"全场鸟瞰","top":"俯视布局","press_front":"压装 · 正面校准","storage_reverse":"入库 · 反向观察",
       "inspection_exit":"探伤 · 出口侧视","identification_close":"识别 · 近景观察","opening_wide":"开盖 · 门架与低位工位",
       "component_overview":"轴承料库 · 装配区全景","bearing_loading":"轴承识别上料","closing_wide":"关盖 · 侧面广角",
       "run_in":"轴承磨合 · 双工位","painting":"轮对自动涂漆","back_ring":"后挡测量与存取",
       "exterior_check":"轮对外观检查","parts_transfer":"附件料筐搬运",
       "axle_check":"轴端识别与油封检查","reinspection":"复探 · 出口俯视",
       "cache_transfer":"压装出口 · 短缓存转运","cache_reverse":"短缓存 · 反向总览",
       "greasing":"轴颈与防尘板座涂脂","cross_transfer":"跨区转运 · 侧室与开口"}


def floorplan(scene,library):
    W,H=2400,1500;image=Image.new("RGB",(W,H),"#f4f6f7");d=ImageDraw.Draw(image)
    font_path="C:/Windows/Fonts/msyh.ttc"
    font=ImageFont.truetype(font_path,26);small=ImageFont.truetype(font_path,21);title=ImageFont.truetype(font_path,44)
    d.text((90,35),"轮轴检修线 · 全场基础布局",font=title,fill="#152c3d")
    d.text((92,100),"视频参考重建 / 初始估计布局 / 设备静态 / 尺寸待实测确认",font=font,fill="#536776")
    width,height=scene['bounds_m'];scale=min(2208/width,1152/height);ox,oy=96,1325
    def pos(x,y):return (ox+x*scale,oy-y*scale)
    d.rectangle([pos(0,height),pos(width,0)],fill="#d8dfe2",outline="#88959b",width=2)
    zone_colors=["#e9edf0","#e0eaf0","#e5edef","#d9efec","#ede6dd","#e0eaf1","#e2e9e9","#dfe6eb","#f1edda"]
    for i,zone in enumerate(scene['zones']):
        color=zone_colors[i%len(zone_colors)]
        x0,y0,x1,y1=zone["bounds"];d.rectangle([pos(x0,y1),pos(x1,y0)],fill=color)
    for road in scene["agv_routes"]:
        d.line([pos(*road["start"]),pos(*road["end"])],fill="#bc9b00",width=3)
    for rail in scene["rail_lines"]:
        for offset in [-rail["gauge_m"]/2,rail["gauge_m"]/2]:
            a,b=rail["start"],rail["end"];d.line([pos(a[0],a[1]+offset),pos(b[0],b[1]+offset)],fill="#596870",width=2)
    for item in scene["boxes"]:
        if item["id"]=="floor":continue
        x,y,z=item["center"];sx,sy,sz=item["size"]
        d.rectangle([pos(x-sx/2,y+sy/2),pos(x+sx/2,y-sy/2)],fill="#5c666b")
    asset_colors={"industrial_arm":"#d97a26","overhead_robot":"#d97a26","wheelset_agv":"#273037",
                  "bearing_rack":"#2d7e9c","roller_conveyor":"#2d7e9c","parts_tray":"#2d7e9c",
                  "unknown_cyan_cell":"#3fb4ad","wheelset_storage_bay":"#78929f",
                  "storage_buffer":"#2d7e9c","bearing_feed_module":"#2d7e9c"}
    for item in scene["instances"]:
        name=item["asset"]
        if name in {"track_segment","inspection_track"}:continue
        x,y,z,_,__,yaw=item["pose"]
        if name=="wheelset" and z>1:continue
        low,high=library["assets"][name]["bounds"]
        c,s=math.cos(yaw),math.sin(yaw)
        corners=[pos(x+c*a-s*b,y+s*a+c*b) for a,b in [(low[0],low[1]),(high[0],low[1]),(high[0],high[1]),(low[0],high[1])]]
        if name=="wheelset":
            for side in [-1,1]:
                wx,wy=x-s*side*.77,y+c*side*.77
                pts=[pos(wx+c*a-s*b,wy+s*a+c*b) for a,b in [(-.46,-.08),(.46,-.08),(.46,.08),(-.46,.08)]]
                d.polygon(pts,fill="#52636c")
            d.line([pos(x+s*1.1,y-c*1.1),pos(x-s*1.1,y+c*1.1)],fill="#52636c",width=2)
        elif "gantry" in name or name=="identification_bridge":
            d.line(corners+[corners[0]],fill="#b6632a",width=2)
        else:
            d.polygon(corners,fill=asset_colors.get(name,"#a3b1b8"),outline="#667882")
    for i,zone in enumerate(scene["zones"],1):
        x0,y0,x1,y1=zone["bounds"];p=pos(x0+.8,y1-.3)
        label=f'{i:02d}  {zone["name_zh"]}'
        box=d.textbbox((0,0),label,font=small);label_width=box[2]+18
        d.rounded_rectangle((p[0]-4,p[1]-3,p[0]+label_width,p[1]+31),radius=5,fill="#f9fbfc")
        d.text((p[0]+4,p[1]),label,font=small,fill="#203847")
    d.text((96,1360),"黄色：转运路线意图     黑线：轮对轨道     橙色：机器人 / 搬运机构     蓝色：料架 / 辊道",font=font,fill="#3c5260")
    d.text((96,1410),f"暂定范围 {width:g} × {height:g} m；此图用于模型审阅，不构成实测布置或通行验收。",font=small,fill="#677781")
    image.save(OUT/"floorplan.png")


def revision_pairs(items):
    """A compact source/capture sheet for the newly rebuilt areas."""
    selected={"identification_close","wiping","greasing","back_ring","press","press_front","cache_reverse","cross_transfer"}
    rows=[v for v in items if v['id'] in selected]
    canvas=Image.new('RGB',(1600,len(rows)*490),'#17232d');draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',23)
    for i,view in enumerate(rows):
        y=i*490
        draw.text((16,y+8),f"{view['name']} · {view['second']} 秒   |   左：原视频   右：Gazebo 静态模型",font=font,fill='#edf4f5')
        for j,key in enumerate(['reference','gazebo']):
            with Image.open(OUT/view[key]) as source:
                source=source.convert('RGB');source.thumbnail((800,450))
                canvas.paste(source,(j*800,y+40))
    canvas.save(OUT/'current_revision_pairs.jpg',quality=92)


def preparation_topology():
    """Annotate visible objects; do not infer hidden dimensions from this crop."""
    with Image.open(OUT/'references/frame_43s.png') as source:
        image=source.convert('RGB').crop((150,115,560,325)).resize((1230,630))
    canvas=Image.new('RGB',(1230,770),'#17232d');canvas.paste(image,(0,90))
    draw=ImageDraw.Draw(canvas);font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',24)
    draw.text((20,16),'43 秒原视频：低位涂脂工位 → 灰色门架 → 高位擦拭工位',font=font,fill='#edf4f5')
    for number,point,color in [(1,(226,194),'#6bd5d1'),(2,(296,205),'#ffd077'),(3,(391,258),'#ff9675')]:
        x,y=(point[0]-150)*3,(point[1]-115)*3+90
        draw.ellipse((x-20,y-20,x+20,y+20),fill='#17232d',outline=color,width=3)
        draw.text((x-8,y-17),str(number),font=font,fill=color)
    draw.text((20,730),'1 涂脂　2 门架　3 擦拭：两工位同时可见；尺寸与运动仍需估计。',font=font,fill='#cad7df')
    canvas.save(OUT/'preparation_topology_43s.png')


def source_coverage(scene,items):
    """Keep model metadata coverage separate from source views actually compared."""
    from .evidence import build_fingerprint
    rows=[]
    for segment in scene['video_segments']:
        start,end=segment['seconds']
        views=[v for v in items if start<=v['second']<=end]
        rows.append({'segment':segment['id'],'name_zh':segment['name_zh'],'seconds':[start,end],
            'associated_instances':sum(segment['id'] in i['source_segments'] for i in scene['instances']),
            'reference_cameras':[v['id'] for v in views],
            'visual_acceptance':'not_independently_accepted'})
    missing=[r['segment'] for r in rows if not r['reference_cameras']]
    report={'build_fingerprint':build_fingerprint(),'reference_pair_count':len(items),
        'segments_with_reference_view':len(rows)-len(missing),'segments_without_reference_view':missing,
        'independent_visual_acceptance':False,'segments':rows,
        'claim_boundary':'Reference times assigned to source intervals only; multiple views can cover one segment. Associated instance counts do not establish visual fidelity or implemented process behavior.'}
    (OUT/'source_coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    lines=['# 视频段与对照镜头覆盖','',
        f'{len(items)}组对照分布在{len(rows)-len(missing)}/{len(rows)}个视频段中。全部20段有关联模型；尚无独立视觉验收。','',
        '下表仅按参考秒数关联工序。相邻区域出现在背景中，不等同于该工序有专用对照。','',
        '| 段 | 视频秒数 | 工序 | 关联实例 | 专用参考镜头 |','| --- | --- | --- | --- | --- |']
    for r in rows:
        views='、'.join(r['reference_cameras']) or '**尚缺**'
        lines.append(f'| {r["segment"]} | {r["seconds"][0]}–{r["seconds"][1]} | {r["name_zh"]} | {r["associated_instances"]} | {views} |')
    lines.extend(['','后续'+('先补缺失镜头，再' if missing else '')+'核对设备数量、比例和跨区连接。静态视觉模型尚未实现工艺状态机或AGV调度。',''])
    (OUT/'source_coverage.md').write_text('\n'.join(lines),encoding='utf-8')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    scene=json.loads(SCENE.read_text(encoding="utf-8"))
    build=json.loads((ROOT/"simulation/wheelset_workshop/build_report.json").read_text(encoding="utf-8"))
    with gzip.open(ROOT/"assets/wheelset_workshop/source/asset_library.json.gz","rt",encoding="utf-8") as f:library=json.load(f)
    floorplan(scene,library)
    refs=OUT/"references";refs.mkdir(exist_ok=True)
    video=Path(r"D:\文件\图片\Screenshots\Bogie Overall Cleaning\车辆段轮轴检修线方案.mp4")
    cap=cv2.VideoCapture(str(video))
    items=[]
    for camera in scene["cameras"]:
        second=camera["reference_second"]
        if second is None:continue
        cap.set(cv2.CAP_PROP_POS_MSEC,second*1000);ok,frame=cap.read()
        if not ok:raise RuntimeError(f"Reference frame unavailable at {second}")
        ref=refs/f"frame_{second:02d}s.png";cv2.imencode(".png",frame)[1].tofile(str(ref))
        actual=OUT/"gazebo"/(camera["id"]+".png")
        if not actual.exists():continue
        items.append({"id":camera["id"],"name":NAMES[camera["id"]],"second":second,
                      "reference":str(ref.relative_to(OUT)).replace("\\","/"),
                      "gazebo":str(actual.relative_to(OUT)).replace("\\","/")})
    cap.release()
    preparation_topology()
    revision_pairs(items)
    source_coverage(scene,items)
    payload=json.dumps(items,ensure_ascii=False)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>轮轴检修线 · 视觉细化审阅</title><style>
*{box-sizing:border-box}body{margin:0;background:#101820;color:#eef4f5;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1540px;margin:auto;padding:38px 32px 70px}.eyebrow{color:#6cc9c8;letter-spacing:3px;font-size:13px}h1{font-size:34px;margin:8px 0 12px;font-weight:600}
p{color:#b6c4cc;max-width:1000px}.badges{display:flex;gap:10px;flex-wrap:wrap}.badge{border:1px solid #35434e;border-radius:99px;padding:5px 14px;font-size:13px;color:#d3dfe4}
.hero{width:100%;display:block;border-radius:12px;background:#242e35;margin:26px 0 12px}.caption{font-size:13px;color:#8fa2b0;margin-bottom:28px}
.toolbar{display:flex;gap:14px;align-items:center;justify-content:space-between;flex-wrap:wrap;margin-top:28px}select,button{font:inherit;background:#20303c;color:#f3f7f9;border:1px solid #465762;padding:8px 14px;border-radius:7px;cursor:pointer}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:15px}.panel{background:#19252e;border:1px solid #30404b;border-radius:10px;overflow:hidden}.panel header{padding:10px 14px;color:#bbccd5;font-size:14px}.panel img{display:block;width:100%;aspect-ratio:16/9;object-fit:contain;background:#3c4347}
.note{border-left:3px solid #dda65e;padding:12px 18px;background:#242827;color:#d3d7d4;margin:24px 0;font-size:14px}a{color:#71d0cb}.links{display:flex;gap:22px;flex-wrap:wrap}
details{margin-top:28px}summary{cursor:pointer;color:#c9dce4}details img{width:100%;border-radius:8px;margin-top:15px}@media(max-width:850px){main{padding:24px 15px}.pair{grid-template-columns:1fr}h1{font-size:26px}}
</style><main><div class="eyebrow">WHEELSET WORKSHOP / VISUAL BUILD REVISION</div><h1>轮轴检修线 · 视觉细化版</h1>
<p>依据 79 秒方案视频搭建整线模型。下方画面来自 Gazebo 的实际相机；模型保留可编辑部件，后续可加入工艺动作与调度。</p>
<div class="badges"><span class="badge">ASSET_COUNT 类设备资产</span><span class="badge">20 个视频参考段</span><span class="badge">尺寸为估计值</span><span class="badge">设备当前为静态</span></div>
<p>0.26 视觉基线：侧室、室内设备与相机共同转向，修正50秒多出的高梁遮挡；压装出口补散热管组，转运车改低圆角底盘。CAMERA_COUNT 个 Gazebo 视角共用同一套设备坐标，其中 PAIR_COUNT 个与视频并列查看。尺寸、绝对朝向、部件数量与跨区连接仍有估计。此页展示静态视觉基线。</p>
<img class="hero" src="gazebo/overview_east.png" alt="Gazebo 实际渲染的全场鸟瞰"><div class="caption">全场鸟瞰 · Gazebo 实际相机图像</div>
<div class="toolbar"><label>查看区域 <select id="view"></select></label><span id="time"></span></div>
<div class="pair"><section class="panel"><header>原视频参考</header><img id="reference" alt="原视频截图"></section><section class="panel"><header>Gazebo 当前模型</header><img id="gazebo" alt="Gazebo 当前区域"></section></div>
<div class="note">视觉基线冻结后转入工艺运行开发，剩余外观问题持续记录；整体视觉尚未验收，不作为后续功能的前置门槛。料库层数、间距、缓存与道路仍是估计；全场地面不代表实测总平面。静态截图与少量标记点匹配不构成工艺、动力学或调度验收。<a href="../../simulation/wheelset_workshop/PROJECT_GOAL.md">长期目标与里程碑</a></div>
<p><a href="source_coverage.md">逐段查看镜头覆盖</a>：PAIR_COUNT 组对照已覆盖全部 20 个视频段。覆盖表示已有相应时间的对照图；设备外形、数量和区域连接仍需逐项核对。</p>
<details><summary>展开全场平面图</summary><img src="floorplan.png" alt="视频参考估计布局"></details>
<details><summary>查看 Gazebo 全场顶视</summary><img src="gazebo/top.png" alt="完整显示当前估计地面范围的 Gazebo 顶视"></details>
<details><summary>查看 62 秒镜头的标记点检查</summary><img src="camera_alignment_press_front.png" alt="原视频上的人工标记与模型投影点，非整体相似度评分"></details>
<details><summary>查看压装上料区两个视角的固定料仓检查</summary><img src="camera_alignment_press_feed.png" alt="58秒固定环件料仓六角点，几何由58和62秒共同估计"><img src="camera_alignment_press_feed_front.png" alt="62秒固定主机相机下的同一料仓角点；无独立检查点，工件与遮挡关系仍有差异"></details>
<details><summary>查看磨合与涂漆区的固定底座检查</summary><img src="camera_alignment_run_in.png" alt="71秒门架脚和磨合底座；轮心为同序列检查，非独立验收"><img src="camera_alignment_painting.png" alt="73秒固定底座和横梁检查；移动轮对位置与当前静态占位不同"></details>
<details><summary>查看探伤区两个视角的标记点与剩余偏差</summary><img src="camera_alignment_inspection.png" alt="30秒标记点检查"><img src="camera_alignment_inspection_exit.png" alt="32秒标记点检查，设备屋顶仍有偏差"></details>
<details><summary>查看入库区固定框架与轨道端点检查</summary><img src="camera_alignment_storage.png" alt="76秒固定框架点检查"><img src="camera_alignment_storage_reverse.png" alt="78秒固定框架与交接轨道检查"></details>
<details><summary>查看青色设备区机座与桥架快照检查</summary><img src="camera_alignment_unknown.png" alt="36秒青色机座角点和静态桥架端点，非独立验证"></details>
<details><summary>查看开盖区两个视角的轮心与门架检查</summary><img src="camera_alignment_opening_wide.png" alt="8秒六个轮心与四个门架顶角拟合，几何使用同一帧调整，无独立检查点"><img src="camera_alignment_opening.png" alt="10秒六个轮心拟合，无独立检查点"></details>
<details><summary>查看分拣料箱与端部工作台检查</summary><img src="camera_alignment_sorting.png" alt="21秒六个料箱口角点拟合、四个工作台角点检查，几何使用同一序列调整"></details>
<details><summary>查看擦拭轮心与关盖料盘检查</summary><img src="camera_alignment_wiping.png" alt="45秒轮心拟合，设备和背景仍为估计"><img src="camera_alignment_closing.png" alt="68秒料盘与固定横梁共同拟合，柜体检查保留偏差"><img src="camera_alignment_closing_wide.png" alt="67秒同一固定梁和料盘，移动吊具不作为固定端点"></details>
<details><summary>查看识别臂静态姿态与移动快照差异</summary><p>4 秒五个姿态点约 15.9 px，参与了模型调整；3 秒采用同一模型时约 194.8 px，保留移动快照差异。最后一点为细探针尖端，罩顶高度间距只检查静态几何。</p><img src="camera_alignment_identification_close_robot.png" alt="4秒识别机械臂静态姿态，末端为细探针，非独立验收"><img src="camera_alignment_identification_robot.png" alt="3秒运动中的源机械臂与同一静态姿态，偏差保留"></details>
<details><summary>查看准备线两工位与三视角门架估计</summary><p>43 秒明确显示两套工位；原先把 45/48 秒当成同一机器的解释已撤回。固定 43 秒相机，联合估计共享顶盖与 45/48 秒相机：43/45 秒四角点误差约 4.3/3.5 px，48 秒远角约 1.6 px、后边缘距离约 1.3 px。全部读点参与拟合，尚无独立验收；近角被字幕遮挡，未补造对应点。涂脂轮心/柜顶约 7.4 px，轮廓和前景工件仍有差异。</p><img src="preparation_topology_43s.png" alt="43秒原图可同时看到独立涂脂工位、门架与擦拭工位"><img src="camera_alignment_component_overview_portal_roof.png" alt="43秒顶盖四角参与共享几何拟合，相机固定"><img src="camera_alignment_wiping_portal_roof.png" alt="45秒门架四顶角参与三视角联合估计"><img src="camera_alignment_greasing_portal_mismatch.png" alt="48秒同一固定门架的可见远角及后边缘参与联合估计"><p><a href="camera_fit_preparation_geometry.json">联合估计参数、残差定义与图像哈希</a>；<a href="camera_alignment_greasing_portal_before_fixed_0_23.png">0.22 失败对应记录</a>。</p></details>
<details><summary>查看擦拭侧板与准备区前段设备细节</summary><p>相机与门架位置沿用 0.23。45 秒两侧三角板六顶点参与外形估计，约 10.6 px；48 秒灰柱近似中心、蓝台端点和散热设备地面中心约 4.1 px。43 秒四个灰柱近似中心检查约 12.7 px，仍来自同一视频，低清对应存在不确定性。蓝台长度、散热管数量、设备工用与内部机构尚未确认。</p><img src="camera_alignment_wiping_preparation_details.png" alt="45秒沿轨道的三角板六顶点，同帧外形估计"><img src="camera_alignment_greasing_preparation_details.png" alt="48秒窄蓝台、灰立柱和散热设备，近似中心仅作局部诊断"><img src="camera_alignment_component_overview_preparation_details.png" alt="43秒同一组灰柱检查，保留跨视角剩余偏差"><p><a href="camera_fit_preparation_details.json">细节对应点与误差记录</a></p></details>
<details><summary>查看料库、地面标线与识别输送线检查</summary><img src="camera_alignment_component_overview.png" alt="43秒14个成对轮心、6个地面交点与3个料架立柱点共同拟合；准备线出口4轮心也用于几何定位，非独立检查"><img src="camera_alignment_bearing_loading.png" alt="54秒窄输送线端点，深度约束较弱，非整体设备验收"></details>
<details><summary>查看后挡储料床与短料台检查</summary><img src="camera_alignment_back_ring.png" alt="55秒储料床四角与假定人员高度拟合，短料台四角同序列检查；比例和设备尺寸未实测"></details>
<details><summary>查看后挡白柜与测量罩外形修订</summary><p>固定原 55 秒相机，白柜七角点约 6.0 px，黑罩七个近似包络角约 9.3 px，均参与了形体调整。圆角包络点不是精确网格角点；机械臂关节高度是假定值，沿像素射线回投所得的零误差不作为精度证据。背景压装的间距与外形仍有差异。</p><img src="camera_alignment_back_ring_column.png" alt="55秒素面白色立柜，七个手读角点参与位置和尺寸估计"><img src="camera_alignment_back_ring_cover.png" alt="55秒圆角黑罩的近似包络角，不代表精确物理角点或独立验收"><p><a href="camera_fit_metrology.json">料床、柜体、测量罩与静态臂假设记录</a></p></details>
<details><summary>查看新增四个镜头与布局偏差</summary><img src="camera_alignment_exterior_check.png" alt="6秒轮心拟合；放大圈为原片叠加；前景分拣横梁遮挡已移除"><img src="camera_alignment_parts_transfer.png" alt="14秒四个修正后的纵梁端角和四个轮心共同估计，约8.4像素；两个料筐中心检查约15.5像素，非独立验收"><img src="camera_alignment_axle_check.png" alt="17秒四个料筐口沿角点与两个轮心约11.4像素；左梁身份与机械臂仍有差异"><img src="camera_alignment_reinspection.png" alt="33秒超探屋顶拟合、磁粉屋顶同序列检查；运动轮对采用静态快照"></details>
<details><summary>查看涂脂、侧室与短缓存的四视角检查</summary><img src="camera_alignment_greasing.png" alt="48秒独立低位涂脂工位轮心与柜顶参与共享门架联合估计，轮缘仅作残差诊断，工位仍简化"><img src="camera_alignment_cross_transfer.png" alt="50秒侧室角点；围挡高度与房间尺度是假设，小设备工序未知"><img src="camera_alignment_cache_transfer.png" alt="65秒短缓存轮心；队列与行距由同序列共同估计"><img src="camera_alignment_cache_reverse.png" alt="66秒同一短缓存的反向观察；压装、关盖和道路布局仍有差异"></details>
<details><summary>查看 0.3 版探伤区（镜头不同，仅作修改记录）</summary><img src="previous_0_3/inspection.png" alt="0.3版探伤区"></details>
<details><summary>查看 0.4 版识别区（镜头不同，仅作修改记录）</summary><img src="previous_0_4/identification.png" alt="0.4版识别区"></details>
<details><summary>查看 0.2 版识别与探伤区域（镜头可能不同，仅作修改记录）</summary><div class="pair"><img src="previous_0_2/identification.png" alt="0.2版识别区"><img src="previous_0_2/inspection.png" alt="0.2版探伤区"></div></details>
<details><summary>查看 0.1 版两个区域（镜头不同，仅作修改记录）</summary><div class="pair"><img src="previous_0_1/press.png" alt="0.1版压装区"><img src="previous_0_1/storage.png" alt="0.1版入库区"></div></details>
<p class="links"><a href="current_revision_pairs.jpg">本轮八组对比总图</a><a href="../../assets/wheelset_workshop/source/wheelset_workshop.blend">可编辑 Blender 工程</a><a href="../../simulation/wheelset_workshop/README.md">进度与启动说明</a><a href="visual_gaps.md">待细化项目</a><a href="gazebo/capture_report.json">相机取证记录</a></p>
</main><script>const items=PAYLOAD;const select=document.getElementById('view');items.forEach((v,i)=>{const o=document.createElement('option');o.value=i;o.textContent=v.name;select.append(o)});select.value=Math.max(0,items.findIndex(v=>v.id==='identification'));function show(){const v=items[Number(select.value)];if(!v)return;document.getElementById('reference').src=v.reference;document.getElementById('gazebo').src=v.gazebo;document.getElementById('time').textContent='参考时间 '+v.second+' 秒'}select.addEventListener('change',show);show();</script></html>'''.replace("PAYLOAD",payload).replace("ASSET_COUNT",str(build["asset_types"])).replace('CAMERA_COUNT',str(len(scene['cameras']))).replace('PAIR_COUNT',str(len(items))).replace("REVISION",scene.get("visual_revision","0.1"))
    (OUT/"review.html").write_text(page,encoding="utf-8")
    (OUT/"reference_index.json").write_text(json.dumps({"video_path":str(video),"video_sha256":hashlib.file_digest(video.open("rb"),"sha256").hexdigest(),"views":items},ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"review_views":len(items),"review_page":str(OUT/"review.html"),"floorplan":str(OUT/"floorplan.png")},ensure_ascii=False))


if __name__=="__main__":main()
