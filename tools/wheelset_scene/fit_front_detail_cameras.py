"""Evidence for 6/14/17/33 s views; sparse points do not certify whole-scene matching."""
import argparse
import hashlib
import json
import math
import numpy as np
from .build import ROOT,SCENE,camera_pose
from .evidence import build_fingerprint
from .fit_camera import project

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'


def landmarks(cid):
    if cid=='exterior_check':
        points=[[x,9+side*.77,z] for x,z in [(31.15,.67),(36.5,.695),(39,.56)] for side in [-1,1]]
        pixels=[[319,273],[420,204],[635,400],[713,319],[902,495],[988,409]]
        checks=[];check_pixels=[]
    elif cid=='parts_transfer':
        from .front_geometry import GANTRY_CENTER, GANTRY_HALF_SPAN, BIN_FIRST_CENTER, BIN_PITCH
        x,y=GANTRY_CENTER;hx,hy=GANTRY_HALF_SPAN
        points=[[x+hx,y-hy,3.16],[x+hx,y+hy,3.16],[x-hx,y+hy,3.16],[x-hx,y-hy,3.16]]
        points.extend([[28,14.23,.67],[28,15.77,.67],[30.1,14.23,.56],[30.1,15.77,.56]])
        pixels=[[134,570],[956,178],[868,129],[88,476],[847,535],[926,474],[943,598],[1017,551]]
        checks=[[BIN_FIRST_CENTER[0],BIN_FIRST_CENTER[1]+n*BIN_PITCH,.30] for n in [0,1]]
        check_pixels=[[315,617],[605,466]]
    elif cid=='axle_check':
        from .front_geometry import BIN_FIRST_CENTER, AXLE_CHECK_X
        x,y=BIN_FIRST_CENTER
        points=[[x+dx,y+dy,.575] for dx,dy in [(-1,-1.25),(1,-1.25),(-1,1.25),(1,1.25)]]
        points.extend([[AXLE_CHECK_X,9+side*.77,.56] for side in [-1,1]])
        pixels=[[1162,446],[1274,351],[981,323],[1076,240],[679,425],[571,330]]
        checks=[];check_pixels=[]
    else:
        points=[];pixels=[]
        for y,px in [(50,[[467,474],[516,479],[485,360],[534,363]]),
                     (56.5,[[490,301],[538,304],[501,215],[548,215]])]:
            points.extend([[44.5+dx,y+dy,2.46] for dx,dy in [(-.805,-1.95),(.805,-1.95),(-.805,1.95),(.805,1.95)]])
            pixels.extend(px)
        checks=[[35.45+dx,50+dy,2.46] for dx,dy in [(-.805,-1.95),(.805,-1.95),(-.805,1.95),(.805,1.95)]]
        check_pixels=[[146,471],[192,477],[194,360],[239,363]]
    return np.array(points),np.array(pixels),np.array(checks),np.array(check_pixels)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    report={'method':'Fixed FOV .92 rad. 6 s: six hubs; 14 s: four corrected runway ends and four hubs, two approximate bin centres checked; 17 s: four bin-rim corners and two hubs; 33 s: eight hood roof corners.',
        'geometry_measured':False,'geometry_adjusted_using_same_sequence':True,
        'pixel_reading_tolerance_estimate':20,'views':[],
        'correspondence_correction':'14 s old [122,481] lay on an end tie, not the near longitudinal beam end; replaced by [134,570]. 17 s approximate whole-bin centres replaced by four visible rim corners. The far 14 s reading may show only an empty half-bin while its loaded half is lifted, so it is excluded.',
        'claim_boundary':'6/33 s callouts are source overlays. Crane dimensions, cameras and workpiece snapshots jointly use this sequence; checks are not independent. Nearest 14 s bin centre is partly caption-covered. Left 17 s beam identity, arms, moving workpieces and backgrounds remain uncertain; no complete-device or full-image acceptance.'}
    captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8')) if args.bind_captures else []
    for cid in ['exterior_check','parts_transfer','axle_check','reinspection']:
        camera=next(c for c in scene['cameras'] if c['id']==cid)
        p=camera_pose(camera);parameters=np.array(p[:3]+[p[5],p[4]])
        points,pixels,checks,check_pixels=landmarks(cid)
        focal=640/math.tan(camera['horizontal_fov']/2);fitted=project(parameters,points,focal)
        checked=project(parameters,checks,focal) if len(checks) else []
        rows=[{'name':f'source_fit_{n+1}','world_point':w.tolist(),'source_pixel':px.tolist(),
               'projection':pr.tolist(),'role':'fit'} for n,(w,px,pr) in enumerate(zip(points,pixels,fitted))]
        rows.extend({'name':f'same_sequence_check_{n+1}','world_point':w.tolist(),'source_pixel':px.tolist(),
            'projection':pr.tolist(),'role':'check'} for n,(w,px,pr) in enumerate(zip(checks,check_pixels,checked)))
        view={'camera':cid,'reference_second':camera['reference_second'],'parameters':parameters.tolist(),
            'fit_count':len(points),'check_count':len(checks),
            'fit_rmse_px':float(np.sqrt(np.mean(np.sum((fitted-pixels)**2,axis=1)))),
            'check_rmse_px':float(np.sqrt(np.mean(np.sum((checked-check_pixels)**2,axis=1)))) if len(checks) else None,
            'landmarks':rows}
        if args.bind_captures:
            capture=next(c for c in captures if c['camera']==cid);pose=capture['requested_pose']
            report['build_fingerprint']=build_fingerprint()
            assert capture['build_fingerprint']==report['build_fingerprint']
            assert math.dist(pose[:3]+[pose[5],pose[4]],parameters)<1e-6
            view['gazebo_image_sha256']=hashlib.sha256((OUT/'gazebo'/capture['file']).read_bytes()).hexdigest()
            assert view['gazebo_image_sha256']==capture['sha256']
            view['reference_sha256']=hashlib.sha256((OUT/'references'/f"frame_{camera['reference_second']:02d}s.png").read_bytes()).hexdigest()
        report['views'].append(view)
    if args.bind_captures:
        from .fit_inspection_cameras import plot
        plot(report,footer='New diagnostic views reveal spacing, overhead occlusion and moving-workpiece differences. No independent or full-scene acceptance.')
    (OUT/'camera_fit_front_detail.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v[k] for k in ['camera','fit_rmse_px','check_rmse_px']} for v in report['views']]))


if __name__=='__main__':main()
