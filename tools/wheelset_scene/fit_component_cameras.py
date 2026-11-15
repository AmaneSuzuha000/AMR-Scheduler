"""Sparse component-store / bearing-line pose diagnostics, not full-scene acceptance."""
import argparse
import hashlib
import json
import math
import numpy as np
from scipy.optimize import least_squares
from .build import ROOT, SCENE, camera_pose
from .fit_camera import project, basis
from .evidence import build_fingerprint

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'
READINGS={
    'bearing_loading':[
        ([106,34.75,.95],[254,184]),([106,35.25,.95],[254,172]),
        ([113,34.75,.95],[731,405]),([113,35.25,.95],[739,390])],
    'component_overview':[
        ([67.5,32.84,0],[276,117]),([100.5,32.84,0],[904,320]),
        ([100.5,32.84,5.4],[920,223]),([104.3,33.95,0],[1002,347]),
        ([87,26,0],[569,339]),([91,26,0],[650,367]),
        ([87,21,0],[510,428]),([91,21,0],[590,460]),
        ([87,16,0],[450,518]),([91,16,0],[530,550])]
}
CHECKS={'component_overview':[
    ([96.4,26,.56],[765,403]),([98.4,26,.56],[812,418]),
    ([96.4,21,.56],[715,505]),([96.4,16,.56],[657,604])]}


def scene_readings(cid,scene):
    offsets=scene.get('layout_offsets',{})
    if cid=='bearing_loading':
        dx,dy=offsets.get('component_end_x',0),offsets.get('component_end_y',0)
        return [([p[0]+dx,p[1]+dy,p[2]],px) for p,px in READINGS[cid]],[]
    if offsets.get('preparation_scale_revision')!=21:return READINGS[cid],CHECKS.get(cid,[])
    from .component_geometry import (HUB_READINGS,LONG_CACHE_FIRST_X,LONG_CACHE_FIRST_Y,
        LONG_CACHE_PITCH,LONG_CACHE_ROW_PITCH,ROAD_X,RACK_START,RACK_END,RACK_FRONT_Y,
        RACK_HEIGHT,PREPARATION_DX,PREPARATION_TOP_Y,PREPARATION_LANE_PITCH)
    rows=[([LONG_CACHE_FIRST_X+n*LONG_CACHE_PITCH,LONG_CACHE_FIRST_Y-r*LONG_CACHE_ROW_PITCH+s*.77,.56],px)
        for n,r,s,px in HUB_READINGS]
    rows.extend(([x,LONG_CACHE_FIRST_Y-r*LONG_CACHE_ROW_PITCH,0],px)
        for (r,x),px in zip([(r,x) for r in range(3) for x in ROAD_X],
            [[569,339],[650,367],[510,428],[590,460],[450,518],[530,550]]))
    rows.extend([([RACK_START,RACK_FRONT_Y,0],[276,117]),
        ([RACK_END,RACK_FRONT_Y,0],[904,320]),([RACK_END,RACK_FRONT_Y,RACK_HEIGHT],[920,223])])
    snapshot_x=82.2
    if offsets.get('preparation_fixed_revision')==23:
        from .component_geometry import PREPARATION_FRONT_SNAPSHOT_X
        snapshot_x=PREPARATION_FRONT_SNAPSHOT_X
    checks=[([snapshot_x+PREPARATION_DX,PREPARATION_TOP_Y-r*PREPARATION_LANE_PITCH+s*.77,.56],px)
        for (r,s),px in zip([(r,s) for r in range(2) for s in [-1,1]],[[438,292],[466,258],[276,455],[307,419]])]
    return rows,checks


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    if args.apply and args.bind_captures:raise SystemExit('Apply before capture; bind afterwards.')
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    report={'method':'0.21: paired cache wheel hubs, six paint crossings and three parallel rack upright points jointly constrain the wide view; preparation exit hubs check adjacency. Bearing-line endpoints use its shared translated coordinates. FOV fixed at 0.92 rad.',
        'pixel_reading_tolerance_estimate':15,'geometry_measured':False,
        'geometry_adjusted_using_same_sequence':True,'views':[],
        'claim_boundary':'Sparse same-sequence fit only. 0.23 preparation exit wheels use the shared 45 s static snapshot, so their 43 s check includes motion mismatch and is not independent layout validation. Wheel span/diameter are assumed scale anchors. Thin conveyor endpoints weakly constrain depth. Rack count, hidden queue counts, dimensions and whole-cell matching remain estimates.'}
    for cid,readings in READINGS.items():
        readings,checks=scene_readings(cid,scene)
        camera=next(c for c in scene['cameras'] if c['id']==cid)
        points=np.array([r[0] for r in readings]);pixels=np.array([r[1] for r in readings])
        p=camera_pose(camera);parameters=np.array(p[:3]+[p[5],p[4]])
        focal=640/math.tan(camera['horizontal_fov']/2)
        if args.apply:
            result=least_squares(lambda q:(project(q,points,focal)-pixels).ravel(),parameters,
                bounds=([80,-35,4,1.6,.2],[140,36,70,3.1,1.3]),max_nfev=3000)
            parameters=result.x;eye,forward,_,_=basis(parameters)
            camera['position']=eye.tolist();camera['target']=(eye+10*forward).tolist()
            camera['registration']='manual_sparse_landmark_fit'
            camera['registration_report']='visual_comparison/wheelset_workshop_v1/camera_fit_component.json'
        projection=project(parameters,points,focal)
        check_projection=project(parameters,np.array([r[0] for r in checks]),focal) if checks else []
        check_error=float(np.sqrt(np.mean(np.sum((check_projection-np.array([r[1] for r in checks]))**2,axis=1)))) if checks else None
        view={'camera':cid,'reference_second':camera['reference_second'],'parameters':parameters.tolist(),
            'fit_rmse_px':float(np.sqrt(np.mean(np.sum((projection-pixels)**2,axis=1)))),
            'check_rmse_px':check_error,'fit_count':len(readings),'check_count':len(checks),
            'landmarks':[{'name':f'fixed_point_{i+1}','world_point':r[0],'source_pixel':r[1],
                'role':'fit','projection':pixel.tolist()} for i,(r,pixel) in enumerate(zip(readings,projection))]}
        view['landmarks'].extend({'name':f'preparation_exit_{i+1}','world_point':r[0],'source_pixel':r[1],
            'role':'check','projection':pixel.tolist()} for i,(r,pixel) in enumerate(zip(checks,check_projection)))
        if cid=='component_overview':
            from .component_geometry import preparation_portal_roof,PREPARATION_ROOF_PIXELS_43
            roof=np.array(preparation_portal_roof(scene));px=np.array(PREPARATION_ROOF_PIXELS_43)
            pr=project(parameters,roof,focal)
            view['portal_roof_diagnostic']={'point_rmse_px':float(np.sqrt(np.mean(np.sum((pr-px)**2,axis=1)))),
                'role':'same_sequence_joint_geometry_fit','independent_check_points':0,
                'claim_boundary':'43 s camera is fixed; these four roof corners helped estimate 0.23 shared portal geometry with 45/48 s, so they are fitting targets.',
                'landmarks':[{'name':f'roof_corner_{n+1}','world_point':w.tolist(),'source_pixel':p.tolist(),
                    'projection':q.tolist(),'role':'fit'} for n,(w,p,q) in enumerate(zip(roof,px,pr))]}
        if args.bind_captures:
            report['build_fingerprint']=build_fingerprint()
            captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8'))
            capture=next(c for c in captures if c['camera']==cid);p=capture['requested_pose']
            assert capture['build_fingerprint']==report['build_fingerprint']
            assert math.dist(p[:3]+[p[5],p[4]],parameters)<1e-6
            view['gazebo_image_sha256']=hashlib.sha256((OUT/'gazebo'/capture['file']).read_bytes()).hexdigest()
            assert view['gazebo_image_sha256']==capture['sha256']
            view['reference_sha256']=hashlib.sha256((OUT/'references'/f"frame_{camera['reference_second']:02d}s.png").read_bytes()).hexdigest()
        report['views'].append(view)
    if args.apply:SCENE.write_text(json.dumps(scene,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.bind_captures:
        from .fit_inspection_cameras import plot
        plot(report,footer='Paired wheel hubs, paint and rack jointly constrain scale; preparation checks also informed geometry. No independent acceptance.')
        wide=next(v for v in report['views'] if v['camera']=='component_overview');roof=wide['portal_roof_diagnostic']
        plot({'views':[dict(wide,diagnostic_id='component_overview_portal_roof',landmarks=roof['landmarks'],
            fit_count=4,fit_rmse_px=roof['point_rmse_px'],check_count=0)]},
            footer='43 s fixed camera; four roof corners used in 43/45/48 s shared geometry fit. Not independent acceptance.')
    (OUT/'camera_fit_component.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'camera':v['camera'],'fit_rmse_px':v['fit_rmse_px']} for v in report['views']]))


if __name__=='__main__':main()
