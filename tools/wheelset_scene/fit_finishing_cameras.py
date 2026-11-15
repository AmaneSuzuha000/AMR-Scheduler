"""Fixed finishing fixtures across 71/73 s; moving wheelsets are not common landmarks."""
import argparse
import hashlib
import json
import math
import numpy as np
from .build import ROOT, SCENE, camera_pose
from .evidence import build_fingerprint
from .fit_camera import project

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'
READINGS={
    'run_in':[[58,450],[204,263],[520,448],[560,331],[670,464],[699,348]],
    'painting':[[637,491],[767,371],[985,655],[1045,543],[1126,720],[1190,607]]}


def landmarks(scene):
    lane=(8 if scene.get('layout_offsets',{}).get('closing_fixed_revision')==16 else 17)+scene.get('layout_offsets',{}).get('finishing_reference_y',0)
    portal=next(i for i in scene['instances'] if i['asset']=='spray_portal' and abs(i['pose'][1]-lane)<1e-6)
    fixtures=sorted([i for i in scene['instances'] if i['asset']=='bearing_run_in_stand' and abs(i['pose'][1]-lane)<1e-6],key=lambda i:-i['pose'][0])
    return np.array([[portal['pose'][0],lane+s*1.9,.03] for s in [1,-1]]+[
        [i['pose'][0],lane+s*1.22,.08] for i in fixtures for s in [1,-1]])


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    scene=json.loads(SCENE.read_text(encoding='utf-8'));points=landmarks(scene)
    report={'method':'Six shared fixed fixture points per view; both camera poses, portal position/span and run-in pitch jointly estimated from these same frames.',
        'geometry_measured':False,'geometry_adjusted_using_same_sequence':True,
        'pixel_reading_tolerance_estimate':15,'views':[],
        'claim_boundary':'Sparse fixed-equipment alignment only. Same-frame checks are not independent; moving wheelsets occupy different positions at 71/73 s. No full-image, process or physics acceptance.'}
    for cid,pixels in READINGS.items():
        camera=next(c for c in scene['cameras'] if c['id']==cid);p=camera_pose(camera)
        parameters=np.array(p[:3]+[p[5],p[4]]);focal=640/math.tan(camera['horizontal_fov']/2)
        pixels=np.array(pixels);fitted=project(parameters,points,focal)
        rows=[{'name':f'fixed_fixture_{n+1}','world_point':w.tolist(),'source_pixel':px.tolist(),
            'projection':pr.tolist(),'role':'fit'} for n,(w,px,pr) in enumerate(zip(points,pixels,fitted))]
        if cid=='run_in':
            lane=(points[0,1]+points[1,1])/2
            dx=scene.get('layout_offsets',{}).get('closing_block_x',0)
            checks=np.array([[x+dx,lane+s*.77,.56] for x in [162.2,158,156.04703562179583] for s in [1,-1]])
            check_pixels=np.array([[198,374],[244,288],[510,408],[545,321],[654,425],[685,337]])
        else:
            checks=np.array([[points[0,0],points[n,1],1.38] for n in [0,1]])
            check_pixels=np.array([[643,435],[778,305]])
        checked=project(parameters,checks,focal)
        rows.extend({'name':f'same_sequence_check_{n+1}','world_point':w.tolist(),'source_pixel':px.tolist(),
            'projection':pr.tolist(),'role':'check'} for n,(w,px,pr) in enumerate(zip(checks,check_pixels,checked)))
        view={'camera':cid,'reference_second':camera['reference_second'],'parameters':parameters.tolist(),
            'fit_count':len(points),'check_count':len(checks),
            'fit_rmse_px':float(np.sqrt(np.mean(np.sum((fitted-pixels)**2,axis=1)))),
            'check_rmse_px':float(np.sqrt(np.mean(np.sum((checked-check_pixels)**2,axis=1)))),
            'landmarks':rows}
        if args.bind_captures:
            captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8'))
            capture=next(c for c in captures if c['camera']==cid);p=capture['requested_pose']
            report['build_fingerprint']=build_fingerprint()
            assert capture['build_fingerprint']==report['build_fingerprint']
            assert math.dist(p[:3]+[p[5],p[4]],parameters)<1e-6
            view['gazebo_image_sha256']=hashlib.sha256((OUT/'gazebo'/capture['file']).read_bytes()).hexdigest()
            assert view['gazebo_image_sha256']==capture['sha256']
            view['reference_sha256']=hashlib.sha256((OUT/'references'/f"frame_{camera['reference_second']:02d}s.png").read_bytes()).hexdigest()
        report['views'].append(view)
    if args.bind_captures:
        from .fit_inspection_cameras import plot
        plot(report,footer='Shared fixed fixtures tuned using both frames. Moving wheelset occupancy differs. Not independent visual acceptance.')
    (OUT/'camera_fit_finishing.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'camera':v['camera'],'fit_rmse_px':v['fit_rmse_px'],'check_rmse_px':v['check_rmse_px']} for v in report['views']]))


if __name__=='__main__':main()
