"""Approximate pose fit to two cyan beds; crane endpoints are same-sequence diagnostics."""
import argparse
import hashlib
import json
import math
import numpy as np
from scipy.optimize import least_squares
from .build import ROOT, SCENE, camera_pose
from .fit_camera import basis, project
from .evidence import build_fingerprint

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'


def landmarks():
    rows=[]
    for x,pixels in [(80,[[171,425],[354,365],[154,402],[325,342]]),
                     (107,[[849,169],[983,128],[834,150],[944,108]])]:
        for number,((dx,dy),pixel) in enumerate(zip([(-3.4,-1),(3.4,-1),(-3.4,1),(3.4,1)],pixels)):
            rows.append({'name':f'bed_{x}_corner_{number}','world_point':[x+dx,63+dy,1.1],
                'source_pixel':pixel,'role':'fit'})
    # Bridges move in the video. These endpoints only diagnose the selected static snapshot.
    for x,pixels in [(85,[[350,270],[550,580]]),(98.5,[[675,155],[955,410]])]:
        for y,pixel in zip([65.5,51],pixels):
            rows.append({'name':f'bridge_{x}_end_{y}','world_point':[x,y,2.65],
                'source_pixel':pixel,'role':'check'})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    if args.apply and args.bind_captures:raise SystemExit('Apply before capture; bind after capture.')
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    camera=next(c for c in scene['cameras'] if c['id']=='unknown')
    rows=landmarks();points=np.array([r['world_point'] for r in rows]);pixels=np.array([r['source_pixel'] for r in rows])
    fit=np.array([r['role']=='fit' for r in rows]);pose=camera_pose(camera)
    parameters=np.array(pose[:3]+[pose[5],pose[4]]);focal=640/math.tan(camera['horizontal_fov']/2)
    if args.apply:
        result=least_squares(lambda v:(project(v,points[fit],focal)-pixels[fit]).ravel(),[77,31,34,1.07,.84],
            bounds=([0,-20,5,.1,.2],[100,60,70,1.5,1.4]),max_nfev=4000)
        parameters=result.x;eye,forward,_,_=basis(parameters)
        camera['position']=eye.tolist();camera['target']=(eye+10*forward).tolist()
        camera['registration']='manual_sparse_landmark_fit'
        camera['registration_report']='visual_comparison/wheelset_workshop_v1/camera_fit_unknown.json'
        SCENE.write_text(json.dumps(scene,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    projection=project(parameters,points,focal)
    for row,pixel in zip(rows,projection):row['projection']=pixel.tolist()
    def rmse(mask):return float(np.sqrt(np.mean(np.sum((projection[mask]-pixels[mask])**2,axis=1))))
    view={'camera':'unknown','reference_second':36,'parameters':parameters.tolist(),
        'fit_rmse_px':rmse(fit),'check_rmse_px':rmse(~fit),'fit_count':int(fit.sum()),
        'check_count':int((~fit).sum()),'landmarks':rows}
    report={'method':'Fixed FOV 0.92 rad; eight approximate cyan-bed corners; five pose parameters',
        'pixel_reading_tolerance_estimate':15,'geometry_measured':False,
        'geometry_adjusted_using_same_sequence':True,'views':[view],
        'claim_boundary':'Cyan bed outlines and snapshot bridge endpoints only; not independent full-image, count, dimension or dynamics acceptance.'}
    if args.bind_captures:
        report['build_fingerprint']=build_fingerprint()
        captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8'))
        capture=next(c for c in captures if c['camera']=='unknown');pose=capture['requested_pose']
        assert math.dist(pose[:3]+[pose[5],pose[4]],parameters)<1e-6
        assert capture['build_fingerprint']==report['build_fingerprint']
        view['gazebo_image_sha256']=hashlib.sha256((OUT/'gazebo'/capture['file']).read_bytes()).hexdigest()
        assert view['gazebo_image_sha256']==capture['sha256']
        view['reference_sha256']=hashlib.sha256((OUT/'references/frame_36s.png').read_bytes()).hexdigest()
        from .fit_inspection_cameras import plot
        plot(report,footer='Approximate low-bed outlines; bridges use one static snapshot. Same sequence informed geometry, not independent acceptance.')
    (OUT/'camera_fit_unknown.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'fit_rmse_px':view['fit_rmse_px'],'check_rmse_px':view['check_rmse_px']}))


if __name__=='__main__':main()
