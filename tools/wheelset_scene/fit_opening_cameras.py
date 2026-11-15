"""Two opening-sequence poses for a common low-tool / portal layout, static snapshot only."""
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


def landmarks(cid):
    rows=[]
    readings=([(28,9,.67,[[349,226],[458,181]]),(31.15,9,.67,[[535,368],[660,300]]),
               (33.25,9,.56,[[709,503],[843,426]])] if cid=='opening_wide' else
              [(28,9,.67,[[579,430],[723,349]]),(30.1,9,.56,[[761,596],[907,488]]),
               (28,15,.67,[[1005,181],[1111,128]])])
    for x,y,z,pixels in readings:
        for side,pixel in zip([-1,1],pixels):
            rows.append({'name':f'wheel_{x}_{y}_{side}','world_point':[x,y+side*.77,z],
                'source_pixel':pixel,'role':'fit'})
    if cid=='opening_wide':
        for n,(dx,dy,pixel) in enumerate([
            (-.25626972,-2.171749535,[350,272]),(.25626972,-2.171749535,[390,300]),
            (-.25626972,2.171749535,[811,141]),(.25626972,2.171749535,[850,164])]):
            rows.append({'name':f'gate_roof_{n+1}','world_point':[31.4971653+dx,9+dy,1.90800289],
                'source_pixel':pixel,'role':'fit'})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    if args.apply and args.bind_captures:raise SystemExit('Apply before capture; bind afterwards.')
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    report={'method':'Fixed FOV 0.92 rad; six wheel centres each; 8 s also uses four gate roof corners with geometry adjusted from this frame',
        'pixel_reading_tolerance_estimate':15,'geometry_measured':False,
        'geometry_adjusted_using_same_sequence':True,'views':[],
        'claim_boundary':'Same-sequence sparse pose fitting without independent check points. Source has moving workpieces; model is one static snapshot.'}
    for cid in ['opening_wide','opening']:
        camera=next(c for c in scene['cameras'] if c['id']==cid);rows=landmarks(cid)
        points=np.array([r['world_point'] for r in rows]);pixels=np.array([r['source_pixel'] for r in rows])
        p=camera_pose(camera);parameters=np.array(p[:3]+[p[5],p[4]]);focal=640/math.tan(camera['horizontal_fov']/2)
        if args.apply:
            initial=parameters
            result=least_squares(lambda v:(project(v,points,focal)-pixels).ravel(),initial,
                bounds=([28,-20,3,1.65,.2],[55,8,40,3.1,1.4]),max_nfev=4000)
            parameters=result.x;eye,forward,_,_=basis(parameters)
            camera['position']=eye.tolist();camera['target']=(eye+10*forward).tolist()
            camera['registration']='manual_sparse_landmark_fit'
            camera['registration_report']='visual_comparison/wheelset_workshop_v1/camera_fit_opening.json'
        projection=project(parameters,points,focal)
        for row,pixel in zip(rows,projection):row['projection']=pixel.tolist()
        view={'camera':cid,'reference_second':camera['reference_second'],'parameters':parameters.tolist(),
            'fit_rmse_px':float(np.sqrt(np.mean(np.sum((projection-pixels)**2,axis=1)))),
            'check_rmse_px':None,'fit_count':len(rows),'check_count':0,'landmarks':rows}
        view['wheel_fit_rmse_px']=float(np.sqrt(np.mean(np.sum((projection[:6]-pixels[:6])**2,axis=1))))
        if cid=='opening_wide':
            view['gate_roof_fit_rmse_px']=float(np.sqrt(np.mean(np.sum((projection[6:]-pixels[6:])**2,axis=1))))
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
    if args.apply:SCENE.write_text(json.dumps(scene,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.bind_captures:
        from .fit_inspection_cameras import plot
        plot(report,footer='8 s: wheel centres plus gate roof corners. 10 s: wheel centres only. Geometry tuned using same frames; no independent checks.')
    (OUT/'camera_fit_opening.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'camera':v['camera'],'fit_rmse_px':v['fit_rmse_px']} for v in report['views']]))


if __name__=='__main__':main()
