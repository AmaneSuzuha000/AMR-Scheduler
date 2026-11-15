"""Sorting bin rim camera fit; same-sequence geometry estimates, not scene acceptance."""
import argparse
import hashlib
import json
import math
import numpy as np
from scipy.optimize import least_squares
from .build import ROOT,SCENE,camera_pose
from .evidence import build_fingerprint
from .fit_camera import basis,project

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'
READINGS=[
    ([-1.86,-.62,.71],[445,505]),([.26,-.62,.71],[592,447]),
    ([.26,-2.42,.71],[645,531]),([-1.86,-2.42,.71],[503,594]),
    ([2.51,-.62,.71],[724,395]),([2.51,-2.42,.71],[774,480])]
CHECKS=[([2.94,-1.9,.95],[791,454]),([6.66,-1.9,.95],[1018,361]),
        ([2.94,-.25,.95],[738,370]),([6.66,-.25,.95],[962,286])]


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    if args.apply and args.bind_captures:raise SystemExit('Apply before capture; bind afterwards.')
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    instance=next(i for i in scene['instances'] if i['id']=='sorting_r09_system')
    x,y,z,_,__,yaw=instance['pose'];co,si=math.cos(yaw),math.sin(yaw)
    def world(p):return [x+co*p[0]-si*p[1],y+si*p[0]+co*p[1],z+p[2]]
    points=np.array([world(r[0]) for r in READINGS]);pixels=np.array([r[1] for r in READINGS])
    checks=np.array([world(r[0]) for r in CHECKS]);check_pixels=np.array([r[1] for r in CHECKS])
    camera=next(c for c in scene['cameras'] if c['id']=='sorting');pose=camera_pose(camera)
    parameters=np.array(pose[:3]+[pose[5],pose[4]]);focal=640/math.tan(camera['horizontal_fov']/2)
    if args.apply:
        result=least_squares(lambda q:(project(q,points,focal)-pixels).ravel(),parameters,
            bounds=([40,25,4,-3.1,.2],[70,55,40,-1.6,1.5]),max_nfev=4000)
        parameters=result.x;eye,forward,_,_=basis(parameters)
        camera.update(position=eye.tolist(),target=(eye+10*forward).tolist(),
            registration='manual_sparse_landmark_fit',
            registration_report='visual_comparison/wheelset_workshop_v1/camera_fit_sorting.json')
        SCENE.write_text(json.dumps(scene,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fitted=project(parameters,points,focal);checked=project(parameters,checks,focal)
    view={'camera':'sorting','reference_second':21,'parameters':parameters.tolist(),
        'fit_count':len(points),'check_count':len(checks),
        'fit_rmse_px':float(np.sqrt(np.mean(np.sum((fitted-pixels)**2,axis=1)))),
        'check_rmse_px':float(np.sqrt(np.mean(np.sum((checked-check_pixels)**2,axis=1)))),
        'landmarks':[]}
    for role,rows,worlds,projections in [('fit',READINGS,points,fitted),('check',CHECKS,checks,checked)]:
        view['landmarks'].extend({'name':f'{role}_{n+1}','world_point':p.tolist(),'source_pixel':r[1],
            'role':role,'projection':pr.tolist()} for n,(r,p,pr) in enumerate(zip(rows,worlds,projections)))
    report={'method':'Six manually read bin rim corners; four table corners excluded from pose fit; FOV fixed 0.92 rad',
        'pixel_reading_tolerance_estimate':15,'geometry_measured':False,
        'geometry_adjusted_using_same_sequence':True,'views':[view],
        'claim_boundary':'Bin and table dimensions estimated from the same sequence. Checks are not independent. No full-scene, count, dynamics or scheduling acceptance.'}
    if args.bind_captures:
        report['build_fingerprint']=build_fingerprint()
        captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8'))
        capture=next(c for c in captures if c['camera']=='sorting');p=capture['requested_pose']
        assert capture['build_fingerprint']==report['build_fingerprint']
        assert math.dist(p[:3]+[p[5],p[4]],parameters)<1e-6
        view['gazebo_image_sha256']=hashlib.sha256((OUT/'gazebo'/capture['file']).read_bytes()).hexdigest()
        assert view['gazebo_image_sha256']==capture['sha256']
        view['reference_sha256']=hashlib.sha256((OUT/'references/frame_21s.png').read_bytes()).hexdigest()
        from .fit_inspection_cameras import plot
        plot(report,footer='Bin rims fit, table corner checks; geometry tuned from same sequence. Not independent visual acceptance.')
    (OUT/'camera_fit_sorting.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:view[k] for k in ['fit_rmse_px','check_rmse_px']}))


if __name__=='__main__':main()
