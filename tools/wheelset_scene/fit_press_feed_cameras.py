"""Fixed ring-magazine corners in 58/62 s; current static layout, not dynamics acceptance."""
import argparse
import hashlib
import json
import math
import numpy as np
from scipy.optimize import least_squares
from .build import ROOT, SCENE, camera_pose
from .evidence import build_fingerprint
from .fit_camera import basis, project

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'
READINGS={
    'press':[[506,422],[426,389],[568,376],[502,461],[434,434],[566,414]],
    'press_front':[[268,533],[184,533],[287,486],[268,569],[188,569],[285,532]]}


def landmarks(scene):
    instance=next(i for i in scene['instances'] if i['id']=='bearing_feed_module_004')
    assert instance['asset']=='bearing_feed_module' and abs(instance['pose'][5])<1e-8
    px,py,pz=instance['pose'][:3]
    x,y,z=px+1.8134601,py+.43881241,pz+.43292097
    sx,sy,sz=.35584701,.38928885,.21919705
    return np.array([[x+a*sx/2,y+b*sy/2,z+c*sz/2]
        for c in [1,-1] for a,b in [(-1,-1),(-1,1),(1,-1)]])


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    if args.apply and args.bind_captures:raise SystemExit('Apply before capture; bind afterwards.')
    scene=json.loads(SCENE.read_text(encoding='utf-8'));points=landmarks(scene)
    report={'method':'Six shared magazine corners per view. Magazine geometry and 58 s camera jointly estimated; 62 s main-machine camera kept fixed.',
        'pixel_reading_tolerance_estimate':15,'geometry_measured':False,
        'geometry_adjusted_using_same_sequence':True,'views':[],
        'claim_boundary':'Same-video sparse fixed-object comparison only. Moving wheelsets/bearings and arm poses do not match both times; no independent full-scene or dynamics acceptance.'}
    for cid in ['press','press_front']:
        camera=next(c for c in scene['cameras'] if c['id']==cid);p=camera_pose(camera)
        parameters=np.array(p[:3]+[p[5],p[4]]);pixels=np.array(READINGS[cid])
        focal=640/math.tan(camera['horizontal_fov']/2)
        if args.apply and cid=='press':
            result=least_squares(lambda q:(project(q,points,focal)-pixels).ravel(),parameters,
                bounds=([120,7,2,.01,.1],[132,18.4,15,1.55,1.4]),max_nfev=2000)
            parameters=result.x;eye,forward,_,_=basis(parameters)
            camera.update(position=eye.tolist(),target=(eye+10*forward).tolist())
        projections=project(parameters,points,focal)
        view={'camera':cid,'diagnostic_id':'press_feed' if cid=='press' else 'press_feed_front',
            'reference_second':camera['reference_second'],'parameters':parameters.tolist(),
            'fit_count':6,'check_count':0,'check_rmse_px':None,
            'fit_rmse_px':float(np.sqrt(np.mean(np.sum((projections-pixels)**2,axis=1)))),
            'landmarks':[{'name':f'magazine_{n+1}','world_point':w.tolist(),'source_pixel':p.tolist(),
                'projection':pr.tolist(),'role':'fit'} for n,(w,p,pr) in enumerate(zip(points,pixels,projections))]}
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
        plot(report,footer='Shared fixed magazine corners; dimensions fitted using both frames. No independent checks; moving parts and occlusions differ.')
    (OUT/'camera_fit_press_feed.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'camera':v['camera'],'fit_rmse_px':v['fit_rmse_px']} for v in report['views']]))


if __name__=='__main__':main()
