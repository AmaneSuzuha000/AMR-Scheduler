"""Fit static gantry corners; moving bridge/clamp never used as camera landmarks."""
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
# World points refer to the stationary frame. Approximate pixel reads at 1280 x 720.
# Both frames also informed geometry. Check points are NOT an independent dataset.
READINGS={
    'storage':[
        ([74,51.4,4.02],[499,668],'fit'),
        ([82,51.4,4.02],[191,379],'fit'),
        ([74,33.4,4.02],[1122,131],'fit'),
        ([82,51.4,.12],[233,508],'check'),
        ([74,33.4,.12],[1058,268],'fit'),
    ],
    'storage_reverse':[
        ([74,51.4,4.02],[253,260],'fit'),
        ([82,51.4,4.02],[570,168],'fit'),
        ([74,51.4,.12],[273,384],'fit'),
        ([82,51.4,.12],[581,291],'fit'),
        ([76.7,49.27,1.165],[408,376],'check'),
        ([87.9,49.27,1.165],[812,244],'check'),
    ],
}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    if args.apply and args.bind_captures:raise SystemExit('Apply before capture, then bind in a separate call.')
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    report={'method':'Four stationary-frame points per camera; fixed FOV 0.92 rad, five pose parameters',
        'pixel_reading_tolerance_estimate':10,'geometry_measured':False,
        'geometry_adjusted_using_same_sequence':True,'views':[],
        'claim_boundary':'Manual sparse pose fit, same-sequence checks; not full-image, independent, motion or dimension acceptance.'}
    for cid,rows in READINGS.items():
        offset=scene.get('layout_offsets',{}).get('storage_y',0)
        offset_x=scene.get('layout_offsets',{}).get('storage_x',0)
        rows=[([point[0]+offset_x,point[1]+offset,point[2]],pixel,role) for point,pixel,role in rows]
        camera=next(c for c in scene['cameras'] if c['id']==cid)
        points=np.array([r[0] for r in rows]);pixels=np.array([r[1] for r in rows]);fit=np.array([r[2]=='fit' for r in rows])
        pose=camera_pose(camera);parameters=np.array(pose[:3]+[pose[5],pose[4]])
        focal=640/math.tan(camera['horizontal_fov']/2)
        if args.apply:
            initial=[66,57,18,-.8,.66] if cid=='storage' else [64,28,17,.78,.63]
            initial[0]+=offset_x;initial[1]+=offset
            result=least_squares(lambda v:(project(v,points[fit],focal)-pixels[fit]).ravel(),initial,
                bounds=([35+offset_x,5,7,-1.5,.2],[78+offset_x,85,55,1.5,1.4]),max_nfev=3000)
            parameters=result.x;eye,forward,_,_=basis(parameters)
            camera['position']=eye.tolist();camera['target']=(eye+10*forward).tolist()
            camera['registration']='manual_sparse_landmark_fit'
            camera['registration_report']='visual_comparison/wheelset_workshop_v1/camera_fit_storage.json'
        projection=project(parameters,points,focal)
        def rmse(mask):return float(np.sqrt(np.mean(np.sum((projection[mask]-pixels[mask])**2,axis=1))))
        report['views'].append({'camera':cid,'reference_second':camera['reference_second'],
            'parameters':parameters.tolist(),'fit_rmse_px':rmse(fit),'check_rmse_px':rmse(~fit),
            'fit_count':int(fit.sum()),'check_count':int((~fit).sum()),
            'landmarks':[{'name':f'static_frame_{i+1}','world_point':r[0],'source_pixel':r[1],
                'role':r[2],'projection':p.tolist()} for i,(r,p) in enumerate(zip(rows,projection))]})
    if args.apply:SCENE.write_text(json.dumps(scene,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.bind_captures:
        report['build_fingerprint']=build_fingerprint()
        captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8'))
        for view in report['views']:
            c=next(c for c in captures if c['camera']==view['camera']);p=c['requested_pose']
            assert math.dist(p[:3]+[p[5],p[4]],view['parameters'])<1e-6
            assert c['build_fingerprint']==report['build_fingerprint']
            assert hashlib.sha256((OUT/'gazebo'/c['file']).read_bytes()).hexdigest()==c['sha256']
            view['gazebo_image_sha256']=c['sha256']
            view['reference_sha256']=hashlib.sha256((OUT/'references'/f"frame_{view['reference_second']:02d}s.png").read_bytes()).hexdigest()
        from .fit_inspection_cameras import plot
        plot(report, footer='Static frame and transfer-rail checks; geometry tuned using this sequence. Moving bridge is one snapshot, not motion acceptance.')
    (OUT/'camera_fit_storage.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'camera':v['camera'],'fit_rmse_px':v['fit_rmse_px'],'check_rmse_px':v['check_rmse_px']} for v in report['views']]))

if __name__=='__main__':main()
