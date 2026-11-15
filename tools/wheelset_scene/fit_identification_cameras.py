"""Approximate view matching to four manually read hubs; moving-scene caveats are explicit."""
import argparse
import hashlib
import json
import math
import numpy as np
from scipy.optimize import least_squares
from .build import ROOT, SCENE, camera_pose
from .fit_camera import project, basis
from .evidence import build_fingerprint
from .front_geometry import (IDENTIFICATION_ROBOT_JOINTS,
    IDENTIFICATION_ROBOT_PIXELS_3,IDENTIFICATION_ROBOT_PIXELS_4)

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'
PIXELS={'identification':[[618,519],[748,434],[993,255],[1052,197]],
        'identification_close':[[554,449],[656,368],[991,270],[1036,206]]}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    offset=scene.get('layout_offsets',{}).get('identification_x',0)
    x=14+offset
    points=np.array([[x,9-.77,.56],[x,9+.77,.56],[x,15-.77,.56],[x,15+.77,.56]])
    report={'method':'Four approximate manual wheel-hub correspondences per view; pose only, fixed FOV',
        'independent_check_points':0,'measured_geometry':False,
        'claim_boundary':'Both frames contain moving equipment. This is a static visual snapshot, not temporal registration or independent scene acceptance.',
        'views':[]}
    from .assets import overhead_robot,identification_end_covers
    arm_low=overhead_robot().bounds()[0][2];hood_high=identification_end_covers().bounds()[1][2]
    minimum_arm_height=min(i['pose'][2]+arm_low for i in scene['instances'] if i['asset']=='overhead_robot')
    maximum_hood_height=max(i['pose'][2]+hood_high for i in scene['instances'] if i['asset']=='identification_hood')
    report['static_robot_hood_height_separation']={'passed':minimum_arm_height>maximum_hood_height,
        'minimum_robot_mesh_height_m':minimum_arm_height,'maximum_hood_mesh_height_m':maximum_hood_height,
        'minimum_z_gap_m':minimum_arm_height-maximum_hood_height,
        'scope':'All two static identification arms and hoods, including probe thickness and cover screws. Z bounds exclude arm/hood visual-mesh intersection in this snapshot; no motion, collision-shape or contact validation.'}
    assert report['static_robot_hood_height_separation']['passed'], 'Static identification arm intersects hood height range'
    for cid,pixels in PIXELS.items():
        camera=next(c for c in scene['cameras'] if c['id']==cid)
        p=camera_pose(camera);initial=np.array(p[:3]+[p[5],p[4]])
        pixels=np.array(pixels);focal=640/math.tan(camera['horizontal_fov']/2)
        if args.apply:
            result=least_squares(lambda v:(project(v,points,focal)-pixels).ravel(),initial,
                bounds=([14+offset,-12,2,1.6,.15],[32+offset,8,18,2.9,1.2]),max_nfev=2000)
            parameters=result.x;eye,forward,_,_=basis(parameters)
            camera['position']=eye.tolist();camera['target']=(eye+10*forward).tolist()
            camera['registration']='manual_sparse_landmark_fit'
            camera['registration_report']='visual_comparison/wheelset_workshop_v1/camera_fit_identification.json'
        else:parameters=initial
        projected=project(parameters,points,focal)
        item=next(i for i in scene['instances'] if i['id']=='overhead_robot_001')
        x,y,z,_,_,yaw=item['pose'];co,si=math.cos(yaw),math.sin(yaw)
        robot_points=np.array([[x+co*a-si*b,y+si*a+co*b,z+c] for a,b,c in IDENTIFICATION_ROBOT_JOINTS])
        robot_pixels=np.array(IDENTIFICATION_ROBOT_PIXELS_4 if cid=='identification_close' else IDENTIFICATION_ROBOT_PIXELS_3)
        robot_projected=project(parameters,robot_points,focal)
        robot_rows=[{'name':name,'world_point':point.tolist(),'source_pixel':pixel.tolist(),
                     'projection':pr.tolist(),'role':'fit' if cid=='identification_close' else 'motion_mismatch'}
                    for name,point,pixel,pr in zip(['shoulder_base','shoulder','elbow','wrist','thin_probe_tip'],
                        robot_points,robot_pixels,robot_projected)]
        report['views'].append({'camera':cid,'reference_second':camera['reference_second'],
            'parameters':parameters.tolist(),'fit_rmse_px':float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1)))),
            'world_points':points.tolist(),'manual_pixels':pixels.tolist(),'projections':projected.tolist(),
            'robot_pose_diagnostic':{'landmarks':robot_rows,'point_rmse_px':float(np.sqrt(np.mean(np.sum((robot_projected-robot_pixels)**2,axis=1)))),
                'probe_tip_height_m':float(robot_points[-1,2]),'hood_top_height_m':1.325,
                'probe_height_above_hood_top_m':float(robot_points[-1,2]-1.325),
                'independent_check_points':0,'manual_pixel_reading_tolerance_estimate':20,
                'role':'same_video_static_pose_fit' if cid=='identification_close' else 'moving_source_vs_shared_static_pose',
                'claim_boundary':'4 s landmarks tune one static arm. 3 s landmarks expose carriage/pose motion mismatch and are not fitting targets. Last point is a thin probe, not a wrist joint. Height separation is a static geometric check, not contact or kinematics validation.'}})
    if args.apply:SCENE.write_text(json.dumps(scene,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.bind_captures:
        captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8'))
        report['build_fingerprint']=build_fingerprint()
        for view in report['views']:
            capture=next(c for c in captures if c['camera']==view['camera']);pose=capture['requested_pose']
            assert math.dist(pose[:3]+[pose[5],pose[4]],view['parameters'])<1e-6
            assert capture['build_fingerprint']==report['build_fingerprint']
            assert hashlib.sha256((OUT/'gazebo'/capture['file']).read_bytes()).hexdigest()==capture['sha256']
            view['gazebo_image_sha256']=capture['sha256']
            view['reference_sha256']=hashlib.sha256((OUT/'references'/f"frame_{view['reference_second']:02d}s.png").read_bytes()).hexdigest()
        from .fit_inspection_cameras import plot
        plot({'views':[dict(v,diagnostic_id=v['camera']+'_robot',
            landmarks=v['robot_pose_diagnostic']['landmarks'],fit_count=5,
            fit_label='Same-video static pose fit' if v['camera']=='identification_close' else 'Static pose vs moving source',
            fit_rmse_px=v['robot_pose_diagnostic']['point_rmse_px'],check_count=0)
            for v in report['views']]},
            footer='4 s tunes the static pose; 3 s retains source motion mismatch. Final landmark is a thin probe. No motion or contact validation.')
    (OUT/'camera_fit_identification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'camera':v['camera'],'fit_rmse_px':v['fit_rmse_px']} for v in report['views']]))


if __name__=='__main__':main()
