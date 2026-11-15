"""Fit one camera to manually identified image landmarks; never a scene-fidelity score."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .build import ROOT, SCENE, camera_pose
from .evidence import build_fingerprint

# Approximate pixel readings from the unchanged 1280x720 source frame at 62 seconds.
# Fit uses visible wheel hubs, the transverse beam ends, and the front motor end.
LANDMARKS = [
    ('left_wheel_hub', [98,17+.77,.78], [502,329], 'fit'),
    ('right_wheel_hub', [98,17-.77,.78], [781,329], 'fit'),
    ('left_beam_end', [98.45,17+2.25,1.45], [219,179], 'fit'),
    ('right_beam_end', [98.45,17-2.25,1.45], [1028,180], 'fit'),
    ('front_motor_end', [97.35,17,.35], [637,470], 'fit'),
    ('left_ram_end', [98,17+1.08,.78], [433,329], 'check'),
    ('right_ram_end', [98,17-1.08,.78], [849,329], 'check'),
]


def basis(parameters):
    x,y,z,yaw,pitch=parameters
    cy,sy,cp,sp=math.cos(yaw),math.sin(yaw),math.cos(pitch),math.sin(pitch)
    return np.array([x,y,z]),np.array([cy*cp,sy*cp,-sp]),np.array([-sy,cy,0]),np.array([cy*sp,sy*sp,cp])


def project(parameters, points, focal):
    eye,forward,left,up=basis(parameters)
    delta=points-eye;depth=delta@forward
    return np.column_stack((640-focal*(delta@left)/depth,360-focal*(delta@up)/depth))


def projection_report(camera, points, pixels, fit, focal):
    """Project the same landmarks with the camera stored in the scene, without moving it."""
    pose=camera_pose(camera)
    parameters=np.array(pose[:3]+[pose[5],pose[4]])
    projection=project(parameters,points,focal)

    def rmse(mask):return float(np.sqrt(np.mean(np.sum((projection[mask]-pixels[mask])**2,axis=1))))
    return parameters,projection,rmse(fit),rmse(~fit)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--verify',action='store_true',help='Re-check the stored pose against current geometry; never moves the camera.')
    args=parser.parse_args()
    if args.apply and args.verify:raise SystemExit('Choose either --apply or --verify.')
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    camera=next(c for c in scene['cameras'] if c['id']=='press_front')
    offset=scene.get('layout_offsets',{}).get('assembly_process_x',0)
    dy=scene.get('layout_offsets',{}).get('press_reference_y',0)
    landmarks=[(name,[point[0]+offset,point[1]+dy,point[2]],pixel,role) for name,point,pixel,role in LANDMARKS]
    points=np.array([v[1] for v in landmarks]);pixels=np.array([v[2] for v in landmarks]);fit=np.array([v[3]=='fit' for v in landmarks])
    pose=camera_pose(camera);initial=np.array(pose[:3]+[pose[5],pose[4]])
    focal=640/math.tan(camera['horizontal_fov']/2)
    if args.verify:
        # Report how the stored pose now projects the same landmarks. This never rewrites the scene.
        stored,after,after_fit,after_check=projection_report(camera,points,pixels,fit,focal)
        report={'camera':'press_front','reference_second':62,'image_size':[1280,720],
                'mode':'verify_stored_pose_against_current_geometry',
                'stored_pose':[camera['position'],camera['target']],
                'fitted_parameters':stored.tolist(),
                'registration':camera.get('registration'),
                'pixel_reading_tolerance_estimate':8,'world_geometry_measured':False,
                'after_fit_rmse_px':after_fit,'after_check_rmse_px':after_check,
                'landmarks':[{'name':v[0],'world_point':v[1],'source_pixel':v[2],'role':v[3],
                              'current_projection':a.tolist()} for v,a in zip(landmarks,after)],
                'geometry_adjusted_using_same_reference':True,
                'check_points_note':'Two additional points excluded from pose fitting, but from the same source frame; not an independent scene-validation set.',
                'applied':True,
                'claim_boundary':'Sparse landmark alignment only; not full-image, full-scene, engineering-dimension, or dynamics acceptance.'}
        report['build_fingerprint']=build_fingerprint()
        directory=ROOT/'visual_comparison/wheelset_workshop_v1'
        (directory/'camera_fit_press_front.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        print(json.dumps({k:report[k] for k in ['after_fit_rmse_px','after_check_rmse_px','applied']}))
        return
    result=least_squares(lambda p:(project(p,points[fit],focal)-pixels[fit]).ravel(),initial,
                         bounds=([87+offset,13,2,-.5,.15],[96+offset,21,10,.5,1.2]),max_nfev=2000)
    before=project(initial,points,focal);after=project(result.x,points,focal)
    def rmse(projection,mask):return float(np.sqrt(np.mean(np.sum((projection[mask]-pixels[mask])**2,axis=1))))
    report={'camera':'press_front','reference_second':62,'image_size':[1280,720],
            'method':'Manual approximate correspondences; five-parameter pose fit; FOV fixed at initial assumption; no lens calibration',
            'pixel_reading_tolerance_estimate':8,'world_geometry_measured':False,
            'before_fit_rmse_px':rmse(before,fit),'after_fit_rmse_px':rmse(after,fit),
            'before_check_rmse_px':rmse(before,~fit),'after_check_rmse_px':rmse(after,~fit),
            'initial_parameters':initial.tolist(),'fitted_parameters':result.x.tolist(),
            'landmarks':[{'name':v[0],'world_point':v[1],'source_pixel':v[2],'role':v[3],
                          'before_projection':b.tolist(),'after_projection':a.tolist()} for v,b,a in zip(landmarks,before,after)],
            'geometry_adjusted_using_same_reference':True,
            'check_points_note':'Two additional points excluded from pose fitting, but from the same source frame; not an independent scene-validation set.',
            'applied':args.apply,'claim_boundary':'Sparse landmark alignment only; not full-image, full-scene, engineering-dimension, or dynamics acceptance.'}
    if args.apply:
        eye,forward,_,_=basis(result.x)
        camera['position']=eye.tolist();camera['target']=(eye+forward*10).tolist()
        camera['registration']='manual_sparse_landmark_fit'
        camera['registration_report']='visual_comparison/wheelset_workshop_v1/camera_fit_press_front.json'
        SCENE.write_text(json.dumps(scene,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    directory=ROOT/'visual_comparison/wheelset_workshop_v1'
    (directory/'camera_fit_press_front.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['before_fit_rmse_px','after_fit_rmse_px','before_check_rmse_px','after_check_rmse_px','applied']}))


if __name__=='__main__':main()
