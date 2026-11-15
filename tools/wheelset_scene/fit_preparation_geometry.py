"""Reproduce the 43/45/48 s shared roof fit with distinct preparation fixtures.

Default: report the applied scene. --optimize also emits a candidate seeded by
the corrected 0.23 geometry. It never edits the scene or rotates a camera seed.
"""
import argparse
import hashlib
import json
import math
import numpy as np
from scipy.optimize import least_squares
from .build import ROOT, SCENE, camera_pose
from .evidence import build_fingerprint
from .fit_camera import project
from .component_geometry import (PREPARATION_PORTAL_HALF_X, PREPARATION_PORTAL_HALF_Y,
    PREPARATION_PORTAL_TOP, PREPARATION_ROOF_PIXELS_43, PREPARATION_PORTAL_ROOF_PIXELS,
    WIPING_CENTRAL_HUB_PIXELS, GREASE_HUB_PIXELS, GREASE_ROOF_PIXELS,
    WIPING_BOX_X, WIPING_BOX_Y, greasing_landmarks, preparation_portal_roof)

OUT = ROOT / 'visual_comparison/wheelset_workshop_v1'
NAMES = ['central45', 'roof45', 'roof43', 'corner48', 'edge48', 'grease48']
WEIGHTS = [.5, 1, 1, 1, 1, .5]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimize', action='store_true')
    parser.add_argument('--bind-captures', action='store_true')
    args = parser.parse_args()
    scene = json.loads(SCENE.read_text(encoding='utf-8'))
    params, focals = {}, {}
    for cid in ['component_overview', 'wiping', 'greasing']:
        camera = next(c for c in scene['cameras'] if c['id'] == cid)
        p = camera_pose(camera)
        params[cid] = np.array(p[:3] + [p[5], p[4]])
        focals[cid] = 640 / math.tan(camera['horizontal_fov'] / 2)
    machine = max((i for i in scene['instances'] if i['asset'] == 'greasing_station'),
                  key=lambda i:i['pose'][1])
    wipe = max((i for i in scene['instances'] if i['asset'] == 'wiping_station'),
               key=lambda i:i['pose'][1])
    portal = max((i for i in scene['instances'] if i['asset'] == 'opening_portal'
                  and i['zone'] == 'preparation'), key=lambda i:i['pose'][1])
    assert abs(machine['pose'][5] - math.pi) < 1e-8
    assert abs(portal['pose'][5]) < 1e-8
    y = machine['pose'][1]
    hubs45 = np.array([[wipe['pose'][0], wipe['pose'][1]+side*.77, .67] for side in [-1,1]])
    def roof(q):
        x,cy,z,hx,hy = q
        return np.array([[x+a,cy+b,z] for a,b in [(-hx,-hy),(hx,-hy),(hx,hy),(-hx,hy)]])
    def grease(x):
        points = [[x,y-.77,.67],[x,y+.77,.67]]
        for side in [1,-1]:
            for a,b in [(-.19,-.19),(-.19,.19),(.19,.19),(.19,-.19)]:
                points.append([x-WIPING_BOX_X+a,y-WIPING_BOX_Y[side]+b,.8])
        return np.array(points)
    def blocks(v):
        c45, rp, c48 = v[:5], roof(v[5:10]), v[10:15]
        p48 = project(c48,rp,focals['greasing'])
        edge = p48[0]-p48[3];delta = np.array([719,571])-p48[3]
        return [(project(c45,hubs45,focals['wiping'])-WIPING_CENTRAL_HUB_PIXELS).ravel(),
            (project(c45,rp,focals['wiping'])-PREPARATION_PORTAL_ROOF_PIXELS).ravel(),
            (project(params['component_overview'],rp,focals['component_overview'])-PREPARATION_ROOF_PIXELS_43).ravel(),
            p48[3]-[1240,284],
            np.array([(edge[0]*delta[1]-edge[1]*delta[0])/np.linalg.norm(edge)]),
            (project(c48,grease(v[15]),focals['greasing'])-np.array(GREASE_HUB_PIXELS+GREASE_ROOF_PIXELS)).ravel()]
    def residual(v):
        return np.concatenate([b*w for b,w in zip(blocks(v),WEIGHTS)])
    v = np.r_[params['wiping'], portal['pose'][:2], portal['pose'][2]+PREPARATION_PORTAL_TOP,
        PREPARATION_PORTAL_HALF_X,PREPARATION_PORTAL_HALF_Y,params['greasing'],machine['pose'][0]]
    assert np.allclose(roof(v[5:10]),preparation_portal_roof(scene),atol=1e-8)
    assert np.allclose(grease(v[15]),greasing_landmarks(scene),atol=1e-8)
    def describe(q):
        return {'parameters':q.tolist(),
            'coordinate_rmse_px':{n:float(np.sqrt(np.mean(b*b))) for n,b in zip(NAMES,blocks(q))},
            'weighted_scalar_rmse_px':float(np.sqrt(np.mean(residual(q)**2)))}
    report = {'build_fingerprint':build_fingerprint(),
        'scene_sha256':hashlib.sha256(SCENE.read_bytes()).hexdigest(),
        'parameter_order':['camera45_x_y_z_yaw_pitch','portal_x_y_top_halfX_halfY',
                           'camera48_x_y_z_yaw_pitch','greasing_center_x'],
        'method':'43 s camera and FOV .92 remain fixed; jointly estimate shared roof, 45/48 s cameras and separate greasing X. Hubs and cabinet roofs have weight .5; fixed roof points/edge have weight 1.',
        'definition':'Coordinate RMSE averages individual x/y residuals. Corner48 point distance is sqrt(2)*coordinate RMSE; edge48 is scalar distance. Other camera reports use point RMSE. These definitions are not interchangeable.',
        'weights':dict(zip(NAMES,WEIGHTS)), 'applied':describe(v),
        'source_interpretation':'43 s shows low greasing station, gray portal, tall wiping station, then exit wheel. Both stations exist in the same static scene for every camera; 48 s views the separate greasing fixture across the portal.',
        'independent_check_points':0, 'geometry_measured':False,
        'claim_boundary':'All targets come from the same proposal animation. Dimensions and wheel scale remain estimates; unseen mechanisms, motion and contact unresolved. Near roof corner48 caption-occluded, not invented. A lower local residual does not establish whole-scene acceptance.',
        'views':[]}
    for cid,second in [('component_overview',43),('wiping',45),('greasing',48)]:
        ref=OUT/f'references/frame_{second:02d}s.png'
        view={'camera':cid,'reference_second':second,
              'reference_sha256':hashlib.sha256(ref.read_bytes()).hexdigest()}
        if args.bind_captures:
            captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8'))
            capture=next(c for c in captures if c['camera']==cid)
            assert capture['build_fingerprint']==report['build_fingerprint']
            p=capture['requested_pose']
            assert math.dist(p[:3]+[p[5],p[4]],params[cid])<1e-6
            view['gazebo_image_sha256']=hashlib.sha256((OUT/'gazebo'/capture['file']).read_bytes()).hexdigest()
            assert view['gazebo_image_sha256']==capture['sha256']
        report['views'].append(view)
    if args.optimize:
        low=[84,12,3,1.8,.2,77,23,1.2,.12,1.4,78,15,2,1.9,.2,73]
        high=[102,26,18,3.1,1.2,83,29,3.5,1.6,3.2,91,29,12,3.4,1.2,80.5]
        result=least_squares(residual,v,bounds=(low,high),max_nfev=15000)
        report['reproduced_candidate']=dict(describe(result.x),optimizer_success=bool(result.success),
            maximum_applied_parameter_difference=float(np.max(np.abs(result.x-v))),scene_modified=False)
    (OUT/'camera_fit_preparation_geometry.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report['applied']))


if __name__=='__main__':main()
