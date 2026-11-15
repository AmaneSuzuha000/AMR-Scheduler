"""Source45 wiping brackets and source48 infeed features; same-video estimates."""
import argparse
import hashlib
import json
import math
import numpy as np
from .build import ROOT,SCENE,camera_pose
from .evidence import build_fingerprint
from .fit_camera import project
from .preparation_details import (WIPING_TRIANGLE_XZ,WIPING_TRIANGLE_SPAN,
    WIPING_TRIANGLE_PIXELS_45,INFEED_POST_Y,INFEED_POST_HEIGHT,
    INFEED_POST_PIXELS_48,INFEED_POST_PIXELS_43,INFEED_BENCH_END_X,
    INFEED_BENCH_Y,INFEED_BENCH_HEIGHT,INFEED_BENCH_PIXELS_48,FIN_UNIT_CENTER)

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    wipe=max((i for i in scene['instances'] if i['asset']=='wiping_station'),key=lambda i:i['pose'][1])
    infeed=max((i for i in scene['instances'] if i['asset']=='preparation_infeed_bench'),key=lambda i:i['pose'][1])
    x,y,z=wipe['pose'][:3]
    triangles=[[x+dx,y+side*WIPING_TRIANGLE_SPAN,z+h] for side in [-1,1] for dx,h in WIPING_TRIANGLE_XZ]
    x,y,z=infeed['pose'][:3]
    posts=[[x,y+dy,z+h] for dy in INFEED_POST_Y for h in [INFEED_POST_HEIGHT,0]]
    benches=[[INFEED_BENCH_END_X,y+dy,INFEED_BENCH_HEIGHT] for dy in INFEED_BENCH_Y]
    readings={
        'wiping':[(f'bracket_vertex_{n+1}',p,px,'fit') for n,(p,px) in enumerate(zip(triangles,WIPING_TRIANGLE_PIXELS_45))],
        'greasing':[(f'post_approx_centre_{n+1}',p,px,'fit') for n,(p,px) in enumerate(zip(posts,INFEED_POST_PIXELS_48))]
            +[(f'bench_end_centre_{n+1}',p,px,'fit') for n,(p,px) in enumerate(zip(benches,INFEED_BENCH_PIXELS_48))]
            +[('fin_unit_assumed_base_centre',FIN_UNIT_CENTER,[886,155],'fit')],
        'component_overview':[(f'post_approx_centre_{n+1}',p,px,'check') for n,(p,px) in enumerate(zip(posts,INFEED_POST_PIXELS_43))]}
    report={'build_fingerprint':build_fingerprint(),
        'method':'Cameras unchanged from0.23.45s six triangle vertices estimate mirrored XZ plates and Y span.48s approximate post roof/foot centres estimate shared X, twoY positions and height; .72m bench height is assumed and end centres set local positions.43s post readings are same-sequence residual checks, with uncertain low-resolution centre correspondences. Fin-unit ground centre derived from48s; envelope and tube counts are estimated.',
        'geometry_measured':False,'geometry_adjusted_using_same_sequence':True,
        'independent_check_points':0,'pixel_reading_tolerance_estimate':15,
        'claim_boundary':'Approximate centres are not precise physical corner correspondences. Local fits do not validate whole equipment or global context. Fin-unit process is unknown. Repeated infeed details on3lanes, stock counts, all dimensions and hidden support/contact remain estimates. No kinematics, dynamics or scheduling acceptance.',
        'views':[]}
    # A centre ray must cross both sides within the hole radius. This checks
    # plate-wall occlusion at the stored45s view, not complete ray tracing.
    camera=next(c for c in scene['cameras'] if c['id']=='wiping');eye=np.array(camera['position'])
    hole_checks=[]
    for side in [-1,1]:
        centre=np.array([wipe['pose'][0]-.025,wipe['pose'][1]+side*WIPING_TRIANGLE_SPAN,1.335])
        ray=centre-eye;distances=[]
        for offset in [-.035/2,.035/2]:
            at=eye+(centre[1]+offset-eye[1])/ray[1]*ray
            distances.append(float(np.linalg.norm((at-centre)[[0,2]])))
        hole_checks.append({'side':side,'world_centre':centre.tolist(),
            'wall_plane_offsets_from_hole_centre_m':distances,'radius_m':.07,
            'centre_ray_clears_both_walls':max(distances)<.07})
    report['static_hole_wall_visibility']={'checks':hole_checks,
        'passed':all(h['centre_ray_clears_both_walls'] for h in hole_checks),
        'scope':'Centre-ray intersection with two plate planes only; other scene occlusion and image appearance checked in actual captures, no dynamic/contact claim.'}
    captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8')) if args.bind_captures else []
    for cid,rows in readings.items():
        camera=next(c for c in scene['cameras'] if c['id']==cid);p=camera_pose(camera)
        parameters=p[:3]+[p[5],p[4]];points=np.array([r[1] for r in rows]);pixels=np.array([r[2] for r in rows])
        projected=project(parameters,points,640/math.tan(camera['horizontal_fov']/2))
        errors=np.sum((projected-pixels)**2,axis=1);fit=np.array([r[3]=='fit' for r in rows])
        view={'camera':cid,'reference_second':camera['reference_second'],'parameters':parameters,
            'fit_count':int(sum(fit)),'check_count':int(sum(~fit)),
            'fit_rmse_px':float(np.sqrt(np.mean(errors[fit]))) if any(fit) else None,
            'check_rmse_px':float(np.sqrt(np.mean(errors[~fit]))) if any(~fit) else None,
            'landmarks':[{'name':r[0],'world_point':r[1],'source_pixel':r[2],'role':r[3],
                'projection':pr.tolist()} for r,pr in zip(rows,projected)]}
        if args.bind_captures:
            capture=next(c for c in captures if c['camera']==cid);p=capture['requested_pose']
            assert capture['build_fingerprint']==report['build_fingerprint']
            assert math.dist(p[:3]+[p[5],p[4]],parameters)<1e-6
            view['gazebo_image_sha256']=hashlib.sha256((OUT/'gazebo'/capture['file']).read_bytes()).hexdigest()
            assert view['gazebo_image_sha256']==capture['sha256']
            view['reference_sha256']=hashlib.sha256((OUT/'references'/f"frame_{camera['reference_second']:02d}s.png").read_bytes()).hexdigest()
        report['views'].append(view)
    if args.bind_captures:
        from .fit_inspection_cameras import plot
        for view in report['views']:
            plot({'views':[dict(view,diagnostic_id=view['camera']+'_preparation_details')]},
                footer='Same-source manual vertices / approximate feature centres.43s checks are not independent. Shapes and hidden mechanisms estimated.')
    (OUT/'camera_fit_preparation_details.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v[k] for k in ['camera','fit_rmse_px','check_rmse_px']} for v in report['views']]))


if __name__=='__main__':main()
