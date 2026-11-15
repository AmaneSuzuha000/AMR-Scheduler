"""Sparse wiping-wheel and closing-tray pose diagnostics, with no full-scene claim."""
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


def landmarks(cid,scene=None):
    rows=[]
    if cid=='wiping':
        for x,z,pixels in [(80,.67,[[525,390],[682,327]]),(82.2,.56,[[687,478],[815,402]]),
                           (76.5,.67,[[374,274],[493,214]])]:
            for side,pixel in zip([-1,1],pixels):
                rows.append({'name':f'wheel_{x}_{side}','world_point':[x,8+side*.77,z],
                    'source_pixel':pixel,'role':'fit'})
    elif scene and scene.get('layout_offsets',{}).get('closing_fixed_revision')==16:
        pixels=([[651,360],[710,464],[800,324],[871,423]] if cid=='closing_wide' else
                [[763,363],[800,482],[906,346],[1001,458]])
        half=1.8674724519341654/2
        for (x,y),pixel in zip([(115.75,18.82+half),(115.75,18.82-half),
                                (117.55,18.82+half),(117.55,18.82-half)],pixels):
            rows.append({'name':f'tray_{x}_{y}','world_point':[x,y,.725],'source_pixel':pixel,'role':'fit'})
        for name,point,pixel in [
            ('fixed_rail_start',[113.27300334368132,19.8,2.9578215181231977],[409,235] if cid=='closing_wide' else [500,184]),
            ('fixed_rail_tip',[118.83258700399617,19.8,2.9578215181231977],[910,142] if cid=='closing_wide' else [1080,132])]:
            rows.append({'name':name,'world_point':point,'source_pixel':pixel,'role':'fit'})
    elif scene and scene.get('layout_offsets',{}).get('closing_side_revision')==12:
        pixels=([[651,360],[710,464],[800,324],[871,423]] if cid=='closing_wide' else
                [[763,363],[800,482],[906,346],[1001,458]])
        for (x,y),pixel in zip([(115.75,19.62),(115.75,18.02),(117.55,19.62),(117.55,18.02)],pixels):
            rows.append({'name':f'tray_{x}_{y}','world_point':[x,y,.725],'source_pixel':pixel,'role':'fit'})
    else:
        for (x,y),pixel in zip([(108.65,19.89),(111.15,19.89),(108.65,21.31),(111.15,21.31)],
                               [[763,363],[800,482],[906,346],[1001,458]]):
            rows.append({'name':f'tray_{x}_{y}','world_point':[x,y,.725],'source_pixel':pixel,'role':'fit'})
    return rows


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--apply',action='store_true')
    parser.add_argument('--cameras',default='wiping,closing,closing_wide')
    parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    if args.apply and args.bind_captures:raise SystemExit('Apply before capture; bind afterwards.')
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    report={'method':'Fixed FOV 0.92 rad; 0.23 wiping uses two central hubs and four portal corners in a joint 43/45/48 s roof fit. Separate greasing fixture confirmed in 43 s. Closing uses four tray corners and two fixed rail ends per view, with cabinet roof checks',
        'pixel_reading_tolerance_estimate':15,'geometry_measured':False,
        'geometry_adjusted_using_same_sequence':True,'views':[],
        'fixed_rail_correspondence_correction':'0.16 reads the visible fixed rail tip; the previous 67 s end reading overlapped the moving carriage and did not consistently identify the rail end.',
        'claim_boundary':'Sparse same-sequence camera and geometry fitting. Cabinet roof checks exclude pose fitting, but come from the same source sequence. No independent full-scene fidelity, kinematics or dimensions acceptance.'}
    for cid in ['wiping','closing']+(['closing_wide'] if any(c['id']=='closing_wide' for c in scene['cameras']) else []):
        camera=next(c for c in scene['cameras'] if c['id']==cid);rows=landmarks(cid,scene)
        lane_offset=(scene.get('layout_offsets',{}).get('wiping_lane_y',8)-8 if cid=='wiping'
            else scene.get('layout_offsets',{}).get('closing_reference_y',0))
        if cid=='wiping':lane_offset+=scene.get('layout_offsets',{}).get('preparation_top_y',0)
        process_offset=scene.get('layout_offsets',{}).get('assembly_process_x',0) if cid.startswith('closing') else 0
        if cid=='wiping':process_offset+=scene.get('layout_offsets',{}).get('preparation_x',0)
        closing_offset=scene.get('layout_offsets',{}).get('closing_block_x',0) if cid.startswith('closing') else 0
        process_offset+=closing_offset
        for row in rows:
            row['world_point'][1]+=lane_offset;row['world_point'][0]+=process_offset
        if cid=='wiping' and scene.get('layout_offsets',{}).get('preparation_fixed_revision')==23:
            from .component_geometry import preparation_portal_roof,WIPING_CENTRAL_HUB_PIXELS
            rows=[{'name':f'central_hub_{side}','world_point':[80+process_offset,8+lane_offset+side*.77,.67],
                'source_pixel':pixel,'role':'fit'} for side,pixel in zip([-1,1],WIPING_CENTRAL_HUB_PIXELS)]
            from .component_geometry import PREPARATION_PORTAL_ROOF_PIXELS
            rows.extend({'name':f'fixed_roof_{n+1}','world_point':point,'source_pixel':pixel,'role':'fit'}
                for n,(point,pixel) in enumerate(zip(preparation_portal_roof(scene),PREPARATION_PORTAL_ROOF_PIXELS)))
        points=np.array([r['world_point'] for r in rows]);pixels=np.array([r['source_pixel'] for r in rows])
        p=camera_pose(camera);parameters=np.array(p[:3]+[p[5],p[4]]);focal=640/math.tan(camera['horizontal_fov']/2)
        if args.apply and cid in args.cameras.split(','):
            initial=[89.3,1.3,6.3,2.49,.47] if cid=='wiping' else [119.6,18.14,4.73,3.08,.35]
            bounds=([82,-20,3,1.65,.2],[110,7,30,3.1,1.4]) if cid=='wiping' else ([111,5,4,1.4,.2],[135,19,25,3.1,1.4])
            initial[1]+=lane_offset;bounds[0][1]+=lane_offset;bounds[1][1]+=lane_offset
            initial[0]+=process_offset;bounds[0][0]+=process_offset;bounds[1][0]+=process_offset
            if cid.startswith('closing') and scene.get('layout_offsets',{}).get('closing_side_revision')==12:
                initial=parameters.tolist();bounds=([125,0,3,.2,.2],[165,22,25,2.3,1.3])
            result=least_squares(lambda v:(project(v,points,focal)-pixels).ravel(),initial,bounds=bounds,max_nfev=4000)
            parameters=result.x;eye,forward,_,_=basis(parameters)
            camera['position']=eye.tolist();camera['target']=(eye+10*forward).tolist()
            camera['registration']='manual_sparse_landmark_fit'
            camera['registration_report']='visual_comparison/wheelset_workshop_v1/camera_fit_service.json'
        projection=project(parameters,points,focal)
        for row,pixel in zip(rows,projection):row['projection']=pixel.tolist()
        view={'camera':cid,'reference_second':camera['reference_second'],'parameters':parameters.tolist(),
            'fit_rmse_px':float(np.sqrt(np.mean(np.sum((projection-pixels)**2,axis=1)))),
            'check_rmse_px':None,'fit_count':len(rows),'check_count':0,'landmarks':rows}
        if cid=='wiping':
            from .component_geometry import preparation_portal_roof,PREPARATION_PORTAL_ROOF_PIXELS
            roof_points=np.array(preparation_portal_roof(scene));roof_pixels=np.array(PREPARATION_PORTAL_ROOF_PIXELS)
            roof_projected=project(parameters,roof_points,focal)
            view['portal_roof_diagnostic']={'point_rmse_px':float(np.sqrt(np.mean(np.sum((roof_projected-roof_pixels)**2,axis=1)))),
                'landmarks':[{'name':f'roof_corner_{n+1}','world_point':w.tolist(),'source_pixel':px.tolist(),
                    'projection':pr.tolist(),'role':'fit'} for n,(w,px,pr) in enumerate(zip(roof_points,roof_pixels,roof_projected))],
                'role':'same_video_geometry_fit','independent_check_points':0,
                'claim_boundary':'0.23 jointly estimates the shared roof and both 45/48 s cameras from 43/45 s four roof corners, 45 s two central hubs, and 48 s far corner/edge plus separate greasing hubs/cabinets; camera43 stays fixed. All are same-video fitting targets, not independent visual acceptance.'}
            if scene.get('layout_offsets',{}).get('preparation_fixed_revision')==23:
                from .component_geometry import PREPARATION_PORTAL_SNAPSHOT_X,PREPARATION_FRONT_SNAPSHOT_X
                moving_points=np.array([[x+process_offset,8+lane_offset+side*.77,z]
                    for x,z in [(PREPARATION_PORTAL_SNAPSHOT_X,.67),(PREPARATION_FRONT_SNAPSHOT_X,.56)] for side in [-1,1]])
                moving_pixels=np.array([[356,277],[478,211],[687,478],[815,402]])
                moving_projection=project(parameters,moving_points,focal)
                view['moving_wheel_snapshot_diagnostic']={'source_pixels':moving_pixels.tolist(),
                    'world_points':moving_points.tolist(),'projected_pixels':moving_projection.tolist(),
                    'point_rmse_px':float(np.sqrt(np.mean(np.sum((moving_projection-moving_pixels)**2,axis=1)))),
                    'role':'same_video_static_snapshot_estimate',
                    'claim_boundary':'These two wheels move in the source and do not determine the fixed camera/roof. Current positions use one 45 s snapshot shared by every camera; 43/48 s motion differences remain.'}
        if cid.startswith('closing') and scene.get('layout_offsets',{}).get('closing_fixed_revision')==16:
            checks=[('cabinet_roof_left',[147.95,19.51,2.585],[365,291] if cid=='closing_wide' else [458,261]),
                    ('cabinet_roof_right',[149.37,19.51,2.465],[510,267] if cid=='closing_wide' else [617,245])]
            checks=[(n,[p[0]+closing_offset,p[1]+lane_offset,p[2]],px) for n,p,px in checks]
            projected=project(parameters,np.array([r[1] for r in checks]),focal)
            view['check_count']=len(checks)
            view['check_rmse_px']=float(np.sqrt(np.mean(np.sum((projected-np.array([r[2] for r in checks]))**2,axis=1))))
            rows.extend({'name':n,'world_point':point,'source_pixel':pixel,'role':'check','projection':pr.tolist()}
                for (n,point,pixel),pr in zip(checks,projected))
        elif cid.startswith('closing') and scene.get('layout_offsets',{}).get('closing_side_revision')==12:
            checks=[('beam_start',[148.4,20.7,2.38],[409,235] if cid=='closing_wide' else [500,184]),
                    ('beam_end',[154,20.7,2.38],[829,130] if cid=='closing_wide' else [1056,142])]
            projected=project(parameters,np.array([r[1] for r in checks]),focal)
            view['check_count']=len(checks)
            view['check_rmse_px']=float(np.sqrt(np.mean(np.sum((projected-np.array([r[2] for r in checks]))**2,axis=1))))
            rows.extend({'name':n,'world_point':point,'source_pixel':pixel,'role':'check','projection':pr.tolist()}
                for (n,point,pixel),pr in zip(checks,projected))
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
        plot(report,footer='Fixed rails and tray fitted jointly; cabinet roof checks retain mismatch. Same sequence, no independent visual acceptance.')
        wiping=next(v for v in report['views'] if v['camera']=='wiping');roof=wiping['portal_roof_diagnostic']
        plot({'views':[dict(wiping,diagnostic_id='wiping_portal_roof',landmarks=roof['landmarks'],
            fit_count=4,fit_rmse_px=roof['point_rmse_px'],check_count=0)]},
            footer='Two central hubs and 43/45/48 s roof jointly estimate geometry and two cameras. Moving wheels excluded. No independent acceptance.')
    (OUT/'camera_fit_service.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'camera':v['camera'],'fit_rmse_px':v['fit_rmse_px']} for v in report['views']]))


if __name__=='__main__':main()
