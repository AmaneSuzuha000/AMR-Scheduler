"""Report sparse alignment for the 55 s back-ring station, with explicit scale assumptions."""
import argparse
import hashlib
import json
import math
import numpy as np
from .build import ROOT, SCENE, camera_pose
from .evidence import build_fingerprint
from .fit_camera import project
from .metrology_assets import BED_LENGTH, BED_WIDTH
from .metrology_assets import (COLUMN_SIZE,COLUMN_PIXELS,COLUMN_CORNER_INDICES,
    COVER_SIZE,COVER_PIXELS,COVER_CORNER_INDICES,ROBOT_JOINTS_WORLD)

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    camera=next(c for c in scene['cameras'] if c['id']=='back_ring')
    bed=next(i for i in scene['instances'] if i['asset']=='back_ring_storage_bed')
    human=next(i for i in scene['instances'] if i['id'].startswith('metrology_r17_') and i['asset']=='workshop_operator')
    x,y=bed['pose'][:2];hx,hy=human['pose'][:2]
    points=np.array([[x+BED_LENGTH/2,y-BED_WIDTH/2,.85],[x+BED_LENGTH/2,y+BED_WIDTH/2,.85],
        [x-BED_LENGTH/2,y+BED_WIDTH/2,.85],[x-BED_LENGTH/2,y-BED_WIDTH/2,.85],[hx,hy,0],[hx,hy,1.7]])
    pixels=np.array([[287,551],[414,672],[962,326],[886,264],[821,273],[854,93]])
    p=camera_pose(camera);parameters=np.array(p[:3]+[p[5],p[4]])
    focal=640/math.tan(camera['horizontal_fov']/2);fitted=project(parameters,points,focal)
    table=next(i for i in scene['instances'] if i['asset']=='back_ring_short_table')
    tx,ty=table['pose'][:2]
    checks=np.array([[tx+.41,ty-.235,.85],[tx+.41,ty+.235,.85],
                     [tx-.41,ty+.235,.85],[tx-.41,ty-.235,.85]])
    check_pixels=np.array([[460,334],[474,363],[591,317],[572,293]])
    checked=project(parameters,checks,focal)
    rows=[{'name':f'bed_or_assumed_human_{n+1}','world_point':w.tolist(),'source_pixel':px.tolist(),
           'projection':pr.tolist(),'role':'fit'} for n,(w,px,pr) in enumerate(zip(points,pixels,fitted))]
    rows.extend({'name':f'same_sequence_table_corner_{n+1}','world_point':w.tolist(),'source_pixel':px.tolist(),
                 'projection':pr.tolist(),'role':'check'} for n,(w,px,pr) in enumerate(zip(checks,check_pixels,checked)))
    view={'camera':'back_ring','reference_second':55,'parameters':parameters.tolist(),
        'fit_count':len(points),'check_count':len(checks),
        'fit_rmse_px':float(np.sqrt(np.mean(np.sum((fitted-pixels)**2,axis=1)))),
        'check_rmse_px':float(np.sqrt(np.mean(np.sum((checked-check_pixels)**2,axis=1)))),
        'landmarks':rows}
    views=[view]
    for asset,size,pixels,indices,diagnostic in [
            ('back_ring_measuring_column',COLUMN_SIZE,COLUMN_PIXELS,COLUMN_CORNER_INDICES,'back_ring_column'),
            ('back_ring_measuring_booth',COVER_SIZE,COVER_PIXELS,COVER_CORNER_INDICES,'back_ring_cover')]:
        item=next(i for i in scene['instances'] if i['asset']==asset);x,y=item['pose'][:2];dx,dy,h=size
        corners=np.array([[x+sx*dx/2,y+sy*dy/2,z] for z in [h,0]
            for sx,sy in [(-1,-1),(1,-1),(1,1),(-1,1)]])[indices]
        projected=project(parameters,corners,focal);pixels=np.array(pixels)
        detail={'camera':'back_ring','diagnostic_id':diagnostic,'reference_second':55,
            'parameters':parameters.tolist(),'fit_count':len(corners),'check_count':0,
            'fit_rmse_px':float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1)))),
            'check_rmse_px':None,'landmarks':[
                {'name':f'{diagnostic}_approx_corner_{index}','world_point':w.tolist(),
                 'source_pixel':px.tolist(),'projection':pr.tolist(),'role':'fit'}
                for index,w,px,pr in zip(indices,corners,pixels,projected)],
            'correspondence_scope':'Visible tower corners' if diagnostic.endswith('column') else
                'Approximate rectangular envelope; rounded cover silhouette corners are not exact mesh vertices'}
        views.append(detail)
    report={'method':'Bed camera and human scale from 0.17. Camera unchanged in0.25; seven white-tower corners and seven approximate black-cover envelope points jointly estimate their positions, footprint and height.',
        'assumed_human_height_m':1.7,'geometry_measured':False,'geometry_adjusted_using_same_sequence':True,
        'bed_dimensions_estimated_m':[BED_LENGTH,BED_WIDTH,.85],
        'pixel_reading_tolerance_estimate':20,'views':views,
        'robot_static_estimate':{'reference_second':55,'assumed_joint_world_heights_m':[p[2] for p in ROBOT_JOINTS_WORLD],
            'manual_approx_joint_centres_px':[[604,351],[632,254],[597,241],[608,207]],
            'method':'XY from source pixel rays at assumed Z. Zero ray reprojection is by construction, not validation; no robot accuracy score.',
            'motion_validated':False},
        'claim_boundary':'One bed corner partly hidden by caption. Human scale, ring count, rail, cabinet dimensions and background are estimates. Black-cover envelope points approximate a rounded shape. Same-sequence fits/checks are not independent; no full-image, process, motion or dimensional acceptance.'}
    if args.bind_captures:
        captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8'))
        capture=next(c for c in captures if c['camera']=='back_ring');pose=capture['requested_pose']
        report['build_fingerprint']=build_fingerprint()
        assert capture['build_fingerprint']==report['build_fingerprint']
        assert math.dist(pose[:3]+[pose[5],pose[4]],parameters)<1e-6
        for v in views:
            v['gazebo_image_sha256']=hashlib.sha256((OUT/'gazebo'/capture['file']).read_bytes()).hexdigest()
            assert v['gazebo_image_sha256']==capture['sha256']
            v['reference_sha256']=hashlib.sha256((OUT/'references/frame_55s.png').read_bytes()).hexdigest()
        from .fit_inspection_cameras import plot
        plot(report,footer='Fixed camera; same-video geometry estimates. Cover uses approximate rectangular envelope; arm Z assumed. No independent acceptance.')
    (OUT/'camera_fit_metrology.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{k:v[k] for k in ['camera','fit_rmse_px','check_rmse_px']} for v in views]))


if __name__=='__main__':main()
