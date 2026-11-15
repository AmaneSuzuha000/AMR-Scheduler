"""Source 48/50/65/66 s diagnostics, retaining unknown scale and snapshot limits."""
import argparse
import hashlib
import json
import math
import numpy as np
from .build import ROOT, SCENE, camera_pose
from .evidence import build_fingerprint
from .fit_camera import project
from .transfer_geometry import CACHE_READINGS, CACHE_FIRST_X, CACHE_FIRST_Y, CACHE_PITCH, CACHE_ROW_PITCH, ROOM_Y_OFFSET

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'


def landmarks(cid,scene=None):
    if cid in CACHE_READINGS:
        rows=CACHE_READINGS[cid]
        points=[[CACHE_FIRST_X+n*CACHE_PITCH,CACHE_FIRST_Y+row*CACHE_ROW_PITCH+side*.77,.56]
            for n,row,side,pixel in rows]
        pixels=[r[3] for r in rows]
    elif cid=='greasing':
        from .component_geometry import greasing_landmarks,GREASE_HUB_PIXELS,GREASE_ROOF_PIXELS
        points=greasing_landmarks(scene)
        pixels=GREASE_HUB_PIXELS+GREASE_ROOF_PIXELS
    else:
        points=[[74.3,35,1.5],[74.3,43,1.5],[82.3,43,1.5],[74.3,43,0]]
        points=[[x,y+ROOM_Y_OFFSET,z] for x,y,z in points]
        if scene and scene.get('layout_offsets',{}).get('transfer_context_revision')==26:
            from .transfer_details import room_point
            points=[room_point(point) for point in points]
        pixels=[[307,677],[702,199],[1260,463],[714,325]]
    return np.array(points),np.array(pixels)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--bind-captures',action='store_true');args=parser.parse_args()
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    report={'method':'Fixed FOV .92 rad. 65/66 s wheel hubs jointly estimate two cameras and short-cache pitch/row spacing. 0.23 corrects 48 s to a separate greasing machine; hubs and cabinet roofs jointly constrain its camera with 43/45/48 s shared portal roof points. Wheel silhouettes are residual diagnostics, not targets of this joint roof fit. 50 s sparse enclosure corners use assumed 1.5 m walls.',
        'geometry_measured':False,'geometry_adjusted_using_same_sequence':True,
        'independent_check_points':0,'pixel_reading_tolerance_estimate':20,'views':[],
        'claim_boundary':'43 s long preparation cache and 65/66 s short press-exit cache are separate. Wheel diameter remains an assumed scale anchor. Counts are static snapshot estimates. 50 s cell process is unknown; wall size, global positions and press-exit exteriors are estimated. Dedicated views do not establish whole-cell visual acceptance, dynamics or scheduling.'}
    captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8')) if args.bind_captures else []
    for cid in ['greasing','cross_transfer','cache_transfer','cache_reverse']:
        camera=next(c for c in scene['cameras'] if c['id']==cid);p=camera_pose(camera)
        parameters=np.array(p[:3]+[p[5],p[4]]);points,pixels=landmarks(cid,scene)
        projected=project(parameters,points,640/math.tan(camera['horizontal_fov']/2))
        view={'camera':cid,'reference_second':camera['reference_second'],'parameters':parameters.tolist(),
            'fit_rmse_px':float(np.sqrt(np.mean(np.sum((projected-pixels)**2,axis=1)))),
            'check_rmse_px':None,'fit_count':len(points),'check_count':0,
            'landmarks':[{'name':f'source_point_{n+1}','world_point':w.tolist(),'source_pixel':px.tolist(),
                'projection':pr.tolist(),'role':'fit'} for n,(w,px,pr) in enumerate(zip(points,pixels,projected))]}
        if cid in {'cache_transfer','cache_reverse'}:
            from .transfer_details import WEST_CROSS_PIXELS,FIN_BANK_PIXELS
            road=next(r for r in scene['agv_routes'] if r['id']=='transfer_r20_cache_cross_141.8')
            context_points=[[road['start'][0],CACHE_FIRST_Y+row*CACHE_ROW_PITCH,.014]
                for row in (range(4) if cid=='cache_transfer' else [3,2,1])]
            context_pixels=WEST_CROSS_PIXELS[cid]
            if cid=='cache_transfer':
                banks=sorted((i for i in scene['instances'] if i['asset']=='press_exit_fin_bank'),key=lambda i:-i['pose'][1])
                context_points.extend([i['pose'][:2]+[.45] for i in banks])
                context_pixels=context_pixels+FIN_BANK_PIXELS if banks else context_pixels
            pp=project(parameters,np.array(context_points),640/math.tan(camera['horizontal_fov']/2))
            view['context_diagnostic']={'points':context_points,'source_pixels':context_pixels,'projections':pp.tolist(),
                'point_rmse_px':float(np.sqrt(np.mean(np.sum((pp-np.array(context_pixels))**2,axis=1)))),
                'role':'same_source_estimated_ground_intersections_and_fin_centres','independent_acceptance':False}
        if cid=='greasing':
            from .assets import wheelset
            from .component_geometry import GREASE_RIM_BOUNDS
            machine=max((i for i in scene['instances'] if i['asset']=='greasing_station'),key=lambda i:i['pose'][1])
            wheel=min((i for i in scene['instances'] if i['asset']=='wheelset' and i['zone']=='preparation'),
                key=lambda i:math.dist(i['pose'][:2],machine['pose'][:2]))
            mesh=wheelset();bounds=[]
            for side in [-1,1]:
                indices={i for face,part in zip(mesh.faces,mesh.parts)
                    if part in {f'wheel_{side}',f'tread_{side}'} for i in face}
                vertices=np.array([mesh.vertices[i] for i in sorted(indices)])
                yaw=wheel['pose'][5];c,s=math.cos(yaw),math.sin(yaw)
                vertices=vertices@np.array([[c,s,0],[-s,c,0],[0,0,1]])+wheel['pose'][:3]
                pp=project(parameters,vertices,640/math.tan(camera['horizontal_fov']/2))
                bounds.append([*pp.min(axis=0),*pp.max(axis=0)])
            view['silhouette_bounds']={'source':GREASE_RIM_BOUNDS,'projected':np.array(bounds).tolist(),
                'coordinate_rmse_px':float(np.sqrt(np.mean((np.array(bounds)-GREASE_RIM_BOUNDS)**2))),
                'role':'same_sequence_residual_diagnostic','world_wheel_id':wheel['id'],
                'note':'Extrema did not participate in the 0.23 joint roof fit. Same source used to develop this geometry; not independent acceptance.'}
            from .component_geometry import preparation_portal_roof
            roof_points=np.array(preparation_portal_roof(scene));roof_projected=project(parameters,roof_points,640/math.tan(camera['horizontal_fov']/2))
            source_corner=np.array([1240,284]);source_edge=np.array([719,571])
            edge=roof_projected[0]-roof_projected[3]
            distance=abs(float(edge[0]*(source_edge-roof_projected[3])[1]-edge[1]*(source_edge-roof_projected[3])[0]))/float(np.linalg.norm(edge))
            view['fixed_portal_diagnostic']={'source_far_corner':source_corner.tolist(),
                'projected_far_corner':roof_projected[3].tolist(),
                'corner_error_px':float(np.linalg.norm(roof_projected[3]-source_corner)),
                'source_back_edge_pixel':source_edge.tolist(),'projected_back_edge':roof_projected[[3,0]].tolist(),
                'point_to_projected_edge_distance_px':distance,'role':'same_sequence_joint_geometry_fit',
                'independent_check_points':0,
                'claim_boundary':'43/45/48 s roof readings jointly tune the shared roof and both 45/48 s cameras; camera43 remains fixed. 48 s observes a separate low greasing fixture. Near corner is caption-occluded; no invented landmark. Foreground exit wheel has a separate estimated static pose; contact/lift and source motion remain unresolved. Same-video fit, not independent whole-image acceptance.'}
        if args.bind_captures:
            capture=next(c for c in captures if c['camera']==cid);pose=capture['requested_pose']
            report['build_fingerprint']=build_fingerprint()
            assert capture['build_fingerprint']==report['build_fingerprint']
            assert math.dist(pose[:3]+[pose[5],pose[4]],parameters)<1e-6
            view['gazebo_image_sha256']=hashlib.sha256((OUT/'gazebo'/capture['file']).read_bytes()).hexdigest()
            assert view['gazebo_image_sha256']==capture['sha256']
            view['reference_sha256']=hashlib.sha256((OUT/'references'/f"frame_{camera['reference_second']:02d}s.png").read_bytes()).hexdigest()
        report['views'].append(view)
    if args.bind_captures:
        from .fit_inspection_cameras import plot
        plot(report,footer='Separate greasing machine confirmed in 43 s; shared portal fit uses 43/45/48 s. No independent acceptance.')
        plot_portal_mismatch(next(v for v in report['views'] if v['camera']=='greasing'))
    (OUT/'camera_fit_transfer_detail.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'camera':v['camera'],'fit_rmse_px':v['fit_rmse_px']} for v in report['views']]))


def plot_portal_mismatch(view):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    data=view['fixed_portal_diagnostic'];fig,axes=plt.subplots(1,2,figsize=(16,5.5),dpi=150)
    for ax,path,title in zip(axes,[OUT/'references/frame_48s.png',OUT/'gazebo/greasing.png'],
            ['Source: visible far corner / back edge','Gazebo: shared fixed portal, joint geometry estimate']):
        ax.imshow(Image.open(path));ax.set_xlim(0,1280);ax.set_ylim(720,0);ax.axis('off');ax.set_title(title)
    for ax,points in zip(axes,[[data['source_far_corner'],data['source_back_edge_pixel']],data['projected_back_edge']]):
        points=np.array(points);ax.plot(points[:,0],points[:,1],color='#ffb938',linewidth=2,marker='o',markersize=5)
    axes[1].annotate('same fixed far corner',xy=data['projected_far_corner'],xytext=(700,90),color='#ffb938',
        arrowprops={'arrowstyle':'->','color':'#ffb938'})
    fig.text(.02,.075,f"Far corner error: {data['corner_error_px']:.1f} px. Source edge pixel to projected edge: {data['point_to_projected_edge_distance_px']:.1f} px.")
    fig.text(.02,.025,'43/45/48 s jointly tune the shared roof and two cameras. Separate machines; near corner caption-occluded. No independent acceptance.',fontsize=10)
    fig.subplots_adjust(left=.01,right=.99,top=.94,bottom=.15,wspace=.04)
    fig.savefig(OUT/'camera_alignment_greasing_portal_mismatch.png',facecolor='white');plt.close(fig)


if __name__=='__main__':main()
