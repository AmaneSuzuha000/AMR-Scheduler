"""Sparse two-view pose checks against one shared estimated layout; no fidelity score."""
import argparse
import hashlib
import json
import math
import numpy as np
from scipy.optimize import least_squares
from .build import ROOT, SCENE, camera_pose
from .fit_camera import basis, project
from .evidence import build_fingerprint
from .assets import INSPECTION_SHELL_WIDTH

OUT=ROOT/'visual_comparison/wheelset_workshop_v1'


def landmarks(scene, camera_id):
    items=[i for i in scene['instances'] if i['zone']=='inspection']
    rows=[]
    def hub(x,y,side,pixel,role):
        item=next(i for i in items if i['asset']=='wheelset' and i['pose'][:2]==[x,y])
        rows.append({'name':f'wheel_{x}_{y}_{side}','instance':item['id'],
            'world_point':[x,y+side*.77,item['pose'][2]+.56],'source_pixel':pixel,'role':role})
    if camera_id=='inspection':
        readings=[(27,56.5,[[329,285],[321,323]]),(27,50,[[289,438],[281,483]]),
                  (42,56.5,[[847,285],[855,323]]),(42,50,[[871,438],[880,483]])]
        for x,y,pixels in readings:
            for side,pixel in zip([1,-1],pixels):hub(x,y,side,pixel,'fit')
        for x,u in [(30,434),(31,469),(32,504)]:hub(x,56.5,1,[u,285],'check')
        for x,u in [(30,412),(31,449),(32,484)]:hub(x,50,1,[u,438],'check')
    else:
        readings=[(32,56.5,[[155,285],[141,323]]),(32,50,[[110,438],[89,483]]),
                  (42,56.5,[[457,285],[450,323]]),(42,50,[[434,438],[427,483]])]
        for x,y,pixels in readings:
            for side,pixel in zip([1,-1],pixels):hub(x,y,side,pixel,'fit')
        for y,pixels in [(56.5,[[208,303],[258,303]]),(50,[[145,478],[205,478]])]:
            item=next(i for i in items if i['asset']=='magnetic_test_hood' and i['pose'][1]==y)
            for sign,pixel in zip([-1,1],pixels):
                rows.append({'name':f'roof_near_{y}_{sign}','instance':item['id'],
                    'world_point':[item['pose'][0]+sign*.805,y-INSPECTION_SHELL_WIDTH/2,2.46],'source_pixel':pixel,'role':'check',
                    'geometry_tuning_point':True})
    return rows


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--apply',action='store_true')
    parser.add_argument('--bind-captures',action='store_true')
    args=parser.parse_args()
    scene=json.loads(SCENE.read_text(encoding='utf-8'))
    report={'method':'Two independent pose fits to a single shared scene; horizontal FOV fixed at 0.92 rad',
            'manual_pixel_reading_tolerance_estimate':8,'scale_measured':False,
            'geometry_adjusted_using_both_frames':True,'roof_points_used_to_adjust_geometry':True,'views':[],
            'claim_boundary':'Manual same-sequence correspondences, including geometry-tuning frames; no independent visual, dimension or physics acceptance.'}
    for cid in ['inspection','inspection_exit']:
        camera=next(c for c in scene['cameras'] if c['id']==cid)
        rows=landmarks(scene,cid)
        points=np.array([r['world_point'] for r in rows]);pixels=np.array([r['source_pixel'] for r in rows])
        fit=np.array([r['role']=='fit' for r in rows])
        pose=camera_pose(camera);initial=np.array(pose[:3]+[pose[5],pose[4]])
        focal=640/math.tan(camera['horizontal_fov']/2)
        if args.apply:
            result=least_squares(lambda p:(project(p,points[fit],focal)-pixels[fit]).ravel(),initial,
                bounds=([15,-40,8,.8,.25],[75,48,65,2.4,1.3]),max_nfev=2000)
            parameters=result.x
            eye,forward,_,_=basis(parameters)
            camera['position']=eye.tolist();camera['target']=(eye+10*forward).tolist()
            camera['registration']='manual_sparse_landmark_fit'
            camera['registration_report']='visual_comparison/wheelset_workshop_v1/camera_fit_inspection.json'
        else:parameters=initial
        after=project(parameters,points,focal)
        for row,pixel in zip(rows,after):row['projection']=pixel.tolist()
        def rmse(mask):return float(np.sqrt(np.mean(np.sum((after[mask]-pixels[mask])**2,axis=1))))
        report['views'].append({'camera':cid,'reference_second':camera['reference_second'],
            'parameters':parameters.tolist(),'fit_rmse_px':rmse(fit),'check_rmse_px':rmse(~fit),
            'landmarks':rows,'fit_count':int(fit.sum()),'check_count':int((~fit).sum())})
    if args.apply:SCENE.write_text(json.dumps(scene,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    if args.bind_captures:
        captures=json.loads((OUT/'gazebo/capture_report.json').read_text(encoding='utf-8'))
        report['build_fingerprint']=build_fingerprint()
        for view in report['views']:
            capture=next(c for c in captures if c['camera']==view['camera'])
            pose=capture['requested_pose']
            assert math.dist(pose[:3]+[pose[5],pose[4]],view['parameters'])<1e-6
            assert capture['build_fingerprint']==report['build_fingerprint']
            image=OUT/'gazebo'/capture['file']
            assert hashlib.sha256(image.read_bytes()).hexdigest()==capture['sha256']
            view['gazebo_image_sha256']=capture['sha256']
            ref=OUT/'references'/f"frame_{view['reference_second']:02d}s.png"
            view['reference_sha256']=hashlib.sha256(ref.read_bytes()).hexdigest()
        plot(report)
    (OUT/'camera_fit_inspection.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps([{'camera':v['camera'],'fit_rmse_px':v['fit_rmse_px'],'check_rmse_px':v['check_rmse_px']} for v in report['views']]))


def plot(report, footer='Geometry adjusted using these same views. Remaining roof/spacing mismatch is retained; no full-scene acceptance.'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image
    for view in report['views']:
        fig,axes=plt.subplots(1,2,figsize=(16,5.5),dpi=150)
        paths=[OUT/'references'/f"frame_{view['reference_second']:02d}s.png",OUT/'gazebo'/f"{view['camera']}.png"]
        for ax,path,title in zip(axes,paths,['Source: manual approximate landmarks','Gazebo: same shared scene, projected landmarks']):
            ax.imshow(Image.open(path));ax.set_xlim(0,1280);ax.set_ylim(720,0);ax.set_title(title);ax.axis('off')
        for i,row in enumerate(view['landmarks'],1):
            color='#ffb938' if row['role']=='fit' else '#35cbbb'
            for ax,key in zip(axes,['source_pixel','projection']):
                x,y=row[key];ax.scatter([x],[y],s=55,facecolors='none',edgecolors=color)
                ax.annotate(str(i),(x,y),xytext=(5,-6),textcoords='offset points',fontsize=8,color=color)
        checks=f"{view['check_rmse_px']:.1f} px RMSE" if view.get('check_count',0) else 'none'
        fit_text=f"{view['fit_rmse_px']:.1f} px RMSE" if view.get('fit_count',0) else 'none'
        fig.text(.02,.075,f"{view.get('fit_label','Fit')}: {view['fit_count']} points, {fit_text}. Same-frame checks: {view['check_count']} points, {checks}.")
        fig.text(.02,.025,footer,fontsize=10)
        fig.subplots_adjust(left=.01,right=.99,top=.94,bottom=.15,wspace=.04)
        fig.savefig(OUT/f"camera_alignment_{view.get('diagnostic_id',view['camera'])}.png",facecolor='white');plt.close(fig)


if __name__=='__main__':main()
