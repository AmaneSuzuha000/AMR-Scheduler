"""Diagnostic plot of manually read image points and corresponding model projections."""
import json
import hashlib
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from .build import ROOT
from .evidence import build_fingerprint

directory=ROOT/'visual_comparison/wheelset_workshop_v1'
report=json.loads((directory/'camera_fit_press_front.json').read_text(encoding='utf-8'))
captures=json.loads((directory/'gazebo/capture_report.json').read_text(encoding='utf-8'))
capture=next(c for c in captures if c['camera']=='press_front')
pose=capture['requested_pose']
assert math.dist(pose[:3]+[pose[5],pose[4]],report['fitted_parameters'])<1e-6, 'Captured camera differs from fitted pose'
assert capture['build_fingerprint']==build_fingerprint(), 'Capture belongs to a different geometry build'
assert hashlib.sha256((directory/'gazebo/press_front.png').read_bytes()).hexdigest()==capture['sha256']
report['build_fingerprint']=capture['build_fingerprint']
report['gazebo_image_sha256']=capture['sha256']
report['reference_frame_sha256']=hashlib.sha256((directory/'references/frame_62s.png').read_bytes()).hexdigest()
(directory/'camera_fit_press_front.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
fig,axes=plt.subplots(1,2,figsize=(16,5.5),dpi=150)
for axis,path,title in zip(axes,[directory/'references/frame_62s.png',directory/'gazebo/press_front.png'],
                           ['Reference video at 62 s: manually read points','Gazebo: projected model landmarks']):
    axis.imshow(Image.open(path));axis.set_xlim(0,1280);axis.set_ylim(720,0);axis.set_title(title,fontsize=12);axis.axis('off')
for i,point in enumerate(report['landmarks'],1):
    color='#ffba38' if point['role']=='fit' else '#42e0c5'
    for axis,key in zip(axes,['source_pixel','after_projection' if 'after_projection' in point else 'current_projection']):
        x,y=point[key]
        axis.scatter([x],[y],s=55,facecolors='none',edgecolors=color,linewidths=1.8)
        axis.annotate(str(i),(x,y),xytext=(7,-11),textcoords='offset points',color='white',fontsize=9,
                      bbox={'boxstyle':'round,pad=.15','facecolor':'#182531','alpha':.8,'edgecolor':'none'})
fig.text(.02,.07,f"Orange: 5 fitting points, RMSE {report['after_fit_rmse_px']:.1f} px.  Teal: 2 extra same-frame checks, RMSE {report['after_check_rmse_px']:.1f} px.",fontsize=11)
fig.text(.02,.025,'Geometry also adjusted using this reference. This checks sparse alignment; it is not an independent full-scene fidelity test.',fontsize=10,color='#555555')
fig.subplots_adjust(left=.01,right=.99,top=.94,bottom=.15,wspace=.04)
fig.savefig(directory/'camera_alignment_press_front.png',facecolor='white')
plt.close(fig)
print('camera_alignment_press_front.png')
