"""Measured full-line replay; physical images always come from the run camera."""
import argparse
from bisect import bisect_right
import hashlib
import json
import math
from pathlib import Path

from .finishing_flow import FULL_STAGES
from .verify_whole_line import verify
from .whole_line_evidence import rows


class Cursor:
    def __init__(self,values,stamp):
        self.values=iter(values); self.stamp=stamp
        self.current=None; self.next=next(self.values,None)

    def at(self,time):
        while self.next is not None and self.stamp(self.next)<=time:
            self.current=self.next;self.next=next(self.values,None)
        return self.current


def replay(directory, *, speed=10.,fps=15):
    import cv2
    import numpy as np
    directory=Path(directory); audit=verify(directory)
    assert math.isfinite(speed) and 0<speed<=30 and 1<=fps<=30
    report=json.loads((directory/'result.json').read_text()); process=report['process_report']
    frames=report['images']; frame_times=[f['sim_time_s'] for f in frames]
    assert all(b>=a for a,b in zip(frame_times,frame_times[1:]))
    sample_cursor=Cursor(rows(directory/'process','samples'),lambda r:r['time_s'])
    fleet_cursor=Cursor(rows(directory,'trace'),lambda r:min(o['sim_time_s'] for o in r['observations'].values()))
    resource_cursor=Cursor(rows(directory/'process','events'),lambda r:r['time_s'])
    supply_cursor=Cursor(rows(directory,'supply_events'),lambda r:r['time_s'])
    start=max(frame_times[0],sample_cursor.next['time_s'])
    end=max(frame_times[-1],max(o['time_s'] for o in process['final_stop'].values()))
    count=math.ceil((end-start)*fps/speed)+1
    path=directory/'whole_line_execution.mp4'
    writer=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,1100));assert writer.isOpened()
    photo_index=-1; photo=None;maximum_age=0.; colours={'ws001':(100,230,245),'ws002':(240,170,100),
        'scout_a':(140,240,140),'scout_b':(210,150,240),'line_supply_tote':(100,190,250)}
    def text(image,value,x,y,scale=.52,colour=(220,225,235)):
        cv2.putText(image,value,(x,y),cv2.FONT_HERSHEY_SIMPLEX,scale,colour,1,cv2.LINE_AA)
    def pixel(xyz): return round(25+xyz[0]*3.35),round(1055-xyz[1]*3.35)
    try:
        for index in range(count):
            stamp=min(start+index*speed/fps,end)
            frame_index=max(0,bisect_right(frame_times,stamp)-1)
            if photo_index!=frame_index:
                photo=cv2.imread(str(directory/frames[frame_index]['file']));assert photo is not None
                photo=cv2.resize(photo,(1280,720));photo_index=frame_index
            image=np.full((1100,1280,3),(25,29,36),dtype=np.uint8);image[:720]=photo
            age=max(0.,stamp-frame_times[frame_index]);maximum_age=max(maximum_age,age)
            cv2.rectangle(image,(0,0),(1280,57),(20,23,30),-1)
            text(image,f"Recorded Gazebo camera | source {frame_times[frame_index]:.3f}s | held {age:.3f}s",15,23,.58)
            text(image,f"Simulation {stamp:.3f}s | playback {speed:g}x | measured data below; no generated physical frames",15,46)
            sample=sample_cursor.at(stamp);fleet=fleet_cursor.at(stamp);resource=resource_cursor.at(stamp);supply=supply_cursor.at(stamp)
            text(image,'Measured XY schematic (equal scale, metres)',20,750,.58)
            for stage in FULL_STAGES:
                p=pixel(stage['target']);cv2.circle(image,p,3,(90,100,115),1)
            if sample:
                for name,obs in sample['observations'].items():
                    if name not in colours:continue
                    p=pixel(obs['xyz']);colour=colours[name];cv2.circle(image,p,6,colour,-1)
                    text(image,name,p[0]+8,p[1]-7,.45,colour)
            if fleet:
                for name,obs in fleet['observations'].items():
                    p=pixel(obs['pose']);cv2.rectangle(image,(p[0]-4,p[1]-4),(p[0]+4,p[1]+4),colours[name],-1)
                    text(image,name,p[0]+8,p[1]-7,.45,colours[name])
            snapshot=resource['snapshot'] if resource else {'jobs':{},'completed':[],'carrier':None}
            for i,job in enumerate(['ws001','ws002']):
                state=snapshot['jobs'].get(job); y=778+i*92
                if state:
                    station='external boundary' if state['state']=='complete' else ('infeed' if state['index']==0 else FULL_STAGES[state['index']-1]['id'])
                    phase=sample['phases'].get(job,'holding') if sample else 'holding'
                    text(image,f"{job}: {station} / {state['state']}",810,y,.58,colours[job])
                    text(image,phase[:53],810,y+24,.45)
                    text(image,f"stage {state['index']}/18",810,y+46,.5)
                else:text(image,job+': waiting for measured admission',810,y,.48,colours[job])
            text(image,f"Measured external releases: {len(snapshot['completed'])}/2",810,978,.53)
            if supply:text(image,'Consumables: '+str(supply['stock']),810,1005,.45)
            if fleet:text(image,'Fleet tasks: '+str(sum(v=='completed' for v in fleet['tasks'].values()))+'/2',810,1032)
            text(image,'Constrained rig / assumed process; contact quality and scene navigation remain separate validations.',20,1084,.50)
            writer.write(image)
    finally:writer.release()
    capture=cv2.VideoCapture(str(path));decoded=0;last=None
    while True:
        ok,frame=capture.read()
        if not ok:break
        assert frame.shape[:2]==(1100,1280);decoded+=1;last=frame
    capture.release();assert decoded==count and last is not None
    cv2.imwrite(str(directory/'whole_line_final_frame.png'),last)
    result=dict(passed=True,file=path.name,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frames=count,fps=fps,playback_speed=speed,source_images=len(frames),simulation_start_s=start,simulation_end_s=end,
        maximum_camera_hold_s=maximum_age,final_camera_hold_s=max(0.,end-frame_times[-1]),
        result_sha256=audit['result_sha256'],process_result_sha256=audit['process_result_sha256'],
        method='Original Gazebo images held at labelled source times. Streaming measured joints, fleet states and resource/material events; equal-scale XY schematic is separate from physical images.')
    (directory/'whole_line_replay.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path)
    p.add_argument('--speed',type=float,default=10.);p.add_argument('--fps',type=int,default=15)
    args=p.parse_args();print(json.dumps(replay(args.directory,speed=args.speed,fps=args.fps)))


if __name__=='__main__':main()
