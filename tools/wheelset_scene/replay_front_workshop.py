"""Replay recorded front-line images and measured states through final stop."""
import argparse
import hashlib
import json
import math
from pathlib import Path

from .front_flow import STAGES,parts
from .assembly_flow import ASSEMBLY_STAGES,components
from .finishing_flow import FULL_STAGES,STORED_JOB
from .verify_front_workshop import verify


def replay(directory):
    import cv2
    import numpy as np
    directory=Path(directory);report=json.loads((directory/'result.json').read_text())
    finishing=report.get('through')=='storage';assembly=report.get('through') in {'assembly','storage'}
    recipe=FULL_STAGES if finishing else ASSEMBLY_STAGES if assembly else STAGES
    def rows(name):return [json.loads(line) for line in (directory/(name+'.jsonl')).read_text().splitlines()]
    samples=rows('samples');events=rows('events');frames=report['images'];items=parts()
    stock=rows('stock_events') if assembly else [];stock_index=-1
    start=max(frames[0]['time_s'],samples[0]['time_s'])
    end=max(frames[-1]['time_s'],samples[-1]['time_s'],events[-1]['time_s'])
    fps=10;count=math.ceil((end-start)*fps)+1;fi=si=0;ei=-1;photo=None
    target=directory/'front_execution.mp4'
    writer=cv2.VideoWriter(str(target),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,1040));assert writer.isOpened()
    initial=dict(station='infeed',completed=[],bins={'cap_bin':[],'bearing_bin':[]})
    colour={'ws001':(80,230,245),**{name:((75,160,255) if item['kind']=='cap' else (230,170,90)) for name,item in items.items()}}
    if assembly:colour.update({n:(125,230,130) for n in components()});colour.update(press_ram_left=(100,80,240),press_ram_right=(100,80,240))
    if finishing:colour.update({STORED_JOB:(150,180,240),'spray_tool':(240,180,120)})
    def pixel(xyz):return (round(40+(xyz[0]-8)*(2.75 if finishing else 4.1 if assembly else 12)),round(290-(xyz[1]-3)*2.5))
    fault=False;recoveries=0
    for tick in range(count):
        stamp=min(start+tick/fps,end)
        while fi+1<len(frames) and frames[fi+1]['time_s']<=stamp:fi+=1;photo=None
        while si+1<len(samples) and samples[si+1]['time_s']<=stamp:si+=1
        while stock_index+1<len(stock) and stock[stock_index+1]['time_s']<=stamp:stock_index+=1
        while ei+1<len(events) and events[ei+1]['time_s']<=stamp:
            ei+=1
            if events[ei]['event']=='FAULT_LATCHED':fault=True
            if events[ei]['event']=='RECOVERY_REISSUED':fault=False;recoveries+=1
        row=samples[si];snap=events[ei]['snapshot'] if ei>=0 else initial;main=row['observations']['ws001']
        if photo is None:photo=cv2.imread(str(directory/frames[fi]['file']));assert photo is not None
        canvas=np.zeros((1040,1280,3),dtype=np.uint8);canvas[:320]=(26,23,20);canvas[320:]=photo
        lines=[f"Front to {'storage' if finishing else 'assembly' if assembly else 'inspection'} | {report['case']} | Gazebo sim {stamp:.2f}s | operations {len(snap['completed'])}/{len(recipe)}",
            f"Station {snap['station']} | {row['phase']} | fault {'latched' if fault else 'clear'} | recoveries {recoveries}",
            f"ws001 measured XYZ {main['xyz'][0]:.3f}, {main['xyz'][1]:.3f}, {main['xyz'][2]:.3f} m | speed {max(abs(v) for v in main['velocity']):.3f} m/s",
            f"Recorded image {frames[fi]['time_s']:.3f}s, held {max(0,stamp-frames[fi]['time_s']):.2f}s | joint sample {row['time_s']:.3f}s",
            'Measured XY schematic (unequal axis scales); independent assumed XYZ carriers; gravity/contact excluded.']
        for y,line in zip([24,49,74,99,124],lines):cv2.putText(canvas,line,(14,y),cv2.FONT_HERSHEY_SIMPLEX,.52,(230,235,240),1,cv2.LINE_AA)
        for stage in recipe:
            p=pixel(stage['target']);c=(85,185,95) if stage['id'] in snap['completed'] else (125,120,110)
            cv2.rectangle(canvas,(p[0]-5,p[1]-5),(p[0]+5,p[1]+5),c,1)
            width=cv2.getTextSize(stage['id'],cv2.FONT_HERSHEY_SIMPLEX,.32,1)[0][0]
            label_y=p[1]-24 if stage['id']=='identification' else p[1]-12
            cv2.putText(canvas,stage['id'],(p[0]-width//2,label_y),cv2.FONT_HERSHEY_SIMPLEX,.32,c,1,cv2.LINE_AA)
        for name,obs in row['observations'].items():
            p=pixel(obs['xyz']);cv2.circle(canvas,p,5 if name=='ws001' else 3,colour[name],-1,cv2.LINE_AA)
        for i,(name,item) in enumerate(items.items()):
            obs=row['observations'][name];state=row['part_state'][name]
            cv2.putText(canvas,f"{name}: {state} | z={obs['xyz'][2]:.2f}m",(670,166+i*28),cv2.FONT_HERSHEY_SIMPLEX,.46,colour[name],1,cv2.LINE_AA)
        cv2.putText(canvas,f"Finite bins: caps {len(snap['bins']['cap_bin'])}/2 | bearings {len(snap['bins']['bearing_bin'])}/2",
            (670,290),cv2.FONT_HERSHEY_SIMPLEX,.48,(230,235,240),1,cv2.LINE_AA)
        if assembly:
            states=stock[stock_index]['snapshot']['component_state'] if stock_index>=0 else {n:'unreceived' for n in components()}
            text='New parts: '+', '.join(f'{state} {list(states.values()).count(state)}' for state in sorted(set(states.values())))
            cv2.putText(canvas,text,(670,143),cv2.FONT_HERSHEY_SIMPLEX,.40,(125,230,130),1,cv2.LINE_AA)
        writer.write(canvas)
    writer.release();capture=cv2.VideoCapture(str(target));ok,_=capture.read()
    actual=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release();assert ok and actual==count
    assert len(events[ei]['snapshot']['completed'])==len(recipe)
    result=dict(file=target.name,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),frames=count,fps=fps,
        simulation_start_s=start,simulation_end_s=end,source_frames=len(frames),camera_end_s=frames[-1]['time_s'],
        final_image_hold_s=end-frames[-1]['time_s'],final_operations=len(recipe),
        method='Recorded Gazebo images, raw measured joints and ordered event/bin snapshots. Camera recording time and held age labelled; XY diagram has unequal scales. No generated physical frames.')
    (directory/'front_replay.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);args=p.parse_args()
    print(json.dumps(verify(args.directory)))
    print(json.dumps(replay(args.directory)))


if __name__=='__main__':main()
