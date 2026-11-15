"""Audit saved Gazebo truth trajectories against road ownership and arrivals."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import xml.etree.ElementTree as ET

from .light_material_network import point_segment


def verify(directory, *, write=True):
    directory=Path(directory)
    report=json.loads((directory/'result.json').read_text())
    assert report['passed'] and report['runtime_error'] is None
    assert report['baseline_session_preserved'] and report['owned_server_stopped']
    for name,digest in report.get('data_sha256',{}).items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,name
    assert hashlib.sha256((directory/'trace.jsonl').read_bytes()).hexdigest()==report['trace_sha256']
    def rows(name):return [json.loads(line) for line in (directory/(name+'.jsonl')).read_text().splitlines()]
    trace=rows('trace');controls=rows('cmd_vel')
    assert len(trace)==report['observations']>100 and len(report['images'])>3
    for frame in report['images']:
        assert hashlib.sha256((directory/frame['file']).read_bytes()).hexdigest()==frame['sha256']
    layout=report['network'];assert report['clearance']['passed']
    robots={r['id']:r for r in layout['robots']};nodes={n['id']:(n['x'],n['y']) for n in layout['nodes']}
    assert set(robots)=={'scout_a','scout_b'}
    edges={tuple(sorted((lane['src'],lane['dst']))) for lane in layout['lanes']}
    routes={c['route_id']:c for c in report['routes']};assert len(routes)==len(report['routes'])
    assert len(routes)>=6 and all(c['source']=='gazebo_truth' and c['frame_id']=='wheelset_world' for c in routes.values())
    assert len({c['run_id'] for c in routes.values()})==1
    recovery=report.get('recovery');replacement={}
    if recovery:
        replacement[recovery['new_command']['route_id']]=recovery['old_command']['route_id']
    previous={};current_nodes={r:robots[r]['start'] for r in robots}
    completed={};seen=set();maximum_error=0.;minimum_separation=math.inf;distance_travelled={r:0. for r in robots}
    waits={r:0 for r in robots};late_samples=0;began={};max_state_time_error=0.;opposing_wait_samples=0
    for index,row in enumerate(trace):
        if index:assert row['time_s']>=trace[index-1]['time_s']
        pins=row['pins']
        assert all(len(owners)==1 and owners[0] in robots for group in pins.values() for owners in group.values())
        observed=row['observations']
        active=row['commands']['scout_a']
        if active and active['nodes']==['west','east'] and row['commands']['scout_b'] is None:
            assert pins['nodes'].get('bay_east')==['scout_b']
            assert observed['scout_b']['linear_speed']<.025
            opposing_wait_samples+=1
        skew=abs(observed['scout_a']['sim_time_s']-observed['scout_b']['sim_time_s'])
        max_state_time_error=max(max_state_time_error,skew)
        assert skew<.15
        separation=math.dist(observed['scout_a']['pose'][:2],observed['scout_b']['pose'][:2])
        conservative_separation=separation-skew*.65
        minimum_separation=min(minimum_separation,conservative_separation)
        assert conservative_separation>=2*(layout['robot_enclosing_radius_m']+layout['safety_margin_m'])
        for robot in robots:
            obs=observed[robot];xy=obs['pose'][:2];command=row['commands'][robot]
            assert all(math.isfinite(v) for v in [*obs['pose'],obs['z'],obs['linear_speed'],obs['angular_speed']])
            assert abs(obs['z'])<.04 and obs['linear_speed']<.65 and obs['angular_speed']<1.
            old=previous.get(robot);old_command=old['command'] if old else None
            old_id=old_command['route_id'] if old_command else None;token=command['route_id'] if command else None
            if old:
                dt=obs['sim_time_s']-old['observation']['sim_time_s'];assert dt>=0
                displacement=math.dist(xy,old['observation']['pose'][:2])
                assert displacement<=.65*dt+.015
                distance_travelled[robot]+=displacement
            if old_id!=token and old_id:
                if replacement.get(token)!=old_id:
                    assert math.dist(xy,old_command['target_xy'])<=.10
                    assert obs['linear_speed']<.025 and obs['angular_speed']<.05
                    assert old_id not in completed
                    completed[old_id]=dict(robot=robot,elapsed_sim_s=row['time_s']-began[old_id],
                        arrival_pose=obs,arrival_time_s=row['time_s'])
                    current_nodes[robot]=old_command['nodes'][1]
                else:assert command['nodes']==old_command['nodes']
            if command:
                assert command==routes[token] and command['robot_id']==robot
                a,b=command['nodes'];assert tuple(sorted((a,b))) in edges and command['target_xy']==list(nodes[b])
                assert all(pins['nodes'].get(n)==[robot] for n in [a,b])
                assert all(pins['edges'].get(f'{s}->{d}')==[robot] for s,d in [(a,b),(b,a)])
                error=point_segment(xy,nodes[a],nodes[b]);maximum_error=max(maximum_error,error)
                assert error<=.15
                if token!=old_id:
                    assert token not in seen;seen.add(token);began[token]=row['time_s']
                    if token not in replacement:
                        assert a==current_nodes[robot] and math.dist(xy,nodes[a])<=.15
                planned_time=math.dist(nodes[a],nodes[b])/command['speed']
                late_samples+=row['time_s']-began[token]>planned_time
            else:
                assert pins['nodes'].get(current_nodes[robot])==[robot]
                assert math.dist(xy,nodes[current_nodes[robot]])<=.15
                if row['tasks'].get('inspection_consumables' if robot=='scout_a' else 'small_tools')!='completed':waits[robot]+=1
            previous[robot]=dict(command=command,observation=obs)
    assert seen==set(routes) and set(completed)==set(routes)-set(replacement.values())
    assert all(value>10 for value in distance_travelled.values()) and late_samples>0 and opposing_wait_samples>10
    assert all(value=='completed' for value in trace[-1]['tasks'].values())
    assert set(report['completed_tasks'])=={'inspection_consumables','small_tools'}
    for robot,target in [('scout_a','receiving'),('scout_b','supply')]:
        stop=report['final_stop'][robot]
        assert math.dist(stop['pose'][:2],nodes[target])<=.10
        assert stop['linear_speed']<.025 and stop['angular_speed']<.05
        model=ET.parse(directory/(robot+'.urdf')).getroot()
        assert model.find("link[@name='base_link']/collision") is not None
        assert len(model.findall(".//plugin[@filename='libgazebo_ros_diff_drive.so']"))==1
        assert model.find(".//plugin[@filename='libgazebo_ros_planar_move.so']") is None
    assert controls and all(abs(c['v'])<=.4 and abs(c['w'])<=.6 for c in controls)
    feedback_checked=0
    if (directory/'feedback.jsonl').exists():
        feedback=rows('feedback');arrivals={};seqs={};timestamps={}
        for event in feedback:
            message=event['message'];answer=event['answer'];robot=message['robot_id'];token=message['route_id']
            obs=event['measured_state'];assert message['pose']==obs['pose'] and message['timestamp']==obs['sim_time_s']
            if event['probe_expected_rejection']:
                assert not answer['accepted'] and answer['reason']=='identity mismatch';continue
            if not answer['accepted']:
                assert answer['reason']=='duplicate source timestamp';continue
            assert message['seq']>seqs.get(robot,-1) and message['timestamp']>=timestamps.get(robot,-math.inf)
            seqs[robot]=message['seq'];timestamps[robot]=message['timestamp'];feedback_checked+=1
            if token:
                assert token in routes
                command=routes[token]
                for key in ['scene_id','run_id','robot_id','frame_id','time_base','source','task_id','route_id']:
                    assert message[key]==command[key]
            if message['event']=='ARRIVED':
                assert token in completed and token not in arrivals
                assert math.dist(obs['pose'][:2],routes[token]['target_xy'])<=.10
                assert obs['linear_speed']<.025 and obs['angular_speed']<.05
                arrivals[token]=event
        assert set(arrivals)==set(completed)
    if report.get('case')=='feedback_recovery':
        assert recovery and recovery['resumed'] and recovery['old_arrival_rejected']
        assert recovery['fault']=='feedback timeout'
        assert recovery['fault_at']-recovery['injected_at']>=1.8
        assert recovery['resumed_at']-recovery['injected_at']>=3.2
        assert recovery['old_command']['route_id']!=recovery['new_command']['route_id']
        robot=recovery['robot'];a,b=recovery['old_command']['nodes']
        for key in ['retained_pins','fault_pins','resumed_pins']:
            pins=recovery[key]
            assert all(pins['nodes'].get(n)==[robot] for n in [a,b])
            assert all(pins['edges'].get(f'{s}->{d}')==[robot] for s,d in [(a,b),(b,a)])
        stop=recovery['stop_evidence'];assert stop['linear_speed']<.025 and stop['angular_speed']<.05
        assert point_segment(stop['pose'][:2],nodes[a],nodes[b])<=.15
        assert math.dist(stop['pose'][:2],nodes[a])>1 and math.dist(stop['pose'][:2],nodes[b])>1
        assert all(c['v']==0 and c['w']==0 for c in controls if c['robot']==robot
            and recovery['injected_at']<=c['wall_s']<=recovery['resumed_at'])
        assert any(e['command']['kind']=='CANCEL' and e['command']['route_id']==recovery['old_command']['route_id']
            for e in rows('dispatches'))
    result=dict(passed=True,case=report.get('case','normal'),robots_checked=2,tasks_completed=2,
        routes_completed=len(completed),trajectory_samples_checked=len(trace),feedback_events_checked=feedback_checked,
        minimum_conservative_robot_separation_m=minimum_separation,maximum_route_center_error_m=maximum_error,
        maximum_pair_state_time_error_s=max_state_time_error,travelled_m=distance_travelled,
        waiting_observations=waits,opposing_access_wait_samples=opposing_wait_samples,
        late_motion_samples_with_occupancy=late_samples,
        feedback_recovery_checked=bool(recovery),claim_boundary=report['claim_boundary'])
    if report.get('fleet_feedback_method')=='native_world_truth':
        from .verify_fleet_truth import verify as verify_truth
        result.update(verify_truth(directory,trace,report))
    if write:
        (directory/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def replay(directory):
    import cv2
    directory=Path(directory);report=json.loads((directory/'result.json').read_text())
    trace=[json.loads(line) for line in (directory/'trace.jsonl').read_text().splitlines()]
    frames=report['images'];fps=10;index=0;observation=0;canvas=None
    layout=report['network'];nodes={n['id']:(n['x'],n['y']) for n in layout['nodes']}
    x0=min(p[0] for p in nodes.values());x1=max(p[0] for p in nodes.values());y0=min(p[1] for p in nodes.values())
    scale=1140/(x1-x0);colors={'scout_a':(70,210,250),'scout_b':(240,165,90)}
    def pixel(point):return (round(70+(point[0]-x0)*scale),round(250-(point[1]-y0)*scale))
    start,end=frames[0]['time_s'],frames[-1]['time_s'];count=math.floor((end-start)*fps)+1
    writer=cv2.VideoWriter(str(directory/'execution.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(1280,720));assert writer.isOpened()
    for tick in range(count):
        stamp=start+tick/fps
        while index+1<len(frames) and frames[index+1]['time_s']<=stamp:index+=1;canvas=None
        while observation+1<len(trace) and trace[observation+1]['time_s']<=stamp:observation+=1
        if canvas is None:canvas=cv2.imread(str(directory/frames[index]['file']));assert canvas is not None
        image=canvas.copy();row=trace[observation]
        cv2.rectangle(image,(0,0),(1280,320),(15,24,30),-1)
        texts=[f"Two wheel-driven SCOUT surrogates | {report.get('case','normal')} | sim +{stamp:.1f}s",
            ' | '.join(f"{r}: {('->'.join(c['nodes'])) if c else 'waiting/stopped'}" for r,c in row['commands'].items()),
            'Gazebo truth feedback; independent light-material corridor. Downstream device handoff excluded.']
        for y,text in zip([24,52,78],texts):cv2.putText(image,text,(14,y),cv2.FONT_HERSHEY_SIMPLEX,.55,(225,235,240),1,cv2.LINE_AA)
        cv2.putText(image,'Measured X/Y and retained reservations (top view); recorded Gazebo camera below',
            (14,115),cv2.FONT_HERSHEY_SIMPLEX,.55,(180,195,205),1,cv2.LINE_AA)
        for lane in layout['lanes']:
            a,b=lane['src'],lane['dst'];owners=row['pins']['edges'].get(a+'->'+b,[])
            color=colors[owners[0]] if owners else (65,75,85)
            cv2.line(image,pixel(nodes[a]),pixel(nodes[b]),color,5,cv2.LINE_AA)
        for name,xy in nodes.items():
            p=pixel(xy);cv2.circle(image,p,4,(205,215,220),-1)
            cv2.putText(image,name,(p[0]-30,p[1]+24),cv2.FONT_HERSHEY_SIMPLEX,.43,(190,200,210),1,cv2.LINE_AA)
        for robot,obs in row['observations'].items():
            p=pixel(obs['pose']);color=colors[robot]
            if row.get('gates',{}).get(robot,{}).get('fault'):color=(90,90,245)
            cv2.circle(image,p,max(8,round(.55*scale)),color,2,cv2.LINE_AA)
            arrow_end=(round(p[0]+16*math.cos(obs['pose'][2])),round(p[1]-16*math.sin(obs['pose'][2])))
            cv2.arrowedLine(image,p,arrow_end,color,2,cv2.LINE_AA)
            cv2.putText(image,f"{robot} {obs['linear_speed']:.2f}m/s",(p[0]-38,p[1]-19),
                cv2.FONT_HERSHEY_SIMPLEX,.43,color,1,cv2.LINE_AA)
        finished=sum(v=='completed' for v in row['tasks'].values())
        cv2.putText(image,f'Completed deliveries: {finished}/2 | occupied links shown in robot colours',
            (14,306),cv2.FONT_HERSHEY_SIMPLEX,.5,(205,220,230),1,cv2.LINE_AA)
        writer.write(image)
    writer.release();capture=cv2.VideoCapture(str(directory/'execution.mp4'))
    ok,_=capture.read();actual=int(capture.get(cv2.CAP_PROP_FRAME_COUNT));capture.release();assert ok and actual==count
    result=dict(file='execution.mp4',sha256=hashlib.sha256((directory/'execution.mp4').read_bytes()).hexdigest(),
        frames=count,fps=fps,source_frames=len(frames),simulation_start_s=start,simulation_end_s=end,
        method='Recorded Gazebo camera frames resampled by simulation reception time; top-view overlay uses measured Gazebo truth and recorded reservations, not generated motion.')
    (directory/'replay.json').write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);parser.add_argument('--replay',action='store_true')
    args=parser.parse_args();print(json.dumps(verify(args.directory)))
    if args.replay:print(json.dumps(replay(args.directory)))


if __name__=='__main__':main()
