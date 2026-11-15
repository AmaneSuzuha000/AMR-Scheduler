"""Exercise multi-wheelset resource flow with an explicitly synthetic executor."""
import argparse
import hashlib
import json
from pathlib import Path

from .resource_runtime import ResourceRuntime


def simulate(case,output,job_count=8):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    runtime=ResourceRuntime(run_id=case,timeout_s=30)
    durations={'derust':8.,'magnetic':18.,'ultrasonic':11.}
    executor={};admitted=0;injected=None;recovered=None;late_rejected=False;output_blocked=False
    faults=[];adapter_events=[]
    for step in range(2401):
        now=step*.25
        if admitted<job_count and runtime.admit(f'ws{admitted+1:03d}',now):admitted+=1
        for token,work in list(executor.items()):
            if work['due']>now:continue
            command=runtime.commands[token]
            inject=(injected is None and command['job']=='ws002' and
                (case=='device_recovery' and command['kind']=='device' and command['target']=='magnetic'
                 or case=='transfer_recovery' and command['kind']=='transfer' and command['target']=='magnetic'))
            if inject:
                runtime.observe(token,now=now,observed_at=now,success=False)
                injected=dict(token=token,time_s=now);faults.append(dict(injected))
            else:
                target=next(p.target for p in runtime.places if p.id==command['target'])
                accepted=runtime.observe(token,now=now,observed_at=now,pose=target,device_complete=True)
                adapter_events.append(dict(time_s=now,command_id=token,event='synthetic_completion',accepted=accepted))
            del executor[token]
        if injected and recovered is None:
            token=injected['token'];elapsed=now-injected['time_s']
            if elapsed>=5 and not runtime.commands[token]['stop_confirmed']:
                runtime.confirm_stopped(token,now=now,observed_at=now,stopped=True)
                adapter_events.append(dict(time_s=now,command_id=token,event='synthetic_stop_feedback'))
            if elapsed>=10:
                recovered=runtime.recover(token,now)
                assert recovered is not None
                target=next(p.target for p in runtime.places if p.id==runtime.commands[token]['target'])
                late_rejected=not runtime.observe(token,now=now,observed_at=now,pose=target,device_complete=True)
                adapter_events.append(dict(time_s=now,command_id=token,event='superseded_completion_rejected',accepted=not late_rejected))
        if case=='blocked_output' and now<150:
            output_blocked |= len(runtime.stock['outfeed'])==runtime.places[-1].capacity
        else:
            for job in sorted(runtime.stock['outfeed']):runtime.release_outfeed(job,now)
        runtime.expire(now)
        for command in runtime.dispatch(now):
            executor[command['id']]=dict(due=now+(2. if command['kind']=='transfer' else durations[command['target']]))
        if len(runtime.completed)==job_count:break
    assert len(runtime.completed)==job_count
    assert case!='blocked_output' or output_blocked
    assert case not in {'device_recovery','transfer_recovery'} or injected and recovered and late_rejected
    event_bytes=''.join(json.dumps(e)+'\n' for e in runtime.events).encode()
    (output/'events.jsonl').write_bytes(event_bytes)
    (output/'adapter.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in adapter_events),encoding='utf-8')
    report=dict(passed=True,case=case,job_count=job_count,completed=runtime.completed,elapsed_sim_s=now,
        flow=[dict(p.__dict__) for p in runtime.places],device_durations_s=durations,transfer_duration_s=2.,
        faults=faults,recovered_command=recovered,superseded_completion_rejected=late_rejected,
        output_saturation_observed=output_blocked,event_count=len(runtime.events),
        source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ['resource_runtime.py','run_resource_demo.py']},
        events_sha256=hashlib.sha256(event_bytes).hexdigest(),final_snapshot=runtime.snapshot(),
        claim_boundary='Deterministic software executor supplies assumed completion/stop feedback and durations. Finite-capacity resource and recovery logic only; no Gazebo multi-workpiece motion, physical stop measurement, real process times or AGV integration.')
    (output/'result.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['normal','blocked_output','device_recovery','transfer_recovery'],default='normal')
    p.add_argument('--jobs',type=int,default=8);args=p.parse_args()
    if args.jobs<3:raise SystemExit('At least three workpieces required.')
    report=simulate(args.case,args.output,args.jobs)
    print(json.dumps({k:report[k] for k in ['passed','case','completed','elapsed_sim_s','event_count']}))


if __name__=='__main__':main()
