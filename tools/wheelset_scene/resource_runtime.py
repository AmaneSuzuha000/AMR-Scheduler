"""Finite-capacity wheelset flow with feedback gates and retained fault locks.

This controller owns logical resources. Adapters must supply measured arrival,
device completion and stopping evidence; dispatch alone never releases a slot.
"""
from dataclasses import dataclass
import math
import uuid


@dataclass(frozen=True)
class Place:
    id: str
    capacity: int
    target: tuple[float, float, float]
    device: bool = False


def inspection_flow():
    """Explicit demo assumptions: two single-slot buffers and one shared carrier."""
    return [Place('infeed',2,(10.,50.,0.)),Place('derust',1,(21.,50.,0.),True),
        Place('buffer_dm',1,(28.,50.,0.)),Place('magnetic',1,(35.45,50.,0.),True),
        Place('buffer_mu',1,(40.,50.,0.)),Place('ultrasonic',1,(44.5,50.,0.),True),
        Place('outfeed',2,(52.5,50.,0.))]


class ResourceRuntime:
    def __init__(self,places=None,*,run_id=None,timeout_s=30.):
        self.places=list(inspection_flow() if places is None else places)
        if len(self.places)<2 or len({p.id for p in self.places})!=len(self.places):
            raise ValueError('unique ordered places required')
        if any(not isinstance(p.capacity,int) or p.capacity<1 or len(p.target)!=3
            or not all(math.isfinite(v) for v in p.target) for p in self.places):
            raise ValueError('invalid place capacity or pose')
        if not math.isfinite(timeout_s) or timeout_s<=0:raise ValueError('invalid timeout')
        self.run_id=run_id or uuid.uuid4().hex;self.timeout=timeout_s
        self.jobs={};self.commands={};self.events=[];self.pending=[]
        self.stock={p.id:set() for p in self.places};self.reserved={p.id:set() for p in self.places}
        self.carrier=None;self.last_time=-math.inf;self.completed=[]

    def _time(self,now):
        if not math.isfinite(now) or now<self.last_time:raise ValueError('monotonic finite clock required')
        self.last_time=now

    def snapshot(self):
        return dict(stock={k:sorted(v) for k,v in self.stock.items()},
            reserved={k:sorted(v) for k,v in self.reserved.items()},carrier=self.carrier,
            jobs={k:dict(v) for k,v in self.jobs.items()},completed=list(self.completed))

    def check(self):
        for p in self.places:
            assert len(self.stock[p.id])+len(self.reserved[p.id])<=p.capacity,p.id
        for job,state in self.jobs.items():
            occupied=[p.id for p in self.places if job in self.stock[p.id]]
            assert occupied==([] if state['state']=='complete' else [self.places[state['index']].id])
        held=[c for c in self.commands.values() if c['kind']=='transfer' and c['status'] in {'active','failed'}]
        assert len(held)<=1 and (held[0]['id'] if held else None)==self.carrier
        expected={(c['target'],c['job']) for c in held}
        assert {(p,j) for p,items in self.reserved.items() for j in items}==expected
        for job in self.jobs:
            assert sum(c['job']==job and c['status'] in {'active','failed'} for c in self.commands.values())<=1

    def _event(self,kind,now,**data):
        self.check()
        self.events.append(dict(seq=len(self.events),event=kind,time_s=now,**data,snapshot=self.snapshot()))

    def admit(self,job,now):
        self._time(now);p=self.places[0]
        if job in self.jobs:raise ValueError('duplicate workpiece')
        if len(self.stock[p.id])+len(self.reserved[p.id])>=p.capacity:return False
        self.jobs[job]=dict(index=0,state='ready',device_done=False,action=None)
        self.stock[p.id].add(job);self._event('ADMITTED',now,job=job);return True

    def _issue(self,job,kind,source,target,now):
        token=f'{self.run_id}:{len(self.commands)}'
        c=dict(id=token,run_id=self.run_id,job=job,kind=kind,source=source,target=target,
            issued_at=now,status='active',stop_confirmed=False)
        self.commands[token]=c;state=self.jobs[job];state.update(state=kind,action=token)
        if kind=='transfer':
            self.reserved[target].add(job);self.carrier=token
        self.pending.append(dict(c));self._event('COMMAND_ISSUED',now,command=dict(c));return c

    def dispatch(self,now):
        self._time(now)
        # Downstream first: move finished work away before admitting another.
        for job,state in sorted(self.jobs.items(),key=lambda pair:(-pair[1]['index'],pair[0])):
            if state['state']!='ready':continue
            p=self.places[state['index']]
            if p.device and not state['device_done']:
                self._issue(job,'device',p.id,p.id,now)
            elif state['index']+1<len(self.places) and self.carrier is None:
                target=self.places[state['index']+1]
                if len(self.stock[target.id])+len(self.reserved[target.id])<target.capacity:
                    self._issue(job,'transfer',p.id,target.id,now)
        result=self.pending;self.pending=[];return result

    def observe(self,token,*,now,observed_at,success=True,pose=None,device_complete=False):
        self._time(now);c=self.commands.get(token)
        if (c is None or c['status']!='active' or not math.isfinite(observed_at)
                or not c['issued_at']<=observed_at<=now):return False
        if now-c['issued_at']>=self.timeout:
            self._fail(c,'feedback_timeout',now);return False
        if not success:self._fail(c,'execution_failed',now);return False
        job=c['job'];state=self.jobs[job]
        if c['kind']=='transfer':
            p=next(p for p in self.places if p.id==c['target'])
            if pose is None or len(pose)!=3 or not all(math.isfinite(v) for v in pose):return False
            if math.dist(pose,p.target)>.03:return False
            self.stock[c['source']].remove(job);self.reserved[p.id].remove(job);self.stock[p.id].add(job)
            self.carrier=None;state.update(index=state['index']+1,device_done=False)
        elif not device_complete:return False
        else:state['device_done']=True
        c['status']='done';state.update(state='ready',action=None)
        self._event('ARRIVAL_CONFIRMED' if c['kind']=='transfer' else 'DEVICE_COMPLETE',now,
            command_id=token,job=job,observed_at=observed_at)
        return True

    def _fail(self,c,reason,now):
        c.update(status='failed',reason=reason);self.jobs[c['job']]['state']='failed'
        self._event('FAULT_LATCHED',now,command_id=c['id'],job=c['job'],reason=reason)

    def expire(self,now):
        self._time(now)
        for c in self.commands.values():
            if c['status']=='active' and now-c['issued_at']>=self.timeout:self._fail(c,'feedback_timeout',now)

    def confirm_stopped(self,token,*,now,observed_at,stopped):
        self._time(now);c=self.commands.get(token)
        if (c is None or c['status']!='failed' or not stopped or not math.isfinite(observed_at)
                or not c['issued_at']<=observed_at<=now):return False
        fault_time=next(e['time_s'] for e in self.events if e['event']=='FAULT_LATCHED' and e['command_id']==token)
        if observed_at<fault_time:return False
        c['stop_confirmed']=True;self._event('STOP_CONFIRMED',now,command_id=token,observed_at=observed_at)
        return True

    def recover(self,token,now):
        self._time(now);c=self.commands.get(token)
        if c is None or c['status']!='failed' or not c['stop_confirmed']:return None
        # Preserve source occupancy and destination reservation across retry.
        c['status']='superseded'
        if c['kind']=='transfer':self.reserved[c['target']].remove(c['job']);self.carrier=None
        replacement=self._issue(c['job'],c['kind'],c['source'],c['target'],now)
        self._event('RECOVERY_REISSUED',now,old_command_id=token,command_id=replacement['id'])
        return replacement['id']

    def release_outfeed(self,job,now):
        self._time(now);state=self.jobs[job]
        if state['index']!=len(self.places)-1 or state['state']!='ready':return False
        if self.places[-1].device and not state['device_done']:return False
        self.stock[self.places[-1].id].remove(job);state['state']='complete';self.completed.append(job)
        self._event('OUTFEED_RELEASED',now,job=job);return True
