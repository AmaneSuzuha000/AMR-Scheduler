"""Single-workpiece process prototype; motion feedback gates software equipment.

This is a local controller/event log, not a new fleet wire protocol. Service
durations are explicit demo assumptions. No contact or joint control is implied.
"""
from dataclasses import dataclass
import math
import uuid


@dataclass(frozen=True)
class Operation:
    id: str
    asset_instance: str
    target: tuple[float, float, float]
    duration_s: float


def inspection_plan(scene, duration_s=2.0):
    """Use the existing lower inspection lane; never infer factory cycle times."""
    result=[]
    for asset in ['derust_hood','magnetic_test_hood','ultrasonic_test_hood']:
        item=min((i for i in scene['instances'] if i['asset']==asset),key=lambda i:i['pose'][1])
        result.append(Operation(asset,item['id'],tuple(item['pose'][:2]+[0]),duration_s))
    result.append(Operation('outfeed','inspection_outfeed',(52.5,result[-1].target[1],0),0))
    return result


class ProcessController:
    """Reserve a station until work completes; failed actions retain their lock."""
    def __init__(self, operations, *, run_id=None, tolerance_m=.03, transfer_timeout_s=120):
        self.operations=list(operations)
        if not self.operations or len({o.id for o in self.operations})!=len(self.operations):
            raise ValueError('unique nonempty operations required')
        for o in self.operations:
            if len(o.target)!=3 or not all(math.isfinite(v) for v in o.target):raise ValueError('invalid target')
            if not math.isfinite(o.duration_s) or o.duration_s<0:raise ValueError('invalid duration')
        if not math.isfinite(tolerance_m) or tolerance_m<=0:raise ValueError('invalid tolerance')
        if not math.isfinite(transfer_timeout_s) or transfer_timeout_s<=0:raise ValueError('invalid timeout')
        self.run_id=run_id or uuid.uuid4().hex
        self.tolerance=tolerance_m;self.timeout=transfer_timeout_s
        self.state='idle';self.index=-1;self.command=None;self.locked_station=None
        self.last_time=-math.inf;self.events=[];self.fault=None;self.due=None

    def _time(self,now):
        if not math.isfinite(now) or now<self.last_time:raise ValueError('clock must be finite and monotonic')
        self.last_time=now

    def _event(self,kind,now,**data):
        self.events.append(dict(data,seq=len(self.events),run_id=self.run_id,event=kind,time_s=now))

    def start(self,now):
        self._time(now)
        if self.state!='idle':raise RuntimeError('cycle already started')
        self.index=0;self._begin_transfer(now)

    def _begin_transfer(self,now):
        operation=self.operations[self.index];self.locked_station=operation.id
        self.command=dict(run_id=self.run_id,command_id=f'{self.run_id}:{self.index}',
            operation=operation.id,target=list(operation.target),issued_at=now)
        self.state='transferring'
        self._event('TRANSFER_REQUESTED',now,**self.command)

    def observe(self,*,run_id,command_id,pose,observed_at,received_at,success=True):
        self._time(received_at)
        if self.state!='transferring':return False
        if run_id!=self.run_id or command_id!=self.command['command_id']:return False
        if not math.isfinite(observed_at) or not self.command['issued_at']<=observed_at<=received_at:return False
        if received_at-self.command['issued_at']>=self.timeout:
            self.fail('transfer_timeout',received_at);return False
        if not success:self.fail('execution_failed',received_at);return False
        if len(pose)!=3 or not all(math.isfinite(v) for v in pose):return False
        error=math.dist(pose,self.command['target'])
        if error>self.tolerance:return False
        self._event('ARRIVAL_CONFIRMED',received_at,operation=self.locked_station,
            command_id=command_id,observed_pose=list(pose),position_error_m=error)
        self.state='processing';self.due=received_at+self.operations[self.index].duration_s
        self._event('DEVICE_BUSY',received_at,operation=self.locked_station,
            duration_s=self.operations[self.index].duration_s,duration_source='assumed_demo_wall_clock')
        return True

    def tick(self,now):
        self._time(now)
        if self.state=='transferring' and now-self.command['issued_at']>=self.timeout:
            self.fail('transfer_timeout',now)
        elif self.state=='processing' and now>=self.due:
            self._event('DEVICE_COMPLETE',now,operation=self.locked_station)
            self._event('DEVICE_IDLE',now,operation=self.locked_station)
            self.locked_station=None;self.command=None;self.index+=1
            if self.index==len(self.operations):
                self.state='complete';self._event('CYCLE_COMPLETE',now)
            else:self._begin_transfer(now)

    def fail(self,reason,now):
        self._time(now)
        if self.state in {'complete','failed'}:return
        self.fault=reason;self.state='failed'
        self._event('CYCLE_FAILED',now,reason=reason,retained_station=self.locked_station)

    def report(self):
        return dict(run_id=self.run_id,state=self.state,fault=self.fault,
            completed_operations=[e['operation'] for e in self.events if e['event']=='DEVICE_COMPLETE'],
            retained_station=self.locked_station,events=self.events,
            claim_boundary='Single-workpiece software process and pose feedback gate; demo durations, no contact dynamics, driven machinery, multi-job resource manager or AGV scheduling.')
