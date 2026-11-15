"""Finite station supplies committed only by an identified measured handoff."""
import math
import threading
from functools import wraps
from .resource_runtime import ResourceRuntime
from .supply_bridge import SOURCES,DESTINATIONS


def locked(method):
    @wraps(method)
    def call(self,*args,**kwargs):
        with self.lock:return method(self,*args,**kwargs)
    return call


class MaterialSupply:
    def __init__(self,run_id,*,initial=None,capacity=4,destinations=None,dock_source='gazebo_truth'):
        if dock_source not in ('gazebo_truth','localized'):raise ValueError('explicit supported dock authority required')
        self.dock_source=dock_source
        self.run_id=run_id;self.initial=dict(derust=2,magnetic=0) if initial is None else dict(initial)
        self.stock=dict(self.initial);self.capacity=capacity;self.events=[];self.deliveries={};self.consumed=set();self.waiting=set()
        self.active=None;self.last_time=-math.inf
        self.destinations=dict(DESTINATIONS if destinations is None else destinations)
        if any(s not in self.destinations or len(self.destinations[s])!=3 or not all(math.isfinite(v) for v in self.destinations[s]) for s in self.stock):
            raise ValueError('finite supply handoff targets required')
        self.lock=threading.RLock()

    @locked
    def shared_state(self):
        """Serializable copies for the optional separate-process executor."""
        return dict(initial=dict(self.initial),stock=dict(self.stock),capacity=self.capacity,
            events=list(self.events),deliveries={k:dict(v) for k,v in self.deliveries.items()},
            consumed=set(self.consumed),waiting=set(self.waiting),destinations=dict(self.destinations),dock_source=self.dock_source)

    @locked
    def event(self,kind,now,**data):
        if not math.isfinite(now) or now<self.last_time-.15:raise ValueError('supply source time skew exceeds 0.15s')
        source=now;now=max(now,self.last_time);self.last_time=now
        self.events.append(dict(seq=len(self.events),event=kind,time_s=now,source_time_s=source,**data,stock=dict(self.stock)))

    @locked
    def arrived(self,task_id,station,quantity,command,observation,*,now,dock_proof=None):
        if task_id in self.deliveries or station not in self.stock:return False
        recipe={'inspection_consumables':('magnetic',4,'receiving'),'small_tools':('derust',2,'supply')}.get(task_id)
        if recipe is None or (station,quantity,command['nodes'][-1])!=recipe:return False
        if (command['run_id']!=self.run_id or command['task_id']!=task_id or command['kind']!='MOVE'
            or command['nodes'][-1] not in {'supply','receiving'} or command['source']!=self.dock_source
            or not isinstance(quantity,int) or quantity<=0):return False
        if list(command['target_xy'])!=list(SOURCES[task_id][:2]):return False
        if self.dock_source=='localized':
            from .navigation_dock import localized_dock_valid
            if not localized_dock_valid(command,observation,dock_proof,self.run_id,now):return False
        if (math.dist(observation['pose'][:2],command['target_xy'])>.10
            or observation['linear_speed']>=.025 or observation['angular_speed']>=.05
            or observation['sim_time_s']>now or now-observation['sim_time_s']>.15):return False
        self.deliveries[task_id]=dict(task_id=task_id,station=station,quantity=quantity,
            route_id=command['route_id'],robot=command['robot_id'],status='at_dock')
        receipt=dict(delivery=dict(self.deliveries[task_id]),command=dict(command),observation=dict(observation))
        if self.dock_source=='localized':receipt.update(dock_source='localized',dock_proof=dict(dock_proof))
        self.event('DOCK_RECEIPT',now,**receipt)
        return True

    @locked
    def begin_handoff(self,task_id,*,now):
        d=self.deliveries.get(task_id)
        if self.active or d is None or d['status']!='at_dock':return None
        token=f'{self.run_id}:handoff:{task_id}'
        self.active=dict(token=token,task_id=task_id,station=d['station'],issued_at=now)
        d['status']='handoff';self.event('HANDOFF_ISSUED',now,command=dict(self.active));return dict(self.active)

    @locked
    def commit(self,token,observation,target,*,now):
        a=self.active
        if a is None or a['token']!=token:return False
        if list(target)!=list(self.destinations[a['station']]):return False
        if (not a['issued_at']<=observation['time_s']<=now or now-observation['time_s']>.15
            or math.dist(observation['xyz'],target)>.025
            or max(abs(v) for v in observation['velocity'])>=.035):return False
        d=self.deliveries[a['task_id']];station=d['station']
        if self.stock[station]+d['quantity']>self.capacity:return False
        self.stock[station]+=d['quantity'];d['status']='committed';self.active=None
        self.event('SUPPLY_COMMITTED',now,token=token,task_id=d['task_id'],station=station,
            quantity=d['quantity'],observation=dict(observation),target=list(target));return True

    @locked
    def can_start(self,job,station,now):
        if station not in self.stock:return True
        key=(job,station)
        if key in self.consumed:return True
        if self.stock[station]>0:return True
        if key not in self.waiting:
            self.waiting.add(key);self.event('MATERIAL_WAIT',now,job=job,station=station)
        return False

    @locked
    def consume(self,job,station,token,now):
        if station not in self.stock:return
        key=(job,station)
        if key in self.consumed:raise ValueError('workpiece material charged twice')
        if self.stock[station]<=0:raise ValueError('device started without supply')
        self.stock[station]-=1;self.consumed.add(key)
        self.event('MATERIAL_CONSUMED',now,job=job,station=station,command_id=token,
            resumed_wait=key in self.waiting)


class SuppliedRuntime(ResourceRuntime):
    def __init__(self,*args,supply,**kwargs):
        super().__init__(*args,**kwargs);self.supply=supply

    def dispatch(self,now):
        self._time(now)
        for job,state in sorted(self.jobs.items(),key=lambda pair:(-pair[1]['index'],pair[0])):
            if state['state']!='ready':continue
            p=self.places[state['index']]
            if p.device and not state['device_done']:
                if not self.supply.can_start(job,p.id,now):continue
                command=self._issue(job,'device',p.id,p.id,now)
                self.supply.consume(job,p.id,command['id'],now)
            elif state['index']+1<len(self.places) and self.carrier is None:
                target=self.places[state['index']+1]
                if len(self.stock[target.id])+len(self.reserved[target.id])<target.capacity:
                    self._issue(job,'transfer',p.id,target.id,now)
        result=self.pending;self.pending=[];return result
