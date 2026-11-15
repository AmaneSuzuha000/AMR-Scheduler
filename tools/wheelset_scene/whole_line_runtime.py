"""Shared resources around identified, feedback-driven eighteen-stage workers.

This layer owns stations, the one assumed wheelset carrier, shared part bins,
consumables and the storage slot. It does not generate device motion evidence.
An execution adapter uses each worker's existing action protocols, then calls
complete with the matching stopped observation. No elapsed-time completion.
"""
import math

from .finishing_flow import FinishingRuntime,FiniteStorage,FULL_STAGES,STORED_JOB,STORAGE_UNLOAD,at
from .front_flow import ORIGIN
from .material_supply import MaterialSupply
from .resource_runtime import Place,ResourceRuntime


def whole_line_places():
    return [Place('infeed',1,ORIGIN)]+[Place(s['id'],1,s['target'],True) for s in FULL_STAGES]


def stopped_sample(observation,now):
    values=[observation['time_s'],*observation['xyz'],*observation['velocity'],
        observation['omega'],observation['probe'],observation['probe_velocity']]
    return all(math.isfinite(v) for v in values) and 0<=now-observation['time_s']<=.15 and (
        max(abs(v) for v in observation['velocity'])<.035 and abs(observation['omega'])<.025
        and abs(observation['probe'])<.005 and abs(observation['probe_velocity'])<.015)


class WholeLineRuntime(ResourceRuntime):
    """Reuse the finite resource protocol and per-workpiece finishing protocol.

    Bin capacities and single carrier/station capacities are explicit layout
    assumptions. Supply uses the same measured dock-to-shuttle interface as
    the existing two-SCOUT joint runner. No fleet/Gazebo success is implied by
    constructing or unit-testing this coordinator.
    """
    def __init__(self,run_id,*,supply=None,bin_capacity=4,timeout_s=600.,worker_geometry=None):
        super().__init__(whole_line_places(),run_id=run_id,timeout_s=timeout_s)
        if not isinstance(bin_capacity,int) or bin_capacity<2:raise ValueError('bin holds at least one identified pair')
        self.workers={};self.storage=FiniteStorage(run_id);self.worker_geometry=worker_geometry or {}
        self.supply=supply if supply is not None else MaterialSupply(run_id)
        self.bins={'cap_bin':[],'bearing_bin':[]};self.bin_capacity=bin_capacity
        self.bin_holds={name:set() for name in self.bins};self.waits=set()

    def admit(self,job,now,*,observation):
        if not isinstance(job,str) or not job or ':' in job:raise ValueError('unique nonempty workpiece identity required')
        if observation.get('entity')!=job or not stopped_sample(observation,now) or not at(observation,ORIGIN,now):return False
        worker=FinishingRuntime(self.run_id+':'+job,job,storage=self.storage)
        if job in self.worker_geometry:
            geometry=self.worker_geometry[job]
            if set(geometry['parts'])!=set(worker.items) or set(geometry['components'])!=set(worker.components):raise ValueError('workpiece geometry identities')
            worker.items=geometry['parts'];worker.components=geometry['components']
        if not super().admit(job,now):return False
        worker.bins=self.bins;worker.capacity=self.bin_capacity;self.workers[job]=worker
        return True

    def _issue(self,job,kind,source,target,now):
        worker=self.workers[job];index=next(i for i,s in enumerate(FULL_STAGES) if s['id']==target)
        old=worker.current
        local=worker.issue(index,kind,now,replacement=old if old is not None else None)
        command=super()._issue(job,kind,source,target,now)
        command['execution_id']=local
        # ResourceRuntime stores copies in the queue/event: bind those too.
        self.pending[-1]['execution_id']=local
        self.events[-1]['command']['execution_id']=local
        if kind=='device' and target in {'opening','demount'}:
            bin_id='cap_bin' if target=='opening' else 'bearing_bin'
            self.bin_holds[bin_id].update(n for n,i in worker.items.items() if i['bin']==bin_id)
        if kind=='device' and (job,target) not in self.supply.consumed:
            self.supply.consume(job,target,local,now)
        return command

    def _bin_available(self,job,station):
        if station not in {'opening','demount'}:return True
        bin_id='cap_bin' if station=='opening' else 'bearing_bin'
        names={n for n,i in self.workers[job].items.items() if i['bin']==bin_id}
        return len(set(self.bins[bin_id])|self.bin_holds[bin_id]|names)<=self.bin_capacity

    def dispatch(self,now):
        self._time(now)
        for job,state in sorted(self.jobs.items(),key=lambda p:(-p[1]['index'],p[0])):
            if state['state']!='ready':continue
            p=self.places[state['index']]
            if p.device and not state['device_done']:
                if not self._bin_available(job,p.id):
                    key=(job,p.id)
                    if key not in self.waits:
                        self.waits.add(key);self._event('PART_BIN_WAIT',now,job=job,station=p.id)
                    continue
                if not self.supply.can_start(job,p.id,now):continue
                self._issue(job,'device',p.id,p.id,now)
            elif state['index']+1<len(self.places) and self.carrier is None:
                target=self.places[state['index']+1]
                if len(self.stock[target.id])+len(self.reserved[target.id])<target.capacity:
                    self._issue(job,'transfer',p.id,target.id,now)
        result=self.pending;self.pending=[];return result

    def observe(self,*args,**kwargs):
        raise ValueError('use complete with identified measured observation, not a device_complete flag')

    def complete(self,token,now,observation):
        self._time(now);c=self.commands.get(token)
        if c is None or c['status']!='active':return False
        if now-c['issued_at']>=self.timeout:
            self._fail(c,'feedback_timeout',now);return False
        if observation.get('entity')!=c['job'] or not stopped_sample(observation,now):return False
        if observation['time_s']<c['issued_at']:return False
        target=next(p.target for p in self.places if p.id==c['target'])
        if not at(observation,target,now):return False
        worker=self.workers[c['job']]
        if not worker.complete(c['execution_id'],now,observation):return False
        accepted=super().observe(token,now=now,observed_at=observation['time_s'],
            pose=observation['xyz'],device_complete=c['kind']=='device')
        assert accepted
        if c['kind']=='device':
            for names in self.bin_holds.values():names.difference_update(worker.items)
        return True

    def _fail(self,c,reason,now):
        worker=self.workers[c['job']]
        assert worker.fail(c['execution_id'],now,reason)
        super()._fail(c,reason,now)

    def fail(self,token,now,reason):
        self._time(now);c=self.commands.get(token)
        if c is None or c['status']!='active':return False
        self._fail(c,reason,now);return True

    def confirm_stopped(self,token,*,now,observations):
        self._time(now);c=self.commands.get(token)
        if c is None or c['status']!='failed':return False
        worker=self.workers[c['job']]
        required={worker.job,*worker.items,*worker.components}
        if c['target']=='press':required.update({'press_ram_left','press_ram_right'})
        if not required.issubset(observations):return False
        fault_time=next(e['time_s'] for e in self.events if e['event']=='FAULT_LATCHED' and e['command_id']==token)
        for name in required:
            s=observations[name]
            if s.get('entity')!=name or not stopped_sample(s,now) or s['time_s']<fault_time:return False
        stamp=min(observations[n]['time_s'] for n in required)
        if not super().confirm_stopped(token,now=now,observed_at=stamp,stopped=True):return False
        assert worker.stopped(c['execution_id'],now,observations)
        return True

    def release_outfeed(self,*args,**kwargs):
        raise ValueError('storage occupancy is released only by complete_unload')

    def begin_unload(self,now):
        self._time(now);job=self.storage.occupant
        if job!=STORED_JOB:
            state=self.jobs.get(job)
            if state is None or state['index']!=len(self.places)-1 or state['state']!='ready' or not state['device_done']:return None
        return self.storage.begin_release(now)

    def complete_unload(self,token,now,observation):
        self._time(now);job=self.storage.occupant
        if token is None or token!=self.storage.release_token or job is None:return False
        if observation.get('entity')!=job or not stopped_sample(observation,now):return False
        issued=next(e['time_s'] for e in self.storage.events if e['event']=='STORAGE_RELEASE_ISSUED' and e['command_id']==token)
        if observation['time_s']<issued or not at(observation,STORAGE_UNLOAD,now):return False
        if not self.storage.released(token,now,observation):return False
        if job!=STORED_JOB:assert super().release_outfeed(job,now)
        return True
