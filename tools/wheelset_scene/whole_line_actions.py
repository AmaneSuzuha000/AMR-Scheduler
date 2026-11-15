"""Cooperative eighteen-stage actions for a shared Gazebo feedback loop.

Each advance consumes measured observations; it never sleeps, starts a server
or fabricates a measurement. Multiple workers can therefore keep their native
servo leases alive through one control loop while one carrier remains shared.
"""
import math

from .actuated_cell import JointSample
from .device_action import DeviceAction
from .finishing_action import RunInAction,RunInSample
from .finishing_flow import PAINT
from .front_flow import ORIGIN,TRAVEL_Z,Waypoints,transfer_points
from .press_action import PressAction,PressSample


class WholeLineAction:
    def __init__(self,command,worker,targets,*,case='normal',recovered=False):
        self.command=command;self.worker=worker;self.targets=targets
        self.token=command['execution_id'];self.job=worker.job;self.stage=worker.recipe[worker.stage]
        self.case=case;self.recovered=recovered;self.disabled=None
        self.phase='initial';self.complete=False;self.fault=None;self.events=[];self.waypoints=[];self.press_samples=[]
        self.detached=set();self.observations={};self.now=None;self.program=self.run()

    def advance(self,observations,now):
        if self.complete or self.fault:return
        self.observations=observations;self.now=now
        try:next(self.program)
        except StopIteration:self.complete=True;self.disabled=None
        except (RuntimeError,TimeoutError) as exc:self.fault=str(exc);self.disabled=None

    def follow(self,name,points,label,timeout=45.):
        self.phase=label;program=Waypoints(self.token,points,self.now,timeout)
        while not program.complete:
            self.targets[name]['xyz']=list(program.target)
            s=self.observations[name]
            attached=[]
            if name==self.job:
                attached=[n for n in self.worker.items if self.worker.part_state[n] in {'mounted','refitted'} and n not in self.detached]
                attached += [n for n in self.worker.components if self.worker.component_state[n]=='fitted']
            settled=all(math.dist(self.observations[n]['xyz'],self.targets[n]['xyz'])<.035 and
                max(abs(v) for v in self.observations[n]['velocity'])<.035 for n in attached)
            index=program.index
            if settled and program.update(s['xyz'],s['velocity'],s['time_s']):
                self.waypoints.append(dict(command_id=self.token,entity=name,label=label,index=index,
                    target=list(points[index]),observation=dict(s),time_s=self.now))
            if not program.complete:yield

    def scan(self):
        for extension in [.22,0.]:
            self.phase='scan_extend' if extension else 'scan_retract'
            self.targets[self.job]['probe']=extension;began=self.now
            while True:
                s=self.observations[self.job]
                if self.now-began>8:raise TimeoutError('scanner_timeout')
                if abs(s['probe']-extension)<.005 and abs(s['probe_velocity'])<.015:
                    if extension and self.stage['id']=='identification':
                        assert self.worker.tag_read(self.token,'DEMO-'+self.job,self.now)
                    self.events.append(dict(command_id=self.token,event=self.phase,time_s=s['time_s'],sample=dict(s)))
                    break
                yield

    def rotate(self):
        self.phase='measured_rotation';a=DeviceAction(self.token,self.job,self.stage['id'],self.stage['target'][0]-ORIGIN[0])
        while a.state not in {'complete','failed'}:
            s=self.observations[self.job]
            if math.dist(s['xyz'],self.stage['target'])>.03:raise RuntimeError('workpiece_left_device')
            a.update(JointSample(s['time_s'],s['xyz'][0]-ORIGIN[0],s['velocity'][0],s['probe'],s['probe_velocity'],s['spin'],s['omega']))
            _,self.targets[self.job]['probe'],self.targets[self.job]['spin']=a.targets()
            if a.state not in {'complete','failed'}:yield
        self.events.extend(a.events)
        if a.state=='failed':raise RuntimeError(a.fault)

    def dismantle(self,kind):
        for name,item in self.worker.items.items():
            if item['kind']!=kind or self.worker.part_state[name]=='binned':continue
            self.detached.add(name)
            if self.worker.part_state[name]=='mounted':
                point=[a+b for a,b in zip(self.stage['target'],item['offset'])]
                point[1]+=item['sign']*item['separation_m']
                yield from self.follow(name,[point],'separate_'+name,12.)
                assert self.worker.separate(self.token,name,self.now,dict(self.observations[name]))
            s=self.observations[name]['xyz'];dest=item['destination']
            yield from self.follow(name,[(*s[:2],TRAVEL_Z),(*dest[:2],TRAVEL_Z),dest],'bin_'+name)
            assert self.worker.deposit(self.token,name,self.now,dict(self.observations[name]))

    def prepare(self):
        for name,item in self.worker.components.items():
            if self.worker.component_state[name]=='unreceived':
                yield from self.follow(name,[item['receiving']],'receive_'+name)
                assert self.worker.receive(name,self.now,dict(self.observations[name]))
        assert self.worker.reserve_stock(self.token,self.now)
        for name,item in self.worker.components.items():
            if self.worker.component_state[name]!='reserved':continue
            r=item['receiving'];s=item['staged'];gauge=[]
            for point in [(*r[:2],r[2]+.12),r]:
                yield from self.follow(name,[point],'gauge_'+name)
                gauge.append(dict(self.observations[name]))
            yield from self.follow(name,[(*r[:2],TRAVEL_Z),(*s[:2],TRAVEL_Z),s],'stage_'+name)
            assert self.worker.prepared(self.token,name,self.now,dict(self.observations[name]),gauge)

    def press(self):
        yield from self.prepare()
        for side in ['left','right']:
            if side in self.worker.fitted_sides:continue
            names=[n for n,i in self.worker.components.items() if i['side']==side]
            for name in names:yield from self.follow(name,[self.worker.components[name]['staged']],'restage_'+name)
            ram='press_ram_'+side;bearing=next(n for n in names if self.worker.components[n]['kind']=='bearing')
            ring=next(n for n in names if self.worker.components[n]['kind']=='ring')
            a=PressAction(self.token,self.job,bearing,ring)
            while a.state not in {'complete','failed'}:
                self.phase='press_'+side+'_'+a.state;self.targets[ram]['probe']=a.target()
                self.disabled=(ram,'probe') if self.case=='press_recovery' and side=='right' and a.state=='insert' and not self.recovered else None
                s=self.observations[ram];fixture=self.observations[self.job]
                q=PressSample(s['time_s'],self.token,s['probe'],s['probe_velocity'],s['applied_effort']['probe'],
                    math.dist(fixture['xyz'],self.stage['target']),max(abs(v) for v in fixture['velocity']))
                self.press_samples.append(dict(side=side,**q.__dict__,fixture=dict(fixture),ram=dict(s)))
                a.update(q)
                if a.state in {'insert','hold','retract'}:
                    fraction=max(0.,min(1.,(s['probe']-.04)/.18))
                    for name in names:
                        item=self.worker.components[name]
                        self.targets[name]['xyz']=[x+fraction*(y-x) for x,y in zip(item['staged'],item['fitted'])]
                if a.state not in {'complete','failed'}:yield
            self.disabled=None;self.events.extend(a.events)
            if a.state=='failed':raise TimeoutError(a.fault)
            for name in names:yield from self.follow(name,[self.worker.components[name]['fitted']],'fit_'+name)
            assert self.worker.fitted(self.token,side,self.now,{n:dict(self.observations[n]) for n in names},a.events)

    def close(self):
        for name,item in self.worker.items.items():
            if item['kind']!='cap' or self.worker.part_state[name]=='refitted':continue
            if self.worker.part_state[name]=='binned':
                assert self.worker.retrieve_cap(self.token,name,self.now,dict(self.observations[name]))
            r=self.observations[name]['xyz'];f=[a+b for a,b in zip(self.stage['target'],item['offset'])]
            s=list(f);s[1]+=item['sign']*.35
            yield from self.follow(name,[(*r[:2],6.),(*s[:2],6.),s,f],'close_'+name,160.)
            assert self.worker.refit_cap(self.token,name,self.now,dict(self.observations[name]));self.detached.discard(name)

    def run_in(self):
        self.phase='bearing_run_in';a=RunInAction(self.token)
        while a.state not in {'complete','failed'}:
            self.targets[self.job]['spin']=a.target();s=self.observations[self.job]
            a.update(RunInSample(s['time_s'],self.token,s['spin'],s['omega'],math.dist(s['xyz'],self.stage['target']),
                max(abs(v) for v in s['velocity']),s['probe'],s['probe_velocity']))
            if a.state not in {'complete','failed'}:yield
        self.events.extend(a.events)
        if a.state=='failed':raise RuntimeError(a.fault)
        assert self.worker.finish_measured(self.token,self.now,dict(kind='run_in',events=a.events))

    def paint(self):
        evidence=[]
        for dx in [-.7,.7,-.7]:
            point=(PAINT[0]+dx,PAINT[1],PAINT[2]+.85)
            yield from self.follow('spray_tool',[point],'spray_traverse')
            fixture=self.observations[self.job]
            if math.dist(fixture['xyz'],PAINT)>.025 or max(abs(v) for v in fixture['velocity'])>.035:raise RuntimeError('spray_fixture_departed')
            evidence.append(dict(tool=dict(self.observations['spray_tool']),fixture=dict(fixture),target=point))
        assert self.worker.finish_measured(self.token,self.now,dict(kind='paint',traverse=evidence))

    def run(self):
        if self.command['kind']=='transfer':
            yield from self.follow(self.job,transfer_points(self.observations[self.job]['xyz'],self.stage),'transfer_'+self.stage['id'])
            return
        action=self.stage['action']
        if action=='scan':yield from self.scan()
        elif action=='rotate':yield from self.rotate()
        elif action in {'caps','bearings'}:yield from self.dismantle('cap' if action=='caps' else 'bearing')
        elif action=='press':yield from self.press()
        elif action=='close':yield from self.close()
        elif action=='run_in':yield from self.run_in()
        elif action=='paint':yield from self.paint()
        elif action=='storage_gate':
            self.phase='storage_wait'
            ledger=self.worker.storage
            if not ledger.reserve(self.job,self.token,self.now):
                while ledger.occupant is not None or ledger.reserved is not None:yield
                assert ledger.reserve(self.job,self.token,self.now)
        elif action=='store':
            reserved=self.worker.storage.reserved
            if reserved is None or reserved['job']!=self.job:raise RuntimeError('storage_reservation_lost')
            assert self.worker.storage.committed(self.job,reserved['command_id'],self.now,dict(self.observations[self.job]))
