"""Video-linked front-to-inspection recipe and identified part accounting."""
import math


ORIGIN=(10.,9.,.8)
TRAVEL_Z=6.
STAGES=[
    dict(id='identification',segment='S02',asset='identification_hood_001',target=(14.5,9.,.8),action='scan'),
    dict(id='exterior',segment='S03',asset='opening_r09_exterior_inspection_rollers_001',target=(36.5,9.,.8),action='rotate'),
    dict(id='opening',segment='S04',asset='opening_r09_opening_tool_station_001',target=(29.4,9.,.8),action='caps'),
    dict(id='axle_end',segment='S06',asset='front_r18_001',target=(19.6,9.,.8),action='scan'),
    dict(id='demount',segment='S07',asset='sorting_r09_demount_0',target=(49.8,9.,.8),action='bearings'),
    dict(id='derust',segment='S08',asset='inspection_r04_derust_hood_001',target=(21.,50.,.8),action='rotate'),
    dict(id='magnetic',segment='S08',asset='inspection_r04_magnetic_test_hood_001',target=(35.45,50.,.8),action='rotate'),
    dict(id='ultrasonic',segment='S08',asset='inspection_r04_ultrasonic_test_hood_001',target=(44.5,50.,.8),action='rotate'),
    dict(id='outfeed',segment='S09',asset=None,target=(52.5,50.,.8),action='output')]


def parts(job='ws001'):
    return {f'{job}_{kind}_{side}':dict(job=job,kind=kind,side=side,sign=sign,
        offset=(0.,sign*(1.14 if kind=='cap' else 1.03),0.),
        bin='cap_bin' if kind=='cap' else 'bearing_bin',
        destination=((23.50 if side=='left' else 23.85),5.70,.9) if kind=='cap'
            else ((53.0 if side=='left' else 53.35),5.5,.9),
        separation_m=.55 if kind=='cap' else .70)
        for kind in ['cap','bearing'] for side,sign in [('left',-1),('right',1)]}


def transfer_points(current,stage):
    target=stage['target']
    if stage['id']=='storage':
        from .finishing_flow import STORAGE_APPROACH
        p=STORAGE_APPROACH
        return [(current[0],current[1],TRAVEL_Z),(p[0],p[1],TRAVEL_Z),p,target]
    if stage['id'] in {'magnetic','ultrasonic'}:return [target]
    if stage['id']=='outfeed':return [(current[0],current[1],1.55),(target[0],target[1],1.55),target]
    if stage['id']=='derust':
        # Enter along the open X corridor after lowering outside its roof.
        return [(current[0],current[1],TRAVEL_Z),(18.,50.,TRAVEL_Z),(18.,50.,.8),target]
    points=[(current[0],current[1],TRAVEL_Z),(target[0],target[1],TRAVEL_Z),target]
    if stage['id'] in {'wiping','greasing','press','assembly_outfeed','closing','run_in','painting','storage_wait'}:
        # Keep each long cross-zone segment within the existing 45s measured
        # waypoint timeout; splitting does not remove its swept-space checks.
        result=[];start=current
        for end in points:
            count=max(1,math.ceil(max(abs(a-b) for a,b in zip(start,end))/30))
            result.extend(tuple(a+(b-a)*i/count for a,b in zip(start,end)) for i in range(1,count+1))
            result[-1]=end
            start=end
        return result
    return points


class FrontRuntime:
    """Identity, retained station ownership and finite, separate part bins.

    The ROS adapter must supply measured completion/stopping. This state store
    never completes a command from elapsed time or a planned waypoint alone.
    """
    def __init__(self,run_id,job='ws001',recipe=None):
        self.recipe=STAGES if recipe is None else recipe
        self.run_id=run_id;self.job=job;self.items=parts(job)
        self.part_state={p:'mounted' for p in self.items};self.bins={'cap_bin':[],'bearing_bin':[]}
        self.capacity=2;self.events=[];self.commands={};self.current=None;self.stage=-1
        self.station='infeed';self.reserved=None;self.completed=[];self.last=-math.inf
        self.tag=None;self.passed=False

    def snapshot(self):
        return dict(job=self.job,station=self.station,reserved=self.reserved,part_state=dict(self.part_state),
            bins={k:list(v) for k,v in self.bins.items()},completed=list(self.completed),tag=self.tag)

    def event(self,event,now,**data):
        if not math.isfinite(now) or now<self.last:raise ValueError('nonmonotonic source time')
        self.last=now
        self.events.append(dict(seq=len(self.events),event=event,time_s=now,**data,snapshot=self.snapshot()))

    def issue(self,stage,kind,now,*,replacement=None):
        if replacement is None:
            if self.current is not None:raise ValueError('command already owns station')
            if stage!=len(self.completed):raise ValueError('operation order')
            if kind=='transfer':
                self.reserved=self.recipe[stage]['id']
            elif self.station!=self.recipe[stage]['id']:raise ValueError('device before arrival')
        else:
            old=self.commands[replacement]
            if old['status']!='failed' or not old.get('stopped'):raise ValueError('retry before stop')
            if (stage,kind)!=(old['stage'],old['kind']):raise ValueError('retry changed ownership')
            old['status']='superseded'
        token=f'{self.run_id}:{len(self.commands)}'
        command=dict(id=token,job=self.job,stage=stage,station=self.recipe[stage]['id'],kind=kind,
            status='active',issued_at=now)
        self.commands[token]=command;self.current=token;self.stage=stage
        self.event('COMMAND_ISSUED',now,command=dict(command))
        if replacement:self.event('RECOVERY_REISSUED',now,old_command_id=replacement,command_id=token)
        return token

    def tag_read(self,token,tag,now):
        c=self.commands[token]
        if token!=self.current or c['station']!='identification' or c['status']!='active':return False
        if tag!=f'DEMO-{self.job}':return False
        self.tag=tag;self.event('SIMULATED_TAG_READ',now,command_id=token,tag=tag)
        return True

    def separate(self,token,part,now,observation):
        c=self.commands[token];item=self.items[part]
        if token!=self.current or c['status']!='active' or self.part_state[part]!='mounted':return False
        if c['station']!=('opening' if item['kind']=='cap' else 'demount'):return False
        self.part_state[part]='separated'
        self.event('PART_SEPARATED',now,command_id=token,part=part,observation=observation);return True

    def deposit(self,token,part,now,observation):
        c=self.commands[token];item=self.items[part];bin_id=item['bin']
        if token!=self.current or c['status']!='active' or self.part_state[part]!='separated':return False
        if len(self.bins[bin_id])>=self.capacity:return False
        self.part_state[part]='binned';self.bins[bin_id].append(part)
        self.event('PART_BIN_COMMITTED',now,command_id=token,part=part,bin=bin_id,observation=observation);return True

    def complete(self,token,now,observation):
        c=self.commands.get(token)
        if c is None or token!=self.current or c['status']!='active':return False
        if c['kind']=='transfer':self.station=c['station'];self.reserved=None
        else:
            station=c['station']
            if station=='identification' and self.tag is None:return False
            if station in {'opening','demount'}:
                kind='cap' if station=='opening' else 'bearing'
                if any(self.part_state[p]!='binned' for p,i in self.items.items() if i['kind']==kind):return False
            self.completed.append(station)
        c['status']='done';self.current=None
        self.event('ARRIVAL_CONFIRMED' if c['kind']=='transfer' else 'DEVICE_COMPLETE',now,
            command_id=token,observation=observation)
        self.passed=len(self.completed)==len(self.recipe);return True

    def fail(self,token,now,reason):
        c=self.commands[token]
        if c['status']!='active':return False
        c['status']='failed';self.event('FAULT_LATCHED',now,command_id=token,reason=reason);return True

    def stopped(self,token,now,observation):
        c=self.commands[token]
        if c['status']!='failed':return False
        c['stopped']=True;self.event('STOP_CONFIRMED',now,command_id=token,observation=observation);return True


class Waypoints:
    def __init__(self,token,points,start,timeout=45):
        self.token=token;self.points=[tuple(p) for p in points];self.index=0;self.entered=start;self.timeout=timeout

    @property
    def target(self):return self.points[min(self.index,len(self.points)-1)]

    def update(self,xyz,velocity,now):
        if now-self.entered>self.timeout:raise TimeoutError('measured_waypoint_timeout')
        # Leave room for the subsequent world-position query and event receipt;
        # the independent audit retains its .025m/.035m/s external limits.
        if math.dist(xyz,self.target)<.012 and max(abs(v) for v in velocity)<.020:
            self.index+=1;self.entered=now;return True
        return False

    @property
    def complete(self):return self.index==len(self.points)
