"""Finite identified replacement stock and front-to-press demonstration recipe."""
import math

from .front_flow import FrontRuntime,STAGES


PRESS=(134.2,11.3,.95)
ASSEMBLY_STAGES=STAGES+[
    dict(id='wiping',segment='S11',asset='service_r09_wiping_station_001',
         target=(82.96953174,10.59752458,.95),action='scan'),
    dict(id='greasing',segment='S12',asset='greasing_r23_greasing_station_1',
         target=(77.162664853673,10.59752458,.95),action='scan'),
    dict(id='press',segment='S16',asset='bearing_press_cell_001',target=PRESS,action='press'),
    dict(id='assembly_outfeed',segment='S17',asset=None,target=(140.,11.3,.95),action='output')]


def components(job='ws001'):
    result={}
    for side,sign in [('left',-1),('right',1)]:
        for kind in ['bearing','ring']:
            receiving=(112.18987+(0 if side=='left' else .45),30.62737,1.7) if kind=='bearing' else (
                131.36012+(0 if side=='left' else .45),29.63309,1.5)
            offset=sign*(1.03 if kind=='bearing' else .92)
            fitted=(PRESS[0],PRESS[1]+offset,PRESS[2])
            staged=(fitted[0],fitted[1]+sign*.18,fitted[2])
            result[f'{job}_new_{kind}_{side}']=dict(job=job,kind=kind,side=side,sign=sign,
                receiving=receiving,origin=(*receiving[:2],6.) if kind=='ring' and side=='right' else receiving,
                staged=staged,fitted=fitted,offset=(0.,offset,0.),
                segment='S14' if kind=='bearing' else 'S15')
    return result


class AssemblyRuntime(FrontRuntime):
    """Stock ownership survives faults; motion evidence gates every stock change."""
    def __init__(self,run_id,job='ws001'):
        super().__init__(run_id,job,ASSEMBLY_STAGES)
        self.components=components(job);self.component_state={n:'unreceived' for n in self.components}
        self.stock_capacity={'bearing':2,'ring':2};self.stock_events=[];self.fitted_sides=[]
        self.stock_owner=None;self.stock_last=-math.inf

    def stock_snapshot(self):
        return dict(component_state=dict(self.component_state),owner=self.stock_owner,
            fitted_sides=list(self.fitted_sides),capacity=dict(self.stock_capacity))

    def stock_event(self,event,now,**data):
        if not math.isfinite(now) or now<self.stock_last:raise ValueError('Stock source time order')
        self.stock_last=now
        self.stock_events.append(dict(seq=len(self.stock_events),event=event,time_s=now,
            snapshot=self.stock_snapshot(),**data))

    @staticmethod
    def at(observation,target,now):
        return 0<=now-observation['time_s']<=.15 and math.dist(observation['xyz'],target)<.025 and (
            max(abs(v) for v in observation['velocity'])<.035)

    def receive(self,name,now,observation):
        item=self.components[name]
        if self.component_state[name]!='unreceived' or not self.at(observation,item['receiving'],now):return False
        if sum(self.component_state[n] in {'available','reserved','prepared'} for n,i in self.components.items()
               if i['kind']==item['kind'])>=self.stock_capacity[item['kind']]:return False
        self.component_state[name]='available'
        self.stock_event('STOCK_RECEIVED',now,component=name,observation=observation);return True

    def reserve_stock(self,token,now):
        c=self.commands[token]
        if token!=self.current or c['kind']!='device' or c['station']!='press' or c['status']!='active':return False
        if self.stock_owner is not None:return self.stock_owner==self.job
        missing=[n for n,s in self.component_state.items() if s!='available']
        if missing:
            self.stock_event('MATERIAL_WAIT',now,command_id=token,missing=missing);return False
        self.stock_owner=self.job
        self.component_state={n:'reserved' for n in self.component_state}
        self.stock_event('STOCK_RESERVED',now,command_id=token);return True

    def prepared(self,token,name,now,observation,gauge_observations):
        c=self.commands[token];item=self.components[name]
        if token!=self.current or c['status']!='active' or c['station']!='press' or self.stock_owner!=self.job:return False
        if self.component_state[name]!='reserved' or not self.at(observation,item['staged'],now):return False
        # This is a measured handling/gauge stroke, not a dimensional quality measurement.
        if len(gauge_observations)!=2:return False
        a,b=gauge_observations
        if not self.at(b,item['receiving'],b['time_s']) or not self.at(a,(*item['receiving'][:2],item['receiving'][2]+.12),a['time_s']):return False
        if not a['time_s']<b['time_s']<=now:return False
        self.component_state[name]='prepared'
        self.stock_event('COMPONENT_PREPARED',now,command_id=token,component=name,
            observation=observation,gauge_observations=gauge_observations,
            identity='DEMO-'+name,measurement='assumed handling stroke, no dimensional/contact quality claim')
        return True

    def fitted(self,token,side,now,observations,press_events):
        c=self.commands[token];names=[n for n,i in self.components.items() if i['side']==side]
        if token!=self.current or c['status']!='active' or c['station']!='press' or side in self.fitted_sides:return False
        if any(self.component_state[n]!='prepared' or not self.at(observations[n],self.components[n]['fitted'],now) for n in names):return False
        expected=['PRESS_STARTED','RAM_HOME','RAM_APPROACH','INSERTION_STROKE_MEASURED','STOPPED_HOLD_MEASURED','PRESS_MOTION_COMPLETE']
        if [e['event'] for e in press_events]!=expected:return False
        if any(e['command_id']!=token for e in press_events) or press_events[-1]['time_s']>now:return False
        bearing=next(n for n in names if self.components[n]['kind']=='bearing')
        ring=next(n for n in names if self.components[n]['kind']=='ring')
        if any((e['bearing_id'],e['ring_id'])!=(bearing,ring) for e in press_events):return False
        for n in names:self.component_state[n]='fitted'
        self.fitted_sides.append(side)
        self.stock_event('PAIR_FITTED',now,command_id=token,side=side,observations=observations,press_events=press_events)
        return True

    def complete(self,token,now,observation):
        c=self.commands.get(token)
        if c and c['kind']=='device' and c['station']=='press' and set(self.fitted_sides)!={'left','right'}:return False
        return super().complete(token,now,observation)
