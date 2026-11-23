"""Same-workpiece closing/finishing and finite measured storage handoff."""
import math

from .assembly_flow import ASSEMBLY_STAGES,AssemblyRuntime


STORAGE=(209.75,30.7,1.70)
STORAGE_APPROACH=(209.75,27.,1.70)
STORAGE_UNLOAD=(205.,26.,.95)
STORED_JOB='storage_prior'
PAINT=(179.5,9.3,.95)
FULL_STAGES=ASSEMBLY_STAGES+[
    dict(id='closing',segment='S18',asset='service_r09_closing_transfer_cell_001',target=(158.2,9.5,.95),action='close'),
    dict(id='run_in',segment='S19',asset='finishing_r15_run_in_8_0',target=(173.,9.3,.95),action='run_in'),
    dict(id='painting',segment='S19',asset='spray_portal_001',target=PAINT,action='paint'),
    dict(id='storage_wait',segment='S20',asset=None,target=(206.,27.,.95),action='storage_gate'),
    dict(id='storage',segment='S20',asset='storage_r07_wheelset_storage_bay_001',target=STORAGE,action='store')]


def at(observation,target,now):
    return 0<=now-observation['time_s']<=.15 and math.dist(observation['xyz'],target)<.025 and (
        max(abs(v) for v in observation['velocity'])<.035 and abs(observation['omega'])<.025)


class FiniteStorage:
    """One demonstrator slot; existing contents leave only after measured unload."""
    def __init__(self,run_id):
        self.run_id=run_id;self.occupant=STORED_JOB;self.reserved=None;self.release_token=None
        self.events=[];self.last=-math.inf;self.release_count=0

    def snapshot(self):return dict(capacity=1,occupant=self.occupant,reserved=self.reserved,release_token=self.release_token)

    def event(self,kind,now,**data):
        if not math.isfinite(now) or now<self.last:raise ValueError('Storage source time order')
        self.last=now;self.events.append(dict(seq=len(self.events),event=kind,time_s=now,**data,snapshot=self.snapshot()))

    def reserve(self,job,token,now):
        if self.occupant is not None or self.reserved is not None:
            self.event('STORAGE_BLOCKED',now,job=job,command_id=token);return False
        self.reserved=dict(job=job,command_id=token)
        self.event('STORAGE_RESERVED',now,job=job,command_id=token);return True

    def begin_release(self,now):
        if self.occupant is None or self.release_token is not None:return None
        self.release_count+=1
        self.release_token=self.run_id+':storage_release'+('' if self.release_count==1 else ':'+str(self.release_count))
        self.event('STORAGE_RELEASE_ISSUED',now,job=self.occupant,command_id=self.release_token)
        return self.release_token

    def released(self,token,now,observation):
        if token!=self.release_token or token is None or self.occupant is None:return False
        if not at(observation,STORAGE_UNLOAD,now):return False
        job=self.occupant;self.occupant=None;self.release_token=None
        self.event('STORAGE_RELEASED',now,job=job,command_id=token,observation=observation);return True

    def committed(self,job,token,now,observation):
        if self.occupant is not None or self.reserved!=dict(job=job,command_id=token):return False
        if not at(observation,STORAGE,now):return False
        self.occupant=job;self.reserved=None
        self.event('STORAGE_COMMITTED',now,job=job,command_id=token,observation=observation);return True


class FinishingRuntime(AssemblyRuntime):
    def __init__(self,run_id,job='ws001',*,storage=None):
        super().__init__(run_id,job);self.recipe=FULL_STAGES;self.storage=storage if storage is not None else FiniteStorage(run_id)
        self.finish_evidence={}

    def retrieve_cap(self,token,name,now,observation):
        c=self.commands[token];i=self.items[name]
        if token!=self.current or c['status']!='active' or c['station']!='closing' or i['kind']!='cap':return False
        if self.part_state[name]!='binned' or not at(observation,i['destination'],now):return False
        self.bins['cap_bin'].remove(name);self.part_state[name]='retrieved'
        self.event('CAP_RETRIEVED',now,command_id=token,part=name,observation=observation);return True

    def refit_cap(self,token,name,now,observation):
        c=self.commands[token];i=self.items[name];target=[a+b for a,b in zip(FULL_STAGES[13]['target'],i['offset'])]
        if token!=self.current or c['status']!='active' or c['station']!='closing' or i['kind']!='cap':return False
        if self.part_state[name]!='retrieved' or not at(observation,target,now):return False
        self.part_state[name]='refitted'
        self.event('CAP_REFITTED',now,command_id=token,part=name,observation=observation);return True

    def finish_measured(self,token,now,evidence):
        c=self.commands[token]
        if token!=self.current or c['status']!='active' or c['kind']!='device':return False
        if c['station'] not in {'run_in','painting'}:return False
        if c['station']=='run_in':
            trace=evidence.get('events',[])
            if evidence.get('kind')!='run_in' or [e.get('event') for e in trace]!=[
                'RUN_IN_STARTED','RUN_IN_SPEED_DWELL_MEASURED','RUN_IN_STOPPED']:return False
            if any(e.get('command_id')!=token for e in trace):return False
            if not c['issued_at']<=trace[0]['time_s']<trace[1]['time_s']<trace[2]['time_s']<=now:return False
            if trace[1].get('stable_s',0)<8 or trace[1].get('rotation_rad',0)<4*math.pi or trace[2].get('stopped_s',0)<.25:return False
        else:
            rows=evidence.get('traverse',[])
            if evidence.get('kind')!='paint' or len(rows)!=3:return False
            for row,dx in zip(rows,[-.7,.7,-.7]):
                target=(PAINT[0]+dx,PAINT[1],PAINT[2]+.85)
                if not at(row['tool'],target,row['tool']['time_s']) or not at(row['fixture'],PAINT,row['fixture']['time_s']):return False
                if abs(row['tool']['time_s']-row['fixture']['time_s'])>.1:return False
            if not c['issued_at']<=rows[0]['tool']['time_s']<rows[1]['tool']['time_s']<rows[2]['tool']['time_s']<=now:return False
        self.finish_evidence[c['station']]=evidence
        self.event('FINISH_MEASURED',now,command_id=token,evidence=evidence);return True

    def complete(self,token,now,observation):
        c=self.commands.get(token)
        if c and c['kind']=='device':
            if c['station']=='closing' and any(self.part_state[n]!='refitted' for n,i in self.items.items() if i['kind']=='cap'):return False
            if c['station'] in {'run_in','painting'} and c['station'] not in self.finish_evidence:return False
            if c['station']=='storage_wait' and self.storage.reserved!=dict(job=self.job,command_id=token):return False
            if c['station']=='storage' and self.storage.occupant!=self.job:return False
        return super().complete(token,now,observation)
