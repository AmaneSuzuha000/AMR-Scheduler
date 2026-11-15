"""Derived two-workpiece world; frozen visual assets are left intact."""
import copy
import json
from pathlib import Path
import xml.etree.ElementTree as ET

from .assembly_flow import components
from .build import save_xml
from .finishing_world import build_finishing_world,finishing_clearance
from .front_flow import ORIGIN,parts
from .front_world import add_body,swept_hits


SUPPLY_HANDOFFS={'magnetic':(35.45,53.,1.),'derust':(21.,53.,1.)}


def worker_geometry(job,index):
    items=parts(job);stock=components(job)
    for item in items.values():
        p=item['destination'];item['destination']=(p[0]+index*.70*(-1 if item['kind']=='cap' else 1),p[1],p[2])
    # The second set waits at height in independent identified staging places;
    # only the current press owner may move into shared receiving/gauge poses.
    if index:
        for item in stock.values():
            p=item['receiving'];item['origin']=(p[0]+index*2.,p[1],6.)
    return dict(parts=items,components=stock)


def build_whole_line_world(output,jobs=('ws001','ws002')):
    if len(jobs)!=2 or jobs[0]!='ws001' or len(set(jobs))!=2:raise ValueError('current derived world requires two distinct workpieces starting with ws001')
    output=Path(output).resolve();meta=build_finishing_world(output);tree=ET.parse(output/'front.world');world=tree.getroot().find('world')
    origins={job:(ORIGIN if i==0 else (7.,9.,.8)) for i,job in enumerate(jobs)}
    geometry={job:worker_geometry(job,i) for i,job in enumerate(jobs)}
    job=jobs[1];origin=origins[job]
    meta['definitions'][job]=add_body(world,job,origin,'wheelset',output/'bare_wheelset.dae')
    for name,item in geometry[job]['parts'].items():
        p=tuple(a+b for a,b in zip(origin,item['offset']))
        meta['definitions'][name]=add_body(world,name,p,item['kind'])
    for name,item in geometry[job]['components'].items():
        meta['definitions'][name]=add_body(world,name,item['origin'],item['kind'])
    meta['definitions']['line_supply_tote']=add_body(world,'line_supply_tote',(6.,69.2,.7),'supply_tote')
    meta.update(jobs=list(jobs),job_origins=origins,worker_geometry=geometry)
    meta['assumptions'].update(multi_job='Two independent measured wheelset rigs, one shared transfer owner and single-capacity stages. Staging at x7 then x10 admission is an assumed external boundary',
        shared_bins='Two pairs per shared part bin; second cap pair shifted -.70m X, second old-bearing pair +.70m X. Finite bin reservations gate opening/demount',
        replacement_staging='Second replacement set waits at z6m, 2m X offset, before shared receiving/gauge poses',
        consumable_handoffs='Assumed equipment-side tote destinations at y53m, outside enclosed hood roofs; receiving there commits station consumables, without modelling internal uptake',
        output='Shared single storage slot. Each outgoing body must reach unload and stop before identity/occupancy release; adapter must explicitly retire outgoing bodies before reusing unload space')
    save_xml(tree.getroot(),output/'front.world');return meta


def whole_line_clearance(meta):
    result=finishing_clearance(meta);checks=result['checks'];boxes=meta['static_obstacles']
    def path(stage,name,points,kind='part'):
        for a,b in zip(points,points[1:]):
            checks.append(dict(stage=stage,entity=name,start=a,end=b,hits=swept_hits(a,b,boxes,kind)))
    # The main recipe path is identical for both jobs after admission. Check
    # differing staging and bins explicitly rather than accepting a duplicate
    # recipe as evidence for their additional geometry.
    for job in meta['jobs'][1:]:
        origin=meta['job_origins'][job]
        path('admission',job,[origin,ORIGIN],'wheelset')
        config=meta['worker_geometry'][job]
        for name,item in config['parts'].items():
            path('admission',name,[[a+b for a,b in zip(p,item['offset'])] for p in [origin,ORIGIN]])
            target=next(s['target'] for s in meta['stages'] if s['id']==('opening' if item['kind']=='cap' else 'demount'))
            released=[a+b for a,b in zip(target,item['offset'])];released[1]+=item['sign']*item['separation_m']
            dest=item['destination']
            path('bin_deposit',name,[released,(*released[:2],6.),(*dest[:2],6.),dest])
            if item['kind']=='cap':
                close=next(s['target'] for s in meta['stages'] if s['id']=='closing')
                fitted=[a+b for a,b in zip(close,item['offset'])];stage=list(fitted);stage[1]+=item['sign']*.35
                path('cap_retrieval',name,[dest,(*dest[:2],6.),(*stage[:2],6.),stage,fitted])
        for name,item in config['components'].items():
            path('replacement_receiving',name,[item['origin'],item['receiving']])
    from .supply_bridge import SOURCES
    current=meta['definitions']['line_supply_tote']['origin']
    for task,station in [('inspection_consumables','magnetic'),('small_tools','derust')]:
        source=SOURCES[task];dest=SUPPLY_HANDOFFS[station]
        path('consumable_handoff','line_supply_tote',[current,(*current[:2],6.),(*source[:2],6.),source,
            (*source[:2],6.),(*dest[:2],6.),dest])
        current=dest
    result['passed']=all(not c['hits'] for c in checks)
    result['boundary']+='; second-job staging/slot differences included; dynamic-pair/carrier/tool sweeps remain separate'
    return result
