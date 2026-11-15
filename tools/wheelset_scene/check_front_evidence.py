"""Causal negative checks with recomputed hashes, preserving original evidence."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .build import ROOT
from .verify_front_workshop import verify


def check(source,output):
    source=Path(source);results=[]
    mutations=['wrong_parent','duplicate_bin','moving_deposit','early_separation','missing_tag',
        'native_sequence_duplicate','native_force_disagreement']
    source_result=json.loads((source/'result.json').read_text())
    if source_result['case']!='normal':mutations.append('lost_partial_bin')
    if source_result.get('through') in {'assembly','storage'}:
        mutations += ['moving_stock_receipt','duplicate_stock_fit','wrong_press_ring','short_press_stroke',
            'missing_stopped_hold','press_force_disagreement','moving_fitted_component']
    if source_result.get('through')=='storage':
        mutations += ['moving_storage_release','duplicate_storage_commit','lost_cap_identity','short_run_in_dwell','short_spray_traverse']
    for mutation in mutations:
        with tempfile.TemporaryDirectory(prefix='front-causal-',dir=ROOT/'out') as temporary:
            assert Path(temporary).resolve().is_relative_to((ROOT/'out').resolve())
            directory=Path(temporary)/'evidence'
            # Unchanged large image/sample files are hard-linked only for reads.
            # Always unlink a candidate file before writing to protect the source.
            shutil.copytree(source,directory,copy_function=os.link)
            report=json.loads((directory/'result.json').read_text())
            events=[json.loads(line) for line in (directory/'events.jsonl').read_text().splitlines()]
            filename='events.jsonl';edited=events
            more={}
            if mutation=='wrong_parent':
                next(e for e in events if e['event']=='COMMAND_ISSUED')['command']['job']='ws999'
            elif mutation=='duplicate_bin':
                i=next(i for i,e in enumerate(events) if e['event']=='PART_BIN_COMMITTED')
                events.insert(i+1,copy.deepcopy(events[i]))
                for seq,e in enumerate(events):e['seq']=seq
            elif mutation=='moving_deposit':
                next(e for e in events if e['event']=='PART_BIN_COMMITTED')['observation']['velocity'][1]=.2
            elif mutation=='early_separation':
                e=next(e for e in events if e['event']=='PART_SEPARATED');e['observation']['xyz'][1]+=.4
            elif mutation=='missing_tag':
                next(e for e in events if e['event']=='SIMULATED_TAG_READ')['tag']='DEMO-ws999'
            elif mutation=='lost_partial_bin':
                next(e for e in events if e['event']=='RECOVERY_REISSUED')['snapshot']['bins']['bearing_bin']=[]
            elif mutation=='native_sequence_duplicate':
                filename='native_commands.jsonl';edited=[json.loads(line) for line in (directory/filename).read_text().splitlines()]
                first=edited[0];next(c for c in edited if c['entity']==first['entity'] and c['sequence']!=first['sequence'])['sequence']=first['sequence']
            elif mutation=='native_force_disagreement':
                filename='efforts.jsonl';edited=[json.loads(line) for line in (directory/filename).read_text().splitlines()]
                edited[0]['effort']+=1.
            elif mutation=='press_force_disagreement':
                filename='press_samples.jsonl';edited=[json.loads(line) for line in (directory/filename).read_text().splitlines()]
                edited[0]['requested_effort_n']+=1.
            elif mutation in {'moving_storage_release','duplicate_storage_commit'}:
                filename='storage_events.jsonl';edited=[json.loads(line) for line in (directory/filename).read_text().splitlines()]
                if mutation=='moving_storage_release':
                    next(e for e in edited if e['event']=='STORAGE_RELEASED')['observation']['velocity'][0]=.2
                else:
                    i=next(i for i,e in enumerate(edited) if e['event']=='STORAGE_COMMITTED')
                    edited.insert(i+1,copy.deepcopy(edited[i]))
                    for seq,e in enumerate(edited):e['seq']=seq
            elif mutation=='lost_cap_identity':
                next(e for e in edited if e['event']=='CAP_REFITTED')['part']='ws001_cap_right'
            elif mutation=='short_run_in_dwell':
                e=next(e for e in edited if e['event']=='FINISH_MEASURED' and e['evidence']['kind']=='run_in')
                next(a for a in e['evidence']['events'] if a['event']=='RUN_IN_SPEED_DWELL_MEASURED')['stable_s']=1.
                more['device_events.jsonl']=[json.loads(line) for line in (directory/'device_events.jsonl').read_text().splitlines()]
                next(a for a in more['device_events.jsonl'] if a['event']=='RUN_IN_SPEED_DWELL_MEASURED')['stable_s']=1.
            elif mutation=='short_spray_traverse':
                e=next(e for e in edited if e['event']=='FINISH_MEASURED' and e['evidence']['kind']=='paint')
                e['evidence']['traverse'][1]['tool']['xyz'][0]-=.5
            else:
                filename='stock_events.jsonl';edited=[json.loads(line) for line in (directory/filename).read_text().splitlines()]
                if mutation=='moving_stock_receipt':
                    next(e for e in edited if e['event']=='STOCK_RECEIVED')['observation']['velocity'][0]=.2
                elif mutation=='duplicate_stock_fit':
                    i=next(i for i,e in enumerate(edited) if e['event']=='PAIR_FITTED');edited.insert(i+1,copy.deepcopy(edited[i]))
                    for seq,e in enumerate(edited):e['seq']=seq
                elif mutation=='moving_fitted_component':
                    next(e for e in edited if e['event']=='PAIR_FITTED')['observations']['ws001_new_bearing_left']['velocity'][0]=.2
                else:
                    actions=[json.loads(line) for line in (directory/'device_events.jsonl').read_text().splitlines()]
                    commit=next(e for e in edited if e['event']=='PAIR_FITTED')
                    token=commit['command_id'];bearing='ws001_new_bearing_left'
                    trace=[e for e in actions if e.get('command_id')==token and e.get('bearing_id')==bearing]
                    for candidate in [trace,commit['press_events']]:
                        if mutation=='wrong_press_ring':
                            for e in candidate:e['ring_id']='ws001_new_ring_right'
                        elif mutation=='short_press_stroke':
                            for e in candidate:
                                if 'stroke_m' in e:e['stroke_m']=.1
                        elif mutation=='missing_stopped_hold':
                            hold=next(e for e in candidate if e['event']=='STOPPED_HOLD_MEASURED')
                            candidate.remove(hold)
                    if mutation=='missing_stopped_hold':
                        actions=[e for e in actions if not(e.get('command_id')==token and e.get('bearing_id')==bearing and e['event']=='STOPPED_HOLD_MEASURED')]
                    more['device_events.jsonl']=actions
            path=directory/filename;path.unlink();path.write_text(''.join(json.dumps(e)+'\n' for e in edited))
            report['data_sha256'][filename]=hashlib.sha256(path.read_bytes()).hexdigest()
            for name,values in more.items():
                path=directory/name;path.unlink();path.write_text(''.join(json.dumps(e)+'\n' for e in values))
                report['data_sha256'][name]=hashlib.sha256(path.read_bytes()).hexdigest()
            path=directory/'result.json';path.unlink();path.write_text(json.dumps(report)+'\n')
            if (directory/'audit.json').exists():(directory/'audit.json').unlink()
            try:verify(directory)
            except AssertionError:results.append(dict(mutation=mutation,rejected=True,hashes_recomputed=True))
            else:raise AssertionError('Causal corruption accepted: '+mutation)
    result=dict(passed=True,source=source.name,cases=results)
    Path(output).write_text(json.dumps(result,indent=2)+'\n');return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();print(json.dumps(check(args.directory,args.output)))


if __name__=='__main__':main()
