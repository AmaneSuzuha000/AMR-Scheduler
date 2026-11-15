"""Rehash causal corruptions of completed whole-line evidence in isolated copies."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .build import ROOT
from .verify_whole_line import verify


def check(source,output):
    source=Path(source)
    baseline=verify(source)
    mutations=['wrong_worker_binding','early_resource_release','lost_source_occupancy',
        'missing_tag','moving_part_deposit','duplicate_part_deposit','wrong_cap_identity',
        'duplicate_stock_fit','wrong_press_ring','short_press_stroke','press_force_disagreement',
        'short_run_in_dwell','short_spray_traverse','moving_storage_release','duplicate_storage_commit',
        'moving_dock_receipt','wrong_supply_quantity','shortcut_tote','native_execution_mismatch','native_sequence_duplicate']
    results=[]
    for mutation in mutations:
        with tempfile.TemporaryDirectory(prefix='whole-line-causal-',dir=ROOT/'out') as temporary:
            directory=Path(temporary)/'evidence'
            shutil.copytree(source,directory,copy_function=os.link)
            top=json.loads((directory/'result.json').read_text()); process=directory/'process'
            child=json.loads((process/'result.json').read_text()); edited={}
            def records(name):
                if name not in edited:
                    edited[name]=[json.loads(line) for line in (directory/name).read_text().splitlines()]
                return edited[name]
            if mutation in {'wrong_worker_binding','early_resource_release','lost_source_occupancy'}:
                events=records('process/events.jsonl')
                if mutation=='wrong_worker_binding':
                    issued=[e for e in events if e['event']=='COMMAND_ISSUED']
                    first=issued[0]; other=next(e for e in issued if e['command']['job']!=first['command']['job'])
                    first['command']['execution_id']=other['command']['execution_id']
                elif mutation=='early_resource_release':
                    e=next(e for e in events if e['event']=='ARRIVAL_CONFIRMED'); e['observed_at']-=1.
                else:
                    e=next(e for e in events if e['event']=='COMMAND_ISSUED')
                    e['snapshot']['stock'][e['command']['source']]=[]
            elif mutation in {'missing_tag','moving_part_deposit','duplicate_part_deposit','wrong_cap_identity','short_run_in_dwell','short_spray_traverse'}:
                events=records('process/worker_events.jsonl')
                if mutation=='missing_tag': next(e for e in events if e['event']=='SIMULATED_TAG_READ')['tag']='DEMO-wrong'
                elif mutation=='moving_part_deposit': next(e for e in events if e['event']=='PART_BIN_COMMITTED')['observation']['velocity'][1]=.2
                elif mutation=='duplicate_part_deposit':
                    i=next(i for i,e in enumerate(events) if e['event']=='PART_BIN_COMMITTED'); events.insert(i+1,copy.deepcopy(events[i]))
                elif mutation=='wrong_cap_identity':
                    e=next(e for e in events if e['event']=='CAP_REFITTED'); e['part']=e['job']+'_cap_right'
                elif mutation=='short_run_in_dwell':
                    e=next(e for e in events if e['event']=='FINISH_MEASURED' and e['evidence']['kind']=='run_in')
                    next(a for a in e['evidence']['events'] if a['event']=='RUN_IN_SPEED_DWELL_MEASURED')['stable_s']=1.
                    next(a for a in records('process/device_events.jsonl') if a['command_id']==e['command_id'] and a['event']=='RUN_IN_SPEED_DWELL_MEASURED')['stable_s']=1.
                else:
                    e=next(e for e in events if e['event']=='FINISH_MEASURED' and e['evidence']['kind']=='paint')
                    e['evidence']['traverse'][1]['tool']['xyz'][0]-=.5
                counters={}
                for e in events:
                    e['seq']=counters.get(e['job'],0); counters[e['job']]=e['seq']+1
            elif mutation in {'duplicate_stock_fit','wrong_press_ring','short_press_stroke'}:
                stock=records('process/stock_events.jsonl')
                i=next(i for i,e in enumerate(stock) if e['event']=='PAIR_FITTED'); commit=stock[i]
                if mutation=='duplicate_stock_fit':
                    stock.insert(i+1,copy.deepcopy(commit))
                    counters={}
                    for e in stock:
                        e['seq']=counters.get(e['job'],0); counters[e['job']]=e['seq']+1
                else:
                    trace=[e for e in records('process/device_events.jsonl') if e.get('command_id')==commit['command_id'] and e.get('bearing_id')==commit['job']+'_new_bearing_left']
                    for group in [trace,commit['press_events']]:
                        for e in group:
                            if mutation=='wrong_press_ring': e['ring_id']=commit['job']+'_new_ring_right'
                            elif 'stroke_m' in e: e['stroke_m']=.1
            elif mutation=='press_force_disagreement': records('process/press_samples.jsonl')[0]['requested_effort_n']+=1.
            elif mutation in {'moving_storage_release','duplicate_storage_commit'}:
                events=records('process/storage_events.jsonl')
                if mutation=='moving_storage_release': next(e for e in events if e['event']=='STORAGE_RELEASED')['observation']['velocity'][0]=.2
                else:
                    i=next(i for i,e in enumerate(events) if e['event']=='STORAGE_COMMITTED'); events.insert(i+1,copy.deepcopy(events[i]))
                    for seq,e in enumerate(events): e['seq']=seq
            elif mutation in {'moving_dock_receipt','wrong_supply_quantity'}:
                events=records('supply_events.jsonl')
                if mutation=='moving_dock_receipt': next(e for e in events if e['event']=='DOCK_RECEIPT')['observation']['linear_speed']=.2
                else: next(e for e in events if e['event']=='SUPPLY_COMMITTED')['quantity']+=1
            elif mutation=='shortcut_tote':
                events=records('process/tote_events.jsonl'); next(e for e in events if e['waypoint']==2)['target'][0]+=1.
            else:
                # Rewrite the large stream line by line, never into a hard link.
                original=process/'native_commands.jsonl'; replacement=process/'native_commands.edited'
                with original.open() as stream,replacement.open('w') as target:
                    for index,line in enumerate(stream):
                        if index==(0 if mutation=='native_execution_mismatch' else 23):
                            e=json.loads(line)
                            if mutation=='native_execution_mismatch': e['execution_id']='wrong-worker'
                            else: e['sequence']=1
                            line=json.dumps(e)+'\n'
                        target.write(line)
                original.unlink(); replacement.rename(original)
                child['data_sha256'][original.name]=hashlib.sha256(original.read_bytes()).hexdigest()
            for name,values in edited.items():
                path=directory/name; path.unlink(); path.write_text(''.join(json.dumps(e)+'\n' for e in values))
                target=child if name.startswith('process/') else top
                target['data_sha256'][Path(name).name]=hashlib.sha256(path.read_bytes()).hexdigest()
            path=process/'result.json'; path.unlink(); path.write_text(json.dumps(child)+'\n')
            top['process_report']=child
            path=directory/'result.json'; path.unlink(); path.write_text(json.dumps(top)+'\n')
            try:
                verify(directory,fleet=False,sweep=False,write=False)
            except AssertionError:
                results.append(dict(mutation=mutation,rejected=True,hashes_recomputed=True))
            else:
                raise AssertionError('Causal corruption accepted: '+mutation)
    result=dict(passed=True,source=source.name,baseline_audit=baseline,cases=results)
    Path(output).write_text(json.dumps(result,indent=2)+'\n'); return result


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('directory',type=Path)
    p.add_argument('--output',type=Path,required=True); args=p.parse_args()
    print(json.dumps(check(args.directory,args.output)))


if __name__=='__main__': main()
