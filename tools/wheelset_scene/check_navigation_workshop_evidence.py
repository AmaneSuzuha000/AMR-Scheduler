"""Rehash cross-component corruptions of a completed same-world workshop run.

Every case starts from an independently accepted full run. Copies share unchanged
files, while replacements never write through a hard link into the source.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .build import ROOT
from .navigation_records import Records
from .verify_navigation_workshop import verify


MUTATIONS = ('wrong_receiving_robot', 'reused_receiving_query', 'wrong_arrival_task',
    'moving_receiving_guard', 'missing_independent_guard', 'failed_state_response',
    'missing_cargo_handoff', 'missing_material_charge', 'duplicate_dock_receipt',
    'wrong_delivery_quantity')


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def replace_json(path, value):
    replacement = path.with_name(path.name+'.causal-replacement')
    with replacement.open('x', encoding='utf-8') as stream:
        json.dump(value, stream); stream.write('\n')
    replacement.replace(path)


def rewrite_rows(path, transform):
    """Stream unchanged sensor rows verbatim and replace only selected records."""
    replacement = path.with_name(path.name+'.causal-replacement'); changed = 0
    with path.open(encoding='utf-8') as source, replacement.open('x', encoding='utf-8') as target:
        for line in source:
            row = json.loads(line); revised = transform(deepcopy(row))
            if revised != row:
                changed += 1
                if revised is not None:
                    target.write(json.dumps(revised)+'\n')
            else:
                target.write(line)
    assert changed, 'mutation did not change any evidence: '+str(path)
    replacement.replace(path)
    return changed


def mutate(directory, mutation):
    directory = Path(directory); assert mutation in MUTATIONS
    top = json.loads((directory/'result.json').read_text())
    receipts = list(Records(directory/'observations.jsonl').only('localized_dock_receipt'))
    assert len(receipts) == 2 and all(r['accepted'] for r in receipts)
    selected = receipts[1] if mutation == 'reused_receiving_query' else receipts[0]
    task = selected['task_id']; robot = selected['robot']; proof = selected['proof']
    touched = {}

    def corrupt_proof(value):
        if mutation == 'wrong_receiving_robot':
            value['state_query']['request']['name'] = 'scout_b' if robot == 'scout_a' else 'scout_a'
        elif mutation == 'reused_receiving_query':
            value['state_query']['query_id'] = receipts[0]['proof']['state_query']['query_id']
        elif mutation == 'wrong_arrival_task': value['feedback']['task_id'] = 'unrelated_task'
        elif mutation == 'moving_receiving_guard': value['guard']['actual_speed'] = .1
        elif mutation == 'failed_state_response' and 'state_reply' in value:
            value['state_reply']['success'] = False

    proof_mutations = {'wrong_receiving_robot', 'reused_receiving_query', 'wrong_arrival_task',
                       'moving_receiving_guard', 'failed_state_response'}
    if mutation in proof_mutations or mutation == 'missing_independent_guard':
        def observations(row):
            if mutation == 'missing_independent_guard':
                if row['kind'] == 'guard' and row.get('robot') == robot and row['message'] == proof['guard']:
                    return None
            elif row['kind'] in ('dock_state_query', 'localized_dock_receipt') and row['task_id'] == task:
                corrupt_proof(row['proof'])
            return row
        touched['observations.jsonl'] = rewrite_rows(directory/'observations.jsonl', observations)

    if mutation != 'missing_independent_guard':
        events = list(Records(directory/'supply_events.jsonl'))
        original = deepcopy(events)
        if mutation in proof_mutations:
            for event in events:
                if event['event'] == 'DOCK_RECEIPT' and event['delivery']['task_id'] == task:
                    corrupt_proof(event['dock_proof'])
        elif mutation == 'missing_cargo_handoff':
            index = next(i for i, e in enumerate(events) if e['event'] == 'HANDOFF_ISSUED')
            del events[index]
        elif mutation == 'missing_material_charge':
            index = next(i for i, e in enumerate(events) if e['event'] == 'MATERIAL_CONSUMED')
            del events[index]
        elif mutation == 'duplicate_dock_receipt':
            index = next(i for i, e in enumerate(events) if e['event'] == 'DOCK_RECEIPT')
            events.insert(index+1, deepcopy(events[index]))
        elif mutation == 'wrong_delivery_quantity':
            next(e for e in events if e['event'] == 'DOCK_RECEIPT')['delivery']['quantity'] += 1
        for seq, event in enumerate(events): event['seq'] = seq
        if events != original:
            path = directory/'supply_events.jsonl'; replacement = path.with_name(path.name+'.causal-replacement')
            with replacement.open('x', encoding='utf-8') as stream:
                for event in events: stream.write(json.dumps(event)+'\n')
            replacement.replace(path); touched['supply_events.jsonl'] = 1
    assert touched
    for name in touched:
        assert name in top['files_sha256']
        top['files_sha256'][name] = digest(directory/name)
    replace_json(directory/'result.json', top)
    return touched


def check(source, output, *, recovery=False):
    source = Path(source); output = Path(output)
    assert not output.exists(), 'refusing to overwrite a prior causal result'
    baseline = verify(source, write=False, recovery=recovery)
    cases = []; original_result = digest(source/'result.json')
    for mutation in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix='navigation-workshop-causal-', dir=ROOT/'out') as temporary:
            directory = Path(temporary)/'evidence'
            shutil.copytree(source, directory, copy_function=os.link)
            touched = mutate(directory, mutation)
            top = json.loads((directory/'result.json').read_text())
            assert all(digest(directory/name) == top['files_sha256'][name] for name in touched)
            try:
                verify(directory, write=False, recovery=recovery)
            except AssertionError as error:
                cases.append(dict(mutation=mutation, rejected=True, hashes_recomputed=True,
                                  changed_files=touched, rejection=str(error)))
            else:
                raise AssertionError('Cross-component corruption accepted: '+mutation)
    assert digest(source/'result.json') == original_result
    result = dict(passed=True, scope='full_same_world_cross_component_causal_rejection',
                  recovery=recovery, source=source.name, baseline_audit=baseline, cases=cases)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2); stream.write('\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--recovery', action='store_true'); args = parser.parse_args()
    print(json.dumps(check(args.directory, args.output, recovery=args.recovery)))


if __name__ == '__main__': main()
