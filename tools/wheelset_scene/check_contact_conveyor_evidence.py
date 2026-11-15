"""Check semantic rejection on rehashed copies of a real conveyor run."""
import argparse
import json
import os
from pathlib import Path
import shutil
import tempfile

from .verify_contact_conveyor import audit, digest


MUTATIONS = ('missing_rail_contacts','zero_support_force','wrong_queried_entity',
             'wheelset_never_translates','pose_jump','missing_stop_phase')


def check(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    assert not output.exists()
    baseline = audit(source); original = {name:digest(source/name) for name in ('records.jsonl','result.json')}
    cases = []
    for mutation in MUTATIONS:
        with tempfile.TemporaryDirectory(prefix='conveyor-causal-') as temporary:
            target = Path(temporary)/'case'
            shutil.copytree(source,target,copy_function=os.link)
            replacement = target/'records.replacement.jsonl'; changes = 0
            with replacement.open('x') as stream, (source/'records.jsonl').open() as raw:
                for line in raw:
                    row = json.loads(line); kind = row['kind']; changed = False
                    if mutation == 'missing_rail_contacts' and kind in ('tread_left_contacts','tread_right_contacts'):
                        changes += 1; continue
                    if mutation == 'zero_support_force' and kind in ('tread_left_contacts','tread_right_contacts'):
                        for contact in row['message']['states']:
                            if 'support_rail' in contact['collision2_name']:
                                contact['total_wrench']['force'] = dict(x=0.,y=0.,z=0.); changed = True
                    if mutation == 'wrong_queried_entity' and kind in ('wheelset_state_request','wheelset_state_reply'):
                        row['query']['request']['name'] = 'powered_conveyor'; changed = True
                    if mutation == 'wheelset_never_translates' and kind == 'wheelset_state_reply':
                        row['response']['state']['pose']['position']['x'] = 0.; changed = True
                    if mutation == 'pose_jump' and kind == 'wheelset_state_reply' and 10. < row['clock_s'] < 10.3:
                        row['response']['state']['pose']['position']['x'] = 1.95; changed = True
                    if mutation == 'missing_stop_phase' and kind == 'phase' and row['value'] == 'stopped':
                        changes += 1; continue
                    changes += int(changed)
                    stream.write(json.dumps(row)+'\n' if changed else line)
            assert changes > 0
            replacement.replace(target/'records.jsonl')
            result = json.loads((source/'result.json').read_text())
            result['files_sha256']['records.jsonl'] = digest(target/'records.jsonl')
            alternate = target/'result.replacement.json'
            alternate.write_text(json.dumps(result,indent=2)+'\n'); alternate.replace(target/'result.json')
            try:
                audit(target)
            except AssertionError as exc:
                cases.append(dict(mutation=mutation,rejected=True,hash_recomputed=True,changes=changes,reason=str(exc)))
            else:
                raise AssertionError('Accepted semantic corruption: '+mutation)
        assert all(digest(source/name) == value for name,value in original.items())
    report = dict(passed=True,scope='isolated_conveyor_semantic_evidence_rejection',baseline=baseline,cases=cases)
    with output.open('x') as stream: stream.write(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('directory',type=Path)
    parser.add_argument('--output',type=Path,required=True); args = parser.parse_args()
    print(json.dumps(check(args.directory,args.output)))


if __name__ == '__main__': main()
