"""Evidence isolation checks, not completed workshop acceptance."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil

import pytest

from Tests.test_wheelset_navigation_dock import receipt
from tools.wheelset_scene.check_navigation_workshop_evidence import digest, mutate, rewrite_rows


def fixture(directory):
    directory.mkdir(); rows = []; events = []
    for index, robot in enumerate(('scout_a', 'scout_b')):
        command, observation, proof = receipt()
        command['robot_id'] = robot; command['task_id'] = 'delivery_'+str(index)
        proof['state_query']['query_id'] = index
        task = command['task_id']
        rows.extend([dict(kind='dock_state_query', robot=robot, task_id=task,
                          proof={k: deepcopy(v) for k, v in proof.items() if k != 'state_reply'}),
                     dict(kind='localized_dock_receipt', robot=robot, task_id=task,
                          proof=deepcopy(proof), accepted=True),
                     dict(kind='guard', robot=robot, message=deepcopy(proof['guard']))])
        events.append(dict(event='DOCK_RECEIPT', delivery=dict(task_id=task, quantity=2), dock_proof=deepcopy(proof)))
    events.extend([dict(event='HANDOFF_ISSUED'), dict(event='MATERIAL_CONSUMED')])
    for seq, event in enumerate(events): event['seq'] = seq
    files = {}
    for name, values in (('observations.jsonl', rows), ('supply_events.jsonl', events)):
        (directory/name).write_text(''.join(json.dumps(row)+'\n' for row in values), encoding='utf-8')
        files[name] = digest(directory/name)
    (directory/'result.json').write_text(json.dumps(dict(files_sha256=files)), encoding='utf-8')


@pytest.mark.parametrize('mutation', ['wrong_receiving_robot', 'missing_independent_guard',
                                    'reused_receiving_query', 'missing_material_charge'])
def test_rehashed_cross_component_edits_preserve_original_hard_linked_evidence(tmp_path, mutation):
    source = tmp_path/'original'; fixture(source)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    target = tmp_path/'copy'; shutil.copytree(source, target, copy_function=os.link)
    changed = mutate(target, mutation)
    report = json.loads((target/'result.json').read_text())
    assert all(digest(target/name) == report['files_sha256'][name] for name in changed)
    assert all((source/name).read_bytes() == raw for name, raw in before.items())
    assert all((target/name).read_bytes() != before[name] for name in changed)


def test_streaming_mutation_keeps_unselected_sensor_record_bytes(tmp_path):
    path = tmp_path/'observations.jsonl'
    retained = '{ "kind" : "scan", "ranges" : [1.0, null, 2.0] }\n'
    path.write_text(retained+'{"kind":"guard"}\n', encoding='utf-8')
    count = rewrite_rows(path, lambda row: None if row['kind'] == 'guard' else row)
    assert count == 1 and path.read_text(encoding='utf-8') == retained
