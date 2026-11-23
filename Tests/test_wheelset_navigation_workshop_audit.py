from copy import deepcopy

import pytest

from Tests.test_wheelset_navigation_dock import receipt
from tools.wheelset_scene.verify_navigation_workshop import audit_dock_guard


@pytest.mark.parametrize('corruption', [None, 'invented_guard', 'other_robot', 'stale_raw', 'changed_header'])
def test_receiving_guard_requires_both_adapter_and_independent_raw_stream(corruption):
    command, _, proof = receipt()
    proof['guard'].update(command_age=.01, lease_age=.02)
    query = dict(robot='scout_a', command=command, wall_s=5.)
    audit = dict(arrival_guards={command['route_id']: deepcopy(proof['guard'])})
    raw = [dict(kind='guard', robot='scout_a', received_wall_s=4.99, message=deepcopy(proof['guard']))]
    if corruption == 'invented_guard': proof['guard']['guard_id'] = 'invented'
    if corruption == 'other_robot': raw[0]['robot'] = 'scout_b'
    if corruption == 'stale_raw': raw[0]['received_wall_s'] = 4.
    if corruption == 'changed_header': proof['guard']['header']['stamp']['nanosec'] = 1
    if corruption:
        with pytest.raises(AssertionError): audit_dock_guard(query, proof, raw, audit)
    else:
        audit_dock_guard(query, proof, raw, audit)
