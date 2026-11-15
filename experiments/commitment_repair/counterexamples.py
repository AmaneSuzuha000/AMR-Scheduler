"""Exact toy counterexamples for commitment-repair design, not benchmark claims.

All neighborhoods use the same integer-time oracle, global capacity validator,
fixed ownership, precedence, release constraints and objective. No production
dispatcher or learned policy is changed by this exploratory script.
"""
from dataclasses import asdict, dataclass
from itertools import combinations, product
from pathlib import Path
import argparse
import hashlib
import json


@dataclass(frozen=True)
class Action:
    name: str
    task: str
    resources: tuple[str, ...]
    start: int
    duration: int
    predecessors: tuple[str, ...] = ()
    due: int | None = None
    locked: bool = False


def overlap(a, b, starts):
    return max(starts[a.name], starts[b.name]) < min(
        starts[a.name]+a.duration, starts[b.name]+b.duration)


def valid(actions, starts, release):
    by_name = {a.name: a for a in actions}
    if set(starts) != set(by_name):
        return False
    for a in actions:
        s = starts[a.name]
        if type(s) is not int or s < max(a.start, release.get(a.name, a.start)):
            return False
        if a.locked and s != a.start:
            return False
        if any(starts[p]+by_name[p].duration > s for p in a.predecessors):
            return False
    return all(not (set(a.resources) & set(b.resources) and overlap(a, b, starts))
               for a, b in combinations(actions, 2))


def objective(actions, starts):
    return (sum(max(0, starts[a.name]+a.duration-a.due) for a in actions if a.due is not None),
            sum(abs(starts[a.name]-a.start) for a in actions),
            max(starts[a.name]+a.duration for a in actions))


def oracle(actions, release, neighborhood, horizon):
    variables = [a for a in actions if a.name in neighborhood and not a.locked]
    domains = [range(max(a.start, release.get(a.name, a.start)), horizon-a.duration+1)
               for a in variables]
    best = None
    examined = 0
    for times in product(*domains):
        starts = {a.name: a.start for a in actions}
        starts.update((a.name, t) for a, t in zip(variables, times))
        examined += 1
        if not valid(actions, starts, release):
            continue
        score = objective(actions, starts)
        key = (*score, tuple(starts[a.name] for a in actions))
        if best is None or key < best[0]:
            best = (key, starts)
    return dict(feasible=best is not None, objective=list(best[0][:3]) if best else None,
                starts=best[1] if best else None, assignments_examined=examined,
                neighborhood=sorted(neighborhood))


def task_neighborhood(actions, seeds):
    tasks = {a.task for a in actions if a.name in seeds}
    return {a.name for a in actions if a.task in tasks}


def conflict_expansion(actions, release, seeds):
    """Explore conflicts of an earliest precedence repair; no optimality claim."""
    affected = task_neighborhood(actions, seeds)
    by_name = {a.name: a for a in actions}
    history = []
    while True:
        starts = {a.name: max(a.start, release.get(a.name, a.start))
                  if a.name in affected and not a.locked else a.start for a in actions}
        for _ in actions:
            for a in actions:
                if a.name in affected and not a.locked:
                    starts[a.name] = max(starts[a.name], max(
                        (starts[p]+by_name[p].duration for p in a.predecessors), default=0))
        outside = set()
        for a, b in combinations(actions, 2):
            if (a.name in affected) == (b.name in affected):
                continue
            if set(a.resources) & set(b.resources) and overlap(a, b, starts):
                outside.add(b.name if a.name in affected else a.name)
        for a in actions:
            if a.name not in affected and any(
                    p in affected and starts[p]+by_name[p].duration > starts[a.name]
                    for p in a.predecessors):
                outside.add(a.name)
        history.append(dict(affected=sorted(affected), proposed=starts,
                            conflicts_outside=sorted(outside)))
        expanded = affected | task_neighborhood(actions, outside)
        if expanded == affected:
            return affected, history
        affected = expanded


def shared_resource_component(actions, seeds):
    affected = task_neighborhood(actions, seeds)
    while True:
        resources = {r for a in actions if a.name in affected for r in a.resources}
        expanded = affected | task_neighborhood(actions,
            {a.name for a in actions if resources & set(a.resources)})
        if expanded == affected:
            return affected
        affected = expanded


def cases():
    return [
        dict(name='new_road_conflict_after_delay', horizon=8, seeds=['a'], release={'a': 2},
             actions=[Action('a', 'A', ('road', 'robot_A', 'material_A'), 0, 2, due=4),
                      Action('b', 'B', ('road', 'robot_B', 'material_B'), 3, 2, due=6)]),
        dict(name='precedence_reaches_finite_buffer', horizon=10, seeds=['a_road'], release={'a_road': 4},
             actions=[Action('loaded_A', 'A', ('robot_A', 'material_A'), 0, 1, locked=True),
                      Action('a_road', 'A', ('road', 'robot_A', 'material_A'), 1, 1, ('loaded_A',)),
                      Action('a_dock', 'A', ('dock', 'robot_A', 'material_A'), 2, 1, ('a_road',)),
                      Action('a_buffer', 'A', ('buffer', 'material_A'), 3, 2, ('a_dock',), due=8),
                      Action('b_buffer', 'B', ('buffer', 'material_B'), 6, 2, due=10)]),
        dict(name='shared_resource_does_not_imply_affected', horizon=15, seeds=['a'], release={'a': 1},
             actions=[Action('a', 'A', ('road', 'material_A'), 0, 2, due=4),
                      Action('b', 'B', ('road', 'material_B'), 8, 2, due=12),
                      Action('c', 'C', ('road', 'material_C'), 12, 2, due=15)]),
        dict(name='immutable_action_cannot_be_retimed', horizon=6, seeds=['fixed'], release={'fixed': 1},
             actions=[Action('fixed', 'A', ('road', 'material_A'), 0, 2, due=4, locked=True),
                      Action('b', 'B', ('road', 'material_B'), 3, 2, due=6)]),
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    inputs = []
    for case in cases():
        actions, release = case['actions'], case['release']
        nominal = {a.name: a.start for a in actions}
        assert valid(actions, nominal, {})
        task = task_neighborhood(actions, set(case['seeds']))
        closure, history = conflict_expansion(actions, release, set(case['seeds']))
        full = {a.name for a in actions}
        row = dict(case=case['name'], nominal_legal=True,
            task_only=oracle(actions, release, task, case['horizon']),
            conflict_expansion=oracle(actions, release, closure, case['horizon']),
            full=oracle(actions, release, full, case['horizon']),
            shared_resource_component=sorted(shared_resource_component(actions, set(case['seeds']))),
            expansion_history=history)
        for method in ('task_only', 'conflict_expansion', 'full'):
            answer = row[method]
            if answer['feasible']:
                assert valid(actions, answer['starts'], release)
                assert tuple(answer['objective']) == objective(actions, answer['starts'])
        inputs.append({**case, 'actions': [asdict(a) for a in actions]})
        results.append(row)
    first, buffer, sparse, locked = results
    checks = dict(
        cross_resource_changes_outcome=first['task_only']['objective'][0] > first['full']['objective'][0],
        same_oracle_matches_full_in_constructed_road_case=first['conflict_expansion']['objective'] == first['full']['objective'],
        buffer_dependency_discovered='b_buffer' in buffer['conflict_expansion']['neighborhood'],
        task_only_is_legal_but_worse=buffer['task_only']['feasible'] and buffer['task_only']['objective'] > buffer['full']['objective'],
        resource_component_can_overexpand=len(sparse['shared_resource_component']) > len(sparse['conflict_expansion']['neighborhood']),
        closure_is_not_minimum_feasibility_modification=first['task_only']['feasible'] and len(first['task_only']['neighborhood']) < len(first['conflict_expansion']['neighborhood']),
        immutable_commitment_preserved=all(not locked[m]['feasible'] for m in ('task_only','conflict_expansion','full')))
    assert all(checks.values()), checks
    raw=json.dumps(inputs,sort_keys=True,separators=(',',':')).encode()
    (args.output/'inputs.json').write_bytes(raw+b'\n')
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    summary=dict(checks=checks,passed=all(checks.values()),cases=len(results),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_sha256=hashlib.sha256(raw+b'\n').hexdigest(),seed=None,
        seed_reason='Four explicit deterministic toy cases; no random generation or training.',
        scope='Exploratory model counterexamples and exact enumeration only; not production integration, industrial calibration, novelty, statistical benefit or RESEARCH_VALIDATED.',
        limitations=['Integer time and unit resource capacities only.',
            'Fixed material ownership is an input restriction; no physical transfer model.',
            'Earliest-proposal expansion is a candidate generator, not a proven safe or minimal repair neighborhood.',
            'Global legality oracle is mandatory; no claim that closure matches full replanning in general.',
            'Objective and four crafted cases are diagnostic, not the preregistered research benchmark.'])
    (args.output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
