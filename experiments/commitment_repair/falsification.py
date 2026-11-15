"""Falsify optimality of earliest-proposal expansion on an exact cascade.

Exploratory deterministic counterexample only. All methods retain the same
constraints and lexicographic objective; no production or novelty claim.
"""
from dataclasses import asdict
from pathlib import Path
import argparse
import hashlib
import json

from counterexamples import Action, conflict_expansion, oracle, task_neighborhood, valid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    actions = [Action('a', 'A', ('road', 'robot_A', 'material_A'), 0, 2, due=3),
               Action('b', 'B', ('road', 'robot_B', 'material_B'), 2, 2, due=5),
               Action('c', 'C', ('road', 'robot_C', 'material_C'), 4, 2, due=7)]
    release = {'a': 1}; seeds = {'a'}; horizon = 9
    assert valid(actions, {a.name: a.start for a in actions}, {})
    closure, history = conflict_expansion(actions, release, seeds)
    answers = {name: oracle(actions, release, neighborhood, horizon)
               for name, neighborhood in dict(task=task_neighborhood(actions, seeds),
                   earliest_expansion=closure, full={a.name for a in actions}).items()}
    for answer in answers.values():
        assert answer['feasible'] and valid(actions, answer['starts'], release)
    checks = dict(outside_c_missed='c' not in closure,
        restricted_answer_is_globally_legal=valid(actions, answers['earliest_expansion']['starts'], release),
        restricted_loses_to_full=answers['earliest_expansion']['objective'] > answers['full']['objective'],
        global_zero_tardiness=answers['full']['objective'][0] == 0,
        full_requires_c_to_move=answers['full']['starts']['c'] != 4)
    assert all(checks.values())
    inputs = dict(actions=[asdict(a) for a in actions], release=release, seeds=sorted(seeds), horizon=horizon)
    (args.output/'inputs.json').write_text(json.dumps(inputs, indent=2)+'\n')
    result = dict(answers=answers, expansion_history=history, checks=checks, passed=True,
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                       for name in ('falsification.py', 'counterexamples.py')},
        inputs_sha256=hashlib.sha256((args.output/'inputs.json').read_bytes()).hexdigest(),
        seed=None, seed_reason='Explicit deterministic counterexample; no sampling.',
        conclusion='Earliest-proposal conflict closure is not sufficient for full-oracle optimality. '
            'A legal restricted repair can miss a better propagation into an initially nonoverlapping task.',
        scope='Diagnostic falsification of this candidate rule only, not all dependency repairs, '
            'production integration, statistical efficacy, industrial calibration or novelty.')
    (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
