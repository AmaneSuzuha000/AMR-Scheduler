"""Post-hoc mechanism stress only; NOT part of frozen primary comparison.

The confirmed primary matrix showed zero additional benefit from production
switching because the shared crew resource was never simultaneously demanded.
This separate, explicitly labelled construct increases P1 processing duration
to create crew contention and tests whether qualified alternative choices can
actually change an end-to-end outcome.
"""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from tools.research.system12_dynamic_matrix import one

def stress():
    methods=('STATIC_ONE','EVENT_FLEET','EVENT_JOINT','PRECOMMITTED_FLEX')
    rows=[one(40,method,p1_duration=140) for method in methods]
    assert all(r['finished'] for r in rows)
    assert all(r['collision_events']==0 and not r['node_capacity_violations'] for r in rows)
    return dict(schema='system12-posthoc-production-contention-v1',
       scientific_status='SUPPLEMENTARY_MECHANISM_STRESS_NOT_CONFIRMATORY',
       change_from_primary='P1 processing duration 50->140 tick, planned after primary results',
       source='single same synthetic network and event, no industry metric',
       episodes=rows,
       makespan_by_method={r['policy']:r['makespan_step'] for r in rows},
       statement='No general or independent superiority inference is allowed')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();result=stress();args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(result['makespan_by_method']))
