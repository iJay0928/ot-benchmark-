"""Audit recorded trajectories without rerunning timed work."""
import hashlib,json,sys
from pathlib import Path

d=Path(sys.argv[1]);manifest=json.loads((d/'manifest.json').read_text())
for name,h in manifest.items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h
summary=json.loads((d/'summary.json').read_text());checks=0
for case in summary['cases']:
    record=json.loads((d/(case['name']+'.json')).read_text())
    runs=record['runs'];assert len(runs)==6
    assert len({r['terminal_sha256'] for r in runs})==1
    reference=runs[0]['trace']
    for run in runs:
        assert run['seconds']>0
        assert [(r['iteration'],r['gap'],r['l1'],r['accepted']) for r in run['trace']]==[(r['iteration'],r['gap'],r['l1'],r['accepted']) for r in reference]
        if run['augmented']:
            for r in run['trace']:
                assert r['observed_gap_distortion']<=r['gap_distortion_bound']+1e-10
                checks+=1
            for label,key in [('l1','l1'),('bound','gap_distortion_bound'),('observed','observed_gap_distortion')]:
                event=next((r for r in run['trace'] if r['accepted'] or r[key]>1e-8),None)
                reported=record['first_events'][label]
                assert (event is None)==(reported is None)
                if event is not None:
                    assert event['iteration']==reported['iteration']
                    assert reported['action']==('accept' if event['accepted'] else 'switch')
base=sum(c['median_seconds']['False'] for c in summary['cases'])
aug=sum(c['median_seconds']['True'] for c in summary['cases'])
result=dict(cases=len(summary['cases']),runs=summary['total_runs'],bound_checks=checks,baseline_sum_medians=base,augmented_sum_medians=aug,overhead_fraction=aug/base-1,passed=True)
(d.parent/(d.name+'_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
