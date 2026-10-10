"""Audit fixed inspection intervals across sizes, retaining per-case regressions."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory
    for name,expected in json.loads((d/'manifest.json').read_text()).items():
        assert hashlib.sha256((d/name).read_bytes()).hexdigest()==expected
    data=json.loads((d/'summary.json').read_text());cases=data['cases'];methods=list(data['totals']);failures=[]
    for c in cases:
        good=[]
        for r in c['runs']:
            q=r['detail']['diagnostics']
            assert abs(r['audit_objective']-q['rounded_primal'])<=1e-12
            assert r['audit_residual']<=1e-12
            assert r['detail']['accepted']==(-1e-10<=q['numerical_gap']<=1e-8 and q['rounded_residual']<=1e-12)
            if r['detail']['accepted']:good.append(q)
            else:failures.append([c['config']['label'],c['seed'],r['method'],r['repeat']])
        if good:assert max(x['dual'] for x in good)<=min(x['rounded_primal'] for x in good)+1e-10
    groups={}
    for label in dict.fromkeys(c['config']['label'] for c in cases):
        subset=[c for c in cases if c['config']['label']==label];groups[label]={}
        for m in methods:
            paired=[c for c in subset if c['metrics'][m]['successes']==3 and c['metrics']['residual_gate']['successes']==3]
            ratios=[c['metrics']['residual_gate']['median_seconds']/c['metrics'][m]['median_seconds'] for c in paired]
            groups[label][m]=dict(successes=sum(c['metrics'][m]['successes'] for c in subset),
                sum_medians=sum(c['metrics'][m]['median_seconds'] for c in subset),paired_cases=len(paired),
                median_speedup=float(np.median(ratios)) if ratios else None,
                faster_cases=sum(x>1 for x in ratios),
                quality_checks=[r['detail']['quality_checks'] for c in subset for r in c['runs'] if r['method']==m and r['repeat']==0],
                ordinary_iterations=[r['detail']['ordinary_iterations'] for c in subset for r in c['runs'] if r['method']==m and r['repeat']==0])
    regressions={m:[dict(label=c['config']['label'],seed=c['seed'],slowdown=c['metrics'][m]['median_seconds']/c['metrics']['residual_gate']['median_seconds'])
        for c in cases if c['metrics'][m]['median_seconds']>c['metrics']['residual_gate']['median_seconds']] for m in methods if m!='residual_gate'}
    result=dict(manifest_verified=True,cases=len(cases),runs=sum(len(c['runs']) for c in cases),
        failures=failures,totals=data['totals'],groups=groups,regressions=regressions,
        note='Exploratory, shared seeds across scales/reg; repeated timing not independent samples; no significance or selected-policy holdout claim')
    (d.parent/(d.name+'_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(totals=data['totals'],failures=failures,groups=groups),indent=2))


if __name__=='__main__':main()
