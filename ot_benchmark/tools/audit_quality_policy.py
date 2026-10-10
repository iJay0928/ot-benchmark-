"""Audit saved policy runs and report paired timings without treating repeats as samples."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory
    manifest=json.loads((d/'manifest.json').read_text())
    for name,expected in manifest.items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==expected
    data=json.loads((d/'summary.json').read_text());cases=data['cases'];ratios=[];failures=[];max_spread=0
    for case in cases:
        good=[]
        for row in case['runs']:
            detail=row['detail'];diag=detail['diagnostics']
            assert abs(row['audit_objective']-diag['rounded_primal'])<=1e-12
            assert row['audit_residual']<=1e-12
            assert detail['accepted']==(-1e-10<=diag['numerical_gap']<=1e-8 and diag['rounded_residual']<=1e-12)
            if detail['accepted']:good.append(diag)
            else:failures.append([case['config']['label'],case['seed'],row['method'],row['repeat']])
        if good:
            # Every accepted method brackets the same objective, up to floating point allowance.
            assert max(r['dual'] for r in good)<=min(r['rounded_primal'] for r in good)+1e-10
            max_spread=max(max_spread,max(r['rounded_primal'] for r in good)-min(r['rounded_primal'] for r in good))
        m=case['metrics']
        if m['periodic']['successes']==3 and m['residual_gate']['successes']==3:
            ratios.append(dict(label=case['config']['label'],seed=case['seed'],
                baseline_over_periodic=m['residual_gate']['median_seconds']/m['periodic']['median_seconds']))
    audit=dict(manifest_verified=True,cases=len(cases),runs=sum(len(c['runs']) for c in cases),
        totals=data['totals'],failures=failures,max_accepted_objective_spread=max_spread,
        paired_cases=len(ratios),paired_speedup_median=float(np.median([r['baseline_over_periodic'] for r in ratios])) if ratios else None,
        faster_cases=sum(r['baseline_over_periodic']>1 for r in ratios),paired_ratios=ratios,
        note='descriptive timing, three repeats per case; shared seeds are dependent; no statistical significance claim')
    out=d.parent/(d.name+'_audit.json');out.write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k!='paired_ratios'},indent=2))


if __name__=='__main__':main()
