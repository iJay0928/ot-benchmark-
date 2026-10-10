"""Audit same-state action experiments, including failures and finite budgets."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory
    for name,h in json.loads((d/'manifest.json').read_text()).items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h
    cases=json.loads((d/'summary.json').read_text())['cases'];rows=[]
    for c in cases:
        good=[]
        for r in c['runs']:
            detail=r['detail'];diag=detail['diagnostics']
            assert r['snapshot_sha256']==c['snapshot_sha256']
            assert r['audit_residual']<=1e-12 and abs(r['audit_objective']-diag['rounded_primal'])<=1e-12
            assert detail['accepted']==(-1e-10<=diag['numerical_gap']<=1e-8 and diag['rounded_residual']<=1e-12)
            if r['action']=='cold_epsilon':assert len(detail['history'])<=1 and detail['additional_work']<=1000
            else:assert len(detail['history'])<=40 and detail['additional_work']<=200000
            if detail['accepted']:good.append(diag)
        if good:assert max(x['dual'] for x in good)<=min(x['rounded_primal'] for x in good)+1e-10
        rows.append(dict(domain=c['input']['domain'],seed=c['input']['seed'],prefix_gap=c['prefix_diagnostics']['numerical_gap'],
            metrics=c['metrics'],first_repeat={r['action']:dict(work=r['detail']['additional_work'],blocks=len(r['detail']['history']),gap=r['detail']['diagnostics']['numerical_gap']) for r in c['runs'] if r['repeat']==0}))
    actions=list(cases[0]['metrics']);totals={m:dict(successes=sum(c['metrics'][m]['successes'] for c in cases),runs=3*len(cases),
        sum_medians=sum(c['metrics'][m]['median_seconds'] for c in cases)) for m in actions}
    output=dict(manifest_verified=True,cases=len(cases),runs=sum(len(c['runs']) for c in cases),totals=totals,rows=rows,
        warning_runs=sum(bool(r['warnings']) for c in cases for r in c['runs']),
        note='development action timings; four selected previously observed fallback states, no holdout or full-policy claim')
    (d.parent/(d.name+'_audit.json')).write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
