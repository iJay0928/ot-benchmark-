"""Audit epsilon-scaling experiments without relabeling failures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(directory):
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    for name,digest in read(directory/'manifest.json').items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest()==digest,name
    protocol=read(directory/'protocol.json');s=read(directory/'summary.json')
    methods=protocol['methods'];cases=s['cases'];counts={};warnings={m:0 for m in methods}
    assert len({(r['config']['label'],r['seed']) for r in cases})==len(cases)
    if protocol['mode']=='new seeds same generator':
        assert len(cases)==len(protocol['configs'])*len(protocol['seeds'])
        assert set(protocol['seeds']).isdisjoint([92001,92002,92003,94001,94002,94003])
    else:assert len(cases)==4
    for r in cases:
        raw=read(directory/f'{r["config"]["label"]}_{r["seed"]}.json')
        assert raw['case']==r
        expected={(m,k) for m in methods for k in range(3)}
        assert len(raw['runs'])==len(expected)
        assert {(v['method'],v['repeat']) for v in raw['runs']}==expected
        for v in raw['runs']:
            residual=v.get('detail',{}).get('residual')
            assert v['success']==(residual is not None and residual<=1e-9)
            warnings[v['method']]+=bool(v['warnings'])
        for m in methods:
            runs=[v for v in raw['runs'] if v['method']==m]
            assert sum(v['success'] for v in runs)==r['metrics'][m]['successes']
            assert float(np.median([v['seconds'] for v in runs]))==r['metrics'][m]['median_seconds']
    for m in methods:
        matched=[r for r in cases if r['metrics'][m]['successes']==3 and r['metrics']['epsilon_scaling']['successes']==3]
        ratios=[r['metrics']['epsilon_scaling']['median_seconds']/r['metrics'][m]['median_seconds'] for r in matched]
        counts[m]=dict(successful_runs=sum(r['metrics'][m]['successes'] for r in cases),runs=len(cases)*3,
            successful_cases=sum(r['metrics'][m]['successes']==3 for r in cases),
            sum_case_median_seconds=sum(r['metrics'][m]['median_seconds'] for r in cases),
            warning_runs=warnings[m],paired_cases=len(matched),
            epsilon_over_method_time_median=float(np.median(ratios)) if ratios else None)
    return dict(cases=len(cases),runs=len(cases)*len(methods)*3,manifest_verified=True,methods=counts,
        failed_cases=[dict(config=r['config']['label'],seed=r['seed'],methods=[m for m in methods if r['metrics'][m]['successes']<3]) for r in cases if any(r['metrics'][m]['successes']<3 for m in methods)])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.input)
    a.output.write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2))
