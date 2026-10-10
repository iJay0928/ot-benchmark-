"""Compare adaptive inspection with fixed intervals, keeping timing repeats paired."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory
    audit=json.loads((d.parent/(d.name+'_audit.json')).read_text())
    assert audit['manifest_verified']
    data=json.loads((d/'summary.json').read_text());cases=data['cases'];comparisons={}
    for method in ['every_250','every_1000','residual_gate']:
        rows=[]
        for c in cases:
            m=c['metrics']
            if m['adaptive']['successes']==3 and m[method]['successes']==3:
                rows.append(dict(label=c['config']['label'],seed=c['seed'],
                    baseline_over_adaptive=m[method]['median_seconds']/m['adaptive']['median_seconds']))
        comparisons[method]=dict(paired_cases=len(rows),faster_cases=sum(r['baseline_over_adaptive']>1 for r in rows),
            median_speedup=float(np.median([r['baseline_over_adaptive'] for r in rows])) if rows else None,
            relative_total_time_reduction=1-data['totals']['adaptive']['sum_case_median_seconds']/data['totals'][method]['sum_case_median_seconds'],
            worst=min(rows,key=lambda x:x['baseline_over_adaptive']) if rows else None,ratios=rows)
    variable=[]
    for c in cases:
        iterations=[]
        for r in c['runs']:
            if r['method']!='adaptive':continue
            trace=r['detail']['inspection_trace'];iterations.append(r['detail']['ordinary_iterations'])
            assert all(t['next_interval']%50==0 and 50<=t['next_interval']<=2000 for t in trace)
            assert all(x['iteration']<y['iteration'] for x,y in zip(trace,trace[1:]))
            assert all(not t['accepted'] or -1e-10<=t['gap']<=1e-8 for t in trace)
        if len(set(iterations))>1:variable.append(dict(label=c['config']['label'],seed=c['seed'],iterations=iterations))
    result=dict(totals=data['totals'],comparisons=comparisons,timing_dependent_iterations=variable)
    (d.parent/(d.name+'_comparisons.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(totals=result['totals'],comparisons={m:{k:v for k,v in r.items() if k!='ratios'} for m,r in comparisons.items()},timing_dependent_cases=len(variable)),indent=2))


if __name__=='__main__':main()
