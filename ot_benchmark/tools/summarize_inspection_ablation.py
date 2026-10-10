"""Report paired ablation evidence without declaring the fastest noisy run a winner."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory
    audit=json.loads((d.parent/(d.name+'_audit.json')).read_text());assert audit['manifest_verified']
    data=json.loads((d/'summary.json').read_text());cases=data['cases'];result={}
    for method in ['adaptive','adaptive_gap','adaptive_cost']:
        comparisons={}
        for base in ['adaptive','every_250','every_1000']:
            if base==method:continue
            rows=[dict(label=c['config']['label'],seed=c['seed'],speedup=c['metrics'][base]['median_seconds']/c['metrics'][method]['median_seconds'])
                for c in cases if c['metrics'][base]['successes']==3 and c['metrics'][method]['successes']==3]
            comparisons[base]=dict(paired_cases=len(rows),faster_cases=sum(r['speedup']>1 for r in rows),
                median_speedup=float(np.median([r['speedup'] for r in rows])) if rows else None,
                relative_total_time_reduction=1-data['totals'][method]['sum_case_median_seconds']/data['totals'][base]['sum_case_median_seconds'],
                worst=min(rows,key=lambda r:r['speedup']) if rows else None)
        varied=0
        for c in cases:
            selected=[r['detail'] for r in c['runs'] if r['method']==method]
            if len({r['ordinary_iterations'] for r in selected})>1:varied+=1
            for r in selected:
                assert all(50<=t['next_interval']<=2000 and t['next_interval']%50==0 for t in r['inspection_trace'])
        result[method]=dict(comparisons=comparisons,timing_dependent_cases=varied)
    output=dict(totals=data['totals'],ablation=result)
    (d.parent/(d.name+'_comparisons.json')).write_text(json.dumps(output,indent=2),encoding='utf-8')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
