"""Audit complete guarded policies, timing on paired successes and bounded rescue."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory
    for name,h in json.loads((d/'manifest.json').read_text()).items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h
    protocol=json.loads((d/'protocol.json').read_text())
    assert hashlib.sha256((Path(protocol['input_directory'])/'manifest.json').read_bytes()).hexdigest()==protocol['input_manifest_sha256']
    data=json.loads((d/'summary.json').read_text());cases=data['cases'];failures=[];triggers=[];groups={}
    for c in cases:
        good=[]
        for r in c['runs']:
            detail=r['detail'];q=detail['diagnostics']
            assert r['audit_residual']<=1e-12 and abs(r['audit_objective']-q['rounded_primal'])<=1e-12
            assert detail['accepted']==(-1e-10<=q['numerical_gap']<=1e-8 and q['rounded_residual']<=1e-12)
            assert detail['ordinary_iterations']<=50000 and len(detail['stages'])<=2
            for stage in detail['stages']:
                if stage['work'] is not None:assert stage['work']<=(1000 if stage['action']=='cold_epsilon' else 200000)
            if detail['reason']=='representation discrepancy':
                t=detail['inspection_trace'][-1];assert t['discrepancy']>1e-8 and not t['accepted']
                if r['repeat']==0:triggers.append(dict(domain=c['input']['domain'],seed=c['seed'],reg=c['config']['reg'],iteration=t['iteration'],discrepancy=t['discrepancy']))
            if detail['accepted']:good.append(q)
            else:failures.append([c['config']['label'],c['seed'],r['method'],r['repeat']])
        if good:assert max(x['dual'] for x in good)<=min(x['rounded_primal'] for x in good)+1e-10
    for key in ['all','movielens','kos']+list(dict.fromkeys(c['config']['label'] for c in cases)):
        subset=[c for c in cases if key=='all' or key==c['input']['domain'] or key==c['config']['label']]
        totals={m:dict(successes=sum(c['metrics'][m]['successes'] for c in subset),runs=len(subset)*3,
            sum_medians=sum(c['metrics'][m]['median_seconds'] for c in subset),
            rescue_runs=sum(r['detail']['used_rescue'] for c in subset for r in c['runs'] if r['method']==m),
            epsilon_runs=sum(r['detail']['used_epsilon'] for c in subset for r in c['runs'] if r['method']==m)) for m in data['totals']}
        comparisons={}
        for base in ['warm_budget','cold_budget']:
            paired=[c for c in subset if c['metrics'][base]['successes']==3 and c['metrics']['guarded']['successes']==3]
            ratios=[c['metrics'][base]['median_seconds']/c['metrics']['guarded']['median_seconds'] for c in paired]
            comparisons[base]=dict(paired_cases=len(paired),faster_cases=sum(x>1 for x in ratios),median_speedup=float(np.median(ratios)) if ratios else None,
                paired_time_reduction=1-sum(c['metrics']['guarded']['median_seconds'] for c in paired)/sum(c['metrics'][base]['median_seconds'] for c in paired) if paired else None,
                worst_speedup=min(ratios) if ratios else None)
        groups[key]=dict(totals=totals,comparisons=comparisons)
    result=dict(manifest_verified=True,groups=groups,failures=failures,trigger_cases=triggers,
        warning_runs=sum(bool(r['warnings']) for c in cases for r in c['runs']),
        note='complete solver timings; shared preprocessing excluded; six pairs per corpus reused across regularizations; no statistical significance claim')
    (d.parent/(d.name+'_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
