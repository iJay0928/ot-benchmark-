"""Audit active support stress tests and distinguish threshold sensitivity from timing noise."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory
    for name,h in json.loads((d/'manifest.json').read_text()).items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h
    protocol=json.loads((d/'protocol.json').read_text())
    data=json.loads((d/'summary.json').read_text());cases=data['cases'];methods=list(data['totals']);failures=[];different=[];groups={};triggers=[]
    for c in cases:
        if 'input' in c:
            source=Path(protocol['input_directory'])
            assert hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()==protocol['input_manifest_sha256']
            assert hashlib.sha256((source/c['input']['file']).read_bytes()).hexdigest()==c['input']['sha256']
        else:
            n=c['config']['n'];rng=np.random.default_rng(c['seed']);x=rng.uniform(size=(n,8));y=rng.uniform(size=(n,8));cost=cdist(x,y,metric='sqeuclidean');cost/=cost.max()
            a=rng.uniform(.5,1.5,n);b=rng.uniform(.5,1.5,n);a/=a.sum();b/=b.sum()
            assert hashlib.sha256(a.tobytes()+b.tobytes()+cost.tobytes()).hexdigest()==c['input_sha256']
        good=[];signatures={}
        for r in c['runs']:
            detail=r['detail'];q=detail['diagnostics']
            assert detail['ordinary_iterations']<=50000 and len(detail['stages'])<=2
            for stage in detail['stages']:
                if stage['work'] is not None:
                    assert stage['work']<=(1000 if stage['action']=='cold_epsilon' else (250000 if r['method']=='direct_stabilized_250' else 200000))
            assert r['audit_residual']<=1e-12 and abs(r['audit_objective']-q['rounded_primal'])<=1e-12
            assert detail['accepted']==(-1e-10<=q['numerical_gap']<=1e-8 and q['rounded_residual']<=1e-12)
            if detail['accepted']:good.append(q)
            else:failures.append([c['config']['label'],c['seed'],r['method'],r['repeat']])
            if detail['reason']=='representation discrepancy':
                threshold=float(r['method'].split('_')[1]);t=detail['inspection_trace'][-1]
                assert t['discrepancy']>threshold and not t['accepted']
                if r['repeat']==0:triggers.append(dict(label=c['config']['label'],seed=c['seed'],method=r['method'],iteration=t['iteration'],discrepancy=t['discrepancy']))
            if r['repeat']==0 and r['method'].startswith('guard_'):
                signatures[r['method']]=[detail['ordinary_iterations'],detail['reason'],[(s['action'],s['work']) for s in detail['stages']]]
        if good:assert max(x['dual'] for x in good)<=min(x['rounded_primal'] for x in good)+1e-10
        if len({json.dumps(x) for x in signatures.values()})>1:different.append(dict(label=c['config']['label'],seed=c['seed'],paths=signatures))
    for label in ['all']+list(dict.fromkeys(c['config']['label'] for c in cases)):
        subset=[c for c in cases if label=='all' or c['config']['label']==label];g={}
        for m in methods:
            paired=[c for c in subset if c['metrics'][m]['successes']==3 and c['metrics']['direct_stabilized_250']['successes']==3]
            ratios=[c['metrics']['direct_stabilized_250']['median_seconds']/c['metrics'][m]['median_seconds'] for c in paired]
            g[m]=dict(successes=sum(c['metrics'][m]['successes'] for c in subset),runs=3*len(subset),sum_medians=sum(c['metrics'][m]['median_seconds'] for c in subset),
                paired_cases=len(paired),median_speedup=float(np.median(ratios)) if ratios else None,faster_cases=sum(x>1 for x in ratios),
                paired_time_reduction=1-sum(c['metrics'][m]['median_seconds'] for c in paired)/sum(c['metrics']['direct_stabilized_250']['median_seconds'] for c in paired) if paired else None)
        groups[label]=g
    output=dict(manifest_verified=True,input_replay_verified=True,groups=groups,failures=failures,threshold_different_paths=different,triggers=triggers,
        note='exploratory threshold sweep; source role defined in protocol; no selected-threshold holdout; repeated timings and shared inputs are dependent')
    (d.parent/(d.name+'_audit.json')).write_text(json.dumps(output,indent=2),encoding='utf-8');print(json.dumps(output,indent=2))


if __name__=='__main__':main()
