"""Audit short-block ablation, paired quality and unchanged trigger paths."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
d=Path(sys.argv[1])
for name,h in json.loads((d/'manifest.json').read_text()).items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h
data=json.loads((d/'summary.json').read_text());groups={};failures=[]
for case in data['cases']:
    for repeat in range(3):
        pair={r['method']:r['detail'] for r in case['runs'] if r['repeat']==repeat}
        assert pair['guarded']['inspection_trace']==pair['guarded_250']['inspection_trace']
        assert pair['guarded']['reason']==pair['guarded_250']['reason']
        for s in pair['guarded_250']['stages']:
            if s['action']=='warm_stabilized' and s['work'] is not None:
                assert s['work']<=200000 and len(s['history'])<=800
                assert all(h['executed']<=250 for h in s['history'])
    for r in case['runs']:
        q=r['detail']['diagnostics']
        assert r['audit_residual']<=1e-12 and abs(r['audit_objective']-q['rounded_primal'])<=1e-12
        assert r['detail']['accepted']==(-1e-10<=q['numerical_gap']<=1e-8 and q['rounded_residual']<=1e-12)
        if not r['detail']['accepted']:failures.append([case['config']['label'],case['seed'],r['method'],r['repeat']])
    good=[r['detail']['diagnostics'] for r in case['runs'] if r['detail']['accepted']]
    if good:assert max(q['dual'] for q in good)<=min(q['rounded_primal'] for q in good)+1e-10
for group in ['all','movielens','kos','kos_reg_0.001']:
    cases=[c for c in data['cases'] if group=='all' or c['input']['domain']==group or c['config']['label']==group]
    totals={m:sum(c['metrics'][m]['median_seconds'] for c in cases) for m in data['totals']}
    comparisons={}
    for baseline in ['guarded','direct_stabilized_250']:
        paired=[c for c in cases if c['metrics']['guarded_250']['successes']==3 and c['metrics'][baseline]['successes']==3]
        ratios=[c['metrics'][baseline]['median_seconds']/c['metrics']['guarded_250']['median_seconds'] for c in paired]
        comparisons[baseline]=dict(paired=len(paired),faster=sum(r>1 for r in ratios),median_speedup=float(np.median(ratios)) if ratios else None,
            reduction=1-sum(c['metrics']['guarded_250']['median_seconds'] for c in paired)/sum(c['metrics'][baseline]['median_seconds'] for c in paired) if paired else None)
    groups[group]=dict(totals=totals,comparisons=comparisons)
slow=[]
for c in data['cases']:
    ratio=c['metrics']['guarded_250']['median_seconds']/c['metrics']['direct_stabilized_250']['median_seconds']
    slow.append(dict(label=c['config']['label'],seed=c['seed'],time_ratio=ratio))
result=dict(groups=groups,failures=failures,trigger_paths_equal=True,worst=sorted(slow,key=lambda x:x['time_ratio'],reverse=True)[:5])
(d.parent/(d.name+'_shortaudit.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
