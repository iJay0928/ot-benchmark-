"""Quality, prefix identity, cold-path equivalence and subgroup ablation."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
d=Path(sys.argv[1]);data=json.loads((d/'summary.json').read_text())
for name,h in json.loads((d/'manifest.json').read_text()).items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h
protocol=json.loads((d/'protocol.json').read_text())
assert hashlib.sha256((Path(protocol['input_directory'])/'manifest.json').read_bytes()).hexdigest()==protocol['input_manifest_sha256']
failures=[];switch=[]
for c in data['cases']:
    for repeat in range(3):
        rs={r['method']:r for r in c['runs'] if r['repeat']==repeat}
        w,k,b=[rs[m]['detail'] for m in ['guarded_250','guarded_cold_250','direct_stabilized_250']]
        assert w['inspection_trace']==k['inspection_trace'] and w['reason']==k['reason']
        if w['stages'] and w['stages'][0]['action']=='warm_stabilized':
            assert k['stages'][0]['action']=='cold_stabilized'
            if k['stages'][0]['accepted'] and b['stages'][0]['work']<=200000:
                assert k['stages'][0]['history']==b['stages'][0]['history']
            if repeat==0:switch.append(dict(label=c['config']['label'],seed=c['seed'],prefix=w['ordinary_iterations'],warm_work=w['stages'][0]['work'],cold_work=k['stages'][0]['work']))
        for method,r in rs.items():
            q=r['detail']['diagnostics']
            assert r['audit_residual']<=1e-12 and abs(r['audit_objective']-q['rounded_primal'])<=1e-12
            assert r['detail']['accepted']==(-1e-10<=q['numerical_gap']<=1e-8 and q['rounded_residual']<=1e-12)
            if not r['detail']['accepted']:failures.append([c['config']['label'],c['seed'],method,repeat])
            for s in r['detail']['stages']:
                if s['work'] is not None:
                    assert s['work']<=(1000 if s['action']=='cold_epsilon' else (250000 if method=='direct_stabilized_250' else 200000))
                if s['action'] in ('warm_stabilized','cold_stabilized'):
                    assert all(h['executed']<=250 for h in s['history'])
        good=[r['detail']['diagnostics'] for r in rs.values() if r['detail']['accepted']]
        if good:assert max(q['dual'] for q in good)<=min(q['rounded_primal'] for q in good)+1e-10
ids={(r['label'],r['seed']) for r in switch};groups={}
for label in ['all','switched','not_switched','kos_reg_0.001']:
    cases=[c for c in data['cases'] if label=='all' or (label=='switched' and (c['config']['label'],c['seed']) in ids) or (label=='not_switched' and (c['config']['label'],c['seed']) not in ids) or c['config']['label']==label]
    totals={m:sum(c['metrics'][m]['median_seconds'] for c in cases) for m in data['totals']}
    comparisons={}
    for base in ['guarded_cold_250','direct_stabilized_250']:
        paired=[c for c in cases if c['metrics']['guarded_250']['successes']==3 and c['metrics'][base]['successes']==3]
        ratios=[c['metrics'][base]['median_seconds']/c['metrics']['guarded_250']['median_seconds'] for c in paired]
        comparisons[base]=dict(paired=len(paired),faster=sum(x>1 for x in ratios),median_speedup=float(np.median(ratios)) if ratios else None,reduction=1-sum(c['metrics']['guarded_250']['median_seconds'] for c in paired)/sum(c['metrics'][base]['median_seconds'] for c in paired) if paired else None)
    groups[label]=dict(totals=totals,comparisons=comparisons)
result=dict(failures=failures,switches=switch,groups=groups,prefix_equal=True,cold_path_equal_when_budget_allows=True)
(d.parent/(d.name+'_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result,indent=2))
