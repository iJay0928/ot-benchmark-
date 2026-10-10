"""Audit fixed versus discrepancy trigger experiments and paired timings."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
d=Path(sys.argv[1]);data=json.loads((d/'summary.json').read_text());protocol=json.loads((d/'protocol.json').read_text())
for name,h in json.loads((d/'manifest.json').read_text()).items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h
assert hashlib.sha256((Path(protocol['input_directory'])/'manifest.json').read_bytes()).hexdigest()==protocol['input_manifest_sha256']
failures=[];safety=[];budget=[]
for c in data['cases']:
    for r in c['runs']:
        q=r['detail'];diag=q['diagnostics'];m=r['method']
        assert r['audit_residual']<=1e-12 and abs(r['audit_objective']-diag['rounded_primal'])<=1e-12
        assert q['accepted']==(-1e-10<=diag['numerical_gap']<=1e-8 and diag['rounded_residual']<=1e-12)
        if not q['accepted']:failures.append([c['seed'],c['config']['label'],m,r['repeat']])
        if m.startswith('fixed_'):
            point=int(m.split('_')[1]);assert q['ordinary_iterations']<=point
            if q['reason']=='fixed switch':assert q['ordinary_iterations']==point
            elif q['reason']!='quality accepted' and r['repeat']==0:safety.append([c['seed'],c['config']['label'],m,q['reason'],q['ordinary_iterations']])
        if m=='guarded_250' and q['reason']=='ordinary budget' and r['repeat']==0:budget.append([c['seed'],c['config']['label']])
        for stage in q['stages']:
            if stage['work'] is not None:assert stage['work']<=(1000 if stage['action']=='cold_epsilon' else (250000 if m=='direct_stabilized_250' else 200000))
            if stage['action'] in ('warm_stabilized','cold_stabilized'):assert all(h['executed']<=250 for h in stage['history'])
    good=[r['detail']['diagnostics'] for r in c['runs'] if r['detail']['accepted']]
    if good:assert max(q['dual'] for q in good)<=min(q['rounded_primal'] for q in good)+1e-10
groups={}
for name in ['all','movielens','kos','kos_reg_0.001']:
    cs=[c for c in data['cases'] if name=='all' or name==c['input']['domain'] or name==c['config']['label']]
    totals={m:sum(c['metrics'][m]['median_seconds'] for c in cs) for m in data['totals']};comparisons={}
    for m in data['totals']:
        if m=='guarded_250':continue
        ps=[c for c in cs if c['metrics'][m]['successes']==3 and c['metrics']['guarded_250']['successes']==3]
        comparisons[m]=dict(paired=len(ps),faster=sum(c['metrics']['guarded_250']['median_seconds']<c['metrics'][m]['median_seconds'] for c in ps),reduction=1-sum(c['metrics']['guarded_250']['median_seconds'] for c in ps)/sum(c['metrics'][m]['median_seconds'] for c in ps) if ps else None)
    groups[name]=dict(totals=totals,comparisons=comparisons)
out=dict(failures=failures,safety_early_switches=safety,guarded_budget_exhaustion=budget,groups=groups)
(d.parent/(d.name+'_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
