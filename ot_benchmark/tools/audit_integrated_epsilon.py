"""Audit identical pre-fallback paths and quality after integrated stopping."""
import hashlib,json,sys
from pathlib import Path
d=Path(sys.argv[1]);data=json.loads((d/'summary.json').read_text());protocol=json.loads((d/'protocol.json').read_text())
for name,h in json.loads((d/'manifest.json').read_text()).items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h
assert hashlib.sha256((Path(protocol['input_directory'])/'manifest.json').read_bytes()).hexdigest()==protocol['input_manifest_sha256']
failures=[];fallback=[]
for c in data['cases']:
    for repeat in range(3):
        rs={r['method']:r for r in c['runs'] if r['repeat']==repeat};old,new=[rs[m]['detail'] for m in ['legacy_fallback','quality_fallback']]
        assert old['inspection_trace']==new['inspection_trace'] and old['reason']==new['reason']
        assert old['used_epsilon']==new['used_epsilon']
        assert [s for s in old['stages'] if s['action']!='cold_epsilon']==[s for s in new['stages'] if s['action']!='cold_epsilon']
        if not new['used_epsilon']:assert old['diagnostics']==new['diagnostics']
        elif repeat==0:
            fallback.append(dict(seed=c['seed'],label=c['config']['label'],old_outer=old['stages'][-1]['work'],new_outer=new['stages'][-1]['work'],old_gap=old['diagnostics']['numerical_gap'],new_gap=new['diagnostics']['numerical_gap']))
        for m,r in rs.items():
            q=r['detail']['diagnostics']
            assert r['audit_residual']<=1e-12 and abs(r['audit_objective']-q['rounded_primal'])<=1e-12
            assert r['detail']['accepted']==(-1e-10<=q['numerical_gap']<=1e-8 and q['rounded_residual']<=1e-12)
            if not r['detail']['accepted']:failures.append([c['seed'],c['config']['label'],m,repeat])
            assert r['detail']['ordinary_iterations']<=50000
            for s in r['detail']['stages']:
                if s['work'] is not None:assert s['work']<=(1000 if s['action']=='cold_epsilon' else 200000)
        good=[r['detail']['diagnostics'] for r in rs.values() if r['detail']['accepted']]
        if good:assert max(q['dual'] for q in good)<=min(q['rounded_primal'] for q in good)+1e-10
ids={(c['seed'],c['label']) for c in fallback};groups={}
for name in ['all','fallback','no_fallback']:
    cs=[c for c in data['cases'] if name=='all' or (((c['seed'],c['config']['label']) in ids)==(name=='fallback'))]
    totals={m:sum(c['metrics'][m]['median_seconds'] for c in cs) for m in data['totals']}
    groups[name]=dict(cases=len(cs),totals=totals,reduction=1-totals['quality_fallback']/totals['legacy_fallback'] if totals['legacy_fallback'] else None,faster=sum(c['metrics']['quality_fallback']['median_seconds']<c['metrics']['legacy_fallback']['median_seconds'] for c in cs))
result=dict(passed=True,failures=failures,fallback_cases=fallback,groups=groups,runs=sum(len(c['runs']) for c in data['cases']))
(d.parent/(d.name+'_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
