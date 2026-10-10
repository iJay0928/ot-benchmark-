"""Audit target quality stopping and paired full-call development timings."""
import hashlib,json,sys
from pathlib import Path
d=Path(sys.argv[1])
for f,h in json.loads((d/'manifest.json').read_text()).items():assert hashlib.sha256((d/f).read_bytes()).hexdigest()==h
data=json.loads((d/'summary.json').read_text());failures=[];outer=[]
for c in data['cases']:
    for r in c['runs']:
        x=r['detail'];q=x['diagnostics']
        assert r['audit_residual']<=1e-12 and abs(r['audit_objective']-q['rounded_primal'])<=1e-12
        assert x['accepted']==(-1e-10<=q['numerical_gap']<=1e-8 and q['rounded_residual']<=1e-12)
        assert x['outer']<=1000
        if not x['accepted']:failures.append([c['input']['seed'],c['reg'],r['method'],r['repeat']])
        if r['method']=='quality10':
            assert all(t['target_reg']==c['reg'] for t in x['trace'])
            assert all(not t['accepted'] for t in x['trace'][:-1])
            if x['reason']=='target quality':assert x['trace'][-1]['accepted'] and x['trace'][-1]['outer']==x['outer']
    qs=[r['detail']['diagnostics'] for r in c['runs'] if r['detail']['accepted']]
    if qs:assert max(q['dual'] for q in qs)<=min(q['rounded_primal'] for q in qs)+1e-10
    outer.append(dict(domain=c['input']['domain'],seed=c['input']['seed'],reg=c['reg'],outer={m:next(r['detail']['outer'] for r in c['runs'] if r['method']==m) for m in ['pot1000','quality10']}))
groups={}
for name in ['all','without_known_hard','known_hard']:
    cs=[c for c in data['cases'] if name=='all' or ((c['input']['domain']=='kos' and c['input']['seed']==111012 and c['reg']==.001)==(name=='known_hard'))]
    totals={m:sum(c['metrics'][m]['median_seconds'] for c in cs) for m in ['pot1000','quality10']}
    groups[name]=dict(cases=len(cs),totals=totals,reduction=1-totals['quality10']/totals['pot1000'],faster=sum(c['metrics']['quality10']['median_seconds']<c['metrics']['pot1000']['median_seconds'] for c in cs))
result=dict(failures=failures,groups=groups,outer=outer,runs=sum(len(c['runs']) for c in data['cases']))
(d.parent/(d.name+'_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
