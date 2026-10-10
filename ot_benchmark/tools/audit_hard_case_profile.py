"""Independent saved-result and timing-accounting audit for the stage probe."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
d=Path(sys.argv[1])
for name,h in json.loads((d/'manifest.json').read_text()).items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h
data=json.loads((d/'summary.json').read_text());runs=data['runs'];assert len(runs)==6
for r in runs:
    q=r['detail']['diagnostics']
    assert r['detail']['accepted'] and -1e-10<=q['numerical_gap']<=1e-8
    assert r['audit_residual']<=1e-12 and abs(r['audit_objective']-q['rounded_primal'])<=1e-12
    if r['method']=='guarded_250':
        assert [s['action'] for s in r['stages']]==['ordinary_continue','cold_epsilon']
        assert r['other_seconds']>=0
        assert abs(r['seconds']-sum(s['seconds'] for s in r['stages'])-r['other_seconds'])<1e-9
qs=[r['detail']['diagnostics'] for r in runs]
assert max(q['dual'] for q in qs)<=min(q['rounded_primal'] for q in qs)+1e-10
assert max(q['rounded_primal'] for q in qs)-min(q['rounded_primal'] for q in qs)<1e-12
g=[r for r in runs if r['method']=='guarded_250']
stage_medians={a:float(np.median([next(s['seconds'] for s in r['stages'] if s['action']==a) for r in g])) for a in ['ordinary_continue','cold_epsilon']}
result=dict(passed=True,runs=6,totals=data['totals'],stage_medians=stage_medians,
    prefix_and_other_median=float(np.median([r['other_seconds'] for r in g])),
    epsilon_fraction_median=float(np.median([next(s['seconds'] for s in r['stages'] if s['action']=='cold_epsilon')/r['seconds'] for r in g])),
    direct_reduction=1-data['totals']['direct_epsilon']['median_seconds']/data['totals']['guarded_250']['median_seconds'],
    limitation='single known case, three timing repeats, no warmup; medians of components need not add to median total')
(d.parent/(d.name+'_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
