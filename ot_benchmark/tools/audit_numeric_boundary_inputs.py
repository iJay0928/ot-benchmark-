"""Regenerate all frozen boundary arrays and report input numerics."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
d=Path(sys.argv[1]);protocol=json.loads((d/'protocol.json').read_text())
for name,h in json.loads((d/'manifest.json').read_text()).items():assert hashlib.sha256((d/name).read_bytes()).hexdigest()==h
rows=[]
for case in json.loads((d/'summary.json').read_text())['cases']:
    rng=np.random.default_rng(case['seed']);family=case['domain']
    if family=='dynamic_mass':
        x=np.sort(rng.uniform(0,1,16));a=10.**rng.uniform(-6,0,16);b=10.**rng.uniform(-6,0,16);a/=a.sum();b/=b.sum()
    else:
        x=np.r_[rng.uniform(0,.02,8),rng.uniform(.98,1.,8)];a=np.ones(16)/16;b=a.copy()
        if family=='near_balanced_blocks':b[:8]+=1e-4/8;b[8:]-=1e-4/8
    c=(x[:,None]-x[None,:])**2;c/=c.max()
    with np.load(d/case['file'],allow_pickle=False) as z:
        for k,v in [('a',a),('b',b),('cost',c),('points',x)]:assert np.array_equal(z[k],v)
    assert np.all(a>0) and np.all(b>0) and abs(a.sum()-1)<1e-12 and abs(b.sum()-1)<1e-12
    rows.append(dict(family=family,seed=case['seed'],min_mass=float(min(a.min(),b.min())),kernel_zeros={str(reg):int(np.count_nonzero(np.exp(-c/reg)==0)) for reg in protocol['reg']}))
out=dict(passed=True,regenerated_pairs=len(rows),rows=rows)
(d.parent/(d.name+'_audit.json')).write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
