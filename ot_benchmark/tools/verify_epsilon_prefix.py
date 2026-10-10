"""Known-hard-case equivalence at the quality-stop endpoint, outside timings."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import json,sys,warnings
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import ot
from ot_benchmark.algorithms.quality_epsilon import solve_quality_epsilon
inp=ROOT/'validation/2026-10-07/quality_real_inputs_round28'
case=next(c for c in json.loads((inp/'summary.json').read_text())['cases'] if c['domain']=='kos' and c['seed']==111012)
with np.load(inp/case['file'],allow_pickle=False) as z:
    a,b,c=z['a'],z['b'],z['cost'];ia,ib=np.flatnonzero(a),np.flatnonzero(b);a,b,c=a[ia]/a.sum(),b[ib]/b.sum(),c[np.ix_(ia,ib)]
with warnings.catch_warnings():
    warnings.simplefilter('ignore');r=solve_quality_epsilon(a,b,c)
    raw,log=ot.bregman.sinkhorn_epsilon_scaling(a,b,c,.001,numItermax=r['outer'],numInnerItermax=1000,epsilon0=1.,stopThr=1e-22,log=True,warn=False)
equal={k:bool(np.array_equal(r[k],v)) for k,v in [('raw_plan',raw),('alpha',log['alpha']),('beta',log['beta'])]}
assert all(equal.values()) and r['outer']==log['niter']+1
out=dict(passed=True,outer=r['outer'],equal=equal,quality_gap=r['diagnostics']['numerical_gap'],trace=r['trace'])
(ROOT/'validation/2026-10-08/epsilon_prefix_round30.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
print({k:v for k,v in out.items() if k!='trace'})
