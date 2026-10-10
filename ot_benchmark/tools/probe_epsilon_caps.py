"""Post-hoc fixed-cap probe; failures retained; one run per cap, no speed claim."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,json,sys,time,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import ot
from ot_benchmark.algorithms.transport_diagnostics import diagnose,round_plan
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);o=p.parse_args().output;o.mkdir(parents=True,exist_ok=False)
    inp=ROOT/'validation/2026-10-07/quality_real_inputs_round28'
    case=next(c for c in json.loads((inp/'summary.json').read_text())['cases'] if c['domain']=='kos' and c['seed']==111012)
    f=inp/case['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==case['sha256']
    with np.load(f,allow_pickle=False) as z:
        a,b,c=z['a'],z['b'],z['cost'];ia,ib=np.flatnonzero(a),np.flatnonzero(b);a,b,c=a[ia]/a.sum(),b[ib]/b.sum(),c[np.ix_(ia,ib)]
    write(o/'protocol.json',dict(role='post-hoc known hard case development; fixed caps, single run each, no tuning or generalization claim',case=case,caps=[100,250,500],inner=1000,reg=.001,epsilon0=1.,stop=1e-22,quality_gap=1e-8))
    with zipfile.ZipFile(o/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    rows=[]
    for cap in [100,250,500]:
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');t=time.perf_counter()
            _,log=ot.bregman.sinkhorn_epsilon_scaling(a,b,c,.001,numItermax=cap,numInnerItermax=1000,epsilon0=1.,stopThr=1e-22,log=True,warn=True)
            q,raw=diagnose(a,b,c,.001,log['alpha']/.001,log['beta']/.001);plan=round_plan(raw,a,b);seconds=time.perf_counter()-t
        positive=plan>0;obj=float(np.sum(plan*c)+.001*np.sum(plan[positive]*(np.log(plan[positive])-1)))
        assert abs(obj-q['rounded_primal'])<=1e-12
        row=dict(cap=cap,outer=int(log['niter'])+1,seconds=seconds,accepted=bool(-1e-10<=q['numerical_gap']<=1e-8 and q['rounded_residual']<=1e-12),diagnostics=q,warnings=sorted(set(str(w.message) for w in ws)))
        rows.append(row);write(o/f'cap_{cap}.json',row);print(cap,seconds,q['numerical_gap'],row['accepted'],flush=True)
    write(o/'summary.json',rows);write(o/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file()})
if __name__=='__main__':main()
