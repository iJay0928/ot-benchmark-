"""Frozen new-seed parameter validation, with all planned outcomes retained."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,json,sys,time,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ot_benchmark.algorithms.gated_epsilon import solve_gated_epsilon
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);o=p.parse_args().output;o.mkdir(parents=True,exist_ok=False)
    protocol=dict(role='new seeds and parameter combinations; same designed synthetic family, dependent paired conditions',seeds=[137001,137002],deltas=[0.,1e-5,1e-3],regs=[.003,.001],n=16,methods={'gated':None,'fixed1':1,'fixed10':10},max_blocks=1000,inner=1000,initial_reg=1.,factor=.5,gap=1e-8,raw_stage_residual=1e-8,rounded_residual=1e-12,repeats=1,order_seed=3700,threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    write(o/'protocol.json',protocol)
    with zipfile.ZipFile(o/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/gated_epsilon.py',ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    cases=[]
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);x=np.r_[rng.uniform(0,.02,8),rng.uniform(.98,1,8)]
        cost=(x[:,None]-x[None,:])**2;cost/=cost.max()
        for k,delta in enumerate(protocol['deltas']):
            a=np.ones(16)/16;b=a.copy();b[:8]+=delta/8;b[8:]-=delta/8
            name=f'seed_{seed}_delta{k}.npz';np.savez_compressed(o/name,a=a,b=b,cost=cost,points=x)
            cases.append(dict(seed=seed,delta=delta,file=name,sha256=hashlib.sha256((o/name).read_bytes()).hexdigest()))
    jobs=[dict(case=case,reg=reg,method=method) for case in cases for reg in protocol['regs'] for method in protocol['methods']]
    np.random.default_rng(protocol['order_seed']).shuffle(jobs)
    write(o/'inputs.json',cases);write(o/'jobs.json',jobs)
    rows=[]
    for index,job in enumerate(jobs):
        case,reg,method=job['case'],job['reg'],job['method']
        with np.load(o/case['file'],allow_pickle=False) as z:a,b,c=z['a'],z['b'],z['cost']
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');start=time.perf_counter()
            try:
                d=solve_gated_epsilon(a,b,c,reg=reg,fixed_hold=protocol['methods'][method]);plan=d.pop('plan');positive=plan>0
                residual=float(max(np.abs(plan.sum(1)-a).sum(),np.abs(plan.sum(0)-b).sum()))
                objective=float(np.sum(c*plan)+reg*np.sum(plan[positive]*(np.log(plan[positive])-1)))
                assert np.isfinite(plan).all() and np.all(plan>=0) and residual<=1e-12
                if d['diagnostics'] is not None:assert abs(objective-d['diagnostics']['rounded_primal'])<=1e-12
                row=dict(job=job,detail=d,audit_residual=residual,audit_objective=objective,net_to_first=float(plan[8:,:8].sum()-plan[:8,8:].sum()))
            except Exception as exc:
                row=dict(job=job,error=repr(exc))
            row.update(seconds=time.perf_counter()-start,warnings=sorted(set(str(w.message) for w in ws)))
        write(o/f'run_{index:03d}.json',row);rows.append(row)
        print(index+1,len(jobs),case['seed'],case['delta'],reg,method,row.get('error') or (row['detail']['accepted'],row['detail']['inner_work']),flush=True)
    write(o/'summary.json',dict(rows=rows));write(o/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file()})
if __name__=='__main__':main()
