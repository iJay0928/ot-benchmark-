"""Development-only, forced ordinary trajectories and diagnostic overhead."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse, hashlib, json, sys, time, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ot_benchmark.algorithms.transport_diagnostics import diagnose
from ot_benchmark.algorithms.representation_budget import representation_budget

def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def run(a,b,c,augmented):
    k=np.exp(-c/.001);u=np.ones(len(a));v=np.ones(len(b));trace=[]
    start=time.perf_counter()
    for i in range(1,50001):
        v=b/(k.T@u);u=a/(k@v)
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            raise ValueError('nonfinite forced trajectory')
        gate=False
        if i%50==0:
            gate=max(np.abs(u*(k@v)-a).sum(),np.abs(v*(k.T@u)-b).sum())<=1e-9
        if gate or i%250==0:
            d,r=diagnose(a,b,c,.001,np.log(u),np.log(v))
            p=u[:,None]*k*v[None,:]
            row=dict(iteration=i,gap=d['numerical_gap'],accepted=bool(-1e-10<=d['numerical_gap']<=1e-8 and d['rounded_residual']<=1e-12),l1=float(np.abs(p-r).sum()))
            if augmented:row.update(representation_budget(a,b,c,.001,p,r))
            trace.append(row)
    seconds=time.perf_counter()-start
    return dict(seconds=seconds,trace=trace,terminal_sha256=hashlib.sha256(u.tobytes()+v.tobytes()).hexdigest())

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    out=args.output;out.mkdir(parents=True,exist_ok=False)
    old=ROOT/'validation/2026-10-06/segmented_rescue_dev_round19'
    cases=[dict(name=f.stem,path=f) for f in sorted(old.glob('*_state.npz'))]
    real=ROOT/'validation/2026-10-06/quality_real_inputs_round20'
    row=next(r for r in json.loads((real/'summary.json').read_text())['cases'] if r['domain']=='movielens' and r['seed']==106011)
    cases.append(dict(name='movielens_106011',path=real/row['file']))
    for case in cases:
        f=case['path'];manifest=json.loads((f.parent/'manifest.json').read_text())
        assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest[f.name]
    write(out/'protocol.json',dict(role='post-hoc development, four round19 states plus known round20 near-threshold case',
        cases=[dict(name=x['name'],path=str(x['path']),sha256=hashlib.sha256(x['path'].read_bytes()).hexdigest()) for x in cases],
        iterations=50000,repetitions=3,reg=.001,thresholds=dict(l1=1e-8,bound=1e-8,observed=1e-8),
        check_every=250,raw_check_every=50,raw_gate=1e-9,order_seed=24001,
        interpretation='Forced trajectories do not stop or rescue; first crossing is counterfactual. Timing compares diagnostic overhead only, not end-to-end switching policies. Acceptance takes precedence.',
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
    with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/transport_diagnostics.py',ROOT/'ot_benchmark/algorithms/representation_budget.py']:z.write(f,f.relative_to(ROOT))
    run(np.array([.5,.5]),np.array([.4,.6]),np.array([[0.,.001],[.001,0.]]),True)
    rng=np.random.default_rng(24001);results=[]
    for case in cases:
        with np.load(case['path'],allow_pickle=False) as s:
            a,b,c=s['a'],s['b'],s['cost'];ia,ib=np.flatnonzero(a),np.flatnonzero(b)
            a,b,c=a[ia]/a.sum(),b[ib]/b.sum(),c[np.ix_(ia,ib)]
        runs=[]
        for repeat in range(3):
            for augmented in rng.permutation([False,True]):
                runs.append(dict(repeat=repeat,augmented=bool(augmented),**run(a,b,c,bool(augmented))))
        assert len({r['terminal_sha256'] for r in runs})==1
        tr=next(r['trace'] for r in runs if r['augmented'])
        first={}
        for label,key in [('l1','l1'),('bound','gap_distortion_bound'),('observed','observed_gap_distortion')]:
            event=next((r for r in tr if r['accepted'] or r[key]>1e-8),None)
            first[label]=None if event is None else dict(iteration=event['iteration'],action='accept' if event['accepted'] else 'switch',value=event[key],observed=event['observed_gap_distortion'],bound=event['gap_distortion_bound'])
        med={str(flag):float(np.median([r['seconds'] for r in runs if r['augmented']==flag])) for flag in [False,True]}
        result=dict(name=case['name'],first_events=first,median_seconds=med,overhead_fraction=med['True']/med['False']-1,runs=runs)
        write(out/(case['name']+'.json'),result);results.append(result)
        print(case['name'],first,med,flush=True)
    write(out/'summary.json',dict(cases=[{k:v for k,v in r.items() if k!='runs'} for r in results],total_runs=sum(len(r['runs']) for r in results)))
    write(out/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()})

if __name__=='__main__':main()
