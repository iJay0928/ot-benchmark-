"""Paired branch measurements from the same ordinary-scaling state.

Development evidence only: measuring every action is an offline oracle, not
an online selection strategy. Prefix/preprocessing times are separate.
"""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,json,sys,time,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import ot
from scipy.spatial.distance import cdist


def ordinary(a,b,kernel,state,budget):
    u,v=(x.copy() for x in state)
    residual=None
    for i in range(1,budget+1):
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            v=b/(kernel.T@u);u=a/(kernel@v)
        if not np.isfinite(u).all() or not np.isfinite(v).all():
            return None,dict(iterations=i,residual=None,accepted=False)
        if i%50==0 or i==budget:
            residual=float(max(np.abs(u*(kernel@v)-a).sum(),np.abs(v*(kernel.T@u)-b).sum()))
            if residual<=1e-9:break
    return u[:,None]*kernel*v[None,:],dict(iterations=i,residual=residual,accepted=residual<=1e-9)


def branch(method,a,b,cost,kernel,state,reg,checkpoint):
    if method=='continue':
        plan,detail=ordinary(a,b,kernel,state,50000-checkpoint)
    else:
        warm=None
        if method=='epsilon_warm':warm=tuple(reg*np.log(x.copy()) for x in state)
        plan,log=ot.bregman.sinkhorn_epsilon_scaling(a,b,cost,reg,numItermax=200,
            numInnerItermax=1000,epsilon0=1.,stopThr=1e-22,warmstart=warm,log=True,warn=True)
        detail=dict(outer_iterations=int(log['niter'])+1)
    finite=plan is not None and np.isfinite(plan).all() and np.all(plan>=0)
    residual=float(max(np.abs(plan.sum(axis=1)-a).sum(),np.abs(plan.sum(axis=0)-b).sum())) if finite else None
    detail.update(accepted=bool(finite and residual<=1e-9),residual=residual,
                  cost=float(np.sum(plan*cost)) if finite else None)
    return detail


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    v=ROOT/'validation/2026-10-06'
    inputs=[v/'switch_probe_round9/summary.json',v/'switch_probe_holdout_round9/summary.json']
    cases=[r for path in inputs for r in json.loads(path.read_text(encoding='utf-8'))['cases']]
    methods=['continue','epsilon_cold','epsilon_warm'];checkpoints=[1000,3000]
    def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    protocol=dict(role='development paired actions; all 28 prior cases; skip branches already converged',
        checkpoints=checkpoints,methods=methods,repetitions=3,acceptance_l1=1e-9,
        ordinary_total_budget=50000,epsilon_outer_budget=200,epsilon_inner_budget=1000,
        epsilon0=1.,epsilon_stop=1e-22,pot_version=ot.__version__,
        warmstart='alpha=reg*log(u), beta=reg*log(v), in original cost units',
        timing='remaining branch including state copy, warmstart conversion, validation; shared prefix and preprocessing separate',
        budget_warning='ordinary iterations and epsilon inner/outer budgets are different work units',
        inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs})
    write(args.output/'protocol.json',protocol)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
    toy=np.ones(2)/2;tc=np.array([[0.,.1],[.1,0.]]);tk=np.exp(-tc/.01)
    for method in methods:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');branch(method,toy,toy,tc,tk,(np.ones(2),np.ones(2)),.01,1000)
    order=np.random.default_rng(98100);allrows=[];screen=[]
    for prior in cases:
        conf=prior['config'];seed=prior['seed'];n,d=conf['n'],conf['d'];reg=conf['reg']
        prep=time.perf_counter();k=max(1,round(n*(1-conf['zero'])))
        points=np.random.default_rng(seed).uniform(size=(n,d));cost=cdist(points,points,metric='sqeuclidean');cost/=max(float(cost.max()),1e-15)
        rng=np.random.default_rng(seed+100000);a=np.zeros(n);b=np.zeros(n)
        a[rng.permutation(n)[:k]]=rng.uniform(.5,1.5,size=k);b[rng.permutation(n)[:k]]=rng.uniform(.5,1.5,size=k)
        a/=a.sum();b/=b.sum();ia,ib=np.flatnonzero(a),np.flatnonzero(b)
        a,b=a[ia],b[ib];cost=cost[np.ix_(ia,ib)];kernel=np.exp(-cost/reg)
        prep=time.perf_counter()-prep;u=np.ones(len(a));vb=np.ones(len(b));prefix_seconds=0.;accepted=False
        for i in range(1,3001):
            start=time.perf_counter();vb=b/(kernel.T@u);u=a/(kernel@vb)
            if i%50==0:residual=float(max(np.abs(u*(kernel@vb)-a).sum(),np.abs(vb*(kernel.T@u)-b).sum()))
            prefix_seconds+=time.perf_counter()-start
            if i%50==0 and residual<=1e-9:accepted=True;break
            if i in checkpoints:
                state=(u.copy(),vb.copy());before=hashlib.sha256(u.tobytes()+vb.tobytes()).hexdigest();runs=[]
                for repeat in range(3):
                    for method in order.permutation(methods):
                        start=time.perf_counter()
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter('always')
                            detail=branch(method,a,b,cost,kernel,state,reg,i)
                        runs.append(dict(method=str(method),repeat=repeat,seconds=time.perf_counter()-start,
                                         detail=detail,warnings=sorted(set(str(w.message) for w in caught))))
                assert before==hashlib.sha256(state[0].tobytes()+state[1].tobytes()).hexdigest()
                metrics={}
                for method in methods:
                    subset=[r for r in runs if r['method']==method]
                    metrics[method]=dict(successes=sum(r['detail']['accepted'] for r in subset),
                        median_seconds=float(np.median([r['seconds'] for r in subset])),
                        cost=float(np.median([r['detail']['cost'] for r in subset if r['detail']['accepted']])) if any(r['detail']['accepted'] for r in subset) else None)
                row=dict(config=conf,seed=seed,checkpoint=i,prefix_seconds=prefix_seconds,preprocessing_seconds=prep,
                         state_sha256=before,state_unchanged=True,metrics=metrics)
                write(args.output/f'{conf["label"]}_{seed}_{i}.json',dict(case=row,runs=runs))
                allrows.append(row);print(conf['label'],seed,i,metrics,flush=True)
        screen.append(dict(config=conf,seed=seed,accepted_by_3000=accepted,iterations=i))
    write(args.output/'summary.json',dict(cases=allrows,screening=screen))
    write(args.output/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()})


if __name__=='__main__':main()
