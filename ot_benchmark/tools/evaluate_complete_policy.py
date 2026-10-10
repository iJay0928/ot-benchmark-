"""Frozen one-factor exploratory sweep against native POT strong baselines."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import ot
from scipy.spatial.distance import cdist
from ot_benchmark.algorithms.scaling_ot import solve_scaling_ot
from ot_benchmark.data_generator import validate_transport_plan


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


from ot_benchmark.algorithms.adaptive_policy import solve_policy

def solve(method,a,b,cost,reg):
    detail=solve_policy(a,b,cost,reg=reg,policy=method)
    return detail['cost'],detail


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--holdout',action='store_true')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); args.output.mkdir(parents=True,exist_ok=False)
    base=dict(n=300,d=8,zero=.9,reg=.01)
    configs=[dict(label='base',**base),dict(label='d_32',**dict(base,d=32)),
             dict(label='d_128',**dict(base,d=128)),dict(label='reg_0.001',**dict(base,reg=.001))]
    methods=['budget','rolling','always']
    protocol=dict(configs=configs,seeds=[98001,98002,98003] if args.holdout else [94001,94002,96001,96002,96003,97001,97002,97003],
        repetitions=3,methods=methods,pot_version=ot.__version__,acceptance_residual=1e-9,
        ordinary_stop_l1=1e-9,ordinary_check_every=50,epsilon0=1.,epsilon_outer_budget=200,epsilon_internal_stop_threshold=1e-22,epsilon_inner_budget=1000,
        mode='new seeds same generator' if args.holdout else '28 previously examined development cases',
        budget_note='different algorithm budgets; compare full observed time, not equal-work claims',
        geometry='shared uniform support, squared Euclidean divided by full matrix maximum',
        reference='budget policy when accepted; EMD separately records unregularized bias',
        policies='budget: 50000 then epsilon; rolling: preceding 500 every 1000 through 10000 predicts finish>50000; always: epsilon from start',
        timing='complete solver policy including kernel build, monitoring, failed prefix and fallback; excludes common preprocessing',
        design='all 28 previously seen cases for development; all 12 new seed cases for validation',
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    import inspect,hashlib
    source=Path(inspect.getsourcefile(ot.bregman.sinkhorn_epsilon_scaling))
    protocol['pot_source_sha256']=hashlib.sha256(source.read_bytes()).hexdigest()
    write(args.output/'protocol.json',protocol)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in list((ROOT/'ot_benchmark').rglob('*.py'))+[Path(__file__)]:z.write(p,p.relative_to(ROOT))
    toy=np.ones(3)/3; tc=np.ones((3,3))-np.eye(3)
    for method in methods:solve(method,toy,toy,tc,.1)
    order=np.random.default_rng(93000); cases=[]
    for conf in configs:
        for seed in protocol['seeds']:
            if not args.holdout and seed in [94001,94002] and (conf['label'],seed) not in [('d_32',94001),('d_128',94001),('reg_0.001',94001),('reg_0.001',94002)]:continue
            started=time.perf_counter()
            n,d=conf['n'],conf['d']; k=max(1,round(n*(1-conf['zero'])))
            points=np.random.default_rng(seed).uniform(size=(n,d))
            cost=cdist(points,points,metric='sqeuclidean');cost/=max(float(cost.max()),1e-15)
            rng=np.random.default_rng(seed+100000)
            a=np.zeros(n);b=np.zeros(n)
            a[rng.permutation(n)[:k]]=rng.uniform(.5,1.5,size=k)
            b[rng.permutation(n)[:k]]=rng.uniform(.5,1.5,size=k)
            a/=a.sum();b/=b.sum()
            prep=time.perf_counter()-started
            ia,ib=np.flatnonzero(a),np.flatnonzero(b)
            m=cost[np.ix_(ia,ib)]
            plan,emdlog=ot.emd(a[ia],b[ib],m,log=True)
            validate_transport_plan(plan,a[ia],b[ib])
            if emdlog.get('warning') or emdlog.get('result_code',1)!=1:raise RuntimeError('invalid exact reference')
            exact=float(np.sum(plan*m));runs=[]
            for repeat in range(3):
                for method in order.permutation(methods):
                    started=time.perf_counter()
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter('always')
                        try:
                            value,detail=solve(method,a,b,cost,conf['reg'])
                            row=dict(success=detail['accepted'],cost=value,detail=detail)
                        except (ValueError,RuntimeError,FloatingPointError) as exc:
                            row=dict(success=False,cost=None,error=str(exc))
                    row.update(method=str(method),repeat=repeat,seconds=time.perf_counter()-started,
                               warnings=sorted(set(str(w.message) for w in caught)))
                    runs.append(row)
            logvalues=[r['cost'] for r in runs if r['method']=='budget' and r['success']]
            logcost=float(np.median(logvalues)) if logvalues else None
            metrics={}
            for method in methods:
                subset=[r for r in runs if r['method']==method]
                good=[r for r in subset if r['success']]
                value=float(np.median([r['cost'] for r in good])) if good else None
                metrics[method]=dict(successes=len(good),median_seconds=float(np.median([r['seconds'] for r in subset])),
                    cost=value,exact_abs_error=abs(value-exact) if value is not None else None,
                    reference_abs_difference=abs(value-logcost) if value is not None and logcost is not None else None,
                    agrees_with_reference=abs(value-logcost)<=1e-4+1e-3*abs(logcost) if value is not None and logcost is not None else None,
                    fallback_runs=sum(r.get('detail',{}).get('used_epsilon',False) for r in subset))
            case=dict(config=conf,seed=seed,active_source=k,active_target=k,preprocessing_seconds=prep,
                      dense_cost_bytes=cost.nbytes,exact_cost=exact,metrics=metrics)
            write(args.output/f'{conf["label"]}_{seed}.json',dict(case=case,runs=runs))
            cases.append(case)
            print(conf['label'],seed,{key:(v['successes'],round(v['median_seconds'],4)) for key,v in metrics.items()},flush=True)
    summary={}
    for conf in configs:
        subset=[r for r in cases if r['config']['label']==conf['label']]
        summary[conf['label']]={}
        for method in methods:
            matched=[r for r in subset if r['metrics'][method]['successes']==3 and r['metrics']['rolling']['successes']==3]
            summary[conf['label']][method]=dict(successes=sum(r['metrics'][method]['successes'] for r in subset),runs=len(subset)*3,
                agree_cases=sum(r['metrics'][method]['agrees_with_reference'] is True for r in subset),
                fallback_runs=sum(r['metrics'][method]['fallback_runs'] for r in subset),paired_cases=len(matched),
                rolling_speedup_median=float(np.median([r['metrics'][method]['median_seconds']/r['metrics']['rolling']['median_seconds'] for r in matched])) if matched else None)
    write(args.output/'summary.json',dict(summary=summary,cases=cases))
    write(args.output/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()})


if __name__=='__main__':main()
