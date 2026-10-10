"""Cost-aware adaptive inspection versus frozen fixed schedules."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,inspect,json,sys,time,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import ot
from scipy.spatial.distance import cdist
from ot_benchmark.algorithms.quality_policy import solve_quality_policy


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--validation',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    configs=[dict(label=f'n_{n}_reg_{reg}',n=n,d=8,zero=.9,reg=reg) for n in [300,1000,2000] for reg in [.01,.001]]
    cases=[dict(config=c,seed=s) for c in configs for s in ([103001,103002,103003] if args.validation else [101001,101002,101003,102001,102002,102003])]
    methods=['adaptive','every_250','every_1000','residual_gate']
    def solve(method,a,b,cost,reg):
        return solve_quality_policy(a,b,cost,reg=reg,policy=method if method in ('adaptive','residual_gate') else 'periodic',
            check_every=250 if method in ('adaptive','residual_gate') else int(method.split('_')[1]))
    protocol=dict(role='18 new same-generator combinations with frozen adaptive rule' if args.validation else '36 previously viewed development cases; no tuning during this run',cases=cases,
        methods=methods,repetitions=3,ordinary_budget=50000,periodic_every=[250,1000],raw_check_every=50,
        raw_gate=1e-9,gap_tolerance=1e-8,rounded_feasibility=1e-12,roundoff_allowance=1e-10,
        epsilon_outer=200,epsilon_inner=1000,epsilon0=1.,epsilon_stop=1e-22,
        objective='sum(C*P)+reg*sum(P*(log(P)-1))',
        timing='entire solver call, including diagnostics, fallback, rounding and full plan allocation; excludes shared input generation and independent output audit',
        policy_note='adaptive first at 250; half log-gap forecast horizon, at least 20x measured check/iteration cost, bounded 50..2000 and rounded to 50; stagnant gap uses 2000; common raw gate and fallback; online timings and trace allocation included',
        limitation='fixed conservative epsilon schedule; no optimal baseline tuning, no interval certificate, dependent seeds across configs',
        pot_version=ot.__version__,pot_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(ot.bregman.sinkhorn_epsilon_scaling)).read_bytes()).hexdigest(),
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    write(args.output/'protocol.json',protocol)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/quality_policy.py',ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    for method in methods:solve(method,[.5,.5],[.4,.6],[[0.,.1],[.1,0.]],.1)
    order=np.random.default_rng(16000);results=[]
    for case in cases:
        c=case['config'];s=case['seed'];n=c['n'];k=max(1,round(n*(1-c['zero'])))
        points=np.random.default_rng(s).uniform(size=(n,c['d']));cost=cdist(points,points,metric='sqeuclidean');cost/=max(float(cost.max()),1e-15)
        rng=np.random.default_rng(s+100000);a=np.zeros(n);b=np.zeros(n)
        a[rng.permutation(n)[:k]]=rng.uniform(.5,1.5,size=k);b[rng.permutation(n)[:k]]=rng.uniform(.5,1.5,size=k);a/=a.sum();b/=b.sum()
        runs=[]
        for repeat in range(3):
            for method in order.permutation(methods):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');t=time.perf_counter()
                    r=solve(method,a,b,cost,c['reg'])
                    seconds=time.perf_counter()-t
                plan=r.pop('plan');positive=plan>0
                residual=float(max(np.abs(plan.sum(1)-a).sum(),np.abs(plan.sum(0)-b).sum()))
                objective=float(np.sum(plan*cost)+c['reg']*np.sum(plan[positive]*(np.log(plan[positive])-1)))
                assert np.isfinite(plan).all() and np.all(plan>=0) and residual<=1e-12
                assert abs(objective-r['diagnostics']['rounded_primal'])<=1e-12
                expected=-1e-10<=r['diagnostics']['numerical_gap']<=1e-8
                assert r['accepted']==expected
                runs.append(dict(method=str(method),repeat=repeat,seconds=seconds,detail=r,
                    audit_residual=residual,audit_objective=objective,warnings=sorted(set(str(w.message) for w in caught))))
        metrics={m:dict(successes=sum(r['detail']['accepted'] for r in runs if r['method']==m),
            median_seconds=float(np.median([r['seconds'] for r in runs if r['method']==m])),
            fallback_runs=sum(r['detail']['used_epsilon'] for r in runs if r['method']==m)) for m in methods}
        row=dict(**case,metrics=metrics,runs=runs);results.append(row)
        write(args.output/f'{c["label"]}_{s}.json',row)
        print(c['label'],s,metrics,flush=True)
    totals={m:dict(successes=sum(r['metrics'][m]['successes'] for r in results),runs=len(results)*3,
        sum_case_median_seconds=sum(r['metrics'][m]['median_seconds'] for r in results),
        fallback_runs=sum(r['metrics'][m]['fallback_runs'] for r in results)) for m in methods}
    write(args.output/'summary.json',dict(totals=totals,cases=results))
    write(args.output/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output.iterdir() if f.is_file()})
    print(totals,flush=True)


if __name__=='__main__':main()
