"""Development diagnostics of residuals, primal/dual progress and plan motion."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,json,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from scipy.spatial.distance import cdist
from ot_benchmark.algorithms.transport_diagnostics import diagnose


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);v=ROOT/'validation/2026-10-06'
    sources=[v/'policy_dev_round11/summary.json',v/'policy_holdout_round11/summary.json']
    oldcases=[r for p in sources for r in json.loads(p.read_text(encoding='utf-8'))['cases']]
    configs={r['config']['label']:r['config'] for r in oldcases}
    cases=[dict(config=c,seed=seed) for c in configs.values() for seed in [99001,99002,99003]]
    def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    protocol=dict(role='12 new seed cases; changed output contract, no speed comparison',
        numerical_gap_tolerance=1e-8,rounded_feasibility_tolerance=1e-12,
        seeds=[99001,99002,99003],
        iteration_budget=50000,solver_check_every=50,solver_acceptance_l1=1e-9,diagnostic_every=250,
        objective='sum(C*P)+reg*sum(P*(log(P)-1)); dual=reg*(a@log_u+b@log_v-sum(exp(log_u+log_v-C/reg)))',
        rounding='AWR 2017 Algorithm 2; numerical feasibility <=1e-12; no interval arithmetic',
        features='centered dual potential motion, raw plan L1 motion, rounded primal-dual numerical gap',
        timing='instrumented diagnostic run, not a performance benchmark',
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources})
    write(args.output/'protocol.json',protocol)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in [Path(__file__),ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(p,p.relative_to(ROOT))
    results=[]
    for prior in cases:
        conf=prior['config'];seed=prior['seed'];n,d=conf['n'],conf['d'];reg=conf['reg'];k=max(1,round(n*(1-conf['zero'])))
        points=np.random.default_rng(seed).uniform(size=(n,d));c=cdist(points,points,metric='sqeuclidean');c/=max(float(c.max()),1e-15)
        rng=np.random.default_rng(seed+100000);a=np.zeros(n);b=np.zeros(n)
        a[rng.permutation(n)[:k]]=rng.uniform(.5,1.5,size=k);b[rng.permutation(n)[:k]]=rng.uniform(.5,1.5,size=k)
        a/=a.sum();b/=b.sum();ia,ib=np.flatnonzero(a),np.flatnonzero(b);a,b=a[ia],b[ib];c=c[np.ix_(ia,ib)]
        kernel=np.exp(-c/reg);u=np.ones(len(a));history=[];last=None;accepted=False
        for i in range(1,50001):
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                vv=b/(kernel.T@u);u=a/(kernel@vv)
            if not np.isfinite(u).all() or not np.isfinite(vv).all():raise RuntimeError('diagnostic case numerical failure')
            if i%50==0:
                residual=float(max(np.abs(u*(kernel@vv)-a).sum(),np.abs(vv*(kernel.T@u)-b).sum()))
                raw_accepted=residual<=1e-9
                if i%250==0 or raw_accepted:
                    lu,lv=np.log(u),np.log(vv);diag,plan=diagnose(a,b,c,reg,lu,lv)
                    f,g=reg*lu,reg*lv;offset=float(a@f);f=f-offset;g=g+offset
                    diag.update(iteration=i,solver_residual=residual,
                        plan_change_l1=float(np.abs(plan-last[0]).sum()) if last else None,
                        potential_change_linf=float(max(np.max(np.abs(f-last[1])),np.max(np.abs(g-last[2])))) if last else None,
                        elapsed_iterations=i-last[3] if last else None)
                    history.append(diag);last=(plan,f,g,i)
                    accepted=diag['rounded_residual']<=1e-12 and -1e-10<=diag['numerical_gap']<=1e-8
                if accepted:break
        row=dict(config=conf,seed=seed,accepted=accepted,iterations=i,history=history)
        write(args.output/f'{conf["label"]}_{seed}.json',row);results.append(row)
        print(conf['label'],seed,accepted,i,'final_gap',history[-1]['numerical_gap'],flush=True)
    write(args.output/'summary.json',dict(cases=results))
    write(args.output/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()})


if __name__=='__main__':main()
