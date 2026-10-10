"""Post-hoc stage profile of round28's single difficult case; not validation."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,inspect,json,sys,time,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import ot
import ot_benchmark.algorithms.numeric_guard_policy as policy
from ot_benchmark.algorithms.segmented_rescue import rescue

def write(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=False)
    inputs=ROOT/'validation/2026-10-07/quality_real_inputs_round28'
    case=next(c for c in json.loads((inputs/'summary.json').read_text())['cases'] if c['domain']=='kos' and c['seed']==111012)
    path=inputs/case['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==case['sha256']
    with np.load(path,allow_pickle=False) as z:a,b,c=z['a'],z['b'],z['cost']
    write(out/'protocol.json',dict(role='post-hoc single known difficult case; no new threshold or independent validation',case=case,reg=.001,
        methods=['guarded_250','direct_epsilon'],repetitions=3,order_seed=29000,
        timing='whole call and separately instrumented rescue calls; direct epsilon includes active extraction, normalization and full plan allocation',
        epsilon=dict(outer=1000,inner=1000,epsilon0=1.,stop=1e-22),quality_gap=1e-8,feasibility=1e-12,
        pot_version=ot.__version__,pot_sha256=hashlib.sha256(Path(inspect.getsourcefile(ot.bregman.sinkhorn_epsilon_scaling)).read_bytes()).hexdigest(),
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
    with zipfile.ZipFile(out/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/numeric_guard_policy.py',ROOT/'ot_benchmark/algorithms/segmented_rescue.py',ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    def direct():
        ia,ib=np.flatnonzero(a),np.flatnonzero(b);aa,bb=a[ia]/a.sum(),b[ib]/b.sum();cc=c[np.ix_(ia,ib)]
        r=rescue(aa,bb,cc,.001,dict(u=np.ones(len(aa)),v=np.ones(len(bb))),'cold_epsilon')
        plan=np.zeros(c.shape);plan[np.ix_(ia,ib)]=r.pop('plan');r['plan']=plan;return r
    rng=np.random.default_rng(29000);runs=[]
    for repeat in range(3):
        for method in rng.permutation(['guarded_250','direct_epsilon']):
            stages=[]
            def timed(*args,**kwargs):
                start=time.perf_counter();r=rescue(*args,**kwargs);elapsed=time.perf_counter()-start
                stages.append(dict(action=args[5],seconds=elapsed,accepted=r['accepted'],work=r['additional_work'],gap=r['diagnostics']['numerical_gap']))
                return r
            original=policy.rescue
            try:
                policy.rescue=timed
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');start=time.perf_counter()
                    r=policy.solve_guard_policy(a,b,c,.001,policy='guarded_250') if method=='guarded_250' else direct()
                    elapsed=time.perf_counter()-start
            finally:policy.rescue=original
            plan=r.pop('plan');positive=plan>0
            residual=float(max(np.abs(plan.sum(1)-a).sum(),np.abs(plan.sum(0)-b).sum()))
            objective=float(np.sum(plan*c)+.001*np.sum(plan[positive]*(np.log(plan[positive])-1)))
            assert np.isfinite(plan).all() and np.all(plan>=0) and residual<=1e-12
            assert abs(objective-r['diagnostics']['rounded_primal'])<=1e-12
            assert r['accepted']==(-1e-10<=r['diagnostics']['numerical_gap']<=1e-8)
            row=dict(method=str(method),repeat=repeat,seconds=elapsed,stages=stages,other_seconds=elapsed-sum(s['seconds'] for s in stages) if stages else None,
                detail=r,audit_residual=residual,audit_objective=objective,warnings=sorted(set(str(w.message) for w in caught)))
            runs.append(row);write(out/f'{method}_{repeat}.json',row)
            print(method,repeat,elapsed,'accepted',r['accepted'],stages,flush=True)
    totals={m:dict(median_seconds=float(np.median([r['seconds'] for r in runs if r['method']==m])),successes=sum(r['detail']['accepted'] for r in runs if r['method']==m)) for m in ['guarded_250','direct_epsilon']}
    write(out/'summary.json',dict(totals=totals,runs=runs));write(out/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in out.iterdir() if f.is_file()});print(totals,flush=True)

if __name__=='__main__':main()
