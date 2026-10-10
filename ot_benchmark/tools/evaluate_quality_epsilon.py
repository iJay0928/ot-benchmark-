"""Full-input development comparison of POT epsilon and target-quality stop."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,inspect,json,sys,time,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import ot
from ot_benchmark.algorithms.quality_epsilon import solve_quality_epsilon
from ot_benchmark.algorithms.transport_diagnostics import diagnose,round_plan

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

def solve(method,a,b,c,reg):
    shape=c.shape;ia,ib=np.flatnonzero(a),np.flatnonzero(b)
    a,b,c=a[ia]/a.sum(),b[ib]/b.sum(),c[np.ix_(ia,ib)]
    if method=='quality10':
        r=solve_quality_epsilon(a,b,c,reg)
        del r['raw_plan'];del r['alpha'];del r['beta']
    else:
        _,log=ot.bregman.sinkhorn_epsilon_scaling(a,b,c,reg,numItermax=1000,numInnerItermax=1000,epsilon0=1.,stopThr=1e-22,log=True,warn=True)
        q,raw=diagnose(a,b,c,reg,log['alpha']/reg,log['beta']/reg)
        r=dict(plan=round_plan(raw,a,b),diagnostics=q,accepted=bool(q['rounded_residual']<=1e-12 and -1e-10<=q['numerical_gap']<=1e-8),outer=int(log['niter'])+1)
    full=np.zeros(shape);full[np.ix_(ia,ib)]=r.pop('plan');r['plan']=full;return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args();o=args.output;o.mkdir(parents=True,exist_ok=False)
    for f,h in json.loads((args.inputs/'manifest.json').read_text()).items():assert hashlib.sha256((args.inputs/f).read_bytes()).hexdigest()==h
    cases=json.loads((args.inputs/'summary.json').read_text())['cases'];methods=['pot1000','quality10']
    write(o/'protocol.json',dict(role='all 12 previously viewed round28 pairs; development only',inputs=str(args.inputs.resolve()),input_manifest_sha256=hashlib.sha256((args.inputs/'manifest.json').read_bytes()).hexdigest(),methods=methods,reg=[.01,.001],repetitions=3,seed=30000,
        schedule='(1-reg)*exp(-i)+reg; POT stabilized inner1000 stop1e-22 tau1000; cap1000',
        quality='target-reg gap [-1e-10,1e-8], rounded residual<=1e-12; check i>=35 and i%10==0 or final; no legacy residual stop in quality mode',
        time='full solve, normalization, diagnostics, rounding and full plan allocation; excludes independent audit',
        pot_version=ot.__version__,pot_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(ot.bregman.sinkhorn_epsilon_scaling)).read_bytes()).hexdigest(),threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
    with zipfile.ZipFile(o/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/quality_epsilon.py',ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    for m in methods:solve(m,np.array([.5,.5]),np.array([.4,.6]),np.array([[0.,.1],[.1,0.]]),.01)
    rng=np.random.default_rng(30000);rows=[]
    for case in cases:
        with np.load(args.inputs/case['file'],allow_pickle=False) as z:a,b,c=z['a'],z['b'],z['cost']
        for reg in [.01,.001]:
            runs=[]
            for repeat in range(3):
                for method in rng.permutation(methods):
                    with warnings.catch_warnings(record=True) as ws:
                        warnings.simplefilter('always');t=time.perf_counter();r=solve(method,a,b,c,reg);seconds=time.perf_counter()-t
                    plan=r.pop('plan');pos=plan>0;res=float(max(np.abs(plan.sum(1)-a).sum(),np.abs(plan.sum(0)-b).sum()))
                    obj=float(np.sum(plan*c)+reg*np.sum(plan[pos]*(np.log(plan[pos])-1)))
                    assert np.isfinite(plan).all() and np.all(plan>=0) and res<=1e-12
                    assert abs(obj-r['diagnostics']['rounded_primal'])<=1e-12
                    row=dict(method=str(method),repeat=repeat,seconds=seconds,detail=r,audit_residual=res,audit_objective=obj,warnings=sorted(set(str(w.message) for w in ws)))
                    runs.append(row);write(o/f"{case['domain']}_{case['seed']}_{reg}_{method}_{repeat}.json",row)
                    print(case['domain'],case['seed'],reg,method,repeat,seconds,r['outer'],r['accepted'],flush=True)
            rows.append(dict(input=case,reg=reg,runs=runs,metrics={m:dict(successes=sum(r['detail']['accepted'] for r in runs if r['method']==m),median_seconds=float(np.median([r['seconds'] for r in runs if r['method']==m]))) for m in methods}))
    write(o/'summary.json',dict(cases=rows));write(o/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file()})
if __name__=='__main__':main()
