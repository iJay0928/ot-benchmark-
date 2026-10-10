"""Post-hoc real-input threshold sensitivity; already viewed round20 inputs."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,inspect,json,sys,time,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import ot
from scipy.spatial.distance import cdist
from ot_benchmark.algorithms.numeric_guard_policy import solve_guard_policy


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--development',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    manifest=json.loads((args.inputs/'manifest.json').read_text())
    for name,h in manifest.items():assert hashlib.sha256((args.inputs/name).read_bytes()).hexdigest()==h
    source_cases=json.loads((args.inputs/'summary.json').read_text())['cases']
    cases=[dict(config=dict(label=f"{r['domain']}_reg_{reg}",reg=reg),seed=r['seed'],input=r) for r in source_cases for reg in [.01,.001]]
    methods=['guard_1e-10','guard_1e-08','guard_1e-06','direct_stabilized_250']
    def solve(method,a,b,cost,reg):
        return solve_guard_policy(a,b,cost,reg=reg,policy='guarded' if method.startswith('guard_') else method,discrepancy_threshold=float(method.split('_')[1]) if method.startswith('guard_') else 1e-8)
    protocol=dict(role='12 previously viewed round20 pairs; development sensitivity only, contains known near-threshold case',input_manifest_sha256=hashlib.sha256((args.inputs/'manifest.json').read_bytes()).hexdigest(),input_directory=str(args.inputs.resolve()),cases=cases,
        methods=methods,repetitions=3,ordinary_budget=50000,periodic_every=250,discrepancy_l1_triggers=[1e-10,1e-8,1e-6],rescue_block=5000,rescue_max_blocks=40,direct_iteration_budget=250000,direct_blocks=[250],stabilized_internal_stop=1e-12,raw_check_every=50,
        raw_gate=1e-9,gap_tolerance=1e-8,rounded_feasibility=1e-12,roundoff_allowance=1e-10,
        epsilon_outer=1000,epsilon_inner=1000,epsilon0=1.,epsilon_stop=1e-22,
        objective='sum(C*P)+reg*sum(P*(log(P)-1))',
        timing='entire solver call, including diagnostics, fallback, rounding and full plan allocation; excludes shared input generation and independent output audit',
        transfer='MovieLens genre geometry and KOS shared cooccurrence geometry; unchanged costs and frozen normalized masses; paired regularizations are dependent',
        policy_note='guarded thresholds 1e-10/1e-8/1e-6, default unchanged; direct stabilized starts from zero dual potentials with no ordinary prefix, block size 5000 or 250, total max 250000, common quality gate and cold epsilon fallback; direct stage counted as used_rescue field for schema compatibility only',
        limitation='fixed conservative epsilon schedule; no optimal baseline tuning, no interval certificate, dependent seeds across configs',
        pot_version=ot.__version__,pot_source_sha256=hashlib.sha256(Path(inspect.getsourcefile(ot.bregman.sinkhorn_epsilon_scaling)).read_bytes()).hexdigest(),
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    write(args.output/'protocol.json',protocol)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/numeric_guard_policy.py',ROOT/'ot_benchmark/algorithms/segmented_rescue.py',ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    for method in methods:solve(method,[.5,.5],[.4,.6],[[0.,.1],[.1,0.]],.1)
    order=np.random.default_rng(22001);results=[]
    for case in cases:
        c=case['config'];s=case['seed']
        path=args.inputs/case['input']['file']
        assert hashlib.sha256(path.read_bytes()).hexdigest()==case['input']['sha256']
        with np.load(path,allow_pickle=False) as z:a,b,cost=z['a'],z['b'],z['cost']
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
