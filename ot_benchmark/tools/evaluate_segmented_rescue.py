"""Development action comparison from the same 50k ordinary state on four real cases."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,json,sys,time,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ot_benchmark.algorithms.segmented_rescue import ordinary_state,rescue
from ot_benchmark.algorithms.transport_diagnostics import diagnose


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();d=args.output;d.mkdir(parents=True,exist_ok=False)
    def write(name,obj):(d/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    source=ROOT/'validation/2026-10-06/quality_real_round18/summary.json'
    cases=[c for c in json.loads(source.read_text())['cases'] if c['metrics']['adaptive_gap']['fallback_runs']]
    inputs=ROOT/'validation/2026-10-06/quality_real_inputs_normalized_round17'
    actions=['ordinary_continue','warm_stabilized','cold_stabilized','cold_epsilon']
    write('protocol.json',dict(role='development only; all four previously observed fallback cases, not new holdout',
        case_selection=[c['input'] for c in cases],source_sha256=sha(source),actions=actions,repetitions=3,
        prefix_iterations=50000,block_size=5000,max_blocks=40,stabilized_stop=1e-12,
        epsilon_outer=1000,epsilon_inner=1000,epsilon_stop=1e-22,epsilon0=1.,
        gap_tolerance=1e-8,feasibility=1e-12,roundoff_allowance=1e-10,
        timing='remaining action time from identical snapshot; includes initial quality check, kernel build, block checks and plan rounding; excludes shared prefix and input generation',
        limitation='different method budgets; outer/inner iterations not interchangeable; not complete application speed or novel method',
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
    with zipfile.ZipFile(d/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/segmented_rescue.py',ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    rng=np.random.default_rng(19000);results=[]
    for c in cases:
        f=inputs/c['input']['file'];assert sha(f)==c['input']['sha256']
        with np.load(f) as z:a,b,cost=z['a'],z['b'],z['cost']
        ia,ib=np.flatnonzero(a),np.flatnonzero(b);a,b,cost=a[ia],b[ib],cost[np.ix_(ia,ib)];a/=a.sum();b/=b.sum();reg=c['config']['reg']
        start=time.perf_counter();state=ordinary_state(a,b,cost,reg);prefix_seconds=time.perf_counter()-start
        diag,_=diagnose(a,b,cost,reg,np.log(state['u']),np.log(state['v']))
        snapshot=hashlib.sha256(state['u'].tobytes()+state['v'].tobytes()).hexdigest()
        label=c['config']['label']+'_'+str(c['seed'])
        np.savez_compressed(d/(label+'_state.npz'),u=state['u'],v=state['v'],a=a,b=b,cost=cost)
        runs=[]
        for repeat in range(3):
            for action in rng.permutation(actions):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');start=time.perf_counter()
                    r=rescue(a,b,cost,reg,state,str(action));seconds=time.perf_counter()-start
                plan=r.pop('plan');positive=plan>0
                residual=float(max(np.abs(plan.sum(1)-a).sum(),np.abs(plan.sum(0)-b).sum()))
                objective=float(np.sum(plan*cost)+reg*np.sum(plan[positive]*(np.log(plan[positive])-1)))
                assert residual<=1e-12 and abs(objective-r['diagnostics']['rounded_primal'])<=1e-12
                assert r['accepted']==(-1e-10<=r['diagnostics']['numerical_gap']<=1e-8)
                assert hashlib.sha256(state['u'].tobytes()+state['v'].tobytes()).hexdigest()==snapshot
                runs.append(dict(action=str(action),repeat=repeat,seconds=seconds,detail=r,audit_objective=objective,audit_residual=residual,
                    snapshot_sha256=snapshot,warnings=sorted(set(str(w.message) for w in caught))))
        metrics={m:dict(successes=sum(r['detail']['accepted'] for r in runs if r['action']==m),
            median_seconds=float(np.median([r['seconds'] for r in runs if r['action']==m]))) for m in actions}
        row=dict(input=c['input'],reg=reg,prefix_seconds=prefix_seconds,prefix_diagnostics=diag,snapshot_sha256=snapshot,runs=runs,metrics=metrics)
        write(label+'.json',row);results.append(row);print(label,metrics,flush=True)
    write('summary.json',dict(cases=results));write('manifest.json',{f.name:sha(f) for f in d.iterdir() if f.is_file()})


if __name__=='__main__':main()
