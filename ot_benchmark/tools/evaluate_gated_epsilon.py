"""Frozen two-case stage-gate development probe; not held-out validation."""
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
    inp=ROOT/'validation/2026-10-09/numeric_boundary_inputs_round33'
    cases=[c for c in json.loads((inp/'summary.json').read_text())['cases'] if c['domain']=='near_balanced_blocks']
    write(o/'protocol.json',dict(role='two previously viewed failures, development only; single run each',cases=cases,reg=.001,initial_reg=1.,factor=.5,
        stage_gap=1e-8,stage_raw_residual=1e-8,max_blocks=1000,inner=1000,target_gap=1e-8,rounded_feasibility=1e-12,
        reference='round34 pace1/10 saved outcomes; compare quality and work only, not cross-run speed',
        threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
    with zipfile.ZipFile(o/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/gated_epsilon.py',ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    rows=[]
    for case in cases:
        f=inp/case['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==case['sha256']
        with np.load(f,allow_pickle=False) as z:a,b,c=z['a'],z['b'],z['cost']
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');start=time.perf_counter();r=solve_gated_epsilon(a,b,c);seconds=time.perf_counter()-start
        plan=r.pop('plan');positive=plan>0
        residual=float(max(np.abs(plan.sum(1)-a).sum(),np.abs(plan.sum(0)-b).sum()))
        objective=float(np.sum(c*plan)+.001*np.sum(plan[positive]*(np.log(plan[positive])-1)))
        assert np.isfinite(plan).all() and np.all(plan>=0) and residual<=1e-12
        if r['diagnostics'] is not None:assert abs(objective-r['diagnostics']['rounded_primal'])<=1e-12
        row=dict(input=case,seconds=seconds,detail=r,audit_residual=residual,audit_objective=objective,
            net_to_first=float(plan[8:,:8].sum()-plan[:8,8:].sum()),warnings=sorted(set(str(w.message) for w in ws)))
        write(o/f"seed_{case['seed']}.json",row);rows.append(row)
        print(case['seed'],r['accepted'],r['blocks'],r['inner_work'],r['reached_reg'],r['diagnostics'],flush=True)
    write(o/'summary.json',dict(cases=rows));write(o/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file()})
if __name__=='__main__':main()
