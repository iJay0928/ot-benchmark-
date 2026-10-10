"""Known-failure fallback probe with exact old-trajectory comparison."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,json,sys,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ot_benchmark.algorithms.gated_epsilon import solve_gated_epsilon
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);o=p.parse_args().output;o.mkdir(parents=True,exist_ok=False)
    inp=ROOT/'validation/2026-10-09/cooling_boundary_round37'
    source=inp/'summary.json';oldrows=json.loads(source.read_text())['rows']
    selected=[r for r in oldrows if r['job']['method']=='gated' and not r['detail']['accepted']]
    write(o/'protocol.json',dict(role='four known failed conditions, development only',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),jobs=[r['job'] for r in selected],keep_best=True,threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
    with zipfile.ZipFile(o/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/gated_epsilon.py',ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    rows=[]
    for index,old in enumerate(selected):
        job=old['job'];case=job['case'];f=inp/case['file'];assert hashlib.sha256(f.read_bytes()).hexdigest()==case['sha256']
        with np.load(f,allow_pickle=False) as z:a,b,c=z['a'],z['b'],z['cost']
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always');new=solve_gated_epsilon(a,b,c,reg=job['reg'],keep_best=True)
        plan=new.pop('plan');q=new.pop('returned_plan_quality')
        assert new==old['detail'] # Includes every stage, work and original diagnostic.
        mask=plan>0;obj=float(np.sum(c*plan)+job['reg']*np.sum(plan[mask]*(np.log(plan[mask])-1)))
        residual=float(max(np.abs(plan.sum(1)-a).sum(),np.abs(plan.sum(0)-b).sum()))
        assert np.isfinite(plan).all() and np.all(plan>=0) and residual<=1e-12
        assert abs(obj-q['rounded_primal'])<=1e-12 and obj<=old['audit_objective']+1e-12
        assert q['numerical_gap']>=-1e-10 and not new['accepted']
        np.savez_compressed(o/f'plan_{index}.npz',plan=plan)
        stages=[]
        for reg in sorted({t['reg'] for t in new['trace']},reverse=True):
            ts=[t for t in new['trace'] if t['reg']==reg]
            stages.append(dict(reg=reg,blocks=len(ts),work=sum(t['executed'] for t in ts),final_gap=ts[-1]['gap'],final_raw_residual=ts[-1]['raw_residual']))
        row=dict(job=job,original_detail=new,returned_plan_quality=q,trajectory_identical=True,audit_objective=obj,audit_residual=residual,stages=stages,warnings=sorted(set(str(w.message) for w in ws)))
        write(o/f'run_{index}.json',row);rows.append(row);print(case['seed'],job['reg'],q,flush=True)
    write(o/'summary.json',dict(rows=rows));write(o/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file()})
if __name__=='__main__':main()
