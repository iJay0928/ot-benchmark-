"""Development-only cooling traces on the two frozen failed block cases."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,json,sys,time,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import ot
from ot_benchmark.algorithms.transport_diagnostics import diagnose,round_plan

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);o=p.parse_args().output;o.mkdir(parents=True,exist_ok=False)
    inputs=ROOT/'validation/2026-10-09/numeric_boundary_inputs_round33'
    cases=[c for c in json.loads((inputs/'summary.json').read_text())['cases'] if c['domain']=='near_balanced_blocks']
    for c in cases:assert hashlib.sha256((inputs/c['file']).read_bytes()).hexdigest()==c['sha256']
    write(o/'protocol.json',dict(role='post-hoc development on both round33 failures; single trajectory each, no new validation',cases=cases,
        paces=[1,10],schedule='(1-reg)*exp(-i/pace)+reg',reg=.001,inner=1000,outer=1000,stopThr=1e-22,
        quality='check final-reg gap every10 at i>=35 and final; same [-1e-10,1e-8] and feasibility1e-12',
        observation='record current-reg quality and signed cross-cluster transport each stage; no state changes by diagnostics',
        limitation='same maximum outer and inner work but no claim of equal actual work or optimal schedule; profiling times are not benchmark results',
        pot_version=ot.__version__,threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']}))
    with zipfile.ZipFile(o/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    summaries=[]
    for case in cases:
        with np.load(inputs/case['file'],allow_pickle=False) as z:a,b,c=z['a'],z['b'],z['cost']
        for pace in [1,10]:
            alpha,beta=np.zeros(len(a)),np.zeros(len(b));trace=[];work=0;accepted=False;start=time.perf_counter()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                for i in range(1000):
                    reg=(1-.001)*np.exp(-i/pace)+.001
                    raw,log=ot.bregman.sinkhorn_stabilized(a,b,c,reg,numItermax=1000,stopThr=1e-22,warmstart=(alpha,beta),verbose=False,print_period=20,tau=1e3,log=True)
                    alpha,beta=log['alpha'],log['beta'];executed=int(log['n_iter'])+1;work+=executed
                    q,_=diagnose(a,b,c,reg,alpha/reg,beta/reg)
                    row=dict(outer=i+1,reg=float(reg),inner=executed,work=work,current_gap=q['numerical_gap'],raw_residual=q['raw_residual'],
                        net_to_first=float(raw[8:,:8].sum()-raw[:8,8:].sum()),required_net=float(b[:8].sum()-a[:8].sum()))
                    if i>=35 and (i%10==0 or i==999):
                        try:
                            target,reconstructed=diagnose(a,b,c,.001,alpha/.001,beta/.001)
                            accepted=target['rounded_residual']<=1e-12 and -1e-10<=target['numerical_gap']<=1e-8
                            row.update(target_gap=target['numerical_gap'],accepted=bool(accepted))
                        except ValueError as exc:row['target_diagnostic_error']=str(exc)
                    trace.append(row)
                    if accepted:break
                target,reconstructed=diagnose(a,b,c,.001,alpha/.001,beta/.001);plan=round_plan(reconstructed,a,b)
                accepted=target['rounded_residual']<=1e-12 and -1e-10<=target['numerical_gap']<=1e-8
                seconds=time.perf_counter()-start
            pos=plan>0;obj=float(np.sum(plan*c)+.001*np.sum(plan[pos]*(np.log(plan[pos])-1)))
            assert abs(obj-target['rounded_primal'])<=1e-12
            if pace==1:
                old=json.loads((ROOT/f"validation/2026-10-09/numeric_boundary_round33/near_balanced_blocks_reg_0.001_{case['seed']}_legacy_fallback_0.json").read_text())
                assert abs(target['numerical_gap']-old['detail']['diagnostics']['numerical_gap'])<=1e-12
            result=dict(seed=case['seed'],pace=pace,outer=i+1,inner_work=work,accepted=bool(accepted),seconds=seconds,diagnostics=target,
                rounded_net_to_first=float(plan[8:,:8].sum()-plan[:8,8:].sum()),trace=trace,warnings=sorted(set(str(w.message) for w in caught)))
            write(o/f"seed_{case['seed']}_pace_{pace}.json",result)
            summaries.append({k:v for k,v in result.items() if k not in ('trace','warnings')})
            print(case['seed'],pace,i+1,work,accepted,target['numerical_gap'],flush=True)
    write(o/'summary.json',dict(rows=summaries));write(o/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file()})
if __name__=='__main__':main()
