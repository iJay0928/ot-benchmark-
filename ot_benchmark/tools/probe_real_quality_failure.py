"""Post-hoc bounded budget probe of one observed real failure; not holdout validation."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,json,sys,time,warnings,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
import ot
from ot_benchmark.algorithms.transport_diagnostics import diagnose


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    def write(name,obj):(args.output/name).write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')
    inp=ROOT/'validation/2026-10-06/quality_real_inputs_normalized_round17/movielens_001.npz'
    configs=[dict(outer=1000,inner=1000),dict(outer=200,inner=5000)]
    write('protocol.json',dict(role='post-hoc diagnosis of MovieLens 105002 reg .001; original failed result retained',
        input_sha256=hashlib.sha256(inp.read_bytes()).hexdigest(),configs=configs,repetitions=1,
        epsilon0=1.,internal_stop=1e-22,gap_tolerance=1e-8,feasibility=1e-12,
        limitation='two cold budget expansions only; exploratory elapsed time not a comparative benchmark'))
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    with np.load(inp) as z:a,b,c=z['a'],z['b'],z['cost']
    ia,ib=np.flatnonzero(a),np.flatnonzero(b);a,b,c=a[ia],b[ib],c[np.ix_(ia,ib)];a/=a.sum();b/=b.sum()
    rows=[]
    for cfg in configs:
        start=time.perf_counter()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            _,log=ot.bregman.sinkhorn_epsilon_scaling(a,b,c,.001,numItermax=cfg['outer'],numInnerItermax=cfg['inner'],epsilon0=1.,stopThr=1e-22,log=True,warn=True)
            diag,_=diagnose(a,b,c,.001,log['alpha']/.001,log['beta']/.001)
        row=dict(config=cfg,seconds=time.perf_counter()-start,outer_iterations=int(log['niter'])+1,diagnostics=diag,
            accepted=diag['rounded_residual']<=1e-12 and -1e-10<=diag['numerical_gap']<=1e-8,
            warnings=sorted(set(str(w.message) for w in caught)))
        rows.append(row);write(f"outer_{cfg['outer']}_inner_{cfg['inner']}.json",row);print(row,flush=True)
    write('summary.json',dict(runs=rows))
    write('manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output.iterdir() if f.is_file()})


if __name__=='__main__':main()
