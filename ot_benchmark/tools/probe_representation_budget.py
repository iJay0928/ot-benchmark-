"""Frozen-state development probe. No solver timing or held-out claim."""
import argparse, hashlib, json, sys, zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ot_benchmark.algorithms.representation_budget import representation_budget


def write(p,obj):
    p.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    files=sorted(args.inputs.glob('*_state.npz'))
    manifest=json.loads((args.inputs/'manifest.json').read_text())
    for f in files:assert hashlib.sha256(f.read_bytes()).hexdigest()==manifest[f.name]
    write(args.output/'protocol.json',dict(role='post-hoc development, all four round19 saved states',
        input_hashes={f.name:manifest[f.name] for f in files},reg=.001,quality_tolerance=1e-8,
        decision='offline only; no threshold selection and no solver modification',
        sources=['https://arxiv.org/pdf/quant-ph/0610146','https://arxiv.org/pdf/1610.06519']))
    with zipfile.ZipFile(args.output/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/representation_budget.py',ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:
            z.write(f,f.relative_to(ROOT))
    rows=[]
    for f in files:
        with np.load(f,allow_pickle=False) as s:
            c,u,v=s['cost'],s['u'],s['v'];k=np.exp(-c/.001)
            direct=u[:,None]*k*v[None,:]
            reconstructed=np.exp(np.log(u)[:,None]+np.log(v)[None,:]-c/.001)
            d=representation_budget(s['a'],s['b'],c,.001,direct,reconstructed)
            assert d['observed_gap_distortion']<=d['gap_distortion_bound']+1e-10
            rows.append(dict(state=f.name,kernel_zeros=int((k==0).sum()),l1_discrepancy=float(np.abs(direct-reconstructed).sum()),**d))
    write(args.output/'summary.json',rows)
    write(args.output/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in args.output.iterdir() if f.is_file()})
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
