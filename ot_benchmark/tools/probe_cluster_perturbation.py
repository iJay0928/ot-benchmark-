"""Controlled fine/coarse trajectory comparison; not a speed benchmark."""
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from ot_benchmark.algorithms.transport_diagnostics import diagnose
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def trajectory(a,b,c,reg):
    k=np.exp(-c/reg);v=np.ones(len(b));out=[]
    for i in range(1,20001):
        u=a/(k@v);v=b/(k.T@u)
        if i in [1,10,100,1000,10000,20000]:
            assert np.isfinite(u).all() and np.isfinite(v).all() and np.all(u>0) and np.all(v>0)
            p=u[:,None]*k*v[None,:];h=len(a)//2
            block=np.array([[p[:h,:h].sum(),p[:h,h:].sum()],[p[h:,:h].sum(),p[h:,h:].sum()]])
            q,reconstructed=diagnose(a,b,c,reg,np.log(u),np.log(v))
            out.append(dict(iteration=i,block=block.tolist(),net=float(block[1,0]-block[0,1]),diagnostics=q,reconstruction_l1=float(np.abs(p-reconstructed).sum())))
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);o=p.parse_args().output;o.mkdir(parents=True,exist_ok=False)
    protocol=dict(role='controlled mechanism development, known cluster labels',seeds=[139001,139002],widths=[0.,.001,.02],deltas=[0.,1e-5,1e-3],regs=[1/32,1/128],n=16,iterations=20000,checkpoints=[1,10,100,1000,10000,20000],zero_width_tolerance=1e-10)
    write(o/'protocol.json',protocol)
    with zipfile.ZipFile(o/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__),ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:z.write(f,f.relative_to(ROOT))
    inputs=[]
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);r=rng.uniform(0,1,16)
        for w in protocol['widths']:
            x=np.r_[w*r[:8],1-w+w*r[8:]];c=(x[:,None]-x[None,:])**2;c/=c.max()
            for delta in protocol['deltas']:
                a=np.ones(16)/16;b=a.copy();b[:8]+=delta/8;b[8:]-=delta/8
                name=f'input_{len(inputs):02d}.npz';np.savez_compressed(o/name,a=a,b=b,cost=c,points=x)
                inputs.append(dict(seed=seed,width=w,delta=delta,file=name,sha256=hashlib.sha256((o/name).read_bytes()).hexdigest()))
    write(o/'inputs.json',inputs)
    coarse={}
    for delta in protocol['deltas']:
        for reg in protocol['regs']:coarse[(delta,reg)]=trajectory(np.array([.5,.5]),np.array([.5+delta,.5-delta]),np.array([[0.,1.],[1.,0.]]),reg)
    rows=[]
    for inp in inputs:
        with np.load(o/inp['file'],allow_pickle=False) as z:a,b,c=z['a'],z['b'],z['cost']
        for reg in protocol['regs']:
            fine=trajectory(a,b,c,reg);reference=coarse[(inp['delta'],reg)];errors=[float(np.max(np.abs(np.array(f['block'])-np.array(g['block'])))) for f,g in zip(fine,reference)]
            if inp['width']==0:assert max(errors)<=1e-10
            final=fine[-1]['diagnostics'];row=dict(input=inp,reg=reg,fine=fine,coarse=reference,block_max_errors=errors,final_accepted=bool(final['rounded_residual']<=1e-12 and -1e-10<=final['numerical_gap']<=1e-8))
            write(o/f'run_{len(rows):02d}.json',row);rows.append(row)
            print(inp['seed'],inp['width'],inp['delta'],reg,row['final_accepted'],max(errors),flush=True)
    write(o/'summary.json',dict(rows=rows));write(o/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file()})
if __name__=='__main__':main()
