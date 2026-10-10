"""Input-only synthetic boundary freeze; no solver-based selection."""
import argparse,hashlib,json,zipfile
from pathlib import Path
import numpy as np

def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False),encoding='utf-8')
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);o=p.parse_args().output;o.mkdir(parents=True,exist_ok=False)
    write(o/'protocol.json',dict(role='new synthetic boundary inputs, no solver outcomes inspected',seeds=[113001,113002],n=16,reg=[.01,.001],
        families=['balanced_blocks','near_balanced_blocks','dynamic_mass'],delta=1e-4,
        generation='two narrow 1D clusters for blocks, squared distance normalized by max; uniform equal masses or cluster transfer delta; dynamic masses 10**uniform(-6,0) on shared uniform 1D support',
        limitations='designed stress families, not representative real population; shared seeds and paired regularizations dependent; no guarantee of fallback activation'))
    with zipfile.ZipFile(o/'source_snapshot.zip','x',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
    cases=[]
    for seed in [113001,113002]:
        for family in ['balanced_blocks','near_balanced_blocks','dynamic_mass']:
            rng=np.random.default_rng(seed)
            if family=='dynamic_mass':
                x=np.sort(rng.uniform(0,1,16));a=10.**rng.uniform(-6,0,16);b=10.**rng.uniform(-6,0,16);a/=a.sum();b/=b.sum()
            else:
                x=np.r_[rng.uniform(0,.02,8),rng.uniform(.98,1.,8)];a=np.ones(16)/16;b=a.copy()
                if family=='near_balanced_blocks':b[:8]+=1e-4/8;b[8:]-=1e-4/8
            c=(x[:,None]-x[None,:])**2;c/=c.max();f=o/f'{family}_{seed}.npz';np.savez_compressed(f,a=a,b=b,cost=c,points=x)
            cases.append(dict(domain=family,seed=seed,file=f.name,sha256=hashlib.sha256(f.read_bytes()).hexdigest(),active_source=16,active_target=16))
    write(o/'summary.json',dict(cases=cases));write(o/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file()})
if __name__=='__main__':main()
