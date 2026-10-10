"""Dense first-passage observations, no solver or timing benchmark changes."""
import argparse, hashlib, json, sys, zipfile
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ot_benchmark.algorithms.transport_diagnostics import diagnose

def write(p, obj):
    p.write_text(json.dumps(obj, indent=2, allow_nan=False), encoding='utf-8')

def trajectory(a, b, c, reg, delta, path):
    k = np.exp(-c/reg); v = np.ones(len(b)); h = len(a)//2
    nets = np.empty(100000)
    for j in range(len(nets)):
        u = a/(k@v); v = b/(k.T@u)
        nets[j] = u[h:] @ k[h:, :h] @ v[:h] - u[:h] @ k[:h, h:] @ v[h:]
    assert np.isfinite(nets).all() and np.all(u > 0) and np.all(v > 0)
    np.save(path, nets)
    arrivals = []
    for fraction in [.1, .5, .9]:
        idx = np.flatnonzero(nets >= fraction*delta)
        i = int(idx[0]) if len(idx) else None
        arrivals.append(dict(fraction=fraction, iteration=None if i is None else i+1,
                             previous_net=None if i is None or i == 0 else float(nets[i-1]),
                             crossing_net=None if i is None else float(nets[i])))
    q, _ = diagnose(a, b, c, reg, np.log(u), np.log(v))
    return dict(arrivals=arrivals, final_net=float(nets[-1]), diagnostics=q,
                final_accepted=bool(q['rounded_residual'] <= 1e-12 and -1e-10 <= q['numerical_gap'] <= 1e-8), trace=path.name)

def main():
    p = argparse.ArgumentParser(); p.add_argument('--output', type=Path, required=True)
    o = p.parse_args().output; o.mkdir(parents=True, exist_ok=False)
    protocol = dict(seeds=[139001,139002], widths=[0.,.001,.02,.1], deltas=[1e-4,1e-3], regs=[1/32,1/128], n=16, budget=100000, fractions=[.1,.5,.9], role='development, known cluster labels')
    write(o/'protocol.json', protocol)
    with zipfile.ZipFile(o/'source_snapshot.zip', 'x', zipfile.ZIP_DEFLATED) as z:
        for f in [Path(__file__), ROOT/'tools/audit_cluster_arrival.py', ROOT/'research/跨簇首次到达时间冻结方案.md', ROOT/'ot_benchmark/algorithms/transport_diagnostics.py']:
            z.write(f, f.relative_to(ROOT))
    inputs = []
    for seed in protocol['seeds']:
        r = np.random.default_rng(seed).uniform(0,1,16)
        for w in protocol['widths']:
            x = np.r_[w*r[:8], 1-w+w*r[8:]]; c = (x[:,None]-x[None,:])**2; c /= c.max()
            for delta in protocol['deltas']:
                a = np.ones(16)/16; b = a.copy(); b[:8] += delta/8; b[8:] -= delta/8
                name = f'input_{len(inputs):02d}.npz'; np.savez_compressed(o/name, a=a,b=b,cost=c,points=x)
                inputs.append(dict(seed=seed,width=w,delta=delta,file=name,sha256=hashlib.sha256((o/name).read_bytes()).hexdigest()))
    write(o/'inputs.json', inputs)
    coarse = {}
    for delta in protocol['deltas']:
        for reg in protocol['regs']:
            coarse[delta,reg] = trajectory(np.array([.5,.5]),np.array([.5+delta,.5-delta]),np.array([[0.,1.],[1.,0.]]),reg,delta,o/f'coarse_{delta}_{reg}.npy')
    rows = []
    for inp in inputs:
        with np.load(o/inp['file']) as z: a,b,c = z['a'],z['b'],z['cost']
        for reg in protocol['regs']:
            fine = trajectory(a,b,c,reg,inp['delta'],o/f'fine_{len(rows):02d}.npy')
            row = dict(input=inp,reg=reg,fine=fine,coarse=coarse[inp['delta'],reg])
            write(o/f'run_{len(rows):02d}.json',row); rows.append(row)
            print(inp['seed'],inp['width'],inp['delta'],reg,[a['iteration'] for a in fine['arrivals']],flush=True)
    write(o/'summary.json',dict(rows=rows))
    write(o/'manifest.json',{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in o.iterdir() if f.is_file()})

if __name__ == '__main__': main()
