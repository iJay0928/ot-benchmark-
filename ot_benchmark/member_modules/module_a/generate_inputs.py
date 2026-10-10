"""Freeze protocol and order before any formal solver result is inspected."""
import argparse
from datetime import datetime, timezone
import itertools
from pathlib import Path
import numpy as np
from core import ROOT, METHODS, array_sha, make_input, sha, write_json

def generate(output):
    output.mkdir(parents=True, exist_ok=False)
    protocol = dict(schema_version=1, frozen_utc=datetime.now(timezone.utc).isoformat(),
        upstream_commit='39d866c', development_seed=210001,
        seeds=[210101,210102,210103], n=[16,64], delta=[0.,1e-5,1e-4,1e-3],
        epsilon=[.01,.001], repeats=3, methods=list(METHODS), order_seed=210099,
        max_iterations=1000000, max_outer=1000, max_inner=1000, stopThr=1e-12,
        timeout_seconds=30., startup_timeout_seconds=60., gap_min=-1e-10,
        gap_max=1e-8, rounded_residual_limit=1e-12,
        reconstruction_rtol=1e-9, reconstruction_atol=1e-12,
        geometry='RNG reset for each (seed,n); same points for all deltas; two narrow 1D clusters',
        timing='solver plus external acceptance; startup/import/warmup/input load/file IO excluded; serial calls',
        grouping='48 dependent input-epsilon conditions; 144 method conditions; 432 repeated calls',
        source_sha256={p.name:sha(p) for p in ROOT.glob('*.py')})
    inputs = []
    for seed, n, delta in itertools.product(protocol['seeds'], protocol['n'], protocol['delta']):
        data = make_input(seed,n,delta)
        a,b,c = (data[k] for k in ('a','b','cost'))
        assert abs(a.sum()-1)<1e-14 and abs(b.sum()-1)<1e-14
        assert np.all(a>0) and np.all(b>0)
        assert abs((b[:n//2]-a[:n//2]).sum()-delta)<1e-14
        identifier = f's{seed}_n{n}_d{delta:g}'
        file = output/'inputs'/f'{identifier}.npz'; file.parent.mkdir(exist_ok=True)
        np.savez_compressed(file, **data)
        inputs.append(dict(id=identifier, seed=seed,n=n,delta=delta,
            sha256=array_sha(data), npz_sha256=sha(file), file=str(file.relative_to(output)),
            kernel_zero_counts={str(e):int(np.count_nonzero(np.exp(-c/e)==0)) for e in protocol['epsilon']}))
    tasks = []
    for inp, eps, method, repeat in itertools.product(inputs, protocol['epsilon'],METHODS,range(1,4)):
        tasks.append(dict(input_id=inp['id'], seed=inp['seed'],n=inp['n'],delta=inp['delta'],
                          epsilon=eps,method=method,repeat=repeat,input_sha256=inp['sha256']))
    np.random.default_rng(protocol['order_seed']).shuffle(tasks)
    for index, task in enumerate(tasks): task['task_id']=f'call_{index+1:04d}'
    write_json(output/'protocol.json',protocol)
    write_json(output/'input_manifest.json',inputs)
    write_json(output/'task_order.json',tasks)
    print(f'Frozen {len(inputs)} inputs, {len(tasks)} calls at {output}',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    generate(parser.parse_args().output)
