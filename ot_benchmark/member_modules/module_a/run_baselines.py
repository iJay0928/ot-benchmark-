"""Serial warmed fresh workers, terminable 30s solver+acceptance budgets."""
import argparse
import csv
import inspect
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path
from core import ROOT, METHODS, array_sha, clean, execute, make_input, ot, np, sha, write_json

def worker(connection, data, task, protocol):
    warm = make_input(210001,4,0.)
    execute(warm,.1,task['method'],protocol)
    connection.send('READY')
    connection.recv()
    payload = execute(data,task['epsilon'],task['method'],protocol)
    connection.send(payload)
    connection.close()

def one(data,task,protocol):
    parent,child = mp.get_context('spawn').Pipe()
    proc = mp.get_context('spawn').Process(target=worker,args=(child,data,task,protocol))
    started=time.perf_counter();proc.start();child.close()
    try:
        if not parent.poll(protocol['startup_timeout_seconds']):
            return dict(status='STARTUP_ERROR',error='worker not ready',timed_out=False),{},{}
        if parent.recv()!='READY': raise RuntimeError('invalid worker handshake')
        startup=time.perf_counter()-started;parent.send('GO');begin=time.perf_counter()
        if not parent.poll(protocol['timeout_seconds']):
            row=dict(status='TIMEOUT',timed_out=True,elapsed_seconds=None,
                     solve_seconds=None,audit_seconds=None,actual_iterations=None,
                     inner_iterations=None,warnings=[],censor_seconds=protocol['timeout_seconds'],
                     error='terminated at common wall-clock deadline')
            arrays,log={},{}
        else: row,arrays,log=parent.recv()
        row['startup_warmup_seconds']=startup
        row['parent_observed_seconds']=time.perf_counter()-begin
        return row,arrays,log
    except (EOFError, OSError) as exc:
        return dict(status='WORKER_ERROR',error=str(exc),timed_out=False),{},{}
    finally:
        if proc.is_alive(): proc.terminate()
        proc.join(timeout=5)
        if proc.is_alive(): proc.kill();proc.join()
        parent.close()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int);args=parser.parse_args();out=args.output
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    if ot.__version__!='0.9.6.post1': raise RuntimeError('POT must be 0.9.6.post1')
    # Fail closed if solver/generator code changed after freezing.
    for name in ('core.py','generate_inputs.py','run_baselines.py'):
        if sha(ROOT/name)!=protocol['source_sha256'][name]: raise RuntimeError('frozen code mismatch: '+name)
    import scipy,psutil
    env=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,POT=ot.__version__,
             os=platform.platform(),cpu=platform.processor(),logical_cpu=psutil.cpu_count(),
             physical_cpu=psutil.cpu_count(logical=False),memory_bytes=psutil.virtual_memory().total,
             pot_source_sha256=sha(inspect.getfile(ot.bregman.sinkhorn_knopp)),threads=1)
    write_json(out/'environment.json',env)
    tasks=json.loads((out/'task_order.json').read_text(encoding='utf-8'))
    manifest={i['id']:i for i in json.loads((out/'input_manifest.json').read_text(encoding='utf-8'))}
    results=out/'results';results.mkdir(exist_ok=True)
    done=set(p.stem for p in (results/'logs').glob('*.json')) if (results/'logs').exists() else set()
    for task in tasks[:args.limit]:
        if task['task_id'] in done: continue
        inp=manifest[task['input_id']]
        with np.load(out/inp['file']) as z: data={k:z[k] for k in z.files}
        assert array_sha(data)==task['input_sha256']
        row,arrays,log=one(data,task,protocol);row={**task,**row}
        row['protocol_sha256']=sha(out/'protocol.json')
        row['kernel_zero_count']=inp['kernel_zero_counts'][str(task['epsilon'])]
        saved=time.perf_counter()
        matrices={k:v for k,v in arrays.items() if v is not None}
        if matrices:
            (results/'plans').mkdir(exist_ok=True)
            np.savez_compressed(results/'plans'/f"{task['task_id']}.npz",**matrices)
        row['artifact_write_seconds']=time.perf_counter()-saved
        write_json(results/'logs'/f"{task['task_id']}.json",dict(result=row,native_log=log))
        print(f"{task['task_id']}/{len(tasks)} {task['method']} n={task['n']} d={task['delta']:g} eps={task['epsilon']:g} {row['status']} {row.get('elapsed_seconds')}",flush=True)
    rows=[json.loads(p.read_text(encoding='utf-8'))['result'] for p in sorted((results/'logs').glob('*.json'))]
    write_json(results/'summary.json',rows)
    fields=sorted(set().union(*(r.keys() for r in rows)))
    with (results/'summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fields);writer.writeheader()
        writer.writerows({k:json.dumps(clean(v),ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows)
    print(f'Saved {len(rows)}/{len(tasks)} calls',flush=True)

if __name__=='__main__':main()
