"""Freeze unused real observation pairs before inspecting solver performance."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,json,sys,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import numpy as np
from ot_benchmark.real_data import load_movielens_100k,load_uci_bag_of_words


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--first-seed',type=int,default=105000);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def write(path,obj):path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    raw=[ROOT/'data_cache/ml-100k/u.data',ROOT/'data_cache/ml-100k/u.item',ROOT/'data_cache/docword.kos.txt.gz']
    if not all(p.is_file() for p in raw):raise RuntimeError('required cached source missing')
    used=set();sources={}
    def collect(obj):
        if isinstance(obj,dict):
            for v in obj.values():collect(v)
        elif isinstance(obj,list):
            for v in obj:collect(v)
        elif isinstance(obj,str) and obj.startswith(('movielens:user:','uci:kos:document:')):used.add(obj)
    # All saved summaries, not only the latest round, contribute exclusions.
    for source in sorted((ROOT/'validation').rglob('summary.json')):
        before=len(used);collect(json.loads(source.read_text(encoding='utf-8')))
        if len(used)>before:sources[str(source.relative_to(ROOT))]=sha(source)
    exclusions=sorted(used)
    params=dict(movielens=dict(n_users=300,n_items=400,min_ratings_per_user=5),
        kos=dict(dataset_name='kos',n_terms=500,max_docs=1000,min_terms_per_doc=20,geometry_seed=42,compress_active=False))
    protocol=dict(role='input-only freeze; no algorithm timing or output inspected',pairs_per_corpus=6,
        first_seed=args.first_seed,params=params,prior_ids=exclusions,prior_summary_hashes=sources,
        raw_sha256={str(p.relative_to(ROOT)):sha(p) for p in raw},
        selection='first six eligible pairs per corpus with distinct unused observation IDs; reject reuse only',
        normalization='preserve loader weights and their sums; renormalize positive totals exactly as quality solver does; loader exact-reference output not used for selection',
        limitation='shared corpus preprocessing and learned KOS geometry; not independent corpora or downstream task accuracy')
    write(args.output/'protocol.json',protocol)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:
        for p in [Path(__file__),ROOT/'ot_benchmark/real_data.py',ROOT/'ot_benchmark/data_generator.py']:z.write(p,p.relative_to(ROOT))
    cases=[];skipped=[]
    for domain,loader in [('movielens',load_movielens_100k),('kos',load_uci_bag_of_words)]:
        accepted=0
        for seed in range(args.first_seed,args.first_seed+1000):
            ds=loader(**params[domain],random_state=seed)
            ids={ds.source_observation_id,ds.target_observation_id}
            if len(ids)!=2 or '' in ids:raise ValueError('invalid observation identifiers')
            if used & ids:
                skipped.append(dict(domain=domain,seed=seed,reason='previously used observation'));continue
            a,b,c=ds.source_dist,ds.target_dist,ds.cost_matrix
            assert np.isfinite(a).all() and np.isfinite(b).all() and np.isfinite(c).all()
            assert np.all(a>=0) and np.all(b>=0) and np.all(c>=0)
            loader_a,loader_b=a.copy(),b.copy();sums=[float(a.sum()),float(b.sum())]
            assert sums[0]>0 and sums[1]>0
            a=a/sums[0];b=b/sums[1]
            assert c.shape==(len(a),len(b)) and abs(a.sum()-1)<1e-12 and abs(b.sum()-1)<1e-12
            filename=f'{domain}_{accepted:03d}.npz'
            np.savez_compressed(args.output/filename,a=a,b=b,cost=c,loader_a=loader_a,loader_b=loader_b)
            cases.append(dict(domain=domain,seed=seed,file=filename,source_id=ds.source_observation_id,
                target_id=ds.target_observation_id,active_source=int(np.count_nonzero(a)),active_target=int(np.count_nonzero(b)),
                stored_shape=list(c.shape),loader_mass_sums=sums,geometry=ds.geometry_description,sha256=sha(args.output/filename)))
            used.update(ids);accepted+=1
            print(domain,accepted,seed,flush=True)
            if accepted==6:break
        if accepted!=6:raise RuntimeError('not enough unused observations')
    write(args.output/'summary.json',dict(cases=cases,skipped=skipped,prior_ids_count=len(exclusions)))
    write(args.output/'manifest.json',{p.name:sha(p) for p in args.output.iterdir() if p.is_file()})


if __name__=='__main__':main()
