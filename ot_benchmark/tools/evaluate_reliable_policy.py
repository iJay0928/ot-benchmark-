"""Development and untouched-observation tests for a bounded OT fallback policy."""
import argparse
import hashlib
import json
import os
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from ot_benchmark.real_data import load_movielens_100k, load_uci_bag_of_words
from ot_benchmark.algorithms.reliable_ot import solve_reliable_ot
from ot_benchmark.algorithms.sinkhorn import run_sinkhorn
from ot_benchmark.algorithms.sparse_ot import run_sparse_sinkhorn
from evaluate_real_selector import clean, write, sha


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=['development','holdout'], required=True)
    parser.add_argument('--pairs', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    old = json.loads((args.previous/'summary.json').read_text(encoding='utf-8'))
    old_plan = json.loads((args.previous/'protocol.json').read_text(encoding='utf-8'))
    excluded = {id for r in old['cases'] for id in (r['source_id'],r['target_id'])}
    methods = ['dense_log', 'legacy_sparse', 'long_sparse_then_log', 'bounded_sparse_then_log']
    protocol = dict(mode=args.mode, repetitions=args.repetitions, pairs_per_domain=args.pairs,
                    methods=methods, reg=.01, sparse_budgets={'long':2000,'bounded':200},
                    fallback_iterations=10000, feasibility_tolerance=1e-7,
                    reference_cost_agreement='absolute difference <= 1e-4 + 1e-3 * abs(dense_log cost)',
                    first_holdout_seed=61000, excluded_observations=sorted(excluded),
                    timing='solver wall time including failed attempts, fallback, validation and kernel build; excludes common input preprocessing',
                    common_preprocessing={k:old_plan[k] for k in ['movielens','bow']},
                    selection='no score-based inclusion; no model training; fixed 200-iteration policy',
                    shared_corpus_caveat='disjoint observations, shared corpus preprocessing',
                    threads={k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    write(args.output/'protocol.json',protocol)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in list((ROOT/'ot_benchmark').rglob('*.py'))+[Path(__file__),ROOT/'tools/evaluate_real_selector.py']:
            archive.write(p,str(p.relative_to(ROOT)))
    rng = np.random.RandomState(62000)
    rows = []
    used = set(excluded) if args.mode == 'holdout' else set()
    for domain, loader in [('movielens',load_movielens_100k),('kos',load_uci_bag_of_words)]:
        prior = [r for r in old['cases'] if r['domain']==domain]
        dev = prior[:2]+[r for r in prior[2:] if r['outcomes']['Sparse Sinkhorn']['successes']<3]
        seeds = [r['seed'] for r in dev] if args.mode=='development' else range(61000,63000)
        limit = len(seeds) if args.mode=='development' else args.pairs
        accepted = 0
        for seed in seeds:
            ds = loader(**old_plan['movielens' if domain=='movielens' else 'bow'],random_state=seed)
            ids = {ds.source_observation_id, ds.target_observation_id}
            if used & ids:
                continue
            if len(ids)!=2 or '' in ids or not ds.ground_truth_is_exact:
                raise ValueError('invalid identity or unavailable exact reference')
            used.update(ids)
            records = []
            # Warm up every method once on a fixed toy case, outside all timings.
            if not rows:
                toy=np.ones(2)/2; cost=np.array([[0.,.1],[.1,0.]])
                run_sinkhorn(toy,toy,cost,reg=.01)
                solve_reliable_ot(toy,toy,cost)
                run_sparse_sinkhorn(toy,toy,cost,reg=.01)
            for repeat in range(args.repetitions):
                for method in rng.permutation(methods):
                    started=time.perf_counter()
                    try:
                        kwargs=dict(source=ds.source_dist,target=ds.target_dist,cost_matrix=ds.cost_matrix,reg=.01)
                        if method=='dense_log':
                            value=run_sinkhorn(**kwargs,max_iter=10000)
                            detail=dict(path='dense_log')
                        elif method=='legacy_sparse':
                            value=run_sparse_sinkhorn(**kwargs,cost_matrix_csr=ds.cost_matrix_csr,max_iter=2000)
                            detail=dict(path='legacy_internal_path_unrecorded')
                        else:
                            result=solve_reliable_ot(**kwargs,sparse_max_iter=200 if method=='bounded_sparse_then_log' else 2000)
                            value=result.cost
                            detail=asdict(result)
                        elapsed=time.perf_counter()-started
                        records.append(dict(method=str(method),repeat=repeat,success=True,cost=value,
                                            seconds=elapsed,detail=detail))
                    except (ValueError,FloatingPointError,RuntimeError) as exc:
                        records.append(dict(method=str(method),repeat=repeat,success=False,cost=None,
                                            seconds=time.perf_counter()-started,error=str(exc)))
            metrics={}
            dense_values=[r['cost'] for r in records if r['method']=='dense_log' and r['success']]
            dense_cost=float(np.median(dense_values)) if dense_values else None
            for method in methods:
                batch=[r for r in records if r['method']==method]
                successful=[r for r in batch if r['success']]
                value=float(np.median([r['cost'] for r in successful])) if successful else None
                metrics[method]=dict(successes=len(successful),attempts=len(batch),
                    median_seconds=float(np.median([r['seconds'] for r in batch])),cost=value,
                    error_to_exact=abs(value-ds.ground_truth_distance)/max(abs(ds.ground_truth_distance),1e-12) if value is not None else None,
                    absolute_difference_to_dense=abs(value-dense_cost) if value is not None and dense_cost is not None else None,
                    agrees_with_dense=(abs(value-dense_cost)<=1e-4+1e-3*abs(dense_cost)) if value is not None and dense_cost is not None else None,
                    fallbacks=sum(r.get('detail',{}).get('path')=='log_fallback' for r in batch))
            row=dict(domain=domain,seed=seed,source_id=ds.source_observation_id,target_id=ds.target_observation_id,
                     reference=ds.ground_truth_distance,reference_method=ds.ground_truth_method,metrics=metrics)
            write(args.output/f'{domain}_{accepted:03d}.json',dict(case=row,runs=records))
            rows.append(row); accepted+=1
            print(f'{domain} {accepted}/{limit}: bounded {metrics[methods[-1]]}',flush=True)
            if accepted==limit:
                break
        if accepted!=limit:
            raise RuntimeError('not enough eligible disjoint pairs')
    summary={}
    for domain in ['movielens','kos','all']:
        subset=[r for r in rows if domain=='all' or r['domain']==domain]
        summary[domain]={}
        for method in methods:
            metric=[r['metrics'][method] for r in subset]
            paired=[r for r in subset if r['metrics'][method]['successes']==args.repetitions
                    and r['metrics']['bounded_sparse_then_log']['successes']==args.repetitions]
            summary[domain][method]=dict(instances=len(metric),runs=sum(m['attempts'] for m in metric),
                successes=sum(m['successes'] for m in metric),
                sum_case_median_seconds=sum(m['median_seconds'] for m in metric),
                dense_agreement_cases=sum(m['agrees_with_dense'] is True for m in metric),
                fallback_runs=sum(m['fallbacks'] for m in metric),
                paired_success_cases=len(paired),
                median_speedup_of_bounded=float(np.median([r['metrics'][method]['median_seconds']/r['metrics']['bounded_sparse_then_log']['median_seconds'] for r in paired])) if paired else None)
    write(args.output/'summary.json',dict(summary=summary,cases=rows))
    write(args.output/'manifest.json',dict(previous_summary_sha256=sha(args.previous/'summary.json'),
          output_sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(clean(summary),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
