"""Frozen synthetic-trained selector evaluation on disjoint real observations."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
import zipfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from ot_benchmark.real_data import load_movielens_100k, load_uci_bag_of_words
from ot_benchmark.algorithm_selector import load_selector_model, feature_vector, select_training_algorithms
from ot_benchmark.algorithms.registry import ALGORITHMS
from ot_benchmark.benchmark_engine import benchmark_dataset, WeightConfig


def clean(value):
    if isinstance(value, dict):
        return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def write(path, value):
    path.write_text(json.dumps(clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=10)
    parser.add_argument('--repetitions', type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    bundle = load_selector_model(args.model)
    model = bundle['model']
    baseline = bundle['evaluation']['fixed_baseline']
    algorithms = select_training_algorithms(ALGORITHMS)
    if baseline not in algorithms or any(c not in algorithms for c in model.classes_):
        raise ValueError('model and candidate algorithms differ')
    # Freeze protocol and source before seeing real-data results.
    plan = dict(pairs_per_corpus=args.pairs, repetitions=args.repetitions,
                first_seed=41000, geometry_seed=42, weights=[.35,.30,.25,.10],
                algorithms=list(algorithms), baseline=baseline, model_sha256=sha(args.model),
                model_training='synthetic only; frozen; no refit on real observations',
                movielens=dict(n_users=300, n_items=400, min_ratings_per_user=5),
                bow=dict(dataset_name='kos', n_terms=500, max_docs=1000, min_terms_per_doc=20,
                         geometry_seed=42, compress_active=False),
                inclusion='first eligible pairs with no observation reused within corpus; no score-based selection',
                inference='conditional on shared corpus preprocessing; not independent corpora',
                python=sys.executable, python_version=platform.python_version(),
                threads={k: os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']})
    write(args.output/'protocol.json', plan)
    with zipfile.ZipFile(args.output/'source_snapshot.zip', 'w', zipfile.ZIP_DEFLATED) as archive:
        for p in list((ROOT/'ot_benchmark').rglob('*.py')) + [Path(__file__)]:
            archive.write(p, str(p.relative_to(ROOT)))
    weights = WeightConfig()
    rng = np.random.RandomState(51000)
    rows = []
    skipped = []
    for domain, loader, params in [('movielens', load_movielens_100k, plan['movielens']),
                                   ('kos', load_uci_bag_of_words, plan['bow'])]:
        used = set()
        accepted = 0
        for seed in range(41000, 42000):
            ds = loader(**params, random_state=seed)
            ids = {ds.source_observation_id, ds.target_observation_id}
            if len(ids) != 2 or '' in ids:
                raise ValueError('missing distinct observation IDs')
            if used & ids:
                skipped.append(dict(domain=domain, seed=seed, reason='observation reused'))
                continue
            if not ds.ground_truth_is_exact:
                raise ValueError('external pilot requires exact reference for every case')
            used.update(ids)
            started = time.perf_counter()
            vector = feature_vector(ds, weights)
            feature_seconds = time.perf_counter() - started
            started = time.perf_counter()
            prediction = str(model.predict(np.asarray([vector]))[0])
            predict_seconds = time.perf_counter() - started
            reports = []
            for repeat in range(args.repetitions):
                order = rng.permutation(list(algorithms))
                reports.append(benchmark_dataset(ds, {str(k): algorithms[k] for k in order},
                                                 weights=weights, verbose=False))
            scores = {k: float(np.mean([dict(r.overall_ranking).get(k, 0.) for r in reports])) for k in algorithms}
            oracle = max(scores, key=scores.get)
            outcomes = {}
            for name in algorithms:
                results = [next(r for r in report.results if r.algorithm_name == name) for report in reports]
                successful = [r for r in results if r.success]
                outcomes[name] = dict(successes=len(successful), attempts=len(results),
                                     mean_runtime_success=float(np.mean([r.runtime_seconds for r in successful])) if successful else None,
                                     mean_error_success=float(np.mean([r.distance_error for r in successful])) if successful else None)
            row = dict(domain=domain, seed=seed, source_id=ds.source_observation_id,
                       target_id=ds.target_observation_id, nominal_dimension=ds.dimension,
                       stored_dimension=len(ds.source_dist), geometry_dimension=ds.geometry_dim,
                       reference=ds.ground_truth_distance, reference_method=ds.ground_truth_method,
                       prediction=prediction, baseline=baseline, oracle=oracle,
                       feature_seconds=feature_seconds, predict_seconds=predict_seconds,
                       scores=scores, outcomes=outcomes,
                       selector_regret=scores[oracle]-scores[prediction], baseline_regret=scores[oracle]-scores[baseline])
            rows.append(row)
            write(args.output/f'{domain}_{accepted:03d}.json', dict(case=row, repeats=[asdict(r) for r in reports]))
            accepted += 1
            print(f'{domain}: {accepted}/{args.pairs}, predicted={prediction}, oracle={oracle}', flush=True)
            if accepted == args.pairs:
                break
        if accepted != args.pairs:
            raise RuntimeError('insufficient disjoint pairs')
    summaries = {}
    for domain in ['movielens', 'kos', 'all']:
        subset = [r for r in rows if domain == 'all' or r['domain'] == domain]
        summaries[domain] = dict(instances=len(subset),
            selector_accuracy=float(np.mean([r['prediction']==r['oracle'] for r in subset])),
            baseline_accuracy=float(np.mean([r['baseline']==r['oracle'] for r in subset])),
            selector_mean_score_regret=float(np.mean([r['selector_regret'] for r in subset])),
            baseline_mean_score_regret=float(np.mean([r['baseline_regret'] for r in subset])),
            selector_failed_runs=sum(r['outcomes'][r['prediction']]['attempts']-r['outcomes'][r['prediction']]['successes'] for r in subset),
            baseline_failed_runs=sum(r['outcomes'][baseline]['attempts']-r['outcomes'][baseline]['successes'] for r in subset))
    write(args.output/'summary.json', dict(summary=summaries, cases=rows, skipped=skipped,
          limitations=['small pilot; shared corpus preprocessing', 'unequal default iteration budgets',
                       'full cost matrix required for features', 'no end-to-end speedup claim']))
    cache = ROOT/'data_cache'
    write(args.output/'manifest.json', dict(
        model_sha256=sha(args.model),
        raw_data_sha256={str(p.relative_to(ROOT)):sha(p) for p in [cache/'ml-100k/u.data', cache/'ml-100k/u.item', cache/'docword.kos.txt.gz']},
        output_sha256={p.name:sha(p) for p in args.output.iterdir() if p.is_file() and p.name!='manifest.json'}))
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
