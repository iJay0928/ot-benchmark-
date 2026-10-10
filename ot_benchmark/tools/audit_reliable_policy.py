"""Read-only checks of frozen fallback-policy evidence and observation separation."""
import argparse
import hashlib
import json
from collections import Counter


def audit(directory, previous, extra=None):
    def read(path):
        return json.loads(path.read_text(encoding='utf-8'))
    manifest = read(directory / 'manifest.json')
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest(previous / 'summary.json') == manifest['previous_summary_sha256']
    for name, expected in manifest['output_sha256'].items():
        assert digest(directory / name) == expected, name
    protocol = read(directory / 'protocol.json')
    summary = read(directory / 'summary.json')
    old_ids = {row[key] for row in read(previous / 'summary.json')['cases']
               for key in ('source_id', 'target_id')}
    if 'extra_exclusion_summary_sha256' in manifest:
        assert extra is not None, 'extra exclusion evidence required'
        assert digest(extra / 'summary.json') == manifest['extra_exclusion_summary_sha256']
        old_ids.update(row[key] for row in read(extra / 'summary.json')['cases']
                       for key in ('source_id', 'target_id'))
    assert set(protocol['excluded_observations']) == old_ids
    ids = [row[key] for row in summary['cases'] for key in ('source_id', 'target_id')]
    assert len(ids) == len(set(ids))
    if protocol['mode'] == 'holdout':
        assert not set(ids) & old_ids
    reasons = Counter()
    count = 0
    case_files = sorted(directory.glob('movielens_*.json')) + sorted(directory.glob('kos_*.json'))
    assert len(case_files) == len(summary['cases'])
    indexed = {(row['domain'], row['seed']): row for row in summary['cases']}
    totals = Counter()
    for path in case_files:
        evidence = read(path)
        row = evidence['case']
        assert row == indexed[(row['domain'], row['seed'])]
        runs = evidence['runs']
        expected = {(method, repeat) for method in protocol['methods']
                    for repeat in range(protocol['repetitions'])}
        assert len(runs) == len(expected)
        assert {(run['method'], run['repeat']) for run in runs} == expected
        for method in protocol['methods']:
            successful = sum(run['success'] for run in runs if run['method'] == method)
            assert successful == row['metrics'][method]['successes']
            totals[method] += successful
        for run in runs:
            assert run['seconds'] >= 0
            if run['method'] == 'bounded_sparse_then_log':
                reasons[run.get('detail', {}).get('fallback_reason', 'solver failed')] += 1
        count += len(runs)
    for method, successes in totals.items():
        assert successes == summary['summary']['all'][method]['successes']
    return dict(cases=len(case_files), runs=count, unique_observations=len(set(ids)),
                prior_overlap=len(set(ids) & old_ids), bounded_reasons=dict(reasons),
                manifest_verified=True)


if __name__ == '__main__':
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', required=True, type=Path)
    parser.add_argument('--previous', required=True, type=Path)
    parser.add_argument('--extra', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory, args.previous, args.extra), indent=2))
