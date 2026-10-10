"""Summarize a selector pilot from saved per-case evidence, not console claims."""
import argparse
import hashlib
import json
import platform
import sys
from collections import Counter
from importlib.metadata import version
from pathlib import Path

root = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('output', type=Path)
args = parser.parse_args()
output = args.output
evaluation = json.loads((output/'selector_evaluation.json').read_text(encoding='utf-8'))
cases = [json.loads(p.read_text(encoding='utf-8')) for p in sorted((output/'case_evidence').glob('*.json'))]
counts = {}
failures = Counter()
for case in cases:
    for result in case['results']:
        record = counts.setdefault(result['algorithm_name'], {'success': 0, 'failed': 0})
        record['success' if result['success'] else 'failed'] += 1
        if not result['success']:
            failures[result['error_message'].split(' exceeds ')[0].split(' residual ')[0]] += 1
summary = {
    'status': 'exploratory, single generation seed, not publication evidence',
    'cases': len(cases),
    'exact_reference_cases': sum(c['ground_truth_is_exact'] for c in cases),
    'algorithms': counts, 'failure_categories': dict(failures),
    'evaluation': evaluation,
    'limitations': ['single solver measurement per case', 'default unequal iteration budgets',
                   'synthetic only', 'full cost matrix features, not lightweight deployment',
                   'same-distribution grouped holdout, not unseen-family validation'],
}
(output/'pilot_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
paths = list((root/'ot_benchmark').rglob('*.py')) + [root/'train_algorithm_selector.py']
manifest = {
    'python': sys.executable, 'python_version': platform.python_version(),
    'packages': {p: version(p) for p in ['numpy', 'scipy', 'POT', 'scikit-learn', 'pytest']},
    'source_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
    'evidence_sha256': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in output.rglob('*') if p.is_file() and p.name != 'reproduction_manifest.json'},
}
(output/'reproduction_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
print(json.dumps({k: v for k, v in summary.items() if k != 'evaluation'}, ensure_ascii=False, indent=2))
for key in ['accuracy', 'fixed_baseline', 'fixed_baseline_accuracy', 'selector_mean_score_regret', 'fixed_mean_score_regret']:
    print(f'{key}: {evaluation[key]}')
