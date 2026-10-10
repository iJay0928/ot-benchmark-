"""Optional discovery replay using externally installed hybrid_scholar.

The tool and its credentials are not vendored. Records are not novelty proof.
Existing outputs are never overwritten.
"""
import argparse
from datetime import date
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

QUERIES = [
    'optimal transport Sinkhorn sparse low rank',
    'optimal transport diffusion image editing',
    'optimal transport attention style',
    'optimal transport spatial attention layout',
    'optimal transport partial domain adaptation',
]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--skill-scripts', type=Path, required=True,
                        help='Directory containing externally installed hybrid_scholar.py')
    parser.add_argument('--output', type=Path, required=True,
                        help='New JSON file; existing discovery records are protected')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Choose a new output path; do not overwrite evidence')
    scripts = args.skill_scripts.resolve(strict=True)
    if not (scripts / 'hybrid_scholar.py').is_file():
        raise FileNotFoundError('hybrid_scholar.py not found in --skill-scripts')
    sys.path.insert(0, str(scripts))
    from hybrid_scholar import HybridScholar

    def search(query):
        scholar = HybridScholar()
        result = scholar.search_papers(query, limit=8)
        return json.loads(scholar.results_to_json(result))

    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(search, QUERIES))
    record = {'date': date.today().isoformat(),
              'note': 'Discovery only; publisher/full text verification separate.',
              'queries': results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(record, stream, ensure_ascii=False, indent=2)
    for result in results:
        print(result['query'], result['stats'])


if __name__ == '__main__':
    main()
