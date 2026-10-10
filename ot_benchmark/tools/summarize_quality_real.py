"""Keep corpus and regularization transfer outcomes separate."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory
    audit=json.loads((d.parent/(d.name+'_audit.json')).read_text());assert audit['manifest_verified']
    protocol=json.loads((d/'protocol.json').read_text());inputs=Path(protocol['input_directory'])
    assert hashlib.sha256((inputs/'manifest.json').read_bytes()).hexdigest()==protocol['input_manifest_sha256']
    data=json.loads((d/'summary.json').read_text());cases=data['cases'];methods=list(data['totals']);groups={}
    for key in ['movielens','kos']+[c['config']['label'] for c in cases]:
        if key in groups:continue
        subset=[c for c in cases if key==c['input']['domain'] or key==c['config']['label']]
        totals={m:dict(successes=sum(c['metrics'][m]['successes'] for c in subset),runs=3*len(subset),
            sum_medians=sum(c['metrics'][m]['median_seconds'] for c in subset),
            fallback_runs=sum(c['metrics'][m]['fallback_runs'] for c in subset)) for m in methods}
        comparisons={}
        for base in ['adaptive','every_250','every_1000','residual_gate']:
            paired=[c for c in subset if c['metrics'][base]['successes']==3 and c['metrics']['adaptive_gap']['successes']==3]
            rows=[dict(seed=c['seed'],reg=c['config']['reg'],ratio=c['metrics'][base]['median_seconds']/c['metrics']['adaptive_gap']['median_seconds'])
                for c in subset if c['metrics'][base]['successes']==3 and c['metrics']['adaptive_gap']['successes']==3]
            comparisons[base]=dict(paired_cases=len(rows),faster_cases=sum(r['ratio']>1 for r in rows),
                median_speedup=float(np.median([r['ratio'] for r in rows])) if rows else None,
                all_attempt_time_reduction=1-totals['adaptive_gap']['sum_medians']/totals[base]['sum_medians'],
                paired_baseline_sum=sum(c['metrics'][base]['median_seconds'] for c in paired),
                paired_gap_sum=sum(c['metrics']['adaptive_gap']['median_seconds'] for c in paired),
                paired_time_reduction=1-sum(c['metrics']['adaptive_gap']['median_seconds'] for c in paired)/sum(c['metrics'][base]['median_seconds'] for c in paired) if paired else None,
                worst=min(rows,key=lambda r:r['ratio']) if rows else None)
        stop_diagnostics=[]
        for c in subset:
            detail={m:next(r['detail'] for r in c['runs'] if r['method']==m and r['repeat']==0) for m in methods}
            stop_diagnostics.append(dict(seed=c['seed'],reg=c['config']['reg'],
                ordinary_iterations={m:r['ordinary_iterations'] for m,r in detail.items()},
                quality_checks={m:r['quality_checks'] for m,r in detail.items()}))
        groups[key]=dict(totals=totals,gap_only_comparisons=comparisons,first_repeat_stop_diagnostics=stop_diagnostics)
    result=dict(groups=groups,warning_runs=sum(bool(r['warnings']) for c in cases for r in c['runs']),
        failures=audit['failures'],note='six independent observation pairs within each corpus, reused at two regularizations; shared preprocessing; no inferential significance claim')
    (d.parent/(d.name+'_transfer.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
