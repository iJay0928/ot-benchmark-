"""Audit whole-policy timing and outcomes; no retrospective oracle selection."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    for name,digest in read(args.input/'manifest.json').items():assert hashlib.sha256((args.input/name).read_bytes()).hexdigest()==digest
    protocol=read(args.input/'protocol.json');s=read(args.input/'summary.json');cases=s['cases'];methods=protocol['methods']
    assert len({(r['config']['label'],r['seed']) for r in cases})==len(cases)
    if protocol['mode']=='new seeds same generator':
        assert len(cases)==12 and set(protocol['seeds']).isdisjoint([92001,92002,92003,94001,94002,94003,96001,96002,96003,97001,97002,97003])
    else:assert len(cases)==28
    extra=[];avoided=[]
    for r in cases:
        raw=read(args.input/f'{r["config"]["label"]}_{r["seed"]}.json');assert raw['case']==r
        expected={(m,i) for m in methods for i in range(3)}
        assert len(raw['runs'])==len(expected) and {(x['method'],x['repeat']) for x in raw['runs']}==expected
        for m in methods:
            rows=[x for x in raw['runs'] if x['method']==m]
            assert sum(x['success'] for x in rows)==r['metrics'][m]['successes']
            assert sum(x['detail']['used_epsilon'] for x in rows)==r['metrics'][m]['fallback_runs']
            assert float(np.median([x['seconds'] for x in rows]))==r['metrics'][m]['median_seconds']
            for x in rows:assert x['success']==(x['detail']['residual'] is not None and x['detail']['residual']<=1e-9)
        out=dict(config=r['config']['label'],seed=r['seed'],rolling_over_budget=r['metrics']['rolling']['median_seconds']/r['metrics']['budget']['median_seconds'])
        if r['metrics']['rolling']['fallback_runs'] and not r['metrics']['budget']['fallback_runs']:extra.append(out)
        if r['metrics']['rolling']['fallback_runs'] and r['metrics']['budget']['fallback_runs']:avoided.append(out)
    totals={m:dict(successes=sum(r['metrics'][m]['successes'] for r in cases),runs=len(cases)*3,
        epsilon_runs=sum(r['metrics'][m]['fallback_runs'] for r in cases),
        sum_case_median_seconds=sum(r['metrics'][m]['median_seconds'] for r in cases)) for m in methods}
    paired=[r for r in cases if r['metrics']['rolling']['successes']==r['metrics']['budget']['successes']==3]
    ratios=[r['metrics']['budget']['median_seconds']/r['metrics']['rolling']['median_seconds'] for r in paired]
    report=dict(cases=len(cases),runs=len(cases)*9,manifest_verified=True,totals=totals,
        budget_over_rolling_median=float(np.median(ratios)),budget_over_rolling_min=min(ratios),budget_over_rolling_max=max(ratios),
        extra_epsilon_cases=extra,both_epsilon_cases=avoided,
        note='same-seed configurations dependent; complete solver policy time excludes common preprocessing; no significance claim')
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
