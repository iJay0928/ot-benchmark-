"""Descriptive cold/warm comparison; original pass/fail labels are preserved."""
import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--development',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();read=lambda x:json.loads(x.read_text(encoding='utf-8'))
    s=read(a.input/'summary.json');protocol=read(a.input/'protocol.json')
    dev=read(a.development/'protocol.json')
    if protocol['seed_role']=='new seeds same generator':
        assert not set(protocol['seeds'])&set(dev['seeds'])
    cases=s['cases'];switched=[]
    for r in cases:
        raw=read(a.input/f'{r["config"]["label"]}_{r["seed"]}.json')
        warm=[v for v in raw['runs'] if v['method']=='bounded_full']
        if any(v.get('detail',{}).get('switched') for v in warm):
            m=r['metrics'];good=all(m[k]['successes']==3 for k in ['bounded_full','cold_switch','matched_full'])
            switched.append(dict(label=r['config']['label'],seed=r['seed'],all_three_successful=good,
                cold_over_warm=m['cold_switch']['median_seconds']/m['bounded_full']['median_seconds'] if good else None,
                warm_over_full=m['bounded_full']['median_seconds']/m['matched_full']['median_seconds'] if good else None,
                warm_iterations=[v.get('detail',{}).get('iterations') for v in warm],
                cold_iterations=[v.get('detail',{}).get('iterations') for v in raw['runs'] if v['method']=='cold_switch']))
    overall={}
    for method in protocol['methods']:
        rows=[r['metrics'][method] for r in cases]
        overall[method]=dict(successes=sum(r['successes'] for r in rows),runs=len(rows)*3,
                            sum_case_median_seconds=sum(r['median_seconds'] for r in rows))
    result=dict(seed_disjoint=set(protocol['seeds']).isdisjoint(dev['seeds']),
                note='Only matched_full/matched_log/cold_switch/bounded_full share L1=1e-9 stopping and total budget; native acceptance remains 1e-7',
                overall=overall,successful_switched_cases=switched)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
