"""Audit frozen scan evidence and export a derived table and scientific figure."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(args.input/'manifest.json')
    for name,digest in manifest.items():
        assert hashlib.sha256((args.input/name).read_bytes()).hexdigest()==digest,name
    s=read(args.input/'summary.json');protocol=read(args.input/'protocol.json')
    methods=protocol['methods'];cases=s['cases']
    assert len(cases)==len(protocol['configs'])*len(protocol['seeds'])
    assert len({(r['config']['label'],r['seed']) for r in cases})==len(cases)
    run_count=0
    for r in cases:
        label=r['config']['label'];seed=r['seed']
        raw=read(args.input/f'{label}_{seed}.json')
        assert raw['case']==r
        expected={(method,repeat) for method in methods for repeat in range(protocol['repetitions'])}
        assert len(raw['runs'])==len(expected)
        assert {(x['method'],x['repeat']) for x in raw['runs']}==expected
        for method in methods:
            rows=[x for x in raw['runs'] if x['method']==method]
            assert sum(x['success'] for x in rows)==r['metrics'][method]['successes']
            assert float(np.median([x['seconds'] for x in rows]))==r['metrics'][method]['median_seconds']
        run_count+=len(raw['runs'])
    overall={}
    for method in methods:
        paired=[r for r in cases if r['metrics'][method]['successes']==3 and r['metrics']['bounded_full']['successes']==3]
        ratios=[r['metrics'][method]['median_seconds']/r['metrics']['bounded_full']['median_seconds'] for r in paired]
        overall[method]=dict(successes=sum(r['metrics'][method]['successes'] for r in cases),runs=len(cases)*3,
            agreement_cases=sum(r['metrics'][method]['agrees_with_log'] is True for r in cases),
            fallback_runs=sum(r['metrics'][method]['fallback_runs'] for r in cases),
            candidate_speedup_median=float(np.median(ratios)) if ratios else None,
            candidate_speedup_min=min(ratios) if ratios else None,
            candidate_speedup_max=max(ratios) if ratios else None,paired_cases=len(paired))
    result=dict(cases=len(cases),runs=run_count,manifest_verified=True,overall=overall,
                inference='descriptive only; repeated seeds across configurations are dependent')
    (args.output/'audit_and_summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    lines=['| 场景 | '+' | '.join(methods)+' | 候选回退 | 普通/候选耗时比中位数 |',
           '|---|'+'---:|'*(len(methods)+2)]
    for config in protocol['configs']:
        label=config['label'];m=s['summary'][label]
        ratios=[r['metrics']['sinkhorn']['median_seconds']/r['metrics']['bounded_full']['median_seconds'] for r in cases if r['config']['label']==label and r['metrics']['sinkhorn']['successes']==3 and r['metrics']['bounded_full']['successes']==3]
        ratio=f'{np.median(ratios):.3f}' if ratios else 'NA'
        lines.append('| '+label+' | '+' | '.join(str(m[k]['successes'])+'/9' for k in methods)+' | '+str(m['bounded_full']['fallback_runs'])+' | '+ratio+' |')
    (args.output/'场景结果表.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(10,5.5))
    labels=[c['label'] for c in protocol['configs']]
    for method,offset,color in [('sinkhorn',-.2,'#3377aa'),('sinkhorn_stabilized',0,'#228855'),('sinkhorn_log',.2,'#bb6633')]:
        xs=[];ys=[]
        for idx,label in enumerate(labels):
            vals=[r['metrics'][method]['median_seconds']/r['metrics']['bounded_full']['median_seconds'] for r in cases if r['config']['label']==label and r['metrics'][method]['successes']==3 and r['metrics']['bounded_full']['successes']==3]
            xs.extend([idx+offset]*len(vals));ys.extend(vals)
        ax.scatter(xs,ys,label=method,color=color,alpha=.7,s=26)
    ax.axhline(1,color='black',linestyle='--',linewidth=1)
    ax.set_yscale('log');ax.set_xticks(range(len(labels)),labels,rotation=35,ha='right')
    ax.set_ylabel('Baseline time / candidate time (paired successful cases)')
    ax.set_title('Exploratory one-factor scan; each dot is one seed\nAbove 1: candidate faster; below 1: baseline faster')
    ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout()
    fig.savefig(args.output/'solver_boundary.png',dpi=180)
    fig.savefig(args.output/'solver_boundary.pdf')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
