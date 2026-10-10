"""Audit saved stage gates and quality; does not certify exact arithmetic."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/2026-10-09'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    reference=read(BASE/'epsilon_cooling_dev_round34/summary.json')['rows']
    outputs=[]
    for name in ['gated_epsilon_dev_round35','gated_epsilon_recheck_round35']:
        folder=BASE/name
        for file,digest in read(folder/'manifest.json').items():
            assert hashlib.sha256((folder/file).read_bytes()).hexdigest()==digest
        rows=read(folder/'summary.json')['cases']
        for row in rows:
            d=row['detail'];trace=d['trace'];total=0
            assert len(trace)==d['blocks']<=1000 and trace[0]['reg']==1.
            for i,t in enumerate(trace):
                total+=t['executed'];assert 1<=t['executed']<=1000 and total==t['work']
                quality=-1e-10<=t['gap']<=1e-8
                expected='accept' if t['reg']==.001 and quality else ('cool' if t['reg']>.001 and quality and t['raw_residual']<=1e-8 else 'hold')
                assert t['action']==expected
                if i:
                    prev=trace[i-1]
                    assert prev['action']!='accept'
                    assert t['reg']==(max(.001,prev['reg']*.5) if prev['action']=='cool' else prev['reg'])
            assert total==d['inner_work']<=1000000 and trace[-1]['reg']==d['reached_reg']
            q=d['diagnostics'];assert d['accepted'] and d['reached_reg']==.001
            assert -1e-10<=q['numerical_gap']<=1e-8 and row['audit_residual']<=1e-12
            assert abs(row['net_to_first']-1e-4)<=1e-12
            assert abs(row['audit_objective']-q['rounded_primal'])<=1e-12
            comparisons=[r['diagnostics'] for r in reference if r['seed']==row['input']['seed']]+[q]
            assert max(x['dual'] for x in comparisons)<=min(x['rounded_primal'] for x in comparisons)+1e-10
            outputs.append(dict(run=name,seed=row['input']['seed'],work=total,gap=q['numerical_gap'],stages=[dict(reg=r,blocks=sum(t['reg']==r for t in trace)) for r in sorted({t['reg'] for t in trace},reverse=True)]))
    a=read(BASE/'gated_epsilon_dev_round35/summary.json')['cases']
    b=read(BASE/'gated_epsilon_recheck_round35/summary.json')['cases']
    for x,y in zip(a,b):assert x['input']==y['input'] and x['detail']==y['detail']
    out=dict(audit_passed=True,repeat_trajectory_identical=True,independent_cases=2,rows=outputs)
    (BASE/'gated_epsilon_round35_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
