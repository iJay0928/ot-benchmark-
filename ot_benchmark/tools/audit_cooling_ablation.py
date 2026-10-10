"""Check matched schedules, work budgets, saved binding and quality decisions."""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/2026-10-09'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    folder=BASE/'cooling_ablation_round36'
    for name,digest in read(folder/'manifest.json').items():
        assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
    protocol=read(folder/'protocol.json')
    assert all(v=='1' for v in protocol['threads'].values())
    rows=read(folder/'summary.json')['cases']
    assert {(r['input']['seed'],r['method']) for r in rows}=={(s,m) for s in [113001,113002] for m in ['gated','fixed1','fixed10']}
    old={r['input']['seed']:r for r in read(BASE/'gated_epsilon_recheck_round35/summary.json')['cases']}
    result=[]
    for row in rows:
        case=row['input'];d=row['detail'];trace=d['trace'];hold=protocol['methods'][row['method']]
        f=BASE/'numeric_boundary_inputs_round33'/case['file']
        assert hashlib.sha256(f.read_bytes()).hexdigest()==case['sha256']
        assert read(folder/f"seed_{case['seed']}_{row['method']}.json")==row
        assert len(trace)==d['blocks']<=1000 and trace[0]['reg']==1.
        current=1.;stage=0;work=0
        for i,t in enumerate(trace):
            assert t['block']==i+1 and t['reg']==current
            stage+=1;work+=t['executed']
            assert 1<=t['executed']<=1000 and work==t['work']
            quality=-1e-10<=t['gap']<=1e-8
            advance=current>.001 and (quality and t['raw_residual']<=1e-8 if hold is None else stage>=hold)
            expected='accept' if current==.001 and quality else ('cool' if advance else 'hold')
            assert expected==t['action']
            if expected=='accept':assert i==len(trace)-1
            if advance and i+1<len(trace):current=max(.001,current*.5);stage=0
        q=d['diagnostics'];assert q is not None
        assert work==d['inner_work']<=1000000 and current==d['reached_reg']
        accepted=current==.001 and q['rounded_residual']<=1e-12 and -1e-10<=q['numerical_gap']<=1e-8
        assert d['accepted']==accepted and (accepted or len(trace)==1000)
        assert row['audit_residual']<=1e-12 and abs(row['net_to_first']-1e-4)<=1e-12
        assert abs(row['audit_objective']-q['rounded_primal'])<=1e-12
        if hold is None:assert d==old[case['seed']]['detail']
        result.append(dict(seed=case['seed'],method=row['method'],accepted=accepted,work=work,gap=q['numerical_gap']))
    for seed in [113001,113002]:
        qs=[r['detail']['diagnostics'] for r in rows if r['input']['seed']==seed]
        assert max(q['dual'] for q in qs)<=min(q['rounded_primal'] for q in qs)+1e-10
    out=dict(audit_passed=True,unique_cases=2,trajectories=6,rows=result)
    (BASE/'cooling_ablation_round36_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
