"""Rebuild inputs and audit all saved validation trajectories."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'validation/2026-10-09'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    folder=BASE/'cooling_boundary_round37'
    for name,digest in read(folder/'manifest.json').items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==digest
    p=read(folder/'protocol.json');inputs=read(folder/'inputs.json');jobs=read(folder/'jobs.json');rows=read(folder/'summary.json')['rows']
    assert len(inputs)==6 and len(jobs)==len(rows)==36 and all(v=='1' for v in p['threads'].values())
    for case in inputs:
        rng=np.random.default_rng(case['seed']);x=np.r_[rng.uniform(0,.02,8),rng.uniform(.98,1,8)];c=(x[:,None]-x[None,:])**2;c/=c.max()
        a=np.ones(16)/16;b=a.copy();b[:8]+=case['delta']/8;b[8:]-=case['delta']/8
        with np.load(folder/case['file'],allow_pickle=False) as z:
            for key,value in dict(a=a,b=b,cost=c,points=x).items():assert np.array_equal(z[key],value)
        assert hashlib.sha256((folder/case['file']).read_bytes()).hexdigest()==case['sha256']
    expected=[dict(case=c,reg=r,method=m) for c in inputs for r in p['regs'] for m in p['methods']]
    np.random.default_rng(p['order_seed']).shuffle(expected);assert expected==jobs
    compact=[]
    for i,(row,job) in enumerate(zip(rows,jobs)):
        assert read(folder/f'run_{i:03d}.json')==row and row['job']==job
        if 'error' in row:
            compact.append(dict(**job,error=row['error'],accepted=False));continue
        d=row['detail'];reg=job['reg'];hold=p['methods'][job['method']];trace=d['trace'];current=1.;stage=0;work=0
        assert len(trace)==d['blocks']<=1000
        for k,t in enumerate(trace):
            assert t['block']==k+1 and t['reg']==current
            stage+=1;work+=t['executed'];assert 1<=t['executed']<=1000 and work==t['work']
            quality=-1e-10<=t['gap']<=1e-8
            advance=current>reg and (quality and t['raw_residual']<=1e-8 if hold is None else stage>=hold)
            action='accept' if current==reg and quality else ('cool' if advance else 'hold')
            assert t['action']==action
            if action=='accept':assert k==len(trace)-1
            if advance and k+1<len(trace):current=max(reg,current*.5);stage=0
        assert work==d['inner_work']<=1000000 and current==d['reached_reg']
        q=d['diagnostics'];accept=current==reg and q is not None and q['rounded_residual']<=1e-12 and -1e-10<=q['numerical_gap']<=1e-8
        assert d['accepted']==accept and (accept or len(trace)==1000)
        assert row['audit_residual']<=1e-12 and abs(row['net_to_first']-job['case']['delta'])<=1e-12
        if q:assert abs(row['audit_objective']-q['rounded_primal'])<=1e-12
        compact.append(dict(seed=job['case']['seed'],delta=job['case']['delta'],reg=reg,method=job['method'],accepted=accept,work=work,reached_reg=current,gap=q['numerical_gap'] if q else None))
    for case in inputs:
        for reg in p['regs']:
            qs=[r['detail']['diagnostics'] for r in rows if r['job']['case']==case and r['job']['reg']==reg and 'detail' in r and r['detail']['diagnostics'] is not None]
            if qs:assert max(q['dual'] for q in qs)<=min(q['rounded_primal'] for q in qs)+1e-10
    totals={m:dict(accepted=sum(r['accepted'] for r in compact if r['method']==m),total=12) for m in p['methods']}
    out=dict(audit_passed=True,totals=totals,rows=compact)
    (BASE/'cooling_boundary_round37_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
