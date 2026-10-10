"""Independent saved-plan feasibility/objective and reference audit."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'validation/2026-10-09'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    folder=BASE/'best_return_round38';inp=BASE/'cooling_boundary_round37'
    for name,h in read(folder/'manifest.json').items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h
    protocol=read(folder/'protocol.json');assert hashlib.sha256((inp/'summary.json').read_bytes()).hexdigest()==protocol['source_sha256']
    assert all(v=='1' for v in protocol['threads'].values())
    rows=read(folder/'summary.json')['rows'];old=read(inp/'summary.json')['rows'];assert len(rows)==len(protocol['jobs'])==4
    for i,row in enumerate(rows):
        job=row['job'];q=row['returned_plan_quality'];assert job==protocol['jobs'][i] and row==read(folder/f'run_{i}.json')
        reference=next(r for r in old if r['job']==job);assert row['original_detail']==reference['detail']
        assert not row['original_detail']['accepted'] and q['source']=='best rounded stage'
        with np.load(inp/job['case']['file'],allow_pickle=False) as z:a,b,c=z['a'],z['b'],z['cost']
        with np.load(folder/f'plan_{i}.npz',allow_pickle=False) as z:p=z['plan']
        assert np.isfinite(p).all() and np.all(p>=0)
        residual=max(np.abs(p.sum(1)-a).sum(),np.abs(p.sum(0)-b).sum());mask=p>0
        objective=float(np.sum(p*c)+job['reg']*np.sum(p[mask]*(np.log(p[mask])-1)))
        assert residual<=1e-12 and abs(objective-q['rounded_primal'])<=1e-12
        assert objective<=q['original_primal'] and abs(q['original_primal']-reference['audit_objective'])<=1e-12
        assert abs(objective-q['dual']-q['numerical_gap'])<=1e-12 and q['numerical_gap']>=-1e-10
        assert abs(p[8:,:8].sum()-p[:8,8:].sum()-job['case']['delta'])<=1e-12
        assert sum(s['work'] for s in row['stages'])==1000000
        # All original methods' target dual values must be compatible with this feasible return.
        ds=[r['detail']['diagnostics']['dual'] for r in old if r['job']['case']==job['case'] and r['job']['reg']==job['reg']]
        assert max(ds)<=objective+1e-10
    out=dict(audit_passed=True,conditions=4,trajectory_unchanged=True,accepted=0,all_returned_objectives_improved=True)
    (BASE/'best_return_round38_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(out)
if __name__=='__main__':main()
