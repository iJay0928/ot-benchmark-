import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'validation/2026-10-10'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def main():
    folder=BASE/'cluster_perturbation_round40';protocol=read(folder/'protocol.json');rows=read(folder/'summary.json')['rows']
    for name,h in read(folder/'manifest.json').items():assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==h
    inputs=read(folder/'inputs.json');assert len(inputs)==18 and len(rows)==36
    for inp in inputs:
        rng=np.random.default_rng(inp['seed']);r=rng.uniform(0,1,16);w=inp['width'];x=np.r_[w*r[:8],1-w+w*r[8:]]
        c=(x[:,None]-x[None,:])**2;c/=c.max();a=np.ones(16)/16;b=a.copy();b[:8]+=inp['delta']/8;b[8:]-=inp['delta']/8
        with np.load(folder/inp['file'],allow_pickle=False) as z:
            for key,val in dict(a=a,b=b,cost=c,points=x).items():assert np.array_equal(z[key],val)
        assert hashlib.sha256((folder/inp['file']).read_bytes()).hexdigest()==inp['sha256']
    groups={};max_relative=0.;example=None
    seen=set()
    for i,row in enumerate(rows):
        assert row==read(folder/f'run_{i:02d}.json');inp=row['input'];key=(inp['seed'],inp['width'],inp['delta'],row['reg']);assert key not in seen;seen.add(key)
        w=inp['width'];g=groups.setdefault(str(w),dict(conditions=0,accepted=0,max_block_error=0.,max_reconstruction_l1=0.,max_first_raw_residual_balanced=0.))
        g['conditions']+=1;g['accepted']+=row['final_accepted'];g['max_block_error']=max(g['max_block_error'],max(row['block_max_errors']))
        for j,(f,c) in enumerate(zip(row['fine'],row['coarse'])):
            assert f['iteration']==c['iteration']==protocol['checkpoints'][j]
            pf,pc=np.array(f['block']),np.array(c['block']);assert abs(float(np.max(np.abs(pf-pc)))-row['block_max_errors'][j])<=1e-20
            assert pf.min()>=0 and np.isfinite(pf).all()
            q=f['diagnostics'];assert q['rounded_residual']<=1e-12 and q['numerical_gap']>=-1e-10
            assert abs(q['rounded_primal']-q['dual']-q['numerical_gap'])<=1e-12
            g['max_reconstruction_l1']=max(g['max_reconstruction_l1'],f['reconstruction_l1'])
            if inp['delta']==0 and j==0:g['max_first_raw_residual_balanced']=max(g['max_first_raw_residual_balanced'],q['raw_residual'])
            reference=pc[0,1]+pc[1,0];fine=pf[0,1]+pf[1,0]
            if reference>0:
                rel=abs(fine-reference)/reference
                if rel>max_relative:max_relative=rel;example=dict(condition=key,iteration=f['iteration'],coarse_cross_total=reference,fine_cross_total=fine,relative_error=rel)
        q=row['fine'][-1]['diagnostics'];assert row['final_accepted']==(-1e-10<=q['numerical_gap']<=1e-8 and q['rounded_residual']<=1e-12)
    assert all(g['conditions']==12 for g in groups.values()) and groups['0.0']['max_block_error']<=1e-10
    out=dict(audit_passed=True,groups=groups,posthoc_cross_flow_relative_error=example,limitation='relative-flow diagnostic added after outputs; not a preregistered pass criterion')
    (BASE/'cluster_perturbation_round40_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
