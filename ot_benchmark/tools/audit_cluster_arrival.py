import argparse, hashlib, json
from pathlib import Path
import numpy as np

def read(p): return json.loads(p.read_text(encoding='utf-8'))

def main():
    p=argparse.ArgumentParser(); p.add_argument('folder',type=Path); o=p.parse_args().folder
    for name,h in read(o/'manifest.json').items(): assert hashlib.sha256((o/name).read_bytes()).hexdigest()==h
    protocol=read(o/'protocol.json'); rows=read(o/'summary.json')['rows']; seen=set(); groups={}
    for i,row in enumerate(rows):
        assert row==read(o/f'run_{i:02d}.json')
        inp=row['input']; key=(inp['seed'],inp['width'],inp['delta'],row['reg']); assert key not in seen; seen.add(key)
        r=np.random.default_rng(inp['seed']).uniform(0,1,16); w=inp['width']; x=np.r_[w*r[:8],1-w+w*r[8:]]
        c=(x[:,None]-x[None,:])**2; c/=c.max(); a=np.ones(16)/16; b=a.copy(); b[:8]+=inp['delta']/8; b[8:]-=inp['delta']/8
        with np.load(o/inp['file']) as z:
            for k,v in dict(a=a,b=b,cost=c,points=x).items(): assert np.array_equal(z[k],v)
        g=groups.setdefault(str(w),dict(conditions=0,accepted=0,both_observed=0,both_censored=0,discordant_censoring=0,relative_errors=[]))
        g['conditions']+=1; g['accepted']+=row['fine']['final_accepted']
        for side in ['fine','coarse']:
            result=row[side]; nets=np.load(o/result['trace']); assert nets.shape==(protocol['budget'],) and np.isfinite(nets).all()
            for event,f in zip(result['arrivals'],protocol['fractions']):
                idx=np.flatnonzero(nets>=f*inp['delta']); expected=None if not len(idx) else int(idx[0])+1
                assert event['fraction']==f and event['iteration']==expected
                if expected:
                    assert event['crossing_net']==nets[expected-1]
                    assert expected==1 or event['previous_net']==nets[expected-2]<f*inp['delta']
            q=result['diagnostics']; assert q['rounded_residual']<=1e-12 and q['numerical_gap']>=-1e-10
            assert abs(q['rounded_primal']-q['dual']-q['numerical_gap'])<1e-12
            assert result['final_accepted']==(-1e-10<=q['numerical_gap']<=1e-8)
        for f,c in zip(row['fine']['arrivals'],row['coarse']['arrivals']):
            fi,ci=f['iteration'],c['iteration']
            if w==0: assert fi==ci
            if fi is None and ci is None: g['both_censored']+=1
            elif fi is None or ci is None: g['discordant_censoring']+=1
            else: g['both_observed']+=1; g['relative_errors'].append(abs(fi-ci)/ci)
    expected={(s,w,d,e) for s in protocol['seeds'] for w in protocol['widths'] for d in protocol['deltas'] for e in protocol['regs']}
    assert seen==expected
    for g in groups.values():
        errors=g.pop('relative_errors'); g['max_relative_error']=max(errors) if errors else None; g['median_relative_error']=float(np.median(errors)) if errors else None
    out=dict(audit_passed=True,groups=groups,limitation='dense traces audited; floating-point diagnostic, not interval certificate; development seeds')
    (o.parent/'cluster_arrival_round41_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8'); print(json.dumps(out,indent=2))

if __name__=='__main__': main()
