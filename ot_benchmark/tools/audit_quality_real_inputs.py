"""Audit frozen real-input identity exclusions, normalization and source hashes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory
    root=Path(__file__).resolve().parents[1]
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name,h in json.loads((d/'manifest.json').read_text()).items():assert sha(d/name)==h
    protocol=json.loads((d/'protocol.json').read_text());data=json.loads((d/'summary.json').read_text())
    for name,h in protocol['raw_sha256'].items():assert sha(root/name)==h
    for name,h in protocol['prior_summary_hashes'].items():assert sha(root/name)==h
    used=set(protocol['prior_ids']);original_count=len(used);rows=[]
    for c in data['cases']:
        ids={c['source_id'],c['target_id']};assert len(ids)==2 and not used&ids;used.update(ids)
        assert sha(d/c['file'])==c['sha256']
        with np.load(d/c['file'],allow_pickle=False) as z:
            a,b,cost=z['a'],z['b'],z['cost']
            assert cost.shape==(len(a),len(b))
            assert all(np.isfinite(x).all() and np.all(x>=0) for x in [a,b,cost])
            assert abs(a.sum()-1)<1e-12 and abs(b.sum()-1)<1e-12
            assert np.array_equal(a,z['loader_a']/z['loader_a'].sum())
            assert np.array_equal(b,z['loader_b']/z['loader_b'].sum())
            rows.append(dict(domain=c['domain'],source_active=int(np.count_nonzero(a)),target_active=int(np.count_nonzero(b)),
                original_mass_defect=max(abs(z['loader_a'].sum()-1),abs(z['loader_b'].sum()-1))))
    result=dict(manifest_verified=True,raw_sources_verified=True,prior_ids_excluded=original_count,
        pairs=len(rows),distinct_new_observations=len(used)-original_count,rows=rows,
        max_loader_mass_defect=max(r['original_mass_defect'] for r in rows),solver_performance_evaluated=False)
    (d.parent/(d.name+'_audit.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
