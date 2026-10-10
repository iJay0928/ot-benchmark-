"""Compare directly scaled floating kernels with log reconstruction of frozen states."""
import argparse,json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();d=args.directory;rows=[]
    protocol=json.loads((d/'protocol.json').read_text())
    # All four frozen cases in this specific experiment use reg=.001.
    cases=json.loads((d/'summary.json').read_text())['cases']
    for case in cases:
        reg=case['reg'];name=f"{case['input']['domain']}_reg_{reg}_{case['input']['seed']}_state.npz"
        with np.load(d/name) as z:
            c,u,v=z['cost'],z['u'],z['v'];k=np.exp(-c/reg)
            direct=u[:,None]*k*v[None,:]
            reconstructed=np.exp(np.log(u)[:,None]+np.log(v)[None,:]-c/reg)
            rows.append(dict(state=name,kernel_zero_entries=int(np.count_nonzero(k==0)),entries=k.size,
                direct_mass=float(direct.sum()),reconstructed_mass=float(reconstructed.sum()),
                reconstructed_mass_on_zero_kernel=float(reconstructed[k==0].sum()),
                plan_l1_discrepancy=float(np.abs(direct-reconstructed).sum())))
    out=d.parent/'segmented_rescue_numeric_round19.json';out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))


if __name__=='__main__':main()
