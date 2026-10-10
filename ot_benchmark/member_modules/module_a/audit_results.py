"""Read-only recomputation: input identity, completeness and all saved plans."""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
from core import ROOT, array_sha, config, diagnostics, make_input, objective, sha, write_json

def audit(out, require_complete=True):
    protocol=json.loads((out/'protocol.json').read_text(encoding='utf-8'))
    manifest=json.loads((out/'input_manifest.json').read_text(encoding='utf-8'))
    tasks=json.loads((out/'task_order.json').read_text(encoding='utf-8'))
    rows=json.loads((out/'results/summary.json').read_text(encoding='utf-8'))
    expected={t['task_id']:t for t in tasks};data={};errors=[];checks=Counter()
    for inp in manifest:
        original=make_input(inp['seed'],inp['n'],inp['delta'])
        with np.load(out/inp['file']) as z: saved={k:z[k] for k in z.files}
        assert all(np.array_equal(saved[k],original[k]) for k in original)
        assert array_sha(original)==inp['sha256'] and sha(out/inp['file'])==inp['npz_sha256']
        data[inp['id']]=original;checks['inputs_rebuilt']+=1
    assert len(rows)==len({r['task_id'] for r in rows})
    missing=set(expected)-{r['task_id'] for r in rows}
    if require_complete and missing: errors.append(f'{len(missing)} missing calls')
    for row in rows:
        try:
            task=expected[row['task_id']]
            assert all(row[k]==v for k,v in task.items())
            assert row['protocol_sha256']==sha(out/'protocol.json')
            raw=json.loads((out/'results/logs'/f"{row['task_id']}.json").read_text(encoding='utf-8'))
            assert raw['result']==row
            assert row.get('config',config(row['method'],protocol))==config(row['method'],protocol)
            if row['status']=='TIMEOUT':
                assert row['timed_out'] and row['elapsed_seconds'] is None
                checks['censored_calls']+=1
                continue
            file=out/'results/plans'/f"{row['task_id']}.npz"
            if file.exists():
                with np.load(file) as z: arr={k:z[k] for k in z.files}
                if 'log_u' in arr and 'log_v' in arr:
                    d=data[row['input_id']];a,b,c=(d[k] for k in ('a','b','cost'))
                    diag,p=diagnostics.diagnose(a,b,c,row['epsilon'],arr['log_u'],arr['log_v'])
                    for field in ('numerical_gap','dual','raw_residual','rounding_l1','rounded_primal'):
                        assert np.isclose(diag[field],row[field],rtol=1e-10,atol=1e-13),field
                    assert np.allclose(arr['plan'],p,rtol=1e-9,atol=1e-12)
                    rounded=diagnostics.round_plan(arr['plan'],a,b)
                    gap=objective(rounded,c,row['epsilon'])-diag['dual']
                    accepted=(protocol['gap_min']<=gap<=protocol['gap_max'] and
                              protocol['gap_min']<=diag['numerical_gap']<=protocol['gap_max'])
                    assert accepted==(row['status']=='SUCCESS')
                    checks['dual_plan_recomputed']+=1
            if row['status']=='SUCCESS':
                assert row['rounded_residual']<=protocol['rounded_residual_limit']
                assert row['elapsed_seconds'] is not None and row['solve_seconds']>=0
                checks['accepted_calls']+=1
            checks['bound_calls']+=1
        except Exception as exc: errors.append(row['task_id']+': '+repr(exc))
    result=dict(passed=not errors,checks=dict(checks),expected_calls=len(tasks),observed_calls=len(rows),
                shared_diagnostics_sha256=sha(ROOT.parents[1]/'ot_benchmark/algorithms/transport_diagnostics.py'),
                missing_calls=sorted(missing),errors=errors,
                meaning='audit consistency is not solver success; floating numerical diagnostics, not interval certificates')
    write_json(out/'results/audit.json',result)
    print(json.dumps(result,ensure_ascii=False,indent=2))
    if errors: raise SystemExit(1)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true');a=p.parse_args()
    audit(a.output,not a.allow_partial)
