"""Replay quality thresholds on existing trajectories, without timing claims."""
import argparse,json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    cases=json.loads((args.input/'summary.json').read_text(encoding='utf-8'))['cases']
    results={}
    for tol in [1e-6,1e-8]:
        hits=[(r,next((h for h in r['history'] if h['rounded_residual']<=1e-12 and -1e-10<=h['numerical_gap']<=tol),None)) for r in cases]
        results[str(tol)]=dict(accepted_cases=sum(h is not None for r,h in hits),
            earlier_than_original_stop=sum(h is not None and h['iteration']<r['iterations'] for r,h in hits),
            previous_budget_failures_now_meeting_different_contract=sum(h is not None and not r['accepted'] for r,h in hits),
            sum_observed_iterations=sum(h['iteration'] if h is not None else r['iterations'] for r,h in hits))
    print(json.dumps(results,indent=2))


if __name__=='__main__':main()
