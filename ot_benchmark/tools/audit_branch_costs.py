"""Audit paired remaining-cost observations, separating failure from latency."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    for name,digest in read(args.input/'manifest.json').items():
        assert hashlib.sha256((args.input/name).read_bytes()).hexdigest()==digest
    protocol=read(args.input/'protocol.json');s=read(args.input/'summary.json');rows=s['cases']
    root=Path(__file__).resolve().parents[1];trajectories={}
    for name,digest in protocol['inputs'].items():
        source=root/name
        assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
        for r in read(source)['cases']:trajectories[(r['config']['label'],r['seed'])]=r['history']
    keys={(r['config']['label'],r['seed'],r['checkpoint']) for r in rows};assert len(keys)==len(rows)
    assert len(s['screening'])==28
    for r in s['screening']:
        for checkpoint in protocol['checkpoints']:
            key=(r['config']['label'],r['seed'],checkpoint)
            expected=not r['accepted_by_3000'] or r['iterations']>checkpoint
            assert (key in keys)==expected
    methods=protocol['methods'];warnings={m:0 for m in methods};cost_deltas=[]
    outcomes=[];learning_rows=[]
    for row in rows:
        path=args.input/f'{row["config"]["label"]}_{row["seed"]}_{row["checkpoint"]}.json'
        raw=read(path);assert raw['case']==row and row['state_unchanged']
        expected={(m,k) for m in methods for k in range(3)}
        assert len(raw['runs'])==len(expected)
        assert {(r['method'],r['repeat']) for r in raw['runs']}==expected
        for m in methods:
            batch=[r for r in raw['runs'] if r['method']==m]
            assert sum(r['detail']['accepted'] for r in batch)==row['metrics'][m]['successes']
            assert float(np.median([r['seconds'] for r in batch]))==row['metrics'][m]['median_seconds']
            for r in batch:
                detail=r['detail'];assert detail['accepted']==(detail['residual'] is not None and detail['residual']<=1e-9)
                warnings[m]+=bool(r['warnings'])
        good=[m for m in methods if row['metrics'][m]['successes']==3]
        if len(good)>1:
            costs=[row['metrics'][m]['cost'] for m in good];cost_deltas.append(max(costs)-min(costs))
        outcome='no successful action'
        if row['metrics']['continue']['successes']==3:
            best_switch=min((row['metrics'][m]['median_seconds'] for m in ['epsilon_cold','epsilon_warm'] if m in good),default=float('inf'))
            outcome='continue faster' if row['metrics']['continue']['median_seconds']<=best_switch else 'switch faster'
        elif good:outcome='switch restores acceptance'
        outcomes.append(dict(config=row['config']['label'],seed=row['seed'],checkpoint=row['checkpoint'],outcome=outcome,
            continue_seconds=row['metrics']['continue']['median_seconds'],cold_seconds=row['metrics']['epsilon_cold']['median_seconds'],warm_seconds=row['metrics']['epsilon_warm']['median_seconds']))
        history=[r for r in trajectories[(row['config']['label'],row['seed'])] if row['checkpoint']-500<=r['iteration']<=row['checkpoint']]
        assert history and history[-1]['iteration']==row['checkpoint']
        slope=float(np.polyfit([r['iteration'] for r in history],np.log([r['residual'] for r in history]),1)[0])
        learning_rows.append(dict(group_id=f'seed_{row["seed"]}',case_id=f'{row["config"]["label"]}_{row["seed"]}',
            features=dict(n=row['config']['n'],d=row['config']['d'],zero=row['config']['zero'],reg=row['config']['reg'],
                          iteration=row['checkpoint'],residual=history[-1]['residual'],recent_log_slope=slope),
            offline_targets=row['metrics'],offline_outcome=outcome))
    paired=[r for r in rows if all(r['metrics'][m]['successes']==3 for m in ['epsilon_cold','epsilon_warm'])]
    report=dict(screened_cases=28,branch_states=len(rows),runs=len(rows)*9,manifest_verified=True,
        repeated_state_note='two states of same case are dependent; not independent test samples',
        methods={m:dict(successes=sum(r['metrics'][m]['successes'] for r in rows),runs=len(rows)*3,warning_runs=warnings[m]) for m in methods},
        cold_over_warm_time_median=float(np.median([r['metrics']['epsilon_cold']['median_seconds']/r['metrics']['epsilon_warm']['median_seconds'] for r in paired])) if paired else None,
        max_cost_difference_between_accepted_actions=max(cost_deltas) if cost_deltas else None,
        outcome_counts={name:sum(r['outcome']==name for r in outcomes) for name in ['continue faster','switch faster','switch restores acceptance','no successful action']},
        states=outcomes,learning_data=dict(role='development only; no fitted predictor or validation score',
            grouping='same seed across checkpoints and configurations must stay in one partition',
            target_warning='failed-action runtime is a censored budget cost, not time to an accepted solution',rows=learning_rows))
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
