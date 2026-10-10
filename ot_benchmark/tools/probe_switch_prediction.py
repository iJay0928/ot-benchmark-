"""Development-only test of a residual-rate switching hypothesis, not a solver."""
import os
os.environ.setdefault('POT_BACKEND_DISABLE_TENSORFLOW','1')
import argparse,hashlib,json,time,zipfile
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist


def forecast(history,target=1e-9,budget=50000):
    recent=[r for r in history if r['iteration']>=500]
    if history[-1]['residual']<=target:
        return dict(switch=False,reason='already accepted',predicted_finish=history[-1]['iteration'])
    if len(recent)<2:return dict(switch=False,reason='insufficient observations',predicted_finish=None)
    x=np.array([r['iteration'] for r in recent]);y=np.log([max(r['residual'],1e-300) for r in recent])
    slope=float(np.polyfit(x,y,1)[0])
    finish=float(x[-1]+(np.log(target)-y[-1])/slope) if slope<0 else None
    return dict(switch=finish is None or finish>budget,reason='log residual linear extrapolation',
                predicted_finish=finish,slope=slope)


def rolling_decision(history):
    for t in range(1000,10001,1000):
        window=[r for r in history if t-500<=r['iteration']<=t]
        if not window or window[-1]['iteration']!=t:break
        if forecast(window)['switch']:return t
    return None


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--new-seeds',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root=Path(__file__).resolve().parents[1];v=root/'validation/2026-10-06'
    sources=[v/'epsilon_strict_dev_round8/summary.json',v/'epsilon_holdout_round8/summary.json']
    cases=[r for path in sources for r in json.loads(path.read_text(encoding='utf-8'))['cases']]
    if args.new_seeds:
        configs={r['config']['label']:r['config'] for r in cases}
        cases=[dict(config=conf,seed=seed) for conf in configs.values() for seed in [97001,97002,97003]]
    protocol=dict(role='all 16 cases previously examined; development only',decision_iteration=1000,
        fit_window=[500,1000],check_every=50,total_budget=50000,tolerance=1e-9,
        hypothesis='residual log-slope by iteration 1000 predicts failure to reach target by 50000',
        limitation='predicts budget failure, not which solver is faster; no deployed switching or runtime saving claim',
        source_sha256={str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sources})
    protocol.update(role='new seed replay validation; no actual solver switching' if args.new_seeds else 'previously examined development cases',
                    rolling_rule='every 1000 through 10000, fit last 500, switch if projected finish >50000',
                    cases=[dict(config=r['config'],seed=r['seed']) for r in cases])
    def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
    write(args.output/'protocol.json',protocol)
    with zipfile.ZipFile(args.output/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as z:z.write(Path(__file__),Path(__file__).name)
    results=[]
    for prior in cases:
        conf=prior['config'];seed=prior['seed'];n,d=conf['n'],conf['d'];k=max(1,round(n*(1-conf['zero'])))
        points=np.random.default_rng(seed).uniform(size=(n,d));c=cdist(points,points,metric='sqeuclidean');c/=max(float(c.max()),1e-15)
        rng=np.random.default_rng(seed+100000);a=np.zeros(n);b=np.zeros(n)
        a[rng.permutation(n)[:k]]=rng.uniform(.5,1.5,size=k);b[rng.permutation(n)[:k]]=rng.uniform(.5,1.5,size=k)
        a/=a.sum();b/=b.sum();ia,ib=np.flatnonzero(a),np.flatnonzero(b)
        a,b=a[ia],b[ib];kernel=np.exp(-c[np.ix_(ia,ib)]/conf['reg']);u=np.ones(len(a))
        history=[];decision=None;accepted=False;reason='budget exhausted';start=time.perf_counter()
        for iteration in range(1,50001):
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                vb=b/(kernel.T@u);u=a/(kernel@vb)
            if not np.isfinite(u).all() or not np.isfinite(vb).all():
                reason='numerical failure';decision=decision or dict(switch=True,reason=reason,predicted_finish=None);break
            if iteration%50==0:
                residual=float(max(np.abs(u*(kernel@vb)-a).sum(),np.abs(vb*(kernel.T@u)-b).sum()))
                history.append(dict(iteration=iteration,residual=residual))
                if iteration==1000:decision=forecast(history)
                if residual<=1e-9:
                    accepted=True;reason='accepted';decision=decision or forecast(history);break
        row=dict(config=conf,seed=seed,decision=decision,accepted=accepted,reason=reason,iterations=iteration,
                 observed_seconds=time.perf_counter()-start,history=history,rolling_switch_iteration=rolling_decision(history))
        write(args.output/f'{conf["label"]}_{seed}.json',row);results.append(row)
        print(conf['label'],seed,'switch',decision['switch'],'accepted',accepted,'iterations',iteration,flush=True)
    tp=sum(r['decision']['switch'] and not r['accepted'] for r in results)
    fp=sum(r['decision']['switch'] and r['accepted'] for r in results)
    fn=sum(not r['decision']['switch'] and not r['accepted'] for r in results)
    tn=sum(not r['decision']['switch'] and r['accepted'] for r in results)
    rolling=dict(true_switch=sum(r['rolling_switch_iteration'] is not None and not r['accepted'] for r in results),
                 false_switch=sum(r['rolling_switch_iteration'] is not None and r['accepted'] for r in results),
                 missed_failure=sum(r['rolling_switch_iteration'] is None and not r['accepted'] for r in results),
                 true_continue=sum(r['rolling_switch_iteration'] is None and r['accepted'] for r in results))
    write(args.output/'summary.json',dict(cases=results,confusion=dict(true_switch=tp,false_switch=fp,missed_failure=fn,true_continue=tn),rolling_confusion=rolling))
    write(args.output/'manifest.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.iterdir() if p.is_file()})


if __name__=='__main__':main()
