"""Experimental policies with a shared rounded-plan quality contract."""
import numpy as np
import ot
import time
from .transport_diagnostics import diagnose, round_plan


def inspection_interval(previous, current, check_seconds, iteration_seconds, tolerance, mode='combined'):
    """Bounded heuristic using only observed gaps and costs; not a convergence bound."""
    if mode not in ('combined','gap','cost'):raise ValueError('invalid inspection mode')
    iteration,gap=current
    horizon=250.
    if previous is not None:
        old_iteration,old_gap=previous
        if gap>0 and old_gap>gap and iteration>old_iteration:
            rate=np.log(old_gap/gap)/(iteration-old_iteration)
            horizon=.5*max(0.,np.log(max(gap/tolerance,1.)))/rate
        else:
            horizon=2000.
    cost_floor=20.*max(check_seconds,0.)/max(iteration_seconds,1e-12)
    if mode=='gap':cost_floor=0.
    if mode=='cost':horizon=50.
    return int(50*np.ceil(np.clip(max(horizon,cost_floor),50,2000)/50))


def solve_quality_policy(source, target, cost_matrix, reg=.01, policy='periodic',
                         ordinary_budget=50000, check_every=250, gap_tolerance=1e-8):
    a, b, cost = (np.asarray(x, dtype=float) for x in (source, target, cost_matrix))
    if (a.ndim != 1 or b.ndim != 1 or cost.shape != (len(a), len(b))
        or any(not np.isfinite(x).all() or np.any(x < 0) for x in (a,b,cost))
        or a.sum() <= 0 or b.sum() <= 0 or not np.isfinite(reg) or reg <= 0
        or policy not in ('periodic','residual_gate','always','adaptive','adaptive_gap','adaptive_cost')
        or not isinstance(ordinary_budget,int) or ordinary_budget < 1
        or not isinstance(check_every,int) or check_every < 1
        or not np.isfinite(gap_tolerance) or gap_tolerance <= 0):
        raise ValueError('invalid quality policy inputs')
    shape=cost.shape; ia,ib=np.flatnonzero(a),np.flatnonzero(b)
    a,b=a[ia]/a.sum(),b[ib]/b.sum(); cost=cost[np.ix_(ia,ib)]
    checks=0; iteration=0; outer=0; used_epsilon=False; result=None; raw=None
    trace=[];previous=None;next_check=250;diagnostic_seconds=0.
    adaptive=policy in ('adaptive','adaptive_gap','adaptive_cost')
    mode={'adaptive':'combined','adaptive_gap':'gap','adaptive_cost':'cost'}.get(policy)

    def assess(lu,lv):
        nonlocal checks,result,raw
        checks+=1
        result,raw=diagnose(a,b,cost,reg,lu,lv)
        return result['rounded_residual'] <= 1e-12 and -1e-10 <= result['numerical_gap'] <= gap_tolerance

    accepted=False
    if policy != 'always':
        kernel=np.exp(-cost/reg);u=np.ones(len(a))
        loop_start=time.perf_counter() if adaptive else None
        for iteration in range(1,ordinary_budget+1):
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                v=b/(kernel.T@u);u=a/(kernel@v)
            if not np.isfinite(u).all() or not np.isfinite(v).all() or np.any(u<=0) or np.any(v<=0):
                break
            raw_gate=False
            if iteration%50==0:
                residual=max(np.abs(u*(kernel@v)-a).sum(),np.abs(v*(kernel.T@u)-b).sum())
                raw_gate=residual<=1e-9
            if raw_gate or iteration==ordinary_budget or (policy=='periodic' and iteration%check_every==0) or (adaptive and iteration>=next_check):
                check_start=time.perf_counter() if adaptive else None
                accepted=assess(np.log(u),np.log(v))
                if adaptive:
                    duration=time.perf_counter()-check_start;diagnostic_seconds+=duration
                    per_iteration=max((time.perf_counter()-loop_start-diagnostic_seconds)/iteration,1e-12)
                    current=(iteration,result['numerical_gap'])
                    interval=inspection_interval(previous,current,duration,per_iteration,gap_tolerance,mode=mode)
                    trace.append(dict(iteration=iteration,gap=current[1],check_seconds=duration,
                        iteration_seconds=per_iteration,next_interval=interval,accepted=bool(accepted)))
                    next_check=iteration+interval;previous=current
                if accepted:break
    if not accepted:
        used_epsilon=True
        _,log=ot.bregman.sinkhorn_epsilon_scaling(a,b,cost,reg,numItermax=200,
            numInnerItermax=1000,epsilon0=1.,stopThr=1e-22,log=True,warn=True)
        outer=int(log['niter'])+1
        # Reconstruct at the target regularization, using POT's final dual potentials.
        accepted=assess(log['alpha']/reg,log['beta']/reg)
    rounded=round_plan(raw,a,b)
    plan=np.zeros(shape);plan[np.ix_(ia,ib)]=rounded
    return dict(plan=plan,accepted=bool(accepted),diagnostics=result,
                ordinary_iterations=iteration,epsilon_outer_iterations=outer,
                used_epsilon=used_epsilon,quality_checks=checks,inspection_trace=trace)
