"""Executable baseline policies. Rolling heuristic is experimental, not default."""
import numpy as np
import ot


def solve_policy(source,target,cost_matrix,reg=.01,policy='budget',ordinary_budget=50000):
    a,b,cost=(np.asarray(x,dtype=float) for x in (source,target,cost_matrix))
    if (a.ndim!=1 or b.ndim!=1 or cost.shape!=(len(a),len(b))
        or not all(np.isfinite(x).all() for x in (a,b,cost))
        or any(np.any(x<0) for x in (a,b,cost)) or a.sum()<=0 or b.sum()<=0
        or not np.isfinite(reg) or reg<=0 or policy not in ('budget','rolling','always')
        or not isinstance(ordinary_budget,int) or ordinary_budget<1):raise ValueError('invalid policy inputs')
    ia,ib=np.flatnonzero(a),np.flatnonzero(b)
    a,b=a[ia]/a.sum(),b[ib]/b.sum();cost=cost[np.ix_(ia,ib)]
    iteration=0;reason='always epsilon';history=[];plan=None
    if policy!='always':
        kernel=np.exp(-cost/reg);u=np.ones(len(a));v=np.ones(len(b));reason='ordinary budget exhausted'
        for iteration in range(1,ordinary_budget+1):
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                v=b/(kernel.T@u);u=a/(kernel@v)
            if not np.isfinite(u).all() or not np.isfinite(v).all():
                reason='numerical failure';break
            if iteration%50==0 or iteration==ordinary_budget:
                residual=float(max(np.abs(u*(kernel@v)-a).sum(),np.abs(v*(kernel.T@u)-b).sum()))
                if residual<=1e-9:
                    plan=u[:,None]*kernel*v[None,:];reason='ordinary accepted';break
                if policy=='rolling':
                    history.append((iteration,residual));history=history[-11:]
                    if iteration%1000==0 and iteration<=10000:
                        x=np.array([r[0] for r in history]);y=np.log([max(r[1],1e-300) for r in history])
                        slope=float(np.polyfit(x,y,1)[0])
                        finish=float(iteration+(np.log(1e-9)-y[-1])/slope) if slope<0 else float('inf')
                        if finish>ordinary_budget:reason='rolling predicted budget failure';break
    fallback=plan is None;outer=0
    if fallback:
        plan,log=ot.bregman.sinkhorn_epsilon_scaling(a,b,cost,reg,numItermax=200,
            numInnerItermax=1000,epsilon0=1.,stopThr=1e-22,log=True,warn=True)
        outer=int(log['niter'])+1
    finite=bool(np.isfinite(plan).all() and np.all(plan>=0))
    residual=float(max(np.abs(plan.sum(axis=1)-a).sum(),np.abs(plan.sum(axis=0)-b).sum())) if finite else None
    return dict(accepted=finite and residual<=1e-9,residual=residual,cost=float(np.sum(plan*cost)) if finite else None,
                ordinary_iterations=iteration,epsilon_outer_iterations=outer,used_epsilon=fallback,reason=reason)
