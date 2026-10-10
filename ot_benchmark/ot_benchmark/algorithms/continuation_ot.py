"""Matched-criterion scaling experiments; no novelty or speed guarantee."""
from dataclasses import dataclass
import numpy as np
from scipy.special import logsumexp
from ..data_generator import validate_transport_plan


@dataclass
class ContinuationResult:
    cost: float
    iterations: int
    residual: float
    switched: bool
    warm_started: bool


def solve_continuation_ot(source,target,cost_matrix,reg=.01,mode='warm',
                          max_iter=10000,switch_iter=2000,check_every=10,tolerance=1e-9):
    """Shared total budget and marginal L1 criterion for all four modes.

    full: ordinary scaling; log: log from start; cold/warm: switch after the
    declared budget or numerical trouble, discarding/preserving finite state.
    Failed ordinary updates consume budget but do not replace last finite state.
    """
    a,b,cost=(np.asarray(x,dtype=float) for x in (source,target,cost_matrix))
    if (a.ndim!=1 or b.ndim!=1 or cost.shape!=(len(a),len(b))
        or not all(np.isfinite(x).all() for x in (a,b,cost))
        or any(np.any(x<0) for x in (a,b,cost)) or a.sum()<=0 or b.sum()<=0
        or not np.isfinite(reg) or reg<=0 or mode not in ('full','log','cold','warm')
        or any(not isinstance(x,int) or x<1 for x in (max_iter,switch_iter,check_every))
        or not 0<tolerance<=1e-7):raise ValueError('invalid inputs or solver settings')
    ia,ib=np.flatnonzero(a>0),np.flatnonzero(b>0)
    a,b=a[ia]/a.sum(),b[ib]/b.sum(); cost=cost[np.ix_(ia,ib)]
    logk=-cost/reg; la,lb=np.log(a),np.log(b)
    with np.errstate(under='ignore'):kernel=np.exp(logk) if mode!='log' else None
    u,v=np.ones(len(a)),np.ones(len(b))
    lu,lv=np.zeros(len(a)),np.zeros(len(b))
    in_log=mode=='log'; switched=False;warm=False;residual=float('inf')
    for iteration in range(1,max_iter+1):
        if not in_log and mode in ('cold','warm') and iteration>switch_iter:
            in_log=True;switched=True;warm=mode=='warm'
            if warm:lu,lv=np.log(u),np.log(v)
        if in_log:
            lv=lb-logsumexp(logk+lu[:,None],axis=0)
            lu=la-logsumexp(logk+lv[None,:],axis=1)
        else:
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                nv=b/(kernel.T@u);nu=a/(kernel@nv)
            if not (np.isfinite(nu).all() and np.isfinite(nv).all() and np.all(nu>0) and np.all(nv>0)):
                if mode=='full':raise ValueError('non-finite ordinary scaling')
                in_log=True;switched=True;warm=mode=='warm'
                if warm:lu,lv=np.log(u),np.log(v)
                continue
            u,v=nu,nv
        if iteration%check_every==0 or iteration==max_iter:
            if in_log:
                rows=np.exp(lu+logsumexp(logk+lv[None,:],axis=1))
                cols=np.exp(lv+logsumexp(logk+lu[:,None],axis=0))
            else:rows=u*(kernel@v);cols=v*(kernel.T@u)
            residual=float(max(np.abs(rows-a).sum(),np.abs(cols-b).sum()))
            if np.isfinite(residual) and residual<=tolerance:
                plan=np.exp(logk+lu[:,None]+lv[None,:]) if in_log else u[:,None]*kernel*v[None,:]
                validate_transport_plan(plan,a,b,tolerance=tolerance)
                return ContinuationResult(float(np.sum(plan*cost)),iteration,residual,switched,warm)
    raise ValueError(f'total iteration budget exhausted; marginal residual={residual}')
