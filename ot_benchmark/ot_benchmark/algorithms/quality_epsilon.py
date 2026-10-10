"""Experimental NumPy outer schedule matching installed POT epsilon scaling.

Uses POT's stabilized inner solver. Schedule, state transfer, minimum outer
count and legacy stopping semantics follow POT 0.9.6.post1's implementation.
The optional quality stop evaluates the FINAL regularization objective.
This is a stopping-rule experiment, not a new epsilon-scaling algorithm.
"""
import numpy as np
import ot
from .transport_diagnostics import diagnose,round_plan


def solve_quality_epsilon(a,b,cost,reg=.001,max_outer=1000,inner=1000,
                          check_every=10,quality_stop=True,legacy_quality_check=False):
    a,b,cost=[np.asarray(x,dtype=float) for x in (a,b,cost)]
    if (a.ndim!=1 or b.ndim!=1 or cost.shape!=(len(a),len(b)) or a.size==0 or b.size==0
        or any(not np.isfinite(x).all() for x in (a,b,cost))
        or np.any(a<=0) or np.any(b<=0) or np.any(cost<0)
        or abs(a.sum()-1)>1e-12 or abs(b.sum()-1)>1e-12
        or not np.isfinite(reg) or not 0<reg<=1
        or any(not isinstance(x,int) or isinstance(x,bool) or x<1 for x in (max_outer,inner,check_every))):
        raise ValueError('invalid positive normalized inputs or schedule')
    alpha,beta=np.zeros(len(a)),np.zeros(len(b));err=1.;trace=[];work=0;accepted=False
    cap=max(35,max_outer);reason='outer budget'
    for i in range(cap):
        current=(1.-reg)*np.exp(-i)+reg
        raw,log=ot.bregman.sinkhorn_stabilized(a,b,cost,current,numItermax=inner,
            stopThr=1e-22,warmstart=(alpha,beta),verbose=False,print_period=20,tau=1e3,log=True)
        alpha,beta=log['alpha'],log['beta'];work+=int(log['n_iter'])+1
        if i%10==0:err=float(np.linalg.norm(raw.sum(0)-b)**2+np.linalg.norm(raw.sum(1)-a)**2)
        legacy_due=err<=1e-22 and i>35
        if i>=35 and (i%check_every==0 or i==cap-1 or (legacy_quality_check and quality_stop and legacy_due)):
            diag,reconstructed=diagnose(a,b,cost,reg,alpha/reg,beta/reg)
            accepted=diag['rounded_residual']<=1e-12 and -1e-10<=diag['numerical_gap']<=1e-8
            trace.append(dict(outer=i+1,schedule_reg=float(current),target_reg=reg,gap=diag['numerical_gap'],accepted=bool(accepted),inner_work=work,legacy_due=bool(legacy_due)))
            if quality_stop and accepted:reason='target quality';break
        if not quality_stop and err<=1e-22 and i>35:reason='legacy residual';break
    diag,reconstructed=diagnose(a,b,cost,reg,alpha/reg,beta/reg)
    accepted=diag['rounded_residual']<=1e-12 and -1e-10<=diag['numerical_gap']<=1e-8
    return dict(plan=round_plan(reconstructed,a,b),raw_plan=raw,alpha=alpha,beta=beta,
        accepted=bool(accepted),diagnostics=diag,outer=i+1,inner_work=work,trace=trace,reason=reason)
