"""Experimental stage-gated epsilon continuation; no convergence guarantee."""
import numpy as np
import ot
from .transport_diagnostics import diagnose,round_plan

def solve_gated_epsilon(a,b,cost,reg=.001,max_blocks=1000,inner=1000,fixed_hold=None,keep_best=False):
    if not isinstance(keep_best,bool):raise ValueError('keep_best must be boolean')
    if fixed_hold is not None and (not isinstance(fixed_hold,int) or isinstance(fixed_hold,bool) or fixed_hold<1):
        raise ValueError('fixed_hold must be a positive integer or None')
    a,b,cost=[np.asarray(x,dtype=float) for x in (a,b,cost)]
    if (a.ndim!=1 or b.ndim!=1 or not a.size or not b.size or cost.shape!=(len(a),len(b))
        or any(not np.isfinite(x).all() for x in (a,b,cost)) or np.any(a<=0) or np.any(b<=0)
        or np.any(cost<0) or abs(a.sum()-1)>1e-12 or abs(b.sum()-1)>1e-12
        or not np.isfinite(reg) or not 0<reg<=1
        or any(not isinstance(x,int) or isinstance(x,bool) or x<1 for x in (max_blocks,inner))):
        raise ValueError('invalid active normalized inputs or budget')
    alpha,beta=np.zeros(len(a)),np.zeros(len(b));current=1.;trace=[];work=0;accepted=False;stage_blocks=0
    best_plan=None;best_value=float('inf');best_block=None
    def objective(p):
        positive=p>0
        return float(np.sum(cost*p)+reg*np.sum(p[positive]*(np.log(p[positive])-1)))
    for block in range(max_blocks):
        raw,log=ot.bregman.sinkhorn_stabilized(a,b,cost,current,numItermax=inner,stopThr=1e-22,
            warmstart=(alpha,beta),verbose=False,print_period=20,tau=1e3,log=True)
        alpha,beta=log['alpha'],log['beta'];spent=int(log['n_iter'])+1;work+=spent
        q,reconstructed=diagnose(a,b,cost,current,alpha/current,beta/current)
        if keep_best:
            candidate=round_plan(reconstructed,a,b);value=objective(candidate)
            if value<best_value:best_plan,best_value,best_block=candidate,value,block+1
        quality=q['rounded_residual']<=1e-12 and -1e-10<=q['numerical_gap']<=1e-8
        accepted=current==reg and quality
        stage_blocks+=1
        advance=current>reg and (quality and q['raw_residual']<=1e-8 if fixed_hold is None else stage_blocks>=fixed_hold)
        action='accept' if accepted else ('cool' if advance else 'hold')
        trace.append(dict(block=block+1,reg=current,executed=spent,work=work,gap=q['numerical_gap'],raw_residual=q['raw_residual'],action=action))
        if accepted:break
        if advance and block+1<max_blocks:
            current=max(reg,current*.5);stage_blocks=0
    error=None
    try:
        final,raw_target=diagnose(a,b,cost,reg,alpha/reg,beta/reg);plan=round_plan(raw_target,a,b)
    except ValueError as exc:
        final=None;error=str(exc);plan=round_plan(reconstructed,a,b)
    # An intermediate stage is never advertised as completed target solving.
    accepted=bool(current==reg and final is not None and final['rounded_residual']<=1e-12 and -1e-10<=final['numerical_gap']<=1e-8)
    result=dict(plan=plan,accepted=accepted,diagnostics=final,target_diagnostic_error=error,
        reached_reg=current,blocks=block+1,inner_work=work,trace=trace,reason='target quality' if accepted else 'block budget')
    if keep_best:
        original=objective(plan);selected='final reconstruction'
        # Preserve successful output exactly; this is only a failed-result fallback.
        if not accepted and best_value<original:
            plan=best_plan;selected='best rounded stage';result['plan']=plan
        value=objective(plan)
        result['returned_plan_quality']=dict(source=selected,best_stage_block=best_block,
            original_primal=original,rounded_primal=value,
            rounded_residual=float(max(np.abs(plan.sum(1)-a).sum(),np.abs(plan.sum(0)-b).sum())),
            dual=final['dual'] if final else None,
            numerical_gap=value-final['dual'] if final else None)
        # diagnostics remains the original final-reconstruction diagnosis.
        # A better feasible return does not change convergence or acceptance.
    return result
