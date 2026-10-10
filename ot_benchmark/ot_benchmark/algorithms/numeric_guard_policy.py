"""Experimental complete policies with frozen numerical-discrepancy triggering."""
import numpy as np
from .transport_diagnostics import diagnose,round_plan
from .segmented_rescue import rescue


def solve_guard_policy(source,target,cost_matrix,reg=.01,policy='guarded',ordinary_budget=50000,discrepancy_threshold=1e-8,switch_iteration=250,epsilon_stop='legacy'):
    a,b,cost=(np.asarray(x,dtype=float) for x in (source,target,cost_matrix))
    if (a.ndim!=1 or b.ndim!=1 or cost.shape!=(len(a),len(b))
        or any(not np.isfinite(x).all() or np.any(x<0) for x in (a,b,cost))
        or a.sum()<=0 or b.sum()<=0 or not np.isfinite(reg) or reg<=0
        or policy not in ('guarded','guarded_250','guarded_cold_250','fixed_warm_250','warm_budget','cold_budget','direct_stabilized','direct_stabilized_250')
        or not isinstance(ordinary_budget,int) or ordinary_budget<1
        or not np.isfinite(discrepancy_threshold) or discrepancy_threshold<=0
        or not isinstance(switch_iteration,int) or isinstance(switch_iteration,bool) or switch_iteration<1
        or epsilon_stop not in ('legacy','quality') or (epsilon_stop=='quality' and reg>1)):raise ValueError('invalid policy input')
    shape=cost.shape;ia,ib=np.flatnonzero(a),np.flatnonzero(b)
    a,b=a[ia]/a.sum(),b[ib]/b.sum();cost=cost[np.ix_(ia,ib)]
    direct=policy.startswith('direct_stabilized')
    guarded=policy in ('guarded','guarded_250','guarded_cold_250')
    kernel=None if direct else np.exp(-cost/reg);u=np.ones(len(a));v=np.ones(len(b))
    accepted=False;trace=[];reason='ordinary budget';stages=[];raw=None;diag=None
    iteration=0
    if direct:reason='direct stabilized'
    for iteration in range(1,(0 if direct else ordinary_budget)+1):
        old_u,old_v=u,v
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            v=b/(kernel.T@u);u=a/(kernel@v)
        if not np.isfinite(u).all() or not np.isfinite(v).all() or np.any(u<=0) or np.any(v<=0):
            u,v=old_u,old_v;reason='nonfinite ordinary update';break
        raw_gate=False
        if iteration%50==0:
            residual=max(np.abs(u*(kernel@v)-a).sum(),np.abs(v*(kernel.T@u)-b).sum())
            raw_gate=residual<=1e-9
        fixed_due=policy=='fixed_warm_250' and iteration>=min(switch_iteration,ordinary_budget)
        if raw_gate or iteration%250==0 or iteration==ordinary_budget or fixed_due:
            try:diag,raw=diagnose(a,b,cost,reg,np.log(u),np.log(v))
            except ValueError:
                reason='invalid reconstruction';break
            accepted=diag['rounded_residual']<=1e-12 and -1e-10<=diag['numerical_gap']<=1e-8
            discrepancy=float(np.abs(u[:,None]*kernel*v[None,:]-raw).sum()) if guarded else None
            trace.append(dict(iteration=iteration,gap=diag['numerical_gap'],discrepancy=discrepancy,accepted=bool(accepted)))
            if accepted:reason='quality accepted';break
            if fixed_due:reason='fixed switch';break
            if guarded and discrepancy>discrepancy_threshold:reason='representation discrepancy';break
    if accepted:
        active_plan=round_plan(raw,a,b)
    else:
        state=dict(u=u,v=v,iterations=iteration)
        if direct:action='cold_stabilized'
        elif policy=='cold_budget':action='cold_epsilon'
        elif policy=='warm_budget' or reason!='ordinary budget':action='warm_stabilized'
        else:action='ordinary_continue'
        cold_state=dict(u=np.ones(len(a)),v=np.ones(len(b)),iterations=0)
        def run_stage(stage_state,stage_action,**options):
            if stage_action=='cold_epsilon' and epsilon_stop=='quality':
                from .quality_epsilon import solve_quality_epsilon
                q=solve_quality_epsilon(a,b,cost,reg,legacy_quality_check=True)
                return dict(plan=q['plan'],accepted=q['accepted'],diagnostics=q['diagnostics'],
                    additional_work=q['outer'],history=[dict(block=1,executed=q['outer'],unit='epsilon_outer',
                    gap=q['diagnostics']['numerical_gap'],accepted=q['accepted'])],quality_trace=q['trace'])
            return rescue(a,b,cost,reg,stage_state,stage_action,**options)
        try:
            options=dict(block_size=250,max_blocks=1000) if policy=='direct_stabilized_250' else (dict(block_size=5000,max_blocks=50) if direct else {})
            if policy in ('guarded_250','guarded_cold_250','fixed_warm_250') and action=='warm_stabilized':
                options=dict(block_size=250,max_blocks=800)
                if policy=='guarded_cold_250':
                    action='cold_stabilized'
                    state=cold_state
            r=run_stage(cold_state if action=='cold_epsilon' else state,action,**options)
            stages.append(dict(action=action,accepted=r['accepted'],work=r['additional_work'],history=r['history']))
            if 'quality_trace' in r:stages[-1]['quality_trace']=r['quality_trace']
        except (ValueError,FloatingPointError) as exc:
            if action=='cold_epsilon':raise
            r=None;stages.append(dict(action=action,accepted=False,work=None,history=[],error=str(exc)))
        if (r is None or not r['accepted']) and action!='cold_epsilon':
            r=run_stage(cold_state,'cold_epsilon')
            stages.append(dict(action='cold_epsilon',accepted=r['accepted'],work=r['additional_work'],history=r['history']))
            if 'quality_trace' in r:stages[-1]['quality_trace']=r['quality_trace']
        accepted=r['accepted'];diag=r['diagnostics'];active_plan=r['plan']
    plan=np.zeros(shape);plan[np.ix_(ia,ib)]=active_plan
    return dict(plan=plan,accepted=bool(accepted),diagnostics=diag,ordinary_iterations=iteration,
        used_epsilon=any(s['action']=='cold_epsilon' for s in stages),used_rescue=bool(stages),reason=reason,
        inspection_trace=trace,stages=stages)
