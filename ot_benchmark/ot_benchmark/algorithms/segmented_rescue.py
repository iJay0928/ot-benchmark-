"""Experimental bounded actions from a shared ordinary-scaling state."""
import numpy as np
import ot
from .transport_diagnostics import diagnose,round_plan


def ordinary_state(a,b,cost,reg,iterations=50000):
    a,b,cost=(np.asarray(x,dtype=float) for x in (a,b,cost))
    if (a.ndim!=1 or b.ndim!=1 or cost.shape!=(len(a),len(b))
        or any(not np.isfinite(x).all() for x in (a,b,cost)) or np.any(a<=0) or np.any(b<=0)
        or np.any(cost<0) or not np.isfinite(reg) or reg<=0
        or abs(a.sum()-1)>1e-12 or abs(b.sum()-1)>1e-12
        or not isinstance(iterations,int) or iterations<1):raise ValueError('invalid active normalized inputs')
    kernel=np.exp(-cost/reg);u=np.ones(len(a));v=np.ones(len(b))
    for _ in range(iterations):
        with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
            v=b/(kernel.T@u);u=a/(kernel@v)
        if not np.isfinite(u).all() or not np.isfinite(v).all() or np.any(u<=0) or np.any(v<=0):
            raise FloatingPointError('ordinary snapshot cannot represent this state')
    return dict(u=u,v=v,iterations=iterations)


def rescue(a,b,cost,reg,state,action,block_size=5000,max_blocks=40):
    if action not in ('ordinary_continue','warm_stabilized','cold_stabilized','cold_epsilon'):
        raise ValueError('unknown rescue action')
    if any(not isinstance(x,int) or x<1 for x in (block_size,max_blocks)):raise ValueError('invalid block budget')
    u=state['u'].copy();v=state['v'].copy();lu,lv=np.log(u),np.log(v)
    def assess():
        diag,raw=diagnose(a,b,cost,reg,lu,lv)
        good=diag['rounded_residual']<=1e-12 and -1e-10<=diag['numerical_gap']<=1e-8
        return diag,raw,good
    diag,raw,accepted=assess();history=[];spent=0
    kernel=np.exp(-cost/reg) if action=='ordinary_continue' else None
    if action=='cold_stabilized':lu,lv=np.zeros(len(a)),np.zeros(len(b))
    for block in range(max_blocks):
        if accepted:break
        if action=='cold_epsilon':
            _,log=ot.bregman.sinkhorn_epsilon_scaling(a,b,cost,reg,numItermax=1000,
                numInnerItermax=1000,epsilon0=1.,stopThr=1e-22,log=True,warn=True)
            lu,lv=log['alpha']/reg,log['beta']/reg
            executed=int(log['niter'])+1;unit='epsilon_outer'
        elif action=='ordinary_continue':
            for _ in range(block_size):
                with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                    v=b/(kernel.T@u);u=a/(kernel@v)
                if not np.isfinite(u).all() or not np.isfinite(v).all() or np.any(u<=0) or np.any(v<=0):
                    raise FloatingPointError('ordinary continuation numerical failure')
            lu,lv=np.log(u),np.log(v);executed=block_size;unit='scaling_iteration'
        else:
            _,log=ot.bregman.sinkhorn_stabilized(a,b,cost,reg,numItermax=block_size,
                stopThr=1e-12,warmstart=(reg*lu,reg*lv),log=True,warn=True)
            lu,lv=log['logu'],log['logv'];executed=int(log['n_iter'])+1;unit='stabilized_iteration'
        spent+=executed;diag,raw,accepted=assess()
        history.append(dict(block=block+1,executed=executed,unit=unit,gap=diag['numerical_gap'],accepted=bool(accepted)))
        if action=='cold_epsilon':break
    return dict(plan=round_plan(raw,a,b),accepted=bool(accepted),diagnostics=diag,
        additional_work=spent,history=history,action=action)
