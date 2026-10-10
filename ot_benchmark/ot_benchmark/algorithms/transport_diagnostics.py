"""Numerical primal/dual diagnostics; rounding follows AWR 2017 Algorithm 2.

Objective: <C,P> + reg * sum(P * (log(P)-1)). Floating-point diagnostics
are not interval-arithmetic certificates and do not alter solver acceptance.
"""
import numpy as np


def round_plan(plan,a,b):
    p=np.asarray(plan,dtype=float).copy();a=np.asarray(a,dtype=float);b=np.asarray(b,dtype=float)
    if (p.shape!=(len(a),len(b)) or any(not np.isfinite(x).all() or np.any(x<0) for x in (p,a,b))
        or not np.isclose(a.sum(),1.,rtol=0,atol=1e-12) or not np.isclose(b.sum(),1.,rtol=0,atol=1e-12)):
        raise ValueError('invalid normalized transport inputs')
    rows=p.sum(axis=1);scale=np.ones_like(a)
    np.divide(a,rows,out=scale,where=rows>0);p*=np.minimum(scale,1)[:,None]
    cols=p.sum(axis=0);scale=np.ones_like(b)
    np.divide(b,cols,out=scale,where=cols>0);p*=np.minimum(scale,1)[None,:]
    ra=np.maximum(a-p.sum(axis=1),0);rb=np.maximum(b-p.sum(axis=0),0)
    if ra.sum()>0 and rb.sum()>0:p+=np.outer(ra,rb)/ra.sum()
    residual=float(max(np.abs(p.sum(axis=1)-a).sum(),np.abs(p.sum(axis=0)-b).sum()))
    if residual>1e-12:raise ValueError('rounding failed numerical feasibility')
    return p


def diagnose(a,b,cost,reg,log_u,log_v):
    a,b,cost,lu,lv=(np.asarray(x,dtype=float) for x in (a,b,cost,log_u,log_v))
    if reg<=0 or not np.isfinite(reg) or any(not np.isfinite(x).all() for x in (cost,lu,lv)):
        raise ValueError('invalid potentials or regularization')
    with np.errstate(over='ignore',under='ignore'):
        raw=np.exp(lu[:,None]+lv[None,:]-cost/reg)
    if not np.isfinite(raw).all():raise ValueError('non-finite reconstructed plan')
    rounded=round_plan(raw,a,b);positive=rounded>0
    entropy=float(np.sum(rounded[positive]*(np.log(rounded[positive])-1)))
    primal=float(np.sum(rounded*cost)+reg*entropy)
    dual=float(reg*(a@lu+b@lv-raw.sum()))
    gap=primal-dual
    if gap < -1e-10:raise ValueError('negative primal-dual gap beyond roundoff allowance')
    result=dict(raw_residual=float(max(np.abs(raw.sum(axis=1)-a).sum(),np.abs(raw.sum(axis=0)-b).sum())),
                rounded_residual=float(max(np.abs(rounded.sum(axis=1)-a).sum(),np.abs(rounded.sum(axis=0)-b).sum())),
                rounded_primal=primal,dual=dual,numerical_gap=gap,
                rounding_l1=float(np.abs(rounded-raw).sum()),
                raw_transport_cost=float(np.sum(raw*cost)),rounded_transport_cost=float(np.sum(rounded*cost)))
    return result,raw
