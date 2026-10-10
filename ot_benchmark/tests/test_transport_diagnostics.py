import numpy as np
import pytest
from ot_benchmark.algorithms.transport_diagnostics import round_plan,diagnose


def test_rounding_restores_both_marginals():
    a=np.array([.2,.8]);b=np.array([.3,.7])
    for p in [np.zeros((2,2)),np.array([[2.,0.],[0.,.1]])]:
        q=round_plan(p,a,b)
        assert np.all(q>=0)
        assert np.allclose(q.sum(axis=1),a,atol=1e-12,rtol=0)
        assert np.allclose(q.sum(axis=0),b,atol=1e-12,rtol=0)


def test_zero_cost_analytic_optimum_and_gauge_invariance():
    a=np.array([.2,.8]);b=np.array([.3,.7]);c=np.zeros((2,2));reg=.1
    r,_=diagnose(a,b,c,reg,np.log(a),np.log(b))
    shifted,_=diagnose(a,b,c,reg,np.log(a)+7,np.log(b)-7)
    expected=reg*(np.sum(a*np.log(a))+np.sum(b*np.log(b))-1)
    assert r['rounded_primal']==pytest.approx(expected,abs=1e-12)
    assert abs(r['numerical_gap'])<1e-12
    assert shifted['dual']==pytest.approx(r['dual'],abs=1e-12)


def test_gap_is_not_computed_from_infeasible_primal():
    a=np.array([.2,.8]);b=np.array([.3,.7])
    r,_=diagnose(a,b,np.zeros((2,2)),.1,np.zeros(2),np.zeros(2))
    assert r['raw_residual']>1
    assert r['rounded_residual']<1e-12
    assert r['numerical_gap']>0


def test_nonzero_cost_bounds_independent_scalar_optimum():
    from scipy.optimize import minimize_scalar
    a=np.array([.4,.6]);b=np.array([.7,.3]);c=np.array([[0.,.2],[.2,0.]]);reg=.1
    def objective(x):
        p=np.array([[x,a[0]-x],[b[0]-x,a[1]-b[0]+x]])
        positive=p>0
        return float(np.sum(p*c)+reg*np.sum(p[positive]*(np.log(p[positive])-1)))
    optimum=minimize_scalar(objective,bounds=(.1,.4),method='bounded',options={'xatol':1e-14})
    assert optimum.success
    kernel=np.exp(-c/reg);u=np.ones(2);v=b/(kernel.T@u);u=a/(kernel@v)
    r,_=diagnose(a,b,c,reg,np.log(u),np.log(v))
    assert r['dual']<=optimum.fun+1e-12
    assert r['rounded_primal']>=optimum.fun-1e-12
