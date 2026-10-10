import numpy as np
import pytest
from ot_benchmark.algorithms.quality_policy import solve_quality_policy


@pytest.mark.parametrize('policy',['periodic','residual_gate','always'])
def test_returns_actual_feasible_plan_and_correct_objective(policy):
    a=np.array([.3,0,.7]);b=np.array([0,.4,.6]);c=np.array([[.2,.8,.1],[.1,.5,.4],[.3,.2,.9]])
    r=solve_quality_policy(a,b,c,reg=.1,policy=policy)
    p=r['plan'];positive=p>0
    assert r['accepted']
    assert np.max(np.abs(p.sum(1)-a))<1e-12
    assert np.max(np.abs(p.sum(0)-b))<1e-12
    objective=np.sum(c*p)+.1*np.sum(p[positive]*(np.log(p[positive])-1))
    assert abs(objective-r['diagnostics']['rounded_primal'])<1e-12
    assert r['diagnostics']['numerical_gap']<=1e-8


def test_numerically_underflowing_ordinary_path_gets_fallback():
    a=np.array([.4,.6]);b=np.array([.7,.3]);c=np.array([[1.,2.],[2.,1.]])
    r=solve_quality_policy(a,b,c,reg=.001,ordinary_budget=1)
    assert r['used_epsilon']
    assert r['accepted']


def test_invalid_policy_rejected():
    with pytest.raises(ValueError):
        solve_quality_policy([1.],[1.],[[0.]],policy='unknown')
